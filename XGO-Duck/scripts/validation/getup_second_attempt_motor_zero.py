"""Check real delayed BAM dispatch on archived configurations without integration."""

import numpy as np
import torch

from getup_release_fork_state import digest, signature
from getup_release_scene_state import SceneReleaseSnapshot
from getup_second_attempt import apply_processed_targets
from getup_second_attempt_archive import controller
from getup_shared_release import cpu


ROLES = ("reference","repeat","retreat","fixed-voltage")


def measure(env,memory,spec,names,role_fixture,edge):
    p = spec["environment"]
    cfg = spec["draft"]
    robot = env.scene["robot"]
    joint_ids,actual_names = robot.find_joints(names,preserve_order=True)
    assert actual_names == names and len(robot.actuators) == 1
    actuator = robot.actuators[0]
    inner = actuator._bam_model.actuator
    model = env.sim.mj_model
    motor_slots = []
    for name in names:
        joint = model.joint(f"robot/{name}").id
        slot, = np.flatnonzero((model.actuator_trntype == 0)&(model.actuator_trnid[:,0] == joint))
        motor_slots.append(int(slot))
    actuator_slots = cpu(actuator._global_ctrl_ids).tolist()
    assert set(motor_slots) == set(actuator_slots)
    order = [actuator_slots.index(i) for i in motor_slots]
    manager = env.action_manager
    term = manager.get_term("joint_pos")
    phase = controller(env.num_envs,env.device,spec["screen"],cfg["return"],p,spec["neck_column"])
    phase.boundary = spec["screen"]["last_decision_control"]
    k = len(role_fixture["previous"])
    assert k == 24 and edge in {"first","last"}
    phase.trigger[:k] = phase.boundary-(0 if edge == "first" else phase.duration-1)
    phase.supine[:k] = True
    control_state = phase.snapshot()
    start_hash = digest(signature(control_state))
    snapshot = SceneReleaseSnapshot(env,memory,cfg["maximum_warp_snapshot_bytes"])
    origin = snapshot.verify()
    assert origin["counters"] == [0,0]
    original_delay = actuator.apply_delay
    delayed = None

    def inspect_delay(command):
        nonlocal delayed
        result = original_delay(command)
        delayed = cpu(result.position_target)[:,order]
        return result

    def put(value,source):
        value[:k].copy_(torch.as_tensor(source,device=env.device,dtype=value.dtype))

    def path(enabled,voltages=None):
        data = env.sim.data
        put(data.qpos,role_fixture["qpos"])
        put(data.qvel,role_fixture["qvel"])
        np.testing.assert_allclose(cpu(robot.data.joint_pos[:,joint_ids])[:k],role_fixture["measured"],atol=2e-6,rtol=0.)
        physics = [cpu(item) for item in (data.qpos,data.qvel,data.time)]
        env.episode_length_buf.fill_(phase.boundary)
        raw = torch.full_like(manager._action,.01)
        put(raw,role_fixture["raw"])
        manager.process_action(raw)
        put(term._processed_actions,role_fixture["previous"])
        for _ in range(env.cfg.decimation):
            manager.apply_action()
            env.scene.write_data_to_sim()
        term.previous_target.copy_(term._processed_actions)
        previous = term.previous_target.clone()
        manager.process_action(raw)
        np.testing.assert_array_equal(cpu(term.requested_target)[:k],role_fixture["requested"])
        apply_processed_targets(term,phase,phase.boundary,previous,enabled)
        assert torch.equal(manager._action,raw) and torch.equal(term._raw_actions,raw)
        saved = dict(previous=cpu(previous),request=cpu(term.requested_target),target=cpu(term._processed_actions),
            history=cpu(torch.stack([manager._action,manager._prev_action,manager._prev_prev_action])),
            nominal=cpu(actuator.vin_tensor),gain=cpu(actuator.vin_drop_gain))
        assert torch.equal(term.previous_target,term._processed_actions)
        if voltages is not None:
            actuator.vin_drop_gain.zero_()
        writes = []
        for j in range(env.cfg.decimation):
            if voltages is not None:
                actuator.vin_tensor.copy_(torch.as_tensor(voltages[j],device=env.device))
            manager.apply_action()
            assert torch.equal(robot.data.joint_pos_target[:,joint_ids],term._processed_actions)
            old = cpu(actuator._prev_motor_torque)
            env.scene.write_data_to_sim()
            writes.append(dict(delayed=delayed,vin=cpu(inner.vin),previous_motor=old,ctrl=cpu(data.ctrl)[:,motor_slots]))
        for previous_array,value in zip(physics,(data.qpos,data.qvel,data.time),strict=True):
            np.testing.assert_array_equal(cpu(value),previous_array)
        saved.update({key:np.stack([line[key] for line in writes]) for key in writes[0]})
        assert snapshot.restore() == origin
        assert digest(signature(phase.snapshot())) == start_hash
        phase.restore(control_state)
        return saved

    actuator.apply_delay = inspect_delay
    try:
        with torch.no_grad():
            ref = path(False)
            repeat = path(False)
            natural = path(True)
            fixed = path(True,ref["vin"])
    finally:
        del actuator.apply_delay
    meta = dict(scene=origin,controller_sha256=start_hash,names=names,fixture_worlds=k,
        fixed_joint_order_motor_ids=motor_slots,vin_floor=actuator.cfg.vin_min,physical_substeps=0,
        measurement_writes=4*env.cfg.decimation,warm_writes=4*env.cfg.decimation)
    return meta,{f"{role}__{key}":value for role,rows in zip(ROLES,(ref,repeat,natural,fixed),strict=True)
                 for key,value in rows.items()}


def audit(meta,data,fixture):
    assert meta["physical_substeps"] == 0 and meta["scene"]["counters"] == [0,0]
    k = meta["fixture_worlds"]
    assert meta["measurement_writes"] == meta["warm_writes"] == 20
    ref,repeat,natural,fixed = [{key.split("__",1)[1]:val for key,val in data.items() if key.startswith(role+"__")}
                               for role in ROLES]
    assert all(np.isfinite(value).all() for value in data.values())
    for key,value in ref.items():
        np.testing.assert_array_equal(value,repeat[key])
    for arm in (natural,fixed):
        for key in ("request","history","previous","nominal","gain"):
            np.testing.assert_array_equal(arm[key],ref[key])
        np.testing.assert_array_equal(arm["target"][:k],fixture["candidate"])
        np.testing.assert_array_equal(arm["target"][k:],ref["target"][k:])
        for key in ("ctrl","delayed","vin"):
            np.testing.assert_array_equal(arm[key][:,k:],ref[key][:,k:])
        np.testing.assert_array_equal(arm["ctrl"][0],ref["ctrl"][0])
        np.testing.assert_array_equal(arm["delayed"][0],ref["delayed"][0])
        np.testing.assert_array_equal(arm["delayed"][1:],np.broadcast_to(arm["target"],arm["delayed"][1:].shape))
    for arm in (ref,natural):
        vin = arm["nominal"]-arm["gain"]*abs(arm["previous_motor"]).sum(axis=-1,keepdims=True)
        np.testing.assert_allclose(arm["vin"],vin.clip(min=meta["vin_floor"]),atol=1e-6,rtol=0.)
    np.testing.assert_array_equal(ref["vin"],fixed["vin"])
    np.testing.assert_array_equal(natural["delayed"],fixed["delayed"])
    allowed = natural["delayed"] != ref["delayed"]
    direct = fixed["ctrl"]-ref["ctrl"]
    coupled = natural["ctrl"]-ref["ctrl"]
    changed_voltage = natural["vin"] != ref["vin"]
    assert not direct[~allowed].any()
    assert not np.where(allowed|changed_voltage,0.,coupled).any()
    assert not changed_voltage[:2].any()
    neck = meta["names"].index("neck_pitch")
    return dict(status="passed",fixture_worlds=k,untouched_worlds_exact=True,raw_action_history_exact=True,
        target_dispatch_exact=True,first_write_preserves_previous_delay=True,
        max_neck_torque_change_nm=float(abs(coupled[...,neck]).max()),
        max_direct_motor_change_nm=float(abs(direct).max()),max_same_world_supply_change_v=float(abs(natural["vin"]-ref["vin"]).max()),
        target_modified_worlds=int((natural["target"]!=ref["target"]).any(axis=1).sum()),
        writes_with_neck_motor_effect=(np.flatnonzero(coupled[...,neck].any(axis=1))+1).tolist())
