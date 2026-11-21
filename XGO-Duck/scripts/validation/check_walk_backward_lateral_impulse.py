"""Check body-lateral free-joint velocity changes on the real CPU walking plant."""
from copy import deepcopy
import inspect
from pathlib import Path

import torch

import collect_walk_backward_onset as onset
from record_io import write
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT/'configs/experiments/common_walk_backward_lateral_probe_draft.json'
OUT = ROOT/'records/reports/common-walk-backward-start-pulse/lateral-probe-cpu.json'


def lateral_velocity_delta(env,ids,speed):
    entity = env.scene['robot']
    address = entity.data.indexing.free_joint_q_adr
    quat = env.sim.data.qpos[ids[:,None],address[3:7]]
    w,x,y,z = quat.unbind(dim=1)
    yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    side = torch.stack((-torch.sin(yaw),torch.cos(yaw)),dim=1)
    expected = side*speed[:,None]
    # The free joint stores linear velocity in world axes. Restrict the write
    # to the two horizontal components, preserving angular values bitwise.
    vadr = entity.data.indexing.free_joint_v_adr
    env.sim.data.qvel[ids[:,None],vadr[:2]] += expected
    return expected, yaw


def main():
    from mjlab.envs import ManagerBasedRlEnv

    assert not OUT.exists() and not torch.cuda.is_initialized()
    draft = read(DRAFT)
    evidence = read(ROOT/draft['evidence'])
    assert evidence['status']=='passed-saved-geometry-review' and not evidence['heldout_used']
    spec = read(ROOT/draft['scene_and_scoring_reference'])
    p = deepcopy(spec['environment'])
    p.update(seed=1075,parallel_worlds=8,episodes_per_command=8,groups={'backward':list(range(8))})
    torch.set_num_threads(2)
    cfg,_ = onset.config(p)
    env = ManagerBasedRlEnv(cfg=cfg,device='cpu',render_mode=None)
    try:
        env.reset(seed=p['seed'])
        entity = env.scene['robot']
        address = entity.data.indexing.free_joint_q_adr
        vadr = entity.data.indexing.free_joint_v_adr
        ids = torch.tensor([0,2,4,6])
        angles = torch.tensor([0.,1.5707963267948966,-1.5707963267948966,3.141592653589793])
        quats = torch.zeros(4,4)
        quats[:,0],quats[:,3] = torch.cos(angles/2),torch.sin(angles/2)
        env.sim.data.qpos[ids[:,None],address[3:7]] = quats
        qpos,qvel,ctrl = (getattr(env.sim.data,name).clone() for name in ('qpos','qvel','ctrl'))
        action = env.action_manager.get_term('joint_pos')
        history = action._raw_actions.clone(),action._processed_actions.clone()
        command = env.command_manager.get_command('twist').clone()
        raw_write = type(entity.data).write_root_velocity
        source = Path(inspect.getfile(raw_write))
        body = inspect.getsource(raw_write)
        assert 'velocity_qvel = torch.cat([velocity[:, :3], ang_vel_b], dim=-1)' in body
        assert 'self.indexing.free_joint_v_adr' in body
        speed = torch.tensor([1.,-1.,1.,-1.])*draft['design']['dose_draft_delta_lateral_m_s']
        intended,yaw = lateral_velocity_delta(env,ids,speed)
        expected = qvel.clone()
        expected[ids[:,None],vadr[:2]] += intended
        torch.testing.assert_close(env.sim.data.qvel.clone(),expected,atol=0,rtol=0)
        measured = (env.sim.data.qvel.clone()-qvel)[ids[:,None],vadr[:2]]
        forward = torch.stack((torch.cos(yaw),torch.sin(yaw)),dim=1)
        lateral = torch.stack((-torch.sin(yaw),torch.cos(yaw)),dim=1)
        torch.testing.assert_close((measured*forward).sum(dim=1),torch.zeros_like(speed),atol=2e-7,rtol=0)
        torch.testing.assert_close((measured*lateral).sum(dim=1),speed,atol=2e-7,rtol=0)
        for actual,original in ((env.sim.data.qpos,qpos),(env.sim.data.ctrl,ctrl),
            (action._raw_actions,history[0]),(action._processed_actions,history[1]),
            (env.command_manager.get_command('twist'),command)):
            torch.testing.assert_close(actual.clone(),original,atol=0,rtol=0)
        assert env._sim_step_counter==env.common_step_counter==0
        assert not torch.cuda.is_initialized()
        write(OUT,dict(status='passed-real-cpu-zero-step',source_sha256=sha(Path(__file__)),draft_sha256=sha(DRAFT),
            installed_velocity_source=source.as_posix(),installed_velocity_source_sha256=sha(source),
            worlds=8,affected_worlds=ids.tolist(),world_horizontal_delta=measured.tolist(),
            body_lateral_delta=(measured*lateral).sum(dim=1).tolist(),
            maximum_body_forward_delta=float((measured*forward).sum(dim=1).abs().max()),
            all_other_qvel_fields_bitwise_equal=True,controller_history_unchanged=True,physical_substeps=0,
            heldout_used=False,learning_updates=0))
        print('WALK_LATERAL_IMPULSE_CPU_PASSED',measured.tolist(),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
