"""Verify a two-coordinate root velocity write and restoration on the CPU plant."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write
from check_continuous_lateral_push_geometry import DRAFT, DEST as GEOMETRY, lateral_world, recovery_window
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import FORWARD
from run_continuous_roll_stand_walk import configuration


OUT = ROOT/'records/reports/continuous-forward-yaw-pair'
DEST = OUT/'lateral-push-writer-cpu.json'


def apply_lateral_push(env,eligible,signed_speed):
    """Update the free joint's world horizontal components at one boundary."""
    assert np.asarray(eligible).dtype==np.bool_ and np.asarray(eligible).shape==(env.num_envs,)
    assert abs(signed_speed)<=.08
    robot = env.scene['robot']
    index = robot.data.indexing
    data = env.sim.data
    ids = torch.from_numpy(np.flatnonzero(eligible)).to(env.device)
    horizontal = index.free_joint_v_adr[:2]
    qpos = data.qpos[ids[:,None],index.free_joint_q_adr[3:7]]
    _,delta = lateral_world(qpos.cpu().numpy(),signed_speed)
    previous = data.qvel[ids[:,None],horizontal]
    change = torch.as_tensor(delta[:,:2],dtype=previous.dtype,device=env.device)
    if signed_speed:
        # The pinned library writes world linear components directly to these
        # slots. Skip its angular round-trip to preserve the other four slots.
        data.qvel[ids[:,None],horizontal] = previous+change
    actual = data.qvel[ids[:,None],horizontal]-previous
    torch.testing.assert_close(actual,change,rtol=0,atol=5e-8)
    return dict(world_ids=ids.cpu().numpy(),target_world_xy=change.cpu().numpy(),
                measured_world_xy=actual.cpu().numpy())


def main():
    from mjlab.entity.data import EntityData
    from mjlab.envs import ManagerBasedRlEnv
    import inspect

    assert not DEST.exists()
    draft,geometry = read(DRAFT),read(GEOMETRY)
    assert geometry['nominal_screen_passed'] and draft['status']=='draft-not-executable'
    proof_path = ROOT/draft['required_independent_result']
    proof = read(proof_path)
    assert proof['passed_development_gate'] and not proof['heldout_used']
    folder = ROOT/proof['record']
    protocol,archive = folder/'protocol.json',folder/'trace.npz'
    assert sha(protocol)==proof['protocol_sha256']
    assert sha(archive)==proof['source_sha256'][archive.relative_to(ROOT).as_posix()]
    p = read(protocol)
    k = geometry['fork_control']
    with np.load(archive) as saved:
        qpos,qvel = (saved[key][k].copy() for key in ('qpos','qvel'))
        active = saved['active'][k].copy()
        eligible = active&(saved['phase_next'][k]==FORWARD)
        recovery = recovery_window({key:saved[key] for key in ('active','velocity','joint_envelope_ok')},k,p)
    assert int(eligible.sum())==proof['counts']['entered_forward']==125
    both = np.array([row['success'] for row in proof['forward']])&recovery
    assert int(both.sum())>=p['minimum_continuous_successes']
    library = Path(inspect.getsourcefile(EntityData))
    assert library.is_relative_to(WORK) and 'torch.cat([velocity[:, :3], ang_vel_b]' in inspect.getsource(EntityData.write_root_velocity)
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=configuration(p),device='cpu',render_mode=None)
    checks = {}
    try:
        env.seed(p['seed'])
        env.reset()
        env.sim.data.qpos.copy_(torch.from_numpy(qpos))
        env.sim.data.qvel.copy_(torch.from_numpy(qvel))
        memory = SimpleNamespace(previous=torch.zeros(env.num_envs,14),history=torch.zeros(env.num_envs,14),
                                 previous_features=np.zeros((env.num_envs,2),dtype=np.float32))
        fork = RollEntrySnapshot(env,memory,268435456)
        receipt = fork.verify()
        assert receipt['counters']==[0,0]
        addresses = env.scene['robot'].data.indexing.free_joint_v_adr[:2].tolist()
        mask = np.ones(qvel.shape,dtype=bool)
        mask[np.ix_(eligible,addresses)] = False
        for role,value in draft['proposed_roles_body_horizontal_lateral_m_s'].items():
            applied = apply_lateral_push(env,eligible,value)
            np.testing.assert_array_equal(env.sim.data.qpos.numpy(),qpos)
            np.testing.assert_array_equal(env.sim.data.qvel.numpy()[mask],qvel[mask])
            after = fork.live_signatures()
            changed = [key for key,digest in after.items() if digest!=fork.expected[key]]
            assert changed==(['warp.qvel'] if value else []),changed
            assert fork.restore()==receipt
            checks[role] = dict(touched_fields=changed,affected_worlds=len(applied['world_ids']) if value else 0,
                maximum_xy_error_m_s=float(np.abs(applied['measured_world_xy']-applied['target_world_xy']).max()))
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    write(DEST,dict(status='passed-actual-cpu-writer-and-restore',fixture_source_control=k,physical_substeps=0,
        nominal_full_and_recovery_successes=int(both.sum()),fixture_forward_worlds=int(eligible.sum()),snapshot=receipt,
        readback=checks,heldout_used=False,library_path=str(library),library_sha256=sha(library),
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in
                       (DRAFT,GEOMETRY,proof_path,protocol,archive,Path(__file__))},
        scope='Real CPU buffer readback and restoration using saved qpos/qvel fixture. CUDA zero step and continuous physical response are pending.'))
    print('CONTINUOUS_LATERAL_PUSH_WRITER_CPU',int(both.sum()),checks,receipt,flush=True)


if __name__=='__main__':
    main()
