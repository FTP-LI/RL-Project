"""Qualify the exact horizontal write on CUDA before permitting any rollout."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from check_continuous_lateral_push_geometry import DRAFT, DEST as GEOMETRY
from check_continuous_lateral_push_writer import DEST as CPU, apply_lateral_push
from roll_endpoint_readback_capture import EndpointReadback
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import FORWARD
from run_continuous_roll_stand_walk import STARTUP, actors, configuration, cpu, install_guard


OUT = ROOT/'records/reports/continuous-forward-lateral-push'
PARENT = ROOT/'configs/experiments/continuous_forward_yaw_replication.json'
SEED = 1097
WARP_LIMIT = 268435456


def main():
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    p,draft,proof = read(PARENT),read(DRAFT),read(CPU)
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==draft['stage'] and not project['pause_requested']
    assert not project['current_protocol_frozen'] and not draft['heldout_used'] and not draft['hardware_io']
    assert proof['status']=='passed-actual-cpu-writer-and-restore' and proof['physical_substeps']==0
    assert sha(Path(proof['library_path']))==proof['library_sha256']
    for name,expected in proof['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    geometry = read(GEOMETRY)
    result_path = ROOT/draft['required_independent_result']
    previous = read(result_path)
    assert previous[draft['required_independent_gate']] and sha(PARENT)==previous['protocol_sha256']
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    archive = ROOT/previous['record']/'trace.npz'
    k = geometry['fork_control']
    with np.load(archive) as values:
        qpos,qvel = (values[key][k].copy() for key in ('qpos','qvel'))
        eligible = values['active'][k]&(values['phase_next'][k]==FORWARD)
    assert int(eligible.sum())==proof['fixture_forward_worlds']==125
    config = dict(p,seed=SEED,roll=dict(p['roll'],seed=SEED))
    folder = OUT/'zero'
    folder.mkdir(parents=True,exist_ok=False)
    source_files = (PARENT,DRAFT,CPU,GEOMETRY,result_path,archive,Path(__file__))
    sources = {path.relative_to(ROOT).as_posix():sha(path) for path in source_files}
    write(folder/'fixture-protocol.json',dict(status='frozen-before-zero-step',seed=SEED,archive_control=k,
        expected_worlds=int(eligible.sum()),physical_substep_budget=0,sources=sources,
        roles=draft['proposed_roles_body_horizontal_lateral_m_s']))
    configure_torch_backends(allow_tf32=False)
    cfg = configuration(config)
    dump_yaml(folder/'env.yaml',asdict(cfg))
    guard = install_guard(config['roll'],folder,STARTUP,'rollout')
    env = guard(cfg=cfg,device='cuda:0',render_mode=None)
    try:
        env.seed(SEED)
        obs,_ = env.reset()
        actors(env,obs,config,folder)
        endpoints = EndpointReadback(env,folder)
        endpoints.capture()
        assert not endpoints.first_mismatches
        endpoints.save(1)
        home = env.action_manager.get_term('joint_pos')._processed_actions
        memory = SimpleNamespace(previous=torch.zeros_like(home),history=torch.zeros_like(home),
                                 previous_features=np.zeros((env.num_envs,2),dtype=np.float32))
        initial = RollEntrySnapshot(env,memory,WARP_LIMIT)
        initial_receipt = initial.verify()
        env.sim.data.qpos.copy_(torch.from_numpy(qpos))
        env.sim.data.qvel.copy_(torch.from_numpy(qvel))
        fixture = RollEntrySnapshot(env,memory,WARP_LIMIT)
        fixture_receipt = fixture.verify()
        coordinates = env.scene['robot'].data.indexing.free_joint_v_adr[:2].tolist()
        untouched = np.ones(qvel.shape,dtype=bool)
        untouched[np.ix_(eligible,coordinates)] = False
        cases = {}
        for role,speed in draft['proposed_roles_body_horizontal_lateral_m_s'].items():
            seen = apply_lateral_push(env,eligible,speed)
            np.testing.assert_array_equal(cpu(env.sim.data.qpos),qpos)
            np.testing.assert_array_equal(cpu(env.sim.data.qvel)[untouched],qvel[untouched])
            after = fixture.live_signatures()
            changed = [name for name in fixture.expected if after[name]!=fixture.expected[name]]
            assert changed==(['warp.qvel'] if speed else []),changed
            np.savez_compressed(folder/f'{role}-velocity.npz',**seen)
            assert fixture.restore()==fixture_receipt
            cases[role] = dict(changed_fields=changed,affected_worlds=len(seen['world_ids']) if speed else 0,
                maximum_readback_error_m_s=float(np.abs(seen['target_world_xy']-seen['measured_world_xy']).max()),
                archive_sha256=sha(folder/f'{role}-velocity.npz'))
        assert initial.restore()==initial_receipt
        assert env.common_step_counter==env._sim_step_counter==env.guard_steps==0 and len(env.guard_resets)==1
    finally:
        env.close()
    write(OUT/'cuda-zero.json',dict(status='passed-actual-cuda-zero-step',seed=SEED,physical_substeps=0,
        fixture_source_control=k,fixture_forward_worlds=int(eligible.sum()),cases=cases,fixture=fixture_receipt,
        initial=initial_receipt,source_sha256=sources,initial_archives={name:sha(folder/name) for name in (
            'startup.npz','endpoint-contact.npz','runtime-contract.json','fixture-protocol.json')},
        library_sha256=proof['library_sha256'],heldout_used=False,hardware_io=False))
    print('CONTINUOUS_LATERAL_PUSH_CUDA_ZERO',cases,fixture_receipt,flush=True)


if __name__=='__main__':
    main()
