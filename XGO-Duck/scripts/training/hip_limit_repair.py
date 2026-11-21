"""One-factor S2 hip-posture diagnostic; keep the official reward function.

Why: saved self-trained trajectories spend about 99.6% of valid frames with
both hip rolls inward beyond 20 degrees. The previous command-side penalty
did not fix actual posture. Add the same hard-limit proximity term in both
arms, with weight 0 / -2 and margin 0.15 rad. It penalizes BOTH ends of the
selected joints; it neither clamps actions nor locks joints. A 50-update
comparison tests a trend, not convergence or deployable walking.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

JOINTS=('left_hip_roll','right_hip_roll')


def prepare_hip_limit(cfg, arm):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab_microduck.tasks.mdp import joint_pos_limit_proximity
    if arm not in ('control','candidate'):raise ValueError('Unknown hip repair arm')
    if 'hip_limit_proximity' in cfg.rewards:raise ValueError('Reward already exists')
    patched=deepcopy(cfg);weight=0. if arm=='control' else -2.
    patched.rewards['hip_limit_proximity']=RewardTermCfg(
        func=joint_pos_limit_proximity,weight=weight,
        params={'asset_cfg':SceneEntityCfg('robot',joint_names=list(JOINTS),preserve_order=True),'margin':.15})
    return patched,dict(term='hip_limit_proximity',weight=weight,margin_rad=.15,
        joints=list(JOINTS),official_function='mjlab_microduck.tasks.mdp.joint_pos_limit_proximity',
        actual_position_not_requested_target=True,hard_limits=True,
        action_clamping=False,expected_effect='Less sustained hip-limit posture; untested',
        reason='One-factor test of the measured inward posture; retain the reachable-neck parent')


def capture_start(runner, obs, action, packet, record, plan):
    """Record without sampling randomness; candidate must match before stepping."""
    from guarded_training import sha
    from training_watch import read,write
    packet['actions']=action.detach().cpu().numpy().copy()
    packet['common_step']=np.asarray([runner.env.unwrapped.common_step_counter],dtype=np.int64)
    for key,value in zip(('distribution_mean','distribution_std'),runner.alg.actor.output_distribution_params,strict=True):
        packet[key]=value.detach().cpu().numpy().copy()
    path=record/'first-action-inputs.npz'
    np.savez_compressed(path,**packet)
    write(record/'first-action.json',dict(packet_sha256=sha(path),parent_sha256=plan['parent_sha256'],
        protocol_sha256=sha(record/'protocol.json'),joint_names=list(runner.env.unwrapped.scene['robot'].joint_names)))
    if plan['arm']=='candidate':
        control=Path(plan['matched_control_record'])
        metadata=read(control/'first-action.json')
        assert metadata['parent_sha256']==plan['parent_sha256']
        assert sha(control/'protocol.json')==plan['paired_control_sha256']
        assert metadata['packet_sha256']==sha(control/'first-action-inputs.npz')
        with np.load(control/'first-action-inputs.npz',allow_pickle=False) as reference:
            assert set(reference.files)==set(packet)
            for key,value in packet.items():
                assert np.isfinite(value).all(),key
                np.testing.assert_array_equal(value,reference[key],err_msg=key)
        write(record/'matched-start.json',dict(status='passed',comparison='all stored arrays exactly equal',
            keys=sorted(packet),control_packet_sha256=metadata['packet_sha256'],
            candidate_packet_sha256=sha(path),checked_before_first_environment_step=True))
        print('S2_MATCHED_START_PASSED: physical state, observations, RNG and first action identical',flush=True)
