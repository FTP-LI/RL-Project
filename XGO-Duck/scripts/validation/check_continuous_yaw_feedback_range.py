"""Screen small feedback-cap increases on actual post-roll forward inputs."""
from pathlib import Path

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from common_walk_startup import make_actor, td
from record_io import write
from walking_actuator_yaw_feedback import ActuatorYawFeedback, projected_feedback_target


OUT = ROOT/'records/reports/continuous-roll-projected-entry'
CAPS = (.12,.15,.18)


def main():
    evidence = OUT/'forward-yaw-localization.json'
    localization = read(evidence)
    proof_path = OUT/'analysis.json'
    proof = read(proof_path)
    assert localization['status']=='passed-saved-forward-yaw-localization' and not localization['heldout_used']
    assert sha(proof_path)==localization['sources'][proof_path.relative_to(ROOT).as_posix()]
    record = ROOT/proof['record']
    plan,source = record/'protocol.json',record/'commands.npz'
    p = read(plan)
    assert sha(plan)==proof['protocol_sha256'] and sha(source)==proof['source_sha256'][source.relative_to(ROOT).as_posix()]
    assert p['feedback']['maximum_yaw_input_rad_s']==CAPS[0]
    worlds,indices,failures = [],[],[]
    for row in proof['forward']:
        if row['entered_forward']:
            start = row['forward_start']
            local = np.arange(start+9,start+p['forward_controls'],10)
            worlds.extend([row['world']]*len(local))
            indices.extend(local.tolist())
            failures.extend([not row['success']]*len(local))
    indices,worlds,failures = np.asarray(indices),np.asarray(worlds),np.asarray(failures)
    assert len(worlds)==12600 and int(failures.sum())==1500
    with np.load(source) as data:
        observed = data['observed']
        phase_input = data['actor_input'][indices,worlds]
        raw_target = data['requested_targets'][indices,worlds]
        prior_yaw = data['feedback_yaw'][indices,worlds]
        prior_alternative = data['alternate_action'][indices,worlds]
        prior_targets = data['applied_targets'][indices,worlds]
    hip = [20+p['walking']['joint_names'].index(f'{side}_hip_pitch') for side in ('left','right')]
    current = np.stack((observed[...,2],observed[...,hip].sum(axis=-1)),axis=-1)
    features = np.concatenate((current[indices,worlds],current[indices,worlds]-current[indices-1,worlds]),axis=-1)
    screen = ROOT/p['feedback_screen']
    screen_proof = read(screen)
    cap = p['feedback']['maximum_joint_correction_rad']
    torch.set_num_threads(4)
    actor = make_actor(p['walking'],'cpu')
    limits = torch.tensor(p['walking']['joint_limits'],dtype=torch.float32)
    home = torch.tensor(p['walking']['model_home'],dtype=torch.float32)
    base = torch.clamp(torch.from_numpy(raw_target),min=limits[:,0],max=limits[:,1])
    results = {}
    saturated = np.abs(prior_yaw)>=CAPS[0]-1e-6
    for value in CAPS:
        yaw = ActuatorYawFeedback(screen_proof,value).commands(features).astype(np.float32)
        obs = phase_input.copy()
        obs[:,50] = yaw
        with torch.no_grad():
            actions = torch.cat([actor(td(part)) for part in torch.from_numpy(obs).split(4096)])
            actual,scale = projected_feedback_target(base,actions,home,limits,cap)
        if value==CAPS[0]:
            np.testing.assert_allclose(yaw,prior_yaw,atol=2e-7,rtol=0)
            np.testing.assert_allclose(actions.numpy(),prior_alternative,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(actual.numpy(),prior_targets,atol=2e-5,rtol=2e-5)
        assert torch.all(actual>=limits[:,0]-1e-7) and torch.all(actual<=limits[:,1]+1e-7)
        assert (actual-base).abs().max()<=cap+1e-7
        delta = np.abs(actual.numpy()-prior_targets).max(axis=-1)
        results[str(value)] = dict(
            fraction_joint_cap_limited=float((scale<1.).float().mean()),
            max_target_change_vs_original_rad=float(delta.max()),
            median_change_when_original_saturated_rad=float(np.median(delta[saturated])),
            median_change_failed_worlds_rad=float(np.median(delta[failures])),
            maximum_corrected_joint_delta_rad=float((actual-base).abs().max()))
    output = OUT/'forward-yaw-feedback-range.json'
    assert not output.exists() and not torch.cuda.is_initialized()
    write(output,dict(status='passed-static-feedback-range',samples=len(indices),failed_world_samples=int(failures.sum()),
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (evidence,proof_path,plan,source,screen,Path(__file__))},
        results=results,physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Saved observations only. Does not predict actual yaw reduction or select a deployment setting.'))
    print('CONTINUOUS_YAW_RANGE',results,flush=True)


if __name__=='__main__':
    main()
