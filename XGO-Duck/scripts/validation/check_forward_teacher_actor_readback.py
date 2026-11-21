"""Check the archived teacher-constrained walk export on saved forward inputs."""

import numpy as np
import onnxruntime as ort

from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import MASTER, PLANS, ROOT, P, put, read, rel, require, sha


TEACHER = ROOT/'artifacts/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414'
OUTPUT = ROOT/'records/reports/forward-correction-limit-probe/offline-teacher-actor-readback.json'


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runner = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs, outputs = runner.get_inputs(),runner.get_outputs()
    require(len(inputs)==len(outputs)==1 and inputs[0].name=='obs' and inputs[0].shape[-1]==61
        and outputs[0].shape[-1]==14, 'Expected a 61-input and 14-action walking model')
    return runner


def geometry(runner):
    meta = runner.get_modelmeta().custom_metadata_map
    return dict(joints=meta['joint_names'].split(','),home=[float(x) for x in meta['default_joint_pos'].split(',')],
                scale=float(meta['action_scale']))


def infer(runner, inputs):
    return np.concatenate([runner.run(None,{'obs':row[None]})[0] for row in inputs])


def apply(raw, alternate, home, limits, cap):
    lower,upper = limits.T
    target = np.clip(raw+home,lower,upper)
    correction = np.clip(alternate+home,lower,upper)-target
    coefficient = np.minimum(1.,cap/np.maximum(np.abs(correction).max(axis=1),1e-8))
    return target+coefficient[:,None]*correction


def main():
    project,master = read(P),read(MASTER)
    track = project['goal_execution']['roll_friction_stand_forward']
    probe = project['goal_execution']['forward_correction_limit_probe']
    require(track['status']==probe['status']=='reviewed' and sha(MASTER)==track['master_sha256']
        and not any(project['staged_training_request'][name] for name in ('simulation_running','learning_running','deployment_authorized_now')),
        'Finish the direct walking and limit trials first')
    require(sha(ROOT/probe['result'])==probe['result_sha256'] and not read(ROOT/probe['result'])['passed_original_gate'],
            'The correction-limit record changed')
    manifest = read(TEACHER/'manifest.json')
    model = TEACHER/'policy.onnx'
    require(sha(model)==manifest['model_sha256'] and manifest['teacher_anchor'], 'Archived alternate actor changed')
    alternate_actor = session(model)
    saved = []
    for key in master['order']:
        plan_path = PLANS/(key+'.json')
        plan = read(plan_path)
        mark = track['scored'][key]
        receipt = ROOT/mark['path']
        proof = read(receipt)
        require(sha(plan_path)==master['protocols'][key] and sha(receipt)==mark['sha256']
            and proof['run_id']==plan['run_id'] and proof['forward_entries']==128, 'Saved forward case changed')
        baseline = ROOT/plan['walking']['model']
        require(sha(baseline)==plan['walking']['model_sha256'], 'Official walking actor changed')
        official = session(baseline)
        geometry_original = geometry(official)
        require(geometry_original==geometry(alternate_actor) and geometry_original['joints']==plan['walking']['joint_names']
                and geometry_original['scale']==1., 'Candidate changes the physical joint contract')
        folder = ROOT/'records/evaluations'/plan['run_id']
        path = folder/'commands.npz'
        recording = read(folder/'guarded-evaluation.json')
        require(sha(path)==recording['command_sha256'], 'Saved forward commands changed')
        indices = plan['initial_stand_controls']+np.linspace(0,plan['forward_controls']-1,12,dtype=np.int32)
        with np.load(path,allow_pickle=False) as data:
            require((data['phase_used'][indices]==FORWARD).all(), 'A sampled command is outside forward phase')
            obs = data['actor_input'][indices].reshape(-1,61)
            control = data['raw_policy_actions'][indices].reshape(-1,14)
            addon = data['alternate_action'][indices].reshape(-1,14)
            actual = data['applied_targets'][indices].reshape(-1,14)
            yaw = data['feedback_yaw'][indices].reshape(-1)
        changed_inputs = obs.copy()
        changed_inputs[:,50] = yaw
        old = infer(official,obs)
        old_yaw = infer(official,changed_inputs)
        np.testing.assert_allclose(old,control,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(old_yaw,addon,atol=2e-5,rtol=2e-5)
        home = np.asarray(geometry_original['home'],np.float32)
        limits = np.asarray(plan['walking']['joint_limits'],np.float32)
        cap = plan['feedback']['maximum_joint_correction_rad']
        np.testing.assert_allclose(apply(old,old_yaw,home,limits,cap),actual,atol=2e-5,rtol=2e-5)
        proposed = apply(infer(alternate_actor,obs),infer(alternate_actor,changed_inputs),home,limits,cap)
        require(np.isfinite(proposed).all() and np.all(proposed>=limits[:,0]-2e-7) and np.all(proposed<=limits[:,1]+2e-7),
                'Candidate joint targets leave the projected envelope')
        delta = np.abs(proposed-actual)
        row = dict(case=key,seed=plan['seed'],friction=plan['parameter_factors']['foot_friction'],
            recorded_control_indices=indices.tolist(),sampled_inputs=len(obs),command_sha256=recording['command_sha256'],
            official_maximum_replay_error_rad=float(np.max(np.abs(old-control))),
            target_mean_absolute_change_rad=float(delta.mean()),target_maximum_change_rad=float(delta.max()))
        saved.append(row)
        print('TEACHER_FORWARD_READBACK',row,flush=True)
    result = dict(cases=saved,archived_candidate=rel(model),candidate_sha256=manifest['model_sha256'],
                  candidate_checkpoint_sha256=manifest['checkpoint_sha256'],input_width=61,actions=14,
                  matched_joint_metadata=True,hardware_io=False,learning_updates=0,new_physics_substeps=0,
                  scope='Same archived input samples; no closed-loop gait effect measured')
    if OUTPUT.exists():
        require(read(OUTPUT)==result,'Repeated archived actor readback differs')
    else:
        put(OUTPUT,result)


if __name__=='__main__':
    main()
