"""Run both checkpoint actors on actual archived roll inputs using CUDA."""

from pathlib import Path

import numpy as np
import torch
from mjlab.utils.torch import configure_torch_backends

from record_io import write
from review_roll_forward_input_stop import ROOT, arrays, read, sha
from roll_stand_walk_sequence import STAND_START, ROLL
from run_continuous_roll_stand_walk import cpu, make_actor, onnx_session, td


def main():
    project = read(ROOT/'project.json')
    task = project['staged_training_request']
    state = project['goal_execution']['roll_checkpoint_selection']
    assert task['current_stage']=='low-friction-roll-checkpoint-selection-preflight'
    assert not (task['pause_requested'] or task['simulation_running'] or task['learning_running'] or task['deployment_authorized_now'])
    draft,receipt = (ROOT/state[key] for key in ('draft','checkpoint_assessment'))
    assert sha(draft)==state['draft_sha256'] and sha(receipt)==state['checkpoint_assessment_sha256']
    plan, prior = read(draft),read(receipt)
    comparison = ROOT/plan['previous_comparison']
    assert sha(comparison)==plan['previous_comparison_sha256'] and not read(comparison)['passed_development_gate']
    sources = read(comparison)['arms']
    config = read(ROOT/'records/evaluations'/sources['seed-1247']['run_id']/'protocol.json')['roll']
    earlier = dict(config,model=prior['exported_model'],model_sha256=prior['export']['model_sha256'])
    assert config['model']==prior['final_model']
    torch.set_num_threads(2)
    configure_torch_backends(allow_tf32=False)
    actors, sessions = {},{}
    for role,meta in (('final-checkpoint',config),('earlier-checkpoint',earlier)):
        assert sha(ROOT/meta['model'])==meta['model_sha256']
        actors[role] = make_actor(meta,'cuda:0')
        sessions[role] = onnx_session(ROOT/meta['model'])
    assert (sessions['earlier-checkpoint'].get_modelmeta().custom_metadata_map
            == sessions['final-checkpoint'].get_modelmeta().custom_metadata_map)
    steps = np.asarray([51,64,70])
    results = {}
    with torch.inference_mode():
        for key,row in sources.items():
            root = ROOT/'records/evaluations'/row['run_id']
            physical,contract = root/'guarded-evaluation.json',root/'protocol.json'
            assert sha(physical)==row['collection_sha256'] and sha(contract)==row['protocol_sha256']
            hashes = read(physical)['prefix_sha256']
            data = root/'shared/commands.npz'
            assert sha(data)==hashes['commands.npz']
            recorded = arrays(data)
            assert (recorded['phase_used'][plan['plant']['shared_snapshot_control']]==STAND_START).all()
            selected = recorded['phase_used'][steps-1]==ROLL
            assert selected.sum(axis=1).min()>=116
            observations = recorded['actor_input'][steps-1][selected]
            saved = recorded['raw_policy_actions'][steps-1][selected]
            inputs = torch.as_tensor(observations,device='cuda:0')
            outputs = {}
            error = {}
            for role,actor in actors.items():
                outputs[role] = cpu(actor(td(inputs)))
                expected = np.concatenate([sessions[role].run(None,{'obs':entry[None]})[0] for entry in observations])
                np.testing.assert_allclose(outputs[role],expected,atol=2e-5,rtol=2e-5)
                error[role] = float(np.abs(outputs[role]-expected).max())
            np.testing.assert_allclose(outputs['final-checkpoint'],saved,atol=2e-5,rtol=2e-5)
            difference = outputs['earlier-checkpoint']-outputs['final-checkpoint']
            results[key] = dict(actual_roll_inputs=len(observations),roll_worlds_by_control=selected.sum(axis=1).tolist(),
                cuda_cpu_max_abs_error_rad=error,final_policy_vs_saved_max_abs_error_rad=float(np.abs(outputs['final-checkpoint']-saved).max()),
                earlier_vs_final_rms_rad=float(np.sqrt(np.mean(difference**2))),trace_sha256=sha(data))
    target = receipt.with_name('cuda-policy-readback.json')
    assert not target.exists()
    write(target,dict(observation_controls=steps.tolist(),scope='Actor inference on actual archived CUDA observations',
        gpu=torch.cuda.get_device_name(0),sources={file.relative_to(ROOT).as_posix():sha(file) for file in (draft,receipt,comparison,Path(__file__))},
        seeds=results,earlier_actor_sha256=earlier['model_sha256'],final_actor_sha256=config['model_sha256'],
        new_scene_resets=0,new_physical_steps=0,new_learning_updates=0,heldout_used=False,hardware_io=False))
    print('ROLL_CHECKPOINT_CUDA_POLICY_READBACK', {key:row['actual_roll_inputs'] for key,row in results.items()},flush=True)


if __name__=='__main__':
    main()
