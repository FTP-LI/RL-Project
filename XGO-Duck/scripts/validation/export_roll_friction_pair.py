"""Export final, read-back roll actors with the original device-facing interface."""

import argparse
from pathlib import Path
import tempfile

import run_roll_friction_pair as pair


def export(plan, arm, checkpoint, destination):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from verify_roll_exposure_parent import finite
    from walk_backward_export import export_walking_actor

    root, require = pair.ROOT, pair.require
    recipe, checks = plan['recipe'], plan['training_checks']
    parent = torch.load(root / recipe['parent_checkpoint'], map_location='cpu', weights_only=True)
    trained = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actor_state = trained['actor_state_dict']
    require(finite(trained) and trained['iter']+1 == arm['updates'] and all(
        torch.equal(value, parent['actor_state_dict'][key])
        for key,value in actor_state.items() if key.startswith('obs_normalizer.')), 'Checkpoint or normalization is incompatible')
    require(any(not torch.equal(value, parent['actor_state_dict'][key])
                for key,value in actor_state.items() if key.startswith('mlp.')), 'Export must contain actual actor updates')
    actor = MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]), {'actor':['actor']}, 'actor', 14,
                   hidden_dims=(512,256,128), activation='elu', obs_normalization=True,
                   distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(actor_state, strict=True)
    actor.obs_normalizer.until = 0
    actor.eval()
    trace = root / checks['drift_trace']
    require(pair.sha(trace)==checks['drift_trace_sha256'], 'Fixed actor inputs changed')
    with np.load(trace, allow_pickle=False) as raw:
        observations = raw['observations'][checks['drift_frames']].reshape(-1,61).copy()
    original = root / recipe['official_onnx']
    # The official roll and walk models use the same normalized 61→14
    # graph. Reuse its verified tensor-only exporter and compare metadata.
    original_graph = onnx.load(original)
    result = export_walking_actor(actor, original, recipe['official_sha256'], destination, observations)
    new_graph = onnx.load(destination)
    saved_meta = [(item.key,item.value) for item in new_graph.metadata_props]
    require(saved_meta == [(item.key,item.value) for item in original_graph.metadata_props]
        and len(dict(saved_meta)['joint_names'].split(',')) == 14
        and not torch.cuda.is_initialized(), 'Export changed joint metadata or used the training GPU')
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=opts, providers=['CPUExecutionProvider'])
    require(session.get_inputs()[0].name == 'obs' and session.get_inputs()[0].shape == [1,61]
        and session.get_outputs()[0].shape == [1,14], 'Written ONNX interface changed')
    return dict(result, original_sha256=recipe['official_sha256'], checkpoint_sha256=pair.sha(checkpoint),
                input_width=61, output_width=14, metadata_preserved=True, hardware_io=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-check', action='store_true')
    args = parser.parse_args()
    root, read, sha = pair.ROOT, pair.read, pair.sha
    if args.capacity_check:
        path = pair.live.REPORT / 'phase-1-pass-1-e3584/independent.json'
        evidence, plan = read(path), read(pair.live.PLAN)
        arm = next(item for item in plan['profiles'].values() if item['run_id']==evidence['run_id'])
        historical = 'records/evaluations/duck_roll_action-memory_control_s1003_20261103T004639/forward-roll-trace.npz'
        plan['training_checks'] = dict(drift_trace=historical, drift_frames=[1,16,31,46,61,76,151,301,450],
            drift_trace_sha256=read(root/'configs/experiments/roll_top_contact/control.json')['drift_probe_trace_sha256'])
        checkpoint = root / 'records/experiments' / arm['run_id'] / 'profile-final.pt'
        pair.require(sha(checkpoint)==evidence['checkpoint_sha256'], 'Capacity example changed')
        with tempfile.TemporaryDirectory(prefix='xgo-roll-export-') as directory:
            print('ROLL_ONNX_EXPORT_CHECK', export(plan, arm, checkpoint, Path(directory)/'policy.onnx'))
        return
    project, plan = pair.checked({'evaluation-pending'})
    state = project['goal_execution']['roll_friction_paired']
    for role, arm in plan['arms'].items():
        report = pair.REPORT / role
        proof = read(root / state['reviews'][role])
        checkpoint = root / 'records/experiments' / arm['run_id'] / 'profile-final.pt'
        pair.require(proof['status']=='passed-paired-training-readback' and proof['run_id']==arm['run_id']
                    and proof['checkpoint_sha256']==sha(checkpoint), 'Paired checkpoint is not verified')
        folder = root / 'artifacts' / arm['run_id']
        folder.mkdir()
        temporary = folder / 'policy.partial.onnx'
        result = export(plan, arm, checkpoint, temporary)
        temporary.replace(folder / 'policy.onnx')
        pair.put(report/'export.json', dict(result, role=role, run_id=arm['run_id'],
            model=pair.rel(folder/'policy.onnx'), checkpoint=pair.rel(checkpoint), status='exported-for-action-evaluation'))
        print('ROLL_PAIRED_ONNX_EXPORTED', role, result['model_sha256'], flush=True)


if __name__=='__main__':
    main()
