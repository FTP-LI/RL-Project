"""Verify recoverable getup components and publish only after the reserved pass."""

from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import numpy as np
import onnxruntime as ort
import torch

import holdout_getup_right_robustness as holdout


ROOT, read, sha, write = holdout.ROOT, holdout.read, holdout.sha, holdout.write
PRE = holdout.OUT.parent / 'delivery-preflight.json'
DELIVERY = ROOT / 'artifacts/delivery'
DEST = DELIVERY / 'getup-simulation.json'
SOURCES = {
    'fallback': ('records/reports/getup-side-to-supine-replay/training/candidate-verification.json', None),
    'supine': ('records/reports/getup-latched-head-corrective-ppo/signature-recovery/training-verification.json', 'control'),
    'right-side': ('records/reports/getup-local-ppo-20261110T011523/throughput-training-verification.json', None),
}


def ref(path):
    return dict(path=holdout.base.rel(path), sha256=sha(path))


def copy_verified(source, target, digest):
    if not target.exists():
        with NamedTemporaryFile(dir=target.parent, prefix=f'.{target.name}.', delete=False) as out:
            temporary = Path(out.name)
            try:
                with source.open('rb') as inp:
                    copyfileobj(inp, out)
                out.close()
                assert sha(temporary) == digest
                if not target.exists():
                    temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    assert sha(target) == digest


def preflight():
    assert not PRE.exists()
    spec = read(holdout.PLAN)
    development = ROOT / spec['holdout_admission']['development_analysis'][0]
    score = read(development)
    assert sha(development) == spec['holdout_admission']['development_analysis'][1] and not score['heldout_used']
    inputs = ROOT / score['record'] / 'first-control-reference-a.npz'
    with np.load(inputs, allow_pickle=False) as saved:
        x = saved['actor_input'][::24].copy()
    assert x.shape == (128, 61) and np.isfinite(x).all()
    policies = dict(fallback=spec['common_fallback'], supine=spec['policies']['reference-a'],
        **{'right-side': spec['policies']['candidate']})
    torch.set_num_threads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    models = {}
    joints = None
    for label, (source, arm) in SOURCES.items():
        path = ROOT / source
        receipt = read(path)
        assert receipt['status'].startswith('passed')
        evidence = receipt['arms'][arm] if arm else receipt
        policy = policies[label]
        assert evidence['model'] == policy['model'] and evidence['model_sha256'] == policy['model_sha256']
        onnx = ROOT / policy['model']
        checkpoint = ROOT / 'records/experiments' / onnx.parent.name / 'model-final.pt'
        assert sha(onnx) == policy['model_sha256'] and sha(checkpoint) == evidence['checkpoint_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['actor_state_dict'] and state['critic_state_dict']
        for field in ('actor_optimizer_state_dict', 'critic_optimizer_state_dict'):
            assert state[field]['state'] and state[field]['param_groups']
        actor = holdout.base.shared.actor_model(policy)
        actor.load_state_dict(state['actor_state_dict'], strict=True)
        session = ort.InferenceSession(str(onnx), sess_options=options, providers=['CPUExecutionProvider'])
        current_joints = session.get_modelmeta().custom_metadata_map['joint_names'].split(',')
        if joints is None:
            joints = current_joints
        assert len(current_joints) == 14 and joints == current_joints
        expected = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in x])
        with torch.inference_mode():
            actual = holdout.base.shared.cpu(actor(holdout.base.shared.td('actor', torch.from_numpy(x))))
        assert actual.shape == expected.shape == (len(x), len(joints))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        models[label] = dict(onnx=ref(onnx), checkpoint=ref(checkpoint), training_evidence=ref(path),
            checkpoint_completed_updates=state['completed_updates'],
            max_checkpoint_onnx_error_rad=float(abs(actual-expected).max()))
    origin = ROOT / 'records/reports/official-reference-refresh/getup-roll-blob-check.json'
    upstream = read(origin)
    official = DELIVERY / 'official/xgoduck_getup.onnx'
    canonical = ROOT / 'official_resources/xgoduck_runtime_arduino/python/xgoduck_getup.onnx'
    model, = (row for row in upstream['models'] if row['model'] == 'getup')
    assert model['same_bytes_by_git_blob'] and sha(canonical) == sha(official) == model['local_sha256']
    assert not torch.cuda.is_initialized()
    write(PRE, dict(status='passed-recoverable-component-preflight', protocol=ref(holdout.PLAN),
        inputs=ref(inputs), development_inputs_per_component=len(x), joint_order=joints, components=models,
        official=dict(file=ref(official), upstream=ref(canonical), upstream_receipt=ref(origin), commit=upstream['commit']),
        learning_updates=0, physical_substeps=0, heldout_used=False))
    print('GETUP_DELIVERY_COMPONENT_PREFLIGHT', len(models), len(x), flush=True)


def publish():
    assert not DEST.exists()
    proof = read(PRE)
    assert proof['status'] == 'passed-recoverable-component-preflight'
    assert proof['protocol'] == ref(holdout.PLAN) and not proof['heldout_used']
    decision_path = holdout.OUT / 'decision.json'
    analysis_path = holdout.OUT / 'analysis.json'
    consumption_path = holdout.OUT / 'consumption.json'
    release_path = holdout.OUT / 'release-verification.json'
    decision, analysis, consumption, release = map(read, (decision_path, analysis_path, consumption_path, release_path))
    assert decision['final_holdout_passed'] and decision['heldout_used'] and not decision['holdout_eligible_for_tuning']
    assert decision['analysis_sha256'] == release['analysis_sha256'] == sha(analysis_path)
    assert consumption['protocol_sha256'] == analysis['protocol_sha256'] == sha(holdout.PLAN)
    assert consumption['heldout_used'] and consumption['status'] == 'completed'
    spec = read(holdout.PLAN)
    assert consumption['actual']['substeps'] == spec['maximum_physics_substeps']
    assert release['counts']['right-side'] >= spec['holdout_admission']['minimum_actual_right_coverage']
    official = proof['official']
    for key in ('file', 'upstream', 'upstream_receipt'):
        assert ref(ROOT / official[key]['path']) == official[key]
    assets = DELIVERY / 'trained/getup-pose-routing'
    assets.mkdir(exist_ok=True)
    components = {}
    for label, component in proof['components'].items():
        copied = {}
        for kind, suffix in (('onnx', 'onnx'), ('checkpoint', 'pt')):
            original = component[kind]
            source = ROOT / original['path']
            assert ref(source) == original
            target = assets / f'{label}.{suffix}'
            copy_verified(source, target, original['sha256'])
            copied[kind] = dict(ref(target), source=original['path'])
        components[label] = dict(component, files=copied)
    e = spec['environment']
    c = analysis['arms']['candidate']['combined']
    write(DEST, dict(status='verified-four-pose-getup-simulation-components',
        scope='fixed 11-second four-pose getup on the shared simulation plant',
        hardware_io=False, continuous_skill_handoff_verified=False, single_onnx_contains_routing=False,
        official=official, components=components, joint_order=proof['joint_order'],
        control_contract=dict(control_hz=50, input_size=61, output_size=14, projected_gravity_slice=[3, 6],
            shared_release_control=spec['prefix_controls'], alignment=spec['dual_route']['alignment'],
            simulation_home=e['simulation_home'], model_home=e['model_home'], joint_limits_rad=e['joint_limits'],
            late_target_slew_rad_s=e['handoff_settings']['rate_rad_s'],
            late_target_slew_start_seconds=e['handoff_settings']['start_seconds'],
            decision_order=['right-side', 'supine', 'fallback'], fixed_during_episode=True,
            implementation=[ref(ROOT / path) for path in ('scripts/training/getup_pose_router.py',
                'scripts/training/getup_supine_release_actor.py', 'scripts/validation/evaluate_getup_dual_pose_route.py')]),
        heldout=dict(protocol=ref(holdout.PLAN), decision=ref(decision_path), score=ref(analysis_path),
            consumption=ref(consumption_path), release=ref(release_path), overall=c['overall'], actual=c['actual'],
            requested=c['requested']),
        component_preflight=ref(PRE), iteration_record=holdout.base.rel(DELIVERY / 'iteration_record.md')))
    print('GETUP_DELIVERY_PUBLISHED', sha(DEST), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'publish'))
    (preflight if parser.parse_args().mode == 'preflight' else publish)()
