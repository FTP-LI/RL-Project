"""Screen earlier support geometry without integrating or consulting the final holdout."""
from pathlib import Path

from mjlab.scene import Scene
import numpy as np
import torch

from common_scene_contract import compare_arrays, model_arrays
from getup_shared_release import prepare
from record_io import write
from review_getup_supine_support_timing import first_run
from review_getup_support_geometry import geometry_reader
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


PLAN = ROOT/'configs/experiments/getup_supine_early_geometry_screen.json'
OUT = ROOT/'records/reports/getup-supine-early-geometry'


def extract(batch, source, role, spec, measure, compiled):
    record = ROOT/batch['record']
    proof = read(ROOT/source)
    assert sha(ROOT/source)==batch['source_sha256'] and proof['record']==batch['record']
    scene = record/'scene-static-fields.npz'
    with np.load(scene) as saved:
        compare_arrays(compiled,{name:saved[name] for name in saved.files})
    branch = record/f'trace-{role}.npz'
    assert sha(branch)==batch['roles'][role]['trace_sha256']==proof['arms'][role]['trace_sha256']
    examples = batch['roles'][role]['rows']
    worlds = [row['world'] for row in examples]
    start,end = spec['jam_window_controls']
    chosen = np.asarray(spec['controls'])-start
    assert min(chosen)>=2 and max(chosen)<end-start+1
    fields = ('quaternion','qpos','qvel','foot_contact','native_head_leg_contact')
    with np.load(branch) as saved:
        steps = np.rint(saved['time']/batch['criteria']['step_dt']).astype(int)
        np.testing.assert_array_equal(steps[:end-start+1],np.arange(start,end+1))
        trace = {name:saved[name][:end-start+1,worlds] for name in fields}
    z = projected_gravity(trace['quaternion'])[...,2]
    tilt = np.rad2deg(np.arccos(np.clip(-z,-1.,1.)))
    lo,hi = spec['jam_tilt_deg']
    jam = (tilt>lo)&(tilt<=hi)&trace['foot_contact'].all(axis=-1)&trace['native_head_leg_contact']
    rows = []
    for col,case in enumerate(examples):
        contact = first_run(jam[:,col],spec['jam_consecutive_samples'])
        entry = dict(world=case['world'],requested=case['requested'],passed=case['passed'],
                     jam_onset=None if contact is None else int(steps[contact[0]]),features={})
        for k in chosen:
            data = measure(trace['qpos'][k,col],trace['qvel'][k,col])
            data.update(tilt_deg=float(tilt[k,col]),tilt_change_40ms_deg=float(tilt[k,col]-tilt[k-2,col]))
            entry['features'][str(steps[k])] = {name:data[name] for name in spec['features']}
        rows.append(entry)
    return dict(seed=batch['seed'],role=role,trace_sha256=sha(branch),scene_sha256=sha(scene),rows=rows)


def feature_arrays(batch,control,feature,lead):
    rows = batch['rows']
    values = np.array([row['features'][str(control)][feature] for row in rows])
    passes = np.array([row['passed'] for row in rows])
    early = np.array([row['jam_onset'] is not None and row['jam_onset']-control>=lead for row in rows])
    assert np.isfinite(values).all() and passes.any() and not passes.all()
    return values,passes,early


def score(arrays, direction, threshold):
    values,passes,early = arrays
    active = values>=threshold if direction=='at-least' else values<=threshold
    misses = ~passes
    useful, false = int((active & early & misses).sum()), int((active & passes).sum())
    return dict(failed=int(misses.sum()),successful=int(passes.sum()),useful_failures=useful,
                failure_activations=int((active&misses).sum()),possible_useful_failures=int((misses&early).sum()),
                success_activations=false,useful_fraction=useful/int(misses.sum()),success_activation_rate=false/int(passes.sum()))


def rank(rule):
    return (-min(row['useful_fraction'] for row in rule['references']),
            -sum(row['useful_failures'] for row in rule['references']),
            max(row['success_activation_rate'] for row in rule['references']),
            rule['control'],rule['feature'],rule['direction'],rule['threshold'])


def screen(reference, spec):
    best = []
    for control in spec['controls']:
        for feature in spec['features']:
            arrays = [feature_arrays(batch,control,feature,spec['minimum_lead_controls']) for batch in reference]
            thresholds = np.unique(np.concatenate([row[0] for row in arrays]))
            # Two exterior thresholds allow an explicit zero-activation result.
            thresholds = np.r_[np.nextafter(thresholds[0],-np.inf),thresholds,
                               np.nextafter(thresholds[-1],np.inf)]
            for direction in ('at-least','at-most'):
                winner = None
                for threshold in thresholds:
                    scores = [score(row,direction,threshold) for row in arrays]
                    if max(row['success_activation_rate'] for row in scores)>spec['maximum_success_activation_rate']:
                        continue
                    rule = dict(control=control,feature=feature,direction=direction,threshold=float(threshold),references=scores)
                    if winner is None or rank(rule)<rank(winner):
                        winner = rule
                assert winner is not None
                best.append(winner)
    return sorted(best,key=rank)


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-geometry-only'
    assert not any(spec[k] for k in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==spec['stage'] and ROOT/stage['current_protocol']==PLAN
    for key in ('timing_evidence','geometry_reader'):
        file,expected = spec[key]
        assert sha(ROOT/file)==expected
    timing = read(ROOT/spec['timing_evidence'][0])
    plan = read(ROOT/spec['timing_plan'])
    assert timing['protocol_sha256']==sha(ROOT/spec['timing_plan'])
    assert timing['status']=='passed-saved-timing-review' and not any(timing['signal_admission'].values())
    batches = {row['seed']:row for row in timing['batches']}
    role = plan['primary_role']
    first = batches[spec['reference_seeds'][0]]
    protocol = read(ROOT/first['record']/'protocol.json')
    assert sha(ROOT/protocol['policies'][role]['model'])==plan['parent_sha256']
    cfg,_,_ = prepare(protocol['environment'])
    model = Scene(cfg.env.scene,device='cpu').compile()
    measure,compiled = geometry_reader(model),model_arrays(model)

    def saved_batch(seed):
        return extract(batches[seed],plan['archives'][str(seed)][0],role,spec,measure,compiled)

    reference = [saved_batch(seed) for seed in spec['reference_seeds']]
    ranked = screen(reference,spec)
    winner = ranked[0]
    locked = min(row['useful_fraction'] for row in winner['references'])>=spec['minimum_useful_failure_coverage']
    transfer,transfer_score = None,None
    if locked:
        transfer = saved_batch(spec['transfer_seed'])
        arrays = feature_arrays(transfer,winner['control'],winner['feature'],spec['minimum_lead_controls'])
        transfer_score = score(arrays,winner['direction'],winner['threshold'])
    passed = bool(transfer_score and transfer_score['useful_fraction']>=spec['minimum_useful_failure_coverage']
                  and transfer_score['success_activation_rate']<=spec['maximum_success_activation_rate'])
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-saved-geometry-screen',protocol_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),references=reference,ranked_reference_rules=ranked,rule_locked=locked,
          transfer=transfer,transfer_score=transfer_score,signal_admitted=passed,
          candidate_dynamic_execution_authorized=False,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('SUPINE_EARLY_GEOMETRY',winner,'locked',locked,'transfer',transfer_score,flush=True)


if __name__=='__main__':
    main()
