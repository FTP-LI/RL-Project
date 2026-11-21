"""Check causal pre-bilateral support features across preserved nominal trajectories."""

from itertools import product
from pathlib import Path

import numpy as np

from audit_nominal_supine_early_support import FIELDS, consecutive, first, reconstruct
from audit_nominal_supine_terminal_stance import attitude
from record_io import write
from review_nominal_skill_composition_handoff import read
from roll_stand_walk_sequence import ROLL
from run_nominal_skill_composition import PROJECT, ROOT, allowed, rel, sha
from verify_nominal_skill_composition import read_trace


KEY = 'nominal_supine_pre_bilateral_support'
REPORT = ROOT/'records/reports/nominal-skill-composition/supine-pre-bilateral-support'


def saved_arrays(row):
    path = ROOT/row['samples']
    assert sha(path)==row['samples_sha256']
    with np.load(path,allow_pickle=False) as data:
        return dict(data)


def repeat_arrays(project, design, parent_rule):
    node = project['goal_execution']['nominal_supine_early_support']
    assert node['collection_sha256']==design['repeat_collection_sha256'] and node['status']=='reviewed'
    collection = ROOT/node['collection']
    raw = read(collection,node['collection_sha256'])
    route = read(ROOT/node['protocol'],node['protocol_sha256'])
    assert route['environment']['seed']==1257 and not raw['reserved_seed_used']
    assert route['roll_route']['standing_dwell_controls']==design['original_pre_roll_dwell_controls']
    assert route['roll_route']['roll']['criteria']['tilt_max_deg']==design['original_pre_roll_tilt_deg']
    branch = raw['branches']['original-getup']
    folder = collection.parent
    split = raw['prefix']['controls']
    partial = [chunk for chunk in branch['chunks'] if chunk['first']<=parent_rule['last_control']]
    prefix = read_trace(folder/'shared',raw['prefix']['chunks'],1,split,FIELDS,sha)
    segment = read_trace(folder/'original-getup',partial,split+1,partial[-1]['last'],FIELDS,sha)
    stepped = {k:np.concatenate((prefix[k],segment[k])) for k in FIELDS}
    poses = folder/'release-poses.npz'
    assert sha(poses)==raw['release_sha256']
    with np.load(poses,allow_pickle=False) as data:
        ids = np.flatnonzero(data['poses']=='supine')
    assert len(ids)==raw['actual_release_counts']['supine']
    final = branch['chunks'][-1]
    final_path = folder/'original-getup'/final['file']
    assert sha(final_path)==final['sha256']
    with np.load(final_path,allow_pickle=False) as data:
        assert data['control'][-1]==600
        passes = data['roll_phase'][-1,ids]==ROLL
    a,b = parent_rule['first_control'],parent_rule['last_control']
    data = {k:v[a-1:b,ids] for k,v in stepped.items()}
    geometry = reconstruct(route,data)
    pitch,roll,tilt = np.moveaxis(attitude(data['quaternion']),-1,0)
    clip = parent_rule['alignment']
    dual = data['active']&data['foot_contact'].all(axis=-1)
    phase = dual&(tilt<=clip['maximum_root_tilt_deg'])&(abs(roll)<=clip['maximum_absolute_roll_deg'])
    return dict(worlds=ids,original_roll_admitted=passes,controls=np.arange(a,b+1),
        aligned=first(consecutive(phase,clip['minimum_dual_foot_reads'])),native_feet=data['foot_contact'],
        active=data['active'],head_foot_reconstructed=geometry['head_foot'],
        foot_flags_match=(data['foot_contact']==geometry['reconstructed_feet']).all(axis=-1),
        front_margin_mm=geometry['front_margin_mm'],rear_margin_mm=geometry['rear_margin_mm'],
        tilt_deg=tilt,pitch_deg=pitch,roll_deg=roll)


def metrics(row, control, settings):
    k, = np.flatnonzero(row['controls']==control)
    ready = np.where(row['aligned']>=0,row['aligned']+row['controls'][0],-1)
    current = row['tilt_deg'][k]
    still_early = (ready<0)|(ready>control)
    active = (row['active'][k]&still_early&(current>=settings['phase_min_tilt_deg'])&
              (current<=settings['phase_max_tilt_deg'])&(abs(row['roll_deg'][k])<=settings['phase_max_abs_roll_deg']))
    feet = row['native_feet'][k].any(axis=-1)&row['foot_flags_match'][k]
    return dict(active=active,tilt=current,drop=row['tilt_deg'][k-settings['tilt_history_controls']]-current,
        foot=feet,margin=row['front_margin_mm'][k],passing=row['original_roll_admitted'],lead=ready-control)


def trigger(values, rule):
    match = values['active'].copy()
    if 'tilt' in rule:
        match &= values['tilt']>=rule['tilt']
    if 'drop' in rule:
        match &= values['drop']<=rule['drop']
    if 'margin' in rule:
        match &= values['foot']&(values['margin']<=rule['margin'])
    return match


def measurement(values, rule, gate):
    hits = trigger(values,rule)
    passed = values['passing']
    useful = ~passed&hits&(values['lead']>=gate['minimum_lead_controls'])
    failures,successes = int((~passed).sum()),int(passed.sum())
    coverage = float(useful.sum()/failures)
    exposure = float((hits&passed).sum()/successes)
    return dict(failed=failures,successful=successes,useful_failures=int(useful.sum()),
        failure_activations=int((hits&~passed).sum()),success_activations=int((hits&passed).sum()),
        coverage=coverage,exposure=exposure,lead=[int(x) for x in values['lead'][useful]],
        passes_gate=coverage>=gate['minimum_failure_coverage'] and exposure<=gate['maximum_success_activation_rate'])


def opportunity(row, lookback):
    t = np.arange(len(row['controls']))[:,None]
    until = row['aligned']
    before = (until>=0)&(t<until)&(t>=until-lookback)&row['active']
    flags = row['native_feet'].sum(axis=-1)
    single = (flags==1)&before&row['foot_flags_match']
    either = (flags>=1)&before&row['foot_flags_match']
    inside = (row['front_margin_mm']>=0.)&(row['rear_margin_mm']>=0.)
    head = row['head_foot_reconstructed'].any(axis=-1)
    miss = ~row['original_roll_admitted']
    return dict(failed=int(miss.sum()),was_aligned=int((miss&(until>=0)).sum()),
        any_verified_foot_contact=int((miss&either.any(axis=0)).sum()),
        single_foot_with_sagittal_com_inside=int((miss&(single&inside).any(axis=0)).sum()),
        any_foot_with_sagittal_com_inside=int((miss&(either&inside).any(axis=0)).sum()),
        any_foot_with_com_inside_and_head_contact=int((miss&(either&inside&head).any(axis=0)).sum()))


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    config = ROOT/node['design']
    gate = read(config)
    assert node['status']=='archive-audit-ready' and allowed(project,gate['stage'])
    assert not project['staged_training_request']['simulation_running'] and not REPORT.exists()
    proof = read(ROOT/gate['previous_analysis'],gate['previous_analysis_sha256'])
    parent = project['goal_execution']['nominal_supine_early_support']
    registered = read(ROOT/parent['design'],parent['design_sha256'])
    cases = [saved_arrays(item) for item in proof['cohorts']]
    cases.append(repeat_arrays(project,gate,registered))
    controls = gate['decision_controls']
    features = [{k:metrics(rows,k,gate) for k in controls} for rows in cases]
    lists = (gate['tilt_thresholds_deg'],gate['tilt_drop_thresholds_deg'],gate['front_margin_thresholds_mm'])
    bank = []
    for k in controls:
        for thresholds in product(*([None,*x] for x in lists)):
            rule = {name:v for name,v in zip(('tilt','drop','margin'),thresholds,strict=True) if v is not None}
            if not rule:
                continue
            sample = measurement(features[0][k],rule,gate)
            if sample['exposure']<=gate['maximum_success_activation_rate']:
                bank.append(dict(control=k,rule=rule,training=sample))
    bank.sort(key=lambda r:(-r['training']['useful_failures'],r['training']['success_activations'],
        -sum(r['training']['lead']),len(r['rule']),r['control'],tuple(r['rule'].items())))
    chosen = bank[0]
    k,rule = chosen['control'],chosen['rule']
    validation = [measurement(batch[k],rule,gate) for batch in features[1:]]
    historical = [measurement(batch[122],{'drop':14.15},gate) for batch in features]
    tags = ['seed1256-archive','seed1257-archive','seed1257-same-seed-repeat']
    all_temporal = [opportunity(row,gate['support_window_controls']) for row in cases]
    passed = chosen['training']['passes_gate'] and all(row['passes_gate'] for row in validation)
    REPORT.mkdir()
    repeat = REPORT/'seed-1257-repeat-early.npz'
    np.savez_compressed(repeat,**cases[-1])
    output = REPORT/'analysis.json'
    write(output,dict(protocol_sha256=sha(config),source_script_sha256=sha(Path(__file__)),
        previous_analysis_sha256=gate['previous_analysis_sha256'],repeated_collection_sha256=gate['repeat_collection_sha256'],
        repeat_samples=rel(repeat),repeat_samples_sha256=sha(repeat),
        candidates_within_training_exposure=len(bank),selected_on_seed1256=chosen,
        unchanged_rule_validation=dict(zip(tags[1:],validation,strict=True)),
        prior_fixed_time_drop_rule=dict(zip(tags,historical,strict=True)),
        geometric_opportunities=dict(zip(tags,all_temporal,strict=True)),
        qualifies_observation_trigger=passed,new_physics_steps=0,learning_updates=0,reserved_seeds_used=False,hardware_io=False))
    node.update(status='archive-audit-reviewed',design_sha256=sha(config),analysis=rel(output),analysis_sha256=sha(output),
        qualifies_observation_trigger=passed)
    project['staged_training_request']['current_evidence'] = project['current_evidence'] = rel(output)
    write(PROJECT,project)
    print('SUPINE_PRE_BILATERAL_SCREEN',sha(output),chosen,validation,all_temporal,flush=True)


if __name__=='__main__':
    main()
