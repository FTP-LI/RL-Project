"""Pair similar releases and compare tilt around first sustained foot support."""

from pathlib import Path

import numpy as np

from review_nominal_skill_composition_handoff import read
from review_nominal_supine_early_load import confirmed, first_event
from roll_metrics import rotation
from run_nominal_supine_early_load import KEY as SOURCE_KEY, PROJECT, REPORT, ROOT, parent, rel, sha, write


KEY = 'nominal_supine_support_rotation_alignment'


def quantiles(values):
    return dict(n=len(values), median=float(np.median(values)) if len(values) else None,
                quartiles=np.quantile(values,(.25,.75)).tolist() if len(values) else None)


def pair(gravity, rate, admitted, spec):
    bad, good = np.flatnonzero(~admitted), np.flatnonzero(admitted)
    assert len(good) >= len(bad) > 0
    direction = (gravity[bad,None,:]-gravity[None,good,:])/spec['gravity_vector_matching_scale']
    speed = (rate[bad,None]-rate[None,good])/spec['upright_speed_matching_scale_deg_s']
    cost = np.sum(direction**2,axis=2) + speed**2
    bad_indices, good_indices = [], []
    for _ in bad:
        i,j = np.unravel_index(np.argmin(cost),cost.shape)
        bad_indices.append(bad[i])
        good_indices.append(good[j])
        cost[i,:] = cost[:,j] = np.inf
    return np.array(bad_indices), np.array(good_indices)


def review(entry, spec, design):
    collection = read(ROOT/entry['collection'],entry['collection_sha256'])
    folder = (ROOT/entry['collection']).parent
    source = ROOT/entry['output_clock']
    assert sha(source) == entry['clock_sha256'] and entry['seed'] not in spec['reserved_seeds']
    poses = folder/'release-poses.npz'
    assert sha(poses) == collection['release_sha256']
    with np.load(poses,allow_pickle=False) as data:
        worlds = np.flatnonzero(data['poses']=='supine')
    with np.load(source,allow_pickle=False) as data:
        steps = data['substep']
        load = data['normal_force_n'][:,worlds]/entry['nominal_weight_n']
        gravity = rotation(data['after'][:,worlds])[...,2,:]
    tilt = np.rad2deg(np.arccos(np.clip(gravity[...,2],-1.,1.)))
    missed = entry['cohorts']['missed']['individual_failed_worlds']
    admitted = ~np.isin(worlds,missed)
    assert int(admitted.sum()) == entry['passes_by_release']['supine']['admission']
    release_step = design['release_control']*5
    release_index = int(np.searchsorted(steps,release_step))
    assert steps[release_index] == release_step
    dt = design['physics_period_s']
    lookback = round(spec['release_speed_lookback_ms']/(dt*1000))
    release_rate = (tilt[release_index-lookback]-tilt[release_index])/(lookback*dt)
    bad, good = pair(gravity[release_index],release_rate,admitted,spec)
    foot_sum = load[:,:,2:].sum(axis=2)
    limit = design['any_foot_load_fraction']
    bilateral = (load[:,:,2:]>limit).all(axis=2) & (foot_sum>=design['bilateral_sum_load_fraction'])
    bilateral &= (steps>release_step)[:,None]
    first = first_event(confirmed(bilateral,design['minimum_consecutive_bilateral_solves']),steps)
    assert (first>=0).all()
    head = load[:,:,1] > limit
    rows = []
    by_offset = {}
    for offset in spec['bilateral_confirmation_offsets_ms']:
        bd = first[bad] + round(offset/(dt*1000)) - steps[0]
        gd = first[good] + round(offset/(dt*1000)) - steps[0]
        valid = (bd>=0)&(gd>=0)&(bd<len(steps))&(gd<len(steps))
        bc,gc = (bd[valid],bad[valid]),(gd[valid],good[valid])
        rows.append(dict(offset_ms=offset, pairs=int(valid.sum()),
                        missed_minus_admitted_tilt_deg=quantiles(tilt[bc]-tilt[gc]),
                        missed_minus_admitted_foot_load_body_weights=quantiles(foot_sum[bc]-foot_sum[gc]),
                        head_loaded_missed=int(head[bc].sum()), head_loaded_admitted=int(head[gc].sum())))
        by_offset[offset] = dict(valid=valid, bad=bd, good=gd)
    before,after = [by_offset[k] for k in spec['recovery_delta_measurement']]
    both = before['valid'] & after['valid']
    b,g = bad[both],good[both]
    impulse = ((tilt[before['bad'][both],b]-tilt[after['bad'][both],b])
               -(tilt[before['good'][both],g]-tilt[after['good'][both],g]))
    delta_gravity = (gravity[release_index,bad] * gravity[release_index,good]).sum(axis=1)
    angle_diff = np.rad2deg(np.arccos(np.clip(delta_gravity,-1.,1.)))
    events = (first[bad]-first[good])*dt*1000
    return dict(seed=entry['seed'], clock_sha256=entry['clock_sha256'], distinct_pairs=len(bad),
                release_gravity_angle_difference_deg=quantiles(angle_diff),
                release_upright_speed_abs_difference_deg_s=quantiles(abs(release_rate[bad]-release_rate[good])),
                bilateral_confirmation_delay_ms=quantiles(events), aligned=rows,
                tilt_drop_missed_minus_admitted_deg=quantiles(impulse),
                worlds=[dict(missed_world=int(worlds[x]),admitted_world=int(worlds[y]),
                             release_gravity_difference_deg=float(a),confirmation_delay_ms=float(d))
                        for x,y,a,d in zip(bad,good,angle_diff,events,strict=True)])


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    draft = ROOT/node['design']
    spec = read(draft)
    assert node['status']=='archive-audit-ready' and parent.allowed(project,spec['stage'])
    assert not project['staged_training_request']['simulation_running']
    recorded = project['goal_execution'][SOURCE_KEY]
    analysis = read(ROOT/recorded['analysis'],spec['source_analysis_sha256'])
    design = read(ROOT/recorded['design'],recorded['design_sha256'])
    entries = analysis['seed_results']
    assert [row['seed'] for row in entries] == spec['development_seeds']
    results = [review(entry,spec,design) for entry in entries]
    output = REPORT/'rotation-aligned-to-load.json'
    assert not output.exists()
    write(output,dict(design_sha256=sha(draft),script_sha256=sha(Path(__file__)),
        source_analysis_sha256=spec['source_analysis_sha256'],seed_results=results,
        new_physical_steps=0,learning_updates=0,hardware_io=False))
    node.update(status='reviewed',design_sha256=sha(draft),analysis=rel(output),analysis_sha256=sha(output))
    task = project['staged_training_request']
    task.update(current_evidence=rel(output),current_protocol_execution='nominal-supine-support-rotation-alignment-reviewed')
    project.update(current_evidence=rel(output),current_protocol_execution=task['current_protocol_execution'])
    write(PROJECT,project)
    for row in results:
        print('SUPINE_LOAD_ROTATION_ALIGNED',row['seed'],row['distinct_pairs'],row['bilateral_confirmation_delay_ms'],
              [(item['offset_ms'],item['missed_minus_admitted_tilt_deg']['median']) for item in row['aligned']],
              row['tilt_drop_missed_minus_admitted_deg'],flush=True)
    print('SUPINE_LOAD_ROTATION_COMPLETE',sha(output),flush=True)


if __name__ == '__main__':
    main()
