"""Read actual 4 ms support and annotate the six verified phase-control videos."""

import argparse
from unittest.mock import patch

import numpy as np

import review_roll_neck_support_response as visual
from record_io import write
from review_roll_support_target_authority import load
from roll_metrics import progress
from run_roll_observed_neck_support import CANDIDATE, KEY, LABELS, MASTER, PLANS, REFERENCES, REPORT, ROLES
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


def phase_measurements(plan, prefix, motion, clock, meta):
    dt = .004
    start, end = (plan['force_capture'][key] for key in ('first_control','last_control'))
    shared = plan['checkpoint_selection']['plant']['shared_snapshot_control']
    np.testing.assert_array_equal(clock['substep'],np.arange(start*5+1,end*5+1))
    np.testing.assert_array_equal(clock['before'][::5],motion['qpos'][start-shared:end-shared,:,3:7])
    np.testing.assert_array_equal(clock['after'][4::5],motion['qpos'][start-shared+1:end-shared+1,:,3:7])
    np.testing.assert_array_equal(clock['after'][:-1],clock['before'][1:])
    roll_start = plan['initial_stand_controls']-shared
    initial = motion['qpos'][roll_start:start-shared+1,:,3:7]
    q = np.concatenate((initial,clock['after']),axis=0)
    angle = np.rad2deg(progress(dict(quaternion=q,ground_contact=np.ones(q.shape[:2],bool)))[0][len(initial)-1:])
    forward = np.maximum(np.minimum(angle[1:],270.)-np.maximum(angle[:-1],180.),0.)
    head = clock['normal_force_n'][...,1]
    require(meta['complete'] and meta['labels']==['ground','head','left_foot','right_foot'] and len(prefix['qpos'])==shared+1
            and head.shape==forward.shape and np.isfinite(head).all() and np.all(head>=0.)
            and meta['force_threshold_n']==meta['nominal_weight_n']*.01,'Measured head loads differ from the frozen clock')
    loaded = head>meta['force_threshold_n']
    endpoint = np.repeat(motion['ground_contact'][start-shared+1:end-shared+1],5,axis=0)
    return dict(head_loaded_duration_ms=(loaded & (forward>0.)).sum(axis=0)*dt*1000.,
                head_normal_impulse_ns=np.where(forward>0.,head,0.).sum(axis=0)*dt,
                head_loaded_forward_angle_deg=(forward*loaded).sum(axis=0),
                endpoint_excluded_forward_angle_deg=(forward*~endpoint).sum(axis=0),
                maximum_observed_forward_angle_deg=angle.max(axis=0))


def statistics(values, mask):
    sample = values[mask]
    if not len(sample):
        return dict(worlds=0,median=None,q25=None,q75=None)
    q25,median,q75 = np.quantile(sample,[.25,.50,.75])
    return dict(median=float(median),q25=float(q25),q75=float(q75),worlds=len(sample))


def summarize(result, hashes):
    frames, results = {},{}
    for case,row in result['arms'].items():
        case_file = PLANS/(case+'.json')
        plan = load(case_file,row['protocol_sha256'],hashes)
        folder = ROOT/'records/evaluations'/row['run_id']
        stamp = load(folder/'guarded-evaluation.json',row['collection_sha256'],hashes)
        keys = ('qpos','ground_contact','head_contact','foot_contact','phase_next')
        prefix = load(folder/'shared/trace.npz',stamp['prefix_sha256']['trace.npz'],hashes,arrays=keys)
        scores, motions = {},{}
        for role in ROLES:
            branch = folder/'clips'/role
            motion = load(branch/'trace.npz',stamp['branches'][role]['trace_sha256']['trace.npz'],hashes,arrays=keys)
            marker = branch/'contact-clock.json'
            meta = load(marker,sha(marker),hashes)
            clock = load(branch/'contact-clock.npz',meta['sha256'],hashes,arrays=('substep','before','after','normal_force_n'))
            scores[role] = phase_measurements(plan,prefix,motion,clock,meta)
            motions[role] = {key:np.concatenate((prefix[key],motion[key][1:])) for key in keys if key!='qpos'}
        a,b = REFERENCES
        common = np.asarray(row['arms'][a]['roll_pass']) & np.asarray(row['arms'][b]['roll_pass'])
        lost = common & ~np.asarray(row['arms'][CANDIDATE]['roll_pass'])
        masks = dict(all_worlds=np.ones(plan['worlds'],bool),both_references_pass=common,
                     both_references_pass_candidate_fails=lost)
        metrics = tuple(scores[a])[:-1]
        cohorts = {name:{
            'measurements':{role:{field:statistics(scores[role][field],mask) for field in metrics} for role in ROLES},
            'candidate_minus_reference':{role:{field:statistics(scores[CANDIDATE][field]-scores[role][field],mask)
                                              for field in metrics} for role in REFERENCES},
            'reference_b_minus_reference_a':{field:statistics(scores[b][field]-scores[a][field],mask) for field in metrics},
        } for name,mask in masks.items()}
        results[case] = dict(cohorts=cohorts,paired_regression_worlds=np.flatnonzero(lost).tolist(),
            phase_reached_270={role:int((scores[role]['maximum_observed_forward_angle_deg']>=270.).sum()) for role in ROLES},
            per_world={role:{name:values.tolist() for name,values in scores[role].items()} for role in ROLES})
        frames[case] = motions
    return results,frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    source,summary = REPORT/'comparison.json', REPORT/'physical-support-analysis.json'
    state=read(P)['goal_execution'][KEY]
    require(state['status']=='reviewed' and sha(source)==state['result_sha256'] and sha(MASTER)==state['master_sha256'],
            'Both independently scored cases are required')
    comparison=read(source)
    hashes={rel(path):sha(path) for path in (source,MASTER)}
    cases,frames=summarize(comparison,hashes)
    payload=dict(scope='positive angle overlap within 180–270 degrees; physical support from each 4 ms solver',
                 sources_sha256=hashes, cases=cases,new_physical_steps=0,learning_updates=0,reserved_seed_used=False)
    if args.check:
        require(payload==read(summary),'Same-source physical support recomputation differs')
        picture=REPORT/'neck-support-keyframes.jpg'
        media=read(REPORT/'keyframe-source.json')
        require(sha(picture)==media['picture_sha256'] and sha(summary)==state['physical_support_result_sha256'],
                'Recorded result or annotated image differs')
    else:
        put(summary,payload)
        with patch.multiple(visual,REPORT=REPORT,ROLES=ROLES,LABELS=tuple(LABELS[role] for role in ROLES)):
            put(REPORT/'keyframe-source.json',visual.render(comparison,frames))
        project=read(P)
        stage=project['staged_training_request']
        state=project['goal_execution'][KEY]
        state.update(physical_support_result=rel(summary),physical_support_result_sha256=sha(summary),observed_videos=6)
        stage.update(current_evidence=rel(summary),current_protocol_execution='roll-observed-neck-support-reviewed')
        write(P,project)
    print('ROLL_OBSERVED_NECK_SUPPORT_VERIFIED',sha(summary),len(cases),6,flush=True)


if __name__=='__main__':
    main()
