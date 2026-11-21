"""Check which paired changes occurred in worlds that actually used the teacher."""

from pathlib import Path

import numpy as np

from audit_nominal_supine_early_support import reconstruct
from audit_nominal_supine_terminal_stance import attitude
from roll_stand_walk_sequence import ROLL
from run_nominal_supine_early_takeover import END, FORK, KEY, PROJECT, REPORT, ROLES, ROOT, read, sha, write
from verify_nominal_skill_composition import read_trace


def main():
    node = read(PROJECT)['goal_execution'][KEY]
    assert node['status']=='reviewed'
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    scores = read(ROOT/node['pair_analysis'],node['pair_analysis_sha256'])
    collection = ROOT/node['collection']
    raw = read(collection,node['collection_sha256'])
    folder = collection.parent
    base, candidate = ROLES
    effects = {}
    for arm in ROLES:
        branch = raw['branches'][arm]
        path = folder/arm/'takeover.npz'
        assert sha(path)==branch['takeover_sha256']
        with np.load(path,allow_pickle=False) as items:
            effects[arm] = dict(items)
    x = effects[candidate]
    worlds = x['worlds']
    assert sorted(worlds)==scores['roles'][candidate]['applied']
    controls = x['first_applied'][worlds]
    # Align both arms to the first command that actually used the incoming
    # actor. With the same absolute clock an earlier baseline divergence is visible.
    timing = {'before':controls-1,'plus_100ms':controls+4,'plus_500ms':controls+24,
              'end':np.full(len(worlds),END,dtype=np.int32)}
    names = ('qpos','quaternion','foot_contact','roll_phase')
    summaries, details, passed = {},{},{}
    actual = np.stack(tuple(timing.values()))
    for arm in ROLES:
        branch = raw['branches'][arm]
        trace = read_trace(folder/arm,branch['chunks'],FORK+1,END,names,sha)
        passed[arm] = trace['roll_phase'][-1,worlds]==ROLL
        samples = {k:trace[k][actual-FORK-1,worlds] for k in names}
        angles = attitude(samples['quaternion'])[...,2]
        physical = reconstruct(plan,samples)
        both = samples['foot_contact'].all(axis=-1)
        match = (samples['foot_contact']==physical['reconstructed_feet']).all(axis=-1)
        front = physical['front_margin_mm']
        head = physical['head_foot'].any(axis=-1)
        details[arm] = dict(tilt_deg=angles.tolist(),head_foot=head.tolist(),dual_foot=both.tolist(),contact_flags_match=match.tolist())
        rows = {}
        for i,label in enumerate(timing):
            used = both[i]&match[i]&np.isfinite(front[i])
            rows[label] = dict(tilt_deg_median=float(np.median(angles[i])),dual_foot=int(both[i].sum()),
                head_foot_reconstructed=int(head[i].sum()),margins_count=int(used.sum()),
                front_margin_mm_median=None if not used.any() else float(np.median(front[i,used])))
        summaries[arm] = rows
    gained, lost = set(scores['gained_roll_entry']), set(scores['lost_roll_entry'])
    controlled = set(int(i) for i in worlds)
    newly_admitted = sorted(gained&controlled)
    lost_admission = sorted(lost&controlled)
    assert newly_admitted==sorted(worlds[passed[candidate]&~passed[base]].tolist())
    assert lost_admission==sorted(worlds[passed[base]&~passed[candidate]].tolist())
    delta = np.abs(x['raw']-x['parent_raw'])
    actual_step = np.abs(x['applied_target']-x['previous_target'])
    result = dict(protocol_sha256=node['protocol_sha256'],collection_sha256=node['collection_sha256'],
        pair_analysis_sha256=node['pair_analysis_sha256'],source_script_sha256=sha(Path(__file__)),
        physically_controlled_worlds=worlds.tolist(),first_completed_teacher_control=controls.tolist(),
        median_first_action_maximum_absolute_teacher_vs_parent_rad=float(np.median(delta.max(axis=-1))),
        first_applied_target_maximum_observed_step_rad=float(actual_step.max()),
        roll_admissions_in_controlled_worlds={k:int(v.sum()) for k,v in passed.items()},
        gained_after_taking_over=newly_admitted,lost_after_taking_over=lost_admission,
        gained_without_taking_over=sorted(gained-controlled),lost_without_taking_over=sorted(lost-controlled),
        paired_readouts_at_completed_controls={k:x.tolist() for k,x in timing.items()},paired_summary=summaries,
        per_world_readouts=details,new_physical_steps=0)
    target = REPORT/'takeover-effect.json'
    assert not target.exists()
    write(target,result)
    print('SUPINE_TAKEOVER_EFFECT',sha(target),len(worlds),newly_admitted,lost_admission,flush=True)


if __name__=='__main__':
    main()
