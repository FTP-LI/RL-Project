"""Index exact difficult initial releases from development-only references."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from verify_walk_backward_onset import score


PLAN = ROOT/'configs/experiments/common_walk_backward_start_curriculum_draft.json'
OUT = ROOT/'records/reports/common-walk-backward-start-curriculum/initial-cases.json'
FIELDS = ('active','velocity','contact','terminated','truncated','fell_over','out_of_bounds','nan_state')
PROTOCOL_DIFFERENCES = {'seed','stage','groups','parallel_worlds','episodes_per_command','maximum_transitions'}


def checked_array(path, digest, keys):
    assert sha(path)==digest
    with np.load(path) as data:
        return {key:data[key] for key in keys}


def archive_for(proof):
    record = ROOT/proof['record']
    if record.is_dir():
        return record
    archived = ROOT/'records/training-archive/evaluations'/record.name
    assert archived.is_dir(),proof['record']
    return archived


def review(path, digest, plan, base):
    assert sha(path)==digest
    evidence = read(path)
    assert evidence['status']=='passed-integrity' and not evidence['heldout_used'] and not evidence['learning_updates']
    record = archive_for(evidence)
    protocol = record/'protocol.json'
    assert sha(protocol)==evidence['protocol_sha256']
    spec = read(protocol)
    p = spec['environment']
    changed = {key for key in (p.keys()|base.keys()) if p.get(key)!=base.get(key)}
    assert not (changed-PROTOCOL_DIFFERENCES)
    assert p['commands']=={'backward':[-.1,0,0]} and p['groups']['backward']==list(range(p['parallel_worlds']))
    scene = {v for k,v in spec['reference_sha256'].items() if k.endswith('/cpu-scene-fields.npz')}
    assert scene=={plan['scene_sha256']}
    paired = 'scores' in evidence
    roles = ('backward','reference-repeat') if paired else ('backward',)
    prefix = (checked_array(record/'trace-shared-prefix.npz',evidence['shared_prefix_sha256'],FIELDS) if paired else None)
    references = {}
    for role in roles:
        trace = checked_array(record/f'trace-{role}.npz',
                    evidence['trace_sha256'][role] if paired else evidence['source_sha256'],FIELDS)
        if prefix is not None:
            trace = {key:np.concatenate((prefix[key],trace[key])) for key in FIELDS}
        assert trace['active'].shape==(p['maximum_control_steps'],p['parallel_worlds'])
        recalculated = score(trace,p,spec)
        reported = evidence['scores'][role] if paired else evidence['analysis']
        assert recalculated['categories']==reported['categories']
        assert recalculated['near_static_worlds']==reported['near_static_worlds']
        references[role] = set(recalculated['near_static_worlds'])
    worlds = sorted(set.intersection(*references.values()))
    previous_intervention_worlds = None
    if paired:
        archived = evidence['common_near_static'] if 'common_near_static' in evidence else evidence['gate']['common_near_static']
        previous_intervention_worlds = archived
        # Actuator and roll studies scored their candidate only on triggered
        # worlds. Reset coverage uses both original branches without that filter.
        eligible = sorted(set(worlds)&set(evidence['triggered'])) if 'eligible' in evidence else worlds
        assert eligible==archived
    initial_path = record/'physical-initial.npz'
    initial_digest = read(record/'evaluation.json')['initial_state_sha256']
    initial = checked_array(initial_path,initial_digest,('qpos','qvel','xy'))
    pos,vel,xy = (initial[key] for key in ('qpos','qvel','xy'))
    n = p['parallel_worlds']
    assert pos.ndim==vel.ndim==2 and len(pos)==len(vel)==n and xy.shape==(n,2)
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    np.testing.assert_allclose(pos[:,:2],xy,atol=1e-6,rtol=0.)
    np.testing.assert_allclose(np.linalg.norm(pos[:,3:7],axis=1),1.,atol=1e-6,rtol=0.)
    rows = []
    for world in worlds:
        state = sha256(np.ascontiguousarray(pos[world,2:]).tobytes()+np.ascontiguousarray(vel[world]).tobytes())
        rows.append(dict(world=world,canonical_qpos_qvel_sha256=state.hexdigest()))
    return dict(seed=evidence['seed'],record=record.relative_to(ROOT).as_posix(),
                source=path.relative_to(ROOT).as_posix(),source_sha256=digest,
                physical_initial=initial_path.relative_to(ROOT).as_posix(),physical_initial_sha256=initial_digest,
                configuration_sha256=evidence['protocol_sha256'],source_worlds=n,reference_roles=list(roles),
                previous_intervention_worlds=previous_intervention_worlds,rows=rows)


def main():
    assert not OUT.exists()
    plan = read(PLAN)
    assert plan['status']=='draft-initial-state-audit-only'
    assert not any(plan[key] for key in ('heldout_used','hardware_io','physical_substeps_authorized','learning_updates_authorized'))
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==plan['stage'] and ROOT/project['current_protocol']==PLAN
    reference = ROOT/plan['reference_plant']
    assert sha(reference)==plan['reference_plant_sha256']
    base = read(reference)['environment']
    assert sha(ROOT/base['model'])==base['model_sha256']
    cohorts = [review(ROOT/path,digest,plan,base) for path,digest in plan['development_sources']]
    assert len({c['seed'] for c in cohorts})==len(cohorts)
    rows = [row for cohort in cohorts for row in cohort['rows']]
    count = len({row['canonical_qpos_qvel_sha256'] for row in rows})
    seeds = sum(bool(c['rows']) for c in cohorts)
    passed = count>=plan['minimum_unique_difficult_initial_states'] and seeds>=plan['minimum_source_seeds']
    OUT.parent.mkdir()
    write(OUT,dict(status='passed-initial-case-integrity',draft_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
                   model_sha256=base['model_sha256'],scene_sha256=plan['scene_sha256'],
                   inspected_releases=sum(c['source_worlds'] for c in cohorts),difficult_releases=len(rows),
                   unique_difficult_releases=count,contributing_seeds=seeds,cohorts=cohorts,
                   reset_readback_admitted=passed,learning_admitted=False,heldout_used=False,physical_substeps=0,
                   learning_updates=0,raw_states_duplicated=False))
    print('BACKWARD_INITIAL_CASES',count,seeds,passed,[(c['seed'],len(c['rows'])) for c in cohorts],flush=True)


if __name__=='__main__':
    main()
