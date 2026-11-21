"""Audit contact at release and the first later interval with no feet touching."""

from pathlib import Path

import numpy as np

from audit_nominal_supine_pre_first_touch import KEY, PROJECT, ROOT, features, read, rel, sha, source_arrays, write


def summarize(trace, passed, spec):
    _, _, valid = features(trace, spec)
    feet = trace['foot_contact'].any(axis=-1) & trace['active']
    clear = trace['active'] & ~feet
    confirmed_airborne = np.zeros_like(clear)
    confirmed_airborne[1:] = clear[1:] & clear[:-1]
    was_clear = np.logical_or.accumulate(confirmed_airborne, axis=0)
    retouch = (was_clear & feet)
    retouch_observed = np.logical_or.accumulate(retouch, axis=0)
    result = {}
    for label, group in (('admitted',passed), ('missed',~passed)):
        available = []
        for control in spec['feature_controls']:
            k = control-100
            in_air = group & was_clear[k] & clear[k] & ~retouch_observed[k]
            j = k+spec['minimum_lead_to_first_touch_controls']-1
            opportunity = in_air & valid[k] & trace['active'][j] & ~retouch_observed[j]
            available.append(dict(control=control, gravity_and_active=int((group & valid[k]).sum()),
                                  already_touched=int((group & trace['touched_by'][k]).sum()),
                                  confirmed_clear_before_first_retouch=int(in_air.sum()), with_lead=int(opportunity.sum())))
        result[label] = dict(n=int(group.sum()), contact_at_release=int((group & feet[0]).sum()),
                             at_least_two_consecutive_clear=int((group & was_clear[-1]).sum()),
                             retouch_after_confirmed_clear=int((group & retouch_observed[-1]).sum()), controls=available)
    return result


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status'] == 'archive-audit-reviewed' and not node['admitted_causal_trigger']
    spec = read(ROOT/node['design'],node['design_sha256'])
    previous = ROOT/node['analysis']
    audited = read(previous,node['analysis_sha256'])
    assert all(row['admitted']['eligible_at_feature'] + row['missed']['eligible_at_feature'] == 0
               for cohort in audited['cohorts'] for row in cohort['reads'])
    cohorts = []
    for source in spec['sources']:
        trace, passed, _, _ = source_arrays(project,source,spec)
        groups = summarize(trace,passed,spec)
        cohorts.append(dict(seed=source['seed'],collection_sha256=source['collection_sha256'],groups=groups))
    output = previous.with_name('release-contact-chronology.json')
    assert not output.exists()
    write(output,dict(previous_analysis_sha256=sha(previous),script_sha256=sha(Path(__file__)),
        minimum_consecutive_clear_controls=2,cohorts=cohorts,new_physics_steps=0,learning_updates=0))
    node.update(contact_chronology=rel(output),contact_chronology_sha256=sha(output))
    project['staged_training_request']['current_evidence'] = project['current_evidence'] = rel(output)
    write(PROJECT,project)
    print('SUPINE_RELEASE_CONTACT_CHRONOLOGY',sha(output),cohorts,flush=True)


if __name__ == '__main__':
    main()
