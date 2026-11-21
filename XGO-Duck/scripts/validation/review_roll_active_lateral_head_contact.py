"""Localize the first lost support to head or foot contact in saved branches."""

import json

import numpy as np

from review_roll_active_lateral_signal import OUT, PLAN, ROOT, arrays, read, sha


def main():
    analysis_path, signal_path = OUT/'analysis.json', OUT/'support-signal-review.json'
    analysis, signal = read(analysis_path), read(signal_path)
    assert signal['analysis_sha256'] == sha(analysis_path) and signal['protocol_sha256'] == sha(PLAN)
    plan = read(PLAN)
    offset = analysis['shared_controls']-plan['initial_stand_controls']
    root = ROOT/analysis['record']/'clips'
    limit = max(item['first_airborne_control'] for row in signal['groups'].values() for item in row['events'])-offset+5
    fields = ('ground_contact', 'head_contact', 'foot_contact')
    cases = {role: arrays(root/role/'trace.npz', fields, limit, analysis['source_sha256'])
             for role in ('reference', *signal['groups'])}
    results = {}
    for role, group in signal['groups'].items():
        worlds = np.asarray([item['world'] for item in group['events']])
        gap = np.asarray([item['first_airborne_control']-offset for item in group['events']])
        counts = {}
        for label in ('reference', role):
            trace = cases[label]
            counts[label] = {str(k): {field: int(trace[field][gap+k, worlds].any(axis=-1).sum()
                    if field == 'foot_contact' else trace[field][gap+k, worlds].sum()) for field in fields}
                for k in range(-1, 5)}
        assert counts[role]['-1']['head_contact'] == len(worlds) == counts[role]['4']['head_contact']
        assert not any(counts[role]['0'].values())
        results[role] = dict(worlds=worlds.tolist(), counts_by_control_from_disturbed_gap=counts)
    result = dict(status='verified-matched-head-contact-readback', physical_substeps=0, heldout_used=False,
        protocol_sha256=sha(PLAN), signal_sha256=sha(signal_path), analysis_sha256=sha(analysis_path),
        step_seconds=plan['roll']['criteria']['step_dt'], groups=results)
    with (OUT/'head-contact-review.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print('ROLL_HEAD_CONTACT_GAP', {role: (len(row['worlds']), row['counts_by_control_from_disturbed_gap']['reference']['0'])
                                    for role, row in results.items()})


if __name__ == '__main__':
    main()
