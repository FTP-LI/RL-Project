"""Replay the pulse gate on saved actor observations without advancing physics."""

from pathlib import Path
import json

import numpy as np

from review_roll_active_lateral_signal import OUT, PLAN, ROOT, read, sha
from roll_head_support_pulse import HeadSupportPulse, projected_head_roll_target
from roll_stand_walk_sequence import ROLL


DRAFT = ROOT/'configs/experiments/continuous_roll_head_support_pulse_draft.json'


def commands(folder, path, names, expected):
    file = folder/path
    assert sha(file) == expected[file.relative_to(ROOT).as_posix()]
    with np.load(file,allow_pickle=False) as stored:
        return {name:stored[name] for name in names}


def main():
    d, plan = read(DRAFT), read(PLAN)
    assert not any(d[name] for name in ('physical_substeps_authorized','heldout_used','hardware_io','learning_updates'))
    assert all(sha(ROOT/name) == checksum for name,checksum in d['pinned'].items())
    prior, signals = read(OUT/'analysis.json'), read(OUT/'support-signal-review.json')
    assert prior['protocol_sha256'] == signals['protocol_sha256'] == sha(PLAN)
    names = ('observed','phase_used','applied_targets')
    record, expected = ROOT/prior['record'], prior['source_sha256']
    prefix = commands(record,Path('shared-commands.npz'),names,expected)
    world_count = d['worlds']
    candidate = d['candidate']
    column = plan['walking']['joint_names'].index(candidate['joint'])
    lower, upper = plan['walking']['joint_limits'][column]
    results = {}
    for role in ('reference','left-push','right-push'):
        suffix = commands(record/'clips'/role,Path('commands.npz'),names,expected)
        source = {name:np.concatenate((prefix[name],suffix[name])) for name in names}
        active = source['phase_used'] == ROLL
        tracker = HeadSupportPulse(world_count,candidate['entry_degrees'],candidate['controls'])
        selected = np.stack([tracker.advance(row[:,3:6],mode) for row,mode in zip(source['observed'],active,strict=True)])
        assert not (selected & ~active).any()
        assert (selected.sum(axis=0) <= candidate['controls']).all()
        initial = np.argmax(selected,axis=0)
        counts = selected.sum(axis=0)
        assert np.array_equal(counts, np.where(selected.any(axis=0),candidate['controls'],0))
        original = source['applied_targets'][...,column]
        target = projected_head_roll_target(original,selected,lower,candidate['target_increment_rad'])
        np.testing.assert_array_equal(target[~selected],original[~selected])
        assert (target[selected] >= lower-1e-8).all() and (target[selected] <= upper+1e-8).all()
        delta = target-original
        assert (delta[selected] <= 0.).all() and (delta[selected] >= candidate['target_increment_rad']-1e-7).all()
        result = dict(pulse_worlds=int((counts > 0).sum()), selected_controls=int(selected.sum()),
            active_roll_controls=int(active.sum()), maximum_target_change_rad=float(abs(delta).max()))
        if role in signals['groups']:
            events = signals['groups'][role]['events']
            worlds = [row['world'] for row in events]
            lead = plan['initial_stand_controls']+np.array([row['first_airborne_control'] for row in events])-initial[worlds]
            assert (counts[worlds] == candidate['controls']).all() and (lead >= candidate['controls']).all()
            result.update(matched_losses=len(worlds), minimum_lead_controls=int(lead.min()), maximum_lead_controls=int(lead.max()))
        results[role] = result
    target = OUT/'head-roll-pulse-cpu.json'
    with target.open('x',encoding='utf-8') as file:
        json.dump(dict(status='passed-cpu-causal-pulse-replay',physical_substeps=0,heldout_used=False,
            protocol_sha256=sha(PLAN),draft_sha256=sha(DRAFT),script_sha256=sha(Path(__file__)),
            pulse_sha256=sha(Path(__file__).with_name('roll_head_support_pulse.py')),groups=results),file,indent=2)
        file.write('\n')
    print('ROLL_HEAD_PULSE_CPU',results,flush=True)


if __name__ == '__main__':
    main()
