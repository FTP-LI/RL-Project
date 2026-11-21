"""Distinguish observed roll losses from unobserved forward windows."""

from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'records/reports/continuous-roll-early-lateral-push'
PLAN = ROOT / 'configs/experiments/continuous_roll_early_lateral_push.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main():
    source = OUT/'analysis.json'
    result = read(source)
    assert result['status'] == 'passed-independent-verification' and result['protocol_sha256'] == sha(PLAN)
    assert not result['heldout_used']
    reference = result['branches']['reference']['roll_outcomes']
    record = ROOT/result['record']
    for name in ('fork.json', 'evaluation.json', 'runtime-contract.json'):
        path = record/name
        assert sha(path) == result['source_sha256'][path.relative_to(ROOT).as_posix()]
    eligible = set(read(record/'fork.json')['eligible_worlds'])
    groups = {}
    for role, data in result['branches'].items():
        rolls = data['roll_outcomes']
        assert len(reference) == len(rolls)
        assert [row['env_index'] for row in rolls] == list(range(len(rolls)))
        assert sum(row['success'] for row in rolls) == data['roll_completed']
        lost = [b for a,b in zip(reference, rolls, strict=True) if a['success'] and not b['success']]
        gained = [b for a,b in zip(reference, rolls, strict=True) if not a['success'] and b['success']]
        failures = Counter(data['first_failed_stage'])
        assert sum(failures.values()) == len(rolls) and failures['passed'] == data['original_successes']
        no_revolution = [row for row in lost if 'no-supported-forward-revolution' in row['reasons']]
        rows = [dict(world=row['env_index'], reasons=row['reasons'], head_contact_seconds=row['head_contact_seconds'],
            completion_seconds=row['completion_seconds'], supported_max_deg=row['supported_max_deg'],
            reference_supported_max_deg=reference[row['env_index']]['supported_max_deg'],
            injected=row['env_index'] in eligible and role != 'reference') for row in lost]
        groups[role] = dict(roll_completed=data['roll_completed'], forward_window_completed=data['forward_completed'],
            complete_sequence=data['original_successes'], first_failed_stage_counts=dict(failures),
            roll_lost=rows, roll_gained=[row['env_index'] for row in gained], stop=data['stop'],
            overlapping_roll_reasons=dict(Counter(reason for row in lost for reason in row['reasons'])),
            median_supported_degrees_on_lost_without_revolution=median(
                row['supported_max_deg'] for row in no_revolution) if no_revolution else None)
    target = OUT/'stage-loss-review.json'
    assert not target.exists()
    target.write_text(json.dumps(dict(status='verified-stage-loss-readback', analysis_sha256=sha(source),
        protocol_sha256=sha(PLAN), controls=result['controls'], physical_substeps=0, heldout_used=False,
        forward_window_is_censored_on_early_stop=True, groups=groups), indent=2)+'\n', encoding='utf-8')
    print('ROLL_ACTIVE_LATERAL_LOSSES', {k:(v['roll_completed'],len(v['roll_lost']),len(v['roll_gained']),
        v['first_failed_stage_counts']) for k,v in groups.items()}, flush=True)


if __name__ == '__main__':
    main()
