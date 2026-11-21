"""Recount the verified reserved holdout and summarize paired outcomes."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'records/reports/getup-monitored-pair-holdout-s4268'
ANALYSIS = OUT / 'analysis.json'
POSES = ('prone', 'supine', 'left-side', 'right-side')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    output = OUT / 'summary.json'
    if output.exists():
        raise FileExistsError('Keep the previous independent summary')
    a = read(ANALYSIS)
    if (a['status'] != 'passed-independent-verification' or a['heldout_used'] is not True
            or a['holdout_for_tuning'] is not False):
        raise ValueError('This is not the verified reserved holdout')
    record = ROOT / a['record']
    spec = read(record / 'protocol.json')
    if digest(record / 'protocol.json') != a['protocol_sha256']:
        raise ValueError('Holdout protocol changed')
    roles = spec['branches']
    outcomes, classes, arms = {}, None, {}
    for role in roles:
        arm = a['arms'][role]
        episodes = arm['timing']['episodes']
        actual = [row['release_pose'] for row in episodes]
        if classes is None:
            classes = actual
        if actual != classes:
            raise ValueError('Release classes differ across policies')
        outcomes[role] = [row['hold_start_seconds'] is not None and stable and row['release_pose'] in POSES
                          for row, stable in zip(episodes, arm['end_hold']['stable_by_environment'], strict=True)]
        if sum(outcomes[role]) != arm['combined_successes']:
            raise ValueError('Combined result does not match saved episode outcomes')
        for grouping, key in (('requested', 'requested_pose'), ('actual', 'release_pose')):
            counts = Counter(row[key] for row in episodes)
            success = Counter(row[key] for row, ok in zip(episodes, outcomes[role], strict=True) if ok)
            for pose in POSES:
                item = arm['timing'][grouping][pose]
                if item['episodes'] != counts[pose] or item['timely_holds'] != success[pose]:
                    raise ValueError('Grouped result differs: ' + role + '/' + grouping + '/' + pose)
        arms[role] = dict(combined_successes=sum(outcomes[role]),
                          all_motion_gates_passed=arm['all_motion_gates_passed'],
                          requested=arm['timing']['requested'], actual=arm['timing']['actual'])
    paired = {}
    for ref in roles[:-1]:
        paired[ref] = {}
        for pose in ('all', *POSES):
            indices = [i for i, value in enumerate(classes) if pose == 'all' or value == pose]
            wins = [i for i in indices if outcomes['candidate'][i] and not outcomes[ref][i]]
            losses = [i for i in indices if outcomes[ref][i] and not outcomes['candidate'][i]]
            paired[ref][pose] = dict(episodes=len(indices), wins=wins, losses=losses, net=len(wins)-len(losses))
        if paired[ref]['all']['wins'] != a['comparisons'][ref]['all']['wins'] or paired[ref]['all']['losses'] != a['comparisons'][ref]['all']['losses']:
            raise ValueError('Paired recount differs: ' + ref)
    telemetry = read(record / 'gpu-telemetry.json')
    if telemetry['failure'] is not None or not telemetry['samples']:
        raise ValueError('GPU telemetry incomplete')
    samples = telemetry['samples']
    state = read(record / 'evaluation.json')
    videos = [dict(role=v['role'], requested_pose=v['requested_pose'], actual_release_pose=v['actual_release_pose'],
                   path=v['path'], frames=v['frames'], fps=v['fps']) for v in a['videos']]
    if len(videos) != 12 or not all(v['frames'] == 276 and v['fps'] == 25 for v in videos):
        raise ValueError('Video verification is incomplete')
    result = dict(status='recounted-from-verified-holdout-episodes', checked_at=datetime.now(timezone.utc).isoformat(),
                  source_analysis_sha256=digest(ANALYSIS), protocol_sha256=a['protocol_sha256'],
                  script_sha256=digest(Path(__file__)), record=a['record'], environments=spec['environment']['num_envs'],
                  arms=arms, paired=paired, videos=videos, heldout_used=True, holdout_for_tuning=False,
                  model_promotion_allowed=False,
                  performance=dict(transitions=state['actual_control_transitions'],
                      collection_and_save_seconds=state['sampling_wall_seconds'],
                      transitions_per_second=state['actual_control_transitions']/state['sampling_wall_seconds'],
                      gpu_samples=len(samples), peak_used_mib=max(s['used_mib'] for s in samples),
                      median_gpu_percent=statistics.median(s['gpu_percent'] for s in samples)),
                  new_physics_substeps=0, new_learning_updates=0)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('HELDOUT_SUMMARY_VERIFIED', {r: arms[r]['combined_successes'] for r in roles})


if __name__ == '__main__':
    main()
