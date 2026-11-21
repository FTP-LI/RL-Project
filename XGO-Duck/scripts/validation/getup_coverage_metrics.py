"""Prospective recovery coverage, keeping the earlier strict score separate.

An initial side fall can settle into supine during the mandatory freeze. Test
both the requested starting groups and enough actual release groups, without
resampling, discarding failures, or relabeling the old strict score as a pass.
Recovery timing, support, joint and posture criteria stay in standup_metrics.
"""
from math import ceil, sqrt
from standup_metrics import summarize

POSES = ('prone', 'supine', 'left-side', 'right-side')


def rate_row(rows, minimum, coverage):
    n = len(rows)
    k = sum(r['hold_start_seconds'] is not None and r['release_pose'] in POSES for r in rows)
    z = 1.959963984540054
    center = (k / n + z*z/(2*n))/(1 + z*z/n) if n else 0
    half = z*sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n))/(1+z*z/n) if n else 0
    return dict(episodes=n, timely_holds=k, required_successes=ceil(minimum*n-1e-10),
                coverage_passed=n >= coverage, rate=None if not n else k/n,
                wilson95=None if not n else [max(0., center-half), min(1., center+half)],
                passed=n >= coverage and k >= ceil(minimum*n-1e-10))


def group_rows(rows, minimum=.95, actual_minimum=32):
    requested = {pose: rate_row([r for r in rows if r['requested_pose'] == pose], minimum, 1) for pose in POSES}
    actual = {pose: rate_row([r for r in rows if r['release_pose'] == pose], minimum, actual_minimum) for pose in POSES}
    overall = rate_row(rows, minimum, 1)
    other = sum(r['release_pose'] not in POSES for r in rows)
    return dict(requested=requested, actual=actual, overall=overall, other_release_episodes=other,
                functional_coverage_gate=other == 0 and overall['passed'] and
                    all(r['passed'] for r in [*requested.values(), *actual.values()]),
                strict_requested_successes=sum(r['success'] for r in rows),
                wrong_release_episodes=sum(not r['release_pose_valid'] for r in rows))


def evaluate(trace, plan):
    strict = summarize(trace, plan['criteria'])
    result = group_rows(strict['episodes'], plan['success_rate_min'], plan['actual_release_minimum'])
    result.update(episodes=strict['episodes'], strict_requested_score_retained=True,
                  scope='Fresh development coverage only; no heldout, skill-switch or hardware acceptance')
    return result
