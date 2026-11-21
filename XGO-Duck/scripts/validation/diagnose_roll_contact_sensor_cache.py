"""Replay the cached-versus-current contact read and verify time alignment."""
import inspect
from pathlib import Path

import numpy as np
import torch

from review_roll_contact_clock_boundaries import OUT, summarize
from roll_contact_endpoint import found_after_forward
from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write


def check_sensor_cache():
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensor, ContactSensorCfg, _ContactSlot

    config = ContactSensorCfg(name='zero-step-cache-check', primary=ContactMatch(mode='geom', pattern='head'),
                             fields=('found',), num_slots=1, reduce='none')
    sensor = ContactSensor(config)
    live = torch.tensor([[0.], [1.]])
    sensor._slots = [_ContactSlot('head', 'found', 'fixture', live)]
    first = sensor.data
    torch.testing.assert_close(first.found, live)
    live.copy_(1.-live)
    assert sensor.data is first
    stale = sensor.data.found.clone()
    fresh = found_after_forward(sensor, 1)
    assert torch.equal(stale, torch.tensor([[0.], [1.]]))
    assert torch.equal(fresh, torch.tensor([[True], [False]]))
    assert sensor.data is first and torch.equal(sensor.data.found, stale)
    sensor.update(.004)
    assert torch.equal(sensor.data.found > 0, fresh)
    return dict(cached_before_refresh=stale[:, 0].tolist(), direct_read=fresh[:, 0].tolist(),
                cache_unchanged_by_direct_read=True, update_invalidates_cache=True)


def main():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor.contact_sensor import ContactSensor
    from mjlab.sensor.sensor import Sensor

    proof_path = OUT/'analysis.json'
    boundary_path = OUT/'boundary-review.json'
    proof, boundary = read(proof_path), read(boundary_path)
    assert proof['status'] == 'passed-independent-diagnostic-verification' and not proof['heldout_used']
    assert boundary['status'] == 'passed-saved-data-only' and boundary['clock_analysis_sha256'] == sha(proof_path)
    folder = ROOT/proof['record']
    trace_path, clock_path = folder/'forward-roll-trace.npz', folder/'contact-clock.npz'
    assert sha(trace_path) == proof['trace_sha256'] and sha(clock_path) == proof['substeps_sha256']
    with np.load(trace_path) as data:
        trace = {key: data[key] for key in ('qpos', 'head_contact', 'foot_contact', 'ground_contact', 'active')}
    with np.load(clock_path) as data:
        clock = {key: data[key] for key in ('substep', 'before', 'after', 'points', 'normal_force_n')}
    prev = np.flatnonzero((clock['substep'] % 5 == 0) & (clock['substep'] < clock['substep'][-1]))
    steps = clock['substep'][prev]//5
    np.testing.assert_array_equal(steps, np.arange(11, 70))
    np.testing.assert_array_equal(clock['after'][prev], clock['before'][prev+1])
    np.testing.assert_array_equal(clock['after'][prev], trace['qpos'][steps, :, 3:7])
    active = trace['active'][steps]
    thresholds = (0., proof['force_threshold_n'])
    source = [trace['ground_contact'][steps], trace['head_contact'][steps], trace['foot_contact'][steps].any(-1)]
    stats = {}
    for kind, column, threshold in zip(('candidate_points', 'normal_force'), ('points', 'normal_force_n'), thresholds, strict=True):
        values = clock[column]
        stats[kind] = {}
        for tag, offset in (('before_last_integration', 0), ('post_integration_pose', 1)):
            hit = values[prev+offset] > threshold
            observed = [hit[..., 0], hit[..., 1], hit[..., 2:].any(-1)]
            stats[kind][tag] = {key: summarize(sensor[active], raw[active]) for key, sensor, raw in
                                 zip(('ground', 'head', 'either_foot'), source, observed, strict=True)}

    rng = torch.get_rng_state().clone()
    fixture = check_sensor_cache()
    assert not torch.cuda.is_initialized() and torch.equal(rng, torch.get_rng_state())
    files = [Path(inspect.getsourcefile(cls)) for cls in (Sensor, ContactSensor, ManagerBasedRlEnv)]
    files.append(WORK/'src/mjlab_microduck/tasks/mdp.py')
    runtime_sources = {str(path): sha(path) for path in files}
    target = OUT/'sensor-cache-review.json'
    assert not target.exists(), 'Preserve the original cache audit'
    write(target, dict(status='passed-archived-alignment-and-cpu-fixture',
        compared_control_boundaries=len(steps), active_world_boundaries=int(active.sum()), comparison=stats,
        fixture=fixture, step_order=['scene.update invalidates the cache', 'reward can cache contact',
          'sim.forward recomputes at post-integration pose', 'cached data remains until a later sensor update'],
        clock_analysis_sha256=sha(proof_path), boundary_review_sha256=sha(boundary_path),
        source_sha256=runtime_sources, script_sha256=sha(Path(__file__)),
        reader_sha256=sha(Path(__file__).with_name('roll_contact_endpoint.py')),
        new_physical_steps=0, learning_updates=0, heldout_used=False))
    print('ROLL_SENSOR_CACHE', fixture, stats, flush=True)


if __name__ == '__main__':
    main()
