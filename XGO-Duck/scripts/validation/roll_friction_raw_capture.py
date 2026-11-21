"""Record selected zero-step arrays after the normal scene fingerprint.

The preceding comparison retained only hashes. This read-only hook records
the affected numeric arrays plus contact counters, checks that each array
matches the already saved fingerprint, and verifies counters and RNG again.
It never takes a simulation step or modifies the policy.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import warp as wp
import roll_entry_snapshot
import roll_factor_zero_recorded_hook as recorded
from getup_release_fork_state import signature
from roll_factor_start_fingerprint import _rng

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT/'records/reports/roll-matched-factor-preflight-20261116/friction-cache-identity-review.json'
CONTEXT = {'warp.nefc', 'warp.nacon', 'warp.contact.worldid', 'warp.contact.geom',
           'warp.contact.geomcollisionid', 'warp.contact.dist', 'warp.contact.friction',
           'warp.contact.dim', 'warp.contact.efc_address'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record_after_capture(original, env, memory, output, run_id, protocol):
    output = Path(output)
    result = original(env, memory, output, run_id, protocol)
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    if review['status'] != 'saved-cache-identity-and-cpu-repeat-reviewed':
        raise ValueError('Saved friction review is missing')
    wanted = set(review['gpu_other_changed_fields']) | CONTEXT
    if (int(env.common_step_counter), int(env._sim_step_counter)) != (0, 0):
        raise RuntimeError('Numeric capture requires zero physics steps')
    slots, generators = roll_entry_snapshot.roll_slots(env, memory)
    before = signature(_rng(env, generators))
    raw = {}
    for name, owner, field in slots:
        if name not in wanted:
            continue
        value = getattr(owner, field)
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy().copy()
        elif isinstance(value, wp.array):
            array = value.numpy().copy()
        else:
            raise ValueError('Selected field is not an array: ' + name)
        if array.dtype.hasobject:
            raise ValueError('Object array cannot be retained: ' + name)
        raw[name] = array
    if set(raw) != wanted:
        raise ValueError('Selected fields missing: ' + ', '.join(sorted(wanted - set(raw))))
    if torch.device(env.device).type == 'cuda':
        torch.cuda.synchronize(env.device)
    if signature(_rng(env, generators)) != before or (int(env.common_step_counter), int(env._sim_step_counter)) != (0, 0):
        raise RuntimeError('Numeric capture changed RNG or counters')
    saved = json.loads(output.read_text(encoding='utf-8'))
    if any(signature(array) != saved['fields'][name] for name, array in raw.items()):
        raise RuntimeError('Numeric array differs from its saved fingerprint')
    destination = output.parent/'selected-zero-numeric.npz'
    with destination.open('xb') as stream:
        np.savez_compressed(stream, **raw)
    metadata = {'status': 'captured-selected-zero-numeric-arrays', 'run_id': run_id,
        'protocol_sha256': sha(protocol), 'fingerprint_sha256': sha(output), 'arrays_sha256': sha(destination),
        'fields': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in sorted(raw.items())},
        'physical_steps': 0, 'learning_updates': 0, 'rng_unchanged': True, 'hardware_io': False}
    with (output.parent/'selected-zero-numeric.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


@contextmanager
def raw_capture():
    original = recorded.capture
    def wrapped(env, memory, output, run_id, protocol):
        return record_after_capture(original, env, memory, output, run_id, protocol)
    with patch.object(recorded, 'capture', side_effect=wrapped):
        yield
