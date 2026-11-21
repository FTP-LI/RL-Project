"""Locate the response of common stalls during and after the short pulse."""
from pathlib import Path

import numpy as np

from collect_walk_backward_start_pulse import OUT, PLAN, ROLES
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def review(seed,spec):
    proof_path = OUT/f'seed{seed}-analysis.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-integrity' and proof['protocol_sha256'] == sha(PLAN) and not proof['heldout_used']
    ids = proof['gate']['common_near_static']
    record = ROOT/proof['record']
    fork = spec['phase']['warmup_controls']
    controls = spec['pulse']['length']
    windows = {'during':slice(0,controls),'next':slice(controls,50),'late':slice(50,None)}
    arms = {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        assert sha(path) == proof['trace_sha256'][role]
        with np.load(path) as data:
            signed = -data['velocity'][:,ids,0]
            active = data['active'][:,ids]
            contact = data['contact'][:,ids]
            one_foot = contact[...,0]^contact[...,1]
        arms[role] = {}
        for name,part in windows.items():
            velocities, support, used = signed[part], one_foot[part], active[part]
            if used.any():
                arms[role][name] = dict(mean_reverse_speed_m_s=float(velocities[used].mean()),
                                         single_support_fraction=float(support[used].mean()))
    return dict(seed=seed,source_sha256=sha(proof_path),matched_stalls=ids,rescued=proof['gate']['rescued'],
                start_seconds=fork*.02,stop_seconds=(fork+controls)*.02,arms=arms)


if __name__ == '__main__':
    spec = read(PLAN)
    result = [review(int(seed),spec) for seed in spec['runs']]
    write(OUT/'response-review.json',dict(status='passed-saved-development-review',cohorts=result,source_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),physical_substeps=0,learning_updates=0,heldout_used=False,
          scope='Common near-static controls, paired by the shared physical fork; 6-second developmental outcomes.'))
    print('WALK_BACKWARD_PULSE_RESPONSE',result,flush=True)
