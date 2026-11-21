"""Locate the CUDA zero-step friction difference without stepping any world."""

from pathlib import Path
from unittest.mock import patch

import fcntl
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import Simulation
from mjlab.utils.torch import configure_torch_backends

import run_continuous_roll_stand_walk as original
import run_forward_yaw_reward_zero as zero
from selective_friction_guard import RESET_FIELDS, resolve_foot_geometry_ids


D, ROOT = zero.D, zero.ROOT
OUT = zero.REPORT / 'zero-field-compare.json'


def read_config(cfg, route, reference, resets):
    def blocked(*_args, **_kwargs):
        raise RuntimeError('The field comparison disables physical steps')

    with patch.object(ManagerBasedRlEnv, 'step', blocked), patch.object(Simulation, 'step', blocked):
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
        try:
            feet = resolve_foot_geometry_ids(env.sim.mj_model)
            rows = {}
            for resets_done in range(resets + 1):
                if resets_done > 0:
                    env.seed(cfg.seed)
                    env.reset()
                row = {}
                for name in RESET_FIELDS:
                    actual = getattr(env.sim.model, name).cpu().numpy()
                    expected = reference[name].copy()
                    if name == 'geom_friction':
                        expected[list(feet), 0] = route['parameter_factors']['foot_friction']
                    delta = np.abs(actual - expected)
                    indices = np.flatnonzero((delta != 0).any(axis=0)).tolist() if actual.ndim == 2 else []
                    row[name] = dict(max_abs=float(delta.max()), unequal_entries=int((delta != 0).sum()),
                                     matching_worlds=int((delta.reshape(len(actual), -1) == 0).all(axis=1).sum()),
                                     differing_indices=indices, first_world=actual[0].tolist(),
                                     target=expected.tolist())
                rows[str(resets_done)] = row
            D.require(env.common_step_counter == env._sim_step_counter == 0, 'Field check advanced dynamics')
            return rows
        finally:
            env.close()


def main():
    project, plan = D.read(zero.PROJECT), D.read(zero.PLAN)
    state = project['goal_execution'][zero.check.KEY]
    D.require(D.pair.trial.allowed(project, zero.STAGE) and state['status'] == 'cuda-zero-review'
              and D.sha(zero.PLAN) == state['zero_protocol_sha256'] and not OUT.exists(),
              'Use the archived configuration failure as the investigation source')
    issue_path = ROOT / state['zero_evidence']
    D.require(D.sha(issue_path) == state['zero_evidence_sha256']
              and 'dof_frictionloss' in D.read(issue_path)['error'], 'Expected the observed field difference')
    zero.verify_sources(plan)
    route, draft = D.read(ROOT / plan['route']), D.read(ROOT / plan['draft'])
    base = ROOT / plan['nominal_root'] / 'nominal-primary-fields.npz'
    with np.load(base, allow_pickle=False) as fields:
        reference = {key: fields[key] for key in RESET_FIELDS}
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    n = route['worlds']
    official = original.configuration(dict(route, maximum_controls=route['physics_reference_controls']))
    official.events['foot_friction'].params['ranges'] = (draft['foot_sliding_friction'],) * 2
    walk, _, _ = zero.learner_config(route, draft, n, 'control', draft['pair_budget']['updates_per_arm'])
    findings = {}
    for role, cfg in (('direct-reference', official), ('training-forward', walk)):
        findings[role] = read_config(cfg, route, reference, resets=2)
        summary = {n: {field: row[field]['max_abs'] for field in RESET_FIELDS} for n, row in findings[role].items()}
        print('FORWARD_YAW_ZERO_FIELD_READBACK', role, summary, flush=True)
    D.put(OUT, dict(worlds=n, sources={D.rel(f): D.sha(f) for f in
        (issue_path, zero.PLAN, base, Path(__file__).resolve())}, readings=findings,
        physics_substeps=0, learning_updates=0, hardware_io=False))


if __name__ == '__main__':
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
