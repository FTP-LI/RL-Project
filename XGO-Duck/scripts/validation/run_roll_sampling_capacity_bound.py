"""Retry the zero-step benchmark stop with valid immutable model-HOME binding."""

import ast
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import run_roll_sampling_capacity as original

ROOT, sha, read, put, require, rel = (original.ROOT,original.sha,original.read,original.put,original.require,original.rel)
PRIOR_PLAN, PRIOR_REPORT, PRIOR_ENTRY = original.PLAN, original.REPORT, original.ENTRY
PLAN = ROOT/'configs/experiments/roll_friction_sampling_capacity_bound.json'
REPORT = ROOT/'records/reports/roll-friction-sampling-capacity-bound'
ENTRY = Path(__file__).resolve()
ZERO_PROOF = PRIOR_REPORT/'startup-stop-review.json'
OLD_FREEZE = original.freeze


def freeze():
    p = read(ROOT/'project.json')
    s,g = p['staged_training_request'],p['goal_execution']
    x = g['roll_sampling_capacity']
    stop, receipt = (read(PRIOR_REPORT/n) for n in ('result.json','execution-receipt.json'))
    prior=read(PRIOR_PLAN)
    expected_root=original.folder_for(prior)
    require(original.active(p) and s['current_stage']=='roll-sampling-capacity-review' and not s['simulation_running']
            and sha(PRIOR_PLAN)==x['sha256']==stop['plan_sha256'] and x['status']=='stopped-review-required'
            and receipt['exit_code']==2 and receipt['log_sha256']==sha(PRIOR_REPORT/'execution.log')
            and stop['reason']=="FrozenInstanceError: cannot assign to field 'env'"
            and sorted(n.name for n in expected_root.iterdir())==['native-state.json','pass-01-e2048']
            and not list((expected_root/'pass-01-e2048').iterdir()), 'Prior zero-step stop is not substantiated')
    source = PRIOR_ENTRY.read_text(encoding='utf-8')
    declared='def collect(spec,folder,bridge):'
    require(source.count(declared)==1 and source.count('cfg.env=adapt(cfg.env,metadata,False)')==1, 'Unexpected frozen source')
    t=ast.parse(source)
    setup=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='trial')
    failure_line=next(n.lineno for n in ast.walk(setup) if isinstance(n,ast.Assign) and any(
        isinstance(target,ast.Attribute) and target.attr=='env' and isinstance(target.value,ast.Name) and target.value.id=='cfg' for target in n.targets))
    require(all(n.lineno>failure_line for n in ast.walk(setup) if isinstance(n,ast.Call)
                and ((isinstance(n.func,ast.Name) and n.func.id=='ManagerBasedRlEnv') or
                     (isinstance(n.func,ast.Attribute) and n.func.attr=='step'))), 'Cannot prove failure precedes stepping')
    put(ZERO_PROOF,dict(status='verified-stopped-before-environment-construction', starting_line=failure_line,
        result_sha256=sha(PRIOR_REPORT/'result.json'), receipt_sha256=sha(PRIOR_REPORT/'execution-receipt.json'),
        physical_controls=0, physics_substeps=0, learner_updates=0, hardware_io=False))
    s.update(current_stage=original.STAGE+'-preflight',
             next_gate='Recheck the VS Code observer before the same sampling-only benchmark after a zero-step startup fix.')
    g['roll_sampling_capacity_startup_history']=deepcopy(x)
    original.write(ROOT/'project.json',p)
    # Add immutable provenance to the new plan before the original routine
    # registers its digest; this never rewrites the stopped plan or input.
    with add_provenance():
        OLD_FREEZE()


@contextmanager
def add_provenance():
    def enriched_put(path,value):
        if path == PLAN:
            parents = (PRIOR_ENTRY,PRIOR_PLAN,ZERO_PROOF)
            value['source_sha256'].update({rel(p):sha(p) for p in parents})
        put(path,value)

    with patch.object(original,'put',enriched_put):
        yield


def bound_collect(*args):
    # The sampler needs only an env slot. Wrap it rather than attempting to
    # assign TrainConfig.env, which is frozen. The learner config is untouched.
    import check_roll_friction_exposure as preflight
    make = preflight.make
    with patch.object(preflight,'make',lambda *a: SimpleNamespace(env=make(*a).env)):
        return original_collect(*args)


if __name__=='__main__':
    original.PLAN,original.REPORT,original.ENTRY = PLAN,REPORT,ENTRY
    import argparse
    argv=argparse.ArgumentParser(description=__doc__)
    argv.add_argument('action',choices=('freeze','run','child','confirm'))
    argv.add_argument('--screenshot')
    args=argv.parse_args()
    original.freeze=freeze
    if args.action=='child':
        require(read(PLAN)['source_sha256'][rel(PRIOR_ENTRY)]==sha(PRIOR_ENTRY),'Frozen runner changed')
        original_collect = original.collect
        original.collect = bound_collect
    sys.exit(original.main())
