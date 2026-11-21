"""Use the already verified predictor RNG isolation for the friction readback."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path
from unittest.mock import patch

import torch

import preflight_nominal_timestep_low_friction as base
import recover_nominal_skill_composition_zero as isolated


ROOT, PROJECT = base.ROOT, base.PROJECT
read, rel, sha, write = base.read, base.rel, base.sha, base.write
KEY, STAGE = base.KEY+'_rng_recovery', base.STAGE+'-rng-recovery'
PLAN = base.PLAN.with_stem(KEY)
REPORT = base.REPORT/'rng-recovery'
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    failed = goal[base.KEY]
    assert base.parent.allowed(project,base.STAGE) and not task['simulation_running']
    assert KEY not in goal and failed['status']=='frozen' and not (PLAN.exists() or REPORT.exists())
    source = ROOT/failed['protocol']
    plan = read(source,failed['protocol_sha256'])
    stop = base.REPORT/'zero-readback-command.log'
    log = stop.read_text(encoding='utf-8')
    assert 'assert torch.equal(rng[0],torch.get_rng_state())' in log
    assert 'NOMINAL_FRICTION_PREFLIGHT_VERIFY_EXIT=1' in log
    assert (base.REPORT/'nominal').is_dir() and not list((base.REPORT/'nominal').iterdir())
    assert not (base.REPORT/'lower-friction').exists()
    check = isolated.diagnosis(plan['walking'])
    assert not torch.cuda.is_initialized()
    plan.update(stage=STAGE,loading_readback='Preserve CPU RNG during temporary yaw predictor initialization.')
    plan['references'].update({rel(path):sha(path) for path in (
        ENTRY,source,stop,Path(isolated.__file__).resolve())})
    base.parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'rng-readback.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),previous_log=rel(stop),previous_log_sha256=sha(stop),
                       actual_physical_steps_at_prior_stop=0,diagnostic=check))
    failed.update(status='closed-zero-step-rng-guard',recovery_evidence=rel(receipt),physical_steps=0)
    goal[KEY] = dict(status='frozen',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(receipt),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Record actual CUDA zero-step coefficients through the isolated predictor loader.')
    write(PROJECT,project)
    print('NOMINAL_FRICTION_RNG_RECOVERY_FROZEN',sha(PLAN),flush=True)


def verify():
    with (patch.multiple(base,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT),
          patch.object(isolated.feedback,'components',isolated.isolated_components)):
        base.verify()


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','verify'))
    action = cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if action=='freeze' else verify)()
