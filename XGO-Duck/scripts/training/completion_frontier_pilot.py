"""Test paying landing rewards only after a qualified 330-degree roll.

Common: verified official actor import, source22-degree reset, model24-degree
coordinate adapter,14critic/36actor updates and all previous stop guards.
Only five landing reward terms receive the new completion mask. The early
rise-velocity reward, progress, head-pivot reward and physical limits remain
as in the source. Control uses the identical wrapper and returns its input.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import os
import shutil
import sys
import numpy as np
import torch
import roll_top_contact_pilot as core
from training_watch import ROOT,read
from record_io import write
from guarded_training import sha

LANDING_TERMS=('roulade_landing_composite','roulade_upright_after_roll',
               'roulade_height_after_roll','roulade_landing_sharp','roulade_stand_tax')


def completion_mask(value,qualified,only_complete):
    return value*(qualified>=np.deg2rad(330.)).to(value.dtype) if only_complete else value


def landing_reward(env,source_function,only_complete=False,**params):
    # The source reward updates its own qualified-progress state first.
    value=source_function(env,**params)
    return completion_mask(value,env._roulade_max,only_complete)


def prepare(plan):
    common=deepcopy(plan);common['arm']='control'
    cfg,display,metadata=core.prepare(common)
    for name in LANDING_TERMS:
        term=cfg.env.rewards[name]
        assert term.func.__module__=='mjlab_microduck.tasks.mdp'
        assert np.isclose(term.params['gate_lo'],np.deg2rad(260.))
        assert np.isclose(term.params['gate_hi'],np.deg2rad(330.))
        term.params=dict(term.params,source_function=term.func,only_complete=plan['arm']=='candidate')
        term.func=landing_reward
    assert cfg.env.rewards['roulade_rise_velocity'].func.__name__=='roulade_rise_velocity'
    assert not cfg.env.rewards['roulade_head_pivot'].params['only_top']
    return cfg,display,metadata


def runner_class(base,plan,record,metadata):
    parent=core.runner_class(base,plan,record,metadata)
    class CompletionFrontier(parent):
        def _live_check(self,imported):
            parent._live_check(self,imported)
            env=self.env.unwrapped
            fields={k:getattr(env.sim.model,k).detach().cpu().numpy().copy() for k in sorted(env.sim.expanded_fields)}
            np.savez_compressed(record/'training-expanded-initial.npz',**fields)
            if plan['arm']=='candidate':
                control=ROOT/'records/experiments'/plan['matching_control_run']/'training-expanded-initial.npz'
                with np.load(control) as prior:
                    assert set(prior.files)==set(fields)
                    for k,value in fields.items():np.testing.assert_array_equal(value,prior[k])
            write(record/'completion-reward-contract.json',dict(status='passed',physical_steps=0,
                arm=plan['arm'],qualified_frontier_deg=330.,terms=list(LANDING_TERMS),
                source_head_pivot_retained=True,early_rise_velocity_retained=True,
                expanded_initial_sha256=sha(record/'training-expanded-initial.npz'),
                matched_captured_fields=plan['arm']=='candidate'))
    return CompletionFrontier


def run(planfile):
    planfile=Path(planfile);plan=read(planfile)
    assert os.isatty(1),'Use one visible VS Code terminal'
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-completion-frontier-pilot'
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(core.WORK/name)==digest,name
    for name,digest in plan['environment_source_sha256'].items():assert sha(Path(name))==digest,name
    cfg,display,metadata=prepare(plan)
    record=ROOT/'records/experiments'/plan['run_id'];record.mkdir(exist_ok=False)
    write(record/'protocol.json',plan);write(record/'display-state.json',display)
    source_dir=record/'sources';source_dir.mkdir()
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,source_dir/Path(name).name)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(planfile)))
    write(record/'run-event.json',dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        foreground_tty=os.isatty(1),protocol=str(planfile),protocol_sha256=sha(planfile)))
    write(record/'state.json',dict(status='starting',run_id=plan['run_id'],completed_updates=0,clips=[]))
    from evaluate_standup_reference import Tee
    with (record/'training.log').open('a',buffering=1) as output:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,output),Tee(olderr,output)
        try:
            from mjlab.scripts import train
            base=train.load_runner_cls(plan['task']);train.load_runner_cls=lambda task:runner_class(base,plan,record,metadata)
            train.launch_training(plan['task'],cfg)
        except BaseException as error:
            state=read(record/'state.json');state.update(status='stopped-review-required',error=repr(error))
            write(record/'state.json',state);raise
        finally:sys.stdout,sys.stderr=oldout,olderr


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('protocol');run(parser.parse_args().protocol)
