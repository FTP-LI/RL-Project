"""Freeze and run the full bridge-disabled stand-roll-stand-forward route."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

import run_continuous_roll_stand_walk as base
from record_io import write
from s1_reference_inputs import WORK, read, sha


DRAFT = ROOT / 'configs/experiments/router_full_sequence_s1218_draft.json'
PLAN = ROOT / 'configs/experiments/router_full_sequence_s1218.json'
OUT = ROOT / 'records/reports/continuous-roll-full-sequence-s1218'
PAGE = ROOT / 'records/router-full-sequence-s1218-live.html'


def redirect():
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = DRAFT, PLAN, OUT, PAGE


def freeze():
    redirect(); base.freeze()
    p = read(PLAN); p.update(stage='S5-router-full-sequence', cross_model_bridge_controls=0,
        diagnostic_scope='Full no-reset stand→roll→stand→forward route with bridge disabled and original forward scoring.',
        next_gate='Review roll latch, stand handoff and the original forward gate before any perturbation.')
    p['frozen_sha256'][str(Path(__file__).relative_to(ROOT)).replace('\\','/')] = sha(Path(__file__))
    write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),physical_substeps=0,
        shared_compiled_scene=True,seed=p['seed'],maximum_controls=p['maximum_controls'],bridge_controls=0,
        observation=PAGE.relative_to(ROOT).as_posix()))
    write(OUT/'windows-entry.json',dict(status='prepared-observer-entry',protocol_sha256=sha(PLAN),
        page=PAGE.relative_to(ROOT).as_posix(),browser_visual_verified=False))
    print('ROUTER_FULL_SEQUENCE_FROZEN',p['run_id'],sha(PLAN),flush=True)


def checked():
    redirect(); p=read(PLAN)
    assert p['status']=='frozen-before-dynamics' and p['stage']=='S5-router-full-sequence'
    assert p['cross_model_bridge_controls']==0 and p['maximum_controls']==1550
    assert p['landing_stand_controls']==50 and p['forward_controls']==1000
    assert not p['hardware_io'] and not p['heldout_used'] and not p['learning_updates']
    for file,digest in p['frozen_sha256'].items(): assert sha(ROOT/file)==digest, file
    for file,digest in p['training_source_sha256'].items(): assert sha(WORK/file)==digest, file
    for file,digest in p['library_source_sha256'].items(): assert sha(Path('/')/file)==digest, file
    return p


def run(zero):
    redirect(); base.checked=checked; base.run(zero)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('freeze','zero','sample')); args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:
        import torch
        with torch.no_grad():run(args.command=='zero')


if __name__=='__main__':main()
