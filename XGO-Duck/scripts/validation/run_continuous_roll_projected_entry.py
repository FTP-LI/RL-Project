"""Retest the complete continuous sequence with the admitted roll target."""
import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write
from roll_entry_impossibility import maximum_possible
from roll_stand_walk_sequence import FAILED, ROLL
import run_continuous_roll_stand_walk as baseline


PLAN = ROOT/'configs/experiments/continuous_roll_projected_entry.json'
DRAFT = PLAN.with_name('continuous_roll_projected_entry_draft.json')
OUT = ROOT/'records/reports/continuous-roll-projected-entry'
PAGE = ROOT/'records/roll-projected-entry-live.html'
PARENT = ROOT/'configs/experiments/continuous_roll_stand_walk.json'
BASE_COLLECT = baseline.collect


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft, parent = read(DRAFT), baseline.checked()
    assert draft['status']=='draft-not-executable' and ROOT/draft['parent_protocol']==PARENT
    assert all(not draft[key] for key in ('heldout_used','hardware_io','learning_updates','automatic_retry','automatic_extension'))
    source = ROOT/draft['paired_entry_evidence']
    proof = read(source)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    assert proof['same_first_input_raw_and_delayed_motor'] and proof['worlds']==parent['worlds']==128
    projected = proof['branches']['projection-only']
    assert projected['meets_development_gate'] and projected['certified']>=parent['minimum_roll_completion']
    assert projected['certified']>=proof['branches']['original-roll-target']['certified']
    assert not proof['branches']['projection-and-slew']['meets_development_gate']
    for file,digest in proof['source_sha256'].items():
        assert sha(ROOT/file)==digest,file
    for key in ('worlds','maximum_controls','maximum_control_transitions','maximum_physics_substeps'):
        assert draft[key]==parent[key],key
    seed = draft['seed']
    assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    p = dict(parent,status='frozen-before-dynamics',stage=draft['stage'],seed=seed,
        roll=dict(parent['roll'],seed=seed),
        run_id=f'duck_switch_roll-projected-entry_development_s{seed}_{datetime.now(baseline.TZ):%Y%m%dT%H%M%S}',
        projected_roll_without_slew=True,paired_entry_evidence=draft['paired_entry_evidence'])
    sources = dict(parent['frozen_sha256'])
    paths = [DRAFT,PARENT,source,Path(__file__),
             ROOT/'scripts/validation/roll_entry_impossibility.py',
             ROOT/'scripts/validation/verify_roll_entry_isolation.py',
             ROOT/'scripts/validation/verify_continuous_roll_projected_entry.py']
    sources.update({file.relative_to(ROOT).as_posix():sha(file) for file in paths})
    p['frozen_sha256'] = sources
    OUT.mkdir()
    write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),paired_entry_sha256=sha(source),
        admitted_paired_score=projected['certified'],physical_substeps=0,
        observation=PAGE.relative_to(ROOT).as_posix()))
    print('ROLL_PROJECTED_ENTRY_FROZEN',p['run_id'],sha(PLAN),flush=True)


def checked():
    p = read(PLAN)
    assert p['status']=='frozen-before-dynamics' and p['projected_roll_without_slew'] and not p['hardware_io']
    for root,key in ((ROOT,'frozen_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for path,digest in p[key].items():
            assert sha(root/path)==digest,path
    return p


class ProjectedRollPolicy(baseline.PolicyControl):
    def next(self,step,obs,sequence):
        target,row = super().next(step,obs,sequence)
        rolling = row['phase_used']==ROLL
        # Remove only the entry slew. The joint-envelope projection and raw
        # model history remain identical to the corresponding isolation branch.
        roll_request = torch.from_numpy(row['requested_targets'][rolling]).to(target.device)
        target[rolling] = torch.clamp(roll_request,min=self.limits[:,0],max=self.limits[:,1])
        row['applied_targets'][rolling] = baseline.cpu(target[rolling])
        row['bridge'][rolling] = False
        self.previous = target.clone()
        return target,row


class SequenceImpossibility(RuntimeError):
    pass


def collect(env,obs,p,folder,*args):
    class SequenceWithBudget(baseline.Sequence):
        def __init__(self,spec):
            super().__init__(spec)
            self.stop = None

        def update(self,step,state):
            changed = super().update(step,state)
            if self.roll_begin<=step<=self.roll_end:
                possible,certified = maximum_possible(self.roll_latch,p['roll_controls'])
                if possible<p['minimum_roll_completion']:
                    self.stop = dict(reason='original-roll-threshold-impossible',control=step,possible=possible,
                                     certified=certified,required=p['minimum_roll_completion'])
            # A full sequence may still pass if its 20 s forward scoring window
            # already ended before a later diagnostic termination.
            possible = ((self.phase!=FAILED) | ((self.forward_at>=0) &
                        (step>self.forward_at+p['forward_controls']))).sum()
            if self.stop is None and possible<p['minimum_continuous_successes']:
                self.stop = dict(reason='sequence-threshold-impossible',control=step,possible=int(possible),
                                 required=p['minimum_continuous_successes'])
            return changed

    active, cameras = [], []
    action_video = baseline.ActionVideo

    def make_camera(*args,**kwargs):
        cameras.append(action_video(*args,**kwargs))
        return cameras[-1]

    def sequence(spec):
        active.append(SequenceWithBudget(spec))
        return active[0]

    original_step = env.step

    def step(target):
        if active[0].stop is not None:
            assert active[0].stop['control']==env.common_step_counter
            if cameras[0].writer is not None:
                cameras[0]._sample_terminal()
            raise SequenceImpossibility(active[0].stop['reason'])
        result = original_step(target)
        k = env.common_step_counter
        if k%50==25:
            view = read(folder/'state.json')
            view.update(controls=k,transitions=k*p['worlds'],
                frame=cameras[0].snapshot.capture()['file'],updated_at=datetime.now(baseline.TZ).isoformat())
            with patch.object(baseline.live,'PAGE',PAGE):
                baseline.live.publish(folder,view)
        return result

    with (patch.object(baseline,'Sequence',sequence),
          patch.object(baseline,'PolicyControl',ProjectedRollPolicy),
          patch.object(baseline,'ActionVideo',make_camera),patch.object(env,'step',step)):
        try:
            BASE_COLLECT(env,obs,p,folder,*args)
        except SequenceImpossibility:
            stop = active[0].stop
            write(folder/'planned-auto-stop.json',dict(stop,protocol_sha256=sha(PLAN),
                completed_controls=env.guard_steps,completed_transitions=env.guard_steps*p['worlds'],
                physical_substeps=env._sim_step_counter))
            print('ROLL_PROJECTED_ENTRY_AUTO_STOP',stop,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','zero','sample'))
    args = parser.parse_args()
    if args.command=='freeze':
        return freeze()
    spec = checked()
    with patch.multiple(baseline,PLAN=PLAN,OUT=OUT,PAGE=PAGE,checked=lambda:spec,collect=collect),torch.no_grad():
        baseline.run(args.command=='zero')


if __name__=='__main__':
    main()
