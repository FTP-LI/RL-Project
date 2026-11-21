"""Run the admitted causal direction rule against two restored native references."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from calibrate_walk_forward_yaw_prediction import restore
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver
import run_walk_forward_stance_response as previous


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_direction_pilot.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
OUT = previous.STAGE/"yaw-direction-pilot"
PAGE = ROOT/"records/walk-forward-yaw-direction-pilot-live.html"
ROLES = ("reference-a","reference-b","causal-feedback")
LABELS = dict(zip(ROLES,("原反馈一","原反馈二","方向反馈"),strict=True))
native, rollout, feedback = previous.native,previous.rollout,previous.feedback


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    source, draft = previous.checked(), read(DRAFT)
    assert draft["status"] == "draft-not-executable" and tuple(draft["branches"]) == ROLES
    assert not any(draft[k] for k in ("heldout_used","hardware_io","learning_updates","automatic_retry","physical_substeps_authorized"))
    model, evidence = (ROOT/draft[key] for key in ("direction_predictor","previous_response"))
    fitted, measured = read(model),read(evidence)
    fit_plan = ROOT/"configs/experiments/common_walk_forward_yaw_direction_fit.json"
    assert fitted["protocol_sha256"] == sha(fit_plan) and fitted["admitted_for_new_mixed_pulse"]
    assert measured["protocol_sha256"] == sha(previous.PLAN) and measured["both_directions_admitted"]
    assert not measured["admitted_for_sustained_feedback_preflight"]
    seed, prefix, h = (draft[k] for k in ("seed","shared_controls","controls_in_each_complete_short_episode"))
    assert (seed,prefix,h) == (1152,84,200)
    suffix, total = h-prefix,prefix+len(ROLES)*(h-prefix)
    n = source["cases"][str(seed)]["environment"]["parallel_worlds"]
    assert total*n == draft["maximum_transitions"] and total*5 == draft["maximum_batched_substeps"]
    assert source["feedback"]["maximum_yaw_input_rad_s"] == draft["maximum_total_input_rad_s"] == .12
    assert source["feedback"]["maximum_joint_correction_rad"] == draft["maximum_joint_correction_rad"] == .03
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    name = f"duck_walk_yaw-direction-pilot_s{seed}_{stamp}"
    folder = ROOT/"records/evaluations"/name
    p = dict(source["cases"][str(seed)]["environment"],run_id=name,stage=draft["stage"],maximum_control_steps=h)
    spec = deepcopy(source)
    spec.update(stage=p["stage"],cases={str(seed):dict(seed=seed,record=native.rel(folder),
        page=native.rel(PAGE),environment=p)},arms=list(ROLES),index=native.rel(PAGE),
        shared_prefix_controls=prefix,controls_per_suffix=suffix,maximum_controls_per_case=total,
        maximum_transitions=total*n,maximum_batched_substeps=total*5,physical_substeps_authorized=total*5,
        direction_pilot=draft,direction_predictor=native.rel(model))
    for path in (DRAFT,previous.PLAN,model,evidence,fit_plan,ROOT/read(fit_plan)["base_protocol"]):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY,ENTRY.with_name("observe_walk_forward_yaw_direction_pilot.py"),
                 ENTRY.with_name("verify_walk_forward_yaw_direction_pilot.py"),
                 ENTRY.with_name("calibrate_walk_forward_yaw_prediction.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    rollout.check_sources(dict(spec,environment=p))
    prediction = restore(fitted["model"], read(ROOT/read(fit_plan)["base_protocol"]))
    assert prediction(np.zeros((1,4*48),np.float32)).shape == (1,) and not torch.cuda.is_initialized()
    OUT.mkdir()
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"preflight.json",dict(status="passed-direction-pilot-cpu-freeze",
        protocol_sha256=sha(PLAN),physical_substeps=0,learning_updates=0))
    (folder/"README.md").write_text("# 连续方向反馈短试验\n\n"
        f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{PAGE.name})\n",encoding="utf-8")
    native.publish(folder,PAGE,dict(status="starting",run_id=name,purpose="连续方向反馈 · 四秒诊断",
        controls=0,maximum_controls=total,transitions=0,learning_updates=0,updates=[]))
    print("FORWARD_DIRECTION_PILOT_FROZEN",sha(PLAN),native.rel(folder),total*n,flush=True)


def checked():
    spec,proof = read(PLAN),read(OUT/"preflight.json")
    assert spec["stage"] == "S2-forward-yaw-direction-pilot" and proof["status"] == "passed-direction-pilot-cpu-freeze"
    assert proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used","hardware_io","learning_updates","automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec,environment=item["environment"]))
    return spec


def components(spec):
    support, observer_plan = (read(ROOT/spec[key]) for key in ("observer_receipt","observer_protocol"))
    fit_plan = read(ROOT/"configs/experiments/common_walk_forward_yaw_prediction.json")
    result = read(ROOT/spec["direction_predictor"])
    thresholds = np.array([np.inf,*[result["thresholds_logit"][side] for side in ("left-only","right-only")]])
    return (StanceObserver(support["model"],observer_plan["minimum_score_separation"]),
            restore(result["model"],fit_plan),thresholds,fit_plan["window"],fit_plan["observation_columns"])


def collect(env,actor,spec,p,record,state,desired,progress):
    prefix = spec["shared_prefix_controls"]
    rule, warmup = spec["direction_pilot"],spec["phase"]["warmup_controls"]
    observer,predict,thresholds,window,width = components(spec)
    base_controller,base_law = feedback.Controller,feedback.ActuatorYawFeedback
    base_collect,base_snapshot = feedback.original_collect,rollout.WalkingSnapshot
    context,rows,coverage = {},[],{}
    branch = "shared-prefix"

    class MatchedRestore:
        def __init__(self,env,limit):
            self.snapshot = base_snapshot(env,limit)
            self.expected = self.snapshot.expected
            self.initial = True

        def verify(self):
            if self.initial:
                self.initial = False
                return self.snapshot.verify()
            return self.restore()

        def restore(self):
            return self.snapshot.restore()

    class HistoryController(base_controller):
        def __init__(self,plan,settings,desired):
            super().__init__(plan,settings,desired)
            self.history = np.zeros((len(desired),window,width),np.float32)
            self.age = np.zeros(len(desired),np.int32)

        def map(self,observed,step,candidate,active):
            self.history[:,:-1] = self.history[:,1:].copy()
            self.history[:,-1] = observed[:,:width]
            self.age += 1
            ids = np.flatnonzero(self.forward & active)
            context.update(step=step,ids=ids,current=observed[ids],history=self.history[ids],ready=self.age[ids] >= window)
            return super().map(observed,step,candidate,active)

        def reset(self,ids):
            super().reset(ids)
            self.history[ids] = 0
            self.age[ids] = 0

        def snapshot(self):
            return super().snapshot(),self.history.copy(),self.age.copy()

        def restore(self,saved):
            old,history,age = saved
            super().restore(old)
            self.history[:],self.age[:] = history,age

    class DirectionLaw(base_law):
        def commands(self,features):
            baseline = super().commands(features).astype(np.float32)
            ids,step = context["ids"],context["step"]
            phase = observer.predict(context["current"])
            logit = predict(context["history"].reshape(len(ids),window*width))
            eligible = context["ready"] & (abs(logit) >= thresholds[phase])
            requested = np.clip(baseline-np.sign(logit)*rule["maximum_extra_input_rad_s"],-self.cap,self.cap)
            adjusted = np.where(eligible,requested,baseline)
            if step == prefix+1:
                contact = rollout.cpu(env.scene["feet_ground_contact"].data.found).reshape(len(desired),2,-1).any(axis=2)[ids]
                actual = contact[:,0].astype(int)+2*contact[:,1]
                changed = adjusted != baseline
                counts = [int(((phase == i) & changed & (actual == i)).sum()) for i in (1,2)]
                wrong = int((changed & (phase != actual)).sum())
                coverage[branch] = dict(changed_verified_per_side=counts,false_side=wrong)
                write(record/"direction-live-coverage.json",coverage)
                assert min(counts) >= rule["minimum_first_step_changed_worlds_per_side"] and wrong == 0,"Insufficient direction coverage"
            output = adjusted if branch == ROLES[-1] else baseline
            index = ROLES.index(branch) if branch in ROLES else -1
            rows.append((np.full(len(ids),index),np.full(len(ids),step),ids.copy(),phase,logit,baseline,output))
            return output

    def write_fork(path,data):
        nonlocal branch
        write(path,data)
        if path == record/"fork-readback.json":
            branch = next(reversed(data))
            state["branch"] = branch
            progress()

    def moved_fork(env,actor,local,record,state,desired):
        class OriginalWarmup(rollout.ForwardPhase):
            def map(self,observed,step,candidate,active):
                return super().map(observed,step,step > warmup,active)

        physical = dict(local,phase=dict(local["phase"],warmup_controls=prefix))
        with patch.object(rollout,"ForwardPhase",OriginalWarmup):
            base_collect(env,actor,physical,record,state,desired)

    try:
        with (
            patch.multiple(feedback,Controller=HistoryController,ActuatorYawFeedback=DirectionLaw,original_collect=moved_fork),
            patch.multiple(rollout,ROLES=ROLES,available=progress,publish=lambda *_:progress(),write=write_fork,
                           WalkingSnapshot=MatchedRestore),
        ):
            feedback.collect(env,actor,dict(spec,environment=p,maximum_physics_substeps=spec["maximum_batched_substeps"]),
                             record,state,desired)
    finally:
        if rows:
            fields = ("role","step","world","phase","logit","baseline_yaw","applied_yaw")
            np.savez_compressed(record/"direction-readback.npz",
                **dict(zip(fields,map(np.concatenate,zip(*rows,strict=True)),strict=True)))
    assert set(coverage) == set(ROLES) and all(row == coverage[ROLES[0]] for row in coverage.values())


def sample(spec,key):
    original = native.publish

    def publish(record,page,view):
        view["purpose"] = view["purpose"].replace("两秒诊断录像","四秒诊断录像")
        original(record,page,view)

    with (
        patch.multiple(previous,PLAN=PLAN,OUT=OUT,PAGE=PAGE,ROLES=ROLES,LABELS=LABELS,collect=collect),
        patch.object(native,"publish",publish),
    ):
        previous.sample(spec,key)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        with patch.multiple(native,OUT=OUT,ENTRY=ENTRY):
            native.run(checked())
    else:
        sample(checked(),args.case)


if __name__ == "__main__":
    main()
