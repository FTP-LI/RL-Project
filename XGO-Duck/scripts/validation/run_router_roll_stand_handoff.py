"""Run the bridge-disabled roll-to-stand handoff screen without learning."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

from action_video import ActionVideo
from common_scene_contract import compare_arrays, install_guard, model_arrays
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback, check_geom_layout
from roll_metrics import rotation
from run_continuous_roll_stand_walk import PolicyControl, actors, capture, configuration
from roll_stand_walk_sequence import FAILED, FORWARD, ROLL, STAND_LANDING, Sequence
from s1_reference_inputs import ROOT as SOURCE_ROOT, read, sha
from training_watch import Snapshot
import training_live_page as live


TZ = ZoneInfo('Asia/Shanghai')
BASE = ROOT / 'configs/experiments/router_bridge_disabled_boundary_s1216.json'
DRAFT = ROOT / 'configs/experiments/router_roll_stand_handoff_s1217_draft.json'
PLAN = ROOT / 'configs/experiments/router_roll_stand_handoff_s1217.json'
OUT = ROOT / 'records/reports/continuous-roll-stand-handoff-s1217'
STARTUP = ROOT / 'records/reports/common-plant-screen/startup'
PAGE = ROOT / 'records/router-roll-stand-handoff-s1217-live.html'


def freeze() -> None:
    assert DRAFT.exists() and not PLAN.exists() and not OUT.exists()
    draft = read(DRAFT); base = read(BASE)
    assert draft['status'] == 'draft-not-executable' and base['status'] == 'frozen-before-dynamics'
    p = deepcopy(base)
    p.update(status='frozen-before-dynamics', stage='S5-router-roll-stand-handoff-screen', seed=int(draft['seed']),
             initial_stand_controls=50, roll_controls=450, landing_stand_controls=50, forward_controls=0,
             maximum_controls=550, maximum_control_transitions=70400, maximum_physics_substeps=2750,
             cross_model_bridge_controls=0, learning_updates=0, hardware_io=False, heldout_used=False,
             automatic_retry=False, automatic_extension=False, diagnostic_scope='One no-reset roll-to-stand handoff with bridge disabled.',
             next_gate='Review roll latch, stand hold and forward-entry counts before any forward extension.')
    p['roll']['seed'] = int(draft['seed']); p['roll']['episodes'] = int(draft['worlds'])
    p['run_id'] = f'duck_router-roll-stand-handoff_development_s{p["seed"]}_{datetime.now(TZ).strftime("%Y%m%dT%H%M%S")}'
    files = [DRAFT, BASE, Path(__file__), ROOT/'scripts/validation/run_continuous_roll_stand_walk.py',
             ROOT/'scripts/validation/roll_stand_walk_sequence.py', ROOT/'scripts/validation/roll_landing_latch.py',
             ROOT/'scripts/validation/roll_metrics.py', ROOT/'scripts/validation/common_scene_contract.py',
             ROOT/'scripts/validation/roll_endpoint_readback_capture.py', ROOT/'scripts/training/action_video.py',
             ROOT/'scripts/training/training_watch.py', ROOT/'scripts/training/training_live_page.py',
             ROOT/'configs/environments/action_recording.json']
    p['frozen_sha256'] = {f.relative_to(ROOT).as_posix(): sha(f) for f in files}
    cfg = configuration(p)
    from mjlab.scene import Scene
    model = Scene(cfg.scene, device='cpu').compile(); check_geom_layout(model)
    with np.load(STARTUP/'roll/scene-static-fields.npz') as data: compare_arrays(model_arrays(model), {key:data[key] for key in data.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True)
    write(PLAN, p)
    write(OUT/'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), physical_substeps=0,
        shared_compiled_scene=True, seed=p['seed'], maximum_controls=p['maximum_controls'], bridge_controls=0,
        observation=PAGE.relative_to(ROOT).as_posix()))
    write(OUT/'windows-entry.json', dict(status='prepared-observer-entry', protocol_sha256=sha(PLAN),
        page=PAGE.relative_to(ROOT).as_posix(), browser_visual_verified=False))
    print('ROUTER_HANDOFF_FROZEN', p['run_id'], sha(PLAN), flush=True)


def checked() -> dict:
    p = read(PLAN)
    assert p['status']=='frozen-before-dynamics' and p['stage']=='S5-router-roll-stand-handoff-screen'
    assert p['maximum_controls']==550 and p['landing_stand_controls']==50 and p['cross_model_bridge_controls']==0
    assert not p['hardware_io'] and not p['heldout_used'] and not p['learning_updates']
    for file,digest in p['frozen_sha256'].items(): assert sha(ROOT/file)==digest, file
    return p


def run(zero: bool) -> None:
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    p = checked(); task=read(ROOT/'project.json')['staged_training_request']
    assert task['current_stage']==p['stage'] and task['current_protocol_frozen'] and not task['pause_requested']
    assert ROOT/task['current_protocol']==PLAN
    folder = OUT/'zero' if zero else ROOT/'records/evaluations'/p['run_id']; folder.mkdir(parents=True,exist_ok=False)
    write(folder/'protocol.json',p); cfg=configuration(p); dump_yaml(folder/'env.yaml',asdict(cfg))
    configure_torch_backends(allow_tf32=False); cls=install_guard(p['roll'],folder,STARTUP,'rollout'); env=cls(cfg=cfg,device='cuda:0',render_mode=None)
    endpoint=None
    try:
        env.seed(p['seed']); obs,_=env.reset(); networks,homes,sim_home,limits=actors(env,obs,p,folder)
        endpoint=EndpointReadback(env,folder); endpoint.capture(); assert not endpoint.first_mismatches
        if zero:
            endpoint.save(1); print('ROUTER_HANDOFF_ZERO',p['seed'],flush=True); return
        robot=env.scene['robot']; joints,names=robot.find_joints(p['walking']['joint_names'],preserve_order=True); feet,sites=robot.find_sites(('left_foot','right_foot'),preserve_order=True); head=robot.find_bodies('jaw_soft')[0][0]
        assert list(names)==p['walking']['joint_names'] and list(sites)==['left_foot','right_foot']
        tolerance=np.deg2rad(p['roll']['actual_joint_limit_tolerance_deg']); limits_np=limits.detach().cpu().numpy(); sequence=Sequence(p); policy=PolicyControl(p,networks,homes,sim_home,limits); alive=np.ones(p['worlds'],dtype=bool)
        states=[]; commands=[]; snapshot=Snapshot(env,folder); video=ActionVideo(snapshot,'switch',folder,p['maximum_controls']*.02); log=folder/'execution.log'; started=monotonic()
        view=dict(status='running',run_id=p['run_id'],purpose='官方组件滚翻 → 站立接力 · bridge disabled',controls=0,maximum_controls=p['maximum_controls'],transitions=0,learning_updates=0,updates=[])
        def publish(k,status='running',frame=None):
            view.update(status=status,updated_at=datetime.now(TZ).isoformat(),controls=k,transitions=k*p['worlds'])
            if frame:view['frame']=frame
            if video.clips and video.clips[0].get('complete_clip'):view['latest_clip']=video.clips[0]['file']
            with patch.object(live,'PAGE',PAGE):live.publish(folder,view)
        def record(k,frame,changed):
            r=rotation(frame['quaternion']); tilt=np.rad2deg(np.arccos(np.clip(r[:,2,2],-1,1)))
            stable=(frame['active']&frame['foot_contact'].all(axis=-1)&frame['joint_envelope_ok']&(tilt<=p['roll']['criteria']['tilt_max_deg'])&(np.abs(frame['height']-p['roll']['criteria']['nominal_height_m'])<=p['roll']['criteria']['height_tolerance_m'])&(np.linalg.norm(frame['angular_velocity'],axis=-1)<=p['roll']['criteria']['angular_speed_max_rad_s']))
            states.append(dict(control=k,phase=sequence.phase.copy(),phase_changed=changed.copy(),active=frame['active'].copy(),stable=stable,dwell=sequence.dwell.copy(),stand_at=sequence.stand_at.copy(),forward_at=sequence.forward_at.copy(),height=frame['height'].copy(),tilt_deg=tilt,angular_velocity=frame['angular_velocity'].copy(),foot_contact=frame['foot_contact'].copy(),head_contact=frame['head_contact'].copy(),head_penetration_m=frame['head_penetration_m'].copy(),joint_envelope_ok=frame['joint_envelope_ok'].copy()))
        frame=capture(env,endpoint,alive,joints,head,feet,limits_np,tolerance); changed=sequence.update(0,frame); record(0,frame,changed); video.sample(); image=snapshot.capture(); publish(0,frame=image['file'])
        for k in range(1,p['maximum_controls']+1):
            phase_before=sequence.phase.copy(); base_obs=obs['actor'].clone(); target,saved=policy.next(k,base_obs,sequence); obs,_,term,timeout,_=env.step(target-sim_home); received=env.action_manager.get_term('joint_pos')._processed_actions; torch.testing.assert_close(received,target,rtol=0,atol=1e-6); alive&=~(term|timeout).detach().cpu().numpy(); commands.append(dict(saved,observed=base_obs.detach().cpu().numpy().copy(),terminated=term.detach().cpu().numpy().copy(),truncated=timeout.detach().cpu().numpy().copy(),actual_motor_target=received.detach().cpu().numpy().copy())); endpoint.capture(); assert not endpoint.first_mismatches
            frame=capture(env,endpoint,alive,joints,head,feet,limits_np,tolerance); changed=sequence.update(k,frame); record(k,frame,changed); video.sample()
            if k%50==0:
                image=snapshot.capture(); row=f'ROUTER_HANDOFF {k}/{p["maximum_controls"]} active {alive.sum()} roll {(sequence.phase==ROLL).sum()} landing {(sequence.phase==STAND_LANDING).sum()} forward {(sequence.phase==FORWARD).sum()} failed {(sequence.phase==FAILED).sum()}'; print(row,flush=True); log.open('a',encoding='utf-8').write(row+'\n'); view['updates'].append(dict(control=k,active=int(alive.sum()),roll=int((sequence.phase==ROLL).sum()),landing=int((sequence.phase==STAND_LANDING).sum()),forward=int((sequence.phase==FORWARD).sum()),failed=int((sequence.phase==FAILED).sum()),transitions_per_second=k*p['worlds']/max(monotonic()-started,1e-9))); publish(k,frame=image['file'])
        arrays={key:np.asarray([row[key] for row in states]) for key in states[0]}; np.savez_compressed(folder/'handoff.npz',**arrays); np.savez_compressed(folder/'commands.npz',**{key:np.asarray([row[key] for row in commands]) for key in commands[0]}); endpoint.save(p['maximum_controls']+1)
        analysis=dict(status='completed-roll-stand-handoff-screen',protocol=PLAN.relative_to(ROOT).as_posix(),run=folder.relative_to(ROOT).as_posix(),seed=p['seed'],worlds=p['worlds'],control_steps=p['maximum_controls'],transitions=p['maximum_control_transitions'],physical_substeps=int(env._sim_step_counter),learning_updates=0,hardware_io=False,roll_entry_worlds=int(np.any(arrays['phase']==ROLL,axis=0).sum()),landing_worlds=int((arrays['stand_at'][-1]>=0).sum()),forward_entry_worlds=int((arrays['forward_at'][-1]>=0).sum()),phase_at_end={str(v):int((arrays['phase'][-1]==v).sum()) for v in (FAILED,0,ROLL,STAND_LANDING,FORWARD)},trace=folder.joinpath('handoff.npz').relative_to(ROOT).as_posix(),commands=folder.joinpath('commands.npz').relative_to(ROOT).as_posix(),video=video.clips,interpretation='No reset is used at the roll-to-stand handoff. The forward count is only a phase-entry signal; a full forward gate remains a separate stage.',no_promotion=True); write(OUT/'analysis.json',analysis); publish(p['maximum_controls'],status='completed',frame=image['file'])
    finally:
        if endpoint is not None: endpoint.save(len(endpoint.rows))
        if 'video' in locals():
            try:video.close()
            except BaseException as exc:write(folder/'video-close-error.json',dict(status='captured-and-continued',error=repr(exc)))
        if 'snapshot' in locals():snapshot.close()
        env.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('freeze','zero','sample')); args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:
        with torch.no_grad():run(args.command=='zero')


if __name__=='__main__':main()
