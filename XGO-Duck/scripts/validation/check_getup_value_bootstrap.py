"""CPU admission: finite returns, group split and exact native contact pairs.

Use existing saved physical states for the contact fixtures. mj_forward only
updates geometry; no learning, rendering, integration or CUDA is involved.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from getup_contact_training import (split_worlds,discounted_returns,train_reward_scale,
    remaining_time,late_contact_mask,prepare,POSES)
from common_scene_contract import model_arrays,compare_arrays
from record_io import write


def check(plan):
    from types import SimpleNamespace
    from mjlab.scene import Scene
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    split=split_worlds();assert split==plan['split']
    train,valid=split['train_ids'],split['validation_ids']
    for group in split['groups'].values():
        assert len(set(group)&set(train))==240 and len(set(group)&set(valid))==16
    actual=discounted_returns(torch.ones(3,2),.5)
    torch.testing.assert_close(actual,torch.tensor([[1.75]*2,[1.5]*2,[1.]*2],dtype=torch.float64),atol=0,rtol=0)
    values=torch.arange(1,3073,dtype=torch.float64).reshape(3,1024)
    scale=train_reward_scale(values,train);values[:,valid]*=100000
    assert train_reward_scale(values,train)==scale
    clock=remaining_time(SimpleNamespace(episode_length_buf=torch.tensor([0,100,399,400,401]),step_dt=.02))
    torch.testing.assert_close(clock.flatten(),torch.tensor([1.,.75,.0025,0.,0.]),atol=3e-8,rtol=0)
    mask=late_contact_mask(torch.tensor([2.,2.02,3.,3.,3.]),torch.tensor([-1.,-1.,-.6,-1.,-1.]),
        torch.tensor([True,True,True,False,True]),torch.tensor([True,True,True,True,False]))
    assert mask.tolist()==[False,True,False,False,False]
    cfg,display,_=prepare(plan);scene=Scene(cfg.env.scene,device='cpu');model=scene.compile()
    with np.load(ROOT/'records/reports/common-plant-screen/getup-cpu-scene-fields.npz') as f:
        compare_arrays(model_arrays(model),{k:f[k] for k in f.files})
    pairs=[];sensor_ids=[]
    head=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/head_collision')
    legs={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+name) for name in
        ('left_foot_collision','right_foot_collision','left_hip_yaw_collision','right_hip_yaw_collision')}
    for name in ('head_feet_contact','head_hips_contact'):
        sensor=scene[name]
        for slot in sensor._slots:
            sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,slot.sensor_name)
            assert model.sensor_objtype[sid]==model.sensor_reftype[sid]==int(mujoco.mjtObj.mjOBJ_GEOM)
            pair=(int(model.sensor_objid[sid]),int(model.sensor_refid[sid]))
            assert pair[0] in legs and pair[1]==head
            pairs.append(pair);sensor_ids.append(sid)
    assert len(pairs)==4 and {p[0] for p in pairs}==legs
    study=read(ROOT/'records/reports/getup-contact-release-recovery/analysis.json')
    source=ROOT/study['arms']['full']['record']/'physical-contact-trace.npz'
    with np.load(source) as f:
        qpos=f['qpos'].copy();qvel=f['qvel'].copy();counts=f['head_contact_pair_counts'].copy()
    negative=np.argwhere((counts[:,:,2]>0)&(counts[:,:,:2].sum(axis=2)==0))
    assert len(negative)>0
    negative_index=negative[len(negative)//2].tolist()
    fixtures=[];data=mujoco.MjData(model)
    for label,(t,e),expected in (('folded-head-leg',(499,15),True),('floor-only',negative_index,False)):
        data.qpos[:]=qpos[t,e];data.qvel[:]=qvel[t,e];saved=data.qpos.copy()
        mujoco.mj_forward(model,data);np.testing.assert_array_equal(data.qpos,saved)
        found=[bool(data.sensordata[model.sensor_adr[s]]) for s in sensor_ids]
        assert any(found)==expected,(label,found)
        fixtures.append(dict(label=label,trace_index=int(t),world=int(e),found=found,expected=expected))
    assert not torch.cuda.is_initialized()
    return dict(status='passed',finite_return_fixture=True,clock_fixture=True,phase_gate_fixture=True,
        validation_excluded_from_scale=True,train_worlds=len(train),validation_worlds=len(valid),
        world_groups={name:dict(train=240,validation=16) for name in POSES},
        complete_scene_physical_fields_exact=True,native_pairs=pairs,contact_fixtures=fixtures,
        source_trace_sha256=sha(source),actor_input_unchanged=True,critic_clock_only=True,
        physics_steps=0,optimizer_steps=0,cuda_initialized=False,script_sha256=sha(Path(__file__)))


if __name__=='__main__':
    from getup_value_bootstrap import make_plan
    result=check(make_plan())
    write(ROOT/'records/reports/getup-adaptation/value-cpu-preflight.json',result)
    print(__import__('json').dumps(result,indent=2),flush=True)
