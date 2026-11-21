"""Kinematic contact audit of saved recovery failures, CPU only.

Use recorded orientation, height and all encoder positions. Root x/y were not
recorded, so place the robot at the origin of the same infinite flat floor.
Run mj_forward at those poses without integrating or producing new rollouts.
Candidate contact pairs are geometric diagnostics, not captured GPU forces or
proof of causation. Keep the source and common collision masks separate.
Validation: exact scene fields against the frozen archive; joint addresses and
root pose round trip; source/common field differences recorded; CUDA stays off.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import mujoco
import torch
from mjlab.scene import Scene
from common_plant_screen import prepare
from evaluate_standup_reference import config as original_config
from common_scene_contract import model_arrays,compare_arrays
from review_common_plant_screen import ROOT,OUT,read,sha,load


def compile_model(cfg):
    return Scene(cfg.scene,device='cpu').compile()


def main():
    assert not torch.cuda.is_initialized()
    review=read(OUT/'failure-review.json');proof=read(OUT/'getup/common-verification.json')
    folder=ROOT/proof['record'];plan=read(folder/'protocol.json');report=read(folder/'evaluation.json')
    common=compile_model(prepare('getup',plan));source=compile_model(original_config(plan))
    compare_arrays(model_arrays(common),load(OUT/'getup-cpu-scene-fields.npz'))
    fields={}
    for key,a in model_arrays(common).items():
        b=model_arrays(source)[key]
        if not np.array_equal(a,b):fields[key]=dict(common_shape=list(a.shape),source_shape=list(b.shape),
            maximum_absolute_difference=float(np.max(abs(a-b))) if a.shape==b.shape and a.size else None)
    # All physical coordinates must refer to the same articulated structure.
    assert common.nq==source.nq==21 and common.nv==source.nv==20
    names=report['joint_names'];rows=[]
    for condition in review['getup']['conditions']:
        pose=condition['condition'];t=load(folder/f'{pose}-trace.npz')
        failed={e['env_index'] for e in condition['no_timely_hold']}
        targets=[(i,'no-timely-hold') for i in sorted(failed)]+[(next(i for i in range(32) if i not in failed),'timely-hold-example')]
        for index,label in targets:
            poses=[]
            for model,kind in ((source,'source-masks'),(common,'common-masks')):
                data=mujoco.MjData(model)
                ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
                assert min(ids)>=0
                addresses=model.jnt_qposadr[ids]
                root=int(np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0]);root_address=int(model.jnt_qposadr[root]);root_body=int(model.jnt_bodyid[root])
                pairs=Counter();late=Counter();ground=Counter();minimum={};samples=0
                for k in range(100,len(t['time']),10):
                    data.qpos[:]=model.qpos0
                    data.qpos[root_address:root_address+3]=[0,0,t['height'][k,index]]
                    data.qpos[root_address+3:root_address+7]=t['quaternion'][k,index]
                    data.qpos[addresses]=t['joint_positions'][k,index]
                    data.qvel[:]=0
                    mujoco.mj_forward(model,data)
                    np.testing.assert_allclose(data.xpos[root_body,2],t['height'][k,index],atol=1e-7,rtol=0)
                    np.testing.assert_allclose(abs(np.dot(data.xquat[root_body],t['quaternion'][k,index])),1,atol=2e-5,rtol=0)
                    present={}
                    for c in data.contact:
                        if c.dist>=0:continue
                        geom=list(map(int,c.geom))
                        labels=sorted(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in geom)
                        pair=' | '.join(labels)
                        if not all(s.startswith('robot/') for s in labels):
                            if t['time'][k]>=9:ground[pair]+=1
                            continue
                        present[pair]=min(present.get(pair,0),float(c.dist))
                    for pair,distance in present.items():
                        pairs[pair]+=1;minimum[pair]=min(minimum.get(pair,0),distance)
                        if t['time'][k]>=9:late[pair]+=1
                    samples+=1
                poses.append(dict(model=kind,samples=samples,self_contact_sample_counts=dict(pairs),
                    final_two_seconds_counts=dict(late),final_two_seconds_floor_contact_counts=dict(ground),
                    minimum_contact_distance_m=minimum))
            rows.append(dict(condition=pose,env_index=index,outcome=label,models=poses))
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',physics_steps=0,cuda_initialized=False,source_common_different_fields=fields,rows=rows,
        source_record=proof['record'],source_review_sha256=sha(OUT/'failure-review.json'),script_sha256=sha(Path(__file__)),
        root_xy='not captured; both set to0 on the same flat infinite floor',
        scope='CPU mj_forward candidate contact geometry only; no forces, integration, causal effect or new success-rate evidence')
    (OUT/'contact-geometry-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',different_fields=fields,
        late_contact_pairs=[dict(condition=r['condition'],env_index=r['env_index'],outcome=r['outcome'],
            source=r['models'][0]['final_two_seconds_counts'],common=r['models'][1]['final_two_seconds_counts']) for r in rows],physics_steps=0),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
