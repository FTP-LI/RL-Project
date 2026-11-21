"""CPU common-plant configuration and compiled-robot audit; no physics steps.

Entity builds the same collision editors, actuators and initial keyframe that
the simulation uses. Compile its MuJoCo model and compare numeric fields;
this is stronger than a factory-name comparison, but does not build the full
scene, initialize CUDA, test contacts or validate any continuous transition.
"""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from mjlab.entity import Entity
from common_skill_plant import TASKS,make_config
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write

OUT=ROOT/'records/reports/common-plant'
FIELDS=('body_mass','body_inertia','body_pos','body_quat','jnt_type','jnt_range',
        'jnt_limited','dof_armature','dof_damping','dof_frictionloss','geom_type',
        'geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity',
        'geom_condim','geom_friction','geom_priority','geom_solref','geom_solimp',
        'exclude_signature','actuator_trnid','actuator_gear','actuator_gainprm','actuator_biasprm',
        'body_ipos','body_iquat','geom_dataid','mesh_vert','mesh_face','mesh_normal','mesh_texcoord')


def stable(value):
    if callable(value):return dict(module=value.__module__,name=value.__qualname__)
    if isinstance(value,dict):return {str(k):stable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [stable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,slice):return dict(slice=[value.start,value.stop,value.step])
    if isinstance(value,(torch.dtype,torch.device)):return str(value)
    if isinstance(value,Path):return str(value)
    return value


def digest(value):return hashlib.sha256(json.dumps(stable(value),sort_keys=True).encode()).hexdigest()


def normalized_xml(xml):
    root=ET.fromstring(xml)
    for mesh in root.findall('./asset/mesh'):
        kind=mesh.attrib.get('content_type')
        if kind is not None:
            assert kind=='model/stl' and mesh.attrib['file'].endswith('.stl')
            mesh.attrib.pop('content_type')
    return ET.tostring(root,encoding='unicode')


def main():
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True)
    configs={};models={};rows=[];xmls={}
    for skill,task in TASKS.items():
        cfg=make_config(skill);cfg.scene.num_envs=cfg.scene.terrain.num_envs=32
        cfg.seed=1009
        physical=dict(robot=asdict(cfg.scene.entities['robot']),terrain=asdict(cfg.scene.terrain),
            sim=asdict(cfg.sim),decimation=cfg.decimation,sensors=[asdict(s) for s in cfg.scene.sensors],
            physical_events={k:asdict(cfg.events[k]) for k in ('foot_friction','expand_bam_friction_fields')})
        configs[skill]=physical
        entity=Entity(cfg.scene.entities['robot']);xml=entity.spec.to_xml();model=entity.spec.compile()
        (OUT/f'{skill}-robot.xml').write_text(xml)
        xmls[skill]=xml
        fields={k:np.asarray(getattr(model,k)).copy() for k in FIELDS}
        assert all(np.isfinite(v).all() for v in fields.values())
        models[skill]=fields;np.savez_compressed(OUT/f'{skill}-robot-fields.npz',**fields)
        row=dict(skill=skill,task=task,physical_config_sha256=digest(physical),
            compiled_field_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in fields.items()},
            robot_xml_sha256=sha(OUT/f'{skill}-robot.xml'),compiled_fields_archive_sha256=sha(OUT/f'{skill}-robot-fields.npz'),
            dimensions=dict(nq=model.nq,nv=model.nv,nu=model.nu,ngeom=model.ngeom,nbody=model.nbody,nexclude=model.nexclude),
            remaining_events=list(cfg.events),sensor_names=[s.name for s in cfg.scene.sensors],
            physics_dt=cfg.sim.mujoco.timestep,control_dt=cfg.sim.mujoco.timestep*cfg.decimation,nconmax=cfg.sim.nconmax,
            actor_terms=list(cfg.observations['actor'].terms),action_terms=list(cfg.actions))
        rows.append(row)
    assert len({r['physical_config_sha256'] for r in rows})==1
    for skill in ('getup','roll'):
        for k in FIELDS:np.testing.assert_array_equal(models['walk'][k],models[skill][k],err_msg=skill+':'+k)
    # MuJoCo may omit the optional model/stl MIME attribute in later writes.
    # Keep raw XML, compare every other declaration, and require exact compiled
    # vertices/faces/normals/UVs above before accepting this one normalization.
    assert len({normalized_xml(x) for x in xmls.values()})==1
    assert all(r['dimensions']['nu']==14 and r['dimensions']['nexclude']==3 for r in rows)
    assert not torch.cuda.is_initialized()
    sources=['src/mjlab_microduck/robot/xgoduck_constants.py','src/mjlab_microduck/tasks/xgoduck_tricks_env_cfg.py',
             'src/mjlab_microduck/robot/xgoduck/robot_allcollisions.xml','src/mjlab_microduck/robot/xgoduck/params/1910_m6.json']
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),tasks=rows,
        physical_config_exact=True,compiled_robot_fields_exact=True,
        raw_robot_xml_exact=len(set(xmls.values()))==1,
        xml_exact_after_verified_mesh_type_normalization=True,
        xml_comparison='Only optional model/stl content_type on .stl mesh declarations normalized; compiled mesh arrays and every other XML declaration exact',
        physical_randomization=False,pushes_enabled=False,physics_steps=0,optimizer_steps=0,cuda_initialized=False,
        common_voltage_v=7.7,voltage_droop_gain=.1,actuator_delay_seconds=.004,
        input_recipes='Per-task original play observation/action recipe; model-HOME/raw-history adaptation still required before rollout',
        unverified=['Full scene compiled fields','Live nominal startup fields','Actual contact capacity','All skill success and continuous switching','Robustness and hardware'],
        source_sha256={n:sha(WORK/n) for n in sources},
        helper_sha256=sha(ROOT/'scripts/validation/common_skill_plant.py'),script_sha256=sha(Path(__file__)))
    write(OUT/'source-contract.json',result);write(OUT/'physical-configs.json',stable(configs))
    print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','source_sha256')},indent=2),flush=True)


if __name__=='__main__':main()
