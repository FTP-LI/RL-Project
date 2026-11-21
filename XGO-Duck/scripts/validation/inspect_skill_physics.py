"""CPU task-configuration audit for choosing one skill-switching plant.

Do not create ManagerBasedRlEnv, initialize CUDA or step physics. Capture the
source task timing, actuator delays and requested collision masks; these are
configuration facts, not measured contact dynamics or transition validation.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows=[]
    for task in ('Mjlab-Velocity-Flat-XgoDuck','Mjlab-StandUp-Flat-XgoDuck','Mjlab-Roulade-Flat-XgoDuck'):
        cfg=load_env_cfg(task,play=True);robot=cfg.scene.entities['robot']
        dt=cfg.sim.mujoco.timestep
        collisions=[{name:getattr(c,name) for name in
            ('geom_names_expr','contype','conaffinity','condim','friction','priority')} for c in robot.collisions]
        actuators=[dict(type=a.__class__.__name__,delay_min_steps=a.delay_min_lag,
            delay_max_steps=a.delay_max_lag,delay_min_ms=a.delay_min_lag*dt*1000,
            delay_max_ms=a.delay_max_lag*dt*1000,vin_range=a.vin_range,
            target_names_expr=a.target_names_expr) for a in robot.articulation.actuators]
        rows.append(dict(task=task,physics_dt=dt,decimation=cfg.decimation,
            control_dt=dt*cfg.decimation,training_timeout=cfg.episode_length_s,
            robot_factory=robot.spec_fn.__name__,nconmax=cfg.sim.nconmax,
            requested_collisions=collisions,actuators=actuators,
            sensor_names=[s.name for s in cfg.scene.sensors],events=list(cfg.events)))
    assert not torch.cuda.is_initialized()
    files=['src/mjlab_microduck/robot/xgoduck_constants.py','src/mjlab_microduck/tasks/xgoduck_physics.py',
           'src/mjlab_microduck/tasks/__init__.py','src/mjlab_microduck/tasks/xgoduck_tricks_env_cfg.py']
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),tasks=rows,
        source_sha256={f:sha(WORK/f) for f in files},script_sha256=sha(Path(__file__)),
        cuda_initialized=False,physics_stepped=False,training_started=False,
        configuration_equal=False,scope='CPU source configuration inspection; no runtime contacts or skill-switch success')
    out=ROOT/'records/reports/skill-interfaces';out.mkdir(exist_ok=True)
    (out/'physics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
