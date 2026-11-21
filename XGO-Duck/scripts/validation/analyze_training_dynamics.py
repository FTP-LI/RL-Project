"""Read recorded PPO scalars and final exploration scales; no new learning.

Purpose: distinguish reward/behavior drift from numerical optimizer failure.
Loss or training reward cannot establish heldout motion quality. Shrinking
noise is a candidate mechanism, not causal proof; no new budget is implied.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
RUNS=[
    ('bootstrap','duck_walk_official-actor-bootstrap_candidate_s42_20261102T180647',49),
    ('adaptation','duck_walk_official-adaptation-recovery_candidate_s42_20261102T184726',176),
    ('width-control','duck_walk_velocity-precision_control_s42_20261102T190657',99),
    ('width-candidate','duck_walk_velocity-precision_candidate_s42_20261102T190657',99),
]


def main():
    rows=[]
    for label,run,iteration in RUNS:
        record=ROOT/'records/experiments'/run
        folder=Path(read(record/'runner.json')['log_dir']);checkpoint=folder/f'model_{iteration}.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['iter']==iteration
        std=saved['actor_state_dict']['distribution.std_param'].numpy()
        assert std.shape==(14,) and np.isfinite(std).all() and (std>0).all()
        events=sorted(folder.glob('events.out.tfevents.*'));assert events
        accumulator=EventAccumulator(str(folder),size_guidance={'scalars':0});accumulator.Reload()
        scalars={}
        for tag in ['Loss/value','Loss/surrogate','Loss/entropy','Loss/learning_rate',
                    'Train/mean_reward','Train/mean_episode_length',
                    'Episode_Reward/pose','Episode_Reward/air_time','Episode_Reward/track_linear_velocity']:
            values=accumulator.Scalars(tag);assert values and all(np.isfinite(v.value) for v in values)
            # Whole-window metrics are retained, avoiding selection of a favorable update.
            scalars[tag]=[dict(iteration=v.step,value=v.value,wall_time=v.wall_time) for v in values]
        rows.append(dict(label=label,run_id=run,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            event_sha256={str(p):sha(p) for p in events},
            std_rad=std.tolist(),std_geometric_mean_rad=float(np.exp(np.log(std).mean())),
            std_min_rad=float(std.min()),std_max_rad=float(std.max()),
            actor_normalizer_count=int(saved['actor_state_dict']['obs_normalizer.count']),scalars=scalars))
    base=rows[0]['std_geometric_mean_rad']
    for row in rows:row['noise_change_from_bootstrap_fraction']=row['std_geometric_mean_rad']/base-1
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),rows=rows,
        script_sha256=sha(Path(__file__)),new_learning_updates=0,
        interpretation=['Recorded value loss decreases while fixed-command behavior can worsen.',
                        'Exploration scale shrinks; insufficient exploration is an untested hypothesis, not demonstrated causality.',
                        'No explained-variance or per-command advantage was recorded; value loss alone does not establish critic adequacy.',
                        'These are saved learned checkpoints, not four independent training seeds.'])
    out=ROOT/'records/reports/reward-incentives/training-dynamics.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',rows=[{k:v for k,v in row.items() if k not in ('scalars','event_sha256','std_rad')} for row in rows]),indent=2))


if __name__=='__main__':main()
