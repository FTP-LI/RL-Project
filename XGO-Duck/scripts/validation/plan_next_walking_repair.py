"""Choose the next bounded hypothesis from verified traces, without learning.

The reachable-neck model is an engineering starting point because the
unreachable command was repaired, not a claim of generally better walking.
The sole next candidate change is the existing official actual-position
limit-proximity cost on both hip rolls. Weight -2 is a conservative initial
scale relative to the pose ceiling; its benefit must be measured in a pair.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import __future__
import ast
import hashlib
import json
import sys
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
OUT=ROOT/'records/reports/s1-evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    result=read(OUT/'analysis.json');assert result['status']=='passed'
    forward=next(r for r in result['rows'] if r['role']=='reachable_head' and r['condition']=='forward')
    stand=next(r for r in result['rows'] if r['role']=='reachable_head' and r['condition']=='stand')
    assert forward['inward_over20deg_fraction']>.5 and stand['aggregate_tracking_pass'], 'Reconsider the hypothesis or parent'
    state=read(OUT/'progress.json');assert state['status']=='passed'
    paired=read(ROOT/'records/reports/s1-inputs/paired-inputs.json');assert paired['status']=='passed'
    inputs=read(ROOT/'records/reports/s1-inputs/progress.json')
    chosen=next(m for m in inputs['models'] if m['role']=='reachable_head')
    record=ROOT/'records/experiments/head_command_repair_reachable_head_20261101T213026_s42'
    manifest=read(record/'artifact-manifest.json')
    checkpoint=Path(manifest['checkpoint']);assert sha(checkpoint)==manifest['checkpoint_sha256']
    metadata=read(record/'training-model-metadata.json')
    names=metadata['joint_names'];hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    limits=np.asarray(metadata['joint_ranges_rad'])[hips]
    source=ROOT/'official_resources/xgoduck_rl/src/mjlab_microduck/tasks/mdp.py'
    assert sha(source)==sha(WORK/'src/mjlab_microduck/tasks/mdp.py')
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='joint_pos_limit_proximity')
    space={'torch':torch,'_DEFAULT_ASSET_CFG':SceneEntityCfg('robot')}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),space)
    rows=[];max_error=0.;bindings={str(OUT.relative_to(ROOT)/'analysis.json'):sha(OUT/'analysis.json')}
    for role in state['models']:
        for part in ('five','lateral'):
            folder=Path(state['models'][role][part]);evaluation=read(folder/'evaluation.json')
            for c in evaluation['conditions']:
                path=folder/(c['condition']+'-trace.npz');bindings[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path,allow_pickle=False) as trace:
                    q=trace['joint_positions'][:,:,hips];active=trace['active'].copy();active[:50]=False
                    numpy_cost=(np.maximum(limits[:,0]+.15-q,0)+np.maximum(q-limits[:,1]+.15,0)).sum(axis=-1)
                    env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(
                        joint_pos=torch.tensor(q.reshape(-1,2)),joint_pos_limits=torch.tensor(limits[None],dtype=torch.float32)))})
                    cost=space['joint_pos_limit_proximity'](env,asset_cfg=SceneEntityCfg('robot',joint_ids=[0,1]),margin=.15).numpy().reshape(active.shape)
                    np.testing.assert_allclose(cost[active],numpy_cost[active],atol=2e-6,rtol=2e-5)
                    max_error=max(max_error,float(np.max(np.abs(cost[active]-numpy_cost[active]))))
                    episode_cost=(cost*active).sum(axis=0)/active.sum(axis=0)
                    rows.append(dict(role=role,condition=c['condition'],episodes=32,
                        cost_mean_rad=float(episode_cost.mean()),weight=-2.,
                        weighted_reward_rate=-2*float(episode_cost.mean()),
                        weighted_step_reward=-2*.02*float(episode_cost.mean()),
                        fraction_of_existing_pose_ceiling=2*float(episode_cost.mean())/1.3))
    assert not torch.cuda.is_initialized()
    decision=dict(status='bounded-hypothesis-selected-not-trained',checked_at=datetime.now(timezone.utc).isoformat(),
        parent_role='reachable_head',parent_checkpoint=str(checkpoint),parent_sha256=sha(checkpoint),
        parent_selection_reason='Preserves the verified reachable neck-command repair and full optimizer/normalizer; gait is still defective, so use only for a short single-factor repair pair',
        official_reference_role='External qualitative and quantitative target; only ONNX is available, no verified full-state PPO checkpoint to resume',
        single_change=dict(function='mjlab_microduck.tasks.mdp.joint_pos_limit_proximity',
            joints=['left_hip_roll','right_hip_roll'],margin_rad=.15,candidate_weight=-2.,control_weight=0.),
        weight_reason='Initial moderate scale: compare with pose reward ceiling 1.3; smaller than the tested arithmetic -4 scale. This is not an optimized or proven weight',
        primary_metric='Forward-command bilateral inward-over20deg fraction after the first second, with paired direction-speed/xy-error and stand regression',
        diagnostic_budget_per_arm=50,automatic_budget_extension=False,
        evaluation='All seven frozen development commands, 32 first episodes each; no holdout tuning',
        progression='Only extend after finite/save/interface checks and a paired posture improvement without new falls or worse tracking; otherwise revise hypothesis, not the update budget',
        scale_rows=rows,source_sha256=sha(source),numpy_function_max_abs_error=max_error,
        input_sha256=bindings,training_started=False,gait_improvement_verified=False)
    (OUT/'next-repair-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps({k:v for k,v in decision.items() if k not in ('input_sha256','scale_rows')},indent=2))


if __name__=='__main__':main()
