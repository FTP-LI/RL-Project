"""Read the stopped zero attempt and trace the reset-to-actuator source path.

The failed hook did not save its inputs, so this analysis must not claim which
condition actually failed. It records a source-supported hypothesis and any
runtime source files that the failed plan did not pin.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl')
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
RUN=ROOT/'records/reports/duck_roll-factor_nominal_s1242_20261116T133133'
PLAN=ROOT/'configs/experiments/roll_factor_fp_zero_nominal_s1242_20261116.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'nominal-zero-stop-analysis.json';assert not out.exists()
    plan=json.loads(PLAN.read_text());receipt=json.loads((RUN/'native-zero-execution.json').read_text())
    result=json.loads((RUN/'native-zero-run/result.json').read_text());runtime=json.loads((RUN/'zero/runtime-contract.json').read_text())
    assert receipt['status']=='child-failed' and receipt['exit_code']==2
    assert receipt['log_sha256']==sha(RUN/'native-zero-execution.log')
    assert result['status']=='stopped-review-required' and result['error']=='Actual startup voltage or actuator state mismatch'
    assert runtime['status']=='closed' and runtime['control_steps']==runtime['physics_substeps']==0
    assert not (RUN/'zero/actuator-start-state.npz').exists() and not (RUN/'zero/start-fingerprints.json').exists()
    others=[ROOT/'records/reports'/f'duck_roll-factor_{role}_s1242_20261116T133133' for role in ('friction-low','voltage-high')]
    assert all(not (path/'native-zero-run').exists() for path in others)
    files={'environment':WORK/'.venv/lib/python3.12/site-packages/mjlab/envs/manager_based_rl_env.py',
           'scene':WORK/'.venv/lib/python3.12/site-packages/mjlab/scene/scene.py',
           'entity':WORK/'.venv/lib/python3.12/site-packages/mjlab/entity/entity.py',
           'actuator':WORK/'vendor/bam/bam/mjlab.py'}
    snippets={'environment':['self._reset_idx(env_ids)','self.scene.write_data_to_sim()','self.sim.forward()'],
              'scene':['def write_data_to_sim(self)','ent.write_data_to_sim()'],
              'entity':['def write_data_to_sim(self)','self._apply_actuator_controls()',
                        'command = act.apply_delay(command)','self._data.write_ctrl(act.compute(command), act.ctrl_ids)'],
              'actuator':['self._prev_motor_torque.zero_()','self._prev_motor_torque = motor_torque.detach()']}
    source=[]
    for label,path in files.items():
        lines=path.read_text(encoding='utf-8').splitlines();hits={}
        for needle in snippets[label]:
            positions=[i+1 for i,line in enumerate(lines) if needle in line];assert positions,(label,needle)
            hits[needle]=positions
        key=str(path);recorded=plan['library_source_sha256'].get(key)
        if label=='actuator':recorded=plan['training_source_sha256'].get('vendor/bam/bam/mjlab.py')
        if recorded is not None:assert sha(path)==recorded
        source.append({'label':label,'path':key,'sha256':sha(path),'pinned_by_failed_plan':recorded is not None,'line_matches':hits})
    force=RUN/'zero/actuator-force-limits.npz';meta=json.loads((RUN/'zero/actuator-force-limits.json').read_text())
    assert meta['arrays_sha256']==sha(force)
    with np.load(force,allow_pickle=False) as a:
        limits=a['actuator_forcerange'];flags=a['actuator_forcelimited'];trntype=a['actuator_trntype']
        assert limits.shape==(14,2) and flags.all() and np.all(trntype==0)
        first_range=limits[0].tolist()
    record={'status':'saved-stop-and-source-path-reviewed','checked_at':datetime.now(timezone.utc).isoformat(),
            'run_id':plan['run_id'],'protocol_sha256':sha(PLAN),'log_sha256':receipt['log_sha256'],
            'zero_control_steps':0,'zero_physics_substeps':0,'learning_updates':0,'other_arms_started':False,
            'compiled_force_range_first':first_range,'actuators':14,
            'failed_check_inputs_saved':False,'actual_failing_condition_identified':False,
            'source_supported_hypothesis':'Reset writes custom actuator controls before any physics step; actuator compute can update previous torque. The zero-torque assumption may therefore be invalid, but this run did not save the actual torque or voltage arrays.',
            'source_checks':source,'unpinned_runtime_sources':[x['path'] for x in source if not x['pinned_by_failed_plan']],
            'next_diagnostic':'Save actual values before individual checks; accept finite correctly shaped previous torque and compare across matched starts. Pin the scene and entity sources in the new plan.',
            'automatic_retry':False,'hardware_io':False}
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
