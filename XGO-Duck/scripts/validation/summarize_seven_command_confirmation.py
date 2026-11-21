"""Close the fixed-controller simulation milestone from verified raw records.

Check all independent seeds, then package unchanged ONNX bytes and the three
small controller modules. The package describes simulation semantics; it is
not a hardware deployment and does not mark the nine-motion goal complete.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import onnx
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/seven-command'


def main():
    suite=read(OUT/'suite-progress.json');master=read(ROOT/'configs/experiments/seven_command_confirmation.json')
    assert suite['status']=='all-seven-functional-confirmed' and suite['completed_seeds']==master['ordered_seeds']
    results=[]
    for seed in master['ordered_seeds']:
        out=OUT/f's{seed}';a=read(out/'analysis.json');v=read(out/'verification.json')
        assert a['functional_development_passed'] and a['verification_sha256']==sha(out/'verification.json')
        folder=ROOT/a['evaluator'];assert v['report_sha256']==sha(folder/'evaluation.json')
        assert a['protocol_sha256']==master['protocol_sha256'][str(seed)]==sha(folder/'protocol.json')
        assert sum(r['episodes'] for r in a['rows'])==224 and all(r['successes']>=31 for r in a['rows'])
        assert a['heldout_used']==(seed==2001) and v['maximum_inference_error']<=2e-5
        results.append(dict(seed=seed,heldout=seed==2001,record=a['evaluator'],analysis_sha256=sha(out/'analysis.json'),
            successes=sum(r['successes'] for r in a['rows']),episodes=224,shared_seconds=a['shared_seconds'],
            rows=a['rows'],replayed_inputs=v['replayed_inputs']))
    rows=[]
    for name in results[0]['rows']:
        condition=name['condition'];samples=[next(r for r in a['rows'] if r['condition']==condition) for a in results]
        rows.append(dict(condition=condition,successes=sum(r['successes'] for r in samples),episodes=96,
            per_seed_successes=[r['successes'] for r in samples],per_seed_wilson95=[r['success_wilson95'] for r in samples],
            signed_speed_range=[min(r['signed_speed'] for r in samples),max(r['signed_speed'] for r in samples)],
            block_error_range=[min(r['block_error'] for r in samples),max(r['block_error'] for r in samples)],
            yaw_error_range=[min(r['yaw_error'] for r in samples),max(r['yaw_error'] for r in samples)],
            original_instantaneous_flags=[r['legacy_instantaneous_pass'] for r in samples],
            maximum_posture_bad_fraction=max(r['posture_bad_fraction'] for r in samples),
            maximum_close_feet_fraction=max(r['close_feet_fraction'] for r in samples)))
    origin=read(ROOT/master['protocols']['1007']);model=ROOT/origin['model']
    assert sha(model)==origin['model_sha256']
    bundle=ROOT/'artifacts/simulation/duck_walk_command-gyro_reference_20261103';bundle.mkdir(parents=True,exist_ok=True)
    files=[(model,'policy.onnx')]+[(ROOT/'scripts/validation'/f,f) for f in
        ('walking_command_controller.py','gyro_turn_controller.py','command_input_mapping.py')]
    packaged=[]
    for source,name in files:
        target=bundle/name
        if target.exists():assert sha(target)==sha(source),'Existing bundle differs: do not overwrite'
        else:shutil.copy2(source,target)
        packaged.append(dict(file=name,sha256=sha(target),source=str(source.relative_to(ROOT))))
    meta={x.key:x.value for x in onnx.load(model).metadata_props}
    contract=dict(status='simulation-functional-confirmed-hardware-not-validated',input_shape=[1,61],output_shape=[1,14],
        joint_names=meta['joint_names'].split(','),model_home=list(map(float,meta['default_joint_pos'].split(','))),
        action_scale=float(meta['action_scale']),control_dt=.02,physics_dt=.005,
        action_history='Previous raw ONNX output, before physical target conversion or optional runtime output filters',
        targets='model_home + raw ONNX output; no external output EMA in these walking tests',
        command_slots=[48,49,50],command_gains=[1.4,3.,1.5],command_min=[-.2,-.24,-1.],command_max=[.4,.24,1.],
        yaw_feedback=dict(gain=.75,time_constant_seconds=.2,input='Existing actor gyro Z',turn_only=True,final_cap=1.),
        reset='Zero raw history and controller EMA readiness for reset worlds',
        original_runtime_differences='Original runtime output EMA and shared HOME must be separately validated; do not flash this package',
        files=packaged,protocol_sha256=sha(ROOT/'configs/experiments/seven_command_confirmation.json'))
    write(bundle/'contract.json',contract)
    (bundle/'README.md').write_text('''# 行走仿真参照

此包保留官方 ONNX 原字节，并加入固定输入映射与转向 gyro 反馈。适用于已记录的行走仿真环境；没有执行真机部署。

在现有环境中，将本目录的三个 Python 文件放在同一导入路径。调用 `walking_command_controller.create_controller(num_envs, 0.02)`；每个控制步用 `controller.map(desired_commands, actor_observations)` 得到第 48–50 项指令输入，再进行 ONNX 推理。其余观测保持原义。

原始输出保存为下一步动作历史；物理目标为模型元数据 HOME 加原始输出。重置相应环境时清零动作历史，并调用 `controller.reset(env_ids)`。切勿将已滤波目标写入原始动作历史。

安装、VS Code 练习入口、三种子结果和原始视频见工程中的 `docs/reports/development_report.md`。完整接口及来源哈希见 `contract.json`。后续共同物理环境、技能切换和官方运行程序的动作滤波仍须验证。
''',encoding='utf-8')
    summary=dict(status='all-seven-functional-confirmed',checked_at=datetime.now(timezone.utc).isoformat(),
        seeds=results,rows=rows,episodes=672,successes=sum(a['successes'] for a in results),
        controller_parameters_changed_during_suite=False,learning_updates=0,heldout_consumed_once=True,
        old_instantaneous_verdicts_preserved=True,full_goal_complete=False,
        package=str(bundle.relative_to(ROOT)),package_contract_sha256=sha(bundle/'contract.json'),script_sha256=sha(Path(__file__)),
        scope='S2 functional simulation milestone only; original instantaneous failures, S3/S4/S5 and hardware boundaries remain')
    write(OUT/'confirmation-summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('seeds',)},indent=2))


if __name__=='__main__':main()
