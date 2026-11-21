"""Compare the trained contact-timing candidate to two original-reward repeats."""

import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import collect_roll_checkpoint_selection as sampler
import run_roll_checkpoint_selection as previous
import verify_roll_checkpoint_selection as scorer
from native_nominal_baseline import verify_sources
from record_io import write
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


KEY = 'roll_endpoint_progress_evaluation'
TRAINING = 'roll_endpoint_progress_training'
STAGE = 'roll-endpoint-progress-evaluation'
ENTRY = Path(__file__).resolve()
PLANS = previous.PLANS.with_name('endpoint-progress-evaluation')
MASTER = PLANS / 'master.json'
REPORT = previous.REPORT.with_name('endpoint-progress-training') / 'action-evaluation'
CASES = previous.ORDER
CANDIDATE = 'endpoint-reward-candidate'
REFERENCES = ('first-original-reward-reference', 'second-original-reward-reference')
ROLES = (*REFERENCES, CANDIDATE)
LABELS = {'shared':'共同站立', REFERENCES[0]:'原奖励第一次参照', REFERENCES[1]:'原奖励第二次参照',
          CANDIDATE:'末端接触奖励候选'}


class RollEvaluationView(RecordedObserver):
    def start(self):
        self.state['purpose'] = '滚翻进度奖励对照｜固定策略'
        super().start()


def freeze():
    project = read(P)
    task, goals = project['staged_training_request'], project['goal_execution']
    training = goals[TRAINING]
    require(previous.direct.pair.trial.allowed(project,'roll-endpoint-progress-training')
            and not task['learning_running'] and not task['simulation_running']
            and training['status']=='candidate-action-evaluation-pending' and not PLANS.exists() and not REPORT.exists(),
            'Candidate training and export must be independently reviewed before action evaluation')
    evidence, training_plan = (ROOT/training[name] for name in ('candidate_result','protocol'))
    result, plan = read(evidence), read(training_plan)
    control = plan['matched_archived_control']
    candidate, checkpoint = (ROOT/result[name] for name in ('model','checkpoint'))
    require(sha(evidence)==training['candidate_result_sha256'] and sha(training_plan)==training['protocol_sha256']
            and result['transitions']==control['transitions']==5529600 and result['historical_control']==control
            and sha(candidate)==result['export']['model_sha256'] and sha(checkpoint)==result['export']['checkpoint_sha256']
            and result['export']['metadata_preserved'], 'Trained candidate or original-reward control does not match')
    recovery = evidence.with_name('verification-recovery.json')
    if recovery.is_file():
        note = read(recovery)
        require(note['candidate_review_sha256']==sha(evidence) and note['frozen_protocol_sha256']==sha(training_plan)
                and note['new_learning_updates']==note['new_physical_steps']==0, 'Read-only review recovery differs')
    old_master = read(previous.MASTER)
    require(sha(previous.MASTER)==goals['roll_checkpoint_selection']['master_sha256']
            and plan['development_gate']['seeds']==[1247,1248]
            and not old_master['reserved_seed_used'] and old_master['reserved_seed']==plan['development_gate']['reserved_seed']==2246,
            'Historical action protocol or seed split differs')
    sources = (ENTRY, Path(sampler.__file__), Path(scorer.__file__), previous.MASTER, evidence, training_plan,
               candidate, checkpoint, ROOT/control['model'])
    if recovery.is_file():
        sources += (recovery,)
    hashes = {rel(path):sha(path) for path in sources}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for case in CASES:
        source = previous.PLANS / (case+'.json')
        spec = read(source)
        require(sha(source)==old_master['protocols'][case] and spec['roll']['model']==control['model']
                and spec['roll']['model_sha256']==sha(ROOT/control['model']) and spec['minimum_roll_completion']==116
                and spec['worlds']==128 and spec['roll_controls']+spec['initial_stand_controls']==500,
                'Action gate or original-reward actor changed')
        display = spec['checkpoint_selection']
        spec['checkpoint_selection'] = dict(plant=display['plant'],assessment=display['assessment'],
            role_source={**dict.fromkeys(REFERENCES,control['model']),CANDIDATE:rel(candidate)})
        out = REPORT/case
        spec.update(stage=STAGE,run_id=f'duck_roll_endpoint_reward_evaluation_s{spec["seed"]}_{stamp}',
            route_scope='Three roll prefixes from the same standing snapshot, with original and endpoint training rewards',
            guarded_output=rel(out),page=rel(out/'route-view.html'),
            earlier_roll=dict(spec['roll'],model=rel(candidate),model_sha256=sha(candidate)))
        spec['frozen_sha256'].update(hashes | {rel(source):sha(source)})
        verify_sources(spec)
        prepared[case] = spec
    admission = REPORT/'admission.json'
    REPORT.mkdir()
    PLANS.mkdir()
    put(admission,dict(observation=preflight(),source_sha256=hashes,initial_physical_steps=0,roll_gate=116))
    protocols = {}
    for case,spec in prepared.items():
        directory, path = REPORT/case, PLANS/(case+'.json')
        directory.mkdir()
        spec['frozen_sha256'][rel(admission)] = sha(admission)
        put(path,spec)
        protocols[case] = sha(path)
        put(directory/'windows-entry.json',dict(protocol_sha256=protocols[case], evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=list(CASES),protocols=protocols,physical_transition_budget=old_master['physical_transition_budget'],
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False,automatic_retry=False))
    goals[KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),admission=rel(admission),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
                current_protocol_execution='roll-endpoint-reward-evaluation-ready',current_evidence=rel(admission),
                next_gate='Compare fixed learned actors under the original low-friction standing-to-roll gate in two seeds.')
    write(P,project)
    print('ROLL_ENDPOINT_EVALUATION_FROZEN',sha(MASTER),flush=True)


def publish(rows, active=None):
    lines = ['# 滚翻进度奖励动作对照','',
             '每个种子128场景。候选须达到116/128，并较两次原奖励参照各净增至少4个。','',
             '| 种子 | 原奖励第一次参照 | 原奖励第二次参照 | 末端接触奖励候选 | 候选净增 |',
             '| --- | ---: | ---: | ---: | ---: |']
    links = []
    for case,row in rows.items():
        counts = ' | '.join(f'{row["arms"][role]["roll_success"]}/128' for role in ROLES)
        changes = '／'.join(str(row['comparisons'][role]['net']) for role in REFERENCES)
        lines.append(f'| {row["seed"]} | {counts} | {changes} |')
        media = [f'[{LABELS[role]}原片](../../../../../{row["arms"][role]["video"]["path"]})' for role in ROLES]
        links.extend(('',f'[种子{row["seed"]}独立评分]({case}/independent.json) · '+' · '.join(media)))
    if active:
        lines.extend(('',f'正在采样{active}，已复核{len(rows)}/2个种子。','',
            f'[状态]({active}/native-view/native-status.txt) · [实景]({active}/native-view/native-latest.jpg) · '
            f'[命令日志]({active}/collect-command.log)'))
    else:
        lines.extend(('', '[完整评分](comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines+links)+'\n',encoding='utf-8')


@contextmanager
def stage_bindings():
    paths = dict(STAGE=STAGE,KEY=KEY,PLANS=PLANS,REPORT=REPORT,MASTER=MASTER,ENTRY=ENTRY)
    roles = dict(ROLES=ROLES,CANDIDATE=CANDIDATE,REFERENCES=REFERENCES)
    with (patch.multiple(previous,**paths,publish=publish,RecordedObserver=RollEvaluationView),
          patch.multiple(previous.direct,**paths),patch.multiple(sampler,**roles,LABELS=LABELS),patch.multiple(scorer,**roles)):
        yield


def render():
    import numpy as np

    import review_roll_neck_support_response as visual
    from review_roll_forward_input_stop import arrays

    evidence = REPORT/'comparison.json'
    state = read(P)['goal_execution'][KEY]
    require(state['status']=='reviewed' and sha(evidence)==state['result_sha256'],'Independently scored comparison required')
    result = read(evidence)
    traces = {}
    fields = ('head_contact','foot_contact','ground_contact','phase_next')
    for case,row in result['arms'].items():
        raw = ROOT/'records/evaluations'/row['run_id']
        stored = raw/'guarded-evaluation.json'
        require(sha(stored)==row['collection_sha256'],'Collection evidence changed')
        manifest = read(stored)
        prefix = raw/'shared/trace.npz'
        require(sha(prefix)==manifest['prefix_sha256']['trace.npz'],'Shared motion source changed')
        common = arrays(prefix)
        traces[case] = {}
        for role in ROLES:
            path = raw/'clips'/role/'trace.npz'
            require(sha(path)==manifest['branches'][role]['trace_sha256']['trace.npz'],'Roll motion source changed')
            branch = arrays(path)
            traces[case][role] = {field:np.concatenate((common[field],branch[field][1:])) for field in fields}
    with patch.multiple(visual,REPORT=REPORT,ROLES=ROLES,LABELS=tuple(LABELS[role] for role in ROLES)):
        receipt = visual.render(result,traces)
    image = REPORT/'endpoint-reward-keyframes.jpg'
    (REPORT/'neck-support-keyframes.jpg').replace(image)
    require(sha(image)==receipt['picture_sha256'],'Keyframe image changed after composition')
    put(REPORT/'endpoint-reward-media.json',dict(receipt,picture=image.name,comparison_sha256=sha(evidence),additional_physical_steps=0))
    print('ROLL_ENDPOINT_KEYFRAMES_VERIFIED',len(traces),18,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score','render'))
    parser.add_argument('case',nargs='?',choices=CASES)
    args = parser.parse_args()
    if args.action=='freeze':
        return freeze()
    if args.action=='render':
        return render()
    with stage_bindings():
        if args.action=='batch':
            previous.batch()
            project, outcome = read(P), read(REPORT/'comparison.json')
            state, task = project['goal_execution'][TRAINING], project['staged_training_request']
            state.update(status='candidate-action-evaluation-reviewed',action_comparison=rel(REPORT/'comparison.json'),
                         action_comparison_sha256=sha(REPORT/'comparison.json'),passed_development_gate=outcome['passed_development_gate'])
            task['current_protocol_execution'] = 'roll-endpoint-reward-action-comparison-reviewed'
            write(P,project)
            return
        if args.case is None:
            parser.error('A frozen action cohort is required')
        {'collect':previous.sample,'score':previous.score}[args.action](args.case)


if __name__=='__main__':
    main()
