"""Close the saved walking/report phase; the whole nine-motion goal stays open."""
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
OUT=ROOT/'records/reports/common-walk-confirmation'


def main():
    review=read(OUT/'closure-review.json')
    assert review['status']=='passed-integrity-functional-failed'
    assert review['total_transitions']==913696 and review['new_successes']==663
    frozen={}
    for name in ('common_walk_confirmation.json','common_walk_reset_recovery.json','getup_teacher_anchor_pair.json'):
        path=ROOT/'configs/experiments'/name;p=read(path)
        for source,digest in p['source_sha256'].items():assert sha(ROOT/source)==digest,source
        frozen[name]=dict(sha256=sha(path),sources=len(p['source_sha256']))
    for row in review['full_evaluations']:
        folder=ROOT/'records/evaluations'/row['run_id']
        assert read(folder/'command.json')['foreground_tty']
        assert read(folder/'startup.json')['status']=='passed'
        assert read(folder/'verification.json')['status']=='passed'
        assert read(folder/'runtime-contract.json')['learning_updates']==0
    media=read(ROOT/'records/reports/stage-media/visual-review.json')
    assert media['status']=='passed' and media['reviewed_action_videos']==421 and media['new_action_videos']==35
    assert sha(ROOT/'records/reports/stage-media/manifest.json')==media['manifest_sha256']
    assert sha(ROOT/'docs/reports/assets/all-video-overview.jpg')==media['overview_sha256']
    duration=read(ROOT/'records/reports/getup-teacher-anchor-pair/contact-duration/analysis.json')
    assert duration['episodes']==960 and duration['learning_updates']==duration['physics_steps']==0
    for source,digest in duration['source_sha256'].items():assert sha(ROOT/source)==digest
    figure=read(ROOT/'records/reports/getup-teacher-anchor-pair/contact-duration/figure-review.json')
    assert figure['status']=='passed' and sha(ROOT/'records/reports/getup-teacher-anchor-pair/contact-duration/analysis.json')==figure['analysis_sha256']
    docs=read(ROOT/'records/reports/development-report-check.json')
    assert docs['status']=='passed' and docs['source_sha256']==sha(ROOT/'docs/reports/development_report.md')
    assert docs['html_sha256']==sha(ROOT/'docs/reports/development_report.html')
    assert read(ROOT/'records/reports/report-media-check.json')['status']=='passed'
    layout=read(ROOT/'records/reports/project-organization/verification.json')
    assert layout['status']=='passed' and not layout['missing'] and layout['daily_tasks']==5
    workers=[]
    names={'recover_common_walk_resets.py','common_walk_confirmation.py','getup_teacher_anchor_pair.py','train.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(arg).name in names for arg in args if arg):workers.append(dict(pid=int(path.parent.name),argv=args))
    assert not workers,workers
    p=read(ROOT/'project.json');request=p['staged_training_request']
    assert len(request['motions'])==9 and not request['simulation_running'] and not request['learning_running']
    assert not request['deployment_authorized_now'] and not p['goal_execution']['complete']
    result=dict(status='passed-phase-closure',checked_at=datetime.now(timezone.utc).isoformat(),
        whole_goal_status='active',whole_goal_complete=False,turn_classification='progress',
        walking_functional_confirmation_passed=False,new_evaluation_transitions=672000,
        combined_evaluation_transitions=913696,learning_updates=0,real_dynamics_workers=workers,
        frozen=frozen,original_failure_retained=True,heldout2040_consumed=True,
        actual_reset_branch_exercised=False,reset_helper_cpu_only=True,
        action_video_contract=read(ROOT/'records/reports/action-video/verification.json') if (ROOT/'records/reports/action-video/verification.json').exists() else 'See check_action_video.py output',
        report_local_links=docs['local_links_checked'],layout_references=layout['local_references'],
        report_source_sha256=docs['source_sha256'],report_html_sha256=docs['html_sha256'],
        media_count=421,annotated_frames=842,new_media_count=35,source_sha256=sha(Path(__file__)),
        remaining=['contact-age and privileged-value preparation','reliable getup','reliable roll',
            'common-plant walking reliability','continuous transitions and disturbances','final reproduction and hardware preparation'])
    write(OUT/'final-status.json',result)
    p['goal_execution'].update(previous_turn_classification='progress',
        previous_turn_evidence='records/reports/common-walk-confirmation/final-status.json',
        current_evidence='records/reports/common-walk-confirmation/final-status.json',updated_at=result['checked_at'])
    write(ROOT/'project.json',p)
    print('COMMON_WALK_FINAL',result['status'],'goalactive',docs['local_links_checked'],'reportlinks',layout['local_references'],'navigationlinks','workers',workers,flush=True)


if __name__=='__main__':main()
