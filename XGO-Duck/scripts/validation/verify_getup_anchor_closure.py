"""Final saved-evidence and worker audit; no simulator or optimizer calls.

Verify immutable protocols, completed artifact checks, native media coverage,
explicit cleanup and the real current process list before reporting this
stage closed. The application goal remains active with all unfinished stages.
"""
from contextlib import redirect_stdout
from datetime import datetime,timezone
from io import StringIO
from pathlib import Path
import json,re,sys
import torch
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from getup_teacher_anchor_pair import checked,OUT


def main():
    assert not torch.cuda.is_initialized()
    plan=checked();a=read(OUT/'analysis.json')
    assert a['selection']['action']=='close' and a['selection']['selected'] is None
    assert a['actual_learning_transitions']==2015232 and a['actual_evaluation_transitions']==528000
    assert a['timely_counts']==dict(control=287,original=285,candidate=303)
    for role in ('control','candidate'):
        proof=read(OUT/role/'training-verification.json')
        assert proof['status']=='passed' and proof['rollout_updates']==41
        assert proof['all_video_qpos_hashes_verified']
    for role in ('control','original','candidate'):
        proof=read(OUT/f'{role}-analysis.json')
        assert proof['status']=='passed' and not proof['all_motion_gates_passed']
        assert len(proof['videos'])==4
    # Parent zero-step failure and recovery remain separate immutable records.
    from recover_getup_anchor_scale import activate
    activate()
    from getup_anchor_value_check import checked as value_checked
    value=value_checked();vp=read(ROOT/'records/reports/getup-anchor-value-check/scale-recovery/analysis.json')
    assert vp['status']=='passed' and vp['value_gate_passed'] and vp['actor_updates']==vp['critic_updates']==0
    names=['getup_late_feasibility_pair.json','getup_early_guard_study.json',
           'roll_common_confirmation.json','roll_common_rng_recovery.json','getup_anchor_value_check.json']
    preserved={}
    for name in names:
        p=read(ROOT/'configs/experiments'/name)
        for path,digest in p['source_sha256'].items():assert sha(ROOT/path)==digest,path
        preserved[name]=len(p['source_sha256'])
    cleanup=read(OUT/'intermediate-cleanup.json')
    assert cleanup['status']=='deleted-verified' and cleanup['total_bytes']==9706142
    for row in cleanup['items']:assert not (ROOT/row['path']).exists()
    for row in cleanup['retained']:assert sha(ROOT/row['path'])==row['sha256']
    atlas=read(ROOT/'records/reports/stage-media/all-video-overview.json')
    visual=read(ROOT/'records/reports/stage-media/visual-review.json')
    assert atlas['status']==visual['status']=='passed' and atlas['videos']==386 and atlas['selected_frames']==772
    assert visual['manifest_sha256']==sha(ROOT/'records/reports/stage-media/manifest.json')
    assert atlas['current_manifest_sha256']==visual['manifest_sha256']
    assert sha(ROOT/atlas['image'])==atlas['image_sha256']
    assert read(OUT/'figure-review.json')['status']=='passed'
    report=read(ROOT/'records/reports/development-report-check.json')
    assert report['status']=='passed' and report['source_sha256']==sha(ROOT/'docs/reports/development_report.md')
    assert report['html_sha256']==sha(ROOT/'docs/reports/development_report.html')
    assert read(ROOT/'records/reports/project-organization/verification.json')['status']=='passed'
    links=0
    for file in (OUT/'README.md',ROOT/'records/reports/getup-anchor-value-check/scale-recovery/README.md'):
        text=file.read_text();assert not re.search(r'\bV\d+\b|修订稿|版本号',text)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            path=(file.parent/target.split('#')[0]).resolve()
            assert path.is_relative_to(ROOT) and path.exists(),target
            links+=1
    import current_status
    output=StringIO()
    with redirect_stdout(output):current_status.main()
    workers=json.loads(output.getvalue());assert not workers['workers'] and not workers['learning_running']
    write(OUT/'worker-status.json',workers)
    result=dict(status='passed-stage-closure',checked_at=datetime.now(timezone.utc).isoformat(),
        application_goal_status='active',all_required_stages_complete=False,
        counts=a['timely_counts'],actual_learning_transitions=2015232,actual_evaluation_transitions=528000,
        new_seed_value_transitions=409600,value_admitted=True,stage_functional_gate_passed=False,
        unchanged_current_source_files=len(plan['source_sha256']),unchanged_value_source_files=len(value['source_sha256']),
        older_frozen_sources_preserved=preserved,action_videos=386,selected_frames=772,
        extra_index_links_checked=links,deleted_intermediate_bytes=9706142,
        real_project_workers=workers['workers'],cuda_initialized=False,new_physics_steps=0,new_optimizer_steps=0,
        remaining=['common-noise/target-transition walking regression','reliable recovery',
            'reliable roll','continuous skill sequences and disturbances','final reproduction and hardware preparation'])
    write(OUT/'final-status.json',result)
    project=read(ROOT/'project.json')
    project['goal_execution'].update(previous_turn_classification='progress',
        previous_turn_evidence='records/reports/getup-teacher-anchor-pair/final-status.json',
        scoped_application_goal_status='active',long_term_goal_complete=False)
    write(ROOT/'project.json',project)
    print('GETUP_ANCHOR_STAGE_CLOSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
