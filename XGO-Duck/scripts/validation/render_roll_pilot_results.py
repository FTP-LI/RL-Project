"""Plot actual fixed evaluations and bounded-learning drift; CPU only.

The 32-world intervals describe one development seed, not generalization over
training seeds. Drift is a recorded tripwire, not a teacher loss or gait score.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from figure_style import apply_figure_style
from s1_reference_inputs import ROOT, read, sha
from record_io import write

OUT=ROOT/'records/reports/roll-top-contact'
ASSETS=ROOT/'docs/reports/assets'


def checked_save(fig,path):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip()]
    collisions=[(a.get_text(),b.get_text()) for i,(a,aa) in enumerate(texts)
                for b,bb in texts[i+1:] if aa.overlaps(bb)]
    own_ticks={ax:set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both')) for ax in fig.axes}
    collisions += [(t.get_text(),'spine') for t,b in texts for ax in fig.axes for s in ax.spines.values()
                   if s.get_visible() and t not in own_ticks[ax] and b.overlaps(s.get_window_extent(renderer))]
    outside=[t.get_text() for t,b in texts if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1)]
    assert not collisions and not outside,(collisions,outside)
    fig.savefig(path,dpi=300,bbox_inches=None,facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches=None,facecolor='white')
    plt.close(fig)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),text_collisions=collisions,outside=outside)


def main():
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    colors={'original':'#808892','control':'#2479a8','candidate':'#be771c'}
    labels={'original':'官方模型','control':'原奖励短训','candidate':'仅头顶奖励'}
    evaluations={k:read(OUT/k/'evaluation-analysis.json') for k in colors}
    assert all(r['status']=='passed' for r in evaluations.values())
    summary=read(OUT/'analysis.json');assert not summary['candidate_continue']
    fig,ax=plt.subplots(figsize=(7,4.8));fig.subplots_adjust(left=.14,right=.94,bottom=.20,top=.86)
    for x,(role,r) in enumerate(evaluations.items()):
        f=r['successes']/r['episodes'];lo,hi=r['wilson95']
        ax.errorbar(x,f*100,yerr=np.array([[f-lo],[hi-f]])*100,fmt='o',color=colors[role],capsize=5,markersize=7)
        ax.text(x+.12,f*100,f'{r["successes"]}/32',va='center',fontsize=10)
    ax.axhline(90,color='#89939b',linestyle='--',linewidth=1)
    ax.text(2.53,90,'90% 门槛',ha='right',va='bottom',fontsize=10)
    ax.set(xticks=range(3),xticklabels=list(labels.values()),ylim=(60,103),xlim=(-.4,2.6),ylabel='滚翻成功比例（%）')
    ax.set_title('滚翻开发筛查',loc='left')
    figures=[checked_save(fig,ASSETS/'roll-pilot-success.png')]
    drift={};failure_counts={};phase={}
    fig,ax=plt.subplots(figsize=(7,4.8));fig.subplots_adjust(left=.14,right=.95,bottom=.17,top=.86)
    for role in ('control','candidate'):
        train=read(OUT/role/'training-verification.json');folder=ROOT/'records/experiments'/train['run_id']
        rows=[json.loads(line) for line in (folder/'policy-drift.jsonl').read_text().splitlines()]
        assert [r['completed_updates'] for r in rows]==list(range(1,51))
        ax.plot([r['completed_updates'] for r in rows],[r['rms'] for r in rows],label=labels[role],color=colors[role])
        drift[role]=dict(maximum_rms_rad=max(r['rms'] for r in rows),maximum_abs_rad=max(r['maximum'] for r in rows))
        assert drift[role]['maximum_rms_rad']<=.15 and drift[role]['maximum_abs_rad']<=.6
        phase[role]={str(k):read(folder/f'phase-watch-{k}.json') for k in (14,28,42)}
    ax.axvline(14,color='#89939b',linestyle='--',linewidth=1)
    ax.set(xlabel='完成更新次数',ylabel='固定观测上的动作偏离 RMS（rad）',xlim=(0,52),xticks=range(0,51,10),ylim=(0,None))
    ax.set_title('短训动作偏离',loc='left');ax.legend(loc='upper left')
    figures.append(checked_save(fig,ASSETS/'roll-pilot-drift.png'))
    for role,r in evaluations.items():
        failed=[x for x in r['rows'] if not x['success']]
        failure_counts[role]=dict(missing_head_top=sum(x['head_contact_seconds'] is None for x in failed),
            incomplete_supported_rotation=sum(x['head_contact_seconds'] is not None and x['completion_seconds'] is None for x in failed),
            remaining=sum(x['completion_seconds'] is not None for x in failed),total=len(failed))
    a=np.array([r['success'] for r in evaluations['control']['rows']]);b=np.array([r['success'] for r in evaluations['candidate']['rows']])
    result=dict(status='rendered-awaiting-visual-review',figures=figures,drift=drift,late_phase=phase,
        failures=failure_counts,candidate_wins=int((~a&b).sum()),candidate_losses=int((a&~b).sum()),
        evidence_sha256={k:sha(OUT/k/'evaluation-analysis.json') for k in evaluations},
        replication='32 matched physical development scenarios per model; one training seed',script_sha256=sha(Path(__file__)))
    write(OUT/'figure-analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('late_phase','evidence_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
