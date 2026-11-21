"""Plot the verified, immutable continuation prefix; never run a simulator.

Raw TensorBoard values are shown without smoothing. Cumulative completed
updates include the saved parent updates, while event timings describe this
invocation only. Training reward and episode length are not gait acceptance.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from figure_style import apply_figure_style


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = root/'records/experiments'/args.run_id
    assert run.parent == root/'records/experiments'
    check = json.loads((run/'resume-live-check.json').read_text())
    assert check['status'] == 'passed'
    evidence = Path(check['evidence_directory'])
    assert evidence.is_relative_to(run)
    for name, expected in check['input_hashes'].items():
        assert sha(evidence/name) == expected
    ea = EventAccumulator(str(evidence/'tensorboard'), size_guidance={'scalars': 0}).Reload()
    tags = {'reward': 'Train/mean_reward', 'episode_steps': 'Train/mean_episode_length',
            'transitions_per_second': 'Perf/total_fps'}
    events = {key:{e.step:e.value for e in ea.Scalars(tag)} for key,tag in tags.items()}
    labels = range(check['first_iteration'], check['last_verified_iteration']+1)
    rows = [dict(iteration=label, completed_updates=label+1,
                 **{key:values[label] for key,values in events.items()}) for label in labels]
    out = evidence/'figures'
    out.mkdir(exist_ok=True)
    with (out/'progress.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    apply_figure_style(sizes=(11,10,9), font='DejaVu Sans')
    matplotlib.rcParams['savefig.bbox'] = None
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
    fig.subplots_adjust(left=.16, right=.96, top=.96, bottom=.10, hspace=.24)
    keys = [('reward', 'Training reward'), ('episode_steps', 'Episode length (steps)'),
            ('transitions_per_second', 'Transitions / s')]
    for axis, (key, label) in zip(axes, keys):
        axis.plot([r['completed_updates'] for r in rows], [r[key] for r in rows],
                  color='#0072B2', linewidth=1.4)
        axis.set_ylabel(label)
        axis.margins(x=.04, y=.12)
        axis.yaxis.set_major_locator(MaxNLocator(4, prune='both'))
    axes[-1].yaxis.set_major_formatter(FuncFormatter(lambda value, _:f'{value/1000:g}k'))
    axes[-1].xaxis.set_major_locator(MaxNLocator(6, integer=True, prune='both'))
    axes[-1].set_xlabel('Cumulative completed updates')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    collisions = []
    for i, (a, box) in enumerate(texts):
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1), a.get_text()
        for b, other in texts[i+1:]:
            if box.overlaps(other):
                collisions.append([a.get_text(),b.get_text()])
    for axis in axes:
        own = set(axis.get_xticklabels(which='both')+axis.get_yticklabels(which='both'))
        for spine in axis.spines.values():
            if spine.get_visible():
                box = spine.get_window_extent(renderer)
                collisions.extend([t.get_text(),'spine'] for t,tbox in texts if t not in own and tbox.overlaps(box))
    assert not collisions, collisions
    figure = out/'continuation-progress.png'
    fig.savefig(figure, dpi=300, bbox_inches=None, facecolor='white')
    picture = Image.open(figure)
    # Crop each actual axes extent. Fixed fractional strips could include the
    # next panel's spine, even when the full figure was laid out correctly.
    sx,sy=picture.width/fig.bbox.width,picture.height/fig.bbox.height
    for index,axis in enumerate(axes):
        box=axis.get_tightbbox(renderer)
        picture.crop((max(0,int((box.x0-4)*sx)),max(0,int(picture.height-(box.y1+4)*sy)),
                      min(picture.width,int((box.x1+4)*sx)+1),min(picture.height,int(picture.height-box.y0*sy)+1))).save(out/f'panel-{index+1}.png')
    result = dict(status='passed', figure_sha256=sha(figure), data_sha256=sha(out/'progress.csv'),
                  source_check_sha256=sha(evidence/'check.json'), text_collisions=collisions,
                  points=len(rows), first_completed_updates=rows[0]['completed_updates'],
                  last_completed_updates=rows[-1]['completed_updates'], smoothing=False,
                  visual_inspection='pending', scope='Training prefix, not fixed-command gait evaluation')
    (out/'plot-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'figure':str(figure),**result}, indent=2))


if __name__ == '__main__':
    main()
