"""Compare contact exit and support in the pre-contact matched pairs."""
from pathlib import Path

import numpy as np

from audit_getup_contact_entry import OUT, RELEASE, first_run, source
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


COHORTS = ('1016', '1050', '1054')
DT = .02
OFFSETS = np.arange(-5, 31)
CHECK_OFFSETS = (0, 5, 8, 12, 15, 25)
RIGHT = 'right_foot_collision'
METRICS = ('head_contact', 'both_feet', 'tilt_deg', 'pitch_rate_rad_s', 'height_m')


def same_contact_pairs(cohort):
    cases = {row['world']: row for row in cohort['cases']}
    pairs = []
    for row in cohort['matched_pairs']:
        success = cases[row['successful_world']]
        failure = cases[row['failed_world']]
        if RIGHT in success['pairs_at_head_entry'] and RIGHT in failure['pairs_at_head_entry']:
            pairs.append((success, failure))
    return pairs


def extract(trace, rows, hold, pairs):
    gravity = projected_gravity(trace['quaternion'])
    tilt = np.rad2deg(np.arccos((-gravity[..., 2]).clip(-1, 1)))
    values = dict(head_contact=trace['native_head_leg_contact'], both_feet=trace['foot_contact'].all(axis=-1),
                  tilt_deg=tilt, pitch_rate_rad_s=trace['angular_velocity'][..., 1], height_m=trace['height'])
    out = {role: {key: [] for key in METRICS} for role in ('success', 'failure')}
    reviewed_pairs = []
    for pair in pairs:
        reviewed = {}
        for i, (role, case) in enumerate(zip(out, pair, strict=True)):
            world, head = case['world'], case['first_head_contact_step'] - 1
            assert rows[world]['release_pose'] == 'supine'
            assert bool(rows[world]['hold_start_seconds'] is not None and hold[world]) == (i == 0)
            assert head >= RELEASE and values['head_contact'][head, world]
            assert not values['head_contact'][RELEASE:head, world].any()
            window = head + OFFSETS
            assert trace['active'][window, world].all()
            for key in METRICS:
                out[role][key].append(values[key][window, world])
            clear = ~values['head_contact'][:, world]
            clear[:head + 1] = False
            first_clear = first_run(clear, 5)
            reviewed[role] = dict(world=world, first_head_step=head + 1,
                contact_clear_delay_s=None if first_clear is None else round((first_clear - head) * DT, 2),
                first_both_feet_step=case['first_both_feet_step'], first_upright_step=case['first_upright_step'])
        reviewed_pairs.append(reviewed)
    stacked = {role: {key: np.stack(series) for key, series in values.items()} for role, values in out.items()}
    return stacked, reviewed_pairs


def summarize(series, pairs):
    success, failure = series['success'], series['failure']
    detail = {}
    for offset in CHECK_OFFSETS:
        i = offset - OFFSETS[0]
        entry = {}
        for role, group in series.items():
            entry[role] = dict(head_contacts=int(group['head_contact'][:, i].sum()),
                both_feet=int(group['both_feet'][:, i].sum()),
                median_tilt_deg=float(np.median(group['tilt_deg'][:, i])),
                median_pitch_rate_rad_s=float(np.median(group['pitch_rate_rad_s'][:, i])),
                median_height_m=float(np.median(group['height_m'][:, i])))
        entry['median_paired_failure_minus_success_tilt_deg'] = float(
            np.median(failure['tilt_deg'][:, i] - success['tilt_deg'][:, i]))
        detail[f'{offset * DT:.2f}'] = entry
    clear = {}
    for role in series:
        observed = [row[role]['contact_clear_delay_s'] for row in pairs
                    if row[role]['contact_clear_delay_s'] is not None]
        clear[role] = dict(count=len(observed), median_delay_seconds=None if not observed else float(np.median(observed)))
    return dict(pairs=len(pairs), by_delay_seconds=detail, clear_before_5s=clear, paired_events=pairs,
                aligned={role: {key: values.tolist() for key, values in group.items()} for role, group in series.items()})


def plot(cohorts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from figure_style import apply_figure_style
    from render_roll_angle_results import checked_save

    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.3), sharex=True, sharey='row')
    fig.subplots_adjust(left=.098, right=.975, bottom=.117, top=.87, wspace=.17, hspace=.24)
    fig.supxlabel('距各自首次头腿接触的时间（秒）', y=.042)
    colors = dict(success='#25769e', failure='#ad5d2b')
    names = dict(success='及时起身且末段持稳', failure='未通过')
    x = DT * OFFSETS
    variables = ('head_contact', 'both_feet', 'tilt_deg')
    labels = ('头腿接触（%）', '双脚同时着地（%）', '躯干倾角（°）')
    for column, seed in enumerate(COHORTS):
        values = cohorts[seed]['aligned']
        axes[0, column].set_title(f'{seed}（{cohorts[seed]["pairs"]} 对）')
        for row, (key, label) in enumerate(zip(variables, labels, strict=True)):
            ax = axes[row, column]
            ax.axvline(0, color='#8a969d', linestyle='--', linewidth=.8, zorder=0)
            if column == 0:
                ax.set_ylabel(label)
            for role, color in colors.items():
                data = np.asarray(values[role][key])
                if row < 2:
                    ax.step(x, 100 * data.mean(axis=0), where='post', color=color, label=names[role], linewidth=1.4)
                else:
                    lo, median, hi = np.percentile(data, [25, 50, 75], axis=0)
                    ax.fill_between(x, lo, hi, color=color, alpha=.14, linewidth=0)
                    ax.plot(x, median, color=color, linewidth=1.4)
            ax.set(xlim=(-.112, .62), xticks=(0, .2, .4, .6))
            if row < 2:
                ax.set(ylim=(-5, 105), yticks=(0, 50, 100))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .98), ncol=2)
    return checked_save(fig, ROOT / 'docs/reports/assets/getup-contact-exit.png')


def main():
    import torch

    parent = OUT / 'analysis.json'
    matched = read(parent)
    assert matched['status'] == 'passed-saved-data-only'
    cohorts, sources = {}, {parent.relative_to(ROOT).as_posix(): sha(parent)}
    for seed in COHORTS:
        branch = 'reference-a' if seed == '1054' else 'original'
        _, trace, rows, hold, _, fingerprints = source(seed, branch)
        for path, digest in fingerprints.items():
            relative = path.relative_to(ROOT).as_posix()
            assert digest == matched['source_sha256'][relative], relative
            sources[relative] = digest
        pairs = same_contact_pairs(matched['cohorts'][seed])
        series, events = extract(trace, rows, hold, pairs)
        cohorts[seed] = summarize(series, events)
        print('CONTACT_EXIT', seed, {k: v for k, v in cohorts[seed].items()
              if k not in ('aligned', 'paired_events')}, flush=True)
    assert not torch.cuda.is_initialized()
    write(OUT / 'exit-analysis.json', dict(status='passed-saved-data-only', cohorts=cohorts,
        source_sha256=sources, script_sha256=sha(Path(__file__)), aligned_offsets_s=(DT * OFFSETS).round(2).tolist(),
        scope='Pre-contact matched pairs with right foot among CPU head contact pairs at both entries. Contact delay describes the native any-head-leg sensor.',
        figure=plot(cohorts), new_physical_steps=0, learning_updates=0, heldout_used=False))


if __name__ == '__main__':
    main()
