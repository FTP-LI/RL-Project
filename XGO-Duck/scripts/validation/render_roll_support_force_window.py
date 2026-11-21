"""Plot the solved loads inside contact gaps in the unchanged roll score."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import PercentFormatter

from figure_style import apply_figure_style
from record_io import write
from render_roll_angle_results import checked_save
from review_roll_support_force_timing import REPORT, ROOT, read, sha


def main():
    timing = REPORT/'contact-sample-timing.json'
    evidence = REPORT/'force-readback.json'
    receipt = read(timing)
    assert receipt['source_sha256']==sha(evidence)
    gain = receipt['groups']['roll-fail']
    count = gain['endpoint_gap_intervals']
    load = gain['nominal_weight_fraction_median_by_substep']
    times = [0,4,8,12,16]
    threshold = .01
    assert len(load)==len(times) and gain['captured_next_solve_intervals']==count and not gain['next_solve_loaded_intervals']
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(12,11,10))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.2,3.8))
    fig.subplots_adjust(left=.14,right=.94,bottom=.20,top=.82)
    ax.plot(times,load,'o-',color='#155A7B',lw=2,ms=6,label='子步法向力中位数')
    ax.axvline(20,color='#94623F',ls='--',lw=1.2)
    ax.plot([20],[threshold],marker='v',ms=7,mfc='white',mec='#94623F',ls='none',label='下一次求解低于1%体重')
    ax.set(xlim=(-.9,21.7),ylim=(-.06,1.03),xticks=[0,4,8,12,16,20],yticks=[0,.25,.5,.75,1],
           xlabel='20毫秒评分间隔内的时间（ms）',ylabel='法向力占标称重量')
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_title(f'滚翻失分场景的{count}个评分间隔',loc='left',pad=10)
    ax.legend(loc='center left',bbox_to_anchor=(.02,.28))
    result = checked_save(fig,REPORT/'contact-window-force.png')
    write(REPORT/'contact-window-figure.json',dict(source_sha256=sha(timing),figure=result,
        endpoint_intervals=count,worlds=gain['worlds_with_endpoint_gaps'],median_nominal_weight_fraction=load,
        physical_read_times_ms=times,next_solve_threshold_fraction=threshold,next_solve_time_ms=20,
        next_solve_below_threshold=count,script_sha256=sha(Path(__file__)),new_physical_steps=0))
    print('ROLL_CONTACT_WINDOW_FIGURE', (REPORT/'contact-window-force.png').relative_to(ROOT))


if __name__=='__main__':
    main()
