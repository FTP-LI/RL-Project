"""Read the real PPO page and originals on Windows as learning proceeds."""
from datetime import datetime
import hashlib
from pathlib import Path
import json
import platform
import sys
from time import monotonic, sleep
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/getup_early_support_ppo.json'
OUT = ROOT/'records/reports/getup-early-support-ppo'
PAGE = ROOT/'records/getup-early-support-ppo-live.html'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(data, indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def preflight():
    assert platform.system()=='Windows' and not PAGE.exists()
    protocol = read(PLAN)
    proof = read(ROOT/read(OUT/'preflight.json')['prior_windows_observation'])
    assert proof['status']=='passed' and proof['running_snapshots']>=3
    previous = ROOT/proof['page']
    prior_html = previous.read_text(encoding='utf-8')
    known = next(row['latest_clip'] for row in reversed(proof['observations']) if 'latest_clip' in row)
    original = (ROOT/read(ROOT/'records/reports/getup-early-support-value/analysis.json')['record'])/known['path']
    assert sha(original)==known['sha256'] and quote(known['path']) in prior_html
    PAGE.write_text('<!doctype html><meta charset="utf-8"><title>XGO-Duck 起身 PPO</title>'
        '<h1>XGO-Duck 起身 PPO</h1><p>已核验观察入口，等待正式运行。</p>'
        '<a href="getup-early-support-value-live.html">查看已完成的价值检查和原片</a>',encoding='utf-8')
    assert 'getup-early-support-value-live.html' in PAGE.read_text(encoding='utf-8')
    write(OUT/'windows-entry.json',dict(status='passed', protocol_sha256=sha(PLAN), observation=protocol['observation'],
        windows_can_resolve_page=True, verified_prior_clip_sha256=sha(original)))
    print('EARLY_SUPPORT_PPO_WINDOWS_ENTRY_PASSED',flush=True)


def watch():
    assert platform.system()=='Windows'
    assert read(OUT/'windows-entry.json')['protocol_sha256']==sha(PLAN)
    protocol = read(PLAN)
    seen = {}
    deadline = monotonic()+2100
    result_path = OUT/'windows-observations.json'
    while monotonic()<deadline:
        status_file = OUT/'coordinator.json'
        if not status_file.is_file():
            sleep(2)
            continue
        try:
            state = read(status_file)
            html = PAGE.read_text(encoding='utf-8')
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            sleep(2)
            continue
        role = state['role']
        name = protocol['arms'][role]
        folder = ROOT/'records/experiments'/name
        path = folder/'observation.json'
        if path.is_file() and name in html:
            try:
                snap = read(path)
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                sleep(2)
                continue
            key = (role,snap['status'],snap['completed_updates'])
            if key not in seen and (snap['rows'] or snap['status']!='training' or snap['frame']):
                entry = dict(role=role,run_id=name,status=snap['status'],completed_updates=snap['completed_updates'],
                    observed_at=datetime.now().astimezone().isoformat(), source_at=snap['updated_at'],
                    auto_refresh='http-equiv="refresh"' in html, curve_points=len(snap['rows']),
                    has_plot='<polyline' in html)
                for field in ('frame','latest_clip'):
                    media = snap.get(field)
                    if media and quote(media) in html and (folder/media).is_file():
                        entry[field] = dict(file=media,sha256=sha(folder/media))
                seen[key] = entry
                write(result_path,dict(status='observing',observations=list(seen.values())))
                print('EARLY_SUPPORT_PPO_VIEW',role,snap['status'],snap['completed_updates'],flush=True)
        if state['status'] in ('collected-review-required','stopped-review-required'):
            break
        sleep(2)
    summary = {}
    for role in protocol['arms']:
        rows = [row for row in seen.values() if row['role']==role]
        summary[role] = dict(running_curves=sum(row['status']=='training' and row['has_plot'] and
            row['auto_refresh'] and row['curve_points']>0 for row in rows),
            original_frames=sum('frame' in row for row in rows),complete_clips=sum('latest_clip' in row for row in rows))
    passed = all(row['running_curves']>=3 and row['original_frames']>0 and row['complete_clips']>0 for row in summary.values())
    write(result_path,dict(status='passed' if passed else 'partial-observation',roles=summary,observations=list(seen.values())))
    print('EARLY_SUPPORT_PPO_VIEW_RESULT',passed,summary,flush=True)


if __name__=='__main__':
    (preflight if sys.argv[1:]==['--preflight'] else watch)()
