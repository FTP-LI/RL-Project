"""Read a running local evaluation page and verify that its evidence changes."""
from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from urllib.parse import quote


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('page', type=Path)
    parser.add_argument('--timeout', type=float, default=800.)
    args = parser.parse_args()
    assert args.record.is_dir() and args.timeout > 0.
    seen = {}
    deadline = time.monotonic()+args.timeout
    while time.monotonic() < deadline:
        try:
            state = json.loads((args.record/'state.json').read_text(encoding='utf-8'))
            html = args.page.read_text(encoding='utf-8')
        except (OSError, ValueError):
            time.sleep(2)
            continue
        frame = state.get('frame')
        if state['run_id'] not in html or (frame and quote(frame) not in html):
            time.sleep(2)
            continue
        status, controls = state['status'], state['controls']
        key = (status, controls)
        if key not in seen:
            entry = dict(status=status, controls=controls, transitions=state['transitions'],
                         source_updated_at=state['updated_at'], observed_at=datetime.now().astimezone().isoformat(),
                         automatically_refreshes='http-equiv="refresh"' in html, throughput_points=len(state['updates']))
            for field in ('frame', 'latest_clip'):
                media = state.get(field)
                if media and quote(media) in html and (args.record/media).is_file():
                    content = (args.record/media).read_bytes()
                    assert len(content) > 48
                    entry[field] = dict(path=media, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            seen[key] = entry
        if status in {'completed', 'stopped-review-required'}:
            break
        time.sleep(2)
    running = [entry for entry in seen.values() if entry['status'] == 'running']
    final = next(reversed(seen.values())) if seen else {}
    original_frames = sum('frame' in row for row in running)
    clips = sum('latest_clip' in row for row in seen.values())
    complete = final.get('status') == 'completed' and 'latest_clip' in final
    passed = len(running) >= 3 and original_frames > 0 and complete
    result = dict(status='passed' if passed else 'review-required', observer_host=platform.system(),
                  running_snapshots=len(running), running_snapshots_with_original_frame=original_frames,
                  snapshots_with_complete_clip=clips, page=args.page.as_posix(), observations=list(seen.values()))
    target = args.record/'windows-view-verification.json'
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('WALK_LIVE_VIEW', result['status'], len(running), original_frames, clips, flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
