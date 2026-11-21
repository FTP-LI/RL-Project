"""Keep the native observer alive through a brief missing-source read.

Only FileNotFoundError for the exact status source is retried, for at most
half a second. Missing media, malformed/stale status, an exited observer, and
all other errors still follow the original observer's stop behaviour. This
adapter does not run or resume physics or training.
"""
from pathlib import Path
import time

import native_run_observer as observer

ORIGINAL_PUBLISH = observer.publish
RETRY_SECONDS = .5
RETRY_INTERVAL = .025


def bounded_publish(source, output, run_id, mode, max_age, last_chart, chart_interval):
    deadline = time.monotonic() + RETRY_SECONDS
    while True:
        try:
            return ORIGINAL_PUBLISH(source, output, run_id, mode, max_age, last_chart, chart_interval)
        except FileNotFoundError as error:
            missing = Path(error.filename).resolve() if error.filename else None
            if missing != Path(source).resolve() or time.monotonic() >= deadline:
                raise
            time.sleep(min(RETRY_INTERVAL, max(0., deadline - time.monotonic())))


def main():
    observer.publish = bounded_publish
    return observer.main()


if __name__ == '__main__':
    raise SystemExit(main())
