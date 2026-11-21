"""Retry only the transient disappearance of the current PPO status file."""

import native_run_observer as observer
from native_ppo_viewer import main
from native_run_observer_bounded import bounded_publish


if __name__=='__main__':
    observer.publish=bounded_publish
    raise SystemExit(main())
