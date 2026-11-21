"""Display actual PPO metrics and detect an exited worker promptly.

This only changes the metric selection in this process.  The historical viewer
and previously frozen experiments keep their original source bytes.
"""
import argparse
import os
import sys

import native_run_observer as observer


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--owner-pid", required=True, type=int)
    args, remaining = parser.parse_known_args()
    if args.owner_pid <= 1:
        raise ValueError("Invalid PPO worker process identifier")
    original_publish = observer.publish

    def guarded_publish(*values, **kwargs):
        # Check on both sides of rendering so a worker exit during a slow chart
        # cannot leave its last update presented as a healthy live status.
        if os.getppid() != args.owner_pid:
            raise ValueError("PPO worker process exited")
        result = original_publish(*values, **kwargs)
        if os.getppid() != args.owner_pid:
            raise ValueError("PPO worker process exited")
        return result

    observer.publish = guarded_publish
    observer.METRICS = (
        ("environment_reward_mean", "环境回报", "平均回报"),
        ("policy_loss", "策略损失", "损失"),
        ("value_loss", "价值损失", "损失"),
        ("net_transitions_per_second", "采样速度", "转移 / 秒"),
    )
    sys.argv = [sys.argv[0], *remaining]
    return observer.main()


if __name__ == "__main__":
    raise SystemExit(main())
