"""Render aligned right-turn replays from already sampled and verified state."""

from argparse import ArgumentParser

from replay_walk_right_startup_seven_command import render
from run_walk_right_support_seven_command import ARMS, LABELS, OUT, PLAN, SEEDS, checked


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    seed = parser.parse_args().seed
    render(checked(), seed, PLAN, OUT, ARMS, tuple(LABELS[arm] for arm in ARMS))
