"""Confirm Windows observes the real run for one frozen development cohort."""

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import observe_walk_right_startup_seven_command as observer


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("known", "fresh"), required=True)
    cohort = parser.parse_args().cohort
    plan = Path(__file__).resolve().parents[2] / "configs/experiments/common_walk_right_support_forward_margin.json"
    protocol = plan.read_bytes()
    read_json = observer.json.loads

    def only_requested_cohort(data, *args, **kwargs):
        spec = read_json(data, *args, **kwargs)
        if data == protocol:
            spec["cases"] = {key: item for key, item in spec["cases"].items() if item["cohort"] == cohort}
            assert len(spec["cases"]) == 3
        return spec

    with patch.object(observer, "PLAN", plan), patch.object(observer.json, "loads", only_requested_cohort):
        observer.main()
