"""Repeat the frozen native sampler at intermediate, evidence-sized scales."""

from pathlib import Path
from unittest.mock import patch

import run_walk_backward_native_scale as runner


ROOT = runner.ROOT
SOURCE = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_backward_native_scale_refinement.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
PAGE = ROOT / "records/walk-backward-native-scale-refinement-live.html"


def sources(recipe):
    assert recipe["status"] == "draft-refinement-no-physics"
    assert not any(recipe[key] for key in ("heldout_used", "hardware_io", "physical_substeps_authorized",
                                            "learning_updates_authorized"))
    hashes = dict(recipe[name] for name in ("reference_summary", "reference_verification",
                                           "capacity_stop_telemetry"))
    for path, digest in hashes.items():
        assert runner.sha(ROOT / path) == digest, path
    summary = runner.read(ROOT / recipe["reference_summary"][0])
    verification = runner.read(ROOT / recipe["reference_verification"][0])
    assert summary["status"] == "passed-scale-comparison"
    assert verification["status"] == "passed-independent-scale-verification"
    assert verification["summary_sha256"] == recipe["reference_summary"][1]
    assert summary["selected"]["worlds"] == verification["selected_worlds"] == recipe["reference_worlds"]
    stopped, = [trial for trial in summary["trials"] if trial["status"] == "capacity-stop"]
    assert stopped["worlds"] == 4096 and stopped["actual_transitions"] == 0
    telemetry = runner.read(ROOT / recipe["capacity_stop_telemetry"][0])
    peak = max(row["used_mib"] for row in telemetry)
    total, = set(row["total_mib"] for row in telemetry)
    cap = total * recipe["memory_cap_fraction"]
    assert peak > cap == recipe["memory_budget_mib"]
    low = max(row["memory"]["peak_mib"] for row in summary["trials"]
              if row["worlds"] == 1024 and row["status"] == "passed")
    projections = {str(n): low + (peak - low) * (n - 1024) / (4096 - 1024)
                   for n in recipe["proposed_worlds"]}
    assert all(value < cap for value in projections.values())
    for path in (DRAFT, runner.PLAN, runner.SOURCE):
        hashes[runner.rel(path)] = runner.sha(path)
    return hashes, projections


def main():
    recipe = runner.read(DRAFT)
    sizes = tuple(recipe["proposed_worlds"])
    hashes, projections = sources(recipe)
    original_write = runner.write

    def write(path, value):
        if value.get("status") == "frozen-native-backward-scale":
            assert value["warmup_controls"] == recipe["warmup_controls"]
            assert value["timed_controls"] == recipe["timed_controls"]
            assert value["memory_cap_fraction"] == recipe["memory_cap_fraction"]
            assert value["minimum_paired_gain"] == recipe["minimum_paired_gain"]
            value["source_sha256"].update(hashes)
            value["refinement"] = dict(reference_worlds=recipe["reference_worlds"], projected_mib=projections)
        original_write(path, value)

    with patch.multiple(runner, SOURCE=SOURCE, PLAN=PLAN, PAGE=PAGE, SIZES=sizes, SEED=recipe["seed"], write=write):
        runner.main()


if __name__ == "__main__":
    main()
