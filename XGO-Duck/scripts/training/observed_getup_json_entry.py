"""Normalize published metrics to their JSON representation before comparison.

The first actual observed run exposed a tuple in replay_drift_before_rad.
JSON stores that tuple as a list, so comparing decoded disk data with the
Python object incorrectly stopped training. Normalize only the representation;
keep every frozen identity, value, progress, source and media check intact.
The companion offline check reproduces the old error with the saved metric.
"""
import json
from pathlib import Path

import observed_getup_entry as entry
from native_ppo_observation import NativePPOObservation


class JsonSnapshotObservation(NativePPOObservation):
    def publish(self, record, snapshot):
        normalized = json.loads(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
        return super().publish(record, normalized)


def main():
    # Apply only within this newly named process. Closed source bytes and old
    # protocols remain unchanged, and the new entry must itself be frozen.
    entry.ENTRY = Path(__file__).resolve()
    entry.REQUIRED_SOURCES = entry.REQUIRED_SOURCES | {entry.ENTRY.relative_to(entry.ROOT).as_posix()}
    entry.NativePPOObservation = JsonSnapshotObservation
    return entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
