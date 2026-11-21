"""Compare saved supported-rotation evidence for successful and failed worlds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_metrics import rotation


def metrics(trace: dict) -> dict:
    q = trace["quaternion"]
    r = rotation(q)
    lateral = np.maximum(np.linalg.norm(r[..., :2, 1], axis=-1), 1e-6)
    forward = (r[..., 0, 0] * r[..., 1, 1] - r[..., 1, 0] * r[..., 0, 1]) / lateral
    pitch = np.arctan2(-r[..., 2, 0], forward)
    delta = np.zeros_like(pitch)
    delta[1:] = (np.diff(pitch, axis=0) + np.pi) % (2*np.pi) - np.pi
    flat = np.clip((.866 - np.abs(r[..., 2, 1])) / (.866 - .5), 0, 1)
    flat = flat * flat * (3 - 2 * flat)
    supported_increment = np.where(delta < 0, delta, delta * flat * trace["ground_contact"])
    geometric = np.cumsum(delta, axis=0)
    supported = np.cumsum(supported_increment, axis=0)
    return dict(geometric=geometric, supported=supported, delta=delta, flat=flat,
                supported_increment=supported_increment, rotation=r)


def group_summary(rows: list[dict]) -> dict:
    keys = ('supported_max_deg', 'geometric_max_deg', 'positive_delta_last120_deg',
            'supported_delta_last120_deg', 'support_ratio_last120', 'ground_contact_last120',
            'foot_contact_last120', 'flatness_last120', 'head_contact_last120')
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = dict(count=len(values), median=float(np.median(values)),
                           q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                           minimum=float(values.min()), maximum=float(values.max()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    args = parser.parse_args(); record = args.record.resolve()
    with np.load(record / "trace.npz") as data: trace = {key: data[key] for key in data.files}
    raw = json.loads((record / "raw-score.json").read_text(encoding="utf-8"))
    successful = np.asarray(raw["roll_switch_at"], dtype=int) >= 0
    m = metrics(trace)
    rows = []
    n = trace["quaternion"].shape[1]
    for world in range(n):
        geom = m["geometric"][:, world]; sup = m["supported"][:, world]
        region = (geom >= np.deg2rad(240)) & (geom <= np.deg2rad(390))
        if not region.any(): region = np.ones_like(geom, dtype=bool)
        positive = np.maximum(m["delta"][:, world], 0) * region
        supported_positive = np.maximum(m["supported_increment"][:, world], 0) * region
        ground = trace["ground_contact"][:, world][region]
        feet = trace["foot_contact"][:, world].all(axis=-1)[region]
        flat = m["flat"][:, world][region]
        head = trace["head_contact"][:, world][region]
        rows.append(dict(world=world, outcome='success' if successful[world] else 'failure',
                         supported_max_deg=float(np.rad2deg(sup.max())), geometric_max_deg=float(np.rad2deg(geom.max())),
                         positive_delta_last120_deg=float(np.rad2deg(positive.sum())),
                         supported_delta_last120_deg=float(np.rad2deg(supported_positive.sum())),
                         support_ratio_last120=float(supported_positive.sum()/max(positive.sum(),1e-9)),
                         ground_contact_last120=float(ground.mean()), foot_contact_last120=float(feet.mean()),
                         flatness_last120=float(flat.mean()), head_contact_last120=float(head.mean()),
                         first_ground_contact=int(np.flatnonzero(trace["ground_contact"][:, world])[0]) if trace["ground_contact"][:, world].any() else None))
    failed_rows = [row for row in rows if row['outcome']=='failure']; success_rows = [row for row in rows if row['outcome']=='success']
    result = dict(status='completed-saved-data-only', record=record.relative_to(ROOT).as_posix(),
                  physical_steps_added=0, learning_updates=0, failed_worlds=[row['world'] for row in failed_rows],
                  groups={'success':group_summary(success_rows),'failure':group_summary(failed_rows)},
                  failed_rows=failed_rows,
                  interpretation=(
                      'The gap is measured from the frozen evaluator support definition. Compare ground contact and '
                      'support_ratio_last120 before changing the actor or the acceptance gate; do not treat geometric '
                      'rotation alone as a roll success.'),
                  no_gate_relaxation=True)
    out = ROOT / 'records/reports/continuous-roll-bridge-disabled-boundary-s1216/supported-rotation-review.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    analysis = ROOT / 'records/reports/continuous-roll-bridge-disabled-boundary-s1216/analysis.json'
    merged=json.loads(analysis.read_text(encoding='utf-8')); merged['supported_rotation_review']=out.relative_to(ROOT).as_posix(); analysis.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
