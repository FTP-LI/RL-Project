"""Classify support from current policy input and bound a stance-specific bias."""

import numpy as np


class StanceObserver:
    def __init__(self, model, separation):
        assert model["class_order"] == ["other", "left-only", "right-only"]
        self.columns = np.asarray(model["columns"], dtype=np.int64)
        self.mean = np.asarray(model["means"], dtype=np.float64)
        self.scale = np.asarray(model["scales"], dtype=np.float64)
        self.weights = np.asarray(model["weights"], dtype=np.float64)
        self.separation = separation
        count = len(self.columns)
        assert self.weights.shape == (count+1, 3) and self.mean.shape == self.scale.shape == (count,)
        assert (self.scale > 0).all() and np.isfinite(self.weights).all()

    def predict(self, observed):
        z = (observed[..., self.columns]-self.mean)/self.scale
        scores = z@self.weights[:-1]+self.weights[-1]
        best = scores.argmax(axis=-1)
        ordered = np.sort(scores, axis=-1)
        margin = ordered[..., -1]-ordered[..., -2]
        return np.where(margin >= self.separation, best, 0)


def phase_commands(original, phase, directions, maximum_delta, limit):
    """Leave abstentions exact and limit both the added input and total input."""
    assert original.shape == phase.shape and np.isfinite(original).all()
    assert (abs(original) <= limit+1e-7).all() and np.isin(phase, (0, 1, 2)).all()
    offset = np.asarray(directions, dtype=original.dtype)[phase]*maximum_delta
    bounded = np.clip(original+offset, -limit, limit)
    return np.where(phase == 0, original, bounded)
