"""Causal roll phase from projected gravity, initialized once at the trigger.

The source roll projects pitch after removing lateral tilt. Match its geometry
using only the actor's three gravity components, unwrap consecutive samples,
and subtract the trigger orientation. Real IMU fusion remains unvalidated.
"""
import numpy as np


class GravityPhase:
    def __init__(self):
        self.previous = None
        self.total = None

    def update(self, gravity):
        g = np.asarray(gravity, dtype=float)
        assert g.ndim == 2 and g.shape[1] == 3 and np.isfinite(g).all()
        g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        lateral = np.sqrt(np.maximum(1-g[:,1]**2, 1e-12))
        angle = np.arctan2(g[:,0], -g[:,2]/lateral)
        if self.previous is None:
            self.total = np.zeros_like(angle)
        else:
            assert angle.shape == self.previous.shape
            delta = angle-self.previous
            wrapped = (delta+np.pi) % (2*np.pi)-np.pi
            wrapped = np.where((wrapped == -np.pi) & (delta > 0), np.pi, wrapped)
            self.total += wrapped
        self.previous = angle.copy()
        return self.total.copy()
