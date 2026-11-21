"""Track forward-roll phase from policy gravity and request one bounded head-roll pulse."""

import numpy as np


class HeadSupportPulse:
    def __init__(self, worlds, entry_degrees, controls):
        self.entry = np.deg2rad(entry_degrees)
        self.controls = controls
        self.active = np.zeros(worlds,bool)
        self.latched = np.zeros(worlds,bool)
        self.previous = np.zeros(worlds)
        self.progress = np.zeros(worlds)
        self.remaining = np.zeros(worlds,np.int32)

    def advance(self, gravity, active):
        """Use the current actor sample and the already active roll mode only."""
        x,y,z = np.asarray(gravity,dtype=np.float64).T
        lateral = np.sqrt(np.maximum(1.-y*y,0.))
        angle = np.arctan2(x,-z/np.maximum(lateral,1e-6))
        step = (angle-self.previous+np.pi)%(2*np.pi)-np.pi
        self.progress = np.where(active & self.active,self.progress+step,0.)
        self.previous = angle
        self.active = np.asarray(active,bool).copy()
        self.latched &= self.active
        trigger = self.active & ~self.latched & (self.progress >= self.entry)
        self.latched |= trigger
        self.remaining = np.where(trigger,self.controls,np.where(self.active,self.remaining,0))
        selected = self.remaining > 0
        self.remaining -= selected
        return selected


def projected_head_roll_target(original, selected, lower, increment):
    return np.where(selected,np.maximum(original+increment,lower),original)
