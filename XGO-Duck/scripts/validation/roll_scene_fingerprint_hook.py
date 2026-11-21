"""Temporarily connect the scene-aware read-only fingerprint inventory."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import roll_entry_snapshot
import roll_plane_fingerprint_inventory as plane
from roll_scene_fingerprint_inventory import scene_roll_slots


@contextmanager
def scene_inventory():
    old = roll_entry_snapshot.roll_slots
    with patch.object(plane, 'roll_entry_snapshot', SimpleNamespace(roll_slots=old)), \
         patch.object(roll_entry_snapshot, 'roll_slots', side_effect=scene_roll_slots):
        yield
