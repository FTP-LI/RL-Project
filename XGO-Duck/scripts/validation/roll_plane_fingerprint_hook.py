"""Temporarily use the plane-aware read-only inventory for fingerprinting."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import roll_entry_snapshot
import roll_plane_fingerprint_inventory as plane


@contextmanager
def plane_inventory():
    # Preserve the old inventory function behind the adapter. Patching both
    # references avoids recursion and restores the original binding on exit.
    old = roll_entry_snapshot.roll_slots
    with patch.object(plane, 'roll_entry_snapshot', SimpleNamespace(roll_slots=old)), \
         patch.object(roll_entry_snapshot, 'roll_slots', side_effect=plane.plane_roll_slots):
        yield
