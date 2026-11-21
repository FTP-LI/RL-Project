"""Render verified frames from the ramped sequence's full original."""

from argparse import ArgumentParser
from unittest.mock import patch

import run_routed_getup_ramped_forward as ramped
import render_routed_getup_sequence as original


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', action='store_true')
    out = ramped.OUT / 'holdout' if parser.parse_args().holdout else ramped.OUT
    with patch.object(original.study, 'OUT', out):
        original.main()
