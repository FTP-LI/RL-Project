"""Run the unchanged fixed-policy collector for this pair's frozen screen."""
import argparse
import evaluate_official_bootstrap as entry

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=('control','candidate'));arm=p.parse_args().arm
    entry.OUT=entry.ROOT/'records/reports/planar-filter'/arm
    entry.PLAN=entry.OUT/'screen-protocol.json'
    entry.main()
