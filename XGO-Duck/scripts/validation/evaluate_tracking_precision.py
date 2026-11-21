"""Run the common fixed three-command evaluator for one precision arm."""
import argparse
import evaluate_official_bootstrap as entry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=('control','candidate'));args=parser.parse_args()
    entry.OUT=entry.ROOT/'records/reports/tracking-precision'/args.arm
    entry.PLAN=entry.OUT/'screen-protocol.json'
    entry.main()


if __name__=='__main__':main()
