#!/usr/bin/env python3
"""R13 accuracy run, with explicit L4 VM and RTX 3060 desktop profiles."""
import argparse
from pathlib import Path
import sys
import run_r10

PROFILES = {
    'vm': dict(threads=30, ce_batch=64, ce_accumulation=1, ce_checkpointing=False,
               ce_score_batch=1024, encode_batch=1024, shard_pairs=6000000,
               ce_train_businesses=250000, max_hours=11, reserve_gb=20),
    'desktop': dict(threads=12, ce_batch=16, ce_accumulation=4, ce_checkpointing=True,
                    ce_score_batch=128, encode_batch=256, shard_pairs=2000000,
                    ce_train_businesses=180000, max_hours=24, reserve_gb=12),
}


def parser(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--profile', choices=PROFILES, default='vm')
    selected, _ = selector.parse_known_args(argv)
    p = run_r10.parser()
    p.description = __doc__
    p.add_argument('--profile', choices=PROFILES, default=selected.profile)
    p.set_defaults(**PROFILES[selected.profile], work=Path('work/r13'), output=Path('output/r13'),
        ann='gpu-exact', r13_features=True, r11_features=True, encoder_pairs=1000000,
        target_local=.99, target_leaderboard=.985)
    return p


def main():
    run_r10.main(argument_parser=parser(sys.argv[1:]))


if __name__ == '__main__':
    main()
