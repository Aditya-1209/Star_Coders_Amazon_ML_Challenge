#!/usr/bin/env python3
"""R16: gated neural pair and business models for the Mumbai L4 / 128 GB VM."""
from pathlib import Path
import sys
import run_r10
import run_r13


def parser(argv=None):
    p = run_r13.parser(argv)
    p.description = __doc__
    p.set_defaults(work=Path('work/r16'), output=Path('output/r16'),
                   target_local=.99, target_leaderboard=.99, max_hours=16)
    return p


def commands(args):
    if not args.r13_features:
        raise ValueError('R16 requires sibling evidence')
    stages = run_r10.commands(args)
    # Preserve the measured R12 CE sample recipe while isolating the effect of
    # new final models. A second expensive encoder/CE is unnecessary here.
    ce, outputs = stages['ce_train']
    stages['ce_train'] = ([x for x in ce if x != '--mine-stage2'], outputs)
    for stage in ('fit', 'select', 'evaluate', 'inference'):
        command, _ = stages[stage]
        command = ['er_v2.r16' if x == 'er_v2.r10' else x for x in command if x != '--r13-features']
        w = args.work
        outputs = {
            'fit': [w / 'r16_models.json', *[w / 'models' / f'r16_{family}_{kind}.json'
                    for family in ('reference', 'enhanced') for kind in ('pair', 'business')],
                    w / 'models/r16_presence.json'],
            'select': [w / 'selection.json'], 'evaluate': [w / 'metrics.json'],
            'inference': [w / 'r16_test_predictions.parquet', args.output / 'candidate_pairs.tsv',
                          args.output / 'matching_results.tsv'],
        }[stage]
        stages[stage] = (command, outputs)
    return stages


if __name__ == '__main__':
    run_r10.main(argument_parser=parser(sys.argv[1:]), command_builder=commands)
