"""Regression checks for the third-stage model and output selection."""
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import polars as pl

from er_v2.graph import HOP_SCHEMA, SUPPORT_SCHEMA, expand, support_features
from er_v2.stage3 import main


class Stage3Tests(TestCase):
    def test_no_graph_anchors_or_support(self):
        empty = pl.DataFrame(schema={'sidx': pl.UInt32, 'a': pl.UInt32, 'pa': pl.Float32})
        self.assertEqual(dict(expand(empty, pl.DataFrame(), pl.DataFrame()).schema), HOP_SCHEMA)
        pairs = pl.DataFrame({'sidx': [0], 'tidx': [1]}, schema={'sidx': pl.UInt32, 'tidx': pl.UInt32})
        self.assertEqual(dict(support_features(pairs, empty, pl.DataFrame(), workers=1).schema), SUPPORT_SCHEMA)

    def test_missing_addresses_are_not_positive_support(self):
        pairs = pl.DataFrame({'sidx': [0], 'tidx': [1]})
        anchors = pl.DataFrame({'sidx': [0], 'a': [2], 'pa': [.9]})
        text = pl.DataFrame({'idx': [1, 2], 'core_r': ['alpha', 'alpha'],
                             'cc_r': ['alpha', 'alpha'], 'addr_r': ['', '']})
        result = support_features(pairs, anchors, text, workers=1)
        self.assertEqual(result['sup_addr_ratio'][0], 0)
        self.assertEqual(result['sup_addr_valid'][0], 0)

    def test_stage2_winner_copies_exact_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models = root/'models'
            earlier = root/'stage2'
            final = root/'output'
            models.mkdir()
            earlier.mkdir()
            (models/'stage3_metrics.json').write_text(
                json.dumps({'feature_version': 'r5', 'selected_stage': 'stage2'}))
            for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
                (earlier/name).write_bytes((name+'\n').encode())
            argv = ['er_v2.stage3', '--split', 'test', '--model-dir', str(models),
                    '--work', str(root), '--output', str(final), '--stage2-output', str(earlier)]
            with mock.patch.object(sys, 'argv', argv):
                main()
            for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
                self.assertEqual((final/name).read_bytes(), (earlier/name).read_bytes())
