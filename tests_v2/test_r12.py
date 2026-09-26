"""R12: exhaustive retrieval, label-free graph features and VM run controls."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from er_v2.neural import exact_batches, block
from er_v2.r10_retrieval import unit
from er_v2.stage3 import competition_features, feature_cols
from er_v2.features import SPLIT_DEPENDENT
from er_v2.train import record_competition, record_competition_frame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_r10


class R12Tests(unittest.TestCase):
    def test_tiled_exact_search_matches_full_dot_products_with_global_ids(self):
        rng = np.random.default_rng(12)
        es, et = unit(rng.normal(size=(11, 16))), unit(rng.normal(size=(23, 16)))
        si, ti = np.array([1, 4, 7]), np.array([2, 5, 8, 12, 15, 19, 22])
        expected = ti[np.argsort(-(es[si] @ et[ti].T), axis=1)[:, :5]]
        for tile in (2, 7):
            batches = list(exact_batches(es, et, si, ti, 5, 2, 'cpu', target_batch=tile))
            np.testing.assert_array_equal(np.concatenate([x[0] for x in batches]), si)
            np.testing.assert_array_equal(np.concatenate([x[2] for x in batches]), expected)
            np.testing.assert_allclose(np.concatenate([x[1] for x in batches]),
                np.take_along_axis(es[si] @ et[ti].T, np.argsort(-(es[si] @ et[ti].T), axis=1)[:, :5], 1), atol=1e-6)
        small = list(exact_batches(es, et, si, ti[:2], 24, 1, 'cpu', target_batch=1))
        self.assertEqual(np.concatenate([x[2] for x in small]).shape, (3, 2))
        self.assertEqual(list(exact_batches(es, et, si, ti[:0], 5, 2, 'cpu')), [])

    def test_exact_block_is_country_scoped_and_atomically_replaces_output(self):
        with TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / 'emb').mkdir()
            (w / 'norm').mkdir()
            for side in (1, 2, 3):
                pl.DataFrame({'idx': np.arange(2, dtype=np.uint32), 'country': ['France', 'India']}).write_parquet(
                    w / 'norm' / f'test_source{side}.parquet')
            np.save(w / 'emb/test_s1.npy', np.eye(2, dtype=np.float16))
            np.save(w / 'emb/test_tg.npy', np.tile(np.eye(2, dtype=np.float16), (2, 1)))
            block(w, 'test', 24, 1, 'cpu')
            pairs = pl.read_parquet(w / 'ncands_test.parquet')
            self.assertEqual(set(pairs.filter(pl.col('sidx') == 0)['tidx']), {0, 2})
            self.assertEqual(set(pairs.filter(pl.col('sidx') == 1)['tidx']), {1, 3})
            original = (w / 'ncands_test.parquet').read_bytes()
            def interrupted(*args, **kwargs):
                raise RuntimeError('interrupted')
            # A MagicMock retains the mmap arguments, artificially keeping files
            # open past block()'s cleanup on Windows. Use a real failing callable.
            with patch('er_v2.neural.exact_batches', new=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    block(w, 'test', 24, 1, 'cpu')
            self.assertEqual((w / 'ncands_test.parquet').read_bytes(), original)

    def test_competition_is_global_across_shards_and_graph_features_exclude_idf(self):
        x = pl.DataFrame({'sidx': [0, 0, 1, 1], 'tidx': [0, 1, 0, 2],
                          'core_tset': [100., 95., 100., 20.], 'addr_tset': [0., 80., 40., 0.],
                          'first_num_eq': [0, 1, 0, 0]}).with_columns(pl.col('sidx', 'tidx').cast(pl.UInt32))
        expected = record_competition_frame(x).sort('sidx', 'tidx')
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for i, part in enumerate(x.iter_slices(2)):
                part.write_parquet(folder / f'part_{i:03d}.parquet')
            actual = record_competition(folder, x.select('sidx', 'tidx')).sort('sidx', 'tidx')
            self.assertTrue(actual.equals(expected))
        graph = competition_features(x, True, True)
        self.assertEqual(graph['rc_n_claim'].to_list(), [2, 1, 2, 1])
        self.assertEqual(graph['rc_name_gap_other'].to_list(), [0., 95., 0., 20.])
        self.assertIn('la_house_unique', graph.columns)
        with_counts = graph.with_columns([pl.lit(100.).alias(c) for c in SPLIT_DEPENDENT])
        self.assertFalse(set(feature_cols(with_counts)) & SPLIT_DEPENDENT)
        self.assertTrue(competition_features(x).equals(x))

    def test_vm_profile_plan_propagates_data_limits_and_device(self):
        args = run_r10.parser().parse_args(['--encoder-pairs', '2000000', '--ce-train-businesses', '250000',
            '--ann', 'gpu-exact', '--r11-features', '--dataset', '/other/student_resource/dataset',
            '--validator', '/other/student_resource/utils/validate_submission.py'])
        stages = run_r10.commands(args)
        self.assertEqual(stages['encoder'][0][stages['encoder'][0].index('--pairs') + 1], '2000000')
        self.assertEqual(stages['ce_train'][0][stages['ce_train'][0].index('--train-businesses') + 1], '250000')
        self.assertIn('--device', stages['ann_train'][0])
        self.assertEqual(stages['validate'][0][1], str(args.validator))
        self.assertIn('--record-competition', stages['train'][0])
        self.assertLess(list(stages).index('select'), list(stages).index('evaluate'))

    def test_invalid_accumulation_shard_size_and_neighbour_count_fail_before_preflight(self):
        for option, value in (('--ce-accumulation', '0'), ('--shard-pairs', '0'), ('--neural-k', '65535')):
            with self.subTest(option=option), patch.object(sys, 'argv', ['run_r10.py', option, value, '--plan']):
                with self.assertRaises(SystemExit):
                    run_r10.main()


if __name__ == '__main__':
    unittest.main()
