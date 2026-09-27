"""Small offline checks: no dataset, model download or neural fitting."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from er_v2 import normalize, r12_france
from er_v2.folds import fold_expr
from er_v2.france_features import canonical, records, similarities, transfer_weights
from er_v2.predict import write_lists

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_r12_france


class FranceTests(unittest.TestCase):
    def test_indic_map_cannot_rewrite_latin_accented_names(self):
        with patch.dict(normalize.TRANSLIT, {'mam': 'maa', 'je': 'jay', 'bhart': 'india'}, clear=True):
            self.assertEqual(normalize.norm_name('MAM École')[1], 'mam ecole')
            self.assertEqual(normalize.norm_name('JE Énergie')[1], 'je energie')
            self.assertEqual(normalize.norm_name('भारत')[1], 'india')

    def test_canonical_keeps_locality_numbers_and_avoids_state_and_indic_maps(self):
        self.assertEqual(canonical("SAS L’École d’Évreux"), 'ecole evreux')
        self.assertEqual(canonical("l ecole d evreux"), 'ecole evreux')
        self.assertEqual(canonical('Maine 12 Association des Jardins'), 'maine 12 association jardins')
        self.assertEqual(canonical('12 bis Rue du Maine 01230', True), '12 bis street maine 01230')
        self.assertNotEqual(canonical('Comite de Paris'), canonical('Comite de Lyon'))

    def test_similarity_preserves_global_ids_order_and_missing_evidence(self):
        left = records(pl.DataFrame({'idx': [0, 1], 'business_name': ["L’École d’Évreux", ''],
                                      'business_address': ['12 bis rue du Maine 01230', '']}))
        right = records(pl.DataFrame({'idx': [30, 40], 'business_name': ['Ecole Evreux', ''],
                                       'business_address': ['12 bis r du Maine 01230', '']}))
        pairs = pl.DataFrame({'sidx': [1, 0], 'tidx': [40, 30]})
        actual = similarities(pairs, left, right, 1)
        self.assertTrue(actual.select('sidx', 'tidx').equals(pairs))
        self.assertEqual(actual['fr_name_ratio'].to_list(), [0., 100.])
        self.assertEqual(actual['fr_addr_exact'].to_list(), [0., 1.])
        self.assertTrue(np.isfinite(actual.drop('sidx', 'tidx').to_numpy()).all())
        with self.assertRaisesRegex(ValueError, 'Missing record mapping'):
            similarities(pairs.with_columns(tidx=pl.lit(500)), left, right, 1)

    def test_domain_odds_are_clipped_and_business_balance_is_preserved(self):
        weights, diag = transfer_weights([0, .5, 1], [1, 1, 1])
        np.testing.assert_allclose(weights, np.array([1/3, 1, 3]) / (13/9), rtol=1e-6)
        self.assertEqual(diag['ratio_min'], 1/3)
        self.assertEqual(diag['ratio_max'], 3)
        self.assertLessEqual(diag['effective_sample_size'], 3)
        weights, _ = transfer_weights([.5, .5], [1, 2])
        self.assertAlmostEqual(float(weights[1] / weights[0]), 2.)

    def test_patch_cannot_change_india_us_or_claim_their_targets(self):
        base = pl.DataFrame({'sidx': [0, 1, 2], 'tidx': [10, 11, 12]})
        country = pl.DataFrame({'sidx': [0, 1, 2], 'country': ['US', 'India', 'France']})
        new = pl.DataFrame({'sidx': [2, 2], 'tidx': [12, 13]})
        result = r12_france.replace_france(base, new, country)
        self.assertTrue(result.filter(pl.col('sidx') != 2).equals(base.head(2)))
        with self.assertRaisesRegex(ValueError, 'non-France business'):
            r12_france.replace_france(base, base, country)
        with self.assertRaisesRegex(ValueError, 'non-France target'):
            r12_france.replace_france(base, new.with_columns(tidx=pl.lit(10)), country)

    def test_plan_does_not_rebuild_neural_models_or_read_holdout_before_selection(self):
        args = run_r12_france.parser().parse_args([])
        stages = run_r12_france.commands(args)
        self.assertEqual(list(stages), ['features_train', 'features_test', 'fit', 'select',
                                       'features_holdout', 'evaluate', 'inference', 'validate'])
        self.assertEqual(args.max_hours, 3.5)
        self.assertEqual(args.rounds, 900)
        for stage, (command, _) in stages.items():
            self.assertNotIn('er_v2.neural', command)
            self.assertNotIn('er_v2.r10_ce', command)
        self.assertIn('--check-ids', stages['validate'][0])
        with patch.object(sys, 'argv', ['run_r12_france.py', '--max-hours', '5', '--plan']):
            with self.assertRaises(SystemExit):
                run_r12_france.main()

    def test_fit_never_uses_gate_or_holdout_labels_including_domain_fit(self):
        ids = pl.DataFrame({'sidx': np.arange(1000, dtype=np.uint32)}).with_columns(fold_expr())
        rows = ids.with_columns(tidx=pl.col('sidx'), label=(pl.col('sidx') % 2).cast(pl.Int8))
        rows = rows.with_columns([pl.lit(0.5).alias(c) for c in r12_france.DOMAIN_FEATURES])
        calls = []
        class Model:
            best_iteration = 0
            def save_model(self, path):
                path.write_text('{}')
        def fake_fit(train, features, valid, rounds, params):
            calls.append((train.clone(), features, valid.clone() if valid is not None else None))
            return Model()
        with TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / 'models').mkdir()
            rows.write_parquet(w / 'features_train.parquet')
            rows.select(r12_france.DOMAIN_FEATURES).write_parquet(w / 'features_test.parquet')
            args = SimpleNamespace(work=w, device='cpu', threads=1, rounds=2)
            with patch.object(r12_france, 'fit', side_effect=fake_fit), \
                 patch.object(r12_france, 'predict_frame', side_effect=lambda m, f, c, b: np.full(len(f), .5)):
                r12_france.fit_models(args)
        self.assertEqual(len(calls), 3)
        self.assertEqual(set(calls[0][0].columns), set(r12_france.DOMAIN_FEATURES) | {'label'})
        self.assertEqual(set(calls[0][0]['label']), {0, 1})
        for train, features, valid in calls[1:]:
            self.assertEqual(set(train['fold']), {6, 7})
            self.assertEqual(set(valid['fold']), {3})
            self.assertTrue(valid.select(r12_france.r10.half()).to_series().eq(0).all())
            self.assertFalse(set(features) & {'sidx', 'tidx', 'fold', 'label', 'w'})

    def test_cache_change_is_rejected_before_reuse(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'scores.parquet'
            p.write_bytes(b'completed')
            state = {'completed': {'inference': {'outputs': run_r12_france.fingerprint([p])}}}
            with patch.object(run_r12_france, 'parent_paths', return_value=[p]):
                run_r12_france.verify_parent_files(None, state)
                p.write_bytes(b'different-file')
                with self.assertRaisesRegex(ValueError, 'no longer matches'):
                    run_r12_france.verify_parent_files(None, state)

    def test_feature_preparation_uses_france_only_and_full_source3_offset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, work = root / 'base', root / 'work'
            (base / 'norm').mkdir(parents=True)
            work.mkdir()
            for side in (1, 2, 3):
                pl.DataFrame({'idx': np.arange(2, dtype=np.uint32), 'country': ['US', 'France'],
                    'entity_id': [f'{side}u', f'{side}f'], 'business_name': ['Shop', "L’École d’Évreux"],
                    'business_address': ['1 Main St', '12 bis rue du Maine 01230']}).write_parquet(
                        base / f'norm/test_source{side}.parquet')
            frame = pl.DataFrame({'sidx': np.array([0, 1, 1], dtype=np.uint32),
                'tidx': np.array([0, 1, 3], dtype=np.uint32), 'direct': [1, 1, 1],
                'sup_both_max': [0., 0., 0.], 'sup_addr_valid': [0, 0, 0], 'sup_name_tset': [0., 0., 0.]})
            frame.write_parquet(base / 'stage3_test.parquet')
            frame.select('sidx', 'tidx').with_columns(ce_logit=pl.Series([1., 2., 3.])).write_parquet(base / 'ce_test.parquet')
            args = SimpleNamespace(base_work=base, work=work, dataset=root, device='cpu', threads=1, split='test')
            r12_france.prepare(args)
            actual = pl.read_parquet(work / 'features_test.parquet')
            self.assertEqual(actual['sidx'].to_list(), [1, 1])
            self.assertEqual(actual['tidx'].to_list(), [1, 3])
            self.assertEqual(actual['fr_name_exact'].to_list(), [1., 1.])
            self.assertEqual(actual['ce_logit'].to_list(), [2., 3.])

    def test_selection_only_reads_fold3_and_rejects_a_tie(self):
        c = pl.DataFrame({'sidx': np.arange(100, dtype=np.uint32)}).filter(fold_expr() == 3)
        c = c.with_columns(country=pl.lit('India'))
        scores = c.select('sidx', tidx=pl.col('sidx')).with_columns(score=pl.lit(.9))
        target = scores.select('sidx', 'tidx')
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            scores.with_columns(fold=pl.lit(3)).write_parquet(work / 'features_train.parquet')
            args = SimpleNamespace(work=work, base_work=work, dataset=work, device='cpu', threads=1)
            with patch.object(r12_france, 'countries', return_value=c), \
                 patch.object(r12_france, 'baseline', return_value=scores) as base_call, \
                 patch.object(r12_france, 'base_decision', return_value=scores), \
                 patch.object(r12_france, 'proposed_scores', return_value=scores), \
                 patch.object(r12_france.r10, 'truth', return_value=target) as truth_call, \
                 patch('builtins.print'):
                r12_france.select(args)
                self.assertEqual(truth_call.call_args.args[1], [3])
                self.assertEqual(base_call.call_args.args[1:], ('train', 3))
            selected = json.loads((work / 'selection.json').read_text())
            self.assertEqual(selected['selected'], 'r12')
            self.assertFalse(selected['gate']['passed'])

    def test_inference_fallback_is_byte_identical_and_patch_changes_only_france(self):
        from er_v2.decision import decide_country
        c = pl.DataFrame({'sidx': [0, 1, 2, 3], 'country': ['US', 'India', 'France', 'France']},
                        schema={'sidx': pl.UInt32, 'country': pl.String})
        scores = pl.DataFrame({'sidx': [0, 1, 2, 2, 3], 'tidx': [0, 1, 2, 3, 4],
                              'score': [.9, .9, .8, .3, .1]},
                             schema={'sidx': pl.UInt32, 'tidx': pl.UInt32, 'score': pl.Float32})
        s1 = c.select(idx='sidx').with_columns(entity_id=pl.Series(['u', 'i', 'f', 'empty']))
        tg = pl.DataFrame({'idx': np.arange(5, dtype=np.uint32), 'entity_id': ['a', 'b', 'c', 'd', 'e']})
        def decide(args, pred, country):
            return decide_country(pred, .7, country, {}, 'score')
        with TemporaryDirectory() as tmp:
            w, out = Path(tmp), Path(tmp) / 'output'
            args = SimpleNamespace(work=w, output=out, base_work=w / 'base', device='cpu', threads=1)
            selection = {'selected': 'r12', 'proposal': {'model': 'balanced', 'weight': 1., 'threshold': .7}}
            (w / 'selection.json').write_text(json.dumps(selection))
            expected = w / 'expected.tsv'
            write_lists(expected, s1, decide(None, scores, c), tg['entity_id'], 'matched_entity_ids')
            with patch.object(r12_france, 'countries', return_value=c), \
                 patch.object(r12_france, 'baseline', return_value=scores), \
                 patch.object(r12_france, 'base_decision', side_effect=decide), \
                 patch.object(r12_france, 'load_split', return_value=(s1, tg)):
                r12_france.inference(args)
                self.assertEqual(expected.read_bytes(), (out / 'matching_results.tsv').read_bytes())
                selection['selected'] = 'france_transfer'
                (w / 'selection.json').write_text(json.dumps(selection))
                updated = scores.filter(pl.col('sidx') >= 2).with_columns(score=pl.Series([.9, .9, .1], dtype=pl.Float32))
                updated.write_parquet(w / 'features_test.parquet')
                with patch.object(r12_france, 'proposed_scores', return_value=updated):
                    r12_france.inference(args)
                result = (out / 'matching_results.tsv').read_text().splitlines()
                self.assertEqual(result[:3], expected.read_text().splitlines()[:3])
                self.assertEqual(result[3], 'f\tc,d')
                self.assertEqual(result[4], 'empty\t')
                self.assertTrue(json.loads((w / 'changes.json').read_text())['non_france_unchanged'])


if __name__ == '__main__':
    unittest.main()
