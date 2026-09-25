"""r5 regression tests: per-country IDF features, genericness counts, hop rule, LB estimate.

Synthetic data only; runs in seconds with no GPU.
"""
import unittest

import polars as pl

from er_v2.features import compute, pair_frame, record_frames, token_idf
from er_v2.graph import hop_keep
from er_v2.metrics import TEST_MIX, by_country, leaderboard_estimate


def records(rows, prefix):
    return pl.DataFrame(
        {"idx": list(range(len(rows))),
         "entity_id": [f"{prefix}-{i}" for i in range(len(rows))],
         "business_name": [r[0] for r in rows], "business_address": [r[1] for r in rows],
         "country": [r[2] for r in rows], "name_n": [r[0] for r in rows],
         "core_n": [r[0] for r in rows], "addr_n": [r[1] for r in rows]},
        schema_overrides={"idx": pl.UInt32})


class R5Tests(unittest.TestCase):
    def setUp(self):
        self.s1 = records([("tir club", "42 rue jean bouin nantes", "France"),
                           ("tir club", "9 rue ducourouble lille", "France")], "S1")
        self.tg = records([("tir club", "42 rue jean bouin nantes", "France"),
                           ("club nautique", "42 rue jean bouin nantes", "France"),
                           ("club", "1 rue x nantes", "France"),
                           ("club", "2 rue y nantes", "France")], "S2")

    def features(self, sidx, tidx):
        left, right = record_frames(self.s1, self.tg)
        idf = token_idf(self.tg)
        cands = pl.DataFrame({"sidx": sidx, "tidx": tidx}, schema={"sidx": pl.UInt32, "tidx": pl.UInt32})
        cands = cands.with_columns(*[pl.lit(1.0, pl.Float32).alias(c) for c in (
            "bscore", "nkeys", "brank", "b_rel_s", "b_rel_t", "t_rank", "t_nc", "s_nc")],
            rescue=pl.lit(0, pl.Int8))
        return compute(pair_frame(cands, left, right, 10), workers=1, idf=idf)

    def test_rare_tokens_outweigh_generic_ones(self):
        f = self.features([0, 0], [0, 1])
        exact, generic_only = f.row(0, named=True), f.row(1, named=True)
        self.assertAlmostEqual(exact["wn_jacc"], 1.0, places=5)
        # "club" is in every target name, so sharing only "club" is weak evidence
        self.assertLess(generic_only["wn_jacc"], 0.5)
        self.assertGreater(generic_only["wn_unmatched_r"], 0.0)
        self.assertAlmostEqual(exact["wa_jacc"], 1.0, places=5)

    def test_genericness_counts_are_per_split_and_country(self):
        f = self.features([0], [0]).row(0, named=True)
        self.assertGreater(f["name_cnt_l"], 0.0)   # two Source 1 rows share "tir club"
        self.assertGreater(f["name_frequency_l"], 0.0)  # and one target carries it

    def test_hop_rule_keeps_direct_supported_and_blank_address_name_matches(self):
        df = pl.DataFrame({
            "direct": [1, 0, 0, 0, 0],
            "sup_both_max": [0.0, 55.0, 10.0, 0.0, 0.0],
            "sup_addr_valid": [1, 1, 1, 0, 0],
            "sup_name_tset": [0.0, 0.0, 95.0, 95.0, 60.0],
        })
        self.assertEqual(df.filter(hop_keep()).height, 3)

    def test_leaderboard_estimate_weights_countries_by_test_mix(self):
        per = {"US": {"macro_f05": 1.0}, "India": {"macro_f05": 0.0}}
        est = leaderboard_estimate(per, france=0.0)["estimate"]
        self.assertAlmostEqual(est, TEST_MIX["US"], places=6)
        self.assertAlmostEqual(sum(TEST_MIX.values()), 1.0, places=6)

    def test_by_country_splits_anchors(self):
        pred = pl.DataFrame({"sidx": [0], "tidx": [5]})
        anchors = pl.DataFrame({"sidx": [0, 1], "country": ["US", "India"]})
        out = by_country(pred, pred, anchors)
        self.assertEqual(out["US"]["macro_f05"], 1.0)
        self.assertEqual(out["India"]["macro_f05"], 1.0)  # singleton predicted empty


if __name__ == "__main__":
    unittest.main()
