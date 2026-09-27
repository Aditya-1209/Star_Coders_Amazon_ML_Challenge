"""Transfer features from raw text; no learned transliteration or US state map.

Supplement the cached R12 features rather than changing the inputs of frozen
models. Preserve numbers and locality words. Empty fields are never agreement.
"""
from __future__ import annotations

import re
import numpy as np
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz, process

# Whole tokens only. Do NOT drop association, commune, school, city names, etc.
FORMS = set('inc incorporated llc ltd limited pvt private corp corporation llp '
            'sarl sas sasu eurl sci snc'.split())
ARTICLES = set('the of and le la les de du des et l d'.split())
STREETS = {'rue': 'street', 'r': 'street', 'st': 'street', 'street': 'street',
           'avenue': 'avenue', 'av': 'avenue', 'ave': 'avenue',
           'boulevard': 'boulevard', 'bd': 'boulevard', 'blvd': 'boulevard',
           'route': 'route', 'rte': 'route', 'chemin': 'chemin', 'chem': 'chemin',
           'allee': 'allee', 'all': 'allee', 'impasse': 'impasse', 'imp': 'impasse',
           'saint': 'saint', 'sainte': 'sainte', 'ste': 'sainte'}


def canonical(value: str | None, address: bool = False) -> str:
    # Apostrophes become boundaries, so d'Évreux and d Evreux agree. Unlike
    # norm_addr, retain leading zeroes, house suffixes, bis/ter and unit numbers.
    words = re.findall(r'[a-z0-9]+', anyascii(value or '').lower())
    if address:
        words = [STREETS.get(w, w) for w in words if w not in ARTICLES]
    else:
        words = [w for w in words if w not in FORMS and w not in ARTICLES]
    return ' '.join(words)


def records(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select('idx',
        fn=pl.col('business_name').map_elements(canonical, return_dtype=pl.String, skip_nulls=False),
        fa=pl.col('business_address').map_elements(lambda x: canonical(x, True),
                                                 return_dtype=pl.String, skip_nulls=False)
    ).with_columns(ft=pl.col('fn').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique(),
                   tail=pl.col('fn').str.split(' ').list.tail(2).list.join(' '))


def similarities(pairs: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame,
                 workers: int) -> pl.DataFrame:
    """Bounded pair window, preserving original pair order and global IDs."""
    x = pairs.select('sidx', 'tidx').join(left.rename({c: c+'_l' for c in left.columns if c != 'idx'}),
        left_on='sidx', right_on='idx', how='left', maintain_order='left', validate='m:1')
    x = x.join(right.rename({c: c+'_r' for c in right.columns if c != 'idx'}),
        left_on='tidx', right_on='idx', how='left', maintain_order='left', validate='m:1')
    if x['fn_l'].null_count() or x['fn_r'].null_count():
        raise ValueError('Missing record mapping in France transfer features')
    out = x.select('sidx', 'tidx')
    for field, label, scorer in [('fn', 'name_ratio', fuzz.ratio), ('fn', 'name_sort', fuzz.token_sort_ratio),
                                  ('tail', 'name_tail', fuzz.ratio), ('fa', 'addr_ratio', fuzz.ratio),
                                  ('fa', 'addr_sort', fuzz.token_sort_ratio)]:
        values = process.cpdist(x[field+'_l'].to_list(), x[field+'_r'].to_list(),
                                scorer=scorer, workers=workers, dtype=np.float32)
        present = ((x[field+'_l'] != '') & (x[field+'_r'] != '')).to_numpy()
        values[~present] = 0
        out = out.with_columns(pl.Series('fr_'+label, values))
    common = pl.col('ft_l').list.set_intersection(pl.col('ft_r')).list.len()
    denom = pl.max_horizontal(pl.col('ft_l').list.len(), pl.col('ft_r').list.len(), pl.lit(1))
    out = out.hstack(x.select(
        fr_name_cont_min=(common / denom).cast(pl.Float32),
        fr_name_exact=((pl.col('fn_l') != '') & (pl.col('fn_l') == pl.col('fn_r'))).cast(pl.Float32),
        fr_addr_exact=((pl.col('fa_l') != '') & (pl.col('fa_l') == pl.col('fa_r'))).cast(pl.Float32)))
    return out.with_columns(fr_both=pl.min_horizontal('fr_name_sort', 'fr_addr_sort'))


def transfer_weights(probability, business_weights):
    """Equal-prior density odds, clipped BEFORE business balancing; never labels."""
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1-1e-5)
    ratio = np.clip(p / (1-p), 1/3, 3)
    w = ratio * np.asarray(business_weights, dtype=np.float64)
    w /= w.mean()
    return w.astype(np.float32), {'ratio_min': float(ratio.min()), 'ratio_max': float(ratio.max()),
        'effective_sample_size': float(w.sum() ** 2 / np.square(w).sum()), 'rows': len(w)}
