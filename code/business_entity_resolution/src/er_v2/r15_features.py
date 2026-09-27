"""Label-free within-business CE margins and cross-source corroboration for R15.

Compute context on a whole country before filtering model folds. No corpus
counts, IDFs, row IDs, country IDs or labels become model features.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

KEYS = ['sidx', 'tidx']


def alternatives(frame, n_source2):
    """Compare with the strongest OTHER target in this business and source.

    Only two rows per business/source are needed. IDs break ties for joins,
    but are never returned as features. A tied winner has margin zero.
    """
    evidence = frame.select(*KEYS, 'ce_logit', 'core_tset', 'addr_tset').with_columns(
        _side=(pl.col('tidx') >= n_source2).cast(pl.Int8))
    groups = ['sidx', '_side']
    best = (evidence.sort([*groups, 'ce_logit', 'tidx'], descending=[False, False, True, False])
            .group_by(groups, maintain_order=True).head(2))
    top = best.group_by(groups, maintain_order=True).first()
    second = best.join(top.select(*KEYS), on=KEYS, how='anti')
    cols = ['tidx', 'ce_logit', 'core_tset', 'addr_tset']
    context = top.rename({c: c + '_first' for c in cols}).join(
        second.rename({c: c + '_second' for c in cols}), on=groups, how='left')
    out = evidence.join(context, on=groups, how='left', validate='m:1')
    is_first = pl.col('tidx') == pl.col('tidx_first')
    other = lambda c: pl.when(is_first).then(pl.col(c + '_second')).otherwise(pl.col(c + '_first'))
    return out.select(*KEYS,
        r15_ce_alternative_margin=(pl.col('ce_logit') - other('ce_logit')).cast(pl.Float32),
        r15_alternative_logit=other('ce_logit').cast(pl.Float32),
        r15_alternative_name_margin=(pl.col('core_tset') - other('core_tset')).cast(pl.Float32),
        r15_alternative_address_margin=(pl.col('addr_tset') - other('addr_tset')).cast(pl.Float32))


def sibling_features(frame, targets, n_source2, workers=1, batch_rows=100_000, progress=None):
    """Compare each target with the strongest CE candidate in the OTHER source.

    The support pair is selected solely by CE logit, never by ground truth or a
    downstream model fitted on folds 6/7. It is necessarily a different record.
    Null means no corroborating source; an absent address never means agreement.
    """
    rows = frame.select(*KEYS, 'ce_logit').with_columns(
        _side=(pl.col('tidx') >= n_source2).cast(pl.Int8))
    strongest = (rows.sort(['sidx', '_side', 'ce_logit', 'tidx'],
                           descending=[False, False, True, False])
                  .unique(['sidx', '_side'], keep='first', maintain_order=True)
                  .select('sidx', _side=1 - pl.col('_side'), _support='tidx',
                          r15_sibling_ce='ce_logit'))
    records = targets.select(tidx=pl.col('idx').cast(pl.UInt32),
        _name=pl.col('core_n').fill_null(''), _address=pl.col('addr_n').fill_null(''))
    records = records.with_columns(_numbers=pl.col('_address').str.extract_all(r'\d+').list.unique())
    context = strongest.join(records.rename({'tidx': '_support', '_name': '_support_name',
        '_address': '_support_address', '_numbers': '_support_numbers'}), on='_support', how='left')
    # Join each country once, rather than rebuilding a multi-million-record
    # hash table for every 100k-pair similarity batch. Countries are processed
    # separately on the 128 GB VM; Python string conversion stays bounded.
    joined = (rows.join(context, on=['sidx', '_side'], how='left', validate='m:1')
              .join(records, on='tidx', how='left', validate='m:1'))
    result = []
    for start in range(0, len(rows), batch_rows):
        batch = joined.slice(start, batch_rows)
        expressions = []
        for field in ('name', 'address'):
            valid = ((pl.col('_' + field).str.len_chars() > 0)
                     & (pl.col('_support_' + field).str.len_chars() > 0)).fill_null(False)
            for metric, scorer in (('ratio', fuzz.ratio), ('tset', fuzz.token_set_ratio)):
                name = f'r15_sibling_{field}_{metric}'
                values = process.cpdist(batch['_' + field].fill_null('').to_list(),
                    batch['_support_' + field].fill_null('').to_list(), scorer=scorer,
                    workers=workers, dtype=np.float32)
                batch = batch.with_columns(pl.Series(name, values))
                expressions.append(pl.when(valid).then(pl.col(name)).otherwise(None).alias(name))
        has_numbers = ((pl.col('_numbers').list.len() > 0)
                       & (pl.col('_support_numbers').list.len() > 0)).fill_null(False)
        result.append(batch.select(*KEYS,
            pl.col('r15_sibling_ce').cast(pl.Float32), *expressions,
            r15_sibling_number_overlap=pl.when(has_numbers).then(
                pl.col('_numbers').list.set_intersection('_support_numbers').list.len() > 0
            ).otherwise(None).cast(pl.Float32),
            r15_sibling_ce_margin=(pl.col('ce_logit') - pl.col('r15_sibling_ce')).cast(pl.Float32)))
        if progress:
            progress(min(start + batch_rows, len(rows)), len(rows))
    if not result:
        names = ['r15_sibling_ce', 'r15_sibling_name_ratio', 'r15_sibling_name_tset',
                 'r15_sibling_address_ratio', 'r15_sibling_address_tset',
                 'r15_sibling_number_overlap', 'r15_sibling_ce_margin']
        return frame.select(*KEYS).with_columns(*[pl.lit(None, pl.Float32).alias(c) for c in names])
    return pl.concat(result)


def enrich(frame, targets, n_source2, workers=1, progress=None):
    if frame.select(KEYS).n_unique() != len(frame):
        raise ValueError('Duplicate R15 candidate keys')
    rivals = alternatives(frame, n_source2)
    sibling = sibling_features(frame, targets, n_source2, workers, progress=progress)
    return (frame.join(rivals, on=KEYS, how='left', validate='1:1')
            .join(sibling, on=KEYS, how='left', validate='1:1'))
