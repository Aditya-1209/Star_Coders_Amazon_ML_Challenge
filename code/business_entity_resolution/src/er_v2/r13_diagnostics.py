"""Report-only error counts; never used for fitting or selecting the model."""
import polars as pl


def error_report(work, scores, chosen, truth):
    keys = ['sidx', 'tidx']
    missing = pl.concat([pl.read_parquet(work / f'norm/train_source{i}.parquet', columns=['addr_n'])
        .select(missing_address=pl.col('addr_n').fill_null('').str.strip_chars() == '')
        for i in (2, 3)]).with_row_index('tidx')
    predictions = chosen.select(keys).unique()
    target = truth.select(keys).unique()
    rejected = target.join(predictions, on=keys, how='anti')
    retrieved = rejected.join(scores.select(keys), on=keys, how='semi')
    result = {'missed_pairs': len(rejected), 'retrieved_but_rejected': len(retrieved),
              'not_retrieved': len(rejected) - len(retrieved),
              'rejected_with_missing_address': int(retrieved.join(missing, on='tidx')['missing_address'].sum()),
              'by_target_address': {}}
    for blank, name in ((True, 'missing'), (False, 'present')):
        ids = missing.filter(pl.col('missing_address') == blank).select('tidx')
        p = predictions.join(ids, on='tidx', how='semi')
        t = target.join(ids, on='tidx', how='semi')
        tp = len(p.join(t, on=keys))
        result['by_target_address'][name] = {'predicted_pairs': len(p), 'true_pairs': len(t),
            'true_positive_pairs': tp, 'false_positive_pairs': len(p) - tp,
            'missed_pairs': len(t) - tp, 'pair_precision': tp / len(p) if len(p) else None,
            'pair_recall': tp / len(t) if len(t) else None}
    return result
