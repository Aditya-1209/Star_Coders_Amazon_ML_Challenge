"""R10 field-aware multilingual cross-encoder with bounded token caches.

Train: folds 0/1/8. Early stopping: fold 9. No fold 2/5/6/7/3/4 labels
enter the neural optimizer. Final stacker trains on 6/7; 4 is report-only.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from .neural import BASE_MODEL, BASE_REVISION
from .r10_retrieval import iter_records, side_rows, save_json
from .run_block import load_split
from .run_features import ground_truth_pairs
from .folds import fold_expr

TRAIN_FOLDS = [0, 1, 8]
VALID_FOLDS = [9]


def mined_negatives(work, anchors, truth, lexical_pool, ann_pool, limit=5):
    """Mine stage-2 false positives, with lexical/ANN/random diversity.

    Stage 2 learns on 2/5; these scores are out of sample for CE folds 0/1/8/9.
    Read ONLY pair keys and predictions: a stored stage-2 label is not CE truth.
    Remove every labelled positive before ranking, including positives outside
    the pruned candidate set. Keep the R12 ceiling of five negatives/business.
    """
    keys = ['sidx', 'tidx']
    model = (pl.scan_parquet(work / 'stage2_train.parquet').select(*keys, 'p2')
        .join(anchors.lazy(), on='sidx', how='semi').collect(engine='streaming'))
    if model.select(keys).n_unique() != len(model) or model['p2'].null_count() or not model['p2'].is_finite().all():
        raise ValueError('Invalid stage-2 hard-negative scores')
    model = model.join(truth, on=keys, how='anti').sort(['sidx', 'p2', 'tidx'], descending=[False, True, False])
    lexical = lexical_pool.sort('sidx', 'brank', 'tidx')
    ann = ann_pool.join(truth, on=keys, how='anti').sort('sidx', 'nrank', 'tidx')
    random = lexical_pool.with_columns(key=pl.struct(keys).hash(seed=1313)).sort('sidx', 'key', 'tidx')
    chosen = anchors.head(0).with_columns(tidx=pl.lit(None, pl.UInt32))
    parts = []
    # Select each source AFTER removing already chosen pairs, so overlap does
    # not consume the diversity slots. A final lexical refill covers sparse ANN.
    for priority, (pool, count) in enumerate(((model, 2), (lexical, 1), (ann, 1), (random, 1), (lexical, limit))):
        take = pool.select(keys).join(chosen, on=keys, how='anti').group_by('sidx', maintain_order=True).head(count)
        parts.append(take.with_columns(priority=pl.lit(priority)))
        chosen = pl.concat([chosen, take])
    return (pl.concat(parts).sort('sidx', 'priority', maintain_order=True).group_by('sidx', maintain_order=True)
            .head(limit).select(keys))


def samples(work, dataset, folds, limit, negatives=4, mine_stage2=False):
    """Sample businesses first; retain truth plus lexical and ANN hard negatives."""
    s1, tg = load_split(work, 'train', columns=['idx', 'entity_id', 'country'])
    anchors = s1.select(sidx=pl.col('idx').cast(pl.UInt32)).filter(fold_expr().is_in(folds))
    if mine_stage2 and not set(folds) <= set(TRAIN_FOLDS + VALID_FOLDS):
        raise ValueError('Stage-2 mining is restricted to out-of-sample CE folds 0/1/8/9')
    anchors = anchors.with_columns(key=pl.col('sidx').hash(seed=10)).sort('key').head(limit).drop('key')
    truth = ground_truth_pairs(dataset, s1, tg).join(anchors, on='sidx', how='semi')
    del s1, tg
    candidates = (pl.scan_parquet(work / 'cands_train.parquet').join(anchors.lazy(), on='sidx', how='semi')
                  .select('sidx', 'tidx', 'brank').collect(engine='streaming'))
    neg = candidates.join(truth, on=['sidx', 'tidx'], how='anti')
    ann = (pl.scan_parquet(work / 'ncands_train.parquet').join(anchors.lazy(), on='sidx', how='semi')
           .select('sidx', 'tidx', 'nrank').collect(engine='streaming'))
    if mine_stage2:
        neg = mined_negatives(work, anchors, truth, neg, ann)
    else:
        lexical = neg.sort('sidx', 'brank', 'tidx').group_by('sidx', maintain_order=True).head(max(1, negatives // 2))
        ann = ann.join(truth, on=['sidx', 'tidx'], how='anti').sort('sidx', 'nrank', 'tidx').group_by('sidx', maintain_order=True).head(max(1, negatives // 2))
        random = neg.with_columns(key=pl.struct('sidx', 'tidx').hash(seed=88)).sort('sidx', 'key').group_by('sidx', maintain_order=True).head(1)
        neg = pl.concat([x.select('sidx', 'tidx') for x in (lexical, ann, random)]).unique()
    neg = neg.with_columns(label=pl.lit(0, pl.Int8))
    # Business sampling already bounds size. Keeping the first four targets
    # discards valid variants and systematically favours Source 2 (lower IDs).
    # Retain every labelled positive; per-business weights still balance loss.
    pos = truth.with_columns(label=pl.lit(1, pl.Int8))
    result = pl.concat([pos, neg]).sort('sidx', 'tidx')
    if result.is_empty() or result['label'].n_unique() < 2:
        raise ValueError('Cross-encoder needs both labels in its train/validation business sample')
    return result.with_columns(w=(len(result) / result['sidx'].n_unique() / pl.len().over('sidx')).cast(pl.Float32))


def cache_tokens(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=BASE_REVISION if args.base_model == BASE_MODEL else None)
    # Equal field budgets per record preserve both names in long-address pairs.
    width = (args.max_length - tokenizer.num_special_tokens_to_add(pair=True)) // 2
    if width < 8:
        raise ValueError('max-length leaves insufficient record tokens')
    root = args.work / 'ce_tokens'
    root.mkdir(exist_ok=True)
    for side in ('s1', 'tg'):
        n = side_rows(args.work, args.split, side)
        path = root / f'{args.split}_{side}.npy'
        lengths_path = root / f'{args.split}_{side}_lengths.npy'
        temp, lentemp = path.with_suffix('.partial.npy'), lengths_path.with_suffix('.partial.npy')
        tokens = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint32, shape=(n, width))
        lengths = np.lib.format.open_memmap(lentemp, mode='w+', dtype=np.uint16, shape=(n,))
        for frame in iter_records(args.work, args.split, side, rows=4096):
            names = tokenizer(frame['business_name'].fill_null('').to_list(), add_special_tokens=False,
                              truncation=True, max_length=max(4, width // 2))['input_ids']
            addresses = tokenizer(frame['business_address'].fill_null('').to_list(), add_special_tokens=False,
                                  truncation=True, max_length=width)['input_ids']
            # A SEP inside each record distinguishes name and address; no new/untrained special tokens.
            for idx, name, address in zip(frame['idx'], names, addresses):
                value = (name + [tokenizer.sep_token_id] + address)[:width]
                tokens[idx, :len(value)] = value
                lengths[idx] = len(value)
        tokens.flush()
        lengths.flush()
        del tokens, lengths
        temp.replace(path)
        lentemp.replace(lengths_path)
    tokenizer.save_pretrained(root / 'tokenizer')
    save_json(root / f'{args.split}.json', {'base_model': args.base_model, 'max_length': args.max_length, 'width': width})


class PairTokens:
    def __init__(self, work, split):
        from transformers import AutoTokenizer
        root = work / 'ce_tokens'
        self.tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer')
        self.arrays = {s: np.load(root / f'{split}_{s}.npy', mmap_mode='r') for s in ('s1', 'tg')}
        self.lengths = {s: np.load(root / f'{split}_{s}_lengths.npy', mmap_mode='r') for s in ('s1', 'tg')}

    def batch(self, frame, device, swap=False, drop_address=False):
        tok = self.tokenizer
        examples = []
        drop = np.broadcast_to(np.asarray(drop_address, dtype=bool), (len(frame),))
        for i, (s, t) in enumerate(frame.select('sidx', 'tidx').iter_rows()):
            a = self.arrays['s1'][s, :self.lengths['s1'][s]].tolist()
            b = self.arrays['tg'][t, :self.lengths['tg'][t]].tolist()
            if drop[i]:
                # Real data has complete Source 1 addresses and missing target
                # addresses. Preserve the field SEP exactly as cache_tokens
                # does for a naturally blank target, including after swapping.
                b = b[:b.index(tok.sep_token_id) + 1] if tok.sep_token_id in b else b
            if swap:
                a, b = b, a
            # Transformers 5 removed build_inputs_with_special_tokens. E5-small
            # uses the BERT pair template; also support the XLM-R four-token template.
            special = tok.num_special_tokens_to_add(pair=True)
            if special not in (3, 4) or tok.cls_token_id is None or tok.sep_token_id is None:
                raise ValueError('Expected BERT or XLM-R pair-token template')
            middle = [tok.sep_token_id] * (special - 2)
            ids = [tok.cls_token_id] + a + middle + b + [tok.sep_token_id]
            item = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
            if 'token_type_ids' in tok.model_input_names:
                item['token_type_ids'] = ([0] * (len(a) + 2) + [1] * (len(b) + 1)
                                          if special == 3 else [0] * len(ids))
            examples.append(item)
        return tok.pad(examples, padding=True, pad_to_multiple_of=8, return_tensors='pt').to(device)

    def pair_lengths(self, frame):
        return (self.lengths['s1'][frame['sidx'].to_numpy()].astype(np.int32)
                + self.lengths['tg'][frame['tidx'].to_numpy()].astype(np.int32))


def amp_dtype(device, precision='fp16'):
    """L4 uses native BF16; preserve FP16 defaults for older experiment profiles."""
    import torch
    if precision not in ('auto', 'bf16', 'fp16'):
        raise ValueError('Unknown CE precision')
    if device != 'cuda':
        return torch.float32
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'fp16'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('BF16 requested but not supported by this GPU; use --precision fp16')
    return torch.bfloat16 if precision == 'bf16' else torch.float16


def score_logits(model, cache, frame, device, batch, precision='fp16'):
    """Length-bucket inference, restored to input order; retry CUDA OOM losslessly.

    Cache a reduced batch ceiling across parquet windows and validation epochs.
    Never truncate text, skip a pair, or move computation silently onto the CPU.
    """
    import torch
    dtype = amp_dtype(device, precision)
    order = np.argsort(cache.pair_lengths(frame), kind='stable')
    values = np.empty(len(frame), dtype=np.float32)
    size = min(batch, getattr(cache, 'score_batch_limit', batch))
    start = 0
    with torch.inference_mode():
        while start < len(order):
            ids = order[start:start + size]
            try:
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == 'cuda'):
                    logits = model(**cache.batch(frame[ids], device)).logits.reshape(-1)
                values[ids] = logits.float().cpu().numpy()
                del logits
            except torch.cuda.OutOfMemoryError:
                if device != 'cuda' or len(ids) == 1:
                    raise
                size = max(1, len(ids) // 2)
                cache.score_batch_limit = size
                print(f'CE inference CUDA OOM: retrying the same pairs at batch {size}', flush=True)
            else:
                start += len(ids)
                continue
            # Outside the exception handler: release its traceback before retry.
            gc.collect()
            torch.cuda.empty_cache()
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite cross-encoder scores')
    return values


def weighted_bce(logits, labels, weights):
    import torch.nn.functional as F
    # Sampling changes the class prior: this signal is calibrated by the held-out stacker.
    return (F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none') * weights).mean()


def fit_model(args, train, valid):
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a CUDA PyTorch wheel')
    precision = getattr(args, 'precision', 'fp16')
    dtype = amp_dtype(args.device, precision)
    rng = np.random.default_rng(args.seed)
    cache = PairTokens(args.work, 'train')
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1,
        revision=BASE_REVISION if args.base_model == BASE_MODEL else None).to(args.device)
    model.config.problem_type = 'regression'  # custom binary-logit loss below, not model MSE
    if args.device == 'cuda' and args.checkpointing:
        model.gradient_checkpointing_enable()
    fused = args.device == 'cuda' and getattr(args, 'fused_optimizer', False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                 **({'fused': True} if fused else {}))
    steps = math.ceil(math.ceil(len(train) / args.batch) / args.accumulation) * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * .06)), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda' and dtype == torch.float16)
    best, stale, history = float('inf'), 0, []
    target = args.work / 'ce_model'
    target.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        optimizer.zero_grad(set_to_none=True)
        loss_sum, rows = 0., 0
        batches = math.ceil(len(train) / args.batch)
        for j in range(batches):
            part = train[order[j * args.batch:(j + 1) * args.batch]]
            # Independent target-address dropout matches the missing-field case;
            # swapping still prevents the model relying on pair orientation.
            inputs = cache.batch(part, args.device, swap=bool(rng.integers(2)),
                                 drop_address=rng.random(len(part)) < .10)
            y = torch.tensor(part['label'].to_numpy(), device=args.device, dtype=torch.float32)
            w = torch.tensor(part['w'].to_numpy(), device=args.device)
            with torch.autocast(device_type=args.device, dtype=dtype, enabled=args.device == 'cuda'):
                logits = model(**inputs).logits.reshape(-1)
                raw_loss = weighted_bce(logits, y, w)
                # Correct denominator for the last, shorter accumulation group.
                group_size = min(args.accumulation, batches - (j // args.accumulation) * args.accumulation)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            if (j + 1) % args.accumulation == 0 or j + 1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before:
                    schedule.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(raw_loss.detach()) * len(part)
            rows += len(part)
            if j % 200 == 0:
                print(f'CE epoch {epoch + 1}: {rows:,}/{len(train):,}, loss {loss_sum / rows:.5f}', flush=True)
        model.eval()
        logits = torch.from_numpy(score_logits(model, cache, valid, args.device, args.score_batch, precision))
        value = float(weighted_bce(logits, torch.tensor(valid['label'].to_numpy()),
                                   torch.tensor(valid['w'].to_numpy())))
        if not np.isfinite(value):
            raise RuntimeError('Non-finite CE validation loss')
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / rows, 'fold9_loss': value})
        print(history[-1], flush=True)
        if value < best:
            best, stale = value, 0
            model.save_pretrained(target, safe_serialization=True)
            cache.tokenizer.save_pretrained(target)
        else:
            stale += 1
            if stale >= args.patience:
                break
    save_json(target / 'training.json', {'base_model': args.base_model, 'revision': BASE_REVISION if args.base_model == BASE_MODEL else None, 'train_folds': TRAIN_FOLDS,
        'early_stopping_folds': VALID_FOLDS, 'train_pairs': len(train), 'valid_pairs': len(valid),
        'max_length': args.max_length, 'batch': args.batch, 'accumulation': args.accumulation,
        'amp_dtype': str(dtype), 'fused_optimizer': fused,
        'positive_sampling': 'all positives for selected businesses',
        'negative_sampling': ('stage2 top-2 + lexical + ANN + random, refill, at most 5/business'
                              if getattr(args, 'mine_stage2', False) else 'lexical + ANN + random'),
        'stage2_mining': getattr(args, 'mine_stage2', False),
        'address_dropout': {'probability': .10, 'side': 'target', 'per_pair': True, 'preserve_field_separator': True},
        'seed': args.seed, 'history': history, 'calibrated_probability': False})


def train(args):
    mining = getattr(args, 'mine_stage2', False)
    tr = samples(args.work, args.dataset, TRAIN_FOLDS, args.train_businesses, mine_stage2=mining)
    va = samples(args.work, args.dataset, VALID_FOLDS, args.valid_businesses, mine_stage2=mining)
    tr.write_parquet(args.work / 'ce_train_sample.parquet')
    va.write_parquet(args.work / 'ce_valid_sample.parquet')
    fit_model(args, tr, va)


def score(args):
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(args.threads)
    cache = PairTokens(args.work, args.split)
    model = AutoModelForSequenceClassification.from_pretrained(args.work / 'ce_model').to(args.device).eval()
    path = args.work / f'ce_{args.split}.parquet'
    temp = path.with_suffix('.partial.parquet')
    writer = None
    from .graph import hop_keep
    source = args.work / f'stage3_{args.split}.parquet'
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=10000):
            frame = pl.from_arrow(batch).filter(hop_keep()).select('sidx', 'tidx')
            if frame.is_empty():
                continue
            values = score_logits(model, cache, frame, args.device, args.score_batch, getattr(args, 'precision', 'fp16'))
            result = frame.with_columns(ce_logit=pl.Series(values, dtype=pl.Float32))
            if not np.isfinite(result['ce_logit'].to_numpy()).all():
                raise RuntimeError('Non-finite cross-encoder scores')
            if writer is None:
                writer = pq.ParquetWriter(temp, result.to_arrow().schema, compression='zstd')
            writer.write_table(result.to_arrow())
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pl.DataFrame(schema={'sidx': pl.UInt32, 'tidx': pl.UInt32, 'ce_logit': pl.Float32}).write_parquet(temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['tokens', 'train', 'score'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--base-model', default=BASE_MODEL)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--score-batch', type=int, default=128)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--no-checkpointing', dest='checkpointing', action='store_false',
                   help='disable activation checkpointing (faster; needs more VRAM)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--patience', type=int, default=1)
    p.add_argument('--train-businesses', type=int, default=180000)
    p.add_argument('--valid-businesses', type=int, default=15000)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=1010)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--precision', choices=['fp16', 'bf16', 'auto'], default='fp16')
    p.add_argument('--fused-optimizer', action='store_true', help='use fused CUDA AdamW; CPU remains unfused')
    p.add_argument('--mine-stage2', action='store_true', help='R13: mine out-of-sample stage-2 false positives')
    args = p.parse_args()
    if min(args.max_length, args.batch, args.score_batch, args.accumulation, args.epochs,
           args.patience, args.train_businesses, args.valid_businesses, args.threads) < 1 or args.lr <= 0:
        p.error('Counts and learning rate must be positive')
    {'tokens': cache_tokens, 'train': train, 'score': score}[args.stage](args)

if __name__ == '__main__':
    main()
