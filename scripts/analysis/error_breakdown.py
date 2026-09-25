"""Where the holdout loss comes from (fold 4): error buckets and missed-pair types.

usage: python scripts/analysis/error_breakdown.py <model_dir> [eval_preds.parquet]
"""
import sys
import polars as pl, json
from pathlib import Path
from er_v2.train import fold_expr, decide
from er_v2.run_block import load_split
from er_v2.run_features import ground_truth_pairs
thr=json.load(open(sys.argv[1]+'/metrics.json'))['threshold']
ev=pl.read_parquet(sys.argv[2] if len(sys.argv)>2 else 'work/eval_preds.parquet').filter(pl.col('fold')==4)
s1,tg=load_split(Path('work'),'train')
truth=ground_truth_pairs(Path('student_resource/dataset'),s1,tg)
a=s1.select(sidx=pl.col('idx').cast(pl.UInt32),country='country').with_columns(fold_expr()).filter(pl.col('fold')==4)
truth=truth.join(a.select('sidx'),on='sidx')
pred=decide(ev,thr).select('sidx','tidx')
cand=ev.select('sidx','tidx')
nt=truth.group_by('sidx').agg(nt=pl.len()); tp=pred.join(truth,on=['sidx','tidx']).group_by('sidx').agg(tp=pl.len()); npred=pred.group_by('sidx').agg(np_=pl.len())
d=a.join(nt,on='sidx',how='left').join(tp,on='sidx',how='left').join(npred,on='sidx',how='left').fill_null(0)
d=d.with_columns(f=pl.when((pl.col('nt')==0)&(pl.col('np_')==0)).then(1.0).when(pl.col('tp')==0).then(0.0)
   .otherwise(1.25*(pl.col('tp')/pl.col('np_'))*(pl.col('tp')/pl.col('nt'))/(0.25*pl.col('tp')/pl.col('np_')+pl.col('tp')/pl.col('nt'))))
N=len(d); loss=(1-d['f']).sum()
print(f'fold4 S1={N:,} macroF={d["f"].mean():.4f}  total loss={loss:,.0f} S1-equivalents')
def bucket(name,mask):
    x=d.filter(mask); print(f'  {name:48s} S1={len(x):7,}  loss={(1-x["f"]).sum():8,.0f} ({100*(1-x["f"]).sum()/loss:5.1f}% of loss)')
bucket('singletons predicted non-empty (F=0)',(pl.col('nt')==0)&(pl.col('np_')>0))
bucket('has matches, predicted empty (F=0)',(pl.col('nt')>0)&(pl.col('np_')==0))
bucket('has matches, predicted only wrong ones (F=0)',(pl.col('nt')>0)&(pl.col('np_')>0)&(pl.col('tp')==0))
bucket('partial: some FP',(pl.col('tp')>0)&(pl.col('np_')>pl.col('tp')))
bucket('partial: only FN (all preds right, missed some)',(pl.col('tp')>0)&(pl.col('np_')==pl.col('tp'))&(pl.col('tp')<pl.col('nt')))
fn=truth.join(pred,on=['sidx','tidx'],how='anti')
fn_c=fn.join(cand,on=['sidx','tidx']); print(f'missed true pairs: {len(fn):,}  (in candidates but rejected: {len(fn_c):,}; never retrieved: {len(fn)-len(fn_c):,})')
fp=pred.join(truth,on=['sidx','tidx'],how='anti'); print(f'false positive pairs: {len(fp):,}')
fpo=fp.join(truth.rename({'sidx':'owner'}),on='tidx',how='left'); print(f'  FP whose target truly belongs to another S1: {fpo["owner"].is_not_null().sum():,}; to no S1 (distractor): {fpo["owner"].is_null().sum():,}')
print(d.group_by('country').agg(pl.len(),F=pl.col('f').mean(),single=(pl.col('nt')==0).mean(),single_acc=((pl.col('nt')==0)&(pl.col('np_')==0)).sum()/(pl.col('nt')==0).sum()))
tgf=tg.select(tidx=pl.col('idx').cast(pl.UInt32),nonlatin=~pl.col('business_name').str.contains(r'^[\x00-\x7FÀ-ɏ]*$'),noaddr=pl.col('addr_n')=='')
m=fn.join(cand.with_columns(inc=pl.lit(True)),on=['sidx','tidx'],how='left').join(tgf,on='tidx').join(a.select('sidx','country'),on='sidx')
print(m.group_by('country',pl.col('inc').fill_null(False).alias('in_cands')).agg(pl.len(),nonlatin=pl.col('nonlatin').mean(),no_addr=pl.col('noaddr').mean()).sort('country','in_cands'))
