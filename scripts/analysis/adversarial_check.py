"""Adversarial train-vs-test check for one country.

Trains a quick classifier to tell training candidate pairs from test candidate
pairs of the same country, using a model's stage-1 features. AUC near 0.5 means
no shift. The original r2 features give ~0.70-0.75, which is the natural
difference. Anything much higher (r6: 0.93-0.97) means some feature depends on
the split itself and will not transfer to the leaderboard.

usage: python scripts/analysis/adversarial_check.py <country> <model_dir>/metrics.json
"""
import glob
import json
import sys

import numpy as np
import polars as pl
import xgboost as xgb

feats = json.load(open(sys.argv[2]))["stage1_features"]


def samp(split,n_files=8,per=150000):
    fs=sorted(glob.glob(f'work/feats_{split}/part_*.parquet'))
    c=pl.read_parquet(f'work/norm/{split}_source1.parquet',columns=['idx','country']).select(sidx=pl.col('idx').cast(pl.UInt32),country='country')
    pick=[fs[i] for i in np.linspace(0,len(fs)-1,n_files).astype(int)]
    out=[]
    for f in pick:
        d=pl.read_parquet(f,columns=feats+['sidx']).join(c,on='sidx').filter(pl.col('country')==sys.argv[1])
        if len(d): out.append(d.sample(min(per,len(d)),seed=1))
    return pl.concat(out).select(feats)
tr=samp('train'); te=samp('test')
X=np.vstack([tr.select(pl.all().cast(pl.Float32)).to_numpy(),te.select(pl.all().cast(pl.Float32)).to_numpy()])
y=np.r_[np.zeros(len(tr)),np.ones(len(te))]
idx=np.random.RandomState(0).permutation(len(y)); X,y=X[idx],y[idx]; k=int(.8*len(y))
d=xgb.DMatrix(X[:k],y[:k],feature_names=feats); v=xgb.DMatrix(X[k:],y[k:],feature_names=feats)
m=xgb.train({'objective':'binary:logistic','eval_metric':'auc','max_depth':6,'eta':0.2,'device':'cuda','nthread':8},d,150,evals=[(v,'v')],verbose_eval=False)
from sklearn.metrics import roc_auc_score
print('train-vs-test AUC (0.5 = no shift):',round(roc_auc_score(y[k:],m.predict(v)),4))
g=m.get_score(importance_type='gain'); tot=sum(g.values())
for f,s in sorted(g.items(),key=lambda x:-x[1])[:12]: print(f'{f:24s} {100*s/tot:5.1f}%  train mean {tr[f].cast(pl.Float64).mean():.3f}  test mean {te[f].cast(pl.Float64).mean():.3f}')
