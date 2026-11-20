"""Webinar 13: drift between fixed v3 train and validation, never tune on test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURES=('amount','tx_count_24h','device_age_days','avg_amount_30d')

def psi(reference,current,bins=10):
    a=np.asarray(reference,dtype=float);b=np.asarray(current,dtype=float)
    if len(a)<20 or len(b)<20 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Each sample needs >=20 finite observations')
    edges=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
    if len(edges)<3:return float('nan')
    edges[0],edges[-1]=-np.inf,np.inf
    x=np.maximum(np.histogram(a,bins=edges)[0]/len(a),1e-6)
    y=np.maximum(np.histogram(b,bins=edges)[0]/len(b),1e-6)
    return float(np.sum((y-x)*np.log(y/x)))

def report(train,validation):
    result={}
    for feature in FEATURES:
        a=train[feature];b=validation[feature]
        stat,p=ks_2samp(a,b)
        value=psi(a,b)
        result[feature]={'ks_statistic':float(stat),'ks_pvalue':float(p),'psi':value,'psi_alert':bool(np.isfinite(value) and value>0.2)}
    return {'reference_rows':len(train),'current_rows':len(validation),'features':result,'action':'investigate_only_no_auto_retraining'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default='webinar13_drift_result.json');args=p.parse_args()
    d=Path(args.data_dir)
    result=report(pd.read_csv(d/'train.csv'),pd.read_csv(d/'validation.csv'))
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
