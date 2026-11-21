"""Controlled feature-group ablation; NEVER replaces the deployed ExtraTrees v4.
Run with accepted chronological v3 CSV: PYTHONPATH=src python experiments/ablate_device_v4.py --data /path/to/v3.csv
"""
import argparse,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,recall_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from antifraud.extratrees_v4 import BASE,ENGINEERED

def design(df):
    x=df[list(BASE)].copy()
    x["amount_to_avg"]=x.amount/(x.avg_amount_30d+1)
    x["log_amount"]=np.log1p(x.amount)
    x["log_avg_amount"]=np.log1p(x.avg_amount_30d)
    x["tx24_share"]=x.tx_count_24h/(x.tx_count_7d+1)
    x["merchants_per_tx"]=x.distinct_merchants_7d/(x.tx_count_7d+1)
    x["tx_per_day"]=x.tx_count_7d/7
    x["log_device_age"]=np.log1p(x.device_age_days)
    return pd.get_dummies(x,columns=["channel","country"],dtype=int)

def run(path):
    d=pd.read_csv(path)
    assert len(d)==7000 and d.split.value_counts().to_dict()=={"train":5000,"validation":1000,"test":1000}
    assert d.split.tolist()==["train"]*5000+["validation"]*1000+["test"]*1000,"Expected chronological split order"
    assert d.is_fraud.isin([0,1]).all()
    x=design(d);y=d.is_fraud.to_numpy()
    with zipfile.ZipFile("extratrees-v4-artifact.zip") as z:
        from io import BytesIO
        baseline=joblib.load(BytesIO(z.read("extratrees-v4/model.joblib")))
    original=baseline["model"]
    params=original.get_params()
    print("Same model hyperparameters across experiments:",{k:params[k] for k in ("n_estimators","max_depth","min_samples_leaf")})
    groups={
      "full_20_features":[],
      "without_device_age_only":["device_age_days"],
      "without_device_family":["device_age_days","is_new_device","log_device_age"],
    }
    result=[]
    for name,drop in groups.items():
        columns=[c for c in x if c not in drop]
        scaler=StandardScaler().fit(x.iloc[:5000][columns])
        train=scaler.transform(x.iloc[:5000][columns])
        val=scaler.transform(x.iloc[5000:6000][columns])
        test=scaler.transform(x.iloc[6000:][columns])
        model=ExtraTreesClassifier(**params).fit(train,y[:5000])
        val_score=model.predict_proba(val)[:,1]
        test_score=model.predict_proba(test)[:,1]
        # Select threshold on validation, never on test.
        negatives=val_score[y[5000:6000]==0]
        thresholds=np.unique(np.r_[val_score,1.0+1e-9])
        threshold=next(float(t) for t in thresholds if np.mean(negatives>=t)<=.10)
        pred=test_score>=threshold
        cm=confusion_matrix(y[6000:],pred,labels=[0,1])
        row=dict(experiment=name,removed=drop,validation_pr_auc=float(average_precision_score(y[5000:6000],val_score)),
                 test_pr_auc=float(average_precision_score(y[6000:],test_score)),
                 test_roc_auc=float(roc_auc_score(y[6000:],test_score)),
                 test_recall=float(cm[1,1]/max(1,cm[1].sum())),
                 test_fpr=float(cm[0,1]/max(1,cm[0].sum())),
                 validation_threshold=threshold)
        result.append(row);print(json.dumps(row,ensure_ascii=False))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("artifacts/device_ablation_report.json"))
    args=p.parse_args();result=run(args.data);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved:",args.output)
