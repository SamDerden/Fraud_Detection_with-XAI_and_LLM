"""Three tracked RF runs on accepted v3; no test-set selection."""
import argparse,hashlib
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--tracking-uri',default='file:./mlruns');args=p.parse_args()
    try:import mlflow
    except ImportError as e:raise SystemExit('MLflow not installed: pip install mlflow') from e
    import joblib
    d=pd.read_csv(args.data)
    assert d.split.value_counts().to_dict()=={'train':5000,'validation':1000,'test':1000}
    assert (d.split.iloc[:5000]=='train').all() and (d.split.iloc[5000:6000]=='validation').all() and (d.split.iloc[6000:]=='test').all()
    features=['amount','hour','day_of_week','avg_amount_30d','tx_count_24h','tx_count_7d','distinct_merchants_7d','distinct_countries_30d','device_age_days','is_new_device','is_night_tx','channel','country']
    x=pd.get_dummies(d[features],dtype=int);y=d.is_fraud
    train=d.split=='train';validation=d.split=='validation'
    mlflow.set_tracking_uri(args.tracking_uri);mlflow.set_experiment('webinar05-antifraud-v3')
    rows=[];digest=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    for trees,depth in [(100,6),(200,8),(350,12)]:
        with mlflow.start_run(run_name=f'rf_{trees}_{depth}') as run:
            model=RandomForestClassifier(n_estimators=trees,max_depth=depth,random_state=42,class_weight='balanced',n_jobs=2)
            model.fit(x.loc[train],y.loc[train]);pred=model.predict_proba(x.loc[validation])[:,1]
            pr=float(average_precision_score(y.loc[validation],pred));roc=float(roc_auc_score(y.loc[validation],pred))
            mlflow.log_params({'n_estimators':trees,'max_depth':depth,'seed':42,'data_sha256':digest,'train_rows':5000,'validation_rows':1000,'split':'explicit_chronological'})
            mlflow.log_metrics({'validation_pr_auc':pr,'validation_roc_auc':roc})
            artifact=Path('webinar05_model.joblib');joblib.dump(model,artifact);mlflow.log_artifact(str(artifact));artifact.unlink()
            rows.append({'run_id':run.info.run_id,'n_estimators':trees,'max_depth':depth,'validation_pr_auc':pr,'validation_roc_auc':roc})
    result=pd.DataFrame(rows).sort_values('validation_pr_auc',ascending=False)
    print(result.to_string(index=False));result.to_csv('webinar05_runs.csv',index=False)
    print('Selection uses validation only. Test set not accessed for tuning.')
if __name__=='__main__':main()
