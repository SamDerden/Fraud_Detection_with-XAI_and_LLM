"""Deep ExtraTrees optimization on chronological training folds. No test access.
Run: python experiments/deep_extratrees.py --data-dir /path/to/v3 --trials 80
Dependencies: optuna pandas numpy scikit-learn joblib
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,optuna,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
BASE=['amount','hour','day_of_week','avg_amount_30d','tx_count_24h','tx_count_7d','distinct_merchants_7d','distinct_countries_30d','device_age_days','is_new_device','is_night_tx','channel','country']
def features(df,engineered):
 x=df[BASE].copy()
 if engineered:
  x['amount_to_avg']=x.amount/(x.avg_amount_30d+1)
  x['log_amount']=np.log1p(x.amount)
  x['log_avg_amount']=np.log1p(x.avg_amount_30d)
  x['tx24_share']=x.tx_count_24h/(x.tx_count_7d+1)
  x['merchants_per_tx']=x.distinct_merchants_7d/(x.tx_count_7d+1)
  x['tx_per_day']=x.tx_count_7d/7
  x['log_device_age']=np.log1p(x.device_age_days)
 return pd.get_dummies(x,columns=['channel','country'],dtype=int)
def run(data_dir,trials,out):
 tr=pd.read_csv(data_dir/'train.csv');va=pd.read_csv(data_dir/'validation.csv')
 results=[];optuna.logging.set_verbosity(optuna.logging.WARNING)
 for engineered in (False,True):
  x=features(tr,engineered);v=features(va,engineered).reindex(columns=x.columns,fill_value=0)
  y=tr['is_fraud'].to_numpy();yv=va['is_fraud'].to_numpy()
  folds=[(0,2500,2500,3250),(0,3250,3250,4000),(0,4000,4000,5000)]
  def objective(t):
   params=dict(n_estimators=t.suggest_int('n_estimators',150,650,step=100),max_depth=t.suggest_int('max_depth',3,18),min_samples_leaf=t.suggest_int('min_samples_leaf',1,25),min_samples_split=t.suggest_int('min_samples_split',2,30),max_features=t.suggest_float('max_features',.4,1.),bootstrap=t.suggest_categorical('bootstrap',[False,True]),class_weight=t.suggest_categorical('class_weight',[None,'balanced','balanced_subsample']))
   scores=[]
   for a,b,c,d in folds:
    m=ExtraTreesClassifier(**params,n_jobs=2,random_state=42).fit(x.iloc[a:b],y[a:b]);scores.append(average_precision_score(y[c:d],m.predict_proba(x.iloc[c:d])[:,1]))
   t.set_user_attr('fold_scores',scores)
   return float(np.mean(scores))
  study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=2026));study.optimize(objective,n_trials=trials)
  model=ExtraTreesClassifier(**study.best_params,n_jobs=2,random_state=42).fit(x,y)
  p=model.predict_proba(v)[:,1]
  record=dict(features='engineered' if engineered else 'base',trials=trials,inner_cv_pr_auc=study.best_value,fold_scores=study.best_trial.user_attrs['fold_scores'],validation_pr_auc=average_precision_score(yv,p),validation_roc_auc=roc_auc_score(yv,p),params=study.best_params,columns=list(x.columns))
  results.append(record);print(json.dumps(record,ensure_ascii=False,default=float),flush=True)
  joblib.dump(model,out/('extratrees_'+record['features']+'.joblib'))
 (out/'deep_extratrees_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=float))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--trials',type=int,default=80);ap.add_argument('--out',type=Path,default=Path('artifacts/deep-extratrees'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);run(args.data_dir,args.trials,args.out)
