"""Finalists: fair chronological CV. Run: python experiments/finalists_cv.py --data-dir PATH --trials 50
No test access; CatBoost and LightGBM receive equal Optuna budgets. ExtraTrees v4 frozen.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
BASE=['amount','hour','day_of_week','avg_amount_30d','tx_count_24h','tx_count_7d','distinct_merchants_7d','distinct_countries_30d','device_age_days','is_new_device','is_night_tx','channel','country']
def features(d):
 x=d[BASE].copy()
 x['amount_to_avg']=x.amount/(x.avg_amount_30d+1)
 x['log_amount']=np.log1p(x.amount)
 x['log_avg_amount']=np.log1p(x.avg_amount_30d)
 x['tx24_share']=x.tx_count_24h/(x.tx_count_7d+1)
 x['merchants_per_tx']=x.distinct_merchants_7d/(x.tx_count_7d+1)
 x['tx_per_day']=x.tx_count_7d/7
 x['log_device_age']=np.log1p(x.device_age_days)
 return pd.get_dummies(x,columns=['channel','country'],dtype=int)
def main(data_dir,trials,out):
 d=pd.read_csv(data_dir/'train.csv');x=features(d);y=d.is_fraud.to_numpy()
 folds=[(0,2500,2500,3250),(0,3250,3250,4000),(0,4000,4000,5000)]
 def evaluate(family,params):
  scores=[]
  for a,b,c,e in folds:
   scaler=StandardScaler().fit(x.iloc[a:b]);xt=scaler.transform(x.iloc[a:b]);xv=scaler.transform(x.iloc[c:e])
   if family=='ExtraTrees':m=ExtraTreesClassifier(n_estimators=200,max_depth=4,min_samples_leaf=6,n_jobs=2,random_state=42)
   elif family=='CatBoost':m=CatBoostClassifier(**params,loss_function='Logloss',verbose=False,thread_count=2,random_seed=42,allow_writing_files=False)
   else:m=LGBMClassifier(**params,random_state=42,n_jobs=2,verbosity=-1)
   m.fit(xt,y[a:b]);scores.append(float(average_precision_score(y[c:e],m.predict_proba(xv)[:,1])))
  return scores
 results={'ExtraTrees':{'folds':evaluate('ExtraTrees',{}),'params':{'n_estimators':200,'max_depth':4,'min_samples_leaf':6}}}
 for family in ('CatBoost','LightGBM'):
  def objective(t):
   if family=='CatBoost':p=dict(iterations=t.suggest_int('iterations',150,450,step=50),depth=t.suggest_int('depth',3,8),learning_rate=t.suggest_float('learning_rate',.015,.16,log=True),l2_leaf_reg=t.suggest_float('l2_leaf_reg',1,20,log=True))
   else:p=dict(n_estimators=t.suggest_int('n_estimators',100,400,step=50),num_leaves=t.suggest_int('num_leaves',7,31),min_child_samples=t.suggest_int('min_child_samples',20,100),learning_rate=t.suggest_float('learning_rate',.015,.16,log=True),reg_lambda=t.suggest_float('reg_lambda',.1,10,log=True))
   vals=evaluate(family,p);t.set_user_attr('folds',vals);return float(np.mean(vals))
  study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=2026))
  study.optimize(objective,n_trials=trials)
  results[family]={'folds':study.best_trial.user_attrs['folds'],'params':study.best_params,'trials':trials}
 for k,v in results.items():v['mean']=float(np.mean(v['folds']));v['std']=float(np.std(v['folds']))
 if (data_dir/'validation.csv').exists():
  val=pd.read_csv(data_dir/'validation.csv');v=features(val).reindex(columns=x.columns,fill_value=0)
  scaler=StandardScaler().fit(x);xt=scaler.transform(x);xv=scaler.transform(v)
  for family in ('CatBoost','LightGBM'):
   p=results[family]['params']
   m=CatBoostClassifier(**p,loss_function='Logloss',verbose=False,thread_count=2,random_seed=42,allow_writing_files=False) if family=='CatBoost' else LGBMClassifier(**p,random_state=42,n_jobs=2,verbosity=-1)
   m.fit(xt,y);pred=m.predict_proba(xv)[:,1]
   results[family]['validation_pr_auc']=float(average_precision_score(val.is_fraud,pred))
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,ensure_ascii=False,indent=2))
 print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--trials',type=int,default=50);ap.add_argument('--out',type=Path,default=Path('experiments/finalists_cv_results.json'));args=ap.parse_args();main(args.data_dir,args.trials,args.out)
