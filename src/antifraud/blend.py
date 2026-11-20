"""Probability-weighted CatBoost + LightGBM ensemble and permutation Shapley."""
import json
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REFERENCE={"amount":2129.16,"hour":12,"day_of_week":3,"avg_amount_30d":3081.42,"tx_count_24h":0,"tx_count_7d":3,"distinct_merchants_7d":3,"distinct_countries_30d":1,"device_age_days":69.1422,"is_new_device":0,"is_night_tx":0,"channel":"mobile","country":"RU"}

@lru_cache(maxsize=2)
def load_blend(directory):
    p=Path(directory)
    meta=json.loads((p/"metadata.json").read_text())
    if meta["model_version"]!="blend-catboost-lightgbm-v3":
        raise ValueError("Unexpected ensemble version")
    cat=CatBoostClassifier()
    cat.load_model(str(p/"catboost.cbm"))
    light=joblib.load(p/"lightgbm.joblib")
    return cat,light,meta

def predict_batch(directory, rows):
    cat,light,meta=load_blend(directory)
    x=pd.DataFrame([{k:r[k] for k in meta["features"]} for r in rows])
    for k in meta["categorical"]:x[k]=x[k].astype(str)
    z=pd.get_dummies(x,columns=meta["categorical"],dtype=int)
    z=z.reindex(columns=meta["lightgbm_columns"],fill_value=0)
    return meta["catboost_weight"]*cat.predict_proba(x)[:,1]+meta["lightgbm_weight"]*light.predict_proba(z)[:,1]

def score_and_explain(directory, features, top_k=5):
    _,_,meta=load_blend(directory)
    keys=meta["features"]
    rng=np.random.default_rng(42)
    perms=[]
    for _ in range(16):
        p=rng.permutation(len(keys))
        perms.extend((p,p[::-1]))
    rows=[]
    for perm in perms:
        state=dict(REFERENCE)
        rows.append(state.copy())
        for j in perm:
            state[keys[j]]=features[keys[j]]
            rows.append(state.copy())
    probabilities=predict_batch(directory,rows).reshape(len(perms),len(keys)+1)
    contributions=np.zeros(len(keys))
    for perm,values in zip(perms,probabilities):
        for step,j in enumerate(perm):
            contributions[j]+=values[step+1]-values[step]
    contributions/=len(perms)
    score=float(probabilities[0,-1])
    if not np.isclose(probabilities[0,0]+contributions.sum(),score,atol=1e-9):
        raise RuntimeError("Shapley reconstruction failed")
    ranked=sorted(zip(keys,contributions),key=lambda p:abs(p[1]),reverse=True)
    factors=[{"feature":name,"value":features[name],"shap_log_odds":round(float(v),6),"direction":"increases_risk" if v>0 else "decreases_risk"} for name,v in ranked[:top_k]]
    return score,meta,factors
