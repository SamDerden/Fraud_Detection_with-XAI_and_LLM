"""Webinar 7: six checks of v3; skip if data unavailable in CI."""
import os
from pathlib import Path
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,f1_score,accuracy_score

@pytest.fixture(scope='module')
def prepared():
    path=os.environ.get('ANTIFRAUD_V3_CSV')
    if not path or not Path(path).is_file():pytest.skip('Set ANTIFRAUD_V3_CSV to accepted v3 CSV')
    d=pd.read_csv(path)
    assert d.split.value_counts().to_dict()=={'train':5000,'validation':1000,'test':1000}
    features=['amount','hour','day_of_week','avg_amount_30d','tx_count_24h','tx_count_7d','distinct_merchants_7d','distinct_countries_30d','device_age_days','is_new_device','is_night_tx','channel','country']
    x=pd.get_dummies(d[features],dtype=int);train=d.split=='train';validation=d.split=='validation'
    m=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=2)
    m.fit(x[train],d.is_fraud[train]);p=m.predict_proba(x[validation])[:,1]
    return d,m,x,p,validation

def test_01_functionality_probabilities(prepared):
    _,_,_,p,_=prepared
    assert len(p)==1000 and all(0<=v<=1 for v in p)

def test_02_quality_better_than_prevalence(prepared):
    d,_,_,p,v=prepared
    assert average_precision_score(d.is_fraud[v],p)>d.is_fraud[v].mean()

def test_03_stability_repeated_inference(prepared):
    _,m,x,_,v=prepared
    assert (m.predict_proba(x[v])[:,1]==m.predict_proba(x[v])[:,1]).all()

def test_04_boundary_positive_amount(prepared):
    d,_,_,_,_=prepared
    assert d.amount.gt(0).all()

def test_05_data_integrity_history(prepared):
    d,_,_,_,_=prepared
    assert d.tx_count_7d.ge(d.tx_count_24h).all()
    assert d.distinct_merchants_7d.le(d.tx_count_7d).all()

def test_06_classification_metrics_defined(prepared):
    d,_,_,p,v=prepared
    y=d.is_fraud[v];pred=p>=.5
    assert 0<=accuracy_score(y,pred)<=1 and 0<=f1_score(y,pred,zero_division=0)<=1
