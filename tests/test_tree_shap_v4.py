"""TreeSHAP attribution invariants on the actual saved ExtraTrees v4 artifact."""
import numpy as np
from antifraud.extratrees_v4 import _feature_group, _positive_class_shap, _tree_explainer, load_extratrees, score_and_explain_extratrees, transform
from test_webinar07_extratrees_v4 import final_model

def test_tree_shap_reconstructs_probability(final_model):
    root,bundle,meta=final_model
    row=dict(amount=850,hour=13,day_of_week=5,avg_amount_30d=1700,tx_count_24h=1,tx_count_7d=5,distinct_merchants_7d=3,distinct_countries_30d=1,device_age_days=220,is_new_device=0,is_night_tx=0,channel="mobile",country="RU")
    scaled=bundle["scaler"].transform(transform(row,meta["encoded_columns"]))
    values,base=_positive_class_shap(_tree_explainer(str(root)),scaled)
    probability=float(bundle["model"].predict_proba(scaled)[0,1])
    assert len(values)==len(meta["encoded_columns"])
    assert np.isclose(base+values.sum(),probability,atol=1e-5)
    score,details,factors=score_and_explain_extratrees(str(root),row)
    assert np.isclose(score,probability)
    assert details["explanation_type"]=="tree_shap_probability"
    assert len(factors)==5
    assert all("shap_value" in f and "score_delta" not in f for f in factors)
    assert np.isclose(details["shap_reconstruction"],score,atol=1e-5)

def test_encoded_and_derived_groups():
    assert _feature_group("channel_mobile")=="channel"
    assert _feature_group("country_RU")=="country"
    assert _feature_group("log_device_age")==_feature_group("is_new_device")=="device_history"
    assert _feature_group("amount_to_avg")==_feature_group("log_amount")=="amount_context"
