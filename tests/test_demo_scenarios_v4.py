"""Three real ExtraTrees v4 demo scenarios, including non-multiple-of-100 amount."""
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from antifraud.app import app
from test_webinar07_extratrees_v4 import final_model as final_model_fixture

@pytest.mark.parametrize("name,amount,hour,history,device,channel,country,average",[
    ("ordinary",850,13,(1,5,3,1),220,"mobile","RU",1700),
    ("suspicious",45000,22,(8,15,5,2),2,"mobile","RU",5000),
    ("night",68000,2,(12,22,8,4),0,"web","TR",2200),
])
def test_demo_v4(name,amount,hour,history,device,channel,country,average,final_model,monkeypatch):
    root,_,_=final_model
    monkeypatch.setenv("ANTIFRAUD_MODEL_DIR",str(root))
    monkeypatch.setenv("ANTIFRAUD_MODEL_KIND","extratrees_v4")
    monkeypatch.setenv("ANTIFRAUD_REQUIRE_ARTIFACT","1")
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED","false")
    payload=dict(transaction_id="demo-"+name,user_id="demo",amount=amount,
        timestamp=f"2026-11-21T{hour:02d}:00:00Z",avg_amount_30d=average,
        tx_count_24h=history[0],tx_count_7d=history[1],
        distinct_merchants_7d=history[2],distinct_countries_30d=history[3],
        device_age_days=device,is_new_device=int(device<7),is_night_tx=int(hour<=5),
        channel=channel,country=country)
    with TestClient(app) as client:
        response=client.post("/v1/score",json=payload)
    assert response.status_code==200,response.text
    body=response.json()
    assert body["model_version"]=="extratrees-engineered-v4"
    assert 0<=body["risk_score"]<=1
    assert body["decision"] in ("approve","review")
    print(name,body["risk_score"],body["decision"])
