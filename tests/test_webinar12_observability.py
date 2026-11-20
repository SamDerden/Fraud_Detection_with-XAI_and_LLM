"""Webinar 12: JSON structured logs and Prometheus HTTP metrics."""
import json
import logging
from fastapi.testclient import TestClient
from antifraud.app import app

client=TestClient(app)

def test_structured_json_log_and_422(caplog):
    with caplog.at_level(logging.INFO,logger='antifraud.api'):
        response=client.post('/v1/score',json={'amount':-1})
    assert response.status_code==422
    records=[]
    for record in caplog.records:
        if record.name!='antifraud.api':continue
        try:records.append(json.loads(record.message))
        except json.JSONDecodeError:continue
    assert any(x['status_code']==422 and x['method']=='POST' and x['duration_ms']>=0 and 'route' in x for x in records)
    assert all('amount' not in x and 'user_id' not in x for x in records)

def test_prometheus_counters_after_422():
    client.post('/v1/score',json={'amount':-1})
    response=client.get('/metrics')
    assert response.status_code==200
    assert 'antifraud_http_requests_total' in response.text
    assert 'antifraud_http_request_duration_seconds_bucket' in response.text
