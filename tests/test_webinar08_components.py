"""Webinar 8: unit and mock integration exercises, independent of banking API."""
from unittest.mock import Mock
import re
import pytest

def clean_text(value: str) -> str:
    return ' '.join(re.sub(r'<[^>]*>', ' ', value).lower().split())

@pytest.mark.parametrize('raw,expected',[
    ('<b>FRAUD</b>','fraud'),('  Many   spaces ','many spaces'),
    ('UPPER Case','upper case'),('',''),('<p> Hi </p>  WORLD','hi world'),
])
def test_clean_text(raw,expected):
    assert clean_text(raw)==expected

def handle_score(amount:float,predict_risk):
    if amount<=0:raise ValueError('amount must be positive')
    score=float(predict_risk(amount))
    if not 0<=score<=1:raise ValueError('invalid model score')
    return {'score':score,'decision':'review' if score>=.5 else 'approve'}

@pytest.mark.parametrize('score,decision',[(.1,'approve'),(.5,'review'),(.9,'review')])
def test_mock_model_decisions(score,decision):
    model=Mock(return_value=score);result=handle_score(100,model)
    assert result['decision']==decision
    assert set(result)=={'score','decision'}
    model.assert_called_once_with(100)

def test_mock_model_failure():
    model=Mock(side_effect=RuntimeError('model unavailable'))
    with pytest.raises(RuntimeError,match='model unavailable'):handle_score(100,model)

def test_contract_rejects_invalid_amount():
    with pytest.raises(ValueError,match='positive'):handle_score(-1,Mock(return_value=.5))

def test_contract_rejects_invalid_probability():
    with pytest.raises(ValueError,match='invalid model score'):handle_score(100,Mock(return_value=1.2))
