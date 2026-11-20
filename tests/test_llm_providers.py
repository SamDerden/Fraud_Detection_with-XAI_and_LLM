"""Provider selection, safety and failure modes without network."""
import json
import pytest
import antifraud.llm as llm

FACTORS = [{"feature": "amount", "direction": "increases_risk", "shap_log_odds": 0.8}]


@pytest.mark.parametrize("provider,key,model,url", [
    ("groq", "GROQ_API_KEY", "GROQ_MODEL", "https://api.groq.com/openai/v1/chat/completions"),
    ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "https://openrouter.ai/api/v1/chat/completions"),
    ("grok", "XAI_API_KEY", "GROK_MODEL", "https://api.x.ai/v1/chat/completions"),
])
def test_provider_routing_and_grounded_selection(monkeypatch, provider, key, model, url):
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "true")
    monkeypatch.setenv("ANTIFRAUD_LLM_PROVIDER", provider)
    monkeypatch.setenv(key, "dummy-secret")
    monkeypatch.setenv(model, "test-model")

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": '{"factor_names":["amount"]}'}}]}

    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def post(self, endpoint, json, headers):
            assert endpoint == url
            assert headers["Authorization"] == "Bearer dummy-secret"
            assert "user_id" not in str(json)
            return Response()

    monkeypatch.setattr(llm.httpx, "Client", Client)
    explanation, method = llm.explain_with_optional_grok(FACTORS)
    assert method == provider + "_grounded_selection"
    assert "amount" in explanation


def test_unknown_provider_never_calls_network(monkeypatch):
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "true")
    monkeypatch.setenv("ANTIFRAUD_LLM_PROVIDER", "https://attacker.invalid")
    _, method = llm.explain_with_optional_grok(FACTORS)
    assert method == "shap_template_unsupported_provider"


def test_missing_key_is_safe(monkeypatch):
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "true")
    monkeypatch.setenv("ANTIFRAUD_LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _, method = llm.explain_with_optional_grok(FACTORS)
    assert method == "shap_template_no_credentials"
