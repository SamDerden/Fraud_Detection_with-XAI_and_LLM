"""Optional provider-neutral, grounded explanation layer.

LLM selects existing diagnostic factor names (score-delta for v4); no personal data is sent.
If provider is unavailable, deterministic explanation is always returned.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

from antifraud.explain import deterministic_explanation

log = logging.getLogger("antifraud.llm")

PROVIDERS = {
    "grok": ("https://api.x.ai/v1/chat/completions", "XAI_API_KEY", "GROK_MODEL"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "GROQ_MODEL"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"),
}
# Hostnames and URLs are fixed: no user-controlled URL / SSRF.


def explain_with_optional_grok(factors: list[dict]) -> tuple[str, str]:
    """Backward-compatible public function; supports grok, groq, openrouter."""
    fallback = deterministic_explanation(factors)
    if os.getenv("ANTIFRAUD_LLM_ENABLED", "").lower() not in {"1", "true", "yes"}:
        return fallback, "shap_template"
    provider = os.getenv("ANTIFRAUD_LLM_PROVIDER", "grok").lower()
    if provider not in PROVIDERS:
        return fallback, "shap_template_unsupported_provider"
    endpoint, key_name, model_name = PROVIDERS[provider]
    token = os.getenv(key_name, "")
    if not token:
        return fallback, "shap_template_no_credentials"
    model = os.getenv(model_name, "")
    if not model:
        return fallback, "shap_template_no_model"
    allowed = {f["feature"]: f for f in factors}
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": (
                'Select existing feature names for an antifraud operator explanation. '
                'Reply ONLY JSON: {"factor_names":["name"]}. '
                "Use only supplied names, at most five, no extra fields. "
                "Do not infer fraud, invent reasons, or include personal data."
            )},
            {"role": "user", "content": json.dumps([
                {"feature": f["feature"], "direction": f["direction"],
                 "contribution": f.get("score_delta", f.get("shap_log_odds"))} for f in factors
            ], ensure_ascii=False)},
        ],
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=False) as client:
            response = client.post(
                endpoint, json=payload,
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        selection = json.loads(content)
        if not isinstance(selection, dict) or set(selection) != {"factor_names"}:
            raise ValueError("Invalid LLM response schema")
        names = selection["factor_names"]
        if (not isinstance(names, list) or not 1 <= len(names) <= min(len(factors), 5)
                or any(not isinstance(n, str) or n not in allowed for n in names)
                or len(names) != len(set(names))):
            raise ValueError("LLM selected unapproved or repeated factors")
        return deterministic_explanation([allowed[n] for n in names]), provider + "_grounded_selection"
    except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as exc:
        log.warning("LLM explanation unavailable: %s (%s)", type(exc).__name__, str(exc)[:180])
        return fallback, "shap_template_fallback"
