"""Optional Grok explanation layer with strict allowlisted output and safe fallback.

Only non-PII SHAP feature names/directions are sent. LLM may select/order
grounded factors; the final explanatory sentences are rendered deterministically.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

from antifraud.explain import deterministic_explanation

log = logging.getLogger("antifraud.llm")
ENDPOINT = "https://api.x.ai/v1/chat/completions"


def explain_with_optional_grok(factors: list[dict]) -> tuple[str, str]:
    fallback = deterministic_explanation(factors)
    if os.getenv("ANTIFRAUD_LLM_ENABLED", "").lower() not in {"1", "true", "yes"}:
        return fallback, "shap_template"
    token = os.getenv("XAI_API_KEY", "")
    if not token:
        return fallback, "shap_template_no_credentials"
    model = os.getenv("GROK_MODEL", "")
    if not model:
        return fallback, "shap_template_no_model"
    allowed = {f["feature"]: f for f in factors}
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": (
                "You select feature names for an antifraud operator explanation. "
                "Reply ONLY JSON: {\\\"factor_names\\\":[\\\"name\\\"]}. "
                "Use only supplied names, at most five, no extra fields. "
                "Do not infer fraud, invent reasons, or include personal data."
            )},
            {"role": "user", "content": json.dumps([
                {"feature": f["feature"], "direction": f["direction"],
                 "shap_log_odds": f["shap_log_odds"]} for f in factors
            ], ensure_ascii=False)},
        ],
    }
    try:
        with httpx.Client(timeout=3.0, follow_redirects=False) as client:
            response = client.post(
                ENDPOINT, json=payload,
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        selection = json.loads(content)
        if not isinstance(selection, dict) or set(selection) != {"factor_names"}:
            raise ValueError("Invalid LLM response schema")
        names = selection["factor_names"]
        if (not isinstance(names, list) or not 1 <= len(names) <= len(factors)
                or any(not isinstance(n, str) or n not in allowed for n in names)
                or len(names) != len(set(names))):
            raise ValueError("LLM selected unapproved or repeated factors")
        selected = [allowed[n] for n in names]
        return deterministic_explanation(selected), "grok_grounded_selection"
    except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as exc:
        log.warning("LLM explanation unavailable: %s", type(exc).__name__)
        return fallback, "shap_template_fallback"
