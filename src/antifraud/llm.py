"""Optional provider-neutral, grounded explanation layer.

LLM generates grounded narrative from verified SHAP factors; no personal data is sent.
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
    provider = os.getenv("ANTIFRAUD_LLM_PROVIDER", "groq").lower()
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
    if not allowed:
        return fallback, "shap_template_no_factors"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": (
                'Составь короткое содержательное объяснение на русском языке для аналитика. '
                'Верни только JSON {"factor_names":["name"],"explanation":"text"}. '
                'Используй только переданные SHAP-факторы и направления их влияния; '
                'названия факторов сохраняй точно, не добавляй чисел, других признаков, '
                'сведений об операции, причинных утверждений или вердикта о мошенничестве. '
                'Укажи, что вклады SHAP не доказывают мошенничество.'
            )},
            {"role": "user", "content": json.dumps([
                {"feature": f["feature"], "direction": f["direction"],
                 "contribution": f.get("shap_value", f.get("shap_log_odds"))}
                for f in factors
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
        if not isinstance(selection, dict) or set(selection) != {"factor_names", "explanation"}:
            raise ValueError("Invalid LLM response schema")
        names = selection["factor_names"]
        if (not isinstance(names, list) or not 1 <= len(names) <= min(len(factors), 5)
                or any(not isinstance(n, str) or n not in allowed for n in names)
                or len(names) != len(set(names))):
            raise ValueError("LLM selected unapproved or repeated factors")
        explanation = selection["explanation"]
        if not isinstance(explanation, str) or not 30 <= len(explanation) <= 900:
            raise ValueError("Invalid explanation text")
        import re
        if re.search(r"\d", explanation):
            raise ValueError("Unverified numeric claims")
        if not all(name.lower() in explanation.lower() for name in names):
            raise ValueError("Selected factors not grounded in text")
        if any(name.lower() in explanation.lower() for name in allowed if name not in names):
            raise ValueError("Unselected factor in explanation")
        return explanation.strip(), provider + "_grounded_generation"
    except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as exc:
        log.warning("LLM explanation unavailable: %s (%s)", type(exc).__name__, str(exc)[:180])
        return fallback, "shap_template_fallback"
