# Вебинар 16 — AI code review, безопасность и ограничения

## Проверка рисков
- **Security:** earlier notebook `Семинар_9.ipynb` contained an embedded ngrok auth token. Current version reads `NGROK_AUTHTOKEN` from the environment. **The exposed token must be revoked by its owner**; editing the file does not erase Git history.
- **Correctness:** model-training notebooks and CSV dataset have different origins and class ratios. Results must not be mixed.
- **Best practices:** explicit validation and tests, versioned artifact metadata, deterministic seeds, train/validation/test split.
- **Data protection:** avoid logging customer identifiers, commit no real transaction data or model artifacts. `.gitignore` excludes CSVs, artifacts, secrets.
- **Bias:** geographic feature `country` is included in the educational candidate; inspect subgroup false-positive rates before considering any real use.
- **LLM hallucination:** Grok/SHAP are architectural intentions, not integrated in the current API. Current explanation explicitly says so.

## AI-assisted review checklist
1. Are secrets stored only in environment/secret manager?
2. Are data provenance and licenses confirmed?
3. Are train/validation/test splits leakage-free?
4. Is the API contract validated with negative tests?
5. Does fallback avoid silently blocking real transactions?
6. Are explanations grounded in model outputs?
7. Are drift alerts separated from retraining triggers?
8. Are model metrics based on independent holdout labels?

## Limitations
The educational dataset is not validated as representative of banking fraud. Score is uncalibrated. No production deployment, real fraud investigation agent, SHAP or Grok integration has been demonstrated. The system must not authorize or block real payments.


## Реализованная защита LLM-слоя
`src/antifraud/llm.py`: внешнему API передаются только имена, направления и SHAP-вклады; пользовательские и платёжные идентификаторы не передаются. Ответ Grok валидируется как JSON с единственным полем `factor_names`; разрешены только признаки из полученного SHAP, без повторов. Свободный текст модели **не используется**, итоговая фраза строится детерминированно. Ошибки сети, отсутствие ключа или недопустимые ответы ведут к безопасному fallback. Данные реальных клиентов использовать без согласования нельзя. Тесты с mock HTTP не подтверждают доступность настоящего Grok API.
