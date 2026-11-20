from antifraud.model_artifact import MODEL_FEATURES, CATEGORICAL


def test_model_schema_has_no_identifier_or_target():
    assert "is_fraud" not in MODEL_FEATURES
    assert "transaction_id" not in MODEL_FEATURES
    assert "user_id" not in MODEL_FEATURES


def test_categorical_features_in_schema():
    assert set(CATEGORICAL).issubset(MODEL_FEATURES)
