import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import ENGINEERED_NUMERIC, MODEL_PATH, build_pipeline


@pytest.fixture
def small_dataset():
    """Dataset mínimo para testar o pipeline sem treino completo."""
    np.random.seed(42)
    n = 30
    ipp = np.where(np.random.rand(n) > 0.5, np.random.uniform(3, 10, n), np.nan)
    return pd.DataFrame({
        "INDE 22": np.random.uniform(4, 9, n),
        "IAA": np.random.uniform(3, 10, n),
        "IEG": np.random.uniform(3, 10, n),
        "IPS": np.random.uniform(3, 10, n),
        "IPP": ipp,  # metade NaN, como ocorre com dados de 2022
        "IDA": np.random.uniform(3, 10, n),
        "IPV": np.random.uniform(3, 10, n),
        "Matem": np.random.uniform(2, 10, n),
        "Portug": np.random.uniform(2, 10, n),
        "Fase": np.random.randint(0, 8, n),
        "Idade 22": np.random.randint(8, 20, n),
        "Gênero": np.random.choice(["Menino", "Menina"], n),
        "Pedra 22": np.random.choice(["Ametista", "Ágata", "Quartzo", "Topázio"], n),
    })


@pytest.fixture
def small_target(small_dataset):
    return pd.Series(np.random.choice([0, 1, 2], len(small_dataset)))


def test_build_pipeline_returns_pipeline():
    from sklearn.linear_model import LogisticRegression
    pipeline = build_pipeline(LogisticRegression(max_iter=10))
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_steps():
    from sklearn.linear_model import LogisticRegression
    pipeline = build_pipeline(LogisticRegression(max_iter=10))
    step_names = [name for name, _ in pipeline.steps]
    assert "feature_engineer" in step_names
    assert "preprocessor" in step_names
    assert "classifier" in step_names


def test_pipeline_fit_predict(small_dataset, small_target):
    from sklearn.linear_model import LogisticRegression
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(small_dataset, small_target)
    preds = pipeline.predict(small_dataset)
    assert len(preds) == len(small_dataset)
    assert set(preds).issubset({0, 1, 2})


def test_pipeline_predict_proba(small_dataset, small_target):
    from sklearn.ensemble import RandomForestClassifier
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=5, random_state=42))
    pipeline.fit(small_dataset, small_target)
    probas = pipeline.predict_proba(small_dataset)
    assert probas.shape == (len(small_dataset), 3)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_engineered_numeric_contains_derived_features():
    assert "Media_Academica" in ENGINEERED_NUMERIC
    assert "Media_Notas" in ENGINEERED_NUMERIC
