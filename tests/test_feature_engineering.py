import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer, engineer_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "INDE 22": [7.0, 5.5],
        "IAA": [8.0, 6.0],
        "IEG": [7.0, 5.0],
        "IDA": [7.0, 4.0],
        "Matem": [6.5, 4.0],
        "Portug": [7.0, 5.0],
        "Fase": [3, 2],
    })


def test_media_academica_created(sample_df):
    result = engineer_features(sample_df)
    assert "Media_Academica" in result.columns


def test_media_academica_values(sample_df):
    result = engineer_features(sample_df)
    expected = (sample_df["IAA"] + sample_df["IEG"] + sample_df["IDA"]) / 3
    pd.testing.assert_series_equal(
        result["Media_Academica"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_media_notas_created(sample_df):
    result = engineer_features(sample_df)
    assert "Media_Notas" in result.columns


def test_media_notas_values(sample_df):
    result = engineer_features(sample_df)
    expected = (sample_df["Matem"] + sample_df["Portug"]) / 2
    pd.testing.assert_series_equal(
        result["Media_Notas"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_original_columns_preserved(sample_df):
    result = engineer_features(sample_df)
    for col in sample_df.columns:
        assert col in result.columns


def test_does_not_modify_original(sample_df):
    original_cols = list(sample_df.columns)
    engineer_features(sample_df)
    assert list(sample_df.columns) == original_cols


def test_feature_engineer_fit_returns_self(sample_df):
    fe = FeatureEngineer()
    result = fe.fit(sample_df)
    assert result is fe


def test_handles_missing_grade_columns():
    df = pd.DataFrame({
        "IAA": [8.0],
        "IEG": [7.0],
        "IDA": [7.0],
        "Fase": [3],
    })
    result = engineer_features(df)
    assert "Media_Academica" in result.columns
    assert "Media_Notas" not in result.columns
