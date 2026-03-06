import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    _parse_fase,
    create_target,
    get_features,
    load_data,
    prepare_data,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "INDE 22": [7.0, 5.5, 8.2],
        "IAA": [8.0, 6.0, 7.5],
        "IEG": [7.0, 5.0, 9.0],
        "IPS": [6.5, 5.5, 8.0],
        "IPP": [None, 7.2, None],  # opcional - NaN em 2022
        "IDA": [7.0, 4.0, 8.5],
        "IPV": [7.5, 5.0, 9.0],
        "IAN": [6.0, 5.0, 8.0],
        "Matem": [6.5, 4.0, 9.0],
        "Portug": [7.0, 5.0, 8.0],
        "Fase": [3, 2, 5],
        "Idade 22": [14, 13, 17],
        "Gênero": ["Menino", "Menina", "Menino"],
        "Pedra 22": ["Ametista", "Ágata", "Quartzo"],
        "Defas": [0, -1, -3],
    })


def test_create_target_sem_risco(sample_df):
    y = create_target(sample_df)
    assert y.iloc[0] == 0


def test_create_target_risco_medio(sample_df):
    y = create_target(sample_df)
    assert y.iloc[1] == 1


def test_create_target_alto_risco(sample_df):
    y = create_target(sample_df)
    assert y.iloc[2] == 2


def test_create_target_boundaries():
    df = pd.DataFrame({"Defas": [1, 2, 0, -1, -2, -5]})
    y = create_target(df)
    assert list(y) == [0, 0, 0, 1, 2, 2]


def test_get_features_returns_correct_columns(sample_df):
    X = get_features(sample_df)
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(X.columns) == expected


def test_get_features_excludes_target(sample_df):
    X = get_features(sample_df)
    assert TARGET_COL not in X.columns


def test_get_features_handles_missing_columns():
    df = pd.DataFrame({"INDE 22": [7.0], "Defas": [0]})
    X = get_features(df)
    assert "INDE 22" in X.columns
    assert "Defas" not in X.columns


def test_load_data_excel(tmp_path):
    # cria Excel com sheet PEDE2022 (formato esperado)
    df = pd.DataFrame({
        "INDE 22": [7.0, 5.5], "IAA": [8.0, 6.0], "IEG": [7.0, 5.0],
        "IPS": [6.5, 5.5], "IDA": [7.0, 4.0], "IPV": [7.5, 5.0],
        "Matem": [6.5, 4.0], "Portug": [7.0, 5.0], "Fase": [3, 2],
        "Idade 22": [14, 13], "Gênero": ["Menino", "Menina"],
        "Pedra 22": ["Ametista", "Ágata"], "Defas": [0, -1],
    })
    path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="PEDE2022", index=False)
    loaded = load_data(str(path))
    assert len(loaded) == 2


def test_load_data_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    loaded = load_data(str(path))
    assert loaded.shape == (2, 2)


def _make_pede_df(n=3, year=2022):
    """Helper: cria DataFrame no formato canonico para testes."""
    d = {
        "INDE 22": np.random.uniform(4, 9, n),
        "IAA": np.random.uniform(3, 10, n),
        "IEG": np.random.uniform(3, 10, n),
        "IPS": np.random.uniform(3, 10, n),
        "IDA": np.random.uniform(3, 10, n),
        "IPV": np.random.uniform(3, 10, n),
        "Matem": np.random.uniform(2, 10, n),
        "Portug": np.random.uniform(2, 10, n),
        "Fase": ([1, 2, 3] * (n // 3 + 1))[:n],
        "Idade 22": ([12, 14, 16] * (n // 3 + 1))[:n],
        "Gênero": (["Menino", "Menina"] * (n // 2 + 1))[:n],
        "Pedra 22": (["Ametista", "Ágata", "Quartzo"] * (n // 3 + 1))[:n],
        "Defas": ([0, -1, -2] * (n // 3 + 1))[:n],
    }
    if year == 2022:
        return pd.DataFrame(d)
    # 2023 format: rename columns
    d23 = {k: v for k, v in d.items() if k not in ("INDE 22", "Pedra 22", "Idade 22", "Matem", "Portug", "Defas")}
    d23[f"INDE {year}"] = d["INDE 22"]
    d23[f"Pedra {year}"] = d["Pedra 22"]
    d23["Idade"] = d["Idade 22"]
    d23["Mat"] = d["Matem"]
    d23["Por"] = d["Portug"]
    d23["Defasagem"] = d["Defas"]
    d23["IPP"] = np.random.uniform(3, 10, n)
    d23["Gênero"] = (["Masculino", "Feminino"] * (n // 2 + 1))[:n]
    return pd.DataFrame(d23)


def test_load_data_multi_sheet_excel(tmp_path):
    df22 = _make_pede_df(3, 2022)
    df23 = _make_pede_df(3, 2023)
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as w:
        df22.to_excel(w, sheet_name="PEDE2022", index=False)
        df23.to_excel(w, sheet_name="PEDE2023", index=False)
    loaded = load_data(str(path))
    assert len(loaded) == 6
    assert "IPP" in loaded.columns
    # genero deve ser normalizado
    assert set(loaded["Gênero"].dropna().unique()).issubset({"Menino", "Menina"})


def test_harmonize_removes_incluir(tmp_path):
    df24 = _make_pede_df(3, 2023)
    df24[f"Pedra 2023"] = ["Ametista", "INCLUIR", "Quartzo"]
    path = tmp_path / "incluir.xlsx"
    with pd.ExcelWriter(path) as w:
        df24.to_excel(w, sheet_name="PEDE2023", index=False)
    loaded = load_data(str(path))
    assert len(loaded) == 2  # INCLUIR removido


def test_parse_fase_numeric():
    assert _parse_fase(3) == 3
    assert _parse_fase(0) == 0


def test_parse_fase_alfa():
    assert _parse_fase("ALFA") == 0
    assert _parse_fase("alfa") == 0


def test_parse_fase_text():
    assert _parse_fase("FASE 3") == 3
    assert _parse_fase("FASE 8") == 8


def test_parse_fase_letter_suffix():
    assert _parse_fase("1A") == 1
    assert _parse_fase("3G") == 3
    assert _parse_fase("7E") == 7


def test_parse_fase_nan():
    assert pd.isna(_parse_fase(float("nan")))


def test_prepare_data_returns_X_y(tmp_path):
    df = _make_pede_df(6, 2022)
    path = tmp_path / "prep.xlsx"
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="PEDE2022", index=False)
    X, y = prepare_data(str(path))
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1, 2})
    assert TARGET_COL not in X.columns
