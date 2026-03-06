import re

import numpy as np
import pandas as pd
from pathlib import Path

NUMERIC_FEATURES = [
    "INDE 22", "IAA", "IEG", "IPS", "IPP", "IDA", "IPV",
    "Matem", "Portug", "Fase", "Idade 22",
]

CATEGORICAL_FEATURES = ["Gênero", "Pedra 22"]
TARGET_COL = "Defas"

_GENERO_MAP = {"Masculino": "Menino", "Feminino": "Menina"}
_PEDRA_MAP = {"Agata": "Ágata"}


def _harmonize_sheet(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Normaliza colunas de cada planilha para o formato canonico (base 2022)."""
    df = df.copy()

    if year == 2022:
        df["IPP"] = np.nan
        return df

    df = df.drop(columns=[c for c in ["INDE 22", "Pedra 22"] if c in df.columns])

    df = df.rename(columns={
        f"INDE {year}": "INDE 22",
        f"Pedra {year}": "Pedra 22",
        "Idade": "Idade 22",
        "Mat": "Matem",
        "Por": "Portug",
        "Defasagem": "Defas",
        "Fase Ideal": "Fase ideal",
    })

    df["Gênero"] = df["Gênero"].map(lambda x: _GENERO_MAP.get(x, x))

    if "Pedra 22" in df.columns:
        df["Pedra 22"] = df["Pedra 22"].map(lambda x: _PEDRA_MAP.get(x, x))
        df = df[df["Pedra 22"] != "INCLUIR"]

    return df


def _parse_fase(val):
    """Converte valores de Fase para inteiro (suporta 'ALFA', 'FASE 3', '3G' etc)."""
    if pd.isna(val):
        return val
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().upper()
    if s == "ALFA":
        return 0
    if s.startswith("FASE "):
        try:
            return int(s.split()[1])
        except (ValueError, IndexError):
            pass
    m = re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if path.suffix not in (".xlsx", ".xls"):
        return pd.read_csv(filepath)

    xl = pd.ExcelFile(filepath)
    frames = []
    for sheet in xl.sheet_names:
        try:
            year = int("".join(filter(str.isdigit, sheet)))
        except ValueError:
            continue
        df = pd.read_excel(xl, sheet_name=sheet)
        frames.append(_harmonize_sheet(df, year))

    combined = pd.concat(frames, ignore_index=True)
    combined["Idade 22"] = pd.to_numeric(combined["Idade 22"], errors="coerce")
    combined["Fase"] = combined["Fase"].apply(_parse_fase)
    return combined

def create_target(df: pd.DataFrame) -> pd.Series:
    def classify(defas):
        if defas >= 0:
            return 0
        elif defas == -1:
            return 1
        return 2

    return df[TARGET_COL].apply(classify)


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available = [f for f in all_features if f in df.columns]
    return df[available].copy()


def prepare_data(filepath: str):
    """Carrega dados de todas as planilhas e retorna X (features) e y (target)."""
    df = load_data(filepath)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, "Idade 22"])
    return get_features(df), create_target(df)
