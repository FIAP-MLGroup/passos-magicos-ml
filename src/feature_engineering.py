import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

PEDRA_ORDER = {"Ametista": 1, "Ágata": 2, "Quartzo": 3, "Topázio": 4}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        academic = [c for c in ["IAA", "IEG", "IDA"] if c in X.columns]
        if academic:
            X["Media_Academica"] = X[academic].mean(axis=1)

        grades = [c for c in ["Matem", "Portug"] if c in X.columns]
        if grades:
            X["Media_Notas"] = X[grades].mean(axis=1)

        return X


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer()
    return fe.transform(df)
