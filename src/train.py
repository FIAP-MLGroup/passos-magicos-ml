import sys
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.evaluate import evaluate_model, print_evaluation
from src.feature_engineering import FeatureEngineer
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_data,
)
from src.utils import setup_logging

MODEL_PATH = Path("models/model.joblib")
logger = setup_logging("train")

ENGINEERED_NUMERIC = NUMERIC_FEATURES + ["Media_Academica", "Media_Notas"]


def build_pipeline(classifier) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ),
        ),
    ])
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, ENGINEERED_NUMERIC),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train(data_path: str):
    logger.info(f"Carregando dados de: {data_path}")
    X, y = prepare_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    best_pipeline = None
    best_score = 0.0
    best_name = ""

    for name, clf in candidates.items():
        pipeline = build_pipeline(clf)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="f1_macro"
        )
        mean_f1 = scores.mean()
        logger.info(
            f"{name}: F1-macro CV = {mean_f1:.4f} (+/- {scores.std():.4f})"
        )
        if mean_f1 > best_score:
            best_score = mean_f1
            best_pipeline = pipeline
            best_name = name

    logger.info(f"\nMelhor modelo: {best_name} (F1-macro CV: {best_score:.4f})")

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    logger.info("=== Métricas no conjunto de teste ===")
    print_evaluation(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info(f"Modelo salvo em: {MODEL_PATH}")

    return best_pipeline, metrics


if __name__ == "__main__":
    data_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    )
    train(data_path)
