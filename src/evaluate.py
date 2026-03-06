from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

RISK_LABELS = {0: "Sem Risco", 1: "Risco Médio", 2: "Alto Risco"}


def evaluate_model(y_true, y_pred) -> Dict[str, Any]:
    target_names = [RISK_LABELS[i] for i in sorted(RISK_LABELS)]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_evaluation(metrics: Dict[str, Any]) -> None:
    print(f"Acurácia       : {metrics['accuracy']:.4f}")
    print(f"F1-Score Macro : {metrics['f1_macro']:.4f}")
    print(f"F1-Score Weighted: {metrics['f1_weighted']:.4f}")
    print("\nRelatório por classe:")
    for label, vals in metrics["classification_report"].items():
        if isinstance(vals, dict):
            print(
                f"  {label:12s} → precision={vals['precision']:.3f} "
                f"recall={vals['recall']:.3f} f1={vals['f1-score']:.3f}"
            )
