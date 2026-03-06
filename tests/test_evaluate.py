import pytest

from src.evaluate import RISK_LABELS, evaluate_model


@pytest.fixture
def perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    return y_true, y_pred


def test_evaluate_model_returns_all_keys(perfect_predictions):
    y_true, y_pred = perfect_predictions
    metrics = evaluate_model(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics


def test_perfect_prediction_accuracy(perfect_predictions):
    y_true, y_pred = perfect_predictions
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_perfect_prediction_f1(perfect_predictions):
    y_true, y_pred = perfect_predictions
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_imperfect_prediction_accuracy_below_one(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["accuracy"] < 1.0


def test_confusion_matrix_is_list(perfect_predictions):
    y_true, y_pred = perfect_predictions
    metrics = evaluate_model(y_true, y_pred)
    assert isinstance(metrics["confusion_matrix"], list)


def test_risk_labels_complete():
    assert 0 in RISK_LABELS
    assert 1 in RISK_LABELS
    assert 2 in RISK_LABELS


def test_print_evaluation_runs_without_error(capsys, perfect_predictions):
    from src.evaluate import print_evaluation
    y_true, y_pred = perfect_predictions
    metrics = evaluate_model(y_true, y_pred)
    print_evaluation(metrics)
    captured = capsys.readouterr()
    assert "Acurácia" in captured.out
    assert "F1-Score" in captured.out
