import json

import pytest

from src.utils import load_prediction_logs, log_prediction


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    import src.utils as utils_module
    monkeypatch.setattr(utils_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(utils_module, "PREDICTION_LOG", tmp_path / "predictions.jsonl")
    yield tmp_path


def test_log_prediction_creates_file(tmp_logs):
    log_prediction({"Fase": 3}, {"risk_class": 1})
    assert (tmp_logs / "predictions.jsonl").exists()


def test_log_prediction_valid_json(tmp_logs):
    log_prediction({"Fase": 3}, {"risk_class": 1, "risk_label": "Risco Médio"})
    content = (tmp_logs / "predictions.jsonl").read_text()
    record = json.loads(content.strip())
    assert "timestamp" in record
    assert record["input"]["Fase"] == 3
    assert record["prediction"]["risk_class"] == 1


def test_log_prediction_appends(tmp_logs):
    log_prediction({"Fase": 1}, {"risk_class": 0})
    log_prediction({"Fase": 2}, {"risk_class": 2})
    lines = (tmp_logs / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_load_prediction_logs_empty(tmp_logs):
    logs = load_prediction_logs()
    assert logs == []


def test_load_prediction_logs_returns_records(tmp_logs):
    log_prediction({"Fase": 3}, {"risk_class": 1})
    logs = load_prediction_logs()
    assert len(logs) == 1
    assert logs[0]["input"]["Fase"] == 3
