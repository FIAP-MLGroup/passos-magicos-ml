import json
import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"


def setup_logging(name: str = "passos-magicos") -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "app.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


def log_prediction(input_data: dict, prediction: dict) -> None:
    """Persiste cada predição em formato JSONL para monitoramento."""
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
    }
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prediction_logs() -> list:
    """Carrega todos os logs de predição."""
    if not PREDICTION_LOG.exists():
        return []
    records = []
    with open(PREDICTION_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
