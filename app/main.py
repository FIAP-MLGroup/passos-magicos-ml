from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from app.routes import router
from src.utils import setup_logging

MODEL_PATH = Path("models/model.joblib")
logger = setup_logging("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando modelo...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o treinamento primeiro: python -m src.train <dados>"
        )
    app.state.model = joblib.load(MODEL_PATH)
    logger.info("Modelo carregado com sucesso.")
    yield
    logger.info("Encerrando API.")


app = FastAPI(
    title="Passos Mágicos — API de Risco Escolar",
    description=(
        "Predição de risco de defasagem escolar para alunos da ONG Passos Mágicos. "
        "Classifica o risco em: Sem Risco, Risco Médio ou Alto Risco."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["Status"])
def health_check():
    """Verifica se a API e o modelo estão operacionais."""
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}
