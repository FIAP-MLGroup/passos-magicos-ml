from typing import Optional

import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.utils import log_prediction

router = APIRouter()

RISK_LABELS = {0: "Sem Risco", 1: "Risco Médio", 2: "Alto Risco"}

RECOMMENDATIONS = {
    0: "Aluno no caminho certo. Manter acompanhamento regular.",
    1: "Atenção necessária. Recomenda-se reforço e acompanhamento próximo.",
    2: "Intervenção urgente recomendada. Avaliar suporte individualizado.",
}


class StudentInput(BaseModel):
    INDE_22: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional 2022")
    IAA: float = Field(..., ge=0, le=10, description="Índice de Auto Avaliação")
    IEG: float = Field(..., ge=0, le=10, description="Índice de Engajamento")
    IPS: float = Field(..., ge=0, le=10, description="Índice Psicossocial")
    IPP: Optional[float] = Field(None, ge=0, le=10, description="Índice Psicossocial Participativo (disponível a partir de 2023)")
    IDA: float = Field(..., ge=0, le=10, description="Índice de Desempenho Acadêmico")
    IPV: float = Field(..., ge=0, le=10, description="Índice de Ponto de Virada")
    Matem: Optional[float] = Field(None, ge=0, le=10, description="Nota de Matemática")
    Portug: Optional[float] = Field(None, ge=0, le=10, description="Nota de Português")
    Fase: int = Field(..., ge=0, le=8, description="Fase atual do aluno (0–8)")
    Genero: str = Field(..., description="'Menino' ou 'Menina'")
    Pedra_22: str = Field(..., description="'Ametista', 'Ágata', 'Quartzo' ou 'Topázio'")
    Atingiu_PV: str = Field(..., description="'Sim' ou 'Não'")
    Indicado: str = Field(..., description="'Sim' ou 'Não'")
    Idade_22: int = Field(..., ge=5, le=25, description="Idade do aluno em 2022")

    model_config = {"json_schema_extra": {
        "example": {
            "INDE_22": 7.0,
            "IAA": 6.5,
            "IEG": 5.0,
            "IPS": 6.0,
            "IDA": 5.5,
            "IPV": 7.0,
            "Matem": 6.0,
            "Portug": 5.5,
            "Fase": 3,
            "Genero": "Menino",
            "Pedra_22": "Ametista",
            "Atingiu_PV": "Não",
            "Indicado": "Não",
            "Idade_22": 14,
        }
    }}


class PredictionOutput(BaseModel):
    risk_class: int
    risk_label: str
    probabilities: dict
    recommendation: str


@router.post("/predict", response_model=PredictionOutput, tags=["Predição"])
def predict(student: StudentInput, request: Request):
    """
    Recebe os dados de um aluno e retorna a predição de risco de defasagem escolar.

    - **risk_class**: 0 = Sem Risco | 1 = Risco Médio | 2 = Alto Risco
    - **probabilities**: probabilidade estimada para cada classe
    - **recommendation**: ação recomendada com base no risco
    """
    model = request.app.state.model

    input_dict = {
        "INDE 22": student.INDE_22,
        "IAA": student.IAA,
        "IEG": student.IEG,
        "IPS": student.IPS,
        "IPP": student.IPP,
        "IDA": student.IDA,
        "IPV": student.IPV,
        "Matem": student.Matem,
        "Portug": student.Portug,
        "Fase": student.Fase,
        "Gênero": student.Genero,
        "Pedra 22": student.Pedra_22,
        "Atingiu PV": student.Atingiu_PV,
        "Indicado": student.Indicado,
        "Idade 22": student.Idade_22,
    }

    df = pd.DataFrame([input_dict])
    risk_class = int(model.predict(df)[0])
    probas = model.predict_proba(df)[0]

    result = PredictionOutput(
        risk_class=risk_class,
        risk_label=RISK_LABELS[risk_class],
        probabilities={
            RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(probas)
        },
        recommendation=RECOMMENDATIONS[risk_class],
    )

    log_prediction(student.model_dump(), result.model_dump())
    return result
