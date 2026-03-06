# Passos Mágicos — Predição de Risco de Defasagem Escolar

Sistema de Machine Learning para predição do risco de defasagem escolar de alunos da ONG Passos Mágicos, desenvolvido como trabalho de conclusão da pós-graduação em Machine Learning Engineering (FIAP).

---

## Visão Geral

A ONG **Passos Mágicos** atende crianças e jovens em situação de vulnerabilidade social por meio de uma metodologia educacional própria, dividida em **Fases** progressivas. O projeto desenvolve um modelo preditivo capaz de classificar o **risco de defasagem escolar** de cada aluno em três categorias:

| Classe | Risco | Defasagem |
|--------|-------|-----------|
| 0 | Sem Risco | Fase atual ≥ Fase ideal |
| 1 | Risco Médio | 1 fase abaixo do esperado |
| 2 | Alto Risco | 2 ou mais fases abaixo |

**Dataset:** 860 alunos com dados educacionais de 2022 (PEDE 2024 Datathon).

---

## Estrutura do Projeto

```
passos-magicos-ml/
├── app/
│   ├── main.py                  # Aplicação FastAPI (lifespan, configuração)
│   └── routes.py                # Endpoint /predict com validação Pydantic
├── src/
│   ├── preprocessing.py         # Carregamento e preparação dos dados
│   ├── feature_engineering.py   # Engenharia de features (FeatureEngineer sklearn)
│   ├── train.py                 # Pipeline de treinamento com seleção de modelos
│   ├── evaluate.py              # Métricas e relatório de avaliação
│   └── utils.py                 # Logging e persistência de predições
├── tests/
│   ├── test_preprocessing.py    # Testes unitários do módulo de pré-processamento
│   ├── test_feature_engineering.py  # Testes da engenharia de features
│   ├── test_evaluate.py         # Testes das métricas de avaliação
│   ├── test_api.py              # Testes dos endpoints da API
│   └── test_utils.py            # Testes de logging e utilitários
├── monitoring/
│   └── drift_dashboard.py       # Dashboard Streamlit para monitoramento de drift
├── models/
│   └── model.joblib             # Modelo treinado (GradientBoosting)
├── data/                        # Dados brutos (não versionados)
├── logs/                        # Logs de predição em runtime
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

---

## Pipeline de Machine Learning

### 1. Pré-processamento (`src/preprocessing.py`)
- Carregamento do dataset Excel/CSV
- Criação da variável target (3 classes de risco)
- Seleção das features relevantes
- **Nota:** `IAN` foi excluído pois é derivado diretamente de `Defas` (data leakage)

### 2. Engenharia de Features (`src/feature_engineering.py`)
- `Media_Academica`: média de IAA, IEG e IDA
- `Media_Notas`: média de Matemática e Português
- Implementado como `FeatureEngineer` (compatível com sklearn Pipeline)

### 3. Treinamento (`src/train.py`)
- Três algoritmos avaliados por **Cross-Validation 5-fold (F1-macro)**:
  - `RandomForestClassifier` (balanced class weights)
  - `GradientBoostingClassifier`
  - `LogisticRegression` (balanced class weights)
- Melhor modelo salvo automaticamente em `models/model.joblib`

### 4. Resultados do Modelo (GradientBoosting)

| Métrica | Valor |
|---------|-------|
| F1-Score Macro (CV) | 0.9745 |
| Acurácia (teste) | 0.9826 |
| F1-Score Macro (teste) | 0.9841 |
| F1-Score Weighted (teste) | 0.9825 |

**Justificativa da métrica:** F1-Score Macro foi escolhido por tratar todas as classes igualmente, sendo robusto ao desbalanceamento presente na base (71% dos alunos estão defasados). É a métrica mais adequada para priorizar a detecção correta do Alto Risco.

---

## Como Executar

### Pré-requisitos
- Docker e Docker Compose instalados
- Python 3.11+ (apenas para treinamento local)

### 1. Treinamento do Modelo (executar uma vez)

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar com os dados
python -m src.train data/BASE\ DE\ DADOS\ PEDE\ 2024\ -\ DATATHON.xlsx
```

O modelo será salvo automaticamente em `models/model.joblib`.

### 2. Subir com Docker Compose

```bash
docker-compose up --build
```

Serviços disponíveis:
- **API:** http://localhost:8000
- **Documentação interativa:** http://localhost:8000/docs
- **Dashboard de monitoramento:** http://localhost:8501

### 3. Parar os serviços

```bash
docker-compose down
```

---

## Exemplos de Chamadas à API

### Health Check

```bash
curl http://localhost:8000/health
```

Resposta:
```json
{"status": "ok", "model_loaded": true}
```

### Predição de Risco

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "INDE_22": 6.5,
    "IAA": 6.0,
    "IEG": 5.0,
    "IPS": 5.5,
    "IDA": 5.0,
    "IPV": 6.5,
    "Matem": 5.0,
    "Portug": 5.5,
    "Fase": 3,
    "Genero": "Menino",
    "Pedra_22": "Ametista",
    "Atingiu_PV": "Não",
    "Indicado": "Não",
    "Idade_22": 15
  }'
```

Resposta:
```json
{
  "risk_class": 1,
  "risk_label": "Risco Médio",
  "probabilities": {
    "Sem Risco": 0.1234,
    "Risco Médio": 0.6543,
    "Alto Risco": 0.2223
  },
  "recommendation": "Atenção necessária. Recomenda-se reforço e acompanhamento próximo."
}
```

### Schema completo dos campos de entrada

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| INDE_22 | float [0-10] | Sim | Índice de Desenvolvimento Educacional |
| IAA | float [0-10] | Sim | Índice de Auto Avaliação |
| IEG | float [0-10] | Sim | Índice de Engajamento |
| IPS | float [0-10] | Sim | Índice Psicossocial |
| IDA | float [0-10] | Sim | Índice de Desempenho Acadêmico |
| IPV | float [0-10] | Sim | Índice de Ponto de Virada |
| Matem | float [0-10] | Não | Nota de Matemática |
| Portug | float [0-10] | Não | Nota de Português |
| Fase | int [0-8] | Sim | Fase atual do aluno |
| Genero | string | Sim | "Menino" ou "Menina" |
| Pedra_22 | string | Sim | "Ametista", "Ágata", "Quartzo" ou "Topázio" |
| Atingiu_PV | string | Sim | "Sim" ou "Não" |
| Indicado | string | Sim | "Sim" ou "Não" |
| Idade_22 | int [5-25] | Sim | Idade do aluno em 2022 |

---

## Testes

```bash
# Executar todos os testes com relatório de cobertura
pytest

# Relatório HTML de cobertura
pytest --cov-report=html
open htmlcov/index.html
```

Cobertura mínima configurada: **80%**

---

## Monitoramento

O dashboard de monitoramento (http://localhost:8501) exibe:

- **Distribuição de classes** preditas ao longo do tempo
- **Volume de predições** por dia
- **Probabilidades médias** por classe (indicador de drift)
- **Distribuição das features** de entrada segmentadas por risco
- **Tabela de predições** recentes

Todas as predições são registradas em `logs/predictions.jsonl` em formato JSONL para auditoria e análise posterior.

---

## Tecnologias Utilizadas

| Componente | Tecnologia |
|-----------|-----------|
| API | FastAPI 0.115 + Uvicorn |
| Modelo | scikit-learn (GradientBoosting) |
| Serialização | joblib |
| Dados | pandas + openpyxl |
| Testes | pytest + pytest-cov |
| Monitoramento | Streamlit + Plotly |
| Deploy | Docker + Docker Compose |

---

## Sobre o Projeto

**ONG Passos Mágicos** transforma a vida de crianças e jovens de baixa renda por meio da educação, oferecendo suporte psicológico, apoio social e desenvolvimento educacional de qualidade.

Este sistema foi desenvolvido como parte do **Datathon FIAP 2024** com o objetivo de apoiar a equipe da ONG na identificação precoce de alunos em risco de defasagem escolar, permitindo intervenções mais direcionadas e eficientes.
