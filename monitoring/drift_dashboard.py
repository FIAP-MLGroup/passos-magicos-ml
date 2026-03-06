"""
Dashboard de Monitoramento - Passos Mágicos
Visualiza estatísticas das predições e detecta desvio do modelo.

Execução:
    streamlit run monitoring/drift_dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_prediction_logs

RISK_LABELS = {0: "Sem Risco", 1: "Risco Médio", 2: "Alto Risco"}
RISK_COLORS = {"Sem Risco": "#2ECC71", "Risco Médio": "#F39C12", "Alto Risco": "#E74C3C"}

st.set_page_config(
    page_title="Passos Mágicos — Monitoramento",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Monitoramento do Modelo — Passos Mágicos")
st.markdown("Acompanhamento das predições de risco de defasagem escolar em tempo real.")

logs = load_prediction_logs()

if not logs:
    st.warning("Nenhuma predição registrada ainda. Realize chamadas ao endpoint `/predict` para visualizar os dados.")
    st.stop()

# Prepara DataFrame de logs
records = []
for log in logs:
    row = {
        "timestamp": pd.to_datetime(log["timestamp"]),
        "risk_class": log["prediction"]["risk_class"],
        "risk_label": log["prediction"]["risk_label"],
        **{f"prob_{k.replace(' ', '_')}": v for k, v in log["prediction"]["probabilities"].items()},
        **{k: v for k, v in log["input"].items() if isinstance(v, (int, float))},
    }
    records.append(row)

df = pd.DataFrame(records).sort_values("timestamp")

# ── Métricas gerais ──────────────────────────────────────────────────────────
st.subheader("Resumo Geral")
col1, col2, col3, col4 = st.columns(4)

total = len(df)
sem_risco = (df["risk_class"] == 0).sum()
risco_medio = (df["risk_class"] == 1).sum()
alto_risco = (df["risk_class"] == 2).sum()

col1.metric("Total de Predições", total)
col2.metric("Sem Risco", f"{sem_risco} ({sem_risco/total*100:.1f}%)")
col3.metric("Risco Médio", f"{risco_medio} ({risco_medio/total*100:.1f}%)")
col4.metric("Alto Risco", f"{alto_risco} ({alto_risco/total*100:.1f}%)", delta_color="inverse")

# ── Distribuição de predições ─────────────────────────────────────────────────
st.subheader("Distribuição de Classes Preditas")
col_a, col_b = st.columns(2)

with col_a:
    dist = df["risk_label"].value_counts().reset_index()
    dist.columns = ["Classe", "Quantidade"]
    fig_pie = px.pie(
        dist, names="Classe", values="Quantidade",
        color="Classe", color_discrete_map=RISK_COLORS,
        title="Proporção por Classe de Risco",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    fig_bar = px.bar(
        dist, x="Classe", y="Quantidade",
        color="Classe", color_discrete_map=RISK_COLORS,
        title="Contagem por Classe de Risco",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Predições ao longo do tempo ──────────────────────────────────────────────
st.subheader("Predições ao Longo do Tempo")
df["data"] = df["timestamp"].dt.date
time_series = df.groupby(["data", "risk_label"]).size().reset_index(name="count")

fig_time = px.line(
    time_series, x="data", y="count", color="risk_label",
    color_discrete_map=RISK_COLORS,
    title="Volume de Predições por Dia",
    markers=True,
)
st.plotly_chart(fig_time, use_container_width=True)

# ── Probabilidades médias (drift proxy) ──────────────────────────────────────
st.subheader("Probabilidades Médias — Indicador de Drift")
st.markdown(
    "Variações significativas nas probabilidades médias ao longo do tempo "
    "podem indicar **data drift** (mudança no perfil dos alunos)."
)

prob_cols = [c for c in df.columns if c.startswith("prob_")]
if prob_cols:
    prob_means = df[prob_cols].mean().reset_index()
    prob_means.columns = ["Classe", "Probabilidade Média"]
    prob_means["Classe"] = prob_means["Classe"].str.replace("prob_", "").str.replace("_", " ")
    fig_prob = px.bar(
        prob_means, x="Classe", y="Probabilidade Média",
        title="Probabilidade Média por Classe (todas as predições)",
        color="Classe", color_discrete_map=RISK_COLORS,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Features de entrada ──────────────────────────────────────────────────────
st.subheader("Distribuição das Features de Entrada")
numeric_input_cols = ["INDE_22", "IAA", "IEG", "IPS", "IPP", "IDA", "IPV", "Fase"]
available = [c for c in numeric_input_cols if c in df.columns]

if available:
    selected = st.selectbox("Selecione uma feature:", available)
    fig_hist = px.histogram(
        df, x=selected, color="risk_label",
        color_discrete_map=RISK_COLORS,
        nbins=20, title=f"Distribuição de {selected} por Classe de Risco",
        barmode="overlay", opacity=0.7,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tabela de logs recentes ──────────────────────────────────────────────────
with st.expander("Ver últimas 50 predições"):
    display_cols = ["timestamp", "risk_label"] + available
    st.dataframe(
        df[display_cols].tail(50).sort_values("timestamp", ascending=False),
        use_container_width=True,
    )
