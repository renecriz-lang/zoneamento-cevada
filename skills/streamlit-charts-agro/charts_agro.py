"""
SKILL: Streamlit Charts para Agronomia
Gráficos analíticos de aptidão agrícola com Plotly
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plotar_top_municipios(
    df,
    municipio_col="Municipio",
    uf_col="UF",
    score_col="Score_Aptidao",
    top_n=20,
    titulo="Top Municípios Aptos para Cevada"
):
    """
    Gráfico de barras com top municípios
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados
    municipio_col : str
        Coluna com nome do município
    uf_col : str
        Coluna com UF
    score_col : str
        Coluna com score de aptidão
    top_n : int
        Número de top municípios
    titulo : str
        Título do gráfico
    """
    
    df_top = df.nlargest(top_n, score_col).copy()
    df_top['Municipio_UF'] = df_top[municipio_col] + ", " + df_top[uf_col]
    
    fig = px.bar(
        df_top,
        x=score_col,
        y='Municipio_UF',
        orientation='h',
        title=titulo,
        labels={score_col: 'Score de Aptidão', 'Municipio_UF': ''},
        color=score_col,
        color_continuous_scale='RdYlGn',
        hover_data=[municipio_col, uf_col],
        height=600
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plotar_por_estado(
    df,
    uf_col="UF",
    score_col="Score_Aptidao",
    titulo="Distribuição de Aptidão por Estado"
):
    """
    Box plot de distribuição por estado
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados
    uf_col : str
        Coluna com UF
    score_col : str
        Coluna com score
    titulo : str
        Título
    """
    
    fig = px.box(
        df,
        x=uf_col,
        y=score_col,
        title=titulo,
        labels={uf_col: 'Estado', score_col: 'Score de Aptidão'},
        color=uf_col,
        height=500
    )
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def scatter_aptidao_3d(
    df,
    x_col="Altitude",
    y_col="Temperatura",
    z_col="Precipitacao",
    cor_col="Score_Aptidao",
    municipio_col="Municipio",
    titulo="Análise 3D de Aptidão"
):
    """
    Scatter plot 3D (Altitude × Temperatura × Precipitação)
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados
    x_col, y_col, z_col : str
        Colunas para eixos
    cor_col : str
        Coluna para cor
    municipio_col : str
        Coluna com nome do município
    titulo : str
        Título
    """
    
    df_clean = df.dropna(subset=[x_col, y_col, z_col, cor_col])
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df_clean[x_col],
        y=df_clean[y_col],
        z=df_clean[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df_clean[cor_col],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Score de Aptidão"),
            line=dict(width=0)
        ),
        text=df_clean[municipio_col],
        hovertemplate="<b>%{text}</b><br>" +
                      f"{x_col}: %{{x:.0f}}<br>" +
                      f"{y_col}: %{{y:.1f}}°C<br>" +
                      f"{z_col}: %{{z:.0f}}mm<br>" +
                      "<extra></extra>"
    )])
    
    fig.update_layout(
        title=titulo,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plotar_feature_importance(
    features,
    importances,
    titulo="Importância das Variáveis (Random Forest)",
    top_n=None
):
    """
    Gráfico de importância de features
    
    Parâmetros:
    -----------
    features : list
        Nomes das features
    importances : array-like
        Valores de importância
    titulo : str
        Título
    top_n : int
        Mostrar top N features (None = todas)
    """
    
    df_imp = pd.DataFrame({
        'Feature': features,
        'Importancia': importances
    }).sort_values('Importancia', ascending=False)
    
    if top_n:
        df_imp = df_imp.head(top_n)
    
    fig = px.bar(
        df_imp,
        x='Importancia',
        y='Feature',
        orientation='h',
        title=titulo,
        labels={'Importancia': 'Importância (%)', 'Feature': ''},
        color='Importancia',
        color_continuous_scale='Viridis',
        height=400
    )
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def calendario_plantio(
    df_decendios,
    municipio,
    variavel="precipitacao",
    ano=None
):
    """
    Visualiza janela de plantio por decêndio
    
    Parâmetros:
    -----------
    df_decendios : pd.DataFrame
        DataFrame com dados por decêndio
    municipio : str
        Nome do município
    variavel : str
        "precipitacao" ou "temperatura"
    ano : int
        Ano específico (None = todos)
    """
    
    # Filtrar dados do município
    df_mun = df_decendios[df_decendios['Municipio'] == municipio].copy()
    
    if df_mun.empty:
        st.error(f"Município '{municipio}' não encontrado")
        return
    
    if ano:
        df_mun = df_mun[df_mun['Ano'] == ano]
    
    # Criar decêndio label (1-36)
    df_mun['Decendio_Label'] = df_mun['Decendio_Numero'].astype(str)
    
    if variavel == "precipitacao":
        coluna_valor = "Precipitacao_Total_mm"
        titulo = f"Precipitação em {municipio} por Decêndio"
        ylim_low, ylim_high = 0, df_mun[coluna_valor].max() * 1.2
        cor = "Blues"
    else:
        coluna_valor = "Temperatura_Media_C"
        titulo = f"Temperatura em {municipio} por Decêndio"
        ylim_low, ylim_high = df_mun[coluna_valor].min() - 5, df_mun[coluna_valor].max() + 5
        cor = "Reds"
    
    fig = px.line(
        df_mun,
        x='Decendio_Label',
        y=coluna_valor,
        title=titulo,
        markers=True,
        color_discrete_sequence=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis_title='Decêndio',
        yaxis_title=coluna_valor,
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def histograma_scores(
    df,
    score_col="Score_Aptidao",
    titulo="Distribuição de Scores de Aptidão"
):
    """
    Histograma de distribuição de scores
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame
    score_col : str
        Coluna com scores
    titulo : str
        Título
    """
    
    fig = px.histogram(
        df,
        x=score_col,
        title=titulo,
        labels={score_col: 'Score de Aptidão'},
        nbins=50,
        color_discrete_sequence=['#2E7D32'],
        height=400
    )
    
    fig.update_layout(
        xaxis_title='Score de Aptidão',
        yaxis_title='Número de Municípios',
        hovermode='x',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
