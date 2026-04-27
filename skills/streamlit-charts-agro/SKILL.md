# SKILL: Streamlit Charts para Agronomia

## Descrição
Componentes Streamlit para exibir **gráficos analíticos** de aptidão agrícola com Plotly. Inclui top 20, distribuição por estado, calendário de plantio e feature importance.

## Funções Principais

### `plotar_top_municipios()`
Gráfico de barras com top municípios aptos

**Parâmetros:**
- `df`: DataFrame com dados
- `municipio_col`: Coluna com nome do município
- `score_col`: Coluna com score
- `top_n`: Número de top municípios (default: 20)
- `titulo`: Título do gráfico

**Exemplo:**
```python
import streamlit as st
from streamlit_charts_agro import plotar_top_municipios

resultado = st.read_csv("resultado_zoneamento.csv")
plotar_top_municipios(
    df=resultado,
    municipio_col="Municipio",
    score_col="Score_Aptidao",
    top_n=20
)
```

### `plotar_por_estado()`
Box plot de distribuição por estado

**Parâmetros:**
- `df`: DataFrame
- `uf_col`: Coluna com UF
- `score_col`: Coluna com score
- `titulo`: Título

**Exemplo:**
```python
plotar_por_estado(
    df=resultado,
    uf_col="UF",
    score_col="Score_Aptidao"
)
```

### `scatter_aptidao_3d()`
Scatter plot 3D (Altitude × Temperatura × Precipitação)

**Parâmetros:**
- `df`: DataFrame
- `x_col`: Coluna X (Altitude)
- `y_col`: Coluna Y (Temperatura)
- `z_col`: Coluna Z (Precipitação)
- `cor_col`: Coluna para colorir (Score)

**Exemplo:**
```python
scatter_aptidao_3d(
    df=resultado,
    x_col="Altitude",
    y_col="Temperatura",
    z_col="Precipitacao",
    cor_col="Score_Aptidao"
)
```

### `plotar_feature_importance()`
Gráfico de importância de variáveis (Random Forest)

**Parâmetros:**
- `features`: Nomes das features
- `importances`: Valores de importância
- `titulo`: Título

**Exemplo:**
```python
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Após treinar RF
feature_names = ["Temperatura", "Precipitacao", "Altitude", "Solo"]
importances = modelo.feature_importances_

plotar_feature_importance(
    features=feature_names,
    importances=importances
)
```

### `calendario_plantio()`
Visualiza janela de plantio por decêndio

**Parâmetros:**
- `df_decendios`: DataFrame com dados por decêndio
- `municipio`: Nome do município
- `variavel`: "precipitacao" ou "temperatura"

**Exemplo:**
```python
calendario_plantio(
    df_decendios=precipitacao_municipios,
    municipio="Guarapuava",
    variavel="precipitacao"
)
```

## Instalação no Streamlit
```python
import sys
sys.path.append('./skills/user/streamlit-charts-agro')
from charts_agro import plotar_top_municipios, scatter_aptidao_3d
```

## Requisitos
- streamlit
- plotly
- pandas
- numpy
