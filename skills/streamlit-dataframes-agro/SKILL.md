# SKILL: Streamlit DataFrames para Agronomia

## Descrição
Componentes Streamlit para exibir **tabelas interativas** com filtros, cores condicionais, paginação e exportação de dados agrícolas.

## Funções Principais

### `df_interativo_agro()`
Exibe DataFrame com filtros, ordenação e exportação

**Parâmetros:**
- `df`: DataFrame com dados
- `colunas_exibir`: Lista de colunas a mostrar (default: todas)
- `coluna_score`: Nome da coluna de aptidão (para colorir)
- `permite_export`: Ativar download (default: True)
- `permite_filtro`: Ativar filtros (default: True)
- `itens_por_pagina`: Itens por página (default: 50)
- `coluna_municipio`: Nome da coluna com município
- `coluna_uf`: Nome da coluna com UF

**Retorno:**
- None (renderiza direto no Streamlit)

**Exemplo:**
```python
import streamlit as st
from streamlit_dataframes_agro import df_interativo_agro

resultado = st.read_csv("resultado_zoneamento.csv")

df_interativo_agro(
    df=resultado,
    colunas_exibir=["Municipio", "UF", "Score_Aptidao", "Temperatura"],
    coluna_score="Score_Aptidao",
    permite_export=True
)
```

### `filtro_por_intervalo()`
Cria slider para filtrar por range de valores

**Parâmetros:**
- `df`: DataFrame
- `coluna`: Nome da coluna numérica
- `label`: Label do filtro
- `min_val`, `max_val`: Limites

**Retorno:**
- DataFrame filtrado

**Exemplo:**
```python
resultado_filtrado = filtro_por_intervalo(
    df=resultado,
    coluna="Score_Aptidao",
    label="Aptidão mínima",
    min_val=0.0,
    max_val=1.0
)
```

### `export_csv_excel()`
Cria botões de download para CSV e Excel

**Parâmetros:**
- `df`: DataFrame
- `nome_arquivo`: Nome base do arquivo
- `coluna1`: Usar 2 colunas para download side-by-side

**Exemplo:**
```python
import streamlit as st
from streamlit_dataframes_agro import export_csv_excel

resultado = st.read_csv("resultado.csv")

col1, col2 = st.columns(2)
with col1:
    export_csv_excel(resultado, "municipios_aptos", coluna1=True)
with col2:
    st.write("Formatos disponíveis ↑")
```

## Instalação no Streamlit
```python
import sys
sys.path.append('./skills/user/streamlit-dataframes-agro')
from dataframes_agro import df_interativo_agro, export_csv_excel
```

## Requisitos
- streamlit
- pandas
- openpyxl (para Excel)
