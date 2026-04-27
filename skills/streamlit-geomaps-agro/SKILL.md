# SKILL: Streamlit Geomaps para Agronomia

## Descrição
Componentes Streamlit prontos para renderizar **mapas interativos** de aptidão agrícola com Folium e Pydeck. Marca municípios aptos, cria heatmaps de clima e exporta visualizações.

## Funções Principais

### `renderizar_mapa_aptidao()`
Cria mapa Folium com marcadores de municípios aptos

**Parâmetros:**
- `df`: DataFrame com dados de municípios
- `lat_col`: Nome da coluna com latitude
- `lon_col`: Nome da coluna com longitude
- `aptidao_col`: Nome da coluna com score de aptidão (0-1)
- `municipio_col`: Nome da coluna com nome do município
- `cor_apto`: Cor para municípios aptos (default: "green")
- `cor_candidato`: Cor para candidatos (default: "yellow")
- `raio`: Tamanho do marcador (default: 5)

**Retorno:**
- Objeto Folium Map

**Exemplo:**
```python
import streamlit as st
from streamlit_geomaps_agro import renderizar_mapa_aptidao

resultado = st.read_csv("resultado_zoneamento.csv")
mapa = renderizar_mapa_aptidao(
    df=resultado,
    lat_col="Latitude",
    lon_col="Longitude",
    aptidao_col="Score_Aptidao",
    municipio_col="Municipio"
)
st.folium_static(mapa, width=1200, height=600)
```

### `criar_heatmap_clima()`
Cria heatmap de temperatura ou precipitação

**Parâmetros:**
- `df`: DataFrame com dados
- `lat_col`, `lon_col`: Colunas geográficas
- `valor_col`: Coluna com valores (temperatura/precipitação)
- `colormap`: "RdYlBu" (temperatura) ou "Blues" (chuva)

**Exemplo:**
```python
heatmap = criar_heatmap_clima(
    df=resultado,
    lat_col="Latitude",
    lon_col="Longitude",
    valor_col="Temperatura",
    colormap="RdYlBu"
)
st.folium_static(heatmap)
```

### `adicionar_municipios_ao_mapa()`
Adiciona clusters de municípios ao mapa (performance)

**Parâmetros:**
- `mapa`: Objeto Folium Map
- `df`: DataFrame
- `lat_col`, `lon_col`: Colunas geográficas
- `municipio_col`: Coluna com nome

**Retorno:**
- Mapa com cluster layer adicionado

## Instalação no Streamlit
```python
import sys
sys.path.append('./skills/user/streamlit-geomaps-agro')
from geomaps_agro import renderizar_mapa_aptidao
```

## Requisitos
- streamlit
- folium
- streamlit-folium
- pandas
- numpy
