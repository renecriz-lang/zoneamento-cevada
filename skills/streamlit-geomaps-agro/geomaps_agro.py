"""
SKILL: Streamlit Geomaps para Agronomia
Renderiza mapas interativos de aptidão agrícola
"""

import folium
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster
import pandas as pd
import numpy as np


def renderizar_mapa_aptidao(
    df,
    lat_col="Latitude",
    lon_col="Longitude",
    aptidao_col="Score_Aptidao",
    municipio_col="Municipio",
    uf_col="UF",
    cor_apto="green",
    cor_candidato="yellow",
    cor_inapto="red",
    raio=5,
    centro_lat=-15,
    centro_lon=-55,
    zoom=4
):
    """
    Cria mapa Folium com marcadores de municípios por aptidão
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados dos municípios
    lat_col : str
        Nome da coluna com latitude
    lon_col : str
        Nome da coluna com longitude
    aptidao_col : str
        Nome da coluna com score de aptidão (0-1)
    municipio_col : str
        Nome da coluna com nome do município
    uf_col : str
        Nome da coluna com UF
    cor_apto : str
        Cor para municípios aptos (score > 0.7)
    cor_candidato : str
        Cor para candidatos (0.4 < score <= 0.7)
    cor_inapto : str
        Cor para inaptos (score <= 0.4)
    raio : int
        Tamanho dos marcadores
    centro_lat : float
        Latitude inicial do mapa
    centro_lon : float
        Longitude inicial do mapa
    zoom : int
        Zoom inicial
    
    Retorno:
    --------
    folium.Map
        Mapa pronto pra usar em Streamlit
    """
    
    # Criar mapa base
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )
    
    # Remover linhas com NaN em colunas essenciais
    df_clean = df.dropna(subset=[lat_col, lon_col, aptidao_col])
    
    # Adicionar marcadores por aptidão
    for idx, row in df_clean.iterrows():
        # Determinar cor e classificação
        score = row[aptidao_col]
        
        if score > 0.7:
            cor = cor_apto
            classificacao = "✓ APTO"
        elif score > 0.4:
            cor = cor_candidato
            classificacao = "◐ CANDIDATO"
        else:
            cor = cor_inapto
            classificacao = "✗ INAPTO"
        
        # Criar popup
        municipio = row.get(municipio_col, "Desconhecido")
        uf = row.get(uf_col, "")
        popup_text = f"""
        <b>{municipio}, {uf}</b><br>
        Score: {score:.1%}<br>
        Status: {classificacao}
        """
        
        # Adicionar marcador
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=raio,
            popup=folium.Popup(popup_text, max_width=200),
            color=cor,
            fill=True,
            fillColor=cor,
            fillOpacity=0.7,
            weight=2
        ).add_to(mapa)
    
    # Adicionar legenda
    legenda_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0;"><b>Legenda de Aptidão</b></p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:green"></i> Apto (> 70%)</p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:yellow"></i> Candidato (40-70%)</p>
    <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:red"></i> Inapto (< 40%)</p>
    </div>
    '''
    mapa.get_root().html.add_child(folium.Element(legenda_html))
    
    return mapa


def criar_heatmap_clima(
    df,
    lat_col="Latitude",
    lon_col="Longitude",
    valor_col="Temperatura",
    colormap="RdYlBu",
    centro_lat=-15,
    centro_lon=-55,
    zoom=4,
    intensidade_min=None,
    intensidade_max=None
):
    """
    Cria heatmap de temperatura ou precipitação
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados
    lat_col, lon_col : str
        Colunas geográficas
    valor_col : str
        Coluna com valores numéricos
    colormap : str
        "RdYlBu" para temperatura, "Blues" para chuva
    intensidade_min, intensidade_max : float
        Limites de intensidade (auto se None)
    
    Retorno:
    --------
    folium.Map
        Mapa de calor
    """
    
    # Criar mapa
    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=zoom,
        tiles="OpenStreetMap"
    )
    
    # Preparar dados para heatmap
    df_clean = df.dropna(subset=[lat_col, lon_col, valor_col])
    
    heat_data = [
        [row[lat_col], row[lon_col], row[valor_col]]
        for idx, row in df_clean.iterrows()
    ]
    
    if not heat_data:
        return mapa
    
    # Normalizar intensidade
    valores = df_clean[valor_col].values
    v_min = intensidade_min or valores.min()
    v_max = intensidade_max or valores.max()
    
    heat_data_norm = [
        [lat, lon, (val - v_min) / (v_max - v_min) if v_max > v_min else 0.5]
        for lat, lon, val in heat_data
    ]
    
    # Adicionar heatmap
    HeatMap(
        heat_data_norm,
        min_opacity=0.3,
        max_zoom=18,
        radius=15,
        blur=15,
        gradient={
            0.0: 'blue',
            0.5: 'yellow',
            1.0: 'red'
        }
    ).add_to(mapa)
    
    return mapa


def adicionar_municipios_ao_mapa(
    mapa,
    df,
    lat_col="Latitude",
    lon_col="Longitude",
    municipio_col="Municipio",
    uf_col="UF",
    usar_cluster=True
):
    """
    Adiciona marcadores de municípios ao mapa com clustering
    
    Parâmetros:
    -----------
    mapa : folium.Map
        Mapa base
    df : pd.DataFrame
        DataFrame com municípios
    usar_cluster : bool
        Usar MarkerCluster para performance
    
    Retorno:
    --------
    folium.Map
        Mapa com marcadores adicionados
    """
    
    df_clean = df.dropna(subset=[lat_col, lon_col])
    
    if usar_cluster:
        cluster = MarkerCluster().add_to(mapa)
        
        for idx, row in df_clean.iterrows():
            municipio = row.get(municipio_col, "Desconhecido")
            uf = row.get(uf_col, "")
            
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f"{municipio}, {uf}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(cluster)
    else:
        for idx, row in df_clean.iterrows():
            municipio = row.get(municipio_col, "Desconhecido")
            uf = row.get(uf_col, "")
            
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f"{municipio}, {uf}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(mapa)
    
    return mapa


def exportar_mapa_html(mapa, caminho_saida="mapa_aptidao.html"):
    """
    Exporta mapa para arquivo HTML
    
    Parâmetros:
    -----------
    mapa : folium.Map
        Mapa Folium
    caminho_saida : str
        Caminho do arquivo HTML
    """
    mapa.save(caminho_saida)
    print(f"✓ Mapa exportado: {caminho_saida}")
