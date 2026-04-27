"""
SKILL: Streamlit DataFrames para Agronomia
Tabelas interativas com filtros, cores e exportação
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO


def df_interativo_agro(
    df,
    colunas_exibir=None,
    coluna_score="Score_Aptidao",
    permite_export=True,
    permite_filtro=True,
    itens_por_pagina=50,
    coluna_municipio="Municipio",
    coluna_uf="UF",
    ordenar_por=None
):
    """
    Exibe DataFrame interativo com filtros, ordenação e exportação
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com dados
    colunas_exibir : list
        Colunas a mostrar (None = todas)
    coluna_score : str
        Coluna com scores para colorir
    permite_export : bool
        Mostrar botões de download
    permite_filtro : bool
        Mostrar filtros dinâmicos
    itens_por_pagina : int
        Itens por página
    coluna_municipio : str
        Coluna com nome do município
    coluna_uf : str
        Coluna com UF
    ordenar_por : str
        Coluna para ordenação inicial
    """
    
    if df is None or df.empty:
        st.warning("DataFrame vazio ou inválido")
        return
    
    # Definir colunas a exibir
    if colunas_exibir is None:
        colunas_exibir = df.columns.tolist()
    
    df_display = df[colunas_exibir].copy()
    
    # Filtros
    if permite_filtro:
        st.subheader("🔍 Filtros")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if coluna_municipio in df.columns:
                municipio_filter = st.multiselect(
                    "Município",
                    options=sorted(df[coluna_municipio].unique()),
                    max_selections=10,
                    label_visibility="collapsed"
                )
                if municipio_filter:
                    df_display = df_display[df[coluna_municipio].isin(municipio_filter)]
        
        with col2:
            if coluna_uf in df.columns:
                uf_filter = st.multiselect(
                    "UF",
                    options=sorted(df[coluna_uf].unique()),
                    label_visibility="collapsed"
                )
                if uf_filter:
                    df_display = df_display[df[coluna_uf].isin(uf_filter)]
        
        with col3:
            if coluna_score in df.columns:
                score_range = st.slider(
                    f"Min. {coluna_score}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    label_visibility="collapsed"
                )
                df_display = df_display[df[coluna_score] >= score_range]
    
    # Ordenação
    if ordenar_por and ordenar_por in df_display.columns:
        df_display = df_display.sort_values(ordenar_por, ascending=False)
    elif coluna_score in df_display.columns:
        df_display = df_display.sort_values(coluna_score, ascending=False)
    
    # Paginação
    total_linhas = len(df_display)
    num_paginas = (total_linhas + itens_por_pagina - 1) // itens_por_pagina
    
    if num_paginas > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            pagina = st.selectbox(
                "Página",
                range(1, num_paginas + 1),
                index=0,
                label_visibility="collapsed"
            )
    else:
        pagina = 1
    
    inicio = (pagina - 1) * itens_por_pagina
    fim = inicio + itens_por_pagina
    df_pagina = df_display.iloc[inicio:fim]
    
    # Estilo condicional para scores
    def colorir_score(val):
        if isinstance(val, (int, float)):
            if val > 0.7:
                return f"background-color: #90EE90; color: black"
            elif val > 0.4:
                return f"background-color: #FFD700; color: black"
            else:
                return f"background-color: #FFB6C6; color: black"
        return ""
    
    # Exibir tabela
    st.subheader(f"📊 Resultados ({total_linhas} municípios)")
    
    if coluna_score in df_pagina.columns:
        styled_df = df_pagina.style.applymap(
            colorir_score,
            subset=[coluna_score]
        ).format(precision=2)
        st.dataframe(styled_df, use_container_width=True, height=500)
    else:
        st.dataframe(df_pagina, use_container_width=True, height=500)
    
    # Exportação
    if permite_export:
        st.subheader("📥 Exportar Dados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📄 CSV",
                data=csv,
                file_name="municipios_aptos.csv",
                mime="text/csv"
            )
        
        with col2:
            buffer = BytesIO()
            df_display.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                label="📋 Excel",
                data=buffer,
                file_name="municipios_aptos.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        with col3:
            json_str = df_display.to_json(orient='records', indent=2)
            st.download_button(
                label="🔗 JSON",
                data=json_str,
                file_name="municipios_aptos.json",
                mime="application/json"
            )
    
    # Resumo
    st.write(f"**Total exibido:** {len(df_pagina)} / {total_linhas} municípios")


def filtro_por_intervalo(
    df,
    coluna,
    label,
    min_val=None,
    max_val=None
):
    """
    Cria slider para filtrar por range
    
    Retorno:
    --------
    pd.DataFrame
        DataFrame filtrado
    """
    
    if min_val is None:
        min_val = df[coluna].min()
    if max_val is None:
        max_val = df[coluna].max()
    
    range_val = st.slider(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        value=(float(min_val), float(max_val)),
        step=(max_val - min_val) / 100
    )
    
    return df[(df[coluna] >= range_val[0]) & (df[coluna] <= range_val[1])]


def export_csv_excel(
    df,
    nome_arquivo="dados",
    coluna1=False
):
    """
    Cria botões de download CSV e Excel
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame a exportar
    nome_arquivo : str
        Nome base do arquivo
    coluna1 : bool
        Usar primeira coluna em layout lado-a-lado
    """
    
    csv = df.to_csv(index=False)
    
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    
    if coluna1:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Baixar CSV",
                data=csv,
                file_name=f"{nome_arquivo}.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="📋 Baixar Excel",
                data=buffer,
                file_name=f"{nome_arquivo}.xlsx",
                mime="application/vnd.ms-excel"
            )
    else:
        st.download_button(
            label="📄 CSV",
            data=csv,
            file_name=f"{nome_arquivo}.csv",
            mime="text/csv"
        )
        st.download_button(
            label="📋 Excel",
            data=buffer,
            file_name=f"{nome_arquivo}.xlsx",
            mime="application/vnd.ms-excel"
        )
