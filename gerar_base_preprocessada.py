"""
Pré-processamento: transforma o parquet bruto (128 MB, long-format)
em bases precomputadas compactas (~3 MB cada) para o app Streamlit.

Execute UMA VEZ antes de subir o app ao Streamlit Cloud:
    python gerar_base_preprocessada.py

Saída:
    plataforma_agro_inteligente/data/Base_Clima_media_geral.parquet
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_PARQUET = os.path.join(ROOT, "NOVO APP", "DADOS_Clima_alt_solos_nino",
                           "DADOS_Clima_alt_solos_nino.parquet")
COORDS_FILE = os.path.join(ROOT, "municipios_coords.parquet")
OUT_DIR     = os.path.join(ROOT, "plataforma_agro_inteligente", "data")

# ── Modos disponíveis ──────────────────────────────────────────────────────
MODES = {
    "media_geral": {"ano_ini": 2010, "ano_fim": 2025},
    # Futuros modos serão adicionados aqui
}


def gerar_base(mode_key: str, ano_ini: int, ano_fim: int) -> None:
    out_file = os.path.join(OUT_DIR, f"Base_Clima_{mode_key}.parquet")
    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} já existe. Delete para regenerar.")
        return

    print(f"[{mode_key}] Carregando parquet bruto ({ano_ini}–{ano_fim})…")
    df = pd.read_parquet(RAW_PARQUET, columns=[
        "codigo_ibge", "nome", "estado", "ano", "decendio",
        "prec_media", "tmax_media", "tmed_media", "tmin_media",
        "altitude_media", "solo_1_ordem",
    ])

    df = df[(df["ano"] >= ano_ini) & (df["ano"] <= ano_fim)].copy()
    print(f"  Linhas após filtro de anos: {len(df):,}")

    # Média histórica por município × decêndio
    print("  Agregando por município × decêndio…")
    agg = (
        df.groupby(["codigo_ibge", "decendio"], observed=True)
        .agg(
            prec=("prec_media", "mean"),
            tmax=("tmax_media", "mean"),
            tmed=("tmed_media", "mean"),
            tmin=("tmin_media", "mean"),
        )
        .reset_index()
    )

    # Pivot para wide: colunas Prec_D1..D36, Tmax_D1..D36, etc.
    print("  Pivotando para formato wide…")
    def pivot_var(var_col, prefix):
        wide = agg.pivot(index="codigo_ibge", columns="decendio", values=var_col)
        wide.columns = [f"{prefix}_D{int(c)}" for c in wide.columns]
        return wide

    prec_w = pivot_var("prec", "Prec")
    tmax_w = pivot_var("tmax", "Tmax")
    tmed_w = pivot_var("tmed", "Tmed")
    tmin_w = pivot_var("tmin", "Tmin")

    wide = pd.concat([prec_w, tmax_w, tmed_w, tmin_w], axis=1).reset_index()

    # Campos estáticos: nome, estado, altitude, solo dominante
    static = (
        df.groupby("codigo_ibge", observed=True)
        .agg(
            nome=("nome", "first"),
            estado=("estado", "first"),
            altitude_media=("altitude_media", "mean"),
            solo_1_ordem=("solo_1_ordem", lambda x: x.dropna().iloc[0]
                          if x.notna().any() else None),
        )
        .reset_index()
    )

    result = static.merge(wide, on="codigo_ibge", how="left")

    # Merge lat/lon
    coords = pd.read_parquet(COORDS_FILE)
    coords["CD_MUN"] = coords["CD_MUN"].astype(str)
    result["codigo_ibge"] = result["codigo_ibge"].astype(str)
    result = result.merge(coords, left_on="codigo_ibge", right_on="CD_MUN", how="left")
    result.drop(columns=["CD_MUN"], inplace=True, errors="ignore")

    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_parquet(out_file, index=False)

    mb = os.path.getsize(out_file) / 1_048_576
    print(f"  Salvo em {out_file}  ({result.shape[0]:,} municípios × {result.shape[1]} colunas, {mb:.1f} MB)")


if __name__ == "__main__":
    for key, params in MODES.items():
        gerar_base(key, params["ano_ini"], params["ano_fim"])
    print("\nPré-processamento concluído.")
