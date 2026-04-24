"""
Zoneamento Agroclimático da Cevada — Streamlit App
Arquitetura: Processamento → Arquivo Temporário → Visualização
"""

import os
import csv
import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# CONTADOR DE ACESSOS
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acessos_log.csv")
_ADMIN_SENHA = "cevada2025"  # troque aqui para a senha que quiser


def _registrar_acesso() -> None:
    """Grava uma linha no log a cada nova sessão."""
    novo = not os.path.exists(_LOG_FILE)
    with open(_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["data_hora", "data", "hora"])
        agora = datetime.datetime.now()
        w.writerow([agora.strftime("%Y-%m-%d %H:%M:%S"),
                    agora.strftime("%Y-%m-%d"),
                    agora.strftime("%H:%M")])


def _painel_admin() -> None:
    """Painel de estatísticas de acesso — visível só com senha."""
    with st.sidebar.expander("⚙️ Config"):
        senha = st.text_input("Senha", type="password", key="_admin_pw")
        if senha != _ADMIN_SENHA:
            return
        if not os.path.exists(_LOG_FILE):
            st.info("Nenhum acesso registrado ainda.")
            return
        df_log = pd.read_csv(_LOG_FILE, parse_dates=["data_hora"])
        total = len(df_log)
        hoje = datetime.date.today().strftime("%Y-%m-%d")
        hoje_ct = (df_log["data"] == hoje).sum()
        st.success(f"Total de acessos: **{total}**")
        st.info(f"Hoje ({hoje}): **{hoje_ct}**")
        por_dia = (
            df_log.groupby("data")
            .size()
            .reset_index(name="acessos")
            .sort_values("data", ascending=False)
        )
        st.dataframe(por_dia, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — Estrutura dos Decêndios (ano não-bissexto)
# ─────────────────────────────────────────────────────────────────────────────
DECENDIO_DAYS = [10, 10, 11,   # Jan (D1–D3)
                 10, 10,  8,   # Fev (D4–D6)
                 10, 10, 11,   # Mar (D7–D9)
                 10, 10, 10,   # Abr (D10–D12)
                 10, 10, 11,   # Mai (D13–D15)
                 10, 10, 10,   # Jun (D16–D18)
                 10, 10, 11,   # Jul (D19–D21)
                 10, 10, 11,   # Ago (D22–D24)
                 10, 10, 10,   # Set (D25–D27)
                 10, 10, 11,   # Out (D28–D30)
                 10, 10, 10,   # Nov (D31–D33)
                 10, 10, 11]   # Dez (D34–D36)

assert sum(DECENDIO_DAYS) == 365

# Dias acumulados (0-indexed): CUMUL[i] = primeiro dia do decêndio i na régua anual
CUMUL = np.zeros(37, dtype=int)
for _i, _d in enumerate(DECENDIO_DAYS):
    CUMUL[_i + 1] = CUMUL[_i] + _d

# Lookup: para cada dia do ano (0..364), qual decêndio (0-indexed)?
DAY_TO_DEC = np.empty(365, dtype=int)
for _pos in range(365):
    for _di in range(36):
        if CUMUL[_di] <= _pos < CUMUL[_di + 1]:
            DAY_TO_DEC[_pos] = _di
            break

# ─────────────────────────────────────────────────────────────────────────────
# DICIONÁRIO TEMPORAL — D1..D36 → rótulo legível
# ─────────────────────────────────────────────────────────────────────────────
MONTHS_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
             "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# DEC_LABEL[n] onde n é 1-indexed (D1=1 … D36=36) → "01-10 Jan", etc.
DEC_LABEL: dict[int, str] = {}
for _m in range(12):
    _mn = MONTHS_PT[_m]
    _md = DAYS_PER_MONTH[_m]
    _base = _m * 3  # 0-indexed offset para o mês
    DEC_LABEL[_base + 1] = f"01-10 {_mn}"
    DEC_LABEL[_base + 2] = f"11-20 {_mn}"
    DEC_LABEL[_base + 3] = f"21-{_md:02d} {_mn}"

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES DE CALENDÁRIO
# ─────────────────────────────────────────────────────────────────────────────
def get_harvest_month(start_dec_1: int, cycle_total_days: int) -> str:
    """Retorna o mês estimado de colheita (ex: 'Nov')."""
    start_dec_0 = start_dec_1 - 1
    annual_pos = (CUMUL[start_dec_0] + cycle_total_days - 1) % 365
    harvest_dec_1 = int(DAY_TO_DEC[annual_pos]) + 1
    return DEC_LABEL[harvest_dec_1].split()[-1]  # extrai apenas o mês


def build_janelas_str(apt_dec_list: list[int], cycle_total_days: int) -> str:
    """
    Converte lista de decêndios de plantio aptos (1-indexed) em string com
    rótulos amigáveis e estimativa de colheita.
    Ex: "11-20 Jul (Colheita: ~Nov) | 21-31 Jul (Colheita: ~Nov)"
    """
    parts = []
    for dec_1 in apt_dec_list:
        label = DEC_LABEL[dec_1]
        harvest = get_harvest_month(dec_1, cycle_total_days)
        parts.append(f"{label} (Colheita: ~{harvest})")
    return " | ".join(parts)


def build_limitantes_str(failures: list[tuple]) -> str:
    """
    failures: lista de (dec_label_str, motivo_str) para decêndios inaptos.
    Retorna resumo agrupado por motivo, ex:
    'D1, D2, D3: Germinação (Tmin: 6.2°C < 8.0°C) | D8-D12: Tmed alta'
    """
    if not failures:
        return ""
    reason_to_decs: dict[str, list[str]] = defaultdict(list)
    for dec_label, reason in failures:
        reason_to_decs[reason].append(dec_label)

    # Ordena por frequência (mais comum primeiro), toma top 3
    sorted_reasons = sorted(reason_to_decs.items(), key=lambda x: -len(x[1]))[:3]
    parts = []
    for reason, decs in sorted_reasons:
        if len(decs) <= 4:
            decs_str = ", ".join(decs)
        else:
            decs_str = f"{decs[0]}-{decs[-1]} ({len(decs)} dec.)"
        parts.append(f"{decs_str}: {reason}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
TEMP_FILE    = os.path.join(_HERE, "resultado_zoneamento_temp.parquet")
PARQUET_PATH = os.path.join(_HERE, "Base_Zoneamento_BR.parquet")
COORDS_PATH  = os.path.join(_HERE, "municipios_coords.parquet")

ESTAGIOS = [
    "Germinação e Emergência",
    "Perfilhamento",
    "Alongamento",
    "Emborrachamento",
    "Espigamento e Floração",
    "Enchimento de Grãos e Maturação",
    "Colheita",
]
STAGE_ICONS = ["🌱", "🌿", "📏", "🌾", "🌸", "🌰", "🚜"]

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS (cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_base() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["Altitude_m"] = df["Altitude_m"].fillna(-1).astype(int)
    return df


@st.cache_data
def load_coords() -> pd.DataFrame:
    """Carrega coordenadas pré-computadas dos municípios (centroides IBGE)."""
    return pd.read_parquet(COORDS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MATEMÁTICA — Pré-computação dos pesos por (decêndio-início, fase)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_phase_weights(start_dec_0: int, day_start: int, day_end: int) -> dict:
    """
    Retorna pesos vetorizados para cálculo de variáveis climáticas em uma fase.

    prec_w[dec]  = dias_da_fase_no_dec / dias_totais_do_dec
                   → phase_prec = prec_mat @ prec_w  (acumulado proporcional)
    tmed_w[dec]  = dias_da_fase_no_dec / dur_fase
                   → phase_tmed = tmed_mat @ tmed_w  (média ponderada)
    tmin_idx / tmax_idx: decêndios tocados (para min/max)
    """
    phase_dur = day_end - day_start + 1
    dec_day_count = np.zeros(36, dtype=int)

    for sim_day in range(day_start, day_end + 1):
        annual_pos = (CUMUL[start_dec_0] + sim_day - 1) % 365
        dec_day_count[DAY_TO_DEC[annual_pos]] += 1

    touched = np.where(dec_day_count > 0)[0]
    prec_w = np.zeros(36)
    tmed_w = np.zeros(36)
    for di in touched:
        prec_w[di] = dec_day_count[di] / DECENDIO_DAYS[di]
        tmed_w[di] = dec_day_count[di] / phase_dur

    return {
        "prec_w": prec_w,
        "tmed_w": tmed_w,
        "tmin_idx": touched.tolist(),
        "tmax_idx": touched.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE CÁLCULO PRINCIPAL — com rastreamento de fatores limitantes
# ─────────────────────────────────────────────────────────────────────────────
def run_zoneamento(df_filtered: pd.DataFrame, phases: list[dict],
                   cycle_total_days: int) -> pd.DataFrame:
    """
    Varre D1-D36 para cada município filtrado.
    Retorna DataFrame com decêndios aptos, janelas amigáveis,
    estimativas de colheita e fatores limitantes dos decêndios inaptos.
    """
    N = len(df_filtered)
    if N == 0:
        return pd.DataFrame()

    # Matrizes climáticas — shape (N, 36)
    prec_mat = df_filtered[[f"Prec_D{i}" for i in range(1, 37)]].values.astype(float)
    tmed_mat = df_filtered[[f"Tmed_D{i}" for i in range(1, 37)]].values.astype(float)
    tmax_mat = df_filtered[[f"Tmax_D{i}" for i in range(1, 37)]].values.astype(float)
    tmin_mat = df_filtered[[f"Tmin_D{i}" for i in range(1, 37)]].values.astype(float)

    # Régua de dias das fases
    phase_ranges: list[tuple[int, int]] = []
    cursor = 1
    for ph in phases:
        phase_ranges.append((cursor, cursor + ph["dur"] - 1))
        cursor += ph["dur"]

    # Acumuladores por município
    apt_dec_raw: list[list[int]] = [[] for _ in range(N)]   # decêndios aptos (1-indexed)
    all_failures: list[list[tuple]] = [[] for _ in range(N)]  # (dec_label, motivo)

    progress = st.progress(0, text="Iniciando…")

    for start_dec_0 in range(36):
        progress.progress(
            (start_dec_0 + 1) / 36,
            text=f"Varrendo decêndio inicial D{start_dec_0 + 1} / 36…",
        )

        weights = [precompute_phase_weights(start_dec_0, ds, de)
                   for ds, de in phase_ranges]

        apt = np.ones(N, dtype=bool)
        first_failure = np.full(N, "", dtype=object)

        for ph_idx, ph in enumerate(phases):
            if not apt.any():
                break

            w = weights[ph_idx]
            stage = ESTAGIOS[ph_idx]

            # ── Calcula condições de TODAS as variáveis habilitadas nesta fase ──
            # (para municípios ainda aptos — computamos para todos N por eficiência
            #  vetorial; municípios já inaptos não afetam o resultado)
            var_results: dict[str, tuple] = {}

            if ph["prec_en"]:
                vals = prec_mat @ w["prec_w"]
                cond = (vals >= ph["prec_min"]) & (vals <= ph["prec_max"])
                var_results["Prec. Acum."] = (vals, cond, ph["prec_min"], ph["prec_max"], "mm")

            if ph["tmed_en"]:
                vals = tmed_mat @ w["tmed_w"]
                cond = (vals >= ph["tmed_min"]) & (vals <= ph["tmed_max"])
                var_results["Tmed"] = (vals, cond, ph["tmed_min"], ph["tmed_max"], "°C")

            if ph["tmin_en"] and w["tmin_idx"]:
                vals = tmin_mat[:, w["tmin_idx"]].min(axis=1)
                cond = (vals >= ph["tmin_min"]) & (vals <= ph["tmin_max"])
                var_results["Tmin"] = (vals, cond, ph["tmin_min"], ph["tmin_max"], "°C")

            if ph["tmax_en"] and w["tmax_idx"]:
                vals = tmax_mat[:, w["tmax_idx"]].max(axis=1)
                cond = (vals >= ph["tmax_min"]) & (vals <= ph["tmax_max"])
                var_results["Tmax"] = (vals, cond, ph["tmax_min"], ph["tmax_max"], "°C")

            # Máscara combinada desta fase
            phase_pass = np.ones(N, dtype=bool)
            for _, (_, cond, _, _, _) in var_results.items():
                phase_pass &= cond

            # Municípios que falham AGORA (eram aptos antes desta fase)
            new_fail_mask = apt & ~phase_pass
            fail_indices = np.where(new_fail_mask)[0]

            for idx in fail_indices:
                if not first_failure[idx]:          # registra apenas a PRIMEIRA falha
                    parts: list[str] = []
                    for var, (vals, cond_arr, vmin, vmax, unit) in var_results.items():
                        if not cond_arr[idx]:
                            v = float(vals[idx])
                            direction = f"{v:.1f}{unit} < {vmin:.1f}{unit}" \
                                        if v < vmin else \
                                        f"{v:.1f}{unit} > {vmax:.1f}{unit}"
                            parts.append(f"{var}: {direction}")
                    first_failure[idx] = f"{stage} ({'; '.join(parts)})" if parts \
                        else stage

            apt &= phase_pass

        # ── Registra aptos e inaptos ─────────────────────────────────────────
        dec_label_str = f"D{start_dec_0 + 1}"
        apt_arr = apt  # snapshot final desta iteração
        for idx in range(N):
            if apt_arr[idx]:
                apt_dec_raw[idx].append(start_dec_0 + 1)
            else:
                motivo = first_failure[idx] if first_failure[idx] \
                    else "Critérios climáticos não atendidos"
                all_failures[idx].append((dec_label_str, motivo))

    progress.empty()

    # ── Monta DataFrame final ────────────────────────────────────────────────
    rows = []
    for i, row in enumerate(df_filtered.itertuples(index=False)):
        if not apt_dec_raw[i]:
            continue
        rows.append({
            "Codigo_IBGE":        row.Codigo_IBGE,
            "Municipio":          row.Municipio,
            "UF":                 row.UF,
            "Altitude_m":         row.Altitude_m,
            "Solo_Dominante":     row.Solo_Dominante,
            "Decendios_Aptos":    ", ".join(f"D{d}" for d in apt_dec_raw[i]),
            "Janelas_Plantio":    build_janelas_str(apt_dec_raw[i], cycle_total_days),
            "Num_Decendios_Aptos": len(apt_dec_raw[i]),
            "Fatores_Limitantes": build_limitantes_str(all_failures[i]),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — popup HTML para o Folium
# ─────────────────────────────────────────────────────────────────────────────
def build_popup_html(row: pd.Series) -> str:
    """Gera HTML rico para o popup do marcador Folium."""
    janelas_html = "".join(
        f"<li style='margin:2px 0'>🗓️ {j.strip()}</li>"
        for j in row["Janelas_Plantio"].split("|")
    )

    # Top 2 fatores limitantes para o popup (para não ficar enorme)
    limitantes_parts = row["Fatores_Limitantes"].split("|") if row["Fatores_Limitantes"] else []
    if limitantes_parts:
        lim_html = "".join(
            f"<li style='margin:2px 0;color:#c0392b'>⚠️ {p.strip()}</li>"
            for p in limitantes_parts[:2]
        )
        lim_block = (
            "<hr style='margin:6px 0;border-color:#ddd'>"
            "<b style='color:#c0392b'>Fatores Restritivos (principais):</b>"
            f"<ul style='margin:4px 0 0 0;padding-left:14px'>{lim_html}</ul>"
        )
    else:
        lim_block = ""

    return (
        f"<div style='font-family:Arial,sans-serif;font-size:13px;"
        f"min-width:260px;max-width:380px;line-height:1.4'>"
        f"<b style='font-size:14px'>📍 {row['Municipio']} / {row['UF']}</b>"
        f"<span style='color:#555'> | ⛰️ {row['Altitude_m']} m</span><br>"
        f"<span style='color:#666;font-size:12px'>🌱 Solo: {row['Solo_Dominante']}</span>"
        f"<hr style='margin:6px 0;border-color:#ddd'>"
        f"<b style='color:#27ae60'>🌾 Janelas de Plantio Aptas "
        f"({row['Num_Decendios_Aptos']}):</b>"
        f"<ul style='margin:4px 0 0 0;padding-left:14px'>{janelas_html}</ul>"
        f"{lim_block}"
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE — CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zoneamento Agroclimático da Cevada",
    page_icon="🌾",
    layout="wide",
)

# Registra acesso uma vez por sessão e exibe painel admin na sidebar
if "acesso_registrado" not in st.session_state:
    _registrar_acesso()
    st.session_state["acesso_registrado"] = True
_painel_admin()

st.title("🌾 Zoneamento Agroclimático da Cevada")
st.markdown(
    "Defina os filtros e os requisitos climáticos de cada estádio fenológico. "
    "O sistema varre os **36 decêndios** possíveis de plantio para cada município "
    "e rastreia os fatores limitantes dos decêndios inaptos."
)

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Carregando base de dados…"):
    df_base = load_base()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTROS GUILHOTINA
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Filtros Primários (Guilhotina)")

alt_min_val = int(df_base["Altitude_m"][df_base["Altitude_m"] >= 0].min())
alt_max_val = int(df_base["Altitude_m"].max())

alt_range = st.sidebar.slider(
    "Altitude (m)", min_value=alt_min_val, max_value=alt_max_val,
    value=(alt_min_val, alt_max_val), step=10,
    help="Filtra municípios dentro da faixa de altitude.",
)

solos_disponiveis = sorted(df_base["Solo_Dominante"].dropna().unique().tolist())
solos_sel = st.sidebar.multiselect(
    "Solo Dominante", options=solos_disponiveis, default=solos_disponiveis,
    help="Selecione os tipos de solo permitidos.",
)

df_filtered = df_base[
    (df_base["Altitude_m"] >= alt_range[0])
    & (df_base["Altitude_m"] <= alt_range[1])
    & (df_base["Solo_Dominante"].isin(solos_sel))
].reset_index(drop=True)

st.sidebar.metric("Municípios após filtros", len(df_filtered))

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÕES FENOLÓGICAS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Estádios Fenológicos da Cevada")
st.caption(
    "Informe a duração (obrigatória) e habilite os limites climáticos opcionais "
    "para cada estádio. Os dias são cumulativos: o estádio seguinte começa "
    "imediatamente após o fim do anterior."
)

phase_inputs: list[dict] = []

for s_idx, stage_name in enumerate(ESTAGIOS):
    with st.expander(f"{STAGE_ICONS[s_idx]} {stage_name}", expanded=(s_idx == 0)):
        col_dur, col_info = st.columns([1, 3])
        with col_dur:
            dur = st.number_input(
                "Duração (dias) *",
                min_value=1, max_value=365, value=None, step=1,
                key=f"dur_{s_idx}", placeholder="Obrigatório",
            )
        with col_info:
            if dur:
                st.info(f"Este estádio terá **{dur} dia(s)**.")
            else:
                st.warning("Preencha a duração para habilitar o processamento.")

        prec_en = st.checkbox("Limitar Precipitação Acumulada (mm)", key=f"prec_en_{s_idx}")
        if prec_en:
            c1, c2 = st.columns(2)
            prec_min = c1.number_input("Prec. Mín (mm)", value=0.0, step=1.0, key=f"prec_min_{s_idx}")
            prec_max = c2.number_input("Prec. Máx (mm)", value=500.0, step=1.0, key=f"prec_max_{s_idx}")
        else:
            prec_min, prec_max = None, None

        tmed_en = st.checkbox("Limitar Temperatura Média (°C)", key=f"tmed_en_{s_idx}")
        if tmed_en:
            c1, c2 = st.columns(2)
            tmed_min = c1.number_input("Tmed Mín (°C)", value=5.0, step=0.5, key=f"tmed_min_{s_idx}")
            tmed_max = c2.number_input("Tmed Máx (°C)", value=30.0, step=0.5, key=f"tmed_max_{s_idx}")
        else:
            tmed_min, tmed_max = None, None

        tmax_en = st.checkbox("Limitar Temperatura Máxima (°C)", key=f"tmax_en_{s_idx}")
        if tmax_en:
            c1, c2 = st.columns(2)
            tmax_min = c1.number_input("Tmax Mín (°C)", value=0.0, step=0.5, key=f"tmax_min_{s_idx}")
            tmax_max = c2.number_input("Tmax Máx (°C)", value=40.0, step=0.5, key=f"tmax_max_{s_idx}")
        else:
            tmax_min, tmax_max = None, None

        tmin_en = st.checkbox("Limitar Temperatura Mínima (°C)", key=f"tmin_en_{s_idx}")
        if tmin_en:
            c1, c2 = st.columns(2)
            tmin_min = c1.number_input("Tmin Mín (°C)", value=-5.0, step=0.5, key=f"tmin_min_{s_idx}")
            tmin_max = c2.number_input("Tmin Máx (°C)", value=20.0, step=0.5, key=f"tmin_max_{s_idx}")
        else:
            tmin_min, tmin_max = None, None

        phase_inputs.append(dict(
            dur=dur,
            prec_en=prec_en, prec_min=prec_min, prec_max=prec_max,
            tmed_en=tmed_en, tmed_min=tmed_min, tmed_max=tmed_max,
            tmax_en=tmax_en, tmax_min=tmax_min, tmax_max=tmax_max,
            tmin_en=tmin_en, tmin_min=tmin_min, tmin_max=tmin_max,
        ))

# ─────────────────────────────────────────────────────────────────────────────
# RESUMO DA RÉGUA DE DIAS
# ─────────────────────────────────────────────────────────────────────────────
durations_ok = all(ph["dur"] is not None and ph["dur"] > 0 for ph in phase_inputs)

if durations_ok:
    total_days = sum(ph["dur"] for ph in phase_inputs)
    cursor = 1
    summary_rows = []
    for s_idx, ph in enumerate(phase_inputs):
        d = ph["dur"]
        summary_rows.append({
            "Estádio": f"{STAGE_ICONS[s_idx]} {ESTAGIOS[s_idx]}",
            "Dia Início": cursor,
            "Dia Fim": cursor + d - 1,
            "Duração (dias)": d,
        })
        cursor += d

    st.markdown("---")
    st.subheader("Régua de Dias da Simulação")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.metric("Total de Dias do Ciclo", total_days)

    if total_days > 365:
        st.error("O ciclo total ultrapassa 365 dias. Reduza as durações dos estádios.")
        durations_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# BOTÃO 1 — PROCESSAR ZONEAMENTO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    btn_processar = st.button(
        "1. Processar Zoneamento",
        type="primary",
        disabled=not durations_ok or len(df_filtered) == 0,
        help="Executa a varredura para todos os municípios filtrados.",
    )

if btn_processar:
    if not durations_ok:
        st.error("Preencha a duração de TODOS os estádios antes de processar.")
    elif len(df_filtered) == 0:
        st.error("Nenhum município passou pelos filtros guilhotina.")
    else:
        def _safe(v, default):
            return v if v is not None else default

        phases_norm = [
            dict(
                dur=ph["dur"],
                prec_en=ph["prec_en"],
                prec_min=_safe(ph["prec_min"], 0.0),
                prec_max=_safe(ph["prec_max"], 1e9),
                tmed_en=ph["tmed_en"],
                tmed_min=_safe(ph["tmed_min"], -99.0),
                tmed_max=_safe(ph["tmed_max"],  99.0),
                tmax_en=ph["tmax_en"],
                tmax_min=_safe(ph["tmax_min"], -99.0),
                tmax_max=_safe(ph["tmax_max"],  99.0),
                tmin_en=ph["tmin_en"],
                tmin_min=_safe(ph["tmin_min"], -99.0),
                tmin_max=_safe(ph["tmin_max"],  99.0),
            )
            for ph in phase_inputs
        ]
        cycle_days = sum(ph["dur"] for ph in phase_inputs)

        with st.spinner("Varrendo decêndios e municípios…"):
            df_result = run_zoneamento(df_filtered, phases_norm, cycle_days)

        if df_result.empty:
            st.warning(
                "Nenhum município apto encontrado com os critérios definidos. "
                "Considere relaxar os limites climáticos."
            )
        else:
            df_result.to_parquet(TEMP_FILE, index=False)
            st.success(
                f"Processamento concluído. **{len(df_result)} municípios aptos** encontrados. "
                f"Resultados salvos em `resultado_zoneamento_temp.parquet`."
            )

# ─────────────────────────────────────────────────────────────────────────────
# BOTÃO 2 — GERAR MAPA E TABELA
# ─────────────────────────────────────────────────────────────────────────────
with col_btn2:
    btn_mapa = st.button(
        "2. Gerar Mapa e Tabela",
        disabled=not os.path.exists(TEMP_FILE),
        help="Lê o arquivo de resultados e exibe mapa e tabela.",
    )

if btn_mapa:
    st.session_state["show_results"] = True

if st.session_state.get("show_results") and os.path.exists(TEMP_FILE):
    st.markdown("---")
    st.subheader("Resultados do Zoneamento")

    df_res = pd.read_parquet(TEMP_FILE)

    with st.spinner("Carregando coordenadas geográficas…"):
        coords = load_coords()

    df_map = df_res.merge(
        coords, left_on="Codigo_IBGE", right_on="CD_MUN", how="left"
    ).dropna(subset=["lat", "lon"])

    # ── Métricas ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municípios Aptos", len(df_res))
    c2.metric("Estados Contempl.", df_res["UF"].nunique())
    c3.metric("Máx. Janelas / Mun.", int(df_res["Num_Decendios_Aptos"].max()))
    verde_pct = (df_res["Num_Decendios_Aptos"] >= 3).mean() * 100
    c4.metric("Municípios Baixo Risco (≥3 jan.)", f"{verde_pct:.0f}%")

    # ── Mapa Folium ───────────────────────────────────────────────────────────
    st.subheader("Mapa Interativo dos Municípios Aptos")

    # Legenda de cores
    col_leg1, col_leg2, _ = st.columns([1, 1, 3])
    col_leg1.markdown(
        "<span style='background:#27ae60;color:white;padding:3px 10px;"
        "border-radius:4px;font-size:13px'>● Verde — ≥ 3 janelas (Baixo Risco)</span>",
        unsafe_allow_html=True,
    )
    col_leg2.markdown(
        "<span style='background:#e67e22;color:white;padding:3px 10px;"
        "border-radius:4px;font-size:13px'>● Laranja — 1-2 janelas (Risco Moderado)</span>",
        unsafe_allow_html=True,
    )

    lat_c = df_map["lat"].mean()
    lon_c = df_map["lon"].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=5, tiles="CartoDB positron")

    for _, row in df_map.iterrows():
        n = int(row["Num_Decendios_Aptos"])
        color = "#27ae60" if n >= 3 else "#e67e22"   # Verde ou Laranja
        radius = 5 + min(n, 12)                       # tamanho proporcional às janelas

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.80,
            weight=1.2,
            popup=folium.Popup(build_popup_html(row), max_width=400),
            tooltip=(
                f"<b>{row['Municipio']} — {row['UF']}</b><br>"
                f"{n} janela(s) de plantio"
            ),
        ).add_to(m)

    st_folium(m, width="100%", height=580, returned_objects=[])

    # ── Tabela de Resultados ─────────────────────────────────────────────────
    st.subheader("Tabela de Municípios Aptos")

    ufs_res = ["Todos"] + sorted(df_res["UF"].unique().tolist())
    col_f1, col_f2 = st.columns([1, 2])
    uf_filter = col_f1.selectbox("Filtrar por UF:", ufs_res)
    min_jan = col_f2.slider("Mínimo de janelas aptas:", 1, int(df_res["Num_Decendios_Aptos"].max()), 1)

    df_show = df_res.copy()
    if uf_filter != "Todos":
        df_show = df_show[df_show["UF"] == uf_filter]
    df_show = df_show[df_show["Num_Decendios_Aptos"] >= min_jan]
    df_show = df_show.sort_values(
        ["Num_Decendios_Aptos", "Municipio"], ascending=[False, True]
    ).reset_index(drop=True)

    st.dataframe(
        df_show[[
            "Municipio", "UF", "Altitude_m", "Solo_Dominante",
            "Num_Decendios_Aptos", "Janelas_Plantio", "Fatores_Limitantes",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Municipio":           st.column_config.TextColumn("Município"),
            "Altitude_m":          st.column_config.NumberColumn("Altitude (m)", format="%d m"),
            "Num_Decendios_Aptos": st.column_config.NumberColumn("Janelas Aptas", format="%d"),
            "Janelas_Plantio":     st.column_config.TextColumn(
                "Janelas de Plantio (Colheita)", width="large"
            ),
            "Fatores_Limitantes":  st.column_config.TextColumn(
                "Principais Fatores Limitantes", width="large"
            ),
        },
    )

    st.caption(f"Exibindo **{len(df_show)}** município(s).")

    # ── Download CSV ─────────────────────────────────────────────────────────
    csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️  Baixar resultado como CSV",
        data=csv_bytes,
        file_name="zoneamento_cevada.csv",
        mime="text/csv",
        help="Exporta a tabela atual (com filtros aplicados) em formato CSV.",
    )

    # ── Distribuição por Estado ───────────────────────────────────────────────
    st.subheader("Distribuição por Estado")
    uf_counts = (
        df_res.groupby("UF")
        .agg(
            Municipios=("Municipio", "count"),
            Media_Janelas=("Num_Decendios_Aptos", "mean"),
            Max_Janelas=("Num_Decendios_Aptos", "max"),
        )
        .sort_values("Municipios", ascending=False)
        .reset_index()
    )
    uf_counts["Media_Janelas"] = uf_counts["Media_Janelas"].round(1)
    st.dataframe(uf_counts, use_container_width=True, hide_index=True,
                 column_config={
                     "Media_Janelas": st.column_config.NumberColumn("Média Janelas", format="%.1f"),
                     "Max_Janelas":   st.column_config.NumberColumn("Máx. Janelas", format="%d"),
                 })
