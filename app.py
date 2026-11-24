"""Aplicación web interactiva para la predicción horaria de PM2.5 en Pekín.

Proyecto Final — Materia: Data Science
Universidad Autónoma de Chihuahua (UACH) — Facultad de Ingeniería
Programa: Ingeniería en Ciencias de la Computación
Alumno: Alexis Aidan Leon Rodelas (Matrícula: 377171)
Docente: Olanda Prieto Ordaz
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.model_loader import (  # noqa: E402
    load_all_models,
    load_metadata,
    predecir,
    predecir_todos,
)
from src.preprocessing import (  # noqa: E402
    CBWD_OPTIONS,
    HISTORICAL_RANGES,
    PHYSICAL_LIMITS,
    RAW_COLUMNS,
    validar_entrada,
)

# ---------------------------------------------------------------------------
# Constantes de dominio y formato en español
# ---------------------------------------------------------------------------

PRIMARY_COLOR = "#2b5c8f"
ACCENT_COLOR = "#d95f02"
SECONDARY_TEAL = "#41b6c4"

MESES_ES: dict[int, str] = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

DIAS_SEMANA_ES: dict[int, str] = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}

EPA_BANDS: list[dict[str, Any]] = [
    {
        "categoria": "Buena",
        "rango": "0.0 – 12.0 µg/m³",
        "min": 0.0,
        "max": 12.0,
        "color": "#2ecc71",
        "text_color": "#0f5132",
        "resumen": "Riesgo mínimo o nulo para la población.",
    },
    {
        "categoria": "Moderada",
        "rango": "12.1 – 35.4 µg/m³",
        "min": 12.1,
        "max": 35.4,
        "color": "#f1c40f",
        "text_color": "#5c4800",
        "resumen": "Aceptable; precaución leve en personas extremadamente sensibles.",
    },
    {
        "categoria": "Dañina para grupos sensibles",
        "rango": "35.5 – 55.4 µg/m³",
        "min": 35.5,
        "max": 55.4,
        "color": "#e67e22",
        "text_color": "#ffffff",
        "resumen": "Niños, adultos mayores y personas con asma deben reducir esfuerzo intenso.",
    },
    {
        "categoria": "Dañina",
        "rango": "55.5 – 150.4 µg/m³",
        "min": 55.5,
        "max": 150.4,
        "color": "#e74c3c",
        "text_color": "#ffffff",
        "resumen": "Efectos respiratorios posibles en población general; limitar actividad exterior.",
    },
    {
        "categoria": "Muy dañina",
        "rango": "150.5 – 250.4 µg/m³",
        "min": 150.5,
        "max": 250.4,
        "color": "#8e44ad",
        "text_color": "#ffffff",
        "resumen": "Alerta sanitaria; suspender ejercicio exterior y usar filtración/mascarilla.",
    },
    {
        "categoria": "Peligrosa",
        "rango": "> 250.4 µg/m³",
        "min": 250.5,
        "max": 500.0,
        "color": "#7f1d1d",
        "text_color": "#ffffff",
        "resumen": "Contingencia severa por smog; permanecer en interiores filtrados.",
    },
]

DEFAULT_INPUT_VALUES: dict[str, Any] = {
    "year": 2014,
    "month": 1,
    "day": 15,
    "hour": 22,
    "TEMP": -1.0,
    "DEWP": -2.0,
    "PRES": 1022.0,
    "cbwd": "cv",
    "Iws": 1.8,
    "Is": 0,
    "Ir": 0,
}


# ---------------------------------------------------------------------------
# Carga en caché de metadatos y modelos (CERO reentrenamiento / recarga)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def obtener_metadatos() -> dict[str, Any]:
    """Carga una única vez el diccionario de metadatos desde models/metadata.json."""
    return load_metadata()


@st.cache_resource(show_spinner="Cargando modelos preentrenados de regresión...")
def obtener_modelos() -> dict[str, Any]:
    """Carga en memoria compartida los 3 Pipelines preentrenados sin reentrenarlos jamás."""
    return load_all_models()


# ---------------------------------------------------------------------------
# Utilidades de extracción defensiva de métricas y estructuras
# ---------------------------------------------------------------------------


def _extract_metric(metrics: dict[str, Any] | None, metric_type: str) -> float | None:
    """Extrae una métrica (mae, mse, rmse, r2, std, tiempo) tolerando variantes de claves."""
    if not isinstance(metrics, dict):
        return None

    lower_map = {str(k).lower(): v for k, v in metrics.items() if v is not None}

    candidates_by_type: dict[str, tuple[str, ...]] = {
        "mae": ("mae", "mae_cv", "mae_test"),
        "mae_std": ("mae_std",),
        "mse": ("mse", "mse_cv", "mse_test"),
        "rmse": ("rmse", "rmse_cv", "rmse_test"),
        "rmse_std": ("rmse_std",),
        "r2": ("r2", "r2_cv", "r2_test", "r2_score"),
        "r2_std": ("r2_std",),
        "rmse_train": ("rmse_train",),
        "r2_train": ("r2_train",),
        "fit_time": ("fit_time_s", "tiempo_s", "tiempo_fit_s", "fit_time"),
    }

    for cand in candidates_by_type.get(metric_type, (metric_type.lower(),)):
        if cand in lower_map:
            try:
                return float(lower_map[cand])
            except (TypeError, ValueError):
                continue

    # Búsqueda flexible secundaria evitando colisiones entre mse y rmse o std/train
    for k, v in lower_map.items():
        if metric_type == "mse" and "mse" in k and "rmse" not in k and "std" not in k and "train" not in k:
            return float(v)
        if metric_type == "rmse" and "rmse" in k and "std" not in k and "train" not in k and "brecha" not in k and "diff" not in k:
            return float(v)
        if metric_type == "mae" and "mae" in k and "std" not in k and "train" not in k:
            return float(v)
        if metric_type == "r2" and "r2" in k and "std" not in k and "train" not in k and "diff" not in k:
            return float(v)

    return None


def _fmt_num(val: float | None, decimals: int = 2, suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/D"
    return f"{val:,.{decimals}f}{suffix}"


def _normalize_all_models_df(metadata: dict[str, Any]) -> pd.DataFrame:
    """Convierte metadata['all_models_summary'] (lista o dict) en un DataFrame homogéneo."""
    raw = metadata.get("all_models_summary", [])
    rows: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        items = [{"name": k, **(v if isinstance(v, dict) else {})} for k, v in raw.items()]
    elif isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    else:
        items = []

    for item in items:
        name = str(item.get("name") or item.get("model") or item.get("modelo") or "Modelo")
        family = str(item.get("family") or item.get("familia") or "—")
        selected = bool(
            item.get("selected_top3")
            or item.get("selected_for_app")
            or item.get("is_top3")
            or item.get("selected", False)
        )
        rows.append(
            {
                "Modelo": name,
                "Familia": family,
                "MAE_CV": _extract_metric(item, "mae"),
                "MSE_CV": _extract_metric(item, "mse"),
                "RMSE_CV": _extract_metric(item, "rmse"),
                "RMSE_std": _extract_metric(item, "rmse_std"),
                "R2_CV": _extract_metric(item, "r2"),
                "RMSE_Train": _extract_metric(item, "rmse_train"),
                "Tiempo_s": _extract_metric(item, "fit_time"),
                "Peso_MB": item.get("xz3_size_mb"),
                "Estado_App": "Integrado" if selected else "Referencia",
                "Nota": str(item.get("note") or ""),
                "Seleccionado_Top3": selected,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "RMSE_CV" in df.columns:
        df = df.sort_values("RMSE_CV", ascending=True).reset_index(drop=True)
    return df


def _normalize_features_table(metadata: dict[str, Any]) -> pd.DataFrame:
    """Construye una tabla descriptiva legible de las 11 variables crudas de entrada."""
    raw_feats = metadata.get("features", {})
    feat_map: dict[str, dict[str, Any]] = {}

    if isinstance(raw_feats, dict):
        for k, v in raw_feats.items():
            if isinstance(v, dict):
                feat_map[k] = v
    elif isinstance(raw_feats, list):
        for item in raw_feats:
            if isinstance(item, dict) and ("name" in item or "id" in item or "column" in item):
                key = str(item.get("name") or item.get("id") or item.get("column"))
                feat_map[key] = item

    default_meta: dict[str, dict[str, str]] = {
        "year": {"label": "Año del registro", "unit": "año", "type": "Discreta temporal", "desc": "Año calendario de la observación (entrenamiento: 2010–2014)."},
        "month": {"label": "Mes del año", "unit": "1–12", "type": "Discreta temporal", "desc": "Mes calendario (captura la estacionalidad invernal y monzónica)."},
        "day": {"label": "Día del mes", "unit": "1–31", "type": "Discreta temporal", "desc": "Día del mes; interviene en día de semana e inicio/fin de calefacción."},
        "hour": {"label": "Hora del día", "unit": "0–23 h", "type": "Discreta temporal", "desc": "Hora local en Pekín (ciclo diurno de tráfico e inversión térmica nocturna)."},
        "DEWP": {"label": "Punto de rocío", "unit": "°C", "type": "Continua meteorológica", "desc": "Temperatura a la cual el aire alcanza saturación de vapor de agua (DEWP ≤ TEMP)."},
        "TEMP": {"label": "Temperatura ambiente", "unit": "°C", "type": "Continua meteorológica", "desc": "Temperatura del aire medida en el Aeropuerto Internacional de Pekín."},
        "PRES": {"label": "Presión atmosférica", "unit": "hPa", "type": "Continua meteorológica", "desc": "Presión barométrica a nivel de estación (alta en anticiclones invernales)."},
        "cbwd": {"label": "Dirección combinada del viento", "unit": "categoría", "type": "Categórica nominal", "desc": "SE (Sureste), NW (Noroeste limpiador), NE (Noreste) o cv (Calmo/Variable)."},
        "Iws": {"label": "Velocidad acumulada del viento", "unit": "m/s", "type": "Continua meteorológica", "desc": "Velocidad de viento acumulada mientras se mantiene una misma dirección."},
        "Is": {"label": "Horas acumuladas de nieve", "unit": "horas", "type": "Discreta meteorológica", "desc": "Número de horas continuas con caída de nieve."},
        "Ir": {"label": "Horas acumuladas de lluvia", "unit": "horas", "type": "Discreta meteorológica", "desc": "Número de horas continuas con precipitación pluvial (lavado atmosférico)."},
    }

    rows: list[dict[str, Any]] = []
    for col in RAW_COLUMNS:
        info = feat_map.get(col, {})
        d_info = default_meta.get(col, {})
        h_range = HISTORICAL_RANGES.get(col)
        p_lim = PHYSICAL_LIMITS.get(col)

        min_hist = info.get("min", h_range[0] if h_range else "—")
        med_hist = info.get("median", "—")
        max_hist = info.get("max", h_range[1] if h_range else "—")

        if col == "cbwd":
            rango_hist_str = "SE (34.9%), NW (32.2%), cv (21.5%), NE (11.4%)"
            rango_val_str = "SE, NW, cv, NE"
        else:
            rango_hist_str = f"{min_hist} a {max_hist}" + (f" (mediana: {med_hist})" if med_hist != "—" else "")
            if p_lim is not None:
                rango_val_str = f"{p_lim[0]} a {p_lim[1]}"
            else:
                rango_val_str = "≥ 0"

        rows.append(
            {
                "Variable": col,
                "Etiqueta en Español": str(info.get("label") or d_info.get("label") or col),
                "Unidad": str(info.get("unit") or d_info.get("unit") or "—"),
                "Naturaleza": str(info.get("type") or d_info.get("type") or "Numérica"),
                "Rango Histórico Train (2010–2014)": rango_hist_str,
                "Rango Físico Válido": rango_val_str,
                "Descripción": str(info.get("description") or d_info.get("desc") or ""),
            }
        )

    return pd.DataFrame(rows)


def _get_eng_value(eng: dict[str, Any], candidates: tuple[str, ...], default: Any = None) -> Any:
    """Obtiene una característica derivada admitiendo snake_case o CamelCase."""
    for key in candidates:
        if key in eng:
            return eng[key]
    lower_eng = {str(k).lower(): v for k, v in eng.items()}
    for key in candidates:
        if key.lower() in lower_eng:
            return lower_eng[key.lower()]
    return default


# ---------------------------------------------------------------------------
# Estilos visuales académicos (paleta #2b5c8f y #d95f02)
# ---------------------------------------------------------------------------


def _inject_custom_styles() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }}
        .inst-banner {{
            background: linear-gradient(90deg, #f3f6fa 0%, #ffffff 100%);
            border: 1px solid #d8e2ee;
            border-left: 5px solid {PRIMARY_COLOR};
            border-radius: 8px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1rem;
            font-size: 0.92rem;
            color: #1e293b;
        }}
        .inst-banner-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 1.6rem;
            margin-top: 0.35rem;
        }}
        .problem-card {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1.1rem;
            line-height: 1.55;
        }}
        .model-summary-box {{
            background-color: #f3f6fa;
            border: 1px solid #cbd5e1;
            border-left: 4px solid {ACCENT_COLOR};
            border-radius: 6px;
            padding: 0.75rem 0.95rem;
            margin-top: 0.35rem;
            margin-bottom: 0.9rem;
            font-size: 0.9rem;
        }}
        .aqi-result-card {{
            border-radius: 10px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            border: 1px solid #dce4ee;
            background-color: #ffffff;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }}
        .aqi-badge {{
            display: inline-block;
            padding: 0.32rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            letter-spacing: 0.01em;
        }}
        .aqi-scale-wrapper {{
            margin-top: 0.9rem;
            margin-bottom: 0.6rem;
        }}
        .aqi-scale-bar {{
            position: relative;
            height: 18px;
            border-radius: 9px;
            display: flex;
            overflow: hidden;
            border: 1px solid rgba(15, 23, 42, 0.15);
        }}
        .aqi-marker-row {{
            position: relative;
            height: 22px;
            margin-top: 2px;
        }}
        .aqi-marker {{
            position: absolute;
            transform: translateX(-50%);
            font-size: 0.78rem;
            font-weight: 700;
            color: #0f172a;
            white-space: nowrap;
        }}
        .health-callout {{
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-top: 0.75rem;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
        }}
        .just-box {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid {PRIMARY_COLOR};
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.65rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Componentes visuales reutilizables
# ---------------------------------------------------------------------------


def _render_sidebar_institutional(metadata: dict[str, Any]) -> None:
    """Renderiza en la barra lateral la ficha institucional obligatoria y resumen técnico."""
    with st.sidebar:
        st.markdown("### Datos Institucionales")
        st.markdown(
            """
- **Universidad:** Universidad Autónoma de Chihuahua (UACH) — Facultad de Ingeniería
- **Carrera:** Ingeniería en Ciencias de la Computación
- **Materia:** Data Science
- **Alumno:** Alexis Aidan Leon Rodelas
- **Matrícula:** 377171
- **Docente:** Olanda Prieto Ordaz
            """.strip()
        )
        st.divider()
        st.markdown("### Ficha del Conjunto de Datos")
        ds = metadata.get("dataset_summary", {})
        n_valid = ds.get("valid_rows", 41757)
        n_train = ds.get("train_rows", 33405)
        n_test = ds.get("test_rows", 8352)
        st.markdown(
            f"""
- **Fuente:** UCI Machine Learning Repository (*Beijing PM2.5 Data*, ID 381)
- **Periodo histórico:** 2010 – 2014 (medición horaria)
- **Registros válidos:** {n_valid:,} observaciones
- **Partición 80/20:** {n_train:,} Train / {n_test:,} Test (`RANDOM_STATE = 42`)
- **Inferencia en producción:** Modelos preentrenados en caché (`@st.cache_resource`), sin reentrenamiento.
            """.strip()
        )


def _render_header_and_problem() -> None:
    """Renderiza el encabezado institucional, título descriptivo y explicación no técnica."""
    st.markdown(
        """
        <div class="inst-banner">
            <strong>Universidad Autónoma de Chihuahua (UACH) — Facultad de Ingeniería</strong>
            &nbsp;|&nbsp; Ingeniería en Ciencias de la Computación
            <div class="inst-banner-grid">
                <span><strong>Materia:</strong> Data Science</span>
                <span><strong>Alumno:</strong> Alexis Aidan Leon Rodelas</span>
                <span><strong>Matrícula:</strong> 377171</span>
                <span><strong>Docente:</strong> Olanda Prieto Ordaz</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Predicción Horaria de Contaminación del Aire (PM2.5) en Pekín")

    st.markdown(
        """
        <div class="problem-card">
            <strong>¿Qué problema resuelve esta aplicación?</strong><br>
            Las partículas finas <strong>PM<sub>2.5</sub></strong> tienen un diámetro menor o igual a
            2.5 micrómetros (unas 30 veces más delgadas que un cabello humano), por lo que penetran
            profundamente en los pulmones y el sistema cardiovascular. En Pekín, su concentración cambia
            drásticamente hora a hora según la meteorología y la actividad estacional: el <strong>aire calmo
            o húmedo</strong> atrapa el smog cerca de la superficie, los <strong>vientos secos del noroeste (NW)</strong>
            limpian la cuenca atmosférica, y la <strong>temporada oficial de calefacción invernal (15 de noviembre
            al 15 de marzo)</strong> incrementa las emisiones.<br><br>
            Esta herramienta estima la <strong>concentración horaria esperada de PM<sub>2.5</sub> en microgramos
            por metro cúbico (µg/m³)</strong> utilizando únicamente condiciones meteorológicas y de calendario
            —sin requerir mediciones previas de sensores de calidad del aire— y traduce el valor numérico a una
            categoría sanitaria clara con recomendaciones prácticas para la población.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_air_quality_visual(pm25_pred: float, calidad_aire: dict[str, Any]) -> None:
    """Renderiza la tarjeta con insignia de color, barra visual 0-500+ µg/m³ y recomendación."""
    categoria = html.escape(str(calidad_aire.get("categoria", "Sin clasificar")))
    color_hex = str(calidad_aire.get("color_hex", PRIMARY_COLOR))
    rango_str = html.escape(str(calidad_aire.get("rango_ugm3", "")))
    descripcion = html.escape(str(calidad_aire.get("descripcion", "")))
    recomendacion = html.escape(str(calidad_aire.get("recomendacion", "")))

    # Color de texto legible sobre el badge
    text_on_badge = "#1e293b" if color_hex.lower() in {"#f1c40f", "#2ecc71"} else "#ffffff"

    # Posición porcentual en escala visual no lineal por tramos EPA (0 a 500 µg/m³)
    # Usamos escala proporcional acotada [0, 500] para fidelidad física directa
    clamped_scale = min(max(pm25_pred, 0.0), 500.0)
    pct_pos = (clamped_scale / 500.0) * 100.0
    # Evitar recorte de etiqueta en extremos
    pct_label = min(max(pct_pos, 4.0), 94.0)

    # Anchos proporcionales reales sobre una escala de 0 a 500 µg/m³:
    # 0-12 (2.4%), 12-35.4 (4.68%), 35.4-55.4 (4.0%), 55.4-150.4 (19.0%), 150.4-250.4 (20.0%), 250.4-500 (49.92%)
    st.markdown(
        f"""
        <div class="aqi-result-card" style="border-top: 5px solid {color_hex};">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <span style="font-size:0.86rem; color:#475569; font-weight:600; text-transform:uppercase;">
                        Clasificación de Calidad del Aire (Referencia EPA)
                    </span>
                    <div style="font-size:1.75rem; font-weight:800; color:#0f172a; margin-top:0.1rem;">
                        {pm25_pred:.2f} µg/m³
                    </div>
                </div>
                <div>
                    <span class="aqi-badge" style="background-color:{color_hex}; color:{text_on_badge};">
                        Nivel {calidad_aire.get('nivel', '')}: {categoria} ({rango_str})
                    </span>
                </div>
            </div>
            <div class="aqi-scale-wrapper">
                <div class="aqi-scale-bar" title="Escala de concentración PM2.5 de 0 a 500+ µg/m³">
                    <div style="width:2.40%; background:#2ecc71;" title="Buena (0–12)"></div>
                    <div style="width:4.68%; background:#f1c40f;" title="Moderada (12.1–35.4)"></div>
                    <div style="width:4.00%; background:#e67e22;" title="Grupos sensibles (35.5–55.4)"></div>
                    <div style="width:19.00%; background:#e74c3c;" title="Dañina (55.5–150.4)"></div>
                    <div style="width:20.00%; background:#8e44ad;" title="Muy dañina (150.5–250.4)"></div>
                    <div style="width:49.92%; background:#7f1d1d;" title="Peligrosa (>250.4)"></div>
                </div>
                <div class="aqi-marker-row">
                    <div class="aqi-marker" style="left:{pct_label:.2f}%;">
                        ▲ {pm25_pred:.1f} µg/m³
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#64748b;">
                    <span>0 µg/m³ (Buena)</span>
                    <span>55 µg/m³</span>
                    <span>150 µg/m³</span>
                    <span>250 µg/m³</span>
                    <span>500+ µg/m³ (Peligrosa)</span>
                </div>
            </div>
            <div class="health-callout" style="border-left: 4px solid {color_hex};">
                <div style="margin-bottom:0.35rem;"><strong>Interpretación ambiental:</strong> {descripcion}</div>
                <div><strong>Recomendación de salud:</strong> {recomendacion}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_three_models_comparison(
    resultados_todos: list[dict[str, Any]],
    selected_model_id: str,
    observed_pm25: float | None = None,
) -> plt.Figure:
    """Genera gráfica de barras horizontales comparando las predicciones de los 3 modelos."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2))

    nombres = [r["model_name"] for r in reversed(resultados_todos)]
    valores = [float(r["pm25_pred"]) for r in reversed(resultados_todos)]
    ids = [r["model_id"] for r in reversed(resultados_todos)]
    colores = [ACCENT_COLOR if m_id == selected_model_id else PRIMARY_COLOR for m_id in ids]

    bars = ax.barh(nombres, valores, color=colores, edgecolor="white", height=0.52, alpha=0.9)

    max_val = max(valores + ([observed_pm25] if observed_pm25 is not None else [50.0]))
    x_upper = max(max_val * 1.24, 45.0)
    ax.set_xlim(0, x_upper)

    for bar, val in zip(bars, valores):
        ax.text(
            val + x_upper * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} µg/m³",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#1e293b",
        )

    # Líneas de referencia EPA visibles dentro del rango actual
    umbrales_epa = [
        (12.0, "Buena/Mod (12)", "#2ecc71"),
        (35.4, "Sensibles (35.4)", "#d4a017"),
        (55.4, "Dañina (55.4)", "#e67e22"),
        (150.4, "Muy dañina (150.4)", "#c0392b"),
        (250.4, "Peligrosa (250.4)", "#7f1d1d"),
    ]
    for umbral, etiq, col_u in umbrales_epa:
        if umbral < x_upper * 0.92:
            ax.axvline(umbral, color=col_u, linestyle=":", linewidth=1.2, alpha=0.75)

    if observed_pm25 is not None:
        ax.axvline(
            observed_pm25,
            color="#111827",
            linestyle="--",
            linewidth=1.6,
            label=f"PM2.5 real observado ({observed_pm25:.1f} µg/m³)",
        )
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    ax.set_title(
        "Comparación Simultánea de los 3 Mejores Modelos Integrados",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Concentración predicha de PM2.5 (µg/m³)", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_all_models_benchmark(
    df_all: pd.DataFrame,
    integrated_names: set[str],
) -> plt.Figure:
    """Grafica el ranking completo de modelos evaluados en main.ipynb (RMSE_CV y R2_CV)."""
    df_plot = df_all.sort_values("RMSE_CV", ascending=False).reset_index(drop=True)
    n_models = len(df_plot)
    fig_height = max(5.2, 0.36 * n_models + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, fig_height), sharey=True)

    is_sel = [
        bool(row.get("Seleccionado_Top3")) or (str(row["Modelo"]) in integrated_names)
        for _, row in df_plot.iterrows()
    ]
    bar_colors = [ACCENT_COLOR if sel else PRIMARY_COLOR for sel in is_sel]

    # Panel 1: RMSE_CV
    axes[0].barh(df_plot["Modelo"], df_plot["RMSE_CV"], color=bar_colors, edgecolor="white", alpha=0.88)
    axes[0].set_title("RMSE en 5-Fold CV (menor es mejor)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("RMSE (µg/m³)", fontsize=10)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)
    max_rmse = float(df_plot["RMSE_CV"].max()) if not df_plot.empty else 100.0
    axes[0].set_xlim(0, max_rmse * 1.18)
    for i, val in enumerate(df_plot["RMSE_CV"]):
        if pd.notna(val):
            axes[0].text(val + max_rmse * 0.015, i, f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold")

    # Panel 2: R2_CV
    axes[1].barh(df_plot["Modelo"], df_plot["R2_CV"], color=bar_colors, edgecolor="white", alpha=0.88)
    axes[1].set_title("R² en 5-Fold CV (mayor es mejor)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Coeficiente de determinación R²", fontsize=10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)
    min_r2 = float(df_plot["R2_CV"].min()) if not df_plot.empty else 0.0
    axes[1].set_xlim(min(0.0, min_r2 - 0.05), 1.05)
    for i, val in enumerate(df_plot["R2_CV"]):
        if pd.notna(val):
            axes[1].text(max(float(val), 0.0) + 0.015, i, f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Comparativa Experimental en Validación Cruzada (Naranja: 3 Modelos Integrados en la App)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gestión de estado de sesión y presets
# ---------------------------------------------------------------------------


def _apply_values_to_session(values: dict[str, Any]) -> None:
    """Copia un diccionario de variables crudas a las claves de los controles en session_state."""
    st.session_state["input_year"] = int(round(float(values.get("year", DEFAULT_INPUT_VALUES["year"]))))
    st.session_state["input_month"] = int(round(float(values.get("month", DEFAULT_INPUT_VALUES["month"]))))
    st.session_state["input_day"] = int(round(float(values.get("day", DEFAULT_INPUT_VALUES["day"]))))
    st.session_state["input_hour"] = int(round(float(values.get("hour", DEFAULT_INPUT_VALUES["hour"]))))
    st.session_state["input_temp"] = float(values.get("TEMP", DEFAULT_INPUT_VALUES["TEMP"]))
    st.session_state["input_dewp"] = float(values.get("DEWP", DEFAULT_INPUT_VALUES["DEWP"]))
    st.session_state["input_pres"] = float(values.get("PRES", DEFAULT_INPUT_VALUES["PRES"]))
    cbwd_val = str(values.get("cbwd", DEFAULT_INPUT_VALUES["cbwd"]))
    st.session_state["input_cbwd"] = cbwd_val if cbwd_val in CBWD_OPTIONS else "NW"
    st.session_state["input_iws"] = float(values.get("Iws", DEFAULT_INPUT_VALUES["Iws"]))
    st.session_state["input_is"] = int(round(float(values.get("Is", DEFAULT_INPUT_VALUES["Is"]))))
    st.session_state["input_ir"] = int(round(float(values.get("Ir", DEFAULT_INPUT_VALUES["Ir"]))))


def _init_session_defaults(presets: list[dict[str, Any]]) -> None:
    """Inicializa los valores por defecto una sola vez por sesión sin sobrescribir cambios del usuario."""
    if "_inputs_initialized" not in st.session_state:
        initial_vals = presets[0]["values"] if presets and "values" in presets[0] else DEFAULT_INPUT_VALUES
        _apply_values_to_session(initial_vals)
        st.session_state["preset_selector"] = "personalizado"
        st.session_state["_last_preset_id"] = "personalizado"
        st.session_state["_inputs_initialized"] = True


# ---------------------------------------------------------------------------
# Pestaña 1: Predicción Interactiva
# ---------------------------------------------------------------------------


def _render_tab_prediccion(metadata: dict[str, Any], _modelos_cargados: dict[str, Any]) -> None:
    models_meta: list[dict[str, Any]] = sorted(
        metadata.get("models", []),
        key=lambda m: m.get("rank", 99),
    )
    presets_list: list[dict[str, Any]] = metadata.get("presets", [])

    model_by_id = {m["id"]: m for m in models_meta}
    model_ids = [m["id"] for m in models_meta]
    preset_by_id = {str(p.get("id", f"preset_{i}")): p for i, p in enumerate(presets_list)}
    preset_options = ["personalizado"] + list(preset_by_id.keys())

    _init_session_defaults(presets_list)

    col_inputs, col_results = st.columns([5, 7], gap="large")

    with col_inputs:
        st.subheader("1. Selección de Modelo y Escenario")

        selected_model_id = st.selectbox(
            "Modelo preentrenado para la predicción principal",
            options=model_ids,
            format_func=lambda mid: (
                f"#{model_ids.index(mid) + 1} — {model_by_id[mid]['name']} "
                f"(RMSE Test: {_fmt_num(_extract_metric(model_by_id[mid].get('test_metrics'), 'rmse'))} µg/m³)"
            ),
            key="model_selector",
            help="Elija cualquiera de los 3 mejores modelos seleccionados en el estudio experimental. No se realiza reentrenamiento.",
        )

        active_m = model_by_id[selected_model_id]
        rmse_cv = _extract_metric(active_m.get("cv_metrics"), "rmse")
        rmse_test = _extract_metric(active_m.get("test_metrics"), "rmse")
        mae_test = _extract_metric(active_m.get("test_metrics"), "mae")
        r2_test = _extract_metric(active_m.get("test_metrics"), "r2")
        size_mb = active_m.get("file_size_mb")
        desempeno_txt = html.escape(str(active_m.get("justification", {}).get("desempeno", "")))

        st.markdown(
            f"""
            <div class="model-summary-box">
                <strong>{html.escape(active_m['name'])}</strong>
                <span style="color:#475569;">({html.escape(str(active_m.get('family', 'Ensamble')))})</span><br>
                <span>
                    <strong>RMSE CV:</strong> {_fmt_num(rmse_cv)} µg/m³ &nbsp;|&nbsp;
                    <strong>RMSE Test:</strong> {_fmt_num(rmse_test)} µg/m³ &nbsp;|&nbsp;
                    <strong>MAE Test:</strong> {_fmt_num(mae_test)} µg/m³ &nbsp;|&nbsp;
                    <strong>R² Test:</strong> {_fmt_num(r2_test, 4)}
                    {f"&nbsp;|&nbsp; <strong>Tamaño:</strong> {float(size_mb):.2f} MB" if size_mb is not None else ""}
                </span>
                <div style="margin-top:0.35rem; color:#334155; font-size:0.86rem;">
                    {desempeno_txt}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_preset_id = st.selectbox(
            "Escenario meteorológico (plantillas rápidas o ingreso manual)",
            options=preset_options,
            format_func=lambda pid: (
                "Personalizado (ingreso manual)"
                if pid == "personalizado"
                else f"{preset_by_id[pid].get('name', pid)}"
            ),
            key="preset_selector",
            help="Cargue en un clic condiciones reales representativas de Pekín o mantenga 'Personalizado' para ajustar libremente cualquier variable.",
        )

        # Sincronizar controles solo cuando el usuario cambia el selector de escenario
        if selected_preset_id != st.session_state.get("_last_preset_id"):
            st.session_state["_last_preset_id"] = selected_preset_id
            if selected_preset_id in preset_by_id:
                _apply_values_to_session(preset_by_id[selected_preset_id]["values"])

        active_preset = preset_by_id.get(selected_preset_id)
        if active_preset is not None:
            obs_pm = active_preset.get("observed_pm25")
            obs_txt = f" (PM2.5 real registrado: {float(obs_pm):.1f} µg/m³)" if obs_pm is not None else ""
            st.caption(f"Escenario cargado: {active_preset.get('description', '')}{obs_txt}")

        st.subheader("2. Variables Temporales")
        c_y, c_m = st.columns(2)
        with c_y:
            year_val = st.number_input(
                "Año (year)",
                min_value=1990,
                max_value=2100,
                step=1,
                key="input_year",
                help="Año del registro. El entrenamiento comprende 2010–2014; otros años generan una advertencia de extrapolación.",
            )
        with c_m:
            month_val = st.selectbox(
                "Mes (month)",
                options=list(range(1, 13)),
                format_func=lambda m: f"{m:02d} — {MESES_ES[m]}",
                key="input_month",
                help="Mes del año (1 = Enero a 12 = Diciembre).",
            )

        c_d, c_h = st.columns(2)
        with c_d:
            day_val = st.number_input(
                "Día del mes (day)",
                min_value=1,
                max_value=31,
                step=1,
                key="input_day",
                help="Día calendario (1–31). Se valida que corresponda a una fecha real del mes y año elegidos.",
            )
        with c_h:
            hour_val = st.number_input(
                "Hora del día (hour, 0–23 h)",
                min_value=0,
                max_value=23,
                step=1,
                key="input_hour",
                help="Hora en formato 24 horas (0 = medianoche, 12 = mediodía, 23 = 11:00 p. m.).",
            )

        st.subheader("3. Variables Meteorológicas")
        c_t, c_dw = st.columns(2)
        with c_t:
            temp_val = st.number_input(
                "Temperatura ambiente — TEMP (°C)",
                min_value=-50.0,
                max_value=55.0,
                step=0.5,
                format="%.1f",
                key="input_temp",
                help="Temperatura del aire en grados Celsius (rango histórico en Pekín: -19 °C a 42 °C).",
            )
        with c_dw:
            dewp_val = st.number_input(
                "Punto de rocío — DEWP (°C)",
                min_value=-60.0,
                max_value=45.0,
                step=0.5,
                format="%.1f",
                key="input_dewp",
                help="Punto de rocío en °C. Por termodinámica atmosférica, DEWP no puede superar a TEMP.",
            )

        c_pr, c_wd = st.columns(2)
        with c_pr:
            pres_val = st.number_input(
                "Presión atmosférica — PRES (hPa)",
                min_value=940.0,
                max_value=1070.0,
                step=0.5,
                format="%.1f",
                key="input_pres",
                help="Presión barométrica en hectopascales (rango histórico en Pekín: 991 a 1046 hPa).",
            )
        with c_wd:
            cbwd_val = st.selectbox(
                "Dirección combinada del viento — cbwd",
                options=list(CBWD_OPTIONS.keys()),
                format_func=lambda k: CBWD_OPTIONS.get(k, k),
                key="input_cbwd",
                help="NW suele limpiar el aire; cv (calmo/variable) y SE favorecen acumulación de contaminantes.",
            )

        c_iws, c_is, c_ir = st.columns(3)
        with c_iws:
            iws_val = st.number_input(
                "Viento acum. — Iws (m/s)",
                min_value=0.0,
                max_value=700.0,
                step=0.5,
                format="%.2f",
                key="input_iws",
                help="Velocidad acumulada del viento en m/s desde el último cambio de dirección (mediana histórica: 5.37 m/s).",
            )
        with c_is:
            is_val = st.number_input(
                "Nieve acum. — Is (h)",
                min_value=0,
                max_value=72,
                step=1,
                key="input_is",
                help="Horas continuas acumuladas de caída de nieve (0 si no nieva).",
            )
        with c_ir:
            ir_val = st.number_input(
                "Lluvia acum. — Ir (h)",
                min_value=0,
                max_value=72,
                step=1,
                key="input_ir",
                help="Horas continuas acumuladas de precipitación pluvial (0 si no llueve).",
            )

        datos_actuales: dict[str, Any] = {
            "year": year_val,
            "month": month_val,
            "day": day_val,
            "hour": hour_val,
            "DEWP": dewp_val,
            "TEMP": temp_val,
            "PRES": pres_val,
            "cbwd": cbwd_val,
            "Iws": iws_val,
            "Is": is_val,
            "Ir": ir_val,
        }

        clicked_predict = st.button(
            "Calcular predicción de PM2.5",
            type="primary",
            key="btn_predecir",
            width="stretch",
        )

    # Procesamiento explícito de validación e inferencia al pulsar el botón
    if clicked_predict:
        es_valido, errores, advertencias = validar_entrada(datos_actuales)
        if not es_valido:
            st.session_state["last_pred_state"] = {
                "status": "error",
                "errores": errores,
                "datos": dict(datos_actuales),
            }
        else:
            try:
                res_principal = predecir(datos_actuales, selected_model_id)
                res_todos = predecir_todos(datos_actuales)
                obs_ref = (
                    float(active_preset["observed_pm25"])
                    if active_preset is not None
                    and active_preset.get("observed_pm25") is not None
                    and datos_actuales == {
                        "year": int(round(float(active_preset["values"]["year"]))),
                        "month": int(round(float(active_preset["values"]["month"]))),
                        "day": int(round(float(active_preset["values"]["day"]))),
                        "hour": int(round(float(active_preset["values"]["hour"]))),
                        "DEWP": float(active_preset["values"]["DEWP"]),
                        "TEMP": float(active_preset["values"]["TEMP"]),
                        "PRES": float(active_preset["values"]["PRES"]),
                        "cbwd": str(active_preset["values"]["cbwd"]),
                        "Iws": float(active_preset["values"]["Iws"]),
                        "Is": int(round(float(active_preset["values"]["Is"]))),
                        "Ir": int(round(float(active_preset["values"]["Ir"]))),
                    }
                    else None
                )
                st.session_state["last_pred_state"] = {
                    "status": "ok",
                    "datos": dict(datos_actuales),
                    "model_id": selected_model_id,
                    "resultado": res_principal,
                    "todos": res_todos,
                    "advertencias": advertencias,
                    "observed_pm25": obs_ref,
                }
            except ValueError as exc:
                st.session_state["last_pred_state"] = {
                    "status": "error",
                    "errores": [str(exc)],
                    "datos": dict(datos_actuales),
                }
            except Exception as exc:  # pragma: no cover - salvaguarda defensiva sin traceback
                st.session_state["last_pred_state"] = {
                    "status": "error",
                    "errores": [f"No fue posible completar la predicción: {exc}"],
                    "datos": dict(datos_actuales),
                }

    with col_results:
        st.subheader("Resultados de la Estimación Ambiental")
        pred_state = st.session_state.get("last_pred_state")

        if pred_state is None:
            st.info(
                "Configure las variables meteorológicas (o elija un escenario predefinido) y presione "
                "**«Calcular predicción de PM2.5»** para ejecutar los modelos preentrenados y visualizar "
                "el nivel de calidad del aire."
            )
            st.markdown("#### Niveles de Referencia de Calidad del Aire para PM2.5 (EPA)")
            df_epa_ref = pd.DataFrame(
                [
                    {
                        "Categoría": b["categoria"],
                        "Concentración PM2.5": b["rango"],
                        "Implicación Sanitaria": b["resumen"],
                    }
                    for b in EPA_BANDS
                ]
            )
            st.dataframe(df_epa_ref, width="stretch", hide_index=True)
            return

        if pred_state.get("status") == "error":
            st.error(
                "**No se pudo calcular la predicción debido a datos de entrada inválidos o físicamente incompatibles:**\n\n"
                + "\n".join(f"- {err}" for err in pred_state.get("errores", []))
            )
            st.caption(
                "Corrija los campos señalados en el panel izquierdo (por ejemplo, asegúrese de que la fecha exista "
                "en el calendario y que el punto de rocío DEWP sea menor o igual a la temperatura ambiente TEMP) y "
                "presione nuevamente «Calcular predicción de PM2.5»."
            )
            return

        # Estado válido ("ok")
        saved_datos = pred_state["datos"]
        todos_list: list[dict[str, Any]] = pred_state["todos"]
        todos_by_id = {r["model_id"]: r for r in todos_list}

        # Si las entradas siguen iguales y el usuario cambió el selector de modelo, reflejar ese modelo
        if datos_actuales == saved_datos and selected_model_id in todos_by_id:
            res_activo = todos_by_id[selected_model_id]
        else:
            res_activo = pred_state["resultado"]

        if datos_actuales != saved_datos:
            st.info(
                "Ha modificado los valores de entrada. Presione **«Calcular predicción de PM2.5»** para "
                "actualizar los resultados con las nuevas condiciones."
            )

        for adv in pred_state.get("advertencias", []):
            st.warning(adv)

        pm25_pred = float(res_activo["pm25_pred"])
        calidad = res_activo["calidad_aire"]
        obs_pm25 = pred_state.get("observed_pm25")

        m1, m2, m3 = st.columns(3)
        with m1:
            if obs_pm25 is not None:
                delta_str = f"{pm25_pred - obs_pm25:+.1f} vs. real ({obs_pm25:.1f})"
            else:
                delta_str = f"{pm25_pred - 72.0:+.1f} vs. mediana Pekín (72.0)"
            st.metric(
                label="PM2.5 Estimado",
                value=f"{pm25_pred:.2f} µg/m³",
                delta=delta_str,
                delta_color="inverse",
            )
        with m2:
            st.metric(
                label="Calidad del Aire (EPA)",
                value=str(calidad.get("categoria", "—")),
                delta=str(calidad.get("rango_ugm3", "")),
                delta_color="off",
            )
        with m3:
            rmse_m = _extract_metric(res_activo.get("test_metrics"), "rmse")
            st.metric(
                label="Modelo Seleccionado",
                value=str(res_activo["model_name"]).split("(")[0].strip(),
                delta=f"RMSE Test: {_fmt_num(rmse_m)} µg/m³",
                delta_color="off",
            )

        if res_activo.get("was_clamped"):
            st.caption(
                f"Nota técnica: la salida cruda del regresor ({float(res_activo['pm25_raw']):.2f} µg/m³) fue "
                "ligeramente negativa y se acotó físicamente a 0.00 µg/m³."
            )

        _render_air_quality_visual(pm25_pred, calidad)

        # Características derivadas por ingeniería determinística
        st.markdown("#### Variables Derivadas Automáticamente (`crear_caracteristicas`)")
        eng = res_activo.get("engineered_features", {})
        dow_val = _get_eng_value(eng, ("day_of_week", "DayOfWeek"), 0)
        wknd_val = _get_eng_value(eng, ("is_weekend", "IsWeekend"), 0)
        heat_val = _get_eng_value(eng, ("is_heating_season", "IsHeatingSeason"), 0)
        diff_val = _get_eng_value(
            eng,
            ("temp_dewp_diff", "TempDewpDiff"),
            float(saved_datos["TEMP"]) - float(saved_datos["DEWP"]),
        )
        prec_val = _get_eng_value(
            eng,
            ("is_precipitating", "IsPrecipitating"),
            int(float(saved_datos["Ir"]) > 0 or float(saved_datos["Is"]) > 0),
        )

        dia_nombre = DIAS_SEMANA_ES.get(int(dow_val), str(dow_val))
        df_eng_view = pd.DataFrame(
            [
                {
                    "Característica Derivada": "Día de la semana (DayOfWeek / day_of_week)",
                    "Valor Calculado": f"{dia_nombre} ({int(dow_val)})",
                    "Significado Físico / Urbano": "Captura variaciones semanales de tráfico e industria (0 = Lunes ... 6 = Domingo).",
                },
                {
                    "Característica Derivada": "Fin de semana (IsWeekend / is_weekend)",
                    "Valor Calculado": "Sí (1)" if int(wknd_val) == 1 else "No (0)",
                    "Significado Físico / Urbano": "Distingue sábados y domingos frente a jornadas laborales.",
                },
                {
                    "Característica Derivada": "Temporada de calefacción (IsHeatingSeason / is_heating_season)",
                    "Valor Calculado": "Activa (1)" if int(heat_val) == 1 else "Inactiva (0)",
                    "Significado Físico / Urbano": "Periodo oficial invernal (15 nov – 15 mar); eleva el promedio de PM2.5 un +22.2% en Pekín.",
                },
                {
                    "Característica Derivada": "Depresión del punto de rocío (TEMP − DEWP / temp_dewp_diff)",
                    "Valor Calculado": f"{float(diff_val):.1f} °C",
                    "Significado Físico / Urbano": "Proxy de sequedad del aire (r = -0.4391 con PM2.5; valores cercanos a 0 °C indican alta humedad y niebla).",
                },
                {
                    "Característica Derivada": "Presencia de precipitación (is_precipitating)",
                    "Valor Calculado": "Con lluvia o nieve (1)" if int(prec_val) == 1 else "Sin precipitación (0)",
                    "Significado Físico / Urbano": "Indica deposición húmeda activa (Ir > 0 o Is > 0).",
                },
            ]
        )
        st.dataframe(df_eng_view, width="stretch", hide_index=True)

        # Comparación entre los 3 mejores modelos
        st.markdown("#### Comparación en Vivo entre los 3 Mejores Modelos")
        fig_comp = _plot_three_models_comparison(
            todos_list,
            selected_model_id=res_activo["model_id"],
            observed_pm25=obs_pm25,
        )
        st.pyplot(fig_comp, width="stretch")
        plt.close(fig_comp)

        df_comp_3 = pd.DataFrame(
            [
                {
                    "Ranking": f"#{idx}",
                    "Modelo": r["model_name"],
                    "PM2.5 Predicho (µg/m³)": round(float(r["pm25_pred"]), 2),
                    "Categoría EPA": r["calidad_aire"]["categoria"],
                    "MAE Test (µg/m³)": round(_extract_metric(r.get("test_metrics"), "mae") or 0.0, 2),
                    "RMSE Test (µg/m³)": round(_extract_metric(r.get("test_metrics"), "rmse") or 0.0, 2),
                    "R² Test": round(_extract_metric(r.get("test_metrics"), "r2") or 0.0, 4),
                }
                for idx, r in enumerate(todos_list, start=1)
            ]
        )
        st.dataframe(df_comp_3, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Pestaña 2: Selección y Justificación de los 3 Mejores Modelos
# ---------------------------------------------------------------------------


def _render_tab_modelos(metadata: dict[str, Any]) -> None:
    st.subheader("Selección y Justificación Experimental de los 3 Mejores Modelos")
    st.markdown(
        """
        En el notebook principal del proyecto (`main.ipynb`) se entrenaron y compararon **19 configuraciones
        de regresión supervisada** pertenecientes a cuatro familias algorítmicas (Regresión Lineal / Polinomial con
        `ElasticNet`, Support Vector Regression `SVR`, `RandomForestRegressor`, y Ensambles de Boosting, `Voting` y
        `Stacking`), siguiendo estrictamente la secuencia metodológica sin fuga de información:
        partición **80 % Train (33,405 registros) / 20 % Test (8,352 registros)** con `RANDOM_STATE = 42`,
        optimización mediante **Validación Cruzada de 5 Folds (`KFold(n_splits=5, shuffle=True, random_state=42)`)**
        con criterio rector de **menor $\\text{RMSE}_{\\text{CV}}$**, y evaluación final única sobre el conjunto `Test`.
        """
    )

    models_meta: list[dict[str, Any]] = sorted(
        metadata.get("models", []),
        key=lambda m: m.get("rank", 99),
    )

    selection_summary = metadata.get("dataset_summary", {}).get("selection_summary")
    if selection_summary:
        st.info(f"**Síntesis de selección para producción web:** {selection_summary}")

    # Tabla resumen de métricas CV y Test de los 3 mejores modelos integrados
    resumen_top3_rows: list[dict[str, Any]] = []
    for idx, m in enumerate(models_meta, start=1):
        cv_m = m.get("cv_metrics", {})
        te_m = m.get("test_metrics", {})
        rmse_cv = _extract_metric(cv_m, "rmse")
        rmse_te = _extract_metric(te_m, "rmse")
        resumen_top3_rows.append(
            {
                "Posición": f"#{idx}",
                "Modelo": m["name"],
                "Familia": m.get("family", "Ensamble"),
                "MAE CV": round(_extract_metric(cv_m, "mae") or 0.0, 2),
                "MSE CV": round(_extract_metric(cv_m, "mse") or 0.0, 2),
                "RMSE CV": round(rmse_cv or 0.0, 2),
                "R² CV": round(_extract_metric(cv_m, "r2") or 0.0, 4),
                "MAE Test": round(_extract_metric(te_m, "mae") or 0.0, 2),
                "MSE Test": round(_extract_metric(te_m, "mse") or 0.0, 2),
                "RMSE Test": round(rmse_te or 0.0, 2),
                "R² Test": round(_extract_metric(te_m, "r2") or 0.0, 4),
                "Dif. RMSE (Test − CV)": round((rmse_te or 0.0) - (rmse_cv or 0.0), 2),
                "Archivo (.joblib)": f"{float(m.get('file_size_mb', 0.0)):.2f} MB",
            }
        )

    st.markdown("### 1. Métricas en Validación Cruzada (5-Fold CV) y Conjunto de Prueba (Test)")
    st.dataframe(pd.DataFrame(resumen_top3_rows), width="stretch", hide_index=True)

    st.markdown("### 2. Justificación Detallada por Criterio de la Rúbrica")
    for idx, m in enumerate(models_meta, start=1):
        just = m.get("justification", {})
        with st.expander(
            f"Modelo #{idx}: {m['name']} — {m.get('family', '')} ({float(m.get('file_size_mb', 0.0)):.2f} MB)",
            expanded=(idx == 1),
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="just-box">
                        <strong>Desempeño del modelo</strong><br>
                        {html.escape(str(just.get('desempeno', '')))}
                    </div>
                    <div class="just-box">
                        <strong>Capacidad para resolver el problema planteado</strong><br>
                        {html.escape(str(just.get('capacidad', '')))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="just-box">
                        <strong>Métricas de evaluación (MAE, MSE, RMSE y R²)</strong><br>
                        {html.escape(str(just.get('metricas', '')))}
                    </div>
                    <div class="just-box">
                        <strong>Viabilidad en una aplicación web en producción</strong><br>
                        {html.escape(str(just.get('viabilidad', '')))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("**Hiperparámetros óptimos entrenados (`main.ipynb`):**")
            st.json(m.get("best_params", {}), expanded=False)

    # Tabla y gráfica global de todos los modelos evaluados en main.ipynb
    df_all = _normalize_all_models_df(metadata)
    if not df_all.empty:
        st.markdown("### 3. Comparativa Global de Todos los Modelos Evaluados en `main.ipynb`")
        integrated_names = {str(m["name"]) for m in models_meta}
        fig_all = _plot_all_models_benchmark(df_all, integrated_names)
        st.pyplot(fig_all, width="stretch")
        plt.close(fig_all)

        cols_show = [
            c
            for c in (
                "Estado_App",
                "Modelo",
                "Familia",
                "MAE_CV",
                "MSE_CV",
                "RMSE_CV",
                "RMSE_std",
                "R2_CV",
                "RMSE_Train",
                "Tiempo_s",
                "Peso_MB",
                "Nota",
            )
            if c in df_all.columns and df_all[c].notna().any()
        ]
        st.dataframe(df_all[cols_show].round(4), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Pestaña 3: Pipeline de Preprocesamiento y Variables
# ---------------------------------------------------------------------------


def _render_tab_pipeline(metadata: dict[str, Any]) -> None:
    st.subheader("Arquitectura del Pipeline de Preprocesamiento e Ingeniería de Características")

    st.markdown(
        """
        Para garantizar **consistencia exacta entre entrenamiento e inferencia** y **cero fuga de información
        (*data leakage*)**, cada entrada del usuario atraviesa exactamente las mismas etapas determinísticas y el
        mismo `ColumnTransformer` ajustado exclusivamente sobre `X_train` en `main.ipynb`:
        """
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="just-box">
                <strong>Etapa 1: Validación Física y Calendario</strong><br>
                <code>validar_entrada(datos)</code> verifica que no existan nulos, que la fecha exista en el calendario
                gregoriano (rechazando p. ej. <code>2014-02-30</code> o <code>2014-04-31</code>), que
                <code>DEWP ≤ TEMP</code> por coherencia termodinámica, y que viento/precipitación no sean negativos.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="just-box">
                <strong>Etapa 2: Ingeniería Determinística</strong><br>
                <code>crear_caracteristicas(df)</code> construye 5 predictores físicos y temporales sin aprender
                parámetros estadísticos: <code>day_of_week</code>, <code>is_weekend</code>,
                <code>is_heating_season</code> (15 nov – 15 mar), <code>temp_dewp_diff</code> (<code>TEMP − DEWP</code>)
                e <code>is_precipitating</code>, pasando de 11 a 16 columnas.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="just-box">
                <strong>Etapa 3: ColumnTransformer + Estimador</strong><br>
                Dentro del <code>Pipeline</code> serializado (<code>.joblib</code>):<br>
                • <strong>15 numéricas:</strong> <code>SimpleImputer(median)</code> → <code>StandardScaler()</code>.<br>
                • <strong>1 categórica (<code>cbwd</code>):</strong> <code>SimpleImputer(most_frequent)</code> →
                <code>OneHotEncoder(drop="first", handle_unknown="ignore")</code> (18 variables finales).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Diccionario de Variables de Entrada (`RAW_COLUMNS`) y Dominios de Referencia")
    df_feats = _normalize_features_table(metadata)
    st.dataframe(df_feats, width="stretch", hide_index=True)

    st.markdown("### Escala Interpretativa de Calidad del Aire (Puntos de Corte EPA para PM2.5)")
    df_epa = pd.DataFrame(
        [
            {
                "Nivel": idx,
                "Categoría": b["categoria"],
                "Rango PM2.5 (µg/m³)": b["rango"],
                "Color Hex": b["color"],
                " Guía Sanitaria Resumida": b["resumen"],
            }
            for idx, b in enumerate(EPA_BANDS, start=1)
        ]
    )
    st.dataframe(df_epa, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Predicción de PM2.5 en Pekín | UACH Data Science",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_styles()

    metadata = obtener_metadatos()
    modelos_cargados = obtener_modelos()

    _render_sidebar_institutional(metadata)
    _render_header_and_problem()

    tab_pred, tab_models, tab_pipe = st.tabs(
        [
            "Predicción Interactiva",
            "Selección y Justificación de los 3 Modelos",
            "Pipeline de Preprocesamiento y Variables",
        ]
    )

    with tab_pred:
        _render_tab_prediccion(metadata, modelos_cargados)

    with tab_models:
        _render_tab_modelos(metadata)

    with tab_pipe:
        _render_tab_pipeline(metadata)


if __name__ == "__main__":
    main()
