"""Suite automatizada de pruebas para la aplicación Beijing PM2.5 (UACH - Data Science).

Verifica:
1. Carga sin reentrenamiento de metadata.json y los 3 modelos preentrenados (.joblib < 45 MB).
2. Consistencia de la ingeniería de características (crear_caracteristicas) y preparar_entrada_usuario.
3. Ejecución de predicciones (predecir y predecir_todos) y clasificación EPA de calidad del aire.
4. Validación exhaustiva de entradas (validar_entrada): fechas inválidas, incompatibilidad física
   DEWP > TEMP, valores negativos/fuera de rango, nulos/NaN y advertencias por extrapolación temporal.
5. Pruebas integrales de interfaz con streamlit.testing.v1.AppTest sobre app.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.model_loader import (  # noqa: E402
    load_all_models,
    load_metadata,
    load_model,
    predecir,
    predecir_todos,
)
from src.preprocessing import (  # noqa: E402
    CBWD_OPTIONS,
    RAW_COLUMNS,
    clasificar_calidad_aire,
    crear_caracteristicas,
    preparar_entrada_usuario,
    validar_entrada,
)

MAX_SINGLE_MODEL_MB = 45.0
MAX_TOTAL_MODELS_MB = 90.0


@pytest.fixture(scope="module")
def sample_valid_input() -> dict:
    """Escenario válido representativo dentro del rango histórico 2010-2014."""
    return {
        "year": 2014,
        "month": 1,
        "day": 15,
        "hour": 22,
        "DEWP": -2.0,
        "TEMP": -1.0,
        "PRES": 1022.0,
        "cbwd": "cv",
        "Iws": 1.8,
        "Is": 0,
        "Ir": 0,
    }


@pytest.fixture(scope="module")
def metadata() -> dict:
    return load_metadata()


@pytest.fixture(scope="module")
def all_models() -> dict:
    return load_all_models()


# ---------------------------------------------------------------------------
# 1. Metadatos, artefactos .joblib (< 45 MB) y carga sin reentrenamiento
# ---------------------------------------------------------------------------


def test_metadata_structure_and_top3_justifications(metadata: dict) -> None:
    """Verifica que metadata.json contenga los 3 mejores modelos, métricas y justificaciones."""
    assert "project_title" in metadata
    assert "dataset_summary" in metadata
    assert "models" in metadata
    assert "all_models_summary" in metadata
    assert "features" in metadata
    assert "presets" in metadata

    models_list = metadata["models"]
    assert len(models_list) == 3, "Deben integrarse exactamente los 3 mejores modelos."

    models_dir = APP_DIR / "models"
    total_mb = 0.0

    for idx, m in enumerate(models_list, start=1):
        for required_key in (
            "id",
            "name",
            "filename",
            "family",
            "best_params",
            "cv_metrics",
            "test_metrics",
            "file_size_mb",
            "justification",
        ):
            assert required_key in m, f"Falta clave '{required_key}' en el modelo #{idx}"

        # Verificar las 4 dimensiones de justificación exigidas por la rúbrica
        just = m["justification"]
        for j_key in ("desempeno", "metricas", "capacidad", "viabilidad"):
            assert j_key in just and len(str(just[j_key]).strip()) > 20

        # Verificar métricas CV y Test (MAE, MSE, RMSE, R2)
        for metric_group in ("cv_metrics", "test_metrics"):
            metrics_dict = m[metric_group]
            keys_lower = {k.lower(): v for k, v in metrics_dict.items()}
            assert any("mae" in k for k in keys_lower), f"Falta MAE en {metric_group}"
            assert any("mse" in k and "rmse" not in k for k in keys_lower), f"Falta MSE en {metric_group}"
            assert any("rmse" in k for k in keys_lower), f"Falta RMSE en {metric_group}"
            assert any("r2" in k for k in keys_lower), f"Falta R2 en {metric_group}"

        # Verificar existencia y límite estricto de tamaño en GitHub (< 45 MB por archivo)
        joblib_path = models_dir / m["filename"]
        assert joblib_path.exists(), f"No existe el archivo serializado {joblib_path}"
        size_mb = joblib_path.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        assert size_mb < MAX_SINGLE_MODEL_MB, (
            f"El modelo {m['filename']} pesa {size_mb:.2f} MB y excede {MAX_SINGLE_MODEL_MB} MB"
        )

    assert total_mb < MAX_TOTAL_MODELS_MB, (
        f"El peso total de models/ ({total_mb:.2f} MB) excede {MAX_TOTAL_MODELS_MB} MB"
    )
    assert len(metadata["presets"]) >= 3


def test_models_load_already_fitted_without_retraining(
    metadata: dict, all_models: dict, sample_valid_input: dict
) -> None:
    """Confirma que los 3 modelos ya están entrenados y NUNCA llaman a .fit() al predecir."""
    assert len(all_models) == 3
    for m_info in metadata["models"]:
        model_id = m_info["id"]
        assert model_id in all_models
        pipeline = load_model(model_id)
        check_is_fitted(pipeline)

        with patch.object(
            type(pipeline),
            "fit",
            side_effect=AssertionError("PROHIBIDO reentrenar el modelo en inferencia"),
        ):
            res = predecir(sample_valid_input, model_id)
            assert isinstance(res["pm25_pred"], float)


# ---------------------------------------------------------------------------
# 2. Ingeniería de características y preparación de entrada
# ---------------------------------------------------------------------------


def _get_col_ci(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    """Obtiene una columna verificando variantes snake_case o CamelCase."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"Ninguna de las columnas {candidates} existe en {list(df.columns)}")


def test_crear_caracteristicas_consistency() -> None:
    """Verifica cálculo exacto de día de semana, fin de semana y temporada de calefacción."""
    filas = pd.DataFrame(
        [
            # Miércoles 15 de enero de 2014 -> calefacción activa (1), entre semana (0)
            {"year": 2014, "month": 1, "day": 15, "hour": 10, "DEWP": -10, "TEMP": -2.0, "PRES": 1025.0, "cbwd": "NW", "Iws": 20.0, "Is": 0, "Ir": 0},
            # Viernes 14 de noviembre de 2014 -> antes de calefacción (0), viernes (4), no fin de semana (0)
            {"year": 2014, "month": 11, "day": 14, "hour": 12, "DEWP": -2, "TEMP": 8.0, "PRES": 1018.0, "cbwd": "SE", "Iws": 5.0, "Is": 0, "Ir": 0},
            # Sábado 15 de noviembre de 2014 -> inicia calefacción (1), sábado (5), fin de semana (1)
            {"year": 2014, "month": 11, "day": 15, "hour": 12, "DEWP": -1, "TEMP": 7.0, "PRES": 1019.0, "cbwd": "cv", "Iws": 2.0, "Is": 0, "Ir": 0},
            # Sábado 15 de marzo de 2014 -> último día de calefacción (1), sábado (5), fin de semana (1)
            {"year": 2014, "month": 3, "day": 15, "hour": 15, "DEWP": 1, "TEMP": 14.0, "PRES": 1012.0, "cbwd": "NE", "Iws": 10.0, "Is": 0, "Ir": 2},
            # Domingo 16 de marzo de 2014 -> fuera de calefacción (0), domingo (6), fin de semana (1)
            {"year": 2014, "month": 3, "day": 16, "hour": 16, "DEWP": 2, "TEMP": 16.0, "PRES": 1010.0, "cbwd": "NW", "Iws": 15.0, "Is": 0, "Ir": 0},
        ]
    )
    df_fe = crear_caracteristicas(filas)

    dow = _get_col_ci(df_fe, ("day_of_week", "DayOfWeek")).tolist()
    weekend = _get_col_ci(df_fe, ("is_weekend", "IsWeekend")).tolist()
    heating = _get_col_ci(df_fe, ("is_heating_season", "IsHeatingSeason")).tolist()

    assert dow == [2, 4, 5, 5, 6]
    assert weekend == [0, 0, 1, 1, 1]
    assert heating == [1, 0, 1, 1, 0]

    if "temp_dewp_diff" in df_fe.columns:
        assert np.allclose(df_fe["temp_dewp_diff"], filas["TEMP"] - filas["DEWP"])
    if "is_precipitating" in df_fe.columns:
        assert df_fe["is_precipitating"].tolist() == [0, 0, 0, 1, 0]


def test_preparar_entrada_usuario_valid_and_invalid(sample_valid_input: dict) -> None:
    """Verifica que preparar_entrada_usuario genere 1 fila lista para predict y falle en inválidos."""
    assert RAW_COLUMNS == ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    assert set(CBWD_OPTIONS.keys()) == {"SE", "NW", "cv", "NE"}

    df_ready = preparar_entrada_usuario(sample_valid_input)
    assert isinstance(df_ready, pd.DataFrame)
    assert len(df_ready) == 1
    for col in RAW_COLUMNS:
        assert col in df_ready.columns

    bad_input = dict(sample_valid_input, DEWP=15.0, TEMP=5.0)
    with pytest.raises(ValueError):
        preparar_entrada_usuario(bad_input)


# ---------------------------------------------------------------------------
# 3. Predicciones en los 3 modelos y clasificación de calidad del aire
# ---------------------------------------------------------------------------


def test_predecir_and_predecir_todos_on_presets(metadata: dict) -> None:
    """Ejecuta predecir y predecir_todos sobre todos los presets verificando salidas no negativas."""
    model_ids = [m["id"] for m in metadata["models"]]

    for preset in metadata["presets"]:
        resultados = predecir_todos(preset["values"])
        assert len(resultados) == 3
        assert [r["model_id"] for r in resultados] == model_ids

        for r in resultados:
            for key in (
                "model_id",
                "model_name",
                "pm25_pred",
                "pm25_raw",
                "was_clamped",
                "calidad_aire",
                "engineered_features",
                "advertencias",
                "cv_metrics",
                "test_metrics",
            ):
                assert key in r
            assert r["pm25_pred"] >= 0.0
            assert not math.isnan(r["pm25_pred"])
            assert r["calidad_aire"]["categoria"] in {
                "Buena",
                "Moderada",
                "Dañina para grupos sensibles",
                "Dañina",
                "Muy dañina",
                "Peligrosa",
            }


def test_physical_coherence_stagnant_winter_vs_nw_wind() -> None:
    """Verifica coherencia física: noche invernal calma supera ampliamente a día ventoso NW."""
    invierno_calmo = {
        "year": 2014, "month": 1, "day": 15, "hour": 22,
        "DEWP": -2.0, "TEMP": -1.0, "PRES": 1022.0, "cbwd": "cv",
        "Iws": 1.8, "Is": 0, "Ir": 0,
    }
    invierno_viento_nw = {
        "year": 2014, "month": 1, "day": 16, "hour": 14,
        "DEWP": -18.0, "TEMP": -3.0, "PRES": 1034.0, "cbwd": "NW",
        "Iws": 120.0, "Is": 0, "Ir": 0,
    }
    preds_calmo = predecir_todos(invierno_calmo)
    preds_nw = predecir_todos(invierno_viento_nw)

    for r_calmo, r_nw in zip(preds_calmo, preds_nw):
        assert r_calmo["pm25_pred"] > r_nw["pm25_pred"] + 80.0


@pytest.mark.parametrize(
    ("pm25_val", "expected_cat", "expected_hex"),
    [
        (0.0, "Buena", "#2ecc71"),
        (12.0, "Buena", "#2ecc71"),
        (25.0, "Moderada", "#f1c40f"),
        (45.0, "Dañina para grupos sensibles", "#e67e22"),
        (100.0, "Dañina", "#e74c3c"),
        (200.0, "Muy dañina", "#8e44ad"),
        (320.0, "Peligrosa", "#7f1d1d"),
    ],
)
def test_clasificar_calidad_aire_breakpoints(
    pm25_val: float, expected_cat: str, expected_hex: str
) -> None:
    info = clasificar_calidad_aire(pm25_val)
    for k in ("categoria", "nivel", "color_hex", "rango_ugm3", "descripcion", "recomendacion"):
        assert k in info
    assert info["categoria"] == expected_cat
    assert info["color_hex"].lower() == expected_hex.lower()


# ---------------------------------------------------------------------------
# 4. Validación de entradas (validar_entrada)
# ---------------------------------------------------------------------------


def test_validar_entrada_accepts_valid_and_warns_on_extrapolated_year(
    sample_valid_input: dict,
) -> None:
    es_valido, errores, advertencias = validar_entrada(sample_valid_input)
    assert es_valido is True
    assert errores == []
    assert advertencias == []

    # Año bisiesto válido (29 de febrero de 2012)
    bisiesto_ok = dict(sample_valid_input, year=2012, month=2, day=29)
    ok_leap, err_leap, _ = validar_entrada(bisiesto_ok)
    assert ok_leap is True
    assert err_leap == []

    # Año fuera del rango de entrenamiento (2026) -> válido pero con advertencia no bloqueante
    futuro_2026 = dict(sample_valid_input, year=2026)
    ok_2026, err_2026, adv_2026 = validar_entrada(futuro_2026)
    assert ok_2026 is True
    assert err_2026 == []
    assert len(adv_2026) >= 1


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"DEWP": 15.0, "TEMP": 10.0}, "DEWP mayor que TEMP"),
        ({"year": 2014, "month": 2, "day": 30}, "Fecha inexistente 2014-02-30"),
        ({"year": 2014, "month": 4, "day": 31}, "Fecha inexistente 2014-04-31"),
        ({"year": 2014, "month": 2, "day": 29}, "Año no bisiesto 2014-02-29"),
        ({"Iws": -2.5}, "Iws negativo"),
        ({"Is": -1}, "Is negativo"),
        ({"Ir": -3}, "Ir negativo"),
        ({"hour": 25}, "Hora fuera de rango 0..23"),
        ({"hour": -1}, "Hora negativa"),
        ({"PRES": 500.0}, "Presión atmosférica fuera de rango"),
        ({"PRES": 1100.0}, "Presión atmosférica excesiva"),
        ({"TEMP": 75.0}, "Temperatura fuera de rango"),
        ({"DEWP": -75.0}, "Punto de rocío fuera de rango"),
        ({"year": 1850}, "Año fuera de rango"),
        ({"cbwd": "SUR"}, "Dirección de viento inválida"),
        ({"TEMP": None}, "Valor None"),
        ({"DEWP": float("nan")}, "Valor NaN"),
        ({"Iws": "no_numerico"}, "Texto no numérico"),
    ],
)
def test_validar_entrada_rejects_invalid_cases(
    sample_valid_input: dict, overrides: dict, reason: str
) -> None:
    bad = dict(sample_valid_input, **overrides)
    es_valido, errores, _ = validar_entrada(bad)
    assert es_valido is False, f"Debió rechazar el caso: {reason}"
    assert len(errores) >= 1, f"Debió reportar mensaje de error en español para: {reason}"


# ---------------------------------------------------------------------------
# 5. Pruebas integrales de interfaz con Streamlit AppTest
# ---------------------------------------------------------------------------


def _collect_all_rendered_text(at: AppTest) -> str:
    """Concatena todo el texto visible renderizado en el árbol principal y sidebar."""
    chunks: list[str] = []
    for attr in ("title", "header", "subheader", "markdown", "caption", "info", "success", "warning", "error", "text"):
        for elem in getattr(at, attr, []):
            val = getattr(elem, "value", None) or getattr(elem, "body", None)
            if val:
                chunks.append(str(val))
    for elem in getattr(at, "metric", []):
        if getattr(elem, "label", None):
            chunks.append(str(elem.label))
        if getattr(elem, "value", None):
            chunks.append(str(elem.value))
    return "\n".join(chunks)


def _find_predict_button(at: AppTest):
    """Localiza el botón principal de predicción en la aplicación."""
    for btn in at.button:
        label = (btn.label or "").lower()
        if any(tok in label for tok in ("predecir", "predicción", "prediccion", "calcular", "estimar")):
            return btn
    assert len(at.button) >= 1, "No se encontró ningún botón de predicción en app.py"
    return at.button[0]


def _find_model_selector(at: AppTest, metadata: dict):
    """Localiza el control (selectbox o radio) que permite elegir entre los 3 mejores modelos."""
    model_ids = {m["id"] for m in metadata["models"]}
    model_names = {m["name"] for m in metadata["models"]}

    for widget_list in (at.selectbox, at.radio):
        for w in widget_list:
            opts = list(w.options)
            if len(opts) == 3 and (
                set(map(str, opts)) == model_ids
                or set(map(str, opts)) == model_names
                or any(any(m_name in str(o) for m_name in model_names) for o in opts)
            ):
                return w
    return None


def test_streamlit_app_boots_and_displays_institutional_info() -> None:
    """Verifica carga limpia de app.py y presencia obligatoria de datos institucionales UACH."""
    app_path = str(APP_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)

    assert len(at.exception) == 0, f"Excepciones al iniciar app.py: {at.exception}"

    full_text = _collect_all_rendered_text(at)
    assert ("Universidad Autónoma de Chihuahua" in full_text) or ("UACH" in full_text)
    assert "Data Science" in full_text
    assert "Alexis Aidan Leon Rodelas" in full_text
    assert "377171" in full_text
    assert "Olanda Prieto Ordaz" in full_text


def test_streamlit_app_switch_3_models_and_predict(metadata: dict) -> None:
    """Cambia entre los 3 mejores modelos y ejecuta la predicción sin errores."""
    app_path = str(APP_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert len(at.exception) == 0

    selector = _find_model_selector(at, metadata)
    assert selector is not None, "No se encontró selector para los 3 modelos en app.py"

    options = list(selector.options)
    assert len(options) == 3

    for opt in options:
        selector.select(opt)
        at.run(timeout=30)
        assert len(at.exception) == 0

        btn = _find_predict_button(at)
        btn.click()
        at.run(timeout=30)
        assert len(at.exception) == 0
        assert len(at.error) == 0

        rendered = _collect_all_rendered_text(at)
        assert ("µg/m³" in rendered) or ("ug/m3" in rendered.lower()) or ("pm2.5" in rendered.lower())


def test_streamlit_app_invalid_input_shows_error_cleanly() -> None:
    """Ingresa datos físicamente incompatibles (DEWP > TEMP) y fecha inválida (30/02) verificando st.error."""
    app_path = str(APP_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Caso 1: Incompatibilidad física DEWP > TEMP
    dewp_widget = None
    temp_widget = None
    for num in at.number_input:
        ident = f"{num.key or ''} {num.label or ''}".lower()
        if ("dewp" in ident or "rocío" in ident or "rocio" in ident) and dewp_widget is None:
            dewp_widget = num
        elif ("temp" in ident or "temperatura" in ident) and "roc" not in ident and temp_widget is None:
            temp_widget = num

    assert dewp_widget is not None and temp_widget is not None, (
        "No se localizaron controles independientes para TEMP y DEWP en app.py"
    )
    temp_widget.set_value(5.0)
    dewp_widget.set_value(20.0)
    at.run(timeout=30)

    btn = _find_predict_button(at)
    btn.click()
    at.run(timeout=30)

    assert len(at.exception) == 0, f"Excepción no controlada ante DEWP > TEMP: {at.exception}"
    assert len(at.error) >= 1, "Se esperaba al menos un mensaje st.error al ingresar DEWP > TEMP"

    # Caso 2: Fecha inexistente en el calendario (30 de febrero de 2014)
    at_date = AppTest.from_file(app_path, default_timeout=30)
    at_date.run(timeout=30)
    assert len(at_date.exception) == 0

    at_date.number_input(key="input_year").set_value(2014)
    at_date.selectbox(key="input_month").select(2)
    at_date.number_input(key="input_day").set_value(30)
    at_date.run(timeout=30)

    _find_predict_button(at_date).click()
    at_date.run(timeout=30)

    assert len(at_date.exception) == 0, f"Excepción no controlada ante fecha 2014-02-30: {at_date.exception}"
    assert len(at_date.error) >= 1, "Se esperaba st.error al ingresar la fecha inválida 2014-02-30"


def test_streamlit_app_extrapolation_warning_and_presets(metadata: dict) -> None:
    """Verifica advertencia st.warning en año fuera de rango (2026) y carga de escenarios reales."""
    app_path = str(APP_DIR / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Probar carga de un escenario predefinido desde el selector
    preset_id = metadata["presets"][-1]["id"]
    at.selectbox(key="preset_selector").select(preset_id)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Extrapolación al año 2026 (físicamente válido, fuera de 2010-2014)
    at.number_input(key="input_year").set_value(2026)
    at.number_input(key="input_day").set_value(15)
    at.run(timeout=30)

    _find_predict_button(at).click()
    at.run(timeout=30)

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.warning) >= 1, "Se esperaba st.warning al extrapolar para el año 2026"

