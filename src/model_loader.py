"""Carga en memoria y ejecución de inferencia sobre los 3 mejores Pipelines preentrenados.

Garantiza que los modelos serializados en `models/*.joblib` se carguen una única vez
desde disco sin reentrenamiento en tiempo de predicción.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.preprocessing import (
    clasificar_calidad_aire,
    preparar_entrada_usuario,
    validar_entrada,
)

BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"

_METADATA_CACHE: dict[str, Any] | None = None
_MODELS_CACHE: dict[str, Any] = {}


def load_metadata() -> dict[str, Any]:
    """Carga y retorna el diccionario de metadatos `models/metadata.json`."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo de metadatos en: {METADATA_PATH}")
        with METADATA_PATH.open("r", encoding="utf-8") as fh:
            _METADATA_CACHE = json.load(fh)
    return _METADATA_CACHE


def _get_model_meta(model_id: str) -> dict[str, Any]:
    """Obtiene el bloque de metadatos de un modelo por su `id`."""
    metadata = load_metadata()
    for item in metadata.get("models", []):
        if item.get("id") == model_id:
            return item
    disponibles = [m.get("id") for m in metadata.get("models", [])]
    raise ValueError(f"Identificador de modelo desconocido: {model_id!r}. Disponibles: {disponibles}")


def load_model(model_id: str) -> Any:
    """Carga (con caché en memoria) el Pipeline preentrenado correspondiente a `model_id`."""
    if model_id not in _MODELS_CACHE:
        model_meta = _get_model_meta(model_id)
        model_path = MODELS_DIR / model_meta["filename"]
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {model_path}")
        _MODELS_CACHE[model_id] = joblib.load(model_path)
    return _MODELS_CACHE[model_id]


def load_all_models() -> dict[str, Any]:
    """Carga los 3 modelos integrados en orden de ranking y retorna `{model_id: pipeline}`."""
    metadata = load_metadata()
    modelos_ordenados = sorted(metadata.get("models", []), key=lambda m: m.get("rank", 99))
    return {m["id"]: load_model(m["id"]) for m in modelos_ordenados}


def predecir(datos: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Ejecuta validación, ingeniería de características e inferencia con el modelo `model_id`."""
    es_valido, errores, advertencias = validar_entrada(datos)
    if not es_valido:
        raise ValueError("Datos de entrada inválidos: " + " | ".join(errores))

    model_meta = _get_model_meta(model_id)
    pipeline = load_model(model_id)
    df_fe = preparar_entrada_usuario(datos)

    pm25_raw = float(pipeline.predict(df_fe)[0])
    was_clamped = bool(pm25_raw < 0.0)
    pm25_pred = max(0.0, pm25_raw)
    calidad_aire = clasificar_calidad_aire(pm25_pred)

    fila_fe = df_fe.iloc[0]
    engineered_features = {
        "day_of_week": int(fila_fe["day_of_week"]),
        "is_weekend": int(fila_fe["is_weekend"]),
        "is_heating_season": int(fila_fe["is_heating_season"]),
        "temp_dewp_diff": round(float(fila_fe["temp_dewp_diff"]), 4),
        "is_precipitating": int(fila_fe["is_precipitating"]),
    }

    return {
        "model_id": str(model_meta["id"]),
        "model_name": str(model_meta["name"]),
        "pm25_pred": float(pm25_pred),
        "pm25_raw": float(pm25_raw),
        "was_clamped": was_clamped,
        "calidad_aire": calidad_aire,
        "engineered_features": engineered_features,
        "advertencias": list(advertencias),
        "cv_metrics": dict(model_meta["cv_metrics"]),
        "test_metrics": dict(model_meta["test_metrics"]),
    }


def predecir_todos(datos: dict[str, Any]) -> list[dict[str, Any]]:
    """Ejecuta `predecir` sobre los 3 modelos integrados siguiendo su orden de ranking (1..3)."""
    metadata = load_metadata()
    modelos_ordenados = sorted(metadata.get("models", []), key=lambda m: m.get("rank", 99))
    return [predecir(datos, m["id"]) for m in modelos_ordenados]
