"""Módulo de validación, ingeniería de características y clasificación de calidad del aire.

Reproduce exactamente la transformación determinística entrenada en `Proyecto Final/main.ipynb`
para alimentar los Pipelines preentrenados sin fuga de información ni reentrenamiento.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd

RAW_COLUMNS: list[str] = [
    "year",
    "month",
    "day",
    "hour",
    "DEWP",
    "TEMP",
    "PRES",
    "cbwd",
    "Iws",
    "Is",
    "Ir",
]

ENGINEERED_COLUMNS: list[str] = [
    "day_of_week",
    "is_weekend",
    "is_heating_season",
    "temp_dewp_diff",
    "is_precipitating",
]

CBWD_OPTIONS: dict[str, str] = {
    "SE": "Sureste (SE)",
    "NW": "Noroeste (NW)",
    "cv": "Calmo / Variable (cv)",
    "NE": "Noreste (NE)",
}

# Límites físicos admisibles para validación estricta (errores bloqueantes)
PHYSICAL_LIMITS: dict[str, tuple[float, float]] = {
    "year": (1990, 2100),
    "month": (1, 12),
    "day": (1, 31),
    "hour": (0, 23),
    "DEWP": (-60.0, 45.0),
    "TEMP": (-50.0, 55.0),
    "PRES": (940.0, 1070.0),
}

# Rangos históricos observados en el dataset Beijing PM2.5 (2010–2014) para advertencias
HISTORICAL_RANGES: dict[str, tuple[float, float]] = {
    "year": (2010, 2014),
    "DEWP": (-40.0, 28.0),
    "TEMP": (-19.0, 42.0),
    "PRES": (991.0, 1046.0),
    "Iws": (0.45, 585.60),
    "Is": (0.0, 27.0),
    "Ir": (0.0, 36.0),
}

FIELD_LABELS_ES: dict[str, str] = {
    "year": "Año (year)",
    "month": "Mes (month)",
    "day": "Día (day)",
    "hour": "Hora (hour)",
    "DEWP": "Punto de rocío (DEWP)",
    "TEMP": "Temperatura ambiente (TEMP)",
    "PRES": "Presión atmosférica (PRES)",
    "cbwd": "Dirección del viento (cbwd)",
    "Iws": "Velocidad acumulada del viento (Iws)",
    "Is": "Horas acumuladas de nieve (Is)",
    "Ir": "Horas acumuladas de lluvia (Ir)",
}


def crear_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables temporales y físicas determinísticas idénticas a `main.ipynb`.

    Genera las 5 características derivadas esperadas por el `ColumnTransformer`:
    - `day_of_week`: día de la semana (0 = lunes ... 6 = domingo)
    - `is_weekend`: indicador binario de fin de semana (sábado o domingo)
    - `is_heating_season`: temporada oficial de calefacción invernal en Pekín (15 nov – 15 mar)
    - `temp_dewp_diff`: depresión del punto de rocío (`TEMP - DEWP`), proxy de sequedad del aire
    - `is_precipitating`: indicador binario de lluvia o nieve activa (`Ir > 0` o `Is > 0`)
    """
    X_out = df.copy()

    # 1. Variables de calendario a partir de year, month y day
    fechas = pd.to_datetime(X_out[["year", "month", "day"]])
    X_out["day_of_week"] = fechas.dt.day_of_week
    X_out["is_weekend"] = X_out["day_of_week"].isin([5, 6]).astype(int)

    # 2. Temporada oficial de calefacción invernal en Pekín (15 nov - 15 mar)
    X_out["is_heating_season"] = (
        ((X_out["month"] == 11) & (X_out["day"] >= 15))
        | (X_out["month"].isin([12, 1, 2]))
        | ((X_out["month"] == 3) & (X_out["day"] <= 15))
    ).astype(int)

    # 3. Depresión del punto de rocío (proxy físico de sequedad / humedad relativa)
    X_out["temp_dewp_diff"] = X_out["TEMP"] - X_out["DEWP"]

    # 4. Indicador binario de precipitación (lluvia o nieve)
    X_out["is_precipitating"] = ((X_out["Ir"] > 0) | (X_out["Is"] > 0)).astype(int)

    return X_out


def _parse_numeric(val: Any) -> float | None:
    """Convierte un valor escalar numérico válido a float, rechazando None, bool, NaN e inf."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float, np.number)):
        fval = float(val)
    elif isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return None
        try:
            fval = float(stripped)
        except ValueError:
            return None
    else:
        return None

    if math.isnan(fval) or math.isinf(fval):
        return None
    return fval


def validar_entrada(datos: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Valida un diccionario de entrada cruda antes de ejecutar la predicción.

    Retorna `(es_valido, errores, advertencias)` con mensajes claros en español:
    - `errores`: violaciones físicas, de tipo o de calendario que impiden predecir.
    - `advertencias`: valores físicamente válidos pero fuera del dominio histórico 2010–2014.
    """
    errores: list[str] = []
    advertencias: list[str] = []

    if not isinstance(datos, dict):
        return False, ["Los datos de entrada deben proporcionarse como un diccionario."], []

    # 1. Verificar presencia de todas las columnas requeridas
    for col in RAW_COLUMNS:
        if col not in datos or datos[col] is None:
            etiqueta = FIELD_LABELS_ES.get(col, col)
            errores.append(f"El campo '{etiqueta}' es obligatorio y no puede ser nulo.")

    if errores:
        return False, errores, advertencias

    # 2. Validar variable categórica cbwd
    cbwd_val = datos["cbwd"]
    if not isinstance(cbwd_val, str) or cbwd_val not in CBWD_OPTIONS:
        opciones_str = ", ".join(CBWD_OPTIONS.keys())
        errores.append(
            f"La dirección del viento (cbwd = {cbwd_val!r}) no es válida. "
            f"Opciones permitidas: {opciones_str}."
        )

    # 3. Validar variables numéricas (tipos y finitud)
    parsed_num: dict[str, float] = {}
    num_cols = [c for c in RAW_COLUMNS if c != "cbwd"]
    for col in num_cols:
        fval = _parse_numeric(datos[col])
        if fval is None:
            etiqueta = FIELD_LABELS_ES.get(col, col)
            errores.append(f"El campo '{etiqueta}' debe ser un número válido (recibido: {datos[col]!r}).")
        else:
            parsed_num[col] = fval

    # 4. Validar variables temporales enteras y fecha real de calendario
    int_time_cols = ["year", "month", "day", "hour"]
    valid_time_ints: dict[str, int] = {}
    for col in int_time_cols:
        if col in parsed_num:
            fval = parsed_num[col]
            etiqueta = FIELD_LABELS_ES[col]
            if not fval.is_integer():
                errores.append(f"El campo '{etiqueta}' debe ser un número entero (recibido: {datos[col]!r}).")
                continue
            ival = int(fval)
            min_lim, max_lim = PHYSICAL_LIMITS[col]
            if ival < int(min_lim) or ival > int(max_lim):
                errores.append(
                    f"El campo '{etiqueta}' ({ival}) está fuera del rango permitido ({int(min_lim)} a {int(max_lim)})."
                )
            else:
                valid_time_ints[col] = ival

    # Verificar existencia en el calendario gregoriano (p. ej. rechazar 2014-02-30 o 2014-04-31)
    if all(k in valid_time_ints for k in ("year", "month", "day")):
        y_int = valid_time_ints["year"]
        m_int = valid_time_ints["month"]
        d_int = valid_time_ints["day"]
        try:
            datetime.date(y_int, m_int, d_int)
        except ValueError:
            errores.append(
                f"La fecha {y_int:04d}-{m_int:02d}-{d_int:02d} no es una fecha válida del calendario "
                f"(verifique el número de días del mes seleccionado)."
            )

    # 5. Validar rangos físicos meteorológicos
    if "PRES" in parsed_num:
        pres = parsed_num["PRES"]
        if not (940.0 <= pres <= 1070.0):
            errores.append(
                f"La presión atmosférica (PRES = {pres:.1f} hPa) está fuera del rango físico terrestre "
                f"permitido (940.0 a 1070.0 hPa)."
            )

    if "TEMP" in parsed_num:
        temp = parsed_num["TEMP"]
        if not (-50.0 <= temp <= 55.0):
            errores.append(
                f"La temperatura ambiente (TEMP = {temp:.1f} °C) está fuera del rango permitido (-50.0 a 55.0 °C)."
            )

    if "DEWP" in parsed_num:
        dewp = parsed_num["DEWP"]
        if not (-60.0 <= dewp <= 45.0):
            errores.append(
                f"El punto de rocío (DEWP = {dewp:.1f} °C) está fuera del rango permitido (-60.0 a 45.0 °C)."
            )

    # Incompatibilidad termodinámica: DEWP > TEMP
    if "DEWP" in parsed_num and "TEMP" in parsed_num:
        if parsed_num["DEWP"] > parsed_num["TEMP"]:
            errores.append(
                f"Incompatibilidad física: el punto de rocío (DEWP = {parsed_num['DEWP']:.1f} °C) "
                f"no puede superar la temperatura del aire (TEMP = {parsed_num['TEMP']:.1f} °C)."
            )

    # Variables acumuladas no negativas
    for col, unidad in [("Iws", "m/s"), ("Is", "horas"), ("Ir", "horas")]:
        if col in parsed_num and parsed_num[col] < 0.0:
            etiqueta = FIELD_LABELS_ES[col]
            errores.append(
                f"El campo '{etiqueta}' ({parsed_num[col]:.2f} {unidad}) no puede ser negativo."
            )

    # 6. Generar advertencias no bloqueantes si se extrapola fuera del rango histórico 2010–2014
    if not errores:
        if "year" in valid_time_ints:
            yr = valid_time_ints["year"]
            if yr < 2010 or yr > 2014:
                advertencias.append(
                    f"El año seleccionado ({yr}) está fuera del periodo histórico de entrenamiento "
                    f"(2010–2014); la predicción representa una extrapolación temporal."
                )

        hist_checks = [
            ("DEWP", "°C"),
            ("TEMP", "°C"),
            ("PRES", "hPa"),
            ("Iws", "m/s"),
            ("Is", "horas"),
            ("Ir", "horas"),
        ]
        for col, unidad in hist_checks:
            val = parsed_num[col]
            h_min, h_max = HISTORICAL_RANGES[col]
            if val < h_min or val > h_max:
                etiqueta = FIELD_LABELS_ES[col]
                advertencias.append(
                    f"El valor de '{etiqueta}' ({val:g} {unidad}) está fuera del rango observado en "
                    f"Pekín entre 2010 y 2014 ({h_min:g} a {h_max:g} {unidad})."
                )

    return len(errores) == 0, errores, advertencias


def preparar_entrada_usuario(datos: dict[str, Any]) -> pd.DataFrame:
    """Valida los datos del usuario, construye un DataFrame de 1 fila con dtypes exactos y aplica ingeniería."""
    es_valido, errores, _ = validar_entrada(datos)
    if not es_valido:
        raise ValueError("Datos de entrada inválidos: " + " | ".join(errores))

    fila = {
        "year": int(float(datos["year"])),
        "month": int(float(datos["month"])),
        "day": int(float(datos["day"])),
        "hour": int(float(datos["hour"])),
        "DEWP": float(datos["DEWP"]),
        "TEMP": float(datos["TEMP"]),
        "PRES": float(datos["PRES"]),
        "cbwd": str(datos["cbwd"]),
        "Iws": float(datos["Iws"]),
        "Is": int(round(float(datos["Is"]))) if float(datos["Is"]).is_integer() else float(datos["Is"]),
        "Ir": int(round(float(datos["Ir"]))) if float(datos["Ir"]).is_integer() else float(datos["Ir"]),
    }

    df_raw = pd.DataFrame([fila], columns=RAW_COLUMNS)
    return crear_caracteristicas(df_raw)


def clasificar_calidad_aire(pm25: float) -> dict[str, Any]:
    """Clasifica la concentración de PM2.5 (µg/m³) según los puntos de corte de referencia EPA."""
    valor = max(0.0, float(pm25))

    if valor <= 12.0:
        return {
            "categoria": "Buena",
            "nivel": 1,
            "color_hex": "#2ecc71",
            "rango_ugm3": "0.0 – 12.0 µg/m³",
            "descripcion": "La calidad del aire es satisfactoria y la contaminación por partículas finas representa un riesgo mínimo o nulo.",
            "recomendacion": "Condiciones ideales para realizar actividades al aire libre y ventilar espacios interiores con normalidad.",
        }
    if valor <= 35.4:
        return {
            "categoria": "Moderada",
            "nivel": 2,
            "color_hex": "#f1c40f",
            "rango_ugm3": "12.1 – 35.4 µg/m³",
            "descripcion": "La calidad del aire es aceptable; sin embargo, puede existir una preocupación moderada para personas excepcionalmente sensibles.",
            "recomendacion": "La población general puede realizar actividades al aire libre; personas con sensibilidad respiratoria severa deben vigilar síntomas.",
        }
    if valor <= 55.4:
        return {
            "categoria": "Dañina para grupos sensibles",
            "nivel": 3,
            "color_hex": "#e67e22",
            "rango_ugm3": "35.5 – 55.4 µg/m³",
            "descripcion": "Niños, adultos mayores y personas con asma o enfermedades cardiopulmonares pueden experimentar efectos adversos.",
            "recomendacion": "Los grupos sensibles deben reducir el esfuerzo físico prolongado o intenso al aire libre.",
        }
    if valor <= 150.4:
        return {
            "categoria": "Dañina",
            "nivel": 4,
            "color_hex": "#e74c3c",
            "rango_ugm3": "55.5 – 150.4 µg/m³",
            "descripcion": "Toda la población puede comenzar a experimentar molestias respiratorias, y los grupos sensibles enfrentan efectos más graves.",
            "recomendacion": "Evitar ejercicio intenso al aire libre; grupos sensibles deben permanecer en interiores y considerar mascarilla filtrante (KN95/N95).",
        }
    if valor <= 250.4:
        return {
            "categoria": "Muy dañina",
            "nivel": 5,
            "color_hex": "#8e44ad",
            "rango_ugm3": "150.5 – 250.4 µg/m³",
            "descripcion": "Alerta sanitaria: el riesgo de efectos adversos en la salud respiratoria y cardiovascular aumenta significativamente para toda la población.",
            "recomendacion": "Suspender actividades físicas al aire libre, mantener ventanas cerradas, usar purificador de aire y mascarilla KN95/N95 al salir.",
        }
    return {
        "categoria": "Peligrosa",
        "nivel": 6,
        "color_hex": "#7f1d1d",
        "rango_ugm3": "> 250.4 µg/m³",
        "descripcion": "Contingencia ambiental severa (emergencia de salud pública): toda la población está expuesta a efectos graves por smog denso.",
        "recomendacion": "Permanecer en espacios interiores filtrados, evitar cualquier exposición externa innecesaria y utilizar protección respiratoria N95/KN95.",
    }
