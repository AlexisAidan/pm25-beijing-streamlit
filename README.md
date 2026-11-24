# Predicción Horaria de la Contaminación del Aire (PM2.5) en Pekín

Aplicación web interactiva desarrollada en **Python** y **Streamlit** para estimar la concentración horaria de material particulado fino ($\text{PM}_{2.5}$, en $\mu\text{g/m}^3$) en Pekín a partir de variables meteorológicas y temporales, integrando los **tres mejores modelos preentrenados de regresión supervisada** obtenidos en el estudio experimental del Proyecto Final.

---

## Portada Institucional

| Información General | Detalle Académico |
| :--- | :--- |
| **Universidad:** | Universidad Autónoma de Chihuahua (UACH) — Facultad de Ingeniería |
| **Programa Académico:** | Ingeniería en Ciencias de la Computación |
| **Materia:** | Data Science |
| **Actividad:** | Proyecto Final — Despliegue de Modelos de Regresión Supervisada (Beijing PM2.5) |
| **Nombre del Alumno:** | Alexis Aidan Leon Rodelas |
| **Matrícula:** | 377171 |
| **Docente:** | Olanda Prieto Ordaz |
| **Fecha:** | Noviembre de 2026 |

---

## Enlace Público de la Aplicación en Producción

La aplicación se encuentra desplegada en producción para acceso público directo desde cualquier navegador —sin necesidad de instalar dependencias locales— a través de **Streamlit Community Cloud**:

- **URL pública de despliegue:** [https://pm25-beijing-app-gchzkpnreicekvgxkwamwx.streamlit.app/](https://pm25-beijing-app-gchzkpnreicekvgxkwamwx.streamlit.app/)
- **Repositorio público en GitHub:** [https://github.com/AlexisAidan/pm25-beijing-streamlit](https://github.com/AlexisAidan/pm25-beijing-streamlit)
- **Archivo de entrada configurado:** `app.py`
- **Versión de Python en la nube:** `3.13` (definida en `.python-version`)

---

## 1. Descripción del Proyecto y Planteamiento del Problema

### Contexto ambiental y de salud pública
Las partículas en suspensión $\text{PM}_{2.5}$ poseen un diámetro aerodinámico menor o igual a $2.5\ \mu\text{m}$ (aproximadamente treinta veces menor que el grosor de un cabello humano), lo que les permite penetrar profundamente en los alvéolos pulmonares y el torrente sanguíneo, incrementando el riesgo de enfermedades respiratorias y cardiovasculares agudas. En la cuenca atmosférica de Pekín, la concentración de $\text{PM}_{2.5}$ experimenta fluctuaciones drásticas hora a hora gobernadas por la interacción entre emisiones estacionales y condiciones sinópticas:
- Las situaciones de **calma o viento variable (`cv`)** y el flujo húmedo del **sureste (`SE`)** favorecen la inversión térmica y el estancamiento de contaminantes cerca de la superficie.
- Los frentes fríos y secos provenientes del **noroeste (`NW`)** limpian rápidamente la atmósfera urbana (reduciendo la mediana histórica de contaminación en casi un $70\ \%$).
- La **temporada oficial de calefacción invernal** (del 15 de noviembre al 15 de marzo) incrementa significativamente las emisiones base de combustión (`+22.2 %` en el promedio horario de $\text{PM}_{2.5}$).

### Fuente de datos y formulación en Ciencia de Datos
El proyecto emplea el conjunto de datos **Beijing PM2.5 Data** del **UCI Machine Learning Repository (ID: 381)** documentado por Liang et al. (2015), que integra cinco años consecutivos de registros horarios (del 1 de enero de 2010 al 31 de diciembre de 2014):
- **Mediciones de calidad del aire:** Estación de monitoreo de la Embajada de Estados Unidos en Pekín (`pm2.5` en $\mu\text{g/m}^3$, variable objetivo cuantitativa continua $y \ge 0$).
- **Mediciones meteorológicas:** Aeropuerto Internacional de Pekín (`DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws`, `Is`, `Ir`) junto con variables de calendario (`year`, `month`, `day`, `hour`).
- **Depuración y partición libre de fuga de información (*data leakage*):** De los `43,824` registros originales (`0` duplicados y `0` nulos en las predictoras), se eliminaron antes del particionado las `2,067` filas sin etiqueta objetivo (`4.72 %`), evitando fabricar etiquetas artificiales. Las **`41,757` observaciones válidas** se dividieron mediante una partición aleatoria **80 % Entrenamiento (`33,405` filas) / 20 % Prueba (`8,352` filas)** con semilla fija `RANDOM_STATE = 42`, manteniendo el bloque `Test` estrictamente aislado hasta la evaluación final.
- **Enfoque puramente meteorológico-temporal:** A diferencia de modelos autorregresivos que exigen conocer la lectura de $\text{PM}_{2.5}$ de la hora anterior ($t - 1$), esta aplicación predice la contaminación **exclusivamente a partir del estado atmosférico y temporal**, permitiendo estimar el nivel de riesgo sanitario incluso cuando no existen sensores químicos activos o ante pronósticos meteorológicos futuros.

---

## 2. Selección y Justificación de los 3 Mejores Modelos Integrados

En el estudio experimental previo (`main.ipynb`) se entrenaron, optimizaron con `GridSearchCV` y compararon **19 configuraciones** bajo **Validación Cruzada de 5 Folds** (`KFold(n_splits=5, shuffle=True, random_state=42)`) sobre el conjunto de entrenamiento, abarcando cuatro familias algorítmicas: Regresión Lineal/Polinomial regularizada con `ElasticNet`, Máquinas de Vectores de Soporte (`LinearSVR` y `SVR` Polinomial/RBF), `RandomForestRegressor` y Métodos de Ensamble (`AdaBoost`, `Gradient Boosting`, `XGBoost`, `Voting` y `Stacking`).

Para integrar en producción web los **3 mejores modelos**, se evaluaron conjuntamente los cuatro criterios técnicos de la rúbrica: **desempeño comparativo**, **métricas en CV y Test (`MAE`, `MSE`, `RMSE` y $R^2$)**, **capacidad para resolver la no linealidad del problema** y **viabilidad operativa en GitHub y Streamlit Community Cloud** (archivos preentrenados `.joblib` comprimidos con LZMA `xz-3` de tamaño individual **`< 45 MB`**, peso total **`38.53 MB < 90 MB`**, carga en caché con `@st.cache_resource` y **cero reentrenamiento** en tiempo de ejecución).

### 2.1 Tabla comparativa de los 3 modelos desplegados en la aplicación

| Ranking | ID (`models/`) | Modelo Preentrenado | MAE CV | MSE CV | RMSE CV ($\pm\sigma$) | $R^2$ CV | MAE Test | MSE Test | RMSE Test | $R^2$ Test | Latencia Inferencia | Tamaño (`.joblib`) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `xgb_opt` | **XGBoost (Optimizado)** | 21.21 | 1,096.57 | 33.11 $\pm$ 0.40 | 0.8692 | **20.56** | **1,182.73** | **34.39** | **0.8658** | ~3.2 ms | **4.62 MB** |
| **#2** | `rf_base` | **Random Forest (Base)** | 24.00 | 1,434.42 | 37.87 $\pm$ 0.56 | 0.8289 | **22.61** | **1,432.41** | **37.85** | **0.8374** | ~6.5 ms | **32.74 MB** |
| **#3** | `gb_opt` | **Gradient Boosting (Optimizado)** | 25.46 | 1,435.98 | 37.89 $\pm$ 0.45 | 0.8287 | **25.21** | **1,561.84** | **39.52** | **0.8228** | ~3.0 ms | **1.17 MB** |

*(Unidades: `MAE` y `RMSE` en $\mu\text{g/m}^3$; `MSE` en $(\mu\text{g/m}^3)^2$; $R^2$ adimensional.)*

### 2.2 Justificación detallada por modelo

#### Modelo #1: `XGBoost (Optimizado)` — `models/xgb_opt.joblib` (`4.62 MB`)
- **Hiperparámetros óptimos:** `n_estimators=400`, `learning_rate=0.05`, `max_depth=10`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
- **Desempeño del modelo:** Mejor modelo individual de todo el estudio experimental (2.º lugar absoluto entre las 19 configuraciones evaluadas, únicamente detrás de `StackingRegressor` por `0.60 µg/m³` en CV y `0.87 µg/m³` en Test). La búsqueda `GridSearchCV` redujo su `RMSE_CV` en un **`-36.63 %`** respecto a `XGBoost (Base)` (`52.25` $\rightarrow$ `33.11 µg/m³`).
- **Métricas de evaluación:** Alcanza `MAE = 21.21 µg/m³`, `MSE = 1,096.57`, `RMSE = 33.11 ± 0.40 µg/m³` y $R^2 = 0.8692$ en 5-Fold CV, confirmando generalización sólida sobre el conjunto `Test` reservado con **`MAE = 20.56 µg/m³`**, **`MSE = 1,182.73`**, **`RMSE = 34.39 µg/m³`** y **$R^2 = 0.8658`** (`Diff_R² = -0.0034`).
- **Capacidad para resolver el problema:** Mediante aproximación de gradiente de segundo orden, regularización explícita en las hojas y submuestreo estocástico de instancias y columnas, modela con gran fidelidad los umbrales abruptos entre humedad relativa (`temp_dewp_diff`), calefacción invernal y limpieza por viento del noroeste.
- **Viabilidad en producción web:** Ofrece el balance óptimo entre exactitud y ligereza: el `Pipeline` completo comprimido con `xz-3` ocupa solo **`4.62 MB`**, se carga en `~0.25 s` y responde en **`~3.2 ms`** por consulta.

#### Modelo #2: `Random Forest (Base)` — `models/rf_base.joblib` (`32.74 MB`)
- **Hiperparámetros entrenados:** `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `max_features=1.0`, `random_state=42`.
- **Desempeño del modelo:** Forma parte del bloque superior de ensambles con más del `82.8 %` de varianza explicada, superando por casi `20 µg/m³` en `RMSE` al mejor `SVR RBF (Optimizado)` (`57.11 µg/m³`) y por casi `40 µg/m³` a `LinearRegression` (`76.97 µg/m³`). Su `RMSE_CV` (`37.87`) difiere en apenas `0.77 %` frente a `Random Forest (Optimizado)` (`37.58`) e incluso presenta mejor `MAE_CV` (`24.00` vs. `24.15 µg/m³`).
- **Métricas de evaluación:** En 5-Fold CV obtuvo `MAE = 24.00 µg/m³`, `MSE = 1,434.42`, `RMSE = 37.87 ± 0.56 µg/m³` y $R^2 = 0.8289$; en `Test` mejoró ligeramente hasta **`MAE = 22.61 µg/m³`**, **`MSE = 1,432.41`**, **`RMSE = 37.85 µg/m³`** y **$R^2 = 0.8374`**.
- **Capacidad para resolver el problema:** Aporta diversidad estructural genuina al representar el paradigma de **Bagging paralelo** (*bootstrap aggregating*): promedia 100 árboles profundos independientes que reducen la varianza sin sobre-ponderar secuencialmente valores extremos de smog (`> 500 µg/m³`).
- **Viabilidad en producción web:** Reduce el peso en disco un **`60.4 %`** (`32.74 MB` bajo `xz-3`, dentro del límite de `< 45 MB` por archivo en GitHub) frente a la variante de 250 árboles (`82.74 MB` en `xz-3` / `124.48 MB` en `zlib`), conservando inferencia inferior a `7 ms`.

#### Modelo #3: `Gradient Boosting (Optimizado)` — `models/gb_opt.joblib` (`1.17 MB`)
- **Hiperparámetros óptimos:** `n_estimators=250`, `learning_rate=0.1`, `max_depth=7`, `subsample=0.8`, `random_state=42`.
- **Desempeño del modelo:** Logró la mayor mejora porcentual tras la optimización de hiperparámetros en todo el estudio (**`-38.79 %`** en `RMSE_CV`, bajando de `61.90` a `37.89 µg/m³`), superando ampliamente a `AdaBoost (Optimizado)` (`54.00 µg/m³`), `SVR RBF (Optimizado)` (`57.11 µg/m³`) y `Poly(2) + ElasticNet` (`67.07 µg/m³`).
- **Métricas de evaluación:** Registró `MAE = 25.46 µg/m³`, `MSE = 1,435.98`, `RMSE = 37.89 ± 0.45 µg/m³` y $R^2 = 0.8287$ en CV, con una brecha entrenamiento-validación notablemente más contenida (`13.96 µg/m³` frente a `~23.7 µg/m³` de los árboles sin poda). En `Test` alcanzó **`MAE = 25.21 µg/m³`**, **`MSE = 1,561.84`**, **`RMSE = 39.52 µg/m³`** y **$R^2 = 0.8228`**.
- **Capacidad para resolver el problema:** El ajuste secuencial de 250 árboles de profundidad moderada (`max_depth=7`) con submuestreo estocástico (`subsample=0.8`) sobre pseudorresiduales proporciona estimaciones estables y regularizadas en regímenes meteorológicos de transición.
- **Viabilidad en producción web:** Es el modelo más liviano de los tres (**`1.17 MB`** con `xz-3`), descomprime en `~0.07 s` e infiere en **`~3.0 ms`** empleando exclusivamente `scikit-learn` estándar.

### 2.3 Nota técnica de viabilidad sobre ensambles pesados (`> 45 MB`)
En `main.ipynb`, las configuraciones que anidan internamente un `RandomForestRegressor(n_estimators=250, max_depth=None)` sin cota de profundidad en hojas —específicamente `StackingRegressor (cv=5)` (`RMSE_Test = 33.52`, $R^2_{\text{Test}} = 0.8725$), `VotingRegressor (Ponderado)` (`RMSE_Test = 35.81`, $R^2_{\text{Test}} = 0.8545$) y `Random Forest (Optimizado)` (`RMSE_CV = 37.58`)— generan artefactos serializados de **`124.48 MB` a `133.18 MB`** con compresión `zlib-3` y de **`82.74 MB` a `88.53 MB`** incluso bajo compresión máxima LZMA `xz-3` (además de requerir `184.67 s` de ajuste en el caso del Stacking). Dichos tamaños rebasan el umbral de seguridad de GitHub sin Git LFS y degradan el arranque en frío en contenedores gratuitos de Streamlit Community Cloud. Por ello, sus resultados íntegros de CV y Test se preservan como referencia experimental en `models/metadata.json` y en la pestaña analítica de la aplicación, desplegándose para inferencia en vivo los tres mejores modelos viables (`38.53 MB` combinados).

---

## 3. Pipeline de Preprocesamiento, Ingeniería de Características y Validación

Cada predicción solicitada por el usuario se procesa **sin reentrenar ningún estimador**, siguiendo tres etapas determinísticas idénticas al entrenamiento en `main.ipynb`:

```text
Entrada del usuario (11 variables crudas)
   │
   ▼
[1] validar_entrada(datos) ──► Verifica tipos, calendario real, DEWP <= TEMP y rangos físicos
   │                           (Emite errores bloqueantes o advertencias de extrapolación)
   ▼
[2] crear_caracteristicas(df) ► Deriva 5 variables temporales y termodinámicas (16 columnas)
   │
   ▼
[3] Pipeline preentrenado (.joblib cargado en caché @st.cache_resource)
    ├── ColumnTransformer (ajustado solo con X_train: 33,405 filas)
    │    ├── 15 numéricas: SimpleImputer(strategy="median") ──► StandardScaler()
    │    └── 1 categórica (cbwd): SimpleImputer("most_frequent") ──► OneHotEncoder(drop="first")
    └── Estimador (XGBoost / Random Forest / Gradient Boosting) ──► max(0.0, y_pred) + Categoría EPA
```

### 3.1 Variables crudas de entrada (`RAW_COLUMNS`)
1. **Temporales (`int`):** `year` (histórico `2010–2014`; válido `1990–2100`), `month` (`1–12`), `day` (`1–31`, validado contra calendario gregoriano real, incluyendo años bisiestos), `hour` (`0–23`).
2. **Termodinámicas y barométricas (`float`):** `TEMP` (`-50.0` a `55.0 °C`), `DEWP` (`-60.0` a `45.0 °C`, sujeto a la restricción física $\text{DEWP} \le \text{TEMP}$), `PRES` (`940.0` a `1070.0 hPa`).
3. **Eólicas y de precipitación:** `cbwd` $\in \{\text{SE}, \text{NW}, \text{cv}, \text{NE}\}$, `Iws` ($\ge 0\ \text{m/s}$), `Is` ($\ge 0\ \text{h}$ de nieve), `Ir` ($\ge 0\ \text{h}$ de lluvia).

### 3.2 Variables derivadas por ingeniería de características (`crear_caracteristicas`)
- **`day_of_week`** (`0` = lunes a `6` = domingo) e **`is_weekend`** (`1` en sábado/domingo, `0` entre semana).
- **`is_heating_season`** (`1` entre el 15 de noviembre y el 15 de marzo; `0` el resto del año).
- **`temp_dewp_diff`** ($\text{TEMP} - \text{DEWP}$): depresión del punto de rocío, indicador físico directo de sequedad atmosférica y **variable #1 en importancia predictiva** ($r = -0.4391$ con `pm2.5`).
- **`is_precipitating`** (`1` cuando `Ir > 0` o `Is > 0`, capturando deposición húmeda por lluvia o nieve).

### 3.3 Clasificación interpretativa de calidad del aire (EPA $\text{PM}_{2.5}$)
La concentración estimada se traduce en seis niveles sanitarios con código cromático y recomendaciones para usuarios no técnicos: **Buena** (`0.0–12.0 µg/m³`), **Moderada** (`12.1–35.4`), **Dañina para grupos sensibles** (`35.5–55.4`), **Dañina** (`55.5–150.4`), **Muy dañina** (`150.5–250.4`) y **Peligrosa** (`> 250.4 µg/m³`).

---

## 4. Estructura del Repositorio

```text
app_pm25_beijing/
├── .streamlit/
│   └── config.toml                 # Tema visual claro académico (#2b5c8f / #d95f02)
├── models/
│   ├── metadata.json               # Métricas CV/Test, justificaciones, rangos y 4 escenarios reales
│   ├── xgb_opt.joblib              # Modelo #1: Pipeline XGBoost Optimizado (4.62 MB)
│   ├── rf_base.joblib              # Modelo #2: Pipeline Random Forest Base (32.74 MB)
│   └── gb_opt.joblib               # Modelo #3: Pipeline Gradient Boosting Optimizado (1.17 MB)
├── src/
│   ├── __init__.py                 # Inicialización del paquete de lógica de dominio
│   ├── preprocessing.py            # Validación física/calendario, crear_caracteristicas y escala EPA
│   └── model_loader.py             # Carga única con caché e inferencia individual y comparativa
├── tests/
│   └── test_app_and_models.py      # Suite automatizada de pruebas (pytest + Streamlit AppTest)
├── .gitignore                      # Exclusión de cachés/entornos/secretos conservando models/ y config
├── .python-version                 # Fijación de versión Python 3.13 para Streamlit Community Cloud
├── app.py                          # Aplicación principal multipestaña en Streamlit
├── requirements.txt                # Dependencias fijadas compatibles con Linux y Windows
└── README.md                       # Documentación académica, técnica y guía de ejecución
```

---

## 5. Instrucciones de Instalación y Ejecución Local

### Requisitos previos
- **Python 3.11 a 3.13** (desarrollado y verificado sobre **Python 3.13**).
- `pip` actualizado. No se requiere conexión al repositorio UCI ni reentrenamiento local, ya que los tres `Pipeline`s preentrenados están incluidos en `models/`.

### Paso 1: Clonar el repositorio y ubicarse en la carpeta del proyecto
```bash
git clone <URL_DEL_REPOSITORIO_GITHUB>
cd app_pm25_beijing
```

### Paso 2: Crear y activar un entorno virtual (recomendado)

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar las dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar la aplicación web con Streamlit
```bash
streamlit run app.py
```
Una vez iniciado el servidor local, abra en su navegador la dirección mostrada en consola (por defecto `http://localhost:8501`).

---

## 6. Ejecución de Pruebas Automatizadas

El proyecto incluye una suite de verificación integral en `tests/test_app_and_models.py` basada en `pytest` y `streamlit.testing.v1.AppTest` que comprueba:
1. Integridad de `models/metadata.json`, límite de peso `< 45 MB` por archivo `.joblib` y carga de los 3 `Pipeline`s ya ajustados garantizando **cero llamadas a `.fit()`** durante la predicción.
2. Exactitud de `crear_caracteristicas` (días de semana, fines de semana, bordes exactos del 15 de noviembre y 15 de marzo en `is_heating_season`, `temp_dewp_diff` e `is_precipitating`).
3. Inferencia no negativa en `predecir` y `predecir_todos`, coherencia física ante escenarios opuestos (noche invernal calma vs. viento fuerte `NW`) y los 6 niveles EPA de `clasificar_calidad_aire`.
4. Rechazo limpio en `validar_entrada` ante incompatibilidad física ($\text{DEWP} > \text{TEMP}$), fechas inválidas (`2014-02-30`, `2014-04-31`, `2014-02-29`), valores negativos, fuera de rango o nulos/`NaN`, y generación de advertencias en años extrapolados (`2026`).
5. Arranque completo de `app.py` con `AppTest`, presencia de datos institucionales de la UACH, cambio entre los 3 modelos, carga de escenarios predefinidos y renderizado limpio de `st.error` / `st.warning` sin excepciones no controladas.

Para ejecutar todas las pruebas desde la raíz de `app_pm25_beijing`:
```bash
pytest tests/test_app_and_models.py -v
```

---

## 7. Despliegue en Streamlit Community Cloud

1. Subir el contenido de este directorio a un repositorio público de GitHub verificando que `models/*.joblib`, `models/metadata.json`, `.streamlit/config.toml`, `.python-version` y `requirements.txt` estén versionados (`.gitignore` ya está configurado para conservarlos y excluir `.env` o `.streamlit/secrets.toml`).
2. Ingresar a [share.streamlit.io](https://share.streamlit.io/) e iniciar sesión con GitHub.
3. Seleccionar **Create app**, elegir el repositorio, la rama principal (`main`) y especificar `app.py` como **Main file path**.
4. Confirmar la versión de Python (`3.13`, detectada automáticamente desde `.python-version`) y pulsar **Deploy**.

---

## 8. Referencias Bibliográficas

1. **Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., & Chen, S. X. (2015).** *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating.* Proceedings of the Royal Society A, 471(2182), 20150257. DOI: `10.1098/rspa.2015.0257`.
2. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.
3. **Breiman, L. (2001).** *Random Forests.* Machine Learning, 45(1), 5–32.
4. **Friedman, J. H. (2001).** *Greedy function approximation: A gradient boosting machine.* The Annals of Statistics, 29(5), 1189–1232.
5. **Fernández-Delgado, M., Sirsat, M. S., Cernadas, E., Alawadi, S., Barro, S., & Febrero-Bande, M. (2019).** *An extensive experimental survey of regression methods.* Neural Networks, 111, 11–34.
6. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
