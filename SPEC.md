# SPEC — Cuban Inflation Predictor (AutoGluon)

> Documento de especificación técnica completo. Dirigido a un modelo ejecutor:
> implementar TODO lo aquí descrito sin tomar decisiones de diseño adicionales.
> Si algo es ambiguo, aplicar la opción más simple que cumpla el contrato.

---

## 1. Contexto

Repo: `cuban_inflation_predictor`. Predice la tasa de cambio informal USD/CUP
(fuente: eltoque.com vía API cubanomic). Sistema anterior: XGBoost en
`sample.ipynb` (queda DEPRECADO, no borrar). Sistema nuevo: AutoGluon
TimeSeries, CPU-only, con pipelines automatizados y frontend público.

Rama de trabajo: `AutoGluon` (ya creada y activa).

## 2. Decisiones congeladas (NO cambiar)

| # | Decisión | Valor |
|---|----------|-------|
| D1 | Series | Solo USD. Arquitectura multi-moneda lista: añadir ECU = añadir `"ECU"` a lista `CURRENCIES` en `src/data_loader.py`, sin tocar nada más |
| D2 | Target | `median` diario, frecuencia `D`, horizonte de predicción = 7 días (valor que publica el sitio eltoque.com) |
| D3 | Re-exploración AutoGluon | MANUAL únicamente (flag CLI `--reexplore` + celda final del notebook). NUNCA automática en CI |
| D4 | Frontend | Sitio estático (GitHub Pages), sin backend, consume `output/predictions.json` + `output/history.jsonl` |
| D5 | Modelos en git | Commitear `models/predictor/` directo (sin Git LFS). La receta congelada se restringe a modelos ligeros si el artefacto supera 100 MB |
| D6 | fit() completo | Se ejecuta UNA sola vez en el notebook (exploración). Producción solo reentrena la receta congelada |
| D7 | Disclaimer | Todo output público lleva: "No es consejo financiero" |

## 3. Entorno

- Python 3.12 (venv existente en `.venv/`, ya tiene torch CPU + autogluon.timeseries)
- Instalación reproducible (documentar en README):

```bash
pip install -U pip setuptools wheel
pip install autogluon.timeseries --extra-index-url https://download.pytorch.org/whl/cpu
```

- Verificación: `python -c "import torch; assert not torch.cuda.is_available()"`
- `requirements.txt`: `autogluon.timeseries`, `pandas`, `numpy`, `requests`, `matplotlib`, `jupyter`, `nbformat`, `nbconvert`. Sin versiones pineadas salvo que haya conflicto.

## 4. Contrato de datos

### 4.1 API origen

```
GET https://api.cubanomic.com/api/v1/x-rates-by-date-range-history?trmi=true&cur={CUR}&token={CUBANOMIC_TOKEN}&period=730D
```

`{CUR}` ∈ {`USD`, `ECU`}. Respuesta: lista JSON de registros diarios.

### 4.2 Schema registro API

```json
{
  "_id": "2024-09-13",          // fecha, string YYYY-MM-DD (clave primaria)
  "avg": 320.68,                // promedio del día (no usado como target)
  "median": 320.0,              // TARGET (valor publicado por el sitio)
  "median": 320.0,
  "min": 310.0,
  "max": 345.0,
  "count_values": 536,
  "cur": "USD",
  "trmi": 321.5,
  "first": {"date": "...", "value": 325.0},
  "last":  {"date": "...", "value": 322.0}
}
```

### 4.3 Cache local

`data/tasas_de_cambio.json` — lista de 2 elementos: `[registros_USD, registros_ECU]`
(orden = orden de `CURRENCIES` al hacer fetch). Formato actual del repo, mantener.

### 4.4 Formato interno canónico (AutoGluon TimeSeriesDataFrame)

Toda serie en formato LARGO:

| item_id | timestamp | target |
|---------|-----------|--------|
| USD | 2024-09-13 | 320.68 |

- `item_id`: str, código moneda
- `timestamp`: pd.Timestamp, frecuencia diaria regular (rellenar huecos con `resample('D').ffill()`)
- `target`: float, columna `median` (D2)

## 5. Estructura de archivos final

```
cuban_inflation_predictor/
├── SPEC.md                          # este archivo
├── README.md                        # generar (sección 10)
├── requirements.txt
├── sample.ipynb                     # DEPRECADO, no tocar
├── data/
│   └── tasas_de_cambio.json         # cache API (existe)
├── src/
│   ├── __init__.py                  # vacío
│   ├── data_loader.py               # sección 6.1
│   ├── predict.py                   # sección 6.3
│   ├── retrain.py                   # sección 6.4
│   └── monitor.py                   # sección 6.5
├── notebooks/
│   └── autogluon_explore.ipynb      # sección 6.2
├── models/
│   ├── recipe.json                  # receta congelada (sección 6.2 paso 8)
│   └── predictor/                   # artefactos AutoGluon (commiteado, <100MB)
├── output/
│   ├── predictions.json             # consumo frontend (sección 7.1)
│   └── history.jsonl                # histórico (sección 7.2)
├── docs/
│   └── DECISIONS.md                 # decisiones + limitaciones + disclaimer
├── .github/
│   └── workflows/
│       ├── daily_predict.yml        # sección 8.1
│       ├── weekly_retrain.yml       # sección 8.2
│       └── monthly_check.yml        # sección 8.3
└── frontend/
    └── index.html                   # sección 9 (single-file, JS vanilla + Chart.js CDN)
```

## 6. Spec por componente

### 6.1 `src/data_loader.py`

Responsabilidad única: fetch + limpieza + formato canónico. Reusar lógica de
`sample.ipynb` celdas 2 y 5 (dedup por `_id`, `pd.json_normalize`, datetime,
sort, `resample('D').ffill()`).

```python
API_URL = "https://api.cubanomic.com/api/v1/x-rates-by-date-range-history"
TOKEN = os.environ["CUBANOMIC_TOKEN"]  # env var, NUNCA hardcodear
CURRENCIES = ["USD"]          # D1: añadir "ECU" aquí escala a multi-moneda
CACHE_PATH = "data/tasas_de_cambio.json"

def fetch_rates(currencies: list[str] = CURRENCIES, period: str = "730D") -> list[list[dict]]:
    """GET por moneda. Devuelve lista paralela a `currencies`.
    Escribe/actualiza CACHE_PATH (lista en mismo orden). Lanza excepción si falla requests."""

def load_cache(path: str = CACHE_PATH) -> list[list[dict]]:
    """Lee cache local. FileNotFoundError si no existe."""

def to_long_dataframe(raw: list[list[dict]], currencies: list[str] = CURRENCIES) -> pd.DataFrame:
    """raw[i] = registros de currencies[i]. Dedup por _id, ordena por fecha,
    resample('D').ffill(), devuelve formato largo [item_id, timestamp, target].
    Solo columnas: item_id, timestamp, target (float)."""

def load_series(refresh: bool = False) -> pd.DataFrame:
    """Punto de entrada único. refresh=True → fetch_rates + cache; si fetch
    falla, fallback a load_cache con warning en stderr. refresh=False → cache directo."""
```

Test mínimo (en `if __name__ == "__main__":`): cargar cache, assert columnas,
assert sin NaN en target, assert frecuencia diaria sin huecos, print rango
fechas + n registros por item_id.

### 6.2 `notebooks/autogluon_explore.ipynb`

Kind: experiment. Crear con helper del skill jupyter-notebook si disponible;
si no, JSON nbformat 4 estándar. Celdas en este orden exacto:

1. **md — Título + objetivo + criterio de éxito.** Objetivo: predecir median USD
   7 días. Criterio éxito: ensemble AutoGluon supera baseline naive en MAE y
   MASE sobre últimos 30 días.
2. **code — Setup.** Imports (pandas, numpy, matplotlib, autogluon.timeseries),
   seed fija (`np.random.seed(42)`), celda config única:
   `PREDICTION_LENGTH = 7`, `TEST_DAYS = 30`, `FREQ = "D"`, paths.
3. **code — Datos.** `from src.data_loader import load_series` (ejecutar desde
   raíz repo o ajustar `sys.path`). `load_series(refresh=True)` →
   `TimeSeriesDataFrame`. Print shape + head.
4. **md + code — EDA.** (a) plot serie completa; (b) media móvil 7d/30d
   (tendencia); (c) descomposición o ACF de residuos (estacionalidad semanal);
   (d) outliers: z-score > 3 marcados en plot. Markdown breve por hallazgo.
5. **code — Split.** Últimos `TEST_DAYS` = test, resto = train. AutoGluon:
   `train, test = ts_df.slice_by_timestep(None, -TEST_DAYS)` y
   `ts_df.slice_by_timestep(-TEST_DAYS, None)` (o equivalente versión instalada).
6. **code — Baselines.** (a) naive: último valor train repetido 7d rolling;
   (b) media móvil 7d. Métricas sobre test: MAE, RMSE, MASE. Dict `baselines`
   + tabla.
7. **code — fit() ÚNICO (D6).**
   ```python
   predictor = TimeSeriesPredictor(
       prediction_length=PREDICTION_LENGTH, freq=FREQ,
       eval_metric="MASE", path="models/predictor",
   )
   predictor.fit(train, presets="medium_quality", time_limit=1800,
                 hyperparameters=None)  # exploración completa, CPU
   ```
   Excluir Chronos si tarda demasiado en CPU (`excluded_model_types=["Chronos"]`).
8. **code — Leaderboard + ensemble.** `predictor.leaderboard(test)`. Print
   modelos, score, pesos del ensemble (`predictor.info()`). Markdown: análisis
   del ganador.
9. **code — Congelar receta.** Escribir `models/recipe.json`:
   ```json
   {
     "frozen_at": "<ISO8601>",
     "eval_metric": "MASE",
     "prediction_length": 7,
     "freq": "D",
     "hyperparameters": {<modelos del ensemble ganador + hiperparámetros>},
     "ensemble_weights": {<modelo: peso>},
    "leaderboard": [{"model": "...", "score_val": ..., "score_test": ...}],
     "baselines": {"naive": {"MAE": ...}, "moving_avg_7": {"MAE": ...}},
     "test_days": 30
   }
   ```
   Si `du -sh models/predictor` > 100 MB (D5): refit con `hyperparameters`
   limitado a modelos ligeros (ETS, ARIMA, Theta, SeasonalNaive, RecursiveTabular,
   DirectTabular) y guardar ESA receta. Documentar cuál camino se tomó en markdown.
10. **code — Predicción demo.** `predictor.predict(train)` → 7 días. Plot
    histórico reciente + forecast + intervalos.
11. **md — Next steps.** Bullets: pipelines (sección 6.3-6.5), CI, frontend.
12. **md + code — Re-exploración manual (D3).** Markdown explica: ejecutar esta
    celda SOLO a voluntad (cada 3-6 meses o si monitor detecta degradación).
    Celda code idéntica a 7-9 pero con `time_limit` mayor y comentada por
    defecto con instrucción clara de descomentar.

Ejecutar top-to-bottom antes de commit (nbconvert). Outputs ruidosos largos:
recortar.

### 6.3 `src/predict.py`

CLI diario. Usa receta congelada, NUNCA re-entrena.

```
python -m src.predict [--out output/predictions.json]
```

Pasos: `load_series(refresh=True)` → `TimeSeriesPredictor.load("models/predictor")`
→ `predict()` 7d → escribir `output/predictions.json` (schema 7.1) →
append `output/history.jsonl` (schema 7.2). Exit 0 OK, exit 1 con mensaje
stderr si falla (CI lo usa para alerta).

### 6.4 `src/retrain.py`

CLI semanal. Reentrena receta congelada SIN re-explorar (D3).

```
python -m src.retrain [--reexplore]
```

- Default: lee `models/recipe.json` → `TimeSeriesPredictor` con mismos
  `hyperparameters` congelados → `fit()` sobre serie completa actualizada →
  sobrescribe `models/predictor/` → actualiza `frozen_at` NO (añade
  `last_retrain` en recipe.json) → conserva `best_model`, recalcula los
  baselines actuales y guarda un evento en `history.jsonl` tipo `"retrain"`.
- `--reexplore`: imprime mensaje "Re-exploración manual: usar notebook celda 12"
  y sale. (La re-exploración real vive en el notebook, D3.)

### 6.5 `src/monitor.py`

```
python -m src.monitor
```

- Lee `output/history.jsonl`, compara predicciones pasadas vs valores reales
  ya observados → calcula MAE rolling 7d y 30d.
- Degradación = MAE_7d > 1.5 × `baselines.naive.MAE` (baseline actual del
  recipe.json).
- Exit 0 sano; exit 2 degradado (mensaje stderr). CI mensual lo usa para alerta.
- Print tabla resumen stdout (últimas 10 predicciones vs real).

## 7. Contratos de salida

### 7.1 `output/predictions.json`

```json
{
  "generated_at": "<ISO8601 UTC>",
  "item_id": "USD",
  "horizon_days": 7,
  "last_observed": {"date": "YYYY-MM-DD", "value": 320.68},
  "history_series": [
    {"date": "YYYY-MM-DD", "value": 320.68}
  ],
  "forecast": [
    {"date": "YYYY-MM-DD", "mean": 321.0, "p10": 315.0, "p90": 327.0}
  ],
  "model": {
    "recipe_frozen_at": "<ISO8601>",
    "best_model": "<modelo congelado>",
    "last_retrain": "<ISO8601>",
    "data_last_date": "YYYY-MM-DD",
    "eval_metric": "MASE",
    "score_val": 0.83
  },
  "disclaimer": "No es consejo financiero. Modelo experimental."
}
```

### 7.2 `output/history.jsonl`

Una línea JSON por evento:

```json
{"type": "prediction", "generated_at": "...", "date": "...", "predicted": 321.0, "actual": null}
{"type": "prediction", "generated_at": "...", "date": "...", "predicted": 321.0, "actual": 322.1}
{"type": "retrain", "at": "...", "mae_30d": 4.2, "mase_30d": 0.81}
{"type": "error", "at": "...", "source": "predict|scraper", "message": "..."}
```

`actual` se rellena en ejecuciones posteriores cuando el dato real ya existe
(monitor.py o predict.py lo actualizan al inicio de cada corrida).

## 8. GitHub Actions

Setup común en los 3 workflows: `actions/checkout@v4`, `setup-python@v5`
(3.12), `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`,
ejecutar script, `git config user.name "github-actions[bot]"`, commit+push de
`output/`, `data/`, `models/` si hay cambios (`git diff --quiet || git commit`).

### 8.1 `daily_predict.yml`

- `schedule: cron: "0 12 * * *"` (07:00 Cuba) + `workflow_dispatch`
- `python -m src.predict` → commit `output/predictions.json`, `output/history.jsonl`, `data/tasas_de_cambio.json`
- Si exit≠0: append evento `error` a history.jsonl, commit, y fallar el job
  (GitHub notifica al owner = alerta gratis).

### 8.2 `weekly_retrain.yml`

- `schedule: cron: "0 13 * * 0"` (domingo) + `workflow_dispatch`
- `python -m src.retrain` → commit `models/` + `output/history.jsonl`

### 8.3 `monthly_check.yml`

- `schedule: cron: "0 14 1 * *"` (día 1 del mes) + `workflow_dispatch`
- `python -m src.monitor` → exit 2 = degradación → job falla (alerta) +
  mensaje indica ejecutar re-exploración manual (notebook celda 12).

## 9. Frontend (`frontend/index.html`)

Single-file. JS vanilla + Chart.js vía CDN. Sin build. GitHub Pages lo sirve.

**Estética**: oscuro futurista tipo crypto. Fondo `#0a0e17`/`#0d1220`, acentos
cyan `#00e5ff` + verde `#00ff9d`, glow sutil (`box-shadow`/`text-shadow`),
tipografía monoespaciada para números (`JetBrains Mono` o `ui-monospace`),
grid CSS. Opcional: grid de fondo con gradiente radial.

**Secciones**:
1. Header: título + badge "EXPERIMENTAL" + última actualización (`generated_at`).
2. Hero: valor actual (`last_observed.value`) grande + predicción día 7 + delta %.
3. Gráfico principal: histórico (desde history.jsonl, línea real) + forecast
   7d (línea punteada cyan) + banda p10-p90 (área semitransparente).
4. Panel auditoría pública: tabla últimas 30 predicciones vs real + error
   absoluto; métricas rolling (MAE 7d/30d desde history); fecha congelado receta.
5. Footer: disclaimer D7 + link repo GitHub.

Fetch de `../output/predictions.json` y `../output/history.jsonl` (rutas
relativas; si Pages sirve solo `frontend/`, copiar JSONs a `frontend/data/`
en el workflow diario — elegir lo más simple y documentarlo en README).

Manejo de error: si fetch falla, mostrar banner "datos no disponibles".

## 10. Documentación a generar

### `README.md`

- Qué predice + disclaimer visible arriba
- Install (sección 3) + verificación CPU
- Uso: notebook exploración, predict, retrain, monitor, re-exploración manual
- CI: 3 workflows explicados en 3 líneas
- Frontend: cómo se publica
- Estructura repo (árbol sección 5)

### `docs/DECISIONS.md`

- Tabla D1-D7 con justificación 1 línea cada una
- Limitaciones: serie corta (~2 años), mercado informal con shocks
  estructurales, datos de una sola fuente, sin variables exógenas
- Disclaimer completo: no consejo financiero, fines educativos/experimentales

## 11. Criterios de aceptación (checklist ejecutor)

- [ ] `python -m src.data_loader` pasa asserts (sección 6.1)
- [ ] Notebook ejecuta top-to-bottom sin error, `models/recipe.json` generado
- [ ] Leaderboard muestra ensemble superando naive en MAE (o markdown justifica)
- [ ] `python -m src.predict` genera `output/predictions.json` válido (schema 7.1)
- [ ] `python -m src.retrain` reentrena y actualiza `last_retrain`
- [ ] `python -m src.monitor` exit 0 con datos actuales
- [ ] 3 workflows con sintaxis válida (`actionlint` o YAML parse)
- [ ] `frontend/index.html` renderiza con JSONs de ejemplo (abrir local)
- [ ] README + DECISIONS.md completos, disclaimer presente en README,
      predictions.json y frontend
- [ ] Todo en rama `AutoGluon`, commits atómicos por componente

## 12. Non-goals (NO implementar)

- GPU / Chronos / modelos LLM
- Backend, base de datos, API propia
- Re-exploración automática en CI
- Multi-moneda activa (solo la PREPARACIÓN para ella, D1)
- Tests con framework (asserts en `__main__` bastan)
- Alertas por email/Telegram (fallo de job GitHub = alerta suficiente)

## 13. Orden de implementación sugerido

1. `src/data_loader.py` (+ verificar asserts)
2. Notebook celdas 1-10 (fit puede tardar ~30 min CPU)
3. `src/predict.py` → `src/retrain.py` → `src/monitor.py`
4. Workflows (validar YAML)
5. `frontend/index.html`
6. README + DECISIONS.md
7. Commit final en rama `AutoGluon`
