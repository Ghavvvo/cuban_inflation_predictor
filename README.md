# Cuban Inflation Predictor

Predicción de la tasa de cambio informal **USD/CUP** (eltoque.com vía API
cubanomic) con **AutoGluon TimeSeries**, horizonte 7 días, CPU-only.

> **No es consejo financiero.** Modelo experimental con fines educativos.

## Instalación

Python 3.12. Instalar CPU-only (sin GPU):

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Verificar:

```bash
python -c "import torch; assert not torch.cuda.is_available()"
python -c "from autogluon.timeseries import TimeSeriesPredictor; print('ok')"
```

## Uso

| Comando | Qué hace |
|---------|----------|
| `python -m src.data_loader` | Carga datos + asserts (sanity) |
| `notebooks/autogluon_explore.ipynb` | Exploración completa: EDA → baselines → fit() único → congelar receta → predicción |
| `python -m src.predict` | Predice 7 días con la receta congelada → `output/predictions.json` + `output/history.jsonl` |
| `python -m src.retrain` | Reentrena la receta (semanal, sin re-explorar) |
| `python -m src.retrain --reexplore` | Avisa que la re-exploración es manual (notebook celda final) |
| `python -m src.monitor` | Actualiza reales, calcula MAE, detecta degradación (exit 2) |

### Re-exploración manual

Cada 3-6 meses o si `src.monitor` detecta degradación (MAE_7d > 1.5× baseline):
ejecutar la **celda final** del notebook (re-exploración completa, `time_limit`
mayor). Nunca automática en CI.

## Automatización (GitHub Actions)

| Workflow | Cron | Acción |
|----------|------|--------|
| `daily_predict.yml` | diario 07:00 Cuba | predice + commit JSONs |
| `weekly_retrain.yml` | domingo | reentrena receta + commit |
| `monthly_check.yml` | día 1 del mes | monitorea degradación (falla = alerta) |

## Frontend

Sitio estático en `frontend/index.html` (GitHub Pages). Consume
`../output/predictions.json` y `../output/history.jsonl`. Publicar Pages desde
la raíz del repo (Settings → Pages → Deploy from a branch → root `/`).

## Estructura

```
src/          data_loader, predict, retrain, monitor
notebooks/    autogluon_explore.ipynb
models/       recipe.json (receta congelada) + predictor/ (artefactos)
output/       predictions.json + history.jsonl
frontend/     index.html
docs/         DECISIONS.md
```

Ver `SPEC.md` para especificación técnica completa.
