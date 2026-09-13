# AutoGluon Time Series - instalación sin GPU (CPU-only)

Requiere Python 3.10 - 3.13. Linux / macOS / Windows.

## Comando (recomendado)

```bash
pip install -U pip setuptools wheel
pip install autogluon.timeseries --extra-index-url https://download.pytorch.org/whl/cpu
```

`--extra-index-url` fuerza PyTorch CPU (~200 MB). Sin él, pip baja PyTorch CUDA (~2 GB+, exige GPU NVIDIA).

## Alternativas

uv (más rápido):

```bash
pip install -U uv
python -m uv pip install autogluon.timeseries --extra-index-url https://download.pytorch.org/whl/cpu
```

Conda:

```bash
conda create -n ag python=3.10
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon.timeseries
```

Instalar todo AutoGluon (no recomendado, pesa más):

```bash
pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu
```

## Notas

- Solo `autogluon.timeseries`, no el paquete completo → menos dependencias.
- Modelos clásicos (ETS, ARIMA, Theta, LightGBM, deep learning) funcionan en CPU normal.
- Modelos tipo Chronos (LLM-based) van lentos sin GPU.
- macOS no soporta GPU en AutoGluon; mismo comando CPU.
- Verificar:

```bash
python -c "import torch; print(torch.cuda.is_available())"  # debe dar False
python -c "from autogluon.timeseries import TimeSeriesPredictor; print('ok')"
```