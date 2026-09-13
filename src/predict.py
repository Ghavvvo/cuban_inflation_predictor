"""Predicción diaria con la receta congelada. NUNCA re-entrena."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.data_loader import load_series

PREDICTION_LENGTH = 7
PREDICTOR_PATH = "models/predictor"
RECIPE_PATH = "models/recipe.json"
OUTPUT_PATH = "output/predictions.json"
HISTORY_PATH = "output/history.jsonl"


def log_error(source: str, msg: str) -> None:
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as fh:
        fh.write(json.dumps({
            "type": "error",
            "at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "message": msg,
        }) + "\n")


def upsert_predictions(path: Path, forecast: list[dict], generated_at: str) -> list[dict]:
    """Histórico con las fechas del forecast reemplazadas (sin duplicados)."""
    dates = {p["date"] for p in forecast}
    lines = path.read_text().splitlines() if path.exists() else []
    events = []
    for line in lines:
        if not line:
            continue
        event = json.loads(line)
        is_stale = event["type"] == "prediction" and event["date"] in dates
        if not is_stale:
            events.append(event)
    for p in forecast:
        events.append({
            "type": "prediction",
            "generated_at": generated_at,
            "date": p["date"],
            "predicted": p["mean"],
            "actual": None,
        })
    return events


def main(out: str = OUTPUT_PATH) -> int:
    df_long = load_series(refresh=True)
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df_long, id_column="item_id", timestamp_column="timestamp"
    )
    predictor = TimeSeriesPredictor.load(PREDICTOR_PATH)
    forecast = predictor.predict(tsdf)
    f = forecast.loc["USD"]

    last_ts = tsdf.index.get_level_values("timestamp").max()
    last_val = float(tsdf.loc["USD"]["target"].iloc[-1])

    recipe = json.load(open(RECIPE_PATH))
    test_mase = None
    lb = recipe.get("leaderboard") or []
    if lb:
        test_mase = lb[0].get("score_test")

    pred = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "item_id": "USD",
        "horizon_days": PREDICTION_LENGTH,
        "last_observed": {"date": str(last_ts.date()), "value": last_val},
        "forecast": [
            {
                "date": str(ts.date()),
                "mean": round(float(f.loc[ts, "mean"]), 2),
                "p10": round(float(f.loc[ts, "0.1"]), 2),
                "p90": round(float(f.loc[ts, "0.9"]), 2),
            }
            for ts in f.index
        ],
        "model": {
            "recipe_frozen_at": recipe.get("frozen_at"),
            "eval_metric": recipe.get("eval_metric"),
            "test_mase": test_mase,
        },
        "disclaimer": "No es consejo financiero. Modelo experimental.",
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(pred, fh, indent=2, ensure_ascii=False)

    events = upsert_predictions(
        Path(HISTORY_PATH), pred["forecast"], pred["generated_at"]
    )
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(HISTORY_PATH).write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )

    print(f"predictions escritas en {out} ({len(pred['forecast'])} días)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 - CI usa exit!=0 como alerta
        print("ERROR:", e, file=sys.stderr)
        log_error("predict", str(e))
        sys.exit(1)
