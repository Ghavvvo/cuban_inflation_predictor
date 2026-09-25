"""Reentrenamiento semanal de la receta congelada (sin re-explorar)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.data_loader import load_series

RECIPE_PATH = "models/recipe.json"
PREDICTOR_PATH = "models/predictor"
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


def calculate_baselines(tsdf: TimeSeriesDataFrame,
                        prediction_length: int,
                        window_days: int = 30) -> dict:
    values = tsdf.loc["USD"]["target"]
    evaluation_values = values.tail(window_days)
    baseline_values = evaluation_values.iloc[:-prediction_length]
    test_values = evaluation_values.iloc[-prediction_length:]
    naive_prediction = baseline_values.iloc[-1]
    moving_avg_prediction = baseline_values.tail(7).mean()

    return {
        "naive": {
            "MAE": float((test_values - naive_prediction).abs().mean()),
        },
        "moving_avg_7": {
            "MAE": float((test_values - moving_avg_prediction).abs().mean()),
        },
    }


def main(reexplore: bool = False) -> int:
    if reexplore:
        # D3: re-exploración es MANUAL, vive en el notebook (celda final).
        print("Re-exploración manual: usar notebook celda final (D3).",
              file=sys.stderr)
        return 0

    recipe = json.load(open(RECIPE_PATH))
    df_long = load_series(refresh=True)
    tsdf = TimeSeriesDataFrame.from_data_frame(
        df_long, id_column="item_id", timestamp_column="timestamp"
    )

    predictor = TimeSeriesPredictor(
        prediction_length=recipe["prediction_length"],
        freq=recipe["freq"],
        eval_metric=recipe["eval_metric"],
        path=PREDICTOR_PATH,
    )
    predictor.fit(
        tsdf,
        presets=recipe["presets"],
        time_limit=1800,
        excluded_model_types=recipe.get("excluded_model_types", ["Chronos"]),
        verbosity=1,
    )

    best_model = recipe["best_model"]
    leaderboard = predictor.leaderboard()
    leaderboard_rows = leaderboard.reset_index().to_dict(orient="records")
    for row in leaderboard_rows:
        score_val = row.get("score_val")
        if score_val is not None:
            row["score_val"] = abs(score_val)
        score_test = row.get("score_test")
        if score_test is not None:
            row["score_test"] = abs(score_test)

    frozen_score_val = None
    for row in leaderboard_rows:
        if row.get("model") == best_model:
            frozen_score_val = row.get("score_val")
            break

    recipe["last_retrain"] = pd.Timestamp.utcnow().isoformat()
    recipe["data_last_date"] = str(
        tsdf.index.get_level_values("timestamp").max().date()
    )
    recipe["baselines"] = calculate_baselines(
        tsdf, recipe["prediction_length"]
    )
    recipe["leaderboard"] = leaderboard_rows
    with open(RECIPE_PATH, "w") as f:
        json.dump(recipe, f, indent=2, default=str)

    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as fh:
        fh.write(json.dumps({
            "type": "retrain",
            "at": recipe["last_retrain"],
            "best_model": best_model,
            "score_val": frozen_score_val,
        }) + "\n")

    print("retrain OK. best_model:", best_model)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reexplore", action="store_true")
    args = parser.parse_args()
    try:
        sys.exit(main(reexplore=args.reexplore))
    except Exception as e:  # noqa: BLE001
        print("ERROR:", e, file=sys.stderr)
        log_error("retrain", str(e))
        sys.exit(1)
