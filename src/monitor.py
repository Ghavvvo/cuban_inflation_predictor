"""Monitoreo: histórico, actualización de reales, detección de degradación.

Exit 0 = sano. Exit 2 = degradado (MAE_7d > 1.5× baseline naive/moving_avg).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_loader import load_series

HISTORY_PATH = "output/history.jsonl"
RECIPE_PATH = "models/recipe.json"
METRICS_PATH = "output/metrics.json"


def write_metrics(mae7: float | None, mae30: float | None,
                  baseline_mae: float | None,
                  degraded: bool) -> None:
    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as fh:
        json.dump({
            "mae_7d": mae7,
            "mae_30d": mae30,
            "baseline_mae": baseline_mae,
            "degraded": degraded,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }, fh, indent=2)


def main() -> int:
    history = Path(HISTORY_PATH)
    if not history.exists():
        write_metrics(None, None, None, False)
        print("sin historial aún")
        return 0

    events = [json.loads(l) for l in history.read_text().splitlines() if l.strip()]
    preds = [e for e in events if e.get("type") == "prediction"]
    if not preds:
        write_metrics(None, None, None, False)
        print("sin predicciones en historial")
        return 0

    # actualizar `actual` con datos reales ya observados
    df = load_series()
    usd = df[df["item_id"] == "USD"]
    actual_map = dict(zip(
        usd["timestamp"].dt.strftime("%Y-%m-%d"), usd["target"]
    ))
    changed = False
    for e in preds:
        if e.get("actual") is None and e["date"] in actual_map:
            e["actual"] = float(actual_map[e["date"]])
            changed = True
    if changed:
        history.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    done = pd.DataFrame([e for e in preds if e.get("actual") is not None])
    if done.empty:
        write_metrics(None, None, None, False)
        print("aún sin predicciones con dato real")
        return 0
    done["err"] = (done["actual"] - done["predicted"]).abs()
    mae7 = float(done["err"].tail(7).mean())
    mae30 = float(done["err"].tail(30).mean())
    print(f"predicciones con real: {len(done)} | MAE_7d={mae7:.2f} "
          f"| MAE_30d={mae30:.2f}")

    recipe = json.load(open(RECIPE_PATH))
    baseline_mae = None
    for k in ("naive", "moving_avg_7"):
        v = recipe.get("baselines", {}).get(k, {}).get("MAE")
        if v is not None:
            baseline_mae = v
            break
    if baseline_mae is not None and mae7 > 1.5 * baseline_mae:
        write_metrics(mae7, mae30, baseline_mae, True)
        print("DEGRADACIÓN: MAE_7d > 1.5× baseline. Re-explorar (notebook celda final).",
              file=sys.stderr)
        return 2
    write_metrics(mae7, mae30, baseline_mae, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
