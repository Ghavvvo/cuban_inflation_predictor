"""Monitoreo: histórico, actualización de reales, detección de degradación.

Exit 0 = sano. Exit 2 = degradado (MAE_7d > 1.5× baseline naive/moving_avg).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import load_series

HISTORY_PATH = "output/history.jsonl"
RECIPE_PATH = "models/recipe.json"


def main() -> int:
    history = Path(HISTORY_PATH)
    if not history.exists():
        print("sin historial aún")
        return 0

    events = [json.loads(l) for l in history.read_text().splitlines() if l.strip()]
    preds = [e for e in events if e.get("type") == "prediction"]
    if not preds:
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
    if baseline_mae and mae7 > 1.5 * baseline_mae:
        print("DEGRADACIÓN: MAE_7d > 1.5× baseline. Re-explorar (notebook celda final).",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
