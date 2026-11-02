"""
Subprocess entry point: read params JSON → write template outputs → same nexus core → CSV.
Demonstrates file-based coupling (SWAT/AquaCrop-style workflow: prepare inputs, run, ingest).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Project root = parent of model_templates
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus_core import engine_metrics_dict  # noqa: E402


def main() -> None:
    param_path = Path(sys.argv[1])
    raw = json.loads(param_path.read_text(encoding="utf-8"))
    ctx = raw["ctx"]
    f_len = float(raw["f_len"])
    f_slp = float(raw["f_slp"])
    workdir = Path(raw["workdir"])

    m = engine_metrics_dict(
        ctx["year"],
        ctx["country"],
        ctx["p_mod"],
        ctx["s_mod"],
        ctx["d_meth"],
        ctx["schedule"],
        ctx["sim_eng"],
        f_len,
        f_slp,
        ctx["crop_type"],
        ctx["use_ml_surrogate"],
        ctx["use_lstm"],
        ctx["climate_scenario"],
        ctx["climate_sensitivity"],
        ctx["use_breeding"],
    )

    out = workdir / "sim_output.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pop", "supply", "demand", "ag_demand", "ag_cons", "yield", "wp", "temp_rise", "eff"],
        )
        w.writeheader()
        w.writerow(
            {
                "pop": m["pop"],
                "supply": m["supply"],
                "demand": m["demand"],
                "ag_demand": m["ag_demand"],
                "ag_cons": m["ag_cons"],
                "yield": m["yield"],
                "wp": m["wp"],
                "temp_rise": m["temp_rise"],
                "eff": m["eff"],
            }
        )


if __name__ == "__main__":
    main()
