"""
Shared nexus engine and helpers for optimization (pymoo), sensitivity (SALib),
and file-coupled simulation stubs. Keeps one analytical core for the DSS.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_TEMPLATES = PROJECT_ROOT / "model_templates"
COUPLED_RUNNER = MODEL_TEMPLATES / "coupled_runner.py"


def run_integrated_engine(
    year: int,
    country_name: str,
    p_mod: str,
    s_mod: str,
    d_meth: str,
    schedule: str,
    sim_eng: str,
    f_len: float,
    f_slp: float,
    crop_type: str,
    use_ml_surrogate: bool,
    use_lstm: bool,
    climate_scenario: str,
    climate_sensitivity: float,
    use_breeding: bool,
    *,
    temp_scale: float = 1.0,
    supply_perturb: float = 1.0,
    demand_perturb: float = 1.0,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Integrated supply–demand–crop nexus (original formulation + perturbation hooks)."""
    years_ahead = max(0, year - datetime.now().year)
    stats_map = {
        "Ethiopia": {"pop": 126.0, "supply": 122.0, "growth": 0.026, "ag_ratio": 0.82, "per_capita": 250},
        "Kenya": {"pop": 55.0, "supply": 30.0, "growth": 0.019, "ag_ratio": 0.75, "per_capita": 280},
        "Sudan": {"pop": 48.0, "supply": 35.0, "growth": 0.024, "ag_ratio": 0.80, "per_capita": 300},
        "Egypt": {"pop": 112.0, "supply": 58.0, "growth": 0.016, "ag_ratio": 0.85, "per_capita": 550},
    }
    base = stats_map.get(
        country_name,
        {"pop": 50.0, "supply": 40.0, "growth": 0.02, "ag_ratio": 0.80, "per_capita": 300},
    )

    p_growth = base["growth"] + (0.005 if p_mod == "Bayesian" else 0)
    proj_pop = base["pop"] * ((1 + p_growth) ** years_ahead)
    scenario_multiplier = {"SSP1-2.6": 0.5, "SSP2-4.5": 1.0, "SSP5-8.5": 2.5}.get(climate_scenario, 1.0)
    temp_rise = years_ahead * 0.045 * scenario_multiplier * climate_sensitivity * temp_scale
    climate_factor = max(0.1, 1 - (temp_rise * 0.02))

    supply_mult = {"SWAT": 1.15, "MODFLOW": 0.95, "SWAT + MODFLOW": 1.25}.get(s_mod, 1.0)
    if use_ml_surrogate:
        supply_mult *= 1.08
    total_water_supply = base["supply"] * supply_mult * climate_factor * supply_perturb

    d_factor = {"System Dynamics-WEAP": 1.0, "Agent-Based": 1.10}.get(d_meth, 1.0)
    if use_lstm:
        d_factor *= 0.94
    total_water_demand = (
        (proj_pop * base["per_capita"] / 1000) * d_factor * (1.02 ** (years_ahead / 10)) * demand_perturb
    )

    ag_water_demand = total_water_demand * base["ag_ratio"]
    sys_eff = ((1.0 - (f_len / 1800)) + (f_slp * 0.3)) * {
        "Soil moisture based": 0.92,
        "ET-based": 0.82,
        "Traditional": 0.55,
    }.get(schedule, 0.70)
    ag_water_consumed = ag_water_demand / max(sys_eff, 1e-6)

    wp_base = {"Wheat": 1.20, "Maize": 1.85, "Teff": 0.75}.get(crop_type, 1.0)
    wp_index = wp_base * sys_eff * climate_factor
    breeding_gain = 1.25 if use_breeding else 1.0
    pred_yield = (wp_index * (ag_water_consumed * 0.45)) * breeding_gain

    return (
        proj_pop,
        total_water_supply,
        total_water_demand,
        ag_water_demand,
        ag_water_consumed,
        pred_yield,
        wp_index,
        temp_rise,
        sys_eff,
    )


def engine_metrics_dict(
    year: int,
    country_name: str,
    p_mod: str,
    s_mod: str,
    d_meth: str,
    schedule: str,
    sim_eng: str,
    f_len: float,
    f_slp: float,
    crop_type: str,
    use_ml_surrogate: bool,
    use_lstm: bool,
    climate_scenario: str,
    climate_sensitivity: float,
    use_breeding: bool,
    **kwargs: Any,
) -> Dict[str, float]:
    t = run_integrated_engine(
        year,
        country_name,
        p_mod,
        s_mod,
        d_meth,
        schedule,
        sim_eng,
        f_len,
        f_slp,
        crop_type,
        use_ml_surrogate,
        use_lstm,
        climate_scenario,
        climate_sensitivity,
        use_breeding,
        **kwargs,
    )
    pop, supply, demand, ag_demand, ag_cons, yld, wp, temp_rise, eff = t
    return {
        "pop": pop,
        "supply": supply,
        "demand": demand,
        "ag_demand": ag_demand,
        "ag_cons": ag_cons,
        "yield": yld,
        "wp": wp,
        "temp_rise": temp_rise,
        "eff": eff,
    }


def _pack_context(
    year: int,
    country: str,
    p_mod: str,
    s_mod: str,
    d_meth: str,
    schedule: str,
    sim_eng: str,
    crop_type: str,
    use_ml_surrogate: bool,
    use_lstm: bool,
    climate_scenario: str,
    climate_sensitivity: float,
    use_breeding: bool,
) -> Dict[str, Any]:
    return {
        "year": year,
        "country": country,
        "p_mod": p_mod,
        "s_mod": s_mod,
        "d_meth": d_meth,
        "schedule": schedule,
        "sim_eng": sim_eng,
        "crop_type": crop_type,
        "use_ml_surrogate": use_ml_surrogate,
        "use_lstm": use_lstm,
        "climate_scenario": climate_scenario,
        "climate_sensitivity": climate_sensitivity,
        "use_breeding": use_breeding,
    }


def pymoo_objective_vector(
    f_len: float,
    f_slp: float,
    ctx: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Multi-objective: minimize agricultural water consumed (BCM), maximize water productivity.
    pymoo minimizes both → second objective is (-wp).
    """
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
    return m["ag_cons"], -m["wp"]


def run_pymoo_nsga2(
    ctx: Dict[str, Any],
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """NSGA-II on (f_len, f_slp). Returns (X, F) arrays."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    class NexusMOO(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=2, n_ieq_constr=0, xl=np.array([50.0, 0.01]), xu=np.array([400.0, 0.5]))

        def _evaluate(self, x, out, *args, **kwargs):
            f0, f1 = pymoo_objective_vector(float(x[0]), float(x[1]), ctx)
            out["F"] = np.array([f0, f1])

    problem = NexusMOO()
    algorithm = NSGA2(pop_size=pop_size, seed=seed)
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=seed, verbose=False)
    if res.X is None or res.F is None:
        raise RuntimeError("NSGA-II did not produce any Pareto solutions.")
    return res.X, res.F


def run_pymoo_ga_single_objective(
    ctx: Dict[str, Any],
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """GA style: single objective minimize (ag_cons / wp) — water per unit productivity."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    class NexusSOO(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, xl=np.array([50.0, 0.01]), xu=np.array([400.0, 0.5]))

        def _evaluate(self, x, out, *args, **kwargs):
            m = engine_metrics_dict(
                ctx["year"],
                ctx["country"],
                ctx["p_mod"],
                ctx["s_mod"],
                ctx["d_meth"],
                ctx["schedule"],
                ctx["sim_eng"],
                float(x[0]),
                float(x[1]),
                ctx["crop_type"],
                ctx["use_ml_surrogate"],
                ctx["use_lstm"],
                ctx["climate_scenario"],
                ctx["climate_sensitivity"],
                ctx["use_breeding"],
            )
            ag_cons, wp = m["ag_cons"], m["wp"]
            out["F"] = np.array([ag_cons / max(wp, 1e-6)])

    problem = NexusSOO()
    algorithm = GA(pop_size=pop_size, seed=seed)
    res = minimize(
        problem, algorithm, get_termination("n_gen", n_gen), seed=seed, verbose=False, save_history=True
    )
    if res.X is None or res.F is None:
        raise RuntimeError("The genetic algorithm did not produce a solution.")
    hist = []
    if res.history:
        for h in res.history:
            opt = getattr(h, "opt", None)
            if opt is not None and opt.get("F") is not None:
                hist.append(float(np.min(opt.get("F"))))
            else:
                pop = h.pop
                if pop.get("F") is not None:
                    hist.append(float(np.min(pop.get("F"))))
    return res.X, res.F, hist


# --- SALib problem spec (names aligned with evaluate_sobol_row) ---
def salib_problem_spec() -> Dict[str, Any]:
    return {
        "num_vars": 5,
        "names": ["climate_sensitivity", "f_len", "f_slp", "temp_scale", "supply_perturb"],
        "bounds": [[0.5, 3.0], [50.0, 400.0], [0.01, 0.5], [0.8, 1.2], [0.85, 1.15]],
    }


def evaluate_sobol_batch(
    X: np.ndarray,
    ctx: Dict[str, Any],
    output: str = "wp",
) -> np.ndarray:
    """Vector model evaluation for SALib samples. X shape (N, 5)."""
    Y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        row = X[i]
        cs = float(row[0])
        fl = float(row[1])
        fs = float(row[2])
        ts = float(row[3])
        sp = float(row[4])
        m = engine_metrics_dict(
            ctx["year"],
            ctx["country"],
            ctx["p_mod"],
            ctx["s_mod"],
            ctx["d_meth"],
            ctx["schedule"],
            ctx["sim_eng"],
            fl,
            fs,
            ctx["crop_type"],
            ctx["use_ml_surrogate"],
            ctx["use_lstm"],
            ctx["climate_scenario"],
            cs,
            ctx["use_breeding"],
            temp_scale=ts,
            supply_perturb=sp,
            demand_perturb=1.0,
        )
        if output == "wp":
            Y[i] = m["wp"]
        elif output == "ag_cons":
            Y[i] = m["ag_cons"]
        else:
            Y[i] = m["yield"]
    return Y


def run_sobol_analysis(
    ctx: Dict[str, Any],
    n_samples: int = 256,
    calc_second_order: bool = False,
    output: str = "wp",
    seed: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray]:
    from SALib.sample import saltelli as saltelli_sample
    from SALib.analyze import sobol as sobol_analyze

    problem = salib_problem_spec()
    rng = np.random.default_rng(seed)
    X = saltelli_sample.sample(problem, n_samples, calc_second_order=calc_second_order)
    Y = evaluate_sobol_batch(X, ctx, output=output)
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
    return Si, Y


def run_morris_analysis(
    ctx: Dict[str, Any],
    n_trajectories: int = 40,
    num_levels: int = 4,
    output: str = "wp",
    seed: int = 42,
) -> Dict[str, Any]:
    from SALib.analyze import morris as morris_analyze
    from SALib.sample.morris import sample as morris_sample

    problem = salib_problem_spec()
    X = morris_sample(problem, N=n_trajectories, num_levels=num_levels, seed=seed)
    Y = evaluate_sobol_batch(X, ctx, output=output)
    Mi = morris_analyze.analyze(problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=num_levels)
    return Mi


def monte_carlo_multiparam(
    ctx: Dict[str, Any],
    n: int = 1000,
    seed: int = 42,
) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    bounds = salib_problem_spec()["bounds"]
    rows = []
    for _ in range(n):
        climate_sensitivity = rng.uniform(bounds[0][0], bounds[0][1])
        f_len = rng.uniform(bounds[1][0], bounds[1][1])
        f_slp = rng.uniform(bounds[2][0], bounds[2][1])
        temp_scale = rng.uniform(bounds[3][0], bounds[3][1])
        supply_perturb = rng.uniform(bounds[4][0], bounds[4][1])
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
            climate_sensitivity,
            ctx["use_breeding"],
            temp_scale=temp_scale,
            supply_perturb=supply_perturb,
            demand_perturb=1.0,
        )
        rows.append(
            {
                "climate_sensitivity": climate_sensitivity,
                "f_len": f_len,
                "f_slp": f_slp,
                "temp_scale": temp_scale,
                "supply_perturb": supply_perturb,
                "wp": m["wp"],
                "ag_cons": m["ag_cons"],
                "yield": m["yield"],
            }
        )
    return pd.DataFrame(rows)


# --- File-based coupling (optimizer ↔ subprocess ↔ read outputs) ---

AQUACROP_TEMPLATE = """# AquaCrop-style parameter stub (replace tokens)
FURROW_LENGTH_M = {F_LEN}
FURROW_SLOPE_PCT = {F_SLP}
CROP = {CROP}
CLIMATE_SENSITIVITY = {CLIM}
SIM_ENGINE = {ENG}
"""


def write_template_inputs(workdir: Path, f_len: float, f_slp: float, crop: str, climate_sensitivity: float, sim_eng: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    inp = workdir / "aquacrop_run.txt"
    text = AQUACROP_TEMPLATE.format(
        F_LEN=f_len,
        F_SLP=f_slp,
        CROP=crop,
        CLIM=climate_sensitivity,
        ENG=sim_eng,
    )
    inp.write_text(text, encoding="utf-8")
    return inp


def run_file_coupled_simulation(
    ctx: Dict[str, Any],
    f_len: float,
    f_slp: float,
    timeout_s: int = 60,
) -> Dict[str, float]:
    """
    Optimizer → write params → subprocess runner → read CSV.
    Runner re-invokes the same analytical core for a reproducible closed loop.
    """
    if not COUPLED_RUNNER.is_file():
        return engine_metrics_dict(
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

    workdir = Path(tempfile.mkdtemp(prefix="tokuma_coupled_"))
    write_template_inputs(workdir, f_len, f_slp, ctx["crop_type"], ctx["climate_sensitivity"], ctx["sim_eng"])
    params = {
        "ctx": ctx,
        "f_len": f_len,
        "f_slp": f_slp,
        "workdir": str(workdir),
    }
    param_file = workdir / "params.json"
    param_file.write_text(json.dumps(params, default=str), encoding="utf-8")

    cmd = [sys.executable, str(COUPLED_RUNNER), str(param_file)]
    subprocess.run(cmd, check=True, cwd=str(MODEL_TEMPLATES.parent), timeout=timeout_s)

    out_csv = workdir / "sim_output.csv"
    if not out_csv.is_file():
        raise FileNotFoundError(f"Expected output {out_csv}")
    import csv

    with open(out_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        row = next(iter(r))
    return {k: float(row[k]) for k in row}


def pymoo_with_file_coupling(
    ctx: Dict[str, Any],
    pop_size: int = 30,
    n_gen: int = 40,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """NSGA-II using file-coupled objective (subprocess each eval). Slower but demonstrates the loop."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    class NexusFileMOO(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=2, n_ieq_constr=0, xl=np.array([50.0, 0.01]), xu=np.array([400.0, 0.5]))

        def _evaluate(self, x, out, *args, **kwargs):
            m = run_file_coupled_simulation(ctx, float(x[0]), float(x[1]))
            ag_cons, wp = m["ag_cons"], m["wp"]
            out["F"] = np.array([ag_cons, -wp])

    problem = NexusFileMOO()
    algorithm = NSGA2(pop_size=pop_size, seed=seed)
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=seed, verbose=False)
    if res.X is None or res.F is None:
        raise RuntimeError("File-coupled NSGA-II did not produce any Pareto solutions.")
    return res.X, res.F
