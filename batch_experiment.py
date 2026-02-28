"""
Batch Experiment CLI for Monte Carlo parameter sweeps.

Runs the simulation multiple times across a grid of parameter values
and exports aggregated results to CSV.

Usage examples:
    # Sweep prestige sensitivity from 0.1 to 0.5 (step 0.05), 10 runs each
    python batch_experiment.py --param w_prestige --min 0.1 --max 0.5 --step 0.05 --runs 10

    # Sweep regional wage multiplier
    python batch_experiment.py --param regional_wage --min 1.0 --max 1.5 --step 0.1 --runs 5

    # Multi-parameter sweep from a JSON config file
    python batch_experiment.py --config experiments/sweep_config.json

    # Quick sweep of Tokyo tax premium
    python batch_experiment.py --param tokyo_tax --min 0 --max 2000 --step 250 --runs 5 --years 10
"""

import argparse
import json
import time
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from model.config import SimulationConfig, ScaleConfig, disable_gpu
from model.model import ExodusModel


PARAM_SETTERS = {
    "w_prestige": lambda cfg, v: setattr(cfg.weights, "w_prestige", v),
    "w_anomie": lambda cfg, v: setattr(cfg.weights, "w_anomie", v),
    "w_convenience": lambda cfg, v: setattr(cfg.weights, "w_convenience", v),
    "w_financial_friction": lambda cfg, v: setattr(cfg.weights, "w_financial_friction", v),
    "status_quo_bias": lambda cfg, v: setattr(cfg.behavior, "status_quo_bias", v),
    "tokyo_tax": lambda cfg, v: setattr(cfg.policy, "circular_tax_per_child_monthly", v),
    "regional_wage": lambda cfg, v: setattr(cfg.policy, "regional_wage_multiplier", v),
    "remote_work_pct": lambda cfg, v: setattr(cfg.policy, "remote_work_penetration", v),
    "childcare_subsidy": lambda cfg, v: setattr(cfg.policy, "childcare_subsidy_ratio", v),
    "housing_subsidy_peri": lambda cfg, v: setattr(cfg.policy, "housing_subsidy_periphery", v),
    "immigration_annual": lambda cfg, v: (
        setattr(cfg.policy, "immigration_active", v > 0),
        setattr(cfg.policy, "immigration_annual", int(v)),
    ),
    "ice_act": lambda cfg, v: setattr(cfg.policy, "hq_relocation_active", bool(v)),
}

PARAM_DESCRIPTIONS = {
    "w_prestige": "Utility weight: prestige sensitivity",
    "w_anomie": "Utility weight: anomie penalty",
    "w_convenience": "Utility weight: convenience",
    "w_financial_friction": "Utility weight: financial friction",
    "status_quo_bias": "Migration: status quo bias",
    "tokyo_tax": "Policy: circular tax (¥/month/child)",
    "regional_wage": "Policy: regional wage multiplier",
    "remote_work_pct": "Policy: remote work penetration",
    "childcare_subsidy": "Policy: childcare subsidy ratio",
    "housing_subsidy_peri": "Policy: periphery housing subsidy",
    "immigration_annual": "Policy: annual immigration count",
    "ice_act": "Policy: ICE Act enabled (0/1)",
}


def make_config(scale: int, years: int, seed: int) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.scale = ScaleConfig(agent_scale=scale, n_years=years, random_seed=seed)
    return cfg


def run_single_experiment(cfg: SimulationConfig, param_name: str,
                          param_value: float, run_idx: int) -> dict:
    """Run one simulation and extract key metrics."""
    model = ExodusModel(cfg)
    start = time.time()
    results = model.run()
    elapsed = time.time() - start

    history = results["history"]
    scale = cfg.scale.agent_scale
    stats = results["final_agent_stats"]

    last_year = history.iloc[-1] if not history.empty else {}

    n_alive = stats.get("n_alive", 0)
    total_pop = n_alive * scale
    mean_age = stats.get("mean_age", 0)
    pct_married = stats.get("pct_married", 0)

    annual = history.groupby("year")
    total_births = int(history["births"].sum()) * scale
    total_deaths = int(history["deaths"].sum()) * scale
    total_migrations = int(history["migrations"].sum())

    tokyo_share = float(last_year.get("tokyo_pop_share", 0))
    peri_share = float(last_year.get("peri_pop_share", 0))

    # TFR estimate from final year
    final_year_births = history[history["year"] == history["year"].max()]["births"].sum() * scale
    n_fertile_approx = total_pop * 0.22
    tfr_est = (final_year_births / max(n_fertile_approx, 1)) * 35 if n_fertile_approx > 0 else 0

    return {
        "param_name": param_name,
        "param_value": param_value,
        "run": run_idx,
        "seed": cfg.scale.random_seed,
        "final_population": total_pop,
        "mean_age": round(mean_age, 2),
        "pct_married": round(pct_married, 4),
        "tokyo_share": round(tokyo_share, 4),
        "periphery_share": round(peri_share, 4),
        "total_births": total_births,
        "total_deaths": total_deaths,
        "total_migrations": total_migrations,
        "tfr_estimate": round(tfr_est, 3),
        "runtime_s": round(elapsed, 1),
    }


def run_sweep(param_name: str, values: list, n_runs: int,
              scale: int, years: int, base_seed: int,
              output_path: str) -> pd.DataFrame:
    """Run a full parameter sweep with Monte Carlo repetitions."""
    all_results = []
    total = len(values) * n_runs
    done = 0

    print(f"\n{'='*70}")
    print(f"  BATCH EXPERIMENT: {param_name}")
    print(f"  Values: {[round(v, 4) for v in values]}")
    print(f"  Runs per value: {n_runs}")
    print(f"  Total simulations: {total}")
    print(f"{'='*70}\n")

    for vi, val in enumerate(values):
        for run_i in range(n_runs):
            seed = base_seed + run_i * 1000 + vi
            cfg = make_config(scale, years, seed)

            setter = PARAM_SETTERS.get(param_name)
            if setter is None:
                print(f"  [ERROR] Unknown parameter: {param_name}")
                print(f"  Available: {list(PARAM_SETTERS.keys())}")
                sys.exit(1)
            setter(cfg, val)

            done += 1
            print(f"  [{done}/{total}] {param_name}={val:.4f}, run={run_i+1}/{n_runs}, seed={seed} ... ", end="", flush=True)

            result = run_single_experiment(cfg, param_name, val, run_i)
            all_results.append(result)

            print(f"pop={result['final_population']:,.0f}, "
                  f"tokyo={result['tokyo_share']:.3f}, "
                  f"tfr={result['tfr_estimate']:.2f} "
                  f"({result['runtime_s']:.1f}s)")

    df = pd.DataFrame(all_results)

    # Aggregated summary
    summary = df.groupby("param_value").agg(
        mean_pop=("final_population", "mean"),
        std_pop=("final_population", "std"),
        mean_tokyo=("tokyo_share", "mean"),
        std_tokyo=("tokyo_share", "std"),
        mean_peri=("periphery_share", "mean"),
        mean_age=("mean_age", "mean"),
        mean_tfr=("tfr_estimate", "mean"),
        mean_births=("total_births", "mean"),
        mean_deaths=("total_deaths", "mean"),
        mean_migrations=("total_migrations", "mean"),
        n_runs=("run", "count"),
    ).reset_index()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary_path = output_path.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"  RESULTS SAVED")
    print(f"  Raw:     {output_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n  {'Value':>10} | {'Mean Pop':>12} | {'Tokyo %':>8} | {'Peri %':>8} | {'TFR':>6} | {'Migrations':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")
    for _, row in summary.iterrows():
        print(f"  {row['param_value']:>10.4f} | {row['mean_pop']:>12,.0f} | "
              f"{row['mean_tokyo']:>8.4f} | {row['mean_peri']:>8.4f} | "
              f"{row['mean_tfr']:>6.3f} | {row['mean_migrations']:>10,.0f}")

    return df


def run_multi_sweep(config_path: str, scale: int, years: int, base_seed: int):
    """Run multiple parameter sweeps defined in a JSON config file."""
    with open(config_path, "r") as f:
        experiments = json.load(f)

    for exp in experiments.get("experiments", []):
        param = exp["param"]
        vmin = exp.get("min", 0.1)
        vmax = exp.get("max", 0.5)
        vstep = exp.get("step", 0.05)
        n_runs = exp.get("runs", 10)

        values = list(np.arange(vmin, vmax + vstep * 0.5, vstep))
        output = f"output/batch/{param}_sweep.csv"

        run_sweep(param, values, n_runs, scale, years, base_seed, output)


def main():
    parser = argparse.ArgumentParser(
        description="Batch Monte Carlo Experiment Runner for Japan Exodus Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--param", type=str,
                        help=f"Parameter to sweep. Options: {list(PARAM_SETTERS.keys())}")
    parser.add_argument("--min", type=float, default=0.1, help="Minimum value")
    parser.add_argument("--max", type=float, default=0.5, help="Maximum value")
    parser.add_argument("--step", type=float, default=0.05, help="Step size")
    parser.add_argument("--runs", type=int, default=10, help="Runs per value (Monte Carlo)")
    parser.add_argument("--years", type=int, default=10, help="Simulation years")
    parser.add_argument("--scale", type=int, default=200, help="Agent scale (higher = faster)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file for multi-parameter sweeps")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only")
    parser.add_argument("--list-params", action="store_true",
                        help="List all sweepable parameters")

    args = parser.parse_args()

    if args.no_gpu:
        disable_gpu()

    if args.list_params:
        print("\nAvailable sweep parameters:")
        print(f"{'Parameter':<25} Description")
        print(f"{'-'*25} {'-'*45}")
        for k, desc in PARAM_DESCRIPTIONS.items():
            print(f"  {k:<23} {desc}")
        return

    if args.config:
        run_multi_sweep(args.config, args.scale, args.years, args.seed)
        return

    if not args.param:
        parser.print_help()
        print("\nError: --param is required (or use --config for multi-sweep)")
        sys.exit(1)

    values = list(np.arange(args.min, args.max + args.step * 0.5, args.step))
    output = args.output or f"output/batch/{args.param}_sweep.csv"

    run_sweep(args.param, values, args.runs, args.scale, args.years, args.seed, output)


if __name__ == "__main__":
    main()
