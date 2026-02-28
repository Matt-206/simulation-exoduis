"""
Japan Exodus Simulation -- Main Entry Point.

Supports multiple scenarios and batch comparison runs.

Usage:
    python run.py                      # Default baseline, 10 years
    python run.py --years 50           # Full 50-year projection
    python run.py --scenario optimistic
    python run.py --scenario pessimistic
    python run.py --compare            # Run all scenarios and compare
    python run.py --scale 500          # Faster: 1 agent = 500 people
    python run.py --no-gpu             # Force CPU-only mode
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.config import SimulationConfig, ScaleConfig, PolicyConfig, GPU_AVAILABLE, disable_gpu
from model.model import ExodusModel
from analysis.metrics import generate_full_report
from analysis.visualize import generate_all_plots, plot_population_pyramid
from data.generator import save_synthetic_datasets


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
def make_baseline_config(scale: int = 100, years: int = 10, seed: int = 42) -> SimulationConfig:
    """Status quo: current policies continue unchanged."""
    cfg = SimulationConfig()
    cfg.scale = ScaleConfig(agent_scale=scale, n_years=years, random_seed=seed)
    return cfg


def make_optimistic_config(scale: int = 100, years: int = 10, seed: int = 42) -> SimulationConfig:
    """Aggressive policy intervention scenario."""
    cfg = make_baseline_config(scale, years, seed)

    cfg.policy.hq_relocation_active = True
    cfg.policy.hq_relocation_prestige_boost = 0.25
    cfg.policy.remote_work_penetration = 0.40
    cfg.policy.remote_work_annual_growth = 0.04
    cfg.policy.childcare_subsidy_ratio = 0.80
    cfg.policy.childcare_expansion_rate = 0.06
    cfg.policy.housing_subsidy_periphery = 0.35
    cfg.policy.housing_subsidy_core = 0.20
    cfg.policy.medical_dx_rollout_rate = 0.08
    cfg.policy.level4_pod_deployment_rate = 0.07
    cfg.policy.regional_university_investment = 0.05

    # New policy levers
    cfg.policy.university_decentralization = True
    cfg.policy.n_regional_universities = 20
    cfg.policy.immigration_active = True
    cfg.policy.immigration_annual = 300_000
    cfg.policy.enterprise_zones_active = True
    cfg.policy.n_enterprise_zones = 30
    cfg.policy.shinkansen_expansion_active = True

    return cfg


def make_pessimistic_config(scale: int = 100, years: int = 10, seed: int = 42) -> SimulationConfig:
    """Policy stagnation: minimal intervention, Tokyo concentration continues."""
    cfg = make_baseline_config(scale, years, seed)

    cfg.policy.hq_relocation_active = False
    cfg.policy.remote_work_penetration = 0.15
    cfg.policy.remote_work_annual_growth = 0.005
    cfg.policy.childcare_subsidy_ratio = 0.30
    cfg.policy.childcare_expansion_rate = 0.01
    cfg.policy.housing_subsidy_periphery = 0.05
    cfg.policy.housing_subsidy_core = 0.02
    cfg.policy.medical_dx_rollout_rate = 0.02
    cfg.policy.level4_pod_deployment_rate = 0.01
    cfg.policy.circular_tax_per_child_monthly = 800

    cfg.policy.university_decentralization = False
    cfg.policy.immigration_active = False
    cfg.policy.enterprise_zones_active = False
    cfg.policy.shinkansen_expansion_active = False

    return cfg


def make_immigration_focus_config(scale: int = 100, years: int = 10, seed: int = 42) -> SimulationConfig:
    """High immigration scenario: open doors to compensate declining births."""
    cfg = make_baseline_config(scale, years, seed)

    cfg.policy.immigration_active = True
    cfg.policy.immigration_annual = 500_000
    cfg.policy.enterprise_zones_active = True
    cfg.policy.n_enterprise_zones = 40

    return cfg


def make_decentralization_config(scale: int = 100, years: int = 10, seed: int = 42) -> SimulationConfig:
    """Max decentralization: everything possible to move people out of Tokyo."""
    cfg = make_baseline_config(scale, years, seed)

    cfg.policy.hq_relocation_active = True
    cfg.policy.hq_relocation_prestige_boost = 0.30
    cfg.policy.university_decentralization = True
    cfg.policy.n_regional_universities = 25
    cfg.policy.enterprise_zones_active = True
    cfg.policy.n_enterprise_zones = 50
    cfg.policy.shinkansen_expansion_active = True
    cfg.policy.housing_subsidy_periphery = 0.40
    cfg.policy.housing_subsidy_core = 0.25
    cfg.policy.remote_work_penetration = 0.50
    cfg.policy.remote_work_annual_growth = 0.05

    return cfg


def make_takaichi_config(scale: int = 100, years: int = 10, seed: int = 42) -> SimulationConfig:
    """Takaichi 2.0: Full ICE Act + Digital Infrastructure + Regional Revival package."""
    cfg = make_baseline_config(scale, years, seed)

    p = cfg.policy
    # ICE Act levers
    p.hq_relocation_active = True
    p.hq_relocation_prestige_boost = 0.30
    p.ice_act_tax_credit = 0.12
    p.ice_act_investment_threshold = 2_000_000_000

    # Digital infrastructure
    p.medical_dx_rollout_rate = 0.10
    p.level4_pod_deployment_rate = 0.09
    p.regional_university_investment = 0.06

    # Remote work push
    p.remote_work_penetration = 0.45
    p.remote_work_annual_growth = 0.05

    # Family support
    p.childcare_subsidy_ratio = 0.85
    p.childcare_expansion_rate = 0.07
    p.circular_tax_per_child_monthly = 300

    # Housing
    p.housing_subsidy_periphery = 0.40
    p.housing_subsidy_core = 0.25

    # Regional revival
    p.university_decentralization = True
    p.n_regional_universities = 25
    p.enterprise_zones_active = True
    p.n_enterprise_zones = 40
    p.shinkansen_expansion_active = True
    p.regional_wage_multiplier = 1.15

    # Immigration
    p.immigration_active = True
    p.immigration_annual = 350_000

    return cfg


SCENARIOS = {
    "baseline": make_baseline_config,
    "takaichi": make_takaichi_config,
    "optimistic": make_optimistic_config,
    "pessimistic": make_pessimistic_config,
    "immigration": make_immigration_focus_config,
    "decentralization": make_decentralization_config,
}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(config: SimulationConfig, label: str = "baseline") -> dict:
    """Execute a single simulation run."""
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label.upper()}")
    print(f"{'='*60}")

    model = ExodusModel(config)
    start = time.time()
    results = model.run()
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Final population: {results['final_agent_stats']['n_alive']:,} agents "
          f"({results['final_agent_stats']['n_alive'] * config.scale.agent_scale:,} real)")
    print(f"Mean age: {results['final_agent_stats']['mean_age']:.1f}")
    print(f"Marriage rate: {results['final_agent_stats']['pct_married']:.1%}")

    # Generate report
    report = generate_full_report(results, config.scale.agent_scale)
    print(f"\nPopulation change: {report['population_change_pct']:+.1f}%")
    print(f"At-risk municipalities: {report['municipality_viability']['at_risk_municipalities']}"
          f" / {report['municipality_viability']['total_periphery']}"
          f" ({report['municipality_viability']['pct_at_risk']:.1%})")
    print(f"Population Gini: {report['population_gini']:.4f}")

    # Save legacy plots
    output_dir = f"output/{label}"
    figures = generate_all_plots(results, output_dir, config.scale.agent_scale)

    # Population pyramid
    pool = model.agents_pool
    n = pool.next_id
    fig_pyr = plot_population_pyramid(
        pool.age[:n], pool.sex[:n], pool.alive[:n],
        title=f"Population Pyramid -- {label.title()} ({model.current_year})",
        save_path=f"{output_dir}/population_pyramid.png",
    )
    figures.append(fig_pyr)

    # Generate publication-quality GeoJSON charts
    try:
        from visualization.charts import generate_all_charts
        chart_dir = f"output/{label}/charts"
        generate_all_charts(results, chart_dir, config.scale.agent_scale)
    except Exception as e:
        print(f"  [warn] Chart generation skipped: {e}")

    # Save history CSV
    results["history"].to_csv(f"{output_dir}/history.csv", index=False)
    if not results["migration_flows"].empty:
        results["migration_flows"].to_csv(f"{output_dir}/migration_flows.csv", index=False)

    plt.close("all")
    return results


# ---------------------------------------------------------------------------
# Comparison run
# ---------------------------------------------------------------------------
def run_comparison(scale: int, years: int, seed: int):
    """Run all scenarios and produce comparison plots."""
    all_results = {}

    for name, builder in SCENARIOS.items():
        config = builder(scale, years, seed)
        all_results[name] = run_single(config, label=name)

    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Total population
    ax = axes[0, 0]
    for name, res in all_results.items():
        h = res["history"]
        annual = h.groupby("year")["total_population"].last()
        ax.plot(annual.index, annual.values * scale, label=name.title(), linewidth=2.5)
    ax.set_title("Total Population Comparison")
    ax.set_ylabel("Population")
    ax.legend()

    # Tokyo share
    ax = axes[0, 1]
    for name, res in all_results.items():
        h = res["history"]
        annual = h.groupby("year")["tokyo_pop_share"].last()
        ax.plot(annual.index, annual.values * 100, label=name.title(), linewidth=2.5)
    ax.set_title("Tokyo Population Share")
    ax.set_ylabel("%")
    ax.legend()

    # Periphery share
    ax = axes[1, 0]
    for name, res in all_results.items():
        h = res["history"]
        annual = h.groupby("year")["peri_pop_share"].last()
        ax.plot(annual.index, annual.values * 100, label=name.title(), linewidth=2.5)
    ax.set_title("Periphery Population Share")
    ax.set_ylabel("%")
    ax.legend()

    # Migrations
    ax = axes[1, 1]
    for name, res in all_results.items():
        h = res["history"]
        annual = h.groupby("year")["migrations"].sum()
        ax.plot(annual.index, annual.values, label=name.title(), linewidth=2.5)
    ax.set_title("Annual Migrations")
    ax.set_ylabel("Count")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("Year")

    plt.tight_layout()
    Path("output/comparison").mkdir(parents=True, exist_ok=True)
    fig.savefig("output/comparison/scenario_comparison.png", dpi=200, bbox_inches="tight")
    print(f"\nComparison plot saved to output/comparison/scenario_comparison.png")
    plt.close("all")


# ---------------------------------------------------------------------------
# A/B Scenario Comparison
# ---------------------------------------------------------------------------
def run_ab_comparison(scenario_a: str, scenario_b: str, scale: int, years: int, seed: int):
    """Run two scenarios side-by-side and produce a detailed comparative report."""
    builder_a = SCENARIOS[scenario_a]
    builder_b = SCENARIOS[scenario_b]

    config_a = builder_a(scale, years, seed)
    config_b = builder_b(scale, years, seed)

    results_a = run_single(config_a, label=f"ab_{scenario_a}")
    results_b = run_single(config_b, label=f"ab_{scenario_b}")

    out_dir = Path(f"output/ab_{scenario_a}_vs_{scenario_b}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Detailed comparison charts ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    colors = {"A": "#d32f2f", "B": "#1565c0"}

    ha = results_a["history"]
    hb = results_b["history"]

    # 1. Population trajectory
    ax = axes[0, 0]
    ann_a = ha.groupby("year")["total_population"].last() * scale
    ann_b = hb.groupby("year")["total_population"].last() * scale
    ax.plot(ann_a.index, ann_a.values, color=colors["A"], linewidth=2.5,
            label=f"A: {scenario_a.title()}")
    ax.plot(ann_b.index, ann_b.values, color=colors["B"], linewidth=2.5,
            label=f"B: {scenario_b.title()}")
    ax.fill_between(ann_a.index, ann_a.values, ann_b.values, alpha=0.15,
                    color="#888888")
    ax.set_title("Total Population")
    ax.set_ylabel("Population")
    ax.legend()

    # 2. Tokyo share
    ax = axes[0, 1]
    ts_a = ha.groupby("year")["tokyo_pop_share"].last() * 100
    ts_b = hb.groupby("year")["tokyo_pop_share"].last() * 100
    ax.plot(ts_a.index, ts_a.values, color=colors["A"], linewidth=2.5,
            label=f"A: {scenario_a.title()}")
    ax.plot(ts_b.index, ts_b.values, color=colors["B"], linewidth=2.5,
            label=f"B: {scenario_b.title()}")
    ax.set_title("Tokyo Population Share (%)")
    ax.legend()

    # 3. Periphery share
    ax = axes[1, 0]
    ps_a = ha.groupby("year")["peri_pop_share"].last() * 100
    ps_b = hb.groupby("year")["peri_pop_share"].last() * 100
    ax.plot(ps_a.index, ps_a.values, color=colors["A"], linewidth=2.5,
            label=f"A: {scenario_a.title()}")
    ax.plot(ps_b.index, ps_b.values, color=colors["B"], linewidth=2.5,
            label=f"B: {scenario_b.title()}")
    ax.set_title("Periphery Population Share (%)")
    ax.legend()

    # 4. Annual births
    ax = axes[1, 1]
    births_a = ha.groupby("year")["births"].sum() * scale
    births_b = hb.groupby("year")["births"].sum() * scale
    ax.bar(births_a.index - 0.2, births_a.values, 0.4, color=colors["A"],
           alpha=0.7, label=f"A: {scenario_a.title()}")
    ax.bar(births_b.index + 0.2, births_b.values, 0.4, color=colors["B"],
           alpha=0.7, label=f"B: {scenario_b.title()}")
    ax.set_title("Annual Births")
    ax.legend()

    # 5. Cumulative migrations
    ax = axes[2, 0]
    mig_a = ha.groupby("year")["migrations"].sum().cumsum()
    mig_b = hb.groupby("year")["migrations"].sum().cumsum()
    ax.plot(mig_a.index, mig_a.values, color=colors["A"], linewidth=2.5,
            label=f"A: {scenario_a.title()}")
    ax.plot(mig_b.index, mig_b.values, color=colors["B"], linewidth=2.5,
            label=f"B: {scenario_b.title()}")
    ax.set_title("Cumulative Internal Migrations")
    ax.legend()

    # 6. Difference in population
    ax = axes[2, 1]
    diff = ann_a - ann_b
    pos = diff.clip(lower=0)
    neg = diff.clip(upper=0)
    ax.bar(diff.index, pos.values, color=colors["A"], alpha=0.7, label=f"A ahead")
    ax.bar(diff.index, neg.values, color=colors["B"], alpha=0.7, label=f"B ahead")
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_title(f"Population Difference (A - B)")
    ax.set_ylabel("People")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("Year")
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"A/B Scenario Comparison: {scenario_a.title()} vs {scenario_b.title()}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = out_dir / "ab_comparison.png"
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    print(f"\nA/B comparison saved to {save_path}")
    plt.close("all")

    # Text summary
    summary_path = out_dir / "ab_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"A/B Scenario Comparison: {scenario_a} vs {scenario_b}\n")
        f.write(f"{'='*60}\n\n")
        final_a = int(ann_a.iloc[-1]) if len(ann_a) else 0
        final_b = int(ann_b.iloc[-1]) if len(ann_b) else 0
        f.write(f"Final Population  A ({scenario_a}): {final_a:,}\n")
        f.write(f"Final Population  B ({scenario_b}): {final_b:,}\n")
        f.write(f"Difference: {final_a - final_b:+,} ({(final_a - final_b) / max(final_b, 1) * 100:+.1f}%)\n\n")
        f.write(f"Final Tokyo Share  A: {ts_a.iloc[-1]:.1f}%\n")
        f.write(f"Final Tokyo Share  B: {ts_b.iloc[-1]:.1f}%\n")
        f.write(f"Final Periphery Share  A: {ps_a.iloc[-1]:.1f}%\n")
        f.write(f"Final Periphery Share  B: {ps_b.iloc[-1]:.1f}%\n")
    print(f"A/B summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Japan Exodus Agent-Based Microsimulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="baseline",
                        help="Scenario to run (default: baseline)")
    parser.add_argument("--years", type=int, default=10,
                        help="Number of years to simulate (default: 10)")
    parser.add_argument("--scale", type=int, default=50,
                        help="Agent scale factor: 1 agent = N people (default: 50 → 2.5M agents)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all scenarios and compare")
    parser.add_argument("--compare-ab", nargs=2, metavar=("SCENARIO_A", "SCENARIO_B"),
                        help="A/B comparison of two scenarios (e.g. --compare-ab baseline optimistic)")
    parser.add_argument("--visual", action="store_true",
                        help="Launch real-time Pygame visualization")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only computation")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate synthetic datasets and exit")

    args = parser.parse_args()

    if args.no_gpu:
        disable_gpu()

    from model.config import GPU_AVAILABLE as GPU_STATUS
    print(f"Japan Exodus Simulation v3.0 (Tokyo Black Hole Edition)")
    print(f"GPU: {'CUDA available (CuPy)' if GPU_STATUS else 'CPU only (NumPy + Numba)'}")

    if args.generate_data:
        save_synthetic_datasets()
        return

    if args.visual:
        from visualization.live_view import LiveSimulationView
        visual_scale = args.scale
        builder = SCENARIOS[args.scenario]
        config = builder(visual_scale, args.years, args.seed)
        print(f"\nLaunching live visualization ({config.scale.n_agents:,} agents)...")
        model = ExodusModel(config)
        viewer = LiveSimulationView(model)
        viewer.run()
    elif args.compare_ab:
        sa, sb = args.compare_ab
        if sa not in SCENARIOS or sb not in SCENARIOS:
            print(f"Invalid scenario. Choose from: {list(SCENARIOS.keys())}")
            sys.exit(1)
        run_ab_comparison(sa, sb, args.scale, args.years, args.seed)
    elif args.compare:
        run_comparison(args.scale, args.years, args.seed)
    else:
        builder = SCENARIOS[args.scenario]
        config = builder(args.scale, args.years, args.seed)
        run_single(config, label=args.scenario)


if __name__ == "__main__":
    main()
