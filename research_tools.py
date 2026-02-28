"""
Research Instrument Suite for the Japan Exodus Simulation.

Provides publication-grade tools for academic analysis:
  1. Scenario Branching: counterfactual A/B from identical state
  2. Cannibalism Matrix Export: origin-tier migration CSV + launchpad ratio
  3. Human Warehouse Correlation: bivariate C vs A at 5-year intervals
  4. Monte Carlo Batch Runner: N-run averaging with confidence intervals
  5. Headless Heatmap Capture: auto-export net migration maps every 5 years

Usage:
    python research_tools.py branch --branch-year 2030 --end-year 2065
    python research_tools.py montecarlo --runs 50 --scenario baseline --years 40
    python research_tools.py heatmaps --scenario baseline --years 40
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model.config import SimulationConfig, ScaleConfig, PolicyConfig, GPU_AVAILABLE, disable_gpu
from model.model import ExodusModel


TIER_NAMES = {0: "Tokyo", 1: "Core", 2: "Periphery"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. SCENARIO BRANCHING
# ═══════════════════════════════════════════════════════════════════════════
def apply_intervention_policy(config: SimulationConfig):
    """Apply 'Takaichi 2.0' intervention package to a config."""
    p = config.policy
    p.hq_relocation_active = True
    p.hq_relocation_prestige_boost = 0.25
    p.remote_work_penetration = 0.40
    p.remote_work_annual_growth = 0.04
    p.childcare_subsidy_ratio = 0.80
    p.childcare_expansion_rate = 0.06
    p.housing_subsidy_periphery = 0.35
    p.housing_subsidy_core = 0.20
    p.university_decentralization = True
    p.n_regional_universities = 20
    p.immigration_active = True
    p.immigration_annual = 300_000
    p.enterprise_zones_active = True
    p.n_enterprise_zones = 30
    p.shinkansen_expansion_active = True


def run_scenario_branch(
    branch_year: int = 2030,
    end_year: int = 2065,
    scale: int = 200,
    seed: int = 42,
    output_dir: str = "output/branch",
):
    """Run baseline to branch_year, snapshot, then diverge into two futures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap_path = str(out / "branch_snapshot.pkl")

    # ── Phase 1: Shared history to branch point ──
    print(f"\n{'='*70}")
    print(f"  SCENARIO BRANCHING: {branch_year} -> {end_year}")
    print(f"  Phase 1: Shared baseline 2020 -> {branch_year}")
    print(f"{'='*70}\n")

    cfg_shared = SimulationConfig()
    start = cfg_shared.scale.start_year
    shared_years = branch_year - start
    if shared_years < 1:
        shared_years = 1
    cfg_shared.scale = ScaleConfig(
        agent_scale=scale,
        n_years=shared_years,
        random_seed=seed,
    )
    model_shared = ExodusModel(cfg_shared)
    t0 = time.time()
    model_shared.run()
    t_shared = time.time() - t0
    print(f"  Shared phase complete in {t_shared:.1f}s")
    print(f"  Population at {branch_year}: {model_shared.agents_pool.n_alive:,} agents")

    model_shared.save_snapshot(snap_path)
    print(f"  Snapshot saved: {snap_path}")

    # ── Phase 2a: Branch A — Baseline (laissez-faire) ──
    print(f"\n  Phase 2a: Branch A -- BASELINE -> {end_year}")
    cfg_a = SimulationConfig()
    remaining = end_year - branch_year
    cfg_a.scale = ScaleConfig(agent_scale=scale, n_years=remaining, random_seed=seed)
    model_a = ExodusModel(cfg_a)
    model_a.load_snapshot(snap_path)
    cfg_a.scale.n_years = remaining
    t0 = time.time()
    for _ in range(remaining * cfg_a.scale.steps_per_year):
        model_a.step()
    results_a = model_a.get_results()
    t_a = time.time() - t0
    print(f"  Branch A complete in {t_a:.1f}s")

    # ── Phase 2b: Branch B — Intervention ──
    print(f"\n  Phase 2b: Branch B -- INTERVENTION -> {end_year}")
    cfg_b = SimulationConfig()
    cfg_b.scale = ScaleConfig(agent_scale=scale, n_years=remaining, random_seed=seed)
    apply_intervention_policy(cfg_b)
    model_b = ExodusModel(cfg_b)
    model_b.load_snapshot(snap_path)
    cfg_b.scale.n_years = remaining
    t0 = time.time()
    for _ in range(remaining * cfg_b.scale.steps_per_year):
        model_b.step()
    results_b = model_b.get_results()
    t_b = time.time() - t0
    print(f"  Branch B complete in {t_b:.1f}s")

    # ── Export ──
    results_a["history"].to_csv(str(out / "branch_A_baseline.csv"), index=False)
    results_b["history"].to_csv(str(out / "branch_B_intervention.csv"), index=False)

    if not results_a["migration_flows"].empty:
        results_a["migration_flows"].to_csv(str(out / "branch_A_migrations.csv"), index=False)
    if not results_b["migration_flows"].empty:
        results_b["migration_flows"].to_csv(str(out / "branch_B_migrations.csv"), index=False)

    # ── Cannibalism Matrix export ──
    export_cannibalism_matrix(results_a, str(out / "cannibalism_A.csv"), "Baseline")
    export_cannibalism_matrix(results_b, str(out / "cannibalism_B.csv"), "Intervention")

    # ── Human Warehouse Correlation ──
    export_warehouse_correlation(model_a, str(out / "warehouse_corr_A.csv"))
    export_warehouse_correlation(model_b, str(out / "warehouse_corr_B.csv"))

    # ── Comparison report ──
    _write_branch_comparison(results_a, results_b, scale, branch_year, end_year, out)

    # ── Spreadsheet export ──
    try:
        from export_spreadsheet import FullDataCollector, build_workbook
        for label, model, results, rt in [("A_baseline", model_a, results_a, t_a),
                                           ("B_intervention", model_b, results_b, t_b)]:
            coll = FullDataCollector(model, scale)
            coll._collect_annual(model.current_year)
            wb = build_workbook(results["history"], results["migration_flows"],
                                coll, model, scale, label, rt)
            wb.save(str(out / f"branch_{label}.xlsx"))
            print(f"  Spreadsheet: branch_{label}.xlsx")
    except Exception as e:
        print(f"  [warn] Spreadsheet export failed: {e}")

    print(f"\n  All outputs saved to {out}/")
    return results_a, results_b


def export_cannibalism_matrix(results: dict, path: str, label: str = ""):
    """Export the full tier-to-tier migration matrix with origin tracking."""
    flows = results.get("migration_flows", pd.DataFrame())
    if flows.empty:
        print(f"  [warn] No migration flows to export for {label}")
        return

    # Aggregate by (from_tier, to_tier, origin_tier)
    if "origin_tier" in flows.columns:
        matrix = flows.groupby(["from_tier", "to_tier", "origin_tier"]).size().reset_index(name="count")
    else:
        matrix = flows.groupby(["from_tier", "to_tier"]).size().reset_index(name="count")

    matrix.to_csv(path, index=False)

    # Summary stats
    tier_map = {0: "Tokyo", 1: "Core", 2: "Periphery"}
    summary = flows.groupby(["from_tier", "to_tier"]).size().unstack(fill_value=0)
    summary.index = [tier_map.get(i, str(i)) for i in summary.index]
    summary.columns = [tier_map.get(i, str(i)) for i in summary.columns]

    # Launchpad ratio
    c2t = len(flows[(flows.from_tier == 1) & (flows.to_tier == 0)])
    c2t_peri_origin = len(flows[(flows.from_tier == 1) & (flows.to_tier == 0) &
                                (flows.origin_tier == 2)]) if "origin_tier" in flows.columns else 0
    launchpad = c2t_peri_origin / max(c2t, 1)

    print(f"\n  Cannibalism Matrix [{label}]:")
    print(f"    {summary.to_string()}")
    print(f"    Launchpad Ratio (Core->Tokyo from Periphery origin): {launchpad:.1%}")
    print(f"    Saved: {path}")


def export_warehouse_correlation(model: ExodusModel, path: str):
    """Export per-municipality Convenience vs Anomie bivariate data."""
    loc = model.loc_state
    n_locs = len(loc.tier)

    rows = []
    for i in range(n_locs):
        rows.append({
            "location_id": i,
            "tier": int(loc.tier[i]),
            "tier_name": TIER_NAMES.get(int(loc.tier[i]), "?"),
            "population": int(loc.population[i]),
            "convenience": round(float(loc.convenience[i]), 4),
            "anomie": round(float(loc.anomie[i]), 4),
            "prestige": round(float(loc.prestige[i]), 4),
            "financial_friction": round(float(loc.financial_friction[i]), 4),
            "healthcare_score": round(float(loc.healthcare_score[i]), 4),
            "digital_access": round(float(loc.digital_access[i]), 4),
            "vacancy_rate": round(float(loc.vacancy_rate[i]), 4),
            "rent_index": round(float(loc.rent_index[i]), 4),
            "year": model.current_year,
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Warehouse correlation data: {path}")


def _write_branch_comparison(results_a, results_b, scale, branch_year, end_year, out_dir):
    """Generate comparison charts and text summary."""
    ha = results_a["history"]
    hb = results_b["history"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    colors = {"A": "#c62828", "B": "#1565c0"}

    def annual(df, col):
        return df.groupby("year")[col].last()

    # Population
    ax = axes[0, 0]
    pa = annual(ha, "total_population") * scale
    pb = annual(hb, "total_population") * scale
    ax.axvline(branch_year, color="gray", ls="--", lw=1, label=f"Branch @ {branch_year}")
    ax.plot(pa.index, pa.values, color=colors["A"], lw=2.5, label="A: Baseline")
    ax.plot(pb.index, pb.values, color=colors["B"], lw=2.5, label="B: Intervention")
    ax.fill_between(pa.index, pa.values, pb.values, alpha=0.1, color="gray")
    ax.set_title("Total Population", fontweight="bold")
    ax.set_ylabel("People")
    ax.legend(fontsize=9)

    # Tokyo share
    ax = axes[0, 1]
    ta = annual(ha, "tokyo_pop_share") * 100
    tb = annual(hb, "tokyo_pop_share") * 100
    ax.axvline(branch_year, color="gray", ls="--", lw=1)
    ax.plot(ta.index, ta.values, color=colors["A"], lw=2.5, label="A: Baseline")
    ax.plot(tb.index, tb.values, color=colors["B"], lw=2.5, label="B: Intervention")
    ax.set_title("Tokyo Population Share (%)", fontweight="bold")
    ax.legend(fontsize=9)

    # Periphery share
    ax = axes[1, 0]
    pera = annual(ha, "peri_pop_share") * 100
    perb = annual(hb, "peri_pop_share") * 100
    ax.axvline(branch_year, color="gray", ls="--", lw=1)
    ax.plot(pera.index, pera.values, color=colors["A"], lw=2.5, label="A: Baseline")
    ax.plot(perb.index, perb.values, color=colors["B"], lw=2.5, label="B: Intervention")
    ax.set_title("Periphery Population Share (%)", fontweight="bold")
    ax.legend(fontsize=9)

    # TFR proxy
    ax = axes[1, 1]
    tfr_a = annual(ha, "tfr_proxy")
    tfr_b = annual(hb, "tfr_proxy")
    ax.axvline(branch_year, color="gray", ls="--", lw=1)
    ax.plot(tfr_a.index, tfr_a.values, color=colors["A"], lw=2.5, label="A: Baseline")
    ax.plot(tfr_b.index, tfr_b.values, color=colors["B"], lw=2.5, label="B: Intervention")
    ax.set_title("TFR Proxy", fontweight="bold")
    ax.legend(fontsize=9)

    # Cannibalism ratio
    ax = axes[2, 0]
    if "cannibalism_ratio" in ha.columns:
        cr_a = annual(ha, "cannibalism_ratio")
        cr_b = annual(hb, "cannibalism_ratio")
        ax.axvline(branch_year, color="gray", ls="--", lw=1)
        ax.plot(cr_a.index, cr_a.values, color=colors["A"], lw=2.5, label="A: Baseline")
        ax.plot(cr_b.index, cr_b.values, color=colors["B"], lw=2.5, label="B: Intervention")
    ax.set_title("Cannibalism Ratio (P->C / P->T)", fontweight="bold")
    ax.legend(fontsize=9)

    # Launchpad ratio
    ax = axes[2, 1]
    if "launchpad_ratio" in ha.columns:
        lp_a = annual(ha, "launchpad_ratio")
        lp_b = annual(hb, "launchpad_ratio")
        ax.axvline(branch_year, color="gray", ls="--", lw=1)
        ax.plot(lp_a.index, lp_a.values, color=colors["A"], lw=2.5, label="A: Baseline")
        ax.plot(lp_b.index, lp_b.values, color=colors["B"], lw=2.5, label="B: Intervention")
    ax.set_title("Launchpad Ratio (Core->Tokyo from Peri origin / all C->T)", fontweight="bold")
    ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("Year")
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"Counterfactual Analysis: Baseline vs Intervention\n"
        f"Branch point: {branch_year} | Projection: {end_year}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_dir / "branch_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Text summary
    with open(str(out_dir / "branch_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"COUNTERFACTUAL ANALYSIS: Baseline vs Intervention\n")
        f.write(f"{'='*60}\n")
        f.write(f"Branch year: {branch_year} | End year: {end_year}\n\n")
        final_a = int(pa.iloc[-1]) if len(pa) else 0
        final_b = int(pb.iloc[-1]) if len(pb) else 0
        f.write(f"Final Pop  A (Baseline):      {final_a:>14,}\n")
        f.write(f"Final Pop  B (Intervention):  {final_b:>14,}\n")
        f.write(f"Difference:                   {final_b - final_a:>+14,} ({(final_b - final_a)/max(final_a,1)*100:+.1f}%)\n\n")
        f.write(f"Final Tokyo Share A: {ta.iloc[-1]:.1f}%\n")
        f.write(f"Final Tokyo Share B: {tb.iloc[-1]:.1f}%\n")
        f.write(f"Final Peri Share  A: {pera.iloc[-1]:.1f}%\n")
        f.write(f"Final Peri Share  B: {perb.iloc[-1]:.1f}%\n\n")

        if "launchpad_ratio" in ha.columns:
            f.write(f"Final Launchpad Ratio A: {lp_a.iloc[-1]:.3f}\n")
            f.write(f"Final Launchpad Ratio B: {lp_b.iloc[-1]:.3f}\n")

    print(f"\n  Comparison saved: {out_dir / 'branch_comparison.png'}")
    print(f"  Summary saved:    {out_dir / 'branch_summary.txt'}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. HUMAN WAREHOUSE CORRELATION (bivariate C vs A)
# ═══════════════════════════════════════════════════════════════════════════
class WarehouseCorrelationCollector:
    """Collects C vs A data at 5-year intervals during simulation."""

    def __init__(self, model: ExodusModel, interval_years: int = 5):
        self.model = model
        self.interval = interval_years
        self.snapshots = []
        self._last_year = None

    def check_and_collect(self):
        yr = self.model.current_year
        q = self.model.current_step_in_year
        if q != 0:
            return
        if yr == self._last_year:
            return
        if yr % self.interval != 0:
            return

        loc = self.model.loc_state
        n_locs = len(loc.tier)
        for i in range(n_locs):
            self.snapshots.append({
                "year": yr,
                "location_id": i,
                "tier": int(loc.tier[i]),
                "convenience": round(float(loc.convenience[i]), 4),
                "anomie": round(float(loc.anomie[i]), 4),
                "prestige": round(float(loc.prestige[i]), 4),
                "population": int(loc.population[i]),
                "digital_access": round(float(loc.digital_access[i]), 4),
                "healthcare_score": round(float(loc.healthcare_score[i]), 4),
            })
        self._last_year = yr

    def export(self, path: str):
        df = pd.DataFrame(self.snapshots)
        df.to_csv(path, index=False)
        print(f"  Warehouse correlation time series: {path} ({len(df)} rows)")
        return df

    def plot(self, path: str):
        df = pd.DataFrame(self.snapshots)
        if df.empty:
            return

        years = sorted(df["year"].unique())
        n_years = len(years)
        cols = min(3, n_years)
        rows = (n_years + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        tier_colors = {0: "#e53935", 1: "#1e88e5", 2: "#43a047"}
        tier_names = {0: "Tokyo", 1: "Core", 2: "Periphery"}

        for idx, yr in enumerate(years):
            ax = axes[idx // cols][idx % cols]
            sub = df[df["year"] == yr]
            for tier in [0, 1, 2]:
                ts = sub[sub["tier"] == tier]
                if len(ts) == 0:
                    continue
                ax.scatter(ts["convenience"], ts["anomie"],
                           c=tier_colors[tier], s=ts["population"] * 0.01 + 2,
                           alpha=0.5, label=tier_names[tier], edgecolors="none")
            ax.set_xlabel("Convenience (C)")
            ax.set_ylabel("Anomie (A)")
            ax.set_title(f"Year {yr}", fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, markerscale=0.5)
            ax.grid(alpha=0.15)

            # Correlation line for periphery
            peri = sub[sub["tier"] == 2]
            if len(peri) > 10:
                z = np.polyfit(peri["convenience"], peri["anomie"], 1)
                xs = np.linspace(0, 1, 50)
                ax.plot(xs, np.polyval(z, xs), "--", color="#43a047", alpha=0.7, lw=1.5)
                r = np.corrcoef(peri["convenience"], peri["anomie"])[0, 1]
                ax.text(0.02, 0.95, f"r(Peri)={r:.2f}", transform=ax.transAxes,
                        fontsize=9, va="top", color="#43a047")

        # Hide unused
        for idx in range(len(years), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle("Human Warehouse Paradox: Convenience vs Anomie", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  C-A scatter plot: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def run_monte_carlo(
    n_runs: int = 50,
    years: int = 40,
    scale: int = 200,
    base_seed: int = 42,
    scenario: str = "baseline",
    output_dir: str = "output/montecarlo",
):
    """Run N simulations and compute mean/std/CI for key metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from run import SCENARIOS
    builder = SCENARIOS.get(scenario)
    if builder is None:
        print(f"Unknown scenario '{scenario}'. Available: {list(SCENARIOS.keys())}")
        sys.exit(1)

    all_runs = []
    annual_series = defaultdict(list)

    print(f"\n{'='*70}")
    print(f"  MONTE CARLO: {n_runs} runs x {years} years")
    print(f"  Scenario: {scenario}")
    print(f"  Scale: 1:{scale}")
    print(f"{'='*70}\n")

    for run_i in range(n_runs):
        seed = base_seed + run_i * 7919
        cfg = builder(scale, years, seed)
        t0 = time.time()
        model = ExodusModel(cfg)
        results = model.run()
        elapsed = time.time() - t0

        h = results["history"]
        stats = results["final_agent_stats"]

        n_alive = stats.get("n_alive", 0)
        total_pop = n_alive * scale

        # Periphery extinction count: municipalities with pop < 50 agents
        peri_mask = model.loc_state.tier == 2
        peri_pops = model.loc_state.population[peri_mask]
        n_extinct = int((peri_pops < 10).sum())
        n_periphery = int(peri_mask.sum())

        annual = h.groupby("year")
        final_year = h[h["year"] == h["year"].max()]
        tokyo_share = float(final_year["tokyo_pop_share"].iloc[-1]) if len(final_year) else 0
        peri_share = float(final_year["peri_pop_share"].iloc[-1]) if len(final_year) else 0
        tfr = float(final_year["tfr_proxy"].iloc[-1]) if len(final_year) else 0

        run_data = {
            "run": run_i,
            "seed": seed,
            "final_population": total_pop,
            "tokyo_share": tokyo_share,
            "periphery_share": peri_share,
            "tfr": tfr,
            "mean_age": stats.get("mean_age", 0),
            "periphery_extinct": n_extinct,
            "periphery_total": n_periphery,
            "pct_extinct": n_extinct / max(n_periphery, 1),
            "runtime_s": round(elapsed, 1),
        }
        all_runs.append(run_data)

        # Collect annual time series for averaging
        for _, row in h.groupby("year").last().iterrows():
            yr = int(row.get("year", 0))
            annual_series[yr].append({
                "total_population": int(row["total_population"]) * scale,
                "tokyo_pop_share": float(row["tokyo_pop_share"]),
                "peri_pop_share": float(row["peri_pop_share"]),
                "tfr_proxy": float(row.get("tfr_proxy", 0)),
            })

        print(f"  [{run_i+1}/{n_runs}] pop={total_pop:,.0f} tokyo={tokyo_share:.3f} "
              f"tfr={tfr:.3f} extinct={n_extinct}/{n_periphery} ({elapsed:.1f}s)")

    # ── Aggregate ──
    df = pd.DataFrame(all_runs)
    df.to_csv(str(out / "montecarlo_runs.csv"), index=False)

    summary = {
        "n_runs": n_runs,
        "scenario": scenario,
        "years": years,
        "pop_mean": df["final_population"].mean(),
        "pop_std": df["final_population"].std(),
        "pop_ci95_lo": df["final_population"].mean() - 1.96 * df["final_population"].std() / np.sqrt(n_runs),
        "pop_ci95_hi": df["final_population"].mean() + 1.96 * df["final_population"].std() / np.sqrt(n_runs),
        "tokyo_mean": df["tokyo_share"].mean(),
        "tokyo_std": df["tokyo_share"].std(),
        "peri_mean": df["periphery_share"].mean(),
        "peri_std": df["periphery_share"].std(),
        "tfr_mean": df["tfr"].mean(),
        "tfr_std": df["tfr"].std(),
        "extinct_mean": df["periphery_extinct"].mean(),
        "extinct_std": df["periphery_extinct"].std(),
        "pct_extinct_mean": df["pct_extinct"].mean(),
    }

    # Annual mean/std time series
    ts_rows = []
    for yr in sorted(annual_series.keys()):
        vals = annual_series[yr]
        for key in ["total_population", "tokyo_pop_share", "peri_pop_share", "tfr_proxy"]:
            arr = np.array([v[key] for v in vals])
            ts_rows.append({
                "year": yr, "metric": key,
                "mean": arr.mean(), "std": arr.std(),
                "ci95_lo": arr.mean() - 1.96 * arr.std() / np.sqrt(len(arr)),
                "ci95_hi": arr.mean() + 1.96 * arr.std() / np.sqrt(len(arr)),
                "min": arr.min(), "max": arr.max(),
            })

    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_csv(str(out / "montecarlo_timeseries.csv"), index=False)

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO RESULTS ({n_runs} runs)")
    print(f"{'='*70}")
    print(f"  Population:     {summary['pop_mean']:>12,.0f} +/- {summary['pop_std']:>10,.0f}")
    print(f"    95% CI:       [{summary['pop_ci95_lo']:>12,.0f}, {summary['pop_ci95_hi']:>12,.0f}]")
    print(f"  Tokyo Share:    {summary['tokyo_mean']:>12.4f} +/- {summary['tokyo_std']:>10.4f}")
    print(f"  Periphery:      {summary['peri_mean']:>12.4f} +/- {summary['peri_std']:>10.4f}")
    print(f"  TFR:            {summary['tfr_mean']:>12.3f} +/- {summary['tfr_std']:>10.3f}")
    print(f"  Extinct munis:  {summary['extinct_mean']:>12.1f} +/- {summary['extinct_std']:>10.1f}")
    print(f"    ({summary['pct_extinct_mean']:.1%} of periphery municipalities)")
    print(f"\n  Saved: {out}/montecarlo_runs.csv")
    print(f"         {out}/montecarlo_timeseries.csv")

    # ── Plots ──
    _plot_montecarlo(ts_df, summary, out)

    with open(str(out / "montecarlo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return df, ts_df


def _plot_montecarlo(ts_df, summary, out_dir):
    """Generate Monte Carlo fan charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("total_population", "Total Population", "People"),
        ("tokyo_pop_share", "Tokyo Share", "Share"),
        ("peri_pop_share", "Periphery Share", "Share"),
        ("tfr_proxy", "TFR Proxy", "TFR"),
    ]

    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        sub = ts_df[ts_df["metric"] == metric].sort_values("year")
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["mean"], "b-", lw=2, label="Mean")
        ax.fill_between(sub["year"], sub["ci95_lo"], sub["ci95_hi"],
                        alpha=0.25, color="blue", label="95% CI")
        ax.fill_between(sub["year"], sub["min"], sub["max"],
                        alpha=0.08, color="blue", label="Min-Max")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    fig.suptitle(f"Monte Carlo Results ({summary['n_runs']} runs)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_dir / "montecarlo_fan_charts.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fan charts: {out_dir / 'montecarlo_fan_charts.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. HEADLESS HEATMAP CAPTURE
# ═══════════════════════════════════════════════════════════════════════════
class HeadlessHeatmapCapture:
    """Auto-exports net migration heatmaps at fixed intervals."""

    LON_MIN, LON_MAX = 128.5, 146.5
    LAT_MIN, LAT_MAX = 30.0, 46.0

    def __init__(self, model: ExodusModel, output_dir: str = "output/heatmaps",
                 interval_quarters: int = 20, vmin: float = -0.10, vmax: float = 0.10):
        self.model = model
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.interval = interval_quarters
        self.vmin = vmin
        self.vmax = vmax
        self._prev_pops = model.loc_state.population.copy().astype(np.float64)
        self._frame_count = 0
        self._geojson = self._load_geojson()

    def _load_geojson(self):
        gj_path = Path("data/japan.geojson")
        if not gj_path.exists():
            return None
        with open(gj_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_and_capture(self):
        if self.model.total_steps % self.interval != 0:
            return
        if self.model.total_steps == 0:
            return

        self._capture_frame()

    def _capture_frame(self):
        loc = self.model.loc_state
        current_pops = loc.population.astype(np.float64)
        net_change = (current_pops - self._prev_pops) / np.maximum(self._prev_pops, 1.0)
        self._prev_pops = current_pops.copy()

        yr = self.model.current_year
        q = self.model.current_step_in_year + 1

        fig, ax = plt.subplots(1, 1, figsize=(10, 14))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        # Draw prefecture outlines
        if self._geojson:
            for feature in self._geojson.get("features", []):
                geom = feature.get("geometry", {})
                coords_list = []
                if geom["type"] == "Polygon":
                    coords_list = [geom["coordinates"][0]]
                elif geom["type"] == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        coords_list.append(poly[0])
                for ring in coords_list:
                    lons = [c[0] for c in ring]
                    lats = [c[1] for c in ring]
                    if min(lons) < self.LON_MIN - 1 or max(lons) > self.LON_MAX + 1:
                        continue
                    ax.fill(lons, lats, facecolor="#1a1e2e", edgecolor="#2a3050", lw=0.3)

        # Scatter municipalities by net migration
        cmap = plt.cm.RdYlGn
        norm = mcolors.TwoSlopeNorm(vmin=self.vmin, vcenter=0, vmax=self.vmax)
        sizes = np.clip(current_pops * 0.005, 2, 80)

        sc = ax.scatter(
            loc.lon, loc.lat,
            c=net_change, cmap=cmap, norm=norm,
            s=sizes, alpha=0.85, edgecolors="none", zorder=5,
        )

        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Net Population Change (%)", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xlim(self.LON_MIN, self.LON_MAX)
        ax.set_ylim(self.LAT_MIN, self.LAT_MAX)
        ax.set_aspect(1.2)
        ax.set_title(f"Net Migration Suction -- {yr} Q{q}",
                      color="white", fontsize=14, fontweight="bold", pad=12)
        ax.tick_params(colors="white")
        ax.set_xlabel("Longitude", color="white")
        ax.set_ylabel("Latitude", color="white")

        # Stats annotation
        total = int(current_pops.sum() * self.model.config.scale.agent_scale)
        gaining = int((net_change > 0.001).sum())
        losing = int((net_change < -0.001).sum())
        ax.text(0.02, 0.02,
                f"Pop: {total:,} | Gaining: {gaining} | Losing: {losing} municipalities",
                transform=ax.transAxes, color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1e2e", edgecolor="#2a3050"))

        fname = f"heatmap_{yr}_Q{q}_{self._frame_count:04d}.png"
        fig.savefig(str(self.out / fname), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        self._frame_count += 1
        print(f"    [Heatmap] {fname}")

    def export_all_frames_summary(self):
        print(f"  Total heatmaps exported: {self._frame_count} -> {self.out}/")


def run_with_heatmaps(
    years: int = 40,
    scale: int = 200,
    seed: int = 42,
    scenario: str = "baseline",
    output_dir: str = "output/heatmaps",
):
    """Run simulation headless with automatic heatmap capture every 5 years."""
    from run import SCENARIOS
    builder = SCENARIOS.get(scenario)
    if not builder:
        print(f"Unknown scenario: {scenario}")
        sys.exit(1)

    cfg = builder(scale, years, seed)
    model = ExodusModel(cfg)

    heatmap = HeadlessHeatmapCapture(model, output_dir, interval_quarters=20)
    warehouse = WarehouseCorrelationCollector(model, interval_years=5)

    total_steps = years * cfg.scale.steps_per_year
    print(f"\n  Running {scenario} for {years} years with heatmap capture...")

    for step in range(total_steps):
        model.step()
        heatmap.check_and_capture()
        warehouse.check_and_collect()

        if step % 20 == 0:
            yr = model.current_year
            pop = model.agents_pool.n_alive * scale
            print(f"    Step {step}/{total_steps} -- {yr} -- Pop: {pop:,}")

    results = model.get_results()
    results["history"].to_csv(f"{output_dir}/history.csv", index=False)
    heatmap.export_all_frames_summary()

    warehouse.export(f"{output_dir}/warehouse_correlation.csv")
    warehouse.plot(f"{output_dir}/warehouse_scatter.png")

    export_cannibalism_matrix(results, f"{output_dir}/cannibalism_matrix.csv", scenario)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Research Instrument Suite for Japan Exodus Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Branch
    br = sub.add_parser("branch", help="Run counterfactual scenario branching")
    br.add_argument("--branch-year", type=int, default=2030)
    br.add_argument("--end-year", type=int, default=2065)
    br.add_argument("--scale", type=int, default=200)
    br.add_argument("--seed", type=int, default=42)
    br.add_argument("--output", type=str, default="output/branch")

    # Monte Carlo
    mc = sub.add_parser("montecarlo", help="Run Monte Carlo trials")
    mc.add_argument("--runs", type=int, default=50)
    mc.add_argument("--years", type=int, default=40)
    mc.add_argument("--scale", type=int, default=200)
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--scenario", type=str, default="baseline")
    mc.add_argument("--output", type=str, default="output/montecarlo")

    # Heatmaps
    hm = sub.add_parser("heatmaps", help="Run with automatic heatmap capture")
    hm.add_argument("--years", type=int, default=40)
    hm.add_argument("--scale", type=int, default=200)
    hm.add_argument("--seed", type=int, default=42)
    hm.add_argument("--scenario", type=str, default="baseline")
    hm.add_argument("--output", type=str, default="output/heatmaps")

    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU (use CPU only)")

    args = parser.parse_args()

    if getattr(args, "no_gpu", False):
        disable_gpu()

    from model.config import GPU_AVAILABLE as GPU_STATUS
    print(f"Japan Exodus Research Tools v1.0")
    print(f"GPU: {'CUDA available' if GPU_STATUS else 'CPU only'}")

    if args.command == "branch":
        run_scenario_branch(args.branch_year, args.end_year, args.scale, args.seed, args.output)
    elif args.command == "montecarlo":
        run_monte_carlo(args.runs, args.years, args.scale, args.seed, args.scenario, args.output)
    elif args.command == "heatmaps":
        run_with_heatmaps(args.years, args.scale, args.seed, args.scenario, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
