"""
VHA Evidence Kit — Calibrated Takaichi 2.0 (Dec 2025 Strategy)
==============================================================
Identical structure to vha_evidence_kit.py but uses the
policy-accurate scenario_takaichi_calibrated config.

Produces all five deliverables plus a Policy Validation Report
that confirms every parameter matches the real Dec 2025 document.

Output directory: output/vha_kit_calibrated/
  (does NOT overwrite output/vha_kit/)

Usage:
    python vha_evidence_kit_calibrated.py --no-gpu
    python vha_evidence_kit_calibrated.py --no-gpu --mc-runs 30
"""

import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from model.config import SimulationConfig, ScaleConfig, disable_gpu
from model.model import ExodusModel
from scenario_takaichi_calibrated import (
    make_takaichi_calibrated_config,
    validate_config,
    VALIDATION_TARGETS,
    STIMULUS_TOTAL_JPY,
)
from run import SCENARIOS, make_baseline_config

TIER_NAMES = {0: "Tokyo", 1: "Core", 2: "Periphery"}
OUT = Path("output/vha_kit_calibrated")


# ======================================================================
# Agent Lifetime Tracker (copied from vha_evidence_kit — no imports
# to avoid coupling to the old file)
# ======================================================================
class AgentLifetimeTracker:
    """Track the full migration biography of N sampled agents."""

    def __init__(self, model: ExodusModel, n_track: int = 5000, scale: int = 300):
        self.model = model
        self.scale = scale
        pool = model.agents_pool
        loc = model.loc_state
        n = pool.next_id

        alive_idx = np.where(pool.alive[:n])[0]
        rng = np.random.default_rng(99)

        sampled = []
        tiers = loc.tier[pool.location[:n]]
        for tier_id in [0, 1, 2]:
            tier_alive = alive_idx[tiers[alive_idx] == tier_id]
            young = tier_alive[(pool.age[tier_alive] >= 20) & (pool.age[tier_alive] <= 34)]
            mid = tier_alive[(pool.age[tier_alive] >= 35) & (pool.age[tier_alive] <= 50)]
            n_per = n_track // 6
            if len(young) > 0:
                sampled.extend(rng.choice(young, size=min(n_per, len(young)), replace=False))
            if len(mid) > 0:
                sampled.extend(rng.choice(mid, size=min(n_per, len(mid)), replace=False))

        remaining = n_track - len(sampled)
        if remaining > 0:
            others = np.setdiff1d(alive_idx, sampled)
            sampled.extend(rng.choice(others, size=min(remaining, len(others)), replace=False))

        self.tracked_ids = np.array(sampled[:n_track], dtype=np.int64)
        self.n_track = len(self.tracked_ids)

        self.records = []
        for idx in self.tracked_ids:
            loc_id = int(pool.location[idx])
            self.records.append({
                "agent_id": int(idx), "year": model.current_year,
                "quarter": model.current_step_in_year + 1,
                "step": model.total_steps, "event": "INITIAL",
                "location_id": loc_id,
                "tier": TIER_NAMES.get(int(loc.tier[loc_id]), "?"),
                "origin_tier": TIER_NAMES.get(int(pool.origin_tier[idx]), "?"),
                "age": int(pool.age[idx]),
                "sex": "M" if pool.sex[idx] == 0 else "F",
                "income": round(float(pool.income[idx])),
                "education": int(pool.education[idx]),
                "marital": int(pool.marital_status[idx]),
                "n_children": int(pool.n_children[idx]),
                "remote_worker": bool(pool.remote_worker[idx]),
                "utility_memory": round(float(pool.utility_memory[idx]), 4),
            })
        self._prev_locs = pool.location[self.tracked_ids].copy()

    def record_step(self):
        pool = self.model.agents_pool
        loc = self.model.loc_state
        yr = self.model.current_year
        q = self.model.current_step_in_year + 1
        step = self.model.total_steps

        cur_locs = pool.location[self.tracked_ids]
        alive = pool.alive[self.tracked_ids]
        moved = (cur_locs != self._prev_locs) & alive

        for i in np.where(moved)[0]:
            idx = int(self.tracked_ids[i])
            old_loc = int(self._prev_locs[i])
            new_loc = int(cur_locs[i])
            self.records.append({
                "agent_id": idx, "year": yr, "quarter": q, "step": step,
                "event": "MIGRATION",
                "location_id": new_loc, "from_location_id": old_loc,
                "tier": TIER_NAMES.get(int(loc.tier[new_loc]), "?"),
                "from_tier": TIER_NAMES.get(int(loc.tier[old_loc]), "?"),
                "origin_tier": TIER_NAMES.get(int(pool.origin_tier[idx]), "?"),
                "age": int(pool.age[idx]),
                "sex": "M" if pool.sex[idx] == 0 else "F",
                "income": round(float(pool.income[idx])),
                "education": int(pool.education[idx]),
                "marital": int(pool.marital_status[idx]),
                "n_children": int(pool.n_children[idx]),
                "remote_worker": bool(pool.remote_worker[idx]),
                "utility_memory": round(float(pool.utility_memory[idx]), 4),
            })

        died = (~alive) & (self._prev_locs >= 0)
        for i in np.where(died)[0]:
            idx = int(self.tracked_ids[i])
            self.records.append({
                "agent_id": idx, "year": yr, "quarter": q, "step": step,
                "event": "DEATH",
                "location_id": int(self._prev_locs[i]),
                "tier": TIER_NAMES.get(int(loc.tier[int(self._prev_locs[i])]), "?"),
                "origin_tier": TIER_NAMES.get(int(pool.origin_tier[idx]), "?"),
                "age": int(pool.age[idx]),
                "sex": "M" if pool.sex[idx] == 0 else "F",
            })
            self._prev_locs[i] = -1

        self._prev_locs = cur_locs.copy()
        self._prev_locs[~alive] = -1

    def export(self, path: str):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        print(f"  [D2] Longitudinal log: {path} ({len(df):,} events, {self.n_track:,} agents)")
        return df

    def compute_launchpad_stats(self):
        df = pd.DataFrame(self.records)
        migrations = df[df["event"] == "MIGRATION"].copy()
        if migrations.empty:
            return {}
        agent_paths = {}
        for agent_id, grp in migrations.groupby("agent_id"):
            agent_paths[agent_id] = grp["tier"].tolist()

        launchpad = buffer = direct = 0
        for agent_id, path in agent_paths.items():
            origin = df[df["agent_id"] == agent_id].iloc[0].get("origin_tier", "?")
            if origin != "Periphery":
                continue
            full = [origin] + path
            visited_core = "Core" in full
            ended_tokyo = path[-1] == "Tokyo" if path else False
            if visited_core and ended_tokyo:
                launchpad += 1
            elif visited_core and not ended_tokyo:
                buffer += 1
            elif ended_tokyo and not visited_core:
                direct += 1

        total = launchpad + buffer + direct
        return {
            "total_periphery_origin_movers": total,
            "launchpad_agents": launchpad,
            "buffer_agents": buffer,
            "direct_to_tokyo": direct,
            "launchpad_pct": launchpad / max(total, 1),
            "buffer_pct": buffer / max(total, 1),
            "direct_pct": direct / max(total, 1),
        }


# ======================================================================
# Nomad Collector
# ======================================================================
class NomadCollector:
    def __init__(self, model, scale):
        self.model = model
        self.scale = scale
        self.snapshots = []
        self._last_year = None

    def collect_if_annual(self):
        yr = self.model.current_year
        q = self.model.current_step_in_year
        if yr == self._last_year:
            return
        if q != 0 and self.model.total_steps > 0:
            return
        self._last_year = yr
        self._collect(yr)

    def _collect(self, yr):
        pool = self.model.agents_pool
        loc = self.model.loc_state
        n = pool.next_id
        alive = pool.alive[:n]
        tiers = loc.tier[pool.location[:n]]
        remote = pool.remote_worker[:n]

        for label, mask in [
            ("periphery_nomad", alive & remote & (tiers == 2)),
            ("core_nomad", alive & remote & (tiers == 1)),
        ]:
            idx = np.where(mask)[0]
            if len(idx) == 0:
                self.snapshots.append({"year": yr, "category": label, "count": 0})
                continue
            ages = pool.age[idx]
            incomes = pool.income[idx]
            children = pool.n_children[idx]
            self.snapshots.append({
                "year": yr, "category": label,
                "count": int(len(idx)) * self.scale,
                "mean_age": round(float(ages.mean()), 1),
                "median_age": round(float(np.median(ages)), 1),
                "pct_female": round(float((pool.sex[idx] == 1).mean()), 4),
                "mean_income": round(float(incomes.mean())),
                "median_income": round(float(np.median(incomes))),
                "p25_income": round(float(np.percentile(incomes, 25))),
                "p75_income": round(float(np.percentile(incomes, 75))),
                "mean_children": round(float(children.mean()), 2),
                "pct_married": round(float((pool.marital_status[idx] == 1).mean()), 4),
                "pct_higher_education": round(float((pool.education[idx] >= 2).mean()), 4),
            })

    def export(self, path):
        df = pd.DataFrame(self.snapshots)
        df.to_csv(path, index=False)
        print(f"  [D3] Nomad demographics: {path} ({len(df)} rows)")
        return df


# ======================================================================
# Bivariate Correlation Collector
# ======================================================================
class BivariateCorrelationCollector:
    def __init__(self, model, scale):
        self.model = model
        self.scale = scale
        self.snapshots = []
        self._last_year = None

    def collect_if_annual(self):
        yr = self.model.current_year
        q = self.model.current_step_in_year
        if yr == self._last_year:
            return
        if q != 0 and self.model.total_steps > 0:
            return
        self._last_year = yr
        self._collect(yr)

    def _collect(self, yr):
        pool = self.model.agents_pool
        loc = self.model.loc_state
        n = pool.next_id
        alive = pool.alive[:n]
        n_locs = len(loc.tier)
        locs = pool.location[:n][alive]
        sexes = pool.sex[:n][alive]
        ages = pool.age[:n][alive]
        children = pool.n_children[:n][alive]

        fertile_mask = (sexes == 1) & (ages >= 15) & (ages <= 49)
        fertile_locs = locs[fertile_mask]
        fertile_per_loc = np.bincount(fertile_locs, minlength=n_locs)
        children_per_loc = np.zeros(n_locs)
        np.add.at(children_per_loc, fertile_locs, children[fertile_mask])
        local_birth_rate = np.divide(
            children_per_loc, fertile_per_loc,
            out=np.zeros(n_locs), where=fertile_per_loc > 0,
        )
        pop_per_loc = np.bincount(locs, minlength=n_locs)
        names = getattr(loc, '_municipality_names', None)
        prefs = getattr(loc, '_prefectures', None)

        for i in range(n_locs):
            self.snapshots.append({
                "year": yr, "location_id": i,
                "name": names[i] if names else f"Loc_{i}",
                "prefecture": str(prefs[i]) if prefs is not None else "N/A",
                "tier": TIER_NAMES.get(int(loc.tier[i]), "?"),
                "population": int(pop_per_loc[i]) * self.scale,
                "convenience": round(float(loc.convenience[i]), 4),
                "anomie": round(float(loc.anomie[i]), 4),
                "prestige": round(float(loc.prestige[i]), 4),
                "financial_friction": round(float(loc.financial_friction[i]), 4),
                "fertile_women": int(fertile_per_loc[i]) * self.scale,
                "local_birth_rate_proxy": round(float(local_birth_rate[i]), 4),
                "digital_access": round(float(loc.digital_access[i]), 4),
                "healthcare": round(float(loc.healthcare_score[i]), 4),
                "vacancy_rate": round(float(loc.vacancy_rate[i]), 4),
            })

    def export(self, path):
        df = pd.DataFrame(self.snapshots)
        df.to_csv(path, index=False)
        print(f"  [D4] Bivariate data: {path} ({len(df):,} rows)")
        return df

    def compute_correlations(self, path):
        df = pd.DataFrame(self.snapshots)
        if df.empty:
            return pd.DataFrame()
        last_yr = df["year"].max()
        final = df[(df["year"] == last_yr) & (df["population"] > 0)]

        pairs = [
            ("convenience", "anomie"),
            ("convenience", "local_birth_rate_proxy"),
            ("anomie", "local_birth_rate_proxy"),
            ("digital_access", "anomie"),
            ("digital_access", "local_birth_rate_proxy"),
            ("prestige", "local_birth_rate_proxy"),
            ("financial_friction", "local_birth_rate_proxy"),
            ("healthcare", "local_birth_rate_proxy"),
        ]
        rows = []
        for tier_label in ["All", "Tokyo", "Core", "Periphery"]:
            sub = final if tier_label == "All" else final[final["tier"] == tier_label]
            if len(sub) < 5:
                continue
            for va, vb in pairs:
                a = sub[va].values
                b = sub[vb].values
                valid = np.isfinite(a) & np.isfinite(b)
                a, b = a[valid], b[valid]
                if len(a) < 5:
                    continue
                try:
                    rp, pp = sp_stats.pearsonr(a, b)
                    rs, ps = sp_stats.spearmanr(a, b)
                except Exception:
                    continue
                rows.append({
                    "tier": tier_label, "variable_A": va, "variable_B": vb,
                    "n": len(a),
                    "pearson_r": round(rp, 4), "pearson_p": round(pp, 6),
                    "sig_pearson": "***" if pp < .001 else "**" if pp < .01 else "*" if pp < .05 else "ns",
                    "spearman_rho": round(rs, 4), "spearman_p": round(ps, 6),
                    "sig_spearman": "***" if ps < .001 else "**" if ps < .01 else "*" if ps < .05 else "ns",
                })
        corr_df = pd.DataFrame(rows)
        corr_df.to_csv(path, index=False)
        print(f"  [D4] Correlation table: {path}")
        return corr_df


# ======================================================================
# Immigration Validation Metric
# ======================================================================
def compute_immigration_validation(model, scale, years):
    """Check if the non-metro immigration target of 10,000/yr was met."""
    pool = model.agents_pool
    loc = model.loc_state
    n = pool.next_id
    alive = pool.alive[:n]

    # Count agents born after start who are in periphery and young (proxy for immigrants)
    start_yr = model.config.scale.start_year
    immigrant_proxy = alive & (pool.birth_year[:n] < start_yr - 10) & (pool.origin_tier[:n] == 2)
    n_peri_immigrants = int(immigrant_proxy.sum()) * scale

    annual_rate = n_peri_immigrants / max(years, 1)
    return {
        "periphery_immigrant_stock": n_peri_immigrants,
        "implied_annual_rate": round(annual_rate),
        "target": 10_000,
        "pct_of_target": round(annual_rate / 10_000 * 100, 1),
    }


# ======================================================================
# D1: Calibrated Takaichi Run
# ======================================================================
def run_takaichi_calibrated(scale, years, seed, out_dir):
    print(f"\n{'='*70}")
    print(f"  D1: Takaichi 2.0 CALIBRATED (Dec 2025 Strategy)")
    print(f"  ICE Act: 7% | Stimulus: JPY 17.7T | Wage +15% | Migrant 10k")
    print(f"{'='*70}\n")

    from export_spreadsheet import FullDataCollector, build_workbook

    cfg = make_takaichi_calibrated_config(scale, years, seed)

    # Validate before running
    checks = validate_config(cfg)
    print("  Policy Validation:")
    all_pass = True
    for label, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {label}")
    print(f"  All checks: {'PASS' if all_pass else 'FAIL'}\n")

    model = ExodusModel(cfg)

    collector = FullDataCollector(model, scale)
    collector.collect_if_annual()

    tracker = AgentLifetimeTracker(model, n_track=5000, scale=scale)
    nomad = NomadCollector(model, scale)
    corr = BivariateCorrelationCollector(model, scale)

    total_steps = years * cfg.scale.steps_per_year
    t0 = time.time()

    for step in range(total_steps):
        model.step()
        tracker.record_step()
        collector.collect_if_annual()
        nomad.collect_if_annual()
        corr.collect_if_annual()

        if step % cfg.scale.steps_per_year == 0:
            yr = model.current_year
            pop = model.agents_pool.n_alive * scale
            print(f"    {yr} -- Pop: {pop:,}")

    collector.collect_if_annual()
    nomad.collect_if_annual()
    corr.collect_if_annual()

    runtime = time.time() - t0
    results = model.get_results()

    wb = build_workbook(
        results["history"], results["migration_flows"],
        collector, model, scale, "takaichi_2.0_calibrated", runtime,
    )
    xlsx_path = str(out_dir / f"D1_takaichi_calibrated_{years}yr.xlsx")
    wb.save(xlsx_path)
    print(f"  [D1] Spreadsheet: {xlsx_path}")

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    results["history"].to_csv(str(csv_dir / "takaichi_cal_history.csv"), index=False)
    if not results["migration_flows"].empty:
        results["migration_flows"].to_csv(str(csv_dir / "takaichi_cal_migrations.csv"), index=False)

    return model, results, tracker, nomad, corr, runtime


# ======================================================================
# D5: Monte Carlo (uses calibrated config)
# ======================================================================
def run_monte_carlo_calibrated(n_runs, years, scale, base_seed, out_dir):
    print(f"\n{'='*70}")
    print(f"  D5: Monte Carlo ({n_runs} runs x {years} years, CALIBRATED)")
    print(f"{'='*70}\n")

    all_runs = []
    annual_series = defaultdict(list)

    for run_i in range(n_runs):
        seed = base_seed + run_i * 7919
        cfg = make_takaichi_calibrated_config(scale, years, seed)
        t0 = time.time()
        model = ExodusModel(cfg)
        results = model.run(progress=False)
        elapsed = time.time() - t0

        h = results["history"]
        stats = results["final_agent_stats"]
        n_alive = stats.get("n_alive", 0)
        total_pop = n_alive * scale

        peri_mask = model.loc_state.tier == 2
        peri_pops = model.loc_state.population[peri_mask]
        n_extinct = int((peri_pops < 10).sum())
        n_periphery = int(peri_mask.sum())

        final = h[h["year"] == h["year"].max()]
        tokyo_share = float(final["tokyo_pop_share"].iloc[-1]) if len(final) else 0
        peri_share = float(final["peri_pop_share"].iloc[-1]) if len(final) else 0
        tfr = float(final["tfr_proxy"].iloc[-1]) if len(final) else 0
        mean_age = float(final["mean_age"].iloc[-1]) if len(final) else 0
        launchpad = float(final["launchpad_ratio"].iloc[-1]) if "launchpad_ratio" in final.columns else 0
        cannibalism = float(final["cannibalism_ratio"].iloc[-1]) if "cannibalism_ratio" in final.columns else 0
        warehouse = int(final["n_human_warehouse_towns"].iloc[-1]) if "n_human_warehouse_towns" in final.columns else 0

        total_births = int(h["births"].sum()) * scale
        total_deaths = int(h["deaths"].sum()) * scale

        run_data = {
            "run": run_i, "seed": seed,
            "final_population": total_pop,
            "tokyo_share": tokyo_share, "periphery_share": peri_share,
            "tfr": tfr, "mean_age": mean_age,
            "periphery_extinct": n_extinct,
            "pct_extinct": n_extinct / max(n_periphery, 1),
            "launchpad_ratio": launchpad,
            "cannibalism_ratio": cannibalism,
            "warehouse_towns": warehouse,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "runtime_s": round(elapsed, 1),
        }
        all_runs.append(run_data)

        for _, row in h.groupby("year").last().iterrows():
            yr = int(row.get("year", 0))
            annual_series[yr].append({
                "total_population": int(row["total_population"]) * scale,
                "tokyo_pop_share": float(row["tokyo_pop_share"]),
                "peri_pop_share": float(row["peri_pop_share"]),
                "tfr_proxy": float(row.get("tfr_proxy", 0)),
            })

        print(f"  [{run_i+1:>3}/{n_runs}] pop={total_pop:>12,} tokyo={tokyo_share:.3f} "
              f"tfr={tfr:.3f} extinct={n_extinct}/{n_periphery} ({elapsed:.1f}s)")

    df = pd.DataFrame(all_runs)
    df.to_csv(str(out_dir / "D5_montecarlo_runs_cal.csv"), index=False)

    metrics = ["final_population", "tokyo_share", "periphery_share", "tfr", "mean_age",
               "periphery_extinct", "pct_extinct", "launchpad_ratio", "cannibalism_ratio",
               "warehouse_towns", "total_births", "total_deaths"]
    summary_rows = []
    for m in metrics:
        vals = df[m].values
        n = len(vals)
        summary_rows.append({
            "metric": m, "n_runs": n,
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "ci95_lo": round(float(vals.mean() - 1.96 * vals.std() / np.sqrt(n)), 4),
            "ci95_hi": round(float(vals.mean() + 1.96 * vals.std() / np.sqrt(n)), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(str(out_dir / "D5_montecarlo_summary_cal.csv"), index=False)

    ts_rows = []
    for yr in sorted(annual_series.keys()):
        v = annual_series[yr]
        for key in ["total_population", "tokyo_pop_share", "peri_pop_share", "tfr_proxy"]:
            arr = np.array([x[key] for x in v])
            ts_rows.append({
                "year": yr, "metric": key,
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "ci95_lo": round(float(arr.mean() - 1.96 * arr.std() / np.sqrt(len(arr))), 4),
                "ci95_hi": round(float(arr.mean() + 1.96 * arr.std() / np.sqrt(len(arr))), 4),
            })
    pd.DataFrame(ts_rows).to_csv(str(out_dir / "D5_montecarlo_timeseries_cal.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"  MONTE CARLO RESULTS ({n_runs} runs, Calibrated Takaichi 2.0)")
    print(f"{'='*70}")
    for _, row in summary_df.iterrows():
        m = row["metric"]
        if "population" in m or "births" in m or "deaths" in m:
            print(f"  {m:<25s}: {row['mean']:>14,.0f} +/- {row['std']:>10,.0f}")
        else:
            print(f"  {m:<25s}: {row['mean']:>14.4f} +/- {row['std']:>10.4f}")

    return df, summary_df


# ======================================================================
# Policy Validation Report
# ======================================================================
def write_validation_report(model, scale, years, out_dir, runtime):
    """Write a text report confirming every parameter matches the real doc."""
    cfg = model.config
    p = cfg.policy
    loc = model.loc_state
    checks = validate_config(cfg)

    # Final-year digital access and healthcare in periphery
    peri_mask = loc.tier == 2
    peri_digital = float(loc.digital_access[peri_mask].mean())
    peri_health = float(loc.healthcare_score[peri_mask].mean())

    report_lines = [
        "=" * 70,
        "  POLICY VALIDATION REPORT",
        "  Takaichi 2.0 Calibrated -- Dec 23, 2025 Comprehensive Strategy",
        "=" * 70,
        "",
        "PARAMETER VALIDATION:",
        *[f"  [{'PASS' if v else 'FAIL'}] {k}" for k, v in checks.items()],
        "",
        "FISCAL ENVELOPE:",
        f"  Stimulus total:        JPY {STIMULUS_TOTAL_JPY/1e12:.1f} trillion",
        f"  Digital infra rate:    {p.level4_pod_deployment_rate:.2f}/yr (periphery L4 pods)",
        f"  Medical DX rate:       {p.medical_dx_rollout_rate:.2f}/yr (telemedicine + AI diag)",
        f"  Childcare expansion:   {p.childcare_expansion_rate:.2f}/yr",
        "",
        "CALIBRATED PARAMETERS:",
        f"  ICE Act tax credit:    {p.ice_act_tax_credit*100:.0f}% (target: 7%)",
        f"  Regional wage mult:    {p.regional_wage_multiplier:.2f} (target: 1.15 = +15%)",
        f"  Remote work:           {p.remote_work_penetration*100:.0f}% (target: 30%)",
        f"  Immigration total:     {p.immigration_annual:,}/yr",
        f"  Periphery pref:        {p.immigration_tier_prefs[2]*100:.0f}% = {int(p.immigration_annual * p.immigration_tier_prefs[2]):,}/yr (target: 10,000)",
        f"  University campuses:   {p.n_regional_universities} (target: 15)",
        f"  Enterprise zones:      {p.n_enterprise_zones} (target: 20)",
        "",
        "OUTCOME METRICS (end of simulation):",
        f"  Periphery mean digital access: {peri_digital:.3f} (target: 0.70 by 2030)",
        f"  Periphery mean healthcare:     {peri_health:.3f} (target: 0.75 by 2030)",
        f"  Final population:              {model.agents_pool.n_alive * scale:,}",
        f"  Simulation years:              {years}",
        f"  Runtime:                       {runtime:.0f}s",
        "",
        "=" * 70,
    ]

    path = str(out_dir / "POLICY_VALIDATION_REPORT.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  [VALIDATION] Report: {path}")

    # Also JSON for machine reading
    json_path = str(out_dir / "policy_validation.json")
    with open(json_path, "w") as f:
        json.dump({
            "checks": {k: v for k, v in checks.items()},
            "all_pass": all(checks.values()),
            "targets": VALIDATION_TARGETS,
            "actual": {
                "ice_act_credit": p.ice_act_tax_credit,
                "regional_wage_mult": p.regional_wage_multiplier,
                "remote_work_pct": p.remote_work_penetration,
                "immigration_annual": p.immigration_annual,
                "periphery_immigrant_annual": int(p.immigration_annual * p.immigration_tier_prefs[2]),
                "university_campuses": p.n_regional_universities,
                "enterprise_zones": p.n_enterprise_zones,
                "final_peri_digital": round(peri_digital, 4),
                "final_peri_healthcare": round(peri_health, 4),
            },
        }, f, indent=2)


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="VHA Evidence Kit -- Calibrated Takaichi 2.0")
    parser.add_argument("--scale", type=int, default=300)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-runs", type=int, default=50)
    parser.add_argument("--mc-scale", type=int, default=500)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-mc", action="store_true")
    args = parser.parse_args()

    if args.no_gpu:
        disable_gpu()

    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  VHA EVIDENCE KIT -- CALIBRATED TAKAICHI 2.0")
    print(f"  Dec 23 2025 Comprehensive Strategy Digital Twin")
    print(f"  ICE 7% | JPY 17.7T | Wage +15% | 10k Migrant Target")
    print(f"{'#'*70}")

    t_total = time.time()

    # D1 + D2 + D3 + D4
    model, results, tracker, nomad, corr, runtime = run_takaichi_calibrated(
        args.scale, args.years, args.seed, out,
    )

    # D2
    print(f"\n{'='*70}")
    print(f"  D2: Longitudinal Migration Log")
    print(f"{'='*70}")
    tracker.export(str(out / "D2_longitudinal_migration_log_cal.csv"))
    lp_stats = tracker.compute_launchpad_stats()
    if lp_stats:
        print(f"    Periphery movers: {lp_stats['total_periphery_origin_movers']}")
        print(f"    Launchpad (P->C->T): {lp_stats['launchpad_agents']} ({lp_stats['launchpad_pct']:.1%})")
        print(f"    Buffer (P->C):       {lp_stats['buffer_agents']} ({lp_stats['buffer_pct']:.1%})")
        print(f"    Direct (P->T):       {lp_stats['direct_to_tokyo']} ({lp_stats['direct_pct']:.1%})")
        with open(str(out / "D2_launchpad_stats_cal.json"), "w") as f:
            json.dump(lp_stats, f, indent=2)

    # D3
    print(f"\n{'='*70}")
    print(f"  D3: Nomad Demographic Export")
    print(f"{'='*70}")
    nomad.export(str(out / "D3_nomad_demographics_cal.csv"))

    # D4
    print(f"\n{'='*70}")
    print(f"  D4: Bivariate Correlation Table")
    print(f"{'='*70}")
    corr.export(str(out / "D4_bivariate_data_cal.csv"))
    corr_table = corr.compute_correlations(str(out / "D4_correlation_table_cal.csv"))
    if not corr_table.empty:
        print(f"\n    Key correlations (Periphery, final year):")
        for _, row in corr_table[corr_table["tier"] == "Periphery"].iterrows():
            print(f"      {row['variable_A']:>25s} vs {row['variable_B']:<25s}: "
                  f"r={row['pearson_r']:+.3f} {row['sig_pearson']}")

    # Validation report
    write_validation_report(model, args.scale, args.years, out, runtime)

    # D5
    if not args.skip_mc:
        run_monte_carlo_calibrated(args.mc_runs, args.years, args.mc_scale, args.seed, out)
    else:
        print(f"\n  [D5] Monte Carlo SKIPPED")

    elapsed = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"  VHA EVIDENCE KIT (CALIBRATED) COMPLETE")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*70}")
    print(f"\n  Output: {out}/")
    for f in sorted(out.iterdir()):
        if f.is_file():
            sz = f.stat().st_size
            print(f"    {f.name:<50s} {sz/1000:.0f} KB" if sz < 1e6 else
                  f"    {f.name:<50s} {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
