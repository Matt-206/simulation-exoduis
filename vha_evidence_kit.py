"""
VHA Evidence Kit Generator
==========================
Produces all five deliverables for the HSG VHA submission:

  1. Takaichi 2.0 Policy Run (20yr spreadsheet)
  2. Longitudinal Migration Log (5,000 tracked agents)
  3. Nomad Demographic Export (remote workers in rural zones)
  4. Bivariate Correlation Table (C vs A vs Births per municipality)
  5. Monte Carlo Summary (50-run statistical significance)

Usage:
    python vha_evidence_kit.py --no-gpu
    python vha_evidence_kit.py --no-gpu --mc-runs 30 --scale 400
"""

import argparse
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from model.config import SimulationConfig, ScaleConfig, disable_gpu
from model.model import ExodusModel
from run import SCENARIOS

TIER_NAMES = {0: "Tokyo", 1: "Core", 2: "Periphery"}
SCALE = 300
YEARS = 20
SEED = 42
MC_RUNS = 50
OUT = Path("output/vha_kit")


# ======================================================================
# Deliverable 1: Takaichi 2.0 Policy Run
# ======================================================================
def run_takaichi_spreadsheet(scale, years, seed, out_dir):
    """Full 20-year Takaichi 2.0 run with comprehensive spreadsheet."""
    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 1: Takaichi 2.0 Policy Run ({years} years)")
    print(f"{'='*70}\n")

    from export_spreadsheet import FullDataCollector, build_workbook

    builder = SCENARIOS["takaichi"]
    cfg = builder(scale, years, seed)
    model = ExodusModel(cfg)

    collector = FullDataCollector(model, scale)
    collector.collect_if_annual()

    # Also set up agent tracking and nomad collection for deliverables 2, 3, 4
    tracker = AgentLifetimeTracker(model, n_track=5000, scale=scale)
    nomad_collector = NomadCollector(model, scale)
    corr_collector = BivariateCorrelationCollector(model, scale)

    total_steps = years * cfg.scale.steps_per_year
    t0 = time.time()

    for step in range(total_steps):
        model.step()
        tracker.record_step()
        collector.collect_if_annual()
        nomad_collector.collect_if_annual()
        corr_collector.collect_if_annual()

        if step % cfg.scale.steps_per_year == 0:
            yr = model.current_year
            pop = model.agents_pool.n_alive * scale
            print(f"    {yr} -- Pop: {pop:,}")

    collector.collect_if_annual()
    nomad_collector.collect_if_annual()
    corr_collector.collect_if_annual()

    runtime = time.time() - t0
    results = model.get_results()

    # Build spreadsheet
    wb = build_workbook(
        results["history"], results["migration_flows"],
        collector, model, scale, "takaichi_2.0", runtime,
    )
    xlsx_path = str(out_dir / f"D1_takaichi_2.0_{years}yr.xlsx")
    wb.save(xlsx_path)
    print(f"  [D1] Spreadsheet: {xlsx_path}")

    # CSVs
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    results["history"].to_csv(str(csv_dir / "takaichi_history.csv"), index=False)
    if not results["migration_flows"].empty:
        results["migration_flows"].to_csv(str(csv_dir / "takaichi_migrations.csv"), index=False)

    return model, results, tracker, nomad_collector, corr_collector, runtime


# ======================================================================
# Deliverable 2: Longitudinal Migration Log
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

        # Sample diverse agents: stratify by tier and age
        sampled = []
        tiers = loc.tier[pool.location[:n]]
        for tier_id in [0, 1, 2]:
            tier_alive = alive_idx[tiers[alive_idx] == tier_id]
            # Mix of young (20-34) and middle-aged (35-50)
            young = tier_alive[(pool.age[tier_alive] >= 20) & (pool.age[tier_alive] <= 34)]
            mid = tier_alive[(pool.age[tier_alive] >= 35) & (pool.age[tier_alive] <= 50)]
            n_per = n_track // 6
            if len(young) > 0:
                sampled.extend(rng.choice(young, size=min(n_per, len(young)), replace=False))
            if len(mid) > 0:
                sampled.extend(rng.choice(mid, size=min(n_per, len(mid)), replace=False))

        # Fill remaining from general population
        remaining = n_track - len(sampled)
        if remaining > 0:
            others = np.setdiff1d(alive_idx, sampled)
            sampled.extend(rng.choice(others, size=min(remaining, len(others)), replace=False))

        self.tracked_ids = np.array(sampled[:n_track], dtype=np.int64)
        self.n_track = len(self.tracked_ids)

        # Record initial state
        self.records = []
        for idx in self.tracked_ids:
            loc_id = int(pool.location[idx])
            self.records.append({
                "agent_id": int(idx),
                "year": model.current_year,
                "quarter": model.current_step_in_year + 1,
                "step": model.total_steps,
                "event": "INITIAL",
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

        # Track previous locations to detect moves
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
                "agent_id": idx,
                "year": yr,
                "quarter": q,
                "step": step,
                "event": "MIGRATION",
                "location_id": new_loc,
                "from_location_id": old_loc,
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

        # Record deaths
        died = (~alive) & (self._prev_locs >= 0)
        for i in np.where(died)[0]:
            idx = int(self.tracked_ids[i])
            self.records.append({
                "agent_id": idx,
                "year": yr, "quarter": q, "step": step,
                "event": "DEATH",
                "location_id": int(self._prev_locs[i]),
                "tier": TIER_NAMES.get(int(loc.tier[int(self._prev_locs[i])]), "?"),
                "origin_tier": TIER_NAMES.get(int(pool.origin_tier[idx]), "?"),
                "age": int(pool.age[idx]),
                "sex": "M" if pool.sex[idx] == 0 else "F",
            })
            self._prev_locs[i] = -1  # mark as dead

        self._prev_locs = cur_locs.copy()
        self._prev_locs[~alive] = -1

    def export(self, path: str):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        print(f"  [D2] Longitudinal log: {path} ({len(df):,} events, {self.n_track:,} agents)")
        return df

    def compute_launchpad_stats(self):
        """Analyze tracked agents for launchpad behavior."""
        df = pd.DataFrame(self.records)
        migrations = df[df["event"] == "MIGRATION"].copy()
        if migrations.empty:
            return {}

        # Agents who went Periphery -> Core -> Tokyo
        agent_paths = {}
        for agent_id, grp in migrations.groupby("agent_id"):
            tiers_seq = grp["tier"].tolist()
            agent_paths[agent_id] = tiers_seq

        # Count agents exhibiting launchpad behavior
        launchpad_agents = 0
        buffer_agents = 0
        direct_agents = 0

        for agent_id, path in agent_paths.items():
            origin = df[df["agent_id"] == agent_id].iloc[0].get("origin_tier", "?")
            if origin != "Periphery":
                continue

            full_path = [origin] + path
            visited_core = "Core" in full_path
            ended_tokyo = path[-1] == "Tokyo" if path else False

            if visited_core and ended_tokyo:
                launchpad_agents += 1
            elif visited_core and not ended_tokyo:
                buffer_agents += 1
            elif ended_tokyo and not visited_core:
                direct_agents += 1

        total_peri_movers = launchpad_agents + buffer_agents + direct_agents
        return {
            "total_periphery_origin_movers": total_peri_movers,
            "launchpad_agents": launchpad_agents,
            "buffer_agents": buffer_agents,
            "direct_to_tokyo": direct_agents,
            "launchpad_pct": launchpad_agents / max(total_peri_movers, 1),
            "buffer_pct": buffer_agents / max(total_peri_movers, 1),
            "direct_pct": direct_agents / max(total_peri_movers, 1),
        }


# ======================================================================
# Deliverable 3: Nomad Demographic Export
# ======================================================================
class NomadCollector:
    """Collect demographics of remote workers in periphery zones."""

    def __init__(self, model: ExodusModel, scale: int):
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

        locs = pool.location[:n]
        tiers = loc.tier[locs]
        remote = pool.remote_worker[:n]

        # Remote workers in periphery
        nomad_mask = alive & remote & (tiers == 2)
        # Also collect remote workers in core for comparison
        nomad_core_mask = alive & remote & (tiers == 1)

        for label, mask in [("periphery_nomad", nomad_mask), ("core_nomad", nomad_core_mask)]:
            idx = np.where(mask)[0]
            if len(idx) == 0:
                self.snapshots.append({
                    "year": yr, "category": label,
                    "count": 0,
                })
                continue

            ages = pool.age[idx]
            incomes = pool.income[idx]
            children = pool.n_children[idx]
            edu = pool.education[idx]
            marital = pool.marital_status[idx]
            sexes = pool.sex[idx]

            self.snapshots.append({
                "year": yr,
                "category": label,
                "count": int(len(idx)) * self.scale,
                "mean_age": round(float(ages.mean()), 1),
                "median_age": round(float(np.median(ages)), 1),
                "pct_female": round(float((sexes == 1).mean()), 4),
                "mean_income": round(float(incomes.mean())),
                "median_income": round(float(np.median(incomes))),
                "p25_income": round(float(np.percentile(incomes, 25))),
                "p75_income": round(float(np.percentile(incomes, 75))),
                "income_std": round(float(incomes.std())),
                "mean_children": round(float(children.mean()), 2),
                "pct_zero_children": round(float((children == 0).mean()), 4),
                "pct_1plus_children": round(float((children >= 1).mean()), 4),
                "pct_2plus_children": round(float((children >= 2).mean()), 4),
                "pct_married": round(float((marital == 1).mean()), 4),
                "pct_higher_education": round(float((edu >= 2).mean()), 4),
                "age_20_29": int(((ages >= 20) & (ages <= 29)).sum()) * self.scale,
                "age_30_39": int(((ages >= 30) & (ages <= 39)).sum()) * self.scale,
                "age_40_49": int(((ages >= 40) & (ages <= 49)).sum()) * self.scale,
                "age_50_plus": int((ages >= 50).sum()) * self.scale,
            })

    def export(self, path: str):
        df = pd.DataFrame(self.snapshots)
        df.to_csv(path, index=False)
        print(f"  [D3] Nomad demographics: {path} ({len(df)} rows)")
        return df


# ======================================================================
# Deliverable 4: Bivariate Correlation Table
# ======================================================================
class BivariateCorrelationCollector:
    """Collect C, A, and local births per municipality for correlation analysis."""

    def __init__(self, model: ExodusModel, scale: int):
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

        # Count births per location (babies born to mothers at each location)
        # Use n_children increments as proxy: women with recent births
        # More precisely: count reproductive females and recent births from history
        locs = pool.location[:n][alive]
        ages = pool.age[:n][alive]
        sexes = pool.sex[:n][alive]
        children = pool.n_children[:n][alive]
        marital = pool.marital_status[:n][alive]

        # Women 15-49 per location
        fertile_mask = (sexes == 1) & (ages >= 15) & (ages <= 49)
        fertile_locs = locs[fertile_mask]
        fertile_per_loc = np.bincount(fertile_locs, minlength=n_locs)

        # Children per location (sum of all n_children for women there)
        children_f = children[fertile_mask]
        children_per_loc = np.zeros(n_locs)
        np.add.at(children_per_loc, fertile_locs, children_f)

        # Crude local birth rate proxy: children / fertile women
        local_birth_rate = np.where(fertile_per_loc > 0, children_per_loc / fertile_per_loc, 0)

        # Population per loc
        pop_per_loc = np.bincount(locs, minlength=n_locs)

        names = getattr(loc, '_municipality_names', None)
        prefs = getattr(loc, '_prefectures', None)

        for i in range(n_locs):
            self.snapshots.append({
                "year": yr,
                "location_id": i,
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

    def export(self, path: str):
        df = pd.DataFrame(self.snapshots)
        df.to_csv(path, index=False)
        print(f"  [D4] Bivariate data: {path} ({len(df):,} rows)")
        return df

    def compute_correlations(self, path: str):
        """Compute Pearson/Spearman correlations for final year."""
        df = pd.DataFrame(self.snapshots)
        if df.empty:
            return pd.DataFrame()

        last_yr = df["year"].max()
        final = df[df["year"] == last_yr].copy()
        final = final[final["population"] > 0]

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
            if tier_label == "All":
                sub = final
            else:
                sub = final[final["tier"] == tier_label]

            if len(sub) < 5:
                continue

            for var_a, var_b in pairs:
                a = sub[var_a].values
                b = sub[var_b].values
                valid = np.isfinite(a) & np.isfinite(b)
                a, b = a[valid], b[valid]
                if len(a) < 5:
                    continue

                r_pearson, p_pearson = sp_stats.pearsonr(a, b)
                r_spearman, p_spearman = sp_stats.spearmanr(a, b)

                rows.append({
                    "tier": tier_label,
                    "variable_A": var_a,
                    "variable_B": var_b,
                    "n_municipalities": len(a),
                    "pearson_r": round(r_pearson, 4),
                    "pearson_p": round(p_pearson, 6),
                    "pearson_sig": "***" if p_pearson < 0.001 else "**" if p_pearson < 0.01 else "*" if p_pearson < 0.05 else "ns",
                    "spearman_rho": round(r_spearman, 4),
                    "spearman_p": round(p_spearman, 6),
                    "spearman_sig": "***" if p_spearman < 0.001 else "**" if p_spearman < 0.01 else "*" if p_spearman < 0.05 else "ns",
                })

        corr_df = pd.DataFrame(rows)
        corr_df.to_csv(path, index=False)
        print(f"  [D4] Correlation table: {path}")
        return corr_df


# ======================================================================
# Deliverable 5: Monte Carlo Summary
# ======================================================================
def run_monte_carlo_summary(n_runs, years, scale, base_seed, out_dir):
    """Run N simulations and produce mean/std/CI summary."""
    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 5: Monte Carlo ({n_runs} runs x {years} years)")
    print(f"{'='*70}\n")

    builder = SCENARIOS["takaichi"]
    all_runs = []
    annual_series = defaultdict(list)

    for run_i in range(n_runs):
        seed = base_seed + run_i * 7919
        cfg = builder(scale, years, seed)
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
        total_mig = int(h["migrations"].sum())

        run_data = {
            "run": run_i,
            "seed": seed,
            "final_population": total_pop,
            "tokyo_share": tokyo_share,
            "periphery_share": peri_share,
            "tfr": tfr,
            "mean_age": mean_age,
            "periphery_extinct": n_extinct,
            "pct_extinct": n_extinct / max(n_periphery, 1),
            "launchpad_ratio": launchpad,
            "cannibalism_ratio": cannibalism,
            "warehouse_towns": warehouse,
            "total_births": total_births,
            "total_deaths": total_deaths,
            "total_migrations": total_mig,
            "runtime_s": round(elapsed, 1),
        }
        all_runs.append(run_data)

        # Annual time series
        for _, row in h.groupby("year").last().iterrows():
            yr = int(row.get("year", 0))
            annual_series[yr].append({
                "total_population": int(row["total_population"]) * scale,
                "tokyo_pop_share": float(row["tokyo_pop_share"]),
                "peri_pop_share": float(row["peri_pop_share"]),
                "tfr_proxy": float(row.get("tfr_proxy", 0)),
                "mean_age": float(row.get("mean_age", 0)),
            })

        print(f"  [{run_i+1:>3}/{n_runs}] pop={total_pop:>12,} tokyo={tokyo_share:.3f} "
              f"tfr={tfr:.3f} extinct={n_extinct}/{n_periphery} ({elapsed:.1f}s)")

    # ── Aggregate ──
    df = pd.DataFrame(all_runs)
    df.to_csv(str(out_dir / "D5_montecarlo_runs.csv"), index=False)

    # Compute summary statistics
    metrics = ["final_population", "tokyo_share", "periphery_share", "tfr", "mean_age",
               "periphery_extinct", "pct_extinct", "launchpad_ratio", "cannibalism_ratio",
               "warehouse_towns", "total_births", "total_deaths", "total_migrations"]

    summary_rows = []
    for m in metrics:
        vals = df[m].values
        n = len(vals)
        summary_rows.append({
            "metric": m,
            "n_runs": n,
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "ci95_lo": round(float(vals.mean() - 1.96 * vals.std() / np.sqrt(n)), 4),
            "ci95_hi": round(float(vals.mean() + 1.96 * vals.std() / np.sqrt(n)), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "median": round(float(np.median(vals)), 4),
            "p5": round(float(np.percentile(vals, 5)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(str(out_dir / "D5_montecarlo_summary.csv"), index=False)

    # Annual time series mean/std
    ts_rows = []
    for yr in sorted(annual_series.keys()):
        vals = annual_series[yr]
        for key in ["total_population", "tokyo_pop_share", "peri_pop_share", "tfr_proxy", "mean_age"]:
            arr = np.array([v[key] for v in vals])
            ts_rows.append({
                "year": yr, "metric": key,
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "ci95_lo": round(float(arr.mean() - 1.96 * arr.std() / np.sqrt(len(arr))), 4),
                "ci95_hi": round(float(arr.mean() + 1.96 * arr.std() / np.sqrt(len(arr))), 4),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
            })

    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_csv(str(out_dir / "D5_montecarlo_timeseries.csv"), index=False)

    # ── Print report ──
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO RESULTS ({n_runs} runs, Takaichi 2.0)")
    print(f"{'='*70}")
    for _, row in summary_df.iterrows():
        m = row["metric"]
        if "population" in m or "births" in m or "deaths" in m:
            print(f"  {m:<25s}: {row['mean']:>14,.0f} +/- {row['std']:>10,.0f}  "
                  f"[{row['ci95_lo']:>14,.0f}, {row['ci95_hi']:>14,.0f}]")
        else:
            print(f"  {m:<25s}: {row['mean']:>14.4f} +/- {row['std']:>10.4f}  "
                  f"[{row['ci95_lo']:>14.4f}, {row['ci95_hi']:>14.4f}]")

    print(f"\n  Saved: {out_dir}/D5_montecarlo_*.csv")
    return df, summary_df, ts_df


# ======================================================================
# Master Orchestrator
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="VHA Evidence Kit Generator")
    parser.add_argument("--scale", type=int, default=SCALE)
    parser.add_argument("--years", type=int, default=YEARS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mc-runs", type=int, default=MC_RUNS)
    parser.add_argument("--mc-scale", type=int, default=500, help="Scale for MC runs (larger = faster)")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-mc", action="store_true", help="Skip Monte Carlo (slow)")
    args = parser.parse_args()

    if args.no_gpu:
        disable_gpu()

    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  VHA EVIDENCE KIT GENERATOR")
    print(f"  Takaichi 2.0 | {args.years} years | Scale 1:{args.scale} | MC: {args.mc_runs} runs")
    print(f"{'#'*70}")

    t_total = time.time()

    # ── D1 + D2 + D3 + D4: Run Takaichi scenario (collects all data in one pass) ──
    model, results, tracker, nomad, corr, runtime = run_takaichi_spreadsheet(
        args.scale, args.years, args.seed, out,
    )

    # ── D2: Export longitudinal migration log ──
    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 2: Longitudinal Migration Log")
    print(f"{'='*70}")
    tracker.export(str(out / "D2_longitudinal_migration_log.csv"))
    launchpad_stats = tracker.compute_launchpad_stats()
    if launchpad_stats:
        print(f"    Periphery-origin movers tracked: {launchpad_stats['total_periphery_origin_movers']}")
        print(f"    Launchpad (P->C->T):  {launchpad_stats['launchpad_agents']} ({launchpad_stats['launchpad_pct']:.1%})")
        print(f"    Buffer (P->C, stayed): {launchpad_stats['buffer_agents']} ({launchpad_stats['buffer_pct']:.1%})")
        print(f"    Direct (P->T):         {launchpad_stats['direct_to_tokyo']} ({launchpad_stats['direct_pct']:.1%})")
        with open(str(out / "D2_launchpad_stats.json"), "w") as f:
            json.dump(launchpad_stats, f, indent=2)

    # ── D3: Export nomad demographics ──
    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 3: Nomad Demographic Export")
    print(f"{'='*70}")
    nomad.export(str(out / "D3_nomad_demographics.csv"))

    # ── D4: Export bivariate correlations ──
    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 4: Bivariate Correlation Table")
    print(f"{'='*70}")
    corr.export(str(out / "D4_bivariate_data.csv"))
    corr_table = corr.compute_correlations(str(out / "D4_correlation_table.csv"))
    if not corr_table.empty:
        print(f"\n    Key correlations (final year):")
        for _, row in corr_table[corr_table["tier"] == "Periphery"].iterrows():
            sig = row["pearson_sig"]
            print(f"      {row['variable_A']:>25s} vs {row['variable_B']:<25s}: "
                  f"r={row['pearson_r']:+.3f} {sig}")

    # ── D5: Monte Carlo ──
    if not args.skip_mc:
        mc_df, mc_summary, mc_ts = run_monte_carlo_summary(
            args.mc_runs, args.years, args.mc_scale, args.seed, out,
        )
    else:
        print(f"\n  [D5] Monte Carlo SKIPPED (use without --skip-mc to run)")

    # ── Final manifest ──
    elapsed_total = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"  VHA EVIDENCE KIT COMPLETE")
    print(f"  Total runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'#'*70}")
    print(f"\n  Output directory: {out}/")
    print(f"  Files:")

    for f in sorted(out.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            if size > 1_000_000:
                print(f"    {f.name:<45s} {size/1_000_000:.1f} MB")
            else:
                print(f"    {f.name:<45s} {size/1_000:.0f} KB")


if __name__ == "__main__":
    main()
