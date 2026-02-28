"""
BACKTEST: Run simulation from 2020 and compare against observed 2020-2024 data.

This tests whether the model's demographic processes (mortality, fertility,
migration, housing market, endogenous PCAF) reproduce the real trajectory
of Japan's population over the known historical period.

Validation targets from:
  - IPSS Annual Population Estimates
  - MHLW Vital Statistics (births, deaths)
  - Statistics Bureau Residence Reports (Tokyo concentration)
"""
import sys
import time
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from model.config import SimulationConfig, ScaleConfig, DemographyConfig
from model.model import ExodusModel
from model.real_data import VALIDATION_TARGETS

SCALE = 500  # 1:500 for reasonable speed + statistical significance

print("=" * 70)
print("  JAPAN EXODUS SIMULATION -- HISTORICAL BACKTEST 2020-2024")
print("=" * 70)

cfg = SimulationConfig(
    scale=ScaleConfig(
        agent_scale=SCALE,
        base_population=126_146_000,  # 2020 actual
        n_years=5,
        start_year=2020,
        random_seed=42,
    ),
)

print(f"\nScale: 1:{SCALE}  ({cfg.scale.n_agents:,} agents)")
print(f"Period: 2020-2024 (20 quarterly steps)")
print(f"Validation years: {sorted(VALIDATION_TARGETS.keys())}")

print("\nInitializing model...")
t0 = time.perf_counter()
model = ExodusModel(cfg)
init_time = time.perf_counter() - t0
print(f"Init complete: {init_time:.1f}s")

pool = model.agents_pool
n = pool.next_id
alive = pool.alive[:n]
ages = pool.age[:n]

print(f"\nInitial state (representing year 2020):")
print(f"  Total agents: {pool.n_alive:,} (real: {126_146_000:,})")
print(f"  Mean age: {ages[alive].mean():.1f}")
print(f"  Female %: {(pool.sex[:n][alive] == 1).mean() * 100:.1f}%")
print(f"  Married %: {(pool.marital_status[:n][alive] == 1).mean() * 100:.1f}%")
print(f"  Elderly (65+): {((ages >= 65) & alive).sum() / alive.sum() * 100:.1f}%")
print(f"  Youth (0-14): {((ages < 15) & alive).sum() / alive.sum() * 100:.1f}%")

loc = model.loc_state
tokyo_pop = int(loc.population[loc.tier == 0].sum())
total_pop = int(loc.population.sum())
print(f"  Tokyo share: {tokyo_pop / total_pop * 100:.1f}%")

print("\n" + "-" * 70)
print(f"{'Year':>6} | {'Metric':<16} | {'Simulated':>14} | {'Observed':>14} | {'Error %':>8}")
print("-" * 70)

year_births = {}
year_deaths = {}

print("\nRunning simulation...")
t_run = time.perf_counter()

for step in range(20):
    model.step()
    h = model.history[-1]
    yr = h["year"]
    q = h["quarter"]

    if yr not in year_births:
        year_births[yr] = 0
        year_deaths[yr] = 0
    year_births[yr] += h["births"]
    year_deaths[yr] += h["deaths"]

    if q == 4:
        sys.stdout.write(f"  Completed year {yr} (pop: {h['total_population']:,})\n")
        sys.stdout.flush()

run_time = time.perf_counter() - t_run
print(f"\nSimulation complete: {run_time:.1f}s ({run_time / 20:.2f}s/step)")

print("\n" + "=" * 70)
print("  BACKTEST RESULTS: Simulated vs Observed (2020-2024)")
print("=" * 70)

errors = []

for yr in sorted(VALIDATION_TARGETS.keys()):
    target = VALIDATION_TARGETS[yr]

    yr_data = [h for h in model.history if h["year"] == yr and h["quarter"] == 4]
    if not yr_data:
        yr_data = [h for h in model.history if h["year"] == yr]
    if not yr_data:
        continue

    last = yr_data[-1]

    sim_pop = int(last["total_population"]) * SCALE
    real_pop = target["total_pop"]
    pop_err = (sim_pop - real_pop) / real_pop * 100

    # Use Tokyo Prefecture share (all Tokyo-pref locations, not just 23 wards)
    sim_tokyo = float(last.get("tokyo_pref_share", last["tokyo_pop_share"]))
    real_tokyo = target["tokyo_share"]
    tokyo_err = (sim_tokyo - real_tokyo) / real_tokyo * 100

    sim_births = year_births.get(yr, 0) * SCALE
    real_births = target["births"]
    birth_err = (sim_births - real_births) / real_births * 100

    sim_deaths = year_deaths.get(yr, 0) * SCALE
    real_deaths = target["deaths"]
    death_err = (sim_deaths - real_deaths) / real_deaths * 100

    nat_change = sim_births - sim_deaths
    real_change = real_births - real_deaths

    print(f"\n  Year {yr}:")
    print(f"    {'Population':<16} | {sim_pop:>14,} | {real_pop:>14,} | {pop_err:>+7.1f}%")
    print(f"    {'Tokyo Pref %':<16} | {sim_tokyo:>13.1%} | {real_tokyo:>13.1%} | {tokyo_err:>+7.1f}%")
    print(f"    {'Births':<16} | {sim_births:>14,} | {real_births:>14,} | {birth_err:>+7.1f}%")
    print(f"    {'Deaths':<16} | {sim_deaths:>14,} | {real_deaths:>14,} | {death_err:>+7.1f}%")
    print(f"    {'Nat. change':<16} | {nat_change:>+14,} | {real_change:>+14,} |")

    errors.append({
        "year": yr,
        "pop_err": abs(pop_err),
        "tokyo_err": abs(tokyo_err),
        "birth_err": abs(birth_err),
        "death_err": abs(death_err),
    })

print("\n" + "=" * 70)
print("  SUMMARY: Mean Absolute Errors across 2020-2024")
print("=" * 70)

if errors:
    mean_pop = np.mean([e["pop_err"] for e in errors])
    mean_tokyo = np.mean([e["tokyo_err"] for e in errors])
    mean_birth = np.mean([e["birth_err"] for e in errors])
    mean_death = np.mean([e["death_err"] for e in errors])

    print(f"\n  Population:   {mean_pop:.1f}% mean error")
    print(f"  Tokyo share:  {mean_tokyo:.1f}% mean error")
    print(f"  Births:       {mean_birth:.1f}% mean error")
    print(f"  Deaths:       {mean_death:.1f}% mean error")

    overall = np.mean([mean_pop, mean_tokyo, mean_birth, mean_death])
    print(f"\n  OVERALL:      {overall:.1f}% mean absolute error")

    if overall < 5:
        grade = "EXCELLENT"
    elif overall < 10:
        grade = "GOOD"
    elif overall < 20:
        grade = "FAIR"
    elif overall < 40:
        grade = "NEEDS CALIBRATION"
    else:
        grade = "POOR -- MAJOR CALIBRATION NEEDED"
    print(f"  Grade:        {grade}")

# --- Additional diagnostics ---
print("\n" + "=" * 70)
print("  ENDOGENOUS VARIABLE TRAJECTORIES (2020-2024)")
print("=" * 70)

for yr in sorted(VALIDATION_TARGETS.keys()):
    yr_data = [h for h in model.history if h["year"] == yr and h["quarter"] == 4]
    if not yr_data:
        continue
    h = yr_data[-1]
    print(f"\n  {yr}:")
    print(f"    Prestige    -- Tokyo: {h['mean_prestige_tokyo']:.3f}  Core: {h['mean_prestige_core']:.3f}  Peri: {h['mean_prestige_periphery']:.3f}")
    print(f"    Convenience -- Periphery: {h['mean_convenience_periphery']:.3f}")
    print(f"    Anomie      -- Periphery: {h['mean_anomie_periphery']:.3f}")
    print(f"    Friction    -- Tokyo: {h['mean_friction_tokyo']:.3f}")
    print(f"    Insurance surcharge: JPY {h['insurance_surcharge_jpy']:.0f}/month")
    print(f"    HQs: Tokyo={h['tokyo_hq_count']}  Core={h['core_hq_count']}")
    print(f"    Human Warehouse towns: {h['n_human_warehouse_towns']}")
    print(f"    School closures: {h['n_school_closures']}")
    print(f"    Unemployed: {h['n_unemployed']:,}")

print("\n" + "=" * 70)
print("  BACKTEST COMPLETE")
print("=" * 70)
