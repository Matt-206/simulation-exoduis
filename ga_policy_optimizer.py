"""
Genetic Algorithm Policy Optimizer
===================================
Evolves "Policy DNA" strings to maximise the Demographic Health Score (DHS)
under a fiscal constraint of JPY 20 trillion/year.

Objective:
    DHS = w1 * (TFR / 2.1) + w2 * (1 - TokyoShare) - w3 * (Warehouse / N_peri)

Search space (5 genes):
    ICE Act credit:            0.05 - 0.25
    Regional wage multiplier:  1.00 - 1.35
    Anomie friction credit:    0.00 - 0.40
    Circular tax (JPY/child):  0    - 800
    Immigration cap (/yr):     100k - 1M

Fiscal constraint:  sum(costs) <= JPY 20 trillion / year

Usage:
    python ga_policy_optimizer.py --no-gpu
    python ga_policy_optimizer.py --no-gpu --pop 60 --gen 30 --scale 100
"""

import argparse, time, json, sys, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from model.config import SimulationConfig, ScaleConfig, PolicyConfig, disable_gpu
from model.model import ExodusModel
from run import make_baseline_config

warnings.filterwarnings("ignore")

OUT = Path("output/ga_optimizer")

# =====================================================================
# Policy DNA
# =====================================================================
GENE_SPEC = {
    #  name               min     max
    "ice_credit":        (0.05,  0.25),
    "wage_mult":         (1.00,  1.35),
    "anomie_credit":     (0.00,  0.40),
    "circular_tax":      (0.0,   800.0),
    "immigration":       (100_000, 1_000_000),
}
GENE_NAMES = list(GENE_SPEC.keys())
N_GENES = len(GENE_NAMES)


@dataclass
class PolicyDNA:
    ice_credit: float
    wage_mult: float
    anomie_credit: float
    circular_tax: float
    immigration: float

    def as_array(self) -> np.ndarray:
        return np.array([self.ice_credit, self.wage_mult, self.anomie_credit,
                         self.circular_tax, self.immigration])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PolicyDNA":
        return cls(
            ice_credit=float(arr[0]),
            wage_mult=float(arr[1]),
            anomie_credit=float(arr[2]),
            circular_tax=float(arr[3]),
            immigration=float(arr[4]),
        )

    def to_dict(self):
        return {n: getattr(self, n) for n in GENE_NAMES}


def random_dna(rng: np.random.Generator) -> PolicyDNA:
    vals = []
    for name in GENE_NAMES:
        lo, hi = GENE_SPEC[name]
        vals.append(rng.uniform(lo, hi))
    return PolicyDNA.from_array(np.array(vals))


def clamp_dna(dna: PolicyDNA) -> PolicyDNA:
    arr = dna.as_array()
    for i, name in enumerate(GENE_NAMES):
        lo, hi = GENE_SPEC[name]
        arr[i] = np.clip(arr[i], lo, hi)
    return PolicyDNA.from_array(arr)


# =====================================================================
# Fiscal Cost Model (JPY per year)
# =====================================================================
FISCAL_CAP = 20e12  # JPY 20 trillion

def fiscal_cost_jpy(dna: PolicyDNA) -> float:
    """Approximate annual fiscal cost of a policy bundle in JPY."""
    # ICE Act: credit_rate x qualifying investment volume (~500B/yr)
    ice = dna.ice_credit * 500e9

    # Wage subsidy: govt funds ~30% of the uplift across non-Tokyo workforce
    # 40M workers x 4M avg wage = 160T total wages
    wage = max(0, dna.wage_mult - 1.0) * 0.30 * 160e12

    # Anomie friction credit: direct community revival grants
    # Scales with credit rate; 0.40 -> ~5T/yr across 1625 periphery towns
    anomie = dna.anomie_credit * 12.5e12

    # Circular tax / child benefit: per-child monthly x 15M children x 12
    child = dna.circular_tax * 15e6 * 12

    # Immigration: settlement support ~2M JPY per person
    immig = dna.immigration * 2e6

    return ice + wage + anomie + child + immig


def is_feasible(dna: PolicyDNA) -> bool:
    return fiscal_cost_jpy(dna) <= FISCAL_CAP


# =====================================================================
# DHS Objective Function
# =====================================================================
DHS_W1 = 10.0   # TFR weight (most important demographic signal)
DHS_W2 = 5.0    # Decentralisation weight
DHS_W3 = 3.0    # Human warehouse penalty

N_PERIPHERY = 1625  # total periphery municipalities

def compute_dhs(tfr: float, tokyo_share: float, warehouse_count: int) -> float:
    """
    DHS = w1*(TFR/2.1) + w2*(1 - TokyoShare) - w3*(Warehouse/N_peri)
    Higher is better.
    """
    return (
        DHS_W1 * (tfr / 2.1)
        + DHS_W2 * (1.0 - tokyo_share)
        - DHS_W3 * (warehouse_count / N_PERIPHERY)
    )


# =====================================================================
# Simulation Evaluator
# =====================================================================
def build_config(dna: PolicyDNA, scale: int, years: int, seed: int) -> SimulationConfig:
    cfg = make_baseline_config(scale, years, seed)
    p = cfg.policy

    p.ice_act_tax_credit = dna.ice_credit
    p.hq_relocation_active = True
    p.hq_relocation_prestige_boost = 0.18

    p.regional_wage_multiplier = dna.wage_mult

    p.circular_tax_per_child_monthly = dna.circular_tax

    p.immigration_active = True
    p.immigration_annual = int(dna.immigration)
    total = dna.immigration
    peri_frac = min(0.15, 10_000 / max(total, 1))
    p.immigration_tier_prefs = {0: 0.50, 1: 0.50 - peri_frac, 2: peri_frac}

    # Fixed policy levers (Dec 2025 baseline)
    p.remote_work_penetration = 0.30
    p.medical_dx_rollout_rate = 0.08
    p.level4_pod_deployment_rate = 0.09
    p.childcare_subsidy_ratio = 0.70
    p.childcare_expansion_rate = 0.05
    p.housing_subsidy_periphery = 0.30
    p.housing_subsidy_core = 0.15
    p.university_decentralization = True
    p.n_regional_universities = 15
    p.enterprise_zones_active = True
    p.n_enterprise_zones = 20
    p.shinkansen_expansion_active = True

    return cfg


def evaluate(dna: PolicyDNA, scale: int, years: int, seed: int) -> dict:
    """Run one simulation and return DHS + metrics."""
    if not is_feasible(dna):
        cost = fiscal_cost_jpy(dna)
        penalty = -10.0 * ((cost - FISCAL_CAP) / FISCAL_CAP)
        return {"dhs": penalty, "feasible": False, "cost_t": cost / 1e12,
                "tfr": 0, "tokyo_share": 0, "warehouse": 0, "population": 0}

    cfg = build_config(dna, scale, years, seed)
    model = ExodusModel(cfg)

    total_steps = years * cfg.scale.steps_per_year
    for _ in range(total_steps):
        model.step()
        # Apply anomie friction credit each quarter
        if dna.anomie_credit > 0:
            peri = model.loc_state.tier == 2
            model.loc_state.anomie[peri] = np.clip(
                model.loc_state.anomie[peri] - dna.anomie_credit * 0.25 * 0.1,
                0, 1,
            )

    h = model.history
    if not h:
        return {"dhs": -99, "feasible": True, "cost_t": fiscal_cost_jpy(dna)/1e12,
                "tfr": 0, "tokyo_share": 0, "warehouse": 0, "population": 0}

    final = h[-1]
    tfr = final.get("tfr_proxy", 0)
    tokyo_share = final.get("tokyo_pop_share", 0)
    warehouse = final.get("n_human_warehouse_towns", 0)
    pop = final.get("total_population", 0) * scale

    dhs = compute_dhs(tfr, tokyo_share, warehouse)

    return {
        "dhs": round(dhs, 6),
        "feasible": True,
        "cost_t": round(fiscal_cost_jpy(dna) / 1e12, 2),
        "tfr": round(tfr, 4),
        "tokyo_share": round(tokyo_share, 4),
        "warehouse": warehouse,
        "population": pop,
        "mean_age": final.get("mean_age", 0),
        "peri_share": final.get("peri_pop_share", 0),
    }


# =====================================================================
# GA Operators
# =====================================================================
def tournament_select(pop: List[PolicyDNA], fitness: np.ndarray,
                      rng: np.random.Generator, k: int = 3) -> PolicyDNA:
    idx = rng.choice(len(pop), size=k, replace=False)
    best = idx[np.argmax(fitness[idx])]
    return pop[best]


def crossover(p1: PolicyDNA, p2: PolicyDNA, rng: np.random.Generator) -> PolicyDNA:
    a1, a2 = p1.as_array(), p2.as_array()
    # BLX-alpha blend crossover
    alpha = 0.3
    lo = np.minimum(a1, a2) - alpha * np.abs(a1 - a2)
    hi = np.maximum(a1, a2) + alpha * np.abs(a1 - a2)
    child = rng.uniform(lo, hi)
    return clamp_dna(PolicyDNA.from_array(child))


def mutate(dna: PolicyDNA, rng: np.random.Generator, sigma: float = 0.15) -> PolicyDNA:
    arr = dna.as_array()
    ranges = np.array([GENE_SPEC[n][1] - GENE_SPEC[n][0] for n in GENE_NAMES])
    noise = rng.normal(0, sigma, N_GENES) * ranges
    mask = rng.random(N_GENES) < 0.3  # mutate ~30% of genes
    arr[mask] += noise[mask]
    return clamp_dna(PolicyDNA.from_array(arr))


# =====================================================================
# GA Main Loop
# =====================================================================
def run_ga(pop_size: int, n_elite: int, n_gen: int,
           scale: int, years: int, base_seed: int):
    rng = np.random.default_rng(base_seed)
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    # ── Seed with calibrated Dec 2025 policy to give GA a head start
    calibrated = PolicyDNA(
        ice_credit=0.07, wage_mult=1.15, anomie_credit=0.0,
        circular_tax=500.0, immigration=200_000,
    )

    population = [calibrated]
    for _ in range(pop_size - 1):
        dna = random_dna(rng)
        # Repair: if infeasible, shrink the most expensive gene
        attempts = 0
        while not is_feasible(dna) and attempts < 20:
            arr = dna.as_array()
            # Reduce wage_mult (most expensive) or anomie_credit
            arr[1] -= 0.02
            arr[2] -= 0.02
            dna = clamp_dna(PolicyDNA.from_array(arr))
            attempts += 1
        population.append(dna)

    all_log = []
    best_ever_dhs = -999
    best_ever_dna = calibrated
    best_ever_result = {}

    print(f"\n{'#'*70}")
    print(f"  GENETIC ALGORITHM POLICY OPTIMIZER")
    print(f"  Pop={pop_size} Elite={n_elite} Gen={n_gen}")
    print(f"  Scale={scale} Years={years} Seed={base_seed}")
    print(f"  Fiscal cap: JPY {FISCAL_CAP/1e12:.0f}T/yr")
    print(f"{'#'*70}\n")

    t_total = time.time()

    for gen in range(n_gen):
        t_gen = time.time()
        fitness = np.full(len(population), -999.0)
        results = [None] * len(population)

        for i, dna in enumerate(population):
            seed_i = base_seed + gen * 1000 + i
            res = evaluate(dna, scale, years, seed_i)
            fitness[i] = res["dhs"]
            results[i] = res

            all_log.append({
                "gen": gen, "individual": i,
                **dna.to_dict(), **res,
            })

        # Stats
        feasible_mask = np.array([r["feasible"] for r in results])
        n_feasible = feasible_mask.sum()
        feas_fit = fitness[feasible_mask] if n_feasible > 0 else np.array([0])

        gen_best_idx = np.argmax(fitness)
        gen_best_dna = population[gen_best_idx]
        gen_best_res = results[gen_best_idx]

        if fitness[gen_best_idx] > best_ever_dhs:
            best_ever_dhs = fitness[gen_best_idx]
            best_ever_dna = gen_best_dna
            best_ever_result = gen_best_res

        elapsed = time.time() - t_gen
        print(f"  Gen {gen:>3d}/{n_gen} | best={fitness[gen_best_idx]:+.4f} "
              f"mean={feas_fit.mean():+.4f} | "
              f"TFR={gen_best_res['tfr']:.3f} Tokyo={gen_best_res['tokyo_share']:.3f} "
              f"WH={gen_best_res['warehouse']} "
              f"cost={gen_best_res['cost_t']:.1f}T | "
              f"feasible={n_feasible}/{len(population)} | {elapsed:.0f}s")

        if gen == n_gen - 1:
            break

        # ── Selection: keep elite
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = [population[i] for i in elite_idx]

        # ── Breed next generation
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            p1 = tournament_select(population, fitness, rng, k=3)
            p2 = tournament_select(population, fitness, rng, k=3)
            child = crossover(p1, p2, rng)
            child = mutate(child, rng, sigma=max(0.05, 0.20 * (1 - gen / n_gen)))
            next_pop.append(child)

        population = next_pop

    total_time = time.time() - t_total

    # ── Results
    print(f"\n{'='*70}")
    print(f"  GA COMPLETE -- {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n  OPTIMAL POLICY DNA:")
    print(f"    ICE Act credit:       {best_ever_dna.ice_credit*100:.1f}%")
    print(f"    Regional wage mult:   {best_ever_dna.wage_mult:.3f} (+{(best_ever_dna.wage_mult-1)*100:.1f}%)")
    print(f"    Anomie friction:      {best_ever_dna.anomie_credit:.3f}")
    print(f"    Circular tax:         JPY {best_ever_dna.circular_tax:.0f}/child/month")
    print(f"    Immigration cap:      {best_ever_dna.immigration:,.0f}/yr")
    print(f"\n  FISCAL COST: JPY {fiscal_cost_jpy(best_ever_dna)/1e12:.2f}T / {FISCAL_CAP/1e12:.0f}T cap")
    print(f"\n  DHS = {best_ever_dhs:+.4f}")
    print(f"    TFR:            {best_ever_result.get('tfr', 0):.4f}")
    print(f"    Tokyo share:    {best_ever_result.get('tokyo_share', 0):.4f}")
    print(f"    Warehouse:      {best_ever_result.get('warehouse', 0)}")
    print(f"    Population:     {best_ever_result.get('population', 0):,.0f}")

    # ── Save
    log_df = pd.DataFrame(all_log)
    log_df.to_csv(str(out / "ga_full_log.csv"), index=False)

    winner = {
        "dna": best_ever_dna.to_dict(),
        "dhs": best_ever_dhs,
        "metrics": best_ever_result,
        "fiscal_cost_T": fiscal_cost_jpy(best_ever_dna) / 1e12,
        "ga_params": {"pop": pop_size, "elite": n_elite, "gen": n_gen,
                      "scale": scale, "years": years},
        "runtime_s": total_time,
    }
    with open(str(out / "ga_winner.json"), "w") as f:
        json.dump(winner, f, indent=2, default=str)

    # Best per generation
    gen_best = log_df.groupby("gen").apply(
        lambda g: g.loc[g["dhs"].idxmax()], include_groups=False
    ).reset_index(drop=True)
    gen_best.to_csv(str(out / "ga_best_per_gen.csv"), index=False)

    print(f"\n  Files: {out}/")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"    {f.name}")

    return best_ever_dna, best_ever_dhs, best_ever_result


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="GA Policy Optimizer")
    parser.add_argument("--pop", type=int, default=40, help="Population size")
    parser.add_argument("--elite", type=int, default=10, help="Elite survivors")
    parser.add_argument("--gen", type=int, default=25, help="Generations")
    parser.add_argument("--scale", type=int, default=100, help="Agent scale (lower=faster)")
    parser.add_argument("--years", type=int, default=10, help="Sim years per eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    if args.no_gpu:
        disable_gpu()

    run_ga(args.pop, args.elite, args.gen, args.scale, args.years, args.seed)


if __name__ == "__main__":
    main()
