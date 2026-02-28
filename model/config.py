"""
Central configuration for the Japan Exodus Simulation.

All parameters are empirically calibrated to Japanese demographic,
economic, and sociological data (2020-2025 baselines).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
GPU_AVAILABLE = False
xp = np

try:
    import cupy as cp
    _test = cp.array([1.0, 2.0]) + cp.array([3.0, 4.0])
    cp.asnumpy(_test)
    del _test
    GPU_AVAILABLE = True
    xp = cp
except Exception:
    # CuPy unavailable or CUDA version mismatch -- fall back to NumPy + Numba
    GPU_AVAILABLE = False
    xp = np

_GPU_FORCED_OFF = False


def disable_gpu():
    global GPU_AVAILABLE, xp, _GPU_FORCED_OFF
    GPU_AVAILABLE = False
    xp = np
    _GPU_FORCED_OFF = True


def get_array_module():
    return xp


# ---------------------------------------------------------------------------
# Simulation scale
# ---------------------------------------------------------------------------
@dataclass
class ScaleConfig:
    agent_scale: int = 100            # 1 agent = N real people
    base_population: int = 125_700_000
    n_agents: int = 0                 # computed at init
    n_years: int = 50                 # 2025 -> 2075
    steps_per_year: int = 4           # quarterly ticks
    start_year: int = 2025
    random_seed: int = 42

    def __post_init__(self):
        self.n_agents = self.base_population // self.agent_scale


# ---------------------------------------------------------------------------
# Geographic tiers
# ---------------------------------------------------------------------------
TIER_TOKYO = 0
TIER_CORE = 1
TIER_PERIPHERY = 2

TIER_NAMES = {TIER_TOKYO: "Tokyo Metro", TIER_CORE: "Core City", TIER_PERIPHERY: "Periphery"}


@dataclass
class GeographyConfig:
    n_tokyo_wards: int = 23
    n_core_cities: int = 88
    n_periphery_districts: int = 1625
    n_locations: int = 0

    tokyo_pop_share: float = 0.076     # ~9.7M (23 wards only) / 125.7M
    core_pop_share: float = 0.360      # ~45M in core/designated cities
    periphery_pop_share: float = 0.564 # ~71M rest

    tokyo_capacity_factor: float = 1.3
    core_capacity_factor: float = 1.5
    periphery_capacity_factor: float = 3.0

    base_adjacency_radius_km: float = 50.0
    inter_tier_connection_prob: float = 0.05
    intra_tier_connection_prob: float = 0.15

    use_real_municipalities: bool = True  # use japan_municipalities.py data

    def __post_init__(self):
        if self.use_real_municipalities:
            from .japan_municipalities import MUNICIPALITIES
            self.n_tokyo_wards = sum(1 for m in MUNICIPALITIES if m["tier"] == 0)
            self.n_core_cities = sum(1 for m in MUNICIPALITIES if m["tier"] == 1)
            self.n_periphery_districts = sum(1 for m in MUNICIPALITIES if m["tier"] == 2)
        self.n_locations = (
            self.n_tokyo_wards + self.n_core_cities + self.n_periphery_districts
        )


# ---------------------------------------------------------------------------
# Demographic parameters (Japan 2024 baseline)
# ---------------------------------------------------------------------------
@dataclass
class DemographyConfig:
    base_tfr: float = 1.20
    tokyo_tfr: float = 1.04
    core_tfr: float = 1.30
    periphery_tfr: float = 1.45

    reproductive_age_min: int = 20
    reproductive_age_max: int = 39

    marriage_rate_base: float = 0.045      # annual prob for eligible singles
    divorce_rate_base: float = 0.0017
    max_children: int = 4

    male_life_expectancy: float = 81.5
    female_life_expectancy: float = 87.6

    infant_mortality_rate: float = 0.0018

    age_bins: int = 101  # 0..100

    population_age_shares: Dict[str, float] = field(default_factory=lambda: {
        "0-14": 0.116,
        "15-19": 0.042,
        "20-24": 0.046,
        "25-29": 0.050,
        "30-34": 0.052,
        "35-39": 0.056,
        "40-44": 0.060,
        "45-49": 0.068,
        "50-54": 0.072,
        "55-59": 0.065,
        "60-64": 0.060,
        "65-69": 0.068,
        "70-74": 0.080,
        "75-79": 0.065,
        "80-84": 0.048,
        "85+": 0.052,
    })

    sex_ratio_at_birth: float = 1.05  # M:F


# ---------------------------------------------------------------------------
# Economic parameters
# ---------------------------------------------------------------------------
@dataclass
class EconomyConfig:
    tokyo_wage_premium: float = 0.25       # 25% above national median
    core_wage_level: float = 1.0           # normalized
    periphery_wage_discount: float = 0.15  # 15% below national median

    tokyo_rent_index: float = 2.5          # relative to national median
    core_rent_index: float = 1.0
    periphery_rent_index: float = 0.45

    tokyo_transport_cost: float = 0.08     # share of income
    core_transport_cost: float = 0.06
    periphery_transport_cost: float = 0.12 # car-dependent

    national_median_income_jpy: float = 4_580_000  # annual
    childcare_cost_ratio: float = 0.15              # share of income per child
    child_insurance_surcharge_monthly: float = 500  # JPY/month

    vacancy_rate_national: float = 0.136   # 13.6%
    vacancy_rate_tokyo: float = 0.028
    vacancy_rate_core: float = 0.09
    vacancy_rate_periphery: float = 0.22

    ice_act_tax_credit: float = 0.07       # 7% credit for HQ relocation
    ice_act_investment_threshold: float = 3_500_000_000  # ¥3.5B


# ---------------------------------------------------------------------------
# Utility function weights
# ---------------------------------------------------------------------------
@dataclass
class UtilityWeights:
    w_prestige: float = 0.22          # w1: career prestige pull
    w_convenience: float = 0.25       # w2: lifestyle convenience
    w_anomie: float = 0.20            # w3: social isolation penalty
    w_financial_friction: float = 0.33 # w4: cost-of-living drag (high: Tokyo is expensive)

    noise_std: float = 0.05           # idiosyncratic preference noise

    # Subcomponent weights within each factor
    prestige_career_weight: float = 0.5
    prestige_brand_weight: float = 0.3
    prestige_network_weight: float = 0.2

    convenience_digital_weight: float = 0.3
    convenience_healthcare_weight: float = 0.25
    convenience_childcare_weight: float = 0.25
    convenience_amenity_weight: float = 0.2

    anomie_isolation_weight: float = 0.4
    anomie_cultural_gap_weight: float = 0.35
    anomie_community_loss_weight: float = 0.25


# ---------------------------------------------------------------------------
# Behavioral / sociological parameters
# ---------------------------------------------------------------------------
@dataclass
class BehaviorConfig:
    cultural_exit_probability: float = 0.268  # 26.8% flight rate (women)
    singlehood_inertia_threshold: float = 0.65
    migration_cooldown_years: int = 3
    social_influence_radius: int = 5          # network hops
    social_conformity_weight: float = 0.15

    fertility_decision_threshold: float = 0.55
    marriage_utility_threshold: float = 0.50

    age_mobility_peak: int = 27
    age_mobility_decay: float = 0.04          # logistic decay rate

    education_premium: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,    # no degree
        1: 0.15,   # high school
        2: 0.40,   # university
        3: 0.65,   # graduate
    })

    information_decay: float = 0.85  # how fast agents forget past utility

    # --- Enhanced migration ---
    migration_top_k: int = 5            # evaluate K best candidates, not just 1
    status_quo_bias: float = 0.22       # utility bonus for staying put (calibrated to ~3% annual migration)
    age_moving_cost: float = 0.005      # per year of age above 30
    child_moving_cost: float = 0.08     # per child
    chain_migration_weight: float = 0.10  # bonus for having peers at destination

    # --- Education pipeline ---
    university_enrollment_rate: float = 0.58  # 58% of 18yo enroll
    university_return_rate: float = 0.35      # 35% return to home region after graduation
    university_city_boost: float = 0.15       # prestige boost for university cities

    # --- Housing market ---
    rent_demand_elasticity: float = 0.15   # how much rent rises per 10% overcrowding
    rent_vacancy_decay: float = 0.05       # rent drops with vacancy
    rent_adjustment_speed: float = 0.25    # quarterly adjustment fraction

    # --- "Tokyo Black Hole" paper mechanics ---
    # 1. Regional Cannibalism: stepping-stone migration
    stepping_stone_core_weight: float = 1.50  # Tier 2→1 weighted 50% higher than Tier 2→0
    softmax_temp_by_tier: Dict[int, float] = field(default_factory=lambda: {
        0: 3.0,    # Tokyo: low temp = more deterministic (settled)
        1: 7.0,    # Core: high temp = more "jumpy" (career anxiety escalation)
        2: 4.0,    # Periphery: moderate (limited options anyway)
    })

    # 2. Career Anxiety: dynamic prestige weight
    career_anxiety_rate: float = 0.05    # w1 increases by this per year in Tier 1
    career_anxiety_max_boost: float = 0.25  # caps at w1 + 0.25

    # 3. School Closure trigger
    school_closure_pop_threshold: float = 0.30  # pop < 30% capacity → school closure
    school_closure_anomie_multiplier: float = 2.0  # w3 doubles for parents
    school_closure_youth_ratio_trigger: float = 0.15  # youth/adult < 15% → trigger

    # 4. Circular Tax refinements
    disposable_income_threshold: float = 2_500_000  # JPY: below this, fertility_intent → 0
    singles_tax_perception_multiplier: float = 2.0  # singles feel 2x the tax burden

    # 5. Social Friction Index (tech-removes-glue paradox)
    tech_anomie_coupling: float = 0.12  # high-tech convenience INCREASES anomie
    akiya_maintenance_cost_per_vacancy: float = 15_000  # JPY/yr per vacant unit equivalent

    # 6. Rent-Prestige coupling in Tokyo
    tokyo_rent_prestige_coupling: float = 0.15  # rent rises with prestige in Tokyo

    # 7. Utility memory: faster decay for prestige-driven events
    prestige_memory_decay: float = 0.70  # faster decay for prestige component (vs 0.85 general)


# ---------------------------------------------------------------------------
# Policy levers (can be toggled per scenario)
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    hq_relocation_active: bool = True
    hq_relocation_prestige_boost: float = 0.15
    hq_relocation_target_cities: int = 30

    remote_work_penetration: float = 0.30      # share of workforce
    remote_work_annual_growth: float = 0.02    # +2% per year

    childcare_subsidy_ratio: float = 0.50      # govt covers 50%
    childcare_expansion_rate: float = 0.03     # annual capacity growth

    housing_subsidy_periphery: float = 0.20    # 20% cost reduction
    housing_subsidy_core: float = 0.10

    medical_dx_rollout_rate: float = 0.05      # annual improvement
    level4_pod_deployment_rate: float = 0.04   # annual growth

    regional_university_investment: float = 0.02

    ice_act_tax_credit: float = 0.07       # 7% credit for HQ relocation
    ice_act_investment_threshold: float = 3_500_000_000  # ¥3.5B

    circular_tax_per_child_monthly: float = 500  # JPY

    # --- New policy levers ---
    university_decentralization: bool = False     # move top universities to regions
    n_regional_universities: int = 15             # how many to relocate
    university_prestige_boost: float = 0.12       # prestige added to receiving cities

    immigration_active: bool = False
    immigration_annual: int = 200_000              # net annual immigrants
    immigration_age_mean: float = 28.0
    immigration_age_std: float = 6.0
    immigration_tier_prefs: Dict[int, float] = field(default_factory=lambda: {
        0: 0.45,  # 45% go to Tokyo
        1: 0.40,  # 40% to Core Cities
        2: 0.15,  # 15% to periphery
    })

    enterprise_zones_active: bool = False
    enterprise_zone_tax_break: float = 0.15       # income boost for zone residents
    n_enterprise_zones: int = 20                   # number of periphery zones

    shinkansen_expansion_active: bool = False
    shinkansen_new_routes: List = field(default_factory=lambda: [
        # (from_city_approx_lon, from_city_approx_lat, to_lon, to_lat, year)
        (136.62, 36.56, 135.50, 34.69, 2030),   # Hokuriku extension to Osaka
        (130.42, 33.59, 129.88, 32.75, 2035),   # Nishi-Kyushu to Nagasaki
    ])
    shinkansen_convenience_boost: float = 0.12

    regional_wage_multiplier: float = 1.0  # multiplier for non-Tokyo wages (1.0 = no change)


# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    geography: GeographyConfig = field(default_factory=GeographyConfig)
    demography: DemographyConfig = field(default_factory=DemographyConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    weights: UtilityWeights = field(default_factory=UtilityWeights)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def summary(self) -> str:
        return (
            f"Japan Exodus Simulation\n"
            f"  Agents: {self.scale.n_agents:,} (1:{self.scale.agent_scale})\n"
            f"  Locations: {self.geography.n_locations}\n"
            f"  Duration: {self.scale.n_years} years ({self.scale.n_years * self.scale.steps_per_year} steps)\n"
            f"  GPU: {'CuPy + CUDA' if GPU_AVAILABLE else 'CPU only (NumPy)'}\n"
        )


DEFAULT_CONFIG = SimulationConfig()
