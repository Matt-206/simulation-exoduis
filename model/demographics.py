"""
Demographic engine: mortality tables, fertility schedules, marriage/divorce
transition matrices -- all calibrated to Japanese vital statistics.

Uses Gompertz-Makeham mortality law fitted to Japanese life tables.
Coale-Trussell fertility model for age-specific fertility rates.
"""

import numpy as np
from numba import njit, prange
from .config import DemographyConfig, GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp


# ---------------------------------------------------------------------------
# Gompertz-Makeham mortality: mu(x) = alpha * exp(beta * x) + lambda
# Fitted to Japanese Abridged Life Tables (2023)
# ---------------------------------------------------------------------------
GOMPERTZ_PARAMS = {
    "male":   {"alpha": 0.0000115, "beta": 0.1005, "lam": 0.00028},
    "female": {"alpha": 0.0000033, "beta": 0.1085, "lam": 0.00015},
}


def build_mortality_table(config: DemographyConfig) -> dict:
    """Build annual mortality probabilities q(x) for ages 0..100, by sex."""
    tables = {}
    for sex, label in [(0, "male"), (1, "female")]:
        p = GOMPERTZ_PARAMS[label]
        ages = np.arange(config.age_bins, dtype=np.float64)
        mu = p["alpha"] * np.exp(p["beta"] * ages) + p["lam"]
        mu[0] = config.infant_mortality_rate
        qx = 1.0 - np.exp(-mu)
        qx = np.clip(qx, 0.0, 1.0)
        qx[100] = 1.0  # force death at 100
        tables[sex] = qx.astype(np.float64)
    return tables


# ---------------------------------------------------------------------------
# Coale-Trussell model: f(a) = n(a) * M * exp(m * v(a))
# n(a) = natural fertility schedule, v(a) = fertility control deviations
# Calibrated so that sum gives target TFR
# ---------------------------------------------------------------------------
# Standard Coale-Trussell n(a) and v(a) for 5-year groups
CT_AGES = np.array([17, 22, 27, 32, 37, 42, 47], dtype=np.float64)
CT_NATURAL = np.array([0.411, 0.460, 0.431, 0.395, 0.322, 0.167, 0.024], dtype=np.float64)
CT_V = np.array([0.000, -0.316, -0.814, -1.048, -1.404, -1.670, -1.860], dtype=np.float64)


def build_fertility_schedule(target_tfr: float, m_param: float = 1.2) -> np.ndarray:
    """
    Build single-year age-specific fertility rates (ASFR) for ages 15..49.
    Returns array of shape (101,) with non-zero entries only at 15-49.
    """
    raw = CT_NATURAL * np.exp(m_param * CT_V)
    raw_tfr = np.sum(raw) * 5.0  # 5-year groups
    scale = target_tfr / raw_tfr if raw_tfr > 0 else 0.0

    asfr_full = np.zeros(101, dtype=np.float64)
    for i, center_age in enumerate(CT_AGES.astype(int)):
        for offset in range(-2, 3):
            age = center_age + offset
            if 15 <= age <= 49:
                asfr_full[age] = raw[i] * scale

    return asfr_full


def build_all_fertility_schedules(config: DemographyConfig) -> dict:
    """Build ASFR arrays for each tier."""
    return {
        0: build_fertility_schedule(config.tokyo_tfr),
        1: build_fertility_schedule(config.core_tfr),
        2: build_fertility_schedule(config.periphery_tfr),
    }


# ---------------------------------------------------------------------------
# Marriage model: age-dependent hazard with logistic shape
# Calibrated to Japan's mean first marriage age (~30.7M, ~29.7F)
# ---------------------------------------------------------------------------
def build_marriage_hazard(config: DemographyConfig) -> dict:
    """Annual marriage probability by age and sex for never-married."""
    hazards = {}
    for sex, peak_age, width in [(0, 30.7, 6.0), (1, 29.7, 5.5)]:
        ages = np.arange(101, dtype=np.float64)
        logistic = 1.0 / (1.0 + np.exp(-(ages - peak_age + 3) / width))
        decline = np.exp(-0.08 * np.maximum(ages - peak_age - 5, 0))
        h = config.marriage_rate_base * logistic * decline
        h[:18] = 0.0
        h[60:] = 0.001
        hazards[sex] = h.astype(np.float64)
    return hazards


# ---------------------------------------------------------------------------
# Numba-accelerated demographic transition kernels
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def apply_mortality_kernel(
    alive: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    mortality_male: np.ndarray,
    mortality_female: np.ndarray,
    rng_vals: np.ndarray,
) -> np.ndarray:
    """Kill agents probabilistically based on age/sex mortality tables."""
    n = alive.shape[0]
    deaths = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        if not alive[i]:
            continue
        age = min(ages[i], 100)
        if sexes[i] == 0:
            qx = mortality_male[age]
        else:
            qx = mortality_female[age]
        if rng_vals[i] < qx:
            alive[i] = False
            deaths[i] = 1
    return deaths


@njit(parallel=True, cache=True)
def apply_fertility_kernel(
    alive: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    marital: np.ndarray,
    n_children: np.ndarray,
    location_tiers: np.ndarray,
    agent_locations: np.ndarray,
    asfr_tokyo: np.ndarray,
    asfr_core: np.ndarray,
    asfr_periphery: np.ndarray,
    fertility_threshold: np.ndarray,
    max_children: int,
    rng_vals: np.ndarray,
) -> np.ndarray:
    """Determine which female agents give birth this step."""
    n = alive.shape[0]
    births = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        if not alive[i]:
            continue
        if sexes[i] != 1:  # female only
            continue
        age = ages[i]
        if age < 15 or age > 49:
            continue
        if n_children[i] >= max_children:
            continue

        loc = agent_locations[i]
        tier = location_tiers[loc]
        if tier == 0:
            rate = asfr_tokyo[age]
        elif tier == 1:
            rate = asfr_core[age]
        else:
            rate = asfr_periphery[age]

        # Marriage bonus: married women 2.5x more likely
        if marital[i] == 1:
            rate *= 2.5
        else:
            rate *= 0.3

        # Fertility intention as modulator AROUND 1.0 (not suppressant)
        # Intent > 0.5 boosts, < 0.5 suppresses, relative to calibrated ASFR
        intent_mod = 0.5 + fertility_threshold[i]
        rate *= intent_mod

        if rng_vals[i] < rate:
            births[i] = 1
            n_children[i] += 1

    return births


@njit(parallel=True, cache=True)
def apply_marriage_kernel(
    alive: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    marital: np.ndarray,
    hazard_male: np.ndarray,
    hazard_female: np.ndarray,
    rng_vals: np.ndarray,
) -> np.ndarray:
    """Transition single agents to married state."""
    n = alive.shape[0]
    new_marriages = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        if not alive[i]:
            continue
        if marital[i] != 0:
            continue
        age = min(ages[i], 100)
        if sexes[i] == 0:
            h = hazard_male[age]
        else:
            h = hazard_female[age]
        if rng_vals[i] < h:
            marital[i] = 1
            new_marriages[i] = 1
    return new_marriages


@njit(parallel=True, cache=True)
def apply_aging_kernel(alive: np.ndarray, ages: np.ndarray) -> None:
    """Increment age for all living agents by 1 year."""
    n = alive.shape[0]
    for i in prange(n):
        if alive[i]:
            ages[i] += 1


class DemographicEngine:
    """
    Manages all demographic processes: aging, mortality, fertility,
    marriage, and divorce. Operates on vectorized agent arrays.
    """

    def __init__(self, config: DemographyConfig):
        self.config = config
        self.mortality_tables = build_mortality_table(config)
        self.fertility_schedules = build_all_fertility_schedules(config)
        self.marriage_hazards = build_marriage_hazard(config)

    def get_mortality_arrays(self):
        return self.mortality_tables[0], self.mortality_tables[1]

    def get_fertility_arrays(self):
        return (
            self.fertility_schedules[0],
            self.fertility_schedules[1],
            self.fertility_schedules[2],
        )

    def get_marriage_arrays(self):
        return self.marriage_hazards[0], self.marriage_hazards[1]

    def validate(self):
        """Sanity checks on demographic tables."""
        for sex in [0, 1]:
            qx = self.mortality_tables[sex]
            assert qx.shape == (101,), f"Mortality table shape mismatch for sex={sex}"
            assert 0 <= qx[0] <= 0.01, f"Infant mortality out of range"
            assert qx[100] == 1.0, f"Must die at age 100"

        for tier in [0, 1, 2]:
            asfr = self.fertility_schedules[tier]
            tfr = np.sum(asfr)
            assert 0.5 < tfr < 5.0, f"TFR={tfr} out of plausible range for tier {tier}"

        return True
