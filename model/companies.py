"""
Company Macro-Agents: Nikkei 225 HQ location logic.

Companies are modeled as macro-agents that respond to fiscal incentives.
When the ICE Act tax credit makes Core City relocation cheaper than
staying in Tokyo, a probability-weighted fraction of HQs migrates.
This drives the endogenous Prestige gradient.

Sources:
  - Nikkei 225 composition (2024): HQ distribution by prefecture
  - Establishment and Enterprise Census (MIC): high-tier job distribution
  - POI density proxies: Economic Census establishment counts per km²
"""

import numpy as np
from .config import TIER_TOKYO, TIER_CORE, TIER_PERIPHERY

# ───────────────────────────────────────────────────────────────────────
# NIKKEI 225 HQ DISTRIBUTION (2024, by prefecture)
# Source: Nikkei Inc. constituent list, cross-referenced with EDINET
# ───────────────────────────────────────────────────────────────────────
NIKKEI225_HQ_BY_PREFECTURE = {
    "Tokyo": 131, "Osaka": 32, "Aichi": 14, "Kyoto": 6,
    "Hyogo": 5, "Kanagawa": 4, "Saitama": 3, "Fukuoka": 3,
    "Hiroshima": 2, "Shizuoka": 2, "Chiba": 2, "Niigata": 1,
    "Toyama": 1, "Ishikawa": 1, "Nagano": 1, "Mie": 1,
    "Shiga": 1, "Okayama": 1, "Ehime": 1, "Tokushima": 1,
    "Yamaguchi": 1, "Miyagi": 1, "Hokkaido": 1,
}

# ───────────────────────────────────────────────────────────────────────
# HIGH-TIER JOB SHARES (exec/management roles per total employment)
# Source: 2020 Employment Status Survey (MIC), management positions
# ───────────────────────────────────────────────────────────────────────
HIGH_TIER_JOB_SHARE = {
    "Tokyo": 0.085, "Osaka": 0.058, "Aichi": 0.048, "Kanagawa": 0.052,
    "Kyoto": 0.044, "Hyogo": 0.040, "Fukuoka": 0.042, "Saitama": 0.038,
    "Chiba": 0.036, "Miyagi": 0.035, "Hiroshima": 0.034, "Hokkaido": 0.030,
    "Shizuoka": 0.032, "Niigata": 0.028, "Nagano": 0.027, "Ibaraki": 0.029,
    "Tochigi": 0.026, "Gunma": 0.025, "Toyama": 0.030, "Ishikawa": 0.029,
    "Fukui": 0.027, "Yamanashi": 0.024, "Gifu": 0.026, "Mie": 0.028,
    "Shiga": 0.030, "Nara": 0.025, "Wakayama": 0.023,
    "Tottori": 0.022, "Shimane": 0.022, "Okayama": 0.030, "Yamaguchi": 0.025,
    "Tokushima": 0.024, "Kagawa": 0.026, "Ehime": 0.025, "Kochi": 0.022,
    "Saga": 0.023, "Nagasaki": 0.024, "Kumamoto": 0.026, "Oita": 0.025,
    "Miyazaki": 0.022, "Kagoshima": 0.024, "Okinawa": 0.025,
    "Aomori": 0.022, "Iwate": 0.023, "Akita": 0.021, "Yamagata": 0.023,
    "Fukushima": 0.025,
}
_DEFAULT_HIGH_TIER = 0.020

# ───────────────────────────────────────────────────────────────────────
# POI DENSITY PROXY (establishments per km², for Convenience init)
# Source: 2021 Economic Census, establishment counts / prefecture area
# ───────────────────────────────────────────────────────────────────────
POI_DENSITY_INDEX = {
    "Tokyo": 1.00, "Osaka": 0.78, "Kanagawa": 0.55, "Aichi": 0.45,
    "Saitama": 0.40, "Chiba": 0.38, "Kyoto": 0.42, "Hyogo": 0.38,
    "Fukuoka": 0.40, "Hiroshima": 0.32, "Miyagi": 0.30, "Hokkaido": 0.18,
    "Shizuoka": 0.25, "Ibaraki": 0.22, "Tochigi": 0.20, "Gunma": 0.19,
    "Niigata": 0.15, "Nagano": 0.14, "Toyama": 0.18, "Ishikawa": 0.20,
    "Fukui": 0.16, "Yamanashi": 0.15, "Gifu": 0.18, "Mie": 0.17,
    "Shiga": 0.22, "Nara": 0.20, "Wakayama": 0.14,
    "Tottori": 0.10, "Shimane": 0.09, "Okayama": 0.22, "Yamaguchi": 0.14,
    "Tokushima": 0.12, "Kagawa": 0.18, "Ehime": 0.14, "Kochi": 0.09,
    "Saga": 0.14, "Nagasaki": 0.15, "Kumamoto": 0.16, "Oita": 0.14,
    "Miyazaki": 0.11, "Kagoshima": 0.13, "Okinawa": 0.22,
    "Aomori": 0.11, "Iwate": 0.10, "Akita": 0.09, "Yamagata": 0.11,
    "Fukushima": 0.12,
}
_DEFAULT_POI = 0.08


class CompanyPool:
    """
    Macro-agent pool representing major corporate HQs.

    Each "company" has a location. Quarterly, companies evaluate whether
    the ICE Act tax credit makes relocation to a Core City net-positive.
    When they move, the destination gains HQs and high-tier jobs.
    """

    def __init__(self, n_locations: int, prefectures: np.ndarray, rng: np.random.Generator):
        self.rng = rng
        self.n_locations = n_locations
        self.prefectures = prefectures

        self.hq_location = []
        self.hq_size = []  # 0=small, 1=medium, 2=large (Nikkei 225)

        self._distribute_initial_hqs(prefectures)

        self.total_hqs = len(self.hq_location)
        self.hq_location = np.array(self.hq_location, dtype=np.int32)
        self.hq_size = np.array(self.hq_size, dtype=np.int32)
        self.relocation_cooldown = np.zeros(self.total_hqs, dtype=np.int32)

    def _distribute_initial_hqs(self, prefectures):
        """Place Nikkei 225 HQs at correct prefecture locations, plus ~3000 smaller HQs."""
        rng = self.rng

        pref_to_locs = {}
        for i, pref in enumerate(prefectures):
            pref_to_locs.setdefault(pref, []).append(i)

        for pref, count in NIKKEI225_HQ_BY_PREFECTURE.items():
            locs = pref_to_locs.get(pref, [])
            if not locs:
                continue
            for _ in range(count):
                loc = rng.choice(locs)
                self.hq_location.append(loc)
                self.hq_size.append(2)

        for pref, locs in pref_to_locs.items():
            n_medium = max(1, int(len(locs) * 0.8))
            for _ in range(n_medium):
                loc = rng.choice(locs)
                self.hq_location.append(loc)
                self.hq_size.append(1)

        for pref, locs in pref_to_locs.items():
            n_small = max(1, int(len(locs) * 0.3))
            for _ in range(n_small):
                loc = rng.choice(locs)
                self.hq_location.append(loc)
                self.hq_size.append(0)

    def get_hq_counts(self) -> np.ndarray:
        """Return per-location HQ count."""
        counts = np.zeros(self.n_locations, dtype=np.int32)
        for i in range(self.total_hqs):
            counts[self.hq_location[i]] += 1
        return counts

    def get_weighted_hq_counts(self) -> np.ndarray:
        """Nikkei 225 HQs count 10x, medium 3x, small 1x."""
        counts = np.zeros(self.n_locations, dtype=np.float64)
        weights = {0: 1.0, 1: 3.0, 2: 10.0}
        for i in range(self.total_hqs):
            counts[self.hq_location[i]] += weights[self.hq_size[i]]
        return counts

    def step_relocations(
        self,
        loc_tiers: np.ndarray,
        loc_prestige: np.ndarray,
        loc_rent_index: np.ndarray,
        ice_act_credit: float,
        ice_act_active: bool,
        candidate_core_locs: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """
        Each quarter, companies evaluate relocation.
        Cost of staying in Tokyo = rent_premium - prestige_benefit
        Cost of moving to Core = moving_cost - ice_act_credit

        If net benefit of Core > 0 AND random draw < probability, the HQ moves.
        Returns count of relocations this step.
        """
        if not ice_act_active or len(candidate_core_locs) == 0:
            self.relocation_cooldown = np.maximum(self.relocation_cooldown - 1, 0)
            return 0

        relocations = 0

        in_tokyo = loc_tiers[self.hq_location] == TIER_TOKYO
        eligible = in_tokyo & (self.relocation_cooldown <= 0)
        eligible_idx = np.where(eligible)[0]

        if len(eligible_idx) == 0:
            self.relocation_cooldown = np.maximum(self.relocation_cooldown - 1, 0)
            return 0

        tokyo_avg_rent = loc_rent_index[loc_tiers == TIER_TOKYO].mean()

        for ci in eligible_idx:
            size = self.hq_size[ci]

            base_move_prob = {0: 0.008, 1: 0.004, 2: 0.001}[size]

            cost_staying = tokyo_avg_rent * 0.05
            tax_saving = ice_act_credit * (1.0 + 0.5 * size)

            net_incentive = tax_saving - cost_staying * 0.3
            move_prob = base_move_prob * max(0.1, 1.0 + net_incentive)

            if rng.random() < move_prob:
                core_prestige = loc_prestige[candidate_core_locs]
                weights = core_prestige + 0.1
                weights /= weights.sum()
                dest = rng.choice(candidate_core_locs, p=weights)

                self.hq_location[ci] = dest
                self.relocation_cooldown[ci] = 8  # 2-year cooldown
                relocations += 1

        self.relocation_cooldown = np.maximum(self.relocation_cooldown - 1, 0)
        return relocations

    def get_high_tier_job_share(self, prefectures: np.ndarray) -> np.ndarray:
        """Return high-tier job share per location (from real data + HQ boost)."""
        n = len(prefectures)
        shares = np.zeros(n)
        for i, pref in enumerate(prefectures):
            shares[i] = HIGH_TIER_JOB_SHARE.get(pref, _DEFAULT_HIGH_TIER)

        hq_counts = self.get_weighted_hq_counts()
        hq_max = max(hq_counts.max(), 1.0)
        shares += 0.02 * (hq_counts / hq_max)

        return np.clip(shares, 0, 1)

    @staticmethod
    def get_poi_density(prefecture: str) -> float:
        return POI_DENSITY_INDEX.get(prefecture, _DEFAULT_POI)
