"""
Vectorized agent state management.

Instead of individual Python Agent objects (which would be far too slow
for 1M+ agents), all agent state is stored in columnar NumPy arrays.
A thin AgentPool class provides structured access and bulk operations.
"""

import numpy as np
from typing import Dict, Optional
from .config import (
    SimulationConfig, DemographyConfig, GeographyConfig,
    TIER_TOKYO, TIER_CORE, TIER_PERIPHERY,
)
from .prefecture_demographics import get_prefecture_profile, NATIONAL_DEFAULTS


class AgentPool:
    """
    Columnar storage for all agent attributes.

    Attributes (all shape = (max_agents,)):
        alive:            bool    - is agent alive
        age:              int32   - current age in years
        sex:              int8    - 0=male, 1=female
        location:         int32   - index into location array
        income:           float64 - annual income in JPY
        education:        int8    - 0=none, 1=HS, 2=university, 3=graduate
        marital_status:   int8    - 0=single, 1=married, 2=divorced, 3=widowed
        n_children:       int32   - number of children
        cultural_orient:  float64 - 0=traditional, 1=progressive
        migration_cooldown: int32 - years until eligible to move again
        fertility_intent: float64 - computed fertility intention [0,1]
        utility_memory:   float64 - exponential moving average of utility
        partner_id:       int64   - index of spouse (-1 if none)
        birth_year:       int32   - year of birth
        remote_worker:    bool    - works remotely
    """

    def __init__(self, max_agents: int, rng: np.random.Generator):
        from collections import deque
        self.max_agents = max_agents
        self.n_alive = 0
        self.next_id = 0
        self.rng = rng
        self._free_slots = deque()  # recycled indices from dead agents

        self.alive = np.zeros(max_agents, dtype=np.bool_)
        self.age = np.zeros(max_agents, dtype=np.int32)
        self.sex = np.zeros(max_agents, dtype=np.int8)
        self.location = np.zeros(max_agents, dtype=np.int32)
        self.income = np.zeros(max_agents, dtype=np.float64)
        self.education = np.zeros(max_agents, dtype=np.int8)
        self.marital_status = np.zeros(max_agents, dtype=np.int8)
        self.n_children = np.zeros(max_agents, dtype=np.int32)
        self.cultural_orient = np.zeros(max_agents, dtype=np.float64)
        self.migration_cooldown = np.zeros(max_agents, dtype=np.int32)
        self.fertility_intent = np.zeros(max_agents, dtype=np.float64)
        self.utility_memory = np.zeros(max_agents, dtype=np.float64)
        self.partner_id = np.full(max_agents, -1, dtype=np.int64)
        self.birth_year = np.zeros(max_agents, dtype=np.int32)
        self.remote_worker = np.zeros(max_agents, dtype=np.bool_)
        self.home_location = np.zeros(max_agents, dtype=np.int32)    # where they grew up
        self.in_university = np.zeros(max_agents, dtype=np.bool_)    # currently enrolled
        self.social_links = np.full((max_agents, 5), -1, dtype=np.int64)
        self.years_in_core = np.zeros(max_agents, dtype=np.int32)
        self.fertility_suppressed_quarters = np.zeros(max_agents, dtype=np.int32)
        self.prestige_utility_memory = np.zeros(max_agents, dtype=np.float64)

        # --- New attributes for enhanced mechanics ---
        self.household_id = np.full(max_agents, -1, dtype=np.int64)
        self.has_elderly_parent = np.zeros(max_agents, dtype=np.bool_)
        self.employer_location = np.full(max_agents, -1, dtype=np.int32)  # -1 = same as residence
        self.unemployed = np.zeros(max_agents, dtype=np.bool_)
        self.disaster_displaced = np.zeros(max_agents, dtype=np.bool_)
        self.owns_property = np.zeros(max_agents, dtype=np.bool_)
        self.migration_type = np.zeros(max_agents, dtype=np.int8)  # 0=none,1=u-turn,2=j-turn,3=i-turn
        self.origin_tier = np.full(max_agents, -1, dtype=np.int8)  # tier at birth/entry (for launchpad tracking)

    def initialize_population(
        self,
        config: SimulationConfig,
        location_tiers: np.ndarray,
        location_populations: np.ndarray,
        location_prefectures: np.ndarray = None,
    ):
        """
        Generate the initial synthetic population using real prefecture-level
        demographic calibration data (age, sex, education, marriage, fertility,
        income distributions all vary by geography).
        """
        n = config.scale.n_agents
        if n > self.max_agents:
            raise ValueError(f"n_agents ({n}) exceeds max_agents ({self.max_agents})")

        rng = self.rng
        dem = config.demography
        eco = config.economy

        self.alive[:n] = True
        self.next_id = n
        self.n_alive = n

        # --- Location assignment first (everything else depends on it) ---
        self._assign_locations(n, location_populations)

        # --- Build per-agent prefecture profile lookup ---
        if location_prefectures is not None:
            agent_prefectures = location_prefectures[self.location[:n]]
        else:
            agent_prefectures = np.array(["_national"] * n, dtype=object)

        unique_prefs = np.unique(agent_prefectures)
        pref_profiles = {p: get_prefecture_profile(p) for p in unique_prefs}

        # --- Age distribution (per-prefecture) ---
        self._assign_ages_by_prefecture(n, agent_prefectures, pref_profiles, dem)

        # --- Sex (per-prefecture female ratio, age-adjusted) ---
        self._assign_sex_by_prefecture(n, agent_prefectures, pref_profiles, dem)

        # --- Birth year, home location, origin tier ---
        self.birth_year[:n] = config.scale.start_year - self.age[:n]
        self.home_location[:n] = self.location[:n].copy()
        self.origin_tier[:n] = location_tiers[self.location[:n]]

        # --- Social links: 5 random peers at same location (vectorized) ---
        self._assign_social_links(n)

        # --- Education (per-prefecture attainment rates) ---
        self._assign_education_by_prefecture(n, agent_prefectures, pref_profiles)

        # --- Income (per-prefecture mean income, education, age, sex) ---
        self._assign_income_by_prefecture(n, agent_prefectures, pref_profiles, config, location_tiers)

        # --- Marital status (per-prefecture marriage patterns) ---
        self._assign_marital_status_by_prefecture(n, agent_prefectures, pref_profiles)

        # --- Children (per-prefecture TFR-calibrated) ---
        self._assign_children_by_prefecture(n, agent_prefectures, pref_profiles, dem)

        # --- Cultural orientation ---
        age_norm = self.age[:n].astype(np.float64) / 80.0
        tier_at_loc = location_tiers[self.location[:n]]
        urban_factor = np.where(tier_at_loc == TIER_TOKYO, 0.3,
                       np.where(tier_at_loc == TIER_CORE, 0.15, 0.0))
        self.cultural_orient[:n] = np.clip(
            (1.0 - age_norm) * 0.6 + urban_factor + rng.normal(0, 0.15, n),
            0, 1,
        )

        # --- Remote work (younger, educated, urban) ---
        remote_base = config.policy.remote_work_penetration
        edu_bonus = self.education[:n].astype(np.float64) * 0.08
        age_penalty = np.where(self.age[:n] > 55, -0.1, 0.0)
        remote_prob = np.clip(remote_base + edu_bonus + age_penalty, 0, 0.8)
        self.remote_worker[:n] = rng.random(n) < remote_prob

        # --- Initialize utility memory to 0.5 ---
        self.utility_memory[:n] = 0.5

        # --- Elderly parent obligation (agents 30-60 with living parents ~70+) ---
        has_elderly = (self.age[:n] >= 30) & (self.age[:n] <= 60)
        parent_alive_prob = np.where(self.age[:n] < 40, 0.85,
                           np.where(self.age[:n] < 50, 0.65, 0.35))
        self.has_elderly_parent[:n] = has_elderly & (rng.random(n) < parent_alive_prob)

        # --- Property ownership (age, tier, income dependent) ---
        age_own_prob = np.where(self.age[:n] < 30, 0.05,
                      np.where(self.age[:n] < 40, 0.25,
                      np.where(self.age[:n] < 50, 0.45,
                      np.where(self.age[:n] < 65, 0.60, 0.70))))
        tier_own_adj = np.where(tier_at_loc == TIER_TOKYO, 0.6,
                      np.where(tier_at_loc == TIER_CORE, 0.8, 1.1))
        self.owns_property[:n] = rng.random(n) < (age_own_prob * tier_own_adj)

        # --- Commuting / employer location ---
        self._assign_commuting(n, location_tiers, location_prefectures)

        # --- Household formation (pair married couples) ---
        self._form_initial_households(n)

        # --- Unemployment (age/location dependent) ---
        unemp_base = np.where(tier_at_loc == TIER_TOKYO, 0.028,
                    np.where(tier_at_loc == TIER_CORE, 0.032, 0.038))
        young_penalty = np.where((self.age[:n] >= 18) & (self.age[:n] < 25), 0.04, 0.0)
        self.unemployed[:n] = (self.age[:n] >= 18) & (rng.random(n) < (unemp_base + young_penalty))

    # ------------------------------------------------------------------
    # Prefecture-calibrated initialization helpers (all vectorized)
    # ------------------------------------------------------------------

    _AGE_GROUP_MAP = {
        "0-14": (0, 14), "15-19": (15, 19), "20-24": (20, 24),
        "25-29": (25, 29), "30-34": (30, 34), "35-39": (35, 39),
        "40-44": (40, 44), "45-49": (45, 49), "50-54": (50, 54),
        "55-59": (55, 59), "60-64": (60, 64), "65-69": (65, 69),
        "70-74": (70, 74), "75-79": (75, 79), "80-84": (80, 84),
        "85+": (85, 100),
    }

    @staticmethod
    def _age_shares_to_probs(age_shares: dict) -> np.ndarray:
        """Convert age-group shares dict into single-year probabilities (101,)."""
        probs = np.zeros(101)
        for group, share in age_shares.items():
            lo, hi = AgentPool._AGE_GROUP_MAP[group]
            n_years = hi - lo + 1
            probs[lo:hi + 1] = share / n_years
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs

    def _assign_locations(self, n: int, location_populations: np.ndarray):
        """Assign agents to locations proportional to real population."""
        rng = self.rng
        total_pop = location_populations.sum()
        if total_pop == 0:
            self.location[:n] = rng.integers(0, len(location_populations), size=n)
            return
        loc_probs = location_populations.astype(np.float64) / total_pop
        self.location[:n] = rng.choice(
            len(location_populations), size=n, p=loc_probs
        ).astype(np.int32)

    def _assign_ages_by_prefecture(self, n, agent_prefectures, pref_profiles, dem):
        """Age drawn from the agent's prefecture age pyramid."""
        rng = self.rng
        ages = np.empty(n, dtype=np.int32)

        pref_age_probs = {}
        for pref, profile in pref_profiles.items():
            pref_age_probs[pref] = self._age_shares_to_probs(profile["age_distribution"])

        for pref in pref_age_probs:
            mask = agent_prefectures == pref
            cnt = mask.sum()
            if cnt == 0:
                continue
            ages[mask] = rng.choice(101, size=cnt, p=pref_age_probs[pref])

        self.age[:n] = ages

    def _assign_sex_by_prefecture(self, n, agent_prefectures, pref_profiles, dem):
        """Sex ratio from prefecture data, with age-adjusted female longevity."""
        rng = self.rng
        ages = self.age[:n]
        female_base = np.full(n, 0.514)

        for pref, profile in pref_profiles.items():
            mask = agent_prefectures == pref
            female_base[mask] = profile["pct_female"]

        female_prob = np.where(ages > 75, np.minimum(female_base + 0.08, 0.65),
                     np.where(ages > 65, female_base + 0.04,
                     np.where((ages >= 20) & (ages <= 39), female_base - 0.01,
                              female_base)))

        self.sex[:n] = (rng.random(n) < female_prob).astype(np.int8)

    def _assign_social_links(self, n):
        """5 random peers at same location (vectorized)."""
        rng = self.rng
        locs = self.location[:n]
        sort_idx = np.argsort(locs, kind='mergesort')
        sorted_locs = locs[sort_idx]
        max_loc = sorted_locs[-1] if len(sorted_locs) > 0 else 0
        boundaries = np.searchsorted(sorted_locs, np.arange(max_loc + 2))
        for loc_id in range(len(boundaries) - 1):
            start, end = boundaries[loc_id], boundaries[loc_id + 1]
            count = end - start
            if count < 7:
                continue
            group = sort_idx[start:end]
            rand_offsets = rng.integers(1, count, size=(count, 5))
            base_idx = np.arange(count)[:, None]
            peer_local = (base_idx + rand_offsets) % count
            self.social_links[group] = group[peer_local]

    def _assign_education_by_prefecture(self, n, agent_prefectures, pref_profiles):
        """Education attainment calibrated to each prefecture's real rates."""
        rng = self.rng
        ages = self.age[:n]
        edu = np.zeros(n, dtype=np.int8)
        p = rng.random(n)

        for pref, profile in pref_profiles.items():
            mask = agent_prefectures == pref
            if not mask.any():
                continue

            uni = profile["university_rate"]
            hs = profile["hs_rate"]
            grad = uni * 0.15
            no_edu = max(0.02, 1.0 - uni - hs - grad)
            hs_thresh = no_edu
            uni_thresh = hs_thresh + hs
            grad_thresh = uni_thresh + uni

            m_ages = ages[mask]
            m_p = p[mask]
            m_edu = np.zeros(mask.sum(), dtype=np.int8)

            child = m_ages < 18
            m_edu[child] = 0

            young_adult = (m_ages >= 18) & (m_ages < 22)
            m_edu[young_adult] = np.where(m_p[young_adult] < 0.05, 0, 1).astype(np.int8)

            adult = m_ages >= 22
            pa = m_p[adult]
            m_edu[adult] = np.select(
                [pa < hs_thresh, pa < uni_thresh, pa < grad_thresh],
                [0, 1, 2],
                default=3,
            ).astype(np.int8)

            edu[mask] = m_edu

        self.education[:n] = edu

    def _assign_income_by_prefecture(self, n, agent_prefectures, pref_profiles, config, location_tiers):
        """Income from prefecture mean, adjusted by education, age, sex."""
        rng = self.rng

        base_income = np.full(n, config.economy.national_median_income_jpy)
        for pref, profile in pref_profiles.items():
            mask = agent_prefectures == pref
            base_income[mask] = profile["mean_income_jpy"]

        edu_mult = np.array([0.6, 0.85, 1.0, 1.35])[self.education[:n]]

        ages = self.age[:n].astype(np.float64)
        age_mult = np.where(
            ages < 22, 0.3,
            np.where(ages > 65, 0.4,
                     0.5 + 0.5 * np.sin(np.pi * (ages - 22) / 56))
        )

        sex_mult = np.where(self.sex[:n] == 1, 0.76, 1.0)
        noise = rng.lognormal(0, 0.20, n)

        self.income[:n] = base_income * edu_mult * age_mult * sex_mult * noise

    def _assign_marital_status_by_prefecture(self, n, agent_prefectures, pref_profiles):
        """Marriage probability calibrated to prefecture-level rates and singlehood."""
        rng = self.rng
        ages = self.age[:n]

        marriage_adj = np.ones(n)
        pct_single_30_34 = np.full(n, 0.44)

        for pref, profile in pref_profiles.items():
            mask = agent_prefectures == pref
            marriage_adj[mask] = profile["marriage_rate_adj"]
            pct_single_30_34[mask] = profile["pct_single_30_34"]

        base_married = np.select(
            [ages < 18, ages < 25, ages < 30, ages < 35, ages < 40, ages < 50, ages < 65],
            [0.0, 0.03, 0.18, 0.42, 0.55, 0.62, 0.68],
            default=0.52,
        )

        single_adj = (1.0 - pct_single_30_34) / (1.0 - 0.44 + 1e-9)
        married_prob = np.clip(base_married * marriage_adj * single_adj, 0, 0.85)

        divorced_prob = np.where(ages > 30, 0.08, 0.02)
        widowed_prob = np.where(ages > 70, 0.15, 0.02)

        p = rng.random(n)
        status = np.zeros(n, dtype=np.int8)
        status[p < married_prob] = 1
        in_divorced = (p >= married_prob) & (p < married_prob + divorced_prob)
        status[in_divorced] = 2
        in_widowed = (p >= married_prob + divorced_prob) & (p < married_prob + divorced_prob + widowed_prob)
        status[in_widowed] = 3

        self.marital_status[:n] = status

    def _assign_children_by_prefecture(self, n, agent_prefectures, pref_profiles, dem):
        """Children count calibrated to each prefecture's TFR."""
        rng = self.rng
        ages = self.age[:n]
        eligible = (self.sex[:n] == 1) & (self.marital_status[:n] != 0)

        tfr_scale = np.ones(n)
        for pref, profile in pref_profiles.items():
            mask = agent_prefectures == pref
            tfr_scale[mask] = profile["tfr"] / 1.20

        base_mean = np.select(
            [ages < 25, ages < 30, ages < 35, ages < 40, ages < 50],
            [0.1, 0.5, 1.0, 1.3, 1.5],
            default=1.7,
        )

        mean_children = base_mean * tfr_scale
        children = rng.poisson(mean_children).astype(np.int32)
        children = np.minimum(children, dem.max_children)
        self.n_children[:n] = np.where(eligible, children, 0)

    def _assign_commuting(self, n, location_tiers, location_prefectures):
        """Assign commuter status for bed-town residents."""
        from .real_data import COMMUTER_PREFECTURES
        rng = self.rng

        self.employer_location[:n] = -1  # default: work where you live

        if location_prefectures is None:
            return

        ages = self.age[:n]
        working_age = (ages >= 22) & (ages < 65) & (~self.in_university[:n])

        for pref, info in COMMUTER_PREFECTURES.items():
            pref_mask = (location_prefectures[self.location[:n]] == pref) & working_age
            commuters = pref_mask & (rng.random(n) < info["commuter_share"])
            if not commuters.any():
                continue
            target_pref = info["commute_to"]
            target_locs = np.where(location_prefectures == target_pref)[0]
            if len(target_locs) == 0:
                continue
            n_commuters = commuters.sum()
            self.employer_location[:n][commuters] = rng.choice(target_locs, size=n_commuters)

    def _form_initial_households(self, n):
        """Pair married agents into households with assortative matching (vectorized by location)."""
        rng = self.rng
        ages = self.age[:n]
        locs = self.location[:n]

        married_males = np.where(
            (self.marital_status[:n] == 1) & (self.sex[:n] == 0) & (ages >= 18)
        )[0]
        married_females = np.where(
            (self.marital_status[:n] == 1) & (self.sex[:n] == 1) & (ages >= 18)
        )[0]

        if len(married_males) == 0 or len(married_females) == 0:
            return

        m_locs = locs[married_males]
        f_locs = locs[married_females]

        m_sort = np.argsort(m_locs, kind='mergesort')
        f_sort = np.argsort(f_locs, kind='mergesort')
        sorted_m = married_males[m_sort]
        sorted_f = married_females[f_sort]
        sorted_m_locs = m_locs[m_sort]
        sorted_f_locs = f_locs[f_sort]

        max_loc = max(sorted_m_locs[-1], sorted_f_locs[-1]) + 2
        m_bounds = np.searchsorted(sorted_m_locs, np.arange(max_loc))
        f_bounds = np.searchsorted(sorted_f_locs, np.arange(max_loc))

        hh_id = 0
        for loc_id in range(max_loc - 1):
            ms = sorted_m[m_bounds[loc_id]:m_bounds[loc_id + 1]]
            fs = sorted_f[f_bounds[loc_id]:f_bounds[loc_id + 1]]
            n_pairs = min(len(ms), len(fs))
            if n_pairs == 0:
                continue

            m_ages = ages[ms[:n_pairs]]
            f_ages = ages[fs]

            rng.shuffle(ms[:n_pairs])

            f_available = np.ones(len(fs), dtype=np.bool_)
            for i in range(n_pairs):
                m_idx = ms[i]
                avail_f = np.where(f_available)[0]
                if len(avail_f) == 0:
                    break
                age_diff = np.abs(ages[fs[avail_f]].astype(np.int32) - int(ages[m_idx]) + 2)
                best_local = avail_f[np.argmin(age_diff)]
                f_idx = fs[best_local]

                self.partner_id[m_idx] = f_idx
                self.partner_id[f_idx] = m_idx
                self.household_id[m_idx] = hh_id
                self.household_id[f_idx] = hh_id
                f_available[best_local] = False
                hh_id += 1

    def _alloc_slot(self) -> int:
        """Get a free slot: recycle a dead agent's index, or append."""
        if self._free_slots:
            return self._free_slots.popleft()
        if self.next_id < self.max_agents:
            idx = self.next_id
            self.next_id += 1
            return idx
        return -1

    def add_newborn(
        self,
        mother_idx: int,
        year: int,
        location_tier: int,
    ) -> int:
        """Add a single newborn agent. Returns new agent index or -1 if full."""
        idx = self._alloc_slot()
        if idx < 0:
            return -1
        self.n_alive += 1

        self.alive[idx] = True
        self.age[idx] = 0
        self.sex[idx] = 1 if self.rng.random() < 0.4878 else 0
        self.location[idx] = self.location[mother_idx]
        self.home_location[idx] = self.location[mother_idx]
        self.income[idx] = 0.0
        self.education[idx] = 0
        self.marital_status[idx] = 0
        self.n_children[idx] = 0
        self.cultural_orient[idx] = self.cultural_orient[mother_idx] * 0.5 + self.rng.normal(0.5, 0.15)
        self.cultural_orient[idx] = np.clip(self.cultural_orient[idx], 0, 1)
        self.migration_cooldown[idx] = 0
        self.fertility_intent[idx] = 0.0
        self.utility_memory[idx] = 0.5
        self.partner_id[idx] = -1
        self.birth_year[idx] = year
        self.remote_worker[idx] = False
        self.in_university[idx] = False
        self.social_links[idx] = -1
        self.years_in_core[idx] = 0
        self.fertility_suppressed_quarters[idx] = 0
        self.prestige_utility_memory[idx] = 0.0
        self.household_id[idx] = self.household_id[mother_idx]
        self.has_elderly_parent[idx] = False
        self.employer_location[idx] = -1
        self.unemployed[idx] = False
        self.disaster_displaced[idx] = False
        self.owns_property[idx] = False
        self.migration_type[idx] = 0
        self.origin_tier[idx] = location_tier

        return idx

    def add_newborns_batch(
        self,
        mother_indices: np.ndarray,
        year: int,
        loc_tiers: np.ndarray = None,
    ) -> int:
        """Vectorized batch newborn creation with dead-slot recycling."""
        n_births = len(mother_indices)
        if n_births == 0:
            return 0

        # Gather available indices: first from free list, then from tail
        recycled = []
        while self._free_slots and len(recycled) < n_births:
            recycled.append(self._free_slots.popleft())

        n_recycled = len(recycled)
        n_remaining = n_births - n_recycled
        n_from_tail = min(n_remaining, self.max_agents - self.next_id)
        actual = n_recycled + n_from_tail

        if actual <= 0:
            # Put back any recycled slots we grabbed
            for s in recycled:
                self._free_slots.appendleft(s)
            return 0

        # Build the index array for all new slots
        if n_from_tail > 0:
            tail_indices = np.arange(self.next_id, self.next_id + n_from_tail, dtype=np.int64)
            self.next_id += n_from_tail
        else:
            tail_indices = np.array([], dtype=np.int64)

        if n_recycled > 0:
            recycled_arr = np.array(recycled[:actual], dtype=np.int64)
            slots = np.concatenate([recycled_arr, tail_indices])
        else:
            slots = tail_indices

        slots = slots[:actual]
        mothers = mother_indices[:actual]

        self.alive[slots] = True
        self.age[slots] = 0
        self.sex[slots] = (self.rng.random(actual) >= 0.4878).astype(np.int8)
        self.location[slots] = self.location[mothers]
        self.home_location[slots] = self.location[mothers]
        self.income[slots] = 0.0
        self.education[slots] = 0
        self.marital_status[slots] = 0
        self.n_children[slots] = 0
        self.cultural_orient[slots] = np.clip(
            self.cultural_orient[mothers] * 0.5 + self.rng.normal(0.5, 0.15, size=actual),
            0, 1,
        )
        self.migration_cooldown[slots] = 0
        self.fertility_intent[slots] = 0.0
        self.utility_memory[slots] = 0.5
        self.partner_id[slots] = -1
        self.birth_year[slots] = year
        self.remote_worker[slots] = False
        self.in_university[slots] = False
        self.social_links[slots] = -1
        self.years_in_core[slots] = 0
        self.fertility_suppressed_quarters[slots] = 0
        self.prestige_utility_memory[slots] = 0.0
        self.household_id[slots] = self.household_id[mothers]
        self.has_elderly_parent[slots] = False
        self.employer_location[slots] = -1
        self.unemployed[slots] = False
        self.disaster_displaced[slots] = False
        self.owns_property[slots] = False
        self.migration_type[slots] = 0
        if loc_tiers is not None:
            self.origin_tier[slots] = loc_tiers[self.location[slots]]
        else:
            self.origin_tier[slots] = self.origin_tier[mothers]

        self.n_alive += actual
        return actual

    def kill_agent(self, idx: int):
        """Mark agent as dead and recycle the slot for future newborns."""
        self.alive[idx] = False
        self.n_alive -= 1
        self._free_slots.append(idx)

        # Widowing the partner
        partner = self.partner_id[idx]
        if partner >= 0 and partner < self.max_agents:
            if self.alive[partner]:
                self.marital_status[partner] = 3
                self.partner_id[partner] = -1
        self.partner_id[idx] = -1

    def get_alive_mask(self) -> np.ndarray:
        return self.alive[:self.next_id]

    def get_reproductive_females_mask(self) -> np.ndarray:
        """Females 20-39 who are alive."""
        mask = (
            self.alive[:self.next_id]
            & (self.sex[:self.next_id] == 1)
            & (self.age[:self.next_id] >= 20)
            & (self.age[:self.next_id] <= 39)
        )
        return mask

    def get_population_by_location(self, n_locations: int) -> np.ndarray:
        """Count living agents per location."""
        alive_locs = self.location[:self.next_id][self.alive[:self.next_id]]
        return np.bincount(alive_locs, minlength=n_locations)

    def get_statistics(self) -> Dict:
        """Compute summary statistics for alive agents."""
        alive = self.alive[:self.next_id]
        n = alive.sum()
        if n == 0:
            return {"n_alive": 0}

        ages = self.age[:self.next_id][alive]
        sexes = self.sex[:self.next_id][alive]
        income = self.income[:self.next_id][alive]

        return {
            "n_alive": int(n),
            "mean_age": float(ages.mean()),
            "median_age": float(np.median(ages)),
            "pct_female": float((sexes == 1).mean()),
            "mean_income": float(income.mean()),
            "pct_married": float((self.marital_status[:self.next_id][alive] == 1).mean()),
            "mean_children": float(self.n_children[:self.next_id][alive].mean()),
            "pct_remote": float(self.remote_worker[:self.next_id][alive].mean()),
        }
