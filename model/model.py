"""
Main simulation model: ExodusModel.

Orchestrates the full simulation loop:
  1. Policy application
  2. Location dynamics update
  3. Utility computation (GPU-accelerated)
  4. Migration decisions
  5. Demographic transitions (aging, mortality, fertility, marriage)
  6. Data collection

Uses Mesa 3.5 for model structure and data collection,
but all heavy computation is vectorized through ComputeEngine.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.spatial import cKDTree

import mesa

from .config import SimulationConfig, TIER_TOKYO, TIER_CORE, TIER_PERIPHERY, GPU_AVAILABLE
from .agents import AgentPool
from .geography import LocationState, build_geography, update_location_dynamics
from .demographics import (
    DemographicEngine,
    apply_mortality_kernel, apply_fertility_kernel,
    apply_marriage_kernel, apply_aging_kernel,
)
from .compute_engine import ComputeEngine
from .policies import PolicyEngine


class ExodusModel(mesa.Model):
    """
    Japan Exodus Spatial Agent-Based Microsimulation.

    Simulates population dynamics across a 3-tier geographic network
    (Tokyo / Core Cities / Periphery) with empirically calibrated
    demographic, economic, and behavioral parameters.
    """

    def __init__(self, config: Optional[SimulationConfig] = None, **kwargs):
        super().__init__(**kwargs)

        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.scale.random_seed)

        self.current_year = self.config.scale.start_year
        self.current_step_in_year = 0
        self.total_steps = 0

        # --- Build geography ---
        self.loc_state, self.network, self.dist_matrix = build_geography(
            self.config.geography, self.config.economy, self.rng,
        )

        # --- Initialize agents ---
        buffer_factor = 2.0  # room for births (dead slots are recycled)
        max_agents = int(self.config.scale.n_agents * buffer_factor)
        self.agents_pool = AgentPool(max_agents, self.rng)
        loc_prefectures = getattr(self.loc_state, '_prefectures', None)
        self.agents_pool.initialize_population(
            self.config,
            self.loc_state.tier,
            self.loc_state.population,
            location_prefectures=loc_prefectures,
        )

        # --- Engines ---
        self.demo_engine = DemographicEngine(self.config.demography)
        self.demo_engine.validate()

        self.compute_engine = ComputeEngine(
            self.config.weights, self.config.behavior,
        )
        self.policy_engine = PolicyEngine(self.config.policy, self.rng)

        # --- Company Macro-Agents ---
        from .companies import CompanyPool
        loc_prefs = getattr(self.loc_state, '_prefectures', np.array(['Tokyo'] * self.config.geography.n_locations))
        self.company_pool = CompanyPool(
            self.config.geography.n_locations, loc_prefs, self.rng,
        )
        # Set initial HQ counts and high-tier job shares from company data
        self.loc_state.hq_count = self.company_pool.get_hq_counts()
        self.loc_state.high_tier_job_share = self.company_pool.get_high_tier_job_share(loc_prefs)

        # --- Spatial index for location queries ---
        coords = np.column_stack([self.loc_state.lon, self.loc_state.lat])
        self.location_kdtree = cKDTree(coords)

        # --- Precompute neighbor lists for migration candidates ---
        self._precompute_migration_candidates()

        # --- Data collection ---
        self.history: List[Dict] = []
        self.migration_flows: List[Dict] = []

        # --- Regional Cannibalism tracker (Paper Section 3.1) ---
        self.cannibalism_tracker = {
            "periphery_to_core": 0,
            "periphery_to_tokyo": 0,
            "core_to_tokyo_from_periphery": 0,  # "Launchpad" moves
            "core_to_tokyo": 0,
        }

        # --- Economic state ---
        self.in_recession = False
        self.recession_quarters_left = 0
        self.recession_income_shock = 0.0

        # --- Disaster state ---
        self.disaster_locations = set()
        self.disaster_recovery = {}

        # --- Prefecture lookup for locations ---
        self._loc_prefectures = getattr(self.loc_state, '_prefectures', None)

        # --- Real data: rent index calibration ---
        self._calibrate_rent_from_real_data()

        # --- Real data: disaster risk per location ---
        self._assign_disaster_risk()

        # --- Update initial location populations ---
        self._sync_location_populations()

        print(self.config.summary())
        print(f"  Network: {self.network.number_of_nodes()} nodes, "
              f"{self.network.number_of_edges()} edges")
        print(f"  Agents initialized: {self.agents_pool.n_alive:,}")

    def _calibrate_rent_from_real_data(self):
        """Override rent_index with real prefecture-level rent data."""
        from .real_data import get_rent_index
        if self._loc_prefectures is None:
            return
        for i, pref in enumerate(self._loc_prefectures):
            self.loc_state.rent_index[i] = get_rent_index(pref)

    def _assign_disaster_risk(self):
        """Store per-location disaster risk from real seismic data."""
        from .real_data import get_disaster_risk
        n_locs = self.config.geography.n_locations
        self._disaster_risk = np.zeros(n_locs)
        if self._loc_prefectures is not None:
            for i, pref in enumerate(self._loc_prefectures):
                self._disaster_risk[i] = get_disaster_risk(pref)

    def _precompute_migration_candidates(self):
        """
        For each location, precompute a set of reachable destination candidates
        (neighbors + some long-range options).
        """
        n_locs = self.config.geography.n_locations
        self.migration_candidates = {}

        for node in range(n_locs):
            # Direct neighbors
            neighbors = set(self.network.neighbors(node))

            # 2-hop neighbors (weaker connections)
            two_hop = set()
            for nb in neighbors:
                two_hop.update(self.network.neighbors(nb))
            two_hop -= {node}

            # Sample a subset of 2-hop to keep candidate lists manageable
            all_cands = list(neighbors | two_hop)
            if len(all_cands) > 50:
                core_cands = list(neighbors)
                extra = list(two_hop - neighbors)
                self.rng.shuffle(extra)
                all_cands = core_cands + extra[:50 - len(core_cands)]

            self.migration_candidates[node] = np.array(all_cands, dtype=np.int32)

    def _sync_location_populations(self):
        """Recount agent populations per location."""
        self.loc_state.population = self.agents_pool.get_population_by_location(
            self.config.geography.n_locations
        )

    def step(self):
        """Execute one simulation step (= 1 quarter of a year)."""
        self.total_steps += 1
        n = self.agents_pool.next_id
        pool = self.agents_pool
        loc = self.loc_state
        quarter = self.current_step_in_year + 1

        # ==== 0. STOCHASTIC EVENTS (economic shocks, disasters) ====
        self._process_economic_shocks(n)
        self._process_natural_disasters(n)

        # ==== 0b. COMPANY HQ RELOCATIONS ====
        geo = self.config.geography
        t_end = geo.n_tokyo_wards
        c_end = t_end + geo.n_core_cities
        candidate_core = np.arange(t_end, c_end, dtype=np.int32)
        hq_relocations = self.company_pool.step_relocations(
            loc.tier, loc.prestige, loc.rent_index,
            self.config.policy.ice_act_tax_credit,
            self.config.policy.hq_relocation_active,
            candidate_core, self.rng,
        )

        # ==== 1. POLICY APPLICATION ====
        policy_metrics = self.policy_engine.apply_step(
            self.current_year, self.current_step_in_year,
            loc.prestige, loc.convenience, loc.anomie, loc.financial_friction,
            loc.tier, loc.hq_count, loc.healthcare_score, loc.childcare_score,
            loc.digital_access,
            pool.income, pool.location, pool.remote_worker, pool.alive, n,
            loc_lons=loc.lon, loc_lats=loc.lat,
        )
        policy_metrics["hq_relocations"] = hq_relocations

        # ==== 2. FULLY ENDOGENOUS LOCATION DYNAMICS ====
        # P, C, A, F all computed from emergent state
        update_location_dynamics(
            loc, self.config.policy,
            self.current_year, self.current_step_in_year, self.rng,
            agent_pool=pool, behavior_cfg=self.config.behavior,
            company_pool=self.company_pool,
        )

        # ==== 2b. CAREER ANXIETY + SCHOOL CLOSURE PRE-PROCESSING ====
        self._apply_career_anxiety(n)
        effective_anomie = self._compute_effective_anomie(n, loc)

        # ==== 2c. SOCIAL NETWORK DIFFUSION ====
        self._process_social_diffusion(n)

        # ==== 3. UTILITY COMPUTATION (GPU-accelerated) ====
        current_utility = self.compute_engine.compute_current_utility(
            pool.location[:n], pool.age[:n], pool.sex[:n],
            pool.education[:n], pool.cultural_orient[:n], pool.n_children[:n],
            loc.prestige, loc.convenience, effective_anomie, loc.financial_friction,
            loc.tier, self.rng,
            career_anxiety_boost=pool.years_in_core[:n].astype(np.float64) * self.config.behavior.career_anxiety_rate,
            career_anxiety_cap=self.config.behavior.career_anxiety_max_boost,
        )

        # Elderly care obligation: agents with elderly parents in periphery get retention bonus
        self._apply_elderly_care_retention(n, current_utility)

        # Update utility memory (exponential moving average)
        alpha = self.config.behavior.information_decay
        pool.utility_memory[:n] = alpha * pool.utility_memory[:n] + (1 - alpha) * current_utility

        prestige_alpha = self.config.behavior.prestige_memory_decay
        prestige_component = self.compute_engine.weights.w_prestige * loc.prestige[pool.location[:n]]
        pool.prestige_utility_memory[:n] = (
            prestige_alpha * pool.prestige_utility_memory[:n]
            + (1 - prestige_alpha) * prestige_component
        )

        # ==== 4. MIGRATION DECISIONS (with seasonal, gender, coupled) ====
        from .real_data import SEASONAL_MIGRATION_WEIGHT
        seasonal_mult = SEASONAL_MIGRATION_WEIGHT.get(quarter, 1.0)
        migration_count = self._process_migrations(current_utility, n,
                                                    seasonal_mult=seasonal_mult)

        # ==== 4b. RETURN MIGRATION (U-turn/J-turn/I-turn) ====
        return_count = self._process_return_migration(n)

        # ==== 5. DEMOGRAPHIC TRANSITIONS (seasonal-adjusted) ====
        from .real_data import SEASONAL_BIRTH_WEIGHT
        quarterly_scale = 1.0 / self.config.scale.steps_per_year
        deaths = self._process_mortality(n, quarterly_scale)

        birth_seasonal = SEASONAL_BIRTH_WEIGHT.get(quarter, 1.0)
        births = self._process_fertility(n, quarterly_scale * birth_seasonal)

        marriages = self._process_marriages_with_matching(n, quarterly_scale)

        # ==== 5b. JOB MARKET ====
        self._process_job_market(n)

        # ==== 5c. HOUSING MARKET (dynamic rent) ====
        self._update_housing_market()

        # ==== 5d. EDUCATION PIPELINE ====
        self._process_education(n)

        # ==== 5e. IMMIGRATION (if active) ====
        immigrants = self._process_immigration(n)

        # ==== 5f. COMMUTER INCOME (bed-town effect) ====
        self._apply_commuter_income(n)

        # Aging and income evolve once per year at end of Q4
        if self.current_step_in_year == self.config.scale.steps_per_year - 1:
            self._process_aging(n)
            self._process_income_evolution_real(n)

        # ==== 6. SYNC & COLLECT ====
        self._sync_location_populations()

        step_data = self._collect_step_data(
            births, deaths, marriages, migration_count + return_count, policy_metrics,
        )
        self.history.append(step_data)

        # Advance time
        self.current_step_in_year += 1
        if self.current_step_in_year >= self.config.scale.steps_per_year:
            self.current_step_in_year = 0
            self.current_year += 1

    def _apply_career_anxiety(self, n: int):
        """Increment years_in_core for agents living in Tier 1 (Core Cities).

        Paper Section 1.2: "Career Anxiety" grows the longer an agent stays
        in a professional environment. The more time in Core, the stronger
        the pull toward Tokyo's "Top-Tier" prestige.
        """
        pool = self.agents_pool
        loc = self.loc_state
        alive = pool.alive[:n]

        in_core = alive & (loc.tier[pool.location[:n]] == TIER_CORE)
        pool.years_in_core[:n] = np.where(in_core, pool.years_in_core[:n] + 1, 0)

    def _compute_effective_anomie(self, n: int, loc) -> np.ndarray:
        """Compute per-agent effective anomie with School Closure doubling.

        Paper Section 1.4: When a Tier 2 location's school closes,
        anomie effectively doubles for agents with n_children > 0.
        Returns a per-location anomie array where closed-school locations
        have elevated anomie. The agent-level child sensitivity is handled
        via the school_closure_mask passed to compute_engine.
        """
        pool = self.agents_pool
        effective = loc.anomie.copy()
        bhv = self.config.behavior

        if loc.school_closed is not None:
            # Build per-agent anomie boost for parents at closed-school locations
            agent_locs = pool.location[:n]
            has_children = pool.n_children[:n] > 0
            at_closed_school = np.zeros(n, dtype=np.bool_)

            for i in range(len(effective)):
                if loc.tier[i] == TIER_PERIPHERY and loc.school_closed[i]:
                    at_closed_school[agent_locs == i] = True
                    effective[i] = np.clip(
                        effective[i] * bhv.school_closure_anomie_multiplier, 0, 1
                    )

            # Store the mask so fertility and migration can use it
            self._school_closure_parent_mask = has_children & at_closed_school

        return effective

    def _apply_circular_tax_fertility_suppression(self, n: int):
        """Paper Section 3.4: The "Circular Tax" and "Single's Tax" perception.

        If disposable income after tax + rent falls below threshold,
        set fertility_intent to 0 for 4 quarters. Singles feel 2x the burden.
        """
        pool = self.agents_pool
        loc = self.loc_state
        bhv = self.config.behavior
        pol = self.config.policy
        alive = pool.alive[:n]

        annual_tax = pol.circular_tax_per_child_monthly * 12
        rent_cost = loc.rent_index[pool.location[:n]] * self.config.economy.national_median_income_jpy * 0.15

        # Singles feel 2x the tax burden (psychological "stick")
        is_single = pool.marital_status[:n] == 0
        effective_tax = np.where(
            is_single,
            annual_tax * bhv.singles_tax_perception_multiplier,
            annual_tax,
        )

        disposable = pool.income[:n] - effective_tax - rent_cost
        below_threshold = alive & (disposable < bhv.disposable_income_threshold)

        pool.fertility_suppressed_quarters[:n] = np.where(
            below_threshold,
            4,  # suppress for 4 quarters
            np.maximum(pool.fertility_suppressed_quarters[:n] - 1, 0),
        )

        # Zero out fertility intent for suppressed agents
        suppressed = pool.fertility_suppressed_quarters[:n] > 0
        pool.fertility_intent[:n] = np.where(suppressed, 0.0, pool.fertility_intent[:n])

    def _process_migrations(self, current_utility: np.ndarray, n: int,
                             seasonal_mult: float = 1.0) -> int:
        """Evaluate and execute migration decisions with top-K + chain migration."""
        pool = self.agents_pool
        loc = self.loc_state
        alive = pool.alive[:n]
        bhv = self.config.behavior

        eligible = (
            alive
            & (pool.migration_cooldown[:n] <= 0)
            & (pool.age[:n] >= 18)
            & (pool.age[:n] <= 70)
            & (~pool.in_university[:n])
        )
        eligible_idx = np.where(eligible)[0]

        if len(eligible_idx) == 0:
            pool.migration_cooldown[:n] = np.maximum(pool.migration_cooldown[:n] - 1, 0)
            return 0

        sample_rate = 0.025 * seasonal_mult
        if len(eligible_idx) > 50000:
            sample_size = max(int(len(eligible_idx) * sample_rate), 10000)
            eval_idx = self.rng.choice(eligible_idx, size=sample_size, replace=False)
        else:
            eval_idx = eligible_idx

        best_dest_util = np.full(n, -np.inf, dtype=np.float64)
        best_dest_loc = np.full(n, -1, dtype=np.int64)

        # Precompute chain migration bonus: count of social peers per location
        peer_loc_counts = self._compute_peer_location_counts(n)

        loc_groups = defaultdict(list)
        for idx in eval_idx:
            loc_groups[pool.location[idx]].append(idx)

        top_k = bhv.migration_top_k

        for curr_loc, agent_indices in loc_groups.items():
            cands = self.migration_candidates.get(curr_loc, np.array([], dtype=np.int32))
            if len(cands) == 0:
                continue

            agent_batch = np.array(agent_indices, dtype=np.int64)
            dest_utils = self.compute_engine.compute_destination_utilities(
                agent_batch, cands,
                pool.age[:n], pool.education[:n],
                pool.cultural_orient[:n], pool.n_children[:n],
                loc.prestige, loc.convenience, loc.anomie, loc.financial_friction,
                loc.tier, self.rng,
            )

            # DISTANCE FRICTION: penalize long-distance moves
            # ~80% of real moves in Japan are < 50km (intra-prefecture).
            # Calibrated so that inter-regional moves (>200km) are rare.
            distances = self.dist_matrix[curr_loc, cands]
            dist_penalty = 0.18 * np.log1p(distances / 10.0)
            dest_utils -= dist_penalty[None, :]

            # Add chain migration bonus for each agent-destination pair
            for k, ai in enumerate(agent_batch):
                for j, dest_loc in enumerate(cands):
                    peers_there = peer_loc_counts.get((ai, dest_loc), 0)
                    if peers_there > 0:
                        dest_utils[k, j] += bhv.chain_migration_weight * peers_there

            # STEPPING STONE: Tier 2 agents weight Tier 1 destinations 50% higher than Tier 0
            src_tier = loc.tier[curr_loc]
            if src_tier == TIER_PERIPHERY:
                for j, dest_loc in enumerate(cands):
                    dt = loc.tier[dest_loc]
                    if dt == TIER_CORE:
                        dest_utils[:, j] *= bhv.stepping_stone_core_weight
                    # Tokyo gets no boost -- higher psychological barrier from periphery

            # Tier-dependent softmax temperature (Paper: higher "jumpiness" in Core)
            agent_tier = src_tier
            softmax_temp = bhv.softmax_temp_by_tier.get(agent_tier, 5.0)

            # Top-K selection: evaluate K best, pick one stochastically
            n_cands = dest_utils.shape[1]
            actual_k = min(top_k, n_cands)

            for k, ai in enumerate(agent_batch):
                top_indices = np.argpartition(dest_utils[k], -actual_k)[-actual_k:]
                top_vals = dest_utils[k, top_indices]
                # Softmax with tier-dependent temperature
                shifted = top_vals - top_vals.max()
                exp_vals = np.exp(shifted * softmax_temp)
                probs = exp_vals / (exp_vals.sum() + 1e-12)
                chosen_local = self.rng.choice(actual_k, p=probs)
                chosen_cand = top_indices[chosen_local]
                best_dest_util[ai] = dest_utils[k, chosen_cand]
                best_dest_loc[ai] = cands[chosen_cand]

        decisions = self.compute_engine.decide_migrations(
            current_utility, best_dest_util, best_dest_loc,
            pool.migration_cooldown[:n], alive,
            pool.age[:n], pool.n_children[:n],
            self.rng,
        )

        movers = np.where(decisions >= 0)[0]
        migration_count = len(movers)

        # Housing capacity check: count current population at each location
        # and reject moves to locations already above capacity threshold
        pop_counts = np.bincount(pool.location[:n][alive], minlength=len(loc.capacity))
        capacity_occupancy = pop_counts / np.maximum(loc.capacity, 1)

        for mi in movers:
            # Gender-specific migration modifier
            gender_mod = self._get_gender_migration_modifier(mi, n)
            if gender_mod < 1.0 and self.rng.random() > gender_mod:
                continue

            old_loc = pool.location[mi]
            new_loc = decisions[mi]

            # Reject move if destination is at >95% capacity
            if capacity_occupancy[new_loc] > 0.95:
                continue

            from_tier = int(loc.tier[old_loc])
            to_tier = int(loc.tier[new_loc])
            pool.location[mi] = new_loc
            pool.migration_cooldown[mi] = bhv.migration_cooldown_years * 4

            # Coupled migration: bring partner and household
            self._move_household(mi, new_loc, n)

            # Regional Cannibalism tracking (Paper Section 3.1)
            agent_origin = int(pool.origin_tier[mi])
            if from_tier == TIER_PERIPHERY and to_tier == TIER_CORE:
                self.cannibalism_tracker["periphery_to_core"] += 1
            elif from_tier == TIER_PERIPHERY and to_tier == TIER_TOKYO:
                self.cannibalism_tracker["periphery_to_tokyo"] += 1
            elif from_tier == TIER_CORE and to_tier == TIER_TOKYO:
                self.cannibalism_tracker["core_to_tokyo"] += 1
                if agent_origin == TIER_PERIPHERY:
                    self.cannibalism_tracker["core_to_tokyo_from_periphery"] += 1

            self.migration_flows.append({
                "year": self.current_year,
                "quarter": self.current_step_in_year + 1,
                "step": self.total_steps,
                "agent_id": int(mi),
                "from_loc": int(old_loc),
                "to_loc": int(new_loc),
                "from_tier": from_tier,
                "to_tier": to_tier,
                "origin_tier": agent_origin,
                "agent_age": int(pool.age[mi]),
                "agent_sex": int(pool.sex[mi]),
                "agent_education": int(pool.education[mi]),
            })

        pool.migration_cooldown[:n] = np.maximum(pool.migration_cooldown[:n] - 1, 0)
        return migration_count

    def _compute_peer_location_counts(self, n: int) -> dict:
        """For chain migration: count how many social peers are at each candidate location.

        Uses vectorized approach: flatten valid links, group by (agent, peer_loc).
        """
        pool = self.agents_pool
        links = pool.social_links[:n]

        valid_mask = (links >= 0) & (links < n)
        agent_ids, link_slots = np.where(valid_mask)

        if len(agent_ids) == 0:
            return {}

        peer_ids = links[agent_ids, link_slots]
        peer_alive = pool.alive[peer_ids]
        agent_ids = agent_ids[peer_alive]
        peer_ids = peer_ids[peer_alive]

        if len(agent_ids) == 0:
            return {}

        peer_locs = pool.location[peer_ids]

        result = {}
        for i in range(len(agent_ids)):
            key = (agent_ids[i], peer_locs[i])
            result[key] = result.get(key, 0) + 1
        return result

    def _process_mortality(self, n: int, rate_scale: float = 1.0) -> int:
        """Apply age-sex specific mortality with COVID excess factor."""
        mort_m, mort_f = self.demo_engine.get_mortality_arrays()

        # COVID excess mortality: Japan experienced significant excess
        # deaths starting 2022 (Omicron waves). Factor captures both
        # direct COVID deaths and indirect healthcare disruption.
        covid_excess = {2021: 1.03, 2022: 1.10, 2023: 1.16, 2024: 1.22}
        excess_mult = covid_excess.get(self.current_year, 1.0)

        # Secular aging-driven mortality trend: as baby boomers enter 80+,
        # aggregate deaths rise even at constant age-specific rates.
        years_elapsed = max(0, self.current_year - 2020)
        aging_trend = 1.0 + 0.005 * years_elapsed

        total_scale = rate_scale * excess_mult * aging_trend
        mort_m = mort_m * total_scale
        mort_f = mort_f * total_scale
        rng_vals = self.rng.random(n)

        deaths = apply_mortality_kernel(
            self.agents_pool.alive[:n],
            self.agents_pool.age[:n],
            self.agents_pool.sex[:n],
            mort_m, mort_f, rng_vals,
        )

        death_count = int(deaths.sum())
        self.agents_pool.n_alive -= death_count

        # Handle widowing
        dead_idx = np.where(deaths == 1)[0]
        for di in dead_idx:
            partner = self.agents_pool.partner_id[di]
            if partner >= 0 and partner < n:
                if self.agents_pool.alive[partner]:
                    self.agents_pool.marital_status[partner] = 3
                    self.agents_pool.partner_id[partner] = -1
            self.agents_pool.partner_id[di] = -1

        return death_count

    def _process_fertility(self, n: int, rate_scale: float = 1.0) -> int:
        """Apply fertility: compute intentions, then births.

        Integrates the Circular Tax suppression (Paper 3.4) and
        the Tokyo fertility multiplier (0.75x).
        """
        pool = self.agents_pool
        loc = self.loc_state

        pool.fertility_intent[:n] = self.compute_engine.compute_fertility_intention(
            pool.age[:n], pool.sex[:n], pool.income[:n],
            pool.n_children[:n], pool.cultural_orient[:n],
            loc.childcare_score, pool.location[:n], loc.tier,
        )

        # Apply Circular Tax fertility suppression
        self._apply_circular_tax_fertility_suppression(n)

        asfr_t, asfr_c, asfr_p = self.demo_engine.get_fertility_arrays()

        # Calibration factor + secular TFR decline
        # Observed TFR path: 1.33 (2020) -> 1.30 -> 1.26 -> 1.20 -> 1.15 (2024)
        birth_calibration = 3.35
        years_elapsed = max(0, self.current_year - 2020)
        tfr_secular_decline = max(0.78, 1.0 - 0.010 * years_elapsed - 0.002 * years_elapsed**2)
        cal = rate_scale * birth_calibration * tfr_secular_decline
        asfr_t = asfr_t * cal
        asfr_c = asfr_c * cal
        asfr_p = asfr_p * cal
        rng_vals = self.rng.random(n)

        births_arr = apply_fertility_kernel(
            pool.alive[:n], pool.age[:n], pool.sex[:n],
            pool.marital_status[:n], pool.n_children[:n],
            loc.tier, pool.location[:n],
            asfr_t, asfr_c, asfr_p,
            pool.fertility_intent[:n],
            self.config.demography.max_children,
            rng_vals,
        )

        birth_idx = np.where(births_arr == 1)[0]
        actual_births = pool.add_newborns_batch(birth_idx, self.current_year, loc_tiers=loc.tier)
        return actual_births

    def _process_marriages(self, n: int, rate_scale: float = 1.0) -> int:
        """Process marriage transitions."""
        pool = self.agents_pool
        haz_m, haz_f = self.demo_engine.get_marriage_arrays()
        haz_m = haz_m * rate_scale
        haz_f = haz_f * rate_scale
        rng_vals = self.rng.random(n)

        new_marriages = apply_marriage_kernel(
            pool.alive[:n], pool.age[:n], pool.sex[:n],
            pool.marital_status[:n], haz_m, haz_f, rng_vals,
        )

        married_mask = pool.alive[:n] & (pool.marital_status[:n] == 1)
        divorce_rng = self.rng.random(n)
        divorce_rate = self.config.demography.divorce_rate_base * rate_scale
        divorce_mask = married_mask & (divorce_rng < divorce_rate)
        pool.marital_status[:n] = np.where(divorce_mask, 2, pool.marital_status[:n])

        return int(new_marriages.sum())

    def _update_housing_market(self):
        """Dynamic rent: rises with overcrowding, falls with vacancy."""
        loc = self.loc_state
        bhv = self.config.behavior
        n_locs = self.config.geography.n_locations

        for i in range(n_locs):
            pop = loc.population[i]
            cap = max(loc.capacity[i], 1)
            ratio = pop / cap

            # Target rent adjustment based on demand-supply mismatch
            if ratio > 1.0:
                pressure = (ratio - 1.0) * bhv.rent_demand_elasticity
            else:
                pressure = -(1.0 - ratio) * bhv.rent_vacancy_decay

            loc.rent_index[i] += pressure * bhv.rent_adjustment_speed
            loc.rent_index[i] = max(0.1, loc.rent_index[i])

            # Vacancy rate responds inversely to demand
            loc.vacancy_rate[i] = np.clip(
                loc.vacancy_rate[i] + (1.0 - ratio) * 0.005, 0.0, 0.5,
            )

    def _process_education(self, n: int):
        """University enrollment at 18 and graduation return migration at 22."""
        pool = self.agents_pool
        loc = self.loc_state
        bhv = self.config.behavior
        geo = self.config.geography

        # Enrollment: 18-year-olds enter university
        enrollable = (
            pool.alive[:n]
            & (pool.age[:n] == 18)
            & (~pool.in_university[:n])
            & (pool.education[:n] <= 1)
        )
        enrollable_idx = np.where(enrollable)[0]

        if len(enrollable_idx) > 0:
            enroll_mask = self.rng.random(len(enrollable_idx)) < bhv.university_enrollment_rate
            enrollees = enrollable_idx[enroll_mask]

            # University cities: all Tokyo + top Core cities by prestige
            t_end = geo.n_tokyo_wards
            c_end = t_end + geo.n_core_cities
            core_idx = np.arange(t_end, c_end)
            core_prestige = loc.prestige[core_idx]
            top_core = core_idx[np.argsort(core_prestige)[-30:]]
            uni_cities = np.concatenate([np.arange(t_end), top_core])

            for idx in enrollees:
                pool.in_university[idx] = True
                pool.home_location[idx] = pool.location[idx]
                # Move to university city
                dest = self.rng.choice(uni_cities)
                if pool.location[idx] != dest:
                    pool.location[idx] = dest

        # Graduation: 22-year-olds finish university
        graduates = (
            pool.alive[:n]
            & pool.in_university[:n]
            & (pool.age[:n] >= 22)
        )
        grad_idx = np.where(graduates)[0]

        for idx in grad_idx:
            pool.in_university[idx] = False
            pool.education[idx] = 2  # university degree

            # Return migration: some go back home
            if self.rng.random() < bhv.university_return_rate:
                pool.location[idx] = pool.home_location[idx]

    def _process_immigration(self, n: int) -> int:
        """Inject new immigrants if immigration policy is active."""
        pol = self.config.policy
        if not pol.immigration_active:
            return 0

        pool = self.agents_pool
        loc = self.loc_state
        quarterly_immigrants = pol.immigration_annual // (self.config.scale.agent_scale * self.config.scale.steps_per_year)
        quarterly_immigrants = max(1, quarterly_immigrants)

        created = 0
        tier_prefs = pol.immigration_tier_prefs
        geo = self.config.geography
        t_end = geo.n_tokyo_wards
        c_end = t_end + geo.n_core_cities

        for _ in range(quarterly_immigrants):
            if pool.next_id >= pool.age.shape[0]:
                break

            # Pick tier
            r = self.rng.random()
            if r < tier_prefs[0]:
                dest = self.rng.integers(0, t_end)
            elif r < tier_prefs[0] + tier_prefs[1]:
                dest = self.rng.integers(t_end, c_end)
            else:
                dest = self.rng.integers(c_end, geo.n_locations)

            idx = pool.next_id
            pool.alive[idx] = True
            pool.age[idx] = max(18, int(self.rng.normal(pol.immigration_age_mean, pol.immigration_age_std)))
            pool.sex[idx] = self.rng.integers(0, 2)
            pool.location[idx] = dest
            pool.home_location[idx] = dest
            pool.income[idx] = self.rng.normal(3_500_000, 800_000)
            pool.education[idx] = self.rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            pool.marital_status[idx] = 0
            pool.cultural_orient[idx] = self.rng.uniform(0.3, 0.7)
            pool.birth_year[idx] = self.current_year - pool.age[idx]
            pool.origin_tier[idx] = int(loc.tier[dest])
            pool.years_in_core[idx] = 0
            pool.fertility_suppressed_quarters[idx] = 0
            pool.prestige_utility_memory[idx] = 0.0
            pool.next_id += 1
            pool.n_alive += 1
            created += 1

        return created

    def _process_aging(self, n: int):
        """Age all living agents by 1 year."""
        apply_aging_kernel(self.agents_pool.alive[:n], self.agents_pool.age[:n])

    def _process_income_evolution(self, n: int):
        """Evolve incomes: inflation, career progression, location effects."""
        pool = self.agents_pool
        alive = pool.alive[:n]

        inflation = 1.02
        career_growth = np.where(
            (pool.age[:n] >= 22) & (pool.age[:n] <= 55),
            1.01 + 0.005 * pool.education[:n].astype(np.float64),
            1.0,
        )
        retirement_effect = np.where(pool.age[:n] >= 65, 0.98, 1.0)

        pool.income[:n] = np.where(
            alive,
            pool.income[:n] * inflation * career_growth * retirement_effect,
            pool.income[:n],
        )

    def _process_income_evolution_real(self, n: int):
        """Income evolution using real wage curves by age/sex/education."""
        from .real_data import MALE_WAGE_CURVE, FEMALE_WAGE_CURVE, EDUCATION_WAGE_MULT
        pool = self.agents_pool
        alive = pool.alive[:n]

        ages = np.clip(pool.age[:n], 0, 100)
        male_mult = MALE_WAGE_CURVE[ages]
        female_mult = FEMALE_WAGE_CURVE[ages]
        sex_curve = np.where(pool.sex[:n] == 1, female_mult, male_mult)

        edu_vals = pool.education[:n]
        edu_mult = np.array([EDUCATION_WAGE_MULT.get(e, 1.0) for e in edu_vals])

        base_income = np.ones(n) * 4_580_000
        if self._loc_prefectures is not None:
            from .prefecture_demographics import get_prefecture_profile
            for pref in np.unique(self._loc_prefectures):
                locs_in_pref = np.where(self._loc_prefectures == pref)[0]
                agents_in_pref = np.isin(pool.location[:n], locs_in_pref) & alive
                profile = get_prefecture_profile(pref)
                base_income[agents_in_pref] = profile["mean_income_jpy"]

        target_income = base_income * sex_curve * edu_mult
        recession_factor = 1.0 + self.recession_income_shock if self.in_recession else 1.0
        target_income *= recession_factor

        unemployed_factor = np.where(pool.unemployed[:n], 0.60, 1.0)
        target_income *= unemployed_factor

        # Regional wage multiplier: boost non-Tokyo incomes
        rwm = self.config.policy.regional_wage_multiplier
        if rwm != 1.0:
            non_tokyo = self.loc_state.tier[pool.location[:n]] != 0
            target_income = np.where(non_tokyo, target_income * rwm, target_income)

        blend = 0.15
        pool.income[:n] = np.where(
            alive & (pool.age[:n] >= 18),
            pool.income[:n] * (1 - blend) + target_income * blend,
            pool.income[:n],
        )

    # ------------------------------------------------------------------
    # PARTNER MATCHING (assortative marriage market)
    # ------------------------------------------------------------------
    def _process_marriages_with_matching(self, n: int, rate_scale: float = 1.0) -> int:
        """Marriage with actual partner assignment: assortative on age, education, location."""
        pool = self.agents_pool
        haz_m, haz_f = self.demo_engine.get_marriage_arrays()
        haz_m = haz_m * rate_scale
        haz_f = haz_f * rate_scale

        single_males = np.where(
            pool.alive[:n] & (pool.marital_status[:n] == 0) &
            (pool.age[:n] >= 18) & (pool.age[:n] <= 60) & (pool.sex[:n] == 0)
        )[0]
        single_females = np.where(
            pool.alive[:n] & (pool.marital_status[:n] == 0) &
            (pool.age[:n] >= 18) & (pool.age[:n] <= 60) & (pool.sex[:n] == 1)
        )[0]

        if len(single_males) == 0 or len(single_females) == 0:
            # Still process divorces
            self._process_divorces(n, rate_scale)
            return 0

        male_probs = haz_m[np.clip(pool.age[single_males], 0, len(haz_m) - 1)]
        want_marriage_m = single_males[self.rng.random(len(single_males)) < male_probs]

        female_probs = haz_f[np.clip(pool.age[single_females], 0, len(haz_f) - 1)]
        want_marriage_f = single_females[self.rng.random(len(single_females)) < female_probs]

        if len(want_marriage_m) == 0 or len(want_marriage_f) == 0:
            self._process_divorces(n, rate_scale)
            return 0

        self.rng.shuffle(want_marriage_m)
        self.rng.shuffle(want_marriage_f)

        cap = 3000
        want_marriage_m = want_marriage_m[:cap]
        want_marriage_f = want_marriage_f[:cap]

        m_locs = pool.location[want_marriage_m]
        f_locs = pool.location[want_marriage_f]

        m_sort = np.argsort(m_locs, kind='mergesort')
        f_sort = np.argsort(f_locs, kind='mergesort')
        sorted_m = want_marriage_m[m_sort]
        sorted_f = want_marriage_f[f_sort]
        sorted_m_locs = m_locs[m_sort]
        sorted_f_locs = f_locs[f_sort]

        if len(sorted_m_locs) == 0 or len(sorted_f_locs) == 0:
            self._process_divorces(n, rate_scale)
            return 0

        max_loc = max(sorted_m_locs[-1], sorted_f_locs[-1]) + 2
        m_bounds = np.searchsorted(sorted_m_locs, np.arange(max_loc))
        f_bounds = np.searchsorted(sorted_f_locs, np.arange(max_loc))

        marriages = 0
        rng = self.rng

        for loc_id in range(max_loc - 1):
            ms = sorted_m[m_bounds[loc_id]:m_bounds[loc_id + 1]]
            fs = sorted_f[f_bounds[loc_id]:f_bounds[loc_id + 1]]
            n_pairs = min(len(ms), len(fs))
            if n_pairs == 0:
                continue

            f_available = np.ones(len(fs), dtype=np.bool_)
            for i in range(n_pairs):
                m_idx = ms[i]
                avail = np.where(f_available)[0]
                if len(avail) == 0:
                    break
                age_diff = np.abs(pool.age[fs[avail]].astype(np.int32) - int(pool.age[m_idx]) + 2)
                best_local = avail[np.argmin(age_diff)]
                f_idx = fs[best_local]

                pool.marital_status[m_idx] = 1
                pool.marital_status[f_idx] = 1
                pool.partner_id[m_idx] = f_idx
                pool.partner_id[f_idx] = m_idx
                hh = max(pool.household_id[m_idx], pool.household_id[f_idx])
                if hh < 0:
                    hh = n + marriages
                pool.household_id[m_idx] = hh
                pool.household_id[f_idx] = hh
                f_available[best_local] = False
                marriages += 1

        self._process_divorces(n, rate_scale)
        return marriages

    def _process_divorces(self, n: int, rate_scale: float):
        pool = self.agents_pool
        married_mask = pool.alive[:n] & (pool.marital_status[:n] == 1)
        divorce_rate = self.config.demography.divorce_rate_base * rate_scale
        divorce_mask = married_mask & (self.rng.random(n) < divorce_rate)
        divorce_idx = np.where(divorce_mask)[0]
        for di in divorce_idx:
            pool.marital_status[di] = 2
            partner = pool.partner_id[di]
            if 0 <= partner < n and pool.alive[partner]:
                pool.marital_status[partner] = 2
                pool.partner_id[partner] = -1
            pool.partner_id[di] = -1

    # ------------------------------------------------------------------
    # COUPLED MIGRATION (household joint decisions)
    # ------------------------------------------------------------------
    def _move_household(self, mover_idx, new_loc, n):
        """When an agent migrates, bring their partner and young children."""
        pool = self.agents_pool
        partner = pool.partner_id[mover_idx]
        if 0 <= partner < n and pool.alive[partner]:
            if pool.location[partner] != new_loc:
                pool.location[partner] = new_loc
                pool.migration_cooldown[partner] = self.config.behavior.migration_cooldown_years * 4

    # ------------------------------------------------------------------
    # RETURN MIGRATION (U-turn / J-turn / I-turn)
    # ------------------------------------------------------------------
    def _process_return_migration(self, n: int) -> int:
        """Implement return migration patterns from real survey data."""
        from .real_data import RETURN_MIGRATION
        pool = self.agents_pool
        loc = self.loc_state
        rng = self.rng
        quarterly = 0.25

        alive = pool.alive[:n]
        eligible = alive & (pool.migration_cooldown[:n] <= 0) & (pool.age[:n] >= 25)
        not_home = pool.location[:n] != pool.home_location[:n]
        eligible = eligible & not_home

        eligible_idx = np.where(eligible)[0]
        if len(eligible_idx) == 0:
            return 0

        ages = pool.age[eligible_idx]
        has_parent = pool.has_elderly_parent[eligible_idx]
        home_tiers = loc.tier[pool.home_location[eligible_idx]]
        curr_tiers = loc.tier[pool.location[eligible_idx]]

        base_rate = RETURN_MIGRATION["u_turn_base_rate"] * quarterly
        age_40_boost = np.where(ages >= 40, RETURN_MIGRATION["u_turn_age_40_boost"] * quarterly, 0)
        age_55_boost = np.where(ages >= 55, RETURN_MIGRATION["u_turn_age_55_boost"] * quarterly, 0)
        parent_boost = np.where(has_parent, RETURN_MIGRATION["u_turn_has_elderly_parent"] * quarterly, 0)

        u_turn_prob = base_rate + age_40_boost + age_55_boost + parent_boost
        u_turn_prob *= np.where(pool.owns_property[eligible_idx], 1.3, 1.0)

        u_turns = rng.random(len(eligible_idx)) < u_turn_prob
        u_turn_idx = eligible_idx[u_turns]

        count = 0
        for idx in u_turn_idx:
            old_loc = pool.location[idx]
            new_loc = pool.home_location[idx]
            pool.location[idx] = new_loc
            pool.migration_cooldown[idx] = self.config.behavior.migration_cooldown_years * 4
            pool.migration_type[idx] = 1
            self._move_household(idx, new_loc, n)
            count += 1

        return count

    # ------------------------------------------------------------------
    # ELDERLY CARE RETENTION
    # ------------------------------------------------------------------
    def _apply_elderly_care_retention(self, n: int, utility: np.ndarray):
        """Agents with elderly parents in periphery get utility bonus for staying."""
        pool = self.agents_pool
        loc = self.loc_state

        has_parent = pool.has_elderly_parent[:n]
        in_periphery = loc.tier[pool.location[:n]] == TIER_PERIPHERY
        at_home = pool.location[:n] == pool.home_location[:n]
        care_mask = has_parent & in_periphery & at_home

        utility[care_mask] += 0.08

    # ------------------------------------------------------------------
    # JOB MARKET
    # ------------------------------------------------------------------
    def _process_job_market(self, n: int):
        """Simple job market: hiring/firing based on location economic health."""
        pool = self.agents_pool
        loc = self.loc_state
        rng = self.rng

        working_age = pool.alive[:n] & (pool.age[:n] >= 18) & (pool.age[:n] < 65)

        pop_ratio = np.zeros(n)
        agent_locs = pool.location[:n]
        for i in range(self.config.geography.n_locations):
            mask = agent_locs == i
            cap = max(loc.capacity[i], 1)
            pop_ratio[mask] = loc.population[i] / cap

        hire_prob = np.where(pop_ratio > 0.8, 0.12, 0.08)
        fire_prob = np.where(pop_ratio < 0.5, 0.015, 0.008)

        if self.in_recession:
            fire_prob *= 1.8
            hire_prob *= 0.6

        unemployed = pool.unemployed[:n]

        new_hires = working_age & unemployed & (rng.random(n) < hire_prob)
        pool.unemployed[:n] = np.where(new_hires, False, pool.unemployed[:n])

        new_fires = working_age & ~unemployed & (rng.random(n) < fire_prob)
        pool.unemployed[:n] = np.where(new_fires, True, pool.unemployed[:n])

    # ------------------------------------------------------------------
    # COMMUTER INCOME (bed-town effect)
    # ------------------------------------------------------------------
    def _apply_commuter_income(self, n: int):
        """Commuters earn employer-location wages but pay transport costs."""
        pool = self.agents_pool
        commuters = pool.employer_location[:n] >= 0
        if not commuters.any():
            return

        emp_locs = pool.employer_location[:n][commuters]
        emp_tiers = self.loc_state.tier[emp_locs]

        wage_boost = np.where(emp_tiers == TIER_TOKYO, 1.20,
                    np.where(emp_tiers == TIER_CORE, 1.08, 1.0))

        transport_cost = np.where(emp_tiers == TIER_TOKYO, 0.92,
                        np.where(emp_tiers == TIER_CORE, 0.95, 1.0))

        income_view = pool.income[:n]
        income_view[commuters] *= wage_boost * transport_cost

    # ------------------------------------------------------------------
    # ECONOMIC SHOCKS
    # ------------------------------------------------------------------
    def _process_economic_shocks(self, n: int):
        """Stochastic recession events."""
        from .real_data import RECESSION_PARAMS
        rng = self.rng

        if self.in_recession:
            self.recession_quarters_left -= 1
            if self.recession_quarters_left <= 0:
                self.in_recession = False
                self.recession_income_shock = 0.0
            else:
                self.recession_income_shock *= (1.0 - RECESSION_PARAMS["recovery_speed"])
            return

        quarterly_prob = RECESSION_PARAMS["annual_probability"] / 4.0
        if rng.random() < quarterly_prob:
            self.in_recession = True
            self.recession_income_shock = rng.normal(
                RECESSION_PARAMS["income_shock_mean"],
                RECESSION_PARAMS["income_shock_std"],
            )
            self.recession_quarters_left = RECESSION_PARAMS["duration_quarters"]

    # ------------------------------------------------------------------
    # NATURAL DISASTERS
    # ------------------------------------------------------------------
    def _process_natural_disasters(self, n: int):
        """Stochastic disaster events based on seismic risk."""
        from .real_data import DISASTER_EVENT_PARAMS
        rng = self.rng
        pool = self.agents_pool
        loc = self.loc_state

        for loc_id, recovery_left in list(self.disaster_recovery.items()):
            self.disaster_recovery[loc_id] = recovery_left - 1
            if recovery_left <= 0:
                del self.disaster_recovery[loc_id]
                self.disaster_locations.discard(loc_id)
                loc.anomie[loc_id] = max(loc.anomie[loc_id] - 0.10, 0)

        quarterly_prob = DISASTER_EVENT_PARAMS["annual_probability_per_location"] / 4.0
        rolls = rng.random(self.config.geography.n_locations)
        risk_adjusted = rolls / (self._disaster_risk + 0.01)
        hit_locs = np.where(risk_adjusted < quarterly_prob)[0]

        for loc_id in hit_locs:
            if loc_id in self.disaster_locations:
                continue

            self.disaster_locations.add(loc_id)
            recovery_q = DISASTER_EVENT_PARAMS["recovery_years"] * 4
            self.disaster_recovery[loc_id] = recovery_q

            loc.anomie[loc_id] = min(loc.anomie[loc_id] + 0.25, 1.0)
            loc.convenience[loc_id] = max(loc.convenience[loc_id] - DISASTER_EVENT_PARAMS["infrastructure_damage_factor"], 0)

            affected = np.where(pool.location[:n] == loc_id)[0]
            if len(affected) > 0:
                n_displaced = max(1, int(len(affected) * DISASTER_EVENT_PARAMS["displacement_fraction"]))
                displaced = rng.choice(affected, size=min(n_displaced, len(affected)), replace=False)
                pool.disaster_displaced[displaced] = True

                neighbors = self.migration_candidates.get(loc_id, np.array([], dtype=np.int32))
                safe_neighbors = [nb for nb in neighbors if nb not in self.disaster_locations]
                if safe_neighbors:
                    for di in displaced:
                        pool.location[di] = rng.choice(safe_neighbors)

    # ------------------------------------------------------------------
    # SOCIAL NETWORK DIFFUSION
    # ------------------------------------------------------------------
    def _process_social_diffusion(self, n: int):
        """Peer influence: migration info, fertility norms, cultural values."""
        pool = self.agents_pool
        rng = self.rng

        links = pool.social_links[:n]
        valid = (links >= 0) & (links < n)
        has_peers = valid.any(axis=1)
        agents_with_peers = np.where(pool.alive[:n] & has_peers)[0]

        if len(agents_with_peers) == 0:
            return

        sample_size = min(len(agents_with_peers), 10000)
        sampled = rng.choice(agents_with_peers, size=sample_size, replace=False)

        for ai in sampled:
            peer_ids = links[ai]
            valid_peers = peer_ids[(peer_ids >= 0) & (peer_ids < n)]
            valid_peers = valid_peers[pool.alive[valid_peers]]
            if len(valid_peers) == 0:
                continue

            peer_culture = pool.cultural_orient[valid_peers].mean()
            pool.cultural_orient[ai] += 0.02 * (peer_culture - pool.cultural_orient[ai])
            pool.cultural_orient[ai] = np.clip(pool.cultural_orient[ai], 0, 1)

            peer_fertility = pool.n_children[valid_peers].mean()
            own_children = pool.n_children[ai]
            if peer_fertility > own_children + 0.5:
                pool.fertility_intent[ai] = min(pool.fertility_intent[ai] + 0.01, 1.0)

    # ------------------------------------------------------------------
    # GENDER-SPECIFIC MIGRATION MODIFIER
    # ------------------------------------------------------------------
    def _get_gender_migration_modifier(self, agent_idx, n):
        """Young women leave rural areas faster (real data)."""
        from .real_data import GENDER_MIGRATION_MULTIPLIER as GMM
        pool = self.agents_pool
        loc = self.loc_state

        age = pool.age[agent_idx]
        sex = pool.sex[agent_idx]
        tier = loc.tier[pool.location[agent_idx]]

        if tier == TIER_PERIPHERY and 20 <= age <= 34:
            return GMM["periphery_female_20_34"] if sex == 1 else GMM["periphery_male_20_34"]
        elif tier == TIER_CORE and 20 <= age <= 34:
            return GMM["core_female_20_34"] if sex == 1 else GMM["core_male_20_34"]
        elif tier == TIER_TOKYO and age >= 35:
            return GMM["tokyo_female_35_plus"] if sex == 1 else GMM["tokyo_male_35_plus"]
        return 1.0

    def _collect_step_data(
        self,
        births: int, deaths: int, marriages: int,
        migrations: int, policy_metrics: dict,
    ) -> Dict:
        """Collect comprehensive metrics for this step."""
        pool = self.agents_pool
        loc = self.loc_state
        n = pool.next_id
        alive = pool.alive[:n]

        # Population by tier
        pop_by_loc = pool.get_population_by_location(self.config.geography.n_locations)
        geo = self.config.geography
        t_end = geo.n_tokyo_wards
        c_end = t_end + geo.n_core_cities

        tokyo_pop = int(pop_by_loc[:t_end].sum())
        core_pop = int(pop_by_loc[t_end:c_end].sum())
        peri_pop = int(pop_by_loc[c_end:].sum())

        # Tokyo Prefecture share: all locations in Tokyo pref (not just 23 wards)
        tokyo_pref_pop = 0
        if self._loc_prefectures is not None:
            tokyo_pref_mask = self._loc_prefectures == "Tokyo"
            tokyo_pref_pop = int(pop_by_loc[tokyo_pref_mask].sum())

        # Reproductive cohort (women 20-39)
        repro_mask = pool.get_reproductive_females_mask()
        repro_count = int(repro_mask.sum())

        # Compute TFR proxy
        female_20_39 = repro_count if repro_count > 0 else 1
        tfr_proxy = births * 20.0 / female_20_39 if births > 0 else 0

        # Mean utility by tier
        alive_mask = alive
        utils = pool.utility_memory[:n]

        agent_stats = pool.get_statistics()

        # School closures in periphery
        n_school_closures = 0
        if loc.school_closed is not None:
            n_school_closures = int(loc.school_closed[loc.tier == TIER_PERIPHERY].sum())

        # Fertility suppressed agents (Circular Tax)
        n_fertility_suppressed = int((pool.fertility_suppressed_quarters[:n] > 0).sum())

        # Mean career anxiety (years in core)
        core_mask = alive & (loc.tier[pool.location[:n]] == TIER_CORE)
        mean_career_anxiety = float(pool.years_in_core[:n][core_mask].mean()) if core_mask.any() else 0.0

        data = {
            "year": self.current_year,
            "quarter": self.current_step_in_year + 1,
            "step": self.total_steps,
            "total_population": int(alive.sum()),
            "tokyo_population": tokyo_pop,
            "core_population": core_pop,
            "periphery_population": peri_pop,
            "births": births,
            "deaths": deaths,
            "marriages": marriages,
            "migrations": migrations,
            "reproductive_cohort": repro_count,
            "tfr_proxy": round(tfr_proxy, 3),
            "mean_age": agent_stats.get("mean_age", 0),
            "pct_married": agent_stats.get("pct_married", 0),
            "mean_income": agent_stats.get("mean_income", 0),
            "pct_remote": agent_stats.get("pct_remote", 0),
            "tokyo_pop_share": tokyo_pop / max(alive.sum(), 1),
            "tokyo_pref_pop": tokyo_pref_pop,
            "tokyo_pref_share": tokyo_pref_pop / max(alive.sum(), 1),
            "core_pop_share": core_pop / max(alive.sum(), 1),
            "peri_pop_share": peri_pop / max(alive.sum(), 1),
            # --- Paper-specific metrics ---
            "periphery_to_core_flow": self.cannibalism_tracker["periphery_to_core"],
            "periphery_to_tokyo_flow": self.cannibalism_tracker["periphery_to_tokyo"],
            "core_to_tokyo_flow": self.cannibalism_tracker["core_to_tokyo"],
            "cannibalism_ratio": (
                self.cannibalism_tracker["periphery_to_core"]
                / max(self.cannibalism_tracker["periphery_to_tokyo"], 1)
            ),
            "core_to_tokyo_from_periphery": self.cannibalism_tracker["core_to_tokyo_from_periphery"],
            "launchpad_ratio": (
                self.cannibalism_tracker["core_to_tokyo_from_periphery"]
                / max(self.cannibalism_tracker["core_to_tokyo"], 1)
            ),
            "n_school_closures": n_school_closures,
            "n_fertility_suppressed": n_fertility_suppressed,
            "mean_career_anxiety_years": round(mean_career_anxiety, 2),
            # --- Endogenous PCAF tracking ---
            "mean_prestige_tokyo": float(loc.prestige[loc.tier == TIER_TOKYO].mean()),
            "mean_prestige_core": float(loc.prestige[loc.tier == TIER_CORE].mean()),
            "mean_prestige_periphery": float(loc.prestige[loc.tier == TIER_PERIPHERY].mean()),
            "mean_convenience_periphery": float(loc.convenience[loc.tier == TIER_PERIPHERY].mean()),
            "mean_anomie_periphery": float(loc.anomie[loc.tier == TIER_PERIPHERY].mean()),
            "mean_friction_tokyo": float(loc.financial_friction[loc.tier == TIER_TOKYO].mean()),
            "insurance_surcharge_jpy": float(loc.insurance_surcharge),
            "tokyo_hq_count": int(loc.hq_count[loc.tier == TIER_TOKYO].sum()),
            "core_hq_count": int(loc.hq_count[loc.tier == TIER_CORE].sum()),
            "n_human_warehouse_towns": int(
                ((loc.convenience > 0.6) & (loc.anomie > 0.7) & (loc.tier == TIER_PERIPHERY)).sum()
            ),
            "in_recession": self.in_recession,
            "n_disaster_locations": len(self.disaster_locations),
            "n_unemployed": int(pool.unemployed[:n].sum()),
        }
        data.update(policy_metrics)
        return data

    def run(self, n_years: Optional[int] = None, progress: bool = True):
        """Run the simulation for n_years (default: config.scale.n_years)."""
        if n_years is None:
            n_years = self.config.scale.n_years
        total_steps = n_years * self.config.scale.steps_per_year

        from tqdm import tqdm
        iterator = tqdm(range(total_steps), desc="Simulating") if progress else range(total_steps)

        for _ in iterator:
            self.step()

            if progress and self.total_steps % self.config.scale.steps_per_year == 0:
                pop = self.agents_pool.n_alive
                iterator.set_postfix({
                    "year": self.current_year,
                    "pop": f"{pop:,}",
                    "migrations": self.history[-1].get("migrations", 0),
                })

        return self.get_results()

    def get_results(self) -> Dict:
        """Package simulation results."""
        import pandas as pd

        history_df = pd.DataFrame(self.history)
        flows_df = pd.DataFrame(self.migration_flows) if self.migration_flows else pd.DataFrame()

        return {
            "config": self.config,
            "history": history_df,
            "migration_flows": flows_df,
            "final_agent_stats": self.agents_pool.get_statistics(),
            "final_location_state": self.loc_state,
            "network": self.network,
        }

    # ──────────────────────────────────────────────────────────
    # SNAPSHOT SYSTEM: save/load full simulation state
    # ──────────────────────────────────────────────────────────
    def save_snapshot(self, path: str = "snapshot.pkl") -> str:
        """Serialize the entire simulation state to disk."""
        import pickle, copy
        state = {
            "config": copy.deepcopy(self.config),
            "current_year": self.current_year,
            "current_step_in_year": self.current_step_in_year,
            "total_steps": self.total_steps,
            "rng_state": self.rng.bit_generator.state,
            "pool_arrays": {},
            "loc_arrays": {},
            "history": list(self.history),
            "cannibalism_tracker": dict(self.cannibalism_tracker),
            "in_recession": self.in_recession,
            "recession_quarters_left": self.recession_quarters_left,
            "recession_income_shock": self.recession_income_shock,
        }
        pool = self.agents_pool
        for attr in ["alive", "age", "sex", "location", "income", "education",
                      "marital_status", "n_children", "cultural_orient",
                      "migration_cooldown", "fertility_intent", "utility_memory",
                      "partner_id", "birth_year", "remote_worker", "in_university",
                      "university_origin", "years_in_core",
                      "prestige_utility_memory", "fertility_suppressed_quarters",
                      "household_id", "has_elderly_parent", "employer_location",
                      "unemployed", "disaster_displaced", "owns_property",
                      "migration_type", "origin_tier"]:
            arr = getattr(pool, attr, None)
            if arr is not None:
                state["pool_arrays"][attr] = arr.copy()
        state["pool_n_alive"] = pool.n_alive
        state["pool_next_id"] = pool.next_id
        state["pool_free_slots"] = list(pool._free_slots)

        loc = self.loc_state
        for attr in ["population", "capacity", "prestige", "convenience", "anomie",
                      "financial_friction", "rent_index", "vacancy_rate",
                      "healthcare_score", "childcare_score", "digital_access",
                      "transport_cost", "social_friction", "school_closed",
                      "prev_population", "poi_density", "high_tier_job_share",
                      "insurance_surcharge", "depopulation_rate", "hq_count"]:
            arr = getattr(loc, attr, None)
            if arr is not None and isinstance(arr, np.ndarray):
                state["loc_arrays"][attr] = arr.copy()
            elif arr is not None:
                state["loc_arrays"][attr] = arr

        state["company_hq_location"] = self.company_pool.hq_location.copy()
        state["company_hq_size"] = self.company_pool.hq_size.copy()
        state["company_cooldown"] = self.company_pool.relocation_cooldown.copy()

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load_snapshot(self, path: str = "snapshot.pkl") -> bool:
        """Restore simulation state from a snapshot file."""
        import pickle, copy
        from pathlib import Path
        if not Path(path).exists():
            return False

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.config = copy.deepcopy(state["config"])
        self.current_year = state["current_year"]
        self.current_step_in_year = state["current_step_in_year"]
        self.total_steps = state["total_steps"]
        self.rng.bit_generator.state = state["rng_state"]
        self.history = list(state["history"])
        self.cannibalism_tracker = dict(state["cannibalism_tracker"])
        self.in_recession = state["in_recession"]
        self.recession_quarters_left = state["recession_quarters_left"]
        self.recession_income_shock = state["recession_income_shock"]

        pool = self.agents_pool
        for attr, arr in state["pool_arrays"].items():
            target = getattr(pool, attr, None)
            if target is not None:
                n = min(len(arr), len(target))
                target[:n] = arr[:n]
        pool.n_alive = state["pool_n_alive"]
        pool.next_id = state["pool_next_id"]
        from collections import deque
        pool._free_slots = deque(state.get("pool_free_slots", []))

        loc = self.loc_state
        for attr, val in state["loc_arrays"].items():
            if isinstance(val, np.ndarray):
                target = getattr(loc, attr, None)
                if target is not None and isinstance(target, np.ndarray):
                    n = min(len(val), len(target))
                    target[:n] = val[:n]
            else:
                setattr(loc, attr, val)

        self.company_pool.hq_location[:] = state["company_hq_location"]
        self.company_pool.hq_size[:] = state["company_hq_size"]
        self.company_pool.relocation_cooldown[:] = state["company_cooldown"]

        self.compute_engine.weights = self.config.weights
        self.compute_engine.behavior = self.config.behavior

        self._sync_location_populations()
        return True

    def validate_against_real_data(self) -> dict:
        """Compare simulation outputs to observed 2020-2024 data."""
        from .real_data import VALIDATION_TARGETS
        import pandas as pd

        if not self.history:
            return {"error": "No history to validate"}

        history_df = pd.DataFrame(self.history)
        scale = self.config.scale.agent_scale
        results = {}

        for year, targets in VALIDATION_TARGETS.items():
            year_data = history_df[history_df["year"] == year]
            if year_data.empty:
                continue

            last_q = year_data.iloc[-1]
            sim_pop = int(last_q["total_population"]) * scale
            sim_tokyo_share = float(last_q["tokyo_pop_share"])
            sim_births = int(year_data["births"].sum()) * scale
            sim_deaths = int(year_data["deaths"].sum()) * scale

            results[year] = {
                "population": {"sim": sim_pop, "real": targets["total_pop"],
                              "error_pct": (sim_pop - targets["total_pop"]) / targets["total_pop"] * 100},
                "tokyo_share": {"sim": round(sim_tokyo_share, 4), "real": targets["tokyo_share"],
                               "error_pct": (sim_tokyo_share - targets["tokyo_share"]) / targets["tokyo_share"] * 100},
                "births": {"sim": sim_births, "real": targets["births"],
                          "error_pct": (sim_births - targets["births"]) / targets["births"] * 100},
                "deaths": {"sim": sim_deaths, "real": targets["deaths"],
                          "error_pct": (sim_deaths - targets["deaths"]) / targets["deaths"] * 100},
            }

        return results
