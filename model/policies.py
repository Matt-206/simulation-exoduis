"""
Policy intervention engine.

Models the effect of real Japanese policy instruments on the simulation:
- ICE Act (Industrial Competitiveness Enhancement) HQ relocation incentives
- Childcare subsidy expansion
- Remote work / digital nomad policies
- Housing subsidies for regional migration
- Medical DX (Digital Transformation) rollout
- Level 4 autonomous pod deployment
- Regional university investment
- Circular Tax (child support surcharge) effects
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .config import PolicyConfig, TIER_TOKYO, TIER_CORE, TIER_PERIPHERY


@dataclass
class PolicyEvent:
    """A discrete policy shock applied at a specific year."""
    year: int
    name: str
    description: str
    apply_fn: str  # method name on PolicyEngine


class PolicyEngine:
    """
    Applies and evolves policy effects on locations and agents each step.
    """

    def __init__(self, config: PolicyConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.hq_relocations_cumulative = 0
        self.scheduled_events: List[PolicyEvent] = []
        self._build_default_events()

    def _build_default_events(self):
        """Schedule realistic policy milestones."""
        self.scheduled_events = [
            PolicyEvent(2026, "ICE_Act_Phase2",
                        "Expanded tax credits for HQ relocation to Core Cities",
                        "boost_hq_relocation"),
            PolicyEvent(2028, "Childcare_Expansion",
                        "Universal childcare coverage target",
                        "expand_childcare"),
            PolicyEvent(2030, "Level4_Pods_Launch",
                        "Full autonomous vehicle deployment in Smart Districts",
                        "deploy_level4_pods"),
            PolicyEvent(2032, "Medical_DX_Complete",
                        "Remote diagnostics available in all municipalities",
                        "complete_medical_dx"),
            PolicyEvent(2035, "Remote_Work_Norm",
                        "Remote work becomes standard for 50% of knowledge workers",
                        "normalize_remote_work"),
            PolicyEvent(2040, "Regional_University_Hubs",
                        "Major research universities open satellite campuses",
                        "invest_regional_universities"),
        ]

    def apply_step(
        self,
        year: int,
        step_in_year: int,
        loc_prestige: np.ndarray,
        loc_convenience: np.ndarray,
        loc_anomie: np.ndarray,
        loc_friction: np.ndarray,
        loc_tiers: np.ndarray,
        loc_hq_count: np.ndarray,
        loc_healthcare: np.ndarray,
        loc_childcare: np.ndarray,
        loc_digital: np.ndarray,
        agent_income: np.ndarray,
        agent_locations: np.ndarray,
        agent_remote: np.ndarray,
        agent_alive: np.ndarray,
        n_agents: int,
        loc_lons: np.ndarray = None,
        loc_lats: np.ndarray = None,
    ) -> dict:
        """
        Apply all active policies for this step.
        Returns a dict of policy metrics for data collection.
        """
        metrics = {}

        if step_in_year == 0:
            for event in self.scheduled_events:
                if event.year == year:
                    method = getattr(self, event.apply_fn, None)
                    if method:
                        method(
                            loc_prestige, loc_convenience, loc_anomie, loc_friction,
                            loc_tiers, loc_hq_count, loc_healthcare, loc_childcare,
                            loc_digital,
                        )
                        metrics[f"event_{event.name}"] = True

        metrics.update(self._apply_hq_relocation_continuous(
            loc_prestige, loc_tiers, loc_hq_count, year,
        ))

        metrics.update(self._apply_remote_work_evolution(
            agent_remote, agent_alive, agent_locations, loc_tiers,
            year, n_agents,
        ))

        metrics.update(self._apply_circular_tax(
            agent_income, agent_locations, agent_alive,
            loc_tiers, n_agents,
        ))

        metrics.update(self._apply_housing_subsidies(
            loc_friction, loc_tiers,
        ))

        metrics.update(self._apply_university_decentralization(
            loc_prestige, loc_tiers, year,
        ))

        metrics.update(self._apply_enterprise_zones(
            loc_friction, loc_tiers,
        ))

        if loc_lons is not None and loc_lats is not None:
            metrics.update(self._apply_shinkansen_expansion(
                loc_convenience, loc_tiers, loc_lons, loc_lats, year,
            ))

        return metrics

    def _apply_hq_relocation_continuous(
        self,
        loc_prestige: np.ndarray,
        loc_tiers: np.ndarray,
        loc_hq_count: np.ndarray,
        year: int,
    ) -> dict:
        """
        ICE Act: Gradual HQ relocation from Tokyo to Core Cities.
        7% tax credit makes relocation NPV-positive above ¥3.5B investment.
        """
        if not self.config.hq_relocation_active:
            return {"hq_relocated": 0}

        n_locs = loc_tiers.shape[0]
        relocated_this_step = 0

        # Tokyo locations lose HQs probabilistically
        tokyo_mask = loc_tiers == TIER_TOKYO
        core_mask = loc_tiers == TIER_CORE
        core_indices = np.where(core_mask)[0]

        if len(core_indices) == 0:
            return {"hq_relocated": 0}

        for i in np.where(tokyo_mask)[0]:
            if loc_hq_count[i] <= 10:
                continue
            # Annual relocation probability per HQ
            relocation_prob = 0.002 * self.config.ice_act_tax_credit / 0.07
            n_relocating = self.rng.binomial(loc_hq_count[i], relocation_prob / 4)

            if n_relocating > 0:
                loc_hq_count[i] -= n_relocating

                # Distribute to top-prestige Core Cities
                target = self.rng.choice(core_indices, size=n_relocating)
                for t in target:
                    loc_hq_count[t] += 1
                    loc_prestige[t] = np.clip(
                        loc_prestige[t] + self.config.hq_relocation_prestige_boost * 0.01,
                        0, 1,
                    )

                relocated_this_step += n_relocating
                self.hq_relocations_cumulative += n_relocating

        return {"hq_relocated": relocated_this_step, "hq_total_relocated": self.hq_relocations_cumulative}

    def _apply_remote_work_evolution(
        self,
        agent_remote: np.ndarray,
        agent_alive: np.ndarray,
        agent_locations: np.ndarray,
        loc_tiers: np.ndarray,
        year: int,
        n_agents: int,
    ) -> dict:
        """Gradually increase remote work penetration."""
        years_elapsed = year - 2025
        target_rate = min(
            self.config.remote_work_penetration
            + self.config.remote_work_annual_growth * years_elapsed,
            0.65,
        )

        alive_mask = agent_alive[:n_agents]
        current_rate = agent_remote[:n_agents][alive_mask].mean() if alive_mask.any() else 0

        if current_rate < target_rate:
            # Convert some non-remote workers to remote
            eligible = alive_mask & (~agent_remote[:n_agents])
            eligible_idx = np.where(eligible)[0]
            n_convert = int(len(eligible_idx) * 0.005)
            if n_convert > 0:
                chosen = self.rng.choice(eligible_idx, size=min(n_convert, len(eligible_idx)), replace=False)
                agent_remote[chosen] = True

        return {"remote_work_rate": float(current_rate)}

    def _apply_circular_tax(
        self,
        agent_income: np.ndarray,
        agent_locations: np.ndarray,
        agent_alive: np.ndarray,
        loc_tiers: np.ndarray,
        n_agents: int,
    ) -> dict:
        """
        The ¥500/month child support surcharge.
        Acts as a "singlehood inertia" accelerator -- reduces disposable income
        slightly, making the cost-benefit of children worse.
        """
        annual_tax = self.config.circular_tax_per_child_monthly * 12
        alive = agent_alive[:n_agents]
        agent_income[:n_agents] = np.where(
            alive,
            agent_income[:n_agents] - annual_tax,
            agent_income[:n_agents],
        )
        agent_income[:n_agents] = np.maximum(agent_income[:n_agents], 0)
        return {"circular_tax_total": float(annual_tax * alive.sum())}

    def _apply_housing_subsidies(
        self, loc_friction: np.ndarray, loc_tiers: np.ndarray,
    ) -> dict:
        """Reduce financial friction in subsidized tiers."""
        core_mask = loc_tiers == TIER_CORE
        peri_mask = loc_tiers == TIER_PERIPHERY

        loc_friction[core_mask] *= (1.0 - self.config.housing_subsidy_core * 0.01)
        loc_friction[peri_mask] *= (1.0 - self.config.housing_subsidy_periphery * 0.01)

        return {}

    # -------------------------------------------------------------------
    # Discrete event handlers
    # -------------------------------------------------------------------
    def boost_hq_relocation(self, loc_p, loc_c, loc_a, loc_f, loc_t, loc_hq, loc_h, loc_cc, loc_d):
        """ICE Act Phase 2: doubled tax credits."""
        self.config.ice_act_tax_credit = 0.14

    def expand_childcare(self, loc_p, loc_c, loc_a, loc_f, loc_t, loc_hq, loc_h, loc_cc, loc_d):
        """Universal childcare push: boost childcare scores in Core + Periphery."""
        core_peri = (loc_t == TIER_CORE) | (loc_t == TIER_PERIPHERY)
        loc_cc[core_peri] = np.clip(loc_cc[core_peri] + 0.15, 0, 1)
        self.config.childcare_subsidy_ratio = 0.70

    def deploy_level4_pods(self, loc_p, loc_c, loc_a, loc_f, loc_t, loc_hq, loc_h, loc_cc, loc_d):
        """Autonomous transport in rural areas."""
        peri = loc_t == TIER_PERIPHERY
        loc_d[peri] = np.clip(loc_d[peri] + 0.25, 0, 1)
        self.config.level4_pod_deployment_rate = 0.08

    def complete_medical_dx(self, loc_p, loc_c, loc_a, loc_f, loc_t, loc_hq, loc_h, loc_cc, loc_d):
        """Remote diagnostics everywhere."""
        loc_h[:] = np.clip(loc_h + 0.20, 0, 1)

    def normalize_remote_work(self, loc_p, loc_c, loc_a, loc_f, loc_t, loc_hq, loc_h, loc_cc, loc_d):
        """Remote work becomes default for knowledge workers."""
        self.config.remote_work_penetration = 0.50
        self.config.remote_work_annual_growth = 0.01
        # Periphery becomes more attractive when work is location-agnostic
        peri = loc_t == TIER_PERIPHERY
        loc_f[peri] *= 0.85
        loc_a[peri] = np.clip(loc_a[peri] - 0.05, 0, 1)

    def invest_regional_universities(self, loc_p, loc_c, loc_a, loc_f, loc_t, loc_hq, loc_h, loc_cc, loc_d):
        """Research hub investment boosts prestige in Core Cities."""
        core = loc_t == TIER_CORE
        top_n = min(20, core.sum())
        core_idx = np.where(core)[0]
        if len(core_idx) > 0:
            chosen = self.rng.choice(core_idx, size=min(top_n, len(core_idx)), replace=False)
            loc_p[chosen] = np.clip(loc_p[chosen] + 0.10, 0, 1)
            loc_a[chosen] = np.clip(loc_a[chosen] - 0.05, 0, 1)

    # -------------------------------------------------------------------
    # New policy levers
    # -------------------------------------------------------------------
    def _apply_university_decentralization(
        self, loc_prestige, loc_tiers, year,
    ) -> dict:
        """Move top universities to regional cities, boosting prestige."""
        if not self.config.university_decentralization:
            return {}
        if year < 2028:
            return {}

        core_mask = loc_tiers == TIER_CORE
        core_idx = np.where(core_mask)[0]
        if len(core_idx) == 0:
            return {}

        n_target = min(self.config.n_regional_universities, len(core_idx))
        chosen = self.rng.choice(core_idx, size=n_target, replace=False)
        loc_prestige[chosen] = np.clip(
            loc_prestige[chosen] + self.config.university_prestige_boost * 0.01,
            0, 1,
        )
        return {"university_decentral_active": True}

    def _apply_enterprise_zones(
        self, loc_friction, loc_tiers,
    ) -> dict:
        """Enterprise zones: reduced financial friction in designated periphery nodes."""
        if not self.config.enterprise_zones_active:
            return {}

        peri_idx = np.where(loc_tiers == TIER_PERIPHERY)[0]
        if len(peri_idx) == 0:
            return {}

        if not hasattr(self, '_enterprise_zone_indices'):
            n_zones = min(self.config.n_enterprise_zones, len(peri_idx))
            self._enterprise_zone_indices = self.rng.choice(
                peri_idx, size=n_zones, replace=False,
            )

        loc_friction[self._enterprise_zone_indices] *= (
            1.0 - self.config.enterprise_zone_tax_break * 0.01
        )
        return {"enterprise_zones": len(self._enterprise_zone_indices)}

    def _apply_shinkansen_expansion(
        self, loc_convenience, loc_tiers, loc_lons, loc_lats, year,
    ) -> dict:
        """New Shinkansen routes boost convenience of nearby locations."""
        if not self.config.shinkansen_expansion_active:
            return {}

        activated = 0
        for route in self.config.shinkansen_new_routes:
            from_lon, from_lat, to_lon, to_lat, route_year = route
            if year < route_year:
                continue

            for i in range(len(loc_tiers)):
                dist_from = np.sqrt((loc_lons[i] - from_lon)**2 + (loc_lats[i] - from_lat)**2)
                dist_to = np.sqrt((loc_lons[i] - to_lon)**2 + (loc_lats[i] - to_lat)**2)
                dist_route = min(dist_from, dist_to)

                if dist_route < 0.5:
                    loc_convenience[i] = np.clip(
                        loc_convenience[i] + self.config.shinkansen_convenience_boost * 0.01,
                        0, 1,
                    )
                    activated += 1

        return {"shinkansen_routes_active": activated}
