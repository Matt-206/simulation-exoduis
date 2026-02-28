"""
3-tier geographic network model for Japan.

Tier 0: Tokyo Metropolitan Area (23 special wards)
Tier 1: 88 Core/Designated Cities (中核市 + 政令指定都市)
Tier 2: ~1,625 Periphery municipalities (市町村)

All 1,736 locations correspond to real Japanese municipalities with
coordinates from the Gazetteer of Japan and 2020 Census populations.

Network edges encode migration corridors with distance-based friction.
"""

import numpy as np
import networkx as nx
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from .config import (
    GeographyConfig, EconomyConfig, PolicyConfig,
    TIER_TOKYO, TIER_CORE, TIER_PERIPHERY,
    GPU_AVAILABLE,
)

if GPU_AVAILABLE:
    import cupy as cp


TOKYO_CENTER = (139.6917, 35.6895)
JAPAN_BOUNDS = {
    "lon_min": 129.5, "lon_max": 145.8,
    "lat_min": 30.5,  "lat_max": 45.5,
}


@dataclass
class LocationState:
    """Mutable state arrays for all locations."""
    tier: np.ndarray              # (n_loc,) int
    lon: np.ndarray               # (n_loc,) float
    lat: np.ndarray               # (n_loc,) float
    prestige: np.ndarray          # (n_loc,) float [0, 1]  -- ENDOGENOUS from HQs
    convenience: np.ndarray       # (n_loc,) float [0, 1]  -- ENDOGENOUS from POI + pop
    anomie: np.ndarray            # (n_loc,) float [0, 1]  -- ENDOGENOUS from depop + school
    financial_friction: np.ndarray # (n_loc,) float [0, 1]  -- ENDOGENOUS from demand
    base_wage: np.ndarray         # (n_loc,) float (multiplier)
    rent_index: np.ndarray        # (n_loc,) float
    transport_cost: np.ndarray    # (n_loc,) float
    vacancy_rate: np.ndarray      # (n_loc,) float [0, 1]
    healthcare_score: np.ndarray  # (n_loc,) float [0, 1]
    childcare_score: np.ndarray   # (n_loc,) float [0, 1]
    digital_access: np.ndarray    # (n_loc,) float [0, 1]
    hq_count: np.ndarray          # (n_loc,) int
    population: np.ndarray        # (n_loc,) int (current)
    capacity: np.ndarray          # (n_loc,) int
    social_friction: np.ndarray = None   # (n_loc,) float [0, 1] -- communal glue
    school_closed: np.ndarray = None     # (n_loc,) bool
    # --- Endogenous tracking ---
    prev_population: np.ndarray = None   # last quarter's population (for depop rate)
    poi_density: np.ndarray = None       # baseline POI/service density [0,1]
    high_tier_job_share: np.ndarray = None  # exec/management share of jobs
    insurance_surcharge: float = 500     # evolving JPY/month, starts at ¥500
    depopulation_rate: np.ndarray = None # quarterly pop change rate per location


def _load_land_polygons():
    """Load Japan GeoJSON and return list of (polygon_points, centroid) for land sampling."""
    geojson_path = Path("data/japan.geojson")
    if not geojson_path.exists():
        return None

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        coords_list = []
        if geom["type"] == "Polygon":
            coords_list = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                coords_list.append(poly[0])

        for ring in coords_list:
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            if len(ring) >= 4:
                polygons.append({
                    "ring": ring,
                    "lon_min": min(lons), "lon_max": max(lons),
                    "lat_min": min(lats), "lat_max": max(lats),
                    "centroid": (sum(lons) / len(lons), sum(lats) / len(lats)),
                })
    return polygons


def _point_in_polygon(px, py, ring):
    """Ray-casting point-in-polygon test."""
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_on_land(lon, lat, polygons):
    """Check if a point falls inside any of the land polygons."""
    for poly in polygons:
        if (poly["lon_min"] <= lon <= poly["lon_max"] and
                poly["lat_min"] <= lat <= poly["lat_max"]):
            if _point_in_polygon(lon, lat, poly["ring"]):
                return True
    return False


def _snap_to_land(lon, lat, polygons):
    """If point is not on land, find nearest polygon centroid."""
    if _point_on_land(lon, lat, polygons):
        return lon, lat
    best_dist = float("inf")
    best = (lon, lat)
    for poly in polygons:
        cx, cy = poly["centroid"]
        d = (cx - lon) ** 2 + (cy - lat) ** 2
        if d < best_dist:
            best_dist = d
            best = (cx, cy)
    return best


def _sample_on_land(rng, polygons, center_lon, center_lat, spread, max_tries=300):
    """Sample a point on land near the given center."""
    for _ in range(max_tries):
        lon = center_lon + rng.normal(0, spread)
        lat = center_lat + rng.normal(0, spread * 0.7)
        if _point_on_land(lon, lat, polygons):
            return lon, lat
    return _snap_to_land(center_lon, center_lat, polygons)


def _load_real_municipalities():
    """Load all municipalities from japan_municipalities.py, sorted by tier (0, 1, 2)."""
    from .japan_municipalities import MUNICIPALITIES

    # Sort: Tier 0 first, then Tier 1, then Tier 2 (stable within each tier)
    sorted_munis = sorted(MUNICIPALITIES, key=lambda m: m["tier"])
    return sorted_munis


def _generate_coordinates_real(sorted_munis):
    """Extract coordinates from real municipality data."""
    n = len(sorted_munis)
    lons = np.array([m["lon"] for m in sorted_munis], dtype=np.float64)
    lats = np.array([m["lat"] for m in sorted_munis], dtype=np.float64)
    return lons, lats


def _compute_distance_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Haversine distance matrix in km (vectorized)."""
    lon1 = np.radians(lons[:, None])
    lat1 = np.radians(lats[:, None])
    lon2 = np.radians(lons[None, :])
    lat2 = np.radians(lats[None, :])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_geography(
    geo_cfg: GeographyConfig,
    eco_cfg: EconomyConfig,
    rng: np.random.Generator,
) -> Tuple[LocationState, nx.Graph, np.ndarray]:
    """
    Construct the full geographic network from real municipality data.
    All 1,736 locations use real coordinates and 2020 Census populations.
    Returns: (LocationState, NetworkX graph, distance_matrix)
    """
    sorted_munis = _load_real_municipalities()
    n = len(sorted_munis)

    # Override config counts to match real data
    geo_cfg.n_tokyo_wards = sum(1 for m in sorted_munis if m["tier"] == 0)
    geo_cfg.n_core_cities = sum(1 for m in sorted_munis if m["tier"] == 1)
    geo_cfg.n_periphery_districts = sum(1 for m in sorted_munis if m["tier"] == 2)
    geo_cfg.n_locations = n

    t_end = geo_cfg.n_tokyo_wards
    c_end = t_end + geo_cfg.n_core_cities

    tiers = np.array([m["tier"] for m in sorted_munis], dtype=np.int32)
    lons, lats = _generate_coordinates_real(sorted_munis)

    # Real 2020 Census populations (used as proportional weights for agents)
    real_pops = np.array([m["population_2020"] for m in sorted_munis], dtype=np.float64)

    # --- Initialize location attributes by tier (with population-scaled variation) ---
    prestige = np.zeros(n)
    convenience = np.zeros(n)
    anomie = np.zeros(n)
    financial_friction = np.zeros(n)
    base_wage = np.ones(n)
    rent_index = np.ones(n)
    transport_cost = np.zeros(n)
    vacancy_rate = np.zeros(n)
    healthcare = np.zeros(n)
    childcare = np.zeros(n)
    digital_access = np.zeros(n)
    hq_count = np.zeros(n, dtype=np.int32)

    for i in range(n):
        noise = rng.normal(0, 0.03)
        pop = real_pops[i]

        if tiers[i] == TIER_TOKYO:
            # Scale prestige slightly by ward population (Minato/Shibuya > Adachi)
            pop_factor = np.clip(pop / 500_000, 0.5, 1.5)
            prestige[i] = np.clip(0.85 + 0.10 * pop_factor + noise, 0, 1)
            convenience[i] = np.clip(0.85 + noise, 0, 1)
            anomie[i] = np.clip(0.55 + rng.normal(0, 0.05), 0, 1)
            base_wage[i] = 1.0 + eco_cfg.tokyo_wage_premium
            rent_index[i] = eco_cfg.tokyo_rent_index + rng.normal(0, 0.2)
            transport_cost[i] = eco_cfg.tokyo_transport_cost
            vacancy_rate[i] = eco_cfg.vacancy_rate_tokyo + rng.normal(0, 0.005)
            healthcare[i] = np.clip(0.92 + noise, 0, 1)
            childcare[i] = np.clip(0.70 + noise, 0, 1)
            digital_access[i] = np.clip(0.95 + noise, 0, 1)
            hq_count[i] = rng.integers(50, 200)
            financial_friction[i] = np.clip(0.75 + noise, 0, 1)

        elif tiers[i] == TIER_CORE:
            # Larger core cities get more prestige
            pop_factor = np.clip(pop / 500_000, 0.3, 1.5)
            prestige[i] = np.clip(0.40 + 0.20 * pop_factor + rng.normal(0, 0.06), 0, 1)
            convenience[i] = np.clip(0.55 + 0.10 * pop_factor + rng.normal(0, 0.05), 0, 1)
            anomie[i] = np.clip(0.25 + rng.normal(0, 0.06), 0, 1)
            base_wage[i] = eco_cfg.core_wage_level + 0.05 * pop_factor + rng.normal(0, 0.03)
            rent_index[i] = eco_cfg.core_rent_index + 0.3 * pop_factor + rng.normal(0, 0.1)
            transport_cost[i] = eco_cfg.core_transport_cost
            vacancy_rate[i] = eco_cfg.vacancy_rate_core + rng.normal(0, 0.02)
            healthcare[i] = np.clip(0.70 + 0.10 * pop_factor + rng.normal(0, 0.04), 0, 1)
            childcare[i] = np.clip(0.50 + 0.10 * pop_factor + rng.normal(0, 0.05), 0, 1)
            digital_access[i] = np.clip(0.65 + 0.10 * pop_factor + rng.normal(0, 0.06), 0, 1)
            hq_count[i] = max(1, int(pop_factor * 15) + rng.integers(-3, 5))
            financial_friction[i] = np.clip(0.35 + 0.10 * pop_factor + rng.normal(0, 0.05), 0, 1)

        else:  # PERIPHERY
            # Tiny villages vs small cities have very different profiles
            pop_factor = np.clip(pop / 50_000, 0.1, 1.5)
            prestige[i] = np.clip(0.10 + 0.12 * pop_factor + rng.normal(0, 0.05), 0, 1)
            convenience[i] = np.clip(0.20 + 0.20 * pop_factor + rng.normal(0, 0.06), 0, 1)
            anomie[i] = np.clip(0.50 - 0.10 * pop_factor + rng.normal(0, 0.08), 0, 1)
            base_wage[i] = 1.0 - eco_cfg.periphery_wage_discount + 0.05 * pop_factor + rng.normal(0, 0.03)
            rent_index[i] = eco_cfg.periphery_rent_index + 0.15 * pop_factor + rng.normal(0, 0.08)
            transport_cost[i] = eco_cfg.periphery_transport_cost - 0.02 * pop_factor + rng.normal(0, 0.015)
            vacancy_rate[i] = np.clip(
                eco_cfg.vacancy_rate_periphery - 0.08 * pop_factor + rng.normal(0, 0.04),
                0.01, 0.5,
            )
            healthcare[i] = np.clip(0.25 + 0.20 * pop_factor + rng.normal(0, 0.08), 0, 1)
            childcare[i] = np.clip(0.20 + 0.15 * pop_factor + rng.normal(0, 0.06), 0, 1)
            digital_access[i] = np.clip(0.30 + 0.20 * pop_factor + rng.normal(0, 0.08), 0, 1)
            hq_count[i] = rng.integers(0, max(1, int(pop_factor * 3)))
            financial_friction[i] = np.clip(0.50 + 0.10 * (1.0 - pop_factor) + rng.normal(0, 0.06), 0, 1)

    # Population: use real 2020 Census populations directly.
    # The agent pool normalizes these into probabilities during _assign_locations.
    total_real_pop = real_pops.sum()
    pop_shares = real_pops / total_real_pop
    population = np.maximum(1, real_pops.astype(np.int64))

    capacity = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cap_factor = (
            geo_cfg.tokyo_capacity_factor if tiers[i] == TIER_TOKYO
            else geo_cfg.core_capacity_factor if tiers[i] == TIER_CORE
            else geo_cfg.periphery_capacity_factor
        )
        capacity[i] = max(1, int(population[i] * cap_factor))

    # Social friction: starts high in periphery, low in Tokyo
    social_friction = np.zeros(n)
    school_closed = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if tiers[i] == TIER_PERIPHERY:
            social_friction[i] = np.clip(0.75 + rng.normal(0, 0.08), 0, 1)
        elif tiers[i] == TIER_CORE:
            social_friction[i] = np.clip(0.40 + rng.normal(0, 0.06), 0, 1)
        else:
            social_friction[i] = np.clip(0.15 + rng.normal(0, 0.04), 0, 1)

    # --- POI density from real data ---
    from .companies import CompanyPool
    poi_density = np.zeros(n)
    prefectures_arr = np.array([m["prefecture"] for m in sorted_munis], dtype=object)
    for i, pref in enumerate(prefectures_arr):
        base_poi = CompanyPool.get_poi_density(pref)
        pop_scale = np.clip(real_pops[i] / 200_000, 0.1, 2.0)
        poi_density[i] = np.clip(base_poi * pop_scale + rng.normal(0, 0.02), 0.01, 1.0)

    state = LocationState(
        tier=tiers, lon=lons, lat=lats,
        prestige=prestige, convenience=convenience,
        anomie=anomie, financial_friction=financial_friction,
        base_wage=base_wage, rent_index=rent_index,
        transport_cost=transport_cost, vacancy_rate=vacancy_rate,
        healthcare_score=healthcare, childcare_score=childcare,
        digital_access=digital_access, hq_count=hq_count,
        population=population, capacity=capacity,
        social_friction=social_friction, school_closed=school_closed,
        prev_population=population.copy(),
        poi_density=poi_density,
        depopulation_rate=np.zeros(n),
    )

    # Store real population shares and municipality metadata on the state
    state._real_pop_shares = pop_shares
    state._municipality_names = [m["name_en"] for m in sorted_munis]
    state._municipality_names_jp = [m["name_jp"] for m in sorted_munis]
    state._prefectures = np.array([m["prefecture"] for m in sorted_munis], dtype=object)

    # --- Build NetworkX graph ---
    # With 1,736 nodes, use KD-tree for efficient neighbor finding
    from scipy.spatial import cKDTree

    dist_matrix = _compute_distance_matrix(lons, lats)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, tier=int(tiers[i]), lon=float(lons[i]), lat=float(lats[i]))

    # Use spatial indexing for efficient edge creation
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])
    tree = cKDTree(coords_rad)

    radius_km = geo_cfg.base_adjacency_radius_km
    radius_rad = radius_km / 6371.0  # approximate

    for i in range(n):
        # Find nearby nodes
        nearby = tree.query_ball_point(coords_rad[i], radius_rad * 1.5)
        for j in nearby:
            if j <= i:
                continue
            d = dist_matrix[i, j]
            same_tier = tiers[i] == tiers[j]

            connect = False
            if same_tier and tiers[i] == TIER_TOKYO and d < 20:
                connect = True
            elif same_tier and d < radius_km:
                if rng.random() < geo_cfg.intra_tier_connection_prob * 3:
                    connect = True
            elif not same_tier and d < radius_km * 1.5:
                if rng.random() < geo_cfg.inter_tier_connection_prob:
                    connect = True

            if connect:
                G.add_edge(i, j, distance=d, friction=d / 500.0)

    # Long-range corridors between core+ cities (Shinkansen/air)
    core_nodes = [i for i in range(n) if tiers[i] <= TIER_CORE]
    for idx_a, i in enumerate(core_nodes):
        for j in core_nodes[idx_a + 1:]:
            if G.has_edge(i, j):
                continue
            d = dist_matrix[i, j]
            if d < 800 and rng.random() < 0.02:
                G.add_edge(i, j, distance=d, friction=d / 500.0)

    # Ensure graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        main = max(components, key=len)
        for comp in components:
            if comp is main:
                continue
            node_c = rng.choice(list(comp))
            best_d = float("inf")
            best_m = rng.choice(list(main))
            for node_m in list(main)[:50]:
                d = dist_matrix[node_c, node_m]
                if d < best_d:
                    best_d = d
                    best_m = node_m
            G.add_edge(node_c, best_m, distance=best_d, friction=best_d / 500.0)

    return state, G, dist_matrix


def update_location_dynamics(
    state: LocationState,
    policy: PolicyConfig,
    year: int,
    step_in_year: int,
    rng: np.random.Generator,
    agent_pool=None,
    behavior_cfg=None,
    company_pool=None,
):
    """
    FULLY ENDOGENOUS location dynamics.
    All four PCAF variables are computed from emergent state, not fixed.

    P = f(HQs, high-tier jobs)            -- follows the companies
    C = f(POI density, population, gov)   -- decays with depop, rises with tech
    A = f(depop rate, school closure, tech-anomie seesaw) -- community decay
    F = f(demand-driven rent, vacancy tax, evolving insurance surcharge)
    """
    n = state.tier.shape[0]

    # ────────────────────────────────────────────────────────────────
    # 0. TRACK DEPOPULATION RATE (quarter-over-quarter)
    # ────────────────────────────────────────────────────────────────
    if state.prev_population is not None:
        delta = state.population.astype(np.float64) - state.prev_population.astype(np.float64)
        denom = np.maximum(state.prev_population.astype(np.float64), 1.0)
        state.depopulation_rate = delta / denom
    else:
        state.depopulation_rate = np.zeros(n)
    state.prev_population = state.population.copy()

    # ────────────────────────────────────────────────────────────────
    # 1. ENDOGENOUS PRESTIGE  P_l = 0.6*(HQs_l/HQs_max) + 0.4*(HighTierJobs_l/TotalJobs_l)
    # ────────────────────────────────────────────────────────────────
    if company_pool is not None:
        hq_weighted = company_pool.get_weighted_hq_counts()
        hq_max = max(hq_weighted.max(), 1.0)
        hq_component = hq_weighted / hq_max

        if state.high_tier_job_share is not None:
            htj = state.high_tier_job_share
        else:
            htj = np.full(n, 0.025)

        new_prestige = 0.6 * hq_component + 0.4 * htj
        # Very slow adaptation: 97% inertia, 3% new signal per quarter
        # Prestige reputation is sticky -- takes years to shift
        state.prestige = np.clip(0.97 * state.prestige + 0.03 * new_prestige, 0, 1)

        state.hq_count = company_pool.get_hq_counts()
    else:
        # Fallback: small drift toward population-driven prestige
        pass

    # ────────────────────────────────────────────────────────────────
    # 2. ENDOGENOUS CONVENIENCE  C_l = f(POI, healthcare, childcare, digital, pop)
    #    "15-minute city" score -- decays when population collapses,
    #    rises when gov deploys Level-4 pods / Medical DX
    # ────────────────────────────────────────────────────────────────
    pop_ratio = state.population.astype(np.float64) / np.maximum(state.capacity.astype(np.float64), 1.0)

    # --- Policy-driven improvements ---
    if state.tier is not None:
        peri_mask = state.tier == TIER_PERIPHERY
        # Level-4 pods raise digital access in periphery
        state.digital_access[peri_mask] = np.clip(
            state.digital_access[peri_mask] + policy.level4_pod_deployment_rate / 4.0 * 0.3,
            0, 1,
        )

    # Medical DX
    state.healthcare_score = np.clip(
        state.healthcare_score + policy.medical_dx_rollout_rate / 4.0 * 0.2,
        0, 1,
    )
    # Childcare expansion
    state.childcare_score = np.clip(
        state.childcare_score + policy.childcare_expansion_rate / 4.0 * 0.15,
        0, 1,
    )

    # --- DEPOPULATION DECAY: when people leave, services vanish ---
    # Buses stop running, stores close. POI density decays with population.
    if state.poi_density is not None:
        pop_decay = np.where(
            pop_ratio < 0.5,
            (0.5 - pop_ratio) * 0.02,
            0.0,
        )
        state.poi_density = np.clip(state.poi_density - pop_decay, 0.01, 1.0)
        poi = state.poi_density
    else:
        poi = np.full(n, 0.3)

    # Compute C as weighted composite
    transport_norm = np.clip(1.0 - state.transport_cost / 0.2, 0, 1)
    state.convenience = np.clip(
        0.25 * poi
        + 0.20 * state.digital_access
        + 0.20 * state.healthcare_score
        + 0.20 * state.childcare_score
        + 0.15 * transport_norm,
        0, 1,
    )

    # ────────────────────────────────────────────────────────────────
    # 3. ENDOGENOUS ANOMIE  A_l = Baseline + DepopRate*SchoolCoeff + TechAnomie
    #    "Perception of Societal Breakdown" -- community rituals collapse
    # ────────────────────────────────────────────────────────────────
    depop_rate = state.depopulation_rate
    school_closure_coeff = 0.25

    # --- School Closure trigger ---
    if state.school_closed is not None and behavior_cfg is not None:
        youth_threshold = behavior_cfg.school_closure_pop_threshold
        peri_mask = state.tier == TIER_PERIPHERY
        state.school_closed[peri_mask] = pop_ratio[peri_mask] < youth_threshold

    # --- Build anomie from scratch each step ---
    # Base anomie: low in Tokyo (crowding), mid in Core, higher in periphery
    base_anomie = np.where(
        state.tier == TIER_TOKYO, 0.35,
        np.where(state.tier == TIER_CORE, 0.20, 0.30)
    )

    # Depopulation component: negative depop_rate = people leaving = anomie rises
    depop_component = np.where(
        depop_rate < 0,
        np.abs(depop_rate) * 8.0,
        -depop_rate * 2.0,
    )

    # School closure component
    school_component = np.zeros(n)
    if state.school_closed is not None:
        school_component = np.where(state.school_closed, school_closure_coeff, 0.0)

    # Tokyo overcrowding anomie
    crowd_component = np.where(
        (state.tier == TIER_TOKYO) & (pop_ratio > 0.9),
        (pop_ratio - 0.9) * 0.5,
        0.0,
    )

    # ---- CONVENIENCE-ANOMIE SEESAW: "Human Warehouse" trigger ----
    # When artificial convenience replaces human services, anomie spikes.
    # High digital_access in low-population areas = "socially dead"
    tech_anomie_penalty = np.zeros(n)
    if behavior_cfg is not None and state.social_friction is not None:
        tech_level = state.digital_access

        # Tech erodes communal bonds (snow shoveling, mizo-soji, shared necessity)
        friction_loss = tech_level * 0.005
        state.social_friction = np.clip(state.social_friction - friction_loss, 0, 1)

        # The paradox: low social friction → anomie INCREASES
        glue_loss = np.maximum(0.5 - state.social_friction, 0)
        tech_anomie_penalty = glue_loss * behavior_cfg.tech_anomie_coupling

        # "Human Warehouse": high-tech + low-pop = maximum anomie
        human_warehouse_mask = (state.convenience > 0.6) & (pop_ratio < 0.4) & (state.tier == TIER_PERIPHERY)
        tech_anomie_penalty[human_warehouse_mask] += 0.15

    state.anomie = np.clip(
        base_anomie + depop_component + school_component + crowd_component + tech_anomie_penalty,
        0, 1,
    )

    # ────────────────────────────────────────────────────────────────
    # 4. ENDOGENOUS FINANCIAL FRICTION  F = f(demand-driven rent, akiya tax, evolving surcharge)
    # ────────────────────────────────────────────────────────────────
    # --- DEMAND-DRIVEN RENT: agents flooding in → rent rises ---
    inflow = np.maximum(depop_rate, 0)
    outflow = np.maximum(-depop_rate, 0)
    rent_pressure = np.where(
        state.tier == TIER_TOKYO,
        inflow * 2.0 - outflow * 0.5,
        inflow * 1.0 - outflow * 0.8,
    )
    state.rent_index = np.clip(state.rent_index + rent_pressure * 0.05, 0.1, 5.0)

    # Rent-Prestige coupling in Tokyo: high prestige → more demand → higher rent
    if behavior_cfg is not None:
        tokyo_mask = state.tier == TIER_TOKYO
        prestige_push = state.prestige[tokyo_mask] * behavior_cfg.tokyo_rent_prestige_coupling * 0.01
        state.rent_index[tokyo_mask] += prestige_push

    # --- VACANCY RATE responds to population ---
    state.vacancy_rate = np.clip(
        state.vacancy_rate + (1.0 - pop_ratio) * 0.003,
        0.0, 0.6,
    )

    # --- AKIYA SURVIVAL TAX: fewer people = higher per-capita infrastructure cost ---
    akiya_surcharge = np.zeros(n)
    if behavior_cfg is not None:
        peri_mask = state.tier == TIER_PERIPHERY
        excess_vacancy = np.maximum(state.vacancy_rate[peri_mask] - 0.15, 0)
        akiya_surcharge[peri_mask] = excess_vacancy * 0.4

    # --- EVOLVING INSURANCE SURCHARGE: as elderly:youth ratio rises nationally ---
    if agent_pool is not None:
        alive_mask = agent_pool.alive[:agent_pool.next_id]
        ages = agent_pool.age[:agent_pool.next_id]
        n_elderly = int(((ages >= 65) & alive_mask).sum())
        n_youth = max(int(((ages < 20) & alive_mask).sum()), 1)
        elderly_ratio = n_elderly / n_youth
        # Surcharge grows 2% for every 0.1 increase in elderly:youth ratio above 2.0
        ratio_excess = max(0, elderly_ratio - 2.0)
        state.insurance_surcharge = 500 + ratio_excess * 100

    # Rent subsidies reduce effective rent
    rent_effective = state.rent_index.copy()
    peri_mask = state.tier == TIER_PERIPHERY
    core_mask = state.tier == TIER_CORE
    rent_effective[peri_mask] *= (1.0 - policy.housing_subsidy_periphery)
    rent_effective[core_mask] *= (1.0 - policy.housing_subsidy_core)

    # Compute F from components
    rent_norm = rent_effective / 3.0
    transport_norm_f = state.transport_cost / 0.15
    vacancy_component = state.vacancy_rate * 0.3
    surcharge_norm = state.insurance_surcharge / 5000.0  # normalize to ~0.1 at ¥500

    # Tokyo cost-of-living surcharge: childcare, commute, tiny-apartment premium
    # Tokyo is genuinely 2-3x more expensive than non-metro Japan
    tokyo_col_premium = np.where(state.tier == TIER_TOKYO, 0.20, 0.0)

    state.financial_friction = np.clip(
        0.40 * rent_norm
        + 0.20 * transport_norm_f
        + 0.15 * vacancy_component
        + 0.10 * surcharge_norm
        + 0.15 * akiya_surcharge
        + tokyo_col_premium,
        0, 1,
    )
