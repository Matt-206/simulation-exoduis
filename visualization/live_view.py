"""
Real-time Pygame visualization for the Japan Exodus simulation.

Shows every agent as a colored dot on a map of Japan, with live
animations for migration trails, birth sparkles, death fades,
and a full statistics dashboard.

Controls:
    SPACE       - Pause / Resume
    UP/DOWN     - Speed up / slow down
    +/-         - Zoom in / out
    Arrow keys  - Pan the map (when zoomed)
    R           - Reset zoom/pan
    T           - Toggle agent color mode (action / tier / age / utility)
    H           - Cycle heatmap mode (off / density / prestige / anomie / friction / convenience)
    D           - Toggle dark/light mode
    I           - Toggle info panel
    P           - Toggle Policy Steering Panel
    F5          - Save snapshot
    F6          - Load snapshot
    CLICK       - Inspect agent or location (shows details)
    ESC / Q     - Quit
"""

import pygame
import numpy as np
import json
import sys
import math
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COL_BG_DARK = (12, 15, 22)
COL_BG_LIGHT = (235, 235, 240)
COL_PANEL_BG = (18, 22, 32)
COL_PANEL_BORDER = (45, 55, 75)
COL_TEXT = (220, 225, 235)
COL_TEXT_DIM = (120, 130, 150)
COL_TEXT_ACCENT = (100, 200, 255)
COL_TEXT_WARN = (255, 180, 60)

COL_TOKYO = (230, 57, 70)
COL_CORE = (69, 123, 157)
COL_PERIPHERY = (42, 157, 143)
COL_PERIPHERY_DIM = (25, 80, 72)

COL_AGENT_IDLE = (160, 170, 190)
COL_AGENT_MIGRATE = (255, 220, 50)
COL_AGENT_BIRTH = (80, 255, 120)
COL_AGENT_DEATH = (255, 60, 60)
COL_AGENT_MARRY = (120, 160, 255)
COL_AGENT_REMOTE = (180, 120, 255)

COL_TRAIL = (255, 200, 50, 80)

TIER_COLORS_PG = {0: COL_TOKYO, 1: COL_CORE, 2: COL_PERIPHERY}

AGE_GRADIENT = [
    (120, 220, 255),  # 0-19: light blue (young)
    (100, 255, 180),  # 20-34: green (young adult)
    (255, 255, 100),  # 35-49: yellow (middle)
    (255, 180, 60),   # 50-64: orange (older)
    (255, 80, 80),    # 65-79: red (elderly)
    (200, 60, 200),   # 80+: purple (very old)
]


# ---------------------------------------------------------------------------
# Animation particles
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    life: float = 1.0
    decay: float = 0.03
    color: Tuple = (255, 255, 255)
    radius: float = 2.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        self.radius = max(0.5, self.radius * 0.97)

    @property
    def alive(self):
        return self.life > 0


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out: slow → fast → slow (cubic smoothstep)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class MigrationTrail:
    """A migration animated along a multi-waypoint road path with smooth easing."""
    waypoints: list  # [(x, y), ...] screen coords
    seg_lengths: list = field(default_factory=list)  # cumulative arc lengths
    total_length: float = 0.0
    progress: float = 0.0
    speed: float = 0.018
    color: Tuple = COL_AGENT_MIGRATE
    tier_from: int = 2
    tier_to: int = 2

    def __post_init__(self):
        if len(self.waypoints) >= 2 and not self.seg_lengths:
            cumul = [0.0]
            for i in range(1, len(self.waypoints)):
                x0, y0 = self.waypoints[i - 1]
                x1, y1 = self.waypoints[i]
                d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                cumul.append(cumul[-1] + d)
            self.total_length = cumul[-1] if cumul[-1] > 0 else 1.0
            self.seg_lengths = cumul

    def update(self):
        self.progress += self.speed

    @property
    def alive(self):
        return self.progress < 1.0

    def _arc_pos(self, t_eased: float):
        """Get position along path at arc-length parameter t_eased in [0, 1]."""
        if len(self.waypoints) < 2:
            return self.waypoints[0] if self.waypoints else (0, 0)

        target_dist = t_eased * self.total_length
        for i in range(1, len(self.seg_lengths)):
            if self.seg_lengths[i] >= target_dist:
                seg_start = self.seg_lengths[i - 1]
                seg_end = self.seg_lengths[i]
                seg_len = seg_end - seg_start
                local_t = (target_dist - seg_start) / seg_len if seg_len > 0 else 0
                x0, y0 = self.waypoints[i - 1]
                x1, y1 = self.waypoints[i]
                return x0 + (x1 - x0) * local_t, y0 + (y1 - y0) * local_t

        return self.waypoints[-1]

    @property
    def current_pos(self):
        t = _ease_in_out(min(self.progress, 0.999))
        return self._arc_pos(t)

    @property
    def trail_points(self):
        """Return smoothly spaced points along the traveled portion of the path."""
        if len(self.waypoints) < 2:
            return []
        t_eased = _ease_in_out(min(self.progress, 0.999))
        target_dist = t_eased * self.total_length

        points = [self.waypoints[0]]
        for i in range(1, len(self.seg_lengths)):
            if self.seg_lengths[i] <= target_dist:
                points.append(self.waypoints[i])
            else:
                seg_start = self.seg_lengths[i - 1]
                seg_len = self.seg_lengths[i] - seg_start
                local_t = (target_dist - seg_start) / seg_len if seg_len > 0 else 0
                x0, y0 = self.waypoints[i - 1]
                x1, y1 = self.waypoints[i]
                points.append((x0 + (x1 - x0) * local_t, y0 + (y1 - y0) * local_t))
                break
        return points


# ---------------------------------------------------------------------------
# Main visualization class
# ---------------------------------------------------------------------------
class LiveSimulationView:
    """
    Real-time Pygame-based simulation viewer.
    Renders every agent on a geographic map with animated life events.
    """

    BASE_WINDOW_W = 1500
    WINDOW_H = 920
    MAP_W = 1060
    MAP_H = 920
    PANEL_W = 440
    POLICY_PANEL_W = 320  # overridden at runtime from PolicyPanel.WIDTH

    # Japan geographic bounds
    LON_MIN, LON_MAX = 128.5, 146.5
    LAT_MIN, LAT_MAX = 30.0, 46.0

    FPS = 60
    STEPS_PER_FRAME_OPTIONS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]

    def __init__(self, model):
        self.model = model
        self.pool = model.agents_pool
        self.loc = model.loc_state
        self.config = model.config
        self.network = model.network
        self._path_cache = {}  # (src_road, dst_road) -> [(lon,lat), ...]

        # Build actual road network
        from model.road_network import (
            build_road_graph, precompute_location_road_nodes, EXPRESSWAYS,
        )
        self.road_graph, self.road_node_coords = build_road_graph()
        self.expressway_routes = EXPRESSWAYS
        self.loc_road_nodes = precompute_location_road_nodes(
            self.loc.lon, self.loc.lat, self.road_node_coords,
        )
        print(f"  Road network: {self.road_graph.number_of_nodes()} nodes, "
              f"{self.road_graph.number_of_edges()} segments")

        pygame.init()
        pygame.display.set_caption("Japan Exodus Simulation -- Live View")

        # --- Policy Panel (visible by default, so start window wider) ---
        from visualization.widgets import PolicyPanel
        self.policy_panel = PolicyPanel(self.WINDOW_H, model)
        self.POLICY_PANEL_W = self.policy_panel.WIDTH
        self.WINDOW_W = self.BASE_WINDOW_W + self.POLICY_PANEL_W
        self.PANEL_X = self.POLICY_PANEL_W + self.MAP_W
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_med = pygame.font.SysFont("Consolas", 18)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_tiny = pygame.font.SysFont("Consolas", 11)

        # Map surface (pre-rendered background)
        self.map_surface = pygame.Surface((self.MAP_W, self.MAP_H))
        self.agent_surface = pygame.Surface((self.MAP_W, self.MAP_H), pygame.SRCALPHA)

        # State
        self.running = True
        self.paused = False
        self.speed_idx = 3  # starts at 1 step/frame
        self.step_accumulator = 0.0
        self.color_mode = 0  # 0=action, 1=tier, 2=age, 3=utility
        self.dark_mode = True
        self.show_info = True
        self.show_trails = True

        # Zoom/pan
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Inspection
        self.inspected_loc = None
        self.inspected_agent = None
        self.hover_loc = None

        # Heatmap mode: 0=off, 1=population density, 2=prestige, 3=anomie, 4=rent, 5=convenience, 6=utility
        self.heatmap_mode = 0
        self.heatmap_labels = [
            "Off", "Pop Density", "Prestige", "Anomie",
            "Fin Friction", "Convenience", "Agent Utility",
        ]

        # Timeline scrubber
        self.show_timeline = True
        self.year_summary_timer = 0
        self.year_summary_data = None

        # Animation queues
        self.particles: List[Particle] = []
        self.trails: List[MigrationTrail] = []
        self.recent_births = deque(maxlen=500)
        self.recent_deaths = deque(maxlen=500)
        self.recent_migrations = deque(maxlen=2000)

        # Load prefecture polygons from GeoJSON for map outline
        self.prefecture_polygons = self._load_prefecture_polygons()

        # Pre-compute location screen positions
        self._precompute_locations()

        # Cached map background (polygons + location markers)
        self._bg_cache_surface = pygame.Surface((self.MAP_W, self.MAP_H))
        self._bg_cache_key = None  # (zoom, pan_x, pan_y, dark_mode)

        # Pre-compute jitter for agents (fixed seed, done once)
        max_agents = self.pool.age.shape[0]
        rng = np.random.default_rng(42)
        self._jitter_x = rng.normal(0, 3.0, size=max_agents).astype(np.float32)
        self._jitter_y = rng.normal(0, 3.0, size=max_agents).astype(np.float32)

        # Stats history for sparklines
        self.pop_history = deque(maxlen=200)
        self.birth_history = deque(maxlen=200)
        self.death_history = deque(maxlen=200)
        self.migration_history = deque(maxlen=200)

        # --- HUD: Live metrics tracking ---
        self.tfr_history = deque(maxlen=200)
        self.cannibalism_history = deque(maxlen=200)
        self.peri_to_core_step = 0
        self.core_to_tokyo_step = 0

        # Step tracking
        self.prev_locations = self.pool.location[:self.pool.next_id].copy()
        self.prev_alive = self.pool.alive[:self.pool.next_id].copy()
        self.prev_n_children = self.pool.n_children[:self.pool.next_id].copy()
        self.prev_marital = self.pool.marital_status[:self.pool.next_id].copy()

        self.step_births = 0
        self.step_deaths = 0
        self.step_migrations = 0
        self.step_marriages = 0

        # --- Snapshot path ---
        self.snapshot_path = "snapshots/sim_snapshot.pkl"

    def _load_prefecture_polygons(self):
        """Load GeoJSON and convert prefecture boundaries to screen-space polygons."""
        geojson_path = Path("data/japan.geojson")
        if not geojson_path.exists():
            print("  [warn] japan.geojson not found, map outlines disabled")
            return []

        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        polygons = []
        skipped = 0

        lon_lo, lon_hi = self.LON_MIN, self.LON_MAX
        lat_lo, lat_hi = self.LAT_MIN, self.LAT_MAX

        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            name = feature.get("properties", {}).get("nam", "")
            tier = 2
            if "Tokyo" in name:
                tier = 0
            elif name in {
                "Hokkai Do", "Miyagi Ken", "Kanagawa Ken", "Aichi Ken",
                "Osaka Fu", "Hyogo Ken", "Hiroshima Ken", "Fukuoka Ken",
                "Saitama Ken", "Chiba Ken", "Kyoto Fu", "Shizuoka Ken",
                "Niigata Ken", "Okayama Ken",
            }:
                tier = 1

            coords_list = []
            if geom["type"] == "Polygon":
                coords_list = [geom["coordinates"][0]]
            elif geom["type"] == "MultiPolygon":
                for poly in geom["coordinates"]:
                    coords_list.append(poly[0])

            for ring in coords_list:
                ring_lons = [c[0] for c in ring]
                ring_lats = [c[1] for c in ring]

                # Skip if ANY vertex is far outside view bounds
                if (min(ring_lons) < lon_lo - 0.5 or max(ring_lons) > lon_hi + 0.5 or
                        min(ring_lats) < lat_lo - 0.5 or max(ring_lats) > lat_hi + 0.5):
                    skipped += 1
                    continue

                # Skip tiny islands (< 4 vertices or very small area)
                if len(ring) < 5:
                    skipped += 1
                    continue

                screen_pts = []
                for lon, lat in ring:
                    sx = (lon - lon_lo) / (lon_hi - lon_lo) * self.MAP_W
                    sy = (lat_hi - lat) / (lat_hi - lat_lo) * self.MAP_H
                    screen_pts.append((sx, sy))

                polygons.append({"points": screen_pts, "tier": tier, "name": name})

        print(f"  Loaded {len(polygons)} prefecture polygons ({skipped} outlying islands skipped)")
        return polygons

    def _precompute_locations(self):
        """Convert geographic coords to screen pixel positions."""
        lons = self.loc.lon
        lats = self.loc.lat
        self.loc_screen_x = ((lons - self.LON_MIN) / (self.LON_MAX - self.LON_MIN) * self.MAP_W).astype(np.int32)
        self.loc_screen_y = ((self.LAT_MAX - lats) / (self.LAT_MAX - self.LAT_MIN) * self.MAP_H).astype(np.int32)

    def _geo_to_screen_zoom(self, lon, lat):
        """Convert lon/lat to zoomed screen coordinates."""
        cx, cy = self.MAP_W / 2, self.MAP_H / 2
        sx = (lon - self.LON_MIN) / (self.LON_MAX - self.LON_MIN) * self.MAP_W
        sy = (self.LAT_MAX - lat) / (self.LAT_MAX - self.LAT_MIN) * self.MAP_H
        zx = (sx - cx) * self.zoom + cx + self.pan_x
        zy = (sy - cy) * self.zoom + cy + self.pan_y
        return zx, zy

    def _find_road_path(self, src_loc: int, dst_loc: int):
        """
        Route a migration through the actual expressway network.
        Path: src_location → nearest road node → road path → nearest road node → dst_location.
        Returns list of (screen_x, screen_y) waypoints.
        """
        import networkx as nx

        src_road = int(self.loc_road_nodes[src_loc])
        dst_road = int(self.loc_road_nodes[dst_loc])

        # If either location has no nearby road, fall back to straight line
        if src_road < 0 or dst_road < 0:
            return [
                self._geo_to_screen_zoom(self.loc.lon[src_loc], self.loc.lat[src_loc]),
                self._geo_to_screen_zoom(self.loc.lon[dst_loc], self.loc.lat[dst_loc]),
            ]

        if src_road == dst_road:
            return [
                self._geo_to_screen_zoom(self.loc.lon[src_loc], self.loc.lat[src_loc]),
                self._geo_to_screen_zoom(*self.road_node_coords[src_road]),
                self._geo_to_screen_zoom(self.loc.lon[dst_loc], self.loc.lat[dst_loc]),
            ]

        cache_key = (src_road, dst_road)
        if cache_key in self._path_cache:
            road_coords = self._path_cache[cache_key]
        else:
            try:
                node_path = nx.shortest_path(
                    self.road_graph, src_road, dst_road, weight="distance",
                )
                road_coords = [self.road_node_coords[n] for n in node_path]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                road_coords = [
                    self.road_node_coords[src_road],
                    self.road_node_coords[dst_road],
                ]
            if len(self._path_cache) < 8000:
                self._path_cache[cache_key] = road_coords

        # Build full path: location → road entry → road segments → road exit → location
        waypoints = [self._geo_to_screen_zoom(self.loc.lon[src_loc], self.loc.lat[src_loc])]
        for lon, lat in road_coords:
            waypoints.append(self._geo_to_screen_zoom(lon, lat))
        waypoints.append(self._geo_to_screen_zoom(self.loc.lon[dst_loc], self.loc.lat[dst_loc]))

        return waypoints

    def _geo_to_screen(self, lon, lat):
        """Convert single lon/lat to screen coordinates with zoom/pan."""
        x = (lon - self.LON_MIN) / (self.LON_MAX - self.LON_MIN) * self.MAP_W
        y = (self.LAT_MAX - lat) / (self.LAT_MAX - self.LAT_MIN) * self.MAP_H
        x = (x - self.MAP_W / 2) * self.zoom + self.MAP_W / 2 + self.pan_x
        y = (y - self.MAP_H / 2) * self.zoom + self.MAP_H / 2 + self.pan_y
        return x, y

    def _get_agent_screen_positions(self):
        """Batch compute screen positions for all alive agents."""
        n = self.pool.next_id
        alive = self.pool.alive[:n]
        locs = self.pool.location[:n]

        sx = self.loc_screen_x[locs].astype(np.float32)
        sy = self.loc_screen_y[locs].astype(np.float32)

        # Pre-computed jitter (fixed per agent, scaled by zoom)
        sx += self._jitter_x[:n] * self.zoom
        sy += self._jitter_y[:n] * self.zoom

        # Apply zoom/pan
        cx, cy = self.MAP_W / 2, self.MAP_H / 2
        sx = (sx - cx) * self.zoom + cx + self.pan_x
        sy = (sy - cy) * self.zoom + cy + self.pan_y

        return sx.astype(np.int32), sy.astype(np.int32), alive

    def _get_agent_colors(self, n: int) -> np.ndarray:
        """Compute RGB colors for each agent based on current color mode."""
        colors = np.zeros((n, 3), dtype=np.uint8)
        alive = self.pool.alive[:n]

        if self.color_mode == 0:  # Action-based
            # Default: idle color based on tier
            tiers = self.loc.tier[self.pool.location[:n]]
            colors[tiers == 0] = COL_TOKYO
            colors[tiers == 1] = COL_CORE
            colors[tiers == 2] = COL_PERIPHERY

            # Brighten remote workers
            remote = self.pool.remote_worker[:n] & alive
            colors[remote] = COL_AGENT_REMOTE

        elif self.color_mode == 1:  # Tier-based (more vivid)
            tiers = self.loc.tier[self.pool.location[:n]]
            colors[tiers == 0] = (255, 70, 80)
            colors[tiers == 1] = (80, 150, 220)
            colors[tiers == 2] = (50, 200, 160)

        elif self.color_mode == 2:  # Age-based (vectorized)
            ages = self.pool.age[:n]
            bins = np.digitize(ages, [20, 35, 50, 65, 80])
            for b in range(6):
                mask = (bins == b) & alive
                colors[mask] = AGE_GRADIENT[b]

        elif self.color_mode == 3:  # Utility-based
            utils = self.pool.utility_memory[:n]
            vmin, vmax = utils[alive].min() if alive.any() else 0, utils[alive].max() if alive.any() else 1
            if vmax - vmin < 1e-6:
                vmax = vmin + 1
            normed = (utils - vmin) / (vmax - vmin)
            normed = np.clip(normed, 0, 1)
            # Low utility = red, mid = yellow, high = green
            r = np.where(normed < 0.5, 255, (255 * (1 - (normed - 0.5) * 2)).astype(int))
            g = np.where(normed < 0.5, (255 * normed * 2).astype(int), 255)
            b_ch = np.full(n, 50, dtype=np.uint8)
            colors[alive, 0] = np.clip(r[alive], 0, 255).astype(np.uint8)
            colors[alive, 1] = np.clip(g[alive], 0, 255).astype(np.uint8)
            colors[alive, 2] = b_ch[alive]

        colors[~alive] = (0, 0, 0)
        return colors

    def _detect_events(self):
        """Compare pre/post step state to detect life events for animation."""
        n_prev = len(self.prev_alive)
        n_curr = self.pool.next_id
        n = min(n_prev, n_curr)

        alive_now = self.pool.alive[:n]
        alive_before = self.prev_alive[:n]

        # Deaths: was alive, now dead
        died = alive_before & (~alive_now)
        died_idx = np.where(died)[0]
        self.step_deaths = len(died_idx)

        for idx in died_idx[:100]:  # cap visual effects
            loc = self.prev_locations[idx]
            sx, sy = float(self.loc_screen_x[loc]), float(self.loc_screen_y[loc])
            sx_z = (sx - self.MAP_W / 2) * self.zoom + self.MAP_W / 2 + self.pan_x
            sy_z = (sy - self.MAP_H / 2) * self.zoom + self.MAP_H / 2 + self.pan_y
            for _ in range(4):
                angle = np.random.uniform(0, 2 * math.pi)
                speed = np.random.uniform(0.5, 2.0)
                self.particles.append(Particle(
                    sx_z, sy_z,
                    math.cos(angle) * speed, math.sin(angle) * speed,
                    life=1.0, decay=0.025, color=COL_AGENT_DEATH, radius=3.0,
                ))
            self.recent_deaths.append(time.time())

        # Births: new agents (indices >= n_prev) or children count increased
        new_borns = 0
        if n_curr > n_prev:
            new_borns = n_curr - n_prev
            for idx in range(n_prev, min(n_curr, n_prev + 100)):
                if self.pool.alive[idx]:
                    loc = self.pool.location[idx]
                    sx, sy = float(self.loc_screen_x[loc]), float(self.loc_screen_y[loc])
                    sx_z = (sx - self.MAP_W / 2) * self.zoom + self.MAP_W / 2 + self.pan_x
                    sy_z = (sy - self.MAP_H / 2) * self.zoom + self.MAP_H / 2 + self.pan_y
                    for _ in range(6):
                        angle = np.random.uniform(0, 2 * math.pi)
                        speed = np.random.uniform(1.0, 3.0)
                        self.particles.append(Particle(
                            sx_z, sy_z,
                            math.cos(angle) * speed, math.sin(angle) * speed,
                            life=1.0, decay=0.02, color=COL_AGENT_BIRTH, radius=3.5,
                        ))
                    self.recent_births.append(time.time())
        self.step_births = new_borns

        # Migrations: location changed while alive
        loc_now = self.pool.location[:n]
        loc_before = self.prev_locations[:n]
        migrated = alive_now & alive_before & (loc_now != loc_before)
        migrated_idx = np.where(migrated)[0]
        self.step_migrations = len(migrated_idx)

        for idx in migrated_idx[:50]:
            old_loc = int(loc_before[idx])
            new_loc = int(loc_now[idx])
            waypoints = self._find_road_path(old_loc, new_loc)

            if len(waypoints) < 2:
                continue

            # Sanity: reject paths with any huge single jump (> 40% of screen)
            max_jump = self.MAP_W * 0.4
            sane = True
            for i in range(len(waypoints) - 1):
                dx = abs(waypoints[i + 1][0] - waypoints[i][0])
                dy = abs(waypoints[i + 1][1] - waypoints[i][1])
                if dx > max_jump or dy > max_jump:
                    sane = False
                    break
            if not sane:
                continue

            tier_from = int(self.loc.tier[old_loc])
            tier_to = int(self.loc.tier[new_loc])
            n_segs = len(waypoints) - 1
            speed = max(0.012, min(0.035, 0.06 / max(n_segs, 1)))
            trail_color = TIER_COLORS_PG.get(tier_to, COL_AGENT_MIGRATE)
            self.trails.append(MigrationTrail(
                waypoints=waypoints, speed=speed, color=trail_color,
                tier_from=tier_from, tier_to=tier_to,
            ))
            self.recent_migrations.append(time.time())

        # Marriages
        marital_now = self.pool.marital_status[:n]
        marital_before = self.prev_marital[:n]
        married = alive_now & (marital_now == 1) & (marital_before != 1)
        married_idx = np.where(married)[0]
        self.step_marriages = len(married_idx)

        for idx in married_idx[:50]:
            loc = self.pool.location[idx]
            sx = float(self.loc_screen_x[loc])
            sy = float(self.loc_screen_y[loc])
            sx_z = (sx - self.MAP_W / 2) * self.zoom + self.MAP_W / 2 + self.pan_x
            sy_z = (sy - self.MAP_H / 2) * self.zoom + self.MAP_H / 2 + self.pan_y
            for _ in range(5):
                angle = np.random.uniform(0, 2 * math.pi)
                speed = np.random.uniform(0.8, 2.5)
                self.particles.append(Particle(
                    sx_z, sy_z,
                    math.cos(angle) * speed, math.sin(angle) * speed,
                    life=1.0, decay=0.02, color=COL_AGENT_MARRY, radius=2.5,
                ))

        # Update previous state
        self.prev_locations = self.pool.location[:self.pool.next_id].copy()
        self.prev_alive = self.pool.alive[:self.pool.next_id].copy()
        self.prev_n_children = self.pool.n_children[:self.pool.next_id].copy()
        self.prev_marital = self.pool.marital_status[:self.pool.next_id].copy()

    def _update_animations(self):
        """Update all animation particles and trails."""
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]

        for t in self.trails:
            t.update()
        self.trails = [t for t in self.trails if t.alive]

    def _rebuild_bg_cache(self):
        """Render prefecture polygons + location markers to the cached surface."""
        bg = COL_BG_DARK if self.dark_mode else COL_BG_LIGHT
        self._bg_cache_surface.fill(bg)

        cx, cy = self.MAP_W / 2, self.MAP_H / 2

        tier_fill_dark = {0: (45, 18, 22), 1: (18, 28, 45), 2: (18, 25, 22)}
        tier_fill_light = {0: (255, 235, 238), 1: (227, 242, 253), 2: (245, 240, 232)}
        tier_edge_dark = {0: (120, 40, 40), 1: (50, 70, 100), 2: (40, 60, 55)}
        tier_edge_light = {0: (200, 160, 160), 1: (180, 190, 210), 2: (208, 208, 200)}

        fills = tier_fill_dark if self.dark_mode else tier_fill_light
        edges = tier_edge_dark if self.dark_mode else tier_edge_light

        for poly_data in self.prefecture_polygons:
            pts = poly_data["points"]
            tier = poly_data["tier"]

            zoomed_pts = []
            for px, py in pts:
                zx = (px - cx) * self.zoom + cx + self.pan_x
                zy = (py - cy) * self.zoom + cy + self.pan_y
                zoomed_pts.append((zx, zy))

            xs = [p[0] for p in zoomed_pts]
            ys = [p[1] for p in zoomed_pts]
            if max(xs) < -50 or min(xs) > self.MAP_W + 50 or max(ys) < -50 or min(ys) > self.MAP_H + 50:
                continue

            if len(zoomed_pts) >= 3:
                pygame.draw.polygon(self._bg_cache_surface, fills[tier], zoomed_pts)
                pygame.draw.polygon(self._bg_cache_surface, edges[tier], zoomed_pts, 1)

        # --- Draw expressway road network ---
        road_col = (55, 65, 85) if self.dark_mode else (190, 195, 205)
        road_col_major = (70, 80, 100) if self.dark_mode else (170, 175, 190)

        for route_name, waypoints in self.expressway_routes.items():
            screen_pts = []
            for lon, lat in waypoints:
                sx = (lon - self.LON_MIN) / (self.LON_MAX - self.LON_MIN) * self.MAP_W
                sy = (self.LAT_MAX - lat) / (self.LAT_MAX - self.LAT_MIN) * self.MAP_H
                zx = (sx - cx) * self.zoom + cx + self.pan_x
                zy = (sy - cy) * self.zoom + cy + self.pan_y
                screen_pts.append((int(zx), int(zy)))

            if len(screen_pts) >= 2:
                is_major = route_name in {
                    "tohoku_exp", "tomei_exp", "meishin_exp", "sanyo_exp",
                    "kyushu_exp", "hokkaido_exp", "hokuriku_exp",
                }
                col = road_col_major if is_major else road_col
                width = 2 if is_major else 1
                pygame.draw.lines(self._bg_cache_surface, col, False, screen_pts, width)

        # --- Location markers on top of roads ---
        n_locs = self.config.geography.n_locations
        pops = self.loc.population

        for i in range(n_locs):
            sx = float(self.loc_screen_x[i])
            sy = float(self.loc_screen_y[i])
            sx_z = (sx - cx) * self.zoom + cx + self.pan_x
            sy_z = (sy - cy) * self.zoom + cy + self.pan_y

            if not (-20 <= sx_z < self.MAP_W + 20 and -20 <= sy_z < self.MAP_H + 20):
                continue

            tier = self.loc.tier[i]
            pop = pops[i]

            if tier == 0:
                r = max(3, min(12, int(math.sqrt(pop) * 0.15 * self.zoom)))
                glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*COL_TOKYO, 30), (r * 2, r * 2), r * 2)
                self._bg_cache_surface.blit(glow_surf, (int(sx_z) - r * 2, int(sy_z) - r * 2))
                pygame.draw.circle(self._bg_cache_surface, COL_TOKYO, (int(sx_z), int(sy_z)), r)
            elif tier == 1:
                r = max(2, min(8, int(math.sqrt(pop) * 0.08 * self.zoom)))
                pygame.draw.circle(self._bg_cache_surface, COL_CORE, (int(sx_z), int(sy_z)), r)
            else:
                if pop > 10:
                    r = max(1, min(3, int(math.sqrt(max(pop, 1)) * 0.03 * self.zoom)))
                    col = COL_PERIPHERY_DIM if self.dark_mode else (150, 200, 180)
                    pygame.draw.circle(self._bg_cache_surface, col, (int(sx_z), int(sy_z)), r)

        self._bg_cache_key = (self.zoom, self.pan_x, self.pan_y, self.dark_mode)

    def _render_map_background(self):
        """Blit the cached background (only rebuild when view changes)."""
        key = (self.zoom, self.pan_x, self.pan_y, self.dark_mode)
        if self._bg_cache_key != key:
            self._rebuild_bg_cache()
        self.map_surface.blit(self._bg_cache_surface, (0, 0))

    def _render_heatmap(self):
        """Render a semi-transparent heatmap overlay for the selected metric."""
        if self.heatmap_mode == 0:
            return

        n_locs = self.config.geography.n_locations
        cx, cy = self.MAP_W / 2, self.MAP_H / 2

        # Select data array based on mode
        if self.heatmap_mode == 1:
            values = self.loc.population.astype(np.float64)
            values = values / max(values.max(), 1)
        elif self.heatmap_mode == 2:
            values = self.loc.prestige.copy()
        elif self.heatmap_mode == 3:
            values = self.loc.anomie.copy()
        elif self.heatmap_mode == 4:
            values = self.loc.financial_friction.copy()
        elif self.heatmap_mode == 5:
            values = self.loc.convenience.copy()
        elif self.heatmap_mode == 6:
            # Per-location average utility from agent utility_memory
            n = self.pool.next_id
            alive = self.pool.alive[:n]
            locs = self.pool.location[:n]
            utils = self.pool.utility_memory[:n]
            n_locs_arr = self.config.geography.n_locations
            loc_util_sum = np.bincount(locs[alive], weights=utils[alive], minlength=n_locs_arr)
            loc_util_cnt = np.bincount(locs[alive], minlength=n_locs_arr).astype(np.float64)
            loc_util_cnt[loc_util_cnt == 0] = 1
            values = loc_util_sum / loc_util_cnt
        else:
            return

        v_min = values.min()
        v_max = max(values.max(), v_min + 0.001)
        normed = (values - v_min) / (v_max - v_min)

        heat_surface = pygame.Surface((self.MAP_W, self.MAP_H), pygame.SRCALPHA)

        for i in range(n_locs):
            lx = float(self.loc_screen_x[i])
            ly = float(self.loc_screen_y[i])
            sx_z = (lx - cx) * self.zoom + cx + self.pan_x
            sy_z = (ly - cy) * self.zoom + cy + self.pan_y

            if not (-40 <= sx_z < self.MAP_W + 40 and -40 <= sy_z < self.MAP_H + 40):
                continue

            v = normed[i]
            # Red-yellow-green gradient
            if v < 0.5:
                r = int(220 * (v * 2))
                g = int(60 + 160 * (v * 2))
                b = 60
            else:
                r = int(220 - 180 * ((v - 0.5) * 2))
                g = int(220)
                b = int(60 + 80 * ((v - 0.5) * 2))

            radius = int(max(8, 25 * self.zoom * (0.5 + 0.5 * v)))
            alpha = int(40 + 80 * v)

            glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow, (r, g, b, alpha), (radius * 2, radius * 2), radius * 2)
            pygame.draw.circle(glow, (r, g, b, min(255, alpha + 40)), (radius * 2, radius * 2), radius)
            heat_surface.blit(glow, (int(sx_z) - radius * 2, int(sy_z) - radius * 2))

        self.map_surface.blit(heat_surface, (0, 0))

        # Label in top-left
        label = self.font_med.render(
            f"HEATMAP: {self.heatmap_labels[self.heatmap_mode]}", True, COL_TEXT_ACCENT,
        )
        bg_rect = pygame.Surface((label.get_width() + 16, label.get_height() + 8), pygame.SRCALPHA)
        bg_rect.fill((20, 25, 40, 180))
        self.map_surface.blit(bg_rect, (8, 28))
        self.map_surface.blit(label, (16, 32))

    def _render_agents(self):
        """Render all alive agents as colored dots on the agent surface."""
        self.agent_surface.fill((0, 0, 0, 0))

        n = self.pool.next_id
        sx, sy, alive = self._get_agent_screen_positions()
        colors = self._get_agent_colors(n)

        # Use numpy to batch-filter valid positions
        valid = (
            alive
            & (sx >= 0) & (sx < self.MAP_W)
            & (sy >= 0) & (sy < self.MAP_H)
        )

        valid_x = sx[valid]
        valid_y = sy[valid]
        valid_colors = colors[valid]

        # Batch render using pixel array
        try:
            pixels = pygame.surfarray.pixels3d(self.agent_surface)

            # Draw 2x2 blocks for visibility
            for dx in range(2):
                for dy in range(2):
                    px = np.clip(valid_x + dx, 0, self.MAP_W - 1)
                    py = np.clip(valid_y + dy, 0, self.MAP_H - 1)
                    pixels[px, py] = valid_colors

            del pixels
        except Exception:
            # Fallback: draw individual dots
            for i in range(min(len(valid_x), 50000)):
                x, y = int(valid_x[i]), int(valid_y[i])
                c = (int(valid_colors[i][0]), int(valid_colors[i][1]), int(valid_colors[i][2]))
                self.agent_surface.set_at((x, y), (*c, 220))

        # Also set alpha for the agent surface
        alpha_arr = pygame.surfarray.pixels_alpha(self.agent_surface)
        alpha_arr[alpha_arr > 0] = 200
        del alpha_arr

    def _render_particles(self):
        """Render animation particles (births, deaths, marriages)."""
        for p in self.particles:
            if not (0 <= p.x < self.MAP_W and 0 <= p.y < self.MAP_H):
                continue
            r = max(1, int(p.radius * p.life))
            pygame.draw.circle(self.map_surface, p.color[:3], (int(p.x), int(p.y)), r)

    def _render_trails(self):
        """Render migration trails following road network paths with fading tail."""
        for trail in self.trails:
            cx, cy = trail.current_pos
            if not (-50 <= cx < self.MAP_W + 50 and -50 <= cy < self.MAP_H + 50):
                continue

            pts = trail.trail_points
            if len(pts) >= 2:
                int_pts = [(int(p[0]), int(p[1])) for p in pts]

                # Draw trail segments with fading from tail to head
                n_pts = len(int_pts)
                for i in range(n_pts - 1):
                    frac = i / max(n_pts - 1, 1)
                    brightness = 0.3 + 0.7 * frac
                    r = int(trail.color[0] * brightness)
                    g = int(trail.color[1] * brightness)
                    b = int(trail.color[2] * brightness)
                    width = 1 if frac < 0.5 else 2
                    pygame.draw.line(self.map_surface, (r, g, b),
                                     int_pts[i], int_pts[i + 1], width)

            # Glowing head dot
            r = max(2, int(4 * (1.0 - trail.progress)))
            pygame.draw.circle(self.map_surface, (255, 255, 255), (int(cx), int(cy)), r + 1)
            pygame.draw.circle(self.map_surface, trail.color, (int(cx), int(cy)), r)

    def _resize_window(self):
        """Adjust window width when policy panel toggles."""
        if self.policy_panel.visible:
            self.WINDOW_W = self.BASE_WINDOW_W + self.POLICY_PANEL_W
        else:
            self.WINDOW_W = self.BASE_WINDOW_W
        self.PANEL_X = (self.POLICY_PANEL_W if self.policy_panel.visible else 0) + self.MAP_W
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H), pygame.RESIZABLE)

    def _do_save_snapshot(self):
        try:
            path = self.model.save_snapshot(self.snapshot_path)
            msg = f"Saved @ {self.model.current_year} Q{self.model.current_step_in_year + 1}"
            self.policy_panel.show_snapshot_msg(msg)
            print(f"  [Snapshot] Saved to {path}")
        except Exception as e:
            self.policy_panel.show_snapshot_msg(f"Save FAILED: {e}")
            print(f"  [Snapshot] Save failed: {e}")

    def _do_load_snapshot(self):
        try:
            ok = self.model.load_snapshot(self.snapshot_path)
            if ok:
                self.pool = self.model.agents_pool
                self.loc = self.model.loc_state
                self.config = self.model.config
                self.prev_locations = self.pool.location[:self.pool.next_id].copy()
                self.prev_alive = self.pool.alive[:self.pool.next_id].copy()
                self.prev_n_children = self.pool.n_children[:self.pool.next_id].copy()
                self.prev_marital = self.pool.marital_status[:self.pool.next_id].copy()
                msg = f"Loaded @ {self.model.current_year} Q{self.model.current_step_in_year + 1}"
                self.policy_panel.show_snapshot_msg(msg)
                print(f"  [Snapshot] Loaded from {self.snapshot_path}")
            else:
                self.policy_panel.show_snapshot_msg("No snapshot found")
        except Exception as e:
            self.policy_panel.show_snapshot_msg(f"Load FAILED: {e}")
            print(f"  [Snapshot] Load failed: {e}")

    def _compute_live_tfr(self):
        """Estimate current TFR from recent births and female population."""
        n = self.pool.next_id
        alive = self.pool.alive[:n]
        female = self.pool.sex[:n] == 1
        ages = self.pool.age[:n]
        fertile_mask = alive & female & (ages >= 15) & (ages <= 49)
        n_fertile = max(int(fertile_mask.sum()), 1)
        annual_births = self.step_births * 4  # quarterly → annual estimate
        asfr_approx = annual_births / n_fertile
        tfr = asfr_approx * 35  # 35 years of fertility window
        return round(tfr, 3)

    def _compute_cannibalism_ratio(self):
        """
        Compute real-time "Regional Cannibalism" gauge:
        ratio of Periphery→Core moves to Core→Tokyo moves.
        A ratio > 1 means the Core is draining Periphery faster than Tokyo drains Core.
        """
        n = self.pool.next_id
        alive_now = self.pool.alive[:n]
        alive_before = self.prev_alive[:min(len(self.prev_alive), n)]
        loc_now = self.pool.location[:n]
        loc_before = self.prev_locations[:min(len(self.prev_locations), n)]
        nn = min(len(alive_before), n)

        migrated = alive_now[:nn] & alive_before[:nn] & (loc_now[:nn] != loc_before[:nn])
        if not migrated.any():
            return 0.0

        tier_from = self.loc.tier[loc_before[:nn][migrated]]
        tier_to = self.loc.tier[loc_now[:nn][migrated]]

        peri_to_core = int(((tier_from == 2) & (tier_to == 1)).sum())
        core_to_tokyo = int(((tier_from == 1) & (tier_to == 0)).sum())
        self.peri_to_core_step = peri_to_core
        self.core_to_tokyo_step = core_to_tokyo
        return peri_to_core / max(core_to_tokyo, 1)

    def _render_hud_metrics(self, panel, x, y):
        """Render HUD metrics: TFR sparkline, cannibalism gauge."""
        # TFR sparkline
        self._draw_sparkline(panel, x, y, "Live TFR", self.tfr_history, (255, 180, 60), 35)
        y += 50

        # Cannibalism gauge
        if self.cannibalism_history:
            ratio = self.cannibalism_history[-1]
        else:
            ratio = 0.0
        gauge_w = self.PANEL_W - 2 * x - 10
        lbl = self.font_tiny.render(
            f"Cannibalism Gauge  P→C: {self.peri_to_core_step}  C→T: {self.core_to_tokyo_step}",
            True, COL_TEXT_DIM,
        )
        panel.blit(lbl, (x, y))
        y += 16

        # Draw gauge bar: 0..5+ scale
        capped = min(ratio, 5.0)
        frac = capped / 5.0
        pygame.draw.rect(panel, (40, 45, 60), (x, y, gauge_w, 10), border_radius=4)

        if frac > 0:
            # Green → Yellow → Red gradient
            if frac < 0.4:
                gc = (80, 220, 120)
            elif frac < 0.7:
                gc = (255, 220, 60)
            else:
                gc = (255, 80, 60)
            fill_w = max(1, int(gauge_w * frac))
            pygame.draw.rect(panel, gc, (x, y, fill_w, 10), border_radius=4)

        val_str = f"{ratio:.2f}"
        val_surf = self.font_tiny.render(val_str, True, COL_TEXT_ACCENT)
        panel.blit(val_surf, (x + gauge_w - val_surf.get_width(), y - 16))
        y += 18

        # Cannibalism sparkline
        self._draw_sparkline(panel, x, y, "Cannibalism Ratio", self.cannibalism_history, (255, 100, 80), 30)
        y += 42

        return y

    def _render_panel(self):
        """Render the statistics and control panel on the right side."""
        panel = pygame.Surface((self.PANEL_W, self.WINDOW_H))
        panel.fill(COL_PANEL_BG)
        pygame.draw.line(panel, COL_PANEL_BORDER, (0, 0), (0, self.WINDOW_H), 2)

        y = 15
        pad = 20

        # Title
        title = self.font_large.render("JAPAN EXODUS", True, COL_TEXT)
        panel.blit(title, (pad, y))
        y += 35

        subtitle = self.font_small.render("Agent-Based Microsimulation", True, COL_TEXT_DIM)
        panel.blit(subtitle, (pad, y))
        y += 30

        # Separator
        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 15

        # Year/Quarter
        year = self.model.current_year
        quarter = self.model.current_step_in_year + 1
        year_text = self.font_large.render(f"{year} Q{quarter}", True, COL_TEXT_ACCENT)
        panel.blit(year_text, (pad, y))
        y += 40

        # Speed indicator
        speed = self.STEPS_PER_FRAME_OPTIONS[self.speed_idx]
        speed_str = f"{'PAUSED' if self.paused else f'Speed: {speed}x'}"
        speed_col = COL_TEXT_WARN if self.paused else COL_TEXT_DIM
        panel.blit(self.font_small.render(speed_str, True, speed_col), (pad, y))
        y += 25

        # Step counter
        panel.blit(self.font_tiny.render(f"Step: {self.model.total_steps}", True, COL_TEXT_DIM), (pad, y))
        y += 22

        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 15

        # Population stats
        n_alive = int(self.pool.alive[:self.pool.next_id].sum())
        scale = self.config.scale.agent_scale
        real_pop = n_alive * scale

        self._draw_stat(panel, pad, y, "TOTAL POPULATION",
                        f"{real_pop:,.0f}", COL_TEXT, big=True)
        y += 45
        self._draw_stat(panel, pad, y, "Agents Simulated", f"{n_alive:,}", COL_TEXT_DIM)
        y += 28

        # Population by tier
        pop_by_loc = self.pool.get_population_by_location(self.config.geography.n_locations)
        geo = self.config.geography
        t_end = geo.n_tokyo_wards
        c_end = t_end + geo.n_core_cities

        tokyo_pop = int(pop_by_loc[:t_end].sum())
        core_pop = int(pop_by_loc[t_end:c_end].sum())
        peri_pop = int(pop_by_loc[c_end:].sum())
        total = max(tokyo_pop + core_pop + peri_pop, 1)

        y += 5
        self._draw_bar(panel, pad, y, "Tokyo", tokyo_pop, total, COL_TOKYO, scale)
        y += 32
        self._draw_bar(panel, pad, y, "Core Cities", core_pop, total, COL_CORE, scale)
        y += 32
        self._draw_bar(panel, pad, y, "Periphery", peri_pop, total, COL_PERIPHERY, scale)
        y += 38

        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 15

        # Events this step
        self._draw_stat(panel, pad, y, "EVENTS THIS STEP", "", COL_TEXT_ACCENT)
        y += 22
        self._draw_event(panel, pad, y, "Births", self.step_births, COL_AGENT_BIRTH)
        y += 20
        self._draw_event(panel, pad, y, "Deaths", self.step_deaths, COL_AGENT_DEATH)
        y += 20
        self._draw_event(panel, pad, y, "Migrations", self.step_migrations, COL_AGENT_MIGRATE)
        y += 20
        self._draw_event(panel, pad, y, "Marriages", self.step_marriages, COL_AGENT_MARRY)
        y += 30

        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 15

        # Demographics
        stats = self.pool.get_statistics()
        self._draw_stat(panel, pad, y, "DEMOGRAPHICS", "", COL_TEXT_ACCENT)
        y += 22
        self._draw_stat(panel, pad, y, "Mean Age", f"{stats.get('mean_age', 0):.1f} yrs", COL_TEXT)
        y += 20
        self._draw_stat(panel, pad, y, "% Married", f"{stats.get('pct_married', 0):.1%}", COL_TEXT)
        y += 20
        self._draw_stat(panel, pad, y, "% Remote Work", f"{stats.get('pct_remote', 0):.1%}", COL_TEXT)
        y += 20
        self._draw_stat(panel, pad, y, "Mean Income",
                        f"¥{stats.get('mean_income', 0):,.0f}", COL_TEXT)
        y += 30

        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 15

        # Sparklines
        self._draw_sparkline(panel, pad, y, "Population", self.pop_history, COL_TEXT_ACCENT, 35)
        y += 45
        self._draw_sparkline(panel, pad, y, "Births", self.birth_history, COL_AGENT_BIRTH, 28)
        y += 38
        self._draw_sparkline(panel, pad, y, "Migrations", self.migration_history, COL_AGENT_MIGRATE, 28)
        y += 40

        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 10

        # --- HUD: Live TFR + Cannibalism Gauge ---
        hud_label = self.font_tiny.render("LIVE METRICS HUD", True, COL_TEXT_ACCENT)
        panel.blit(hud_label, (pad, y))
        y += 16
        y = self._render_hud_metrics(panel, pad, y)

        pygame.draw.line(panel, COL_PANEL_BORDER, (pad, y), (self.PANEL_W - pad, y), 1)
        y += 8

        # Controls help
        controls = [
            "SPACE: Pause/Resume",
            "UP/DOWN: Speed   T: Color",
            "H: Heatmap   P: Policy Panel",
            "+/-: Zoom   R: Reset view",
            "F5: Save snap   F6: Load snap",
            "CLICK: Inspect  ESC: Quit",
        ]
        for ctrl in controls:
            panel.blit(self.font_tiny.render(ctrl, True, COL_TEXT_DIM), (pad, y))
            y += 14

        # Color mode indicator
        y += 3
        modes = ["Action/Tier", "Tier (vivid)", "Age gradient", "Utility"]
        mode_text = f"Color: {modes[self.color_mode]}"
        panel.blit(self.font_small.render(mode_text, True, COL_TEXT_ACCENT), (pad, y))

        self.screen.blit(panel, (self.PANEL_X, 0))

    def _draw_stat(self, surface, x, y, label, value, color, big=False):
        font = self.font_med if big else self.font_small
        lbl = self.font_tiny.render(label, True, COL_TEXT_DIM)
        val = font.render(str(value), True, color)
        surface.blit(lbl, (x, y))
        surface.blit(val, (x, y + 14))

    def _draw_event(self, surface, x, y, label, count, color):
        # Color dot
        pygame.draw.circle(surface, color, (x + 6, y + 7), 5)
        lbl = self.font_small.render(f"  {label}: {count:,}", True, COL_TEXT)
        surface.blit(lbl, (x + 14, y))

    def _draw_bar(self, surface, x, y, label, value, total, color, scale):
        bar_w = self.PANEL_W - 2 * x - 10
        pct = value / max(total, 1)
        filled = int(bar_w * pct)

        # Label and value
        lbl = self.font_tiny.render(f"{label}", True, COL_TEXT_DIM)
        val = self.font_tiny.render(f"{value * scale / 1e6:.1f}M ({pct:.1%})", True, COL_TEXT)
        surface.blit(lbl, (x, y))
        surface.blit(val, (x + 100, y))

        # Bar background
        pygame.draw.rect(surface, (40, 45, 60), (x, y + 16, bar_w, 8), border_radius=3)
        if filled > 0:
            pygame.draw.rect(surface, color, (x, y + 16, filled, 8), border_radius=3)

    def _draw_sparkline(self, surface, x, y, label, data, color, height):
        if len(data) < 2:
            return

        lbl = self.font_tiny.render(label, True, COL_TEXT_DIM)
        surface.blit(lbl, (x, y))

        values = list(data)
        w = self.PANEL_W - 2 * x - 10
        h = height
        min_v = min(values)
        max_v = max(values)
        if max_v == min_v:
            max_v = min_v + 1

        points = []
        for i, v in enumerate(values):
            px = x + int(i / max(len(values) - 1, 1) * w)
            py = y + 14 + h - int((v - min_v) / (max_v - min_v) * h)
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 2)

        # Current value
        cur = self.font_tiny.render(f"{values[-1]:,.0f}", True, color)
        surface.blit(cur, (x + w - 60, y))

    def _handle_events(self):
        """Process Pygame events (keyboard, mouse, window)."""
        map_offset_x = self.POLICY_PANEL_W if self.policy_panel.visible else 0

        for event in pygame.event.get():
            # Let policy panel consume mouse events first
            if self.policy_panel.visible and event.type in (
                pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP
            ):
                if self.policy_panel.handle_event(event, offset_x=0):
                    continue

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed_idx = min(self.speed_idx + 1, len(self.STEPS_PER_FRAME_OPTIONS) - 1)
                elif event.key == pygame.K_DOWN:
                    self.speed_idx = max(self.speed_idx - 1, 0)
                elif event.key == pygame.K_t:
                    self.color_mode = (self.color_mode + 1) % 4
                elif event.key == pygame.K_h:
                    self.heatmap_mode = (self.heatmap_mode + 1) % len(self.heatmap_labels)
                elif event.key == pygame.K_d:
                    self.dark_mode = not self.dark_mode
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_p:
                    self.policy_panel.toggle()
                    self._resize_window()
                elif event.key == pygame.K_r:
                    self.zoom = 1.0
                    self.pan_x = 0.0
                    self.pan_y = 0.0
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.zoom = min(self.zoom * 1.3, 8.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.zoom = max(self.zoom / 1.3, 0.5)
                elif event.key == pygame.K_LEFT:
                    self.pan_x += 50
                elif event.key == pygame.K_RIGHT:
                    self.pan_x -= 50
                elif event.key == pygame.K_F5:
                    self._do_save_snapshot()
                elif event.key == pygame.K_F6:
                    self._do_load_snapshot()

            elif event.type == pygame.MOUSEWHEEL:
                old_zoom = self.zoom
                if event.y > 0:
                    self.zoom = min(self.zoom * 1.15, 8.0)
                else:
                    self.zoom = max(self.zoom / 1.15, 0.5)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mx, my = event.pos
                    mx -= map_offset_x
                    if 0 <= mx < self.MAP_W:
                        self._inspect_at(mx, my)

    def _inspect_at(self, mx, my):
        """Find and inspect the nearest location or agent to click position."""
        # Try agent first (more specific)
        n = self.pool.next_id
        sx, sy, alive = self._get_agent_screen_positions()
        if alive.any():
            dx = sx.astype(np.float64) - mx
            dy = sy.astype(np.float64) - my
            dists = dx * dx + dy * dy
            dists[~alive] = 1e12
            best_agent = int(np.argmin(dists))
            if dists[best_agent] < 100:
                self.inspected_agent = best_agent
                self.inspected_loc = None
                return

        self.inspected_agent = None

        # Fall back to location
        best_dist = 999999
        best_loc = None
        for i in range(self.config.geography.n_locations):
            lx = (self.loc_screen_x[i] - self.MAP_W / 2) * self.zoom + self.MAP_W / 2 + self.pan_x
            ly = (self.loc_screen_y[i] - self.MAP_H / 2) * self.zoom + self.MAP_H / 2 + self.pan_y
            d = (mx - lx) ** 2 + (my - ly) ** 2
            if d < best_dist:
                best_dist = d
                best_loc = i

        if best_dist < 400:
            self.inspected_loc = best_loc
        else:
            self.inspected_loc = None

    def _render_inspection(self):
        """Show detailed info for inspected location or agent."""
        if self.inspected_agent is not None:
            self._render_agent_inspection()
            return

        if self.inspected_loc is None:
            return

        i = self.inspected_loc
        tier_names = {0: "TOKYO", 1: "CORE CITY", 2: "PERIPHERY"}
        tier = self.loc.tier[i]
        pop = self.loc.population[i]
        cap = self.loc.capacity[i]
        scale = self.config.scale.agent_scale

        sx = (self.loc_screen_x[i] - self.MAP_W / 2) * self.zoom + self.MAP_W / 2 + self.pan_x
        sy = (self.loc_screen_y[i] - self.MAP_H / 2) * self.zoom + self.MAP_H / 2 + self.pan_y

        box_w, box_h = 240, 210
        bx = min(int(sx) + 15, self.MAP_W - box_w - 5)
        by = max(int(sy) - 50, 5)

        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((20, 25, 40, 220))
        pygame.draw.rect(box, TIER_COLORS_PG[tier], (0, 0, box_w, box_h), 2, border_radius=5)

        occupancy = pop / max(cap, 1)
        occ_color = COL_AGENT_DEATH if occupancy > 1.1 else (COL_TEXT_WARN if occupancy > 0.9 else COL_AGENT_BIRTH)

        lines = [
            (f"Location #{i}", COL_TEXT_ACCENT, self.font_small),
            (f"Type: {tier_names[tier]}", TIER_COLORS_PG[tier], self.font_small),
            (f"Pop: {pop:,} (real: {pop * scale:,})", COL_TEXT, self.font_small),
            (f"Capacity: {cap:,} ({occupancy:.0%})", occ_color, self.font_tiny),
            (f"Prestige:   {self.loc.prestige[i]:.3f}", COL_TEXT, self.font_tiny),
            (f"Convenience:{self.loc.convenience[i]:.3f}", COL_TEXT, self.font_tiny),
            (f"Anomie:     {self.loc.anomie[i]:.3f}", COL_TEXT_WARN, self.font_tiny),
            (f"Fin.Frict:  {self.loc.financial_friction[i]:.3f}", COL_TEXT, self.font_tiny),
            (f"Rent Index: {self.loc.rent_index[i]:.2f}", COL_TEXT, self.font_tiny),
            (f"Vacancy:    {self.loc.vacancy_rate[i]:.1%}", COL_TEXT, self.font_tiny),
            (f"Healthcare: {self.loc.healthcare_score[i]:.3f}", COL_TEXT, self.font_tiny),
            (f"Digital:    {self.loc.digital_access[i]:.3f}", COL_TEXT, self.font_tiny),
        ]

        ty = 8
        for text, color, font in lines:
            rendered = font.render(text, True, color)
            box.blit(rendered, (8, ty))
            ty += 16

        self.map_surface.blit(box, (bx, by))
        pygame.draw.circle(self.map_surface, TIER_COLORS_PG[tier],
                           (int(sx), int(sy)), 12, 2)

    def _render_agent_inspection(self):
        """Show detailed info for a clicked individual agent."""
        ai = self.inspected_agent
        pool = self.pool
        if ai >= pool.next_id or not pool.alive[ai]:
            self.inspected_agent = None
            return

        loc_id = pool.location[ai]
        tier = self.loc.tier[loc_id]
        tier_names = {0: "Tokyo", 1: "Core City", 2: "Periphery"}
        edu_names = {0: "None", 1: "High School", 2: "University", 3: "Graduate"}
        marital_names = {0: "Single", 1: "Married", 2: "Divorced", 3: "Widowed"}
        sex_names = {0: "Male", 1: "Female"}

        sx, sy, _ = self._get_agent_screen_positions()
        ax, ay = int(sx[ai]), int(sy[ai])

        box_w, box_h = 250, 240
        bx = min(ax + 15, self.MAP_W - box_w - 5)
        by = max(ay - 50, 5)

        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((15, 20, 35, 230))
        border_col = (255, 200, 50)
        pygame.draw.rect(box, border_col, (0, 0, box_w, box_h), 2, border_radius=5)

        lines = [
            (f"Agent #{ai}", border_col, self.font_small),
            (f"{sex_names.get(pool.sex[ai], '?')}, Age {pool.age[ai]}", COL_TEXT, self.font_small),
            (f"Education: {edu_names.get(pool.education[ai], '?')}", COL_TEXT, self.font_tiny),
            (f"Status: {marital_names.get(pool.marital_status[ai], '?')}", COL_TEXT, self.font_tiny),
            (f"Children: {pool.n_children[ai]}", COL_TEXT, self.font_tiny),
            (f"Income: ¥{pool.income[ai]:,.0f}", COL_TEXT, self.font_tiny),
            (f"Location: {tier_names.get(tier, '?')} #{loc_id}", TIER_COLORS_PG.get(tier, COL_TEXT), self.font_tiny),
            (f"Cultural Orient: {pool.cultural_orient[ai]:.2f}", COL_TEXT, self.font_tiny),
            (f"Utility Memory: {pool.utility_memory[ai]:.3f}", COL_TEXT, self.font_tiny),
            (f"Remote Worker: {'Yes' if pool.remote_worker[ai] else 'No'}", COL_TEXT, self.font_tiny),
            (f"In University: {'Yes' if pool.in_university[ai] else 'No'}", COL_TEXT_ACCENT if pool.in_university[ai] else COL_TEXT, self.font_tiny),
            (f"Migration CD: {pool.migration_cooldown[ai]}q", COL_TEXT, self.font_tiny),
            (f"Fertility Intent: {pool.fertility_intent[ai]:.2f}", COL_TEXT, self.font_tiny),
        ]

        ty = 8
        for text, color, font in lines:
            rendered = font.render(text, True, color)
            box.blit(rendered, (8, ty))
            ty += 17

        self.map_surface.blit(box, (bx, by))

        # Highlight the agent with a crosshair
        pygame.draw.circle(self.map_surface, border_col, (ax, ay), 8, 2)
        pygame.draw.line(self.map_surface, border_col, (ax - 12, ay), (ax + 12, ay), 1)
        pygame.draw.line(self.map_surface, border_col, (ax, ay - 12), (ax, ay + 12), 1)

    def _render_timeline(self):
        """Render a timeline bar at the bottom of the map."""
        bar_h = 28
        bar_y = self.MAP_H - bar_h
        bar_w = self.MAP_W

        bar_bg = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        bar_bg.fill((15, 18, 28, 200))
        self.map_surface.blit(bar_bg, (0, bar_y))

        total_years = self.config.scale.n_years
        start_year = self.config.scale.start_year
        current_year = self.model.current_year
        current_q = self.model.current_step_in_year

        progress = ((current_year - start_year) * 4 + current_q) / (total_years * 4)
        progress = min(1.0, max(0.0, progress))

        pad = 80
        track_w = bar_w - 2 * pad

        # Track background
        pygame.draw.rect(self.map_surface, (40, 45, 60),
                         (pad, bar_y + 10, track_w, 6), border_radius=3)

        # Filled portion
        fill_w = int(track_w * progress)
        if fill_w > 0:
            pygame.draw.rect(self.map_surface, COL_TEXT_ACCENT,
                             (pad, bar_y + 10, fill_w, 6), border_radius=3)

        # Decade markers
        for yr in range(start_year, start_year + total_years + 1, 10):
            frac = (yr - start_year) / total_years
            mx = pad + int(track_w * frac)
            pygame.draw.line(self.map_surface, COL_TEXT_DIM,
                             (mx, bar_y + 6), (mx, bar_y + 20), 1)
            lbl = self.font_tiny.render(str(yr), True, COL_TEXT_DIM)
            self.map_surface.blit(lbl, (mx - 12, bar_y + 2))

        # Current position marker
        cx = pad + int(track_w * progress)
        pygame.draw.circle(self.map_surface, (255, 255, 255), (cx, bar_y + 13), 5)
        pygame.draw.circle(self.map_surface, COL_TEXT_ACCENT, (cx, bar_y + 13), 3)

        # Year label left
        yr_lbl = self.font_small.render(
            f"{current_year} Q{current_q + 1}", True, COL_TEXT_ACCENT,
        )
        self.map_surface.blit(yr_lbl, (5, bar_y + 5))

    def _render_year_summary(self):
        """Show a brief year-end summary overlay."""
        if not self.year_summary_data:
            return

        d = self.year_summary_data
        alpha = min(255, self.year_summary_timer * 4)

        box_w, box_h = 350, 180
        bx = (self.MAP_W - box_w) // 2
        by = 60

        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((15, 20, 35, min(210, alpha)))
        pygame.draw.rect(box, COL_TEXT_ACCENT, (0, 0, box_w, box_h), 2, border_radius=8)

        ty = 12
        title = self.font_med.render(f"Year {d.get('year', '?')} Summary", True, COL_TEXT_ACCENT)
        box.blit(title, (12, ty))
        ty += 30

        scale = self.config.scale.agent_scale
        lines = [
            f"Population: {d.get('pop', 0) * scale:,.0f}",
            f"Births: {d.get('births', 0):,}  Deaths: {d.get('deaths', 0):,}",
            f"Net Migration: {d.get('net_mig', 0):+,}",
            f"Tokyo Share: {d.get('tokyo_pct', 0):.1%}",
            f"Mean Age: {d.get('mean_age', 0):.1f}",
            f"Marriage Rate: {d.get('married_pct', 0):.1%}",
        ]

        for line in lines:
            txt = self.font_small.render(line, True, COL_TEXT)
            box.blit(txt, (12, ty))
            ty += 22

        self.map_surface.blit(box, (bx, by))

    def _sim_step(self):
        """Execute one simulation step and detect events."""
        prev_year = self.model.current_year

        self.model.step()
        self._detect_events()

        # Update sparkline data
        n_alive = int(self.pool.alive[:self.pool.next_id].sum())
        self.pop_history.append(n_alive * self.config.scale.agent_scale)
        self.birth_history.append(self.step_births)
        self.death_history.append(self.step_deaths)
        self.migration_history.append(self.step_migrations)

        # HUD metrics
        tfr = self._compute_live_tfr()
        self.tfr_history.append(tfr)
        cann = self._compute_cannibalism_ratio()
        self.cannibalism_history.append(cann)

        # Year-end summary trigger
        if self.model.current_year != prev_year:
            stats = self.pool.get_statistics()
            pop_by_loc = self.pool.get_population_by_location(self.config.geography.n_locations)
            geo = self.config.geography
            total_pop = int(self.pool.alive[:self.pool.next_id].sum())
            tokyo_pop = int(pop_by_loc[:geo.n_tokyo_wards].sum())
            self.year_summary_data = {
                "year": prev_year,
                "pop": total_pop,
                "births": sum(list(self.birth_history)[-4:]),
                "deaths": sum(list(self.death_history)[-4:]),
                "net_mig": 0,
                "tokyo_pct": tokyo_pop / max(total_pop, 1),
                "mean_age": stats.get("mean_age", 0),
                "married_pct": stats.get("pct_married", 0),
            }
            self.year_summary_timer = 120

    def run(self):
        """Main visualization loop."""
        print("\n" + "=" * 60)
        print("  LIVE VISUALIZATION STARTED")
        print("  Press SPACE to pause, ESC to quit")
        print("=" * 60 + "\n")

        while self.running:
            self._handle_events()

            # Simulation stepping (based on speed)
            if not self.paused:
                speed = self.STEPS_PER_FRAME_OPTIONS[self.speed_idx]
                self.step_accumulator += speed / self.FPS

                while self.step_accumulator >= 1.0:
                    self.step_accumulator -= 1.0
                    try:
                        self._sim_step()
                    except Exception as e:
                        print(f"Simulation error: {e}")
                        self.paused = True
                        break

            # Update animations
            self._update_animations()

            # === RENDER ===
            self.screen.fill((12, 15, 22))

            # Policy panel on left (if visible)
            map_x_offset = 0
            if self.policy_panel.visible:
                pp_surf = self.policy_panel.draw()
                if pp_surf is not None:
                    self.screen.blit(pp_surf, (0, 0))
                map_x_offset = self.POLICY_PANEL_W

            self._render_map_background()
            self._render_heatmap()
            self._render_agents()

            # Composite layers
            self.map_surface.blit(self.agent_surface, (0, 0))

            if self.show_trails:
                self._render_trails()
            self._render_particles()
            self._render_inspection()

            # FPS counter on map
            fps = self.clock.get_fps()
            fps_text = self.font_tiny.render(f"FPS: {fps:.0f}", True, COL_TEXT_DIM)
            self.map_surface.blit(fps_text, (self.MAP_W - 70, 5))

            # Timeline bar at bottom
            if self.show_timeline:
                self._render_timeline()

            # Year-end summary overlay
            if self.year_summary_timer > 0:
                self._render_year_summary()
                self.year_summary_timer -= 1

            # Blit map to screen (offset by policy panel)
            self.screen.blit(self.map_surface, (map_x_offset, 0))

            # Stats panel (right)
            self._render_panel()

            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
        print("Visualization closed.")
        return self.model.get_results()
