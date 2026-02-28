"""
Japanese expressway and national highway network.

Real waypoints for the major expressway system (高速道路) tracing actual routes.
Each route is a list of (longitude, latitude) waypoints sampled along the road.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict

# ===================================================================
# Major Expressways — waypoints sampled every ~30-50km along route
# ===================================================================
EXPRESSWAYS: Dict[str, List[Tuple[float, float]]] = {

    # --- Tohoku Expressway (東北自動車道) Tokyo → Aomori ---
    "tohoku_exp": [
        (139.74, 35.68),  # Tokyo (Kawaguchi JCT)
        (139.75, 35.86),  # Urawa
        (139.74, 36.05),  # Kuki
        (139.78, 36.25),  # Kazo
        (139.78, 36.39),  # Utsunomiya area
        (139.88, 36.57),  # Utsunomiya
        (139.93, 36.77),  # Yaita
        (139.93, 36.95),  # Nasushiobara
        (140.00, 37.10),  # Shirakawa
        (140.10, 37.28),  # Sukagawa
        (140.20, 37.40),  # Koriyama
        (140.34, 37.58),  # Motomiya
        (140.38, 37.75),  # Fukushima
        (140.50, 38.00),  # Shiroishi
        (140.72, 38.15),  # Murata
        (140.87, 38.27),  # Sendai
        (140.93, 38.45),  # Furukawa
        (140.95, 38.73),  # Ichinoseki
        (141.05, 38.90),  # Mizusawa
        (141.12, 39.10),  # Kitakami
        (141.15, 39.30),  # Hanamaki
        (141.15, 39.50),  # Morioka-minami
        (141.15, 39.70),  # Morioka
        (140.95, 39.90),  # Takizawa
        (140.80, 40.10),  # Kazuno
        (140.60, 40.30),  # Kosaka
        (140.50, 40.50),  # Towada
        (140.45, 40.70),  # Shichinohe
        (140.46, 40.82),  # Aomori
    ],

    # --- Tomei Expressway (東名高速道路) Tokyo → Nagoya ---
    "tomei_exp": [
        (139.74, 35.68),  # Tokyo IC
        (139.66, 35.58),  # Yokohama-Machida
        (139.47, 35.45),  # Atsugi
        (139.30, 35.35),  # Hadano-Nakai
        (139.15, 35.25),  # Gotemba
        (139.00, 35.20),  # Susono
        (138.90, 35.12),  # Numazu
        (138.70, 35.05),  # Fuji
        (138.52, 34.98),  # Shimizu
        (138.38, 34.98),  # Shizuoka
        (138.20, 34.90),  # Yaizu
        (138.05, 34.82),  # Kakegawa
        (137.88, 34.78),  # Fukuroi
        (137.72, 34.75),  # Hamamatsu
        (137.55, 34.72),  # Mikkabi
        (137.40, 34.85),  # Toyohashi area
        (137.25, 34.92),  # Gamagori
        (137.10, 34.98),  # Okazaki
        (136.97, 35.05),  # Toyota
        (136.91, 35.12),  # Nagoya IC
    ],

    # --- Meishin Expressway (名神高速道路) Nagoya → Kobe ---
    "meishin_exp": [
        (136.91, 35.12),  # Komaki (Nagoya)
        (136.75, 35.18),  # Ichinomiya
        (136.60, 35.25),  # Gifu-Hashima
        (136.42, 35.28),  # Sekigahara
        (136.28, 35.22),  # Maibara
        (136.15, 35.10),  # Hikone
        (136.05, 35.02),  # Otsu
        (135.85, 35.00),  # Kyoto-minami
        (135.72, 34.95),  # Kyoto
        (135.58, 34.85),  # Takatsuki
        (135.50, 34.78),  # Suita (Osaka)
        (135.43, 34.72),  # Amagasaki
        (135.30, 34.70),  # Nishinomiya
        (135.20, 34.69),  # Kobe
    ],

    # --- San'yo Expressway (山陽自動車道) Kobe → Shimonoseki ---
    "sanyo_exp": [
        (135.20, 34.69),  # Kobe JCT
        (135.05, 34.65),  # Akashi
        (134.85, 34.68),  # Kakogawa
        (134.69, 34.83),  # Himeji
        (134.45, 34.85),  # Tatsuno
        (134.20, 34.80),  # Bizen
        (133.93, 34.67),  # Okayama
        (133.65, 34.55),  # Kurashiki
        (133.45, 34.50),  # Fukuyama
        (133.20, 34.45),  # Onomichi
        (133.05, 34.42),  # Mihara
        (132.82, 34.38),  # Higashi-Hiroshima
        (132.46, 34.40),  # Hiroshima
        (132.20, 34.35),  # Hatsukaichi
        (132.05, 34.30),  # Otake
        (131.85, 34.20),  # Iwakuni
        (131.60, 34.10),  # Kudamatsu
        (131.45, 34.05),  # Tokuyama
        (131.25, 34.00),  # Hofu
        (131.10, 33.97),  # Yamaguchi
        (130.95, 33.95),  # Ube
        (131.00, 33.95),  # Shimonoseki
    ],

    # --- Chuo Expressway (中央自動車道) Tokyo → Nagoya (inland) ---
    "chuo_exp": [
        (139.66, 35.68),  # Takaido (Tokyo)
        (139.48, 35.69),  # Hachioji
        (139.28, 35.65),  # Hachioji-nishi
        (139.10, 35.63),  # Otsuki
        (138.90, 35.66),  # Kofu area
        (138.57, 35.66),  # Kofu
        (138.35, 35.70),  # Suwa
        (138.15, 35.85),  # Okaya
        (137.97, 35.88),  # Ina
        (137.82, 35.75),  # Komagane
        (137.60, 35.60),  # Iida
        (137.42, 35.45),  # Achi
        (137.28, 35.35),  # Nakatsugawa
        (137.10, 35.25),  # Tajimi
        (136.97, 35.18),  # Komaki (Nagoya)
    ],

    # --- Kan-Etsu Expressway (関越自動車道) Tokyo → Niigata ---
    "kanetsu_exp": [
        (139.68, 35.73),  # Nerima (Tokyo)
        (139.60, 35.83),  # Tokorozawa
        (139.48, 35.95),  # Kawagoe
        (139.38, 36.10),  # Higashi-Matsuyama
        (139.25, 36.25),  # Honjo-Kodama
        (139.02, 36.40),  # Takasaki
        (138.90, 36.55),  # Shibukawa
        (138.85, 36.72),  # Numata
        (138.82, 36.88),  # Minakami
        (138.85, 36.98),  # Yuzawa (tunnel)
        (138.90, 37.12),  # Muikamachi
        (139.00, 37.30),  # Ojiya
        (139.05, 37.50),  # Nagaoka
        (139.08, 37.70),  # Mitsuke
        (139.02, 37.92),  # Niigata
    ],

    # --- Joban Expressway (常磐自動車道) Tokyo → Sendai ---
    "joban_exp": [
        (139.82, 35.76),  # Misato (Tokyo area)
        (139.95, 35.83),  # Kashiwa
        (140.05, 35.90),  # Tsukuba JCT area
        (140.13, 36.05),  # Tsuchiura
        (140.20, 36.20),  # Ishioka
        (140.28, 36.40),  # Mito
        (140.42, 36.58),  # Hitachi-minami
        (140.55, 36.72),  # Hitachi
        (140.65, 36.90),  # Takahagi
        (140.72, 37.05),  # Iwaki
        (140.80, 37.25),  # Tomioka
        (140.88, 37.50),  # Minami-Soma
        (140.92, 37.75),  # Soma
        (140.90, 38.00),  # Watari
        (140.87, 38.27),  # Sendai
    ],

    # --- Hokuriku Expressway (北陸自動車道) Niigata → Maibara ---
    "hokuriku_exp": [
        (139.02, 37.92),  # Niigata
        (138.85, 37.72),  # Tsubame-Sanjo
        (138.72, 37.55),  # Nagaoka
        (138.40, 37.35),  # Kashiwazaki
        (138.22, 37.12),  # Joetsu
        (137.88, 36.95),  # Itoigawa
        (137.60, 36.85),  # Asahi
        (137.40, 36.78),  # Nyuzen
        (137.22, 36.72),  # Kurobe
        (137.10, 36.70),  # Uozu
        (137.00, 36.68),  # Toyama
        (136.82, 36.65),  # Tonami
        (136.65, 36.57),  # Kanazawa area
        (136.62, 36.56),  # Kanazawa
        (136.42, 36.42),  # Komatsu
        (136.28, 36.28),  # Kaga
        (136.18, 36.15),  # Tsuruga area
        (136.15, 35.92),  # Tsuruga
        (136.22, 35.60),  # Nagahama
        (136.28, 35.40),  # Maibara
    ],

    # --- Kyushu Expressway (九州自動車道) Kitakyushu → Kagoshima ---
    "kyushu_exp": [
        (130.85, 33.88),  # Kitakyushu
        (130.72, 33.78),  # Nogata
        (130.60, 33.65),  # Tosu JCT
        (130.52, 33.55),  # Kurume
        (130.62, 33.40),  # Yame
        (130.70, 33.25),  # Yamaga
        (130.74, 33.10),  # Kumamoto
        (130.72, 32.95),  # Uto
        (130.68, 32.80),  # Yatsushiro
        (130.60, 32.60),  # Hitoyoshi area
        (130.62, 32.45),  # Ebino
        (130.72, 32.28),  # Kobayashi
        (130.85, 32.10),  # Miyakonojo
        (130.95, 31.90),  # Kirishima
        (130.72, 31.70),  # Kajiki
        (130.56, 31.60),  # Kagoshima
    ],

    # --- Fukuoka Urban (福岡都市高速 + 西九州) ---
    "fukuoka_nishi": [
        (130.85, 33.88),  # Kitakyushu
        (130.70, 33.80),  # Wakamatsu
        (130.52, 33.70),  # Iizuka
        (130.42, 33.59),  # Fukuoka
        (130.28, 33.50),  # Saga area
        (130.10, 33.35),  # Takeo
        (129.95, 33.20),  # Hasami
        (129.88, 32.75),  # Nagasaki
    ],

    # --- Hokkaido Expressway (道央自動車道) ---
    "hokkaido_exp": [
        (140.73, 41.77),  # Hakodate area (Mori)
        (140.65, 41.90),  # Yakumo
        (140.55, 42.10),  # Oshamanbe
        (140.45, 42.32),  # Date
        (141.00, 42.33),  # Tomakomai
        (141.25, 42.55),  # Eniwa
        (141.35, 42.78),  # Kitahiroshima
        (141.35, 43.06),  # Sapporo
        (141.60, 43.20),  # Ebetsu
        (141.80, 43.32),  # Iwamizawa
        (142.00, 43.35),  # Mikasa
        (142.37, 43.34),  # Asahikawa direction
        (142.37, 43.77),  # Asahikawa
    ],

    # --- Sapporo → Obihiro/Kushiro ---
    "doto_exp": [
        (141.35, 43.06),  # Sapporo
        (141.65, 42.95),  # Chitose
        (142.10, 42.92),  # Yubari area
        (142.45, 42.93),  # Shimizu
        (143.20, 42.92),  # Obihiro
        (143.80, 42.95),  # Ikeda
        (144.37, 42.98),  # Kushiro
    ],

    # --- Shikoku routes ---
    "shikoku_north": [
        (134.05, 34.35),  # Takamatsu
        (133.80, 34.07),  # Niihama
        (133.55, 33.95),  # Saijo
        (132.77, 33.84),  # Matsuyama
        (132.55, 33.60),  # Ozu
        (132.45, 33.35),  # Uwajima
    ],
    "shikoku_south": [
        (134.05, 34.35),  # Takamatsu
        (134.25, 34.07),  # Tokushima direction
        (134.56, 34.07),  # Tokushima
        (134.30, 33.80),  # Anan
        (133.90, 33.65),  # Muroto area
        (133.53, 33.56),  # Kochi
        (133.20, 33.35),  # Shimanto area
    ],

    # --- Seto Bridges ---
    "seto_ohashi": [
        (133.93, 34.67),  # Okayama
        (133.82, 34.50),  # Kojima
        (133.80, 34.38),  # Sakaide (bridge)
        (134.05, 34.35),  # Takamatsu
    ],

    # --- Akashi-Kaikyo Bridge + Naruto ---
    "akashi_naruto": [
        (135.20, 34.69),  # Kobe
        (135.02, 34.60),  # Awaji Island north
        (134.85, 34.40),  # Awaji Island south
        (134.60, 34.18),  # Naruto
        (134.56, 34.07),  # Tokushima
    ],

    # --- Okinawa Expressway ---
    "okinawa_exp": [
        (127.68, 26.33),  # Naha
        (127.73, 26.40),  # Nishihara
        (127.76, 26.50),  # Okinawa City
        (127.78, 26.55),  # Gushikawa
        (127.80, 26.62),  # Nago direction
    ],

    # --- Akita Expressway ---
    "akita_exp": [
        (140.87, 38.27),  # Sendai area
        (140.70, 38.45),  # Furukawa
        (140.50, 38.70),  # Yokote area
        (140.30, 39.00),  # Daisen
        (140.10, 39.30),  # Yokote
        (140.10, 39.72),  # Akita
    ],

    # --- Ban-Etsu Expressway (Koriyama → Niigata) ---
    "banetsu_exp": [
        (140.34, 37.40),  # Koriyama
        (140.10, 37.42),  # Inawashiro
        (139.90, 37.48),  # Aizu area
        (139.70, 37.50),  # Aizuwakamatsu
        (139.50, 37.60),  # Nishiaizu
        (139.30, 37.70),  # Tsugawa
        (139.10, 37.80),  # Gosen
        (139.02, 37.92),  # Niigata
    ],
}


def build_road_graph() -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    Build a road graph from expressway waypoints.

    Returns:
        road_graph: NetworkX graph where nodes are waypoint indices,
                    edges are road segments with distance weights
        node_coords: dict mapping node_id → (lon, lat)
    """
    G = nx.Graph()
    node_coords = {}
    node_id = 0
    coord_to_node = {}  # (rounded_lon, rounded_lat) → node_id for merging nearby points

    def _get_or_create_node(lon, lat):
        nonlocal node_id
        key = (round(lon, 2), round(lat, 2))
        if key in coord_to_node:
            return coord_to_node[key]
        nid = node_id
        node_id += 1
        G.add_node(nid, lon=lon, lat=lat)
        node_coords[nid] = (lon, lat)
        coord_to_node[key] = nid
        return nid

    def _haversine(lon1, lat1, lon2, lat2):
        r = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon / 2) ** 2)
        return r * 2 * np.arcsin(np.sqrt(a))

    for route_name, waypoints in EXPRESSWAYS.items():
        prev_nid = None
        for lon, lat in waypoints:
            nid = _get_or_create_node(lon, lat)
            if prev_nid is not None and prev_nid != nid:
                plon, plat = node_coords[prev_nid]
                d = _haversine(plon, plat, lon, lat)
                if G.has_edge(prev_nid, nid):
                    existing = G[prev_nid][nid]["distance"]
                    if d < existing:
                        G[prev_nid][nid]["distance"] = d
                else:
                    G.add_edge(prev_nid, nid, distance=d, route=route_name)
            prev_nid = nid

    # --- Critical cross-sea links (tunnels, bridges, ferries) ---
    cross_links = [
        # Seikan Tunnel: Aomori ↔ Hakodate
        ((140.46, 40.82), (140.73, 41.77)),
        # Seto-Ohashi already in routes, but ensure Honshu↔Kyushu
        # Kanmon Strait: Shimonoseki ↔ Kitakyushu
        ((131.00, 33.95), (130.85, 33.88)),
    ]
    for (lon1, lat1), (lon2, lat2) in cross_links:
        n1 = _get_or_create_node(lon1, lat1)
        n2 = _get_or_create_node(lon2, lat2)
        d = _haversine(lon1, lat1, lon2, lat2)
        G.add_edge(n1, n2, distance=d, route="cross_link")

    # Ensure full connectivity: link disconnected components to nearest node in main
    components = list(nx.connected_components(G))
    if len(components) > 1:
        main = max(components, key=len)
        main_nodes = list(main)
        main_coords = np.array([(node_coords[n][0], node_coords[n][1]) for n in main_nodes])
        for comp in components:
            if comp is main:
                continue
            comp_nodes = list(comp)
            best_d = float("inf")
            best_pair = None
            for cn in comp_nodes:
                clon, clat = node_coords[cn]
                dists = np.sqrt((main_coords[:, 0] - clon) ** 2 + (main_coords[:, 1] - clat) ** 2) * 111.0
                idx = np.argmin(dists)
                if dists[idx] < best_d:
                    best_d = dists[idx]
                    best_pair = (cn, main_nodes[idx])
            if best_pair:
                n1, n2 = best_pair
                d = _haversine(*node_coords[n1], *node_coords[n2])
                G.add_edge(n1, n2, distance=d, route="bridge_link")

    return G, node_coords


def snap_location_to_road(
    loc_lon: float, loc_lat: float,
    node_coords: Dict[int, Tuple[float, float]],
    max_dist_km: float = 80.0,
) -> int:
    """Find the nearest road node to a location. Returns node_id or -1."""
    best_id = -1
    best_dist = max_dist_km
    for nid, (nlon, nlat) in node_coords.items():
        d = np.sqrt((nlon - loc_lon) ** 2 + (nlat - loc_lat) ** 2) * 111.0
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id


def precompute_location_road_nodes(
    loc_lons: np.ndarray,
    loc_lats: np.ndarray,
    node_coords: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    """For each simulation location, find the nearest road network node."""
    n = len(loc_lons)
    result = np.full(n, -1, dtype=np.int32)

    road_lons = np.array([c[0] for c in node_coords.values()])
    road_lats = np.array([c[1] for c in node_coords.values()])
    road_ids = np.array(list(node_coords.keys()))

    for i in range(n):
        dists = np.sqrt((road_lons - loc_lons[i]) ** 2 + (road_lats - loc_lats[i]) ** 2)
        nearest_idx = np.argmin(dists)
        if dists[nearest_idx] * 111.0 < 100.0:
            result[i] = road_ids[nearest_idx]

    return result
