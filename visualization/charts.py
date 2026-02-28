"""
Publication-quality cartographic charts for the Japan Exodus simulation.

Renders 7 chart types using actual prefecture GeoJSON boundaries,
bilingual labels, and the warm-grey cartographic style.

Charts:
  1. Population density choropleth
  2. Population change choropleth (diverging red-green)
  3. Prestige score map
  4. Anomie / Social risk map
  5. Migration flow map with arcs
  6. Demographic indicators dashboard
  7. Defensive hubs map (Core Cities network)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, Optional

from .style import (
    BG_COLOR, TEXT_DARK, TEXT_MED, TEXT_SUBTITLE, TEXT_SOURCE,
    GRID_COLOR, SPINE_COLOR, BORDER_LIGHT,
    TOKYO_RED, TOKYO_RED_LIGHT, CORE_BLUE, CORE_BLUE_LIGHT,
    PERIPHERY_GREEN, PERIPHERY_GREEN_LIGHT, PURPLE_NOTE, PURPLE_NOTE_LIGHT,
    AMBER_WARN, AMBER_WARN_LIGHT,
    TIER_COLORS, TIER_LABELS_JA, TIER_LABELS_EN,
    cmap_diverging, cmap_heat, cmap_cool, cmap_prestige,
    bilingual_title, source_note, label_with_stroke, callout_box,
    style_axes, style_map_axes, style_figure, style_legend, save_figure,
    get_jp_font, has_jp_font, bilingual,
)

# ---------------------------------------------------------------------------
# Prefecture <-> Tier mapping
# (In the simulation, agents are at 744 locations; we aggregate to 47 prefs)
# ---------------------------------------------------------------------------
TOKYO_PREFECTURES = {"Tokyo To"}
CORE_PREFECTURES = {
    "Hokkai Do", "Miyagi Ken", "Kanagawa Ken", "Aichi Ken", "Osaka Fu",
    "Hyogo Ken", "Hiroshima Ken", "Fukuoka Ken", "Saitama Ken", "Chiba Ken",
    "Kyoto Fu", "Shizuoka Ken", "Niigata Ken", "Okayama Ken",
}

PREF_NAME_JA = {
    "Hokkai Do": "北海道", "Aomori Ken": "青森", "Iwate Ken": "岩手",
    "Miyagi Ken": "宮城", "Akita Ken": "秋田", "Yamagata Ken": "山形",
    "Fukushima Ken": "福島", "Ibaraki Ken": "茨城", "Tochigi Ken": "栃木",
    "Gunma Ken": "群馬", "Saitama Ken": "埼玉", "Chiba Ken": "千葉",
    "Tokyo To": "東京", "Kanagawa Ken": "神奈川", "Niigata Ken": "新潟",
    "Toyama Ken": "富山", "Ishikawa Ken": "石川", "Fukui Ken": "福井",
    "Yamanashi Ken": "山梨", "Nagano Ken": "長野", "Gifu Ken": "岐阜",
    "Shizuoka Ken": "静岡", "Aichi Ken": "愛知", "Mie Ken": "三重",
    "Shiga Ken": "滋賀", "Kyoto Fu": "京都", "Osaka Fu": "大阪",
    "Hyogo Ken": "兵庫", "Nara Ken": "奈良", "Wakayama Ken": "和歌山",
    "Tottori Ken": "鳥取", "Shimane Ken": "島根", "Okayama Ken": "岡山",
    "Hiroshima Ken": "広島", "Yamaguchi Ken": "山口", "Tokushima Ken": "徳島",
    "Kagawa Ken": "香川", "Ehime Ken": "愛媛", "Kochi Ken": "高知",
    "Fukuoka Ken": "福岡", "Saga Ken": "佐賀", "Nagasaki Ken": "長崎",
    "Kumamoto Ken": "熊本", "Oita Ken": "大分", "Miyazaki Ken": "宮崎",
    "Kagoshima Ken": "鹿児島", "Okinawa Ken": "沖縄",
}


def _load_japan_gdf(geojson_path: str = "data/japan.geojson") -> gpd.GeoDataFrame:
    """Load and prepare the prefecture GeoDataFrame."""
    gdf = gpd.read_file(geojson_path)
    gdf["pref_en"] = gdf["nam"].str.replace(" Ken$| Fu$| To$| Do$", "", regex=True)
    gdf["pref_ja"] = gdf["nam"].map(PREF_NAME_JA).fillna("")

    gdf["tier"] = 2  # default periphery
    gdf.loc[gdf["nam"].isin(CORE_PREFECTURES), "tier"] = 1
    gdf.loc[gdf["nam"].isin(TOKYO_PREFECTURES), "tier"] = 0

    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y
    return gdf


def _aggregate_to_prefectures(loc_state, geography_config, scale: int = 100) -> pd.DataFrame:
    """
    Map simulation locations to prefectures using tier + rough geo matching.
    Returns a DataFrame indexed by tier with aggregated stats.
    """
    n = geography_config.n_locations
    tiers = loc_state.tier
    pops = loc_state.population * scale

    records = []
    for i in range(n):
        records.append({
            "loc_id": i,
            "tier": int(tiers[i]),
            "population": float(pops[i]),
            "prestige": float(loc_state.prestige[i]),
            "convenience": float(loc_state.convenience[i]),
            "anomie": float(loc_state.anomie[i]),
            "financial_friction": float(loc_state.financial_friction[i]),
            "vacancy_rate": float(loc_state.vacancy_rate[i]),
            "healthcare": float(loc_state.healthcare_score[i]),
            "childcare": float(loc_state.childcare_score[i]),
            "digital_access": float(loc_state.digital_access[i]),
            "hq_count": int(loc_state.hq_count[i]),
            "lon": float(loc_state.lon[i]),
            "lat": float(loc_state.lat[i]),
        })

    return pd.DataFrame(records)


def _assign_locations_to_prefectures(
    loc_df: pd.DataFrame, gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign simulation location data to prefectures by nearest centroid matching.
    """
    from scipy.spatial import cKDTree

    pref_centroids = np.column_stack([gdf["centroid_x"].values, gdf["centroid_y"].values])
    loc_coords = np.column_stack([loc_df["lon"].values, loc_df["lat"].values])

    tree = cKDTree(pref_centroids)
    _, nearest = tree.query(loc_coords)

    loc_df["pref_idx"] = nearest

    agg = loc_df.groupby("pref_idx").agg(
        population=("population", "sum"),
        mean_prestige=("prestige", "mean"),
        mean_convenience=("convenience", "mean"),
        mean_anomie=("anomie", "mean"),
        mean_friction=("financial_friction", "mean"),
        mean_vacancy=("vacancy_rate", "mean"),
        mean_healthcare=("healthcare", "mean"),
        mean_childcare=("childcare", "mean"),
        mean_digital=("digital_access", "mean"),
        total_hq=("hq_count", "sum"),
        n_locations=("loc_id", "count"),
    ).reset_index()

    gdf = gdf.reset_index(drop=True)
    gdf = gdf.merge(agg, left_index=True, right_on="pref_idx", how="left")
    gdf["population"] = gdf["population"].fillna(0)
    gdf["pop_density"] = gdf["population"] / (gdf.geometry.area * 10000)

    return gdf


# ===================================================================
# CHART 1: Population Density Choropleth
# ===================================================================
def chart_population_density(
    gdf: gpd.GeoDataFrame,
    year: int,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    style_figure(fig)
    style_map_axes(ax)

    gdf.plot(
        ax=ax,
        column="population",
        cmap=cmap_heat,
        edgecolor="#d0d0d0",
        linewidth=0.4,
        legend=False,
        missing_kwds={"color": "#f5f0e8", "edgecolor": "#d0d0d0"},
    )

    vmin = gdf["population"].min()
    vmax = gdf["population"].max()
    sm = plt.cm.ScalarMappable(cmap=cmap_heat, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, aspect=20, pad=0.02,
                        orientation="horizontal", location="bottom")
    cbar.ax.tick_params(labelsize=7, colors=TEXT_MED)
    cbar.set_label(bilingual("人口", "Population"), fontsize=8, color=TEXT_MED)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(BORDER_LIGHT)

    # Labels on key prefectures
    for _, row in gdf.iterrows():
        if row["population"] > gdf["population"].quantile(0.75) or row["tier"] == 0:
            ja = row.get("pref_ja", "")
            en = row["pref_en"]
            label_with_stroke(ax, row["centroid_x"], row["centroid_y"] + 0.12,
                              ja if has_jp_font() else "", fontsize=6.5, weight="bold")
            label_with_stroke(ax, row["centroid_x"], row["centroid_y"] - 0.08,
                              en, fontsize=5.5, color=TEXT_SUBTITLE, weight="normal")

    bilingual_title(ax,
                    f"人口分布 ({year}年)",
                    f"Population Distribution ({year})")

    callout_box(ax, 0.02, 0.25,
                "東京一極集中 — ブラックホール効果",
                "Tokyo Hyper-Concentration — Black Hole Effect",
                bg_color=TOKYO_RED_LIGHT, border_color=TOKYO_RED)

    source_note(ax)

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# CHART 2: Population Change (Diverging)
# ===================================================================
def chart_population_change(
    gdf_initial: gpd.GeoDataFrame,
    gdf_final: gpd.GeoDataFrame,
    start_year: int,
    end_year: int,
    save_path: Optional[str] = None,
) -> plt.Figure:

    gdf = gdf_final.copy()
    pop_init = gdf_initial.set_index("nam")["population"]
    gdf["pop_change_pct"] = gdf.apply(
        lambda r: ((r["population"] - pop_init.get(r["nam"], r["population"]))
                    / max(pop_init.get(r["nam"], 1), 1) * 100),
        axis=1,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    style_figure(fig)
    style_map_axes(ax)

    vmax = max(abs(gdf["pop_change_pct"].min()), abs(gdf["pop_change_pct"].max()), 10)
    gdf.plot(
        ax=ax,
        column="pop_change_pct",
        cmap=cmap_diverging,
        edgecolor="#d0d0d0",
        linewidth=0.4,
        legend=False,
        vmin=-vmax, vmax=vmax,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_diverging, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, aspect=20, pad=0.02,
                        orientation="horizontal", location="bottom")
    cbar.ax.tick_params(labelsize=7, colors=TEXT_MED)
    cbar.set_label(bilingual("人口変化率 (%)", "Population Change (%)"), fontsize=8, color=TEXT_MED)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(BORDER_LIGHT)

    for _, row in gdf.iterrows():
        change = row["pop_change_pct"]
        if abs(change) > gdf["pop_change_pct"].abs().quantile(0.70):
            sign = "+" if change > 0 else ""
            label_with_stroke(ax, row["centroid_x"], row["centroid_y"],
                              f"{sign}{change:.1f}%", fontsize=6, weight="bold",
                              color=PERIPHERY_GREEN if change > 0 else TOKYO_RED)

    bilingual_title(ax,
                    f"人口増減率 ({start_year}→{end_year}年)",
                    f"Population Change ({start_year}→{end_year})")

    callout_box(ax, 0.02, 0.25,
                "赤 = 人口減少 | 緑 = 人口増加",
                "Red = Decline | Green = Growth",
                bg_color="#fff8e1", border_color="#ffa000")

    source_note(ax)

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# CHART 3: Prestige Score Map
# ===================================================================
def chart_prestige_map(
    gdf: gpd.GeoDataFrame,
    year: int,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    style_figure(fig)
    style_map_axes(ax)

    gdf.plot(
        ax=ax, column="mean_prestige", cmap=cmap_prestige,
        edgecolor="#d0d0d0", linewidth=0.4, legend=False,
        missing_kwds={"color": "#f5f0e8"},
    )

    vmin, vmax = gdf["mean_prestige"].min(), gdf["mean_prestige"].max()
    sm = plt.cm.ScalarMappable(cmap=cmap_prestige, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, aspect=20, pad=0.02,
                        orientation="horizontal", location="bottom")
    cbar.ax.tick_params(labelsize=7, colors=TEXT_MED)
    cbar.set_label(bilingual("威信スコア (P)", "Prestige Score (P)"),
                   fontsize=8, color=TEXT_MED)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(BORDER_LIGHT)

    # Mark top-prestige prefectures
    top = gdf.nlargest(8, "mean_prestige")
    for _, row in top.iterrows():
        label_with_stroke(ax, row["centroid_x"], row["centroid_y"] + 0.1,
                          row["pref_ja"] if has_jp_font() else row["pref_en"],
                          fontsize=6.5, weight="bold")
        label_with_stroke(ax, row["centroid_x"], row["centroid_y"] - 0.08,
                          f"P={row['mean_prestige']:.2f}", fontsize=5.5,
                          color=AMBER_WARN, weight="bold")

    bilingual_title(ax,
                    f"威信スコア分布 ({year}年)",
                    f"Prestige Score Distribution ({year})")

    callout_box(ax, 0.02, 0.20,
                "ICE法による本社移転効果",
                "ICE Act HQ Relocation Effect",
                bg_color=AMBER_WARN_LIGHT, border_color=AMBER_WARN)

    source_note(ax)

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# CHART 4: Anomie / Social Risk Map
# ===================================================================
def chart_anomie_map(
    gdf: gpd.GeoDataFrame,
    year: int,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    style_figure(fig)
    style_map_axes(ax)

    gdf.plot(
        ax=ax, column="mean_anomie", cmap=cmap_heat,
        edgecolor="#d0d0d0", linewidth=0.4, legend=False,
        missing_kwds={"color": "#f5f0e8"},
    )

    vmin, vmax = gdf["mean_anomie"].min(), gdf["mean_anomie"].max()
    sm = plt.cm.ScalarMappable(cmap=cmap_heat, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, aspect=20, pad=0.02,
                        orientation="horizontal", location="bottom")
    cbar.ax.tick_params(labelsize=7, colors=TEXT_MED)
    cbar.set_label(bilingual("社会的孤立度 (A)", "Social Anomie Index (A)"),
                   fontsize=8, color=TEXT_MED)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(BORDER_LIGHT)

    # Mark high-anomie areas
    high = gdf.nlargest(10, "mean_anomie")
    for _, row in high.iterrows():
        label_with_stroke(ax, row["centroid_x"], row["centroid_y"],
                          f"A={row['mean_anomie']:.2f}", fontsize=5.5,
                          color=TOKYO_RED, weight="bold")

    bilingual_title(ax,
                    f"社会的アノミー指数 ({year}年)",
                    f"Social Anomie Index ({year})")

    callout_box(ax, 0.02, 0.20,
                "高アノミー = 過疎化 + 社会的孤立",
                "High Anomie = Depopulation + Social Isolation",
                bg_color=TOKYO_RED_LIGHT, border_color=TOKYO_RED)

    source_note(ax)

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# CHART 5: Migration Flow Map
# ===================================================================
def chart_migration_flows(
    gdf: gpd.GeoDataFrame,
    flows_df: pd.DataFrame,
    loc_state,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    style_figure(fig)
    style_map_axes(ax)

    gdf.plot(ax=ax, color="#f5f0e8", edgecolor="#d0d0d0", linewidth=0.4)

    if not flows_df.empty:
        tier_flows = flows_df.groupby(["from_tier", "to_tier"]).size().reset_index(name="count")

        tier_centroids = {}
        for t in [0, 1, 2]:
            mask = gdf["tier"] == t
            if mask.any():
                tier_centroids[t] = (
                    gdf.loc[mask, "centroid_x"].mean(),
                    gdf.loc[mask, "centroid_y"].mean(),
                )

        max_flow = tier_flows["count"].max()
        for _, row in tier_flows.iterrows():
            ft, tt = int(row["from_tier"]), int(row["to_tier"])
            if ft == tt:
                continue
            if ft not in tier_centroids or tt not in tier_centroids:
                continue

            x0, y0 = tier_centroids[ft]
            x1, y1 = tier_centroids[tt]

            width = max(0.5, row["count"] / max_flow * 6)
            alpha = min(0.8, 0.2 + row["count"] / max_flow * 0.6)
            color = TIER_COLORS.get(tt, TEXT_MED)

            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=width,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=5,
            )

            mx = (x0 + x1) / 2 + 0.3
            my = (y0 + y1) / 2
            label_with_stroke(ax, mx, my, f"{int(row['count']):,}",
                              fontsize=6, color=color, weight="bold")

    # Tier region labels
    for t in [0, 1, 2]:
        mask = gdf["tier"] == t
        if mask.any():
            cx = gdf.loc[mask, "centroid_x"].mean()
            cy = gdf.loc[mask, "centroid_y"].mean()
            ja = TIER_LABELS_JA[t]
            en = TIER_LABELS_EN[t]
            label_with_stroke(ax, cx, cy + 0.3, ja if has_jp_font() else en,
                              fontsize=9, weight="bold", color=TIER_COLORS[t])

    bilingual_title(ax,
                    "人口移動フロー",
                    "Migration Flow Map")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TOKYO_RED,
               markersize=8, label=bilingual("東京圏", "Tokyo Metro")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORE_BLUE,
               markersize=8, label=bilingual("中核市", "Core Cities")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PERIPHERY_GREEN,
               markersize=8, label=bilingual("周辺地域", "Periphery")),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower left", fontsize=8)
    style_legend(leg)

    source_note(ax)

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# CHART 6: Demographic Dashboard
# ===================================================================
def chart_demographic_dashboard(
    history: pd.DataFrame,
    scale: int,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    style_figure(fig)

    annual = history.groupby("year").agg({
        "total_population": "last",
        "tokyo_population": "last",
        "core_population": "last",
        "periphery_population": "last",
        "births": "sum",
        "deaths": "sum",
        "marriages": "sum",
        "migrations": "sum",
        "mean_age": "mean",
        "pct_married": "mean",
        "pct_remote": "mean",
        "reproductive_cohort": "mean",
    })

    # (0,0) Population by tier
    ax = axes[0, 0]
    style_axes(ax, "人口推移", "Population by Tier")
    for tier, col in [(0, "tokyo_population"), (1, "core_population"), (2, "periphery_population")]:
        ax.plot(annual.index, annual[col] * scale / 1e6,
                color=TIER_COLORS[tier], label=TIER_LABELS_EN[tier], linewidth=2)
    ax.set_ylabel("Million", fontsize=8, color=TEXT_MED)
    leg = ax.legend(fontsize=7)
    style_legend(leg)

    # (0,1) Births vs Deaths
    ax = axes[0, 1]
    style_axes(ax, "出生・死亡", "Natural Change")
    ax.fill_between(annual.index, annual["births"] * scale, alpha=0.3, color=PERIPHERY_GREEN)
    ax.fill_between(annual.index, annual["deaths"] * scale, alpha=0.3, color=TOKYO_RED)
    ax.plot(annual.index, annual["births"] * scale, color=PERIPHERY_GREEN, lw=2, label="Births")
    ax.plot(annual.index, annual["deaths"] * scale, color=TOKYO_RED, lw=2, label="Deaths")
    leg = ax.legend(fontsize=7)
    style_legend(leg)

    # (0,2) TFR
    ax = axes[0, 2]
    style_axes(ax, "合計特殊出生率", "Total Fertility Rate")
    tfr = annual["births"] * 20.0 / annual["reproductive_cohort"].clip(lower=1)
    ax.plot(annual.index, tfr, color=AMBER_WARN, linewidth=2.5)
    ax.axhline(y=2.1, color=TEXT_SOURCE, linestyle="--", alpha=0.6, linewidth=0.8)
    ax.text(annual.index[-1], 2.15, bilingual("置換水準 2.1", "Replacement 2.1"),
            fontsize=6.5, color=TEXT_SOURCE, ha="right")
    ax.set_ylim(bottom=0)

    # (1,0) Mean age
    ax = axes[1, 0]
    style_axes(ax, "平均年齢", "Mean Population Age")
    ax.plot(annual.index, annual["mean_age"], color=PURPLE_NOTE, linewidth=2.5)
    ax.set_ylabel("Years", fontsize=8, color=TEXT_MED)

    # (1,1) Migration volume
    ax = axes[1, 1]
    style_axes(ax, "年間移住数", "Annual Migrations")
    ax.bar(annual.index, annual["migrations"], color=CORE_BLUE, alpha=0.7, width=0.6)

    # (1,2) Population shares stacked area
    ax = axes[1, 2]
    style_axes(ax, "人口構成比", "Population Share")
    total = annual["total_population"].clip(lower=1)
    ax.stackplot(
        annual.index,
        annual["tokyo_population"] / total * 100,
        annual["core_population"] / total * 100,
        annual["periphery_population"] / total * 100,
        labels=[TIER_LABELS_EN[0], TIER_LABELS_EN[1], TIER_LABELS_EN[2]],
        colors=[TOKYO_RED, CORE_BLUE, PERIPHERY_GREEN],
        alpha=0.7,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("%", fontsize=8, color=TEXT_MED)
    leg = ax.legend(loc="upper right", fontsize=7)
    style_legend(leg)

    for ax in axes.flat:
        ax.set_xlabel("")

    fig.suptitle("")
    source_note(axes[1, 2], x=1.0, y=-0.12)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# CHART 7: Defensive Hubs Map (Core Cities Network)
# ===================================================================
def chart_defensive_hubs(
    gdf: gpd.GeoDataFrame,
    loc_state,
    network,
    save_path: Optional[str] = None,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    style_figure(fig)
    style_map_axes(ax)

    tier_colors_fill = {0: TOKYO_RED_LIGHT, 1: CORE_BLUE_LIGHT, 2: "#f5f0e8"}
    for t in [2, 1, 0]:
        mask = gdf["tier"] == t
        gdf[mask].plot(ax=ax, color=tier_colors_fill[t], edgecolor="#d0d0d0", linewidth=0.4)

    # Draw Core City hub markers
    n_locs = len(loc_state.tier)
    for i in range(n_locs):
        if loc_state.tier[i] != 1:
            continue
        pop = loc_state.population[i]
        r = max(2, min(8, int(np.sqrt(pop) * 0.06)))
        ax.plot(loc_state.lon[i], loc_state.lat[i], "o",
                color=CORE_BLUE, markersize=r, alpha=0.7, zorder=5,
                markeredgecolor="white", markeredgewidth=0.5)

    # Tokyo marker (pulsing red)
    for i in range(n_locs):
        if loc_state.tier[i] != 0:
            continue
        ax.plot(loc_state.lon[i], loc_state.lat[i], "o",
                color=TOKYO_RED, markersize=4, alpha=0.8, zorder=6,
                markeredgecolor="white", markeredgewidth=0.3)

    # Network edges between Core Cities
    for u, v, data in network.edges(data=True):
        if loc_state.tier[u] == 1 and loc_state.tier[v] == 1:
            d = data.get("distance", 100)
            if d < 200:
                ax.plot(
                    [loc_state.lon[u], loc_state.lon[v]],
                    [loc_state.lat[u], loc_state.lat[v]],
                    color=CORE_BLUE, alpha=0.15, linewidth=0.8, zorder=3,
                )

    bilingual_title(ax,
                    "防衛ライン — 中核市ネットワーク",
                    "Defensive Line — Core City Network")

    callout_box(ax, 0.02, 0.32,
                "デジタル田園都市スーパーハイウェイ",
                "Digital Garden City Superhighway",
                bg_color=CORE_BLUE_LIGHT, border_color=CORE_BLUE, fontsize=7)

    callout_box(ax, 0.02, 0.25,
                "● 中核市80市 — 地方の防衛拠点",
                "● 80 Core Cities — Regional Defensive Hubs",
                bg_color=PERIPHERY_GREEN_LIGHT, border_color=PERIPHERY_GREEN, fontsize=7)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TOKYO_RED,
               markersize=10, label=bilingual("東京圏（ブラックホール）", "Tokyo (Black Hole)")),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORE_BLUE,
               markersize=10, label=bilingual("中核市（防衛ライン）", "Core Cities (Defensive Line)")),
        Line2D([0], [0], color="#f5f0e8", marker="s", markersize=10,
               markeredgecolor="#d0d0d0",
               label=bilingual("周辺地域", "Periphery Districts")),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower left", fontsize=8, title_fontsize=9)
    style_legend(leg)

    source_note(ax, text="出典: シミュレーション | MIC / METI — Digital Garden City Superhighway Initiative")

    if save_path:
        save_figure(fig, save_path)
    return fig


# ===================================================================
# Master generation function
# ===================================================================
def generate_all_charts(
    results: Dict,
    output_dir: str = "output/charts",
    scale: int = 100,
    initial_results: Optional[Dict] = None,
) -> list:
    """Generate the full suite of 7 publication-quality charts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gdf = _load_japan_gdf()
    loc_state = results["final_location_state"]
    geo_cfg = results["config"].geography
    history = results["history"]
    flows = results.get("migration_flows", pd.DataFrame())
    network = results.get("network", None)

    loc_df = _aggregate_to_prefectures(loc_state, geo_cfg, scale)
    gdf_final = _assign_locations_to_prefectures(loc_df, gdf)

    start_year = int(history["year"].min())
    end_year = int(history["year"].max())

    figures = []

    print(f"Generating charts to {output_dir}/...")

    # Chart 1: Population density
    fig = chart_population_density(gdf_final, end_year,
                                   f"{output_dir}/01_population_density.png")
    figures.append(fig)
    print("  [1/7] Population density choropleth")

    # Chart 2: Population change (needs initial state)
    if initial_results:
        loc_df_init = _aggregate_to_prefectures(
            initial_results["final_location_state"], geo_cfg, scale)
        gdf_init = _assign_locations_to_prefectures(loc_df_init, _load_japan_gdf())
        fig = chart_population_change(gdf_init, gdf_final, start_year, end_year,
                                      f"{output_dir}/02_population_change.png")
    else:
        fig = chart_population_change(gdf_final, gdf_final, start_year, end_year,
                                      f"{output_dir}/02_population_change.png")
    figures.append(fig)
    print("  [2/7] Population change choropleth")

    # Chart 3: Prestige map
    fig = chart_prestige_map(gdf_final, end_year,
                             f"{output_dir}/03_prestige_map.png")
    figures.append(fig)
    print("  [3/7] Prestige score map")

    # Chart 4: Anomie map
    fig = chart_anomie_map(gdf_final, end_year,
                           f"{output_dir}/04_anomie_map.png")
    figures.append(fig)
    print("  [4/7] Social anomie map")

    # Chart 5: Migration flows
    fig = chart_migration_flows(gdf_final, flows, loc_state,
                                f"{output_dir}/05_migration_flows.png")
    figures.append(fig)
    print("  [5/7] Migration flow map")

    # Chart 6: Demographic dashboard
    fig = chart_demographic_dashboard(history, scale,
                                      f"{output_dir}/06_demographic_dashboard.png")
    figures.append(fig)
    print("  [6/7] Demographic dashboard")

    # Chart 7: Defensive hubs
    if network:
        fig = chart_defensive_hubs(gdf_final, loc_state, network,
                                   f"{output_dir}/07_defensive_hubs.png")
        figures.append(fig)
        print("  [7/7] Defensive hubs network map")

    plt.close("all")
    print(f"All {len(figures)} charts saved to {output_dir}/")
    return figures
