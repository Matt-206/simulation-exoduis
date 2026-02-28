"""
Shared visual style for the Japan Exodus simulation.

Matches the Infrastructure Plan / Digital Garden City Superhighway
cartographic aesthetic: warm grey backgrounds, bilingual labels,
clean typography, custom red-cream-green colormaps.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
BG_COLOR = "#fafafa"
TEXT_DARK = "#212121"
TEXT_MED = "#555555"
TEXT_SUBTITLE = "#757575"
TEXT_SOURCE = "#9e9e9e"
GRID_COLOR = "#e0e0e0"
SPINE_COLOR = "#9e9e9e"
BORDER_LIGHT = "#e0e0e0"

TOKYO_RED = "#c62828"
TOKYO_RED_LIGHT = "#ffebee"
CORE_BLUE = "#1565c0"
CORE_BLUE_LIGHT = "#e3f2fd"
PERIPHERY_GREEN = "#2e7d32"
PERIPHERY_GREEN_LIGHT = "#e8f5e9"
PURPLE_NOTE = "#6a1b9a"
PURPLE_NOTE_LIGHT = "#f3e5f5"
AMBER_WARN = "#e65100"
AMBER_WARN_LIGHT = "#fff3e0"

TIER_COLORS = {0: TOKYO_RED, 1: CORE_BLUE, 2: PERIPHERY_GREEN}
TIER_COLORS_LIGHT = {0: TOKYO_RED_LIGHT, 1: CORE_BLUE_LIGHT, 2: PERIPHERY_GREEN_LIGHT}
TIER_LABELS_JA = {0: "東京圏", 1: "中核市", 2: "周辺地域"}
TIER_LABELS_EN = {0: "Tokyo Metro", 1: "Core Cities", 2: "Periphery"}

# ---------------------------------------------------------------------------
# Custom colormaps
# ---------------------------------------------------------------------------
CMAP_DIVERGING_COLORS = [
    (0.0,  "#8b1a1a"),   # deep red (decline/risk)
    (0.25, "#d32f2f"),   # red
    (0.45, "#ffab91"),   # salmon
    (0.5,  "#f5f0e8"),   # cream center
    (0.55, "#a5d6a7"),   # light green
    (0.75, "#43a047"),   # green
    (1.0,  "#1b5e20"),   # deep green (growth/health)
]

CMAP_HEAT_COLORS = [
    (0.0,  "#f5f0e8"),   # cream (low)
    (0.3,  "#ffcc80"),   # light orange
    (0.5,  "#ff8a65"),   # orange
    (0.7,  "#e53935"),   # red
    (1.0,  "#b71c1c"),   # deep red (high)
]

CMAP_COOL_COLORS = [
    (0.0,  "#f5f0e8"),
    (0.3,  "#90caf9"),
    (0.6,  "#42a5f5"),
    (0.8,  "#1565c0"),
    (1.0,  "#0d47a1"),
]

CMAP_PRESTIGE_COLORS = [
    (0.0,  "#efebe9"),   # light beige (low prestige)
    (0.25, "#bcaaa4"),
    (0.5,  "#ff8f00"),   # amber
    (0.75, "#e65100"),
    (1.0,  "#bf360c"),   # deep orange (high prestige)
]


def _build_cmap(color_stops, name="custom"):
    positions = [c[0] for c in color_stops]
    hex_colors = [c[1] for c in color_stops]
    rgb_colors = [mpl.colors.to_rgb(h) for h in hex_colors]
    return LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)))


cmap_diverging = _build_cmap(CMAP_DIVERGING_COLORS, "exodus_diverging")
cmap_heat = _build_cmap(CMAP_HEAT_COLORS, "exodus_heat")
cmap_cool = _build_cmap(CMAP_COOL_COLORS, "exodus_cool")
cmap_prestige = _build_cmap(CMAP_PRESTIGE_COLORS, "exodus_prestige")


# ---------------------------------------------------------------------------
# Font detection (Japanese with fallback)
# ---------------------------------------------------------------------------
_JP_FONT = None

def _detect_jp_font():
    global _JP_FONT
    if _JP_FONT is not None:
        return _JP_FONT

    import matplotlib.font_manager as fm
    candidates = [
        "Yu Gothic", "YuGothic", "Meiryo", "MS Gothic", "msgothic",
        "Hiragino Sans", "Hiragino Kaku Gothic Pro", "Noto Sans CJK JP",
        "Noto Sans JP", "IPAGothic", "IPAexGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            _JP_FONT = name
            return _JP_FONT

    _JP_FONT = ""
    return _JP_FONT


def get_jp_font():
    return _detect_jp_font()


def has_jp_font():
    return bool(get_jp_font())


def configure_matplotlib_fonts():
    """Set matplotlib rcParams to use detected Japanese font globally."""
    jp = get_jp_font()
    if jp:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [jp, "Arial", "DejaVu Sans"]
    mpl.rcParams["figure.facecolor"] = BG_COLOR
    mpl.rcParams["axes.facecolor"] = BG_COLOR
    mpl.rcParams["savefig.facecolor"] = BG_COLOR
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["savefig.dpi"] = 300


configure_matplotlib_fonts()


# ---------------------------------------------------------------------------
# Text helpers (bilingual pattern)
# ---------------------------------------------------------------------------
def bilingual(ja: str, en: str) -> str:
    if has_jp_font():
        return ja
    return en


def bilingual_title(ax, ja: str, en: str, x=0.0, y=1.06, fontsize_ja=17, fontsize_en=10):
    """Add bilingual title: Japanese bold on top, English grey subtitle below."""
    jp = get_jp_font()
    if jp:
        ax.text(x, y, ja, transform=ax.transAxes, fontsize=fontsize_ja,
                fontweight="bold", color=TEXT_DARK, fontfamily=jp,
                ha="left", va="bottom")
        ax.text(x, y - 0.045, en, transform=ax.transAxes, fontsize=fontsize_en,
                color=TEXT_SUBTITLE, fontfamily="sans-serif",
                ha="left", va="bottom")
    else:
        ax.text(x, y, en, transform=ax.transAxes, fontsize=fontsize_ja,
                fontweight="bold", color=TEXT_DARK, fontfamily="sans-serif",
                ha="left", va="bottom")


def source_note(ax, text="出典: シミュレーション結果 | Source: Simulation Output", x=1.0, y=-0.06):
    """Add source note in bottom corner."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6.5,
            color=TEXT_SOURCE, ha="right", va="top",
            fontfamily=get_jp_font() or "sans-serif")


def label_with_stroke(ax, x, y, text, fontsize=7, color=TEXT_DARK,
                      weight="normal", stroke_color="white", stroke_width=2.5,
                      ha="center", va="center", fontfamily=None, **kwargs):
    """Text label with white outline for readability over maps."""
    ff = fontfamily or get_jp_font() or "sans-serif"
    ax.text(x, y, text, fontsize=fontsize, color=color, fontweight=weight,
            ha=ha, va=va, fontfamily=ff, zorder=10,
            path_effects=[
                pe.withStroke(linewidth=stroke_width, foreground=stroke_color),
            ], **kwargs)


# ---------------------------------------------------------------------------
# Callout box
# ---------------------------------------------------------------------------
def callout_box(ax, x, y, ja_text, en_text, bg_color=TOKYO_RED_LIGHT,
                border_color=TOKYO_RED, fontsize=8, transform=None):
    """Draw a styled callout box with bilingual text."""
    transform = transform or ax.transAxes
    jp = get_jp_font()
    if jp:
        text = f"{ja_text}\n{en_text}"
    else:
        text = en_text

    ax.text(x, y, text, transform=transform, fontsize=fontsize,
            fontfamily=jp or "sans-serif", color=TEXT_DARK,
            ha="left", va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=bg_color,
                edgecolor=border_color,
                linewidth=0.8,
                alpha=0.95,
            ))


# ---------------------------------------------------------------------------
# Axes styling
# ---------------------------------------------------------------------------
def style_axes(ax, title_ja=None, title_en=None, grid=True, spines_left_bottom=True):
    """Apply the standard chart style to a matplotlib axes."""
    ax.set_facecolor(BG_COLOR)

    if title_ja or title_en:
        bilingual_title(ax, title_ja or title_en, title_en or title_ja)

    ax.tick_params(colors=TEXT_MED, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("sans-serif")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if spines_left_bottom:
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_linewidth(0.5)
    else:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    if grid:
        ax.grid(True, color=GRID_COLOR, alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)


def style_map_axes(ax):
    """Style for map-type axes (no spines, no ticks)."""
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def style_figure(fig):
    """Apply figure-level styling."""
    fig.patch.set_facecolor(BG_COLOR)


def save_figure(fig, path, dpi=300):
    """Save with standard padding and DPI."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.5,
                facecolor=BG_COLOR, edgecolor="none")


# ---------------------------------------------------------------------------
# Legend styling
# ---------------------------------------------------------------------------
def style_legend(leg):
    """Apply standard legend styling."""
    frame = leg.get_frame()
    frame.set_facecolor(BG_COLOR)
    frame.set_edgecolor(BORDER_LIGHT)
    frame.set_linewidth(0.5)
    frame.set_alpha(0.9)
    for text in leg.get_texts():
        text.set_color(TEXT_MED)
        text.set_fontsize(8)
