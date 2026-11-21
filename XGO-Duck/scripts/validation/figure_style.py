# Adapted from the figure-style skill kernel (Apache-2.0): style setup only.
# Bundled here so report plots do not require a user-specific skill path.
"""Sidecar helpers for the figure-style skill.

Definition-only module: importing it must not touch the network, filesystem
(beyond font probing at call time), or draw anything. Heavy imports live
inside function bodies so the kernel loads instantly.

Public API (names are stable — other skills reference them):
    apply_figure_style, set_frame, panel_letter, focal_palette,
    bar_with_points, strip_with_median, goodness_arrow, two_tier_label,
    end_of_line_labels, panel_crops
"""

META_GREY = "#888888"

# CJK-capable fonts each OS ships with, probed by file path first (so the
# exact file gets registered with matplotlib) and by family name second.
# Without this, Chinese/Japanese/Korean labels fall back to DejaVu Sans and
# render as tofu boxes (□□□).
_CJK_CANDIDATES = {
    "Windows": [
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc"),
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttf"),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
    ],
    "Darwin": [
        ("PingFang SC", "/System/Library/Fonts/PingFang.ttc"),
        ("Hiragino Sans GB", "/System/Library/Fonts/Hiragino Sans GB.ttc"),
        ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
    ],
    "Linux": [
        ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        ("WenQuanYi Zen Hei", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ("Source Han Sans SC", "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf"),
    ],
}


def _find_cjk_font():
    """Return a CJK-capable family name registered with matplotlib, or None.

    Probes the current OS's known font files first (registering the file so
    the family becomes usable), then any CJK family the font manager already
    knows. Uses whatever the machine has — nothing is bundled.
    """
    import os
    import platform
    import matplotlib.font_manager as fm
    for family, path in _CJK_CANDIDATES.get(platform.system(), []):
        if os.path.isfile(path):
            try:
                fm.fontManager.addfont(path)
                return family
            except Exception:
                continue
    known = {f.name for f in fm.fontManager.ttflist}
    for family in ("Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
                   "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                   "Arial Unicode MS"):
        if family in known:
            return family
    return None


def _register_conda_fonts():
    """Fonts installed via conda (e.g. mscorefonts) land in $CONDA_PREFIX/fonts,
    which matplotlib never scans — register them so `font=` requests resolve."""
    import glob
    import os
    import sys
    import matplotlib.font_manager as fm
    fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
    if not os.path.isdir(fdir):
        return
    known = {f.fname for f in fm.fontManager.ttflist}
    for f in glob.glob(os.path.join(fdir, "*.ttf")):
        if f not in known:
            fm.fontManager.addfont(f)


# ---------------------------------------------------------------- style setup

def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    """Install publication-output rcParams. Call once, before any plotting.

    What this sets is mechanics, not a house look: a three-step font-size
    ladder mapped to text roles, outward ticks, frameless legends, left-flush
    regular-weight titles, 300-dpi tight saves, and Type-42 (editable) fonts
    in vector output. Frame shape, family, and the ladder are parameters.

    frame : 'open' → bottom+left spines only (default); 'boxed' → all four;
            'none' → no spines, no tick marks
    font  : preferred sans-serif family; None keeps the platform default
    sizes : (base, secondary, tick) point sizes — base covers titles, axis
            labels and series identity; secondary covers legends/annotations;
            tick covers tick labels
    grid  : draw axes.grid when True
    """
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        _register_conda_fonts()
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    spined = (frame != "none")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.bottom": spined, "axes.spines.left": spined,
        "axes.grid": bool(grid),
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "figure.dpi": 200,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    # Sans-serif fallback chain: explicit request first, then a CJK family
    # (when the OS has one) so non-Latin labels never render as boxes, then
    # the usual Latin families. Putting CJK ahead of DejaVu matters on older
    # matplotlib versions that lack per-glyph fallback.
    chain = [font] if font else []
    cjk = _find_cjk_font()
    if cjk and cjk not in chain:
        chain.append(cjk)
    chain += ["DejaVu Sans", "Liberation Sans", "Arial"]
    rc["font.sans-serif"] = chain
    rc["axes.unicode_minus"] = False  # a boxed minus sign is still tofu
    mpl.rcParams.update(rc)



