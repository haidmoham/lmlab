"""Shared dark palette for the attention notebook and its generated figures."""

from functools import wraps

import matplotlib as mpl
from cycler import cycler

BACKGROUND = "#181b25"
TEXT = "#e5e7ef"
MUTED = "#b9bfce"
COLORS = ("#d7b887", "#92b9e0", "#a5cea7", "#b9a3df", "#e5a1b7", "#82cbd1")
STYLE = {
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": "#626a7d",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": MUTED,
    "legend.facecolor": BACKGROUND,
    "legend.edgecolor": "#626a7d",
    "axes.prop_cycle": cycler(color=COLORS),
}


def apply_notebook_theme():
    """Use the palette for subsequent scratch figures in this notebook kernel."""
    mpl.rcParams.update(STYLE)


def notebook_figure(function):
    """Scope exported figures to this palette without changing the caller's defaults."""

    @wraps(function)
    def render(*args, **kwargs):
        with mpl.rc_context(STYLE):
            return function(*args, **kwargs)

    return render
