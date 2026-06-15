"""Façade des panneaux Streamlit du dashboard."""

from __future__ import annotations

from ui.panels_herd import render_tab_daily_comparison, render_tab_export, render_tab_herd
from ui.panels_individual import render_tab_individual

__all__ = [
    "render_tab_individual",
    "render_tab_herd",
    "render_tab_daily_comparison",
    "render_tab_export",
]
