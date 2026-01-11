"""Regime Monitor page callbacks."""

from dash import Input, Output
from dash.exceptions import PreventUpdate

from src.ui.layouts.regime_monitor import (
    create_indicators_tab,
    create_history_tab,
)


def register_regime_callbacks(app):
    """Register regime monitor callbacks."""

    @app.callback(
        Output("regime-tab-content", "children"),
        [Input("regime-tabs", "value")],
    )
    def render_regime_tab(tab):
        """Render regime monitor tab content."""
        if tab == "indicators":
            return create_indicators_tab()
        elif tab == "history":
            return create_history_tab()
        return create_indicators_tab()
