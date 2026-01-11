"""Callbacks module - registers all Dash callbacks."""

from .dashboard_cb import register_dashboard_callbacks
from .data_explorer_cb import register_data_explorer_callbacks
from .factor_cb import register_factor_callbacks
from .regime_cb import register_regime_callbacks
from .backtest_cb import register_backtest_callbacks
from .portfolio_cb import register_portfolio_callbacks


def register_callbacks(app):
    """Register all callbacks with the app."""
    register_dashboard_callbacks(app)
    register_data_explorer_callbacks(app)
    register_factor_callbacks(app)
    register_regime_callbacks(app)
    register_backtest_callbacks(app)
    register_portfolio_callbacks(app)
