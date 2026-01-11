"""Page layouts for Dash application."""

from src.ui.layouts.dashboard import create_dashboard_layout
from src.ui.layouts.data_explorer import create_data_explorer_layout
from src.ui.layouts.factor_analysis import create_factor_analysis_layout
from src.ui.layouts.regime_monitor import create_regime_monitor_layout
from src.ui.layouts.backtest import create_backtest_layout
from src.ui.layouts.portfolio import create_portfolio_layout

__all__ = [
    "create_dashboard_layout",
    "create_data_explorer_layout",
    "create_factor_analysis_layout",
    "create_regime_monitor_layout",
    "create_backtest_layout",
    "create_portfolio_layout",
]
