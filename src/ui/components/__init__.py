"""Reusable UI components."""

from src.ui.components.theme import (
    COLORS,
    CHART_COLORS,
    get_chart_config,
    get_table_style,
)
from src.ui.components.navbar import create_navbar, create_breadcrumb
from src.ui.components.metric_card import (
    create_metric_card,
    create_metric_row,
    create_status_badge,
    create_loading_wrapper,
    create_empty_state,
)
from src.ui.components.charts import (
    create_line_chart,
    create_area_chart,
    create_pie_chart,
    create_bar_chart,
    create_histogram,
    create_regime_timeline,
    create_chart_container,
)
from src.ui.components.feedback import (
    create_loading_spinner,
    create_error_alert,
    create_success_toast,
    create_warning_alert,
    create_info_alert,
    create_confirmation_dialog,
)

__all__ = [
    # Theme
    "COLORS",
    "CHART_COLORS",
    "get_chart_config",
    "get_table_style",
    # Navigation
    "create_navbar",
    "create_breadcrumb",
    # Metric Cards
    "create_metric_card",
    "create_metric_row",
    "create_status_badge",
    "create_loading_wrapper",
    "create_empty_state",
    # Charts
    "create_line_chart",
    "create_area_chart",
    "create_pie_chart",
    "create_bar_chart",
    "create_histogram",
    "create_regime_timeline",
    "create_chart_container",
    # Feedback
    "create_loading_spinner",
    "create_error_alert",
    "create_success_toast",
    "create_warning_alert",
    "create_info_alert",
    "create_confirmation_dialog",
]
