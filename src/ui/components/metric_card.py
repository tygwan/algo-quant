"""Reusable metric card components."""

from dash import html
from typing import Optional


def create_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "positive",
    icon: Optional[str] = None,
) -> html.Div:
    """Create a styled metric card.

    Args:
        label: Metric label text
        value: Metric value to display
        delta: Optional change indicator
        delta_type: "positive" or "negative" for styling
        icon: Optional icon to display

    Returns:
        Styled metric card component
    """
    children = []

    # Icon (optional)
    if icon:
        children.append(
            html.Div(
                icon,
                style={
                    "fontSize": "1.5rem",
                    "marginBottom": "0.5rem",
                    "opacity": "0.7",
                },
            )
        )

    # Label
    children.append(html.Div(label, className="metric-label"))

    # Value
    children.append(html.Div(value, className="metric-value"))

    # Delta (optional)
    if delta:
        delta_icon = "↑" if delta_type == "positive" else "↓"
        children.append(
            html.Div(
                f"{delta_icon} {delta}",
                className=f"metric-delta {delta_type}",
            )
        )

    return html.Div(className="metric-card", children=children)


def create_metric_row(metrics: list[dict]) -> html.Div:
    """Create a row of metric cards.

    Args:
        metrics: List of metric configs with keys:
            - label: str
            - value: str
            - delta: Optional[str]
            - delta_type: str
            - icon: Optional[str]

    Returns:
        Row of metric cards
    """
    col_class = f"col-{12 // len(metrics)}" if len(metrics) <= 4 else "col-3"

    return html.Div(
        className="row",
        children=[
            html.Div(
                className=f"col {col_class}",
                children=create_metric_card(**metric),
            )
            for metric in metrics
        ],
    )


def create_status_badge(
    status: str,
    badge_type: str = "expansion",
) -> html.Span:
    """Create a status badge.

    Args:
        status: Status text
        badge_type: "expansion", "contraction", "peak", "trough"

    Returns:
        Styled status badge
    """
    return html.Span(
        status,
        className=f"status-badge {badge_type}",
    )


def create_loading_wrapper(
    component_id: str,
    children,
    loading_type: str = "circle",
) -> html.Div:
    """Wrap component with loading indicator.

    Args:
        component_id: ID for the loading component
        children: Child components
        loading_type: Type of loading indicator

    Returns:
        Component wrapped with loading indicator
    """
    from dash import dcc

    return dcc.Loading(
        id=f"{component_id}-loading",
        type=loading_type,
        color="var(--primary)",
        children=children,
    )


def create_empty_state(
    icon: str = "📊",
    message: str = "No data available",
    action: Optional[html.Button] = None,
) -> html.Div:
    """Create an empty state placeholder.

    Args:
        icon: Icon to display
        message: Message text
        action: Optional action button

    Returns:
        Empty state component
    """
    children = [
        html.Div(icon, className="empty-state-icon"),
        html.Div(message),
    ]

    if action:
        children.append(html.Div(action, style={"marginTop": "1rem"}))

    return html.Div(className="empty-state", children=children)
