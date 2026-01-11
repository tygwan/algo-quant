"""Reusable chart components with dark theme and accessibility features."""

from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional

from src.ui.components.theme import COLORS, CHART_COLORS, get_chart_config


# Consistent hover styling for all charts
HOVER_STYLE = dict(
    bgcolor="rgba(30, 30, 63, 0.95)",
    bordercolor="rgba(255, 255, 255, 0.2)",
    font=dict(
        family="Inter, sans-serif",
        size=13,
        color="#ffffff",
    ),
)


def create_accessible_colorscale() -> dict:
    """Return colors that are distinguishable for color-blind users.

    This colorscale uses colors optimized for deuteranopia (red-green),
    protanopia (red-green), and tritanopia (blue-yellow) color blindness.
    Colors are chosen to have distinct luminance and hue values.

    Returns:
        dict: Dictionary with 'colors' list and 'patterns' for additional distinction
    """
    # Colors optimized for color-blind accessibility
    # Using Wong's color palette + additional high-contrast colors
    colors = [
        "#0077BB",  # Blue (safe for all types)
        "#EE7733",  # Orange (distinguishable from blue)
        "#009988",  # Teal (cyan-ish, good contrast)
        "#CC3311",  # Red-orange (distinct luminance)
        "#33BBEE",  # Cyan (light, high contrast)
        "#EE3377",  # Magenta (distinct from green)
        "#BBBBBB",  # Grey (neutral, always visible)
        "#000000",  # Black (maximum contrast)
    ]

    # Pattern fills for additional distinction (for charts that support it)
    patterns = [
        "",         # Solid (no pattern)
        "/",        # Diagonal lines
        "\\",       # Back-diagonal lines
        "x",        # Crosshatch
        "-",        # Horizontal lines
        "|",        # Vertical lines
        "+",        # Plus pattern
        ".",        # Dots
    ]

    return {
        "colors": colors,
        "patterns": patterns,
        # Pre-defined semantic colors for positive/negative values
        "positive": "#009988",  # Teal - visible to most color-blind users
        "negative": "#CC3311",  # Red-orange - distinct luminance from teal
        "neutral": "#0077BB",   # Blue - universally visible
    }


def create_line_chart(
    data: pd.DataFrame,
    title: str = "",
    height: int = 400,
    accessible_colors: bool = False,
) -> dcc.Graph:
    """Create a line chart with dark theme and enhanced accessibility.

    Args:
        data: DataFrame with datetime index and columns to plot
        title: Chart title
        height: Chart height in pixels
        accessible_colors: Use color-blind friendly palette

    Returns:
        Dash Graph component
    """
    fig = go.Figure()

    # Choose color palette
    if accessible_colors:
        color_palette = create_accessible_colorscale()["colors"]
    else:
        color_palette = CHART_COLORS

    for i, col in enumerate(data.columns):
        # Calculate change from previous value for hover
        series = data[col]
        pct_change = series.pct_change() * 100

        # Build custom data for hover template
        customdata = np.column_stack([
            pct_change.fillna(0).values,
            series.shift(1).fillna(series.iloc[0]).values,
        ])

        fig.add_trace(go.Scatter(
            x=data.index,
            y=series,
            name=col,
            mode="lines",
            line=dict(color=color_palette[i % len(color_palette)], width=2),
            customdata=customdata,
            hovertemplate=(
                f"<b>{col}</b><br>"
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>Value:</b> %{y:,.2f}<br>"
                "<b>Change:</b> %{customdata[0]:+.2f}%<br>"
                "<b>Previous:</b> %{customdata[1]:,.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title if title else None,
        height=height,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        hoverlabel=HOVER_STYLE,
        hovermode="x unified",
    )

    # Generate aria-label description
    columns_desc = ", ".join(data.columns.tolist())
    date_range = f"from {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}" if len(data) > 0 else ""
    aria_label = f"Line chart showing {columns_desc} {date_range}. {title}" if title else f"Line chart showing {columns_desc} {date_range}"

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
        **{"aria-label": aria_label},
    )


def create_area_chart(
    data: pd.Series,
    title: str = "",
    color: str = None,
    fill_color: str = None,
    height: int = 300,
) -> dcc.Graph:
    """Create an area chart (e.g., for drawdown).

    Args:
        data: Series to plot
        title: Chart title
        color: Line color
        fill_color: Fill color (with alpha)
        height: Chart height

    Returns:
        Dash Graph component
    """
    color = color or COLORS["danger"]

    # Create fill color with alpha if not provided
    if not fill_color:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_color = f"rgba({r}, {g}, {b}, 0.3)"

    fig = go.Figure()

    # Calculate change from previous for hover
    pct_change = data.pct_change() * 100

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        fill="tozeroy",
        name=data.name or "Value",
        line=dict(color=color, width=1),
        fillcolor=fill_color,
        customdata=pct_change.fillna(0).values,
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}<br>"
            "<b>Value:</b> %{y:.2f}<br>"
            "<b>Change:</b> %{customdata:+.2f}%"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title if title else None,
        height=height,
        hoverlabel=HOVER_STYLE,
    )

    # Generate aria-label
    name = data.name or "values"
    aria_label = f"Area chart showing {name}. {title}" if title else f"Area chart showing {name}"

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
        **{"aria-label": aria_label},
    )


def create_pie_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    height: int = 350,
    accessible_colors: bool = False,
    center_text: str = "Allocation",
) -> dcc.Graph:
    """Create a donut/pie chart with enhanced accessibility.

    Args:
        labels: Category labels
        values: Category values
        title: Chart title
        height: Chart height
        accessible_colors: Use color-blind friendly palette
        center_text: Text to display in center of donut

    Returns:
        Dash Graph component
    """
    # Choose color palette
    if accessible_colors:
        color_palette = create_accessible_colorscale()["colors"]
    else:
        color_palette = CHART_COLORS

    # Calculate total for percentage display
    total = sum(values)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(color=COLORS["text_secondary"], size=11),
        insidetextorientation="radial",
        marker=dict(
            colors=color_palette[:len(labels)],
            line=dict(color="rgba(0,0,0,0.3)", width=1),
        ),
        # Enhanced hover with both percentage and actual value
        hovertemplate=(
            "<b>%{label}</b><br>"
            "<b>Value:</b> %{value:,.2f}<br>"
            "<b>Percentage:</b> %{percent}<br>"
            f"<b>Total:</b> {total:,.2f}"
            "<extra></extra>"
        ),
        # Improve label positioning with pull for small segments
        pull=[0.02 if v / total < 0.05 else 0 for v in values],
    )])

    fig.update_layout(
        title=title if title else None,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=20, r=20, t=40, b=60),
        hoverlabel=HOVER_STYLE,
    )

    # Center annotation
    fig.add_annotation(
        text=f"<b>{center_text}</b>",
        x=0.5,
        y=0.5,
        font=dict(size=14, color=COLORS["text"]),
        showarrow=False,
    )

    # Generate aria-label
    segments_desc = ", ".join([f"{label}: {value:,.2f}" for label, value in zip(labels, values)])
    aria_label = f"Pie chart showing {segments_desc}. {title}" if title else f"Pie chart showing {segments_desc}"

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
        **{"aria-label": aria_label},
    )


def create_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    horizontal: bool = False,
    height: int = 350,
    show_values: bool = True,
    accessible_colors: bool = False,
    use_patterns: bool = False,
) -> dcc.Graph:
    """Create a bar chart with enhanced accessibility.

    Args:
        labels: Bar labels
        values: Bar values
        title: Chart title
        horizontal: If True, create horizontal bar chart
        height: Chart height
        show_values: Show value labels on top of bars
        accessible_colors: Use color-blind friendly palette
        use_patterns: Add pattern fills for accessibility (color-blind users)

    Returns:
        Dash Graph component
    """
    # Choose colors based on accessibility preference
    if accessible_colors:
        accessible = create_accessible_colorscale()
        colors = [accessible["positive"] if v >= 0 else accessible["negative"] for v in values]
    else:
        colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in values]

    # Pattern configuration for accessibility
    pattern_shape = None
    if use_patterns:
        patterns = create_accessible_colorscale()["patterns"]
        pattern_shape = ["/" if v >= 0 else "\\" for v in values]

    # Format value labels
    text_values = [f"{v:+.2f}" if v != 0 else "0.00" for v in values]

    if horizontal:
        fig = go.Figure(data=[go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(
                color=colors,
                pattern_shape=pattern_shape,
                pattern_fillmode="overlay",
                line=dict(color="rgba(255,255,255,0.3)", width=1),
            ),
            text=text_values if show_values else None,
            textposition="outside",
            textfont=dict(color=COLORS["text_secondary"], size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "<b>Value:</b> %{x:+.2f}<br>"
                "<b>Status:</b> " + "%{customdata}"
                "<extra></extra>"
            ),
            customdata=["Positive" if v >= 0 else "Negative" for v in values],
        )])
    else:
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                pattern_shape=pattern_shape,
                pattern_fillmode="overlay",
                line=dict(color="rgba(255,255,255,0.3)", width=1),
            ),
            text=text_values if show_values else None,
            textposition="outside",
            textfont=dict(color=COLORS["text_secondary"], size=11),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "<b>Value:</b> %{y:+.2f}<br>"
                "<b>Status:</b> " + "%{customdata}"
                "<extra></extra>"
            ),
            customdata=["Positive" if v >= 0 else "Negative" for v in values],
        )])

    # Add zero line with better visibility
    if horizontal:
        fig.add_vline(
            x=0,
            line=dict(color=COLORS["text_secondary"], width=1, dash="dash"),
        )
    else:
        fig.add_hline(
            y=0,
            line=dict(color=COLORS["text_secondary"], width=1, dash="dash"),
        )

    fig.update_layout(
        title=title if title else None,
        height=height,
        bargap=0.3,
        hoverlabel=HOVER_STYLE,
        # Ensure enough space for value labels
        margin=dict(t=60 if not horizontal else 40, b=40, l=60 if horizontal else 40, r=60),
    )

    # Adjust y-axis range to make room for text labels
    if not horizontal and show_values:
        max_val = max(values) if values else 0
        min_val = min(values) if values else 0
        padding = abs(max_val - min_val) * 0.15 if max_val != min_val else abs(max_val) * 0.15
        fig.update_yaxes(range=[min_val - padding if min_val < 0 else min_val, max_val + padding])

    # Generate aria-label
    pos_count = sum(1 for v in values if v >= 0)
    neg_count = len(values) - pos_count
    aria_label = f"Bar chart with {len(values)} bars ({pos_count} positive, {neg_count} negative). {title}" if title else f"Bar chart with {len(values)} bars ({pos_count} positive, {neg_count} negative)"

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
        **{"aria-label": aria_label},
    )


def create_histogram(
    data: pd.Series,
    title: str = "",
    bins: int = 50,
    show_normal: bool = True,
    height: int = 350,
    accessible_colors: bool = False,
) -> dcc.Graph:
    """Create a histogram with optional normal distribution overlay.

    Args:
        data: Data series
        title: Chart title
        bins: Number of bins
        show_normal: Whether to show normal distribution overlay
        height: Chart height
        accessible_colors: Use color-blind friendly palette

    Returns:
        Dash Graph component
    """
    fig = go.Figure()

    # Choose colors
    if accessible_colors:
        colors = create_accessible_colorscale()
        bar_color = colors["colors"][0]  # Blue
        normal_color = colors["colors"][1]  # Orange
        mean_color = colors["colors"][2]  # Teal
    else:
        bar_color = COLORS["primary"]
        normal_color = COLORS["warning"]
        mean_color = COLORS["success"]

    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name="Distribution",
        marker=dict(
            color=bar_color,
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        opacity=0.8,
        hovertemplate=(
            "<b>Range:</b> %{x:.4f}<br>"
            "<b>Count:</b> %{y}<br>"
            f"<b>Total samples:</b> {len(data)}"
            "<extra></extra>"
        ),
    ))

    if show_normal:
        x_range = np.linspace(data.min(), data.max(), 100)
        from scipy import stats
        normal = stats.norm.pdf(x_range, data.mean(), data.std())
        scale = len(data) * (data.max() - data.min()) / bins

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal * scale,
            name="Normal Distribution",
            line=dict(color=normal_color, dash="dash", width=2),
            hoverinfo="skip",
        ))

    # Add mean line
    fig.add_vline(
        x=data.mean(),
        line=dict(color=mean_color, dash="dot", width=2),
        annotation_text=f"Mean: {data.mean():.4f}",
        annotation_position="top",
        annotation_font_color=mean_color,
    )

    # Add standard deviation markers
    std = data.std()
    mean = data.mean()
    for i, mult in enumerate([1, 2]):
        for sign, label in [(1, "+"), (-1, "-")]:
            fig.add_vline(
                x=mean + sign * mult * std,
                line=dict(color=COLORS["text_muted"], dash="dot", width=1),
                annotation_text=f"{label}{mult}σ" if mult == 1 else None,
                annotation_position="top",
                annotation_font_color=COLORS["text_muted"],
                annotation_font_size=10,
            )

    fig.update_layout(
        title=title if title else None,
        height=height,
        bargap=0.1,
        xaxis_title="Value",
        yaxis_title="Frequency",
        hoverlabel=HOVER_STYLE,
    )

    # Generate aria-label
    aria_label = f"Histogram showing distribution of {len(data)} values. Mean: {mean:.4f}, Std: {std:.4f}. {title}" if title else f"Histogram showing distribution of {len(data)} values. Mean: {mean:.4f}, Std: {std:.4f}"

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
        **{"aria-label": aria_label},
    )


def create_regime_timeline(
    dates: pd.DatetimeIndex,
    regimes: list[str],
    title: str = "",
    height: int = 200,
    accessible_colors: bool = False,
) -> dcc.Graph:
    """Create a regime timeline chart with enhanced accessibility.

    Args:
        dates: Date index
        regimes: Regime labels for each date
        title: Chart title
        height: Chart height
        accessible_colors: Use color-blind friendly palette

    Returns:
        Dash Graph component
    """
    # Color maps for standard and accessible modes
    if accessible_colors:
        colors = create_accessible_colorscale()["colors"]
        color_map = {
            "Expansion": colors[0],    # Blue
            "Peak": colors[1],         # Orange
            "Contraction": colors[3],  # Red-orange
            "Trough": colors[6],       # Grey
        }
        # Pattern map for additional accessibility
        pattern_map = {
            "Expansion": "",
            "Peak": "/",
            "Contraction": "\\",
            "Trough": "x",
        }
    else:
        color_map = {
            "Expansion": COLORS["success"],
            "Peak": COLORS["warning"],
            "Contraction": COLORS["danger"],
            "Trough": COLORS["secondary"],
        }
        pattern_map = None

    df = pd.DataFrame({"date": dates, "regime": regimes})

    fig = go.Figure()

    # Create bars for each regime with proper hover
    for regime in df["regime"].unique():
        regime_df = df[df["regime"] == regime]
        pattern = pattern_map.get(regime) if pattern_map else None

        fig.add_trace(go.Bar(
            x=regime_df["date"],
            y=[1] * len(regime_df),
            name=regime,
            marker=dict(
                color=color_map.get(regime, COLORS["primary"]),
                pattern_shape=pattern,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>{regime}</b><br>"
                "<b>Date:</b> %{x|%Y-%m-%d}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title if title else None,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        yaxis_visible=False,
        xaxis_title="",
        bargap=0,
        barmode="overlay",
        hoverlabel=HOVER_STYLE,
    )

    # Generate aria-label
    regime_counts = df["regime"].value_counts().to_dict()
    regime_desc = ", ".join([f"{k}: {v} periods" for k, v in regime_counts.items()])
    aria_label = f"Timeline chart showing market regimes. {regime_desc}. {title}" if title else f"Timeline chart showing market regimes. {regime_desc}"

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
        **{"aria-label": aria_label},
    )


def create_chart_container(
    title: str,
    chart: dcc.Graph,
    subtitle: Optional[str] = None,
    description: Optional[str] = None,
) -> html.Div:
    """Wrap chart in styled container with accessibility features.

    Args:
        title: Chart title
        chart: Chart component
        subtitle: Optional subtitle
        description: Optional description for screen readers

    Returns:
        Styled chart container with ARIA attributes
    """
    header_children = [
        html.H3(
            title,
            className="chart-title",
            style={"margin": "0", "fontSize": "1rem", "fontWeight": "600"},
        )
    ]

    if subtitle:
        header_children.append(
            html.P(
                subtitle,
                style={
                    "color": "var(--text-muted)",
                    "fontSize": "0.875rem",
                    "margin": "0.25rem 0 0 0",
                },
            )
        )

    # Build container with proper ARIA attributes
    container_attrs = {
        "className": "chart-container",
        "role": "figure",
    }

    if description:
        container_attrs["aria-label"] = description

    return html.Div(
        **container_attrs,
        children=[
            html.Div(
                header_children,
                style={"marginBottom": "0.5rem"},
            ),
            chart,
        ],
    )
