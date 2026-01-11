"""Reusable chart components with dark theme."""

from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional

from src.ui.components.theme import COLORS, CHART_COLORS, get_chart_config


def create_line_chart(
    data: pd.DataFrame,
    title: str = "",
    height: int = 400,
) -> dcc.Graph:
    """Create a line chart with dark theme.

    Args:
        data: DataFrame with datetime index and columns to plot
        title: Chart title
        height: Chart height in pixels

    Returns:
        Dash Graph component
    """
    fig = go.Figure()

    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            name=col,
            mode="lines",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{y:,.2f}}<extra></extra>",
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
    )

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
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

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        fill="tozeroy",
        name=data.name or "Value",
        line=dict(color=color, width=1),
        fillcolor=fill_color,
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=title if title else None,
        height=height,
    )

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
    )


def create_pie_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    height: int = 350,
) -> dcc.Graph:
    """Create a donut/pie chart.

    Args:
        labels: Category labels
        values: Category values
        title: Chart title
        height: Chart height

    Returns:
        Dash Graph component
    """
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(color=COLORS["text_secondary"], size=11),
        marker=dict(
            colors=CHART_COLORS[:len(labels)],
            line=dict(color="rgba(0,0,0,0.3)", width=1),
        ),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        title=title if title else None,
        height=height,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Center annotation
    fig.add_annotation(
        text="<b>Allocation</b>",
        x=0.5,
        y=0.5,
        font=dict(size=14, color=COLORS["text"]),
        showarrow=False,
    )

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
    )


def create_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    horizontal: bool = False,
    height: int = 350,
) -> dcc.Graph:
    """Create a bar chart.

    Args:
        labels: Bar labels
        values: Bar values
        title: Chart title
        horizontal: If True, create horizontal bar chart
        height: Chart height

    Returns:
        Dash Graph component
    """
    colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in values]

    if horizontal:
        fig = go.Figure(data=[go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
        )])
    else:
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>",
        )])

    # Add zero line
    if horizontal:
        fig.add_vline(x=0, line=dict(color=COLORS["text_secondary"], width=1))
    else:
        fig.add_hline(y=0, line=dict(color=COLORS["text_secondary"], width=1))

    fig.update_layout(
        title=title if title else None,
        height=height,
        bargap=0.3,
    )

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
    )


def create_histogram(
    data: pd.Series,
    title: str = "",
    bins: int = 50,
    show_normal: bool = True,
    height: int = 350,
) -> dcc.Graph:
    """Create a histogram with optional normal distribution overlay.

    Args:
        data: Data series
        title: Chart title
        bins: Number of bins
        show_normal: Whether to show normal distribution overlay
        height: Chart height

    Returns:
        Dash Graph component
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name="Distribution",
        marker_color=COLORS["primary"],
        opacity=0.8,
        hovertemplate="Value: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))

    if show_normal:
        x_range = np.linspace(data.min(), data.max(), 100)
        from scipy import stats
        normal = stats.norm.pdf(x_range, data.mean(), data.std())
        scale = len(data) * (data.max() - data.min()) / bins

        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal * scale,
            name="Normal",
            line=dict(color=COLORS["warning"], dash="dash", width=2),
            hoverinfo="skip",
        ))

    # Add mean line
    fig.add_vline(
        x=data.mean(),
        line=dict(color=COLORS["success"], dash="dot", width=2),
        annotation_text=f"Mean: {data.mean():.4f}",
        annotation_position="top",
        annotation_font_color=COLORS["success"],
    )

    fig.update_layout(
        title=title if title else None,
        height=height,
        bargap=0.1,
        xaxis_title="Value",
        yaxis_title="Frequency",
    )

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
    )


def create_regime_timeline(
    dates: pd.DatetimeIndex,
    regimes: list[str],
    title: str = "",
    height: int = 200,
) -> dcc.Graph:
    """Create a regime timeline chart.

    Args:
        dates: Date index
        regimes: Regime labels for each date
        title: Chart title
        height: Chart height

    Returns:
        Dash Graph component
    """
    color_map = {
        "Expansion": COLORS["success"],
        "Peak": COLORS["warning"],
        "Contraction": COLORS["danger"],
        "Trough": COLORS["secondary"],
    }

    df = pd.DataFrame({"date": dates, "regime": regimes})

    fig = px.bar(
        df,
        x="date",
        y=[1] * len(df),
        color="regime",
        color_discrete_map=color_map,
    )

    fig.update_layout(
        title=title if title else None,
        height=height,
        showlegend=True,
        yaxis_visible=False,
        xaxis_title="",
        bargap=0,
    )

    return dcc.Graph(
        figure=fig,
        config=get_chart_config(),
        style={"height": f"{height}px"},
    )


def create_chart_container(
    title: str,
    chart: dcc.Graph,
    subtitle: Optional[str] = None,
) -> html.Div:
    """Wrap chart in styled container.

    Args:
        title: Chart title
        chart: Chart component
        subtitle: Optional subtitle

    Returns:
        Styled chart container
    """
    header_children = [html.Div(title, className="chart-title")]

    if subtitle:
        header_children.append(
            html.Div(
                subtitle,
                style={"color": "var(--text-muted)", "fontSize": "0.875rem"},
            )
        )

    return html.Div(
        className="chart-container",
        children=[
            html.Div(header_children),
            chart,
        ],
    )
