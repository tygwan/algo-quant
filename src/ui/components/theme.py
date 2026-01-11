"""Dark theme configuration for Dash application."""

import plotly.graph_objects as go
import plotly.io as pio

# Color palette (Claude-inspired dark theme)
COLORS = {
    "primary": "#8b5cf6",
    "primary_hover": "#a78bfa",
    "secondary": "#6366f1",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "background": "#1a1a2e",
    "background_dark": "#0f0f23",
    "surface": "rgba(255,255,255,0.03)",
    "surface_hover": "rgba(255,255,255,0.05)",
    "border": "rgba(255,255,255,0.06)",
    "border_hover": "rgba(255,255,255,0.1)",
    "text": "#ffffff",
    "text_secondary": "#a0a0a0",
    "text_muted": "#666666",
    "grid": "rgba(255,255,255,0.05)",
}

# Chart color sequence
CHART_COLORS = [
    "#8b5cf6",  # Purple
    "#10b981",  # Green
    "#3b82f6",  # Blue
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#84cc16",  # Lime
]

# CSS Variables for easy theming
CSS_VARIABLES = f"""
:root {{
    --primary: {COLORS['primary']};
    --primary-hover: {COLORS['primary_hover']};
    --secondary: {COLORS['secondary']};
    --success: {COLORS['success']};
    --warning: {COLORS['warning']};
    --danger: {COLORS['danger']};
    --info: {COLORS['info']};
    --bg: {COLORS['background']};
    --bg-dark: {COLORS['background_dark']};
    --surface: {COLORS['surface']};
    --surface-hover: {COLORS['surface_hover']};
    --border: {COLORS['border']};
    --border-hover: {COLORS['border_hover']};
    --text: {COLORS['text']};
    --text-secondary: {COLORS['text_secondary']};
    --text-muted: {COLORS['text_muted']};
    --grid: {COLORS['grid']};

    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --border-radius: 12px;
    --border-radius-lg: 16px;
    --transition: all 0.2s ease;
}}
"""

# Plotly dark theme template
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color=COLORS["text_secondary"],
            size=12,
        ),
        title=dict(
            font=dict(color=COLORS["text"], size=16),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            linecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
            title_font=dict(color=COLORS["text_secondary"]),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            linecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
            title_font=dict(color=COLORS["text_secondary"]),
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_secondary"]),
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(30, 30, 63, 0.95)",
            font_size=12,
            font_family="Inter, sans-serif",
        ),
        margin=dict(l=50, r=20, t=40, b=50),
        colorway=CHART_COLORS,
    )
)

# Register template
pio.templates["algo_quant_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "algo_quant_dark"


def get_chart_config() -> dict:
    """Get common Plotly chart config."""
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "algo_quant_chart",
            "height": 600,
            "width": 1000,
            "scale": 2,
        },
    }


def get_table_style() -> dict:
    """Get DataTable dark theme style."""
    return {
        "style_header": {
            "backgroundColor": COLORS["surface"],
            "color": COLORS["text"],
            "fontWeight": "600",
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "12px 16px",
        },
        "style_cell": {
            "backgroundColor": "transparent",
            "color": COLORS["text_secondary"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "12px 16px",
            "fontFamily": "Inter, sans-serif",
            "fontSize": "14px",
            "textAlign": "left",
        },
        "style_data_conditional": [
            {
                "if": {"state": "active"},
                "backgroundColor": COLORS["surface_hover"],
                "border": f"1px solid {COLORS['border_hover']}",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": f"rgba(139, 92, 246, 0.1)",
                "border": f"1px solid {COLORS['primary']}",
            },
        ],
        "style_table": {
            "borderRadius": "12px",
            "overflow": "hidden",
        },
    }
