"""Theme configuration for Dash application."""

import plotly.graph_objects as go
import plotly.io as pio

# Color palette (Aurora day-light theme)
COLORS = {
    "primary": "#0F766E",
    "primary_hover": "#0D9488",
    "secondary": "#C2410C",
    "success": "#15803D",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#0EA5E9",
    "background": "#F3F7FB",
    "background_dark": "#EAF0F7",
    "surface": "rgba(255,255,255,0.85)",
    "surface_hover": "#FFFFFF",
    "border": "#D3DEE9",
    "border_hover": "#B7C7D9",
    "text": "#102A43",
    "text_secondary": "#486581",
    "text_muted": "#7B8794",
    "grid": "#DFE8F2",
}

# Chart color sequence
CHART_COLORS = [
    "#0F766E",  # Teal
    "#C2410C",  # Ember
    "#0EA5E9",  # Sky
    "#2563EB",  # Blue
    "#16A34A",  # Green
    "#7C3AED",  # Indigo
    "#E11D48",  # Rose
    "#CA8A04",  # Gold
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

    --font-family: 'Space Grotesk', 'Avenir Next', 'Segoe UI', sans-serif;
    --font-mono: 'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace;
    --border-radius: 14px;
    --border-radius-lg: 18px;
    --transition: all 0.2s ease;
}}
"""

# Plotly light theme template
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(
            family="Space Grotesk, sans-serif",
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
            bgcolor="rgba(240, 248, 255, 0.98)",
            bordercolor=COLORS["border_hover"],
            font_color=COLORS["text"],
            font_size=12,
            font_family="Space Grotesk, sans-serif",
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
            "fontFamily": "Space Grotesk, sans-serif",
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
                "backgroundColor": "rgba(15, 118, 110, 0.10)",
                "border": f"1px solid {COLORS['primary']}",
            },
        ],
        "style_table": {
            "borderRadius": "12px",
            "overflow": "hidden",
        },
    }
