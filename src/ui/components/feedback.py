"""Feedback components for loading states, errors, and notifications."""

from dash import html, dcc

from src.ui.components.theme import COLORS


def create_loading_spinner(component_id: str, children=None, spinner_type: str = "circle") -> dcc.Loading:
    """Create a loading spinner wrapper.

    Args:
        component_id: ID for the loading component
        children: Child components to wrap
        spinner_type: Type of spinner ('circle', 'dot', 'default', 'cube', 'graph')

    Returns:
        dcc.Loading component wrapping the children
    """
    return dcc.Loading(
        id=component_id,
        type=spinner_type,
        color=COLORS["primary"],
        children=children or html.Div(),
        style={"minHeight": "100px"},
        parent_style={"minHeight": "100px"},
    )


def create_error_alert(message: str, error_id: str = None, dismissible: bool = True) -> html.Div:
    """Create a styled error message alert.

    Args:
        message: Error message to display
        error_id: Optional ID for the error div
        dismissible: Whether to show a close button (visual only)

    Returns:
        Styled error message div
    """
    close_button = None
    if dismissible:
        close_button = html.Span(
            "x",
            style={
                "cursor": "pointer",
                "marginLeft": "auto",
                "fontSize": "1.25rem",
                "fontWeight": "bold",
                "opacity": "0.7",
            },
        )

    return html.Div(
        id=error_id,
        className="error-alert",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "0.75rem",
                },
                children=[
                    html.Span(
                        "!",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "24px",
                            "height": "24px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["danger"],
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "0.875rem",
                            "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                "Error",
                                style={
                                    "fontWeight": "600",
                                    "color": COLORS["danger"],
                                    "marginBottom": "0.25rem",
                                },
                            ),
                            html.Div(
                                message,
                                style={
                                    "color": COLORS["text_secondary"],
                                    "fontSize": "0.875rem",
                                    "lineHeight": "1.5",
                                },
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    close_button,
                ],
            ),
        ],
        style={
            "backgroundColor": f"rgba(239, 68, 68, 0.1)",
            "border": f"1px solid {COLORS['danger']}",
            "borderRadius": "12px",
            "padding": "1rem",
            "marginBottom": "1rem",
        },
    )


def create_success_toast(message: str, toast_id: str = None) -> html.Div:
    """Create a styled success notification toast.

    Args:
        message: Success message to display
        toast_id: Optional ID for the toast div

    Returns:
        Styled success toast div
    """
    return html.Div(
        id=toast_id,
        className="success-toast",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "0.75rem",
                },
                children=[
                    html.Span(
                        "check",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "24px",
                            "height": "24px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["success"],
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "0.75rem",
                            "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        message,
                        style={
                            "color": COLORS["text"],
                            "fontSize": "0.875rem",
                            "fontWeight": "500",
                        },
                    ),
                ],
            ),
        ],
        style={
            "backgroundColor": f"rgba(16, 185, 129, 0.1)",
            "border": f"1px solid {COLORS['success']}",
            "borderRadius": "12px",
            "padding": "1rem",
            "marginBottom": "1rem",
            "animation": "fadeIn 0.3s ease-in-out",
        },
    )


def create_warning_alert(message: str, alert_id: str = None) -> html.Div:
    """Create a styled warning message alert.

    Args:
        message: Warning message to display
        alert_id: Optional ID for the alert div

    Returns:
        Styled warning message div
    """
    return html.Div(
        id=alert_id,
        className="warning-alert",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "0.75rem",
                },
                children=[
                    html.Span(
                        "!",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "24px",
                            "height": "24px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["warning"],
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "0.875rem",
                            "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        message,
                        style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "0.875rem",
                        },
                    ),
                ],
            ),
        ],
        style={
            "backgroundColor": f"rgba(245, 158, 11, 0.1)",
            "border": f"1px solid {COLORS['warning']}",
            "borderRadius": "12px",
            "padding": "1rem",
            "marginBottom": "1rem",
        },
    )


def create_info_alert(message: str, alert_id: str = None) -> html.Div:
    """Create a styled info message alert.

    Args:
        message: Info message to display
        alert_id: Optional ID for the alert div

    Returns:
        Styled info message div
    """
    return html.Div(
        id=alert_id,
        className="info-alert",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "0.75rem",
                },
                children=[
                    html.Span(
                        "i",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "24px",
                            "height": "24px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["info"],
                            "color": "white",
                            "fontWeight": "bold",
                            "fontSize": "0.875rem",
                            "fontStyle": "italic",
                            "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        message,
                        style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "0.875rem",
                        },
                    ),
                ],
            ),
        ],
        style={
            "backgroundColor": f"rgba(59, 130, 246, 0.1)",
            "border": f"1px solid {COLORS['info']}",
            "borderRadius": "12px",
            "padding": "1rem",
            "marginBottom": "1rem",
        },
    )


def create_confirmation_dialog(
    dialog_id: str,
    title: str,
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
) -> html.Div:
    """Create a confirmation dialog component.

    Args:
        dialog_id: Base ID for the dialog components
        title: Dialog title
        message: Confirmation message
        confirm_text: Text for confirm button
        cancel_text: Text for cancel button

    Returns:
        Confirmation dialog div (hidden by default)
    """
    return html.Div(
        id=f"{dialog_id}-container",
        className="confirmation-dialog",
        style={"display": "none"},
        children=[
            html.Div(
                className="dialog-overlay",
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "right": "0",
                    "bottom": "0",
                    "backgroundColor": "rgba(0, 0, 0, 0.5)",
                    "zIndex": "1000",
                },
            ),
            html.Div(
                className="dialog-content",
                style={
                    "position": "fixed",
                    "top": "50%",
                    "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "backgroundColor": COLORS["background"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "16px",
                    "padding": "1.5rem",
                    "minWidth": "350px",
                    "maxWidth": "450px",
                    "zIndex": "1001",
                    "boxShadow": "0 20px 40px rgba(0, 0, 0, 0.3)",
                },
                children=[
                    html.Div(
                        title,
                        style={
                            "fontSize": "1.25rem",
                            "fontWeight": "600",
                            "color": COLORS["text"],
                            "marginBottom": "0.75rem",
                        },
                    ),
                    html.Div(
                        message,
                        style={
                            "color": COLORS["text_secondary"],
                            "fontSize": "0.9rem",
                            "marginBottom": "1.5rem",
                            "lineHeight": "1.5",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "0.75rem",
                            "justifyContent": "flex-end",
                        },
                        children=[
                            html.Button(
                                cancel_text,
                                id=f"{dialog_id}-cancel",
                                className="btn-secondary",
                                style={
                                    "padding": "0.5rem 1rem",
                                    "borderRadius": "8px",
                                    "border": f"1px solid {COLORS['border']}",
                                    "backgroundColor": "transparent",
                                    "color": COLORS["text_secondary"],
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                confirm_text,
                                id=f"{dialog_id}-confirm",
                                className="btn-primary",
                                style={
                                    "padding": "0.5rem 1rem",
                                    "borderRadius": "8px",
                                    "border": "none",
                                    "backgroundColor": COLORS["primary"],
                                    "color": "white",
                                    "cursor": "pointer",
                                    "fontWeight": "500",
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
