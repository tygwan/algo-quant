"""App-shell callbacks (navigation and responsive behavior)."""

from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate


def register_shell_callbacks(app):
    """Register shell-level callbacks."""

    @app.callback(
        [Output("sidebar", "className"), Output("sidebar-overlay", "className")],
        [
            Input("sidebar-toggle", "n_clicks"),
            Input("sidebar-overlay", "n_clicks"),
            Input("url", "pathname"),
        ],
        [State("sidebar", "className")],
        prevent_initial_call=True,
    )
    def toggle_sidebar(toggle_clicks, overlay_clicks, pathname, sidebar_class):
        """Toggle sidebar on mobile and close overlay on navigation."""
        del toggle_clicks, overlay_clicks, pathname
        trigger = callback_context.triggered_id
        if not trigger:
            raise PreventUpdate

        base_sidebar = "sidebar"
        base_overlay = "sidebar-overlay"

        if trigger == "sidebar-toggle":
            is_open = "open" not in (sidebar_class or "")
            return (
                f"{base_sidebar} open" if is_open else base_sidebar,
                f"{base_overlay} active" if is_open else base_overlay,
            )

        # close on overlay click or route change
        return base_sidebar, base_overlay
