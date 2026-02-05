"""Data Explorer page callbacks."""

import logging
import traceback

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from src.ui.components import (
    create_line_chart,
    create_chart_container,
    create_error_alert,
)
from src.ui.services import DataService
from src.ui.layouts.data_explorer import (
    create_stocks_tab,
    create_macro_tab,
    create_crypto_tab,
)

logger = logging.getLogger(__name__)


def register_data_explorer_callbacks(app):
    """Register data explorer callbacks."""

    @app.callback(
        Output("data-explorer-content", "children"),
        [Input("data-explorer-tabs", "value")],
    )
    def render_data_explorer_tab(tab):
        """Render data explorer tab content."""
        if tab == "stocks":
            return create_stocks_tab()
        elif tab == "macro":
            return create_macro_tab()
        elif tab == "crypto":
            return create_crypto_tab()
        return create_stocks_tab()

    @app.callback(
        Output("stock-chart-area", "children"),
        [Input("generate-stock-btn", "n_clicks")],
        [
            State("stock-symbols-input", "value"),
            State("stock-periods-slider", "value"),
        ],
    )
    def generate_stock_chart(n_clicks, symbols, periods):
        """Generate stock price chart."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # Validate inputs
            if not symbols or not symbols.strip():
                return create_error_alert(
                    "Please enter at least one stock symbol."
                )

            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

            if not symbol_list:
                return create_error_alert(
                    "Please enter valid stock symbols separated by commas."
                )

            service = DataService(demo_mode=False)
            prices = service.get_prices(symbol_list, periods=periods)

            if prices.empty:
                return create_error_alert(
                    f"No price data available for symbols: {', '.join(symbol_list)}"
                )

            chart = create_line_chart(prices, height=450)
            return create_chart_container(
                "Stock Prices",
                chart,
                subtitle=f"Showing {periods} days of data",
            )

        except ValueError as e:
            logger.warning(f"Stock chart validation error: {e}")
            return create_error_alert(f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Stock chart error: {e}\n{traceback.format_exc()}")
            return create_error_alert(
                "Unable to load stock data. Please check your symbols and try again."
            )

    @app.callback(
        Output("macro-chart-area", "children"),
        [Input("macro-indicator-dropdown", "value")],
    )
    def update_macro_chart(indicators):
        """Update macro indicators chart."""
        if not indicators:
            raise PreventUpdate

        service = DataService(demo_mode=False)
        macro_data = service.get_macro_data(48)

        # Filter selected indicators
        selected_data = macro_data[indicators]

        chart = create_line_chart(selected_data, height=400)
        return create_chart_container(
            "Macro Indicators",
            chart,
            subtitle="Monthly economic data",
        )
