"""Portfolio page callbacks."""

import logging
import traceback

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np

from src.ui.components import (
    create_pie_chart,
    create_bar_chart,
    create_chart_container,
    create_empty_state,
    create_error_alert,
    create_warning_alert,
)
from src.ui.services import DataService
from src.ui.layouts.portfolio import (
    create_holdings_tab,
    create_optimization_tab,
)

logger = logging.getLogger(__name__)


def register_portfolio_callbacks(app):
    """Register portfolio callbacks."""

    @app.callback(
        Output("portfolio-tab-content", "children"),
        [Input("portfolio-tabs", "value")],
        [State("portfolio-store", "data")],
    )
    def render_portfolio_tab(tab, portfolio):
        """Render portfolio tab content."""
        if tab == "holdings":
            return create_holdings_tab(portfolio)
        elif tab == "optimization":
            return create_optimization_tab()
        return create_holdings_tab(portfolio)

    @app.callback(
        [
            Output("portfolio-store", "data"),
            Output("portfolio-tabs", "value"),
        ],
        [Input("add-position-btn", "n_clicks")],
        [
            State("add-symbol-input", "value"),
            State("add-weight-input", "value"),
            State("portfolio-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def add_position(n_clicks, symbol, weight, current_portfolio):
        """Add position to portfolio."""
        if not n_clicks or not symbol:
            raise PreventUpdate

        portfolio = current_portfolio or {}
        symbol = symbol.strip().upper()
        weight = float(weight) / 100  # Convert to decimal

        portfolio[symbol] = weight

        return portfolio, "holdings"

    @app.callback(
        Output("optimization-results", "children"),
        [Input("optimize-btn", "n_clicks")],
        [
            State("opt-method-dropdown", "value"),
            State("opt-assets-input", "value"),
            State("optimization-results", "children"),
        ],
    )
    def run_optimization(n_clicks, method, assets, existing_results):
        """Run portfolio optimization."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # Validate inputs
            if not assets or not assets.strip():
                return create_error_alert(
                    "Please enter at least one asset symbol."
                )

            symbols = [s.strip().upper() for s in assets.split(",") if s.strip()]

            if not symbols:
                return create_error_alert(
                    "Please enter valid asset symbols separated by commas."
                )

            if len(symbols) < 2:
                return create_error_alert(
                    "Portfolio optimization requires at least 2 assets."
                )

            n_assets = len(symbols)

            # Generate simulated optimization results based on method
            service = DataService(demo_mode=True)
            returns = service.get_returns(symbols, periods=252)

            if returns.empty:
                return create_error_alert(
                    f"No return data available for symbols: {', '.join(symbols)}"
                )

            if method == "equal_weight":
                weights = {s: 1 / n_assets for s in symbols}

            elif method == "min_variance":
                # Simulate minimum variance (slightly adjusted from equal)
                np.random.seed(42)
                base = np.ones(n_assets) / n_assets
                noise = np.random.randn(n_assets) * 0.05
                weights_arr = base + noise
                weights_arr = np.maximum(weights_arr, 0)
                weights_arr /= weights_arr.sum()
                weights = dict(zip(symbols, weights_arr))

            elif method == "max_sharpe":
                # Simulate max Sharpe (momentum-biased)
                mean_returns = returns.mean()
                ranked = mean_returns.rank()
                weights_arr = ranked / ranked.sum()
                weights = dict(zip(symbols, weights_arr.values))

            elif method == "risk_parity":
                # Simulate risk parity (inverse volatility)
                vol = returns.std()
                if (vol == 0).any():
                    return create_error_alert(
                        "Risk parity optimization failed: some assets have zero volatility."
                    )
                inv_vol = 1 / vol
                weights_arr = inv_vol / inv_vol.sum()
                weights = dict(zip(symbols, weights_arr.values))

            else:
                weights = {s: 1 / n_assets for s in symbols}

            # Calculate expected metrics
            portfolio_return = sum(returns[s].mean() * w for s, w in weights.items()) * 252
            portfolio_vol = sum(returns[s].std() * w for s, w in weights.items()) * np.sqrt(252)
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

            # Build result components
            result_components = []

            # Add warning if overwriting existing results
            if existing_results is not None:
                result_components.append(
                    create_warning_alert(
                        "Previous optimization results have been replaced with new results."
                    )
                )

            result_components.extend([
                create_chart_container(
                    "Optimized Weights",
                    create_bar_chart(
                        list(weights.keys()),
                        [w * 100 for w in weights.values()],
                        height=300,
                    ),
                ),
                html.Div(
                    className="chart-container",
                    style={"marginTop": "1rem"},
                    children=[
                        html.Div("Expected Performance", className="chart-title"),
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="col col-4",
                                    children=[
                                        html.Div("Expected Return", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                        html.Div(f"{portfolio_return:.1%}", style={"fontSize": "1.5rem", "fontWeight": "600", "color": "var(--text-primary)"}),
                                    ],
                                ),
                                html.Div(
                                    className="col col-4",
                                    children=[
                                        html.Div("Expected Volatility", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                        html.Div(f"{portfolio_vol:.1%}", style={"fontSize": "1.5rem", "fontWeight": "600", "color": "var(--text-primary)"}),
                                    ],
                                ),
                                html.Div(
                                    className="col col-4",
                                    children=[
                                        html.Div("Sharpe Ratio", style={"color": "var(--text-secondary)", "fontSize": "0.875rem"}),
                                        html.Div(f"{sharpe:.2f}", style={"fontSize": "1.5rem", "fontWeight": "600", "color": "var(--text-primary)"}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ])

            return html.Div(result_components)

        except ValueError as e:
            logger.warning(f"Portfolio optimization validation error: {e}")
            return create_error_alert(f"Invalid input: {str(e)}")
        except ZeroDivisionError:
            logger.warning("Portfolio optimization: Division by zero encountered")
            return create_error_alert(
                "Optimization failed: Unable to calculate weights due to data issues."
            )
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}\n{traceback.format_exc()}")
            return create_error_alert(
                "An unexpected error occurred during optimization. Please try again."
            )
