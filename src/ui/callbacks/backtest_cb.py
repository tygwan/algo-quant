"""Backtest page callbacks."""

import logging
import traceback

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np

from src.ui.components import (
    create_line_chart,
    create_pie_chart,
    create_area_chart,
    create_histogram,
    create_chart_container,
    create_error_alert,
)
from src.ui.services import DataService
from src.ui.layouts.backtest import (
    create_configure_tab,
    create_results_tab,
    create_analysis_tab,
)

logger = logging.getLogger(__name__)


def register_backtest_callbacks(app):
    """Register backtest callbacks."""

    @app.callback(
        Output("backtest-tab-content", "children"),
        [Input("backtest-tabs", "value")],
        [State("backtest-results-store", "data")],
    )
    def render_backtest_tab(tab, results):
        """Render backtest tab content."""
        if tab == "configure":
            return create_configure_tab()
        elif tab == "results":
            return create_results_tab(results)
        elif tab == "analysis":
            return create_analysis_tab(results)
        return create_configure_tab()

    @app.callback(
        Output("universe-input", "value"),
        [Input("store-screened-universe", "data")],
        [State("universe-input", "value")],
        prevent_initial_call=True,
    )
    def apply_screened_universe(screened_data, current_value):
        """Apply screened tickers from Live Analyzer to backtest universe."""
        if not screened_data:
            raise PreventUpdate

        tickers = screened_data.get("tickers", [])
        if not tickers:
            raise PreventUpdate

        new_value = ",".join(tickers)
        if new_value == (current_value or ""):
            raise PreventUpdate

        return new_value

    @app.callback(
        [
            Output("backtest-results-store", "data"),
            Output("backtest-tabs", "value"),
            Output("run-backtest-btn", "disabled"),
            Output("backtest-error", "children"),
        ],
        [Input("run-backtest-btn", "n_clicks")],
        [
            State("strategy-dropdown", "value"),
            State("universe-input", "value"),
            State("rebalance-dropdown", "value"),
            State("capital-input", "value"),
            State("commission-slider", "value"),
            State("periods-slider", "value"),
        ],
    )
    def run_backtest(n_clicks, strategy, universe, rebalance, capital, commission, periods):
        """Run backtest simulation."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # Validate inputs
            if not universe or not universe.strip():
                return no_update, no_update, False, create_error_alert(
                    "Please enter at least one stock symbol."
                )

            if capital is None or capital <= 0:
                return no_update, no_update, False, create_error_alert(
                    "Initial capital must be greater than zero."
                )

            # Parse symbols
            symbols = [s.strip().upper() for s in universe.split(",") if s.strip()]

            if not symbols:
                return no_update, no_update, False, create_error_alert(
                    "Please enter valid stock symbols separated by commas."
                )

            # Use real market data by default (falls back to sample data on fetch failure)
            service = DataService(demo_mode=False)
            prices = service.get_prices(symbols, periods=periods)

            # Calculate equal weight portfolio returns
            returns = prices.pct_change().dropna()
            portfolio_returns = returns.mean(axis=1)

            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            cagr = (1 + total_return) ** (252 / periods) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = cagr / volatility if volatility > 0 else 0

            # Calculate drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.cummax()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # Create results
            results = {
                "metrics": {
                    "total_return": total_return,
                    "cagr": cagr,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "volatility": volatility,
                },
                "portfolio_value": (capital * (1 + portfolio_returns).cumprod()).tolist(),
                "dates": portfolio_returns.index.strftime("%Y-%m-%d").tolist(),
                "returns": portfolio_returns.tolist(),
                "drawdown": drawdown.tolist(),
                "allocation": {s: 1 / len(symbols) for s in symbols},
                "config": {
                    "strategy": strategy,
                    "symbols": symbols,
                    "rebalance": rebalance,
                    "capital": capital,
                    "commission": commission,
                },
            }

            return results, "results", False, None

        except ValueError as e:
            logger.warning(f"Backtest validation error: {e}")
            return no_update, no_update, False, create_error_alert(
                f"Invalid input: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Backtest error: {e}\n{traceback.format_exc()}")
            return no_update, no_update, False, create_error_alert(
                "An unexpected error occurred while running the backtest. Please try again."
            )

    @app.callback(
        [
            Output("portfolio-value-chart", "children"),
            Output("allocation-chart", "children"),
        ],
        [Input("backtest-results-store", "data")],
    )
    def update_result_charts(results):
        """Update backtest result charts."""
        if not results:
            raise PreventUpdate

        # Portfolio value chart
        dates = pd.to_datetime(results["dates"])
        values = pd.Series(results["portfolio_value"], index=dates)
        values_df = pd.DataFrame({"Portfolio": values})

        portfolio_chart = create_chart_container(
            "Portfolio Value",
            create_area_chart(values_df, height=350),
        )

        # Allocation chart
        allocation = results["allocation"]
        allocation_chart = create_chart_container(
            "Allocation",
            create_pie_chart(
                list(allocation.keys()),
                list(allocation.values()),
                height=350,
            ),
        )

        return portfolio_chart, allocation_chart

    @app.callback(
        Output("analysis-chart-area", "children"),
        [Input("analysis-type-dropdown", "value")],
        [State("backtest-results-store", "data")],
    )
    def update_analysis_chart(analysis_type, results):
        """Update analysis chart based on type."""
        if not results:
            raise PreventUpdate

        dates = pd.to_datetime(results["dates"])

        if analysis_type == "distribution":
            returns = np.array(results["returns"])
            chart = create_histogram(returns, height=400)
            return create_chart_container(
                "Daily Returns Distribution",
                chart,
                subtitle=f"Mean: {np.mean(returns):.4f}, Std: {np.std(returns):.4f}",
            )

        elif analysis_type == "drawdown":
            drawdown = pd.Series(results["drawdown"], index=dates)
            drawdown_df = pd.DataFrame({"Drawdown": drawdown})
            chart = create_area_chart(drawdown_df, height=400)
            return create_chart_container(
                "Drawdown",
                chart,
                subtitle=f"Max Drawdown: {min(results['drawdown']):.2%}",
            )

        elif analysis_type == "rolling":
            returns = pd.Series(results["returns"], index=dates)
            rolling_sharpe = (
                returns.rolling(21).mean() / returns.rolling(21).std() * np.sqrt(252)
            )
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)

            rolling_df = pd.DataFrame({
                "Rolling Sharpe (21d)": rolling_sharpe,
                "Rolling Volatility (21d)": rolling_vol,
            })
            chart = create_line_chart(rolling_df, height=400)
            return create_chart_container(
                "Rolling Metrics",
                chart,
            )

        raise PreventUpdate
