"""Backtest page callbacks."""

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np

from src.ui.components import (
    create_line_chart,
    create_pie_chart,
    create_area_chart,
    create_histogram,
    create_chart_container,
)
from src.ui.services import DataService
from src.ui.layouts.backtest import (
    create_configure_tab,
    create_results_tab,
    create_analysis_tab,
)


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
        [
            Output("backtest-results-store", "data"),
            Output("backtest-tabs", "value"),
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

        # Parse symbols
        symbols = [s.strip().upper() for s in universe.split(",")]

        # Generate simulated backtest results
        service = DataService(demo_mode=True)
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

        return results, "results"

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
