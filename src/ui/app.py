"""Algo-Quant Dashboard - Streamlit Application."""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - must be first
st.set_page_config(
    page_title="Algo-Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import utilities after page config
from src.ui.utils import (
    generate_sample_prices,
    generate_sample_returns,
    plot_price_chart,
    plot_returns_distribution,
    plot_drawdown,
    plot_allocation_pie,
    plot_performance_comparison,
    plot_factor_exposure,
    format_metrics_table,
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}


def main():
    """Main application entry point."""
    # Sidebar
    st.sidebar.title("📈 Algo-Quant")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Data Explorer", "🔬 Factor Analysis",
         "🌡️ Regime Monitor", "📈 Backtest", "💼 Portfolio"],
        label_visibility="collapsed",
    )

    # Demo mode toggle
    st.sidebar.markdown("---")
    demo_mode = st.sidebar.checkbox("🎮 Demo Mode", value=True, help="Use sample data")
    st.session_state.demo_mode = demo_mode

    if demo_mode:
        st.sidebar.success("Using sample data")
    else:
        st.sidebar.warning("Configure API keys")

    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🔬 Factor Analysis":
        show_factor_analysis()
    elif page == "🌡️ Regime Monitor":
        show_regime_monitor()
    elif page == "📈 Backtest":
        show_backtest()
    elif page == "💼 Portfolio":
        show_portfolio()


def show_dashboard():
    """Show main dashboard."""
    st.markdown('<p class="main-header">Dashboard Overview</p>', unsafe_allow_html=True)

    # Generate demo data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    prices = generate_sample_prices(symbols, periods=252)
    returns = prices.pct_change().dropna()
    portfolio_values = 100000 * (1 + returns.mean(axis=1)).cumprod()

    # Metrics
    total_return = (portfolio_values.iloc[-1] / 100000 - 1) * 100
    vol = returns.mean(axis=1).std() * np.sqrt(252) * 100
    sharpe = (returns.mean(axis=1).mean() * 252) / (returns.mean(axis=1).std() * np.sqrt(252))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Portfolio Value", f"${portfolio_values.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
    with col2:
        st.metric("Volatility", f"{vol:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col4:
        st.metric("Current Regime", "Expansion", "Stable")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio Performance")
        fig = plot_price_chart(portfolio_values.to_frame("Portfolio"), "Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Asset Allocation")
        weights = {"AAPL": 0.30, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.20}
        fig = plot_allocation_pie(weights)
        st.plotly_chart(fig, use_container_width=True)

    # Price chart
    st.subheader("📈 Asset Prices")
    fig = plot_price_chart(prices, "Stock Prices (Indexed)")
    st.plotly_chart(fig, use_container_width=True)


def show_data_explorer():
    """Show data exploration page."""
    st.markdown('<p class="main-header">Data Explorer</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Stock Data", "🏛️ Macro Indicators", "💰 Crypto"])

    with tab1:
        col1, col2 = st.columns([1, 3])

        with col1:
            symbols = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
            periods = st.slider("Days", 60, 500, 252)

            if st.button("Generate Data", key="gen_stock"):
                symbol_list = [s.strip() for s in symbols.split(",")]
                prices = generate_sample_prices(symbol_list, periods)
                st.session_state.stock_data = prices

        with col2:
            if "stock_data" in st.session_state:
                prices = st.session_state.stock_data
                fig = plot_price_chart(prices, "Stock Prices")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(prices.tail(10), use_container_width=True)
            else:
                st.info("Click 'Generate Data' to view sample stock prices")

    with tab2:
        st.subheader("Macroeconomic Indicators")

        # Generate demo macro data
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        macro_data = pd.DataFrame({
            "GDP Growth (%)": np.random.normal(2.5, 1.0, 48).cumsum() / 10,
            "Unemployment (%)": 4 + np.random.normal(0, 0.5, 48).cumsum() / 5,
            "10Y-2Y Spread (%)": 1.5 + np.random.normal(0, 0.2, 48).cumsum() / 3,
        }, index=dates)

        selected = st.multiselect(
            "Select Indicators",
            macro_data.columns.tolist(),
            default=["GDP Growth (%)"],
        )

        if selected:
            fig = plot_price_chart(macro_data[selected], "Macro Indicators")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Cryptocurrency Data")

        crypto_symbols = ["BTC", "ETH", "BNB"]
        crypto_prices = generate_sample_prices(crypto_symbols, 365, "2024-01-01")
        crypto_prices = crypto_prices * 300  # Scale to crypto-like prices

        fig = plot_price_chart(crypto_prices, "Crypto Prices (Demo)")
        st.plotly_chart(fig, use_container_width=True)


def show_factor_analysis():
    """Show factor analysis page."""
    st.markdown('<p class="main-header">Factor Analysis</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Factor Returns", "⚖️ Factor Exposure"])

    with tab1:
        st.subheader("Fama-French Factor Returns")

        # Generate demo factor data
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        factor_data = pd.DataFrame({
            "Mkt-RF": np.random.normal(0.0004, 0.01, 252),
            "SMB": np.random.normal(0.0001, 0.005, 252),
            "HML": np.random.normal(0.0001, 0.006, 252),
            "RMW": np.random.normal(0.0001, 0.004, 252),
            "CMA": np.random.normal(0.0001, 0.004, 252),
        }, index=dates)

        model = st.selectbox("Model", ["FF3 (3-Factor)", "FF5 (5-Factor)"])

        if model == "FF3 (3-Factor)":
            factors = ["Mkt-RF", "SMB", "HML"]
        else:
            factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

        # Cumulative returns
        cum_returns = (1 + factor_data[factors]).cumprod()
        fig = plot_price_chart(cum_returns, "Cumulative Factor Returns")
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        stats = pd.DataFrame({
            "Mean (Ann.)": factor_data[factors].mean() * 252,
            "Vol (Ann.)": factor_data[factors].std() * np.sqrt(252),
            "Sharpe": (factor_data[factors].mean() * 252) / (factor_data[factors].std() * np.sqrt(252)),
        }).T
        st.dataframe(stats.style.format("{:.2%}"), use_container_width=True)

    with tab2:
        st.subheader("Portfolio Factor Exposure")

        # Demo factor exposures
        exposures = {
            "Market": 1.05,
            "Size (SMB)": -0.15,
            "Value (HML)": 0.25,
            "Profitability": 0.10,
            "Investment": -0.05,
        }

        fig = plot_factor_exposure(exposures)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - Market β > 1: More volatile than market
        - Negative SMB: Tilted toward large caps
        - Positive HML: Tilted toward value stocks
        """)


def show_regime_monitor():
    """Show regime monitoring page."""
    st.markdown('<p class="main-header">Market Regime Monitor</p>', unsafe_allow_html=True)

    # Demo regime
    regimes = ["Expansion", "Peak", "Contraction", "Trough"]
    current_regime = "Expansion"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Regime", current_regime)
    with col2:
        st.metric("Confidence", "78%")
    with col3:
        st.metric("Duration", "8 months")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Indicators", "📈 History"])

    with tab1:
        st.subheader("Economic Indicators")

        indicators = {
            "GDP Growth": {"value": "+2.3%", "signal": "🟢 Positive"},
            "Unemployment": {"value": "3.8%", "signal": "🟢 Low"},
            "10Y-2Y Spread": {"value": "+0.45%", "signal": "🟢 Normal"},
            "Fed Funds Rate": {"value": "5.25%", "signal": "🟡 High"},
            "PMI": {"value": "52.1", "signal": "🟢 Expanding"},
        }

        for name, data in indicators.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(name)
            with col2:
                st.text(data["value"])
            with col3:
                st.text(data["signal"])

    with tab2:
        st.subheader("Regime History")

        # Generate regime history
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        regime_values = np.random.choice([0, 1, 2, 3], 48, p=[0.5, 0.2, 0.15, 0.15])
        regime_names = ["Expansion", "Peak", "Contraction", "Trough"]

        regime_df = pd.DataFrame({
            "Regime": [regime_names[r] for r in regime_values],
        }, index=dates)

        # Color-coded bar chart
        import plotly.express as px
        fig = px.bar(
            regime_df.reset_index(),
            x="index",
            y=[1] * len(regime_df),
            color="Regime",
            color_discrete_map={
                "Expansion": "green",
                "Peak": "yellow",
                "Contraction": "red",
                "Trough": "orange",
            },
        )
        fig.update_layout(
            title="Regime Timeline",
            showlegend=True,
            yaxis_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def show_backtest():
    """Show backtesting page."""
    st.markdown('<p class="main-header">Strategy Backtesting</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚙️ Configure", "📊 Results", "📈 Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Strategy Settings**")
            strategy = st.selectbox(
                "Strategy Type",
                ["Equal Weight", "Risk Parity", "Momentum", "Mean-Variance"],
            )

            symbols = st.text_input("Universe", "AAPL,MSFT,GOOGL,AMZN,META")
            rebalance = st.selectbox("Rebalance", ["Monthly", "Quarterly", "Weekly"])

        with col2:
            st.markdown("**Backtest Settings**")
            capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
            commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01)
            periods = st.slider("Backtest Period (days)", 60, 756, 252)

        st.markdown("---")

        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                # Generate results
                symbol_list = [s.strip() for s in symbols.split(",")]
                prices = generate_sample_prices(symbol_list, periods)
                returns = prices.pct_change().dropna()

                # Simple equal weight strategy
                n_assets = len(symbol_list)
                portfolio_returns = returns.mean(axis=1)
                portfolio_values = capital * (1 + portfolio_returns).cumprod()

                # Calculate metrics
                total_return = portfolio_values.iloc[-1] / capital - 1
                vol = portfolio_returns.std() * np.sqrt(252)
                sharpe = (portfolio_returns.mean() * 252) / vol

                running_max = portfolio_values.cummax()
                drawdown = (portfolio_values - running_max) / running_max
                max_dd = drawdown.min()

                st.session_state.backtest_result = {
                    "portfolio_values": portfolio_values,
                    "returns": portfolio_returns,
                    "prices": prices,
                    "metrics": {
                        "Total Return": total_return,
                        "CAGR": (1 + total_return) ** (252 / periods) - 1,
                        "Volatility": vol,
                        "Sharpe Ratio": sharpe,
                        "Max Drawdown": max_dd,
                        "Win Rate": (portfolio_returns > 0).mean(),
                    },
                    "weights": {s: 1/n_assets for s in symbol_list},
                }

            st.success("Backtest complete! See Results tab.")

    with tab2:
        st.subheader("Backtest Results")

        if st.session_state.backtest_result:
            result = st.session_state.backtest_result
            metrics = result["metrics"]

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics['Total Return']:.1%}")
            with col2:
                st.metric("CAGR", f"{metrics['CAGR']:.1%}")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig = plot_price_chart(
                    result["portfolio_values"].to_frame("Portfolio"),
                    "Portfolio Value",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = plot_allocation_pie(result["weights"], "Final Allocation")
                st.plotly_chart(fig, use_container_width=True)

            # Full metrics table
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:.2%}" if isinstance(v, float) else str(v)}
                for k, v in metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run a backtest to see results")

    with tab3:
        st.subheader("Performance Analysis")

        if st.session_state.backtest_result:
            result = st.session_state.backtest_result

            analysis = st.selectbox(
                "Analysis Type",
                ["Returns Distribution", "Drawdown", "Rolling Metrics"],
            )

            if analysis == "Returns Distribution":
                fig = plot_returns_distribution(result["returns"])
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == "Drawdown":
                fig = plot_drawdown(result["portfolio_values"])
                st.plotly_chart(fig, use_container_width=True)

            elif analysis == "Rolling Metrics":
                rolling_vol = result["returns"].rolling(21).std() * np.sqrt(252)
                rolling_sharpe = (
                    result["returns"].rolling(63).mean() * 252 /
                    (result["returns"].rolling(63).std() * np.sqrt(252))
                )

                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_price_chart(rolling_vol.to_frame("Vol"), "Rolling Volatility (21d)")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = plot_price_chart(rolling_sharpe.to_frame("Sharpe"), "Rolling Sharpe (63d)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a backtest first")


def show_portfolio():
    """Show portfolio management page."""
    st.markdown('<p class="main-header">Portfolio Management</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Current Portfolio", "⚖️ Optimization"])

    with tab1:
        st.subheader("Holdings")

        # Use backtest result if available
        if st.session_state.backtest_result:
            weights = st.session_state.backtest_result["weights"]
            portfolio_value = st.session_state.backtest_result["portfolio_values"].iloc[-1]

            holdings = []
            for symbol, weight in weights.items():
                holdings.append({
                    "Symbol": symbol,
                    "Weight": f"{weight:.1%}",
                    "Value": f"${weight * portfolio_value:,.0f}",
                })

            st.dataframe(pd.DataFrame(holdings), use_container_width=True, hide_index=True)

            fig = plot_allocation_pie(weights)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run backtest or add positions manually")

            # Manual entry
            with st.form("add_position"):
                col1, col2 = st.columns(2)
                with col1:
                    symbol = st.text_input("Symbol")
                with col2:
                    weight = st.number_input("Weight (%)", 0, 100, 25)

                if st.form_submit_button("Add"):
                    if symbol:
                        st.session_state.portfolio[symbol] = weight / 100
                        st.success(f"Added {symbol}")

            if st.session_state.portfolio:
                st.dataframe(pd.DataFrame([
                    {"Symbol": k, "Weight": f"{v:.1%}"}
                    for k, v in st.session_state.portfolio.items()
                ]))

    with tab2:
        st.subheader("Portfolio Optimization")

        method = st.selectbox(
            "Method",
            ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"],
        )

        symbols = st.text_input("Assets", "AAPL,MSFT,GOOGL,AMZN")

        if st.button("Optimize"):
            symbol_list = [s.strip() for s in symbols.split(",")]
            n = len(symbol_list)

            if method == "Equal Weight":
                opt_weights = {s: 1/n for s in symbol_list}
            elif method == "Minimum Variance":
                # Demo: slightly different weights
                base = 1/n
                opt_weights = {s: base + np.random.uniform(-0.05, 0.05) for s in symbol_list}
                total = sum(opt_weights.values())
                opt_weights = {k: v/total for k, v in opt_weights.items()}
            else:
                opt_weights = {s: 1/n for s in symbol_list}

            st.success("Optimization complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Optimal Weights:**")
                for s, w in opt_weights.items():
                    st.text(f"{s}: {w:.1%}")
            with col2:
                fig = plot_allocation_pie(opt_weights, "Optimized Allocation")
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
