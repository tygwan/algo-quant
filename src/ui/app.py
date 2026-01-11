"""Algo-Quant Dashboard - Modern Dark Theme UI."""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - must be first
st.set_page_config(
    page_title="Algo-Quant",
    page_icon="◉",
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

# Claude-inspired Dark Theme CSS
CLAUDE_THEME = """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e3f 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }

    /* Navigation items */
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent;
        gap: 0.25rem;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        transition: all 0.2s ease;
        color: #a0a0a0;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.1);
        color: #ffffff;
    }

    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-color: transparent;
        color: #ffffff;
        font-weight: 500;
    }

    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 100%;
    }

    /* Headers */
    .page-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.1);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 0.875rem;
        color: #888;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }

    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    .metric-delta.positive {
        color: #10b981;
    }

    .metric-delta.negative {
        color: #ef4444;
    }

    /* Chart containers */
    .chart-container {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .chart-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #888;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background: rgba(255,255,255,0.05);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.1);
        color: #8b5cf6;
        border-bottom: 2px solid #8b5cf6;
    }

    /* Input fields */
    .stTextInput > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: #ffffff;
    }

    .stTextInput > div > div:focus-within {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        color: #ffffff;
    }

    .stSlider > div > div > div {
        background: #8b5cf6;
    }

    /* Data tables */
    .stDataFrame {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stDataFrame"] > div {
        background: transparent;
    }

    /* Info/Success/Warning boxes */
    .stAlert {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border-left: 4px solid;
    }

    .stSuccess {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
    }

    .stInfo {
        border-left-color: #8b5cf6;
        background: rgba(139, 92, 246, 0.1);
    }

    .stWarning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.1);
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 2rem 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.02);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.2);
    }

    /* Section divider with gradient */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        margin: 2rem 0;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .status-badge.expansion {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }

    .status-badge.contraction {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
    }

    .status-badge.peak {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
    }

    .status-badge.trough {
        background: rgba(99, 102, 241, 0.15);
        color: #6366f1;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #888;
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
</style>
"""

# Inject custom CSS
st.markdown(CLAUDE_THEME, unsafe_allow_html=True)


# Session state initialization
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True


def render_metric_card(label: str, value: str, delta: str = None, delta_type: str = "positive"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_type == "positive" else "negative"
        delta_icon = "↑" if delta_type == "positive" else "↓"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_icon} {delta}</div>'

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def render_page_header(title: str, subtitle: str = ""):
    """Render page header with title and subtitle."""
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    return f"""
    <h1 class="page-header">{title}</h1>
    {subtitle_html}
    """


def main():
    """Main application entry point."""
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; color: white;">◉</div>
                <div>
                    <div style="font-weight: 700; font-size: 1.125rem; color: #fff;">Algo-Quant</div>
                    <div style="font-size: 0.75rem; color: #888;">Quantitative Investing</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom: 0.5rem; font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em;">Navigation</div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Dashboard", "Data Explorer", "Factor Analysis",
             "Regime Monitor", "Backtest", "Portfolio"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom: 0.5rem; font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em;">Settings</div>', unsafe_allow_html=True)

        demo_mode = st.toggle("Demo Mode", value=st.session_state.demo_mode, help="Use sample data for demonstration")
        st.session_state.demo_mode = demo_mode

        if demo_mode:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 12px; padding: 0.75rem; margin-top: 0.5rem;">
                <div style="color: #10b981; font-size: 0.875rem; font-weight: 500;">✓ Demo Mode Active</div>
                <div style="color: #888; font-size: 0.75rem; margin-top: 0.25rem;">Using simulated market data</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); border-radius: 12px; padding: 0.75rem; margin-top: 0.5rem;">
                <div style="color: #f59e0b; font-size: 0.875rem; font-weight: 500;">⚠ API Keys Required</div>
                <div style="color: #888; font-size: 0.75rem; margin-top: 0.25rem;">Configure in config/api_keys.yaml</div>
            </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="position: absolute; bottom: 2rem; left: 1rem; right: 1rem;">
            <div style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;">
                <div style="color: #666; font-size: 0.75rem;">Version 0.1.0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Factor Analysis":
        show_factor_analysis()
    elif page == "Regime Monitor":
        show_regime_monitor()
    elif page == "Backtest":
        show_backtest()
    elif page == "Portfolio":
        show_portfolio()


def show_dashboard():
    """Show main dashboard."""
    st.markdown(render_page_header("Dashboard", "Real-time portfolio overview and market insights"), unsafe_allow_html=True)

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
        st.markdown(render_metric_card("Portfolio Value", f"${portfolio_values.iloc[-1]:,.0f}", f"{total_return:+.1f}%", "positive" if total_return > 0 else "negative"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("Annual Volatility", f"{vol:.1f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("Sharpe Ratio", f"{sharpe:.2f}"), unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Regime</div>
            <div style="margin-top: 0.5rem;"><span class="status-badge expansion">Expansion</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Portfolio Performance</div>', unsafe_allow_html=True)
        fig = plot_price_chart(portfolio_values.to_frame("Portfolio"), "")
        st.plotly_chart(fig, width="stretch", key="dashboard_portfolio")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Asset Allocation</div>', unsafe_allow_html=True)
        weights = {"AAPL": 0.30, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.20}
        fig = plot_allocation_pie(weights, "")
        st.plotly_chart(fig, width="stretch", key="dashboard_allocation")
        st.markdown('</div>', unsafe_allow_html=True)

    # Full width price chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Asset Prices (Indexed)</div>', unsafe_allow_html=True)
    fig = plot_price_chart(prices, "")
    st.plotly_chart(fig, width="stretch", key="dashboard_prices")
    st.markdown('</div>', unsafe_allow_html=True)


def show_data_explorer():
    """Show data exploration page."""
    st.markdown(render_page_header("Data Explorer", "Explore market data across stocks, macro indicators, and crypto"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Stocks", "Macro Indicators", "Cryptocurrency"])

    with tab1:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown('<div style="margin-bottom: 1rem; font-weight: 600; color: #fff;">Configuration</div>', unsafe_allow_html=True)
            symbols = st.text_input("Symbols", "AAPL,MSFT,GOOGL,AMZN", help="Comma-separated ticker symbols")
            periods = st.slider("History (days)", 60, 500, 252)

            if st.button("Generate Data", type="primary", key="gen_stock"):
                symbol_list = [s.strip() for s in symbols.split(",")]
                prices = generate_sample_prices(symbol_list, periods)
                st.session_state.stock_data = prices

        with col2:
            if "stock_data" in st.session_state:
                prices = st.session_state.stock_data
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = plot_price_chart(prices, "")
                st.plotly_chart(fig, width="stretch", key="explorer_stocks")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                st.dataframe(prices.tail(10), width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div>Click "Generate Data" to view sample stock prices</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="chart-title">Macroeconomic Indicators</div>', unsafe_allow_html=True)

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
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = plot_price_chart(macro_data[selected], "")
            st.plotly_chart(fig, width="stretch", key="explorer_macro")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="chart-title">Cryptocurrency Prices</div>', unsafe_allow_html=True)

        crypto_symbols = ["BTC", "ETH", "BNB"]
        crypto_prices = generate_sample_prices(crypto_symbols, 365, "2024-01-01")
        crypto_prices = crypto_prices * 300

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = plot_price_chart(crypto_prices, "")
        st.plotly_chart(fig, width="stretch", key="explorer_crypto")
        st.markdown('</div>', unsafe_allow_html=True)


def show_factor_analysis():
    """Show factor analysis page."""
    st.markdown(render_page_header("Factor Analysis", "Fama-French multi-factor model analysis and portfolio exposure"), unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Factor Returns", "Factor Exposure"])

    with tab1:
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        factor_data = pd.DataFrame({
            "Mkt-RF": np.random.normal(0.0004, 0.01, 252),
            "SMB": np.random.normal(0.0001, 0.005, 252),
            "HML": np.random.normal(0.0001, 0.006, 252),
            "RMW": np.random.normal(0.0001, 0.004, 252),
            "CMA": np.random.normal(0.0001, 0.004, 252),
        }, index=dates)

        col1, col2 = st.columns([1, 4])

        with col1:
            model = st.selectbox("Model", ["FF3 (3-Factor)", "FF5 (5-Factor)"])

        if model == "FF3 (3-Factor)":
            factors = ["Mkt-RF", "SMB", "HML"]
        else:
            factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

        cum_returns = (1 + factor_data[factors]).cumprod()

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Cumulative Factor Returns</div>', unsafe_allow_html=True)
        fig = plot_price_chart(cum_returns, "")
        st.plotly_chart(fig, width="stretch", key="factor_returns")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Factor Statistics</div>', unsafe_allow_html=True)
        stats = pd.DataFrame({
            "Mean (Ann.)": factor_data[factors].mean() * 252,
            "Vol (Ann.)": factor_data[factors].std() * np.sqrt(252),
            "Sharpe": (factor_data[factors].mean() * 252) / (factor_data[factors].std() * np.sqrt(252)),
        }).T
        st.dataframe(stats.style.format("{:.2%}"), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        exposures = {
            "Market": 1.05,
            "Size (SMB)": -0.15,
            "Value (HML)": 0.25,
            "Profitability": 0.10,
            "Investment": -0.05,
        }

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Portfolio Factor Exposure</div>', unsafe_allow_html=True)
        fig = plot_factor_exposure(exposures, "")
        st.plotly_chart(fig, width="stretch", key="factor_exposure")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(255,255,255,0.02); border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
            <div style="font-weight: 600; color: #fff; margin-bottom: 0.75rem;">Interpretation Guide</div>
            <div style="color: #888; font-size: 0.875rem; line-height: 1.6;">
                • <strong style="color: #a0a0a0;">Market β > 1:</strong> Portfolio is more volatile than market<br>
                • <strong style="color: #a0a0a0;">Negative SMB:</strong> Tilted toward large-cap stocks<br>
                • <strong style="color: #a0a0a0;">Positive HML:</strong> Tilted toward value stocks
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_regime_monitor():
    """Show regime monitoring page."""
    st.markdown(render_page_header("Regime Monitor", "Economic cycle classification and market regime analysis"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Regime</div>
            <div style="margin-top: 0.5rem;"><span class="status-badge expansion">Expansion</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("Confidence", "78%"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("Duration", "8 months"), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Economic Indicators", "Regime History"])

    with tab1:
        st.markdown('<div class="chart-title">Key Economic Indicators</div>', unsafe_allow_html=True)

        indicators = {
            "GDP Growth": {"value": "+2.3%", "signal": "positive", "label": "Positive"},
            "Unemployment": {"value": "3.8%", "signal": "positive", "label": "Low"},
            "10Y-2Y Spread": {"value": "+0.45%", "signal": "positive", "label": "Normal"},
            "Fed Funds Rate": {"value": "5.25%", "signal": "warning", "label": "High"},
            "PMI": {"value": "52.1", "signal": "positive", "label": "Expanding"},
        }

        st.markdown('<div style="display: grid; gap: 0.75rem;">', unsafe_allow_html=True)
        for name, data in indicators.items():
            signal_color = "#10b981" if data["signal"] == "positive" else "#f59e0b"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border-radius: 12px; padding: 1rem; display: flex; justify-content: space-between; align-items: center;">
                <div style="color: #fff; font-weight: 500;">{name}</div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="color: #a0a0a0; font-family: monospace;">{data["value"]}</div>
                    <div style="background: rgba({signal_color[1:]}, 0.15); color: {signal_color}; padding: 0.25rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 500;">{data["label"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        regime_values = np.random.choice([0, 1, 2, 3], 48, p=[0.5, 0.2, 0.15, 0.15])
        regime_names = ["Expansion", "Peak", "Contraction", "Trough"]

        regime_df = pd.DataFrame({
            "Regime": [regime_names[r] for r in regime_values],
        }, index=dates)

        import plotly.express as px
        fig = px.bar(
            regime_df.reset_index(),
            x="index",
            y=[1] * len(regime_df),
            color="Regime",
            color_discrete_map={
                "Expansion": "#10b981",
                "Peak": "#f59e0b",
                "Contraction": "#ef4444",
                "Trough": "#6366f1",
            },
        )
        fig.update_layout(
            showlegend=True,
            yaxis_visible=False,
            xaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0a0a0'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Regime Timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", key="regime_timeline")
        st.markdown('</div>', unsafe_allow_html=True)


def show_backtest():
    """Show backtesting page."""
    st.markdown(render_page_header("Strategy Backtesting", "Test and evaluate trading strategies with historical data"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Configure", "Results", "Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-title">Strategy Settings</div>', unsafe_allow_html=True)
            strategy = st.selectbox(
                "Strategy Type",
                ["Equal Weight", "Risk Parity", "Momentum", "Mean-Variance"],
            )
            symbols = st.text_input("Universe", "AAPL,MSFT,GOOGL,AMZN,META")
            rebalance = st.selectbox("Rebalance Frequency", ["Monthly", "Quarterly", "Weekly"])

        with col2:
            st.markdown('<div class="chart-title">Backtest Settings</div>', unsafe_allow_html=True)
            capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
            commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01)
            periods = st.slider("Backtest Period (days)", 60, 756, 252)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                symbol_list = [s.strip() for s in symbols.split(",")]
                prices = generate_sample_prices(symbol_list, periods)
                returns = prices.pct_change().dropna()

                n_assets = len(symbol_list)
                portfolio_returns = returns.mean(axis=1)
                portfolio_values = capital * (1 + portfolio_returns).cumprod()

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

            st.success("Backtest complete! View results in the Results tab.")

    with tab2:
        if st.session_state.backtest_result:
            result = st.session_state.backtest_result
            metrics = result["metrics"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                delta_type = "positive" if metrics["Total Return"] > 0 else "negative"
                st.markdown(render_metric_card("Total Return", f"{metrics['Total Return']:.1%}", delta_type=delta_type), unsafe_allow_html=True)
            with col2:
                st.markdown(render_metric_card("CAGR", f"{metrics['CAGR']:.1%}"), unsafe_allow_html=True)
            with col3:
                st.markdown(render_metric_card("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}"), unsafe_allow_html=True)
            with col4:
                st.markdown(render_metric_card("Max Drawdown", f"{metrics['Max Drawdown']:.1%}"), unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Portfolio Value</div>', unsafe_allow_html=True)
                fig = plot_price_chart(result["portfolio_values"].to_frame("Portfolio"), "")
                st.plotly_chart(fig, width="stretch", key="backtest_portfolio")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Final Allocation</div>', unsafe_allow_html=True)
                fig = plot_allocation_pie(result["weights"], "")
                st.plotly_chart(fig, width="stretch", key="backtest_allocation")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="chart-title">Performance Metrics</div>', unsafe_allow_html=True)
            metrics_df = pd.DataFrame([
                {"Metric": k, "Value": f"{v:.2%}" if isinstance(v, float) else str(v)}
                for k, v in metrics.items()
            ])
            st.dataframe(metrics_df, width="stretch", hide_index=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📈</div>
                <div>Configure and run a backtest to see results</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        if st.session_state.backtest_result:
            result = st.session_state.backtest_result

            analysis = st.selectbox(
                "Analysis Type",
                ["Returns Distribution", "Drawdown", "Rolling Metrics"],
            )

            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            if analysis == "Returns Distribution":
                st.markdown('<div class="chart-title">Returns Distribution</div>', unsafe_allow_html=True)
                fig = plot_returns_distribution(result["returns"], "")
                st.plotly_chart(fig, width="stretch", key="backtest_dist")

            elif analysis == "Drawdown":
                st.markdown('<div class="chart-title">Drawdown Analysis</div>', unsafe_allow_html=True)
                fig = plot_drawdown(result["portfolio_values"], "")
                st.plotly_chart(fig, width="stretch", key="backtest_dd")

            elif analysis == "Rolling Metrics":
                st.markdown('<div class="chart-title">Rolling Volatility & Sharpe</div>', unsafe_allow_html=True)
                rolling_vol = result["returns"].rolling(21).std() * np.sqrt(252)
                rolling_sharpe = (
                    result["returns"].rolling(63).mean() * 252 /
                    (result["returns"].rolling(63).std() * np.sqrt(252))
                )

                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_price_chart(rolling_vol.to_frame("Volatility"), "")
                    st.plotly_chart(fig, width="stretch", key="backtest_vol")
                with col2:
                    fig = plot_price_chart(rolling_sharpe.to_frame("Sharpe"), "")
                    st.plotly_chart(fig, width="stretch", key="backtest_sharpe")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div>Run a backtest first to view analysis</div>
            </div>
            """, unsafe_allow_html=True)


def show_portfolio():
    """Show portfolio management page."""
    st.markdown(render_page_header("Portfolio Management", "Monitor holdings and optimize asset allocation"), unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Current Portfolio", "Optimization"])

    with tab1:
        if st.session_state.backtest_result:
            weights = st.session_state.backtest_result["weights"]
            portfolio_value = st.session_state.backtest_result["portfolio_values"].iloc[-1]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown('<div class="chart-title">Holdings</div>', unsafe_allow_html=True)
                holdings = []
                for symbol, weight in weights.items():
                    holdings.append({
                        "Symbol": symbol,
                        "Weight": f"{weight:.1%}",
                        "Value": f"${weight * portfolio_value:,.0f}",
                    })
                st.dataframe(pd.DataFrame(holdings), width="stretch", hide_index=True)

            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = plot_allocation_pie(weights, "")
                st.plotly_chart(fig, width="stretch", key="portfolio_pie")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">💼</div>
                <div>Run a backtest or add positions manually</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("add_position"):
                col1, col2 = st.columns(2)
                with col1:
                    symbol = st.text_input("Symbol")
                with col2:
                    weight = st.number_input("Weight (%)", 0, 100, 25)

                if st.form_submit_button("Add Position"):
                    if symbol:
                        st.session_state.portfolio[symbol] = weight / 100
                        st.success(f"Added {symbol}")

            if st.session_state.portfolio:
                st.dataframe(pd.DataFrame([
                    {"Symbol": k, "Weight": f"{v:.1%}"}
                    for k, v in st.session_state.portfolio.items()
                ]), width="stretch", hide_index=True)

    with tab2:
        st.markdown('<div class="chart-title">Portfolio Optimization</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            method = st.selectbox(
                "Optimization Method",
                ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"],
            )
            symbols = st.text_input("Assets", "AAPL,MSFT,GOOGL,AMZN", key="opt_symbols")

            if st.button("Optimize", type="primary"):
                symbol_list = [s.strip() for s in symbols.split(",")]
                n = len(symbol_list)

                if method == "Equal Weight":
                    opt_weights = {s: 1/n for s in symbol_list}
                elif method == "Minimum Variance":
                    base = 1/n
                    opt_weights = {s: base + np.random.uniform(-0.05, 0.05) for s in symbol_list}
                    total = sum(opt_weights.values())
                    opt_weights = {k: v/total for k, v in opt_weights.items()}
                else:
                    opt_weights = {s: 1/n for s in symbol_list}

                st.session_state.opt_result = opt_weights
                st.success("Optimization complete!")

        with col2:
            if "opt_result" in st.session_state:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Optimized Allocation</div>', unsafe_allow_html=True)
                fig = plot_allocation_pie(st.session_state.opt_result, "")
                st.plotly_chart(fig, width="stretch", key="opt_allocation")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                for s, w in st.session_state.opt_result.items():
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: #a0a0a0;">{s}</span>
                        <span style="color: #fff; font-weight: 500;">{w:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
