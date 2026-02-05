"""Shared state management for Dash application."""

# Store IDs
STORE_IDS = {
    "demo_mode": "store-demo-mode",
    "current_page": "store-current-page",
    "backtest_result": "store-backtest-result",
    "portfolio": "store-portfolio",
    "prices": "store-prices",
}

# Default state values
DEFAULT_STATE = {
    "demo_mode": True,
    "current_page": "dashboard",
    "backtest_result": None,
    "portfolio": {},
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
}

# Page configuration
PAGE_CONFIG = {
    "dashboard": {
        "title": "Dashboard",
        "subtitle": "Real-time portfolio overview and market insights",
        "icon": "◉",
    },
    "live-analyzer": {
        "title": "Live Analyzer",
        "subtitle": "Enter tickers for real-time data and analysis",
        "icon": "⚡",
    },
    "data-explorer": {
        "title": "Data Explorer",
        "subtitle": "Explore market data across stocks, macro indicators, and crypto",
        "icon": "◈",
    },
    "factor-analysis": {
        "title": "Factor Analysis",
        "subtitle": "Fama-French multi-factor model analysis and portfolio exposure",
        "icon": "◇",
    },
    "regime-monitor": {
        "title": "Regime Monitor",
        "subtitle": "Economic cycle classification and market regime analysis",
        "icon": "◆",
    },
    "backtest": {
        "title": "Strategy Backtesting",
        "subtitle": "Test and evaluate trading strategies with historical data",
        "icon": "▣",
    },
    "portfolio": {
        "title": "Portfolio Management",
        "subtitle": "Monitor holdings and optimize asset allocation",
        "icon": "◧",
    },
}
