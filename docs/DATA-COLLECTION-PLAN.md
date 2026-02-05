# Market Data Collection Plan

## Executive Summary

This document outlines the data collection strategy for the algo-quant project, covering US stocks, Korean equities, cryptocurrencies, and macroeconomic indicators. The plan is designed to support multi-factor analysis, regime classification, and automated trading.

---

## 1. Data Sources Overview

### 1.1 Primary Data Sources

| Source | Asset Class | Data Type | Rate Limit | Cost |
|--------|-------------|-----------|------------|------|
| **FMP** | US Stocks | OHLCV, Financials, Metrics | 300/min | Freemium |
| **FRED** | Macro | Economic Indicators | 120/min | Free |
| **KIS** | Korean Stocks | OHLCV, Trading | 60/min | Free (account required) |
| **Kiwoom** | Korean Stocks | OHLCV, Trading | 60/min | Free (account required) |
| **Binance** | Crypto | OHLCV, Trading | 1200/min | Free |
| **Upbit** | Korean Crypto | OHLCV, Trading | 600/min | Free |

### 1.2 Implemented Clients

```
src/data/
├── base_client.py      # Common infrastructure (rate limiting, retries)
├── cache.py            # File-based caching with TTL
├── preprocessor.py     # Data quality processing
├── fmp_client.py       # US stocks
├── fred_client.py      # Macro indicators
├── kis_client.py       # Korean stocks (Korea Investment)
├── kiwoom_client.py    # Korean stocks (Kiwoom)
├── binance_client.py   # Crypto (Binance)
└── upbit_client.py     # Korean crypto (Upbit)
```

---

## 2. Data Collection Strategy

### 2.1 US Stock Data (FMP)

**Target Universe:**
- S&P 500 constituents
- Russell 2000 (small cap exposure for size factor)
- Custom watchlist for specific sectors

**Data Types:**
| Data | Frequency | TTL | Use Case |
|------|-----------|-----|----------|
| Daily OHLCV | Daily EOD | 24h | Factor calculation, backtesting |
| Financial Statements | Quarterly | 7 days | Fundamental analysis |
| Key Metrics | Quarterly | 7 days | Value/Quality factors |
| Company Profile | Monthly | 7 days | Sector classification |

**Collection Schedule:**
```python
# Daily: After market close (4:30 PM ET / 6:30 AM KST)
- Update OHLCV for universe
- Check for new earnings releases

# Weekly: Sunday
- Full universe refresh
- Update company profiles for changes

# Quarterly: After earnings season
- Financial statements bulk update
- Key metrics recalculation
```

### 2.2 Macroeconomic Indicators (FRED)

**Core Indicators:**
| Category | Indicator | FRED Code | Frequency | Lead Time |
|----------|-----------|-----------|-----------|-----------|
| **Growth** | Real GDP | GDPC1 | Quarterly | Lag 2 months |
| **Labor** | Unemployment | UNRATE | Monthly | Lag 1 month |
| **Labor** | Initial Claims | ICSA | Weekly | Lead indicator |
| **Inflation** | CPI | CPIAUCSL | Monthly | Lag 2 weeks |
| **Inflation** | PCE | PCEPI | Monthly | Lag 3 weeks |
| **Monetary** | Fed Funds | FEDFUNDS | Monthly | Same day |
| **Yields** | 10Y Treasury | DGS10 | Daily | Same day |
| **Yields** | 10Y-2Y Spread | T10Y2Y | Daily | Lead indicator |
| **Sentiment** | Consumer Sentiment | UMCSENT | Monthly | Lead indicator |
| **Housing** | Housing Starts | HOUST | Monthly | Lead indicator |

**Regime Indicators Matrix:**
```
EXPANSION:   GDP↑, UNRATE↓, ISM>50, Spread>0
PEAK:        GDP↑ slowing, UNRATE↓ stable, ISM declining
CONTRACTION: GDP↓, UNRATE↑, ISM<50, Spread inverted
TROUGH:      GDP↓ stabilizing, UNRATE↑ peaking, ISM bottoming
```

**Collection Schedule:**
```python
# Daily: 9:00 AM KST
- Treasury yields (DGS10, T10Y2Y)
- Fed Funds effective rate

# Weekly: Monday
- Initial claims (ICSA)

# Monthly: 1st week of month
- All monthly indicators after release
- Update composite indices
```

### 2.3 Korean Stock Data (KIS/Kiwoom)

**Target Universe:**
- KOSPI 200 constituents
- KOSDAQ 150 (growth/tech exposure)
- Custom sector ETFs

**Data Types:**
| Data | Frequency | Source | Use Case |
|------|-----------|--------|----------|
| Daily OHLCV | Daily EOD | KIS/Kiwoom | Backtesting |
| Current Price | Real-time | KIS/Kiwoom | Live trading |
| Account Balance | On-demand | KIS/Kiwoom | Position management |

**Collection Schedule:**
```python
# Daily: After market close (3:30 PM KST)
- Update OHLCV for KOSPI/KOSDAQ universe
- Sync portfolio positions

# Weekly: Saturday
- Universe refresh (constituent changes)
- Corporate action adjustments
```

### 2.4 Cryptocurrency Data (Binance/Upbit)

**Target Pairs:**
- Major: BTC, ETH, BNB vs USDT
- Alts: SOL, ADA, XRP, DOT, AVAX
- Korean market: KRW pairs on Upbit

**Data Types:**
| Data | Frequency | TTL | Use Case |
|------|-----------|-----|----------|
| 1-minute candles | Real-time | 1 min | Intraday signals |
| 1-hour candles | Hourly | 5 min | Short-term analysis |
| Daily candles | Daily | 60 min | Trend analysis |
| Orderbook | Real-time | - | Execution optimization |

**Collection Schedule:**
```python
# Continuous: Every minute
- 1-minute candles for active pairs
- Price ticker updates

# Hourly: On the hour
- 1-hour candles
- 24-hour statistics

# Daily: 00:00 UTC
- Daily candles consolidation
- Volume analysis
```

---

## 3. Data Pipeline Architecture

### 3.1 Collection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       SCHEDULER (APScheduler)                   │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│   │  Daily  │   │ Weekly  │   │ Monthly │   │Real-time│       │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘       │
└────────┼─────────────┼───────────┼─────────────┼───────────────┘
         │             │           │             │
         ▼             ▼           ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       COLLECTORS                                 │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│   │   FMP   │   │  FRED   │   │   KIS   │   │ Binance │       │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘       │
└────────┼─────────────┼───────────┼─────────────┼───────────────┘
         │             │           │             │
         ▼             ▼           ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CACHE LAYER                              │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  Parquet files (.cache/)  │  TTL-based expiry           │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PREPROCESSOR                               │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│   │ Missing │   │ Outlier │   │ Returns │   │ Normalize│       │
│   │ Values  │   │Detection│   │  Calc   │   │         │       │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘       │
└────────┼─────────────┼───────────┼─────────────┼───────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                                │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  data/raw/     - Original API responses                  │ │
│   │  data/processed/ - Cleaned, normalized data              │ │
│   │  data/features/  - Calculated features (factors)         │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
data/
├── raw/                    # Original API data
│   ├── prices/
│   │   ├── us/            # FMP price data
│   │   ├── kr/            # KIS/Kiwoom price data
│   │   └── crypto/        # Binance/Upbit price data
│   ├── fundamentals/
│   │   ├── financials/    # Income, Balance, Cash Flow
│   │   └── metrics/       # Key financial ratios
│   └── macro/             # FRED indicators
│
├── processed/              # Cleaned data
│   ├── returns/           # Simple and log returns
│   ├── factors/           # Factor exposures
│   └── regimes/           # Regime classifications
│
└── features/               # Calculated features
    ├── technical/         # Moving averages, RSI, etc.
    ├── fundamental/       # P/E, P/B, ROE, etc.
    └── composite/         # Combined factor scores
```

---

## 4. Data Quality Controls

### 4.1 Validation Rules

```python
VALIDATION_RULES = {
    "price": {
        "open >= 0": True,
        "high >= low": True,
        "close >= 0": True,
        "volume >= 0": True,
        "max_daily_change": 0.50,  # 50% max daily change
    },
    "financial": {
        "assets >= liabilities + equity": True,
        "revenue >= 0": True,
        "no_future_dates": True,
    },
    "macro": {
        "unemployment_range": (0, 30),  # 0-30%
        "cpi_range": (-5, 20),          # -5% to 20% YoY
    },
}
```

### 4.2 Data Quality Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Completeness | % of non-null values | > 99% |
| Timeliness | Age of latest data point | < 24h |
| Consistency | Cross-source validation | 100% match |
| Accuracy | Price deviation from benchmark | < 0.01% |

### 4.3 Anomaly Detection

```python
# Automatic anomaly detection triggers:
- Daily return > 30% (investigate corporate action)
- Volume spike > 10x average (investigate news)
- Missing data points > 3 consecutive days
- Cross-source price divergence > 1%
```

---

## 5. Implementation Roadmap

### Phase 1: Core Data Collection (Current Sprint)

**Week 1-2:**
- [ ] Create data collection scheduler (APScheduler integration)
- [ ] Implement universe management (S&P 500, KOSPI 200)
- [ ] Set up automated daily OHLCV collection
- [ ] Add data validation checks

**Deliverables:**
```python
src/data/
├── scheduler.py        # Job scheduling
├── universe.py         # Universe management
├── collector.py        # Unified collection interface
└── validator.py        # Data quality checks
```

### Phase 2: Feature Engineering

**Week 3-4:**
- [ ] Calculate rolling returns (1D, 5D, 20D, 60D, 252D)
- [ ] Build factor exposures (Beta, Size, Value, Momentum)
- [ ] Integrate regime classification signals
- [ ] Create composite factor scores

**Deliverables:**
```python
src/features/
├── returns.py          # Return calculations
├── factors.py          # Factor exposures
├── technical.py        # Technical indicators
└── composite.py        # Combined scores
```

### Phase 3: Real-time Pipeline

**Week 5-6:**
- [ ] WebSocket integration for crypto (Binance/Upbit)
- [ ] Real-time price streaming
- [ ] Live regime monitoring
- [ ] Trading signal generation

**Deliverables:**
```python
src/realtime/
├── websocket_client.py # WebSocket connections
├── stream_processor.py # Real-time processing
└── signal_generator.py # Trading signals
```

---

## 6. API Key Configuration

### Required API Keys

```yaml
# config/api_keys.yaml (DO NOT COMMIT)

fmp:
  api_key: "YOUR_FMP_API_KEY"
  # Free tier: 250 requests/day, Premium: unlimited

fred:
  api_key: "YOUR_FRED_API_KEY"
  # Free: 120 requests/minute

binance:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_SECRET"
  # No cost, create at binance.com

upbit:
  api_key: "YOUR_UPBIT_ACCESS_KEY"
  api_secret: "YOUR_UPBIT_SECRET_KEY"
  # No cost, create at upbit.com

kis:
  app_key: "YOUR_KIS_APP_KEY"
  app_secret: "YOUR_KIS_SECRET"
  account_number: "YOUR_ACCOUNT_NUMBER"
  account_product_code: "01"
  # Requires Korea Investment account

kiwoom:
  app_key: "YOUR_KIWOOM_APP_KEY"
  app_secret: "YOUR_KIWOOM_SECRET"
  # Requires Kiwoom account
```

### Key Acquisition Guide

1. **FMP**: https://financialmodelingprep.com/register
2. **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html
3. **Binance**: https://www.binance.com/en/my/settings/api-management
4. **Upbit**: https://upbit.com/open_api
5. **KIS**: https://apiportal.koreainvestment.com
6. **Kiwoom**: https://openapi.kiwoom.com

---

## 7. Monitoring and Alerts

### 7.1 Collection Monitoring

```python
MONITORING_CONFIG = {
    "metrics": {
        "collection_success_rate": "percentage of successful API calls",
        "cache_hit_rate": "percentage of cache hits",
        "data_freshness": "age of latest data point",
        "api_latency": "average response time",
    },
    "alerts": {
        "collection_failure": "notify on 3 consecutive failures",
        "stale_data": "notify if data > 2 days old",
        "rate_limit_warning": "notify at 80% rate limit usage",
    },
}
```

### 7.2 Logging Strategy

```python
import logging

# Log levels by operation type:
# - DEBUG: Individual API calls, cache operations
# - INFO: Collection job start/complete, daily summaries
# - WARNING: Rate limit approaching, data quality issues
# - ERROR: API failures, authentication issues
# - CRITICAL: System-wide failures, data corruption
```

---

## 8. Cost Estimation

### Monthly API Costs

| API | Tier | Cost | Usage Limit |
|-----|------|------|-------------|
| FMP | Free | $0 | 250 calls/day |
| FMP | Starter | $19/mo | Unlimited |
| FRED | Free | $0 | Unlimited |
| Binance | Free | $0 | 1200/min |
| Upbit | Free | $0 | 600/min |
| KIS | Free | $0 | Account required |
| Kiwoom | Free | $0 | Account required |

**Recommended Setup:**
- Start with free tiers for development
- Upgrade FMP when hitting daily limits
- Total production cost: $19-49/month

---

## 9. Next Steps

1. **Immediate (This Week):**
   - Set up API keys in config file
   - Test each client with sample data
   - Verify caching is working correctly

2. **Short-term (Next 2 Weeks):**
   - Implement scheduler for automated collection
   - Build universe management for S&P 500
   - Create data validation pipeline

3. **Medium-term (Next Month):**
   - Integrate factor calculation with data pipeline
   - Set up regime classification monitoring
   - Build backtesting data preparation workflow

---

*Document created: 2026-01-12*
*Last updated: 2026-01-12*
