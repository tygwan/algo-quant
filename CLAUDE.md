# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**algo-quant** is a quantitative investment automation agent for stocks and cryptocurrencies implementing Fama-French multi-factor models, macroeconomic regime classification, and automated portfolio management in Python 3.11+.

**Current Status**: Phase 1-5 complete (100%), Phase 6 in progress (47%). End-to-end pipeline working with free data sources. Dash dashboard with live ticker analysis and paper execution foundation.

## Quick Start

```bash
# Dash 대시보드 실행 (권장)
python scripts/run_dashboard.py
# or: /dashboard

# 전체 파이프라인 실행 (데이터 수집 → 팩터 분석 → 레짐 분류 → 백테스트)
python scripts/run_pipeline.py --watchlist tech --top 10

# 커스텀 티커로 파이프라인 실행
python scripts/run_pipeline.py --symbols AAPL,MSFT,GOOGL,NVDA

# 데이터 수집 (무료 소스)
python scripts/collect_data.py index --indices sp500,dow,nasdaq,vix
python scripts/collect_data.py sectors
python scripts/collect_data.py factors

# 테스트 실행
uv run pytest tests/ -v

# 오프라인 페이퍼 트레이딩 데모 (API 키 불필요)
uv run python scripts/demo_paper_trading.py --symbol AAPL --steps 120 --ma-window 20
```

## Technology Stack

- **Python 3.11+** with pandas, numpy, scipy, scikit-learn, hmmlearn
- **Package Manager**: uv (권장) 또는 pip
- **UI**: Dash + Plotly (interactive), Streamlit (legacy)
- **Free Data Sources**: Yahoo Finance (stocks), Kenneth French Library (FF factors), CBOE (VIX)
- **Premium Data Sources**: FMP API (stocks), FRED API (macro), Binance/Upbit (crypto)

## Architecture

```
src/
├── data/          # API clients (FMP, FRED, YFinance, VIX, Binance, Upbit)
│   ├── yfinance_client.py   # Free stock data (no API key)
│   ├── vix_client.py        # CBOE VIX data
│   ├── collector.py         # Unified data collection
│   └── cache.py             # Data caching
├── factors/       # CAPM, FF3, FF5, factor neutralization
│   └── ff_data.py           # Kenneth French Data Library
├── regime/        # Rule-based, HMM classifiers, signals
│   └── hmm_classifier.py    # Hidden Markov Model
├── strategy/      # Optimizer, factor strategy, regime strategy, risk manager
├── backtest/      # Engine, metrics, walk-forward analysis
└── ui/            # Dash dashboard (live analyzer, factor analysis, backtest)
    └── layouts/live_analyzer.py  # Real-time ticker analysis

scripts/
├── run_pipeline.py    # End-to-end quantitative pipeline
├── run_dashboard.py   # Dash dashboard launcher
└── collect_data.py    # Data collection CLI

config/
└── watchlist.yaml     # Predefined ticker watchlists
```

## CLI Commands

```bash
# Data Collection (무료 소스, API 키 불필요)
python scripts/collect_data.py index --indices sp500,dow,nasdaq,vix
python scripts/collect_data.py sectors
python scripts/collect_data.py factors
python scripts/collect_data.py prices AAPL,MSFT --source yfinance

# Pipeline (데이터 수집 → 팩터 분석 → 레짐 분류 → 백테스트)
python scripts/run_pipeline.py --watchlist tech --top 10
python scripts/run_pipeline.py --symbols NVDA,AMD,INTC,TSM
python scripts/run_pipeline.py --start 2020-01-01 --end 2024-12-31

# Dashboard
python scripts/run_dashboard.py --port 8050
```

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/dashboard` | Dash 대시보드 실행 |
| `/test` | pytest 테스트 실행 |
| `/backtest` | 백테스트 실행 |
| `/phase status` | 현재 Phase 진행률 |

## Watchlists (config/watchlist.yaml)

| Watchlist | Description |
|-----------|-------------|
| `default` | 주요 대형주 (AAPL, MSFT, GOOGL...) |
| `tech` | 기술주 (NVDA, AMD, CRM...) |
| `semiconductor` | 반도체 (NVDA, AMD, INTC, TSM...) |
| `etf` | 주요 ETF (SPY, QQQ, IWM...) |
| `value` | 가치주 (BRK-B, JPM, JNJ...) |
| `growth` | 성장주 (TSLA, SQ, SHOP...) |

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Data Infrastructure | ✅ 100% | API clients, preprocessor, cache |
| 2. Factor Modeling | ✅ 100% | CAPM, FF3, FF5, neutralization |
| 3. Regime Classification | ✅ 100% | Rule-based, HMM, signals |
| 4. Strategy Development | ✅ 100% | Optimizer, factor/regime strategy |
| 5. Backtesting | ✅ 100% | Engine, metrics, walk-forward |
| 6. Production | 🔄 47% | Realtime stream, execution engine, paper broker done; live validation/alerts pending |

## Key Domain Concepts

### Factor Models
- **CAPM**: E(Ri) = Rf + βi(E(Rm) - Rf)
- **FF3**: Market + SMB (size) + HML (value)
- **FF5**: FF3 + RMW (profitability) + CMA (investment)

### Regime Classification
- **Expansion**: GDP↑, Unemployment↓, Yield Curve Normal
- **Peak**: Growth slowing, Yield Curve Flat
- **Contraction**: GDP↓, Unemployment↑, Yield Curve Inverted
- **Trough**: Recovery beginning

### Portfolio Optimization
- Mean-Variance, Minimum Variance, Maximum Sharpe
- Risk Parity, Equal Weight, Inverse Volatility

## Data Sources

**무료 (API 키 불필요)**:
- Yahoo Finance (`yfinance`): 주가, 인덱스, 섹터 ETF
- Kenneth French Library: Fama-French 팩터 데이터
- CBOE: VIX 데이터

**유료 (API 키 필요)** - `config/api_keys.yaml`:
```yaml
fmp:
  api_key: "YOUR_FMP_API_KEY"     # 재무제표, 상세 주가
fred:
  api_key: "YOUR_FRED_API_KEY"   # 거시경제 지표
binance:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_SECRET"
```

## Live Analyzer (Dashboard)

대시보드의 Live Analyzer 페이지에서:
1. 티커 입력 (쉼표로 구분: `AAPL, MSFT, NVDA`)
2. 기간 선택 (1mo ~ 5y)
3. 분석 유형 선택:
   - **Price Chart**: 주가 추이
   - **Returns Distribution**: 수익률 분포
   - **Correlation Matrix**: 상관관계
   - **Factor Analysis**: FF5 팩터 노출도
   - **Risk Metrics**: VaR, CVaR, 변동성

## References

- [퀀트 투자 강의 (FastCampus)](https://fastcampus.co.kr/fin_online_quant01)
- [FRED API Docs](https://fred.stlouisfed.org/docs/api/fred/)
- [FMP API Docs](https://site.financialmodelingprep.com/developer/docs)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
