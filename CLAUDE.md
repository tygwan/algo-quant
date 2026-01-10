# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**algo-quant** is a quantitative investment automation agent for stocks and cryptocurrencies implementing Fama-French multi-factor models, macroeconomic regime classification, and automated portfolio management in Python 3.11+.

**Current Status**: Project initialization phase. Documentation and phase structure complete; application code development pending.

## Technology Stack

- **Python 3.11+** with pandas, numpy, scikit-learn, backtrader/bt, requests/aiohttp
- **Data Sources**: FMP API (stocks), FRED API (macro indicators), Binance/Upbit (crypto)
- **Environment**: Google Colab compatible, local development supported

## Architecture (Planned)

```
src/
├── data/          # API clients (FMP, FRED, crypto) + preprocessor
├── factors/       # Fama-French models (FF3, FF5), factor neutralization
├── regime/        # Business cycle classifier using FRED indicators
├── strategy/      # Factor-based and regime-adaptive allocation
├── backtest/      # Backtesting engine, metrics, reporting
└── execution/     # Live trading (future phase)
```

## Development Workflow

This project uses **cc-initializer** for phase-based development. Progress is tracked in `docs/PROGRESS.md`.

### Workflow Commands

| Command | Purpose |
|---------|---------|
| `/feature start <name>` | Create branch, link to phase task, update tracking |
| `/feature complete` | Run quality checks, commit, create PR |
| `/phase status` | View current phase progress |
| `/validate` | Verify configuration integrity |
| `/quality-gate` | Run pre-commit checks (lint, test, coverage) |

### Development Phases

1. **Data Infrastructure** - API clients with rate limiting, preprocessing, caching
2. **Factor Modeling** - CAPM, FF3, FF5, factor neutralization
3. **Regime Classification** - FRED indicators, business cycle classification
4. **Strategy Development** - Portfolio construction, risk management
5. **Backtesting** - Historical evaluation, walk-forward analysis
6. **Production** - Real-time pipeline, live trading

## Key Domain Concepts

### Factor Models
- **CAPM**: E(Ri) = Rf + βi(E(Rm) - Rf)
- **FF3**: Adds SMB (size) and HML (value)
- **FF5**: Adds RMW (profitability) and CMA (investment)

### Data Preprocessing
Data cleaning is 50%+ of quant work: missing values, outliers, corporate actions, survivorship bias, point-in-time handling.

### Regime Classification
FRED indicators (GDP, unemployment, yield curve) classify market regimes (expansion/contraction/recession/recovery) to drive allocation decisions.

## API Configuration

Store in `config/api_keys.yaml` (never commit):
```yaml
fmp:
  api_key: "YOUR_FMP_API_KEY"
fred:
  api_key: "YOUR_FRED_API_KEY"
binance:
  api_key: "YOUR_BINANCE_API_KEY"
  api_secret: "YOUR_BINANCE_SECRET"
```

## References

- [퀀트 투자 강의 (FastCampus)](https://fastcampus.co.kr/fin_online_quant01)
- [FRED API Docs](https://fred.stlouisfed.org/docs/api/fred/)
- [FMP API Docs](https://site.financialmodelingprep.com/developer/docs)
