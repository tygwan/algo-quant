# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**algo-quant** is a quantitative investment automation agent for stocks and cryptocurrencies implementing Fama-French multi-factor models, macroeconomic regime classification, and automated portfolio management in Python 3.11+.

**Current Status**: Phase 1-5 complete (83%). Data infrastructure, factor models, regime classification, strategy development, and backtesting implemented. Streamlit dashboard available.

## Quick Start

```bash
# 대시보드 실행 (권장)
uv run --with streamlit --with plotly --with altair streamlit run src/ui/app.py

# 또는 명령어 사용
/dashboard

# 테스트 실행
uv run pytest tests/ -v
```

## Technology Stack

- **Python 3.11+** with pandas, numpy, scipy, scikit-learn
- **Package Manager**: uv (권장) 또는 pip
- **UI**: Streamlit + Plotly
- **Data Sources**: FMP API (stocks), FRED API (macro), Binance/Upbit (crypto)

## Architecture

```
src/
├── data/          # API clients (FMP, FRED, KIS, Kiwoom, Binance, Upbit)
├── factors/       # CAPM, FF3, FF5, factor neutralization
├── regime/        # Rule-based, HMM classifiers, signals
├── strategy/      # Optimizer, factor strategy, regime strategy, risk manager
├── backtest/      # Engine, metrics, walk-forward analysis
└── ui/            # Streamlit dashboard
```

## Commands (Slash Commands)

| Command | Purpose |
|---------|---------|
| `/dashboard` | Streamlit 대시보드 실행 |
| `/test` | pytest 테스트 실행 |
| `/backtest` | 백테스트 실행 |
| `/uv-run <script>` | uv로 스크립트 실행 |
| `/uv-sync` | 의존성 동기화 |
| `/phase status` | 현재 Phase 진행률 |

## uv 사용법

```bash
# 의존성 동기화
uv sync

# 스크립트 실행
uv run python script.py

# 임시 의존성과 함께 실행
uv run --with pandas python analyze.py

# Streamlit 앱 실행
uv run --with streamlit streamlit run src/ui/app.py
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Data Infrastructure | ✅ 100% | API clients, preprocessor, cache |
| 2. Factor Modeling | ✅ 100% | CAPM, FF3, FF5, neutralization |
| 3. Regime Classification | ✅ 100% | Rule-based, HMM, signals |
| 4. Strategy Development | ✅ 100% | Optimizer, factor/regime strategy |
| 5. Backtesting | ✅ 100% | Engine, metrics, walk-forward |
| 6. Production | ⏳ 0% | Real-time, live trading (future) |

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
- [uv Documentation](https://docs.astral.sh/uv/)
