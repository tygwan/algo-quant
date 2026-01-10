# 기술 설계서: algo-quant

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         algo-quant Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │   FMP API    │   │   FRED API   │   │ Binance/Upbit│            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         │                  │                  │                      │
│         └──────────────────┼──────────────────┘                      │
│                            ▼                                         │
│                   ┌────────────────┐                                │
│                   │  Data Layer    │                                │
│                   │ (Preprocessor) │                                │
│                   └────────┬───────┘                                │
│                            │                                         │
│         ┌──────────────────┼──────────────────┐                      │
│         ▼                  ▼                  ▼                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │Factor Models │   │   Regime     │   │   Strategy   │            │
│  │ (FF3, FF5)   │   │ Classifier   │   │   Engine     │            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         │                  │                  │                      │
│         └──────────────────┼──────────────────┘                      │
│                            ▼                                         │
│                   ┌────────────────┐                                │
│                   │   Portfolio    │                                │
│                   │  Constructor   │                                │
│                   └────────┬───────┘                                │
│                            │                                         │
│              ┌─────────────┼─────────────┐                          │
│              ▼                           ▼                          │
│       ┌──────────────┐           ┌──────────────┐                  │
│       │  Backtester  │           │   Executor   │                  │
│       └──────────────┘           │   (Future)   │                  │
│                                  └──────────────┘                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 기술 스택

| 구분 | 기술 | 용도 |
|------|------|------|
| Language | Python 3.11+ | 메인 개발 언어 |
| Data | pandas, numpy | 데이터 조작 |
| ML | scikit-learn | 팩터 모델링, 체제 분류 |
| Backtest | backtrader/bt | 백테스팅 프레임워크 |
| API | requests, aiohttp | HTTP 클라이언트 |
| Config | PyYAML | 설정 관리 |
| Test | pytest | 테스트 프레임워크 |

## 모듈 설계

### 1. Data Layer (`src/data/`)

```python
# fmp_client.py
class FMPClient:
    """Financial Modeling Prep API 클라이언트"""
    def __init__(self, api_key: str): ...
    def get_stock_price(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    def get_fundamentals(self, symbol: str) -> dict: ...
    def get_financial_ratios(self, symbol: str) -> pd.DataFrame: ...

# fred_client.py
class FREDClient:
    """FRED API 클라이언트 (거시경제 지표)"""
    def __init__(self, api_key: str): ...
    def get_series(self, series_id: str, start: date, end: date) -> pd.Series: ...
    def get_gdp(self) -> pd.Series: ...
    def get_unemployment(self) -> pd.Series: ...
    def get_yield_curve(self) -> pd.DataFrame: ...

# preprocessor.py
class DataPreprocessor:
    """데이터 전처리 (50%+ 퀀트 작업)"""
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def adjust_corporate_actions(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def ensure_point_in_time(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

### 2. Factor Models (`src/factors/`)

```python
# fama_french.py
class CAPM:
    """Capital Asset Pricing Model: E(Ri) = Rf + βi(E(Rm) - Rf)"""
    def calculate_beta(self, returns: pd.Series, market: pd.Series) -> float: ...
    def expected_return(self, beta: float, risk_free: float, market_premium: float) -> float: ...

class FamaFrench3:
    """FF3: Market + SMB (Size) + HML (Value)"""
    def calculate_smb(self, returns: pd.DataFrame, market_caps: pd.Series) -> pd.Series: ...
    def calculate_hml(self, returns: pd.DataFrame, book_to_market: pd.Series) -> pd.Series: ...
    def factor_regression(self, returns: pd.Series) -> FactorLoadings: ...

class FamaFrench5(FamaFrench3):
    """FF5: FF3 + RMW (Profitability) + CMA (Investment)"""
    def calculate_rmw(self, returns: pd.DataFrame, operating_profit: pd.Series) -> pd.Series: ...
    def calculate_cma(self, returns: pd.DataFrame, asset_growth: pd.Series) -> pd.Series: ...

# factor_neutral.py
class FactorNeutralPortfolio:
    """팩터 중립화된 롱숏 포트폴리오"""
    def neutralize(self, positions: pd.Series, factor_exposures: pd.DataFrame) -> pd.Series: ...
```

### 3. Regime Classification (`src/regime/`)

```python
# classifier.py
class RegimeClassifier:
    """경기 사이클 분류기"""
    EXPANSION = "expansion"
    CONTRACTION = "contraction"
    RECESSION = "recession"
    RECOVERY = "recovery"

    def __init__(self, indicators: MacroIndicators): ...
    def classify(self, date: date) -> str: ...
    def get_regime_history(self, start: date, end: date) -> pd.Series: ...

# indicators.py
class MacroIndicators:
    """FRED 거시경제 지표 처리"""
    def __init__(self, fred_client: FREDClient): ...
    def get_leading_indicators(self) -> pd.DataFrame: ...
    def get_coincident_indicators(self) -> pd.DataFrame: ...
    def get_lagging_indicators(self) -> pd.DataFrame: ...
```

### 4. Strategy (`src/strategy/`)

```python
# base.py
class BaseStrategy(ABC):
    """전략 기본 클래스"""
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series: ...
    @abstractmethod
    def calculate_positions(self, signals: pd.Series, portfolio: Portfolio) -> pd.DataFrame: ...

# factor_strategy.py
class FactorStrategy(BaseStrategy):
    """팩터 기반 자산 배분 전략"""
    def __init__(self, factors: List[str], weights: Dict[str, float]): ...
    def rank_assets(self, factor_scores: pd.DataFrame) -> pd.Series: ...

# regime_strategy.py
class RegimeAdaptiveStrategy(BaseStrategy):
    """체제 적응형 전략"""
    def __init__(self, regime_classifier: RegimeClassifier): ...
    def adjust_allocation(self, regime: str, base_allocation: pd.Series) -> pd.Series: ...
```

### 5. Backtest (`src/backtest/`)

```python
# engine.py
class BacktestEngine:
    """백테스팅 엔진"""
    def __init__(self, strategy: BaseStrategy, data: pd.DataFrame): ...
    def run(self, start: date, end: date) -> BacktestResult: ...
    def walk_forward(self, train_period: int, test_period: int) -> List[BacktestResult]: ...

# metrics.py
class PerformanceMetrics:
    """성과 지표 계산"""
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free: float = 0) -> float: ...
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free: float = 0) -> float: ...
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float: ...
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float: ...
```

## 데이터 모델

### 가격 데이터 스키마
```python
@dataclass
class PriceData:
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float  # 조정 종가 (분할, 배당 반영)
    volume: int
```

### 팩터 로딩 스키마
```python
@dataclass
class FactorLoadings:
    market: float    # 시장 베타
    smb: float       # 소형주 프리미엄
    hml: float       # 가치 프리미엄
    rmw: float       # 수익성 프리미엄 (FF5)
    cma: float       # 투자 프리미엄 (FF5)
    r_squared: float # 설명력
```

## API Rate Limiting

| API | 제한 | 전략 |
|-----|------|------|
| FMP | 300/min (Free) | 지수 백오프 + 로컬 캐시 |
| FRED | 120/min | 배치 요청 + 캐시 |
| Binance | 1200/min | 웹소켓 우선 |

## 테스트 전략

```
tests/
├── unit/              # 단위 테스트
│   ├── test_fmp_client.py
│   ├── test_fred_client.py
│   └── test_factor_models.py
├── integration/       # 통합 테스트
│   └── test_data_pipeline.py
└── backtest/          # 백테스트 검증
    └── test_strategy_performance.py
```

## 설정 관리

```yaml
# config/settings.yaml
data:
  cache_dir: ".cache"
  cache_ttl_days: 1

backtest:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.001

strategy:
  rebalance_frequency: "monthly"
  max_position_size: 0.1
```

## 의존성 관리

```
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
backtrader>=1.9.76
requests>=2.31.0
aiohttp>=3.8.0
pyyaml>=6.0.0
pytest>=7.4.0
python-dotenv>=1.0.0
```
