# Phase 1: Data Infrastructure

## 목표
FMP, FRED, 암호화폐 API 클라이언트 구현 및 데이터 전처리 파이프라인 구축

## 범위

### 포함
- API 클라이언트 (FMP, FRED, Binance/Upbit)
- Rate Limiting 및 재시도 로직
- 데이터 전처리 및 정제
- 로컬 캐싱 시스템

### 제외
- 실시간 스트리밍 데이터
- 고빈도 데이터 처리

## 기술 상세

### FMP Client
```python
class FMPClient:
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(calls=300, period=60)

    def get_stock_price(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """조정 종가 포함 OHLCV 데이터 반환"""

    def get_fundamentals(self, symbol: str) -> dict:
        """재무제표 데이터 반환"""
```

### FRED Client
```python
class FREDClient:
    BASE_URL = "https://api.stlouisfed.org/fred"

    # 주요 시리즈 ID
    SERIES = {
        "gdp": "GDP",
        "unemployment": "UNRATE",
        "yield_10y": "DGS10",
        "yield_2y": "DGS2",
        "cpi": "CPIAUCSL",
    }
```

### Data Preprocessor
```python
class DataPreprocessor:
    """퀀트 작업의 50%+ 차지"""

    def handle_missing(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """결측치 처리: forward fill, interpolation"""

    def adjust_corporate_actions(self, df: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
        """주식 분할, 배당 조정"""

    def handle_outliers(self, df: pd.DataFrame, method: str = "winsorize", threshold: float = 3.0) -> pd.DataFrame:
        """이상치 처리: winsorization, z-score"""
```

### Cache System
```python
class DataCache:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)

    def get(self, key: str, ttl_days: int = 1) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 조회"""

    def set(self, key: str, data: pd.DataFrame) -> None:
        """캐시에 데이터 저장"""
```

## 완료 조건

- [ ] 모든 API 클라이언트 구현 완료
- [ ] Rate Limiting 테스트 통과
- [ ] 데이터 전처리 함수 단위 테스트 80%+ 커버리지
- [ ] 캐싱 시스템 동작 확인
- [ ] 통합 테스트 통과

## 의존성

- requests
- pandas
- numpy
- python-dotenv

## 예상 산출물

```
src/data/
├── __init__.py
├── fmp_client.py
├── fred_client.py
├── crypto_client.py
├── preprocessor.py
├── cache.py
└── rate_limiter.py
```
