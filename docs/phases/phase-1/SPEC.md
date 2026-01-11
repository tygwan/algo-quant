# Phase 1: Data Infrastructure

## 목표

주식, 암호화폐, 거시경제 데이터를 수집하고 전처리하는 견고한 데이터 인프라 구축

## 범위

### 포함
- FMP API 클라이언트 (주식 데이터)
- FRED API 클라이언트 (거시경제 지표)
- Crypto API 클라이언트 (Binance/Upbit)
- 데이터 전처리 파이프라인
- 로컬 캐싱 시스템

### 제외
- 실시간 스트리밍 (Phase 6)
- 브로커 연동 (Phase 6)

## 기술 요구사항

### FMP API 클라이언트
```python
class FMPClient:
    - get_historical_prices(symbol, start, end)
    - get_financial_statements(symbol)
    - get_company_profile(symbol)
    - rate_limit: 300 requests/min
```

### FRED API 클라이언트
```python
class FREDClient:
    - get_series(series_id, start, end)
    - get_series_info(series_id)
    - 주요 지표: GDP, UNRATE, T10Y2Y, FEDFUNDS
```

### Crypto API 클라이언트
```python
class BinanceClient:
    - get_klines(symbol, interval, start, end)
    - get_ticker(symbol)

class UpbitClient:
    - get_candles(market, interval, count)
```

### 데이터 전처리
- 결측치 처리 (forward fill, interpolation)
- 이상치 탐지 및 처리
- 수익률 계산 (단순, 로그)
- 정규화/표준화

### 캐싱 시스템
- SQLite 또는 Parquet 기반
- 캐시 만료 정책 (일별 데이터: 1일, 재무제표: 1주)
- 캐시 무효화 메커니즘

## 성공 기준

1. 모든 API 클라이언트 단위 테스트 통과
2. Rate limiting 정상 작동
3. 캐시 hit/miss 로깅
4. 최소 3년치 히스토리컬 데이터 수집 가능

## 의존성

- 외부: FMP API Key, FRED API Key, Binance API Key
- 내부: 없음 (첫 번째 Phase)

## 예상 산출물

```
src/data/
├── __init__.py
├── base_client.py      # 공통 HTTP 클라이언트
├── fmp_client.py       # FMP API
├── fred_client.py      # FRED API
├── binance_client.py   # Binance API
├── upbit_client.py     # Upbit API
├── preprocessor.py     # 데이터 전처리
└── cache.py            # 캐싱 시스템

tests/data/
├── test_fmp_client.py
├── test_fred_client.py
├── test_binance_client.py
├── test_upbit_client.py
├── test_preprocessor.py
└── test_cache.py
```
