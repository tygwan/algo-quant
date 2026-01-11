# algo-quant

퀀트 투자 자동화 에이전트 - Fama-French 멀티팩터 모델과 거시경제 체제 분류를 활용한 포트폴리오 관리 시스템

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 개요

algo-quant는 **국내/해외 주식 + 암호화폐**를 대상으로 한 통합 퀀트 투자 자동화 에이전트입니다.

### 주요 기능

- **다중 자산 지원**: 미국 주식, 한국 주식, 암호화폐 통합 관리
- **팩터 모델링**: CAPM, Fama-French 3팩터/5팩터 모델
- **거시경제 체제 분류**: FRED 지표 기반 경기 사이클 분석
- **자동화된 백테스팅**: Walk-forward 분석, 성과 지표 계산
- **실거래 연동** (예정): 한국투자증권, 키움증권, Binance, Upbit

## 지원 데이터 소스

| 카테고리 | 데이터 소스 | 기능 |
|----------|-------------|------|
| 미국 주식 | FMP API | 가격, 재무제표, 기업 프로필 |
| 한국 주식 | 한국투자증권 (KIS) | 시세, 주문, 잔고 조회 |
| 한국 주식 | 키움증권 | 시세, 주문, 조건검색 |
| 거시경제 | FRED API | GDP, 실업률, 금리, 수익률곡선 |
| 암호화폐 | Binance | OHLCV, 주문, 잔고 (글로벌) |
| 암호화폐 | Upbit | OHLCV, 주문, 잔고 (국내) |

## 설치

### 요구사항

- Python 3.11 이상
- pip 또는 uv

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/algo-quant.git
cd algo-quant

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### API 키 설정

```bash
# API 키 설정 파일 복사
cp config/api_keys.yaml.example config/api_keys.yaml

# api_keys.yaml 파일을 편집하여 API 키 입력
```

## 프로젝트 구조

```
algo-quant/
├── src/
│   ├── data/           # API 클라이언트 및 데이터 처리
│   │   ├── base_client.py    # 공통 HTTP 클라이언트
│   │   ├── fmp_client.py     # FMP API (미국 주식)
│   │   ├── fred_client.py    # FRED API (거시경제)
│   │   ├── kis_client.py     # 한국투자증권 API
│   │   ├── kiwoom_client.py  # 키움증권 API
│   │   ├── binance_client.py # Binance API
│   │   ├── upbit_client.py   # Upbit API
│   │   ├── preprocessor.py   # 데이터 전처리
│   │   └── cache.py          # 캐싱 시스템
│   ├── factors/        # 팩터 모델 (CAPM, FF3, FF5)
│   ├── regime/         # 경기 체제 분류
│   ├── strategy/       # 투자 전략
│   ├── backtest/       # 백테스팅 엔진
│   └── execution/      # 실거래 (예정)
├── tests/              # 테스트
├── config/             # 설정 파일
├── docs/               # 문서
│   ├── PRD.md          # 제품 요구사항
│   ├── TECH-SPEC.md    # 기술 설계서
│   ├── PROGRESS.md     # 진행 현황
│   └── phases/         # Phase별 문서
└── README.md
```

## 사용법

### 기본 예제

```python
from src.data import FMPClient

# FMP 클라이언트 초기화
client = FMPClient(api_key="your_api_key")

# AAPL 주가 데이터 조회
prices = client.get_historical_prices("AAPL", start="2023-01-01", end="2024-01-01")
print(prices.head())

# 재무제표 조회
financials = client.get_financial_statements("AAPL", statement_type="income")
print(financials.head())
```

### 한국 주식 (한국투자증권)

```python
from src.data import KISClient

# KIS 클라이언트 초기화
client = KISClient(
    app_key="your_app_key",
    app_secret="your_app_secret",
    account_no="your_account",
    is_paper=True  # 모의투자
)

# 삼성전자 현재가 조회
price = client.get_price("005930")
print(price)
```

### 암호화폐 (Binance)

```python
from src.data import BinanceClient

# Binance 클라이언트 초기화
client = BinanceClient(
    api_key="your_api_key",
    api_secret="your_secret"
)

# BTC/USDT 일봉 데이터 조회
klines = client.get_klines("BTCUSDT", interval="1d", limit=100)
print(klines.head())
```

## 개발 현황

### Phase 1: Data Infrastructure (진행 중)

| 구성요소 | 상태 |
|----------|------|
| Base Client | ✅ 완료 |
| FMP Client (미국 주식) | ✅ 완료 |
| FRED Client (거시경제) | 🔄 진행 중 |
| KIS Client (한국투자증권) | ⏳ 대기 |
| Kiwoom Client (키움증권) | ⏳ 대기 |
| Binance Client | ⏳ 대기 |
| Upbit Client | ⏳ 대기 |

### 향후 계획

- **Phase 2**: 팩터 모델링 (CAPM, FF3, FF5)
- **Phase 3**: 거시경제 체제 분류
- **Phase 4**: 투자 전략 개발
- **Phase 5**: 백테스팅
- **Phase 6**: 실거래 연동

## 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=src --cov-report=term-missing
```

## 문서

- [PRD (제품 요구사항)](docs/PRD.md)
- [기술 설계서](docs/TECH-SPEC.md)
- [개발 범위](docs/DEVELOPMENT-SCOPE.md)
- [진행 현황](docs/PROGRESS.md)

## 참고 자료

- [글로벌 퀀트 챔피언십 우승자와 함께 하는 퀀트 투자](https://fastcampus.co.kr/fin_online_quant01)
- [FMP API 문서](https://site.financialmodelingprep.com/developer/docs)
- [FRED API 문서](https://fred.stlouisfed.org/docs/api/fred/)
- [한국투자증권 API](https://apiportal.koreainvestment.com)
- [키움증권 Open API](https://openapi.kiwoom.com)
- [Binance API](https://binance-docs.github.io/apidocs)
- [Upbit API](https://docs.upbit.com)

## 라이선스

MIT License
