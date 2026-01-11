# Phase 1 Tasks

## 작업 목록

### 1. 프로젝트 기본 구조 설정
- [x] src/ 디렉토리 구조 생성
- [x] tests/ 디렉토리 구조 생성
- [x] config/ 디렉토리 및 템플릿 생성
- [x] requirements.txt 작성
- [x] pyproject.toml (pytest 설정)

### 2. Base Client 구현
- [x] `src/data/base_client.py` 생성
- [x] HTTP 요청 래퍼 구현
- [x] Rate limiter 구현
- [x] 재시도 로직 구현
- [x] 에러 핸들링 구현

### 3. FMP API 클라이언트 (미국 주식)
- [x] `src/data/fmp_client.py` 생성
- [x] get_historical_prices() 구현
- [x] get_financial_statements() 구현
- [x] get_company_profile() 구현
- [x] 단위 테스트 작성

### 4. FRED API 클라이언트 (거시경제)
- [ ] `src/data/fred_client.py` 생성
- [ ] get_series() 구현
- [ ] get_series_info() 구현
- [ ] 주요 지표 상수 정의
- [ ] 단위 테스트 작성

### 5. 한국투자증권 API 클라이언트 (국내 주식)
- [ ] `src/data/kis_client.py` 생성
- [ ] OAuth 2.0 인증 구현
- [ ] get_price(), get_daily_prices() 구현
- [ ] get_balance(), create_order() 구현
- [ ] 단위 테스트 작성

### 6. 키움증권 API 클라이언트 (국내 주식)
- [ ] `src/data/kiwoom_client.py` 생성
- [ ] 인증 토큰 관리 구현
- [ ] get_price(), get_daily_prices() 구현
- [ ] get_balance(), create_order() 구현
- [ ] 단위 테스트 작성

### 7. Binance API 클라이언트 (글로벌 암호화폐)
- [ ] `src/data/binance_client.py` 생성
- [ ] HMAC 서명 구현
- [ ] get_klines(), get_ticker() 구현
- [ ] get_balance(), create_order() 구현
- [ ] 단위 테스트 작성

### 8. Upbit API 클라이언트 (국내 암호화폐)
- [ ] `src/data/upbit_client.py` 생성
- [ ] JWT 인증 구현
- [ ] get_candles(), get_ticker() 구현
- [ ] get_balance(), create_order() 구현
- [ ] 단위 테스트 작성

### 9. 데이터 전처리 파이프라인
- [ ] `src/data/preprocessor.py` 생성
- [ ] 결측치 처리 함수
- [ ] 이상치 탐지 함수
- [ ] 수익률 계산 함수
- [ ] 정규화/표준화 함수
- [ ] 단위 테스트 작성

### 10. 캐싱 시스템
- [ ] `src/data/cache.py` 생성
- [ ] 캐시 저장/조회 구현
- [ ] 만료 정책 구현
- [ ] 캐시 무효화 구현
- [ ] 단위 테스트 작성

## 진행 상황

| Task | 상태 | 완료일 |
|------|------|--------|
| 프로젝트 구조 설정 | ✅ | 2026-01-11 |
| Base Client | ✅ | 2026-01-11 |
| FMP Client | ✅ | 2026-01-11 |
| FRED Client | 🔄 | - |
| KIS Client | ⏳ | - |
| Kiwoom Client | ⏳ | - |
| Binance Client | ⏳ | - |
| Upbit Client | ⏳ | - |
| Preprocessor | ⏳ | - |
| Cache System | ⏳ | - |
