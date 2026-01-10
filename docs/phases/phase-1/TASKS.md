# Phase 1: Tasks

## Task List

### T1-01: 프로젝트 기본 구조 생성
- **Status**: ⏳ 대기
- **Description**: src/, tests/, config/ 디렉토리 및 기본 파일 생성
- **Subtasks**:
  - [ ] src/data/ 디렉토리 생성
  - [ ] tests/unit/ 디렉토리 생성
  - [ ] config/settings.yaml 작성
  - [ ] requirements.txt 작성

### T1-02: Rate Limiter 구현
- **Status**: ⏳ 대기
- **Description**: API 호출 제한을 위한 Rate Limiter 클래스
- **Subtasks**:
  - [ ] RateLimiter 클래스 구현
  - [ ] 지수 백오프 재시도 로직
  - [ ] 단위 테스트 작성

### T1-03: FMP API 클라이언트
- **Status**: ⏳ 대기
- **Description**: Financial Modeling Prep API 클라이언트 구현
- **Subtasks**:
  - [ ] FMPClient 기본 클래스
  - [ ] 주가 데이터 조회 (get_stock_price)
  - [ ] 재무제표 조회 (get_fundamentals)
  - [ ] 재무비율 조회 (get_financial_ratios)
  - [ ] 단위 테스트 작성

### T1-04: FRED API 클라이언트
- **Status**: ⏳ 대기
- **Description**: FRED 거시경제 지표 API 클라이언트 구현
- **Subtasks**:
  - [ ] FREDClient 기본 클래스
  - [ ] 시계열 조회 (get_series)
  - [ ] GDP, 실업률, 수익률곡선 편의 메서드
  - [ ] 단위 테스트 작성

### T1-05: 암호화폐 API 클라이언트
- **Status**: ⏳ 대기
- **Description**: Binance/Upbit API 클라이언트 구현
- **Subtasks**:
  - [ ] CryptoClient 기본 클래스
  - [ ] Binance 구현
  - [ ] Upbit 구현 (선택)
  - [ ] 단위 테스트 작성

### T1-06: 데이터 전처리 파이프라인
- **Status**: ⏳ 대기
- **Description**: 데이터 정제 및 전처리 기능 구현
- **Subtasks**:
  - [ ] DataPreprocessor 클래스
  - [ ] 결측치 처리 (handle_missing)
  - [ ] 이상치 처리 (handle_outliers)
  - [ ] 기업 행동 조정 (adjust_corporate_actions)
  - [ ] Point-in-time 데이터 처리
  - [ ] 단위 테스트 작성

### T1-07: 로컬 캐싱 시스템
- **Status**: ⏳ 대기
- **Description**: API 응답 캐싱으로 개발 효율성 확보
- **Subtasks**:
  - [ ] DataCache 클래스
  - [ ] TTL 기반 캐시 만료
  - [ ] 파일 기반 저장
  - [ ] 단위 테스트 작성

### T1-08: 통합 테스트
- **Status**: ⏳ 대기
- **Description**: 데이터 파이프라인 E2E 테스트
- **Subtasks**:
  - [ ] API → 전처리 → 캐시 플로우 테스트
  - [ ] 에러 처리 테스트

## Progress Summary

| Task | Status |
|------|--------|
| T1-01 | ⏳ |
| T1-02 | ⏳ |
| T1-03 | ⏳ |
| T1-04 | ⏳ |
| T1-05 | ⏳ |
| T1-06 | ⏳ |
| T1-07 | ⏳ |
| T1-08 | ⏳ |

**Phase Progress**: 0/8 tasks (0%)
