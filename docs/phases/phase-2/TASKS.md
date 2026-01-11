# Phase 2 Tasks

## 작업 목록

### 1. 팩터 모델 기반 클래스
- [x] `src/factors/base.py` 생성
- [x] FactorModel ABC 정의
- [x] 공통 유틸리티 함수

### 2. CAPM 구현
- [x] `src/factors/capm.py` 생성
- [x] 베타 계산 (OLS 회귀)
- [x] Rolling 베타 계산
- [x] 기대 수익률 계산
- [x] 알파 계산
- [x] 단위 테스트 작성

### 3. Fama-French 데이터 로더
- [x] `src/factors/ff_data.py` 생성
- [x] Kenneth French Data Library 연동
- [x] 3팩터 데이터 로드 (Mkt-RF, SMB, HML)
- [x] 5팩터 데이터 로드 (RMW, CMA 추가)
- [x] 캐싱 지원

### 4. Fama-French 3팩터 모델
- [x] `src/factors/ff3.py` 생성
- [x] 3팩터 회귀분석
- [x] 팩터 로딩 추출
- [x] t-stats, p-values, R² 계산
- [x] 단위 테스트 작성

### 5. Fama-French 5팩터 모델
- [x] `src/factors/ff5.py` 생성
- [x] 5팩터 회귀분석
- [x] 팩터 로딩 추출
- [x] 모델 비교 도구
- [x] 단위 테스트 작성

### 6. 팩터 중립화
- [x] `src/factors/neutralizer.py` 생성
- [x] 단일 팩터 중립화
- [x] 다중 팩터 중립화
- [x] Long-Short 포트폴리오 구성
- [x] 단위 테스트 작성

## 진행 상황

| Task | 상태 | 완료일 |
|------|------|--------|
| Base Class | ✅ | 2026-01-11 |
| CAPM | ✅ | 2026-01-11 |
| FF Data Loader | ✅ | 2026-01-11 |
| FF3 Model | ✅ | 2026-01-11 |
| FF5 Model | ✅ | 2026-01-11 |
| Factor Neutralizer | ✅ | 2026-01-11 |

## Phase 2 완료 ✅

모든 팩터 모델링 작업이 완료되었습니다.
