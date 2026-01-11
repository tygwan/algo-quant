# PRD: algo-quant

## 제품 개요

**algo-quant**는 주식 및 암호화폐를 위한 퀀트 투자 자동화 에이전트입니다.

### 비전
팩터 기반 퀀트 전략과 거시경제 체제 분류를 결합하여 자동화된 포트폴리오 관리 시스템 구축

### 핵심 가치
- **데이터 중심**: 50%+ 시간을 데이터 전처리에 투자
- **체계적 접근**: 팩터 모델 기반의 과학적 투자
- **리스크 관리**: 체제 기반 동적 자산 배분

## 목표 사용자

| 사용자 | 니즈 |
|--------|------|
| 개인 투자자 | 체계적인 퀀트 전략 학습 및 적용 |
| 퀀트 연구자 | 팩터 모델 연구 및 백테스트 |
| 알고리즘 트레이더 | 자동화된 포트폴리오 리밸런싱 |

## 핵심 기능

### 1. 데이터 파이프라인
- [ ] FMP API 클라이언트 (미국 주식 데이터)
- [ ] FRED API 클라이언트 (거시경제 지표)
- [ ] 한국투자증권 API 클라이언트 (국내 주식)
- [ ] 키움증권 API 클라이언트 (국내 주식)
- [ ] Binance API 클라이언트 (글로벌 암호화폐)
- [ ] Upbit API 클라이언트 (국내 암호화폐)
- [ ] 데이터 전처리 파이프라인
- [ ] 로컬 캐싱 시스템

### 2. 팩터 모델링
- [ ] CAPM 구현
- [ ] Fama-French 3팩터 모델 (SMB, HML)
- [ ] Fama-French 5팩터 모델 (RMW, CMA 추가)
- [ ] 팩터 중립화 (Long-Short 전략)
- [ ] 커스텀 멀티팩터 전략

### 3. 거시경제 체제 분류
- [ ] FRED 지표 처리 (GDP, 실업률, 수익률곡선)
- [ ] 경기 사이클 분류 (확장, 수축, 침체, 회복)
- [ ] 체제 기반 시그널 생성
- [ ] 선행/동행/후행 지표 분석

### 4. 투자 전략
- [ ] 팩터 기반 자산 배분
- [ ] 체제 적응형 포트폴리오 구성
- [ ] 리스크 관리 규칙
- [ ] ETF/주식/암호화폐 통합 관리

### 5. 백테스팅
- [ ] 백테스팅 엔진
- [ ] 성과 지표 (Sharpe, Sortino, MDD 등)
- [ ] 파라미터 최적화
- [ ] Walk-forward 분석

### 6. 실전 투자 (Future)
- [ ] 실시간 데이터 파이프라인
- [ ] 자동 리밸런싱
- [ ] 한국투자증권 실거래 연동
- [ ] 키움증권 실거래 연동
- [ ] Binance 실거래 연동
- [ ] Upbit 실거래 연동

## 비기능 요구사항

### 성능
- API 호출 Rate Limiting 준수
- 로컬 캐싱으로 개발 효율성 확보

### 호환성
- Python 3.11+ 지원
- Google Colab 호환
- 로컬 개발 환경 지원

### 데이터 품질
- 생존자 편향 완화
- Point-in-time 데이터 처리
- 기업 행동 조정 (분할, 배당)

## 제외 범위

- 고빈도 매매 (HFT)
- 뉴스/소셜 미디어 감성 분석 (v1)
- 옵션/선물 파생상품 (v1)

## 성공 지표

| 지표 | 목표 |
|------|------|
| 백테스트 Sharpe Ratio | > 1.0 |
| 최대 낙폭 (MDD) | < 20% |
| 테스트 커버리지 | > 80% |
| 코드 문서화율 | > 90% |

## 참조

- [글로벌 퀀트 챔피언십 우승자와 함께 하는 퀀트 투자](https://fastcampus.co.kr/fin_online_quant01)
- [FMP API 문서](https://site.financialmodelingprep.com/developer/docs)
- [FRED API 문서](https://fred.stlouisfed.org/docs/api/fred/)
- [한국투자증권 API](https://apiportal.koreainvestment.com)
- [키움증권 Open API](https://openapi.kiwoom.com)
- [Binance API 문서](https://binance-docs.github.io/apidocs)
- [Upbit API 문서](https://docs.upbit.com)
