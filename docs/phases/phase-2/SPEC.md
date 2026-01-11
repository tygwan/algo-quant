# Phase 2: Factor Modeling Specification

## 목표

Fama-French 다중 팩터 모델을 구현하여 주식 수익률의 위험 요인을 분석하고, 팩터 기반 포트폴리오 구성의 기반을 마련합니다.

## 범위

### 1. CAPM (Capital Asset Pricing Model)
- **공식**: E(Ri) = Rf + βi × (E(Rm) - Rf)
- **구성요소**:
  - Rf: 무위험 이자율 (FRED Treasury 데이터)
  - βi: 시장 베타 (회귀분석)
  - Rm: 시장 수익률 (S&P 500)
- **기능**:
  - 베타 계산 (rolling window 지원)
  - 기대 수익률 계산
  - 알파 계산 (초과 수익률)

### 2. Fama-French 3팩터 모델
- **공식**: Ri - Rf = αi + βi(Rm - Rf) + si(SMB) + hi(HML) + εi
- **팩터**:
  - Market (Rm - Rf): 시장 초과 수익률
  - SMB (Small Minus Big): 소형주 프리미엄
  - HML (High Minus Low): 가치주 프리미엄 (B/M 비율)
- **데이터 소스**: Kenneth French Data Library

### 3. Fama-French 5팩터 모델
- **공식**: Ri - Rf = αi + βi(Rm-Rf) + si(SMB) + hi(HML) + ri(RMW) + ci(CMA) + εi
- **추가 팩터**:
  - RMW (Robust Minus Weak): 수익성 팩터
  - CMA (Conservative Minus Aggressive): 투자 팩터

### 4. 팩터 중립화 (Factor Neutralization)
- 포트폴리오의 특정 팩터 노출도를 0으로 조정
- Long-Short 전략 지원
- 팩터 로딩 분석 및 리스크 분해

## 기술 명세

### 입력
- 가격 데이터 (DataFrame: date, close)
- 팩터 데이터 (Kenneth French Library)
- 무위험 이자율 (FRED)

### 출력
- 팩터 로딩 (베타, SMB, HML, RMW, CMA 계수)
- 알파 (비시장적 초과수익)
- R² (설명력)
- t-statistics 및 p-values

### 클래스 구조

```python
class FactorModel(ABC):
    """팩터 모델 기본 클래스"""
    def fit(self, returns, factor_returns): ...
    def get_loadings(self): ...
    def get_alpha(self): ...
    def predict(self, factor_returns): ...

class CAPM(FactorModel):
    """자본자산가격결정모델"""

class FamaFrench3(FactorModel):
    """FF 3팩터 모델"""

class FamaFrench5(FactorModel):
    """FF 5팩터 모델"""

class FactorNeutralizer:
    """팩터 중립화 도구"""
    def neutralize(self, weights, target_loadings): ...
```

## 의존성

- pandas, numpy: 데이터 처리
- statsmodels: 회귀분석
- scipy: 최적화
- pandas-datareader: Kenneth French 데이터 로드

## 테스트 요구사항

- CAPM 베타 계산 정확도 검증
- 3팩터 모델 팩터 로딩 검증
- 5팩터 모델 팩터 로딩 검증
- 팩터 중립화 결과 검증 (타겟 로딩 달성 여부)

## 참고자료

- Fama, E.F. & French, K.R. (1993). "Common risk factors in the returns on stocks and bonds"
- Fama, E.F. & French, K.R. (2015). "A five-factor asset pricing model"
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
