# Phase 2: Factor Modeling

## 목표
CAPM 및 Fama-French 팩터 모델 구현

## 범위

### 포함
- CAPM (Capital Asset Pricing Model)
- Fama-French 3팩터 모델 (Market, SMB, HML)
- Fama-French 5팩터 모델 (+ RMW, CMA)
- 팩터 중립화 (Long-Short Portfolio)

### 제외
- 대체 팩터 (모멘텀, 품질 등) - v2
- 머신러닝 기반 팩터 - v2

## 기술 상세

### CAPM
```
E(Ri) = Rf + βi × (E(Rm) - Rf)

- Ri: 자산 i의 기대 수익률
- Rf: 무위험 수익률
- βi: 시장 베타
- Rm: 시장 수익률
```

### Fama-French 3 Factor
```
Ri - Rf = αi + βi(Rm - Rf) + si×SMB + hi×HML + εi

- SMB: Small Minus Big (소형주 - 대형주)
- HML: High Minus Low (가치주 - 성장주)
```

### Fama-French 5 Factor
```
+ RMW: Robust Minus Weak (고수익성 - 저수익성)
+ CMA: Conservative Minus Aggressive (저투자 - 고투자)
```

## 완료 조건

- [ ] CAPM 베타 계산 정확도 검증
- [ ] FF3 팩터 계산 검증
- [ ] FF5 팩터 계산 검증
- [ ] 팩터 중립화 포트폴리오 구성 가능
- [ ] 단위 테스트 80%+ 커버리지

## 예상 산출물

```
src/factors/
├── __init__.py
├── capm.py
├── fama_french.py
├── factor_neutral.py
└── multi_factor.py
```
