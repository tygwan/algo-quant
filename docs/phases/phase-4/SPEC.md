# Phase 4: Strategy Development Specification

## 목표

팩터 모델과 체제 분류를 활용한 자산 배분 전략 및 포트폴리오 최적화 시스템을 구현합니다.

## 범위

### 1. 팩터 기반 자산 배분

팩터 노출도에 따른 전략적 자산 배분:

**팩터 틸트 전략**:
- 가치 팩터 (HML): 가치주 비중 조정
- 모멘텀 팩터: 상승 모멘텀 종목 선호
- 퀄리티 팩터 (RMW): 수익성 높은 기업 선호
- 저변동성: 변동성 낮은 자산 선호

**리스크 패리티**:
- 위험 기여도 균등 배분
- 자산군별 변동성 기반 가중치

### 2. 체제 적응형 포트폴리오

시장 체제에 따른 동적 배분:

| 체제 | 주식 | 채권 | 대체자산 | 현금 |
|------|------|------|----------|------|
| Expansion | 70% | 15% | 10% | 5% |
| Peak | 40% | 30% | 15% | 15% |
| Contraction | 20% | 50% | 5% | 25% |
| Trough | 50% | 25% | 10% | 15% |

### 3. 리스크 관리

**포지션 사이징**:
- Kelly Criterion
- Fixed Fractional
- Volatility Targeting

**손절/익절**:
- Trailing Stop
- ATR 기반 스톱
- 시간 기반 청산

**포트폴리오 제약**:
- 최대 섹터 비중
- 최대 단일 종목 비중
- 최소 분산화 요구

## 기술 명세

### 클래스 구조

```python
class PortfolioOptimizer:
    """포트폴리오 최적화"""
    def mean_variance(self, returns, target_return): ...
    def risk_parity(self, returns): ...
    def minimum_variance(self, returns): ...
    def maximum_sharpe(self, returns, rf): ...

class FactorStrategy:
    """팩터 기반 전략"""
    def calculate_factor_scores(self, data): ...
    def generate_weights(self, scores): ...
    def rebalance(self, current_weights, target_weights): ...

class RegimeAdaptiveStrategy:
    """체제 적응형 전략"""
    def get_regime_allocation(self, regime): ...
    def transition_portfolio(self, old_regime, new_regime): ...

class RiskManager:
    """리스크 관리"""
    def calculate_position_size(self, signal, volatility): ...
    def check_risk_limits(self, portfolio): ...
    def generate_stop_levels(self, positions): ...
```

### 입력
- 자산 가격 데이터
- 팩터 로딩/스코어
- 현재 체제 분류
- 리스크 파라미터

### 출력
- 목표 포트폴리오 가중치
- 리밸런싱 지시
- 리스크 지표
- 스톱 레벨

## 의존성

- scipy.optimize: 포트폴리오 최적화
- cvxpy (optional): Convex optimization
- numpy, pandas: 계산

## 참고자료

- Markowitz, H. (1952). "Portfolio Selection"
- Risk Parity: Bridgewater All Weather Strategy
- Kelly, J.L. (1956). "A New Interpretation of Information Rate"
