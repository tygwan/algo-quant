# Phase 4: Strategy Development

## 목표
팩터 기반 자산 배분 및 체제 적응형 포트폴리오 전략 구현

## 범위

### 포함
- 전략 기본 클래스 (BaseStrategy)
- 팩터 기반 자산 배분 전략
- 체제 적응형 포트폴리오 구성
- 리스크 관리 규칙

### 제외
- 실시간 트레이딩
- 고빈도 전략

## 기술 상세

### Strategy Interface
```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def calculate_positions(self, signals: pd.Series) -> pd.DataFrame: ...
```

### Factor Strategy
- 팩터 점수 기반 자산 랭킹
- 상위 N% 매수, 하위 N% 매도
- 월간/분기별 리밸런싱

### Regime-Adaptive Strategy
| 체제 | 주식 | 채권 | 현금 |
|------|------|------|------|
| Expansion | 80% | 15% | 5% |
| Contraction | 40% | 40% | 20% |
| Recession | 20% | 30% | 50% |
| Recovery | 60% | 30% | 10% |

## 완료 조건

- [ ] BaseStrategy 인터페이스 정의
- [ ] FactorStrategy 구현
- [ ] RegimeAdaptiveStrategy 구현
- [ ] 포트폴리오 구성 로직
- [ ] 리스크 관리 규칙
- [ ] 단위 테스트 80%+

## 예상 산출물

```
src/strategy/
├── __init__.py
├── base.py
├── factor_strategy.py
├── regime_strategy.py
├── portfolio.py
└── risk.py
```
