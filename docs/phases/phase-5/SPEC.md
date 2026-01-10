# Phase 5: Backtesting & Optimization

## 목표
백테스팅 엔진 구현 및 전략 최적화

## 범위

### 포함
- 백테스팅 엔진
- 성과 지표 계산 (Sharpe, Sortino, MDD 등)
- 파라미터 최적화
- Walk-forward 분석
- 결과 리포팅

### 제외
- 실시간 시뮬레이션
- Monte Carlo 시뮬레이션 (v2)

## 기술 상세

### 성과 지표
```python
# 수익률 기반
sharpe_ratio = (returns.mean() - risk_free) / returns.std() * np.sqrt(252)
sortino_ratio = (returns.mean() - risk_free) / returns[returns < 0].std() * np.sqrt(252)

# 리스크 기반
max_drawdown = (cumulative - cumulative.cummax()).min()
calmar_ratio = cagr / abs(max_drawdown)

# 기타
win_rate = (returns > 0).sum() / len(returns)
profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum())
```

### Walk-Forward 분석
```
|--Train--|--Test--|
         |--Train--|--Test--|
                  |--Train--|--Test--|
```

## 완료 조건

- [ ] BacktestEngine 구현
- [ ] 모든 성과 지표 구현
- [ ] Walk-forward 분석 구현
- [ ] 리포트 생성
- [ ] 단위 테스트 80%+

## 예상 산출물

```
src/backtest/
├── __init__.py
├── engine.py
├── metrics.py
├── optimizer.py
├── walkforward.py
└── report.py
```
