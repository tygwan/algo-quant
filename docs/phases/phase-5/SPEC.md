# Phase 5: Backtesting

## 목표
과거 데이터를 기반으로 전략 성과를 평가하는 백테스팅 프레임워크 구축

## 주요 컴포넌트

### 1. 백테스팅 엔진 (engine.py)
- 이벤트 기반 백테스터
- 거래 비용 모델 (수수료, 슬리피지)
- 마진/레버리지 지원
- 현금 관리

### 2. 성과 지표 (metrics.py)
- 수익률: CAGR, Total Return
- 위험: Volatility, Max Drawdown, VaR
- 위험 조정 수익: Sharpe, Sortino, Calmar
- 거래 통계: Win Rate, Profit Factor

### 3. Walk-Forward 분석 (walk_forward.py)
- 롤링 윈도우 최적화
- Out-of-Sample 테스트
- 파라미터 안정성 분석

### 4. 리포팅 (reporting.py)
- 성과 요약 리포트
- 수익률 차트
- 드로다운 분석

## 인터페이스

```python
# 백테스팅 실행
from src.backtest import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date="2015-01-01",
    end_date="2023-12-31",
    initial_capital=100000,
    commission=0.001,
)
engine = BacktestEngine(config)
result = engine.run(strategy, data)

# 성과 분석
from src.backtest import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(result)
metrics = analyzer.calculate_all_metrics()
analyzer.generate_report("backtest_report.html")
```

## 의존성
- pandas, numpy (데이터 처리)
- matplotlib, plotly (시각화)
- scipy (통계 분석)
