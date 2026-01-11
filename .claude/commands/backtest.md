---
description: 백테스트 실행 (CLI)
arguments:
  - name: strategy
    description: 전략 유형 (equal_weight, risk_parity, momentum)
    required: false
  - name: symbols
    description: 종목 리스트 (쉼표 구분)
    required: false
  - name: period
    description: 백테스트 기간 (일)
    required: false
---

# 백테스트 CLI

명령줄에서 백테스트를 실행합니다.

## 실행

```bash
# 대시보드에서 실행 권장
uv run --with streamlit streamlit run src/ui/app.py

# 또는 Python 스크립트로 실행
uv run python -c "
from src.backtest import BacktestEngine, BacktestConfig
from src.ui.utils import generate_sample_prices

symbols = '${ARGUMENTS.symbols:-AAPL,MSFT,GOOGL}'.split(',')
periods = int('${ARGUMENTS.period:-252}')

prices = generate_sample_prices(symbols, periods)
# ... 백테스트 로직
print('Run backtest via dashboard for full functionality')
"
```

## 권장 사항

대시보드(`/dashboard`)에서 백테스트 실행을 권장합니다:
- 시각적 결과 확인
- 성과 분석 차트
- 파라미터 조정 용이
