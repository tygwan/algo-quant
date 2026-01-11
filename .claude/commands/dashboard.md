---
description: Algo-Quant 대시보드 실행
arguments:
  - name: port
    description: 포트 번호 (기본값: 8501)
    required: false
---

# Dashboard 실행

Streamlit 대시보드를 uv로 실행합니다.

## 실행

```bash
PORT=${ARGUMENTS.port:-8501}
uv run --with streamlit --with plotly --with altair streamlit run src/ui/app.py --server.port=$PORT
```

## 접속

- Local: http://localhost:${ARGUMENTS.port:-8501}
- 종료: Ctrl+C

## 기능

| 페이지 | 설명 |
|--------|------|
| Dashboard | 포트폴리오 개요 |
| Data Explorer | 데이터 조회 |
| Factor Analysis | 팩터 분석 |
| Regime Monitor | 시장 체제 |
| Backtest | 백테스트 실행 |
| Portfolio | 포트폴리오 관리 |
