---
description: pytest로 테스트 실행
arguments:
  - name: path
    description: 테스트 경로 (기본값: tests/)
    required: false
  - name: options
    description: pytest 옵션 (예: -v, -x, --cov)
    required: false
---

# 테스트 실행

uv와 pytest를 사용하여 테스트를 실행합니다.

## 실행

```bash
PATH_ARG="${ARGUMENTS.path:-tests/}"
OPTIONS="${ARGUMENTS.options:--v}"
uv run pytest $PATH_ARG $OPTIONS
```

## 예시

| 명령어 | 설명 |
|--------|------|
| `/test` | 전체 테스트 |
| `/test tests/backtest` | 백테스트 모듈만 |
| `/test --options="-v --cov=src"` | 커버리지 포함 |
| `/test --options="-x"` | 첫 실패시 중단 |
