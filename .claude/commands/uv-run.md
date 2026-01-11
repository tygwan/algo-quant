---
description: uv로 Python 스크립트 또는 모듈 실행
arguments:
  - name: script
    description: 실행할 스크립트 또는 모듈 (예: src/ui/app.py, pytest)
    required: true
  - name: extra_deps
    description: 추가 의존성 (쉼표 구분)
    required: false
---

# uv run 명령어

uv를 사용하여 Python 스크립트를 실행합니다. 가상환경 없이도 즉시 실행 가능합니다.

## 실행

```bash
# 기본 실행
uv run $ARGUMENTS.script

# 추가 의존성이 있는 경우
uv run --with $ARGUMENTS.extra_deps $ARGUMENTS.script
```

## 일반적인 사용 예시

| 명령어 | 설명 |
|--------|------|
| `/uv-run pytest` | 테스트 실행 |
| `/uv-run src/ui/app.py` | 스크립트 실행 |
| `/uv-run "streamlit run src/ui/app.py"` | Streamlit 앱 실행 |

## 주의사항

- pyproject.toml이 있는 디렉토리에서 실행
- 의존성은 자동으로 설치됨
- 캐시를 사용하여 빠른 재실행
