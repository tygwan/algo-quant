---
name: uv
description: uv 패키지 매니저 사용 가이드. Python 프로젝트 설정, 의존성 관리, 스크립트 실행 시 사용. "uv", "패키지", "의존성", "가상환경", "pip" 키워드에 반응.
---

# uv 패키지 매니저 스킬

uv는 Rust로 작성된 초고속 Python 패키지 매니저입니다.

## 핵심 명령어

### 프로젝트 초기화
```bash
# 새 프로젝트 생성
uv init project-name

# 기존 프로젝트에 uv 설정
uv init
```

### 의존성 관리
```bash
# 패키지 추가
uv add pandas numpy

# 개발 의존성 추가
uv add --dev pytest black

# 패키지 제거
uv remove package-name

# 의존성 동기화
uv sync
```

### 스크립트 실행
```bash
# 프로젝트 스크립트 실행
uv run python script.py

# 임시 의존성과 함께 실행
uv run --with requests python fetch_data.py

# pytest 실행
uv run pytest

# streamlit 앱 실행
uv run --with streamlit streamlit run app.py
```

### lock 파일 관리
```bash
# lock 파일 생성/업데이트
uv lock

# lock 파일 기반 설치
uv sync --locked
```

## algo-quant 프로젝트에서 사용

### 대시보드 실행
```bash
uv run --with streamlit --with plotly --with altair streamlit run src/ui/app.py
```

### 테스트 실행
```bash
uv run pytest tests/ -v
```

### 커버리지 포함 테스트
```bash
uv run pytest tests/ --cov=src --cov-report=html
```

## pip 대비 장점

| 항목 | pip | uv |
|------|-----|-----|
| 설치 속도 | 느림 | 10-100x 빠름 |
| 의존성 해결 | 기본 | 고급 (SAT solver) |
| lock 파일 | 별도 도구 필요 | 내장 |
| 가상환경 | 수동 관리 | 자동 관리 |

## pyproject.toml 설정

```toml
[project]
name = "algo-quant"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
]
ui = [
    "streamlit>=1.29.0",
    "plotly>=5.18.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
]
```

## 주의사항

1. **pyproject.toml 필수**: uv는 pyproject.toml 기반으로 동작
2. **캐시 활용**: `~/.cache/uv`에 패키지 캐시 저장
3. **Python 버전**: `uv python install 3.11`로 Python 설치 가능
