---
description: uv로 프로젝트 의존성 동기화
---

# uv sync 명령어

프로젝트 의존성을 동기화하고 가상환경을 설정합니다.

## 실행

```bash
# 의존성 동기화
uv sync

# 개발 의존성 포함
uv sync --dev

# lock 파일 업데이트
uv lock
```

## 결과

- `.venv/` 디렉토리에 가상환경 생성
- `uv.lock` 파일에 정확한 버전 고정
- 모든 의존성 설치 완료
