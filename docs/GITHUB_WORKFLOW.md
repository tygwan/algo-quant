# GitHub 기록 운영 가이드

이 저장소에서 작업 기록을 남기는 기본 채널은 아래 5가지입니다.

## 1) Issues (무엇을 할지)
- 버그: `Bug Report` 템플릿
- 기능: `Feature Request` 템플릿
- 실험: `Quant Experiment Log` 템플릿

권장 기록 항목:
- 문제/가설
- 재현 절차 또는 실험 셋업
- 측정 지표(Sharpe, MDD, latency 등)
- 완료 조건(acceptance criteria)

## 2) Pull Requests (어떻게 바꿨는지)
- `.github/pull_request_template.md` 기반으로 작성
- 반드시 포함:
  - 변경 요약
  - 검증 명령/결과
  - 리스크와 롤백 계획
  - 비밀정보 미포함 체크

## 3) Code Review (왜 승인/거절했는지)
- 리뷰 코멘트는 아래 기준으로 기록:
  - 버그/회귀 가능성
  - 전략/데이터 가정 타당성
  - 테스트 누락
  - 운영 리스크

## 4) Projects (진행 상태)
- 칸반 권장 컬럼:
  - Backlog
  - Ready
  - In Progress
  - In Review
  - Done
- 각 카드에 Issue/PR 링크 연결

## 5) Releases (언제 배포했는지)
- 배포 단위마다 릴리스 노트 작성:
  - 사용자 영향
  - 마이그레이션/설정 변경
  - 알려진 제한 사항

---

## 최소 운영 루프 (권장)
1. Issue 생성 (문제/가설/완료조건)
2. 브랜치 작업 + PR 생성
3. 리뷰/수정
4. 머지 후 Issue 닫기
5. 실험성 변경이면 `Quant Experiment Log`에 결과 기록
