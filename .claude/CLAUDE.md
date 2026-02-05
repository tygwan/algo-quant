# ultra-cc-init (algo-quant)

> **Type**: Claude Code Configuration Framework
> **Stack**: Markdown + Shell + JSON
> **Docs**: [.claude/docs/](.claude/docs/)

## Overview

ultra-cc-init 기반 통합 개발 워크플로우 프레임워크.
Agents, Skills, Hooks, Commands를 유기적으로 연결하여 효율적인 개발 환경을 제공합니다.

## Quick Start

```bash
# 설정 검증
/validate --full

# 기능 개발 시작
/feature start "기능명"

# Sprint 관리
/sprint start --phase N

# 사용 통계
/analytics
```

## Component Structure

```
.claude/
├── settings.json          # 통합 설정
├── agents/                # 26 specialized agents
│   ├── MANIFEST.md
│   ├── progress-tracker.md
│   ├── phase-tracker.md
│   ├── analytics-reporter.md
│   ├── github-manager.md
│   └── ...
├── skills/                # 28+ skills
│   ├── init/
│   ├── analytics/
│   ├── gh/
│   ├── codex/
│   └── ...
├── commands/              # 11 workflow commands
│   ├── feature.md
│   ├── bugfix.md
│   ├── release.md
│   ├── backtest.md        # project-specific
│   ├── dashboard.md       # project-specific
│   └── ...
├── hooks/                 # 5 automation hooks
│   ├── phase-progress.sh
│   ├── pre-tool-use-safety.sh
│   └── ...
├── analytics/             # Usage metrics (JSONL)
├── scripts/               # CLI tools
│   └── analytics-visualizer.sh
└── docs/                  # Framework documentation
```

## Key Components

| Category | Count | Purpose |
|----------|-------|---------|
| Agents | 26 | 전문화된 작업 수행 |
| Skills | 28+ | 워크플로우 자동화 (Codex 듀얼 AI 포함) |
| Commands | 11 | 통합 개발 플로우 (프로젝트 전용 포함) |
| Hooks | 5 | 자동 트리거 작업 |

## Core Workflows

### Feature Development
```
/feature start → 개발 → /feature progress → /feature complete
```

### Sprint Execution
```
/sprint start --phase N → Task 선택 → /sprint complete → Phase 업데이트
```

## Quick Reference

### Primary Skills

| Skill | Usage |
|-------|-------|
| `/validate` | 설정 검증 |
| `/phase` | Phase 관리 |
| `/sprint` | Sprint 관리 |
| `/agile-sync` | 문서 동기화 |
| `/analytics` | 사용 통계 시각화 |
| `/gh` | GitHub CLI 통합 |
| `/codex` | Codex CLI 실행 |

### Project-Specific Commands

| Command | Usage |
|---------|-------|
| `/backtest` | 백테스트 실행 |
| `/dashboard` | Dash 대시보드 실행 |
| `/test` | pytest 테스트 실행 |
| `/uv-sync` | 의존성 동기화 |
| `/uv-run` | uv run 명령어 |

### Key Agents

| Agent | Purpose |
|-------|---------|
| `progress-tracker` | 전체 진행률 관리 |
| `phase-tracker` | Phase별 상세 추적 |
| `dev-docs-writer` | 개발 문서 생성 |
| `analytics-reporter` | 사용 통계 및 리포트 |
| `github-manager` | GitHub 통합 |
| `readme-helper` | README 작성 및 개선 |
| `agent-writer` | Agent 작성 및 검증 |

## Configuration

핵심 설정: `.claude/settings.json`

```json
{
  "phase": { "enabled": true },
  "sprint": { "enabled": true, "phase_integration": { "enabled": true } },
  "sync": {
    "enabled": true,
    "framework_source": "ultra-cc-init",
    "preserve_project_customizations": true
  },
  "analytics": { "enabled": true },
  "github": { "enabled": true }
}
```

## Links

- [Document Structure](.claude/docs/DOCUMENT-STRUCTURE.md)
- [Sprint-Phase Integration](.claude/docs/SPRINT-PHASE-INTEGRATION.md)
- [Settings](.claude/settings.json)
