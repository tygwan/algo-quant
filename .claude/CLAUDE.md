# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **cc-initializer**, a Claude Code configuration framework for managing AI-assisted development workflows. It provides skills, agents, hooks, and commands that orchestrate documentation, phase-based development, sprint tracking, and quality gates.

This is **not** a traditional application codebase—it's a framework configuration system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Skills (22)    Commands (6)    Agents (20)    Hooks (5)       │
│       └─────────────────────┬──────────────────┘               │
│                             ▼                                   │
│                       settings.json                             │
│                             ▼                                   │
│                     docs/ (standards)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Primary Commands

| Command | Purpose |
|---------|---------|
| `/init [--full\|--quick]` | Initialize/analyze project, create CLAUDE.md |
| `/feature <start\|progress\|complete>` | Feature development workflow |
| `/bugfix <start\|analyze\|complete>` | Bug fix workflow with root cause analysis |
| `/release <prepare\|create\|publish>` | Release management |
| `/phase <status\|update>` | Phase tracking and management |
| `/sprint <add\|start\|close>` | Sprint lifecycle management |
| `/validate [--full\|--fix]` | Configuration validation |
| `/quality-gate` | Pre-commit/merge/release checks |
| `/agile-sync` | Synchronize CHANGELOG, README, progress docs |

## Key Configuration

**settings.json** - Central configuration controlling:
- Hook definitions (PreToolUse, PostToolUse, Notification)
- Phase/Sprint integration settings
- Quality gate thresholds
- Document standard locations
- Safety settings (dangerous command blocking)

## Document Structure Standard

```
docs/
├── PRD.md           # Product requirements
├── TECH-SPEC.md     # Technical specification
├── PROGRESS.md      # Integrated progress tracking
├── CONTEXT.md       # AI context summary
├── phases/          # Phase-based development
│   └── phase-N/
│       ├── SPEC.md
│       ├── TASKS.md
│       └── CHECKLIST.md
├── sprints/         # Sprint tracking (optional)
└── adr/             # Architecture Decision Records
```

## Hook System

Hooks are shell scripts in `hooks/` triggered by tool events:

- **pre-tool-use-safety.sh**: Blocks dangerous commands (rm -rf /, force push to main, DROP TABLE)
- **phase-progress.sh**: Auto-updates PROGRESS.md when TASKS.md changes
- **auto-doc-sync.sh**: Syncs README stats on .claude/ changes
- **post-tool-use-tracker.sh**: Logs tool usage

## Agent System

Agents are specialized AI assistants defined in `agents/`. Key agents:
- **progress-tracker**: Unified progress tracking with Phase system
- **phase-tracker**: Phase-specific task and checklist management
- **dev-docs-writer**: Auto-generates PRD, TECH-SPEC, PROGRESS
- **branch-manager**: GitHub Flow branch management
- **commit-helper**: Conventional Commits message generation
- **pr-creator**: PR creation with templates

## Workflow Integration

The `/feature` workflow demonstrates how components integrate:

```
/feature start "feature name"
    └── branch-manager → Git branch
    └── phase-tracker → Link to Phase task
    └── /sprint add → Add to current sprint
    └── progress-tracker → Update PROGRESS.md
    └── context-optimizer → Load relevant context

/feature complete
    └── quality-gate → Lint, test, coverage
    └── phase-tracker → Mark task complete
    └── commit-helper → Generate commit message
    └── pr-creator → Create GitHub PR
    └── agile-sync → Update CHANGELOG
```

## Validation

Run `/validate --full` to verify configuration integrity:
- settings.json syntax and required sections
- Hook file existence and permissions
- Agent/Skill frontmatter validity
- Document structure compliance
