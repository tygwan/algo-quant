---
name: init
description: Initialize and analyze a new project. Detects tech stack, structure, creates CLAUDE.md, and triggers development documentation chain. Use when starting work on any new codebase.
---

# Project Initialization Skill

## Usage
```
/init [path] [--full|--quick|--docs-only]
```

### Parameters
- `path`: Optional. Project root path (default: current directory)
- `--full`: Complete initialization with Phase structure
- `--quick`: Quick analysis, CLAUDE.md only
- `--docs-only`: Skip tech detection, create docs only

### Examples
```bash
/init
/init ./my-project
/init @./backend --full
```

## Workflow Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        /INIT WORKFLOW CHAIN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  /init                                                                       │
│    │                                                                         │
│    ├── Step 1: Structure Discovery                                           │
│    │     └── Detect tech stack, frameworks, patterns                        │
│    │                                                                         │
│    ├── Step 2: Generate CLAUDE.md                                           │
│    │     └── Project summary, commands, key files                           │
│    │                                                                         │
│    ├── Step 3: Trigger dev-docs-writer (--full mode)                        │
│    │     └── Create PRD.md, TECH-SPEC.md, PROGRESS.md, CONTEXT.md          │
│    │                                                                         │
│    ├── Step 4: Complexity Analysis                                           │
│    │     └── Evaluate project complexity                                     │
│    │                                                                         │
│    └── Step 5: Trigger doc-splitter (if HIGH complexity)                    │
│          └── Create Phase structure in docs/phases/                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Step 1: Structure Discovery

```bash
# Find root indicators
Glob: package.json, requirements.txt, *.csproj, go.mod, Cargo.toml, pom.xml

# Find source directories
Glob: src/**, lib/**, app/**

# Find config files
Glob: *.config.*, .env*, tsconfig.json, setup.py

# Find documentation
Glob: README*, docs/**, *.md
```

## Step 2: Tech Stack Detection

| File | Stack |
|------|-------|
| package.json | Node.js |
| tsconfig.json | TypeScript |
| requirements.txt | Python |
| *.csproj | .NET/C# |
| go.mod | Go |
| Cargo.toml | Rust |

### Framework Detection
```bash
# Web frameworks
Grep: "express|fastify|koa|nest" package.json
Grep: "flask|django|fastapi" requirements.txt
Grep: "gin|echo|fiber" go.mod

# Frontend frameworks
Grep: "react|vue|angular|svelte" package.json
Grep: "next|nuxt|gatsby" package.json
```

## Step 3: Trigger dev-docs-writer

When `--full` flag is used or project appears new:

```markdown
## Auto-Trigger Conditions

1. No docs/ folder exists
2. No CLAUDE.md exists
3. User specifies --full flag
4. Project complexity > LOW

## Triggered Actions

→ dev-docs-writer creates:
  - docs/PRD.md
  - docs/TECH-SPEC.md
  - docs/PROGRESS.md
  - docs/CONTEXT.md
```

## Step 4: Complexity Analysis

```yaml
Complexity Scoring:
  LOW (score < 3):
    - Single module
    - < 10 source files
    - No external dependencies

  MEDIUM (score 3-6):
    - Multiple modules
    - 10-50 source files
    - Some dependencies

  HIGH (score > 6):
    - Complex architecture
    - > 50 source files
    - Multiple frameworks
    - External integrations
```

## Step 5: Trigger doc-splitter (HIGH complexity)

When complexity is HIGH:

```markdown
## Auto-Trigger Conditions

- Complexity score > 6
- Multiple distinct features
- Estimated development > 1 week

## Triggered Actions

→ doc-splitter creates:
  docs/phases/
  ├── phase-1/
  │   ├── SPEC.md
  │   ├── TASKS.md
  │   └── CHECKLIST.md
  └── phase-N/
```

## Generate CLAUDE.md

```markdown
# {Project Name}

> **Type**: {Web App | API | CLI | Library | Plugin}
> **Stack**: {Language + Framework}
> **Docs**: [docs/_INDEX.md](docs/_INDEX.md)

## Quick Start

```bash
{install command}
{run command}
```

## Structure

```
{project structure}
```

## Key Files

| Purpose | File |
|---------|------|
| Entry | {entry_file} |
| Config | {config_file} |

## Development Status

See [docs/PROGRESS.md](docs/PROGRESS.md) for current phase and tasks.

## Skills Reference

| Skill | Usage |
|-------|-------|
| `/init` | Initialize/analyze project |
| `/phase` | Phase management |
| `/agile-sync` | Sync all documentation |
```

## Integration Points

### With dev-docs-writer
- Automatically triggered for new projects
- Creates standardized documentation structure

### With doc-splitter
- Triggered for complex projects
- Creates Phase-based development structure

### With phase-tracker
- Activated after Phase structure is created
- Begins tracking development progress

### With context-optimizer
- CONTEXT.md is created for token optimization
- Phase documents are structured for efficient loading

## Output

After running `/init --full`, you'll get:

1. **Console Output**: Project analysis summary
2. **CLAUDE.md**: Created/updated with project context
3. **docs/**: Full documentation structure
4. **docs/phases/**: Phase structure (if complex project)
5. **Activation**: Phase tracking system enabled

## Best Practices

- Run `/init` when starting on any new project
- Use `--full` for new development projects
- Use `--quick` for exploring existing codebases
- Re-run after major structural changes
