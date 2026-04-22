# CLAUDE.md — DNA (Deep Neural Acceleration)

This file documents the repository structure, conventions, and development workflow for AI assistants (and human contributors) working on this codebase.

---

## Project Overview

| Field   | Value                        |
|---------|------------------------------|
| Name    | DNA — Deep Neural Acceleration |
| License | GNU General Public License v3 |
| Status  | Skeleton — no source code yet |

The project name suggests a focus on accelerating deep neural network workloads, but no implementation exists as of this writing. Do not assume or hallucinate existing code paths, APIs, or dependencies.

---

## Current Repository Contents

```
DNA/
├── CLAUDE.md      # This file
├── LICENSE        # GNU GPL v3
└── README.md      # Two-line stub (title + tagline)
```

No source code, tests, build scripts, dependencies, or CI/CD pipelines have been added yet.

---

## Expected Future Structure

When implementation begins, follow this conventional layout (adjust to match the chosen tech stack):

```
DNA/
├── src/           # Primary source code
├── tests/         # Unit and integration tests
├── docs/          # Extended documentation
├── scripts/       # Utility / automation scripts
├── .github/
│   └── workflows/ # CI/CD pipelines
├── CLAUDE.md
├── LICENSE
└── README.md
```

Update this section and the rest of CLAUDE.md as the actual structure solidifies.

---

## Development Workflow

### Branches

- **`master`** — stable, production-ready commits only
- **Feature branches** — named `<namespace>/<short-description>-<id>` (e.g. `claude/add-claude-documentation-Cj2cU`)
- Never commit directly to `master` without review

### Commits

- Write short, imperative commit messages: `Add training loop`, `Fix memory leak in loader`
- Keep commits focused; one logical change per commit
- Reference issue numbers when applicable: `Fix gradient overflow (#42)`

### Build & Test

No build system or test framework is configured yet. When one is established:

1. Add run commands here (e.g. `make build`, `pytest`, `cargo test`)
2. Document required environment variables or secrets
3. Note any pre-commit hooks or linting requirements

---

## Key Conventions for AI Assistants

- **No code exists yet.** Do not reference, import, or depend on files that are not present in the repository. Verify with `ls` or `find` before assuming a file exists.
- **License compliance.** All contributed code must be compatible with GNU GPL v3. Avoid copying code from permissively-licensed sources without confirming GPL compatibility.
- **Keep CLAUDE.md current.** Whenever significant new patterns, dependencies, or conventions are introduced (new framework, test runner, env vars, etc.), update the relevant section of this file in the same commit.
- **Prefer editing existing files** over creating new ones when adding small changes.
- **No speculative abstractions.** Implement only what is explicitly requested; do not add helper utilities, wrapper classes, or future-proofing code unless asked.

---

## Testing

No test framework has been configured. When tests are added:

- Document the framework and how to run the suite here
- Place tests under `tests/` mirroring the source structure
- Aim for unit tests on core logic and integration tests on I/O boundaries

---

## CI/CD

No CI/CD pipelines exist yet. When added, document:

- Pipeline trigger conditions (push, PR, schedule)
- Required secrets or environment variables
- How to reproduce a CI failure locally
