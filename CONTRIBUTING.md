# Contributing

Thank you for considering contributing to the Clinical Trial Cohort Matching System!

## Getting Started

1. Fork the repository and clone locally.
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `make install`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Workflow

```bash
make lint       # Run ruff linter
make test       # Run the full test suite
make run        # Start the API server locally
```

## Code Standards

- All functions must have type annotations.
- All public classes and functions must have Google-style docstrings.
- Use `logging.getLogger(__name__)` instead of `print()`.
- Keep functions under 40 lines; extract helpers if needed.
- Run `make lint-fix` before committing.

## Pull Request Guidelines

- One logical change per PR.
- Include tests for all new functionality.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Ensure CI passes before requesting review.

## Reporting Issues

Open an issue with:
- A clear description of the problem.
- Steps to reproduce.
- Expected vs actual behaviour.
- Python version and OS.

## Security

Please report security vulnerabilities privately — see `SECURITY.md`.
