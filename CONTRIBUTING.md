# Contributing to NexQuant

We welcome contributions and suggestions to improve NexQuant. Whether it's solving an issue, addressing a bug, enhancing documentation, or even correcting a typo, every contribution is valuable and helps improve the project.

## Getting Started

To get started, you can explore the issues list or search for `TODO:` comments in the codebase by running:
```sh
grep -r "TODO:"
```

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/NexQuant.git
cd NexQuant

# Add upstream remote
git remote add upstream https://github.com/TPTBusiness/NexQuant.git
```

### 2. Create a Branch

```bash
# Use conventional commit prefixes in branch names
git checkout -b feat/your-feature-name
# or
git checkout -b fix/bug-description
git checkout -b docs/documentation-update
git checkout -b refactor/code-cleanup
```

**Branch naming convention:**
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes
- `chore/` - Maintenance tasks

### 3. Make Your Changes

Follow the project conventions:

- **Code style**: Use type hints, docstrings (Google style), and 120 char line limit
- **Language**: All comments and documentation MUST be in English
- **Structure**: Follow the existing module structure

### 4. Write Tests

**MANDATORY:** All new features MUST have tests with >80% coverage.

```bash
# Run tests
pytest test/ -v

# Run with coverage
pytest --cov=rdagent --cov-report=html

# Run integration tests
pytest test/integration/ -v
```

### 5. Run Pre-commit Hooks

Pre-commit hooks run automatically before EVERY commit:

```bash
# Install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### 6. Commit Your Changes

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
git commit -m "type: description"

# Types:
# feat:     New feature
# fix:      Bug fix
# docs:     Documentation
# style:    Formatting
# refactor: Code restructuring
# test:     Tests
# chore:    Maintenance
```

**Examples:**
```bash
git commit -m "feat: Add Optuna hyperparameter optimization"
git commit -m "fix: Resolve database connection timeout"
git commit -m "docs: Update README with new CLI commands"
git commit -m "test: Add integration tests for portfolio optimizer"
```

### 7. Push and Create a Pull Request

```bash
git push origin your-branch-name
```

Then open a Pull Request on GitHub with:
- Clear title (use conventional commit format)
- Description of changes
- Link to related issues
- Screenshots (for UI changes)

## Code Review Process

All PRs are reviewed by maintainers. Expect:
- Automated checks (tests, linting, security scan)
- Code review by maintainers
- Possible requested changes

## Important Rules

### 🚫 NEVER COMMIT

- `.env` files or API keys
- Generated data (`results/`, `*.db`, `*.log`)
- Closed-source assets (`models/local/`, `prompts/local/`)
- JSON strategy files in root directory
- Private credentials or tokens

### ✅ ALWAYS DO

- Write tests for new features
- Update documentation for user-visible changes
- Run `pre-commit run --all-files` before pushing
- Keep commit messages in English
- Follow conventional commit format

## Project Structure

```
NexQuant/
├── rdagent/                 # Core framework (open source)
│   ├── app/                 # CLI and scenario apps
│   ├── components/          # Reusable agent components
│   └── scenarios/           # Domain-specific scenarios
├── test/                    # Test suite
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── prompts/                 # LLM prompts
├── models/                  # ML models (standard only)
├── constraints/             # Python version constraints
└── requirements/            # Dependency files
```

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/TPTBusiness/NexQuant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TPTBusiness/NexQuant/discussions)
- **Documentation**: See `docs/` folder

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
