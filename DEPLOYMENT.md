# Deployment & Release Guide

This document provides comprehensive instructions for testing, deploying, and rolling back changes to the Hybrid Tokenizer project.

## Table of Contents

1. [Local Testing](#local-testing)
2. [Running on Kaggle](#running-on-kaggle)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Release Checklist](#release-checklist)
5. [Deployment Procedures](#deployment-procedures)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Rollback Procedures](#rollback-procedures)

---

## Local Testing

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip and virtualenv
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/sh20022002/small-Language-Model.git
cd small-Language-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make install-dev
```

### Running Tests Locally

#### Quick Test Run
```bash
# Run all tests with verbose output
make test

# Run specific test file
pytest tests/test_model.py -v

# Run specific test class
pytest tests/test_model.py::TestForwardPass -v

# Run specific test method
pytest tests/test_model.py::TestForwardPass::test_output_shape -v
```

#### Coverage Report (80% threshold)
```bash
# Generate coverage report (fails if below 80%)
make coverage

# View HTML coverage report
open htmlcov/index.html

# Run tests with coverage details
pytest tests/ --cov=src/my_slm --cov-report=term-missing
```

#### Testing on Multiple Python Versions
```bash
# Test on Python 3.9
python3.9 -m venv venv39
source venv39/bin/activate
pip install -e ".[dev]"
make test

# Repeat for 3.10 and 3.11
```

### Code Quality Checks

```bash
# Run all quality checks
make lint typecheck

# Format code automatically
make format

# Run full CI locally (test + lint + typecheck + notebook-check)
make ci
```

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit:

```bash
# Install pre-commit hooks
make pre-commit

# Run hooks manually on all files
pre-commit run --all-files

# Skip pre-commit (only when necessary)
git commit --no-verify
```

---

## Running on Kaggle

### Before Uploading

1. **Validate notebook structure:**
   ```bash
   make notebook-check
   ```

2. **Test imports in Kaggle environment simulation:**
   ```bash
   python scripts/validate_imports.py
   ```

3. **Check for hard-coded paths:**
   ```bash
   python scripts/check_notebook_paths.py
   ```

### Uploading to Kaggle

1. **Export notebook:**
   - Ensure all cells execute without errors locally
   - Remove debug cells and personal configurations
   - Validate JSON structure: `make notebook-check`

2. **Kaggle Notebook Steps:**
   - Go to https://www.kaggle.com/settings/account
   - Create/update your Kaggle API token (kaggle.json)
   - Use Kaggle CLI or web interface to push notebook

3. **Set Kaggle Metadata:**
   ```json
   {
     "accelerator": "nvidiaTeslaT4",
     "isInternetEnabled": true,
     "isGpuEnabled": true
   }
   ```

4. **Environment Setup in Kaggle:**
   ```python
   # First cell: install/clone latest code
   !git clone https://github.com/sh20022002/small-Language-Model.git
   %cd small-Language-Model
   !pip install -e .
   ```

### Testing Kaggle Notebook Locally

The GitHub Actions notebook workflow includes simulated Kaggle environment testing:

```bash
# Simulate Kaggle input directory structure
mkdir -p /tmp/kaggle_input/example_model_instance

# Run notebook validation
python scripts/test_notebook_cells.py \
  --notebook kaggle_dual_gpu_finetune.ipynb \
  --output /tmp/extracted_cells.py

# Test extracted cells
python /tmp/extracted_cells.py
```

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. CI Pipeline (.github/workflows/ci.yml)

**Triggers:** Push to main/feature/* branches, pull requests

**Jobs:**
- **test** (Python 3.9, 3.10, 3.11):
  - Install dependencies
  - Run pytest
  - Measure coverage (80% threshold)
  - Upload coverage reports

- **lint** (Python 3.11):
  - Black format check
  - isort import sorting check
  - Flake8 linting
  - Pylint analysis

- **typecheck** (Python 3.11):
  - mypy type checking with strict settings

- **build** (Python 3.11):
  - Build distribution package
  - Validate with twine

#### 2. Notebook Validation (.github/workflows/notebook.yml)

**Triggers:** Changes to *.ipynb, src/**, or tests/**

**Jobs:**
- **validate-notebook-structure**:
  - Validate JSON schema
  - Check notebook format

- **test-kaggle-notebook**:
  - Parse notebook cells
  - Verify imports
  - Check paths

### Monitoring CI/CD

```bash
# View workflow runs
gh workflow list
gh workflow view ci.yml

# View specific run
gh run list --workflow=ci.yml
gh run view <run-id> --log

# Re-run failed workflow
gh run rerun <run-id>
```

---

## Release Checklist

Use this checklist before releasing a new version:

### Code Quality (Automated)
- [ ] All tests pass (pytest tests/ -v)
- [ ] Coverage >= 80% (make coverage)
- [ ] No lint errors (make lint)
- [ ] Type checks pass (make typecheck)
- [ ] Notebook structure valid (make notebook-check)

### Code Review
- [ ] Pull request reviewed and approved
- [ ] No blocking comments
- [ ] Changes documented in FIXES.md or similar

### Documentation
- [ ] README.md updated (if needed)
- [ ] Docstrings added/updated
- [ ] DEPLOYMENT.md updated (if process changed)
- [ ] Changelog entry created

### Testing
- [ ] Feature tested locally on Python 3.9, 3.10, 3.11
- [ ] Kaggle notebook tested
- [ ] Edge cases covered
- [ ] Performance impact assessed

### Deployment
- [ ] Version bumped (follows semantic versioning)
- [ ] pyproject.toml updated
- [ ] Git tag created: `git tag v<version>`
- [ ] Release notes prepared

### Post-Release
- [ ] Monitor error rates and performance metrics
- [ ] Check for reported issues
- [ ] Be prepared to rollback if critical issues arise

---

## Deployment Procedures

### Releasing to PyPI

```bash
# Ensure all checks pass
make ci

# Bump version in pyproject.toml
# Update version = "X.Y.Z"

# Create changelog entry
echo "## v<version>" >> CHANGELOG.md

# Build distributions
python -m build

# Validate packages
twine check dist/*

# Upload to TestPyPI (optional verification)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*

# Create Git tag
git tag v<version>
git push origin main v<version>
```

### Deploying to Kaggle

```bash
# Ensure notebook validates locally
make notebook-check

# Update code in notebook repository
# Option 1: Via Kaggle CLI
kaggle notebooks push -p ./kaggle_dual_gpu_finetune.ipynb -m "Description"

# Option 2: Via web interface
# 1. Go to https://www.kaggle.com/sh20022002/dual-gpu-finetune
# 2. Click "Edit Notebook"
# 3. Paste updated code
# 4. Save and publish
```

### Deploying to Production Model Server (if applicable)

```bash
# Export model checkpoint
python -c "
import torch
from my_slm.transformer import Transformer

model = Transformer(vocab_size=50257, dim=512, depth=8, heads=8)
torch.save(model.state_dict(), 'model_checkpoint.pt')
print('Model exported')
"

# Package for deployment
tar -czf model_checkpoint_v<version>.tar.gz model_checkpoint.pt
aws s3 cp model_checkpoint_v<version>.tar.gz s3://your-bucket/models/
```

---

## Monitoring & Alerts

### Health Checks

After deployment, monitor:

1. **Error Rates**: Watch for exceptions in logs
2. **Performance**: Check inference latency
3. **Resource Usage**: Monitor GPU/CPU utilization
4. **API Availability**: Ensure service endpoints respond

### Metrics to Track

```bash
# Test inference speed
python -c "
import torch
import time
from my_slm.transformer import Transformer

model = Transformer(vocab_size=256, dim=256, depth=6, heads=8)
ids = torch.randint(0, 256, (1, 100))

start = time.time()
for _ in range(100):
    _ = model(ids)
elapsed = time.time() - start
print(f'Inference speed: {elapsed/100*1000:.2f}ms per batch')
"
```

### Alerts Setup

Set up alerts for:
- Build failures
- Test coverage drops below 80%
- Type check failures
- Notebook validation failures

(Configure in GitHub Actions or your monitoring service)

---

## Rollback Procedures

### Immediate Rollback (Critical Issue)

If a release has a critical issue:

```bash
# 1. Identify last known good commit
git log --oneline | head -20

# 2. Revert the problematic commit
git revert <problematic-commit-hash>

# 3. Verify rollback locally
make test coverage lint typecheck

# 4. Push rollback
git commit -m "Rollback: <reason>"
git push origin main

# 5. Create new release with rollback
git tag v<version>-rollback
git push origin v<version>-rollback

# 6. Update PyPI (if necessary)
# Re-release with bumped patch version
```

### Partial Rollback (Revert Feature)

```bash
# If only one feature needs rollback:
git revert <feature-commit-hash>

# Resolve conflicts if any
git add .
git commit -m "Revert feature: <feature-name>"

# Test thoroughly before pushing
make ci

git push origin main
```

### Database/Model State Rollback

For Kaggle notebooks or persistent model state:

```bash
# Restore from backup
gsutil cp gs://your-bucket/backups/model_v<previous>.tar.gz .
tar -xzf model_v<previous>.tar.gz

# Update references to point to previous version
# In notebooks: MODEL_PATH = "gs://bucket/models/v<previous>"

# Verify restoration
python -c "
import torch
checkpoint = torch.load('model_checkpoint.pt')
print(f'Model loaded: {len(checkpoint)} parameters')
"

# Push updated configuration
git commit -am "Restore to model v<previous>"
git push origin main
```

### Verification After Rollback

```bash
# 1. Verify functionality
make test

# 2. Check inference works
python scripts/test_inference.py

# 3. Monitor error rates for 24+ hours
# Look for issues in logs/metrics dashboard

# 4. Communicate status to stakeholders
echo "Rollback complete. Monitoring for issues."

# 5. Post-mortem (if critical)
# Schedule meeting to discuss root cause
# Add regression tests to prevent recurrence
```

---

## Emergency Contacts

- **Primary Maintainer**: shmuel.tor@gmail.com
- **Issue Tracker**: https://github.com/sh20022002/small-Language-Model/issues

## Related Documentation

- [README.md](README.md) - Project overview
- [FIXES.md](FIXES.md) - Recent fixes and improvements
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - Pre-commit hook configuration
- [pyproject.toml](pyproject.toml) - Project metadata and dependencies
