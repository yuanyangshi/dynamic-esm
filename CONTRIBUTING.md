# Contributing to Dynamic-ESM

Thank you for your interest in contributing to **Dynamic-ESM**! We welcome community contributions, ranging from bug reports and documentation enhancements to novel physical modeling operators and algorithmic extensions.

---

## 1. Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all members of the community with respect and professionalism.

---

## 2. Development Setup

### 2.1 Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8+ recommended for GPU acceleration)
- Git

### 2.2 Local Installation
Clone the repository and set up a development environment:

```bash
git clone https://github.com/yuanyangshi/dynamic-esm.git
cd dynamic-esm

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 3. Engineering Guidelines & Coding Standards

### 3.1 All-English Repository Standard
- **100% English Mandate**: All source code, docstrings, variable names, comments, commit messages, PR descriptions, and issue discussions must be written strictly in standard English. Non-ASCII characters are prohibited in the codebase.
- **PEP 8 Compliance**: Adhere to PEP 8 standards. Use `black`, `flake8`, and `isort` for code formatting and linting.
- **Type Annotations**: Ensure complete PEP 484 type annotations for all public functions, classes, and methods.

### 3.2 Academic Rigor & Physical Authenticity
- Under no circumstances should synthetic, dummy, or toy mock data be used for reporting metrics or validating scientific claims.
- All evaluation metrics ($R_p, \rho_s$, RMSE, MAE, $CI$) must be computed strictly on genuine physical datasets or actual model inferences.
- When introducing new physical modeling features (e.g., equivariant geometric operators, SE(3) convolutions), provide rigorous mathematical formulations and references in the documentation.

### 3.3 Code Speaks, Comments Whisper
- Retain complete, authentic implementation details to guarantee exact reproducibility.
- Maintain neutral, objective, and de-highlighted docstrings describing mathematical and tensor operations. Avoid hyperbolic commentary or research trial claims.

---

## 4. Testing & Verification

Before submitting a pull request, run the complete automated test suite to ensure all unit tests pass without regression:

```bash
pytest tests -v
```

All 30+ unit tests across data, models, training, evaluation, interpretability, and visualization must pass.

---

## 5. Submitting Pull Requests

1. **Fork the Repository**: Create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Implement & Test**: Write your code and accompanying unit tests under `tests/`.
3. **Commit Cleanly**: Use clear, concise commit messages following standard conventions:
   ```bash
   git commit -m "feat(models): add equivariant coordinate regularizer"
   ```
4. **Push & Open PR**: Push to your fork and submit a Pull Request targeting `main`. Provide a descriptive overview of changes and test evidence in English.

---

## 6. Reporting Issues & Questions

If you encounter a bug or have a feature suggestion, please open a GitHub Issue with:
- A clear, concise title and description.
- A minimal reproducible example.
- Your OS, Python version, PyTorch version, and hardware specifications (CPU/GPU).
