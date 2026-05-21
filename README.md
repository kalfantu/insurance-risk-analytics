# Insurance Risk Analytics

A data-driven project to assess insurance risk and profitability for AlphaCare Insurance Solutions (ACIS), using historical motor vehicle insurance data from South Africa (Feb 2014 – Aug 2015).

## Business Objective

Optimize marketing strategy and premium pricing by:
- Identifying low-risk customer segments for reduced premiums
- Discovering high-risk segments and geographic patterns
- Building predictive models for claim likelihood and severity

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/   # CI/CD pipeline
├── data/                # Raw data (managed by DVC, not committed to git)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py   # Data ingestion utilities
│   ├── eda_utils.py     # EDA helper functions
│   ├── hypothesis_tests.py
│   └── modeling.py
├── tests/               # Unit tests (pytest)
├── reports/             # Generated reports and figures
├── requirements.txt
└── dvc.yaml             # DVC pipeline definition
```

## Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/KalkidanAsfaw/insurance-risk-analytics.git
cd insurance-risk-analytics
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

Raw data is tracked with [DVC](https://dvc.org/). Place the source CSV (`MachineLearningRating_v3.txt`) inside `data/raw/` before running notebooks.

```bash
# If a DVC remote is configured:
dvc pull
```

## Usage

Run notebooks in order:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Or import utilities directly:

```python
from src.data_loader import load_data, summarize_data
from src.eda_utils import plot_loss_ratio_by_province

df = load_data("data/raw/MachineLearningRating_v3.txt")
```

## Running Tests

```bash
pytest tests/ -v
```

## CI

GitHub Actions runs linting (`flake8`) and tests (`pytest`) on every push and pull request. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Tasks

| Task | Branch | Description |
|------|--------|-------------|
| Task 1 | `task-1` | EDA & data quality assessment |
| Task 2 | `task-2` | Hypothesis testing |
| Task 3 | `task-3` | Predictive modeling |

## License

MIT
