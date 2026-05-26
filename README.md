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
│   ├── prepare_data.py  # DVC pipeline stages (raw → cleaned)
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
git clone https://github.com/kalfantu/insurance-risk-analytics.git
cd insurance-risk-analytics
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data Pipeline

Data is versioned with [DVC](https://dvc.org/). Two dataset versions are tracked:

- **`data/MachineLearningRating_v3.txt`** — the original source file, tracked via `data/MachineLearningRating_v3.txt.dvc` committed to Git.
- **`data/insurance_data_cleaned.csv`** — the analysis-ready cleaned dataset, tracked as a pipeline output in `dvc.yaml` and locked in `dvc.lock`.

| Version | File | Description |
|---|---|---|
| Raw | `data/insurance_data.csv` | Source file converted to standard CSV (1,000,098 rows × 52 cols) |
| Cleaned | `data/insurance_data_cleaned.csv` | 7 high-missing columns dropped, 4 imputed, 628 bad rows removed (999,470 rows × 45 cols) |

#### Reproduce the pipeline from scratch

```bash
# 1. Pull the tracked source file from DVC remote
dvc pull

# 2. Re-run all pipeline stages (raw → cleaned)
dvc repro

# 3. Push any new outputs back to the remote
dvc push
```

#### Pull data only (no reprocessing)

```bash
dvc pull
```

#### Pipeline stages (`dvc.yaml`)

| Stage | Command | Input | Output |
|---|---|---|---|
| `prepare_raw` | `src/prepare_data.py --stage raw` | `MachineLearningRating_v3.txt` | `insurance_data.csv` |
| `clean` | `src/prepare_data.py --stage clean` | `insurance_data.csv` | `insurance_data_cleaned.csv` |

The cleaning strategy applied in the `clean` stage:
- **Dropped** 7 columns with >50% missing values (`CrossBorder`, `NewVehicle`, `WrittenOff`, `Rebuilt`, `Converted`, `NumberOfVehiclesInFleet`, `CustomValueEstimate`)
- **Imputed** `Bank`, `AccountType`, `Gender`, `MaritalStatus` → `"Not specified"`
- **Dropped** 628 rows missing vehicle attributes or `CapitalOutstanding`
- **Removed** exact duplicate rows

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
