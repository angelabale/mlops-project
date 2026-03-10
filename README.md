# Car Price Prediction - MLOps Project

## Project description

This project is part of the MLOps course final assessment.
The goal is to design and implement an end-to-end machine learning pipeline following MLOps best practices, including:
- Reproducibility
- Version control
- Code quality and testing
- Experiment tracking
- CI/CD automation
- Containerization
- Model serving with FastAPI
- Monitoring and drift detection

The emphasis of this project is engineering quality and reliability rather than model complexity.

### Checkpoint 1 (Completed)
- Set up a clean project structure
- Managed the Python environment with UV
- Implemented basic data loading and preprocessing
- Ran a first baseline training pipeline

### Checkpoint 2 (Completed)
- Implement pre-commit hooks for code quality
- Write unit tests covering ≥60% of the code
- Track experiments with MLflow (parameters, metrics, model artifacts)
- Maintain a clear and reproducible project structure

### Checkpoint 3 (Completed)
- FastAPI application exposing prediction endpoint
- Clear request/response schema
- Dockerized application (training + inference)
- CI pipeline including lint, tests, coverage and Docker build
- Application runnable via Docker container
- Basic API tests implemented

### Checkpoint 4 (Completed)
- Streamlit monitoring dashboard (prediction, experiment history, data drift, model lifecycle)
- Data drift detection using KS test on production vs reference data
- Reference data saved as MLflow artifact on each training run
- Production predictions logged to MLflow
- Final project report
- Demo video

## Task definition

We aim to build a supervised regression model that predicts car prices based on structured features.

## Machine Learning Pipeline

The pipeline:

1. Loads and preprocesses raw data
2. Engineers features (e.g., car age)
3. Encodes categorical variables
4. Trains a `RandomForestRegressor`
5. Evaluates performance using:
   - MAE
   - RMSE
   - R²
6. Logs experiments to MLflow:
   - Parameters
   - Metrics
   - Model artifacts
7. Saves reference data as MLflow artifact for drift monitoring

---

## Data Source

The dataset used in this project comes from:
**https://www.kaggle.com/datasets/nalisha/car-price-prediction-dataset/data**

A brief description of the dataset:
- Number of samples: 2500
- Features:
    - Car ID
    - Brand
    - Year
    - Engine Size
    - Fuel Type
    - Transmission
    - Mileage
    - Condition
    - Model
- Target variable: Price

## Project structure


```text
mlops-project/
├── .github/workflows/
│   └── ci.yml
├── artifacts/
│   └── reference_data.csv
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── app/
│   │   ├── api.py
│   │   └── streamlit_app.py
│   ├── data/
│   └── models/
│       └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_load_data.py
│   ├── test_preprocess.py
│   └── test_train.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## System Architecture

The system follows a modular MLOps architecture:

1. **Data Layer** — Raw data in `data/raw`, processed data in `data/processed`
2. **Training Layer** — Preprocessing, training, MLflow experiment tracking, reference data export
3. **Serving Layer** — FastAPI REST API for inference, model loaded from `model.joblib`
4. **Monitoring Layer** — Streamlit dashboard for drift detection, experiment comparison, model lifecycle
5. **Infrastructure Layer** — Docker containerization, GitHub Actions CI pipeline

## MLOps Practices
This project integrates the following MLOps practices:

- **UV**: Python environment and dependency management, ensuring reproducibility across all machines
- **Pre-commit hooks**: Automated code quality checks (linting, formatting) before each commit
- **MLflow**: Experiment tracking with logged parameters, metrics and model artifacts. Each run is named and comparable via the MLflow UI
- **Docker**: The full application (training + inference) is containerized for portability and reproducibility
- **GitHub Actions CI**: Automated pipeline triggered on each push — runs lint, tests, coverage check and Docker build
- **Streamlit**: Interactive dashboard for model validation, experiment history, drift monitoring and model lifecycle management

## Monitoring & Reliability

### API monitoring endpoints

- `GET /health` — Returns the API status and confirms the model is loaded
- `GET /metrics` — Returns runtime statistics:
  - Total number of requests received
  - Total number of successful predictions
  - Total number of errors
  - Average prediction latency (in seconds)

Application logs are handled via Python's `logging` module. All prediction requests and errors are logged at INFO/ERROR level.

### Streamlit monitoring dashboard

The dashboard runs on `http://127.0.0.1:8501` and exposes four pages:

- **Prediction** — Interactive form to test the model and log predictions to MLflow
- **Experiment History** — Table and chart of all MLflow runs with metrics comparison
- **Data Drift** — KS test comparing reference data (from training) vs production data (from predictions). Flags features where p-value < 0.05
- **Model Lifecycle** — Overview of all registered model versions, their stage (Production/Staging/None) and metrics

### Drift detection strategy

On each training run, `x_train_features` is saved as `artifacts/reference_data.csv` and logged as an MLflow artifact. Each prediction made via the Streamlit app is appended to `artifacts/production_samples.csv`. The drift page compares these two distributions using the Kolmogorov-Smirnov test. A p-value below 0.05 on any feature indicates drift and would trigger retraining.

## Limitations & Future Work

**Current limitations:**
- The dataset is small (2,500 samples), which limits model generalization
- Monitoring is in-memory only for the API — metrics reset on each container restart
- Drift detection requires manual review, no automatic retraining trigger yet
- No authentication or rate limiting on the API

**Future work:**
- Automate retraining pipeline triggered by drift detection
- Integrate Grafana for persistent dashboards
- Expand the dataset and experiment with more advanced models
- Add input validation and more robust error handling
- Schedule periodic retraining with GitHub Actions or Airflow

## Team Collaboration

Collaboration was managed through GitHub using branches and pull requests, with each PR reviewed before merging into main. Tasks were distributed across the team throughout the project, with each member contributing to different parts of the pipeline — from data processing and model training to API development, containerization and monitoring. All team members stayed involved in the overall project and maintained a shared understanding of every component.

Git commit history reflects active and balanced participation from all contributors.


## How to Run

### 1. Synchronize environment
```bash
uv sync
```

### 2. Preprocessing
```bash
uv run python -m src.data.preprocess
```

### 3. Train the model
- Default parameters (max_year = 2010, n_estimator = 150, max_depth = 15, min_samples_leaf = 5, max_features = 'sqrt')
```bash
# Default parameters
make train
# or directly
uv run python src/models/train.py

# Custom parameters
uv run python src/models/train.py --max-year 2018 --n-trees 200 --max-depth 20 --min-samples-leaf 3 --max-features log2
```

### 4. MLflow tracking
```bash
mlflow ui
```
Open: http://127.0.0.1:5000

### 5. Run Streamlit dashboard
```bash
uv run streamlit run src/app/streamlit_app.py
```
Open: http://127.0.0.1:8501

### 6. Run tests
```bash
make test
make coverage
```

### 7. Lint and formatting
```bash
make lint
```

### 8. Run pre-commit hooks
```bash
make precommit
```

## Model Serving (FastAPI)

The trained model is served through a FastAPI application exposing a `/predict` endpoint.

### Run API locally

```bash
uv run uvicorn src.app.api:app --reload --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs

### Example request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Brand": "Toyota",
    "Engine Size": 2.0,
    "Fuel Type": "Petrol",
    "Transmission": "Manual",
    "Mileage": 50000,
    "Condition": "Good",
    "Model": "Corolla",
    "Year": 2015
  }'
```

## Run with Docker Compose

```bash
docker compose up --build
```


## Video Link
