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

The emphasis of this project is engineering quality and reliability rather than model complexity.

### Checkpoint 1 (Completed)
- Set up a clean project structure
- Managed the Python environment with UV
- Implemented basic data loading and preprocessing
- Ran a first baseline training pipeline

### Checkpoint 2 (Current)
- Implement pre-commit hooks for code quality
- Write unit tests covering ≥60% of the code
- Track experiments with MLflow (parameters, metrics, model artifacts)
- Maintain a clear and reproducible project structure

### Checkpoint 3 (Completed): 
- FastAPI application exposing prediction endpoint
- Clear request/response schema
- Dockerized application (training + inference)
- CI pipeline including lint, tests, coverage and Docker build
- Application runnable via Docker container
- Basic API tests implemented

### Checkpoint 4 (Ongoing): 
- Monitoring
- final report
- demo video

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
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── data/
│   ├── models/
│   └── api/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── uv.lock
├── .github/workflows/ci.yml
└── README.md
```

## System Architecture

The system follows a modular MLOps architecture:

1. Data Layer
   - Raw data stored in `data/raw`
   - Processed data stored in `data/processed`

2. Training Layer
   - Preprocessing module
   - Training module
   - MLflow experiment tracking

3. Serving Layer
   - FastAPI application
   - Model loaded from saved artifact
   - REST API endpoint for inference

4. Infrastructure Layer
   - Docker containerization
   - GitHub Actions CI pipeline
   - Automated build validation


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
make train
# or directly
uv run python src/models/train.py
```

- Custom parameters
```bash
uv run python src/models/train.py --max-year 2018 --n-trees 200 --max-depth 20 --min-samples-leaf 3 --max-features log2
```

### 4. MLflow tracking
```bash
mlflow ui
```
Open your browser at: http://127.0.0.1:5000/ to view experiment metrics, parameters, and models.

### 5. Run tests
```bash
make test
make coverage
```

### 6. Lint and formatting
```bash
make lint
```

### 7. Run pre-commit hooks
```bash
make precommit
```

## Model Serving (FastAPI)

The trained model is served through a FastAPI application exposing a `/predict` endpoint.

### Run API locally

```bash
uv run uvicorn src.app.api:app --reload --port 8000

Swagger UI (API documentation):

http://127.0.0.1:8000/docs
```

## Run with Docker Compose

Build and start the application:

```bash
docker compose up --build
```
