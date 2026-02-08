# Car Price Prediction - MLOps Project

## Project description

This project is part of the MLOps course final assessment.
The goal is to design and implement an end-to-end machine learning pipeline following MLOps best practices, including:
- Reproducibility
- Version control
- Code quality and testing
- Experiment tracking
- Deployment

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

## Task definition

The objective is to build a supervised learning regression model predicting car prices.

The model will:
- Load and preprocess the dataset
- Engineer simple features (e.g. car age)
- Encode categorical variables
- Train a RandomForestRegressor with hyperparameters
- Evaluate performance using MAE, RMSE and R² score
- Log experiments to MLflow with clear naming conventions

The task and model complexity will remain simple initially, with the goal of emphasizing engineering quality and reproducibility rather than model performance.

The final task definition may evolve during the project.

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
    ├── processed/
    ├── raw/
├── src/            # Source code
    ├── data/
    ├── models/
├── tests/          # Unit tests
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
```

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
