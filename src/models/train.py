"""
Training script for the Car Price ML model with RandomForest and MLflow.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_processed_data():
    """
    Load the preprocessed CSV dataset.
    """
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "car_price_cleaned.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Please run preprocessing first."
        )

    return pd.read_csv(data_path)


def train_model(
    max_year: int | None = None,
    n_estimators: int = 150,
    max_depth: int = 15,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
):  # pragma : no cover
    """
    Train a Random Forest model and track experiment with MLflow.
    """

    # Set MLflow experiment
    mlflow.set_experiment("car-price-experiment")

    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_year", max_year)

        # Load data
        df = load_processed_data()

        # Filter by year if needed
        if max_year is not None:
            df = df[df["Year"] <= max_year]

        # Create new feature: Age of the car
        current_year = df["Year"].max()
        df["Car_Age"] = current_year - df["Year"]

        # Log-transform target
        df["Price_log"] = np.log1p(df["Price"])
        df["Mileage_log"] = np.log1p(df["Mileage"])

        # Separate features and target
        labels = df["Price_log"]
        features = df.drop(
            columns=["Price", "Price_log", "Car ID", "Year"]
        )  # drop original price & ID

        # Identify categorical and numerical features
        categorical_features = features.select_dtypes(
            include=["object"]
        ).columns
        numerical_features = features.select_dtypes(exclude=["object"]).columns

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
                ("num", "passthrough", numerical_features),
            ]
        )

        # RandomForest model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

        # Full pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )

        # Train/test split
        x_train_features, x_test_features, y_train_labels, y_test_labels = (
            train_test_split(features, labels, test_size=0.2, random_state=42)
        )

        # Train model
        pipeline.fit(x_train_features, y_train_labels)

        # Predict and inverse log-transform
        y_pred_log = pipeline.predict(x_test_features)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test_labels)

        # Compute metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="car-price-model",
        )

        # Print results
        print("=" * 40)
        print("Training completed successfully")
        if max_year:
            print(f"Data used: Year <= {max_year}")
        print(f"Number of trees: {n_estimators}")
        print(f"Max depth: {max_depth}")
        print(f"Min samples leaf: {min_samples_leaf}")
        print(f"Max features: {max_features}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R² score: {r2:.3f}")
        print("=" * 40)


if __name__ == "__main__":  # pragma : no cover
    parser = argparse.ArgumentParser(
        description="Train Car Price Model with RandomForest"
    )
    parser.add_argument(
        "--max-year", type=int, default=2010, help="Use data up to this year"
    )
    parser.add_argument(
        "--n-trees", type=int, default=150, help="Number of trees"
    )
    parser.add_argument(
        "--max-depth", type=int, default=15, help="Maximum tree depth"
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples per leaf",
    )
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        help="Max features for splits",
    )

    args = parser.parse_args()
    train_model(
        max_year=args.max_year,
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    )
