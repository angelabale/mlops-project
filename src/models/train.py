import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load_processed_data():
    """
    Load the preprocessed CSV dataset.

    Returns:
        pd.DataFrame: DataFrame containing cleaned car price data.

    Raises:
        FileNotFoundError: If the preprocessed data file does not exist.
    """
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed" / "car_price_cleaned.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Please run preprocessing first."
        )

    return pd.read_csv(data_path)

def train_model(max_year: int | None = None, n_estimators: int = 50):  # pragma: no cover
    """
    Train a Random Forest model on the processed car price dataset.

    Args:
        max_year (int | None): If set, use only data up to this year.
        n_estimators (int): Number of trees in the Random Forest.

    Returns:
        None. Prints the Mean Absolute Error (MAE) on the test set.
    """

    # Load data
    df = load_processed_data()

    # Filter by year is max_year is provided
    if max_year is not None:
        df = df[df["Year"] <= max_year]

    # Separate target from features
    y = df["Price"]
    X = df.drop(columns=["Price", "Car ID"])

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Define the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    # Full pipeline: preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit model
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Print results
    print("=" * 40)
    print(" Training completed successfully")
    if max_year:
        print(f" Data used: Year <= {max_year}")
    print(f" Number of trees: {n_estimators}")
    print(f" Mean Absolute Error (MAE): {mae:.2f}")
    print("=" * 40)

if __name__ == "__main__":  # pragma: no cover
    # CLI argument parsing
    parser = argparse.ArgumentParser(description="Train Car Price Model")
    parser.add_argument("--max-year", type=int, default=2010, help="Use data up to this year")
    parser.add_argument("--n-trees", type=int, default=50, help="Number of trees in RandomForest")
    args = parser.parse_args()
    # Run training with specified parameters
    train_model(max_year=args.max_year, n_estimators=args.n_trees)
