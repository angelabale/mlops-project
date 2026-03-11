"""
Streamlit app for Car Price Model — Validation, Monitoring & Drift Detection.
"""

import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
import streamlit as st

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Car Price ML Explorer", layout="wide")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

PROJECT_ROOT = Path(__file__).parent.parent.parent
REFERENCE_PATH = PROJECT_ROOT / "artifacts" / "reference_data.csv"
PRODUCTION_PATH = PROJECT_ROOT / "artifacts" / "production_samples.csv"
MODEL_PATH = PROJECT_ROOT / "model.joblib"


# ─────────────────────────────────────────────
# Load model via joblib (same as API)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model, "Model loaded from model.joblib"
    except Exception as e:
        return None, f"Could not load model: {e}"


model, model_status = load_model()

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Experiment History", "Data Drift", "Model Lifecycle"],
)

# ═════════════════════════════════════════════
# PAGE 1 — Prediction
# ═════════════════════════════════════════════
if page == "Prediction":
    st.title("Car Price Prediction")
    st.info(model_status)

    st.header("Car Information")
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox(
            "Brand",
            ["Toyota", "BMW", "Mercedes", "Audi", "Ford", "Honda", "Chevrolet", "Other"],
        )
        model_name = st.text_input("Model", value="Corolla")
        year = st.slider("Year", 1990, 2023, 2015)
        mileage = st.number_input(
            "Mileage (km)", min_value=0, max_value=500000, value=50000, step=1000
        )

    with col2:
        engine_size = st.slider("Engine Size (L)", 0.5, 6.0, 2.0, step=0.1)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        condition = st.selectbox("Condition", ["New", "Like New", "Good", "Fair", "Poor"])

    if st.button("Predict Price", type="primary"):
        if model is None:
            st.error("Model not loaded. Check that model.joblib exists.")
        else:
            input_df = pd.DataFrame([{
                "Brand": brand,
                "Engine Size": engine_size,
                "Fuel Type": fuel_type,
                "Transmission": transmission,
                "Mileage": mileage,
                "Condition": condition,
                "Model": model_name,
                "Year": year,
            }])

            # Feature engineering (same as training and API)
            input_df["Car_Age"] = 2023 - input_df["Year"]
            input_df["Mileage_log"] = np.log1p(input_df["Mileage"])
            input_df = input_df.drop(columns=["Year"])

            prediction_log = model.predict(input_df)[0]
            prediction = np.expm1(prediction_log)

            st.subheader("Result")
            st.metric("Estimated Price", f"${prediction:,.0f}")

            # Log prediction to MLflow
            try:
                mlflow.set_experiment("car-price-experiment")
                with mlflow.start_run(run_name="streamlit_prediction"):
                    mlflow.log_param("timestamp", datetime.datetime.now().isoformat())
                    mlflow.log_param("brand", brand)
                    mlflow.log_param("fuel_type", fuel_type)
                    mlflow.log_param("transmission", transmission)
                    mlflow.log_param("condition", condition)
                    mlflow.log_metric("predicted_price", float(prediction))
                    mlflow.log_metric("mileage", float(mileage))
                    mlflow.log_metric("engine_size", float(engine_size))
                st.caption("Prediction logged to MLflow")
            except Exception as e:
                st.caption(f"Could not log to MLflow: {e}")

            # Append to production samples (for drift monitoring)
            try:
                row = {
                    "Car_Age": 2023 - year,
                    "Mileage": mileage,
                    "Engine_Size": engine_size,
                    "predicted_price": float(prediction),
                }
                prod_df = pd.DataFrame([row])
                if PRODUCTION_PATH.exists():
                    existing = pd.read_csv(PRODUCTION_PATH)
                    prod_df = pd.concat([existing, prod_df], ignore_index=True)
                PRODUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
                prod_df.to_csv(PRODUCTION_PATH, index=False)
            except Exception:
                pass

# ═════════════════════════════════════════════
# PAGE 2 — Experiment History
# ═════════════════════════════════════════════
elif page == "Experiment History":
    st.title("Experiment History")

    try:
        experiments = client.search_experiments()
        exp_names = [e.name for e in experiments]
        selected_exp = st.selectbox("Select experiment", exp_names)

        exp = client.get_experiment_by_name(selected_exp)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.r2 DESC"],
        )

        if not runs:
            st.warning("No runs found for this experiment.")
        else:
            data = []
            for r in runs:
                data.append({
                    "run_id": r.info.run_id[:8],
                    "status": r.info.status,
                    "mae": r.data.metrics.get("mae"),
                    "rmse": r.data.metrics.get("rmse"),
                    "r2": r.data.metrics.get("r2"),
                    "n_estimators": r.data.params.get("n_estimators"),
                    "max_depth": r.data.params.get("max_depth"),
                })

            df_runs = pd.DataFrame(data)
            st.dataframe(df_runs, use_container_width=True)

            if "r2" in df_runs.columns and df_runs["r2"].notna().any():
                st.subheader("R2 Score by Run")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(df_runs["run_id"], df_runs["r2"].astype(float), color="steelblue")
                ax.set_xlabel("Run ID")
                ax.set_ylabel("R2")
                ax.set_title("Model R2 across runs")
                plt.xticks(rotation=45)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not fetch experiments: {e}")
        st.info("Make sure MLflow is running: mlflow ui")

# ═════════════════════════════════════════════
# PAGE 3 — Data Drift
# ═════════════════════════════════════════════
elif page == "Data Drift":
    st.title("Data Drift Monitoring")

    if not REFERENCE_PATH.exists():
        st.warning(
            "Reference data not found. Train the model first — "
            "reference_data.csv will be saved in artifacts/."
        )
    elif not PRODUCTION_PATH.exists():
        st.warning(
            "No production data yet. Make a few predictions on the Prediction page first."
        )
    else:
        reference = pd.read_csv(REFERENCE_PATH)
        production = pd.read_csv(PRODUCTION_PATH)

        st.write(
            f"Reference samples: **{len(reference)}** | "
            f"Production samples: **{len(production)}**"
        )

        common_cols = [
            c for c in reference.columns
            if c in production.columns and reference[c].dtype != object
        ]

        if not common_cols:
            st.error("No common numerical features between reference and production data.")
        else:
            feature = st.selectbox("Feature to inspect", common_cols)

            stat, p_value = ks_2samp(
                reference[feature].dropna(), production[feature].dropna()
            )
            drift_detected = p_value < 0.05

            col1, col2 = st.columns(2)
            col1.metric("KS p-value", f"{p_value:.4f}")
            col2.metric(
                "Drift",
                "Detected" if drift_detected else "Not detected",
            )

            st.subheader(f"Reference vs Production — {feature}")
            fig, ax = plt.subplots(figsize=(10, 4))
            reference[feature].hist(ax=ax, alpha=0.6, label="Reference", bins=30, color="steelblue")
            production[feature].hist(ax=ax, alpha=0.6, label="Production", bins=30, color="orange")
            ax.legend()
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
            st.pyplot(fig)

            st.subheader("Drift Summary (all features)")
            drift_results = []
            for col in common_cols:
                try:
                    s, p = ks_2samp(
                        reference[col].dropna(), production[col].dropna()
                    )
                    drift_results.append({
                        "Feature": col,
                        "KS Stat": round(s, 4),
                        "p-value": round(p, 4),
                        "Drift": "Yes" if p < 0.05 else "No",
                    })
                except Exception:
                    pass
            st.dataframe(pd.DataFrame(drift_results), use_container_width=True)

# ═════════════════════════════════════════════
# PAGE 4 — Model Lifecycle
# ═════════════════════════════════════════════
elif page == "Model Lifecycle":
    st.title("Model Lifecycle")

    try:
        versions = client.search_model_versions("name='car-price-model'")

        if not versions:
            st.warning("No registered model versions found in MLflow.")
        else:
            prod_versions = [v for v in versions if v.current_stage == "Production"]
            staging_versions = [v for v in versions if v.current_stage == "Staging"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total versions", len(versions))
            col2.metric("Production version", prod_versions[0].version if prod_versions else "None")
            col3.metric("Staging version", staging_versions[0].version if staging_versions else "None")

            st.subheader("All versions")
            version_data = []
            for v in versions:
                run = None
                try:
                    run = client.get_run(v.run_id)
                except Exception:
                    pass
                version_data.append({
                    "Version": v.version,
                    "Stage": v.current_stage,
                    "Run ID": v.run_id[:8] if v.run_id else "-",
                    "R2": round(run.data.metrics.get("r2", 0), 4) if run else "-",
                    "MAE": round(run.data.metrics.get("mae", 0), 2) if run else "-",
                    "Created": datetime.datetime.fromtimestamp(
                        v.creation_timestamp / 1000
                    ).strftime("%Y-%m-%d %H:%M") if v.creation_timestamp else "-",
                })
            st.dataframe(pd.DataFrame(version_data), use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch model versions: {e}")
        st.info("Make sure MLflow is running: mlflow ui")