from pathlib import Path

import pandas as pd

from src.data.preprocess import preprocess_data


def test_preprocess_data_cleans_and_saves(tmp_path, monkeypatch):
    # 1) Se placer dans un dossier temporaire pour ne pas écrire dans ton repo
    monkeypatch.chdir(tmp_path)

    # 2) Fake dataset (avec NaN exprès)
    df_raw = pd.DataFrame(
        {
            "Car ID": [1, 2, 3],
            "Price": [10000, None, 15000],  # une ligne à drop
            "Brand": ["Toyota", None, "BMW"],
            "Fuel Type": ["Petrol", None, "Diesel"],
            "Transmission": ["Manual", None, "Auto"],
            "Condition": ["Used", None, "New"],
            "Model": [" Corolla ", " X5 ", None],  # strip + astype(str)
            "Year": [2010, None, 2015],
            "Engine Size": [1.8, None, 2.0],
            "Mileage": [120000, None, 90000],
        }
    )

    # 3) Mock load_data() pour retourner notre df
    monkeypatch.setattr("src.data.preprocess.load_data", lambda: df_raw)

    # 4) Lancer preprocess
    df_clean = preprocess_data()

    # 5) Assertions principales
    # La ligne avec Price=None doit être supprimée -> 2 lignes restantes
    assert df_clean.shape[0] == 2

    # Pas de NaN dans les colonnes remplies
    for col in ["Brand", "Fuel Type", "Transmission", "Condition"]:
        assert df_clean[col].isna().sum() == 0

    # Les numériques doivent être remplis (médiane)
    for col in ["Year", "Engine Size", "Mileage"]:
        assert df_clean[col].isna().sum() == 0

    # Model doit être string et strip (pas d'espaces au début/fin)
    assert df_clean["Model"].dtype == object
    assert not df_clean["Model"].astype(str).str.startswith(" ").any()
    assert not df_clean["Model"].astype(str).str.endswith(" ").any()

    saved_file = Path("data/processed/car_price_cleaned.csv")
    assert saved_file.exists()

    df_saved = pd.read_csv(saved_file)
    assert df_saved.shape[0] == 2
