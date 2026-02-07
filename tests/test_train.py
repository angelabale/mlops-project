import pandas as pd
import pytest
from pathlib import Path


from src.models.train import load_processed_data


def test_load_processed_data_raises_if_missing(tmp_path, monkeypatch):
    # On se met dans un faux projet
    fake_project_root = tmp_path / "mlops-project"
    (fake_project_root / "data" / "processed").mkdir(parents=True)

    # Patch __file__ pour que project_root soit fake_project_root
    monkeypatch.setattr(
        "src.models.train.__file__",
        str(fake_project_root / "src" / "models" / "train.py"),
    )

    # Le fichier n'existe pas -> doit lever une erreur
    with pytest.raises(FileNotFoundError):
        load_processed_data()


def test_load_processed_data_returns_dataframe(tmp_path, monkeypatch):
    fake_project_root = tmp_path / "mlops-project"
    processed_dir = fake_project_root / "data" / "processed"
    processed_dir.mkdir(parents=True)

    csv_path = processed_dir / "car_price_cleaned.csv"
    pd.DataFrame({"Car ID": [1], "Price": [10000], "Year": [2010]}).to_csv(
        csv_path, index=False
    )

    monkeypatch.setattr(
        "src.models.train.__file__",
        str(fake_project_root / "src" / "models" / "train.py"),
    )

    df = load_processed_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "Price" in df.columns
