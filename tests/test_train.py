import pandas as pd
import pytest

from src.models import train
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


@pytest.fixture
def sample_df():
    # Colonnes nécessaires: Price, Car ID, Year + mix cat/num
    return pd.DataFrame(
        {
            "Car ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Year": [
                2005,
                2008,
                2012,
                2009,
                2015,
                2007,
                2011,
                2006,
                2010,
                2013,
            ],
            "Brand": ["A", "A", "B", "B", "A", "C", "C", "A", "B", "C"],  # cat
            "Model": [
                "m1",
                "m2",
                "m1",
                "m3",
                "m2",
                "m1",
                "m2",
                "m3",
                "m1",
                "m3",
            ],  # cat
            "Mileage": [10, 20, 15, 5, 8, 25, 30, 12, 18, 7],  # num
            "Price": [
                1000,
                1200,
                1500,
                1100,
                1600,
                900,
                1300,
                1050,
                1400,
                1550,
            ],
        }
    )


def test_load_processed_data_file_not_found(monkeypatch):
    monkeypatch.setattr(train.Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError):
        train.load_processed_data()


def test_train_model_prints_summary_with_filter(
    monkeypatch, capsys, sample_df
):
    monkeypatch.setattr(train, "load_processed_data", lambda: sample_df)

    train.train_model(max_year=2010, n_estimators=10)

    out = capsys.readouterr().out
    assert "Training completed successfully" in out
    assert "Data used: Year <= 2010" in out
    assert "Number of trees: 10" in out
    assert "Mean Absolute Error (MAE):" in out


def test_train_model_prints_summary_without_filter(
    monkeypatch, capsys, sample_df
):
    monkeypatch.setattr(train, "load_processed_data", lambda: sample_df)

    train.train_model(max_year=None, n_estimators=5)

    out = capsys.readouterr().out
    assert "Training completed successfully" in out
    assert "Number of trees: 5" in out
    # Quand max_year=None, cette ligne ne doit pas apparaître
    assert "Data used:" not in out
