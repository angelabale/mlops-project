import pandas as pd

from src.data.load_data import load_data


def test_load_data_uses_existing_csv(tmp_path, monkeypatch):
    """
    If the CSV already exists in data/raw, load_data should read it
    and MUST NOT call kagglehub.dataset_download.
    """

    # Fake project root structure
    fake_project_root = tmp_path / "mlops-project"
    raw_dir = fake_project_root / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Create the expected CSV file
    target_csv = raw_dir / "car_price_prediction_with_missing.csv"
    pd.DataFrame({"price": [10000, 15000], "year": [2010, 2015]}).to_csv(
        target_csv, index=False
    )

    # Make load_data() compute project_root from our fake location
    monkeypatch.setattr(
        "src.data.load_data.__file__",
        str(fake_project_root / "src" / "data" / "load_data.py"),
    )

    # Fail hard if it tries to download
    def _fail_download(*args, **kwargs):
        raise AssertionError(
            "dataset_download should NOT be called when file exists"
        )

    monkeypatch.setattr(
        "src.data.load_data.kagglehub.dataset_download", _fail_download
    )

    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["price", "year"]


def test_load_data_downloads_when_missing(tmp_path, monkeypatch):
    """
    If the CSV is missing, load_data should call kagglehub.dataset_download,
    copy the CSV into data/raw, then read it.
    """

    fake_project_root = tmp_path / "mlops-project"
    raw_dir = fake_project_root / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Ensure target file is missing
    target_csv = raw_dir / "car_price_prediction_with_missing.csv"
    assert not target_csv.exists()

    # Create a fake "downloaded" folder that contains a CSV
    downloaded_dir = tmp_path / "downloaded_dataset"
    downloaded_dir.mkdir()
    source_csv = downloaded_dir / "some_name_from_kaggle.csv"
    pd.DataFrame({"price": [1], "year": [2000]}).to_csv(
        source_csv, index=False
    )

    # Patch __file__ so project_root resolves correctly
    monkeypatch.setattr(
        "src.data.load_data.__file__",
        str(fake_project_root / "src" / "data" / "load_data.py"),
    )

    # Mock kagglehub download to return our downloaded_dir
    def _mock_download(_dataset_name: str):
        return str(downloaded_dir)

    monkeypatch.setattr(
        "src.data.load_data.kagglehub.dataset_download", _mock_download
    )

    # We let shutil.copy run normally, but we could also mock it if needed.
    df = load_data()

    assert target_csv.exists()

    # And dataframe should match the source csv content
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert df.iloc[0]["price"] == 1
    assert df.iloc[0]["year"] == 2000
