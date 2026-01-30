import pandas as pd
import kagglehub
import shutil
import os
from pathlib import Path

def load_data():
    # Définition des chemins
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    file_name = "car_price_prediction_with_missing.csv"
    target_path = raw_data_dir / file_name

    # 1. Créer le dossier data/raw s'il n'existe pas
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Télécharger si le fichier n'est pas là
    if not target_path.exists():
        print("Téléchargement depuis Kaggle via kagglehub...")
        # Télécharge (retourne le chemin du dossier temporaire)
        downloaded_path = kagglehub.dataset_download("nalisha/car-price-prediction-dataset")
        
        # Trouver le fichier .csv dans le dossier téléchargé
        # (On cherche n'importe quel .csv si le nom change un peu)
        source_files = list(Path(downloaded_path).glob("*.csv"))
        
        if source_files:
            shutil.copy(source_files[0], target_path)
            print(f" Fichier déplacé vers : {target_path}")
        else:
            raise FileNotFoundError("Aucun fichier CSV trouvé dans le téléchargement Kaggle.")

    # 3. Charger le DataFrame
    df = pd.read_csv(target_path)
    print(f"Extraction réussie : {df.shape}")
    return df

if __name__ == "__main__":
    try:
        # Exécution du chargement
        df = load_data()

        # --- BLOC ANALYSE DE DATA ---
        print("\n" + "="*40)
        print(" DIAGNOSTIC DU DATASET")
        print("="*40)

        # 1. Dimensions
        print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")

        # 2. Infos techniques (Types & Non-Nuls)
        print("\n Infos et Valeurs Manquantes :")
        print("-" * 20)
        print(df.info())

        # 3. Statistiques descriptives (Numérique + Catégoriel)
        print("\n Statistiques Descriptives :")
        print("-" * 20)
        print(df.describe(include='all')) 

        # 4. Aperçu des données
        print("\n Aperçu des 5 premières lignes :")
        print("-" * 20)
        print(df.head())
        print("="*40)

    except Exception as e:
        print(f" Erreur lors de l'exécution : {e}")