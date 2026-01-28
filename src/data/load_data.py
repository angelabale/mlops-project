import pandas as pd
from pathlib import Path

def load_data(file_name="car_price_prediction_with_missing.csv"):
    """
    Charger le dataset et appliquer le nettoyage
    
    :param file_name: Description
    """
    #On trouve le chemin du ficheir actuel
    current_file = Path(__file__).resolve()

    #on remonte jusqu'a la racine
    project_root=current_file.parent.parent.parent

    #on construit le chemin vers le CSV
    data_path=project_root/"data"/"raw"/file_name

    if not data_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {data_path}")
    
    #chargement
    df=pd.read_csv(data_path)
    

    return df
if __name__=="__main__":
    try:
        df=load_data()
        print(f"Extraction réussie:{df.shape}")
        print("="*30)
        print("Aperçue du dataset :")
        print("="*30)

        #Dimensions
        print(f"Dimensions:{df.shape[0]} lignes, {df.shape[1]} colonnes")

        # 2. Types de colonnes et valeurs manquantes
        print("Infos et Valeurs Manquantes :")
        print(df.info())

        # 3. Statistiques descriptives 
        print("\n Statistiques Descriptives :")
        print(df.describe(include='all')) 

        # 4. Aperçu des premières lignes
        print("\n 5 premières lignes :")
        print(df.head())
    except Exception as e:
        print(f"Erreur lors de l'affichage:{e}")