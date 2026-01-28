import pandas as pd
from src.data.load_data import load_data

def preprocess_data():
    """
    Nettoyage des colonnes 
    """
    # on charge les données du fichier load_data
    df=load_data()

    # Gestion des valeurs manquantes 
    #On supprime les lignes totalement vides ou on remplit les Nan
    #on supprime les lignes ou le price est vide

    initial_shape=df.shape

    df=df.dropna(subset=['Price'])

    for col in ['Brand', 'Fuel Type', 'Transmission', 'Condition']:
        df[col] = df[col].fillna(df[col].mode()[0])

    categorical_cols=['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
    
    print(f"Nettoyage terminé.")
    print(f"Lignes avant: {initial_shape[0]} ; \n Lignes après: {df.shape[0]}")
    return df


if __name__ == "__main__":
    data_cleaned = preprocess_data()
    print("\n Statistiques après nettoyage :")
    print(data_cleaned.describe())
    # Vérification des manquants
    print("\nValeurs manquantes restantes :")
    print(data_cleaned.isnull().sum())