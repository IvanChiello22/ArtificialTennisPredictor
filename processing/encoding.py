from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Directory radice del progetto
base_dir = Path(__file__).resolve().parent.parent  # Risali di un livello

# Percorsi relativi
file_path = base_dir / "data" / "processed" / "Dataset_Prob_v2.csv"
encoded_file_path = base_dir / "data" / "processed" / "encoded_dataset.csv"

#file_path = r"C:\Users\Domenico\PycharmProjects\ArtificialTennisPredictor\data\processed\Dataset_Prob_v2.csv"
#encoded_file_path = r"C:\Users\Domenico\PycharmProjects\ArtificialTennisPredictor\data\processed\encoded_dataset.csv"



df = pd.read_csv(file_path, low_memory=False)

# Mappa categorie in base alla gerarchia
series_mapping = {
    'ATP250': 7,
    'ATP500': 5,
    'International': 8,
    'International Gold': 6,
    'Masters': 4,
    'Masters 1000': 3,
    'Grand Slam': 1,
    'Masters Cup': 2
}

surface_mapping = {
    'Carpet': 1,
    'Clay': 2,
    'Grass': 3,
    'Hard': 4,
}

round_mapping = {
    '1st Round': 1,
    '2nd Round': 2,
    '3rd Round': 3,
    '4th Round': 4,
    'Quarterfinals': 5,
    'Round Robin': 0,
    'Semifinals': 6,
    'The Final': 7
}



def encode_dataset(df):
    df_cleaned = df.copy()

    df_cleaned = df_cleaned.drop(columns=['Tournament'])

    df_cleaned['Series_encoded'] = df_cleaned['Series'].map(series_mapping)
    df_cleaned.insert(2, 'Series_encoded', df_cleaned.pop('Series_encoded'))

    # Applicare One-Hot Encoding alla colonna 'Court'
    df_cleaned = pd.get_dummies(df_cleaned, columns=['Court'])
    df_cleaned.insert(3, 'Court_Indoor', df_cleaned.pop('Court_Indoor'))
    df_cleaned.insert(3, 'Court_Outdoor', df_cleaned.pop('Court_Outdoor'))

    df_cleaned['Surface_encoded'] = df_cleaned['Surface'].map(surface_mapping)
    df_cleaned.insert(6, 'Surface_encoded', df_cleaned.pop('Surface_encoded'))

    df_cleaned['Round_encoded'] = df_cleaned['Round'].map(round_mapping)
    df_cleaned.insert(8, 'Round_encoded', df_cleaned.pop('Round_encoded'))

    #--------------------------------------------------------------------------------------------------------------------
    # Unire tutte le occorrenze di Player_1 e Player_2 per ottenere una codifica unica
    all_players = pd.concat([df_cleaned['Player_1'], df_cleaned['Player_2']]).unique()

    # Creare un LabelEncoder per la codifica comune
    common_encoder = LabelEncoder()

    # Adattare l'encoder sui valori unici combinati
    common_encoder.fit(all_players)

    # Codificare entrambe le colonne con lo stesso encoder
    df_cleaned['Player_1_encoded'] = common_encoder.transform(df_cleaned['Player_1'])
    df_cleaned['Player_2_encoded'] = common_encoder.transform(df_cleaned['Player_2'])

    df_cleaned.insert(12, 'Player_1_encoded', df_cleaned.pop('Player_1_encoded'))
    df_cleaned.insert(15, 'Player_2_encoded', df_cleaned.pop('Player_2_encoded'))

    #--------------------------------------------------------------------------------------------------------------------

    df_cleaned['Winner_encoded'] = df_cleaned.apply(lambda row: 0 if row['Winner'] == row['Player_1'] else 1, axis=1)
    df_cleaned.insert(17, 'Winner_encoded', df_cleaned.pop('Winner_encoded'))
    #--------------------------------------------------------------------------------------------------------------------

    df_cleaned = df_cleaned.drop(columns=['Score']) #HO AGGIUNTO ODD PER CONTROLLARE IL FUNZIONAMENTO DOPO
    #--------------------------------------------------------------------------------------------------------------------

    return df_cleaned

print("processing...")
df_cleaned = encode_dataset(df)

print("salvataggio e scrittura...")

df_cleaned.to_csv(encoded_file_path, index = False)
print(df_cleaned.head())
print("completato")
