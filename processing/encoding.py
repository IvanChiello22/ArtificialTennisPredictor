"""
Script di Codifica ed Elaborazione Avanzata del Dataset di Tennis

Questo script si occupa di eseguire operazioni di **codifica** e **standardizzazione** sul dataset pre-processato nella fase precedente (`Dataset_Prob_v2.csv`),
preparandolo per essere utilizzato nei modelli predittivi.
L'obiettivo è trasformare le colonne categoriali e numeriche in un formato che sia interpretabile dai modelli di machine learning.

Principali operazioni eseguite:
1. **Caricamento dei dati**: Lettura del dataset processato nella fase precedente.
2. **Codifica delle Categorie Ordinali**:
   - La colonna `Series` è codificata utilizzando una mappatura basata sull'importanza dei tornei (es. Grand Slam = 1, ATP250 = 7).
   - La colonna `Surface` è codificata in base alla tipologia della superficie (es. Carpet = 1, Hard = 4).
   - La colonna `Round` è codificata in base al turno del torneo (es. 1st Round = 1, Final = 7).
3. **One-Hot Encoding**:
   - La colonna `Court` (Indoor/Outdoor) è codificata utilizzando il metodo One-Hot Encoding per generare due colonne (`Court_Indoor`, `Court_Outdoor`).
4. **Standardizzazione delle Colonne Numeriche**:
   - I ranking dei giocatori (`Rank_1` e `Rank_2`) sono standardizzati utilizzando `StandardScaler` per migliorare la convergenza nei modelli.
5. **Codifica della Variabile Target**:
   - La colonna `Winner` è codificata in modo binario (`Winner_encoded`): 0 se il vincitore è `Player_1`, 1 se il vincitore è `Player_2`.
6. **Organizzazione delle Colonne**: Le nuove colonne codificate sono inserite in posizioni strategiche all'interno del DataFrame per migliorare la leggibilità.

Struttura del codice:
- **Funzioni definite**:
  - `encode_dataset(df)`: Funzione principale che applica tutti i passaggi di codifica e standardizzazione al dataset.
- Il codice è modulare e progettato per essere facilmente modificabile in caso di aggiornamenti o aggiunte ai dati.
- Utilizza `StandardScaler` per normalizzare le colonne numeriche e garantire coerenza tra diverse scale di dati.

Output:
- Un dataset completamente codificato ed elaborato (`encoded_dataset.csv`), pronto per essere utilizzato nei modelli di machine learning.

Nota:
Questo script rappresenta il terzo passaggio della pipeline di elaborazione. Il dataset risultante è ottimizzato per l'addestramento di modelli predittivi, con dati numerici e categoriali gestiti in modo appropriato.

"""


from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

    scaler = StandardScaler()
    # Applica la standardizzazione alle colonne Rank_1 e Rank_2
    df_cleaned[['Rank_1', 'Rank_2']] = scaler.fit_transform(df_cleaned[['Rank_1','Rank_2']])
    #--------------------------------------------------------------------------------------------------------------------

    df_cleaned['Winner_encoded'] = df_cleaned.apply(lambda row: 0 if row['Winner'] == row['Player_1'] else 1, axis=1)
    df_cleaned.insert(16, 'Winner_encoded', df_cleaned.pop('Winner_encoded'))
    #--------------------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------------------

    return df_cleaned

print("processing...")
df_cleaned = encode_dataset(df)

print("salvataggio e scrittura...")

df_cleaned.to_csv(encoded_file_path, index = False)
print(df_cleaned.head())
print("completato")
