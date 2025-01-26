"""
Script di Pulizia Finale e Riduzione delle Dimensioni del Dataset

Questo script si occupa di eliminare colonne superflue o non necessarie dal dataset codificato nella fase precedente (`encoded_dataset.csv`).
L'obiettivo è ottenere un dataset più compatto e focalizzato, contenente solo le informazioni rilevanti per l'analisi o l'addestramento di modelli predittivi.

Principali operazioni eseguite:
1. **Caricamento dei dati**: Lettura del dataset codificato generato nella fase precedente.
2. **Eliminazione di colonne**:
   - Colonne testuali o categoriali che sono già state codificate e non sono più necessarie, come:
     - `Series`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`, `Score`, `Tournament`.
   - Colonne ridondanti o meno rilevanti per l'analisi, come `Delta_Rank`.
3. **Ottimizzazione delle dimensioni del dataset**: Rimuovendo colonne inutili, il dataset risultante è più compatto e pronto per l'uso nei modelli.

Struttura del codice:
- **Funzioni definite**:
  - `drop_dataset(df)`: Funzione principale che esegue la copia del dataset originale e rimuove le colonne non necessarie.
- Il codice è progettato per essere semplice e modulare, rendendo facile l'aggiornamento o la modifica delle colonne da eliminare.

Output:
- Un dataset ridotto e pulito (`dropped_dataset.csv`), contenente solo le informazioni essenziali per le fasi successive.

Nota:
Questo script rappresenta l'ultimo passaggio della pipeline di elaborazione dei dati. Il dataset risultante è pronto per essere utilizzato in analisi o per l'addestramento di modelli, avendo eliminato ogni ridondanza o informazione non necessaria.

"""


from pathlib import Path

import pandas as pd

#Directory radice del progetto
base_dir = Path(__file__).resolve().parent.parent  # Risali di un livello

# Percorsi relativi
file_path = base_dir / "data" / "processed" / "encoded_dataset.csv"
dropped_file_path = base_dir / "data" / "Finished" / "dropped_dataset.csv"

#file_path = r"C:\Users\Domenico\PycharmProjects\ArtificialTennisPredictor\data\processed\encoded_dataset.csv"
#droped_file_path = r"C:\Users\Domenico\PycharmProjects\ArtificialTennisPredictor\data\Finished\dropped_dataset.csv"


df = pd.read_csv(file_path, low_memory=False)

def drop_dataset(df):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop(columns=['Series','Surface','Round','Player_1','Player_2','Winner', 'Score','Tournament', 'Delta_Rank'])


    return df_cleaned


print("processing...")
df_cleaned = drop_dataset(df)

print("salvataggio e scrittura...")

df_cleaned.to_csv(dropped_file_path, index = False)
print(df_cleaned.head())
print("completato")
