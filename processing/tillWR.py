"""
Script di Pre-Processing dei Dati per Analisi Tennis

Questo script si occupa di effettuare un primo livello di pre-processing del dataset ATP per renderlo pronto
alle successive fasi di analisi e modellazione.
Il file processa un dataset grezzo in formato CSV contenente dati relativi ai match di tennis e produce un file
pulito e arricchito con nuove variabili calcolate.

Principali operazioni eseguite:
1. **Caricamento dei dati**: Lettura del dataset grezzo da file CSV utilizzando pandas.
2. **Conversione delle date**: Trasformazione della colonna "Date" in formato datetime per consentire un'elaborazione basata sul tempo.
3. **Calcolo del Delta Rank**: Creazione di una nuova variabile `Delta_Rank`, che rappresenta la differenza tra i ranking dei due giocatori.
4. **Calcolo del WinRate Dinamico**:
   - Per ogni match, viene calcolato il WinRate recente (ultimi 365 giorni) per entrambi i giocatori.
   - Questa operazione tiene conto dei match giocati da ciascun giocatore nell'anno precedente alla data del match corrente.
5. **Pulizia dei dati**: Eliminazione di righe contenenti valori invalidi o indesiderati (es. `-1`).
6. **Esportazione dei dati**: Salvataggio del dataset processato in un nuovo file CSV, per un accesso rapido nelle fasi successive.

Struttura del codice:
- La funzione `calculate_recent_winrate` è la parte centrale del codice, dove viene implementato il calcolo del WinRate dinamico.
- Le operazioni sul dataset sono strutturate in modo sequenziale e mirano a migliorare la leggibilità e la manutenibilità del codice.
- Viene utilizzata la libreria `pandas` per la manipolazione dei dati e la libreria `pathlib` per la gestione dei percorsi file in modo robusto.

Output:
- Un dataset pulito e arricchito (`Dataset_WR.csv`), pronto per essere utilizzato nelle fasi successive di analisi o modellazione.

Nota:
Questo script è il primo di una sequenza di script progettati per elaborare i dati di tennis. I prossimi file potrebbero utilizzare il dataset generato qui come input per ulteriori trasformazioni o analisi.

"""


from pathlib import Path
import pandas as pd
from datetime import timedelta

#Directory radice del progetto
base_dir = Path(__file__).resolve().parent.parent  # Risali di un livello

# Percorsi relativi
file_path = base_dir / "data" / "raw" / "Dataset_ATP.csv"
cleaned_file_path = base_dir / "data" / "processed" / "Dataset_WR.csv"


#file_path = r"C:\Users\Marco\Desktop\ArtificialTennisPredictorv2\ArtificialTennisPredictorv2\data\raw\Dataset_ATP.csv"
#cleaned_file_path = r"C:\Users\Marco\Desktop\ArtificialTennisPredictorv2\ArtificialTennisPredictorv2\data\processed\Dataset_WR.csv"

# Usa 'sep' per cambiare il delimitatore standard di pandas "," con ";"
df = pd.read_csv(file_path, sep = ";")

df_cleaned = df.copy()

#---------------------------------------------------------------------------


# colonna date trasformata in datetime
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%d/%m/%Y')



#-----------------------------------------------------------------------------------------------------------------------------------------------------

#calcolo del Delta del rank tra player1 e player2


df_cleaned['Delta_Rank'] = df_cleaned['Rank_1'] - df_cleaned['Rank_2']

df_cleaned.insert(12, 'Delta_Rank', df_cleaned.pop('Delta_Rank'))

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#Calcolo del WinRate di Player_1 e player_2
def calculate_recent_winrate(df, time_window=365):
    """
    Calcola il Winrate dinamico limitato all'ultimo anno per Player 1 e Player 2, utilizzando l'indicizzazione su 'Date'.
    """
    df = df.sort_values(by='Date')  # Ordinare per data
    df.set_index('Date', inplace=True, drop=False)  # Impostare 'Date' come indice
    player1_wr = []
    player2_wr = []

    # Itera su ogni riga del dataset
    for i, row in df.iterrows():
        p1, p2, winner, match_date = row['Player_1'], row['Player_2'], row['Winner'], i

        # Limita il dataset alle partite dell'ultimo anno rispetto alla partita corrente
        one_year_ago = match_date - timedelta(days=time_window)

        # Filtro usando l'indice di data (più veloce)
        recent_matches = df[(df.index > one_year_ago) & (df.index < match_date)]

        # Calcola le statistiche di Player 1
        p1_matches = recent_matches[(recent_matches['Player_1'] == p1) | (recent_matches['Player_2'] == p1)]
        p1_wins = p1_matches[p1_matches['Winner'] == p1].shape[0]
        p1_total = p1_matches.shape[0]
        p1_wr = round((p1_wins / p1_total) * 100, 2) if p1_total > 0 else 0
        player1_wr.append(p1_wr)

        # Calcola le statistiche di Player 2
        p2_matches = recent_matches[(recent_matches['Player_1'] == p2) | (recent_matches['Player_2'] == p2)]
        p2_wins = p2_matches[p2_matches['Winner'] == p2].shape[0]
        p2_total = p2_matches.shape[0]
        p2_wr = round((p2_wins / p2_total) * 100, 2) if p2_total > 0 else 0
        player2_wr.append(p2_wr)

    # Aggiungi le colonne calcolate al dataset
    df['Player1_WR'] = player1_wr
    df['Player2_WR'] = player2_wr

    return df
print("processing...")
df_cleaned = calculate_recent_winrate(df_cleaned)
df_cleaned.insert(7, 'Player1_WR', df_cleaned.pop('Player1_WR'))
df_cleaned.insert(9, 'Player2_WR', df_cleaned.pop('Player2_WR'))

#-----------------------------------------------------------------------------------------------------------------------------------------------------


# Eliminazione delle righe con valori nulli ea evitare
df_cleaned = df_cleaned[~df_cleaned.isin([-1]).any(axis=1)]

print("salvataggio e scrittura...")
#------------------------------------------------------------------------------------------------------------------------------------------------------
#salvataggio del dataset intermedio per migliroare l'efficienza
df_cleaned.to_csv(cleaned_file_path, index = False)

print("completato")





# Mostrare le prime righe del DataFrame pulito
print(df_cleaned.head())



