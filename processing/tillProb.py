"""
Script di Pulizia e Calcolo delle Probabilità Implicite delle Quote

Questo script si occupa di pulire e trasformare il dataset generato nella fase precedente (`Dataset_WR.csv`),
focalizzandosi sull'elaborazione delle colonne contenenti le quote (`Odd_1` e `Odd_2`) per calcolare le probabilità implicite dei giocatori.
L'obiettivo è produrre un dataset che includa probabilità normalizzate e pronte per l'analisi o la modellazione.

Principali operazioni eseguite:
1. **Caricamento dei dati**: Lettura del dataset processato nella fase precedente.
2. **Gestione delle anomalie nei dati delle quote**:
   - Conversione delle quote da stringa a formato numerico float, gestendo eventuali separatori decimali (virgola/punto).
   - Rimozione di valori anomali o non validi (es. valori estremamente grandi o piccoli rappresentati in notazione scientifica).
   - Correzione delle quote che presentano valori sproporzionati spostando il punto decimale per normalizzarle.
3. **Calcolo delle Probabilità Implicite**:
   - Le probabilità implicite sono calcolate come l'inverso delle quote (`1 / Odd`).
   - Le probabilità sono normalizzate affinché la loro somma per un match sia pari a 1.
4. **Arrotondamento delle Probabilità**: Le probabilità normalizzate vengono arrotondate a 4 decimali per semplificare l'analisi.
5. **Esportazione dei dati**: Salvataggio del dataset processato in un file CSV (`Dataset_Prob_v2.csv`).

Struttura del codice:
- **Funzioni definite**:
  - `is_scientific(x)`: Controlla se un valore è rappresentato in notazione scientifica.
  - `convert_to_float(value)`: Converte una stringa contenente una quota in formato float, gestendo separatori errati.
  - `move_dot(x)`: Sposta il punto decimale di un valore per correggere eventuali errori nei dati delle quote.
  - `process_Prob(df)`: Applica i passaggi sopra elencati per elaborare l'intero dataset.
- Il codice è modulare e ogni passaggio è chiaramente separato per garantire manutenibilità e chiarezza.

Output:
- Un dataset pulito e arricchito (`Dataset_Prob_v2.csv`), con colonne per le probabilità normalizzate (`Prob_1_norm`, `Prob_2_norm`).

Nota:
Questo script è il secondo di una sequenza. Utilizza il dataset generato nella fase precedente e prepara i dati per le successive analisi, come la modellazione predittiva o l'addestramento di modelli di machine learning.

"""


from pathlib import Path

import pandas as pd


# Directory radice del progetto
base_dir = Path(__file__).resolve().parent.parent  # Risali di un livello

# Percorsi relativi
file_path = base_dir / "data" / "processed" / "Dataset_WR.csv"
cleaned_file_path = base_dir / "data" / "processed" / "Dataset_Prob_v2.csv"


# Funzione per controllare se un numero è rappresentato in notazione scientifica
def is_scientific(x):
    if isinstance(x, float):  # Considera solo i numeri float
       return x >= 1e+10 or x <= 1e-10  # Condizioni per notazione scientifica
    return False

# Funzione per gestire la sostituzione della virgola e conversione in float
def convert_to_float(value):
    if isinstance(value, str):
        # Sostituire la virgola con il punto
        value = value.replace(',', '.')
        try:
            # Provare a convertire in float
            return float(value)
        except ValueError:
            # Se non è possibile, restituire NaN (o puoi gestirlo diversamente)
            return None
    return value

# Funzione per spostare il punto dopo il primo numero se è necessario
def move_dot(x):
    if isinstance(x, float):# Verifica se è una stringa
        if x >= 10000:
            return x / 10000
        elif x >= 1000:
            return x / 1000
        elif x >= 100:
            return x / 100
        elif x >= 10:
            return x / 10

    return x

df = pd.read_csv(file_path, low_memory=False)


def process_Prob(df):
    df_cleaned = df.copy()



    # Applicare la funzione alla colonna 'Odd_1'
    df_cleaned['Odd_1'] = df_cleaned['Odd_1'].apply(convert_to_float)
    df_cleaned['Odd_2'] = df_cleaned['Odd_2'].apply(convert_to_float)

    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: is_scientific(row['Odd_1']) or is_scientific(row['Odd_2']), axis=1)]



    df_cleaned['Odd_1'] = df_cleaned['Odd_1'].apply(move_dot)
    df_cleaned['Odd_2'] = df_cleaned['Odd_2'].apply(move_dot)



    # Calcola le probabilità implicite non normalizzate
    df_cleaned['p1'] = 1 / df_cleaned['Odd_1']
    df_cleaned['p2'] = 1 / df_cleaned['Odd_2']


    # Normalizzare le probabilità in modo che la somma sia uguale a 1
    df_cleaned['Prob_1_norm'] = df_cleaned['p1'] / (df_cleaned['p1'] + df_cleaned['p2'])
    df_cleaned['Prob_2_norm'] = df_cleaned['p2'] / (df_cleaned['p1'] + df_cleaned['p2'])

    df_cleaned.drop(['p1', 'p2'], axis=1, inplace=True)

    df_cleaned['Prob_1_norm'] = df_cleaned['Prob_1_norm'].round(4)
    df_cleaned['Prob_2_norm'] = df_cleaned['Prob_2_norm'].round(4)


    return df_cleaned

print("processing...")
df_cleaned = process_Prob(df)

print("salvataggio e scrittura...")

df_cleaned.to_csv(cleaned_file_path, index = False)
print(df_cleaned.head())
print("completato")
