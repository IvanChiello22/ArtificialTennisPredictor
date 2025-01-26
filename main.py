"""
Questo script consente di utilizzare un modello di classificazione preaddestrato per prevedere l'esito di una partita
di tennis basandosi su input forniti dall'utente.

**Flusso dello script**:
1. **Caricamento del modello**:
    - Il modello addestrato viene caricato da un file `.joblib`.
    - In caso di errore nel caricamento, viene restituito un messaggio di errore.

2. **Input dati utente**:
    - L'utente fornisce i dati per la partita direttamente da tastiera (ad es. Rank, Winrate, Odds, ecc.).
    - I dati sono normalizzati o standardizzati se necessario, per assicurare coerenza con il modello addestrato.

3. **Predizione**:
    - Il modello predice il vincitore della partita (Player 1 o Player 2) sulla base dei dati forniti.
    - Viene stampato a schermo il risultato della predizione.

**Dati richiesti dall'utente**:
- `Player 1 WR` e `Player 2 WR`: Winrate dei giocatori.
- `Rank 1` e `Rank 2`: Posizioni nel ranking dei due giocatori (standardizzate internamente).
- `Pts 1` e `Pts 2`: Punti di ciascun giocatore.
- `Odd 1` e `Odd 2`: Quote per ciascun giocatore (utilizzate per calcolare probabilità normalizzate).
- `Round`, `Series`, `Surface`, `Best of`: Altre caratteristiche del match.

**Note tecniche**:
- Lo script normalizza le quote (`Odd_1` e `Odd_2`) per ottenere le probabilità normalizzate di vincita per ciascun giocatore.
- Lo standard scaler viene utilizzato per standardizzare i rank (`Rank_1` e `Rank_2`).
- È necessario che il modello caricato includa l'ordine delle feature, così da mappare correttamente i dati inseriti dall'utente.

**Output**:
- Lo script stampa il vincitore predetto (Player 1 o Player 2).

**Prerequisiti**:
- Assicurarsi che il file del modello (`model.joblib`) esista nella directory specificata.
- Le feature nel modello addestrato devono corrispondere esattamente ai dati richiesti dall'utente.

**Esecuzione**:
- Eseguire il file e seguire le istruzioni fornite a schermo per inserire i dati richiesti.
"""


from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler


# Funzione per caricare il modello
def load_model(model_path):
    try:
        model = load(model_path)
        print("Modello caricato con successo.")
        return model
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return None


# Funzione per fare la previsione sulla partita
def predict_match(model, match_data):
    try:
        # Predizione
        prediction = model.predict(match_data)
        return prediction
    except Exception as e:
        print(f"Errore nella previsione: {e}")
        return None


# Funzione per chiedere l'input da tastiera
def get_match_data(feature_order):
    print("Inserisci i dati per la partita (i valori devono essere numerici):")



    player1_wr = float(input("Player 1 WR (Ultimo Winrate Disponibile nel dataset): "))

    player2_wr = float(input("Player 2 WR (Ultimo Winrate Disponibile nel dataset): "))


    rank_1 = float(input("Rank 1: "))
    rank_2 = float(input("Rank 2: "))
    #rank_1 = rank_1 / rank_1 + rank_2
    #rank_2 = rank_2 / rank_1 + rank_1
    scaler = StandardScaler()
    # Standardizzazione degli input
    input_data = np.array([[rank_1, rank_2]])

    scaler.fit(input_data)

    # Trasformiamo i dati
    standardized_data = scaler.transform(input_data)


    pts_1 = int(input("Pts 1: "))
    pts_2 = int(input("Pts 2: "))

    odd_1 = float(input("Odd 1: "))
    odd_2 = float(input("Odd 2: "))

    p1= 1 / odd_1
    p2= 1 / odd_2
    prob_1_norm = p1 / p1 + p2
    prob_2_norm = p2 / p1 + p2

    round_encoded = int(input("Round: "))
    series_encoded = int(input("Series: "))
    surface_encoded = int(input("Surface: "))
    best = int(input("Best of: "))
    #court_indoor = bool(input("Court indoor: "))
    #court_outdoor = bool(input("Court outdoor: "))


    # Restituisce un DataFrame con i dati inseriti
    match_data = pd.DataFrame({
        'Player1_WR': [player1_wr],
        'Player2_WR': [player2_wr],
        'Rank_1': [standardized_data[0][0]],
        'Rank_2': [standardized_data[0][1]],
        'Pts_1': [pts_1],
        'Pts_2': [pts_2],
        #'Odd_1': [odd_1],
        #'Odd_2': [odd_2],
        'Prob_1_norm': [prob_1_norm],
        'Prob_2_norm': [prob_2_norm],
        'Round_encoded': [round_encoded],
        'Series_encoded': [series_encoded],
        'Surface_encoded': [surface_encoded],
        'Best of': [best]
        #: [court_indoor],
        #'Court_Outdoor': [court_outdoor]
    })

    match_data = match_data[feature_order]
    return match_data


# Funzione principale che esegue l'intero processo
def main():
    # Directory radice del progetto
    base_dir = Path(__file__).resolve().parent  # Risali di un livello

    # Percorsi relativi
    model_path = base_dir / "modello" / "model.joblib"


    # Carica il modello
    model = load_model(model_path)
    if model is None:
        return

    # Recupera l'ordine delle feature dal modello addestrato
    feature_order = model.feature_names_in_

    # Ottieni i dati della partita tramite tastiera
    match_data = get_match_data(feature_order)



    # Fai la previsione
    prediction = predict_match(model, match_data)

    if prediction is not None:
        # Stampa il risultato predetto
        if prediction[0] == 0:
            print("Il predetto vincitore è Player 1.")
        else:
            print("Il predetto vincitore è Player 2.")


# Esegui lo script
if __name__ == "__main__":
    main()
