from datetime import timedelta

import pandas as pd

def calculate_recent_winrate(df, time_window=365):
    """
    Calcola il Winrate dinamico limitato all'ultimo anno per Player 1 e Player 2, utilizzando l'indicizzazione su 'Date'.
    """
    df = df.sort_values(by='Date')  # Ordinare per data
    df.set_index('Date', inplace=True)  # Impostare 'Date' come indice
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


