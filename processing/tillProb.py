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
