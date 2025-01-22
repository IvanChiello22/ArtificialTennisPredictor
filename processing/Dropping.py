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
    df_cleaned = df_cleaned.drop(columns=['Series','Surface','Round','Player_1','Player_2','Winner', 'Score','Tournament'])


    return df_cleaned


print("processing...")
df_cleaned = drop_dataset(df)

print("salvataggio e scrittura...")

df_cleaned.to_csv(dropped_file_path, index = False)
print(df_cleaned.head())
print("completato")
