import pandas as pd
from datetime import timedelta

from data.raw.winrates import calculate_recent_winrate



file_path = "Dataset_ATP.csv"
cleaned_file_path = r"C:\Users\Ivan\PycharmProjects\ArtificialTennisPredictor\data\processed\Dataset_intermedio_withWR.csv"

# Usa 'sep' per cambiare il delimitatore standard di pandas "," con ";"
df = pd.read_csv(file_path, sep = ";")

df_cleaned = df

#---------------------------------------------------------------------------

df_cleaned = df_cleaned.copy()

# colonna date trasformata in datetime
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%d/%m/%Y')

reference_date = pd.to_datetime('07/04/2005')

#creiamo la colonna timestamp per dare un numero di secondi dal 07-04-2005 che è la prima partita del dataset
df_cleaned['Timestamp_from_07-04-2005'] = df_cleaned['Date'].apply(lambda x: int(x.timestamp()) - reference_date.timestamp())


df_cleaned.insert(2, 'Timestamp_from_07-04-2005', df_cleaned.pop('Timestamp_from_07-04-2005'))

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#calcolo del Delta del rank tra player1 e player2


df_cleaned['Delta_Rank'] = df_cleaned['Rank_1'] - df_cleaned['Rank_2']

df_cleaned.insert(12, 'Delta_Rank', df_cleaned.pop('Delta_Rank'))

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#Calcolo del WinRate di Player_1 e player_2

df_cleaned = calculate_recent_winrate(df_cleaned)
df_cleaned.insert(7, 'Player1_WR', df_cleaned.pop('Player1_WR'))
df_cleaned.insert(9, 'Player2_WR', df_cleaned.pop('Player2_WR'))

#-----------------------------------------------------------------------------------------------------------------------------------------------------

# Eliminazione delle righe con valori nulli
df_cleaned = df_cleaned[~df_cleaned.isin([-1]).any(axis=1)]

#------------------------------------------------------------------------------------------------------------------------------------------------------
#salvataggio del dataset intermedio per migliroare l'efficienza
df_cleaned.to_csv(cleaned_file_path, index = False)





# Mostrare le prime righe del DataFrame pulito
print(df_cleaned.head())



