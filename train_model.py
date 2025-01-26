"""
Questo script esegue l'addestramento e la valutazione di un modello di classificazione basato su Random Forest
per prevedere l'esito di incontri di tennis utilizzando split temporali del dataset.

**Flusso del codice**:
1. **Caricamento e preparazione del dataset**:
    - I dati vengono ordinati cronologicamente in base alla colonna 'Date'.
    - Viene applicata una suddivisione temporale in base a finestre mobili definite da `window_size` (addestramento) e `step_size` (test).

2. **Definizione del modello**:
    - Viene utilizzato un classificatore Random Forest con parametri configurati per gestire la complessità del problema.

3. **Addestramento e valutazione**:
    - Per ogni split temporale:
        - Si addestra il modello sui dati di addestramento.
        - Si valuta il modello sui dati di test calcolando metriche come Accuracy, Precision, Recall e F1-Score.
        - Si stampa la matrice di confusione e si verifica l'importanza delle feature.

4. **Output e salvataggio**:
    - Il modello addestrato viene salvato in un file `.joblib` per utilizzi futuri.
    - Le metriche di performance e l'importanza delle feature vengono salvate in file CSV per ulteriori analisi.

**Output principali**:
- Modello salvato: `model.joblib`
- Importanza delle feature: `information_gain.csv`
- Confronto previsioni/valori reali: `results.csv`

**Nota**:
- Lo script gestisce il tempo come una variabile fondamentale per simulare un ambiente di previsione nel mondo reale.
- Alcune colonne non pertinenti al processo di addestramento sono escluse manualmente per migliorare l'efficienza e la performance del modello.
"""


from pathlib import Path

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Esempio di modello
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Directory radice del progetto
base_dir = Path(__file__).resolve().parent  # Risali di un livello

# Percorsi relativi
file_path = base_dir / "data" / "Finished" / "dropped_dataset.csv"
model_path = base_dir / "modello" / "model.joblib"
feature_importance_path = base_dir / "data" / "Results" /"information_gain.csv"
results_path = base_dir / "data" / "Results" / "results.csv"

df = pd.read_csv(file_path)
df_cleaned = df.copy()
# Assicurati che 'Date' sia in datetime
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%Y-%m-%d')

# Ordina i dati per 'Date'
df_cleaned = df_cleaned.sort_values('Date')


# Funzione aggiornata per eseguire gli split temporali
def time_series_split(df, window_size=3, step_size=3):
    """
    Divide il dataset in training e test set basati su finestre temporali, saltando gli split vuoti.

    :param df: il DataFrame da dividere
    :param window_size: la dimensione della finestra di addestramento in mesi
    :param step_size: la dimensione della finestra di test in mesi
    :return: una lista di tuple (train_set, test_set)
    """
    splits = []
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    # Inizializza la finestra di addestramento
    train_start = min_date
    train_end = train_start + pd.DateOffset(months=window_size)

    # Esegui l'iterazione per creare gli split
    while train_end < max_date:
        # Il periodo di test è successivo alla finestra di addestramento
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=step_size)

        # Estrai il set di addestramento (da train_start a train_end)
        train_set = df[(df['Date'] >= train_start) & (df['Date'] < train_end)]

        # Estrai il set di test (da test_start a test_end)
        test_set = df[(df['Date'] >= test_start) & (df['Date'] < test_end)]

        # Aggiungi lo split solo se il test set non è vuoto
        if not test_set.empty and not train_set.empty:
            splits.append((train_set, test_set))

        # Avanza la finestra temporale
        train_start = train_end
        train_end = train_start + pd.DateOffset(months=window_size)

    return splits


# Crea gli split
splits = time_series_split(df_cleaned)

# Verifica gli split
for i, (train_set, test_set) in enumerate(splits):
    print(f"Split {i + 1} -> Training Period: {train_set['Date'].min()} - {train_set['Date'].max()}, "
          f"Test Period: {test_set['Date'].min()} - {test_set['Date'].max()}")


# Liste per accumulare i valori delle metriche
accuracies = []
precisions = []
recalls = []
f1_scores = []


# Addestra e testa il modello solo sui split validi
for i, (train_set, test_set) in enumerate(splits):
    # Separa le caratteristiche (X) e il target (y)
    X_train = train_set.drop(columns=['Winner_encoded', 'Date', 'Odd_1','Odd_2', 'Court_Indoor', 'Court_Outdoor']) #'Winner' sia la colonna target
    y_train = train_set['Winner_encoded']  # Modifica con il target corretto
    X_test = test_set.drop(columns=['Winner_encoded', 'Date', 'Odd_1','Odd_2', 'Court_Indoor', 'Court_Outdoor',])
    y_test = test_set['Winner_encoded']

    model = RandomForestClassifier(
        n_estimators=300,  # Numero di alberi
        min_samples_split=5,  # Minimo numero di campioni per dividere un nodo
        min_samples_leaf=2,  # Minimo numero di campioni in un nodo foglia
        max_features='sqrt',  # Numero massimo di caratteristiche per ogni split
    )

    model.fit(X_train, y_train)

    # Testa il modello
    y_pred = model.predict(X_test)

    # Calcola le metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Aggiungi i valori delle metriche nelle liste
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Stampa le metriche
    print(f"Accuracy for Split {i + 1}: {accuracy:.4f}")
    print(f"Precision for Split {i + 1}: {precision:.4f}")
    print(f"Recall for Split {i + 1}: {recall:.4f}")
    print(f"F1 Score for Split {i + 1}: {f1:.4f}")

    # Confronta le prime 10 predizioni con i valori reali
    print(f"Prime 10 predizioni for Split{i + 1}:")
    print("Predizioni:", y_pred[:10])  # Prime 10 predizioni
    print("Valori rea:", y_test[:10].values)  # Prime 10 valori reali

# Calcola la media delle metriche
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1_score = np.mean(f1_scores)
# Calcola la mediana delle metriche
median_accuracy = np.median(accuracies)
median_precision = np.median(precisions)
median_recall = np.median(recalls)
median_f1_score = np.median(f1_scores)

# Stampa la media delle metriche
print(f"\nAverage Accuracy: {mean_accuracy:.4f}")

# Crea un DataFrame per ispezionare le predizioni
comparison_df = pd.DataFrame({
    'Valori Reali': y_test,
    'Predizioni': y_pred
})

# Aggiungi una colonna che mostra se la previsione è corretta
comparison_df['Corretto'] = comparison_df['Valori Reali'] == comparison_df['Predizioni']


# Conta le predizioni corrette
correct_predictions = (comparison_df['Corretto']).sum()
incorrect_predictions = len(comparison_df) - correct_predictions

print(f"Predizioni corrette for Split: {correct_predictions}")
print(f"Predizioni errate for Split: {incorrect_predictions}")

# Calcola la matrice di confusione
cm = confusion_matrix(y_test, y_pred)

# Crea un DataFrame per una visualizzazione migliore
cm_df = pd.DataFrame(cm, index=["Player 1", "Player 2"], columns=["Predicted Player 1", "Predicted Player 2"])



# Calcola le probabilità di previsione
y_pred_prob = model.predict_proba(X_test)

# Visualizza le probabilità per le prime 10 predizioni (solo per la classe 1)
print(f"Probabilità per le prime 10 predizioni (Classe 1) for Split")
print(y_pred_prob[:10, 1])  # Probabilità per la classe 1 (Player 2 vince)



# Stampa la matrice di confusione
print(f"Matrice di Confusionefor Split:")
print(cm_df)

# Ottieni l'importanza delle feature dal modello Random Forest
feature_importances = model.feature_importances_

# Associa l'importanza alle feature del dataset
feature_importances_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)



# Salva il modello in un file
dump(model, model_path)

print("Modello salvato correttamente come 'model.joblib'.")


feature_importances_df.to_csv(feature_importance_path, index=False)


comparison_df.to_csv(results_path, index=False)