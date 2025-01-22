from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

#Directory radice del progetto
base_dir = Path(__file__).resolve().parent.parent  # Risali di un livello

# Percorsi relativi
file_path = base_dir / "data" / "Finished" / "dropped_dataset.csv"

data = pd.read_csv(file_path, low_memory=False)

data = data.drop(columns=["Date"])


# Scatter plot: Rank_1 vs Rank_2
plt.scatter(data['Rank_1'], data['Rank_2'], s=50, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Rank_1 e Rank_2")
plt.xlabel("Rank_1")
plt.ylabel("Rank_2")
plt.grid(True)
plt.show()


# Scatter plot: Prob_1_norm vs Prob_2_norm
plt.scatter(data['Prob_1_norm'], data['Prob_2_norm'], s=50, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Prob_1_norm e Prob_2_norm")
plt.xlabel("Prob_1_norm")
plt.ylabel("Prob_2_norm")
plt.grid(True)
plt.show()


# Scatter plot: Odd_1 vs Odd_2
plt.scatter(data['Odd_1'], data['Odd_2'], s=50, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Odd_1 e Odd_2")
plt.xlabel("Odd_1")
plt.ylabel("Odd_2")
plt.grid(True)
plt.show()

# Scatter plot: Pts_1 vs Pts_2 con dimensione basata su Best of
plt.scatter(data['Pts_1'], data['Pts_2'], s=data['Best of'] * 20, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Pts_1 e Pts_2 (Dimensione = Best of)")
plt.xlabel("Pts_1")
plt.ylabel("Pts_2")
plt.grid(True)
plt.show()

# --- 1. Applica il clustering agglomerativo ---
agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)  # Auto-detect cluster
labels = agg_clustering.fit_predict(data)

# --- 2. Distanze gerarchiche (dendrogramma) ---
linked = linkage(data, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode='lastp', p=12, show_leaf_counts=True)
plt.title("Dendrogramma")
plt.xlabel("Campioni")
plt.ylabel("Distanza")
plt.show()

# --- 3. Identifica gli outlier ---
# Gli outlier possono essere considerati punti isolati in piccoli cluster
# Troviamo la dimensione di ciascun cluster
from collections import Counter
cluster_sizes = Counter(labels)

print("Dimensione dei cluster:", cluster_sizes)

# Identifica cluster molto piccoli (soglia: cluster con meno di 3 punti)
small_clusters = [cluster for cluster, size in cluster_sizes.items() if size <= 2]

# Trova gli indici dei punti in questi cluster
outliers = np.where(np.isin(labels, small_clusters))[0]

print(f"Outlier trovati: {len(outliers)}")
print(f"Indici degli outlier: {outliers}")


# Scatter plot: Rank_1 vs Rank_2
plt.scatter(data['Rank_1'], data['Rank_2'], s=50, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Rank_1 e Rank_2")
plt.xlabel("Rank_1")
plt.ylabel("Rank_2")
plt.grid(True)
plt.show()



# Scatter plot: Prob_1_norm vs Prob_2_norm
plt.scatter(data['Prob_1_norm'], data['Prob_2_norm'], s=50, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Prob_1_norm e Prob_2_norm")
plt.xlabel("Prob_1_norm")
plt.ylabel("Prob_2_norm")
plt.grid(True)
plt.show()



# Scatter plot: Odd_1 vs Odd_2
plt.scatter(data['Odd_1'], data['Odd_2'], s=50, alpha=0.7, edgecolors='k')
plt.title("Relazione tra Odd_1 e Odd_2")
plt.xlabel("Odd_1")
plt.ylabel("Odd_2")
plt.grid(True)
plt.show()

