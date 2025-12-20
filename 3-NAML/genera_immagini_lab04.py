"""
Script per generare le immagini per la lezione Lab04
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import os

# Crea cartella img se non esiste
os.makedirs('img', exist_ok=True)

print("Caricamento dataset MovieLens...")
# Carica il dataset
dataset = pd.read_csv(
    "note/Lab04/movielens.csv",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)

n_people = np.unique(dataset.user_id).size
n_movies = np.unique(dataset.item_id).size
n_ratings = len(dataset)

print(f"{n_people} people, {n_movies} movies, {n_ratings} ratings")

# Shuffling
np.random.seed(1)
idxs = np.arange(n_ratings)
np.random.shuffle(idxs)
rows_dupes = dataset.user_id[idxs]
cols_dupes = dataset.item_id[idxs]
vals = dataset.rating[idxs]

# Compatta indici
_, rows = np.unique(rows_dupes, return_inverse=True)
_, cols = np.unique(cols_dupes, return_inverse=True)

# Split train/test
training_data = int(0.8 * n_ratings)
rows_train = rows[:training_data]
cols_train = cols[:training_data]
vals_train = vals[:training_data]
rows_test = rows[training_data:]
cols_test = cols[training_data:]
vals_test = vals[training_data:]

# Crea matrice
X_sparse = csr_matrix((vals_train, (rows_train, cols_train)), shape=(n_people, n_movies))
X_full = X_sparse.toarray()

print("Calcolo predittore banale...")
# Predittore banale
average_rating = np.zeros(n_people)
for i in range(n_people):
    mask = (rows_train == i)
    average_rating[i] = vals_train[mask].mean()

vals_trivial = average_rating[rows_test]
errors_trivial = vals_test - vals_trivial
RMSE_trivial = np.sqrt(np.mean(errors_trivial**2))
rho_trivial = pearsonr(vals_test, vals_trivial)[0]

print(f"Baseline - RMSE: {RMSE_trivial:.3f}, rho: {rho_trivial:.3f}")

print("Esecuzione algoritmo SVT...")
# Algoritmo SVT
n_max_iter = 50  # Ridotto per velocità
threshold = 100.0
increment_tol = 1e-6

RMSE_list = []
rho_list = []

A = X_full.copy()

for i in range(n_max_iter):
    A_old = A.copy()
    
    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Hard Thresholding
    S_thresholded = np.where(S > threshold, S, 0)
    
    # Ricostruzione
    A = U @ np.diag(S_thresholded) @ Vt
    
    # Imponi valori di training
    A[rows_train, cols_train] = vals_train
    
    # Incremento
    increment = np.linalg.norm(A - A_old, ord='fro')
    
    # Predizioni
    vals_pred = A[rows_test, cols_test]
    
    # Metriche
    errors = vals_test - vals_pred
    RMSE = np.sqrt(np.mean(errors**2))
    rho = pearsonr(vals_test, vals_pred)[0]
    
    RMSE_list.append(RMSE)
    rho_list.append(rho)
    
    if (i+1) % 10 == 0:
        print(f"Iter {i+1:02d} | RMSE: {RMSE:.3f} | rho: {rho:.3f}")
    
    if increment < increment_tol:
        break

print("\nGenerazione grafici...")

# GRAFICO 1: RMSE e Correlazione SVT
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(range(1, len(RMSE_list)+1), RMSE_list, 'b-', linewidth=2, label='SVT Algorithm')
ax[0].hlines(RMSE_trivial, 1, len(RMSE_list), 
             colors='red', linestyles='dashed', linewidth=2,
             label='Trivial Predictor')
ax[0].set_xlabel('Iteration', fontsize=12)
ax[0].set_ylabel('RMSE', fontsize=12)
ax[0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
ax[0].legend(fontsize=11)
ax[0].grid(True, alpha=0.3)
ax[0].set_xlim(1, len(RMSE_list))

ax[1].plot(range(1, len(rho_list)+1), rho_list, 'g-', linewidth=2, label='SVT Algorithm')
ax[1].hlines(rho_trivial, 1, len(rho_list), 
             colors='red', linestyles='dashed', linewidth=2,
             label='Trivial Predictor')
ax[1].set_xlabel('Iteration', fontsize=12)
ax[1].set_ylabel('Pearson Correlation (ρ)', fontsize=12)
ax[1].set_title('Correlation Coefficient', fontsize=14, fontweight='bold')
ax[1].legend(fontsize=11)
ax[1].grid(True, alpha=0.3)
ax[1].set_xlim(1, len(rho_list))

plt.tight_layout()
plt.savefig('img/lab04_svt_performance.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_svt_performance.png")
plt.close()

# GRAFICO 2: Convergenza RMSE dettagliata
fig, ax = plt.subplots(figsize=(10, 6))
iterations = range(1, len(RMSE_list)+1)
ax.plot(iterations, RMSE_list, 'b-', linewidth=2.5, marker='o', 
        markersize=4, markevery=5, label='SVT RMSE')
ax.axhline(y=RMSE_trivial, color='red', linestyle='--', linewidth=2, 
           label=f'Baseline RMSE = {RMSE_trivial:.3f}')
ax.fill_between(iterations, RMSE_trivial, RMSE_list, 
                where=np.array(RMSE_list) < RMSE_trivial, 
                alpha=0.3, color='green', label='SVT migliore')
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('RMSE', fontsize=13)
ax.set_title('Convergenza Algoritmo SVT - Root Mean Squared Error', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('img/lab04_svt_convergence.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_svt_convergence.png")
plt.close()

# GRAFICO 3: Distribuzione valutazioni
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Istogramma valutazioni
axes[0].hist(vals_train, bins=5, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Rating', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribuzione delle Valutazioni (Training Set)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Scatter plot predizioni vs reali
n_sample = min(1000, len(vals_test))
sample_indices = np.random.choice(len(vals_test), n_sample, replace=False)

# Converti pandas Series a numpy array per evitare problemi di indicizzazione
vals_test_arr = vals_test.to_numpy() if hasattr(vals_test, 'to_numpy') else np.array(vals_test)
vals_trivial_arr = vals_trivial.to_numpy() if hasattr(vals_trivial, 'to_numpy') else np.array(vals_trivial)

axes[1].scatter(vals_test_arr[sample_indices], vals_trivial_arr[sample_indices], 
                alpha=0.5, s=20, color='coral')
axes[1].plot([1, 5], [1, 5], 'k--', linewidth=2, label='Predizione perfetta')
axes[1].set_xlabel('Valutazione Reale', fontsize=12)
axes[1].set_ylabel('Valutazione Predetta (Baseline)', fontsize=12)
axes[1].set_title('Predizioni Baseline vs Valori Reali', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0.5, 5.5)
axes[1].set_ylim(0.5, 5.5)

plt.tight_layout()
plt.savefig('img/lab04_data_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_data_analysis.png")
plt.close()

print("\n✅ Tutte le immagini sono state generate con successo!")
print(f"   Totale iterazioni SVT: {len(RMSE_list)}")
print(f"   RMSE finale: {RMSE_list[-1]:.4f}")
print(f"   Correlazione finale: {rho_list[-1]:.4f}")
