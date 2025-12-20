"""
Script per generare immagini per Lab07 - ANN Regression su California Housing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import jax
import time

# Configurazione matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

print("Caricamento dataset...")
data = pd.read_csv("note/Lab07/california_housing_train.csv")

# ===== IMMAGINE 1: Distribuzione median_house_value prima del filtro =====
print("Generando immagine 1: distribuzione target con outlier...")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data["median_house_value"], kde=True, ax=ax)
ax.set_title("Distribuzione median_house_value (con outlier a 500k)")
ax.set_xlabel("Median House Value ($)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("img/lab07_distribution_before_filter.png")
plt.close()

# Filtro outlier
data = data[data["median_house_value"] < 500000]

# ===== IMMAGINE 2: Distribuzione median_house_value dopo il filtro =====
print("Generando immagine 2: distribuzione target senza outlier...")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data["median_house_value"], kde=True, ax=ax)
ax.set_title("Distribuzione median_house_value (senza outlier)")
ax.set_xlabel("Median House Value ($)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("img/lab07_distribution_after_filter.png")
plt.close()

# ===== IMMAGINE 3: Mappa geografica con prezzi =====
print("Generando immagine 3: scatter geografico...")
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(data["longitude"], data["latitude"], 
                     c=data["median_house_value"], 
                     cmap="viridis", alpha=0.5, s=10)
ax.set_title("Distribuzione geografica delle case in California")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.colorbar(scatter, ax=ax, label="Median House Value ($)")
plt.tight_layout()
plt.savefig("img/lab07_geographic_scatter.png")
plt.close()

# ===== IMMAGINE 4: Heatmap correlazione =====
print("Generando immagine 4: matrice di correlazione...")
fig, ax = plt.subplots(figsize=(12, 10))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap="vlag_r", vmin=-1, vmax=1, 
            fmt=".2f", square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Matrice di Correlazione delle Feature")
plt.tight_layout()
plt.savefig("img/lab07_correlation_heatmap.png")
plt.close()

# ===== IMMAGINE 5: Violin plot prima della normalizzazione =====
print("Generando immagine 5: violin plot pre-normalizzazione...")
fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(data=data, ax=ax)
ax.set_title("Distribuzione feature (prima della normalizzazione)")
ax.set_ylabel("Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("img/lab07_violin_before_norm.png")
plt.close()

# Normalizzazione
data_mean = data.mean()
data_std = data.std()
data_normalized = (data - data_mean) / data_std

# ===== IMMAGINE 6: Violin plot dopo normalizzazione =====
print("Generando immagine 6: violin plot post-normalizzazione...")
fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(data=data_normalized, ax=ax)
ax.set_title("Distribuzione feature (dopo normalizzazione)")
ax.set_ylabel("Normalized Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("img/lab07_violin_after_norm.png")
plt.close()

# Preparazione dati per training
np.random.seed(0)
data_normalized_np = data_normalized.to_numpy()
np.random.shuffle(data_normalized_np)

fraction_validation = 0.2
num_train = int(data_normalized_np.shape[0] * (1 - fraction_validation))
x_train = data_normalized_np[:num_train, :-1]
y_train = data_normalized_np[:num_train, -1:]
x_valid = data_normalized_np[num_train:, :-1]
y_valid = data_normalized_np[num_train:, -1:]

print(f"Training samples: {x_train.shape[0]}")
print(f"Validation samples: {x_valid.shape[0]}")

# ===== Definizione del modello =====
def initialize_params(layers_size):
    """Inizializza parametri con Xavier/Glorot Normal"""
    params = []
    np.random.seed(42)
    for i in range(len(layers_size) - 1):
        n_in = layers_size[i]
        n_out = layers_size[i + 1]
        # Xavier/Glorot Normal
        coef = np.sqrt(2.0 / (n_in + n_out))
        W = coef * np.random.randn(n_out, n_in)
        b = np.zeros((n_out, 1))
        params.append([W, b])
    return params

def ann(x, params):
    """Forward pass della ANN con tanh"""
    layer = x.T  # (features, samples)
    for i, (W, b) in enumerate(params):
        layer = W @ layer + b
        if i < len(params) - 1:
            layer = jnp.tanh(layer)
    return layer.T

def loss(x, y, params):
    """Mean Squared Error loss"""
    y_pred = ann(x, params)
    error = y_pred - y
    return jnp.mean(error * error)

# ===== TRAINING FULL BATCH =====
print("\n=== Training Full Batch ===")
layers_size = [8, 20, 20, 1]
num_epochs = 2000
learning_rate = 1e-1

params = initialize_params(layers_size)

grad_loss = jax.jit(jax.grad(loss, argnums=2))
loss_jit = jax.jit(loss)

history_train_fb = []
history_valid_fb = []

t0 = time.time()
for epoch in range(num_epochs):
    grads = grad_loss(x_train, y_train, params)
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads
    )
    history_train_fb.append(float(loss_jit(x_train, y_train, params)))
    history_valid_fb.append(float(loss_jit(x_valid, y_valid, params)))
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: train={history_train_fb[-1]:.4f}, val={history_valid_fb[-1]:.4f}")

elapsed_fb = time.time() - t0
print(f"Tempo: {elapsed_fb:.2f}s")
print(f"Loss train: {history_train_fb[-1]:.4e}")
print(f"Loss validation: {history_valid_fb[-1]:.4e}")

# Salvo parametri full batch per il test
params_full_batch = params

# ===== IMMAGINE 7: Andamento loss full batch =====
print("\nGenerando immagine 7: andamento loss full batch...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.loglog(history_train_fb, label="Train", linewidth=2)
ax.loglog(history_valid_fb, label="Validation", linewidth=2)
ax.set_title("Andamento Loss - Full Batch Gradient Descent")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (log scale)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/lab07_loss_full_batch.png")
plt.close()

# ===== TRAINING MINI-BATCH SGD =====
print("\n=== Training Mini-Batch SGD ===")
params = initialize_params(layers_size)

num_epochs_sgd = 2000
learning_rate_max = 1e-1
learning_rate_min = 5e-2
learning_rate_decay = num_epochs_sgd
batch_size = 1000

history_train_sgd = []
history_valid_sgd = []

t0 = time.time()
for epoch in range(num_epochs_sgd):
    # Learning rate decrescente
    lr = max(learning_rate_min, 
             learning_rate_max * (1 - epoch / learning_rate_decay))
    
    # Shuffle dati
    n_samples = x_train.shape[0]
    perm_indices = np.random.permutation(n_samples)
    
    # Mini-batch updates
    for i in range(0, n_samples, batch_size):
        batch_indices = perm_indices[i : i + batch_size]
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        grads = grad_loss(x_batch, y_batch, params)
        params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, params, grads
        )
    
    # Fine epoca: calcolo loss
    history_train_sgd.append(float(loss_jit(x_train, y_train, params)))
    history_valid_sgd.append(float(loss_jit(x_valid, y_valid, params)))
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: train={history_train_sgd[-1]:.4f}, val={history_valid_sgd[-1]:.4f}, lr={lr:.4f}")

elapsed_sgd = time.time() - t0
print(f"Tempo: {elapsed_sgd:.2f}s")
print(f"Loss train: {history_train_sgd[-1]:.4e}")
print(f"Loss validation: {history_valid_sgd[-1]:.4e}")

# Salvo parametri SGD per il test
params_sgd = params

# ===== IMMAGINE 8: Andamento loss SGD =====
print("\nGenerando immagine 8: andamento loss SGD...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.loglog(history_train_sgd, label="Train", linewidth=2, alpha=0.7)
ax.loglog(history_valid_sgd, label="Validation", linewidth=2, alpha=0.7)
ax.set_title("Andamento Loss - Mini-Batch SGD")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (log scale)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/lab07_loss_sgd.png")
plt.close()

# ===== IMMAGINE 9: Confronto Full Batch vs SGD =====
print("\nGenerando immagine 9: confronto full batch vs SGD...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.loglog(history_train_fb, label="Full Batch - Train", linewidth=2, linestyle='--')
ax.loglog(history_valid_fb, label="Full Batch - Validation", linewidth=2, linestyle='--')
ax.loglog(history_train_sgd, label="SGD - Train", linewidth=2, alpha=0.7)
ax.loglog(history_valid_sgd, label="SGD - Validation", linewidth=2, alpha=0.7)
ax.set_title("Confronto: Full Batch vs Mini-Batch SGD")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (log scale)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/lab07_comparison_fb_sgd.png")
plt.close()

# ===== TEST SET EVALUATION =====
print("\n=== Valutazione su Test Set ===")
data_test = pd.read_csv("note/Lab07/california_housing_test.csv")
data_test = data_test[data_test["median_house_value"] < 500000]

# Normalizzazione con statistiche del training
data_test_norm = (data_test - data_mean) / data_std
data_test_np = data_test_norm.to_numpy()
x_test = data_test_np[:, :-1]
y_test_norm = data_test_np[:, -1:]

# Predizione (usando il modello SGD)
y_pred_norm = ann(x_test, params_sgd)

# Denormalizzazione per visualizzazione in dollari
mean_target = data_mean["median_house_value"]
std_target = data_std["median_house_value"]
y_pred = y_pred_norm * std_target + mean_target
y_test_actual = data_test["median_house_value"].values.reshape(-1, 1)

# ===== IMMAGINE 10: Scatter predetto vs vero =====
print("\nGenerando immagine 10: scatter test predictions...")
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y_test_actual, y_pred, alpha=0.3, s=10)
min_val = min(y_test_actual.min(), y_pred.min())
max_val = max(y_test_actual.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Median House Value ($)")
ax.set_ylabel("Predicted Median House Value ($)")
ax.set_title("Test Set: Predicted vs Actual")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
plt.tight_layout()
plt.savefig("img/lab07_test_scatter.png")
plt.close()

# ===== IMMAGINE 11: Joint plot con Seaborn =====
print("\nGenerando immagine 11: joint plot...")
test_df = pd.DataFrame({
    "Actual": y_test_actual.flatten(),
    "Predicted": y_pred.flatten()
})
g = sns.jointplot(data=test_df, x="Actual", y="Predicted", 
                  kind="scatter", alpha=0.3, height=10)
g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
g.fig.suptitle("Test Set: Joint Distribution", y=1.02)
plt.tight_layout()
plt.savefig("img/lab07_test_jointplot.png")
plt.close()

# Calcolo RMSE
error = y_pred.flatten() - y_test_actual.flatten()
rmse = np.sqrt(np.mean(error ** 2))
print(f"\nRMSE sul test set: ${rmse:,.2f}")
print(f"RMSE (in migliaia): ${rmse/1000:.2f}k")

print("\n✓ Tutte le immagini generate con successo!")
print("\nImmagini create:")
print("  1. lab07_distribution_before_filter.png - Distribuzione target con outlier")
print("  2. lab07_distribution_after_filter.png - Distribuzione target filtrato")
print("  3. lab07_geographic_scatter.png - Mappa geografica California")
print("  4. lab07_correlation_heatmap.png - Matrice di correlazione")
print("  5. lab07_violin_before_norm.png - Violin plot pre-normalizzazione")
print("  6. lab07_violin_after_norm.png - Violin plot post-normalizzazione")
print("  7. lab07_loss_full_batch.png - Andamento loss full batch")
print("  8. lab07_loss_sgd.png - Andamento loss SGD")
print("  9. lab07_comparison_fb_sgd.png - Confronto full batch vs SGD")
print(" 10. lab07_test_scatter.png - Scatter predictions vs actual")
print(" 11. lab07_test_jointplot.png - Joint plot con distribuzioni marginali")
