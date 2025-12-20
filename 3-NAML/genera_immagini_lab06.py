"""
Script per generare le immagini del Lab06 - ANN from scratch in JAX
Genera visualizzazioni per:
1. Training XOR con MSE vs Cross-Entropy
2. Decision boundary per problema make_circles
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

# Crea directory img se non esiste
os.makedirs('img', exist_ok=True)

print("=== Lab06: Generazione Immagini ANN from Scratch ===\n")

# ========================
# PARTE 1: XOR con ANN Semplice
# ========================

print("1. Training su dataset XOR...")

# Dataset XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Iperparametri architettura XOR
n1, n2, n3, n4 = 2, 4, 3, 1

# Inizializzazione parametri
np.random.seed(42)

W1 = jnp.array(np.random.randn(n1, n2))
b1 = jnp.zeros((n2, 1))

W2 = jnp.array(np.random.randn(n2, n3))
b2 = jnp.zeros((n3, 1))

W3 = jnp.array(np.random.randn(n3, n4))
b3 = jnp.zeros((n4, 1))

params_xor = [W1, b1, W2, b2, W3, b3]

# Forward pass ANN XOR
def ANN(x, params):
    """Forward pass della rete neurale XOR"""
    W1, b1, W2, b2, W3, b3 = params
    
    # Layer 1
    z1 = jnp.tanh(x @ W1 + b1.T)
    
    # Layer 2
    z2 = jnp.tanh(z1 @ W2 + b2.T)
    
    # Layer 3 con sigmoid
    output = 1.0 / (1.0 + jnp.exp(-(z2 @ W3 + b3.T)))
    
    return output

# Loss functions
def loss_quadratic(x, y, params):
    """MSE Loss"""
    y_pred = ANN(x, params)
    return jnp.mean((y - y_pred) ** 2)

def loss_crossentropy(x, y, params):
    """Binary Cross-Entropy Loss"""
    y_pred = ANN(x, params)
    return -jnp.mean(y * jnp.log(y_pred + 1e-10) + (1 - y) * jnp.log(1 - y_pred + 1e-10))

# JIT compilation
loss_quadratic_jit = jax.jit(loss_quadratic)
loss_crossentropy_jit = jax.jit(loss_crossentropy)

grad_quad_jit = jax.jit(jax.grad(loss_quadratic, argnums=2))
grad_xent_jit = jax.jit(jax.grad(loss_crossentropy, argnums=2))

# Training con Cross-Entropy
grad_function = grad_xent_jit
learning_rate = 0.1
epochs = 2000

history_mse = []
history_xent = []

for epoch in range(epochs):
    # Calcola gradiente
    grads = grad_function(inputs, outputs, params_xor)
    
    # Aggiorna parametri
    for i in range(len(params_xor)):
        params_xor[i] = params_xor[i] - learning_rate * grads[i]
    
    # Salva loss
    history_mse.append(float(loss_quadratic_jit(inputs, outputs, params_xor)))
    history_xent.append(float(loss_crossentropy_jit(inputs, outputs, params_xor)))

# Visualizzazione training XOR
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_mse, label='MSE', linewidth=2)
axes[0].plot(history_xent, label='Cross-Entropy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].set_title('Andamento Loss (scala lineare)', fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_mse, label='MSE', linewidth=2)
axes[1].plot(history_xent, label='Cross-Entropy', linewidth=2)
axes[1].set_yscale('log')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss (log scale)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].set_title('Andamento Loss (scala logaritmica)', fontsize=13)
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('img/lab06_xor_training.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvata: img/lab06_xor_training.png")
plt.close()

# Verifica accuratezza XOR
predictions = ANN(inputs, params_xor)
pred_binary = (predictions > 0.5).astype(int)
accuracy_xor = jnp.mean(pred_binary == outputs)
print(f"   Accuratezza XOR: {accuracy_xor * 100:.1f}%")

# ========================
# PARTE 2: Make Circles con MLP Generalizzato
# ========================

print("\n2. Training su dataset Make Circles...")

# Funzioni generalizzate per MLP
def init_layer_params(key, in_dim, out_dim):
    """Inizializza parametri di un layer"""
    w_key, b_key = jax.random.split(key)
    w = jax.random.normal(w_key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return w, b

def init_mlp_params(key, layer_sizes):
    """Inizializza parametri MLP completo"""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    
    for i in range(len(layer_sizes) - 1):
        w, b = init_layer_params(keys[i], layer_sizes[i], layer_sizes[i+1])
        params.append((w, b))
    
    return params

def sigmoid(x):
    """Funzione sigmoide"""
    return 1.0 / (1.0 + jnp.exp(-x))

def forward(params, x):
    """Forward pass generalizzato"""
    # Strati nascosti con tanh
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    
    # Ultimo strato con sigmoid
    W, B = params[-1]
    return sigmoid(x @ W + B)

def binary_cross_entropy(params, X, Y):
    """Binary cross-entropy loss"""
    y_pred = forward(params, X)
    return -jnp.mean(Y * jnp.log(y_pred + 1e-10) + (1 - Y) * jnp.log(1 - y_pred + 1e-10))

@jax.jit
def update(params, x, y, learning_rate):
    """Update parametri con gradient descent"""
    grad_fn = jax.grad(binary_cross_entropy)
    grads = grad_fn(params, x, y)
    
    # tree_map per aggiornare tutti i parametri
    updated_params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads
    )
    
    return updated_params

# Generazione dataset make_circles
X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Visualizzazione dataset
plt.figure(figsize=(6, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), 
            cmap='coolwarm', s=30, edgecolor='k', label='Train', alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.flatten(), 
            cmap='coolwarm', s=50, marker='x', linewidths=2, label='Test')
plt.title('Dataset Make Circles', fontsize=14)
plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/lab06_dataset_circles.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvata: img/lab06_dataset_circles.png")
plt.close()

# Training MLP
key = jax.random.PRNGKey(42)
layer_sizes = [2, 16, 1]
params = init_mlp_params(key, layer_sizes)
learning_rate = 0.01
epochs = 5000
batch_size = 64

num_batches = X_train.shape[0] // batch_size
train_losses = []

for epoch in range(epochs):
    # Shuffle dati
    key, subkey = jax.random.split(key)
    permutation = jax.random.permutation(subkey, X_train.shape[0])
    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]
    
    # Training su minibatch
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        
        params = update(params, x_batch, y_batch, learning_rate)
    
    # Salva loss test
    if epoch % 100 == 0:
        test_loss = float(binary_cross_entropy(params, X_test, y_test))
        train_losses.append(test_loss)

# Calcolo accuratezza
predictions = forward(params, X_test)
pred_binary = (predictions > 0.5).astype(int)
accuracy = float(jnp.mean(pred_binary == y_test))
print(f"   Accuratezza test: {accuracy * 100:.2f}%")

# Matrice di confusione
conf_mat = confusion_matrix(y_test.flatten(), pred_binary.flatten())
print(f"   Matrice confusione:\n{conf_mat}")

# Visualizzazione decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                      np.linspace(y_min, y_max, 300))
grid = np.column_stack([xx.ravel(), yy.ravel()])

Z = np.array(forward(params, grid))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 8))
# Contour riempito per decision boundary
contour = plt.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='coolwarm')
plt.colorbar(contour, label='Probabilità classe 1')

# Dati training
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), 
            cmap='coolwarm', s=30, edgecolor='k', linewidth=0.5, alpha=0.8)

# Dati test
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.flatten(), 
            cmap='coolwarm', s=80, marker='x', linewidths=2.5)

# Linea di decisione
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')

plt.title(f'Confine Decisionale (Accuratezza Test = {accuracy:.3f})', fontsize=14)
plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/lab06_decision_boundary.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvata: img/lab06_decision_boundary.png")
plt.close()

# Visualizzazione andamento loss durante training
plt.figure(figsize=(8, 5))
epochs_sampled = np.arange(0, epochs, 100)
plt.plot(epochs_sampled, train_losses, linewidth=2, marker='o', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Loss (Cross-Entropy)', fontsize=12)
plt.title('Andamento Loss Test durante Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('img/lab06_training_loss.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvata: img/lab06_training_loss.png")
plt.close()

print("\n=== Generazione completata! ===")
print(f"Immagini totali generate: 4")
print(f"  - lab06_xor_training.png")
print(f"  - lab06_dataset_circles.png")
print(f"  - lab06_decision_boundary.png")
print(f"  - lab06_training_loss.png")
