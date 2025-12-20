"""
Script per generare le immagini del Lab05: Gradient Descent, SGD, SVR e SVM
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("Inizializzazione...")

# ============================================================================
# FUNZIONI BENCHMARK
# ============================================================================

@jax.jit
def rastrigin(x):
    """Funzione di Rastrigin (multi-modale)"""
    return 10 * x.size + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x)) + 1e-10

@jax.jit
def ackley(x):
    """Funzione di Ackley (non convessa)"""
    a = 20
    b = 0.2
    c = 2 * jnp.pi
    sum1 = jnp.sum(x**2)
    sum2 = jnp.sum(jnp.cos(c * x))
    return (
        -a * jnp.exp(-b * jnp.sqrt(sum1 / x.size))
        - jnp.exp(sum2 / x.size)
        + a
        + jnp.exp(1)
    )

quadratic_A = jnp.array([[3.0, 0.5], [0.5, 1.0]])
quadratic_b = jnp.array([-1.0, 2.0])
quadratic_c = jnp.dot(quadratic_b, jnp.linalg.solve(quadratic_A, quadratic_b)) / 2

@jax.jit
def quadratic(x):
    """Funzione quadratica (convessa)"""
    return (
        0.5 * jnp.dot(x.T, jnp.dot(quadratic_A, x))
        + jnp.dot(quadratic_b, x)
        + quadratic_c
    )

# ============================================================================
# ALGORITMI DI OTTIMIZZAZIONE
# ============================================================================

def gradient_descent(grad_func, x0, lr=0.01, tol=1e-6, max_iter=1000):
    """Gradient Descent con passo fisso"""
    x = jnp.copy(x0)
    path = [x]
    
    for _ in range(max_iter):
        g = grad_func(x)
        x = x - lr * g
        path.append(x)
        
        if jnp.linalg.norm(g) < tol:
            break
    
    return x, path

def gradient_descent_backtracking(func, grad_func, x0, alpha=0.3, beta=0.8, tol=1e-6, max_iter=100):
    """Gradient Descent con backtracking"""
    x = jnp.copy(x0)
    path = [x]
    
    for _ in range(max_iter):
        g = grad_func(x)
        
        # Backtracking line search
        t = 1.0
        fx = func(x)
        grad_norm_sq = jnp.dot(g, g)
        
        while func(x - t * g) > fx - alpha * t * grad_norm_sq:
            t = beta * t
        
        x = x - t * g
        path.append(x)
        
        if jnp.linalg.norm(g) < tol:
            break
    
    return x, path

# ============================================================================
# GENERAZIONE IMMAGINI: OTTIMIZZAZIONE 2D
# ============================================================================

print("\n=== Generazione immagini ottimizzazione 2D ===")

x0 = jnp.array([4.0, 4.0])

# Test su Rastrigin
print("Test su funzione Rastrigin...")
grad_rastrigin = jax.jit(jax.grad(rastrigin))
x_gd, path_gd = gradient_descent(grad_rastrigin, x0, lr=0.01, max_iter=200)
x_gd_bt, path_gd_bt = gradient_descent_backtracking(rastrigin, grad_rastrigin, x0, max_iter=100)

# Visualizzazione
x_vals = jnp.linspace(-5, 5, 100)
y_vals = jnp.linspace(-5, 5, 100)
X, Y = jnp.meshgrid(x_vals, y_vals)
Z = jnp.array([[rastrigin(jnp.array([x, y])) for x in x_vals] for y in y_vals])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot contour
cs = axs[0].contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(cs, ax=axs[0])
axs[0].contour(X, Y, Z, colors="white", linewidths=0.5, alpha=0.3)

# Plot traiettorie
path_gd = jnp.array(path_gd)
path_gd_bt = jnp.array(path_gd_bt)
axs[0].plot(path_gd[:, 0], path_gd[:, 1], "r.-", label="GD (lr=0.01)", markersize=3)
axs[0].plot(path_gd_bt[:, 0], path_gd_bt[:, 1], ".-", color="orange", label="GD + backtracking", markersize=3)
axs[0].set_title("Rastrigin Function - Optimization Paths", fontsize=14, fontweight='bold')
axs[0].set_xlabel("x", fontsize=12)
axs[0].set_ylabel("y", fontsize=12)
axs[0].set_xlim([-5, 5])
axs[0].set_ylim([-5, 5])
axs[0].legend(fontsize=10)

# Plot convergenza
axs[1].semilogy([rastrigin(x) for x in path_gd], "ro-", label="GD (lr=0.01)", markersize=4)
axs[1].semilogy([rastrigin(x) for x in path_gd_bt], "o-", color="orange", label="GD + backtracking", markersize=4)
axs[1].set_xlabel("Iteration", fontsize=12)
axs[1].set_ylabel("Function Value (log scale)", fontsize=12)
axs[1].set_title("Convergence Comparison", fontsize=14, fontweight='bold')
axs[1].legend(fontsize=10)
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/lab05_rastrigin_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab05_rastrigin_optimization.png")
plt.close()

# Test su Ackley
print("Test su funzione Ackley...")
grad_ackley = jax.jit(jax.grad(ackley))
x_gd, path_gd = gradient_descent(grad_ackley, x0, lr=0.1, max_iter=200)
x_gd_bt, path_gd_bt = gradient_descent_backtracking(ackley, grad_ackley, x0, max_iter=100)

Z = jnp.array([[ackley(jnp.array([x, y])) for x in x_vals] for y in y_vals])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

cs = axs[0].contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(cs, ax=axs[0])
axs[0].contour(X, Y, Z, colors="white", linewidths=0.5, alpha=0.3)

path_gd = jnp.array(path_gd)
path_gd_bt = jnp.array(path_gd_bt)
axs[0].plot(path_gd[:, 0], path_gd[:, 1], "r.-", label="GD (lr=0.1)", markersize=3)
axs[0].plot(path_gd_bt[:, 0], path_gd_bt[:, 1], ".-", color="orange", label="GD + backtracking", markersize=3)
axs[0].set_title("Ackley Function - Optimization Paths", fontsize=14, fontweight='bold')
axs[0].set_xlabel("x", fontsize=12)
axs[0].set_ylabel("y", fontsize=12)
axs[0].set_xlim([-5, 5])
axs[0].set_ylim([-5, 5])
axs[0].legend(fontsize=10)

axs[1].semilogy([ackley(x) for x in path_gd], "ro-", label="GD (lr=0.1)", markersize=4)
axs[1].semilogy([ackley(x) for x in path_gd_bt], "o-", color="orange", label="GD + backtracking", markersize=4)
axs[1].set_xlabel("Iteration", fontsize=12)
axs[1].set_ylabel("Function Value (log scale)", fontsize=12)
axs[1].set_title("Convergence Comparison", fontsize=14, fontweight='bold')
axs[1].legend(fontsize=10)
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/lab05_ackley_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab05_ackley_optimization.png")
plt.close()

# ============================================================================
# REGRESSIONE LINEARE CON SGD
# ============================================================================

print("\n=== Generazione immagini regressione lineare ===")

@jax.jit
def model(theta, x):
    return theta[0] + theta[1] * x

@jax.jit
def mse_loss(theta, x, y):
    y_pred = model(theta, x)
    return jnp.mean((y - y_pred) ** 2)

grad_mse = jax.jit(jax.grad(mse_loss))

@jax.jit
def sgd_update(theta, x_batch, y_batch, learning_rate):
    gradients = grad_mse(theta, x_batch, y_batch)
    return theta - learning_rate * gradients

# Genera dati
np.random.seed(0)
N = 200
x_data = np.random.uniform(size=(N,)) * 10
y_data = 1.5 * x_data + 3 + np.random.normal(size=(N,))

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

# Training con SGD
theta = jnp.array([0.0, 0.0])
key = jax.random.PRNGKey(0)
epochs = 100
batch_size = 10
learning_rate = 0.01

for epoch in range(epochs):
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, len(X_train))
    
    for i in range(0, len(X_train), batch_size):
        batch_idx = perm[i:i + batch_size]
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        theta = sgd_update(theta, x_batch, y_batch, learning_rate)

# Visualizzazione
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.scatter(X_train, y_train, color='blue', alpha=0.6, s=30, label='Training data')
ax.scatter(X_test, y_test, color='green', marker='x', s=50, label='Test data')

x_plot = jnp.linspace(0, 10, 100)
y_plot = model(theta, x_plot)
ax.plot(x_plot, y_plot, 'r-', linewidth=2, label=f'Learned: y = {theta[0]:.2f} + {theta[1]:.2f}x')

# Vera relazione
y_true = 1.5 * x_plot + 3
ax.plot(x_plot, y_true, 'k--', linewidth=2, alpha=0.5, label='True: y = 3.0 + 1.5x')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Linear Regression with SGD', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

mse_test = jnp.mean((y_test - model(theta, X_test)) ** 2)
ax.text(0.5, 9.5, f'Test MSE: {mse_test:.3f}', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('img/lab05_linear_regression_sgd.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab05_linear_regression_sgd.png")
plt.close()

# ============================================================================
# SUPPORT VECTOR REGRESSION (SVR)
# ============================================================================

print("\n=== Generazione immagini SVR ===")

class SVR:
    def __init__(self, epsilon=0.1, lmbda=1.0):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.w = None
    
    def loss(self, params, X, y):
        w_slope = params[:-1]
        bias = params[-1]
        y_pred = jnp.dot(X, w_slope) + bias
        errors = jnp.abs(y_pred - y)
        epsilon_loss = jnp.maximum(0, errors - self.epsilon)
        reg = self.lmbda * jnp.sum(params ** 2)
        return jnp.mean(epsilon_loss) + reg
    
    def train(self, X, y, lr=1e-2, max_iter=1000):
        n_features = X.shape[1]
        self.w = jnp.zeros(n_features + 1)
        grad_loss = jax.jit(jax.grad(self.loss))
        
        @jax.jit
        def step(w, X, y):
            return w - lr * grad_loss(w, X, y)
        
        for _ in range(max_iter):
            self.w = step(self.w, X, y)
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        w_slope = self.w[:-1]
        bias = self.w[-1]
        return jnp.dot(X, w_slope) + bias

# Genera dati
np.random.seed(0)
X = np.random.uniform(0, 10, size=(100, 1))
y = 2.5 * X.flatten() + 1.0 + np.random.normal(0, 1, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

# Training
svr = SVR(epsilon=1.0, lmbda=0.1)
svr.train(X_train, y_train, lr=1e-2, max_iter=1000)

# Visualizzazione
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.scatter(X_train, y_train, color='blue', alpha=0.6, s=30, label='Training data')
ax.scatter(X_test, y_test, color='green', marker='x', s=50, label='Test data')

x_range = jnp.linspace(0, 10, 100)
y_pred_line = svr.predict(x_range)
ax.plot(x_range, y_pred_line, 'r-', linewidth=2, label='SVR prediction')
ax.fill_between(x_range, y_pred_line - svr.epsilon, y_pred_line + svr.epsilon,
                 color='r', alpha=0.15, label=f'ε-tube (ε={svr.epsilon})')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Support Vector Regression (ε={svr.epsilon}, λ={svr.lmbda})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

y_pred_test = svr.predict(X_test)
mse_test = jnp.mean((y_test - y_pred_test) ** 2)
ax.text(0.5, 26, f'Test MSE: {mse_test:.3f}', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('img/lab05_svr.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab05_svr.png")
plt.close()

# ============================================================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================================================

print("\n=== Generazione immagini SVM ===")

class SVM:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.w = None
    
    def loss(self, params, X, y):
        w_slope = params[:-1].reshape(-1, 1)
        bias = params[-1]
        decision = jnp.dot(X, w_slope).flatten() + bias
        hinge = jnp.maximum(0, 1 - y * decision)
        reg = self.lmbda * jnp.sum(params ** 2)
        return jnp.mean(hinge) + reg
    
    def train(self, X, y, lr=1e-2, max_iter=1000):
        n_features = X.shape[1]
        self.w = jnp.zeros(n_features + 1)
        grad_loss = jax.jit(jax.grad(self.loss))
        
        @jax.jit
        def step(w, X, y):
            return w - lr * grad_loss(w, X, y)
        
        for _ in range(max_iter):
            self.w = step(self.w, X, y)
    
    def predict(self, X):
        w_slope = self.w[:-1].reshape(-1, 1)
        bias = self.w[-1]
        decision = jnp.dot(X, w_slope).flatten() + bias
        return jnp.sign(decision)

# Genera dati
np.random.seed(42)
X = np.random.uniform(0, 10, size=(100, 2))
y = np.where(X.sum(axis=1) > 10, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

# Training
svm = SVM(lmbda=0.001)
svm.train(X_train, y_train, lr=1e-1, max_iter=5000)

# Valutazione
y_pred_test = svm.predict(X_test)
accuracy = jnp.mean(y_pred_test == y_test)

# Visualizzazione
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Meshgrid per decision boundary
t = jnp.linspace(0, 10, 200)
xx1, xx2 = jnp.meshgrid(t, t)
xx = jnp.stack([xx1.flatten(), xx2.flatten()], axis=1)
yy = svm.predict(xx)

# Plot decision regions
ax.contourf(xx1, xx2, yy.reshape(xx1.shape), alpha=0.2, levels=[-1, 0, 1], colors=['blue', 'red'])

# Plot data points
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
           c='red', marker='o', s=50, edgecolors='k', label='Class +1 (train)', alpha=0.7)
ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 
           c='blue', marker='o', s=50, edgecolors='k', label='Class -1 (train)', alpha=0.7)
ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
           c='red', marker='x', s=100, linewidths=2, label='Class +1 (test)')
ax.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], 
           c='blue', marker='x', s=100, linewidths=2, label='Class -1 (test)')

# Decision boundary (contour at 0)
ax.contour(xx1, xx2, yy.reshape(xx1.shape), levels=[0], colors='black', linewidths=2)

ax.set_xlabel('X₁', fontsize=12)
ax.set_ylabel('X₂', fontsize=12)
ax.set_title(f'SVM Binary Classification (λ={svm.lmbda})\nTest Accuracy: {accuracy:.2%}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

plt.tight_layout()
plt.savefig('img/lab05_svm_classification.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab05_svm_classification.png")
plt.close()

print("\n✅ Tutte le immagini Lab05 sono state generate con successo!")
print("   - lab05_rastrigin_optimization.png")
print("   - lab05_ackley_optimization.png")
print("   - lab05_linear_regression_sgd.png")
print("   - lab05_svr.png")
print("   - lab05_svm_classification.png")
