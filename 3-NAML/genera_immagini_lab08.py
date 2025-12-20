import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import jax

# Enable double precision in JAX
jax.config.update("jax_enable_x64", True)

# Create output directory
import os
os.makedirs("img", exist_ok=True)

# ============================================================================
# PART 1: First-order optimization methods (from notebook 1)
# ============================================================================

# Define 1D function
f = lambda x: np.sin(x) * np.exp(-0.1 * x) + 0.1 * np.cos(np.pi * x)
a, b = 0, 10

def get_training_data(N, noise):
    np.random.seed(0)
    x = np.linspace(a, b, N)[:, None]
    y = f(x) + noise * np.random.randn(N, 1)
    return x, y

def initialize_params(layers_size):
    np.random.seed(0)
    params = list()
    for i in range(len(layers_size) - 1):
        W = np.random.randn(layers_size[i + 1], layers_size[i])
        b = np.zeros((layers_size[i + 1], 1))
        params.append(W)
        params.append(b)
    return params

def ANN(x, params):
    layer = (2 * x.T - (a + b)) / (b - a)
    num_layers = int(len(params) / 2 + 1)
    weights = params[0::2]
    biases = params[1::2]
    for i in range(num_layers - 1):
        layer = jnp.dot(weights[i], layer) - biases[i]
        if i < num_layers - 2:
            layer = jnp.tanh(layer)
    return layer.T

def loss(x, y, params):
    error = ANN(x, params) - y
    return jnp.mean(error * error)

# Training data
n_training_points = 100
noise = 0.05
xx, yy = get_training_data(n_training_points, noise)
layers_size = [1, 5, 5, 1]

loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.grad(loss, argnums=2))

# ============================================================================
# Gradient Descent (Full Batch)
# ============================================================================
params_gd = initialize_params(layers_size)
num_epochs = 2000
learning_rate = 1e-1
history_gd = [loss_jit(xx, yy, params_gd)]

for epoch in range(num_epochs):
    grads = grad_jit(xx, yy, params_gd)
    for i in range(len(params_gd)):
        params_gd[i] = params_gd[i] - learning_rate * grads[i]
    history_gd.append(loss_jit(xx, yy, params_gd))

# ============================================================================
# SGD with decay
# ============================================================================
params_sgd = initialize_params(layers_size)
num_epochs_sgd = 20000
learning_rate_max = 1e-1
learning_rate_min = 2e-2
learning_rate_decay = 10000
batch_size = 10
history_sgd = [loss_jit(xx, yy, params_sgd)]

for epoch in range(num_epochs_sgd):
    learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))
    idxs = np.random.choice(n_training_points, batch_size, replace=True)
    grads = grad_jit(xx[idxs, :], yy[idxs, :], params_sgd)
    for i in range(len(params_sgd)):
        params_sgd[i] = params_sgd[i] - learning_rate * grads[i]
    history_sgd.append(loss_jit(xx, yy, params_sgd))

# ============================================================================
# Momentum
# ============================================================================
params_momentum = initialize_params(layers_size)
alpha = 0.9
velocity = [0.0 for _ in range(len(params_momentum))]
history_momentum = [loss_jit(xx, yy, params_momentum)]

for epoch in range(num_epochs_sgd):
    learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))
    idxs = np.random.choice(n_training_points, batch_size, replace=True)
    grads = grad_jit(xx[idxs, :], yy[idxs, :], params_momentum)
    for i in range(len(params_momentum)):
        velocity[i] = alpha * velocity[i] - learning_rate * grads[i]
        params_momentum[i] = params_momentum[i] + velocity[i]
    history_momentum.append(loss_jit(xx, yy, params_momentum))

# ============================================================================
# AdaGrad
# ============================================================================
params_adagrad = initialize_params(layers_size)
cumulated_square_grad = [0.0 for i in range(len(params_adagrad))]
delta = 1e-7
learning_rate_ada = 1e-1
history_adagrad = [loss_jit(xx, yy, params_adagrad)]

for epoch in range(num_epochs_sgd):
    idxs = np.random.choice(n_training_points, batch_size, replace=True)
    grads = grad_jit(xx[idxs, :], yy[idxs, :], params_adagrad)
    for i in range(len(params_adagrad)):
        cumulated_square_grad[i] = cumulated_square_grad[i] + grads[i] * grads[i]
        params_adagrad[i] = params_adagrad[i] - learning_rate_ada / (delta + jnp.sqrt(cumulated_square_grad[i])) * grads[i]
    history_adagrad.append(loss_jit(xx, yy, params_adagrad))

# ============================================================================
# RMSProp
# ============================================================================
params_rmsprop = initialize_params(layers_size)
cumulated_square_grad_rms = [0.0 for i in range(len(params_rmsprop))]
decay_rate = 0.9
learning_rate_rms = 1e-3
batch_size_rms = 50
history_rmsprop = [loss_jit(xx, yy, params_rmsprop)]

for epoch in range(num_epochs_sgd):
    idxs = np.random.choice(n_training_points, batch_size_rms, replace=True)
    grads = grad_jit(xx[idxs, :], yy[idxs, :], params_rmsprop)
    for i in range(len(params_rmsprop)):
        cumulated_square_grad_rms[i] = decay_rate * cumulated_square_grad_rms[i] + (1 - decay_rate) * grads[i] * grads[i]
        params_rmsprop[i] = params_rmsprop[i] - learning_rate_rms / (delta + jnp.sqrt(cumulated_square_grad_rms[i])) * grads[i]
    history_rmsprop.append(loss_jit(xx, yy, params_rmsprop))

# ============================================================================
# Plot comparison of all optimizers
# ============================================================================
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Loss comparison
axs[0].loglog(history_gd[:2001], label='Full-batch GD', linewidth=2)
axs[0].loglog(history_sgd, label='SGD', linewidth=2)
axs[0].loglog(history_momentum, label='Momentum', linewidth=2)
axs[0].loglog(history_adagrad, label='AdaGrad', linewidth=2)
axs[0].loglog(history_rmsprop, label='RMSProp', linewidth=2)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Comparison of First-Order Optimizers')
axs[0].legend()
axs[0].grid(True, which="both", ls="-", alpha=0.2)

# Function approximation
x_fine = np.linspace(a, b, 200)[:, None]
axs[1].plot(x_fine, f(x_fine), 'k-', label='True function', linewidth=2)
axs[1].plot(xx, yy, 'ko', label='Training data', markersize=3, alpha=0.5)
axs[1].plot(x_fine, ANN(x_fine, params_gd), label='Full-batch GD', linewidth=1.5)
axs[1].plot(x_fine, ANN(x_fine, params_sgd), label='SGD', linewidth=1.5)
axs[1].plot(x_fine, ANN(x_fine, params_momentum), label='Momentum', linewidth=1.5)
axs[1].plot(x_fine, ANN(x_fine, params_adagrad), label='AdaGrad', linewidth=1.5)
axs[1].plot(x_fine, ANN(x_fine, params_rmsprop), label='RMSProp', linewidth=1.5)
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Function Approximation Results')
axs[1].legend()
axs[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("img/lab08_optimizer_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("✓ Generated: lab08_optimizer_comparison.png")

# ============================================================================
# Individual optimizer plots
# ============================================================================

# Full-batch GD
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(history_gd, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Full-Batch Gradient Descent')
ax.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("img/lab08_gd_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_gd_loss.png")

# SGD
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(history_sgd, 'g-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('SGD with Learning Rate Decay')
ax.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("img/lab08_sgd_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_sgd_loss.png")

# Momentum
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(history_momentum, 'r-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('SGD with Momentum (α=0.9)')
ax.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("img/lab08_momentum_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_momentum_loss.png")

# ============================================================================
# PART 2: Newton's Method (from notebook 2)
# ============================================================================

# Setup problem
n = 100
np.random.seed(0)
A = np.random.randn(n, n)
x_ex = np.random.randn(n)
b = A @ x_ex

def loss_newton(x):
    return jnp.sum(jnp.square(A @ x - b))

grad_newton = jax.grad(loss_newton)
hess_newton = jax.jacfwd(jax.jacrev(loss_newton))

loss_newton_jit = jax.jit(loss_newton)
grad_newton_jit = jax.jit(grad_newton)
hess_newton_jit = jax.jit(hess_newton)

# Newton's method with full Hessian
np.random.seed(0)
x_guess = np.random.randn(n)
x = x_guess.copy()
num_epochs_newton = 100
eps = 1e-8
history_newton = [loss_newton_jit(x)]

for epoch in range(num_epochs_newton):
    H = hess_newton_jit(x)
    G = grad_newton_jit(x)
    incr = np.linalg.solve(H, -G)
    x = x + incr
    history_newton.append(loss_newton_jit(x))
    if np.linalg.norm(incr) < eps:
        break

# Plot Newton convergence
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(history_newton, 'mo-', linewidth=2, markersize=8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Newton\'s Method Convergence')
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.axhline(y=1e-14, color='r', linestyle='--', label='Machine precision', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.savefig("img/lab08_newton_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_newton_convergence.png")

# ============================================================================
# PART 3: Regularization (from notebook 3)
# ============================================================================

# Load Auto MPG dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
data = pd.read_csv(url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
data = data.dropna()

# MPG distribution
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data["MPG"], kde=True, ax=ax)
ax.set_xlabel('Miles Per Gallon (MPG)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Fuel Efficiency (MPG)')
plt.tight_layout()
plt.savefig("img/lab08_mpg_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_mpg_distribution.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="vlag_r", vmin=-1, vmax=1, ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix of Auto MPG Features')
plt.tight_layout()
plt.savefig("img/lab08_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_correlation_heatmap.png")

# Normalization
data_mean = data.mean()
data_std = data.std()
data_normalized = (data - data_mean) / data_std

# Violin plot
fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(data=data_normalized, ax=ax)
ax.set_ylabel('Normalized Values')
ax.set_title('Distribution of Normalized Features')
plt.tight_layout()
plt.savefig("img/lab08_normalized_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_normalized_violin.png")

# Train-validation split
data_normalized_np = data_normalized.to_numpy()
np.random.seed(0)
np.random.shuffle(data_normalized_np)

fraction_validation = 0.2
num_train = int(data_normalized_np.shape[0] * (1 - fraction_validation))
x_train = data_normalized_np[:num_train, 1:]
y_train = data_normalized_np[:num_train, :1]
x_valid = data_normalized_np[num_train:, 1:]
y_valid = data_normalized_np[num_train:, :1]

# Initialize regularization functions
def initialize_params_reg(layers_size):
    np.random.seed(0)
    params = list()
    for i in range(len(layers_size) - 1):
        W = np.random.randn(layers_size[i + 1], layers_size[i]) * np.sqrt(2 / (layers_size[i + 1] + layers_size[i]))
        b = np.zeros((layers_size[i + 1], 1))
        params.append(W)
        params.append(b)
    return params

activation = jax.nn.relu

def ANN_reg(x, params):
    layer = x.T
    num_layers = int(len(params) / 2 + 1)
    weights = params[0::2]
    biases = params[1::2]
    for i in range(num_layers - 1):
        layer = weights[i] @ layer - biases[i]
        if i < num_layers - 2:
            layer = activation(layer)
    return layer.T

def MSE(x, y, params):
    error = ANN_reg(x, params) - y
    return jnp.mean(error * error)

def MSW(params):
    weights = params[0::2]
    weight_sum = 0
    num_weights = 0
    for w in weights:
        weight_sum += jnp.sum(w * w)
        num_weights += w.shape[0] * w.shape[1]
    return weight_sum / num_weights

def loss_reg(x, y, params, beta):
    mse_val = MSE(x, y, params)
    msw_val = MSW(params)
    return mse_val + beta * msw_val

# Training function
def train_regularization(penalization):
    layers_size = [7, 20, 20, 1]
    num_epochs = 5000
    learning_rate_max = 1e-1
    learning_rate_min = 5e-3
    learning_rate_decay = 1000
    batch_size = 100
    alpha = 0.9
    
    params = initialize_params_reg(layers_size)
    grad = jax.grad(loss_reg, argnums=2)
    MSE_jit = jax.jit(MSE)
    grad_jit = jax.jit(grad)
    
    n_samples = x_train.shape[0]
    velocity = [0.0 for i in range(len(params))]
    
    for epoch in range(num_epochs):
        learning_rate = max(learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay))
        idxs = np.random.choice(n_samples, batch_size)
        grads = grad_jit(x_train[idxs, :], y_train[idxs, :], params, penalization)
        
        for i in range(len(params)):
            velocity[i] = alpha * velocity[i] - learning_rate * grads[i]
            params[i] += velocity[i]
    
    return {
        "MSE_train": float(MSE(x_train, y_train, params)),
        "MSE_valid": float(MSE(x_valid, y_valid, params)),
        "MSW": float(MSW(params))
    }

# Grid search over beta values
results = []
pen_values = np.arange(0, 2.1, 0.25)
for beta in pen_values:
    print(f"Training for beta = {beta:.2f}...")
    result = train_regularization(beta)
    results.append({"beta": beta, **result})

hyper_tuning_df = pd.DataFrame(results)

# Plot hyperparameter tuning results
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# MSE train
axs[0].plot(hyper_tuning_df['beta'], hyper_tuning_df['MSE_train'], 'bo-', linewidth=2, markersize=8)
axs[0].set_xlabel('β (Penalization Parameter)')
axs[0].set_ylabel('Training MSE')
axs[0].set_title('Training MSE vs Regularization')
axs[0].grid(True, alpha=0.3)

# MSE validation
axs[1].plot(hyper_tuning_df['beta'], hyper_tuning_df['MSE_valid'], 'ro-', linewidth=2, markersize=8)
axs[1].set_xlabel('β (Penalization Parameter)')
axs[1].set_ylabel('Validation MSE')
axs[1].set_title('Validation MSE vs Regularization')
axs[1].grid(True, alpha=0.3)

# MSW
axs[2].plot(hyper_tuning_df['beta'], hyper_tuning_df['MSW'], 'go-', linewidth=2, markersize=8)
axs[2].set_xlabel('β (Penalization Parameter)')
axs[2].set_ylabel('Mean Squared Weights (MSW)')
axs[2].set_title('MSW vs Regularization')
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("img/lab08_hyperparameter_tuning.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_hyperparameter_tuning.png")

# Tikhonov L-curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(hyper_tuning_df['MSW'], hyper_tuning_df['MSE_train'], 'mo-', linewidth=2, markersize=8)
for i, beta in enumerate(hyper_tuning_df['beta']):
    ax.annotate(f'β={beta:.2f}', 
                (hyper_tuning_df['MSW'].iloc[i], hyper_tuning_df['MSE_train'].iloc[i]),
                textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
ax.set_xlabel('MSW (Mean Squared Weights)')
ax.set_ylabel('Training MSE')
ax.set_title('Tikhonov L-Curve: Trade-off between Fit and Regularization')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/lab08_tikhonov_lcurve.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated: lab08_tikhonov_lcurve.png")

print("\n" + "="*60)
print("All Lab08 images generated successfully!")
print("="*60)
print("\nGenerated images:")
print("  1. lab08_optimizer_comparison.png - Comparison of all optimizers")
print("  2. lab08_gd_loss.png - Full-batch gradient descent")
print("  3. lab08_sgd_loss.png - SGD with decay")
print("  4. lab08_momentum_loss.png - Momentum optimizer")
print("  5. lab08_newton_convergence.png - Newton's method")
print("  6. lab08_mpg_distribution.png - MPG distribution")
print("  7. lab08_correlation_heatmap.png - Feature correlations")
print("  8. lab08_normalized_violin.png - Normalized features")
print("  9. lab08_hyperparameter_tuning.png - Beta hyperparameter analysis")
print(" 10. lab08_tikhonov_lcurve.png - Tikhonov L-curve")
