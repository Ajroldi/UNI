import os
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(__file__), 'img')

def ensure_img_dir():
    os.makedirs(IMG_DIR, exist_ok=True)

def figura_vista_geometrica_separazione_circolare():
    # Generate circular dataset: inside circle (+1, red), outside (-1, blue)
    rng = np.random.default_rng(42)
    n = 600
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    R = 1.0
    r2 = np.sum(X**2, axis=1)
    y = (r2 <= R**2).astype(int)  # 1 inside, 0 outside

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c='#d62728', label='Classe +1 (dentro cerchio)')
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c='#1f77b4', label='Classe -1 (fuori cerchio)')

    circle = plt.Circle((0, 0), R, color='gray', fill=False, linestyle='--', linewidth=1.5)
    ax.add_artist(circle)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Vista geometrica: separazione circolare in ℝ²')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, alpha=0.2)
    out = os.path.join(IMG_DIR, 'lez12_vista_geometrica_separazione_circolare.png')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def figura_kernel_matrix_heatmap():
    # Create small synthetic dataset and compute RBF kernel matrix heatmap
    rng = np.random.default_rng(7)
    n = 80
    X = rng.normal(0, 1, size=(n, 2))
    # Two clusters for visible structure
    X[: n//2] += np.array([2.0, 0.0])
    X[n//2 :] += np.array([-2.0, 0.0])

    def rbf_kernel(X, gamma=0.5):
        # Compute pairwise squared distances efficiently
        sq = np.sum(X**2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2 * (X @ X.T)
        K = np.exp(-gamma * D2)
        return K

    K = rbf_kernel(X, gamma=0.7)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(K, cmap='viridis', interpolation='nearest')
    ax.set_title('Heatmap matrice di Gram (kernel RBF)')
    ax.set_xlabel('Indice campione j')
    ax.set_ylabel('Indice campione i')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('K(xᵢ, xⱼ)')
    out = os.path.join(IMG_DIR, 'lez12_kernel_matrix_heatmap.png')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def figura_metodo_delle_potenze_convergenza():
    # Demonstrate power method convergence on a stochastic matrix
    # Build a simple Google-like matrix G = αP + (1-α)ee^T/n
    rng = np.random.default_rng(21)
    n = 20
    P = rng.uniform(0, 1, size=(n, n))
    P /= P.sum(axis=0, keepdims=True)  # column-stochastic
    alpha = 0.85
    E = np.ones((n, n)) / n
    G = alpha * P + (1 - alpha) * E

    v = np.ones(n) / n
    history = []
    iters = 40
    for _ in range(iters):
        v = G @ v
        v = v / v.sum()  # normalize to sum-1
        # track residual of fixed-point equation v = G v
        res = np.linalg.norm(v - G @ v)
        history.append(res)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(history, marker='o', ms=4, lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Iterazione')
    ax.set_ylabel('Residuo ||v - Gv||')
    ax.set_title('Convergenza del metodo delle potenze (PageRank)')
    ax.grid(True, alpha=0.3)
    out = os.path.join(IMG_DIR, 'lez12_power_method_convergence.png')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def main():
    ensure_img_dir()
    figura_vista_geometrica_separazione_circolare()
    figura_kernel_matrix_heatmap()
    figura_metodo_delle_potenze_convergenza()
    print('Immagini generate in:', IMG_DIR)

if __name__ == '__main__':
    main()