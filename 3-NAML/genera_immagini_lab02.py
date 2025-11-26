import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
DATA_DIR = os.path.join(BASE_DIR, "note", "Lab02")

os.makedirs(IMG_DIR, exist_ok=True)


def genera_pca_geometrica():
    rng = np.random.default_rng(0)

    n_points = 200
    mean_seed = np.array([0.0, 0.0])
    cov_seed = np.array([[1.0, 0.0], [0.0, 0.2]])
    Z = rng.multivariate_normal(mean_seed, cov_seed, size=n_points).T

    theta = np.deg2rad(35)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X = R @ Z

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_bar = X - X_mean

    U, s, VT = np.linalg.svd(X_bar, full_matrices=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(X[0, :], X[1, :], s=10, alpha=0.5, label="Dati")

    scale = 3.0
    origin = X_mean[:, 0]

    for i, color in enumerate(["black", "black"]):
        v = R[:, i]
        plt.arrow(
            origin[0],
            origin[1],
            scale * v[0],
            scale * v[1],
            head_width=0.1,
            length_includes_head=True,
            color=color,
            alpha=0.8,
        )

    for i, color in enumerate(["red", "red"]):
        v = U[:, i]
        plt.arrow(
            origin[0],
            origin[1],
            scale * v[0],
            scale * v[1],
            head_width=0.1,
            length_includes_head=True,
            color=color,
            alpha=0.8,
            linestyle="--",
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Confronto Direzioni Vere (nero) vs PCA (rosso)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = os.path.join(IMG_DIR, "lab02_pca_geometrica.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Salvata:", out_path)


def carica_mnist_train_small():
    path = os.path.join(DATA_DIR, "mnist_train_small.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    labels_full = data[:, 0].astype(int)
    A_full = data[:, 1:].T
    return A_full, labels_full


def genera_mnist_eigenimages_e_scatter():
    A_full, labels_full = carica_mnist_train_small()

    digits = [0, 9]
    mask = (labels_full == digits[0]) | (labels_full == digits[1])
    A = A_full[:, mask]
    labels = labels_full[mask]

    A_mean = np.mean(A, axis=1)
    A_bar = A - A_mean[:, None]

    U, s, VT = np.linalg.svd(A_bar, full_matrices=False)
    Phi = U.T @ A_bar

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    axes = axes.flatten()

    axes[0].imshow(A_mean.reshape(28, 28), cmap="gray")
    axes[0].set_title("Media (0 e 9)")
    axes[0].axis("off")

    for i in range(4):
        axes[i + 1].imshow(U[:, i].reshape(28, 28), cmap="gray")
        axes[i + 1].set_title(f"PC{i+1}")
        axes[i + 1].axis("off")

    axes[-1].axis("off")
    plt.tight_layout()

    out_path = os.path.join(IMG_DIR, "lab02_mnist_eigenimages.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Salvata:", out_path)

    PC1 = Phi[0, :]
    PC2 = Phi[1, :]

    mean0 = np.mean(PC1[labels == 0])
    mean9 = np.mean(PC1[labels == 9])
    threshold = 0.5 * (mean0 + mean9)

    plt.figure(figsize=(7, 6))
    for digit, color in zip([0, 9], ["tab:blue", "tab:orange"]):
        m = labels == digit
        plt.scatter(PC1[m], PC2[m], s=10, alpha=0.5, label=str(digit), c=color)

    plt.axvline(x=threshold, color="red", linestyle="--", label="threshold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("MNIST 0 vs 9 nello spazio PC1/PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(IMG_DIR, "lab02_mnist_scatter_threshold.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Salvata:", out_path)


def main():
    genera_pca_geometrica()
    genera_mnist_eigenimages_e_scatter()


if __name__ == "__main__":
    main()
