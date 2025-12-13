from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_labels_csv(path: Path) -> list[str]:
    # File is one label per line, no header.
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        labels.append(line)
    return labels


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, positive_label: str) -> dict[str, int]:
    # Confusion matrix for binary labels {positive_label, other}
    other_label = "__OTHER__"
    y_true_bin = np.where(y_true == positive_label, positive_label, other_label)
    y_pred_bin = np.where(y_pred == positive_label, positive_label, other_label)

    tp = int(np.sum((y_true_bin == positive_label) & (y_pred_bin == positive_label)))
    tn = int(np.sum((y_true_bin == other_label) & (y_pred_bin == other_label)))
    fp = int(np.sum((y_true_bin == other_label) & (y_pred_bin == positive_label)))
    fn = int(np.sum((y_true_bin == positive_label) & (y_pred_bin == other_label)))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def generate_ovarian_cancer_pca_figures(lab02_dir: Path, out_dir: Path) -> None:
    obs_path = lab02_dir / "ovariancancer_obs.csv"
    grp_path = lab02_dir / "ovariancancer_grp.csv"

    X = np.loadtxt(obs_path, delimiter=",")  # shape: (n_samples, n_features)
    labels = np.array(_load_labels_csv(grp_path))

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix from {obs_path}, got shape {X.shape}")
    if labels.shape[0] != X.shape[0]:
        raise ValueError(f"Label count mismatch: X has {X.shape[0]} rows but labels has {labels.shape[0]}")

    # Center features (per-feature mean over samples).
    X_centered = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD on centered data.
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = U * S  # equivalent to X_centered @ Vt.T
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]

    # Choose a simple 1D threshold on PC1 (midpoint between class means), then pick direction that maximizes accuracy.
    unique = sorted(set(labels.tolist()))
    if len(unique) != 2:
        raise ValueError(f"Expected a binary dataset, got labels: {unique}")

    # Keep the original label names from file; we treat the first label alphabetically as positive for reporting,
    # but we still pick the direction that yields best accuracy.
    positive_label = unique[0]

    mean0 = float(pc1[labels == unique[0]].mean())
    mean1 = float(pc1[labels == unique[1]].mean())
    threshold = (mean0 + mean1) / 2.0

    # Two possible directions for the same threshold.
    pred_a = np.where(pc1 >= threshold, unique[0], unique[1])
    pred_b = np.where(pc1 >= threshold, unique[1], unique[0])

    acc_a = float(np.mean(pred_a == labels))
    acc_b = float(np.mean(pred_b == labels))

    if acc_b > acc_a:
        y_pred = pred_b
        accuracy = acc_b
        positive_label = unique[1]
    else:
        y_pred = pred_a
        accuracy = acc_a
        positive_label = unique[0]

    counts = _confusion_counts(labels, y_pred, positive_label=positive_label)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure: PC1 vs PC2 scatter + threshold line.
    fig, ax = plt.subplots(figsize=(9, 6))

    # Colors: fixed and readable.
    color_map = {unique[0]: "tab:blue", unique[1]: "tab:orange"}
    for lab in unique:
        mask = labels == lab
        ax.scatter(pc1[mask], pc2[mask], s=18, alpha=0.75, label=lab, c=color_map.get(lab))

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3g}")

    ax.set_title(
        "Ovarian cancer dataset: PCA projection (PC1 vs PC2)\n"
        f"Simple 1D classifier on PC1 → accuracy={accuracy:.1%} | TP={counts['tp']} FP={counts['fp']} FN={counts['fn']} TN={counts['tn']}"
    )
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "lab02_cancer_scatter_threshold.png", dpi=200)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]  # .../UNI/3-NAML
    lab02_dir = Path(__file__).resolve().parent
    out_dir = repo_root / "img"

    generate_ovarian_cancer_pca_figures(lab02_dir=lab02_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
