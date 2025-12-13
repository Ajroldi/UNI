"""
Script per generare immagini visuali per il Lab03 (Lez13-17ott.md)
Regressione lineare, ridge, kernel e PageRank
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
import pandas as pd

# Imposta stile per grafici puliti
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# ==============================================================================
# 1. REGRESSIONE LINEARE: Scatter + Fit
# ==============================================================================
def plot_linear_regression():
    """Visualizza regressione lineare con dati rumorosi"""
    m, q = 2.0, 3.0
    N = 100
    noise = 2.0
    
    X = np.random.randn(N)
    Y = m * X + q + noise * np.random.randn(N)
    
    # Calcolo fit con pseudo-inversa
    Phi = np.column_stack([X, np.ones(N)])
    w = np.linalg.pinv(Phi) @ Y
    m_hat, q_hat = w[0], w[1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, Y, alpha=0.6, s=50, label='Dati con rumore', color='steelblue')
    
    x_line = np.linspace(X.min(), X.max(), 100)
    ax.plot(x_line, m * x_line + q, 'r--', linewidth=2.5, label=f'Modello reale: y = {m}x + {q}')
    ax.plot(x_line, m_hat * x_line + q_hat, 'k-', linewidth=2.5, 
            label=f'Stima LS: y = {m_hat:.2f}x + {q_hat:.2f}')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Regressione ai Minimi Quadrati (Least Squares)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_linear_regression.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_linear_regression.png")
    plt.close()

# ==============================================================================
# 2. RIDGE REGRESSION: Confronto lambda
# ==============================================================================
def plot_ridge_comparison():
    """Confronto tra diversi valori di lambda nella ridge regression"""
    N = 100
    noise = 0.1
    
    X = np.random.randn(N, 1)
    y_true = lambda x: np.tanh(2 * (x - 1))
    Y = y_true(X) + noise * np.random.randn(N, 1)
    
    X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
    Y_test_true = y_true(X_test)
    
    # Matrice delle caratteristiche
    Phi = np.column_stack([X, np.ones((N, 1))])
    Phi_test = np.column_stack([X_test, np.ones((300, 1))])
    
    # Diversi valori di lambda
    lambdas = [0, 0.1, 1.0, 10.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, lam in enumerate(lambdas):
        ax = axes[idx]
        
        if lam == 0:
            w = np.linalg.pinv(Phi) @ Y
        else:
            w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(2), Phi.T @ Y)
        
        Y_pred = Phi_test @ w
        
        ax.scatter(X, Y, alpha=0.5, s=30, color='steelblue', label='Training data')
        ax.plot(X_test, Y_test_true, 'k-', linewidth=2, label='Funzione vera')
        ax.plot(X_test, Y_pred, 'r--', linewidth=2, label=f'Ridge (λ={lam})')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Ridge Regression: λ = {lam}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.5, 1.5])
    
    plt.suptitle('Effetto del parametro di regolarizzazione λ', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_ridge_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_ridge_comparison.png")
    plt.close()

# ==============================================================================
# 3. KERNEL REGRESSION: Confronto kernel diversi
# ==============================================================================
def plot_kernel_comparison():
    """Confronto tra kernel lineare, polinomiale e gaussiano"""
    N = 100
    noise = 0.1
    
    X = np.random.randn(N, 1)
    y_true = lambda x: np.tanh(2 * (x - 1))
    Y = y_true(X) + noise * np.random.randn(N, 1)
    
    X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
    Y_test_true = y_true(X_test)
    
    # Definizione kernel
    def linear_kernel(x1, x2):
        return x1 * x2 + 1
    
    def poly_kernel(x1, x2, q=4):
        return (x1 * x2 + 1) ** q
    
    def rbf_kernel(x1, x2, sigma=0.5):
        return np.exp(-((x1 - x2) ** 2) / (2 * sigma ** 2))
    
    # Funzione di regressione kernel
    def kernel_regression(kernel, X, Y, X_test, lam=1.0):
        N = X.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = kernel(X[i, 0], X[j, 0])
        
        alpha = np.linalg.solve(K + lam * np.eye(N), Y)
        
        N_test = X_test.shape[0]
        K_test = np.zeros((N_test, N))
        for i in range(N_test):
            for j in range(N):
                K_test[i, j] = kernel(X_test[i, 0], X[j, 0])
        
        return K_test @ alpha
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Kernel lineare
    Y_pred_lin = kernel_regression(linear_kernel, X, Y, X_test)
    axes[0].scatter(X, Y, alpha=0.5, s=30, color='steelblue', label='Training data')
    axes[0].plot(X_test, Y_test_true, 'k-', linewidth=2, label='Funzione vera')
    axes[0].plot(X_test, Y_pred_lin, 'g--', linewidth=2, label='Kernel Lineare')
    axes[0].set_title('Kernel Lineare: k(x,z) = xz + 1', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    
    # Kernel polinomiale
    Y_pred_poly = kernel_regression(lambda x1, x2: poly_kernel(x1, x2, q=4), X, Y, X_test)
    axes[1].scatter(X, Y, alpha=0.5, s=30, color='steelblue', label='Training data')
    axes[1].plot(X_test, Y_test_true, 'k-', linewidth=2, label='Funzione vera')
    axes[1].plot(X_test, Y_pred_poly, 'orange', linewidth=2, linestyle='--', label='Kernel Polinomiale (q=4)')
    axes[1].set_title('Kernel Polinomiale: k(x,z) = (xz + 1)⁴', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('y', fontsize=11)
    
    # Kernel gaussiano
    Y_pred_rbf = kernel_regression(lambda x1, x2: rbf_kernel(x1, x2, sigma=0.5), X, Y, X_test)
    axes[2].scatter(X, Y, alpha=0.5, s=30, color='steelblue', label='Training data')
    axes[2].plot(X_test, Y_test_true, 'k-', linewidth=2, label='Funzione vera')
    axes[2].plot(X_test, Y_pred_rbf, 'purple', linewidth=2, linestyle='--', label='Kernel RBF (σ=0.5)')
    axes[2].set_title('Kernel Gaussiano (RBF): k(x,z) = exp(-||x-z||²/2σ²)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('y', fontsize=11)
    
    plt.suptitle('Confronto tra diversi Kernel nella Regressione', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_kernel_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_kernel_comparison.png")
    plt.close()

# ==============================================================================
# 4. SIGMA VARIATION: Effetto di sigma nel kernel gaussiano
# ==============================================================================
def plot_rbf_sigma_effect():
    """Visualizza l'effetto di diversi valori di sigma nel kernel RBF"""
    N = 100
    noise = 0.1
    
    X = np.random.randn(N, 1)
    y_true = lambda x: np.tanh(2 * (x - 1))
    Y = y_true(X) + noise * np.random.randn(N, 1)
    
    X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
    Y_test_true = y_true(X_test)
    
    def rbf_kernel(x1, x2, sigma):
        return np.exp(-((x1 - x2) ** 2) / (2 * sigma ** 2))
    
    def kernel_regression(sigma, X, Y, X_test, lam=1.0):
        N = X.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = rbf_kernel(X[i, 0], X[j, 0], sigma)
        
        alpha = np.linalg.solve(K + lam * np.eye(N), Y)
        
        N_test = X_test.shape[0]
        K_test = np.zeros((N_test, N))
        for i in range(N_test):
            for j in range(N):
                K_test[i, j] = rbf_kernel(X_test[i, 0], X[j, 0], sigma)
        
        return K_test @ alpha
    
    sigmas = [0.1, 0.3, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, sigma in enumerate(sigmas):
        ax = axes[idx]
        Y_pred = kernel_regression(sigma, X, Y, X_test)
        
        ax.scatter(X, Y, alpha=0.5, s=30, color='steelblue', label='Training data')
        ax.plot(X_test, Y_test_true, 'k-', linewidth=2, label='Funzione vera')
        ax.plot(X_test, Y_pred, 'r--', linewidth=2.5, label=f'Kernel RBF (σ={sigma})')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Kernel Gaussiano: σ = {sigma}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.5, 1.5])
    
    plt.suptitle('Effetto del parametro σ nel Kernel RBF', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_rbf_sigma_effect.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_rbf_sigma_effect.png")
    plt.close()

# ==============================================================================
# 5. PAGERANK: Grafo Wikipedia ML con dimensioni nodi proporzionali
# ==============================================================================
def plot_pagerank_graph():
    """Visualizza grafo con PageRank come dimensione dei nodi"""
    # Crea un grafo di esempio (simile al dataset Wikipedia ML)
    np.random.seed(42)
    
    # Genera un grafo diretto casuale
    G = nx.gnp_random_graph(30, 0.15, directed=True, seed=42)
    
    # Calcola PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    pagerank_values = np.array(list(pagerank.values()))
    
    # Layout
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Disegna archi
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, 
                          edge_color='gray', alpha=0.4, width=1.5, ax=ax)
    
    # Disegna nodi con dimensione proporzionale al PageRank
    node_sizes = pagerank_values * 10000
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=pagerank_values, cmap='YlOrRd',
                                   alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
    
    # Top 5 nodi per PageRank
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    top_node_ids = {node for node, _ in top_nodes}
    labels = {node: str(node) for node in top_node_ids}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold',
                           font_color='white', ax=ax)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap='YlOrRd', 
                          norm=plt.Normalize(vmin=min(pagerank_values), 
                                            vmax=max(pagerank_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('PageRank Score', fontsize=12, fontweight='bold')
    
    ax.set_title('Grafo con PageRank\n(dimensione nodi ∝ importanza)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_pagerank_graph.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_pagerank_graph.png")
    plt.close()

# ==============================================================================
# 6. PAGERANK CONVERGENCE: Iterazioni del power method
# ==============================================================================
def plot_pagerank_convergence():
    """Visualizza convergenza dell'algoritmo power iteration"""
    # Crea grafo
    G = nx.gnp_random_graph(50, 0.1, directed=True, seed=42)
    N = len(G)
    
    # Costruisci matrice di transizione M
    M = np.zeros((N, N))
    for u, v in G.edges():
        out_degree = G.out_degree(u)
        if out_degree > 0:
            M[v, u] = 1.0 / out_degree
    
    # Matrice Google G
    d = 0.85
    G_matrix = d * M + (1 - d) / N * np.ones((N, N))
    
    # Power iteration con tracciamento convergenza
    p = np.ones(N) / N
    tol = 1e-10
    max_iter = 100
    
    errors = []
    for i in range(max_iter):
        p_next = G_matrix @ p
        p_next /= np.linalg.norm(p_next, 1)
        
        error = np.linalg.norm(p_next - p, 2)
        errors.append(error)
        
        if error < tol:
            break
        p = p_next
    
    # Plot convergenza
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Errore vs iterazioni (scala lineare)
    ax1.plot(range(len(errors)), errors, 'o-', linewidth=2, markersize=5, color='steelblue')
    ax1.axhline(y=tol, color='r', linestyle='--', linewidth=2, label=f'Tolleranza = {tol}')
    ax1.set_xlabel('Iterazione', fontsize=12)
    ax1.set_ylabel('Errore ||p⁽ᵏ⁺¹⁾ - p⁽ᵏ⁾||₂', fontsize=12)
    ax1.set_title('Convergenza Power Iteration (scala lineare)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Errore vs iterazioni (scala log)
    ax2.semilogy(range(len(errors)), errors, 'o-', linewidth=2, markersize=5, color='darkorange')
    ax2.axhline(y=tol, color='r', linestyle='--', linewidth=2, label=f'Tolleranza = {tol}')
    ax2.set_xlabel('Iterazione', fontsize=12)
    ax2.set_ylabel('Errore ||p⁽ᵏ⁺¹⁾ - p⁽ᵏ⁾||₂ (log)', fontsize=12)
    ax2.set_title('Convergenza Power Iteration (scala logaritmica)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=11)
    
    plt.suptitle(f'PageRank: Convergenza in {len(errors)} iterazioni', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_pagerank_convergence.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_pagerank_convergence.png")
    plt.close()

# ==============================================================================
# 7. PAGERANK VS TRAFFICO: Scatter plot correlazione
# ==============================================================================
def plot_pagerank_traffic_correlation():
    """Scatter plot PageRank vs Traffico (simulato)"""
    np.random.seed(42)
    N = 50
    
    # Simula PageRank e traffico correlato
    pagerank = np.random.gamma(2, 2, N) / 100
    pagerank /= pagerank.sum()
    
    # Traffico correlato con PageRank + rumore
    traffic = pagerank * 1e6 * (1 + 0.5 * np.random.randn(N))
    traffic = np.abs(traffic)
    
    # Calcola correlazione
    corr = np.corrcoef(pagerank, traffic)[0, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scala lineare
    ax1.scatter(traffic, pagerank, s=80, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1)
    ax1.set_xlabel('Traffico Wikipedia (visite giornaliere)', fontsize=12)
    ax1.set_ylabel('PageRank Score', fontsize=12)
    ax1.set_title(f'Correlazione PageRank vs Traffico\nCorrelazione = {corr:.3f}', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scala logaritmica
    ax2.scatter(traffic, pagerank, s=80, alpha=0.7, color='darkorange', edgecolors='black', linewidth=1)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Traffico Wikipedia (scala log)', fontsize=12)
    ax2.set_ylabel('PageRank Score (scala log)', fontsize=12)
    ax2.set_title(f'Correlazione su scala log-log\nCorrelazione = {corr:.3f}', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('3-NAML/img/lab03_pagerank_traffic.png', dpi=150, bbox_inches='tight')
    print("✓ Salvata: lab03_pagerank_traffic.png")
    plt.close()

# ==============================================================================
# MAIN: Genera tutte le immagini
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generazione immagini per Lab03 (Lez13-17ott.md)")
    print("="*60 + "\n")
    
    plot_linear_regression()
    plot_ridge_comparison()
    plot_kernel_comparison()
    plot_rbf_sigma_effect()
    plot_pagerank_graph()
    plot_pagerank_convergence()
    plot_pagerank_traffic_correlation()
    
    print("\n" + "="*60)
    print("✅ Completato! Generate 7 nuove immagini per Lab03")
    print("="*60 + "\n")
