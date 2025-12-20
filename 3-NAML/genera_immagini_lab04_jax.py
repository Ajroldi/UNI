"""
Script per generare le immagini JAX per la lezione Lab04
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Crea cartella img se non esiste
os.makedirs('img', exist_ok=True)

print("Generazione grafici JAX...")

# GRAFICO 1: Funzione SELU
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * np.where(x > 0, x, alpha * np.exp(x) - alpha)

x = np.linspace(-3, 3, 1000)
y = selu(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=2.5, label='SELU(x)')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('SELU(x)', fontsize=13)
ax.set_title('Funzione di Attivazione SELU (Scaled Exponential Linear Unit)', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(-2, 3)
plt.tight_layout()
plt.savefig('img/lab04_jax_selu.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_jax_selu.png")
plt.close()

# GRAFICO 2: Confronto performance (simulato per visualizzazione)
methods = ['Naive\n(for loop)', 'JIT only', 'vmap only', 'vmap + JIT']
times = [500, 1.0, 1.0, 0.1]  # millisecondi
speedups = [1, 500, 500, 5000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Tempo di esecuzione
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars1 = ax1.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Tempo di Esecuzione (ms)', fontsize=13)
ax1.set_title('Confronto Tempo di Esecuzione', fontsize=15, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')
for bar, time in zip(bars1, times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.1f}ms' if time >= 1 else f'{time:.2f}ms',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup rispetto a naive
bars2 = ax2.bar(methods, speedups, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup (×)', fontsize=13)
ax2.set_title('Speedup Rispetto al Metodo Naive', fontsize=15, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup}×',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('img/lab04_jax_performance.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_jax_performance.png")
plt.close()

# GRAFICO 3: Visualizzazione gradiente funzione quadratica
x_vals = np.linspace(-2, 3, 100)
f = lambda x: x**2 + x + 4
dfdx = lambda x: 2*x + 1

y_vals = f(x_vals)
dy_vals = dfdx(x_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Funzione originale
ax1.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='f(x) = x² + x + 4')
ax1.scatter([1], [f(1)], color='red', s=100, zorder=5, label='Punto x=1')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('f(x)', fontsize=13)
ax1.set_title('Funzione Originale', fontsize=15, fontweight='bold')
ax1.legend(fontsize=12)

# Gradiente
ax2.plot(x_vals, dy_vals, 'g-', linewidth=2.5, label="f'(x) = 2x + 1")
ax2.scatter([1], [dfdx(1)], color='red', s=100, zorder=5, label='Gradiente in x=1')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel("f'(x)", fontsize=13)
ax2.set_title('Gradiente (Derivata Prima)', fontsize=15, fontweight='bold')
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('img/lab04_jax_gradient.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_jax_gradient.png")
plt.close()

# GRAFICO 4: Confronto CPU vs GPU (simulato)
scenarios = ['GPU only', 'CPU only', 'GPU + transfer\n(implicit)', 'GPU + transfer\n(explicit)']
times_gpu = [15, 180, 35, 17]  # millisecondi

fig, ax = plt.subplots(figsize=(11, 7))
colors_gpu = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
bars = ax.barh(scenarios, times_gpu, color=colors_gpu, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Tempo di Esecuzione (ms)', fontsize=13)
ax.set_title('JAX: Confronto Performance CPU vs GPU\n(Matrix multiplication 3000×3000)', 
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, time in zip(bars, times_gpu):
    width = bar.get_width()
    ax.text(width + 3, bar.get_y() + bar.get_height()/2.,
            f'{time} ms',
            ha='left', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('img/lab04_jax_cpu_gpu.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_jax_cpu_gpu.png")
plt.close()

# GRAFICO 5: Valore assoluto e suo gradiente
x_vals = np.linspace(-2, 2, 400)
abs_vals = np.abs(x_vals)

# Gradiente (con convenzione JAX per x=0)
grad_vals = np.where(x_vals > 0, 1, np.where(x_vals < 0, -1, 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Funzione valore assoluto
ax1.plot(x_vals, abs_vals, 'b-', linewidth=2.5, label='f(x) = |x|')
ax1.scatter([0], [0], color='red', s=150, zorder=5, 
            marker='o', edgecolors='darkred', linewidths=2,
            label='Punto non differenziabile (x=0)')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('|x|', fontsize=13)
ax1.set_title('Funzione Valore Assoluto', fontsize=15, fontweight='bold')
ax1.legend(fontsize=12)

# Gradiente con discontinuità
ax2.plot(x_vals[x_vals < -0.01], grad_vals[x_vals < -0.01], 
         'g-', linewidth=2.5, label="f'(x) = -1 (x < 0)")
ax2.plot(x_vals[x_vals > 0.01], grad_vals[x_vals > 0.01], 
         'purple', linewidth=2.5, label="f'(x) = 1 (x > 0)")
ax2.scatter([0], [1], color='red', s=150, zorder=5,
            marker='o', edgecolors='darkred', linewidths=2,
            label="f'(0) = 1 (convenzione JAX)")
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel("f'(x)", fontsize=13)
ax2.set_title('Gradiente (con Gestione Punto Non Differenziabile)', 
              fontsize=15, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('img/lab04_jax_abs_gradient.png', dpi=150, bbox_inches='tight')
print("✓ Salvato: img/lab04_jax_abs_gradient.png")
plt.close()

print("\n✅ Tutte le immagini JAX sono state generate con successo!")
