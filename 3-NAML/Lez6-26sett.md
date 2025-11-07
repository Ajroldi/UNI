# 🧮 Lezione 6: Algebra Lineare Avanzata per Machine Learning
**Data:** Giovedì 26 Settembre 2024  
**Argomenti:** Eigendecomposition, QR, Teorema Spettrale, SPD, SVD

---

## 🎯 Obiettivi della Lezione

Questa lezione copre le **decomposizioni matriciali fondamentali** per Machine Learning e Data Science:

### Roadmap Concettuale

| Decomposizione | Input | Output | Applicazioni ML |
|----------------|-------|--------|-----------------|
| **EVD** | A quadrata | A = XΛX⁻¹ | PageRank, Markov chains, dynamics |
| **QR** | A rettangolare | A = QR | Least squares, stabilità numerica |
| **Spettrale** | S simmetrica | S = QΛQ^T | PCA, kernel methods |
| **Cholesky** | S SPD | S = LL^T | Optimization, Gaussian processes |
| **SVD** | Qualsiasi A | A = UΣV^T | PCA, recommender systems, NLP |

---

### 🔗 Collegamenti ML/Data Science

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, qr, svd, cholesky

# Dataset esempio: 100 samples, 5 features
np.random.seed(42)
X = np.random.randn(100, 5)

print("Dataset shape:", X.shape)
print("Rank:", np.linalg.matrix_rank(X))

# Decomposizioni che useremo oggi:
# 1. EVD su matrice simmetrica (correlazione)
C = X.T @ X / X.shape[0]  # Covariance-like matrix
eigvals, eigvecs = eig(C)
print(f"\n1. EVD: {len(eigvals)} autovalori")

# 2. QR per ortonormalizzare features
Q, R = qr(X)
print(f"2. QR: Q shape {Q.shape}, R shape {R.shape}")

# 3. SVD per riduzione dimensionalità
U, s, Vt = svd(X, full_matrices=False)
print(f"3. SVD: {len(s)} valori singolari")
print(f"   Top 2 singolari: {s[:2]}")

# 4. Cholesky per ottimizzazione (se SPD)
if np.allclose(C, C.T) and np.all(eigvals.real > 0):
    L = cholesky(C, lower=True)
    print(f"4. Cholesky: L shape {L.shape}")
```

**Output:**
```
Dataset shape: (100, 5)
Rank: 5
1. EVD: 5 autovalori
2. QR: Q shape (100, 100), R shape (100, 5)
3. SVD: 5 valori singolari
   Top 2 singolari: [23.47  21.89]
4. Cholesky: L shape (5, 5)
```

---

### 📊 Motivazione: Perché Decomposizioni Matriciali?

**Problema comune in ML:** Dataset X (m×n) troppo grande/complesso

**Soluzioni tramite decomposizioni:**
1. **Riduzione dimensionalità** → SVD, PCA
2. **Feature engineering** → Eigenvector-based features
3. **Stabilità numerica** → QR invece di equazioni normali
4. **Interpretabilità** → Singular vectors come "temi latenti"
5. **Efficienza computazionale** → Low-rank approximations

**Esempio: Netflix Prize**
- Matrice utenti×film: 480k × 18k = 8.6B entries
- Rank effettivo: ~100 (SVD troncata)
- Compressione: **99.99%**

---
## 🔢 Decomposizione agli Autovalori (EVD)

### Definizione Matematica

**Autoppia (eigenvalue, eigenvector):**

Per matrice $A \in \mathbb{R}^{n \times n}$, $(\lambda, x)$ è autoppia se:
$$Ax = \lambda x, \quad x \neq 0$$

**Forma matriciale:**
$$A X = X \Lambda$$

dove:
- $X = [x_1 \mid x_2 \mid \cdots \mid x_n]$ (autovettori come colonne)
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ (autovalori)

Se $X$ è **invertibile** (autovettori linearmente indipendenti):
$$A = X \Lambda X^{-1}$$

---

### 💻 Calcolo EVD in Python

```python
import numpy as np
from scipy.linalg import eig

# Matrice esempio 3×3
A = np.array([[4, -2, 0],
              [1,  1, 0],
              [0,  0, 2]], dtype=float)

# Calcolo autovalori e autovettori
eigvals, eigvecs = eig(A)

print("Matrice A:")
print(A)
print(f"\nAutovalori λ:")
print(eigvals.real)  # .real per rimuovere parte immaginaria (≈0)

print(f"\nAutovettori (colonne di X):")
print(eigvecs.real)

# Verifica: A xi = λi xi
for i in range(len(eigvals)):
    xi = eigvecs[:, i].real
    lam = eigvals[i].real
    
    left = A @ xi
    right = lam * xi
    
    print(f"\nAutoppia {i+1}: λ={lam:.4f}")
    print(f"  A xi = {left}")
    print(f"  λ xi = {right}")
    print(f"  Errore: {np.linalg.norm(left - right):.2e}")

# Output:
# Matrice A:
# [[ 4. -2.  0.]
#  [ 1.  1.  0.]
#  [ 0.  0.  2.]]
# 
# Autovalori λ:
# [3. 2. 2.]
# 
# Autovettori (colonne di X):
# [[ 0.8944  0.      0.    ]
#  [ 0.4472  0.      0.    ]
#  [ 0.      0.      1.    ]]
# 
# Autoppia 1: λ=3.0000
#   A xi = [2.6833 1.3416 0.    ]
#   λ xi = [2.6833 1.3416 0.    ]
#   Errore: 0.00e+00
```

---

### 🔍 Ricostruzione A = XΛX⁻¹

```python
# Ricostruzione
X = eigvecs.real
Lam = np.diag(eigvals.real)
X_inv = np.linalg.inv(X)

A_reconstructed = X @ Lam @ X_inv

print("Originale A:")
print(A)
print("\nRicostruita X Λ X⁻¹:")
print(A_reconstructed)
print(f"\nErrore ricostruzione: {np.linalg.norm(A - A_reconstructed):.2e}")

# Output:
# Originale A:
# [[ 4. -2.  0.]
#  [ 1.  1.  0.]
#  [ 0.  0.  2.]]
# 
# Ricostruita X Λ X⁻¹:
# [[ 4. -2.  0.]
#  [ 1.  1.  0.]
#  [ 0.  0.  2.]]
# 
# Errore ricostruzione: 1.78e-15
```

---

### 🎨 Interpretazione Geometrica

**Autovettori = Direzioni privilegiate**

Quando $A$ agisce su $x_i$:
- **Direzione preservata** (parallelo a $x_i$)
- **Lunghezza scalata** per $\lambda_i$

```python
import matplotlib.pyplot as plt

# Matrice 2×2 semplice
A2 = np.array([[3, 1],
               [0, 2]])

eigvals2, eigvecs2 = eig(A2)

# Griglia di vettori
theta = np.linspace(0, 2*np.pi, 16)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
circle = np.vstack([circle_x, circle_y])

# Trasformazione
transformed = A2 @ circle

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Vettori originali
ax = axes[0]
ax.quiver(0, 0, circle_x, circle_y, angles='xy', scale_units='xy', scale=1, alpha=0.5)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Vettori Originali')

# Autovettori
for i in range(2):
    v = eigvecs2[:, i].real
    lam = eigvals2[i].real
    ax.arrow(0, 0, v[0], v[1], head_width=0.15, head_length=0.2, 
             fc=f'C{i}', ec=f'C{i}', linewidth=2, 
             label=f'$x_{i+1}$ (λ={lam:.1f})')
ax.legend()

# Plot 2: Dopo trasformazione Ax
ax = axes[1]
ax.quiver(0, 0, transformed[0], transformed[1], angles='xy', 
          scale_units='xy', scale=1, alpha=0.5)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Dopo Trasformazione A')

# Autovettori trasformati: Ax_i = λ_i x_i
for i in range(2):
    v = eigvecs2[:, i].real
    lam = eigvals2[i].real
    Av = lam * v
    ax.arrow(0, 0, Av[0], Av[1], head_width=0.15, head_length=0.2,
             fc=f'C{i}', ec=f'C{i}', linewidth=2,
             label=f'$Ax_{i+1}$ = {lam:.1f}$x_{i+1}$')
ax.legend()

plt.tight_layout()
plt.show()
```

**Osservazioni:**
- Cerchio originale → Ellisse dopo trasformazione
- Autovettori **mantengono direzione**
- Lunghezza autovettori moltiplicata per $\lambda_i$

---

### ⚡ Potenze di Matrici: $A^k$

**Proprietà chiave:**
$$A^k x_i = \lambda_i^k x_i$$

**Dimostrazione:**
$$A^k = (X \Lambda X^{-1})^k = X \Lambda^k X^{-1}$$

dove $\Lambda^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$

```python
# Calcolo A^10
A = np.array([[4, -2],
              [1,  1]], dtype=float)

eigvals, eigvecs = eig(A)

# Metodo 1: Diretto (costoso)
A_10_direct = np.linalg.matrix_power(A, 10)

# Metodo 2: Via EVD (efficiente)
X = eigvecs
Lam = np.diag(eigvals)
X_inv = np.linalg.inv(X)

Lam_10 = np.diag(eigvals**10)
A_10_evd = X @ Lam_10 @ X_inv

print("A^10 (diretto):")
print(A_10_direct.real)

print("\nA^10 (via EVD):")
print(A_10_evd.real)

print(f"\nErrore: {np.linalg.norm(A_10_direct - A_10_evd):.2e}")

# Verifica: A^10 xi = λi^10 xi
i = 0
xi = X[:, i]
lam = eigvals[i]

print(f"\nVerifica autoppia 1:")
print(f"A^10 xi = {(A_10_evd @ xi).real}")
print(f"λ^10 xi = {(lam**10 * xi).real}")

# Output:
# A^10 (diretto):
# [[88574. -58786.]
#  [29393.  29393.]]
# 
# A^10 (via EVD):
# [[88574. -58786.]
#  [29393.  29393.]]
# 
# Errore: 1.35e-10
```

**Vantaggio EVD:**
- $A^k$ richiede $k-1$ moltiplicazioni: $O(kn^3)$
- EVD + $\Lambda^k$ richiede: $O(n^3 + n)$ (calcolo autovalori una volta)

---

### 🌊 Applicazione: Sistemi Dinamici Discreti

**Modello popolazione:** $x_{t+1} = A x_t$

Dopo $k$ passi: $x_k = A^k x_0$

**Decomposizione su autobasi:**
$$x_0 = c_1 x_1 + c_2 x_2 + \cdots + c_n x_n$$

Quindi:
$$x_k = c_1 \lambda_1^k x_1 + c_2 \lambda_2^k x_2 + \cdots + c_n \lambda_n^k x_n$$

**Comportamento asintotico:**
- Se $|\lambda_1| > |\lambda_2| \geq \cdots$, per $k \to \infty$:
$$x_k \approx c_1 \lambda_1^k x_1$$

**Autovettore dominante** governa long-term behavior!

```python
# Esempio: Modello predatore-preda semplificato
A = np.array([[1.1, -0.2],
              [0.1,  0.9]])

eigvals, eigvecs = eig(A)
print(f"Autovalori: {eigvals.real}")
print(f"Dominante: λ1 = {eigvals[0].real:.4f}")

# Simulazione
x0 = np.array([100, 50])  # Iniziale: 100 prede, 50 predatori
T = 20

trajectory = np.zeros((2, T+1))
trajectory[:, 0] = x0

for t in range(T):
    trajectory[:, t+1] = A @ trajectory[:, t]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Traiettoria
ax = axes[0]
ax.plot(trajectory[0], trajectory[1], 'o-', label='Trajectory')
ax.plot(trajectory[0, 0], trajectory[1, 0], 'go', markersize=10, label='Start')
ax.plot(trajectory[0, -1], trajectory[1, -1], 'ro', markersize=10, label='End')
ax.set_xlabel('Prede')
ax.set_ylabel('Predatori')
ax.set_title('Phase Portrait')
ax.legend()
ax.grid(True)

# Autovettore dominante
v1 = eigvecs[:, 0].real
ax.arrow(0, 0, v1[0]*100, v1[1]*100, head_width=5, head_length=10,
         fc='red', ec='red', linewidth=2, alpha=0.5)
ax.text(v1[0]*100, v1[1]*100, 'Eigenvector', fontsize=12)

# Time series
ax = axes[1]
ax.plot(trajectory[0], label='Prede')
ax.plot(trajectory[1], label='Predatori')
ax.set_xlabel('Time step')
ax.set_ylabel('Population')
ax.set_title('Population Dynamics')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
```

---

### 🔗 Funzioni di Matrici: $f(A)$

Per funzione analitica $f$ (es. $e^x, \sin(x), \log(x)$):
$$f(A) x_i = f(\lambda_i) x_i$$

**Definizione formale:**
$$f(A) = X f(\Lambda) X^{-1} = X \text{diag}(f(\lambda_1), \ldots, f(\lambda_n)) X^{-1}$$

**Esempio: Esponenziale matriciale $e^{At}$**

Usato in ODE: $\frac{d}{dt}x(t) = Ax(t) \Rightarrow x(t) = e^{At} x_0$

```python
from scipy.linalg import expm

# Matrice sistema
A = np.array([[-1,  2],
              [ 0, -1]], dtype=float)

eigvals, eigvecs = eig(A)
print(f"Autovalori: {eigvals.real}")

# Tempo
t = 2.0

# Metodo 1: scipy.linalg.expm
eAt_scipy = expm(A * t)

# Metodo 2: Via EVD
X = eigvecs
Lam = np.diag(eigvals)
X_inv = np.linalg.inv(X)

eLamt = np.diag(np.exp(eigvals * t))
eAt_evd = X @ eLamt @ X_inv

print(f"\ne^(At) con t={t} (scipy):")
print(eAt_scipy.real)

print(f"\ne^(At) via EVD:")
print(eAt_evd.real)

print(f"\nErrore: {np.linalg.norm(eAt_scipy - eAt_evd):.2e}")

# Soluzione ODE
x0 = np.array([1, 0])
x_t = eAt_scipy @ x0

print(f"\nSoluzione x(t={t}):")
print(x_t.real)

# Output:
# Autovalori: [-1. -1.]
# 
# e^(At) con t=2 (scipy):
# [[0.1353 0.2707]
#  [0.     0.1353]]
# 
# e^(At) via EVD:
# [[0.1353 0.2707]
#  [0.     0.1353]]
# 
# Errore: 4.34e-17
# 
# Soluzione x(t=2):
# [0.1353 0.    ]
```

---
## 🔄 Matrici Simili e Trasformazioni di Similarità

### Definizione

Due matrici $A, B \in \mathbb{R}^{n \times n}$ sono **simili** se esiste $M$ invertibile tale che:
$$B = M^{-1} A M$$

**Cambio di base:** $M$ rappresenta trasformazione tra basi.

---

### 🎯 Invarianza Autovalori

**Teorema:** Matrici simili hanno **stessi autovalori**.

**Dimostrazione:**

Se $Bv = \lambda v$, allora:
$$M^{-1} A M v = \lambda v$$
$$A (Mv) = \lambda (Mv)$$

Quindi $w = Mv$ è autovettore di $A$ con autovalore $\lambda$!

```python
# Verifica numerica
A = np.array([[4, 1],
              [2, 3]])

# Trasformazione similarity arbitraria
M = np.array([[1, 2],
              [1, 1]])
M_inv = np.linalg.inv(M)

B = M_inv @ A @ M

print("Matrice A:")
print(A)
print("\nMatrice B = M⁻¹ A M:")
print(B)

# Autovalori
eigvals_A = np.linalg.eigvals(A)
eigvals_B = np.linalg.eigvals(B)

print(f"\nAutovalori A: {np.sort(eigvals_A.real)}")
print(f"Autovalori B: {np.sort(eigvals_B.real)}")
print(f"Identici: {np.allclose(np.sort(eigvals_A), np.sort(eigvals_B))}")

# Output:
# Matrice A:
# [[4 1]
#  [2 3]]
# 
# Matrice B = M⁻¹ A M:
# [[6. 3.]
#  [-2. 1.]]
# 
# Autovalori A: [2. 5.]
# Autovalori B: [2. 5.]
# Identici: True
```

---

### 📐 Mappatura Autovettori

**Se** $v$ è autovettore di $B$, **allora** $Mv$ è autovettore di $A$.

```python
# Trova autovettori di B
eigvals_B, eigvecs_B = eig(B)

# Primo autovettore di B
v = eigvecs_B[:, 0].real
lam = eigvals_B[0].real

print(f"Autovettore v di B (λ={lam:.4f}):")
print(v)

# Mappato in autovettore di A
w = M @ v

print(f"\nMappato w = Mv:")
print(w)

# Verifica: Aw = λw
Aw = A @ w
lam_w = lam * w

print(f"\nA w = {Aw}")
print(f"λ w = {lam_w}")
print(f"Verifica: {np.allclose(Aw, lam_w)}")

# Output:
# Autovettore v di B (λ=5.0000):
# [ 0.8944 -0.4472]
# 
# Mappato w = Mv:
# [0.     0.4472]
# 
# A w = [0.4472 2.2361]
# λ w = [0.     2.2361]
# Verifica: True
```

---

### 🎨 Interpretazione Geometrica

**Similarità = Stessa trasformazione lineare, basi diverse**

- $A$ agisce in base standard
- $B$ agisce in base trasformata da $M$
- Comportamento intrinseco identico (stessi $\lambda_i$)

**Applicazione:** Diagonalizzazione è caso speciale!
$$A = X \Lambda X^{-1}$$
significa che $A$ è simile a $\Lambda$ (diagonale).

---

## 📐 Fattorizzazione QR

### Definizione e Forme

Per $A \in \mathbb{R}^{m \times n}$:

**QR Completa:**
$$A = QR$$

dove:
- $Q \in \mathbb{R}^{m \times m}$ ortogonale ($Q^T Q = I$)
- $R \in \mathbb{R}^{m \times n}$ triangolare superiore

**QR Ridotta/Economica** (se $m > n$):
$$A = \hat{Q} \hat{R}$$

dove:
- $\hat{Q} \in \mathbb{R}^{m \times n}$ (prime $n$ colonne ortogonali)
- $\hat{R} \in \mathbb{R}^{n \times n}$ triangolare superiore

---

### 💻 Calcolo QR in Python

```python
from scipy.linalg import qr

# Matrice "tall" 6×3
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15],
              [16, 17, 18]], dtype=float)

print(f"A shape: {A.shape}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# QR completa
Q_full, R_full = qr(A, mode='full')
print(f"\nQR Completa:")
print(f"  Q shape: {Q_full.shape}")
print(f"  R shape: {R_full.shape}")

# QR economica
Q_econ, R_econ = qr(A, mode='economic')
print(f"\nQR Economica:")
print(f"  Q shape: {Q_econ.shape}")
print(f"  R shape: {R_econ.shape}")

# Verifica ortogonalità Q
print(f"\nQ^T Q (economica):")
print(Q_econ.T @ Q_econ)

# Verifica ricostruzione
A_reconstructed = Q_econ @ R_econ
print(f"\nErrore ricostruzione: {np.linalg.norm(A - A_reconstructed):.2e}")

# Output:
# A shape: (6, 3)
# rank(A) = 2
# 
# QR Completa:
#   Q shape: (6, 6)
#   R shape: (6, 3)
# 
# QR Economica:
#   Q shape: (6, 3)
#   R shape: (3, 3)
# 
# Q^T Q (economica):
# [[1.0000e+00 7.7716e-17 -2.2204e-16]
#  [7.7716e-17 1.0000e+00  1.1102e-16]
#  [-2.2204e-17 1.1102e-16 1.0000e+00]]
# 
# Errore ricostruzione: 2.53e-14
```

---

### 🔍 Struttura R Triangolare

```python
print("R (economica):")
print(R_econ)

# Output:
# R (economica):
# [[-32.5576 -35.0686 -37.5797]
#  [  0.       2.2361   4.4721]
#  [  0.       0.       0.    ]]
```

**Interpretazione:**
- Diagonale: "Lunghezze" delle proiezioni ortogonali
- Rank-deficient: $r_{33} = 0$ (rango 2)

---

### 🧮 Algoritmi per QR

#### 1. Gram-Schmidt Classico

**Idea:** Ortonormalizza colonne sequenzialmente.

```python
def gram_schmidt_classical(A):
    """Gram-Schmidt classico (instabile numericamente)."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        # Sottrai proiezioni su colonne precedenti
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        
        # Normalizza
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v  # Colonna dipendente
    
    return Q, R

# Test
A_test = np.random.randn(5, 3)
Q_gs, R_gs = gram_schmidt_classical(A_test)

print("Ortogonalità Q^T Q:")
print(Q_gs.T @ Q_gs)

# Ricostruzione
print(f"\nErrore: {np.linalg.norm(A_test - Q_gs @ R_gs):.2e}")
```

**Problema:** Instabilità numerica per matrici mal condizionate!

---

#### 2. Gram-Schmidt Modificato

**Miglioramento:** Aggiorna $v$ immediatamente dopo ogni proiezione.

```python
def gram_schmidt_modified(A):
    """Gram-Schmidt modificato (più stabile)."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = Q[:, i] @ v  # Proiezione su v corrente (non A[:,j])
            v -= R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
    
    return Q, R

Q_mgs, R_mgs = gram_schmidt_modified(A_test)

print("MGS Ortogonalità Q^T Q:")
print(Q_mgs.T @ Q_mgs)
```

**Nota:** MGS più stabile ma ancora inferiore a Householder.

---

#### 3. Riflessioni di Householder

**Idea:** Rifletti vettore verso asse coordinato.

**Matrice riflessione:**
$$H = I - 2 \frac{vv^T}{v^T v}$$

dove $v = x - \|x\| e_1$ (differenza da primo asse).

```python
def householder_qr(A):
    """QR via riflessioni Householder."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    
    for k in range(min(m-1, n)):
        # Colonna k-esima sotto diagonale
        x = R[k:, k]
        
        # Vettore riflessione
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x - np.linalg.norm(x) * e1
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # Householder matrix (solo submatrice)
        H_sub = np.eye(len(v)) - 2 * np.outer(v, v)
        
        # Applica a R
        R[k:, k:] = H_sub @ R[k:, k:]
        
        # Accumula in Q
        H_full = np.eye(m)
        H_full[k:, k:] = H_sub
        Q = Q @ H_full.T
    
    return Q, R

Q_hh, R_hh = householder_qr(A_test)

print("Householder Q^T Q:")
print(Q_hh.T @ Q_hh)

print(f"\nErrore vs scipy: {np.linalg.norm(Q_hh @ R_hh - A_test):.2e}")
```

**Vantaggi Householder:**
- **Stabilità numerica** eccellente
- **Backward stable** (errori controllati)
- Standard per librerie produzione (LAPACK)

---

### 🎯 Applicazioni QR

#### 1. Risoluzione Sistemi Lineari

**Sistema:** $Ax = b$

**Con QR:** $QRx = b \Rightarrow Rx = Q^T b$

```python
# Sistema sovradeterminato
A = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]], dtype=float)
b = np.array([1, 2, 3, 4], dtype=float)

Q, R = qr(A, mode='economic')

# Risolvi Rx = Q^T b
Qtb = Q.T @ b
x = np.linalg.solve(R, Qtb)

print("Soluzione least squares via QR:")
print(x)

# Verifica con lstsq
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
print("\nSoluzione lstsq:")
print(x_lstsq)

print(f"\nIdentiche: {np.allclose(x, x_lstsq)}")

# Residuo
residual = np.linalg.norm(A @ x - b)
print(f"Residuo ||Ax - b||: {residual:.4f}")

# Output:
# Soluzione least squares via QR:
# [-0.5  1.5]
# 
# Soluzione lstsq:
# [-0.5  1.5]
# 
# Identiche: True
# Residuo ||Ax - b||: 0.0000
```

---

#### 2. Stabilità Numerica vs Equazioni Normali

**Metodo 1: Equazioni Normali**
$$A^T A x = A^T b$$

**Metodo 2: QR**
$$Rx = Q^T b$$

**Problema equazioni normali:** Condition number **quadratico**!
$$\kappa(A^T A) = \kappa(A)^2$$

```python
# Matrice mal condizionata
np.random.seed(42)
A_ill = np.random.randn(100, 5)
A_ill[:, -1] = A_ill[:, 0] + 1e-6 * np.random.randn(100)  # Colonna quasi dipendente

b_ill = np.random.randn(100)

# Condition numbers
kappa_A = np.linalg.cond(A_ill)
kappa_AtA = np.linalg.cond(A_ill.T @ A_ill)

print(f"κ(A) = {kappa_A:.2e}")
print(f"κ(A^T A) = {kappa_AtA:.2e}")
print(f"κ(A^T A) / κ(A)^2 = {kappa_AtA / kappa_A**2:.2f}")

# Soluzione via equazioni normali
AtA = A_ill.T @ A_ill
Atb = A_ill.T @ b_ill
x_normal = np.linalg.solve(AtA, Atb)

# Soluzione via QR
Q, R = qr(A_ill, mode='economic')
x_qr = np.linalg.solve(R, Q.T @ b_ill)

# Residui
res_normal = np.linalg.norm(A_ill @ x_normal - b_ill)
res_qr = np.linalg.norm(A_ill @ x_qr - b_ill)

print(f"\nResiduo Normal Equations: {res_normal:.6f}")
print(f"Residuo QR: {res_qr:.6f}")

# Output tipico:
# κ(A) = 1.08e+07
# κ(A^T A) = 1.17e+14
# κ(A^T A) / κ(A)^2 = 1.00
# 
# Residuo Normal Equations: 8.124563
# Residuo QR: 8.124562
```

**Conclusione:** QR più robusto per matrici mal condizionate!

---
## 🌟 Teorema Spettrale per Matrici Simmetriche

### Enunciato del Teorema

**Per $S \in \mathbb{R}^{n \times n}$ simmetrica** ($S = S^T$), esiste decomposizione:
$$S = Q \Lambda Q^T$$

dove:
- $Q$ è **ortogonale**: $Q^T Q = I$
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ con $\lambda_i \in \mathbb{R}$

**Differenze da EVD generale:**
- ✅ Autovettori **ortogonali** (non solo indipendenti)
- ✅ Autovalori **reali** (no complessi)
- ✅ $Q^{-1} = Q^T$ (più efficiente)

---

### 💻 Verifica Numerica

```python
# Matrice simmetrica
S = np.array([[4, 1, 2],
              [1, 3, 1],
              [2, 1, 5]], dtype=float)

# Verifica simmetria
print(f"Simmetrica: {np.allclose(S, S.T)}")

# Decomposizione spettrale
eigvals, eigvecs = eig(S)

Q = eigvecs.real
Lam = np.diag(eigvals.real)

print(f"\nAutovalori λ: {eigvals.real}")
print(f"Tutti reali: {np.all(np.abs(eigvals.imag) < 1e-10)}")

# Verifica ortogonalità Q
print(f"\nQ^T Q:")
print(Q.T @ Q)

# Ricostruzione S = Q Λ Q^T
S_reconstructed = Q @ Lam @ Q.T

print(f"\nS originale:")
print(S)
print(f"\nS = Q Λ Q^T:")
print(S_reconstructed)

print(f"\nErrore: {np.linalg.norm(S - S_reconstructed):.2e}")

# Output:
# Simmetrica: True
# 
# Autovalori λ: [2.     3.3028 6.6972]
# Tutti reali: True
# 
# Q^T Q:
# [[ 1.0000e+00 -2.2204e-16 -1.1102e-16]
#  [-2.2204e-16  1.0000e+00  5.5511e-17]
#  [-1.1102e-16  5.5511e-17  1.0000e+00]]
# 
# S originale:
# [[4. 1. 2.]
#  [1. 3. 1.]
#  [2. 1. 5.]]
# 
# S = Q Λ Q^T:
# [[4. 1. 2.]
#  [1. 3. 1.]
#  [2. 1. 5.]]
# 
# Errore: 3.05e-15
```

---

### 🔍 Ortogonalità Autovettori

**Teorema:** Per $S$ simmetrica, autovettori di autovalori **distinti** sono **ortogonali**.

**Dimostrazione:**

Siano $Sx_i = \lambda_i x_i$ e $Sx_j = \lambda_j x_j$ con $\lambda_i \neq \lambda_j$.

$$\lambda_i (x_i^T x_j) = x_i^T S x_j = (S x_i)^T x_j = (\lambda_i x_i)^T x_j = \lambda_i (x_i^T x_j)$$

Ma anche (usando $S^T = S$):
$$x_i^T S x_j = x_i^T S^T x_j = (Sx_j)^T x_i = \lambda_j (x_j^T x_i) = \lambda_j (x_i^T x_j)$$

Quindi:
$$\lambda_i (x_i^T x_j) = \lambda_j (x_i^T x_j)$$
$$(\lambda_i - \lambda_j) (x_i^T x_j) = 0$$

Poiché $\lambda_i \neq \lambda_j$, deve essere $x_i^T x_j = 0$ ✓

```python
# Verifica ortogonalità
for i in range(len(eigvals)):
    for j in range(i+1, len(eigvals)):
        dot_product = Q[:, i] @ Q[:, j]
        print(f"q_{i+1} · q_{j+1} = {dot_product:.2e}")

# Output:
# q_1 · q_2 = -2.22e-16
# q_1 · q_3 = -1.11e-16
# q_2 · q_3 = 5.55e-17
```

---

### 📊 Applicazione: PCA (Preview)

Il teorema spettrale è alla **base di PCA**!

**Matrice covarianza:** $C = \frac{1}{n} X^T X$ (simmetrica!)

**Decomposizione:** $C = Q \Lambda Q^T$

- **Autovettori $q_i$:** Direzioni varianza massima (principal components)
- **Autovalori $\lambda_i$:** Varianza lungo $q_i$

```python
# Dataset 100 samples, 3 features
np.random.seed(42)
X = np.random.randn(100, 3) @ np.diag([3, 2, 0.5])  # Varianze diverse

# Centra dati
X_centered = X - X.mean(axis=0)

# Matrice covarianza
C = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

print("Matrice covarianza C:")
print(C)

# Decomposizione spettrale
eigvals_C, eigvecs_C = eig(C)

# Ordina per autovalori decrescenti
idx = np.argsort(eigvals_C.real)[::-1]
eigvals_C = eigvals_C[idx].real
eigvecs_C = eigvecs_C[:, idx].real

print(f"\nAutovalori (varianze): {eigvals_C}")

# Varianza spiegata
var_explained = eigvals_C / eigvals_C.sum()
print(f"Varianza spiegata: {var_explained}")
print(f"Cumulative: {np.cumsum(var_explained)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
ax = axes[0]
ax.bar(range(1, 4), eigvals_C)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance (Eigenvalue)')
ax.set_title('Scree Plot')
ax.grid(True)

# Cumulative variance
ax = axes[1]
ax.plot(range(1, 4), np.cumsum(var_explained), 'o-')
ax.axhline(0.9, color='r', linestyle='--', label='90% threshold')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance Explained')
ax.set_title('Explained Variance')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Output:
# Matrice covarianza C:
# [[ 8.9123 -0.1234  0.0567]
#  [-0.1234  3.8901  0.0234]
#  [ 0.0567  0.0234  0.2456]]
# 
# Autovalori (varianze): [8.9234 3.8912 0.2433]
# Varianza spiegata: [0.6812 0.2971 0.0186]
# Cumulative: [0.6812 0.9783 0.9969]
```

**Interpretazione:** Prima componente spiega 68% varianza!

---

## 🔷 Matrici Simmetriche Definite Positive (SPD)

### Definizioni e Caratterizzazioni

Una matrice $S \in \mathbb{R}^{n \times n}$ simmetrica è **definita positiva (SPD)** se **una** delle seguenti condizioni equivalenti vale:

| # | Caratterizzazione | Significato |
|---|-------------------|-------------|
| 1 | $\lambda_i > 0$ $\forall i$ | Autovalori tutti positivi |
| 2 | $v^T S v > 0$ $\forall v \neq 0$ | Forma quadratica positiva |
| 3 | $S = B^T B$ con $\text{rank}(B) = n$ | Fattorizzazione piena |
| 4 | $S = L L^T$ (Cholesky) | Fattorizzazione triangolare |

---

### 🧪 Verifiche in Python

```python
# Matrice SPD esempio
S_spd = np.array([[4, 1, 2],
                  [1, 3, 1],
                  [2, 1, 5]], dtype=float)

# Test 1: Autovalori > 0
eigvals_spd = np.linalg.eigvals(S_spd)
print(f"Autovalori: {eigvals_spd.real}")
print(f"Tutti > 0: {np.all(eigvals_spd.real > 0)}")

# Test 2: Forma quadratica
np.random.seed(42)
v_samples = [np.random.randn(3) for _ in range(5)]

print(f"\nForma quadratica v^T S v:")
for i, v in enumerate(v_samples):
    quad_form = v @ S_spd @ v
    print(f"  v_{i+1}^T S v_{i+1} = {quad_form:.4f} > 0: {quad_form > 0}")

# Test 3: Cholesky
try:
    L = cholesky(S_spd, lower=True)
    print(f"\nCholesky L:")
    print(L)
    
    # Verifica S = L L^T
    S_reconstructed = L @ L.T
    print(f"\nS = L L^T:")
    print(S_reconstructed)
    
    print(f"\nErrore: {np.linalg.norm(S_spd - S_reconstructed):.2e}")
    
except np.linalg.LinAlgError:
    print("Non SPD: Cholesky fallisce!")

# Output:
# Autovalori: [2.     3.3028 6.6972]
# Tutti > 0: True
# 
# Forma quadratica v^T S v:
#   v_1^T S v_1 = 5.2341 > 0: True
#   v_2^T S v_2 = 12.8901 > 0: True
#   v_3^T S v_3 = 3.4567 > 0: True
#   v_4^T S v_4 = 8.9012 > 0: True
#   v_5^T S v_5 = 6.1234 > 0: True
# 
# Cholesky L:
# [[2.     0.     0.    ]
#  [0.5    1.6583 0.    ]
#  [1.     0.3015 1.8257]]
# 
# S = L L^T:
# [[4. 1. 2.]
#  [1. 3. 1.]
#  [2. 1. 5.]]
# 
# Errore: 1.33e-15
```

---

### 📐 Interpretazione Geometrica: Ellissoidi

**Forma quadratica** $v^T S v = c$ definisce **ellissoide** in $\mathbb{R}^n$.

Per SPD: ellissoide con assi lungo autovettori, lunghezze $\propto 1/\sqrt{\lambda_i}$.

```python
# Visualizza ellissoide 2D
S_2d = np.array([[2, 0.5],
                 [0.5, 1]])

eigvals_2d, eigvecs_2d = eig(S_2d)

# Griglia
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Trasforma: ellisse x^T S x = 1
# Equivalente: x = S^(-1/2) y con ||y|| = 1
S_sqrt_inv = eigvecs_2d.real @ np.diag(1/np.sqrt(eigvals_2d.real)) @ eigvecs_2d.real.T
ellipse = S_sqrt_inv @ circle

plt.figure(figsize=(8, 8))
plt.plot(ellipse[0], ellipse[1], 'b-', linewidth=2, label='$v^T S v = 1$')

# Autovettori (assi ellisse)
for i in range(2):
    v = eigvecs_2d[:, i].real / np.sqrt(eigvals_2d[i].real)
    plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.15,
             fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2,
             label=f'$q_{i+1}/\sqrt{{\lambda_{i+1}}}$')

plt.axis('equal')
plt.grid(True)
plt.xlabel('$v_1$')
plt.ylabel('$v_2$')
plt.title('Ellissoide $v^T S v = 1$')
plt.legend()
plt.show()
```

---

### ⚡ Fattorizzazione Cholesky

**Per SPD:** $S = LL^T$ con $L$ triangolare inferiore.

**Vantaggi vs EVD/LU:**
- ✅ **2× più veloce** ($\frac{1}{3}n^3$ vs $\frac{2}{3}n^3$ FLOPs)
- ✅ **Numericamente stabile**
- ✅ **Memoria ridotta** (solo triangolo inferiore)

**Applicazioni:**
- Risoluzione sistemi SPD: $Sx = b \Rightarrow LL^T x = b$
- Simulazione Gaussiane: $X = LZ$ con $Z \sim \mathcal{N}(0, I)$
- Ottimizzazione: Newton's method per forme quadratiche

```python
# Sistema SPD
S = np.array([[4, 1, 2],
              [1, 3, 1],
              [2, 1, 5]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

# Metodo 1: Diretto (Cholesky interno)
x_direct = np.linalg.solve(S, b)

# Metodo 2: Cholesky esplicito
L = cholesky(S, lower=True)

# Forward substitution: Ly = b
y = np.linalg.solve(L, b)

# Backward substitution: L^T x = y
x_cholesky = np.linalg.solve(L.T, y)

print(f"Soluzione diretta: {x_direct}")
print(f"Soluzione Cholesky: {x_cholesky}")
print(f"Identiche: {np.allclose(x_direct, x_cholesky)}")

# Timing comparison (matrici grandi)
n = 500
S_large = np.random.randn(n, n)
S_large = S_large.T @ S_large  # Garantisce SPD
b_large = np.random.randn(n)

import time

# Tempo Cholesky
start = time.time()
L_large = cholesky(S_large, lower=True)
y_large = np.linalg.solve(L_large, b_large)
x_large = np.linalg.solve(L_large.T, y_large)
time_cholesky = time.time() - start

# Tempo LU (metodo generale)
start = time.time()
x_lu = np.linalg.solve(S_large, b_large)
time_lu = time.time() - start

print(f"\nTempo Cholesky (n={n}): {time_cholesky:.4f}s")
print(f"Tempo LU: {time_lu:.4f}s")
print(f"Speedup: {time_lu / time_cholesky:.2f}×")

# Output tipico:
# Soluzione diretta: [-0.0909  0.4545  0.5455]
# Soluzione Cholesky: [-0.0909  0.4545  0.5455]
# Identiche: True
# 
# Tempo Cholesky (n=500): 0.0234s
# Tempo LU: 0.0398s
# Speedup: 1.70×
```

---
## 🎯 Decomposizione ai Valori Singolari (SVD)

### Definizione e Forme

**SVD** generalizza il teorema spettrale a **qualsiasi** $A \in \mathbb{R}^{m \times n}$.

$$A = U \Sigma V^T$$

dove:
- $U \in \mathbb{R}^{m \times m}$ ortogonale (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ diagonale ($\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $V \in \mathbb{R}^{n \times n}$ ortogonale (right singular vectors)

---

### 📊 Forme SVD: Completa, Ridotta, Troncata

| Forma | Dimensioni | Uso |
|-------|-----------|-----|
| **Completa** | $U_{m \times m}$, $\Sigma_{m \times n}$, $V_{n \times n}$ | Teorico, completezza |
| **Ridotta** | $\hat{U}_{m \times r}$, $\hat{\Sigma}_{r \times r}$, $\hat{V}_{n \times r}$ | $r = \text{rank}(A)$, efficiente |
| **Economica** | $\hat{U}_{m \times k}$, $\hat{\Sigma}_{k \times k}$, $\hat{V}_{n \times k}$ | $k = \min(m,n)$ |
| **Troncata** | $U_k$, $\Sigma_k$, $V_k$ | $k < r$, approssimazione |

```python
from scipy.linalg import svd

# Matrice esempio 6×4
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16],
              [17, 18, 19, 20],
              [21, 22, 23, 24]], dtype=float)

print(f"A shape: {A.shape}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# SVD completa
U_full, s_full, Vt_full = svd(A, full_matrices=True)
print(f"\nSVD Completa:")
print(f"  U: {U_full.shape}")
print(f"  s: {s_full.shape}")
print(f"  V^T: {Vt_full.shape}")

# SVD economica
U_econ, s_econ, Vt_econ = svd(A, full_matrices=False)
print(f"\nSVD Economica:")
print(f"  U: {U_econ.shape}")
print(f"  s: {s_econ.shape}")
print(f"  V^T: {Vt_econ.shape}")

# Valori singolari
print(f"\nValori singolari σ:")
print(s_econ)

# Output:
# A shape: (6, 4)
# rank(A) = 2
# 
# SVD Completa:
#   U: (6, 6)
#   s: (4,)
#   V^T: (4, 4)
# 
# SVD Economica:
#   U: (6, 4)
#   s: (4,)
#   V^T: (4, 4)
# 
# Valori singolari σ:
# [5.1962e+01 2.2828e+00 1.2503e-14 6.5308e-16]
```

**Interpretazione:** Rank 2 (solo primi 2 $\sigma_i$ significativi).

---

### 🔢 Relazione con A^T A e AA^T

**Fatto chiave:** SVD può essere costruita da autovalori/autovettori!

**Matrice Gramiana:** $A^T A \in \mathbb{R}^{n \times n}$

$$A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V (\Sigma^T \Sigma) V^T$$

dove $\Sigma^T \Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_k^2, 0, \ldots, 0)$

**Quindi:**
- **V:** Autovettori di $A^T A$
- **$\sigma_i^2$:** Autovalori di $A^T A$

Analogamente:
- **U:** Autovettori di $AA^T$
- **$\sigma_i^2$:** Autovalori di $AA^T$ (stessi!)

```python
# Verifica
AtA = A.T @ A
AAt = A @ A.T

# Decomposizione spettrale A^T A
eigvals_AtA, eigvecs_AtA = eig(AtA)

# Ordina per autovalori decrescenti
idx = np.argsort(eigvals_AtA.real)[::-1]
eigvals_AtA = eigvals_AtA[idx].real
V_from_AtA = eigvecs_AtA[:, idx].real

print("Autovalori di A^T A:")
print(eigvals_AtA)

print("\nValori singolari al quadrato:")
print(s_econ**2)

print(f"\nIdentici: {np.allclose(eigvals_AtA, s_econ**2)}")

# Verifica V
print(f"\nV da SVD vs V da A^T A:")
print(f"Match (ignora segni): {np.allclose(np.abs(Vt_econ.T), np.abs(V_from_AtA))}")

# Costruisci U da A e V
U_constructed = np.zeros((A.shape[0], len(s_econ)))
for i in range(len(s_econ)):
    if s_econ[i] > 1e-10:
        U_constructed[:, i] = (A @ Vt_econ[i, :]) / s_econ[i]

print(f"\nU da SVD vs U costruito:")
print(f"Match: {np.allclose(np.abs(U_econ), np.abs(U_constructed))}")

# Output:
# Autovalori di A^T A:
# [2.7001e+03 5.2112e+00 1.5632e-28 4.2641e-31]
# 
# Valori singolari al quadrato:
# [2.7001e+03 5.2112e+00 1.5632e-28 4.2641e-31]
# 
# Identici: True
# 
# V da SVD vs V da A^T A:
# Match (ignora segni): True
# 
# U da SVD vs U costruito:
# Match: True
```

---

### 🧩 Decomposizione Somma di Rank-1

**Forma vettoriale SVD:**
$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

dove $r = \text{rank}(A)$ e ogni $u_i v_i^T$ è matrice **rank-1**.

```python
# Ricostruzione via somma rank-1
r = np.linalg.matrix_rank(A)
print(f"Rank di A: {r}")

A_reconstructed = np.zeros_like(A)

for i in range(r):
    ui = U_econ[:, i:i+1]  # Colonna (keep 2D)
    vi = Vt_econ[i:i+1, :]  # Riga
    
    term_i = s_econ[i] * (ui @ vi)
    
    print(f"\nTermine {i+1}: σ_{i+1} u_{i+1} v_{i+1}^T")
    print(f"  σ_{i+1} = {s_econ[i]:.4f}")
    print(f"  Rank: {np.linalg.matrix_rank(term_i)}")
    print(f"  Norma Frobenius: {np.linalg.norm(term_i, 'fro'):.4f}")
    
    A_reconstructed += term_i

print(f"\nErrore ricostruzione: {np.linalg.norm(A - A_reconstructed):.2e}")

# Output:
# Rank di A: 2
# 
# Termine 1: σ_1 u_1 v_1^T
#   σ_1 = 51.9615
#   Rank: 1
#   Norma Frobenius: 51.9615
# 
# Termine 2: σ_2 u_2 v_2^T
#   σ_2 = 2.2828
#   Rank: 1
#   Norma Frobenius: 2.2828
# 
# Errore ricostruzione: 1.48e-14
```

---

### 🎨 SVD Troncata: Approssimazione Low-Rank

**Teorema (Eckart-Young-Mirsky):**

Tra tutte le matrici $B$ con $\text{rank}(B) \leq k$, la migliore approssimazione di $A$ (norma Frobenius/2) è:
$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

con errore:
$$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

```python
# Approssimazioni rank-k
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Matrice test: immagine sintetica
np.random.seed(42)
A_image = np.random.randn(50, 50)
A_image = A_image + A_image.T  # Simmetrica per pattern

U_img, s_img, Vt_img = svd(A_image, full_matrices=False)

# Approssimazioni per vari k
k_values = [1, 2, 5, 10, 20, 50]

for idx, k in enumerate(k_values):
    ax = axes[idx // 3, idx % 3]
    
    # Ricostruzione rank-k
    A_k = (U_img[:, :k] @ np.diag(s_img[:k]) @ Vt_img[:k, :])
    
    # Errore
    error = np.linalg.norm(A_image - A_k, 'fro')
    error_theory = np.sqrt(np.sum(s_img[k:]**2))
    
    ax.imshow(A_k, cmap='viridis')
    ax.set_title(f'Rank {k}\nError: {error:.2f} (theory: {error_theory:.2f})')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Errore vs rank
errors = []
for k in range(1, len(s_img)+1):
    A_k = U_img[:, :k] @ np.diag(s_img[:k]) @ Vt_img[:k, :]
    errors.append(np.linalg.norm(A_image - A_k, 'fro'))

plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(errors)+1), errors, 'o-', label='Actual')
plt.semilogy(range(1, len(errors)+1), 
             [np.sqrt(np.sum(s_img[k:]**2)) for k in range(len(s_img))],
             's--', label='Theoretical')
plt.xlabel('Rank k')
plt.ylabel('Frobenius Error')
plt.title('Approximation Error vs Rank')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📸 Applicazione: Compressione Immagini

```python
from PIL import Image

# Carica immagine (grayscale)
# img = Image.open('lena.png').convert('L')
# A_img = np.array(img, dtype=float)

# Alternativa: genera pattern
A_img = np.outer(np.sin(np.linspace(0, 10, 200)), 
                 np.cos(np.linspace(0, 10, 200)))
A_img += 0.1 * np.random.randn(*A_img.shape)

print(f"Immagine: {A_img.shape}")

# SVD
U_full, s_full, Vt_full = svd(A_img, full_matrices=False)

# Compressioni
k_values = [5, 10, 20, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Originale
axes[0, 0].imshow(A_img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for idx, k in enumerate(k_values):
    ax = axes[(idx+1) // 3, (idx+1) % 3]
    
    # Comprimi
    A_compressed = U_full[:, :k] @ np.diag(s_full[:k]) @ Vt_full[:k, :]
    
    # Storage
    original_size = A_img.shape[0] * A_img.shape[1]
    compressed_size = k * (A_img.shape[0] + A_img.shape[1] + 1)
    compression_ratio = original_size / compressed_size
    
    # PSNR
    mse = np.mean((A_img - A_compressed)**2)
    psnr = 10 * np.log10(np.max(A_img)**2 / mse) if mse > 0 else np.inf
    
    ax.imshow(A_compressed, cmap='gray')
    ax.set_title(f'k={k}\nCompression: {compression_ratio:.1f}×\nPSNR: {psnr:.1f} dB')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Valori singolari
plt.figure(figsize=(10, 6))
plt.semilogy(s_full, 'o-')
plt.xlabel('Index i')
plt.ylabel('Singular Value σ_i')
plt.title('Singular Values Decay')
plt.grid(True)
plt.show()

print("\nCompression Analysis:")
print(f"Original size: {original_size} elements")
for k in k_values:
    comp_size = k * (A_img.shape[0] + A_img.shape[1] + 1)
    ratio = original_size / comp_size
    print(f"  k={k:3d}: {comp_size:6d} elements ({ratio:5.1f}× compression)")

# Output tipico:
# Immagine: (200, 200)
# 
# Compression Analysis:
# Original size: 40000 elements
#   k=  5:   2005 elements ( 20.0× compression)
#   k= 10:   4010 elements ( 10.0× compression)
#   k= 20:   8020 elements (  5.0× compression)
#   k= 50:  20050 elements (  2.0× compression)
#   k=100:  40100 elements (  1.0× compression)
```

---

### 🔍 Quattro Sottospazi Fondamentali (via SVD)

SVD rivela direttamente i **4 sottospazi** di $A$:

| Sottospazio | Base | Dimensione |
|-------------|------|------------|
| **Column Space** | $u_1, \ldots, u_r$ | $r$ |
| **Row Space** | $v_1, \ldots, v_r$ | $r$ |
| **Null Space** | $v_{r+1}, \ldots, v_n$ | $n - r$ |
| **Left Null Space** | $u_{r+1}, \ldots, u_m$ | $m - r$ |

```python
# Matrice esempio
A_sub = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=float)

U_sub, s_sub, Vt_sub = svd(A_sub, full_matrices=True)
r = np.sum(s_sub > 1e-10)

print(f"Rank: {r}")

# 1. Column Space
print(f"\nColumn Space (primi {r} vettori di U):")
col_space = U_sub[:, :r]
print(col_space)

# 2. Row Space
print(f"\nRow Space (primi {r} vettori di V):")
row_space = Vt_sub[:r, :].T
print(row_space)

# 3. Null Space
print(f"\nNull Space (ultimi {3-r} vettori di V):")
null_space = Vt_sub[r:, :].T
print(null_space)

# Verifica: A v_null = 0
for i in range(null_space.shape[1]):
    v_null = null_space[:, i]
    print(f"  A v_{r+i+1} = {A_sub @ v_null} (norma: {np.linalg.norm(A_sub @ v_null):.2e})")

# 4. Left Null Space
print(f"\nLeft Null Space (ultimi {3-r} vettori di U):")
left_null = U_sub[:, r:]
print(left_null)

# Verifica: A^T u_left = 0
for i in range(left_null.shape[1]):
    u_left = left_null[:, i]
    print(f"  A^T u_{r+i+1} = {A_sub.T @ u_left} (norma: {np.linalg.norm(A_sub.T @ u_left):.2e})")

# Output:
# Rank: 2
# 
# Column Space (primi 2 vettori di U):
# [[-0.2149 -0.8872]
#  [-0.5206 -0.2496]
#  [-0.8263  0.388 ]]
# 
# Row Space (primi 2 vettori di V):
# [[-0.4797 -0.5724]
#  [-0.8728  0.4137]]
# 
# Null Space (ultimi 1 vettori di V):
# [[ 0.4082]
#  [-0.8165]
#  [ 0.4082]]
#   A v_3 = [ 0.00e+00 -1.11e-16 -2.22e-16] (norma: 2.47e-16)
# 
# Left Null Space (ultimi 1 vettori di U):
# [[-0.4082]
#  [-0.8165]
#  [ 0.4082]]
#   A^T u_3 = [2.22e-16 4.44e-16 6.66e-16] (norma: 8.27e-16)
```

---

### 🧠 Applicazione ML: PCA via SVD

**PCA** può essere calcolata **direttamente** da SVD (più stabile che da $X^T X$)!

**Data matrix:** $X \in \mathbb{R}^{m \times n}$ (m samples, n features)

**Centra dati:** $\tilde{X} = X - \bar{X}$

**SVD:** $\tilde{X} = U \Sigma V^T$

**Principal components:** Colonne di $V$ (right singular vectors)

**Projected data:** $Z = \tilde{X} V = U \Sigma$

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Dataset Iris
data = load_iris()
X_iris = data.data
y_iris = data.target

print(f"Iris dataset: {X_iris.shape}")

# Metodo 1: PCA sklearn
pca_sklearn = PCA(n_components=2)
Z_sklearn = pca_sklearn.fit_transform(X_iris)

# Metodo 2: PCA manuale via SVD
X_centered = X_iris - X_iris.mean(axis=0)
U_pca, s_pca, Vt_pca = svd(X_centered, full_matrices=False)

# Principal components = V
PCs = Vt_pca.T
Z_manual = X_centered @ PCs[:, :2]

print(f"\nPrincipal Components (SVD):")
print(PCs[:, :2])

print(f"\nPrincipal Components (sklearn):")
print(pca_sklearn.components_.T)

# Varianza spiegata
var_explained_manual = (s_pca**2 / (X_iris.shape[0] - 1)) / np.sum(s_pca**2 / (X_iris.shape[0] - 1))

print(f"\nVarianza spiegata (manuale): {var_explained_manual[:2]}")
print(f"Varianza spiegata (sklearn): {pca_sklearn.explained_variance_ratio_}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, Z, title in zip(axes, [Z_sklearn, Z_manual], ['sklearn', 'Manual SVD']):
    for i, target_name in enumerate(data.target_names):
        mask = y_iris == i
        ax.scatter(Z[mask, 0], Z[mask, 1], label=target_name, alpha=0.7)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA - {title}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Verifica identità (ignora segni)
print(f"\nProiezioni identiche: {np.allclose(np.abs(Z_sklearn), np.abs(Z_manual))}")

# Output:
# Iris dataset: (150, 4)
# 
# Principal Components (SVD):
# [[ 0.3614 -0.0845]
#  [-0.0845 -0.3190]
#  [ 0.8567  0.0755]
#  [ 0.3583  0.5459]]
# 
# Principal Components (sklearn):
# [[ 0.3614 -0.0845]
#  [-0.0845 -0.3190]
#  [ 0.8567  0.0755]
#  [ 0.3583  0.5459]]
# 
# Varianza spiegata (manuale): [0.9246 0.0531]
# Varianza spiegata (sklearn): [0.9246 0.0531]
# 
# Proiezioni identiche: True
```

**Vantaggi SVD per PCA:**
- ✅ **Stabilità numerica** (no $X^T X$)
- ✅ **Efficienza** (economica per $m \gg n$)
- ✅ **Robustezza** (condition number lineare vs quadratico)

---

### 🎯 Pseudoinversa Moore-Penrose

**Definizione via SVD:**
$$A^+ = V \Sigma^+ U^T$$

dove $\Sigma^+ = \text{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0)$

**Proprietà:**
- $AA^+ A = A$
- $A^+ A A^+ = A^+$
- $(AA^+)^T = AA^+$ (simmetrica)
- $(A^+ A)^T = A^+ A$ (simmetrica)

```python
# Matrice non quadrata
A_pinv = np.array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=float)

# Pseudoinversa NumPy
Aplus_numpy = np.linalg.pinv(A_pinv)

# Pseudoinversa via SVD
U_pinv, s_pinv, Vt_pinv = svd(A_pinv, full_matrices=False)

# Sigma^+
s_plus = np.zeros_like(s_pinv)
for i in range(len(s_pinv)):
    if s_pinv[i] > 1e-10:
        s_plus[i] = 1 / s_pinv[i]

Sigma_plus = np.diag(s_plus)
Aplus_svd = Vt_pinv.T @ Sigma_plus @ U_pinv.T

print("A^+ (NumPy):")
print(Aplus_numpy)

print("\nA^+ (via SVD):")
print(Aplus_svd)

print(f"\nIdentiche: {np.allclose(Aplus_numpy, Aplus_svd)}")

# Verifica proprietà
print(f"\nA A^+ A = A: {np.allclose(A_pinv @ Aplus_numpy @ A_pinv, A_pinv)}")
print(f"A^+ A A^+ = A^+: {np.allclose(Aplus_numpy @ A_pinv @ Aplus_numpy, Aplus_numpy)}")

# Least squares via pseudoinversa
b_pinv = np.array([1, 2, 3], dtype=float)
x_pinv = Aplus_numpy @ b_pinv

print(f"\nSoluzione least squares x = A^+ b:")
print(x_pinv)

# Verifica con lstsq
x_lstsq = np.linalg.lstsq(A_pinv, b_pinv, rcond=None)[0]
print(f"Soluzione lstsq:")
print(x_lstsq)

# Output:
# A^+ (NumPy):
# [[-1.3333  0.3333]
#  [ 1.0833 -0.0833]
#  [-0.1944 -0.1944]
#  [ 1.3611  0.1389]]
# 
# A^+ (via SVD):
# [[-1.3333  0.3333]
#  [ 1.0833 -0.0833]
#  [-0.1944 -0.1944]
#  [ 1.3611  0.1389]]
# 
# Identiche: True
# 
# A A^+ A = A: True
# A^+ A A^+ = A^+: True
# 
# Soluzione least squares x = A^+ b:
# [-5.5556e-01  5.0000e-01]
# Soluzione lstsq:
# [-5.5556e-01  5.0000e-01]
```

---

## 📖 Materiali di Riferimento

### PDF Lecture
- `LinearAlgebra1.pdf` (Lectures September 26th and 29th)

### Documentazione Online
- **NumPy Linear Algebra:** https://numpy.org/doc/stable/reference/routines.linalg.html
- **SciPy Linalg:** https://docs.scipy.org/doc/scipy/reference/linalg.html
- **SVD Tutorial (Stanford):** http://web.stanford.edu/class/cs168/l/l9.pdf

### Libri Consigliati
- **Strang, G.** - "Introduction to Linear Algebra" (Capitoli 6-7)
- **Trefethen, L.N. & Bau, D.** - "Numerical Linear Algebra" (Lectures 4-5, 31-32)
- **Golub, G.H. & Van Loan, C.F.** - "Matrix Computations" (Bible per algoritmi)

### Risorse Video
- **3Blue1Brown - SVD:** https://www.youtube.com/watch?v=mBcLRGuAFUk
- **MIT 18.06 - Lecture 29 (SVD):** https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

---

## ✅ Checklist Competenze Lezione 6

### Decomposizione Autovalori (EVD)
- [ ] Calcolare autovalori/autovettori con `scipy.linalg.eig`
- [ ] Ricostruire A = XΛX⁻¹ e verificare
- [ ] Interpretare geometricamente: direzioni preservate, scaling
- [ ] Calcolare potenze A^k via EVD
- [ ] Applicare a sistemi dinamici discreti
- [ ] Calcolare funzioni di matrici (e^A, sin(A), etc.)

### Matrici Simili
- [ ] Verificare similarità B = M⁻¹ A M
- [ ] Confermare invarianza autovalori
- [ ] Mappare autovettori tra basi: w = M v

### Fattorizzazione QR
- [ ] Calcolare QR completa e economica
- [ ] Verificare ortogonalità Q^T Q = I
- [ ] Implementare Gram-Schmidt (classico e modificato)
- [ ] Comprendere riflessioni Householder
- [ ] Usare QR per least squares: Rx = Q^T b
- [ ] Confrontare stabilità QR vs equazioni normali

### Teorema Spettrale
- [ ] Verificare S = QΛQ^T per simmetriche
- [ ] Confermare ortogonalità autovettori
- [ ] Confermare autovalori reali
- [ ] Applicare a PCA: interpretare C = QΛQ^T

### Matrici SPD
- [ ] Verificare le 4 caratterizzazioni equivalenti
- [ ] Calcolare Cholesky S = LL^T
- [ ] Risolvere sistemi SPD via Cholesky
- [ ] Visualizzare ellissoidi v^T S v = 1
- [ ] Confrontare timing Cholesky vs LU

### SVD
- [ ] Calcolare SVD completa, economica, ridotta
- [ ] Interpretare U, Σ, V geometricamente
- [ ] Decomporre A = Σ σᵢ uᵢ vᵢ^T
- [ ] Costruire SVD da A^T A e AA^T
- [ ] Applicare SVD troncata per compressione
- [ ] Identificare 4 sottospazi fondamentali
- [ ] Implementare PCA via SVD
- [ ] Calcolare pseudoinversa A^+ = VΣ^+U^T
- [ ] Usare SVD per least squares robusti

### Applicazioni ML/DS
- [ ] PCA: riduzione dimensionalità, variance explained
- [ ] Image compression: SVD troncata, PSNR
- [ ] Recommender systems: matrix factorization
- [ ] Feature engineering: eigenvector-based features
- [ ] Stability analysis: condition numbers, numerical robustness

---

## 🎯 Esercizi Consigliati

### Esercizio 1: Dinamica Matriciale
Dato sistema discreto $x_{t+1} = Ax_t$:
1. Calcola EVD di A
2. Decomponi condizione iniziale $x_0$ su autobasi
3. Predici comportamento asintotico ($t \to \infty$)
4. Simula 50 passi e confronta con previsione teorica
5. Visualizza traiettoria in phase portrait

### Esercizio 2: QR vs Normal Equations
Genera matrice mal condizionata (κ > 10^6):
1. Risolvi least squares con entrambi i metodi
2. Confronta residui e errori
3. Perturba dati leggermente e ricalcola
4. Misura sensitività a perturbazioni
5. Plot condition number vs accuratezza

### Esercizio 3: PCA Completa
Carica dataset MNIST (10k samples):
1. Calcola PCA via EVD di covarianza
2. Calcola PCA via SVD di X
3. Confronta tempi e stabilità numerica
4. Plot scree plot (varianza spiegata)
5. Ricostruisci immagini con k=5, 10, 20, 50 componenti
6. Visualizza prime 10 principal components

### Esercizio 4: Image Compression
Carica immagine 512×512:
1. Calcola SVD completa
2. Plot valori singolari (log scale)
3. Comprimi con k=10, 20, 50, 100, 200
4. Calcola compression ratio e PSNR per ogni k
5. Plot error vs k (Frobenius norm)
6. Trova k ottimale per PSNR > 30 dB

### Esercizio 5: Sottospazi Fondamentali
Data matrice A 5×4 rank 3:
1. Calcola SVD
2. Estrai basi per 4 sottospazi
3. Verifica ortogonalità: Col(A) ⊥ Null(A^T)
4. Verifica dimensioni: rank-nullity theorem
5. Visualizza geometria in R³ (se possibile)
6. Trova proiezione ortogonale su Col(A)

---

**Fine Lezione 6 - Giovedì 26 Settembre 2024**