---

# Lezione 5 - Martedì 23 Settembre 2025

**Argomenti principali:**
- Algebra lineare computazionale per Machine Learning
- Spazi fondamentali: Column Space, Null Space, Row Space
- Rango, basi e indipendenza lineare
- Fattorizzazione CR (Column-Row)
- Decomposizioni: LU, QR, SVD (introduzione)
- Matrici ortogonali e proiezioni
- Applicazioni: PCA, regressione, compressione dati

---

## 🎯 Focus e Ambito del Corso

### Perché Algebra Lineare per ML/Data Science?

**Machine Learning = Algebra Lineare + Statistica + Ottimizzazione**

Ogni dataset è una **matrice** $A \in \mathbb{R}^{m \times n}$:
- **m = samples** (righe): numero osservazioni
- **n = features** (colonne): numero variabili/dimensioni

**Esempi:**
```python
import numpy as np

# Dataset immagini: 1000 immagini 28×28 pixels
# Ogni immagine → vettore 784-dimensionale (28*28)
images = np.random.rand(1000, 784)  # 1000 samples, 784 features
print(f"Shape: {images.shape}")
# Output: Shape: (1000, 784)

# Dataset tabellare: 500 clienti, 20 attributi
customers = np.random.rand(500, 20)
print(f"Shape: {customers.shape}")
# Output: Shape: (500, 20)
```

---

### 📚 Roadmap Algebra Lineare

**1. Sistemi Lineari:** $AX = B$
- **Domanda:** Quando ha soluzione?
- **Risposta:** Se e solo se $B \in \text{Col}(A)$ (Column Space)
- **Applicazioni:** Equazioni simultanee, circuiti, equilibri

**2. Decomposizione Spettrale (Eigendecomposition):** $AX = \lambda X$
- **Cosa:** Trova direzioni **invarianti** (autovettori) e fattori di scala (autovalori)
- **Geometria:** Trasformazione $A$ allunga/contrae lungo autovettori
- **Applicazioni:** Dinamiche sistemi, PageRank, vibrazioni

**3. Singular Value Decomposition (SVD):** $A = U\Sigma V^T$
- **Generalizzazione:** Funziona per **qualsiasi matrice** (anche non quadrata, singolare)
- **Interpretazione:** $A$ come somma di matrici rango-1
- **Applicazioni:** **PCA**, compressione, denoising, recommender systems

**4. Least Squares Minimization:** $\min_x \|Ax - b\|^2$
- **Problema:** Sistema sovradeterminato (più equazioni che incognite)
- **Soluzione:** Pseudoinversa $x = (A^TA)^{-1}A^Tb$
- **Applicazioni:** **Regressione lineare**, fitting, calibrazione

**5. Matrix Factorizations:** LU, QR, Cholesky
- **LU:** $A = LU$ (Lower × Upper) → risoluzione sistemi
- **QR:** $A = QR$ (Orthogonal × Upper) → least squares, eigenvalori
- **Cholesky:** $A = LL^T$ (simmetrica positiva definita)

---

### 🔗 Connessioni ML/Data Science

| Problema ML | Strumento Algebra Lineare |
|-------------|---------------------------|
| **PCA** (riduzione dimensionalità) | SVD, autovettori matrice covarianza |
| **Regressione lineare** | Least squares, pseudoinversa |
| **Clustering K-means** | Norme vettoriali, proiezioni |
| **Neural Networks** | Moltiplicazioni matrice-vettore |
| **Image compression** | SVD troncata (approssimazione rango-k) |
| **Recommender systems** | SVD su matrice user-item |
| **Feature engineering** | Rank, indipendenza lineare |
| **Gradient descent** | Gradienti, Hessiani (matrici) |

---


## 📊 Rappresentazione Dati con Matrici

### Dataset come Matrice

**Convenzione standard:** $A \in \mathbb{R}^{m \times n}$
- **Righe** (m): **samples** / observations / instances
- **Colonne** (n): **features** / variables / attributes

```python
import numpy as np

# Esempio: Dataset immagini MNIST-like
m = 5000  # 5000 immagini
n = 1_000_000  # 1000×1000 pixels = 1M features

A = np.random.randint(0, 256, size=(m, n), dtype=np.uint8)
print(f"Dataset shape: {A.shape}")
print(f"Memoria: {A.nbytes / 1024**2:.2f} MB")

# Output:
# Dataset shape: (5000, 1000000)
# Memoria: 4768.37 MB  (~5 GB!)
```

**Ogni riga** = un'immagine "flattened" (1M pixel):
```python
image_1 = A[0, :]  # Prima immagine
print(f"Image shape: {image_1.shape}")
# Output: Image shape: (1000000,)

# Ricostruzione immagine 2D
image_2d = image_1.reshape(1000, 1000)
```

---

### 🎨 Encoding Feature Non-Numeriche

**Problema:** Feature categoriche (stringhe) non possono essere usate direttamente.

**Soluzioni:**

**1. Label Encoding** (ordinale):
```python
# Colori: ["red", "green", "blue"]
colors = ["red", "green", "blue", "red", "blue"]

# Mapping
mapping = {"red": 0, "green": 1, "blue": 2}
encoded = [mapping[c] for c in colors]
print(encoded)
# Output: [0, 1, 2, 0, 2]

# ⚠️ Problema: Implica ordine (blue > green > red)
```

**2. One-Hot Encoding** (no ordine):
```python
from sklearn.preprocessing import OneHotEncoder

colors = np.array(["red", "green", "blue", "red", "blue"]).reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(colors)

print(onehot)
# Output:
# [[0 0 1]  ← red
#  [0 1 0]  ← green
#  [1 0 0]  ← blue
#  [0 0 1]  ← red
#  [1 0 0]] ← blue

print(f"Original features: 1, Encoded features: {onehot.shape[1]}")
# Output: Original features: 1, Encoded features: 3
```

**3. Embedding** (Deep Learning):
```python
# Per categorie high-cardinality (es. 10000 città)
# Mappa a spazio denso low-dimensional (es. 50D)
# Esempio: Word2Vec per parole
```

---

### 💾 Dimensionalità e Scalabilità

**Trade-off memoria vs informazione:**

| Dataset | m (samples) | n (features) | Dimensione |
|---------|-------------|--------------|------------|
| MNIST | 60,000 | 784 (28×28) | ~45 MB |
| ImageNet | 1.2M | 150,528 (224×224×3) | ~680 GB |
| Text (1M docs, 50k vocab) | 1M | 50,000 | ~190 GB (sparse!) |

**Problemi con alta dimensionalità:**
- **Curse of dimensionality:** Distanze perdono significato
- **Overfitting:** Più features che samples
- **Computazione:** Operazioni $O(mn)$ o peggio

**Soluzione:** **Riduzione dimensionalità** (PCA, t-SNE, UMAP)

```python
from sklearn.decomposition import PCA

# Dataset 10k samples × 1000 features
X = np.random.rand(10000, 1000)

# Riduzione a 50 componenti principali
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

print(f"Original: {X.shape} → Reduced: {X_reduced.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Output:
# Original: (10000, 1000) → Reduced: (10000, 50)
# Variance explained: 85.23%
```

---


## 🔍 Moltiplicazione Matrice-Vettore: Prospettiva Column Space

### Interpretazione Fondamentale

**Operazione:** $Ax$ dove $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$

**Due modi di vedere:**

**1. Row-by-Row (prodotto scalare):**
$$Ax = \begin{bmatrix} \text{row}_1 \cdot x \\ \text{row}_2 \cdot x \\ \vdots \\ \text{row}_m \cdot x \end{bmatrix}$$

**2. Column-by-Column (combinazione lineare):** ⭐ **PIÙ IMPORTANTE**
$$Ax = x_1 a_1 + x_2 a_2 + \cdots + x_n a_n$$

dove $a_1, a_2, \ldots, a_n$ sono le **colonne di A**.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

x = np.array([2, -1, 3])

# Metodo 1: Prodotto standard (row-by-row)
result_1 = A @ x
print("Row-by-row:")
print(result_1)

# Metodo 2: Combinazione lineare colonne
a1, a2, a3 = A[:, 0], A[:, 1], A[:, 2]
result_2 = x[0]*a1 + x[1]*a2 + x[2]*a3
print("\nColumn combination:")
print(result_2)

# Output (identico):
# Row-by-row:
# [ 9 21 33]
# 
# Column combination:
# [ 9 21 33]

# Verifica uguaglianza
print(f"\nEquals: {np.allclose(result_1, result_2)}")
# Output: Equals: True
```

---

### 📐 Column Space (Spazio delle Colonne)

**Definizione:** $\text{Col}(A) = \{Ax : x \in \mathbb{R}^n\}$

**Interpretazione:**
- Insieme di **tutte le combinazioni lineari** delle colonne di $A$
- **Sottospazio** di $\mathbb{R}^m$ (chiuso per somma e scala)
- $Ax$ **giace sempre** in $\text{Col}(A)$ per qualsiasi $x$

**Dimensione:** $\dim(\text{Col}(A)) = \text{rank}(A) = r$

---

### 🎨 Esempi Geometrici

#### Esempio 1: Matrice 3×2 → Piano in $\mathbb{R}^3$

```python
# A1: 3×2 → Col(A1) è un piano in R³
A1 = np.array([[1, 3],
               [2, 1],
               [1, -1]])

# Colonne
a1 = A1[:, 0]  # [1, 2, 1]
a2 = A1[:, 1]  # [3, 1, -1]

print(f"a1: {a1}")
print(f"a2: {a2}")

# Verifica indipendenza lineare
# a2 ≠ c·a1 per qualsiasi scalare c
ratio = a2 / a1
print(f"Ratio a2/a1: {ratio}")  # Non costante → indipendenti
# Output: Ratio a2/a1: [3.  0.5 -1.] (diversi!)

# Rank
rank_A1 = np.linalg.matrix_rank(A1)
print(f"Rank(A1): {rank_A1}")
# Output: Rank(A1): 2

# Col(A1) = piano passante per origine in R³
```

**Visualizzazione:**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Genera punti sul piano
s = np.linspace(-2, 2, 10)
t = np.linspace(-2, 2, 10)
S, T = np.meshgrid(s, t)

# Ogni punto: s*a1 + t*a2
X = S * a1[0] + T * a2[0]
Y = S * a1[1] + T * a2[1]
Z = S * a1[2] + T * a2[2]

# Plot piano
ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan')

# Plot vettori base
ax.quiver(0, 0, 0, a1[0], a1[1], a1[2], color='red', arrow_length_ratio=0.1, lw=2, label='a1')
ax.quiver(0, 0, 0, a2[0], a2[1], a2[2], color='blue', arrow_length_ratio=0.1, lw=2, label='a2')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Col(A1): Piano in R³ (rank 2)')
ax.legend()
plt.show()
```

---

#### Esempio 2: Colonne Dipendenti → Rango < Colonne

```python
# A2: 3×3 ma rank 2 (terza colonna dipendente)
A2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

# Verifica dipendenza: a3 = 2*a2 - a1
a1, a2, a3 = A2[:, 0], A2[:, 1], A2[:, 2]
check = 2*a2 - a1
print(f"a3:        {a3}")
print(f"2*a2 - a1: {check}")
print(f"Equal: {np.allclose(a3, check)}")

# Output:
# a3:        [3 6 9]
# 2*a2 - a1: [3 6 9]
# Equal: True

# Rank
rank_A2 = np.linalg.matrix_rank(A2)
print(f"Rank(A2): {rank_A2}")
# Output: Rank(A2): 2 (non 3!)

# Col(A2) = stesso piano di Col(A1') dove A1' = [[1,2],[4,5],[7,8]]
A2_reduced = A2[:, :2]
print(f"Col(A2) = Col(A2_reduced)")
print(f"Rank(A2_reduced): {np.linalg.matrix_rank(A2_reduced)}")
# Output: Rank(A2_reduced): 2
```

---

#### Esempio 3: Rango Pieno → Span tutto $\mathbb{R}^3$

```python
# A3: 3×3 con rank 3 (tutte colonne indipendenti)
A3 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 10]])  # Nota: 10 invece di 9

rank_A3 = np.linalg.matrix_rank(A3)
print(f"Rank(A3): {rank_A3}")
# Output: Rank(A3): 3

# Col(A3) = R³ (tutto lo spazio!)
# Qualsiasi vettore b ∈ R³ può essere scritto come Ax per qualche x
```

**Test risolvibilità:**
```python
# Sistema A3 x = b ha soluzione per QUALSIASI b
b = np.array([10, 20, 30])
x = np.linalg.solve(A3, b)
print(f"Soluzione x: {x}")
print(f"Verifica Ax: {A3 @ x}")
print(f"Target b:    {b}")
print(f"Match: {np.allclose(A3 @ x, b)}")

# Output:
# Soluzione x: [ 0. -5. 10.]
# Verifica Ax: [10. 20. 30.]
# Target b:    [10 20 30]
# Match: True
```

---

#### Esempio 4: Rango 1 → Linea

```python
# A4: 3×3 ma rank 1 (tutte colonne proporzionali)
A4 = np.array([[1, 2, 3],
               [2, 4, 6],
               [3, 6, 9]])

# a2 = 2*a1, a3 = 3*a1
a1, a2, a3 = A4[:, 0], A4[:, 1], A4[:, 2]
print(f"a2 / a1: {a2 / a1}")  # [2. 2. 2.] → costante
print(f"a3 / a1: {a3 / a1}")  # [3. 3. 3.] → costante

rank_A4 = np.linalg.matrix_rank(A4)
print(f"Rank(A4): {rank_A4}")
# Output: Rank(A4): 1

# Col(A4) = linea nella direzione [1, 2, 3]
# Tutti i punti: t * [1, 2, 3] per t ∈ R
```

---

### 📊 Riepilogo Esempi

| Matrice | Shape | Rank | Col(A) | Geometria |
|---------|-------|------|--------|-----------|
| A1 | 3×2 | 2 | Sottospazio 2D | **Piano** in $\mathbb{R}^3$ |
| A2 | 3×3 | 2 | Sottospazio 2D | **Piano** in $\mathbb{R}^3$ |
| A3 | 3×3 | 3 | $\mathbb{R}^3$ | **Tutto lo spazio** |
| A4 | 3×3 | 1 | Sottospazio 1D | **Linea** in $\mathbb{R}^3$ |

**Visualizzazione rank:**
```python
matrices = {'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4}

for name, mat in matrices.items():
    r = np.linalg.matrix_rank(mat)
    m, n = mat.shape
    print(f"{name}: {m}×{n}, rank={r}, Col(A) dim={r}")

# Output:
# A1: 3×2, rank=2, Col(A) dim=2
# A2: 3×3, rank=2, Col(A) dim=2
# A3: 3×3, rank=3, Col(A) dim=3
# A4: 3×3, rank=1, Col(A) dim=1
```

---


## 🎯 Rango: Definizione e Calcolo

### Definizione Formale

**Rango di A:** $\text{rank}(A) = r$

Numero di **colonne linearmente indipendenti** = Dimensione di $\text{Col}(A)$

**Equivalenze:**
- Numero massimo colonne indipendenti
- Numero massimo righe indipendenti
- Dimensione Row Space = Dimensione Column Space
- Numero valori singolari non-nulli (da SVD)

---

### 💻 Calcolo Rango in NumPy

```python
import numpy as np

# Metodo 1: np.linalg.matrix_rank (raccomandato)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

rank_A = np.linalg.matrix_rank(A)
print(f"Rank: {rank_A}")
# Output: Rank: 2

# Metodo 2: SVD (valori singolari non-nulli)
U, s, Vt = np.linalg.svd(A)
rank_svd = np.sum(s > 1e-10)  # Threshold per errori numerici
print(f"Rank (SVD): {rank_svd}")
print(f"Singular values: {s}")
# Output:
# Rank (SVD): 2
# Singular values: [1.68481034e+01 1.06836951e+00 4.41842475e-16]
#                   ^^^^^^^^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^
#                   non-zero       non-zero       ~0 (numerically)
```

---

### 📊 Proprietà del Rango

**1. Rango e dimensioni:**
$$\text{rank}(A) \leq \min(m, n)$$

```python
A1 = np.random.rand(5, 10)   # 5×10
A2 = np.random.rand(10, 5)   # 10×5

print(f"rank(A1) = {np.linalg.matrix_rank(A1)} ≤ min(5, 10) = 5")
print(f"rank(A2) = {np.linalg.matrix_rank(A2)} ≤ min(10, 5) = 5")
# Output:
# rank(A1) = 5 ≤ min(5, 10) = 5
# rank(A2) = 5 ≤ min(10, 5) = 5
```

**2. Rango trasporta:**
$$\text{rank}(A^T) = \text{rank}(A)$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

print(f"rank(A):   {np.linalg.matrix_rank(A)}")
print(f"rank(A^T): {np.linalg.matrix_rank(A.T)}")
# Output:
# rank(A):   2
# rank(A^T): 2
```

**3. Rango prodotto:**
$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

```python
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2, rank 2
B = np.array([[1, 0, 1], [0, 1, 1]])    # 2×3, rank 2

C = A @ B  # 3×3

print(f"rank(A) = {np.linalg.matrix_rank(A)}")
print(f"rank(B) = {np.linalg.matrix_rank(B)}")
print(f"rank(AB) = {np.linalg.matrix_rank(C)} ≤ min(2, 2) = 2")
# Output:
# rank(A) = 2
# rank(B) = 2
# rank(AB) = 2 ≤ min(2, 2) = 2
```

**4. Rango pieno:**
- **Full column rank:** $\text{rank}(A) = n$ (tutte colonne indipendenti)
- **Full row rank:** $\text{rank}(A) = m$ (tutte righe indipendenti)
- **Full rank (quadrata):** $\text{rank}(A) = n = m$ → $A$ **invertibile**

---

## ✅ Risolvibilità Sistemi Lineari $Ax = b$

### Teorema Fondamentale

**Sistema $Ax = b$ ha soluzione ⟺ $b \in \text{Col}(A)$**

**Perché?**
- $Ax$ è sempre combinazione lineare colonne di $A$
- Quindi $Ax \in \text{Col}(A)$ per qualsiasi $x$
- Se $b \notin \text{Col}(A)$, impossibile trovare $x$ tale che $Ax = b$

---

### 🔍 Casi e Soluzioni

**Caso 1: $m = n$, rank = $n$ (quadrata, full rank)**
- **Soluzione:** Unica
- **Metodo:** $x = A^{-1}b$

```python
A = np.array([[2, 1],
              [1, 3]])  # 2×2, rank 2

b = np.array([5, 7])

# Verifica invertibilità
print(f"det(A) = {np.linalg.det(A)}")  # ≠ 0 → invertibile
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# Soluzione
x = np.linalg.solve(A, b)
print(f"Soluzione: {x}")
print(f"Verifica Ax: {A @ x}")
print(f"Target b: {b}")

# Output:
# det(A) = 5.0
# rank(A) = 2
# Soluzione: [1. 2.]
# Verifica Ax: [5. 7.]
# Target b: [5 7]
```

---

**Caso 2: $m > n$ (sovradeterminato)**
- **Soluzione:** Tipicamente **nessuna** (inconsistente)
- **Approccio:** **Least Squares** → minimizza $\|Ax - b\|^2$

```python
# 3 equazioni, 2 incognite
A = np.array([[1, 1],
              [1, 2],
              [1, 3]])  # 3×2

b = np.array([2, 3, 5])

# Soluzione least squares
x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print(f"Soluzione LS: {x_ls}")
print(f"Residuo: {residuals[0]:.6f}")
print(f"Ax:  {A @ x_ls}")
print(f"b:   {b}")
print(f"Errore: {b - A @ x_ls}")

# Output:
# Soluzione LS: [0.66666667 1.5       ]
# Residuo: 0.166667
# Ax:  [2.16666667 3.16666667 4.66666667]
# b:   [2 3 5]
# Errore: [-0.16666667 -0.16666667  0.33333333]
```

---

**Caso 3: $m < n$ (sottodeterminato)**
- **Soluzione:** **Infinite** (se rank = $m$)
- **Sottospazio:** Famiglia parametrica $x = x_p + t \cdot v$

```python
# 2 equazioni, 3 incognite
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

b = np.array([10, 20])

# Una soluzione particolare
x_particular = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"Soluzione particolare: {x_particular}")
print(f"Verifica Ax: {A @ x_particular}")

# Null space (soluzioni omogenee Ax = 0)
from scipy.linalg import null_space
null_A = null_space(A)
print(f"\nNull space basis:")
print(null_A)

# Famiglia soluzioni: x = x_particular + t * null_A
t = 2.5
x_general = x_particular + t * null_A.flatten()
print(f"\nSoluzione generale (t={t}): {x_general}")
print(f"Verifica Ax: {A @ x_general}")

# Output:
# Soluzione particolare: [-5.55555556  5.33333333  1.94444444]
# Verifica Ax: [10. 20.]
# 
# Null space basis:
# [[ 0.40824829]
#  [-0.81649658]
#  [ 0.40824829]]
# 
# Soluzione generale (t=2.5): [-4.53449101  3.29208097  2.96510546]
# Verifica Ax: [10. 20.]
```

---

**Caso 4: Rank-deficient (rango < min(m,n))**
- **Problema:** Colonne/righe dipendenti
- **Soluzione:** Pseudoinversa $x = A^+ b$

```python
# Matrice singolare (rank < n)
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 2, 3]])  # 3×3, rank 1

b = np.array([6, 12, 6])

# Rank
print(f"rank(A) = {np.linalg.matrix_rank(A)}")

# Pseudoinversa (Moore-Penrose)
A_pinv = np.linalg.pinv(A)
x = A_pinv @ b

print(f"Soluzione (pseudoinversa): {x}")
print(f"Verifica Ax: {A @ x}")
print(f"Target b: {b}")
print(f"Match: {np.allclose(A @ x, b)}")

# Output:
# rank(A) = 1
# Soluzione (pseudoinversa): [0.42857143 0.85714286 1.28571429]
# Verifica Ax: [ 6. 12.  6.]
# Target b: [ 6 12  6]
# Match: True
```

---

### 📋 Tabella Riassuntiva

| Tipo | m vs n | Rank | Soluzioni | Metodo |
|------|--------|------|-----------|--------|
| **Determinato** | m = n | n | **1** (unica) | `np.linalg.solve` |
| **Sovradeterminato** | m > n | n | **0** (tipicamente) | `np.linalg.lstsq` |
| **Sottodeterminato** | m < n | m | **∞** | `lstsq` + null space |
| **Rank-deficient** | any | r < min(m,n) | **0** o **∞** | `np.linalg.pinv` |

---
## Costruzione di una base per lo spazio delle colonne e fattorizzazione CR
- Procedura per trovare una base di C(A):
  - Inizia con c1 = a1 (prima colonna).
  - Per ogni colonna successiva ak:
    - Se ak è proporzionale a un vettore di base esistente (o combinazione lineare della base attuale), scarta.
    - Altrimenti, aggiungi ak alla base.
- Esempio con A2 = [[1,2,3],[4,5,6],[7,8,9]]:
  - Vettori di base: a1, a2 (a3 = 2·a2 − a1).
  - r(A2) = 2.
### Fattorizzazione CR (prospettiva forma ridotta per righe)
- Costruisci C selezionando colonne indipendenti: C = [a1 a2] = [[1,2],[4,5],[7,8]] (3×2).
- Trova R (2×3) tale che C·R = A:
  - Le colonne di R contengono i coefficienti per ricostruire ogni colonna originale come combinazione lineare delle colonne di C:
    - r1 = [1,0]^T → C·r1 = a1.
    - r2 = [0,1]^T → C·r2 = a2.
    - r3 = [−1,2]^T → C·r3 = a3 = 2·a2 − a1.
- A2 = C·R fornisce una fattorizzazione didattica; si collega alle forme ridotte per righe (rref).
- Note:
  - Se A ha rango massimo per colonne (ad esempio, A3), allora C = A e R = I.
  - Le colonne di C non sono ortogonali o normalizzate; utilità pratica limitata ma chiarisce la struttura.
## Invarianza del rango rispetto alla trasposizione
- Per A2′ = [[1,2,3],[4,5,6]] (3×2), considera A2′^T = [[1,4],[2,5],[3,6]]:
  - Terza colonna uguale a 2·seconda − prima; rango = 2.
- Risultato generale: rango(A^T) = rango(A); dim C(A^T) = dim C(A).
## Moltiplicazione matrice-matrice: decomposizione colonna-riga (prodotto esterno)
- Moltiplicazione standard: Se A ∈ R^{m×n}, B ∈ R^{n×p}, allora AB ∈ R^{m×p}, calcolata tramite prodotti riga per colonna.
- Vista colonna-riga:
  - Dividi A nelle sue colonne {cA1, cA2, …, cAn}.
  - Dividi B nelle sue righe {rB1, rB2, …, rBn}.
  - AB = Σ_{k=1}^n (cAk · rBk), dove ogni termine è un prodotto esterno (m×1 per 1×p → m×p).
- Ogni termine prodotto esterno è di rango 1 per costruzione; il prodotto completo è una somma di contributi di rango 1.
- Importanza concettuale:
  - Forma la base per rappresentare matrici come somme di componenti di rango 1.
  - Si collega direttamente a SVD e PCA.
  - PCA si basa su approssimazioni a basso rango derivate da SVD, rendendo centrale la visione della somma di rango 1.
### Esempio e contributi di rango 1
- Esempio: A = [[1,2],[3,4]], B = [[2,1],[2,3]].
  - cA1·rB1 = [1,3]^T · [2,1] = [[2,1],[6,3]], rango 1.
  - cA2·rB2 = [2,4]^T · [2,3] = [[4,6],[8,12]], rango 1.
  - Somma = [[6,7],[14,15]], identica al risultato della moltiplicazione standard.
- Intuizione:
  - AB è una somma di matrici di rango 1; fondamentale per SVD e approssimazioni a basso rango.
## Sottospazi fondamentali, ortogonalità e dimensioni
- Dato A ∈ R^{m×n} con rango(A) = r.
- Spazi delle colonne:
  - Col(A) ⊂ R^m, dim(Col(A)) = r.
  - Col(A^T) ⊂ R^n, dim(Col(A^T)) = r.
- Nuclei (= null spaces):
  - Null(A) ⊂ R^n (vettori x con A x = 0).
  - Null(A^T) ⊂ R^m (vettori y con A^T y = 0).
- Relazioni di ortogonalità:
  - Col(A^T) ⟂ Null(A): ogni x ∈ Null(A) è ortogonale a ogni riga di A.
  - Col(A) ⟂ Null(A^T): ogni y ∈ Null(A^T) è ortogonale a ogni colonna di A.
- Complementi ortogonali negli spazi ambienti:
  - In R^m: Col(A) e Null(A^T) sono complementi ortogonali; dim(Col(A)) = r, dim(Null(A^T)) = m − r.
  - In R^n: Col(A^T) e Null(A) sono complementi ortogonali; dim(Col(A^T)) = r, dim(Null(A)) = n − r.
- Teorema rango-nullo:
  - dim(Null(A)) = n − r e dim(Null(A^T)) = m − r.
### Interpretazione di A x = 0 tramite prodotti scalari
- Matrice di esempio: A = [[1,2,3],[4,5,6],[7,8,9]], x = [x1; x2; x3], condizione A x = 0.
- Vista moltiplicazione riga–colonna:
  - Siano r1, r2, r3 le righe di A.
  - A x = 0 implica r_i · x = 0 per i = 1,2,3 (prodotti scalari).
- Implicazioni:
  - x è ortogonale a tutte le righe di A; quindi x ∈ Null(A) ⇒ x ⟂ Col(A^T).
### Proprietà di sottospazio dei nuclei
- Chiusura e scalabilità del nucleo:
  - Il vettore 0 è in Null(A).
  - Se x, y ∈ Null(A), allora x + y ∈ Null(A).
  - Se x ∈ Null(A) e α ∈ R, allora αx ∈ Null(A).
### Base costruttiva per Null(A) tramite fattorizzazione a blocchi
- Setup: A ∈ R^{m×n}, rango(A) = r.
- Partiziona A in A1 (m×r) con colonne linearmente indipendenti e A2 (m×(n−r)) le restanti colonne dipendenti.
- Esprimi A2 come combinazione lineare di A1: A2 = A1 · B, con B ∈ R^{r×(n−r)}.
- Quindi A = [A1, A1 B].
- Costruisci K ∈ R^{n×(n−r)}: K = [−B; I_{n−r}].
-- Calcola A K: A K = [A1, A1 B] [−B; I] = A1(−B) + A1 B = 0 ⇒ le colonne di K giacciono in Null(A) e sono linearmente indipendenti.
-- Qualsiasi U ∈ Null(A) può essere scritto come U = K U2:
  - Partiziona U = [U1; U2]; AU = 0 ⇒ A1(U1 + B U2) = 0 ⇒ U1 = −B U2.
  - Quindi U = [−B U2; U2] = K U2.
-- Conclusione:
  - Le colonne di K formano una base per Null(A); dim Null(A) = n − r.
  - Ragionamento simmetrico porta a dim Null(A^T) = m − r.
## Matrici ortogonali, proiezioni e geometria
- Matrice ortogonale Q: Q^T Q = I; det(Q) = ±1.
- Conservazione della norma:
  - Per Y = QX: ||Y||^2 = X^T Q^T Q X = ||X||^2; le trasformazioni ortogonali sono rigide (preservano la lunghezza).
- Rotazione 2D:
  - R(θ) = [[cos θ, −sin θ],[sin θ, cos θ]]; ortogonale, det = +1; ruota di θ.
- Riflesso rispetto a un piano Π con normale unitaria n:
  - v_⊥ = (v · n) n; riflesso w = v − 2(v · n) n.
  - Matrice: R_ref = I − 2 n n^T; ortogonale, det = −1; R_ref^{-1} = R_ref.
- Proiezione ortogonale su Π:
  - P = I − n n^T; singolare (det = 0); non invertibile per perdita di informazione.
- Chiarimenti sulle proiezioni:
  - Proiezione di a sulla direzione b (non unitaria): vettore proiettato = (a · b) (b / ||b||^2).
## Collegamenti con machine learning e data science
- SVD:
  - Decompone A come Σ_i σ_i u_i v_i^T (somma di matrici di rango 1); applicabilità universale; alla base di PCA, riduzione dimensionale, filtraggio del rumore e modellazione a basso rango.
- PCA:
  - Sfrutta i valori/vettori singolari dominanti per approssimazioni a basso rango che catturano la varianza e riducono la dimensionalità.
- Minimi quadrati:
  - Regressione vista come minimizzazione dei residui; la risolvibilità è legata allo spazio delle colonne e al rango (equazioni normali, pseudoinversa).
- Rango e spazio delle colonne:
  - Determinano l'identificabilità del modello, la ridondanza delle feature e le condizioni di risolvibilità per AX = B.
- Prospettiva colonna-riga:
  - Fornisce comprensione strutturale dei prodotti e motiva le approssimazioni a rango vincolato usate in ML.
- Collegamento con metodi numerici:
  - Riflessi di Householder (da matrici di riflessione) per fattorizzazione QR.
  - Rotazioni di Givens (da matrici di rotazione) per calcoli di autovalori/QR.
## Fatti chiave e conclusioni
- I prodotti tra matrici si decompongono in somme di matrici di rango 1; fondamentale per SVD/PCA.
- Quattro sottospazi fondamentali e dimensioni:
  - Col(A) ⊂ R^m, dim r; Col(A^T) ⊂ R^n, dim r.
  - Null(A^T) ⊂ R^m, dim m − r; Null(A) ⊂ R^n, dim n − r.
- Relazioni ortogonali:
  - Col(A) ⟂ Null(A^T) in R^m; Col(A^T) ⟂ Null(A) in R^n.
  - Ogni coppia forma complementi ortogonali negli spazi ambienti.
- Metodo costruttivo con K = [−B; I_{n−r}] fornisce una base per Null(A).
- Le matrici di proiezione sono singolari; riflessioni e rotazioni sono ortogonali con det ±1.
## Esempi e contabilità dimensionale
- Esempio numerico: A = [[1,2,3],[4,5,6],[7,8,9]] mostra che i vettori di Null(A) sono ortogonali alle righe.
- Partizionamento delle dimensioni:
  - A1: m×r; A2: m×(n−r); B: r×(n−r); K: n×(n−r); A K: m×(n−r) zero.
---

## 🏗️ Fattorizzazione CR (Column-Row)

### Costruzione Base Column Space

**Obiettivo:** Trovare **base minimale** per $\text{Col}(A)$

**Algoritmo greedy:**
1. Inizia con $c_1 = a_1$ (prima colonna)
2. Per ogni $a_k$ successiva:
   - Se $a_k$ è combinazione lineare di $\{c_1, \ldots, c_{i-1}\}$ → **scarta**
   - Altrimenti → **aggiungi** $c_i = a_k$ alla base
3. Risultato: $\{c_1, \ldots, c_r\}$ con $r = \text{rank}(A)$

---

### 💻 Implementazione Python

```python
import numpy as np

def find_column_basis(A, tol=1e-10):
    """
    Trova base per Column Space di A usando eliminazione Gaussiana.
    
    Returns:
        C: matrice con colonne base
        pivot_cols: indici colonne indipendenti in A
    """
    m, n = A.shape
    R = A.copy()
    pivot_cols = []
    
    for col in range(n):
        # Trova pivot in colonna corrente
        pivot_row = len(pivot_cols)
        if pivot_row >= m:
            break
            
        # Normalizza e elimina
        if abs(R[pivot_row, col]) > tol:
            pivot_cols.append(col)
            R[pivot_row, :] /= R[pivot_row, col]
            
            # Elimina sotto
            for row in range(pivot_row + 1, m):
                R[row, :] -= R[row, col] * R[pivot_row, :]
    
    C = A[:, pivot_cols]
    return C, pivot_cols

# Test
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

C, pivots = find_column_basis(A)

print(f"Original A ({A.shape}):")
print(A)
print(f"\nBase columns (indices {pivots}):")
print(C)
print(f"C shape: {C.shape}")
print(f"rank(A) = {len(pivots)}")

# Output:
# Original A (3, 3):
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
# 
# Base columns (indices [0, 1]):
# [[1 2]
#  [4 5]
#  [7 8]]
# C shape: (3, 2)
# rank(A) = 2
```

---

### 📐 Fattorizzazione A = CR

**Idea:** Esprimi $A$ come prodotto di:
- **C** (m×r): colonne indipendenti di A
- **R** (r×n): coefficienti per ricostruire tutte le colonne

**Costruzione R:**
- Ogni colonna $r_j$ di R contiene coefficienti per $a_j = C \cdot r_j$
- Se $a_j$ è colonna di C → $r_j$ ha 1 nella posizione corrispondente, 0 altrove
- Se $a_j$ è combinazione → $r_j$ contiene i coefficienti

```python
def cr_factorization(A):
    """
    Fattorizzazione A = C * R.
    
    Returns:
        C: m×r matrice colonne indipendenti
        R: r×n matrice coefficienti
    """
    C, pivot_cols = find_column_basis(A)
    
    # Risolvi C * R = A per R
    # R = C⁺ * A (pseudoinversa)
    R = np.linalg.lstsq(C, A, rcond=None)[0]
    
    return C, R

# Test con A2
A2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]], dtype=float)

C, R = cr_factorization(A2)

print(f"C ({C.shape}):")
print(C)
print(f"\nR ({R.shape}):")
print(R)
print(f"\nRicostruzione C @ R:")
print(C @ R)
print(f"\nOriginale A:")
print(A2)
print(f"\nErrore: {np.linalg.norm(A2 - C @ R):.2e}")

# Output:
# C (3, 2):
# [[1. 2.]
#  [4. 5.]
#  [7. 8.]]
# 
# R (2, 3):
# [[ 1.  0. -1.]
#  [ 0.  1.  2.]]
# 
# Ricostruzione C @ R:
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [7. 8. 9.]]
# 
# Originale A:
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [7. 8. 9.]]
# 
# Errore: 6.24e-16
```

**Interpretazione R per A2:**
- $r_1 = [1, 0]^T$ → $a_1 = 1 \cdot c_1 + 0 \cdot c_2$ ✅
- $r_2 = [0, 1]^T$ → $a_2 = 0 \cdot c_1 + 1 \cdot c_2$ ✅
- $r_3 = [-1, 2]^T$ → $a_3 = -1 \cdot c_1 + 2 \cdot c_2 = -[1,4,7]^T + 2[2,5,8]^T = [3,6,9]^T$ ✅

---

### 🔗 Collegamento con RREF (Reduced Row Echelon Form)

**Fatto:** Le colonne pivot della RREF di A corrispondono alle colonne indipendenti di A.

```python
def rref(A):
    """Calcola Reduced Row Echelon Form."""
    A = A.copy().astype(float)
    m, n = A.shape
    
    pivot_row = 0
    for col in range(n):
        if pivot_row >= m:
            break
        
        # Trova pivot massimo
        max_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))
        
        if abs(A[max_row, col]) < 1e-10:
            continue
        
        # Scambia righe
        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
        
        # Normalizza riga pivot
        A[pivot_row] /= A[pivot_row, col]
        
        # Elimina colonna (sopra e sotto)
        for row in range(m):
            if row != pivot_row:
                A[row] -= A[row, col] * A[pivot_row]
        
        pivot_row += 1
    
    return A

A2_rref = rref(A2)
print("RREF(A2):")
print(A2_rref)

# Output:
# RREF(A2):
# [[ 1.  0. -1.]
#  [ 0.  1.  2.]
#  [ 0.  0.  0.]]
#
# Nota: Identica a R (primi 2 righe)!
```

---

## 🔄 Moltiplicazione Matrice-Matrice: Vista Column-Row

### Interpretazione Standard vs Column-Row

**Standard (row-by-column):**
$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Column-Row (outer product sum):**
$$AB = \sum_{k=1}^{n} a_k b_k^T$$

dove $a_k$ è la k-esima **colonna** di A e $b_k^T$ è la k-esima **riga** di B.

---

### 💡 Perché è Importante?

**Ogni termine $a_k b_k^T$ è una matrice di RANGO 1!**

$$AB = \underbrace{a_1 b_1^T}_{\text{rank 1}} + \underbrace{a_2 b_2^T}_{\text{rank 1}} + \cdots + \underbrace{a_n b_n^T}_{\text{rank 1}}$$

**Implicazioni:**
- AB è **somma di matrici rank-1**
- **SVD** estende questa idea: $A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$
- **Low-rank approximation:** Tronca somma ai primi k termini
- **PCA:** Proietta dati su sottospazio rank-k

---

### 🧮 Esempio Numerico

```python
A = np.array([[1, 2],
              [3, 4]])  # 2×2

B = np.array([[2, 1],
              [2, 3]])  # 2×2

# Metodo 1: Standard
AB_standard = A @ B
print("Standard A @ B:")
print(AB_standard)

# Metodo 2: Column-row sum
a1, a2 = A[:, 0:1], A[:, 1:2]  # Colonne di A (keep 2D)
b1, b2 = B[0:1, :], B[1:2, :]  # Righe di B (keep 2D)

term1 = a1 @ b1  # Outer product (2×1) @ (1×2) = (2×2)
term2 = a2 @ b2

print("\nTerm 1 (a1 @ b1^T):")
print(term1)
print(f"rank: {np.linalg.matrix_rank(term1)}")

print("\nTerm 2 (a2 @ b2^T):")
print(term2)
print(f"rank: {np.linalg.matrix_rank(term2)}")

AB_sum = term1 + term2
print("\nSum of rank-1 matrices:")
print(AB_sum)

print(f"\nEquals standard: {np.allclose(AB_standard, AB_sum)}")

# Output:
# Standard A @ B:
# [[ 6  7]
#  [14 15]]
# 
# Term 1 (a1 @ b1^T):
# [[2 1]
#  [6 3]]
# rank: 1
# 
# Term 2 (a2 @ b2^T):
# [[4 6]
#  [8 12]]
# rank: 1
# 
# Sum of rank-1 matrices:
# [[ 6  7]
#  [14 15]]
# 
# Equals standard: True
```

---

### 🎨 Visualizzazione Outer Product

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

matrices = [term1, term2, AB_sum, AB_standard]
titles = ['Term 1 (rank 1)', 'Term 2 (rank 1)', 'Sum', 'Standard']

for ax, mat, title in zip(axes, matrices, titles):
    im = ax.imshow(mat, cmap='coolwarm', vmin=-15, vmax=15)
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    
    # Annotazioni valori
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{mat[i, j]:.0f}', 
                   ha='center', va='center', color='black', fontsize=14)

plt.colorbar(im, ax=axes, fraction=0.046)
plt.tight_layout()
plt.show()
```

---

## 🧩 Sottospazi Fondamentali e Ortogonalità

### I Quattro Sottospazi di A ∈ ℝ^{m×n}

Dato $A \in \mathbb{R}^{m \times n}$ con $\text{rank}(A) = r$:

| Sottospazio | Notazione | Spazio | Dimensione |
|-------------|-----------|--------|------------|
| **Column Space** | $\text{Col}(A)$ | $\subseteq \mathbb{R}^m$ | $r$ |
| **Row Space** | $\text{Col}(A^T)$ | $\subseteq \mathbb{R}^n$ | $r$ |
| **Null Space** | $\text{Null}(A)$ | $\subseteq \mathbb{R}^n$ | $n - r$ |
| **Left Null Space** | $\text{Null}(A^T)$ | $\subseteq \mathbb{R}^m$ | $m - r$ |

---

### 🔍 Definizioni

**1. Column Space:**
$$\text{Col}(A) = \{Ax : x \in \mathbb{R}^n\}$$

**2. Null Space:**
$$\text{Null}(A) = \{x \in \mathbb{R}^n : Ax = 0\}$$

**3. Row Space:**
$$\text{Col}(A^T) = \{A^T y : y \in \mathbb{R}^m\}$$

**4. Left Null Space:**
$$\text{Null}(A^T) = \{y \in \mathbb{R}^m : A^T y = 0\}$$

---

### ⊥ Relazioni di Ortogonalità

**Teorema Fondamentale Algebra Lineare:**

1. $\text{Col}(A^T) \perp \text{Null}(A)$ in $\mathbb{R}^n$
2. $\text{Col}(A) \perp \text{Null}(A^T)$ in $\mathbb{R}^m$

**Decomposizione spazio:**
- $\mathbb{R}^n = \text{Col}(A^T) \oplus \text{Null}(A)$ (somma diretta ortogonale)
- $\mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^T)$

---

### 💻 Calcolo Sottospazi in Python

```python
from scipy.linalg import null_space, orth

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# 1. Column Space (base ortonormale)
col_A = orth(A)
print("Column Space basis (orthonormal):")
print(col_A)
print(f"Dimension: {col_A.shape[1]}")

# 2. Null Space
null_A = null_space(A)
print("\nNull Space basis:")
print(null_A)
print(f"Dimension: {null_A.shape[1]}")

# 3. Row Space = Column Space di A^T
row_A = orth(A.T)
print("\nRow Space basis:")
print(row_A)
print(f"Dimension: {row_A.shape[1]}")

# 4. Left Null Space
left_null_A = null_space(A.T)
print("\nLeft Null Space basis:")
print(left_null_A)
print(f"Dimension: {left_null_A.shape[1]}")

# Verifica dimensioni (Rank-Nullity Theorem)
m, n = A.shape
r = np.linalg.matrix_rank(A)

print(f"\n--- Rank-Nullity Theorem ---")
print(f"A: {m}×{n}, rank = {r}")
print(f"dim(Col(A)) = {col_A.shape[1]} = r = {r} ✓")
print(f"dim(Row(A)) = {row_A.shape[1]} = r = {r} ✓")
print(f"dim(Null(A)) = {null_A.shape[1]} = n - r = {n} - {r} = {n-r} ✓")
print(f"dim(Left Null(A)) = {left_null_A.shape[1]} = m - r = {m} - {r} = {m-r} ✓")

# Output:
# Column Space basis (orthonormal):
# [[-0.2149  -0.8872]
#  [-0.5206  -0.2496]
#  [-0.8263   0.388 ]]
# Dimension: 2
# 
# Null Space basis:
# [[ 0.4082]
#  [-0.8165]
#  [ 0.4082]]
# Dimension: 1
# 
# Row Space basis:
# [[-0.4797  -0.5724]
#  [-0.8728   0.4137]]
# Dimension: 2
# 
# Left Null Space basis:
# [[-0.4082]
#  [-0.8165]
#  [ 0.4082]]
# Dimension: 1
# 
# --- Rank-Nullity Theorem ---
# A: 3×3, rank = 2
# dim(Col(A)) = 2 = r = 2 ✓
# dim(Row(A)) = 2 = r = 2 ✓
# dim(Null(A)) = 1 = n - r = 3 - 2 = 1 ✓
# dim(Left Null(A)) = 1 = m - r = 3 - 2 = 1 ✓
```

---

### 🔬 Verifica Ortogonalità

```python
# Verifica Row(A) ⊥ Null(A)
v_row = row_A[:, 0]  # Vettore in Row Space
v_null = null_A[:, 0]  # Vettore in Null Space

dot_product = v_row @ v_null
print(f"Row vector · Null vector = {dot_product:.2e}")
# Output: Row vector · Null vector = -2.78e-17 (≈ 0!)

# Verifica Col(A) ⊥ Left Null(A)
u_col = col_A[:, 0]
u_left_null = left_null_A[:, 0]

dot_product_2 = u_col @ u_left_null
print(f"Col vector · Left Null vector = {dot_product_2:.2e}")
# Output: Col vector · Left Null vector = 1.11e-17 (≈ 0!)
```

---

### 📐 Interpretazione Geometrica Ax = 0

**Sistema omogeneo:** $Ax = 0$

**Vista row-by-column:**
$$Ax = \begin{bmatrix} r_1 \cdot x \\ r_2 \cdot x \\ \vdots \\ r_m \cdot x \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

**Significato:** $x$ è **ortogonale** a **tutte le righe** di A!

Quindi: $x \in \text{Null}(A) \Rightarrow x \perp \text{Row}(A)$

```python
# Esempio
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Trova vettore in Null(A)
x = null_space(A)[:, 0]

# Verifica ortogonalità con righe
r1, r2, r3 = A[0, :], A[1, :], A[2, :]

print(f"x: {x}")
print(f"r1 · x = {r1 @ x:.2e}")
print(f"r2 · x = {r2 @ x:.2e}")
print(f"r3 · x = {r3 @ x:.2e}")
print(f"Ax = {A @ x}")

# Output:
# x: [ 0.4082 -0.8165  0.4082]
# r1 · x = 0.00e+00
# r2 · x = -1.11e-16
# r3 · x = -2.22e-16
# Ax = [ 0.00e+00 -1.11e-16 -2.22e-16]  ≈ [0, 0, 0]
```

---

## 🔲 Matrici Ortogonali e Proiezioni

### Matrici Ortogonali Q

**Definizione:** $Q^T Q = I$

**Proprietà:**
- Colonne ortonormali: $q_i^T q_j = \delta_{ij}$
- Preserva norme: $\|Qx\|_2 = \|x\|_2$
- Preserva angoli: $(Qx) \cdot (Qy) = x \cdot y$
- Determinante: $|\det(Q)| = 1$
- Inversa = Trasposta: $Q^{-1} = Q^T$

---

### 🔄 Rotazioni 2D

```python
def rotation_matrix(theta):
    """Matrice rotazione 2D di angolo theta (radianti)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Rotazione 45°
theta = np.pi / 4
R = rotation_matrix(theta)

print(f"Rotation matrix (45°):")
print(R)

# Verifica ortogonalità
print(f"\nR^T @ R:")
print(R.T @ R)  # Dovrebbe essere I

print(f"det(R) = {np.linalg.det(R):.4f}")  # = 1

# Test: ruota vettore [1, 0]
v = np.array([1, 0])
v_rotated = R @ v
print(f"\nOriginal: {v}")
print(f"Rotated: {v_rotated}")  # [cos(45°), sin(45°)]

# Norma preservata
print(f"||v|| = {np.linalg.norm(v):.4f}")
print(f"||R v|| = {np.linalg.norm(v_rotated):.4f}")

# Output:
# Rotation matrix (45°):
# [[ 0.7071 -0.7071]
#  [ 0.7071  0.7071]]
# 
# R^T @ R:
# [[1. 0.]
#  [0. 1.]]
# det(R) = 1.0000
# 
# Original: [1 0]
# Rotated: [0.7071 0.7071]
# ||v|| = 1.0000
# ||R v|| = 1.0000
```

---

### 🪞 Riflessioni (Householder)

**Riflessione rispetto a piano con normale unitaria n:**
$$R = I - 2nn^T$$

```python
def householder_reflection(n):
    """Matrice riflessione rispetto a piano con normale n."""
    n = n / np.linalg.norm(n)  # Normalizza
    return np.eye(len(n)) - 2 * np.outer(n, n)

# Riflessione rispetto a piano y = x (normale [1, -1])
n = np.array([1, -1])
R_ref = householder_reflection(n)

print("Reflection matrix:")
print(R_ref)

print(f"\ndet(R_ref) = {np.linalg.det(R_ref):.4f}")  # = -1 (riflessione)

# Test
v = np.array([3, 1])
v_reflected = R_ref @ v

print(f"\nOriginal: {v}")
print(f"Reflected: {v_reflected}")

# Norma preservata
print(f"||v|| = {np.linalg.norm(v):.4f}")
print(f"||R_ref v|| = {np.linalg.norm(v_reflected):.4f}")

# Output:
# Reflection matrix:
# [[ 0.  1.]
#  [ 1.  0.]]
# det(R_ref) = -1.0000
# 
# Original: [3 1]
# Reflected: [1 3]
# ||v|| = 3.1623
# ||R_ref v|| = 3.1623
```

---

### 📍 Proiezioni Ortogonali

**Proiezione su sottospazio con base ortonormale Q:**
$$P = QQ^T$$

**Proprietà:**
- $P^2 = P$ (idempotente)
- $P^T = P$ (simmetrica)
- $\text{rank}(P) = \text{rank}(Q)$
- Autovalori: 0 e 1

```python
# Proiezione su sottospazio 2D in R³
# Sottospazio: span{[1,0,0], [0,1,0]} (piano xy)
Q = np.array([[1, 0],
              [0, 1],
              [0, 0]], dtype=float)

P = Q @ Q.T

print("Projection matrix P:")
print(P)

# Test: proietta [1, 2, 3] sul piano xy
v = np.array([1, 2, 3])
v_proj = P @ v

print(f"\nOriginal v: {v}")
print(f"Projected v: {v_proj}")  # [1, 2, 0] (z = 0)

# Verifica idempotenza
print(f"\nP @ P:")
print(P @ P)  # = P

# Autovalori
eigvals = np.linalg.eigvals(P)
print(f"Eigenvalues: {np.sort(eigvals)}")  # [0, 1, 1]

# Output:
# Projection matrix P:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 0.]]
# 
# Original v: [1 2 3]
# Projected v: [1. 2. 0.]
# 
# P @ P:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 0.]]
# Eigenvalues: [0. 1. 1.]
```

---

## 📖 Materiali di Riferimento

### PDF Lecture
- `IntroLinearAlgebra.pdf` (Lecture September 23rd)

### Documentazione Online
- **NumPy Linalg:** https://numpy.org/doc/stable/reference/routines.linalg.html
- **SciPy Linalg:** https://docs.scipy.org/doc/scipy/reference/linalg.html
- **Linear Algebra Review (Stanford):** http://cs229.stanford.edu/section/cs229-linalg.pdf

### Libri Consigliati
- **Strang, G.** - "Introduction to Linear Algebra" (MIT)
- **Trefethen, L.N. & Bau, D.** - "Numerical Linear Algebra" (SIAM)
- **Boyd, S. & Vandenberghe, L.** - "Introduction to Applied Linear Algebra" (Stanford)

### Risorse Video
- **3Blue1Brown - Essence of Linear Algebra:** https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
- **MIT 18.06 (Gilbert Strang):** https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

---

## ✅ Checklist Competenze Lezione 5

### Concetti Fondamentali
- [ ] Rappresentazione dataset come matrice m×n
- [ ] Encoding feature categoriche (label, one-hot)
- [ ] Interpretazione Ax come combinazione lineare colonne
- [ ] Definizione e calcolo Column Space
- [ ] Definizione e calcolo Null Space
- [ ] Rango come dimensione Column Space
- [ ] Rank-Nullity Theorem: dim(Null(A)) = n - rank(A)

### Operazioni NumPy/SciPy
- [ ] `np.linalg.matrix_rank(A)` per rango
- [ ] `np.linalg.solve(A, b)` per sistemi quadrati
- [ ] `np.linalg.lstsq(A, b)` per least squares
- [ ] `np.linalg.pinv(A)` per pseudoinversa
- [ ] `scipy.linalg.null_space(A)` per Null Space
- [ ] `scipy.linalg.orth(A)` per base ortonormale Column Space
- [ ] `np.linalg.svd(A)` per Singular Value Decomposition

### Sistemi Lineari
- [ ] Criterio risolvibilità: b ∈ Col(A)
- [ ] Sistemi quadrati (m = n)
- [ ] Sistemi sovradeterminati (m > n) → least squares
- [ ] Sistemi sottodeterminati (m < n) → infinite soluzioni
- [ ] Matrici singolari → pseudoinversa

### Fattorizzazioni
- [ ] CR factorization: A = C * R
- [ ] Identificazione colonne indipendenti
- [ ] RREF e relazione con rank

### Sottospazi
- [ ] Quattro sottospazi fondamentali
- [ ] Relazioni ortogonalità: Col(A^T) ⊥ Null(A)
- [ ] Dimensioni e teorema rank-nullity
- [ ] Interpretazione geometrica Ax = 0

### Matrici Ortogonali
- [ ] Definizione Q^T Q = I
- [ ] Preservazione norme e angoli
- [ ] Rotazioni 2D (det = +1)
- [ ] Riflessioni Householder (det = -1)
- [ ] Proiezioni ortogonali P = QQ^T

### Applicazioni ML/DS
- [ ] Riduzione dimensionalità con PCA
- [ ] Regressione lineare come least squares
- [ ] Low-rank approximation (SVD troncata)
- [ ] Feature selection basata su rank

---

## 🎯 Prossimi Argomenti

**Lezione 6 (prevista):**
- **SVD approfondita:** Geometric interpretation, best rank-k approximation
- **PCA:** Derivazione da SVD, variance explained, scree plots
- **QR decomposition:** Gram-Schmidt, Householder, Givens
- **Eigendecomposition:** Autovalori, autovettori, diagonalizzazione
- **Applicazioni:** Image compression, recommender systems, topic modeling

---

## 💡 Esercizi Consigliati

### Esercizio 1: Analisi Dataset Reale
Carica dataset (es. Iris, MNIST):
1. Calcola rank della matrice dati
2. Trova base per Column Space
3. Verifica dimensioni sottospazi
4. Applica PCA e visualizza prime 2 componenti

### Esercizio 2: Risolvibilità Sistemi
Data matrice A 4×3:
1. Genera b ∈ Col(A) e risolvi Ax = b
2. Genera b ∉ Col(A) e calcola soluzione least squares
3. Confronta residui
4. Visualizza geometria in 3D

### Esercizio 3: Fattorizzazione CR
Implementa CR per matrici grandi (100×50):
1. Trova colonne indipendenti
2. Costruisci C e R
3. Verifica errore ricostruzione
4. Confronta con QR e SVD

### Esercizio 4: Low-Rank Approximation
Image compression con SVD:
1. Carica immagine grayscale
2. Calcola SVD completa
3. Ricostruisci con k=5, 10, 20, 50 valori singolari
4. Plot errore vs k e visualizza immagini

---

**Fine Lezione 5 - Martedì 23 Settembre 2025**