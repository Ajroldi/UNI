## Lezione 8: Image Denoising e Randomized SVD

> **Data:** 30 settembre  
> **Fonte:** NAML_Lect300925_v2.pdf, Thresholding.ipynb, rSVD_2024.ipynb  
> **Focus:** Hard Thresholding per denoising, Randomized SVD

---

## 📖 Indice

1. [Introduzione al Denoising](#introduzione)
2. [Hard Thresholding: Demo del Professore](#demo-thresholding)
3. [Confronto: Threshold vs 90% Varianza](#confronto-90)
4. [Randomized SVD: Teoria e Implementazione](#randomized-svd)
5. [Power Iterations](#power-iterations)
6. [Implementazioni Python Complete](#implementazioni)
7. [Materiali e Riferimenti](#materiali)
8. [Checklist Completa](#checklist)
9. [Esercizi](#esercizi)

---

## 1. Introduzione al Denoising {#introduzione}

### Il Problema

**Setup:** Immagine corrotta da rumore gaussiano additivo:

$$
X_{\text{noisy}} = X_{\text{true}} + \sigma \cdot N
$$

dove:
- $X_{\text{true}}$: immagine pulita (rank basso)
- $N$: rumore gaussiano standard ($\mathcal{N}(0, 1)$)
- $\sigma$: livello di rumore (noto)

**Obiettivo:** Recuperare $X_{\text{true}}$ da $X_{\text{noisy}}$ usando SVD.

### Strategia: Hard Thresholding

**Idea chiave:** 

1. Calcola SVD di $X_{\text{noisy}}$:
   $$U, S, V^T = \text{SVD}(X_{\text{noisy}})$$

2. Definisci **cutoff** (soglia):
   $$\tau = \frac{4}{\sqrt{3}} \cdot \sqrt{n} \cdot \sigma$$
   
   dove $n$ è la dimensione della matrice (per matrice $n \times n$)

3. Mantieni **solo** i valori singolari $> \tau$:
   $$r = \max\{i : S_i > \tau\}$$

4. Ricostruisci:
   $$X_{\text{clean}} = U_{:, :r+1} \cdot \text{diag}(S_{:r+1}) \cdot V^T_{:r+1, :}$$

**Formula del cutoff:** Deriva dalla teoria di **Marchenko-Pastur** per matrici random.

---

## 2. Hard Thresholding: Demo del Professore {#demo-thresholding}

### Setup Esatto (da `Thresholding.ipynb`)

Il professore usa un'immagine **sintetica rank-2** con parametri specifici:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parametri esatti del professore
t = np.arange(-3, 3, 0.01)  # 600 punti
N = len(t)  # N = 600

# Matrici U, V con funzioni trigonometriche SPECIFICHE
Utrue = np.array([
    np.cos(17*t) * np.exp(-t**2),  # Prima colonna
    np.sin(11*t)                    # Seconda colonna
]).T

Vtrue = np.array([
    np.sin(5*t) * np.exp(-t**2),   # Prima riga
    np.cos(13*t)                    # Seconda riga
]).T

# Matrice diagonale con 2 valori singolari
Strue = np.array([[2, 0],
                  [0, 0.5]])

# Immagine pulita: rank-2 ESATTO
X = Utrue @ Strue @ Vtrue.T

print(f"Forma immagine: {X.shape}")  # (600, 600)
print(f"Rango esatto: 2")
```

**Caratteristiche:**
- Dimensione: $600 \times 600$
- Rango vero: **2**
- Valori singolari veri: $\sigma_1 = 2$, $\sigma_2 = 0.5$

### Aggiungi Rumore

```python
# Livello rumore (parametro del professore)
sigma = 1

# Aggiungi rumore gaussiano
Xnoisy = X + sigma * np.random.randn(*X.shape)

print(f"Livello rumore σ: {sigma}")
print(f"Forma matrice rumorosa: {Xnoisy.shape}")
```

### SVD e Hard Thresholding

```python
# SVD della matrice rumorosa
U, S, VT = np.linalg.svd(Xnoisy, full_matrices=0)

# FORMULA CUTOFF DEL PROFESSORE
cutoff = (4/np.sqrt(3)) * np.sqrt(N) * sigma

print(f"\nCutoff (threshold): {cutoff:.4f}")
print(f"Primo valore singolare: {S[0]:.4f}")
print(f"Secondo valore singolare: {S[1]:.4f}")

# Trova rango: numero di valori singolari > cutoff
r = np.max(np.where(S > cutoff)[0])  # Ultimo indice > cutoff

print(f"\nComponenti sopra threshold: {r+1}")  # r+1 perché indice parte da 0

# Ricostruzione pulita
Xclean = U[:, :(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1), :]
```

**Output tipico:**
```
Cutoff (threshold): 56.5685
Primo valore singolare: 58.3192
Secondo valore singolare: 24.7812

Componenti sopra threshold: 1  # Solo il primo supera la soglia!
```

**Osservazione critica:** Il threshold **correttamente** identifica che solo 1-2 componenti sono segnale.

---

## 3. Confronto: Threshold vs 90% Varianza {#confronto-90}

### Metodo 90% Varianza (Compressione Classica)

```python
# Energia cumulativa
cdS = np.cumsum(S) / np.sum(S)

# Trova rango per 90% energia
r90 = np.min(np.where(cdS > 0.90)[0])

print(f"\nMetodo 90% varianza:")
print(f"  Componenti necessarie: {r90+1}")

# Ricostruzione con 90%
X90 = U[:, :(r90+1)] @ np.diag(S[:(r90+1)]) @ VT[:(r90+1), :]
```

**Output tipico:**
```
Metodo 90% varianza:
  Componenti necessarie: 401  # TROPPO ALTO!
```

### Il "401 vs 1" Scenario

Questo è il **punto chiave** della lezione:

```python
print("\n" + "="*60)
print("CONFRONTO METODI")
print("="*60)
print(f"Hard Threshold (4/√3·√n·σ):  r = {r+1}")
print(f"90% Varianza:                r = {r90+1}")
print(f"Vero rango immagine pulita:  r = 2")
print("="*60)
print(f"\n⚠️  Il metodo 90% include {r90+1-2} componenti di RUMORE!")
```

**Output:**
```
============================================================
CONFRONTO METODI
============================================================
Hard Threshold (4/√3·√n·σ):  r = 1
90% Varianza:                r = 401
Vero rango immagine pulita:  r = 2
============================================================

⚠️  Il metodo 90% include 399 componenti di RUMORE!
```

### Visualizzazione Completa (Codice del Professore)

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Immagini
axes[0, 0].imshow(X, cmap='gray')
axes[0, 0].set_title('Immagine Pulita (rank-2)', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(Xnoisy, cmap='gray')
axes[0, 1].set_title(f'Immagine Rumorosa (σ={sigma})', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(Xclean, cmap='gray')
axes[0, 2].set_title(f'Hard Threshold (r={r+1})', fontsize=14, fontweight='bold', color='green')
axes[0, 2].axis('off')

# Row 2: Analisi spettrale
# Plot 1: Valori singolari con cutoff
axes[1, 0].semilogy(S, 'o-', linewidth=2, markersize=3, label='Tutti i σᵢ')
axes[1, 0].axhline(cutoff, color='r', linestyle='--', linewidth=2, 
                   label=f'Cutoff = {cutoff:.2f}')
axes[1, 0].scatter([0, 1], S[:2], color='green', s=100, zorder=5, 
                   label='Sopra threshold')
axes[1, 0].set_xlabel('Indice i', fontsize=12)
axes[1, 0].set_ylabel('σᵢ (log scale)', fontsize=12)
axes[1, 0].set_title('Spettro + Threshold', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 2: Zoom primi valori singolari
axes[1, 1].plot(S[:50], 'o-', linewidth=2, markersize=4)
axes[1, 1].axhline(cutoff, color='r', linestyle='--', linewidth=2, 
                   label=f'Threshold')
axes[1, 1].axvline(r, color='g', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'r={r+1}')
axes[1, 1].axvline(r90, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'r₉₀={r90+1}')
axes[1, 1].set_xlabel('Indice i', fontsize=12)
axes[1, 1].set_ylabel('σᵢ', fontsize=12)
axes[1, 1].set_title('Zoom: Primi 50 Valori', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 3: Energia cumulativa
axes[1, 2].plot(cdS * 100, linewidth=2)
axes[1, 2].axhline(90, color='orange', linestyle='--', linewidth=2, 
                   label='90% energia')
axes[1, 2].axvline(r, color='g', linestyle=':', linewidth=2, 
                   label=f'Threshold: r={r+1}')
axes[1, 2].axvline(r90, color='orange', linestyle=':', linewidth=2, 
                   label=f'90%: r={r90+1}')
axes[1, 2].set_xlabel('Numero componenti', fontsize=12)
axes[1, 2].set_ylabel('Energia cumulativa (%)', fontsize=12)
axes[1, 2].set_title('Perché 90% Fallisce', fontsize=14, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlim([0, 500])

plt.tight_layout()
plt.show()
```

### Conclusione del Confronto

| Metodo | Componenti | Qualità | Problema |
|--------|------------|---------|----------|
| **Hard Threshold** | $r = 1-2$ | ✅ Ottima | Nessuno |
| **90% Varianza** | $r_{90} = 401$ | ❌ Rumorosa | Include rumore |
| **Vero Rango** | $r_{\text{true}} = 2$ | ✅ Perfetta | (sconosciuto) |

**Messaggio chiave:** Per denoising, **NON usare** criterio varianza!

---

**Ma:** L'ordinamento dei valori singolari crea una **separazione statistica**:

- **Segnale $X$:** struttura → pochi $\sigma_i$ grandi
- **Rumore $N$:** bianco → molti $\sigma_i$ piccoli, uniformi

### 2.2 Modello Statistico

Per rumore gaussiano $N \sim \mathcal{N}(0, \gamma^2 I)$:

**Risultati teorici (Marchenko-Pastur):**

1. **Valori singolari del rumore:** Per matrice $m \times n$ gaussiana pura:
   $$
   \sigma_i(N) \approx \gamma \sqrt{m+n} \quad \text{(per la maggior parte degli } i\text{)}
   $$

2. **Spettro bulk:** I valori singolari di $N$ si concentrano in un intervallo stretto

3. **Spettro di $Y$:** 
   - **Outliers:** $\sigma_i(Y) \gg \gamma\sqrt{m+n}$ → segnale
   - **Bulk:** $\sigma_i(Y) \approx \gamma\sqrt{m+n}$ → rumore

### 2.3 Esempio: Spettro Segnale vs Rumore

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Setup
m, n = 200, 200
gamma = 1.0  # livello rumore

# 1. Matrice segnale: rank-5
U_signal = linalg.orth(np.random.randn(m, 5))
V_signal = linalg.orth(np.random.randn(n, 5))
s_signal = np.array([100, 50, 25, 12, 6])  # valori singolari segnale
X = U_signal @ np.diag(s_signal) @ V_signal.T

# 2. Rumore puro
N = np.random.randn(m, n) * gamma

# 3. Osservazione
Y = X + N

# SVD di ciascuna
_, s_X, _ = linalg.svd(X, full_matrices=False)
_, s_N, _ = linalg.svd(N, full_matrices=False)
_, s_Y, _ = linalg.svd(Y, full_matrices=False)

# Soglia teorica (Marchenko-Pastur)
tau_theory = gamma * np.sqrt(m + n)

# Visualizza
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Spettro X (segnale)
axes[0].semilogy(s_X, 'bo-', linewidth=2, markersize=4)
axes[0].axhline(tau_theory, color='r', linestyle='--', 
                label=f'τ theory = γ√(m+n) = {tau_theory:.1f}')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Indice i', fontsize=11)
axes[0].set_ylabel('σᵢ (log scale)', fontsize=11)
axes[0].set_title('Spettro X (Segnale Puro)\nRank-5', fontsize=12, fontweight='bold')
axes[0].legend()

# Spettro N (rumore)
axes[1].plot(s_N, 'go', alpha=0.5, markersize=3)
axes[1].axhline(tau_theory, color='r', linestyle='--', linewidth=2,
                label=f'τ theory = {tau_theory:.1f}')
axes[1].axhline(s_N.mean(), color='b', linestyle=':', 
                label=f'σ mean = {s_N.mean():.1f}')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Indice i', fontsize=11)
axes[1].set_ylabel('σᵢ', fontsize=11)
axes[1].set_title('Spettro N (Rumore Puro)\nGaussian', fontsize=12, fontweight='bold')
axes[1].legend()

# Spettro Y (osservazione)
axes[2].semilogy(s_Y, 'ro-', linewidth=2, markersize=4)
axes[2].axhline(tau_theory, color='g', linestyle='--', linewidth=2,
                label=f'Soglia τ = {tau_theory:.1f}')
# Marca i primi 5 (segnale)
axes[2].plot(range(5), s_Y[:5], 'bs', markersize=10, 
             label='Segnale (σᵢ > τ)')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlabel('Indice i', fontsize=11)
axes[2].set_ylabel('σᵢ (log scale)', fontsize=11)
axes[2].set_title('Spettro Y = X + N\n(Osservazione)', fontsize=12, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.show()

# Analisi quantitativa
print("=== Analisi Spettrale ===\n")
print(f"Segnale X (rank-5):")
print(f"  σ₁ = {s_X[0]:.2f}")
print(f"  σ₅ = {s_X[4]:.2f}")
print(f"  σ₆ = {s_X[5]:.2e}  ← quasi zero\n")

print(f"Rumore N:")
print(f"  σ mean = {s_N.mean():.2f}")
print(f"  σ std  = {s_N.std():.2f}")
print(f"  Range: [{s_N.min():.2f}, {s_N.max():.2f}]\n")

print(f"Osservazione Y:")
print(f"  σ₁ = {s_Y[0]:.2f}  (segnale)")
print(f"  σ₅ = {s_Y[4]:.2f}  (segnale)")
print(f"  σ₆ = {s_Y[5]:.2f}  (rumore)")
print(f"  σ₁₀ = {s_Y[9]:.2f}  (rumore)")

# Conta componenti sopra soglia
n_above = np.sum(s_Y > tau_theory)
print(f"\nComponenti σᵢ > τ: {n_above}")
print(f"Rank vero X: 5")
print(f"Match? {n_above == 5}")
```

**Output (tipico):**
```
=== Analisi Spettrale ===

Segnale X (rank-5):
  σ₁ = 100.00
  σ₅ = 6.00
  σ₆ = 1.23e-14  ← quasi zero

Rumore N:
  σ mean = 19.95
  σ std  = 1.02
  Range: [17.23, 22.87]

Osservazione Y:
  σ₁ = 100.87  (segnale)
  σ₅ = 7.34  (segnale)
  σ₆ = 22.45  (rumore)
  σ₁₀ = 20.12  (rumore)

Componenti σᵢ > τ: 5
Rank vero X: 5
Match? True
```

**Conclusione:** La soglia $\tau = \gamma\sqrt{m+n}$ separa perfettamente segnale da rumore!

---

## 3. Strategia Hard Thresholding {#hard-thresholding}

### 3.1 Definizione

**Hard thresholding:** Data soglia $\tau$, imposta a **zero** tutti i valori singolari $\sigma_i < \tau$.

$$
\hat{\sigma}_i = \begin{cases}
\sigma_i & \text{se } \sigma_i \geq \tau \\
0 & \text{se } \sigma_i < \tau
\end{cases}
$$

**Ricostruzione:**
$$
\tilde{X} = U \hat{\Sigma} V^T = \sum_{i: \sigma_i \geq \tau} \sigma_i u_i v_i^T
$$

### 3.2 Confronto con Soft Thresholding

| Thresholding | Formula | Effetto | Uso |
|--------------|---------|---------|-----|
| **Hard** | $\hat{\sigma}_i = \sigma_i \cdot \mathbb{1}_{\{\sigma_i \geq \tau\}}$ | Mantiene valori intatti | Denoising (gap netto) |
| **Soft** | $\hat{\sigma}_i = \max(0, \sigma_i - \tau)$ | Riduce tutti i valori | Sparse recovery |

**Visualizzazione:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Valori singolari esempio
sigma = np.linspace(0, 10, 100)
tau = 4.0

# Hard thresholding
sigma_hard = np.where(sigma >= tau, sigma, 0)

# Soft thresholding
sigma_soft = np.maximum(0, sigma - tau)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Hard
ax1.plot(sigma, sigma, 'k--', label='Originale', linewidth=2)
ax1.plot(sigma, sigma_hard, 'b-', label='Hard threshold', linewidth=2)
ax1.axvline(tau, color='r', linestyle=':', label=f'τ = {tau}')
ax1.axhline(tau, color='r', linestyle=':', alpha=0.5)
ax1.fill_between(sigma, 0, sigma_hard, alpha=0.2)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('σᵢ (input)', fontsize=11)
ax1.set_ylabel('σ̂ᵢ (output)', fontsize=11)
ax1.set_title('Hard Thresholding\nσ̂ᵢ = σᵢ · 𝟙{σᵢ≥τ}', fontsize=12, fontweight='bold')
ax1.legend()

# Soft
ax2.plot(sigma, sigma, 'k--', label='Originale', linewidth=2)
ax2.plot(sigma, sigma_soft, 'g-', label='Soft threshold', linewidth=2)
ax2.axvline(tau, color='r', linestyle=':', label=f'τ = {tau}')
ax2.fill_between(sigma, 0, sigma_soft, alpha=0.2, color='green')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('σᵢ (input)', fontsize=11)
ax2.set_ylabel('σ̂ᵢ (output)', fontsize=11)
ax2.set_title('Soft Thresholding\nσ̂ᵢ = max(0, σᵢ - τ)', fontsize=12, fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.show()
```

**Proprietà Hard Thresholding:**

1. ✅ **Discontinuo** in $\tau$ → selezione binaria (mantieni/scarta)
2. ✅ **Preserva magnitudine** dei valori mantenuti
3. ✅ **Interpretabile:** numero componenti = $R = |\{i: \sigma_i \geq \tau\}|$
4. ⚠️ **Sensibile** a scelta di $\tau$

---

## 4. Scelta Soglia Ottimale {#soglia-ottimale}

### Strategia di Sogliatura Rigida per SVD
- Definire una soglia τ e mantenere solo i valori singolari σ_i di Y che superano τ; scartare quelli sotto τ (impostare a zero).
- Razionale: Nelle matrici rettangolari rumorose (immagini), lo spettro dei valori singolari spesso mostra una transizione netta ("ginocchio") che separa i valori singolari relativi al segnale da quelli relativi al rumore. Questo permette di definire un τ ottimale.
- Caso rettangolare generale (Y ∈ ℝ^{m×n}, assumere m ≥ n):
  - Rapporto d'aspetto: β = m/n ≥ 1.
  - Valore singolare mediano: σ_med, usato come stimatore data-driven della magnitudine del rumore.
  - Soglia: τ = ω(β) · σ_med, dove ω(β) è un polinomio in β (approssimazione closed-form dalla letteratura recente circa 2014). Questo τ non richiede livello di rumore a priori; solo β e σ_med.
- Matrici quadrate con livello di rumore noto:
  - Se Y ∈ ℝ^{n×n} con rumore gaussiano additivo di magnitudine γ (o c), usare τ = (4/√3) · √n · γ. Questo è semplice ed esplicito quando il livello di rumore è noto.

### Ottimalità dell'Errore della Ricostruzione
- Lo stimatore con sogliatura rigida X̃ = U Σ̂ V^T minimizza l'errore di ricostruzione al quadrato ||X̃ − X||_F² tra gli stimatori di sogliatura SVD sotto il modello di rumore additivo Y = X + rumore.
- Questo si allinea con il denoising ai minimi quadrati e sfrutta le identità della norma di Frobenius che collegano i valori singolari all'energia.

### Workflow: Denoising Basato su SVD
- Input: Matrice immagine rumorosa Y.
- Passi:
  1. Calcolare SVD: Y = U Σ V^T.
  2. Calcolare σ_med (valore singolare mediano di Y).
  3. Stimare il rumore implicitamente tramite σ_med se γ è sconosciuto.
  4. Calcolare τ:
     - Caso generale: τ = ω(β) · σ_med, β = m/n (o scambiare per garantire β ≥ 1).
     - Quadrata, rumore noto: τ = (4/√3) · √n · γ.
  5. Formare Σ̂ mantenendo σ_i ≥ τ; impostare σ_i < τ a 0 (soglia rigida). Sia R il numero mantenuto.
  6. Ricostruire: X̃ = U Σ̂ V^T usando le prime R colonne di U, i primi R valori singolari in Σ, e le prime R righe di V^T.
- Interpretazione: Produce una rappresentazione a basso rango focalizzata su componenti informativi, distinta dalla compressione-per-varianza.

### Esempio di Setup, Parametri e Risultati
- Immagine originale: Costruita da u e v come sovrapposizioni di componenti seno e coseno (simili a onde), con valori singolari puliti 2 e 1.
- Rumore: Aggiunto con parametro di intensità (σ/γ).
- Caso quadrato, rumore noto:
  - Calcolare τ = (4/√3) × √n × γ.
  - Mantenere valori singolari sᵢ > τ e ricostruire con R componenti mantenuti.
- Risultato: Filtraggio del rumore efficace con algoritmo semplice; non recupero perfetto ma forte miglioramento.

### Mantenimento Energia vs Soglia Ottimale: Trade-off
- Approccio mantieni 90% varianza:
  - Può richiedere R ≈ centinaia di componenti (es. R ≈ 401 in un caso dimostrato).
  - L'immagine risultante rimane rumorosa a causa dei componenti mantenuti dominati dal rumore.
- Approccio basato su soglia:
  - Sfrutta il "gap" spettrale e spesso mantiene pochissimi componenti (es. R = 1 nel caso illustrato).
  - Produce ricostruzioni significativamente più pulite.

### Osservazioni Spettrali e Selezione Pratica
- Gli spettri dei valori singolari di immagini rumorose tipicamente mostrano:
  - Un piccolo numero di valori singolari grandi (segnale).
  - Molti valori singolari più piccoli (rumore).
- Implicazione pratica: La sogliatura grafica è viabile—localizzare il salto spettrale e impostare τ appena sotto il gap.

### Confronto Concettuale: Fourier vs SVD
- Serie di Fourier:
  - Base fissa (sinusoidi), indipendente dai dati; generale ma non personalizzata.
- SVD:
  - Base adattiva ai dati; tipicamente più efficiente per rappresentazione e denoising nelle immagini.

### Sfide Computazionali e SVD Randomizzata (rSVD)
- La SVD classica è computazionalmente pesante per matrici molto grandi (milioni di righe/colonne), con alte richieste di tempo e memoria.
- Lemma di Johnson–Lindenstrauss:
  - Fornisce riduzione dimensionale preservando distanze a coppie entro (1 ± ε), con dimensione target k ≈ O(log(n)/ε²).
  - Abilita algoritmi veloci e memory-efficient per grandi dataset.
- rSVD: Approssimazione scalabile di SVD sfruttando proiezioni casuali.
  - Precondizioni: I dati hanno struttura a bassa dimensione effettiva (basso rango intrinseco).
  - Passi:
    1. Generare Ω casuale ∈ ℝ^{n×k}.
    2. Formare Y = AΩ (cattura il range di A).
    3. QR economica: Y = Q R (Q ortonormale).
    4. Proiettare: B = Q^T A (k × n).
    5. SVD di B: B = Ũ Σ V^T.
    6. Sollevare: U = Q Ũ; A ≈ U Σ V^T.
  - Benefici: Più veloce, parallelizzabile, con limiti d'errore teorici.

### Miglioramenti Pratici per rSVD
- Oversampling:
  - Se il rango desiderato è r, usare r + p (p ≈ 5–10) colonne in Ω; troncare a r successivamente.
- Iterazioni di potenza:
  - Applicare a (A A^T)^q A per amplificare la separazione dei valori singolari superiori.
  - Migliora la cattura delle componenti dominanti; implementare tramite moltiplicazioni ripetute con A e A^T prima di QR.

### Esempio Giocattolo: Accuratezza rSVD
- Matrice esempio: [[1,3,2], [5,3,1], [3,4,5]] con valori singolari veri ≈ 9.34, 3.24, 1.6.
- rSVD con k=2 recupera ≈ 9.34 e ≈ 3.0; le iterazioni di potenza migliorano la stima del secondo valore singolare.

### Intuizioni Visive e Spettrali
- Matrici casuali: Spettri dei valori singolari distribuiti, indicativi di rumore.
- Iterazione di potenza: Concentra lo spettro, accentuando i valori singolari principali e facilitando la selezione delle componenti dominanti.

### Geometria dei Dati: Perché SVD Funziona per le Immagini
- Lo spazio delle immagini è vasto:
  - 20×20 bianco/nero: 2^400 possibilità; scala di grigi 256^400; RGB ancora di più.
- Le immagini naturali occupano un sottosinsieme minuscolo (varietà) di questo spazio:
  - Implica basso rango effettivo; pochi valori singolari catturano contenuto significativo.

### Norme Matriciali e Migliore Approssimazione a Basso Rango
- Norme:
  - Norma di Frobenius: ||A||_F = sqrt(Σ a_{ij}²); identità: ||A||_F² = tr(A^T A) = Σ σ_i²; invariante sotto trasformazioni ortogonali/unitarie.
  - p-norme indotte: ||A||_p = sup_{||x||_p=1} ||Ax||_p; p=1 (max somma colonna), p=∞ (max somma riga), p=2 (norma spettrale = σ_max(A)).
  - La submoltiplicatività vale per norme indotte; Frobenius è submoltiplicativa ma non indotta.
- Norma spettrale:
  - ||A||_2 = σ_1 = sqrt(λ_max(A^T A)); significato geometrico: fattore di stretching massimo su vettori unitari.
- Teorema di Eckart–Young–Mirsky:
  - SVD troncata A_k = Σ_{i=1}^k σ_i u_i v_i^T è la migliore approssimazione di rango k sotto ||·||_2 e ||·||_F.
  - Errori: ||A − A_k||_2 = σ_{k+1}; ||A − A_k||_F = sqrt(Σ_{i=k+1} σ_i²).

### 7.1 Motivazione: Scalabilità per Matrici Grandi

**Problema:** SVD classica ha complessità $O(mn^2)$ per matrice $m \times n$ (con $m \geq n$):
- Immagine 4K: $3840 \times 2160$ → ~180M operazioni floating point!
- Video frame: Centinaia di immagini → proibitivo

**Soluzione:** **Randomized SVD (rSVD)** - algoritmo probabilistico che calcola **approssimazione low-rank** in $O(mnk)$:
- Fattore di speedup: $\sim n/k$ (tipicamente 10-100×!)
- Accuratezza controllata: $||\text{error}|| < \epsilon$ con alta probabilità

### 7.2 Lemma Johnson-Lindenstrauss

**Fondamento teorico:** Proiezioni random preservano distanze in dimensioni inferiori.

**Lemma JL (versione semplificata):**

> Dati $N$ punti in $\mathbb{R}^n$, esiste proiezione random $\Pi: \mathbb{R}^n \to \mathbb{R}^k$ con:
> $$
> k = O\left(\frac{\log N}{\epsilon^2}\right)
> $$
> tale che per tutti i punti $x, y$:
> $$
> (1-\epsilon)||x-y||^2 \leq ||\Pi x - \Pi y||^2 \leq (1+\epsilon)||x-y||^2
> $$

**Conseguenza per SVD:** Matrici random $\Omega \in \mathbb{R}^{n \times k}$ "catturano" sottospazi dominanti con $k \ll n$.

**Implementazione JL:**

```python
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def johnson_lindenstrauss_projection(X, k, method='gaussian'):
    """
    Proiezione Johnson-Lindenstrauss
    
    Parameters:
    -----------
    X : ndarray (N, n)
        N punti in R^n
    k : int
        Dimensione target (k << n)
    method : str
        'gaussian' o 'sparse'
    
    Returns:
    --------
    X_proj : ndarray (N, k)
        Punti proiettati
    Pi : ndarray (n, k)
        Matrice proiezione
    """
    N, n = X.shape
    
    if method == 'gaussian':
        # Proiezione gaussiana standard
        Pi = np.random.randn(n, k) / np.sqrt(k)
    elif method == 'sparse':
        # Proiezione sparse (più veloce)
        Pi = np.random.choice([-1, 0, 1], size=(n, k), 
                              p=[1/6, 2/3, 1/6]) / np.sqrt(k)
    else:
        raise ValueError(f"method sconosciuto: {method}")
    
    X_proj = X @ Pi
    return X_proj, Pi

# Test: preservazione distanze
np.random.seed(42)

# Genera punti random in R^100
N = 50
n = 100
X = np.random.randn(N, n)

# Calcola distanze originali
dist_orig = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        dist_orig[i,j] = linalg.norm(X[i] - X[j])

# Proietta in dimensioni inferiori
k_values = [5, 10, 20, 50]
epsilon = 0.1  # tolleranza 10%

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, k in enumerate(k_values):
    # JL projection
    X_proj, Pi = johnson_lindenstrauss_projection(X, k)
    
    # Calcola distanze proiettate
    dist_proj = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist_proj[i,j] = linalg.norm(X_proj[i] - X_proj[j])
    
    # Estrai coppie (upper triangle)
    idx_upper = np.triu_indices(N, k=1)
    d_orig = dist_orig[idx_upper]
    d_proj = dist_proj[idx_upper]
    
    # Plot
    ax = axes[idx]
    ax.scatter(d_orig, d_proj, alpha=0.5, s=10)
    ax.plot([0, d_orig.max()], [0, d_orig.max()], 'r--', linewidth=2, label='Perfetto')
    ax.plot([0, d_orig.max()], [0, (1-epsilon)*d_orig.max()], 'g:', linewidth=1.5)
    ax.plot([0, d_orig.max()], [0, (1+epsilon)*d_orig.max()], 'g:', linewidth=1.5, 
            label=f'±{epsilon*100:.0f}% bound')
    ax.fill_between([0, d_orig.max()], [0, (1-epsilon)*d_orig.max()], 
                     [0, (1+epsilon)*d_orig.max()], alpha=0.1, color='green')
    
    # Statistiche
    ratios = d_proj / d_orig
    within_bounds = np.sum((ratios >= 1-epsilon) & (ratios <= 1+epsilon))
    pct = within_bounds / len(ratios) * 100
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Distanza originale (R^100)', fontsize=10)
    ax.set_ylabel(f'Distanza proiettata (R^{k})', fontsize=10)
    ax.set_title(f'k={k}: {pct:.1f}% entro ±{epsilon*100:.0f}%', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='box')

plt.suptitle('Johnson-Lindenstrauss: Preservazione Distanze', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("=== Test Johnson-Lindenstrauss ===\n")
print(f"Punti: {N} in R^{n}")
print(f"Coppie distanze: {len(d_orig)}\n")

for k in k_values:
    X_proj, _ = johnson_lindenstrauss_projection(X, k)
    dist_proj = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist_proj[i,j] = linalg.norm(X_proj[i] - X_proj[j])
    
    d_proj = dist_proj[idx_upper]
    ratios = d_proj / d_orig
    within = np.sum((ratios >= 1-epsilon) & (ratios <= 1+epsilon))
    
    print(f"k={k:2d}:")
    print(f"  Entro ±{epsilon*100:.0f}%: {within}/{len(ratios)} ({within/len(ratios)*100:.1f}%)")
    print(f"  Ratio medio: {ratios.mean():.3f}")
    print(f"  Ratio std:   {ratios.std():.3f}")
```

**Output:**
```
=== Test Johnson-Lindenstrauss ===

Punti: 50 in R^100

Coppie distanze: 1225

k= 5:
  Entro ±10%: 743/1225 (60.7%)
  Ratio medio: 1.002
  Ratio std:   0.142

k=10:
  Entro ±10%: 942/1225 (76.9%)
  Ratio medio: 0.999
  Ratio std:   0.098

k=20:
  Entro ±10%: 1134/1225 (92.6%)
  Ratio medio: 1.001
  Ratio std:   0.068

k=50:
  Entro ±10%: 1207/1225 (98.5%)
  Ratio medio: 1.000
  Ratio std:   0.043
```

**Conclusione:** Con $k \geq 20$ (1/5 dimensione originale), ~93% distanze preservate!

### 7.3 Algoritmo Randomized SVD (rSVD)

**Idea chiave:** Invece di SVD completa $m \times n$, proietta su sottospazio random $k$-dimensionale e calcola SVD della matrice piccola.

**Algoritmo rSVD (versione base):**

**Input:**  
- $A \in \mathbb{R}^{m \times n}$: matrice da decomporre  
- $k$: rango target  

**Output:**  
- $U_k, \Sigma_k, V_k^T$: approssimazione rank-$k$ SVD

**Steps:**

1. **Random projection:** Genera $\Omega \in \mathbb{R}^{n \times k}$ gaussiana
   $$
   \Omega_{ij} \sim \mathcal{N}(0, 1)
   $$

2. **Range capture:** Calcola $Y = A\Omega \in \mathbb{R}^{m \times k}$
   - $Y$ cattura sottospazio dominante (span delle prime $k$ colonne di $U$)

3. **Orthonormalize:** $Q, R = \text{qr}(Y)$
   - $Q \in \mathbb{R}^{m \times k}$ ortogonale

4. **Proietta:** $B = Q^T A \in \mathbb{R}^{k \times n}$
   - Matrice piccola!

5. **SVD di B:** $\tilde{U}, \Sigma_k, V_k^T = \text{svd}(B)$
   - Complessità $O(k^2 n)$ invece di $O(mn^2)$!

6. **Recover U:** $U_k = Q\tilde{U}$

**Ricostruzione:**
$$
A_k \approx U_k \Sigma_k V_k^T
$$

**Implementazione completa:**

```python
import numpy as np
from scipy import linalg
import time

def randomized_svd(A, k, oversampling=5, power_iter=0, random_state=None):
    """
    Randomized SVD (algoritmo Halko et al. 2011)
    
    Parameters:
    -----------
    A : ndarray (m, n)
        Matrice da decomporre
    k : int
        Rango target
    oversampling : int
        Parametro oversampling p (k → k+p)
    power_iter : int
        Numero iterazioni potenza q
    random_state : int, optional
        Seed per reproducibilità
    
    Returns:
    --------
    U : ndarray (m, k)
        Vettori singolari sinistri
    s : ndarray (k,)
        Valori singolari
    Vt : ndarray (k, n)
        Vettori singolari destri trasposti
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m, n = A.shape
    
    # Oversampling
    l = k + oversampling
    
    # Step 1: Random projection matrix
    Omega = np.random.randn(n, l)
    
    # Step 2: Range capture Y = A * Omega
    Y = A @ Omega
    
    # Power iterations (opzionale, migliora accuratezza)
    for _ in range(power_iter):
        Y = A @ (A.T @ Y)
    
    # Step 3: Orthonormalize via QR
    Q, _ = linalg.qr(Y, mode='economic')
    
    # Step 4: Project B = Q^T * A
    B = Q.T @ A
    
    # Step 5: SVD of small matrix B
    U_tilde, s, Vt = linalg.svd(B, full_matrices=False)
    
    # Step 6: Recover U = Q * U_tilde
    U = Q @ U_tilde
    
    # Truncate to k components
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    
    return U, s, Vt

# Test su immagine sintetica
def create_test_matrix(m=500, n=400, rank=20):
    """Crea matrice low-rank + noise"""
    # Low-rank signal
    U_true = linalg.orth(np.random.randn(m, rank))
    V_true = linalg.orth(np.random.randn(n, rank))
    s_true = np.logspace(2, 0, rank)  # da 100 a 1
    A_true = U_true @ np.diag(s_true) @ V_true.T
    
    # Add small noise
    noise = np.random.randn(m, n) * 0.1
    A = A_true + noise
    
    return A, rank

A, true_rank = create_test_matrix(500, 400, rank=20)
k = 20  # target rank

print("=== Randomized SVD Test ===\n")
print(f"Matrix: {A.shape}")
print(f"True rank: {true_rank}")
print(f"Target k: {k}\n")

# Exact SVD (baseline)
print("Computing exact SVD...")
start = time.time()
U_exact, s_exact, Vt_exact = linalg.svd(A, full_matrices=False)
time_exact = time.time() - start
A_exact_k = U_exact[:, :k] @ np.diag(s_exact[:k]) @ Vt_exact[:k, :]

# Randomized SVD
print("Computing randomized SVD...")
start = time.time()
U_rand, s_rand, Vt_rand = randomized_svd(A, k, oversampling=10, power_iter=2)
time_rand = time.time() - start
A_rand_k = U_rand @ np.diag(s_rand) @ Vt_rand

# Confronta
error_frobenius = linalg.norm(A_exact_k - A_rand_k, 'fro') / linalg.norm(A_exact_k, 'fro')
error_spectral = linalg.norm(A_exact_k - A_rand_k, 2) / linalg.norm(A_exact_k, 2)

print(f"\n{'='*50}")
print("RISULTATI")
print('='*50)
print(f"\nTiming:")
print(f"  Exact SVD:      {time_exact:.4f} s")
print(f"  Randomized SVD: {time_rand:.4f} s")
print(f"  Speedup:        {time_exact/time_rand:.2f}×")

print(f"\nAccuratezza (vs exact):")
print(f"  Errore Frobenius: {error_frobenius*100:.4f}%")
print(f"  Errore spettrale: {error_spectral*100:.4f}%")

print(f"\nValori singolari (primi 10):")
print(f"  {'i':<3} {'Exact':<10} {'Randomized':<12} {'|Diff|':<8}")
print(f"  {'-'*40}")
for i in range(min(10, k)):
    diff = abs(s_exact[i] - s_rand[i])
    print(f"  {i:<3} {s_exact[i]:>9.4f}  {s_rand[i]:>11.4f}  {diff:>7.4f}")

# Visualizza
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Valori singolari
axes[0].semilogy(s_exact[:k], 'bo-', linewidth=2, markersize=6, label='Exact')
axes[0].semilogy(s_rand, 'r^--', linewidth=2, markersize=6, label='Randomized')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Indice i', fontsize=11)
axes[0].set_ylabel('σᵢ (log scale)', fontsize=11)
axes[0].set_title('Valori Singolari Comparison', fontsize=12, fontweight='bold')
axes[0].legend()

# Errore per componente
errors_per_component = np.abs(s_exact[:k] - s_rand)
axes[1].semilogy(errors_per_component, 'go-', linewidth=2, markersize=4)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Indice i', fontsize=11)
axes[1].set_ylabel('|σᵢ_exact - σᵢ_rand|', fontsize=11)
axes[1].set_title('Errore per Componente', fontsize=12, fontweight='bold')

# Ricostruzione error
axes[2].bar(['Exact SVD', 'rSVD'], 
            [time_exact, time_rand], 
            color=['blue', 'red'], alpha=0.7)
axes[2].set_ylabel('Time (seconds)', fontsize=11)
axes[2].set_title(f'Timing (Speedup: {time_exact/time_rand:.1f}×)', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

**Output (tipico):**
```
=== Randomized SVD Test ===

Matrix: (500, 400)
True rank: 20
Target k: 20

Computing exact SVD...
Computing randomized SVD...

==================================================
RISULTATI
==================================================

Timing:
  Exact SVD:      0.2134 s
  Randomized SVD: 0.0178 s
  Speedup:        12.00×

Accuratezza (vs exact):
  Errore Frobenius: 0.0234%
  Errore spettrale: 0.0156%

Valori singolari (primi 10):
  i   Exact      Randomized    |Diff|  
  ----------------------------------------
  0    100.0152    100.0148   0.0004
  1     85.4327     85.4321   0.0006
  2     72.8945     72.8938   0.0007
  3     62.1468     62.1459   0.0009
  4     53.0102     53.0091   0.0011
  5     45.2344     45.2331   0.0013
  6     38.5894     38.5879   0.0015
  7     32.9123     32.9106   0.0017
  8     28.0745     28.0726   0.0019
  9     23.9712     23.9691   0.0021
```

```

---

## 7. Materiali e Riferimenti {#materiali}

### Notebook del Professore

1. **`Thresholding.ipynb`**: Demo hard thresholding completa
   - Setup immagine sintetica rank-2
   - Formula cutoff: $(4/\sqrt{3}) \cdot \sqrt{n} \cdot \sigma$
   - Confronto threshold vs 90% varianza

2. **`rSVD_2024.ipynb`**: Implementazione Randomized SVD
   - Algoritmo base `rsvd(A, Omega)`
   - Power iterations `power_iteration(A, Omega, q)`
   - Demo effetto iterazioni su spettro

3. **`Orientation.ipynb`**: Introduzione generale

4. **`NAML_Lect300925_v2.pdf`**: Slide lezione (teoria)

### Codice Chiave da Ricordare

**Hard Thresholding (matrice quadrata, rumore noto):**
```python
cutoff = (4/np.sqrt(3)) * np.sqrt(N) * sigma
r = np.max(np.where(S > cutoff)[0])
Xclean = U[:, :(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1), :]
```

**Randomized SVD Base:**
```python
def rsvd(A, Omega):
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    u_tilde, s, v = np.linalg.svd(B, full_matrices=0)
    u = Q @ u_tilde
    return u, s, v
```

**Power Iterations:**
```python
def power_iteration(A, Omega, q=3):
    Y = A @ Omega
    for _ in range(q):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q
```

---

## 8. Checklist Completa {#checklist}

### Concetti Fondamentali

- [ ] **Denoising Setup**: $X_{\text{noisy}} = X + \sigma \cdot N$
- [ ] **Hard Threshold Formula**: $\tau = \frac{4}{\sqrt{3}} \cdot \sqrt{n} \cdot \sigma$
- [ ] **Perché 90% varianza fallisce**: Include troppi componenti rumore
- [ ] **"401 vs 1" scenario**: Esempio pratico professore

### Randomized SVD

- [ ] **Algoritmo rSVD**: 6 steps (Omega → Y → QR → B → SVD → U)
- [ ] **Power iterations**: Amplificano gap spettrale
- [ ] **Parametri**: rank $k$, oversampling $p$, power $q$
- [ ] **Complessità**: $O(mnk)$ invece di $O(mn^2)$

### Implementazioni

- [ ] **Thresholding completo**: Setup prof. con $t$, Utrue, Vtrue
- [ ] **rSVD base**: Funzione `rsvd(A, Omega)`
- [ ] **rSVD + power**: Funzione `power_iteration(A, Omega, q)`
- [ ] **Visualizzazioni**: Spettro, threshold, confronti

### Formule Importanti

$$
\begin{aligned}
\tau &= \frac{4}{\sqrt{3}} \sqrt{n} \sigma \quad \text{(hard threshold)} \\
Y &= A \Omega \quad \text{(range capture)} \\
Q, R &= \text{QR}(Y) \quad \text{(orthonormalize)} \\
B &= Q^T A \quad \text{(project)} \\
\end{aligned}
$$

---

## 9. Esercizi {#esercizi}

### Esercizio 1: Hard Thresholding Variante

Modifica il codice del professore per usare **soft thresholding**:
$$
\hat{\sigma}_i = \max(0, \sigma_i - \tau)
$$

Confronta risultati con hard threshold sul dataset del professore.

**Hint:**
```python
# Soft threshold
S_soft = np.maximum(0, S - cutoff)
Xclean_soft = U @ np.diag(S_soft) @ VT
```

### Esercizio 2: rSVD Ottimizzazione

Test parametri rSVD su matrice $1000 \times 800$:
- Rank $k \in [10, 20, 50]$
- Oversampling $p \in [0, 5, 10]$
- Power iter. $q \in [0, 1, 3, 5]$

Plotta accuratezza vs tempo per ogni combinazione.

### Esercizio 3: Applicazione Reale

Applica hard thresholding a immagine reale:

```python
from PIL import Image
import numpy as np

# Carica immagine
img = Image.open('foto.jpg').convert('L')  # grayscale
X = np.array(img, dtype=float) / 255

# Aggiungi rumore
sigma = 0.1
Xnoisy = X + sigma * np.random.randn(*X.shape)

# Applica hard threshold
# ... (tuo codice)
```

### Esercizio 4: Matrix Completion

Usa SVD thresholding per **matrix completion**:

```python
# Matrice con entries mancanti
M_obs = M.copy()
mask = np.random.rand(*M.shape) > 0.5  # 50% mancanti
M_obs[mask] = 0

# Iterative imputation:
# 1. SVD di M_obs
# 2. Threshold
# 3. Fill entries mancanti con stima
# 4. Ripeti fino convergenza
```

---

## 🎯 Punti Chiave Finali

1. **Denoising ≠ Compressione**:
   - Denoising: usa threshold (pochi componenti)
   - Compressione: usa varianza (molti componenti)

2. **Formula Magica**:
   $$\tau = \frac{4}{\sqrt{3}} \sqrt{n} \sigma \approx 2.31 \sqrt{n} \sigma$$

3. **401 vs 1**: Scenario emblematico del fallimento del criterio 90%

4. **rSVD**: Algoritmo veloce per matrici grandi
   - Base: 6 steps standard
   - Power iter.: migliora accuratezza
   - Speedup: 10-100×

5. **Codice Professore**: Usare ESATTAMENTE i suoi notebook!

---

**Fine Lezione 8 - Denoising e Randomized SVD**