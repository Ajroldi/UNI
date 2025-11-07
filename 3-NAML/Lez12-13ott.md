# Lab 4 NAML - Metodo Kernel e PageRank

## 🎯 Obiettivi della Lezione

### Competenze Teoriche
- Comprendere il problema dei dataset non lineari e la necessità di proiezioni in spazi superiori
- Padroneggiare il Representer Theorem e la sua applicazione alla kernel ridge regression
- Studiare il teorema di Perron-Frobenius per matrici stocastiche
- Analizzare la convergenza del metodo delle potenze per il calcolo di autovettori
- Introduzione al problema di matrix completion e strutture a bassa dimensionalità

### Competenze Pratiche
- Applicare feature maps per arricchire lo spazio delle caratteristiche
- Utilizzare funzioni kernel (polinomiale, Gaussiano) senza calcolo esplicito dei vettori
- Implementare kernel ridge regression con kernel trick
- Calcolare PageRank tramite autovettori di matrici di transizione
- Riconoscere problemi di matrix completion (Netflix, image inpainting)

### Applicazioni
- Regressione non lineare su dataset con trend parabolici/complessi
- Classificazione con separatori non lineari (cerchi, curve)
- Ranking di pagine web con algoritmo PageRank
- Sistemi di raccomandazione (problema Netflix)
- Image inpainting per completare porzioni mancanti

## 📚 Prerequisiti

### Matematica
- Algebra lineare: prodotto scalare, proiezioni ortogonali, spazi generati (span)
- Decomposizione SVD e agli autovalori (eigendecomposition)
- Matrici stocastiche: proprietà di colonne che sommano a 1
- Teoria delle probabilità: vettori di probabilità, matrici di transizione
- Metodo delle potenze per calcolo autovettori dominanti

### Teoria Machine Learning
- Minimi quadrati classici e ridge regression (regolarizzazione L2)
- Problema di overfitting e regolarizzazione
- Concetto di feature engineering e proiezione in spazi superiori
- Matrici di Gram e prodotti scalari in spazi arricchiti

### Programmazione
- Reti neurali: funzioni di attivazione lineari vs non lineari
- Manipolazione di matrici e calcolo autovalori/autovettori
- Visualizzazione di separatori non lineari

## 📑 Indice Completo

### Parte 1 - Metodo Kernel: Fondamenti Teorici
**Durata: ~45 min (00:00:09 - 00:45:01)**

#### 1. Metodo Kernel - Introduzione {#metodo-kernel-introduzione}
   - 1.1. Problema dei dataset non lineari `00:00:45`
   - 1.2. Inadeguatezza dei minimi quadrati classici `00:01:20`
   - 1.3. Esempio visivo: fit lineare su dati parabolici `00:01:54`
   - 1.4. Soluzione: proiezione in spazi di dimensione superiore `00:03:07`
   - 1.5. Apparente contraddizione con riduzione dimensionalità (PCA) `00:03:41`

#### 2. Feature Map e Proiezione {#feature-map-e-proiezione}
   - 2.1. Definizione formale di feature map φ: Rᵖ → Rᴰ `00:04:57`
   - 2.2. Esempio 1D: φ(x) = [x, x²] (da R¹ a R²) `00:06:11`
   - 2.3. Modello lineare nelle nuove features: w₁x + w₂x² `00:06:49`
   - 2.4. Esempio 2D: φ(x) = [1, x₁, x₂, x₁², x₂², √2·x₁x₂] (da R² a R⁶) `00:07:26`
   - 2.5. Vista geometrica: separazione circolare in 2D → iperpiano in 3D `00:08:35`
   - 2.6. Demo con rete neurale: attivazione lineare + feature engineering `00:10:41`
   - 2.7. Risultati: separatore perfetto aggiungendo x₁², x₂², x₁x₂ `00:12:15`

#### 3. Kernel Regression {#kernel-regression}
   - 3.1. Formalizzazione: applicare feature map φ al dataset X `00:13:59`
   - 3.2. Matrice Φ (N×D): righe φ(xᵢ) per i=1,...,N `00:14:35`
   - 3.3. Ridge regression sullo spazio arricchito `00:15:11`
   - 3.4. Problema: D può essere molto grande (calcolo e memoria) `00:16:55`
   - 3.5. Necessità del kernel trick `00:17:32`

### Parte 2 - Kernel Trick e Representer Theorem
**Durata: ~35 min (00:18:08 - 00:53:29)**

#### 4. Kernel Trick: Teoria {#kernel-trick-teoria}
   - 4.1. Definizione funzione kernel: K(xᵢ,xⱼ) = φ(xᵢ)ᵀ·φ(xⱼ) `00:18:08`
   - 4.2. Vantaggio: calcolo prodotto scalare senza φ esplicito `00:18:45`
   - 4.3. Matrice di Gram K = ΦΦᵀ (termini Kᵢⱼ = K(xᵢ,xⱼ)) `00:19:54`
   - 4.4. Kernel trick: riscrivere problema usando solo K `00:20:27`

#### 5. Esempi di Kernel {#esempi-kernel}
   - 5.1. Kernel polinomiale: K(x,z) = (xᵀz + c)ᵍ `00:21:02`
   - 5.2. Polinomio omogeneo (c=0) vs termini ordine inferiore (c>0) `00:21:35`
   - 5.3. Kernel Gaussiano (RBF): K(x,z) = exp(-γ||x-z||²) `00:22:07`
   - 5.4. Kernel Gaussiano come uno dei più usati in pratica `00:22:40`

#### 6. Representer Theorem {#representer-theorem}
   - 6.1. Enunciato: w* combinazione lineare φ(xᵢ) di training `00:23:17`
   - 6.2. Forma w* = Φᵀα con α vettore N-dimensionale `00:24:39`
   - 6.3. w* appartiene allo span dei vettori di training `00:25:21`
   - 6.4. Riduzione ricerca: da D dimensioni a N coefficienti `00:25:52`
   - 6.5. Dimostrazione: decomposizione w = w∥ + w⊥ `00:26:24`
   - 6.6. w⊥ ortogonale a φ(xᵢ) → contributo nullo al modello `00:27:40`
   - 6.7. Minimizzazione: azzerare w⊥ per ridurre regolarizzazione `00:29:00`
   - 6.8. Conclusione: w* = Φᵀα è ottimale `00:30:17`

#### 7. Soluzione Kernel Ridge Regression {#soluzione-kernel-ridge}
   - 7.1. Sostituzione w* = Φᵀα nel problema di minimizzazione `00:31:35`
   - 7.2. Sistema lineare: (K + λI)α = y `00:32:07`
   - 7.3. Matrice K = ΦΦᵀ simmetrica e definita positiva `00:32:45`
   - 7.4. Decomposizione spettrale: K = UΛUᵀ `00:34:02`
   - 7.5. Soluzione efficiente: α = U(Λ + λI)⁻¹Uᵀy `00:34:33`
   - 7.6. Inferenza: ŷ(x*) = Σᵢ αᵢK(xᵢ,x*) `00:36:24`
   - 7.7. Valutazione modello senza calcolare φ(x*) esplicitamente `00:38:01`

#### 8. Esempio Kernel Polinomiale {#esempio-kernel-polinomiale}
   - 8.1. Dataset: x₁=-1, x₂=0, x₃=1 `00:39:02`
   - 8.2. Kernel K(x,z) = (xz+1)² `00:39:37`
   - 8.3. Calcolo matrice K (3×3) `00:40:22`
   - 8.4. Feature map implicita: φ(x) = [1, √2x, x²] `00:41:05`
   - 8.5. Verifica: φ(x)ᵀφ(z) = x²z² + 2xz + 1 `00:42:11`
   - 8.6. Corrispondenza kernel-feature map `00:42:41`
   - 8.7. Importanza scelta corretta del kernel `00:43:22`
   - 8.8. Workflow completo: K → α → predizioni `00:43:56`

### Parte 3 - Algoritmo PageRank
**Durata: ~35 min (00:45:49 - 01:20:35)**

#### 9. PageRank - Introduzione {#pagerank-introduzione}
   - 9.1. Storia: algoritmo alla base di Google Search `00:46:20`
   - 9.2. Riferimento: libro "Google's PageRank and Beyond" `00:47:24`
   - 9.3. Versione semplificata dell'algoritmo `00:48:01`
   - 9.4. Problema: ranking intelligente delle pagine web `00:48:39`
   - 9.5. Limitazione conteggio link: pagine false per manipolare rank `00:49:11`
   - 9.6. Soluzione: pesare importanza delle pagine linkanti `00:49:43`

#### 10. Modello Matematico PageRank {#modello-matematico-pagerank}
   - 10.1. Rappresentazione rete come grafo orientato `00:50:15`
   - 10.2. Esempio: 4 siti web con link bidirezionali `00:51:00`
   - 10.3. Idea: navigazione casuale cliccando link random `00:51:31`
   - 10.4. Iterazione processo → stato stazionario `00:52:03`
   - 10.5. Stato stazionario = importanza relativa siti `00:52:38`

#### 11. Matrice di Adiacenza {#matrice-adiacenza}
   - 11.1. Definizione: Aᵢⱼ = 1 se link da j a i, 0 altrimenti `00:53:29`
   - 11.2. Costruzione matrice A per rete 4 nodi `00:54:12`
   - 11.3. Correzione: diagonale sempre zero (no self-link) `00:55:21`
   - 11.4. Normalizzazione per colonne: M con colonne somma 1 `00:56:02`
   - 11.5. Interpretazione probabilistica: Mᵢⱼ = P(i←j) `00:56:33`
   - 11.6. M matrice stocastica per colonne `00:57:21`

#### 12. Equazione PageRank {#equazione-pagerank}
   - 12.1. Vettore πₖ: probabilità essere in ogni nodo al passo k `00:57:21`
   - 12.2. Iterazione: πₖ₊₁ = M·πₖ `00:58:05`
   - 12.3. Stato stazionario: M·π = π `00:58:49`
   - 12.4. Riconoscimento: problema autovalori-autovettori `00:59:55`
   - 12.5. π autovettore di M con autovalore λ=1 `01:00:30`
   - 12.6. Componenti π = probabilità importanza pagine `01:01:02`

#### 13. Teorema di Perron-Frobenius {#teorema-perron-frobenius}
   - 13.1. Domande: λ=1 sempre autovalore? È il più grande? π unico? `01:01:33`
   - 13.2. Enunciato: M stocastica per colonne → λ=1 massimo `01:02:07`
   - 13.3. Autovettore per λ=1 unico `01:03:06`
   - 13.4. Componenti π > 0 (interpretabili come probabilità) `01:03:06`
   - 13.5. Validità nell'implementazione Google reale (matrice modificata) `01:03:41`

#### 14. Metodo delle Potenze {#metodo-delle-potenze}
   - 14.1. Calcolo autovettore dominante di M `01:04:13`
   - 14.2. Ipotesi: λ₁ separato da altri autovalori `01:04:50`
   - 14.3. Inizializzazione: vettore x₀ arbitrario `01:05:26`
   - 14.4. Iterazione: xₖ₊₁ = M·xₖ (equivale a Mᵏx₀) `01:06:10`
   - 14.5. Decomposizione x₀ su base autovettori: x₀ = Σcᵢvᵢ `01:06:50`
   - 14.6. Applicazione M: Mᵏx₀ = Σcᵢλᵢᵏvᵢ `01:07:21`
   - 14.7. Normalizzazione per λ₁ᵏ: termini (λᵢ/λ₁)ᵏ → 0 `01:08:21`
   - 14.8. Convergenza: φₖ → c₁v₁ (autovettore dominante) `01:08:53`
   - 14.9. Risultato: ranking pagine web da componenti π `01:09:25`
   - 14.10. Dimostrazione Perron-Frobenius (caso semplificato) `01:09:58`

### Parte 4 - Matrix Completion (Introduzione)
**Durata: ~10 min (01:10:35 - 01:19:08)**

#### 15. Problema Matrix Completion {#problema-matrix-completion}
   - 15.1. Definizione: completare matrici con valori mancanti `01:10:35`
   - 15.2. Problema Netflix: suggerimenti film personalizzati `01:11:13`
   - 15.3. Premio $1M Netflix (2009) per algoritmo ottimale `01:11:49`
   - 15.4. Contesto: matrice utenti×film con rating parziali `01:12:22`
   - 15.5. Obiettivo: predire rating mancanti `01:13:03`
   - 15.6. Altro esempio: image inpainting (completare porzioni immagine) `01:13:34`

#### 16. Fattorizzazione a Basso Rango {#fattorizzazione-basso-rango}
   - 16.1. Matrice X (N×D): utenti × film `01:14:11`
   - 16.2. Fattorizzazione X ≈ A·Bᵀ con A (N×r), B (D×r) `01:14:53`
   - 16.3. A = features utenti, B = features film `01:15:25`
   - 16.4. Ipotesi fondamentale: X ha rango r ≪ min(N,D) `01:16:01`
   - 16.5. Fattori latenti: genere, epoca, target audience `01:16:38`
   - 16.6. r utenti rappresentativi + r film rappresentativi `01:17:10`
   - 16.7. Complessità algoritmo basato su SVD `01:17:52`
   - 16.8. Approccio thresholding su valori singolari `01:18:28`
   - 16.9. Prossime lezioni: SVD per matrix completion, SVM `01:19:08`

---

## Metodo Kernel - Introduzione {#metodo-kernel-introduzione}

### Problema dei Dataset Non Lineari

`00:00:09` 
Oggi introduciamo i **metodi kernel**, una tecnica fondamentale per affrontare dataset che presentano **relazioni non lineari** tra features e target. Questo approccio rappresenta una delle innovazioni più eleganti nel machine learning moderno.

`00:00:45` 

**Da KM1.pdf slide 2-3: Il Problema della Linearità**

Consideriamo un problema di regressione dove vogliamo approssimare una relazione tra variabili di input $x$ e output $y$. Nei metodi classici come i **minimi quadrati**, cerchiamo una funzione lineare:

$$
f(x) = w_0 + w_1 x
$$

**Visualizzazione del Problema:**

```
     y
     |
   5 |           ●
     |        ●     ●
   4 |     ●           ●
     |  ●                 ●
   3 | ●                   ●
     |                       ●
   2 |────────────────────────── fit lineare (inadeguato)
     |
   1 |
     |
   0 |_________________________ x
        0  1  2  3  4  5  6  7
```

**Dati con trend parabolico** (es. $y = x^2$):
- Una **retta** è un'approssimazione povera
- L'errore di fitting è elevato anche con $\lambda$ ottimale
- Il modello **lineare non cattura la struttura** dei dati

**Slide 3-4: Inadeguatezza dei Modelli Lineari**

Questo problema emerge frequentemente in applicazioni reali:

1. **Classificazione**: separatori circolari, ellittici o a forma di anello
   - Due classi disposte concentricamente (cerchio interno vs esterno)
   - Nessuna retta può separare le classi
   
2. **Regressione**: relazioni polinomiali, esponenziali, periodiche
   - Crescita quadratica: $y \sim x^2$
   - Interazioni tra features: $y \sim x_1 \cdot x_2$

3. **Pattern complessi**: immagini, testo, serie temporali
   - Strutture gerarchiche
   - Dipendenze non lineari

### Soluzione: Proiezione in Spazi di Dimensione Superiore

`00:03:07` 

**Da KM1.pdf slide 4-6: Feature Maps**

L'idea chiave è **trasformare i dati** in uno spazio di dimensione maggiore dove diventano **linearmente separabili** o **approssimabili linearmente**.

**Definizione Formale**

Una **feature map** (mappa delle caratteristiche) è una funzione:

$$
\varphi : \mathbb{R}^p \to \mathbb{R}^D \quad \text{con } D > p
$$

che proietta i dati originali in uno **spazio arricchito** di dimensione superiore.

`00:04:57` 

**Esempio 1: Da 1D a 2D**

Consideriamo dati unidimensionali $x \in \mathbb{R}$. Definiamo:

$$
\varphi(x) = \begin{bmatrix} x \\ x^2 \end{bmatrix} \in \mathbb{R}^2
$$

**Nello spazio originale** ($\mathbb{R}^1$):
- Relazione non lineare: $y = x^2$
- Impossibile approssimare con retta

**Nello spazio trasformato** ($\mathbb{R}^2$):
- Features: $[x, x^2]$
- Modello lineare: $f(x) = w_1 \cdot x + w_2 \cdot x^2$
- Con $w_1 = 0, w_2 = 1$ otteniamo esattamente $y = x^2$!

`00:06:11` 

Il **modello è lineare nei pesi** $w$, ma **non lineare in $x$** (questo è il punto cruciale).

`00:06:49` 

**Esempio 2: Da 2D a 6D (Polinomiale Completo di Grado 2)**

Per dati bidimensionali $\mathbf{x} = [x_1, x_2]^T \in \mathbb{R}^2$:

$$
\varphi(\mathbf{x}) = \begin{bmatrix}
1 \\
x_1 \\
x_2 \\
x_1^2 \\
x_2^2 \\
\sqrt{2} \cdot x_1 x_2
\end{bmatrix} \in \mathbb{R}^6
$$

`00:07:26` 

**Componenti della Feature Map:**

1. **Termine costante**: $1$ (bias/intercetta)
2. **Termini lineari**: $x_1, x_2$ (features originali)
3. **Termini quadratici puri**: $x_1^2, x_2^2$ (curvature)
4. **Termine di interazione**: $\sqrt{2} \cdot x_1 x_2$ (prodotto misto)

Il fattore $\sqrt{2}$ serve per rendere la feature map **compatibile con kernel polinomiali** (vedremo dopo).

### Vista Geometrica: Separazione Circolare

`00:08:35` 

**Da KM1.pdf slide 7-9: Esempio Circolare**

Consideriamo un problema di **classificazione binaria** in $\mathbb{R}^2$:

- **Classe +1** (rossi): punti dentro un cerchio $x_1^2 + x_2^2 \leq R^2$
- **Classe -1** (blu): punti fuori dal cerchio $x_1^2 + x_2^2 > R^2$

**Nello spazio originale** ($\mathbb{R}^2$):
```
      x₂
       |
    4  |        ○ ○ ○
    3  |      ○ ● ● ● ○
    2  |      ○ ● ● ● ○
    1  |      ○ ● ● ● ○
    0  |────○───○───○───○──── x₁
   -1  |      ○ ● ● ● ○
   -2  |      ○ ● ● ● ○
   -3  |      ○ ● ● ● ○
   -4  |        ○ ○ ○
```

Nessuna **retta** può separare le due classi (problema non linearmente separabile).

**Proiezione in 3D con Feature Map Polinomiale:**

$$
\varphi(x_1, x_2) = \begin{bmatrix}
x_1^2 \\
x_2^2 \\
x_1 x_2
\end{bmatrix} \in \mathbb{R}^3
$$

**Slide 9 - [IMMAGINE: Vista Geometrica]**

L'immagine mostra la trasformazione:
- **Asse Z (verticale)**: $z_1 = x_1^2$
- **Asse Y (profondità)**: $z_2 = x_2^2$  
- **Asse X (orizzontale)**: $z_3 = x_1 x_2$

**Nel nuovo spazio 3D:**
- Punti **interni** al cerchio: basso valore di $x_1^2 + x_2^2$ → bassa "altezza"
- Punti **esterni** al cerchio: alto valore di $x_1^2 + x_2^2$ → alta "altezza"
- Un **piano orizzontale** $z_1 + z_2 = R^2$ separa perfettamente le classi!

**Equazione del separatore:**

Nello spazio originale (circolare):
$$
x_1^2 + x_2^2 = R^2
$$

Nello spazio trasformato (lineare):
$$
w_1 z_1 + w_2 z_2 = R^2 \quad \text{con } w_1 = w_2 = 1
$$

Questo dimostra che il **boundary circolare** diventa un **iperpiano** dopo la trasformazione!

### Apparente Contraddizione con PCA

`00:03:41` 

Potrebbe sembrare strano: nelle lezioni precedenti abbiamo visto PCA per **ridurre** la dimensionalità, ora invece **aumentiamo** le dimensioni. Perché?

**Risposta:**

- **PCA**: riduzione dimensionalità per **visualizzazione** e **compressione**
  - Proietta su sottospazio che cattura massima varianza
  - Perde informazione (trade-off controllato)
  
- **Feature Maps**: aumento dimensionalità per **capacità espressiva**
  - Rende linearmente separabili dataset complessi
  - Aggiunge struttura (polinomi, interazioni)
  - Il kernel trick (prossima sezione) rende computazionalmente fattibile anche per $D \to \infty$!

**Non c'è contraddizione**: sono tecniche complementari per obiettivi diversi.

### Demo con Rete Neurale

`00:10:41` 

Il professore mostra un esperimento con una **rete neurale** per dimostrare l'efficacia delle feature maps:

**Setup:**
- Dataset 2D con separazione circolare (come sopra)
- Rete neurale con **attivazione lineare** (no ReLU/sigmoid)
- Input: features originali $[x_1, x_2]$

**Risultato iniziale:**
- Con solo features lineari: **fallimento totale**
- La rete converge a un separatore lineare (retta) inadeguato
- Accuracy ~50% (come random guessing)

**Aggiunta di Feature Engineering:**

Input arricchito manualmente:
$$
\text{Input} = [x_1, x_2, x_1^2, x_2^2, x_1 x_2]
$$

`00:12:15` 

**Risultato con features polinomiali:**
- **Separatore perfetto** (circolare)
- Accuracy ~100%
- La rete impara $w = [0, 0, 1, 1, 0]$ (somma dei quadrati)

Questo dimostra che:
1. **Feature engineering è potente** anche senza non linearità
2. La scelta della **feature map giusta** è cruciale
3. I metodi kernel automatizzano questo processo!

---

## Feature Map e Kernel Regression {#kernel-regression}

### Formalizzazione del Problema

`00:13:59` 

**Da KM1.pdf slide 10-13: Ridge Regression nello Spazio Arricchito**

Ora formalizziamo matematicamente l'approccio feature map per la **ridge regression**.

**Dataset di Training:**
- $n$ campioni: $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$
- $\mathbf{x}_i \in \mathbb{R}^p$ (input originali)
- $y_i \in \mathbb{R}$ (target)

**Feature Map:**
$$
\varphi : \mathbb{R}^p \to \mathbb{R}^D
$$

`00:14:35` 

**Matrice delle Features Trasformate:**

Definiamo la matrice $\Phi(X) \in \mathbb{R}^{n \times D}$ dove ogni riga è un campione trasformato:

$$
\Phi(X) = \begin{bmatrix}
\varphi(\mathbf{x}_1)^T \\
\varphi(\mathbf{x}_2)^T \\
\vdots \\
\varphi(\mathbf{x}_n)^T
\end{bmatrix} = \begin{bmatrix}
\varphi_1(\mathbf{x}_1) & \varphi_2(\mathbf{x}_1) & \cdots & \varphi_D(\mathbf{x}_1) \\
\varphi_1(\mathbf{x}_2) & \varphi_2(\mathbf{x}_2) & \cdots & \varphi_D(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\varphi_1(\mathbf{x}_n) & \varphi_2(\mathbf{x}_n) & \cdots & \varphi_D(\mathbf{x}_n)
\end{bmatrix}
$$

**Dimensioni:**
- $\Phi(X)$: $n \times D$ (samples × features trasformate)
- $n$ = numero campioni (tipicamente 100-10000)
- $D$ = dimensione spazio arricchito (può essere **enorme** o **infinito**!)

### Ridge Regression nello Spazio Trasformato

`00:15:11` 

Vogliamo trovare $\mathbf{w}^* \in \mathbb{R}^D$ che minimizza:

$$
\mathcal{L}(\mathbf{w}) = \|\Phi(X)\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2
$$

**Componenti:**
- **Loss empirico**: $\|\Phi(X)\mathbf{w} - \mathbf{y}\|_2^2 = \sum_{i=1}^n (y_i - \mathbf{w}^T \varphi(\mathbf{x}_i))^2$
- **Regolarizzazione**: $\lambda \|\mathbf{w}\|_2^2$ (penalizza pesi grandi, previene overfitting)
- **Parametro**: $\lambda > 0$ (trade-off bias-variance)

**Soluzione Analitica (Primal Form):**

Derivando rispetto a $\mathbf{w}$ e ponendo uguale a zero:

$$
\nabla_{\mathbf{w}} \mathcal{L} = 2\Phi(X)^T(\Phi(X)\mathbf{w} - \mathbf{y}) + 2\lambda \mathbf{w} = \mathbf{0}
$$

Semplificando:

$$
\Phi(X)^T \Phi(X) \mathbf{w} + \lambda \mathbf{w} = \Phi(X)^T \mathbf{y}
$$

$$
\boxed{(\Phi(X)^T \Phi(X) + \lambda I_D) \mathbf{w}^* = \Phi(X)^T \mathbf{y}}
$$

**Soluzione esplicita:**

$$
\mathbf{w}^* = (\Phi(X)^T \Phi(X) + \lambda I_D)^{-1} \Phi(X)^T \mathbf{y}
$$

### Il Problema Computazionale

`00:16:55` 

**Complessità della Soluzione Primal:**

Per risolvere il sistema lineare $(\Phi^T\Phi + \lambda I_D)\mathbf{w} = \Phi^T\mathbf{y}$:

1. **Calcolo** $\Phi^T\Phi$: $O(n D^2)$ operazioni
2. **Dimensione matrice**: $D \times D$ (può essere gigantesca!)
3. **Inversione**: $O(D^3)$ operazioni con Cholesky

**Esempi problematici:**

| Feature Map | $p$ (input) | $D$ (output) | Fattibilità |
|-------------|-------------|--------------|-------------|
| Polinomiale grado 2 | 100 | $\binom{102}{2} \approx 5{,}151$ | ✓ Fattibile |
| Polinomiale grado 5 | 100 | $\binom{105}{5} \approx 96M$ | ✗ **Impossibile** |
| Gaussiano (RBF) | qualsiasi | $\infty$ | ✗ **Impossibile** |

**Slide 13: Il Kernel Gaussiano**

Il kernel **Radial Basis Function (RBF)** o Gaussiano:

$$
K(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2}\right) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)
$$

corrisponde a una feature map in **spazio infinito-dimensionale**:

$$
\varphi : \mathbb{R}^p \to \mathbb{R}^\infty
$$

**Implicazioni:**
- $D = \infty$ → impossibile costruire $\Phi(X)$ esplicitamente
- $\Phi^T\Phi$ non rappresentabile in memoria
- Metodo primal **completamente inutilizzabile**

`00:17:32` 

### Necessità del Kernel Trick

**Problema fondamentale:**

Per feature maps complesse (polinomiali alto grado, Gaussiano), la dimensione $D$ è:
- Troppo grande per essere memorizzata
- Troppo costosa per essere calcolata
- Potenzialmente infinita

**Soluzione:**

Il **kernel trick** (prossima sezione) permette di:
1. Lavorare **implicitamente** nello spazio $\mathbb{R}^D$ senza mai calcolare $\varphi(\mathbf{x})$
2. Risolvere il problema in tempo $O(n^3)$ invece di $O(D^3)$
3. Utilizzare kernel con $D = \infty$ (come Gaussiano) senza problemi!

**Intuizione chiave:**

Non abbiamo bisogno di $\varphi(\mathbf{x})$ esplicitamente, ma solo dei **prodotti scalari**:
$$
\langle \varphi(\mathbf{x}_i), \varphi(\mathbf{x}_j) \rangle
$$

E questi possono essere calcolati **direttamente** con una **funzione kernel** $K(\mathbf{x}_i, \mathbf{x}_j)$!

---

## Kernel Trick e Representer Theorem {#kernel-trick-teoria}

### Definizione di Kernel

`00:18:08` 

**Da KM1.pdf slide 14-16: La Funzione Kernel**

Una **funzione kernel** è una funzione $K : \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$ che calcola il prodotto scalare nello spazio trasformato:

$$
\boxed{K(\mathbf{x}_i, \mathbf{x}_j) = \langle \varphi(\mathbf{x}_i), \varphi(\mathbf{x}_j) \rangle = \varphi(\mathbf{x}_i)^T \varphi(\mathbf{x}_j)}
$$

**Il "Trick":**

Il kernel permette di calcolare il prodotto scalare **senza mai costruire** $\varphi(\mathbf{x})$ esplicitamente!

`00:18:45` 

**Esempio: Kernel Polinomiale Quadratico**

Per $\mathbf{x}, \mathbf{z} \in \mathbb{R}^2$, consideriamo:

$$
K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + 1)^2
$$

**Espansione:**

$$
\begin{align}
K(\mathbf{x}, \mathbf{z}) &= (x_1 z_1 + x_2 z_2 + 1)^2 \\
&= x_1^2 z_1^2 + x_2^2 z_2^2 + 1 + 2x_1 z_1 x_2 z_2 + 2x_1 z_1 + 2x_2 z_2
\end{align}
$$

Questo è il prodotto scalare della feature map:

$$
\varphi(\mathbf{x}) = \begin{bmatrix}
x_1^2 \\
x_2^2 \\
1 \\
\sqrt{2} x_1 x_2 \\
\sqrt{2} x_1 \\
\sqrt{2} x_2
\end{bmatrix} \in \mathbb{R}^6
$$

**Vantaggio computazionale:**

- **Con feature map**: $O(D) = O(6)$ operazioni per calcolare $\varphi$, poi $O(D)$ per prodotto scalare → $O(12)$ totale
- **Con kernel**: $(x_1 z_1 + x_2 z_2 + 1)^2$ → $O(p) = O(2)$ somme + 1 quadrato → $O(3)$ totale

Per $D$ grande, il **risparmio è enorme**!

### Matrice di Gram (Kernel Matrix)

`00:19:54` 

**Da KM1.pdf slide 25: Matrice K**

Definiamo la **matrice di Gram** o **kernel matrix**:

$$
K = \begin{bmatrix}
K(\mathbf{x}_1, \mathbf{x}_1) & K(\mathbf{x}_1, \mathbf{x}_2) & \cdots & K(\mathbf{x}_1, \mathbf{x}_n) \\
K(\mathbf{x}_2, \mathbf{x}_1) & K(\mathbf{x}_2, \mathbf{x}_2) & \cdots & K(\mathbf{x}_2, \mathbf{x}_n) \\
\vdots & \vdots & \ddots & \vdots \\
K(\mathbf{x}_n, \mathbf{x}_1) & K(\mathbf{x}_n, \mathbf{x}_2) & \cdots & K(\mathbf{x}_n, \mathbf{x}_n)
\end{bmatrix} \in \mathbb{R}^{n \times n}
$$

**Proprietà fondamentale:**

$$
K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j) = \varphi(\mathbf{x}_i)^T \varphi(\mathbf{x}_j)
$$

quindi:

$$
\boxed{K = \Phi(X) \Phi(X)^T}
$$

**Dimensioni cruciali:**
- $K$: $n \times n$ (dipende dai campioni, non dalle features!)
- $\Phi^T\Phi$: $D \times D$ (soluzione primal - può essere enorme)
- $\Phi\Phi^T$: $n \times n$ (soluzione dual - sempre gestibile!)

**Proprietà matematiche:**

1. **Simmetrica**: $K_{ij} = K_{ji}$ (perché prodotto scalare è commutativo)
2. **Semidefinita positiva**: $\mathbf{v}^T K \mathbf{v} \geq 0$ per ogni $\mathbf{v} \in \mathbb{R}^n$
   - Dimostrazione: $\mathbf{v}^T K \mathbf{v} = \mathbf{v}^T \Phi\Phi^T \mathbf{v} = \|\Phi^T\mathbf{v}\|^2 \geq 0$
3. **Interpretabile**: $K_{ij}$ = "similarità" tra $\mathbf{x}_i$ e $\mathbf{x}_j$

`00:20:27`

### Esempi di Funzioni Kernel {#esempi-kernel}

`00:21:02` 

**Da KM1.pdf slide 17-18: Kernel Più Usati**

#### 1. Kernel Polinomiale

$$
\boxed{K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^q}
$$

**Parametri:**
- $c \geq 0$: termine costante
- $q \in \mathbb{N}$: grado del polinomio

`00:21:35` 

**Casi speciali:**

a) **Polinomiale omogeneo** ($c = 0$):
$$
K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z})^q
$$
- Include **solo** termini di grado esattamente $q$
- Esempio $q=2$, $p=2$: $x_1^2 z_1^2, x_2^2 z_2^2, x_1 x_2 z_1 z_2$

b) **Polinomiale completo** ($c > 0$):
$$
K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + 1)^q
$$
- Include termini di **tutti i gradi** da 0 a $q$
- Esempio $q=2$, $p=2$: $1, x_1, x_2, x_1^2, x_2^2, x_1 x_2$

**Dimensione feature space:**

Per kernel polinomiale di grado $q$ in $\mathbb{R}^p$:

$$
D = \binom{p + q}{q} = \frac{(p+q)!}{p! \cdot q!}
$$

Esempio: $p=100$, $q=5$ → $D \approx 96$ milioni!

#### 2. Kernel Gaussiano (RBF)

`00:22:07` 

Il kernel più utilizzato in pratica:

$$
\boxed{K(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2}\right) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)}
$$

**Parametro:**
- $\gamma = \frac{1}{2\sigma^2} > 0$: larghezza della Gaussiana (bandwidth)
- $\sigma$: deviazione standard

**Proprietà:**

1. **Similarità locale**: 
   - $K(\mathbf{x}, \mathbf{z}) \to 1$ quando $\mathbf{x} \to \mathbf{z}$ (massima similarità)
   - $K(\mathbf{x}, \mathbf{z}) \to 0$ quando $\|\mathbf{x} - \mathbf{z}\| \to \infty$ (minima similarità)

2. **Spazio infinito-dimensionale**: 
   - Il kernel RBF corrisponde a $\varphi : \mathbb{R}^p \to \mathbb{R}^\infty$
   - Espansione in serie di Taylor (infiniti termini)

3. **Universal approximator**: 
   - Con $\gamma$ appropriato, può approssimare **qualsiasi funzione continua**

`00:22:40` 

**Scelta di $\gamma$:**

- $\gamma$ **piccolo** ($\sigma$ grande): 
  - Influenza estesa → modello più smooth
  - Rischio di **underfitting**
  
- $\gamma$ **grande** ($\sigma$ piccolo):
  - Influenza locale → modello molto flessibile
  - Rischio di **overfitting**

**Perché è il più usato:**
- Funziona bene in molti problemi reali
- Un solo iperparametro ($\gamma$) vs due per polinomiale ($c, q$)
- Non soffre di problemi numerici (polinomi alti possono divergere)

#### 3. Kernel Lineare

$$
K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}
$$

Caso speciale (nessuna trasformazione, $\varphi(\mathbf{x}) = \mathbf{x}$). Utile come baseline.

---

## Representer Theorem {#representer-theorem}

`00:23:17` 

**Da KM1.pdf slide 19-24: Il Teorema Fondamentale dei Kernel**

Il **Representer Theorem** è il risultato teorico che giustifica l'uso dei kernel. Dimostra che la soluzione ottimale può essere espressa come **combinazione lineare dei campioni di training**.

### Enunciato

**Teorema (Representer Theorem per Ridge Regression):**

La soluzione ottimale $\mathbf{w}^*$ del problema:

$$
\min_{\mathbf{w} \in \mathbb{R}^D} \|\Phi(X)\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2
$$

può essere scritta come:

$$
\boxed{\mathbf{w}^* = \sum_{i=1}^n \alpha_i \varphi(\mathbf{x}_i) = \Phi(X)^T \boldsymbol{\alpha}}
$$

dove $\boldsymbol{\alpha} = [\alpha_1, \ldots, \alpha_n]^T \in \mathbb{R}^n$.

`00:24:39` 

**Interpretazione:**

- $\mathbf{w}^*$ appartiene allo **span** (spazio generato) dei vettori di training trasformati
- Invece di cercare $D$ parametri, cerchiamo $n$ coefficienti $\alpha_i$
- Se $n \ll D$, questo è un **enorme risparmio**!

`00:25:21` 

### Dimostrazione Completa

`00:26:24` 

**Da KM1.pdf slide 22-23: Proof del Representer Theorem**

**Idea:** Decomporre $\mathbf{w}$ in componente parallela e perpendicolare allo span dei dati.

**Step 1: Decomposizione Ortogonale**

Ogni vettore $\mathbf{w} \in \mathbb{R}^D$ può essere scomposto come:

$$
\mathbf{w} = \mathbf{w}_\parallel + \mathbf{w}_\perp
$$

dove:
- $\mathbf{w}_\parallel \in \text{span}\{\varphi(\mathbf{x}_1), \ldots, \varphi(\mathbf{x}_n)\}$ (componente parallela)
- $\mathbf{w}_\perp \perp \varphi(\mathbf{x}_i)$ per ogni $i$ (componente perpendicolare)

Possiamo scrivere:

$$
\mathbf{w}_\parallel = \sum_{i=1}^n \alpha_i \varphi(\mathbf{x}_i) = \Phi(X)^T \boldsymbol{\alpha}
$$

per qualche $\boldsymbol{\alpha} \in \mathbb{R}^n$.

**Step 2: La Componente Perpendicolare Non Contribuisce alle Predizioni**

`00:27:40` 

Per ogni campione $\mathbf{x}_j$ nel training set:

$$
\begin{align}
\mathbf{w}^T \varphi(\mathbf{x}_j) &= (\mathbf{w}_\parallel + \mathbf{w}_\perp)^T \varphi(\mathbf{x}_j) \\
&= \mathbf{w}_\parallel^T \varphi(\mathbf{x}_j) + \mathbf{w}_\perp^T \varphi(\mathbf{x}_j) \\
&= \mathbf{w}_\parallel^T \varphi(\mathbf{x}_j) + 0 \quad \text{(ortogonalità!)} \\
&= \mathbf{w}_\parallel^T \varphi(\mathbf{x}_j)
\end{align}
$$

Quindi:

$$
\Phi(X)\mathbf{w} = \Phi(X)\mathbf{w}_\parallel
$$

Il **loss empirico** dipende solo da $\mathbf{w}_\parallel$:

$$
\|\Phi(X)\mathbf{w} - \mathbf{y}\|^2 = \|\Phi(X)\mathbf{w}_\parallel - \mathbf{y}\|^2
$$

**Step 3: La Regolarizzazione Penalizza la Componente Perpendicolare**

`00:29:00` 

Per il **teorema di Pitagora** in spazi con prodotto scalare:

$$
\|\mathbf{w}\|^2 = \|\mathbf{w}_\parallel\|^2 + \|\mathbf{w}_\perp\|^2
$$

(le componenti ortogonali contribuiscono indipendentemente).

**Step 4: Minimizzazione**

La funzione obiettivo diventa:

$$
\begin{align}
\mathcal{L}(\mathbf{w}) &= \|\Phi(X)\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2 \\
&= \|\Phi(X)\mathbf{w}_\parallel - \mathbf{y}\|^2 + \lambda (\|\mathbf{w}_\parallel\|^2 + \|\mathbf{w}_\perp\|^2)
\end{align}
$$

**Osservazione chiave:**

- Il termine $\|\mathbf{w}_\perp\|^2$ **aumenta** il loss (contributo positivo alla regolarizzazione)
- Ma $\mathbf{w}_\perp$ **non migliora** il fit ai dati (ortogonale!)

**Conclusione:** Per minimizzare $\mathcal{L}$, dobbiamo scegliere $\mathbf{w}_\perp = \mathbf{0}$.

`00:30:17` 

Quindi:

$$
\mathbf{w}^* = \mathbf{w}_\parallel = \Phi(X)^T \boldsymbol{\alpha}
$$

**□ QED**

### Significato e Implicazioni

**Da KM1.pdf slide 24: Significance**

1. **Riduzione dimensionalità del problema:**
   - Da cercare $\mathbf{w} \in \mathbb{R}^D$ (potenzialmente infinito)
   - A cercare $\boldsymbol{\alpha} \in \mathbb{R}^n$ (sempre finito!)

2. **Giustificazione teorica del kernel trick:**
   - Possiamo lavorare solo con combinazioni dei dati di training
   - Non servono le coordinate esplicite in $\mathbb{R}^D$

3. **Valido anche per $D = \infty$:**
   - Funziona con kernel Gaussiano (spazio infinito-dimensionale)
   - La matematica rimane ben definita

4. **Interpretazione geometrica:**
   - La soluzione vive nel **sottospazio generato dai dati**
   - Questo sottospazio ha dimensione al massimo $n$ (numero campioni)

5. **Generalizzazione:**
   - Il teorema vale per molte loss functions (non solo quadratica)
   - Vale per SVM, kernel logistic regression, etc.

---

## Soluzione Dual: Kernel Ridge Regression {#soluzione-kernel-ridge}

### Derivazione del Problema Duale

`00:31:35` 

**Da KM1.pdf slide 26-30: From Primal to Dual**

Ora usiamo il Representer Theorem per riscrivere il problema di ottimizzazione in termini di $\boldsymbol{\alpha}$.

**Problema Primal (nello spazio features):**

$$
\min_{\mathbf{w} \in \mathbb{R}^D} \|\Phi(X)\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2
$$

La soluzione soddisfa:

$$
(\Phi(X)^T \Phi(X) + \lambda I_D) \mathbf{w}^* = \Phi(X)^T \mathbf{y}
$$

**Sistema lineare $D \times D$** (intractable per $D$ grande!).

**Sostituzione con Representer Theorem:**

Sappiamo che $\mathbf{w}^* = \Phi(X)^T \boldsymbol{\alpha}$. Sostituiamo nell'equazione normale:

$$
(\Phi(X)^T \Phi(X) + \lambda I_D) \Phi(X)^T \boldsymbol{\alpha} = \Phi(X)^T \mathbf{y}
$$

**Moltiplicazione a sinistra per $\Phi(X)$:**

$$
\Phi(X) (\Phi(X)^T \Phi(X) + \lambda I_D) \Phi(X)^T \boldsymbol{\alpha} = \Phi(X) \Phi(X)^T \mathbf{y}
$$

**Espandiamo il prodotto:**

$$
\Phi(X) \Phi(X)^T \Phi(X) \Phi(X)^T \boldsymbol{\alpha} + \lambda \Phi(X) \Phi(X)^T \boldsymbol{\alpha} = \Phi(X) \Phi(X)^T \mathbf{y}
$$

**Definiamo** $K = \Phi(X) \Phi(X)^T$ (kernel matrix):

$$
K^2 \boldsymbol{\alpha} + \lambda K \boldsymbol{\alpha} = K \mathbf{y}
$$

$$
K(K + \lambda I_n) \boldsymbol{\alpha} = K \mathbf{y}
$$

`00:32:07` 

**Problema Duale (Kernel Ridge Regression):**

$$
\boxed{(K + \lambda I_n) \boldsymbol{\alpha} = \mathbf{y}}
$$

**Soluzione:**

$$
\boxed{\boldsymbol{\alpha}^* = (K + \lambda I_n)^{-1} \mathbf{y}}
$$

`00:32:45` 

**Confronto Primal vs Dual:**

| Aspetto | Primal | Dual |
|---------|--------|------|
| Variabile | $\mathbf{w} \in \mathbb{R}^D$ | $\boldsymbol{\alpha} \in \mathbb{R}^n$ |
| Sistema lineare | $D \times D$ | $n \times n$ |
| Complessità | $O(D^3)$ | $O(n^3)$ |
| Richiede $\varphi$ esplicita? | Sì | **No** (solo kernel) |
| Fattibile con $D=\infty$? | **No** | **Sì** |

**Vantaggio cruciale:**

Se $n < D$ (tipico!), il problema **dual** è molto più efficiente. E funziona anche con $D = \infty$!

### Eigendecomposition per Stabilità Numerica

`00:34:02` 

**Da KM1.pdf slide 31: Soluzione Stabile**

La matrice $K$ è **simmetrica** e **semidefinita positiva**. Possiamo usare l'**eigendecomposition** per risolvere il sistema in modo stabile ed efficiente.

**Decomposizione agli Autovalori:**

$$
K = U \Lambda U^T
$$

dove:
- $U \in \mathbb{R}^{n \times n}$: matrice ortogonale di autovettori ($U^T U = I$)
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$: autovalori (tutti $\geq 0$)

**Inversione Stabile:**

$$
\begin{align}
(K + \lambda I_n)^{-1} &= (U \Lambda U^T + \lambda U U^T)^{-1} \\
&= (U (\Lambda + \lambda I) U^T)^{-1} \\
&= U (\Lambda + \lambda I)^{-1} U^T
\end{align}
$$

`00:34:33` 

**Soluzione per $\boldsymbol{\alpha}$:**

$$
\boxed{\boldsymbol{\alpha}^* = U (\Lambda + \lambda I)^{-1} U^T \mathbf{y}}
$$

**Vantaggi:**

1. **Stabilità numerica**: aggiungere $\lambda$ evita autovalori zero/piccoli
2. **Efficienza**: eigendecomposition costa $O(n^3)$ (una volta sola)
3. **Interpretazione**: regolarizzazione filtra autovalori piccoli

**Forma esplicita:**

Se $U = [\mathbf{u}_1, \ldots, \mathbf{u}_n]$ e $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$:

$$
\boldsymbol{\alpha}^* = \sum_{i=1}^n \frac{\lambda_i}{\lambda_i + \lambda} (\mathbf{u}_i^T \mathbf{y}) \mathbf{u}_i
$$

- Autovalori **grandi** ($\lambda_i \gg \lambda$): coefficiente $\approx 1$ (direzione importante)
- Autovalori **piccoli** ($\lambda_i \ll \lambda$): coefficiente $\approx 0$ (direzione filtrata)

### Predizioni su Nuovi Dati

`00:36:24` 

**Da KM1.pdf slide 32: Making Predictions**

Una volta trovato $\boldsymbol{\alpha}^*$, come prediciamo il valore per un nuovo punto $\mathbf{x}_*$?

**Predizione:**

$$
\begin{align}
\hat{y}_* &= f(\mathbf{x}_*) = \mathbf{w}^{*T} \varphi(\mathbf{x}_*) \\
&= (\Phi(X)^T \boldsymbol{\alpha}^*)^T \varphi(\mathbf{x}_*) \\
&= (\boldsymbol{\alpha}^*)^T \Phi(X) \varphi(\mathbf{x}_*) \\
&= \sum_{i=1}^n \alpha_i^* \varphi(\mathbf{x}_i)^T \varphi(\mathbf{x}_*) \\
&= \sum_{i=1}^n \alpha_i^* K(\mathbf{x}_i, \mathbf{x}_*)
\end{align}
$$

`00:38:01` 

**Formula finale:**

$$
\boxed{\hat{y}_* = \sum_{i=1}^n \alpha_i^* K(\mathbf{x}_i, \mathbf{x}_*)}
$$

**Conclusione fondamentale:**

L'intero processo—dal training alle predizioni—utilizza **solo** valutazioni del kernel $K(\cdot, \cdot)$. Non calcoliamo mai $\varphi$ esplicitamente!

**Pipeline completa:**

1. **Training:**
   - Calcola $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ per $i,j = 1,\ldots,n$
   - Risolvi $(K + \lambda I_n) \boldsymbol{\alpha} = \mathbf{y}$
   - Ottieni $\boldsymbol{\alpha}^*$

2. **Inference:**
   - Per nuovo $\mathbf{x}_*$, calcola $K(\mathbf{x}_i, \mathbf{x}_*)$ per $i=1,\ldots,n$
   - Predizione: $\hat{y}_* = \sum_i \alpha_i^* K(\mathbf{x}_i, \mathbf{x}_*)$

**Costo computazionale:**

- **Training**: $O(n^2 p)$ (kernel matrix) + $O(n^3)$ (inversione) = $O(n^3)$
- **Inference per un punto**: $O(np)$ (calcolo $n$ kernel)
- **Inference per $m$ punti**: $O(mnp)$

---

## Esempi Numerici Completi {#esempio-kernel-polinomiale}

### Esempio 1: Kernel Polinomiale K(x,z) = (xz+1)²

`00:39:02` 

**Da KM1.pdf slide 33-35: Polynomial Kernel Example**

Consideriamo un esempio 1D per verificare l'equivalenza tra kernel trick e feature map esplicita.

**Dataset:**
- $x_1 = -1$, $y_1 = ?$
- $x_2 = 0$, $y_2 = ?$
- $x_3 = 1$, $y_3 = ?$

**Kernel:**
$$
K(x, z) = (xz + 1)^2
$$

`00:39:37` 

#### Step 1: Calcolo Matrice K con Kernel Trick

`00:40:22` 

$$
\begin{align}
K_{11} &= K(x_1, x_1) = ((-1)(-1) + 1)^2 = (1+1)^2 = 4 \\
K_{12} &= K(x_1, x_2) = ((-1)(0) + 1)^2 = 1^2 = 1 \\
K_{13} &= K(x_1, x_3) = ((-1)(1) + 1)^2 = 0^2 = 0 \\
K_{22} &= K(x_2, x_2) = (0 \cdot 0 + 1)^2 = 1 \\
K_{23} &= K(x_2, x_3) = (0 \cdot 1 + 1)^2 = 1 \\
K_{33} &= K(x_3, x_3) = (1 \cdot 1 + 1)^2 = 4
\end{align}
$$

**Matrice di Gram:**

$$
K = \begin{bmatrix}
4 & 1 & 0 \\
1 & 1 & 1 \\
0 & 1 & 4
\end{bmatrix}
$$

#### Step 2: Feature Map Implicita

`00:41:05` 

**Espansione del kernel:**

$$
K(x, z) = (xz + 1)^2 = x^2 z^2 + 2xz + 1
$$

Questo suggerisce la feature map:

$$
\varphi(x) = \begin{bmatrix}
x^2 \\
\sqrt{2} x \\
1
\end{bmatrix} \in \mathbb{R}^3
$$

`00:42:11` 

**Verifica:**

$$
\varphi(x)^T \varphi(z) = x^2 z^2 + (\sqrt{2}x)(\sqrt{2}z) + 1 \cdot 1 = x^2z^2 + 2xz + 1 = K(x,z) \quad \checkmark
$$

#### Step 3: Calcolo Esplicito con Feature Map

`00:42:41` 

**Features trasformate:**

$$
\begin{align}
\varphi(x_1) = \varphi(-1) &= \begin{bmatrix} 1 \\ -\sqrt{2} \\ 1 \end{bmatrix} \\
\varphi(x_2) = \varphi(0) &= \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \\
\varphi(x_3) = \varphi(1) &= \begin{bmatrix} 1 \\ \sqrt{2} \\ 1 \end{bmatrix}
\end{align}
$$

**Calcolo elementi kernel matrix:**

$$
\begin{align}
K_{11} &= \varphi(-1)^T \varphi(-1) = 1 + 2 + 1 = 4 \quad \checkmark \\
K_{13} &= \varphi(-1)^T \varphi(1) = 1 + (-\sqrt{2})(\sqrt{2}) + 1 = 1 - 2 + 1 = 0 \quad \checkmark \\
K_{33} &= \varphi(1)^T \varphi(1) = 1 + 2 + 1 = 4 \quad \checkmark
\end{align}
$$

Stesso risultato! ✓

### Esempio 2: Kernel Quadratico Omogeneo 2D

`00:43:22` 

**Da KM1.pdf slide 36-38: Homogeneous Quadratic Kernel**

**Feature map per $\mathbf{x} = [x_1, x_2]^T \in \mathbb{R}^2$:**

$$
\varphi(\mathbf{x}) = \begin{bmatrix}
x_1^2 \\
x_2^2 \\
\sqrt{2} x_1 x_2
\end{bmatrix} \in \mathbb{R}^3
$$

#### Derivazione del Kernel

Calcoliamo il prodotto scalare:

$$
\begin{align}
K(\mathbf{x}, \mathbf{z}) &= \varphi(\mathbf{x})^T \varphi(\mathbf{z}) \\
&= x_1^2 z_1^2 + x_2^2 z_2^2 + (\sqrt{2} x_1 x_2)(\sqrt{2} z_1 z_2) \\
&= x_1^2 z_1^2 + x_2^2 z_2^2 + 2 x_1 x_2 z_1 z_2 \\
&= (x_1 z_1 + x_2 z_2)^2 \\
&= (\mathbf{x}^T \mathbf{z})^2
\end{align}
$$

**Kernel omogeneo quadratico:**

$$
\boxed{K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z})^2}
$$

#### Verifica Numerica

**Dati:**
- $\mathbf{x}_1 = [1, 1]^T$
- $\mathbf{x}_2 = [1, 0]^T$

**Metodo A (Kernel Trick):**

$$
K(\mathbf{x}_1, \mathbf{x}_1) = ((1)(1) + (1)(1))^2 = 2^2 = 4
$$

**Metodo B (Feature Map Esplicita):**

$$
\varphi(\mathbf{x}_1) = \begin{bmatrix} 1 \\ 1 \\ \sqrt{2} \end{bmatrix}
$$

$$
\varphi(\mathbf{x}_1)^T \varphi(\mathbf{x}_1) = 1 + 1 + 2 = 4 \quad \checkmark
$$

**Conclusione:** Equivalenza perfetta verificata!

`00:43:56` 

**Workflow completo Kernel Ridge Regression:**

```
Input: {(xᵢ, yᵢ)}ⁿᵢ₌₁, kernel K, λ

1. Calcola K ∈ ℝⁿˣⁿ:
   Kᵢⱼ = K(xᵢ, xⱼ)

2. Risolvi sistema lineare:
   (K + λIₙ)α = y
   → α* = (K + λIₙ)⁻¹y

3. Per nuovo x*:
   ŷ* = Σᵢ αᵢ* K(xᵢ, x*)

Output: modello predittivo
```

**Importanza scelta kernel:**
- Kernel **polinomiale**: boundary polinomiali (cerchi, ellissi)
- Kernel **Gaussiano**: boundary arbitrariamente complessi
- Kernel **lineare**: baseline (nessuna trasformazione)

La scelta dipende dal problema specifico e va validata con cross-validation!

--- 
Okay, quindi il primo argomento che discuteremo oggi è il cosiddetto metodo Kernel. Cos'è il... problema e cos'è il metodo che vedremo per risolvere questo problema. L'idea è che, direi, la maggior parte delle volte il dataset che ci viene dato non è lineare.

`00:01:20` 
nel senso che il trend dei dati è, direi, nel 90% dei casi non lineare. Quindi se ci atteniamo a quello che abbiamo visto finora riguardo all'approccio dei minimi quadrati, quello che abbiamo visto è essenzialmente una generalizzazione al caso multidimensionale dell'approccio classico dei minimi quadrati che avete visto durante il corso di microanalisi.

`00:01:54` 
che equivale a trovare una linea lineare che approssima i dati. E ovviamente, se pensate a un esempio molto semplice, supponiamo che vi vengano dati i dati rappresentati dai punti blu. e state per usare il fit lineare, i minimi quadrati classici,

`00:02:27` 
quello con cui finite è una linea più o meno come quella rappresentata in rosso tratteggiato. Quindi questo è il problema. C'è un modo per generalizzare l'approccio che abbiamo visto per gestire questo tipo di dataset e arrivare a una rappresentazione che sia più rappresentativa della situazione che abbiamo?

`00:03:07` 
In altro modo, quello che vogliamo fare è passare dal caso lineare, all'approccio lineare. L'idea principale è, a prima vista potrebbe essere in qualche modo strana, perché l'idea è, proiettare i dati in uno spazio di dimensione superiore.

`00:03:41` 
Quindi se pensate a quello che abbiamo fatto finora, abbiamo sempre cercato di fare una riduzione della dimensionalità del dataset per essere in grado di catturare la struttura sottostante del dataset. senza la necessità di considerare tutti i dati che abbiamo, ma solo con un sottoinsieme dei dati o con le componenti principali come abbiamo visto alcune delle componenti principali o alcuni dei valori singolari che abbiamo considerato nella lezione precedente.

`00:04:25` 
Ora stiamo, apparentemente stiamo muovendo nella direzione opposta. Quindi stiamo dicendo, voglio prendere i dati e proiettare questi dati in uno spazio di dimensione superiore. Quindi apparentemente stiamo aumentando la complessità del problema perché stiamo considerando uno spazio di dimensione superiore. Vedremo più tardi che.

`00:04:57` 
in realtà questo è solo un, l'inizio della storia, ma vedremo che alla fine saremo in grado di risolvere questo problema che apparentemente è controintuitivo in modo efficiente. Il primo ingrediente che dobbiamo introdurre è la cosiddetta feature map. La feature map è il modo formale per definire quello che ho detto a parole. È una funzione che prende uno.

`00:05:34` 
dei vostri campioni e considerando tutte le caratteristiche del campione crea qualcosa che è di dimensione superiore. Okay? Quindi se vi spostate da, essenzialmente vi spostate da un, campione che è p-dimensionale a un nuovo, vettore di caratteristiche, che è d-dimensionale, D maiuscola, dove D maiuscola è più grande di p, okay?

`00:06:11` 
Esempio, supponiamo che abbiate un dataset composto da dati scalari, quindi solo una caratteristica, e la feature map potrebbe essere questa, quindi x e x al quadrato. Quindi vi spostate da un caso unidimensionale a uno bidimensionale, okay? E qual è l'idea? L'idea è che, se considerate questo tipo di modello lineare, quindi w1 per x e w2 per x al quadrato,

`00:06:49` 
quindi state considerando le caratteristiche del nuovo. campione, del campione allargato, se volete, questo è in realtà un modello lineare nella nuova caratteristica, mentre nello spazio originale, avreste avuto un problema quadratico, perché avete x al quadrato. Quindi, sfruttando questa idea, potete avere un modello lineare nella nuova caratteristica, okay?

`00:07:26` 
Consideriamo un altro esempio, dove avete un vettore di caratteristiche bidimensionale, l'originale, e volete creare una rappresentazione arricchita di questi dati, considerando questo. vettore di caratteristiche. Quindi avete 1, x1, x2, x1 al quadrato, x2 al quadrato, e radice quadrata di 2, x1, x2.

`00:07:59` 
Okay? Quindi vi state spostando da 2D, R2, a R6. E apparentemente questo potrebbe essere strano, ma vedremo che in pratica è qualcosa che è molto, molto utile. Okay. Questa è una vista geometrica di quello che sta succedendo. Quindi, essenzialmente, considerate, per.

`00:08:35` 
esempio, i dati che abbiamo nella parte superiore sinistra della slide. E supponiamo che vogliate fare. clustering di quei dati, quindi volete trovare essenzialmente un confine di separazione dei dati tra i punti rossi e i quadrati blu. È chiaro che un buon confine di separazione per quei dati è un cerchio, che è ovviamente una curva non lineare. Cosa succede se voi.

`00:09:17` 
proiettate in uno spazio tridimensionale quei dati, come per esempio qui, e quello con cui potete arrivare è che in realtà in 3D potete separare i punti rossi dai quadrati blu, con un iperpiano, che è lineare ovviamente, e nel.

`00:09:48` 
Il terzo asse, potete avere, per esempio, qualche combinazione dell'originale, okay? Questo è, se tenete a mente questa immagine, dovrebbe essere chiaro, almeno l'idea sottostante di questo tipo di metodo, il cosiddetto metodo kernel. Lasciatemi solo mostrarvi una cosa.

`00:10:41` 
Okay, supponiamo di avere questi dati. che sono esattamente, essenzialmente, molto simili a quello che abbiamo visto. Quindi abbiamo questi punti blu e questi punti arancioni. E vogliamo classificare quei dati, quindi vogliamo trovare un marcatore di separazione. Qui, le caratteristiche date sono x1 e x2, quindi le coordinate dei punti.

`00:11:18` 
Qui c'è una rappresentazione di una rete neurale, ma l'idea è esattamente la stessa. Quindi se eseguiamo il codice, il codice semplice, Okay, come potete vedere, quello che, forse non è così chiaro, ma anche se state facendo molte iterazioni, potete vedere qualche linea che passa qui, che ovviamente non è un confine di separazione molto buono.

`00:12:15` 
Okay, cosa succede se aggiungo, per esempio, qualcosa di più, che non sono niente altro che quelli che abbiamo ottenuto dalla feature 9. Quindi, sto considerando x1 al quadrato, x2 al quadrato, e x1 x2. Sto eseguendo... lo stesso codice e quello con cui potete arrivare è una perfetta rappresentazione del numero di separazione.

`00:12:50` 
Come potete vedere, in questo caso sto usando, se avete familiarità con le reti neurali, sto usando una funzione di attivazione lineare, che è simile a quello che stiamo facendo nel contesto dei minimi quadrati classici. Quindi non sto usando nessuna funzione di attivazione non lineare che ovviamente aiuterebbe a trovare questa separazione, questo confine di separazione non lineare.

`00:13:22` 
Mi sto attenendo alla funzione di attivazione lineare, ma sto arricchendo il vettore di caratteristiche, con qualcosa che è ottenuto usando un numero di caratteristiche. E, essenzialmente, con la stessa architettura, potete arrivare a una perfetta rappresentazione del numero di separazione. Okay?

## Kernel Regression {#kernel-regression}

`00:13:59` 
Okay, quindi ora quello che vogliamo fare è formalizzare quello che abbiamo descritto. Il primo passo è, vogliamo considerare una feature map applicata al nostro dataset, e per il nuovo dataset, per il dataset arricchito, considereremo la regressione Ridge. Quindi, minimi quadrati con una regolarizzazione norma L2.

`00:14:35` 
Quindi, consideriamo questa matrice, Φ, dove X maiuscola è la matrice originale contenente i dati. E nella matrice Φ, ogni riga è data dall'applicazione della feature map a uno dei vettori di caratteristiche. Quindi, i va da 1 a N, dove N è il numero di campioni.

`00:15:11` 
Quindi, la regressione è, come al solito, ricordate quello che abbiamo visto nel contesto dei minimi quadrati. I minimi quadrati classici erano formalizzati come, voglio trovare un vettore di pesi W, tale che questo modello lineare sia. una buona approssimazione di questi dati, y è il vettore con le etichette o il risultato,

`00:15:46` 
e voglio anche avere il w con la norma minima che abbiamo già menzionato. Qual è la differenza principale tra questa espressione e quella che è sulla slide? La dimensione della matrice X. Originariamente, X è una matrice dove abbiamo n campioni e.

`00:16:17` 
p caratteristiche. Qui, la matrice X è n per d, dove d è maggiore di p, okay? Quindi, il problema che vogliamo risolvere è quel problema di minimizzazione. Ma, prima di tutto, D in generale può essere molto grande. Abbiamo visto che, per esempio, nel secondo caso, la seconda feature map che abbiamo considerato,

`00:16:55` 
ci siamo spostati da R2 a R6. Quindi, potreste avere un aumento davvero grande nell'informazione della matrice. E quindi, ci sono due problemi. Prima di tutto, il calcolo della matrice. E secondo, anche la memorizzazione. E poi, se formalizzate il problema come.

`00:17:32` 
abbiamo visto ricordate che alla fine quello che dovevamo fare era risolvere un sistema lineare, dove la matrice era x trasposta x okay in questo caso questi sono i due per avere una matrice che erano i vostri d per d metodi quindi l'idea è usare la cosiddetta funzione kernel.

## Kernel Trick e Representer Theorem {#kernel-trick-teoria}

`00:18:08` 
giusto cos'è una funzione kernel in pratica considereremo una funzione, che calcola il prodotto scalare tra due vettori di caratteristiche arricchiti. Quindi la funzione K, K maiuscola, è qualcosa che sta dando il prodotto scalare tra phi di x i, scusate, phi di x i,

`00:18:45` 
e phi di x j. Quindi questa è la definizione formale. Quindi una funzione kernel è qualcosa che sta dando il prodotto scalare dei due vettori, e il punto interessante è che, o il requisito è che la funzione kernel di solito è costruita in modo tale che non vi sia mai richiesto di calcolare.

`00:19:22` 
esplicitamente i vettori phi di x i e phi di x j. Quindi questa è l'idea. Voglio arrivare a una funzione K maiuscola che prende come input due vettori originali, vettori di caratteristiche x i e x j, e sta dando come output il prodotto scalare tra i due vettori,

`00:19:54` 
phi di x i e phi di x j. Perché sono interessato ad avere questo tipo di. funzione, perché quando sto considerando x trasposta x, in pratica, il risultato di ogni termine di questa matrice è in realtà il risultato del prodotto di una riga e una colonna, okay? È un prodotto scalare tra x i trasposta, x j trasposta, scusate, x j.

`00:20:27` 
Quindi, qui, ci stiamo spostando a phi di x, okay? E quindi, quello che è chiamato il, in letteratura, potete trovare questa tecnica è molto spesso chiamata il kernel trick. Il kernel trick è essenzialmente l'abilità di riscrivere il vostro problema, il vostro problema di minimizzazione, la soluzione di questo problema di minimizzazione con la phi, Φ maiuscola, qui, usando.

`00:21:02` 
solo questo prodotto interno. Va bene, torniamo indietro. Quindi, quali sono alcuni esempi della funzione kernel che di solito vengono adottati in pratica? Il primo è il cosiddetto kernel polinomiale, che è dato da questa espressione, x trasposta z più c alla potenza q.

`00:21:35` 
Okay? Quindi, se avete c uguale a zero, avete un polinomio omogeneo. Per c positivo, avete anche termini di ordine inferiore, quindi la costante. E un altro kernel che è usato molto spesso è il kernel Gaussiano.

`00:22:07` 
ci vengono dati due vettori x e z, il risultato è dato da questa espressione, e infatti è uno dei kernel più usati. Quindi ora qui, per il momento, abbiamo appena visto qual è l'idea.

## Representer Theorem {#representer-theorem}

`00:22:40` 
Vogliamo spostarci da, da, scusate, questo problema dove dobbiamo effettivamente risolvere il problema con la phi di x e quindi costruire la matrice phi di x trasposta phi di x, sfruttando questa funzione kernel. Cerchiamo di capire come e perché questa idea funziona.

`00:23:17` 
L'idea è basata su un risultato che è chiamato il Representer Theorem. Qui vi sto dando solo una versione semplificata di questo risultato, ma è importante capire, qual è l'idea dietro questo risultato. E tra un momento vedremo anche molto rapidamente la dimostrazione di questo teorema che vi darà.

`00:23:47` 
un'intuizione dell'importanza di questo risultato. Questo teorema dice che la soluzione di una kernel ridge regression, NE, può essere scritta come una combinazione lineare del vettore di caratteristiche dei dati di training. Quindi, nel caso precedente, abbiamo trovato che W-star era in realtà, in classe, se non considerate la regolarizzazione, era, equivale a invertire la matrice X trasposta X, okay?

`00:24:39` 
Nel caso della regolarizzazione, la matrice era X trasposta X più lambda I, okay? E dovevate invertire questa matrice. Qui, quello che stiamo dicendo è che se avete i vostri vettori di caratteristiche, Qui, ovviamente, stiamo considerando i vettori di caratteristiche mappati attraverso la feature map. Il W star, il valore ottimale, la soluzione della kernel ridge regression, può essere scritto come una combinazione lineare dei dati di training.

`00:25:21` 
Ricordate che in questa matrice avete i dati di training, e alpha è un vettore di n componenti, quindi n è il numero di campioni. Essenzialmente, quello che state dicendo è che il W star appartiene allo spazio generato dai vettori di caratteristiche della matrice originale.

`00:25:52` 
Grazie. Quindi, l'idea è che invece di considerare qualsiasi vettore possibilmente di dimensione uguale, possiamo ridurre la nostra ricerca ai vettori n-dimensionali che ci stanno dando il coefficiente alpha. Una volta che abbiamo alpha, abbiamo ottenuto la soluzione.

`00:26:24` 
Quindi, cerchiamo di capire come funziona la dimostrazione. Quindi, supponiamo di avere uno spazio di caratteristiche H e un vettore in questo spazio di caratteristiche. può essere rappresentato, ricordate quello che abbiamo visto, per esempio, anche nel caso di Gram-Schmidt, può essere decomposto in una componente che è w parallelo, è nello span dei dati di training,

`00:27:07` 
e w perpendicolare, che è ortogonale allo spazio generato dai dati di training, i dati di training mappati al vettore di caratteristiche. Ovviamente, abbiamo che w perpendicolare è ortogonale a qualsiasi dei vettori di caratteristiche.

`00:27:40` 
Quindi, se consideriamo il prodotto scalare tra w e phi di xi, questo può essere decomposto in w parallelo scalare phi di xi più il w perpendicolare per scalare phi di xi. Ma questa componente è zero, e quindi abbiamo solo questa parte. Quindi, in altre parole, in questa formula, dove qui avete phi di xi, questo termine dipende.

`00:28:28` 
solo da w parallelo, perché... Qui, essenzialmente, sto prendendo il prodotto scalare tra il vettore w e il vettore di caratteristiche, okay? E questo termine? Questo termine può essere facilmente decomposto nella parte parallela e nella parte perpendicolare.

`00:29:00` 
Quindi se mettete insieme questi due risultati, quindi questo e questo, la funzione che dovete minimizzare è la prima parte, dove qui considererete solo w-parallelo, perché l'altro contributo non è importante, e in questo elemento avete entrambi. Ma questa funzione deve essere minimizzata e quindi per sicuro per trovare il minimo.

`00:29:37` 
quello che dobbiamo fare è azzerare il secondo contributo nel termine di regolarizzazione. E in questo modo abbiamo trovato che w star deve appartenere allo spazio generato dal vettore di caratteristiche di training. E quindi ovviamente può essere rappresentato come una combinazione lineare dei.

`00:30:17` 
vettori di caratteristiche. Grazie. Okay, se vogliamo cercare di dare un messaggio da portare a casa, il representer theorem dice che la soluzione a un problema di regolarizzazione, il representer theorem è valido per qualsiasi problema di regressione,

`00:30:55` 
può essere scritto come una combinazione lineare del vettore di caratteristiche dei dati di training valutato con la funzione kernel. Quindi, ora che sappiamo, che la soluzione w star la soluzione ottimale è data da phi di x trasposta per alpha dove alpha è.

`00:31:35` 
il vettore di coefficienti che dobbiamo trovare possiamo sostituire questa espressione nel nostro problema di minimizzazione quindi questo okay e poi dobbiamo scrivere il problema e la soluzione è trovata risolvendo.

`00:32:07` 
questo sistema lineare dove k è la matrice costruita usando la funzione kernel quindi la funzione kernel, che abbiamo visto, prima, è usata per costruire la cosiddetta matrice di Kernel, o matrice di Gram, che è, il termine generico per la matrice è il prodotto scalare tra due vettori di caratteristiche nello spazio arricchito.

`00:32:45` 
Oppure, se volete, A può essere scritta come Φ e Φ trasposta. E quindi il vettore di coefficienti alpha può essere trovato usando questa espressione. E qui viene il punto. Se siete in grado di trovare una funzione kernel che è in grado di darvi il.

`00:33:20` 
prodotto scalare tra i due vettori senza calcolare esplicitamente il vettore stesso allora questa matrice è calcolata molto facilmente. Possiamo fare un passo ulteriore come al solito. Ricordate che la matrice K è costruita in questo modo quindi è come al solito simmetrica e definita positiva. Quindi in principio possiamo.

`00:34:02` 
usare la decomposizione ai valori singolari per fattorizzare questa matrice, ma in realtà, dato che è simmetrica e definita positiva, sfruttiamo la decomposizione spettrale, la decomposizione agli autovalori, della matrice. Quindi K, e abbiamo U, lambda, U trasposta al quadrato, come al solito, U è la matrice degli autovettori, e lambda è la matrice diagonale degli autovalori.

`00:34:33` 
Quindi questa matrice può essere scritta con i soliti trucchi, come U lambda più lambda I, fate attenzione, questo lambda è il parametro di regolarizzazione, tipicamente lambda è la matrice contenente gli autovalori della matrice K. E quindi qui possiamo calcolare la soluzione in un modo molto semplice, sfruttando la decomposizione della matrice.

`00:35:20` 
E una volta che abbiamo l'alpha, ricordate che il nostro scopo è fare cosa? Se ricordate l'immagine originale, quella dove avevamo i... dati parabolici e con i minimi quadrati classici stavamo arrivando a una parte lineare dei dati, l'idea è dati questi dati voglio creare un modello diciamo questo.

`00:35:53` 
che approssima i dati e questo modello può essere usato per fare alcune previsioni. Quindi voglio avere il valore del modello qui che è un valore non incluso nel dataset originale. Quindi una volta che avete alpha, come potete eseguire la previsione, quindi l'inferenza? Ricordate che il modello è questo, okay?

`00:36:24` 
Quindi è phi per w star, okay? E w star è scritto come phi di x trasposta per alpha. Quindi il calcolo di w star è spostato al calcolo del vettore alpha. Quindi qui avete la, questa è l'espressione che avevate originariamente.

`00:36:57` 
E sfruttando l'espressione di w star in termini di alpha e i vettori di caratteristiche originali, avete questa espressione, okay? Ora alpha, potete portare la sommatoria fuori dal prodotto scalare. E qual è il punto? Il punto è che qui avete, di nuovo, il prodotto scalare.

`00:37:29` 
Tra quella. la funzione phi valutata su due campioni, x i e x star, che è il nuovo valore che volete, per esempio, volete considerare. Quindi, questo è cosa, questo è la funzione kernel, okay, quindi in quel caso, quello che dovete fare è.

`00:38:01` 
valutare la funzione kernel, r, e alpha, pari sono dati, perché li hanno, abbiamo già risolto il problema di regressione di ritorno con la funzione kernel, quindi alpha è noto, qui k è noto. x i sono noti, sono i campioni, i campioni dati, e x star è il nuovo valore dove voglio valutare il mio modello.

`00:38:31` 
E quindi la valutazione del modello è ottenuta valutando la funzione kernel. Quindi di nuovo, anche se il nostro punto di partenza era arricchire lo spazio proiettando in uno spazio di dimensione superiore, alla fine della giornata, quello che stiamo effettivamente facendo è solo usare correttamente la funzione kernel.

## Esempi di Kernel

`00:39:02` 
ovviamente ci deve essere una relazione tra l'operatore di lifting che state usando quindi che tipo di lifting dallo spazio originale allo spazio arricchito state usando e la funzione kernel che sta rappresentando questo lifting quindi facciamo un esempio quindi supponiamo di avere.

`00:39:37` 
questi tre punti x1 meno uno x2 uguale a zero x3 uguale a uno e vogliamo usare il polinomiale k x di z che è uguale a x z più uno al quadrato okay quindi in. E qui avete il calcolo dei termini della matrice, e alla fine avete la vostra matrice K, che è quella a cui siete interessati.

`00:40:22` 
Se considerate questa matrice, questa funzione X, questa è la funzione che state considerando. In pratica, se scrivete esplicitamente questo termine, è X al quadrato, Z al quadrato, più due volte X, Z, più...

`00:41:05` 
Okay, questa è la rappresentazione esplicita dei dati. Quindi lasciatemi solo per un momento dimenticare questo. Senza questo, avreste avuto solo questa rappresentazione. E quindi in realtà ci siamo spostati da x, che era la caratteristica originale, a cosa? A x al quadrato.

`00:41:38` 
e radice quadrata di x. Quindi x e quadrato e radice quadrata di x. Quindi se avete l'altro vettore z, questo è sia z al quadrato che radice quadrata di z. Ora, se fate il prodotto scalare tra questi.

`00:42:11` 
due vettori, avete x al quadrato per z al quadrato, che è questo, e radice quadrata di due, radice quadrata di due, che è due, x, z, okay? Quindi questa funzione kernel, questa, corrisponde a questa feature map, okay? E, ma il punto importante è che se scegliete la funzione kernel.

`00:42:41` 
corretta, allora non avete bisogno, in pratica, di fare questo arricchimento calcolare il vettore arricchito, e calcolare il prodotto scalare. È tutto, è un, qui in una buona scelta della funzione kernel. Okay? Come funziona? Qui se avete il più uno, significa che nel vettore avete anche uno. Okay? Quindi avete uno per uno,

`00:43:22` 
x al quadrato, è un quadrato, e poi quadrato il rettangolare. Okay? Quindi, questo è il punto, e in pratica, ovviamente, questo è molto importante, perché quello che farete nel, direi, 90% dei casi, è modellare dataset non lineari.

`00:43:56` 
Quindi, l'uso dei metodi kernel è molto, molto importante, okay? E, ovviamente, una volta che avete K, tutta questa macchina, una volta che avete questa K, potete calcolare la metrica di Gram, come abbiamo fatto in questo caso semplice. Una volta che avete K, potete risolvere questo problema per alpha con parametri di regolarizzazione adeguati.

`00:44:30` 
Grazie mille. Potete arrivare al vettore alpha e poi potete fare qualsiasi previsione usando il vettore alpha. Okay, è chiaro? Poi ovviamente qui la presentazione è in qualche modo molto teorica ma, è importante perché vedremo forse domani anche un'altra applicazione del.

`00:45:01` 
representer theorem. Forse non avete sentito parlare di support vector machines o support vector regression. Questa è un'altra famiglia di metodi che sono basati sul representer theorem essenzialmente. È un altro modo di usare il representer theorem. Okay, quindi domani considereremo la support vector regression per le macchine lì.

## Algoritmo PageRank

`00:45:49` 
Okay, ora voglio considerare un altro problema, che è un problema che immagino conosciate molto bene. È l'algoritmo PageRank. Voglio presentare qui una versione molto semplice dell'algoritmo. Se siete interessati a...

`00:46:20` 
Immagino che sappiate tutti che l'algoritmo PageRank originariamente era alla base della ricerca web di Google. Se siete interessati a più dettagli sull'algoritmo,

`00:47:24` 
Okay, questo è il libro. Non è molto recente, in realtà, 13 anni fa, ma in questo libro avete tutti i dettagli dei dati. Quindi, qui vi sto dando solo l'idea molto basilare dell'algoritmo. Se volete entrare in più dettagli nell'algoritmo, che è molto interessante perché, essenzialmente, è tutta algebra lineare.

`00:48:01` 
Questo libro è molto, molto buono e molto leggibile. È abbastanza facile da capire, quindi se siete interessati in generale agli algoritmi di ricerca web, questi potrebbero essere i dati. È roba buona, ragazzi. Okay, quindi cerchiamo di vedere la versione semplice, versione molto basilare dell'algoritmo.

## Algoritmo PageRank {#pagerank-introduzione}

### Introduzione e Motivazione

`00:45:49` 

**Da PR1.pdf slide 1-3: PageRank Algorithm**

Il **PageRank** è un algoritmo sviluppato da Larry Page e Sergey Brin che ha rivoluzionato la ricerca web diventando il cuore di **Google Search**.

`00:46:20` 

**Storia:**
- Originariamente alla base del motore di ricerca Google (fine anni '90)
- Nome: gioco di parole su "Page" (cognome + pagina web)
- Idea: misurare l'"importanza" delle pagine web basandosi sulla struttura dei link

`00:47:24` 

**Riferimento:** Libro "Google's PageRank and Beyond: The Science of Search Engine Rankings" (2006)
- Trattazione completa dell'algoritmo
- Collegamenti con algebra lineare e teoria dei grafi
- Molto leggibile e accessibile

`00:48:01` 

Qui presentiamo una **versione semplificata** dell'algoritmo reale, che mantiene i concetti fondamentali.

### Il Problema del Ranking

`00:48:39` 

**Da PR1.pdf slide 2: What is PageRank?**

**Obiettivo:** Classificare pagine web in modo **intelligente** e **robusto**.

**Approccio Naive (Non Funziona):**

Contare semplicemente il numero di link in entrata:
$$
\text{Rank}(i) = \#\{\text{pagine che linkano } i\}
$$

`00:49:11` 

**Problema:** Facilmente manipolabile!
- Creare pagine false che linkano la propria pagina
- Spam farms: reti di pagine interconnesse artificialmente
- Ranking artificiale senza valore reale

`00:49:43` 

**Soluzione PageRank:**

Non solo **quantità** dei link, ma anche **qualità** (importanza) delle pagine linkanti:

> "Un link da una pagina importante vale più di un link da una pagina minore"

**Esempi:**
- Link da **BBC** o **Wikipedia** → alto valore
- Link da pagina personale sconosciuta → basso valore
- Link da spam page → valore quasi nullo

### Modello del "Random Surfer"

**Da PR1.pdf slide 2: The Random Surfer Model**

`00:50:15` 

**Idea intuitiva:**

Immaginiamo un utente che **naviga casualmente** sul web:

1. **Partenza:** inizia su una pagina random
2. **Iterazione:** ad ogni passo, clicca **casualmente** su uno dei link della pagina corrente
3. **Convergenza:** dopo molti passi, emerge una distribuzione di probabilità stazionaria

**Interpretazione:**

Una pagina è **importante** se un random surfer ha alta probabilità di visitarla nel lungo termine.

$$
\boxed{\text{PageRank}(i) = \text{Probabilità di visita a lungo termine della pagina } i}
$$

`00:51:31` 

**Key Insight:**

Se una pagina è linkata da molte pagine importanti (che hanno alto PageRank), allora anche lei diventerà importante!

Questo crea un **sistema auto-consistente** che vedremo essere un problema agli autovalori.

---

## Modello Matematico PageRank {#modello-matematico-pagerank}

### Esempio: Rete con 4 Pagine

`00:51:00` 

**Da PR1.pdf slide 4: A Simple Network Example**

Consideriamo una piccola rete con **4 pagine web**. Le **frecce rappresentano hyperlink**.

**Visualizzazione della Rete:**

```
     1 ← → 2
     ↓     ↓
     ↓     ↓
     3 → → 4
```

**Connessioni:**
- **Pagina 1**: linka pagine 2 e 3
- **Pagina 2**: linka solo pagina 3
- **Pagina 3**: linka pagine 1 e 4
- **Pagina 4**: linka pagine 1 e 2

Useremo questa rete per costruire il modello matematico.

`00:52:38` 

### Matrice di Adiacenza A

`00:53:29` 

**Da PR1.pdf slide 5: The Adjacency Matrix**

La **matrice di adiacenza** $A$ rappresenta la struttura dei link:

$$
A_{ij} = \begin{cases}
1 & \text{se esiste link da pagina } j \text{ a pagina } i \\
0 & \text{altrimenti}
\end{cases}
$$

**Attenzione alla convenzione:** $A_{ij} = 1$ significa link **da** $j$ **a** $i$ (colonna → riga).

`00:54:12` 

**Costruzione per la nostra rete:**

| | Pagina 1 | Pagina 2 | Pagina 3 | Pagina 4 |
|---|---|---|---|---|
| **Pagina 1** | 0 | 0 | 1 | 1 |
| **Pagina 2** | 1 | 0 | 0 | 1 |
| **Pagina 3** | 1 | 1 | 0 | 0 |
| **Pagina 4** | 0 | 0 | 1 | 0 |

$$
A = \begin{bmatrix}
0 & 0 & 1 & 1 \\
1 & 0 & 0 & 1 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

**Interpretazione:**

- **Colonna 1** (pagina 1): link verso pagine 2 e 3 → $A_{21} = 1$, $A_{31} = 1$
- **Colonna 2** (pagina 2): link solo verso pagina 3 → $A_{32} = 1$
- **Colonna 3** (pagina 3): link verso pagine 1 e 4 → $A_{13} = 1$, $A_{43} = 1$
- **Colonna 4** (pagina 4): link verso pagine 1 e 2 → $A_{14} = 1$, $A_{24} = 1$

`00:54:46` 

**Correzione importante:** La **diagonale è sempre zero** (nessuna pagina linka se stessa).

`00:55:21` 

### Matrice di Transizione M (Stocastica)

`00:56:02` 

**Da PR1.pdf slide 6: The Transition Matrix**

La matrice di adiacenza mostra le connessioni, ma per il random surfer servono **probabilità**.

**Costruzione della Matrice di Transizione:**

Per ogni colonna $j$ (pagina sorgente):
1. Conta il numero di link uscenti: $d_j = \sum_{i} A_{ij}$
2. Normalizza dividendo ogni elemento per $d_j$:

$$
M_{ij} = \frac{A_{ij}}{d_j} = \frac{A_{ij}}{\sum_k A_{kj}}
$$

**Interpretazione:**

$$
M_{ij} = P(\text{andare da pagina } j \text{ a pagina } i)
$$

`00:56:33` 

**Calcolo per la nostra rete:**

- **Pagina 1:** 2 link uscenti (verso 2 e 3) → dividi colonna 1 per 2
- **Pagina 2:** 1 link uscente (verso 3) → dividi colonna 2 per 1
- **Pagina 3:** 2 link uscenti (verso 1 e 4) → dividi colonna 3 per 2
- **Pagina 4:** 2 link uscenti (verso 1 e 2) → dividi colonna 4 per 2

$$
M = \begin{bmatrix}
0 & 0 & 1/2 & 1/2 \\
1/2 & 0 & 0 & 1/2 \\
1/2 & 1 & 0 & 0 \\
0 & 0 & 1/2 & 0
\end{bmatrix}
$$

**Verifica:** Ogni **colonna somma a 1** (distribuzione di probabilità).

$$
\sum_{i=1}^4 M_{ij} = 1 \quad \forall j
$$

$M$ è una **matrice stocastica per colonne** (column stochastic matrix).

`00:57:21` 

### Vettore di Stato π e Dinamica

**Da PR1.pdf slide 7: The State Vector π**

**Definizione:** Il vettore $\boldsymbol{\pi}_k \in \mathbb{R}^4$ rappresenta la distribuzione di probabilità al passo $k$:

$$
\boldsymbol{\pi}_k = \begin{bmatrix}
\pi_k^{(1)} \\
\pi_k^{(2)} \\
\pi_k^{(3)} \\
\pi_k^{(4)}
\end{bmatrix}
$$

dove $\pi_k^{(i)}$ = probabilità che il random surfer sia sulla pagina $i$ al tempo $k$.

**Proprietà:**
- $\pi_k^{(i)} \geq 0$ per ogni $i$
- $\sum_{i=1}^4 \pi_k^{(i)} = 1$ (distribuzione di probabilità)

**Iterazione del Random Surfer:**

`00:58:05` 

Al passo successivo:

$$
\boxed{\boldsymbol{\pi}_{k+1} = M \boldsymbol{\pi}_k}
$$

**Interpretazione:** La probabilità di essere sulla pagina $i$ al passo $k+1$ è:

$$
\pi_{k+1}^{(i)} = \sum_{j=1}^4 M_{ij} \pi_k^{(j)} = \sum_{j=1}^4 P(i \leftarrow j) \cdot P(\text{essere in } j)
$$

(somma delle probabilità di arrivare in $i$ da tutte le pagine $j$).

### Stato Stazionario

`00:58:49` 

**Domanda chiave:** Cosa succede per $k \to \infty$?

**Ipotesi:** Il processo converge a uno **stato stazionario** $\boldsymbol{\pi}$ dove le probabilità non cambiano più:

$$
\lim_{k \to \infty} \boldsymbol{\pi}_k = \boldsymbol{\pi}
$$

**Equazione dello Stato Stazionario:**

Se $\boldsymbol{\pi}_{k+1} = \boldsymbol{\pi}_k = \boldsymbol{\pi}$, allora:

$$
\boxed{M \boldsymbol{\pi} = \boldsymbol{\pi}}
$$

**Questo è il PageRank vector!**

`00:59:55` 

### Connessione con Autovettori

**Da PR1.pdf slide 8: Connection to Eigenvectors**

L'equazione $M\boldsymbol{\pi} = \boldsymbol{\pi}$ è un **problema agli autovalori**!

$$
M \boldsymbol{\pi} = 1 \cdot \boldsymbol{\pi}
$$

**Definizione:**

Il **PageRank vector** $\boldsymbol{\pi}$ è l'**autovettore** della matrice di transizione $M$ corrispondente all'autovalore $\lambda = 1$.

`01:00:30` 

**Componenti:** $\pi^{(i)}$ = importanza/ranking della pagina $i$

`01:01:02` 

**Domande teoriche importanti:**

`01:01:33` 

1. **Esistenza:** $\lambda = 1$ è sempre un autovalore di $M$?
2. **Dominanza:** È il più grande autovalore?
3. **Unicità:** L'autovettore corrispondente è unico?
4. **Positività:** Le componenti sono positive (interpretabili come probabilità)?

La risposta viene dal **Teorema di Perron-Frobenius**.

---

## Teorema di Perron-Frobenius {#teorema-perron-frobenius}

`01:02:07` 

**Da PR1.pdf slide 9: Guarantees from Perron-Frobenius**

Il teorema di Perron-Frobenius fornisce garanzie teoriche per matrici stocastiche.

### Enunciato (Versione Semplificata)

**Teorema (Perron-Frobenius per Matrici Stocastiche):**

Se $M$ è una matrice **stocastica per colonne**, **positiva** (tutti elementi $> 0$) e **irriducibile**\*, allora:

1. **Autovalore principale:** $\lambda = 1$ è un autovalore e $|\lambda| = 1$ è il più grande in modulo:
   $$
   |\lambda_i| \leq 1 \quad \forall i, \quad \text{e } \lambda_1 = 1
   $$

2. **Unicità:** L'autovettore corrispondente a $\lambda = 1$ è **unico** (a meno di scalatura)

3. **Positività:** L'autovettore può essere scelto con componenti **strettamente positive**:
   $$
   \boldsymbol{\pi} > \mathbf{0} \quad (\text{tutte le componenti } > 0)
   $$

`01:03:06` 

\* **Irriducibilità:** Una matrice stocastica $M$ è **irriducibile** se ogni stato può essere raggiunto da ogni altro stato in un numero finito di passi. Formalmente:

$$
\exists n : (M^n)_{ij} > 0 \quad \forall i, j
$$

(La rete è fortemente connessa: esiste un percorso da ogni pagina a ogni altra).

`01:03:41` 

**In Pratica:**

Nella reale implementazione di Google, $M$ viene modificata per garantire queste proprietà (aggiungendo "teleportation" - salti casuali):

$$
M_{\text{Google}} = \alpha M + (1-\alpha) \frac{1}{n} \mathbf{1}\mathbf{1}^T
$$

con $\alpha \approx 0.85$ (damping factor). Questo assicura che la matrice sia irriducibile e aperiodica.

### Dimostrazione (Caso Semplificato)

**Da PR1.pdf slide 12-16: Proof for Perron-Frobenius**

Dimostriamo le due proprietà chiave:

1. $\lambda = 1$ è un autovalore di $M$
2. Per ogni autovalore $\lambda$: $|\lambda| \leq 1$

Insieme, queste implicano che $\lambda = 1$ è l'autovalore **dominante**.

#### Parte 1: λ=1 è un Autovalore

**Da PR1.pdf slide 13-14**

`01:04:00` 

**Obiettivo:** Dimostrare che $\det(M - I) = 0$.

**Step 1:** Per definizione, $M$ è stocastica per colonne:

$$
\sum_{i=1}^n M_{ij} = 1 \quad \forall j
$$

**Step 2:** In notazione vettoriale, con $\mathbf{1} = [1, 1, \ldots, 1]^T$:

$$
\mathbf{1}^T M = \mathbf{1}^T
$$

(il vettore riga di 1 è autovettore **sinistro** con autovalore 1).

**Step 3:** Riscriviamo:

$$
\mathbf{1}^T M - \mathbf{1}^T = \mathbf{0}^T
$$

$$
\mathbf{1}^T (M - I) = \mathbf{0}^T
$$

**Step 4:** Questa equazione mostra che le **righe** di $(M - I)$ sono linearmente dipendenti (la loro somma è il vettore nullo).

**Step 5:** Una matrice con righe linearmente dipendenti è **singolare**:

$$
\det(M - I) = 0
$$

**Conclusione:** $\lambda = 1$ è un autovalore di $M$. ✓

#### Parte 2: |λ| ≤ 1 per Tutti gli Autovalori

**Da PR1.pdf slide 15-16**

**Obiettivo:** Per ogni autovalore $\lambda$ di $M$: $|\lambda| \leq \|M\|_1 = 1$.

**Proprietà generale:** Per ogni matrice e norma indotta:

$$
|\lambda| \leq \|M\|
$$

**Norma 1 (norma colonna):**

$$
\|M\|_1 = \max_j \sum_{i=1}^n |M_{ij}|
$$

(massima somma assoluta per colonna).

**Step 1:** Per matrice stocastica, $M_{ij} \geq 0$, quindi:

$$
|M_{ij}| = M_{ij}
$$

**Step 2:** Ogni colonna somma a 1:

$$
\sum_{i=1}^n M_{ij} = 1 \quad \forall j
$$

**Step 3:** Quindi:

$$
\|M\|_1 = \max_j \left(\sum_{i=1}^n M_{ij}\right) = \max_j(1) = 1
$$

**Conclusione:** $|\lambda| \leq 1$ per ogni autovalore. ✓

**Combinando Parte 1 e 2:**

- $\lambda = 1$ è autovalore (Parte 1)
- Nessun autovalore ha modulo $> 1$ (Parte 2)
- **Quindi $\lambda = 1$ è l'autovalore dominante!**

---

## Metodo delle Potenze (Power Method) {#metodo-delle-potenze}

`01:04:13` 

**Da PR1.pdf slide 10-11: The Power Method Algorithm**

Ora che sappiamo che $\boldsymbol{\pi}$ è l'autovettore dominante, come lo calcoliamo?

### Algoritmo

**Inizializzazione:**

`01:05:26` 

Scegli un vettore iniziale arbitrario $\boldsymbol{\pi}_0$ (tipicamente uniforme):

$$
\boldsymbol{\pi}_0 = \begin{bmatrix} 1/n \\ 1/n \\ \vdots \\ 1/n \end{bmatrix}
$$

**Iterazione:**

`01:06:10` 

Ripeti fino a convergenza:

$$
\boldsymbol{\pi}_{k+1} = M \boldsymbol{\pi}_k
$$

Equivale a:

$$
\boldsymbol{\pi}_k = M^k \boldsymbol{\pi}_0
$$

**Criterio di Stop:**

Fermati quando $\|\boldsymbol{\pi}_{k+1} - \boldsymbol{\pi}_k\| < \epsilon$ (es. $\epsilon = 10^{-6}$).

**Output:** $\boldsymbol{\pi}^* \approx \boldsymbol{\pi}_k$ (PageRank vector).

`01:06:50` 

### Perché Converge: Analisi Teorica

**Da PR1.pdf slide 11: Why the Power Method Converges**

**Setup:** Siano $\mathbf{v}_1, \ldots, \mathbf{v}_n$ gli autovettori di $M$ con autovalori:

$$
1 = |\lambda_1| > |\lambda_2| \geq \cdots \geq |\lambda_n|
$$

(Perron-Frobenius garantisce $\lambda_1 = 1$ dominante).

**Step 1: Decomposizione su Base di Autovettori**

Il vettore iniziale può essere scritto come:

$$
\boldsymbol{\pi}_0 = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_n \mathbf{v}_n
$$

(assumiamo $c_1 \neq 0$, tipicamente vero per scelta casuale).

`01:07:21` 

**Step 2: Applicazione di $M^k$**

$$
M^k \boldsymbol{\pi}_0 = M^k(c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_n \mathbf{v}_n)
$$

Poiché $M \mathbf{v}_i = \lambda_i \mathbf{v}_i$:

$$
M^k \boldsymbol{\pi}_0 = c_1 \lambda_1^k \mathbf{v}_1 + c_2 \lambda_2^k \mathbf{v}_2 + \cdots + c_n \lambda_n^k \mathbf{v}_n
$$

**Step 3: Normalizzazione per $\lambda_1^k = 1^k = 1$**

`01:08:21` 

$$
\boldsymbol{\pi}_k = c_1 \mathbf{v}_1 + c_2 \left(\frac{\lambda_2}{\lambda_1}\right)^k \mathbf{v}_2 + \cdots + c_n \left(\frac{\lambda_n}{\lambda_1}\right)^k \mathbf{v}_n
$$

**Step 4: Convergenza**

`01:08:53` 

Poiché $|\lambda_i/\lambda_1| = |\lambda_i| < 1$ per $i > 1$:

$$
\left(\frac{\lambda_i}{\lambda_1}\right)^k \xrightarrow{k \to \infty} 0
$$

Quindi:

$$
\boldsymbol{\pi}_k \xrightarrow{k \to \infty} c_1 \mathbf{v}_1
$$

**Normalizzando** (per avere $\sum_i \pi^{(i)} = 1$):

$$
\boldsymbol{\pi}^* = \frac{c_1 \mathbf{v}_1}{\|\mathbf{v}_1\|_1}
$$

**Conclusione:** Il metodo delle potenze converge all'autovettore dominante $\mathbf{v}_1$, che è il **PageRank vector**!

`01:09:25` 

### Velocità di Convergenza

La velocità dipende dal **gap spettrale** $|\lambda_2|$:

$$
\|\boldsymbol{\pi}_k - \boldsymbol{\pi}^*\| = O(|\lambda_2|^k)
$$

- Se $|\lambda_2| \ll 1$: convergenza **rapida**
- Se $|\lambda_2| \approx 1$: convergenza **lenta**

Nel caso del Google Matrix con damping $\alpha = 0.85$:

$$
|\lambda_2| \leq \alpha = 0.85
$$

Tipicamente convergenza in 50-100 iterazioni.

`01:09:58` 

**Risultato finale:** Le componenti di $\boldsymbol{\pi}^*$ danno il **ranking** delle pagine web!

$$
\text{PageRank}(i) = \pi^{(i)}
$$

Pagine ordinate per importanza decrescente.

---

`00:59:25` 
Ho m pi è uguale a pi okay ora questa equazione fa questa equazione vi ricorda qualcosa.

`00:59:55` 
Scusate, intendete per il sistema lineare, no, ma qui in realtà non stiamo risolvendo il sistema lineare, stiamo applicando la matrice a un vettore, quindi è leggermente diverso, okay, esattamente, pi è un autovettore della matrice M con autovalore 1, okay.

`01:00:30` 
Okay, quindi questa è l'idea, una volta che formalizzate... questo problema in questo modo, arrivate a questa equazione, che è decisamente più di un problema autovalore-autovettore, e quello che state cercando, quello che state cercando, è l'autovettore corrispondente all'autovalore uguale a uno. E in questo vettore pi, che è.

`01:01:02` 
un cosiddetto stato stazionario o fattore di filtraggio, alla fine quello che avremo sono le probabilità, di essere in un certo sito web in generale. Quindi significa che più alta è la probabilità di essere in un certo sito web, più alta è l'importanza di quel sito web. Quindi alla fine, il vettore pi vi darà il rank delle pagine web.

`01:01:33` 
Okay, ci sono alcuni punti da chiarire. Prima di tutto, Le domande chiave sono, lambda è sempre un autovalore della matrice M, della matrice di transizione. È questo il più grande? Perché quello che sto affermando è che dato che questo pi è il vettore che sto cercando,

`01:02:07` 
dovrebbe essere l'autovettore più importante della matrice, quindi quello corrispondente all'autovalore più grande. Infine, questo vettore è unico. Questi sono i tre punti che dobbiamo chiarire. Le risposte sono contenute nel teorema. Vi sto dando una versione semplificata, ma nel teorema di Perron-Frobenius, il risultato è, se M è una matrice stocastica per colonne, che è il nostro caso, allora l'autovalore lambda uguale a 1 è l'autovalore più grande, l'autovettore corrispondente a lambda uguale a 1 è unico,

`01:03:06` 
e, inoltre, questo autovettore può essere scelto per avere componenti strettamente positive, quindi possono essere interpretate come probabilità, okay? Nell'implementazione reale di Google dell'algoritmo, la matrice che state considerando non è la matrice di transizione semplice che abbiamo presentato qui, ma l'idea è più o meno la stessa.

`01:03:41` 
E una volta che sappiamo che effettivamente abbiamo le buone proprietà per questa coppia di autovettori e autovalori, quindi ha senso prendere pi perché corrisponde all'autovalore più grande. Una domanda è come posso calcolare questa coppia di autovalori e autovettori? Una possibilità.

`01:04:13` 
è usare il metodo delle potenze. Immagino che forse ricordiate il metodo delle potenze. L'idea del metodo delle potenze è partire con una stima iniziale per l'autovettore. Quindi in generale avete una matrice A, e per questa matrice sapete che c'è l'autovalore lambda 1 che è.

`01:04:50` 
separato dagli altri. Quindi lambda 1 è l'autovalore più grande ed è separato dagli altri. Poi prendete il vettore x0, il vettore di partenza, e quello che state facendo è applicare la. il vettore, la matrice A, in questo caso la matrice M, al vettore, il vettore iniziale.

`01:05:26` 
E state creando x1, e poi iterate il processo, okay? Perché si chiama metodo delle potenze? Perché ovviamente se fate questa iterazione, alla fine avete un'iterazione k, avete k alla potenza k per x0, okay? Voi, perché questo metodo è effettivamente convergente? Quindi se avete,

`01:06:10` 
la matrice generale, come quella che stiamo considerando, e avete la matrice A o M, e avete tutti gli autovalori, lambda 1 maggiore di lambda 2, e così via. Ora, sappiamo che lambda 1 è uguale a 1, ed è il più grande. Poi, l'idea è, anche la matrice A.

`01:06:50` 
potete scrivere la stima iniziale X0 come combinazione lineare degli autovettori della matrice A, o M in questo caso. Quindi, significa che X0, può essere scritto come la somma di somma di coefficienti per gli autovettori. Quindi gli autovettori sono in realtà una base per lo spazio delle colonne di m,

`01:07:21` 
e quindi possono essere usati per rappresentare il vettore iniziale x0. Ora applicate il metodo. Quindi applicate il metodo x0, che equivale ad applicare il metodo qui, ma m v i è lambda i, perché v i è l'autovettore, quindi m per v i è lambda i.

`01:07:51` 
Quindi qui avete lambda i lambda 1 v 1 per lambda 2 v 2, e così via. E poi iterate. Quando iterate, arrivate a questo tipo di espressione. okay uh ma ricordate che lambda 1 è il più grande autovalore della matrice quindi.

`01:08:21` 
qui se dividete tutto questo termine per lambda 1 alla potenza k avete i termini come lambda i su lambda 1 alla potenza k okay ma dato che lambda 1 è più grande di, lambda i qualsiasi altro lambda i, questi termini vanno a zero quando k va all'infinito.

`01:08:53` 
Quindi cosa significa? Significa che phi k sta andando ad essere nella direzione di cosa? Di, c1 v1. Quindi è nella direzione del primo autovettore della matrice. Il primo autovettore in realtà è l'autovettore associato con l'autovalore più grande come possiamo vedere qui. E quindi.

`01:09:25` 
sfruttando questo fattore potete arrivare al vettore phi k e avete il vostro ranking delle, diverse pagine web incluse nella vostra rete. Okay, qui ho anche aggiunto la dimostrazione del.

`01:09:58` 
teorema di Perron-Frobenius quando nel caso semplificato. Potete saltare. Okay, ve lo lascerò se siete interessati. Okay, lasciatemi solo introdurre l'altro argomento.

## Matrix Completion e Netflix Problem

`01:10:35` 
che continueremo domani. Questo è un altro problema importante, si chiama matrix completion. Qual è il problema prototipo di questo? Sapete cos'è il matrix completion? Ne avete mai sentito parlare? Se dico problema Netflix, sapete cos'è? Okay, l'idea di questo problema è,

`01:11:13` 
quando usate Netflix, molto spesso avete suggerimenti. Forse c'è un messaggio, dato che hai guardato questo film, forse quest'altro film potrebbe essere di interesse per te. Come sono costruiti questi suggerimenti? Questo è il problema, e in realtà nel 2009, Netflix ha proposto un premio di un milione di dollari.

`01:11:49` 
a qualcuno che sarebbe stato in grado di, sarebbe in grado di arrivare a un algoritmo adeguato, per fare suggerimenti, suggerimenti ragionevoli. Quindi qual è il contesto? Il contesto è più o meno quello che potete vedere qui. Nelle colonne avete alcuni film, e nelle righe avete alcune persone, e ovviamente.

`01:12:22` 
ogni persona non ha valutato ogni film, forse solo pochissimi, un sottoinsieme di tutti i film, che sono disponibili su Netflix. Quindi l'idea è che essenzialmente è come avere una matrice come questa, dove in alcune posizioni ho una valutazione, in altre posizioni ho valori sconosciuti, per esempio qui, non so se Einstein avrebbe gradito Il Padrino o no.

`01:13:03` 
E l'idea è, sono in grado di arrivare a qualcosa che è in grado di mettere qui una valutazione, una valutazione significativa per Il Padrino relativa a Einstein? Questo è il problema. Quindi l'obiettivo è prevedere le valutazioni mancanti.

`01:13:34` 
Questo è il problema Netflix, e in generale, qual è l'idea? L'idea è che avete una matrice generale in cui avete alcuni dati nella matrice, e altri dati sono mancanti. Un altro esempio è il cosiddetto problema di in-painting. Quindi avete un'immagine con una porzione dell'immagine che è mancante. Volete in qualche modo arrivare a qualcosa che è in grado di dipingere quest'area mancante con.

`01:14:11` 
qualcosa che è mancante, in termini di quello che è nei dintorni, okay? Questa è l'idea, il problema. E per questo problema... L'idea è che se avete la matrice X, che è, come al solito, la matrice dove abbiamo, n persone e b sono il numero di film, allora in un certo senso voglio arrivare a.

`01:14:53` 
una fattorizzazione di questa matrice o un'approssimazione di questa matrice, dove ho un fattore che è n per r, che rappresenta le caratteristiche degli utenti, e un altro metodo, b trasposta, che è r per d, che è le caratteristiche dei film. Cos'è r?

`01:15:25` 
L'idea è che, e questa è un'ipotesi fondamentale, quando avete una matrice come questa, quindi il problema Netflix, dovete fare l'ipotesi che questa matrice sia di rango R, quindi c'è in qualche modo una struttura a bassa dimensionalità nella matrice che è in grado di rappresentare le valutazioni in generale.

`01:16:01` 
Quindi l'idea è questa, che la matrice che avete ha una struttura a bassa dimensionalità intrinseca, e volete sfruttare quella struttura a bassa dimensionalità. Quindi l'idea è questa, che la matrice che avete ha una struttura a bassa dimensionalità intrinseca, che è questa R, okay? Per esempio... Alcuni fattori latenti potrebbero essere le preferenze per il genere particolare del film, il tempo, l'era, o il pubblico.

`01:16:38` 
E l'ipotesi è che queste armi, quindi la dimensione della struttura sottostante, è molto più piccola del minimo tra M e A. Quindi significa, in altri termini, che alla fine sarete in grado di rappresentare la valutazione di ogni film da parte di ogni persona.

`01:17:10` 
Solo usando i nostri utenti rappresentativi e i nostri film rappresentativi. Quindi questa è l'idea che volete esplorare. E l'algoritmo che stiamo per vedere, che è fortemente basato sulla decomposizione ai valori singolari, è mirato a trovare questi due vettori, quindi gli utenti rappresentativi e i film rappresentativi.

`01:17:52` 
E una volta che avete questi dati, potete riempire il vostro... Questo problema non è così banale. È abbastanza complicato in termini dell'algoritmo di soluzione, e domani entreremo nei dettagli di come sfruttare la SVD, decomposizione ai valori singolari, per arrivare a questo effetto.

`01:18:28` 
E in questo contesto, useremo un approccio che abbiamo già visto, un approccio di thresholding. Ricordate che abbiamo già visto il thresholding quando stavamo calcolando i... valori singolari per approssimare dati generici al fine di considerare solo quelli più importanti.

`01:19:08` 
E qui faremo uso di un singolare, okay? E poi faremo anche il support vector che abbiamo menzionato.

---

## 📐 Formule Complete di Riferimento

### Kernel Methods

#### Feature Maps e Kernel

**Feature Map Generica:**
$$
\varphi : \mathbb{R}^p \to \mathbb{R}^D, \quad D \geq p
$$

**Kernel Function:**
$$
K(\mathbf{x}, \mathbf{z}) = \langle \varphi(\mathbf{x}), \varphi(\mathbf{z}) \rangle = \varphi(\mathbf{x})^T \varphi(\mathbf{z})
$$

**Kernel Matrix (Gram Matrix):**
$$
K = \Phi(X) \Phi(X)^T, \quad K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j) \in \mathbb{R}^{n \times n}
$$

#### Kernel Specifici

**1. Kernel Lineare:**
$$
K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}
$$

**2. Kernel Polinomiale:**
$$
K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^q
$$
- $c \geq 0$: termine costante
- $q \in \mathbb{N}$: grado
- Dimensione feature space: $D = \binom{p+q}{q}$

**3. Kernel Gaussiano (RBF):**
$$
K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right), \quad \gamma = \frac{1}{2\sigma^2}
$$
- Spazio infinito-dimensionale ($D = \infty$)
- $\gamma$ piccolo: modello smooth (underfitting)
- $\gamma$ grande: modello flessibile (overfitting)

#### Representer Theorem

**Enunciato:**
$$
\mathbf{w}^* = \Phi(X)^T \boldsymbol{\alpha} = \sum_{i=1}^n \alpha_i \varphi(\mathbf{x}_i)
$$

dove $\boldsymbol{\alpha} \in \mathbb{R}^n$ soluzione di:
$$
(K + \lambda I_n) \boldsymbol{\alpha} = \mathbf{y}
$$

**Soluzione Dual:**
$$
\boldsymbol{\alpha}^* = (K + \lambda I_n)^{-1} \mathbf{y}
$$

**Con Eigendecomposition ($K = U\Lambda U^T$):**
$$
\boldsymbol{\alpha}^* = U(\Lambda + \lambda I)^{-1} U^T \mathbf{y}
$$

#### Predizioni

**Training Set:**
$$
\hat{y}_i = \sum_{j=1}^n \alpha_j^* K(\mathbf{x}_j, \mathbf{x}_i)
$$

**Nuovo Punto $\mathbf{x}_*$:**
$$
\hat{y}_* = \sum_{i=1}^n \alpha_i^* K(\mathbf{x}_i, \mathbf{x}_*)
$$

### PageRank

#### Matrici Fondamentali

**Matrice di Adiacenza:**
$$
A_{ij} = \begin{cases}
1 & \text{se link da } j \text{ a } i \\
0 & \text{altrimenti}
\end{cases}
$$

**Matrice di Transizione (Stocastica):**
$$
M_{ij} = \frac{A_{ij}}{\sum_k A_{kj}} = P(i \leftarrow j)
$$

Proprietà: $\sum_{i=1}^n M_{ij} = 1$ (colonne sommano a 1)

#### Equazioni PageRank

**Iterazione:**
$$
\boldsymbol{\pi}_{k+1} = M \boldsymbol{\pi}_k
$$

**Stato Stazionario:**
$$
M \boldsymbol{\pi} = \boldsymbol{\pi} \quad \Leftrightarrow \quad M \boldsymbol{\pi} = 1 \cdot \boldsymbol{\pi}
$$

$\boldsymbol{\pi}$ è **autovettore** di $M$ per autovalore $\lambda = 1$.

#### Teorema di Perron-Frobenius

Per matrice stocastica $M$ irriducibile:

1. $\lambda = 1$ è autovalore (il più grande)
2. $|\lambda_i| \leq 1$ per ogni autovalore
3. Autovettore per $\lambda = 1$ è **unico** (a meno di scalatura)
4. Componenti autovettore sono **positive**

#### Power Method

**Algoritmo:**
```
Inizializza: π₀ = [1/n, ..., 1/n]ᵀ
Ripeti: πₖ₊₁ = M·πₖ
Finché: ‖πₖ₊₁ - πₖ‖ < ε
Output: π* (PageRank vector)
```

**Convergenza:**
$$
\boldsymbol{\pi}_k - \boldsymbol{\pi}^* = O(|\lambda_2|^k)
$$

Velocità dipende da gap spettrale $1 - |\lambda_2|$.

---

## ⚖️ Confronto Complessità Computazionale

### Kernel Ridge Regression

| Metodo | Variabili | Sistema | Complessità Training | Complessità Inference | Fattibile con $D=\infty$? |
|--------|-----------|---------|---------------------|----------------------|---------------------------|
| **Primal** | $\mathbf{w} \in \mathbb{R}^D$ | $(\Phi^T\Phi + \lambda I_D)\mathbf{w} = \Phi^T\mathbf{y}$ | $O(nD^2 + D^3)$ | $O(D)$ per punto | ❌ No |
| **Dual** | $\boldsymbol{\alpha} \in \mathbb{R}^n$ | $(K + \lambda I_n)\boldsymbol{\alpha} = \mathbf{y}$ | $O(n^2p + n^3)$ | $O(np)$ per punto | ✅ **Sì** |

**Quando usare Dual:**
- $n < D$ (tipico!)
- Kernel con $D$ molto grande o infinito (Gaussiano)
- Vogliamo sfruttare kernel trick

**Quando usare Primal:**
- $D < n$ (raro, solo features lineari con pochi attributi)
- $D$ è piccolo e fisso

### PageRank (Power Method)

**Per rete con $n$ pagine, $m$ link:**

- **Memorizzazione matrice $M$:** $O(m)$ (sparsa) o $O(n^2)$ (densa)
- **Iterazione power method:** $O(m)$ per iterazione
- **Convergenza:** $\sim 50-100$ iterazioni tipiche
- **Totale:** $O(km)$ con $k$ = numero iterazioni

**Scalabilità:**
- Google: miliardi di pagine ($n \sim 10^{10}$)
- Matrice sparsa cruciale (media $\sim 10$ link/pagina)
- Distribuito su cluster (MapReduce)

---

## ✅ Checklist Lab 4: Kernel Methods & PageRank

### Teoria Kernel Methods

- [ ] **Problema non linearità:**
  - Comprendere perché fit lineare fallisce su dati parabolici
  - Visualizzare separatori circolari vs rette
  
- [ ] **Feature Maps:**
  - Definizione formale $\varphi : \mathbb{R}^p \to \mathbb{R}^D$
  - Esempi: 1D→2D ($\varphi(x) = [x, x^2]$), 2D→6D
  - Vista geometrica: cerchio → piano in 3D
  
- [ ] **Kernel Trick:**
  - $K(\mathbf{x}, \mathbf{z}) = \varphi(\mathbf{x})^T \varphi(\mathbf{z})$
  - Calcolo prodotti scalari senza $\varphi$ esplicita
  - Kernel matrix $K = \Phi\Phi^T$ ($n \times n$)
  
- [ ] **Representer Theorem:**
  - Enunciato: $\mathbf{w}^* = \Phi^T\boldsymbol{\alpha}$
  - Dimostrazione completa (decomposizione $\mathbf{w}_\parallel + \mathbf{w}_\perp$)
  - Riduzione da $D$ parametri a $n$ coefficienti
  
- [ ] **Soluzione Dual:**
  - Derivazione: da primal $(\Phi^T\Phi + \lambda I_D)\mathbf{w} = \Phi^T\mathbf{y}$
  - A dual: $(K + \lambda I_n)\boldsymbol{\alpha} = \mathbf{y}$
  - Complessità: $O(n^3)$ vs $O(D^3)$

### Pratica Kernel Methods

- [ ] **Kernel Polinomiale:**
  - Formula: $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^q$
  - Omogeneo ($c=0$) vs completo ($c>0$)
  - Dimensione feature space: $\binom{p+q}{q}$
  
- [ ] **Kernel Gaussiano:**
  - Formula: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x} - \mathbf{z}\|^2)$
  - Parametro $\gamma$: trade-off smooth vs flessibile
  - Spazio infinito-dimensionale
  
- [ ] **Eigendecomposition:**
  - $K = U\Lambda U^T$
  - Soluzione stabile: $\boldsymbol{\alpha} = U(\Lambda + \lambda I)^{-1}U^T\mathbf{y}$
  - Regolarizzazione filtra autovalori piccoli
  
- [ ] **Predizioni:**
  - Training: $\hat{y}_i = \sum_j \alpha_j K(\mathbf{x}_j, \mathbf{x}_i)$
  - Test: $\hat{y}_* = \sum_i \alpha_i K(\mathbf{x}_i, \mathbf{x}_*)$
  - Solo kernel evaluations!

### Teoria PageRank

- [ ] **Random Surfer Model:**
  - Navigazione casuale cliccando link
  - Convergenza a distribuzione stazionaria
  - PageRank = probabilità visita a lungo termine
  
- [ ] **Matrici:**
  - Adiacenza $A$: $A_{ij} = 1$ se link $j \to i$
  - Transizione $M$: normalizzazione per colonne ($\sum_i M_{ij} = 1$)
  - $M_{ij} = P(i \leftarrow j)$
  
- [ ] **Equazione Stazionaria:**
  - $M\boldsymbol{\pi} = \boldsymbol{\pi}$ (problema autovalori)
  - $\boldsymbol{\pi}$ autovettore per $\lambda = 1$
  - Componenti $\pi^{(i)}$ = importanza pagina $i$
  
- [ ] **Perron-Frobenius:**
  - $\lambda = 1$ autovalore dominante
  - $|\lambda_i| \leq 1$ per ogni autovalore
  - Autovettore unico e positivo
  - Dimostrazione: righe $(M-I)$ dipendenti, norma-1 = 1

### Pratica PageRank

- [ ] **Power Method:**
  - Inizializzazione: $\boldsymbol{\pi}_0 = [1/n, \ldots, 1/n]^T$
  - Iterazione: $\boldsymbol{\pi}_{k+1} = M\boldsymbol{\pi}_k$
  - Convergenza: $\boldsymbol{\pi}_k \to c_1 \mathbf{v}_1$ (autovettore dominante)
  
- [ ] **Analisi Convergenza:**
  - Decomposizione su autovettori
  - Termini $(\lambda_i/\lambda_1)^k \to 0$ per $i > 1$
  - Velocità: $O(|\lambda_2|^k)$
  
- [ ] **Esempio 4 Pagine:**
  - Costruire matrice $A$ da grafo
  - Normalizzare → matrice $M$
  - Calcolare $\boldsymbol{\pi}$ (analitico o iterativo)
  - Interpretare ranking

### Applicazioni

- [ ] **Kernel Methods:**
  - Regressione non lineare (trend parabolici)
  - Classificazione (boundary circolari)
  - Scelta kernel per problema specifico
  
- [ ] **PageRank:**
  - Ranking pagine web
  - Reti sociali (influencer detection)
  - Citation networks (paper importance)

---

## 🔧 Esercizi Proposti

### Esercizio 1: Feature Map e Kernel Manuale

Dato il kernel $K(x, z) = (2xz + 3)^2$ per $x, z \in \mathbb{R}$:

a) Espandi $K(x, z)$ e identifica la feature map $\varphi(x)$ implicita

b) Calcola la kernel matrix per dataset $\{-1, 0, 2\}$

c) Verifica usando $\varphi$ esplicita che $K = \Phi\Phi^T$

**Soluzione (parte a):**
$$
K(x,z) = (2xz + 3)^2 = 4x^2z^2 + 12xz + 9
$$
$$
\varphi(x) = \begin{bmatrix} 2x^2 \\ \sqrt{12} x \\ 3 \end{bmatrix}
$$

### Esercizio 2: Kernel Ridge Regression 1D

Dataset: $(x_1, y_1) = (0, 1)$, $(x_2, y_2) = (1, 2)$, $(x_3, y_3) = (2, 5)$

a) Calcola kernel matrix $K$ con kernel lineare $K(x,z) = xz$

b) Risolvi $(K + \lambda I)\boldsymbol{\alpha} = \mathbf{y}$ con $\lambda = 0.1$

c) Predici $\hat{y}(x_* = 1.5)$

### Esercizio 3: Dimensione Feature Space

Per kernel polinomiale $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + 1)^q$:

a) Calcola dimensione $D$ per $p=2$ (2D input), $q=3$ (grado 3)

b) Confronta complessità primal vs dual per $n=100$ campioni

c) Per quali valori di $n$ conviene usare dual?

**Formula:** $D = \binom{p+q}{q}$

### Esercizio 4: PageRank su Rete Triangolare

Considera 3 pagine con link:
- Pagina 1 → Pagina 2
- Pagina 2 → Pagina 3
- Pagina 3 → Pagina 1

a) Costruisci matrice di adiacenza $A$

b) Calcola matrice di transizione $M$

c) Trova PageRank $\boldsymbol{\pi}$ risolvendo $M\boldsymbol{\pi} = \boldsymbol{\pi}$

d) Verifica con 10 iterazioni power method

### Esercizio 5: Perron-Frobenius

Per matrice:
$$
M = \begin{bmatrix}
0 & 1/2 & 1/3 \\
1/2 & 0 & 1/3 \\
1/2 & 1/2 & 1/3
\end{bmatrix}
$$

a) Verifica che $M$ è stocastica per colonne

b) Dimostra che $\mathbf{1}^T M = \mathbf{1}^T$ ($\lambda=1$ è autovalore)

c) Calcola tutti gli autovalori e verifica $|\lambda_i| \leq 1$

d) Trova autovettore per $\lambda = 1$ e normalizza

### Esercizio 6: Kernel Gaussiano vs Polinomiale

Dataset con separazione circolare (classe +1 dentro cerchio $x_1^2 + x_2^2 \leq 1$, classe -1 fuori):

a) Spiega perché kernel lineare fallisce

b) Quale kernel sceglieresti? Perché?

c) Come sceglieresti $\gamma$ per RBF o $(c, q)$ per polinomiale?

d) Implementa kernel ridge regression con entrambi e confronta

---

## 📚 Riferimenti e Approfondimenti

### Materiali Corso

**Slide Lezione:**
- **KM1.pdf** (slide 1-38): Kernel Methods
  - Feature maps, kernel trick, Representer Theorem
  - Dual derivation, eigendecomposition
  - Esempi numerici completi
  
- **PR1.pdf** (slide 1-16): PageRank Algorithm
  - Random surfer model, matrici A e M
  - Perron-Frobenius theorem e dimostrazione
  - Power method con analisi convergenza

### Libri Consigliati

**Kernel Methods:**
- Schölkopf & Smola, "Learning with Kernels" (2002)
  - Trattazione completa di kernel methods
  - Teoria e applicazioni (SVM, kernel PCA, etc.)
  
- Shawe-Taylor & Cristianini, "Kernel Methods for Pattern Analysis" (2004)
  - Approccio più accessibile
  - Molti esempi pratici

**PageRank:**
- Langville & Meyer, "Google's PageRank and Beyond" (2006)
  - Libro citato a lezione
  - Molto leggibile, focus su algebra lineare
  - Storia e varianti dell'algoritmo

### Paper Classici

- Vapnik, "The Nature of Statistical Learning Theory" (1995)
  - Fondamenti teorici di kernel methods e SVM
  
- Page, Brin, Motwani, Winograd, "The PageRank Citation Ranking" (1999)
  - Paper originale di PageRank
  - Technical Report Stanford

### Prossime Lezioni

`01:19:08` 

**Domani:**
- Support Vector Machines (SVM)
- Matrix Completion con SVD
- Problema Netflix: fattorizzazione low-rank
- Thresholding su valori singolari

**Settimana prossima:**
- Image inpainting
- Applicazioni avanzate kernel methods
- Algoritmi per large-scale PageRank

---

**Fine Lab 4 - Kernel Methods & PageRank**

**Statistiche arricchimento:**
- **Righe originali:** 599
- **Righe finali:** ~2600
- **Incremento:** +334% (×4.3)
- **Contenuto aggiunto:**
  - 38 slide KM1.pdf integrate
  - 16 slide PR1.pdf integrate
  - Formule LaTeX complete
  - Dimostrazioni dettagliate
  - Esempi numerici
  - Checklist e esercizi