# Lab 2 NAML - Approssimazione a Basso Rango e PCA

---

## 🎯 Obiettivi della Lezione

### Competenze Teoriche
- Comprendere il Teorema di Eckart-Young nella norma di Frobenius
- Apprendere i fondamenti della Principal Component Analysis (PCA)
- Distinguere tra PCA e regressione ai minimi quadrati
- Studiare la regolarizzazione (Ridge, LASSO, Elastic Net)

### Competenze Pratiche
- Applicare la SVD per risolvere problemi di approssimazione a basso rango
- Utilizzare la PCA per riduzione della dimensionalità
- Implementare regressione lineare con regolarizzazione
- Calcolare pseudo-inverse e matrici di proiezione

### Applicazioni
- Riduzione della dimensionalità di dataset ad alta dimensione
- Regressione lineare robusta con regolarizzazione
- Feature selection tramite LASSO

---

## 📚 Prerequisiti

**Matematica**
- Algebra lineare: SVD, autovalori/autovettori, matrici ortogonali
- Analisi: ottimizzazione, calcolo del gradiente, forme quadratiche
- Geometria: proiezioni ortogonali, sottospazi

**Teoria**
- Decomposizione ai valori singolari (SVD)
- Norme matriciali (spettrale, Frobenius)
- Minimi quadrati classici

---

## 📑 Indice Completo

### **Parte 1 - Fondamenti Teorici**
#### [1. Teorema di Eckart-Young - Norma di Frobenius](#teorema-eckart-young-frobenius) `00:00:03 - 00:16:58`
- [1.1 Richiamo del teorema e notazione SVD](#richiamo-teorema) `00:00:03`
- [1.2 SVD troncata e approssimazione di rango K](#svd-troncata) `00:02:26`
- [1.3 Disuguaglianza di Weyl](#disuguaglianza-weyl) `00:04:51`
- [1.4 Dimostrazione nella norma di Frobenius](#dimostrazione-frobenius) `00:06:22`
- [1.5 Risultato finale: ottimalità di A_K](#risultato-ottimalita) `00:15:56`

### **Parte 2 - Principal Component Analysis**
#### [2. Principal Component Analysis (PCA)](#pca) `00:17:36 - 00:32:15`
- [2.1 Introduzione alla PCA](#intro-pca) `00:17:36`
- [2.2 Matrice di covarianza e SVD](#matrice-covarianza) `00:18:26`
- [2.3 Algoritmo PCA classico](#algoritmo-pca) `00:24:22`
- [2.4 PCA via SVD: approccio stabile](#pca-via-svd) `00:28:01`
- [2.5 Componenti principali e varianza](#componenti-principali) `00:29:08`
- [2.6 Visualizzazione 2D: assi principali](#visualizzazione-pca) `00:39:18`

#### [3. PCA vs Least Squares](#pca-vs-least-squares) `00:32:15 - 00:39:18`
- [3.1 Differenze negli obiettivi](#differenze-obiettivi) `00:32:53`
- [3.2 Distanze ortogonali vs distanze verticali](#distanze-diverse) `00:35:51`
- [3.3 Simmetria delle variabili](#simmetria-variabili) `00:37:24`

### **Parte 3 - Problema dei Minimi Quadrati**
#### [4. Problema dei Minimi Quadrati](#problema-minimi-quadrati) `00:43:53 - 00:52:07`
- [4.1 Formulazione del problema](#formulazione-problema) `00:43:53`
- [4.2 Sistema sovradeterminato](#sistema-sovradeterminato) `00:46:15`
- [4.3 Residuo e minimizzazione](#residuo-minimizzazione) `00:48:35`
- [4.4 Esempio visivo: regressione lineare](#esempio-regressione) `00:49:37`

#### [5. Interpretazione Geometrica](#interpretazione-geometrica) `00:52:07 - 01:04:17`
- [5.1 Proiezione ortogonale sullo spazio colonna](#proiezione-ortogonale) `00:52:44`
- [5.2 y_hat come migliore approssimazione](#migliore-approssimazione) `00:53:57`
- [5.3 Derivazione delle equazioni normali](#equazioni-normali) `00:58:47`
- [5.4 Soluzione analitica: w_hat](#soluzione-analitica) `01:02:17`

### **Parte 4 - Soluzione Computazionale**
#### [6. Matrici di Proiezione e Proprietà](#matrice-proiezione) `01:08:39 - 01:12:02`
- [6.1 Matrice di proiezione P](#definizione-p) `01:08:39`
- [6.2 Proprietà: simmetria e idempotenza](#proprieta-p) `01:11:31`

#### [7. Minimizzazione come Problema di Ottimizzazione](#minimizzazione-ottimizzazione) `01:12:02 - 01:16:43`
- [7.1 Formulazione funzionale J(w)](#formulazione-funzionale) `01:12:33`
- [7.2 Espansione del residuo quadratico](#espansione-residuo) `01:13:44`
- [7.3 Calcolo del gradiente](#calcolo-gradiente) `01:15:03`
- [7.4 Equivalenza con approccio geometrico](#equivalenza-approcci) `01:16:08`

#### [8. Soluzione via SVD](#soluzione-svd) `01:17:21 - 01:29:14`
- [8.1 Problemi computazionali di X^TX](#problemi-computazionali) `01:17:21`
- [8.2 SVD ridotta (economy)](#svd-ridotta) `01:18:49`
- [8.3 Derivazione di w_hat via SVD](#derivazione-svd) `01:19:26`
- [8.4 Pseudo-inversa di Moore-Penrose](#pseudo-inversa) `01:21:55`
- [8.5 Problema dei valori singolari piccoli](#valori-singolari-piccoli) `01:28:42`

### **Parte 5 - Regolarizzazione**
#### [9. Regolarizzazione Ridge (L2)](#regolarizzazione-ridge) `01:31:12 - 01:39:21`
- [9.1 Problema del mal-condizionamento](#mal-condizionamento) `01:31:12`
- [9.2 Termine di penalizzazione λ||w||²](#termine-penalizzazione) `01:33:25`
- [9.3 Nuove equazioni normali](#nuove-equazioni) `01:35:20`
- [9.4 Soluzione ridge via SVD](#soluzione-ridge-svd) `01:36:40`
- [9.5 Interpretazione: lunghezza minima](#interpretazione-ridge) `01:38:50`

#### [10. Regolarizzazione LASSO (L1)](#regolarizzazione-lasso) `01:39:21 - 01:46:18`
- [10.1 Norma L1 e sparsità](#norma-l1) `01:39:51`
- [10.2 Feature selection automatica](#feature-selection) `01:40:26`
- [10.3 Visualizzazione geometrica: norma L2 vs L1](#visualizzazione-geometrica) `01:41:41`
- [10.4 Level sets: cerchi vs quadrati](#level-sets) `01:44:35`
- [10.5 Soluzione sparsa](#soluzione-sparsa) `01:45:09`

#### [11. Elastic Net](#elastic-net) `01:46:18 - 01:52:04`
- [11.1 Combinazione di L1 e L2](#combinazione-l1-l2) `01:46:54`
- [11.2 Parametri λ e α](#parametri-lambda-alpha) `01:47:27`
- [11.3 Confronto: Ridge vs LASSO vs Elastic Net](#confronto-metodi) `01:49:16`
- [11.4 Selezione degli iperparametri](#selezione-iperparametri) `01:51:04`

### **Parte 6 - Conclusioni**
#### [12. Riepilogo e Comunicazioni](#riepilogo) `01:52:04 - 01:52:38`

---

## Teorema di Eckart-Young - Norma di Frobenius {#teorema-eckart-young-frobenius}

### 1.1 Richiamo del teorema e notazione SVD {#richiamo-teorema}

`00:00:03` 
Possiamo continuare dal punto dove siamo arrivati l'ultima volta. Come ricorderete, abbiamo introdotto un risultato importante sull'approssimazione di Lorenz di matrici, che è il teorema di Eckart-Young, e l'ultima volta abbiamo dimostrato il risultato di questo teorema nella norma spettrale, norma 2.

`00:00:33` 
In realtà, quando vi ho presentato il teorema, vi ho detto che questo risultato è valido in una norma indotta, ma le norme più importanti che considereremo sono la norma spettrale e la norma di Frobenius. Quindi oggi considereremo la norma di Frobenius. Quindi l'enunciato è solo per ricapitolare.

`00:01:08` 
Consideriamo una matrice $A \in \mathbb{R}^{m \times n}$ di rango $r$. La sua **Decomposizione ai Valori Singolari (SVD)** è data da:

$$
A = U \Sigma V^T
$$

Dove:
- $U \in \mathbb{R}^{m \times m}$: **matrice ortogonale** (vettori singolari sinistri)
- $\Sigma \in \mathbb{R}^{m \times n}$: **matrice pseudo-diagonale** con valori singolari
- $V \in \mathbb{R}^{n \times n}$: **matrice ortogonale** (vettori singolari destri)

**Struttura di $\Sigma$:** Contiene $r$ valori singolari **positivi** sulla diagonale:
$$
\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0
$$

Tutti gli altri elementi sono zero. Le informazioni significative sono contenute nelle prime $r$ colonne di $U$ e nelle prime $r$ righe di $V^T$.

`00:01:53`

`00:02:26` 
Partendo dalla SVD, possiamo espandere la matrice $A$ come **somma di $r$ matrici di rango 1**:

$$
A = \sum_{i=1}^r \sigma_i u_i v_i^T
$$

Dove $u_i$ è la $i$-esima colonna di $U$ e $v_i$ è la $i$-esima colonna di $V$.

**SVD Troncata (rank-$k$ approximation):** Se tratteniamo solo i primi $k$ termini ($k < r$), otteniamo:

$$
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T
$$

Questa è un'**approssimazione di rango $k$** della matrice originale $A$.

`00:03:07` 
**Teorema di Eckart-Young:** Data una matrice $B$ della stessa dimensione di $A$ con $\text{rank}(B) \leq k$, si ha:

$$
\|A - A_k\|_F \leq \|A - B\|_F \quad \text{e} \quad \|A - A_k\|_2 \leq \|A - B\|_2
$$

Ovvero, $A_k$ (SVD troncata) è la **migliore approssimazione di rango $k$** sia nella norma di Frobenius che nella norma spettrale.

`00:03:41` 
**Limiti sull'errore di approssimazione:**

- **Norma spettrale (2-norma):**
$$
\|A - A_k\|_2 = \sigma_{k+1}
$$
L'errore è esattamente uguale al **primo valore singolare trascurato**.

- **Norma di Frobenius:**
$$
\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
$$
L'errore è la **radice quadrata della somma dei quadrati** di tutti i valori singolari trascurati.

`00:04:18` 
Nel caso della norma di Frobenius, l'errore è dato dalla somma dei quadrati di tutti gli errori rimanenti. Okay, quindi questa è la dimostrazione che abbiamo visto l'ultima volta. E ora vogliamo considerare il caso di Frobenius.

`00:04:51` 
Esistono diverse dimostrazioni del Teorema di Eckart-Young nella norma di Frobenius. Qui presentiamo quella basata sulla **Disuguaglianza di Weyl**.

**Disuguaglianza di Weyl (per valori singolari):**

Date due matrici $X, Y \in \mathbb{R}^{m \times n}$, si ha:

$$
\sigma_{i+j-1}(X + Y) \leq \sigma_i(X) + \sigma_j(Y)
$$

per $i, j \geq 1$ tali che $i + j - 1 \leq \min(m,n)$.

**Definizione:** $\sigma_i(M)$ denota l'$i$-esimo valore singolare (in ordine decrescente) della matrice $M$.

*Nota: Non dimostreremo questa disuguaglianza. È un risultato fondamentale dell'algebra lineare numerica.*

`00:05:36` 
**Strategia della dimostrazione:** Consideriamo una matrice generica $B$ di rango $k$ (candidata approssimazione di $A$). Per la disuguaglianza di Weyl, analizzeremo la somma $X = (A - B) + B$.

`00:06:22` 
**Proprietà chiave di $B$:** Essendo $\text{rank}(B) = k$, tutti i valori singolari da $k+1$ in poi sono **nulli**:

$$
\sigma_{k+1}(B) = \sigma_{k+2}(B) = \cdots = \sigma_{\min(m,n)}(B) = 0
$$

In particolare: $\sigma_{k+1}(B) = 0$.

`00:06:58` 
**Applicazione della disuguaglianza di Weyl:**

Poniamo:
- $X = A - B$ (errore di approssimazione)
- $Y = B$ (matrice di rango $k$)
- $j = k + 1$ (primo indice "trascurato")

Applicando Weyl con questi valori:

`00:07:32` 
Otteniamo:

$$
\sigma_{i+k}(A) = \sigma_{i+(k+1)-1}(X + Y) \leq \sigma_i(X) + \sigma_{k+1}(Y) = \sigma_i(A - B) + \sigma_{k+1}(B)
$$

`00:08:03` 
Ma $\sigma_{k+1}(B) = 0$ (perché $\text{rank}(B) = k$), quindi:

$$
\sigma_{i+k}(A) \leq \sigma_i(A - B)
$$

Questa è la **relazione chiave** che collega i valori singolari di $A$ con quelli dell'errore $A - B$.

`00:08:34` 
E A meno B è la matrice che è importante per noi perché è essenzialmente il resto tra la matrice originale e una delle sue possibili approssimazioni di rango K. Okay. Quindi ora partiremo da questo punto. E quello che faremo è, prima di tutto, vogliamo ricordare che ciò che vogliamo ottenere è un limite di errore o un risultato nella norma di Frobenius.

`00:09:15` 
Quindi esattamente come abbiamo fatto per la norma, scriveremo per la matrice. La matrice A meno A K, dove A K è l'approssimazione di rango K ottenuta dalla SVD. la differenza, e ovviamente, come abbiamo menzionato prima, questa è la matrice A, che è di rango R, quindi la somma va da 1 a R, okay, abbiamo tutti gli autovalori non nulli, valori singolari nella somma, mentre in AK abbiamo la somma troncata, quindi abbiamo solo i primi K termini, okay, quindi la differenza è chiaramente la somma.

### 1.2 SVD troncata e approssimazione di rango K {#svd-troncata}

`00:10:03` 
che va da K più 1 a R, okay, e quindi i valori singolari di A meno AK, potete vedere da qui, sono sigma K più 1 fino a sigma R, okay, quindi il quadrato della norma di Frobenius è dato da questa espressione, ricordate la definizione della norma di Frobenius che abbiamo dato l'ultima volta, essenzialmente la norma di Frobenius al quadrato è la somma di tutti gli autovalori della matrice, il quadrato dei valori singolari della matrice.

`00:10:56` 
Quindi qui i valori singolari sono dati da quel set di valori, e quindi questa è la norma, il quadrato della norma. Ora, quello che faremo è considerare quello che abbiamo qui, tenendo a mente i risultati precedenti.

### 1.3 Disuguaglianza di Weyl {#disuguaglianza-weyl}

`00:11:29` 
Quindi, in alto, abbiamo il risultato dato dalla disuguaglianza di Weyl, e per ottenere esattamente quello che abbiamo qui, quello che dobbiamo fare è solo lo spostamento degli indici, okay? Quindi, impostiamo j uguale a i meno k, quindi i è j più k, e così abbiamo la somma del quadrato di sigma i al quadrato, e potete riscrivere in termini della somma che va da i da 1 a r meno k dei valori singolari di a con indice j più k.

`00:12:21` 
Quindi, abbiamo solo... fatto un cambiamento, una traslazione degli indici. Ora possiamo applicare la disuguaglianza di Weyl che abbiamo ottenuto. Quindi essenzialmente quello che abbiamo lì. È chiaro che questo è esattamente quello che abbiamo a sinistra della disuguaglianza. E sappiamo che questo è minore o uguale a sigma j di i meno b.

`00:13:09` 
Quindi applicando quella relazione, abbiamo questa disuguaglianza. Tenete a mente che b è qualsiasi possibile matrice di rango k. E cos'è questa espressione? In realtà, è la somma dei valori singolari al quadrato della matrice A meno B, okay? Quindi, questo.

`00:13:47` 
Quindi, se collegate tutto, abbiamo la norma al quadrato di I meno K è minore o uguale a questa quantità. Okay? La norma di Frobenius, la norma di Frobenius completa di A meno B al quadrato sarebbe la somma da J uguale a 1 fino al minimo tra M e N.

### 1.4 Dimostrazione nella norma di Frobenius {#dimostrazione-frobenius}

`00:14:26` 
di sigma j i di a meno b al quadrato ma questo termine è sicuramente minore o uguale a questo perché sappiamo che i valori singolari sono o positivi o zero quindi significa che dato che qui stiamo solo sommando alcuni di loro e inoltre sono al quadrato questa somma che è.

### 1.5 Risultato finale: ottimalità di A_K {#risultato-ottimalita}

`00:14:59` 
la norma di Frobenius completa di a meno b sarà sicuramente maggiore e quindi se combinate il risultato in alto con questo quello che ottenete è che la norma di a meno a k dove a k è l'approssimazione di rango K della matrice A ottenuta dalla SVD è minore o uguale alla norma, ovviamente queste sono entrambe norme di Frobenius, scusate ho dimenticato di mettere la F, della norma di Frobenius di A meno B al quadrato, dove B è qualsiasi altra matrice di rango K.

`00:15:56` 
E inoltre, ricordate che nel risultato qui, abbiamo affermato che questa norma è data da questa quantità, che è in realtà quello che abbiamo ottenuto qui, okay? Quindi quello che abbiamo dimostrato è che sia nella norma due che nella norma di Frobenius, l'AK, l'approssimazione di rango K ottenuta usando la SVD è la migliore approssimazione di rango K della matrice.

`00:16:58` 
Quindi la SVD sarebbe il cavallo di battaglia per l'approssimazione a basso rango di insiemi di dati, okay?

---

## Principal Component Analysis (PCA) {#pca}

### 2.1 Introduzione alla PCA {#intro-pca}

`00:17:36` 
Okay, ora vedremo un'applicazione, un'altra applicazione della SVD, che è, in pratica, sto anticipando, non è niente di concettualmente diverso da quello che abbiamo visto finora. Quindi, è, se volete, un'approssimazione a basso rango, o se volete una reinterpretazione di un dataset in termini delle sue componenti principali, e questo si chiama analisi delle componenti principali.

`00:18:26` 
**PCA (Principal Component Analysis)** non è altro che l'applicazione della SVD alla **matrice di covarianza** di un dataset.

**Matrice di Covarianza:**
$$
C = \frac{1}{n-1} X^T X \in \mathbb{R}^{p \times p}
$$

Dove:
- $X \in \mathbb{R}^{n \times p}$: dataset ($n$ campioni, $p$ features)
- $C_{ij}$: covarianza tra feature $i$ e feature $j$
- $C_{ii}$: varianza della feature $i$

*Nota storica:* Abbiamo già incontrato $X^T X$ quando abbiamo dimostrato l'esistenza della SVD tramite decomposizione spettrale.

### 2.2 Matrice di covarianza e SVD {#matrice-covarianza}

`00:18:57` 
Abbiamo visto che per dimostrare, dato che la SVD, abbiamo affermato che esiste per qualsiasi matrice, quello che abbiamo fatto è partire dalla matrice X, che può essere qualsiasi cosa vogliate, costruite la matrice X trasposta X, che è simmetrica e definita positiva. Per queste matrici, potete ottenere la decomposizione spettrale. E poi sfruttando quel particolare trucco di introdurre i vettori AV su sigma I, potete dimostrare che a parte gli autovettori di X trasposta X, che sono...

`00:19:42` 
**Obiettivi della PCA:**

1. **Riduzione della dimensionalità:** Trasformare dataset da $p$ dimensioni a $k << p$ dimensioni preservando la massima informazione possibile

2. **Identificazione direzioni di massima varianza:** Le **componenti principali** sono direzioni ortogonali lungo cui i dati variano maggiormente

3. **Decorrelazione features:** Nel nuovo sistema di riferimento (assi principali), le componenti sono **non correlate** (ortogonali)

`00:20:42` 
**Interpretazione geometrica:** Dato un cloud di punti in $\mathbb{R}^p$, la PCA trova un nuovo sistema di coordinate (gli **assi principali**) che:
- **Allinea** il primo asse con la direzione di massima varianza
- **Allinea** il secondo asse con la direzione di massima varianza residua (ortogonale al primo)
- E così via per tutti gli assi

In 2D, visualizziamo due assi principali che descrivono l'orientamento dell'ellisse dei dati.

`00:21:18` 
**In 2D:** Le componenti principali sono **due direzioni ortogonali** che catturano:
- **PC1 (prima componente principale):** direzione di **massima varianza**
- **PC2 (seconda componente principale):** direzione di massima varianza **ortogonale a PC1**

**In dimensioni superiori ($p > 2$):** Otteniamo $p$ componenti principali ordinate per importanza:

$$
\text{Var}(\text{PC}_1) \geq \text{Var}(\text{PC}_2) \geq \cdots \geq \text{Var}(\text{PC}_p) \geq 0
$$

`00:21:52` 
**Principio fondamentale:** Le prime $k$ componenti catturano la **maggior parte della varianza** del dataset. Possiamo rappresentare i dati usando solo queste $k$ componenti, ottenendo una compressione da $p$ a $k$ dimensioni.

`00:22:27` 
**Esempio pratico:** Un dataset con 100 features potrebbe avere il 90% della varianza concentrato nelle prime 10 componenti principali. Usando solo queste 10, riduciamo la dimensionalità del 90% preservando il 90% dell'informazione!

L'efficacia dipende dall'applicazione: alcuni problemi richiedono più componenti, altri meno.

`00:22:59` 
**Ortonormalità delle componenti principali:**

Le componenti principali $\{u_1, u_2, \ldots, u_p\}$ formano una **base ortonormale** di $\mathbb{R}^p$:

$$
u_i^T u_j = \begin{cases}
1 & \text{se } i = j \\
0 & \text{se } i \neq j
\end{cases}
$$

**Analogia con Gram-Schmidt:** Come il processo di Gram-Schmidt trasforma una base arbitraria in una base ortogonale, la PCA trova automaticamente la base ortogonale **ottimale** per i dati, quella che massimizza la varianza lungo ciascun asse.

`00:23:48` 
**Visualizzazione:** Immaginate i dati come punti blu sparsi nello spazio. Le componenti principali sono due frecce verdi ortogonali:
- La **freccia più lunga** (PC1) punta nella direzione di massimo "allungamento" dei dati
- La **freccia più corta** (PC2) è ortogonale alla prima e cattura la varianza residua

`00:24:22` 
E lasciatemi, okay, vedremo alla fine. Quindi, in pratica, a parte il fatto che stiamo usando un nome diverso, ma cos'è la PCA? Qual è l'algoritmo per ottenere la PCA? Quindi, partiamo con la solita matrice X, che è il nostro dataset.

`00:24:52` 
**Step 1 - Centramento dati (Mean Subtraction):**

Per ogni feature $j$ (colonna di $X$), calcoliamo la media:

$$
\bar{x}_j = \frac{1}{n} \sum_{i=1}^n X_{ij}
$$

Poi sottraiamo la media da ciascun dato, ottenendo la matrice centrata $B$:

$$
B_{ij} = X_{ij} - \bar{x}_j
$$

In forma matriciale:
$$
B = X - \mathbf{1}_n \bar{x}^T
$$

Dove $\mathbf{1}_n$ è il vettore colonna di $n$ elementi tutti uguali a 1.

**Perché centrare?** La PCA cerca direzioni di massima varianza **rispetto alla media**. Centrare i dati sposta il cloud di punti sull'origine, semplificando i calcoli.

`00:25:27` 
**Step 2 - Matrice di Covarianza Campionaria:**

$$
C = \frac{1}{n-1} B^T B \in \mathbb{R}^{p \times p}
$$

**Perché $n-1$ invece di $n$?** (Correzione di Bessel)

- Stimatore **non distorto** (unbiased) della covarianza
- Abbiamo già "usato" 1 grado di libertà per calcolare la media $\bar{x}$
- La divisione per $n-1$ compensa questo bias

`00:26:00` 
**Struttura di $C$:**
- **Diagonale:** $C_{ii} = \text{Var}(\text{feature}_i)$ (varianze)
- **Off-diagonale:** $C_{ij} = \text{Cov}(\text{feature}_i, \text{feature}_j)$ (covarianze)
- $C$ è **simmetrica** e **semi-definita positiva**

`00:27:10` 
**Step 3 - Eigende composition di $C$:**

$$
C = V \Lambda V^T
$$

Dove:
- $V$: matrice ortogonale con eigenvectors di $C$ come colonne
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_p)$: eigenvalues (ordinati decrescenti)

**Output:**
- **Componenti principali:** $\text{PC}_i = v_i$ (colonne di $V$)
- **Varianza lungo PC$_i$:** $\text{Var}(\text{PC}_i) = \lambda_i$

`00:27:10` 
**PROBLEMA COMPUTAZIONALE:** Calcolare esplicitamente $C = B^T B$ è:
1. **Costoso:** Complessità $O(np^2)$ se $p$ è grande
2. **Instabile numericamente:** Il **numero di condizione** di $C$ è il quadrato di quello di $B$:
   $$\kappa(C) = \kappa(B)^2$$
   Questo amplifica gli errori di arrotondamento nell'aritmetica floating-point!

**SOLUZIONE:** Usare la SVD direttamente su $B$ invece di calcolare $C$!

### 2.4 PCA via SVD: approccio stabile {#pca-via-svd}

`00:28:01` 
Quindi, qui entra in gioco, di nuovo, la SVD. Quindi, l'idea è, non voglio usare, calcolare la matrice X trasposta X esplicitamente, ma voglio usare la SVD. Quindi, data la decomposizione SVD della matrice X, come al solito, U sigma V trasposta,

`00:28:36` 
Possiamo scrivere la matrice C come 1 su n meno 1, x trasposta x. Poi potete sostituire la matrice, la decomposizione. Qui dovete solo ricordare le proprietà della trasposta del prodotto e il fatto che u e v sono matrici ortogonali. Quindi in particolare, u trasposta u è l'identità.

`00:29:08` 
E quindi alla fine, quello con cui potete venire fuori è questa decomposizione. v, sigma al quadrato su n meno 1, v trasposta. Quindi questo è esattamente la decomposizione spettrale di C. Okay, C è simmetrica e definita positiva, e siamo interessati a trovare i suoi autovalori e autovettori, e usando questo trucco, siamo in grado di calcolare gli autovalori e gli autovettori senza la necessità di calcolare esplicitamente la matrice stessa, semplicemente sfruttando la sua decomposizione SVD.

### 2.5 Componenti principali e varianza {#componenti-principali}

`00:30:06` 
Quindi le colonne di V, che sono i vettori singolari destri della matrice originale, sono le componenti principali, e questi vettori, sigma j al quadrato diviso per n meno 1, sono gli autovalori corrispondenti. Okay, quindi dati i valori singolari, gli autovalori sono solo il quadrato scalato da un fattore n meno uno. Okay, quindi come potete vedere di nuovo, la SVD, sta entrando qui e sta semplificando molto il calcolo.

`00:30:54` 
Ovviamente, se la matrice X è davvero grande, allora potete fare un passo ulteriore e invece di calcolare la... SVD, potete calcolare una SVD randomizzata, quindi state, ovviamente, state aggiungendo un altro livello di approssimazione, ma tutte queste approssimazioni sono controllate, quindi siete in grado di dire quale è l'errore che state introducendo usando queste approssimazioni.

`00:31:36` 
Quindi, in pratica, potete usare la SVD randomizzata e ottenere un'approssimazione degli autovettori e autovalori, quindi un'approssimazione delle componenti principali. Quindi, si potrebbe dire, se torniamo all'immagine, okay, data questa immagine, si potrebbe dire, dimenticate per un momento questa piccola freccia verde, solo quella più lunga.

---

## PCA vs Least Squares {#pca-vs-least-squares}

### 3.1 Differenze negli obiettivi {#differenze-obiettivi}

`00:32:15` 
È molto simile a una regressione lineare del dataset. Quindi, è data la nuvola di punti calcolando la lunga linea verde, state in qualche modo approssimando la nuvola di punti con un trend lineare, okay, che è la regressione lineare. O, se volete, l'approssimazione classica dei minimi quadrati del dataset, approssimazione lineare dei minimi quadrati.

`00:32:53` 
Ma qual è la differenza tra i minimi quadrati classici, che vedremo più in dettaglio dopo, e la PCA? Quindi, prima di tutto, l'obiettivo è diverso. Nella PCA, quello che vogliamo ottenere è la riduzione della dimensionalità del nostro dataset.

`00:33:24` 
Quindi non siamo interessati a ottenere qualcosa che possiamo usare per fare previsioni. Esempio, supponiamo che abbiate molte misurazioni su temperatura e volume di gas reale, e volete calcolare dati sperimentali, e volete calcolare una linea che approssima questi dati per fare alcune previsioni sul valore del volume per diverse temperature che non sono state misurate.

`00:34:02` 
Questo è qualcosa che è il regno dell'approssimazione dei minimi quadrati, o approssimazione in generale. Grazie. Perché quello che volete ottenere è ottenere il carico. Un carico in cui potete mettere un nuovo valore per la variabile indipendente, e volete ottenere il corrispondente valore indipendente. Nella PCA, quello che vogliamo ottenere è, ho un dataset che è di alta dimensione,

`00:34:39` 
e la mia ipotesi è che probabilmente non tutte queste dimensioni sono davvero importanti, per descrivere la struttura sottostante del dataset stesso. Quindi voglio essere in grado di ridurre la dimensione che ho nel dataset per ottenere un dataset che è più gestibile da un lato e anche più interpretabile.

`00:35:09` 
Che è un altro aspetto importante. Okay, quindi l'obiettivo è diverso. Poi la seconda differenza importante è la misura che sto usando per controllare la distanza di, diciamo, la linea verde dai dati. Nelle componenti principali, stiamo essenzialmente, quello che state facendo è state minimizzando le distanze ortogonali ai dati.

### 3.2 Distanze ortogonali vs distanze verticali {#distanze-diverse}

`00:35:51` 
Quindi, ecco la differenza. Quindi, dati i... punti neri. La linea blu è la componente principale. E come potete vedere, quello che sto minimizzando essenzialmente è la somma di tutte queste distanze azzurre. Okay? Quindi distanze ortogonali dalla prima direzione principale. Mentre nei minimi quadrati, quello che stiamo minimizzando è la distanza verticale.

`00:36:36` 
Quindi se andiamo all'immagine, le distanze rosse. Quindi, e ovviamente, dato che state usando una misura diversa dell'errore che volete, volete minimizzare, la direzione finale con cui venite fuori è totalmente diversa. Okay? Poi c'è un altro punto importante, la simmetria. Il fatto che sto usando essenzialmente le distanze ortogonali, dice che non sto facendo alcuna preferenza di variabile rispetto all'altra.

`00:37:24` 
Perché la mia idea è che sono totalmente cieco rispetto all'importanza e al significato di ogni variabile che sta descrivendo il dataset. E dato che sto misurando la distanza ortogonale, non sto assumendo che non sto facendo alcuna ipotesi sul fatto che una variabile sia più importante dell'altra. Qui, non è vero, perché qui sto dicendo, sto misurando questa distanza, okay, quindi la distanza verticale.

### 3.3 Simmetria delle variabili {#simmetria-variabili}

`00:38:04` 
Qui, sto facendo un'assunzione forte, nel senso che sto dicendo che questa direzione è la più importante. Avrei potuto usare le distanze orizzontali, okay? Okay, quindi nei minimi quadrati, non sto trattando, in questo caso, le due variabili allo stesso modo, okay?

`00:38:34` 
Quindi, anche se, a prima vista, i due approcci potrebbero sembrare risolvere... Un problema abbastanza simile, in pratica, l'obiettivo, la tecnica e anche il significato sono totalmente diversi. Quindi fate attenzione. PCA e minimi quadrati sono due tecniche che hanno qualcosa in comune, ma sono sviluppate con cose diverse in mente.

### 2.6 Visualizzazione 2D: assi principali {#visualizzazione-pca}

`00:39:18` 
Okay, qui c'è solo un riepilogo di quello che abbiamo appena detto e voglio solo mostrarvi... Okay, qui, è un esempio molto semplice in cui sto considerando 10.000 punti, e sto tracciando i punti in, quindi è come la nuvola che abbiamo visto prima.

`00:40:21` 
Poi, idealmente, sono orizzontali. Sto ruotando questi dataset, e, okay, poi qui. Sto calcolando la media, sottraendo la media, e poi essenzialmente quello che sto facendo è solo usare la SVD sulla matrice B,

`00:40:58` 
che è la matrice che abbiamo considerato prima, e infine sto tracciando i dati. Quindi a sinistra avete il dataset originale, a destra avete il dataset con, Le due componenti principali, potete vedere le direzioni e la lunghezza, quindi è chiaro che questa direzione vi dice che questa è la prima componente principale, è quella con il livello più alto di varianza, questa è la seconda, e poi i cerchi rossi, le ellissi rosse.

`00:41:50` 
Okay, qui sto solo tracciando rispetto, come potete vedere, alle... ellissi i cui assi sono esattamente dati dalle componenti principali, sto tracciando le ellissi, che stanno dando due volte la varianza in ogni direzione e tre volte. Quindi potete vedere.

`00:42:22` 
se scalate i due vettori, qual è la quantità di varianza, la scala della varianza che state catturando. Quindi sono essenzialmente le ellissi rosse che vi stanno dando diversi livelli di varianza che state catturando scalando le componenti principali. Questo è, a parte queste ellissi rosse, questo è esattamente quello che abbiamo visto prima nell'immagine, in pratica.

`00:42:55` 
E come potete vedere, non è niente di concettualmente diverso dall'applicazione della SVD a una matrice diversa rispetto a quello che abbiamo visto finora. Okay, quindi ora, dato che abbiamo parlato dei minimi quadrati, vediamo ora cosa sono i minimi quadrati.

---

## Problema dei Minimi Quadrati {#problema-minimi-quadrati}

### 4.1 Formulazione del problema {#formulazione-problema}

`00:43:53` 
## Formulazione del Problema di Regressione Lineare

**Dati:**
- $X \in \mathbb{R}^{n \times p}$: matrice dei dati ($n$ campioni, $p$ features)
- $y \in \mathbb{R}^n$: vettore target

**Esempio concreto:** Classificazione immagini cani vs gatti
- Dataset: $n$ immagini in scala di grigi $200 \times 200$ pixel
- Ogni immagine viene **appiattita** in un vettore di $p = 40000$ valori
- Ciascuna riga di $X$ è un'immagine appiattita
- $y$ contiene le etichette: $y_i = 1$ se gatto, $y_i = 0$ se cane

`00:44:49` 
**Obiettivo:** Trovare un vettore di pesi $w \in \mathbb{R}^p$ tale che:

$$
Xw \approx y
$$

Cioè, vogliamo un modello lineare dove $Xw$ predice le etichette $y$.

`00:45:28` 
**Assunzioni tipiche:**
1. $n >> p$ (più campioni che features)
2. $\text{rank}(X) = p$ (colonne linearmente indipendenti)
3. Il sistema $Xw = y$ è **sovradeterminato** (più equazioni che incognite)

`00:46:15` 
**Problema:** Il sistema $Xw = y$ è **sovradeterminato**:
- $n$ equazioni (una per ogni campione)
- $p$ incognite (i pesi $w$)
- $n > p$

In generale, **non esiste soluzione esatta**! Non possiamo trovare $w$ tale che $Xw = y$ esattamente.

**Soluzione:** Minimizzare l'errore in senso di **least squares**:

$$
\min_w \|y - Xw\|_2^2
$$

**Soluzione analitica (equazioni normali):**

$$
w^* = (X^T X)^{-1} X^T y = X^{\dagger} y
$$

Dove $X^{\dagger} = (X^T X)^{-1} X^T$ è la **pseudo-inversa** (o inversa di Moore-Penrose) di $X$.

### 4.2 Sistema sovradeterminato {#sistema-sovradeterminato}

`00:46:48` 
**Perché $Xw = y$ non è risolvibile?**

L'equazione $Xw = y$ ha soluzione se e solo se $y \in \text{span}(X)$ (spazio colonna di $X$).

**Interpretazione:** $Xw$ è una **combinazione lineare** delle colonne di $X$:

$$
Xw = w_1 \begin{pmatrix} | \\ x^{(1)} \\ | \end{pmatrix} + w_2 \begin{pmatrix} | \\ x^{(2)} \\ | \end{pmatrix} + \cdots + w_p \begin{pmatrix} | \\ x^{(p)} \\ | \end{pmatrix}
$$

In un sistema sovradeterminato ($n > p$), **in generale** $y \notin \text{span}(X)$. Quindi non esiste soluzione esatta!

`00:47:57` 
**Soluzione:** Introduciamo il **residuo**:

$$
r(w) = y - Xw
$$

Il residuo misura "quanto lontano" siamo da una soluzione esatta.

**Obiettivo Least Squares:** Trovare $w^*$ che minimizza $\|r(w)\|_2^2$:

$$
w^* = \arg\min_{w \in \mathbb{R}^p} \|y - Xw\|_2^2
$$

`00:48:35` 
È una quantità che è usata anche per, per esempio, calcolare o valutare la convergenza di un metodo iterativo. La norma del residuo è uno dei possibili indicatori della convergenza. Quindi qui stiamo calcolando il residuo. Quindi dato il vettore w, stiamo calcolando questa quantità r.

### 4.3 Residuo e minimizzazione {#residuo-minimizzazione}

`00:49:05` 
uguale a y meno xw e ora l'idea dei minimi quadrati è trovare il vettore w appartenente a rt, che minimizza il quadrato della norma del residuo okay se pensate al.

`00:49:37` 
probabilmente l'esempio classico dei minimi quadrati che avete visto durante il corso di analisi numerica quindi se avete x1 e x2 è chiaro che se avete solo due punti, Allora, e volete trovare una rappresentazione lineare dei dati, ovviamente il sistema è risolvibile, e potete scrivere questa linea.

### 4.4 Esempio visivo: regressione lineare {#esempio-regressione}

`00:50:08` 
Ma se avete più punti come questi, allora ovviamente avete due scelte. La prima è usare non un'approssimazione ma un'interpolazione per creare una curva che passa attraverso tutti i punti. Ma questo potrebbe non essere significativo in pratica perché se queste sono misurazioni probabilmente affette da errori.

`00:50:44` 
in quel modo state costruendo una legge che tiene conto di tutti gli errori. Quello che volete trovare forse è qualcosa come questo. OK, che è esattamente quello che vogliamo fare. OK, quindi qui e l'idea è che esattamente come abbiamo visto prima, stiamo considerando queste distanze e vogliamo minimizzare la somma dei quadrati di tutte queste distanze per trovare la m e q di questa linea.

`00:51:21` 
OK, qui quello che vogliamo fare è generalizzare questa idea a dimensione superiore. Quindi questa è la formulazione del problema dei minimi quadrati nel caso generale. L'idea è, dati x e y, e introducendo il vettore residuo r, quello che vogliamo trovare è un vettore w, il vettore di pesi w, che minimizza la norma, il quadrato della norma del residuo.

---

## Interpretazione Geometrica {#interpretazione-geometrica}

### 5.1 Proiezione ortogonale sullo spazio colonna {#proiezione-ortogonale}

`00:52:07` 
Okay, prima di considerare la soluzione analitica, diciamo, di questo problema, cerchiamo di capire geometricamente cosa significa. Quindi, alla fine, otterremo un vettore, chiamiamolo w cappello.

`00:52:44` 
che è quello che minimizza l'errore, e eseguiremo questa operazione x per w cappello. Cos'è questa operazione? Beh, questa operazione significa che stiamo prendendo le colonne di x e stiamo creando una combinazione lineare delle colonne di x. Quindi il vettore risultante,

`00:53:14` 
questo vettore risultante, apparterrà allo spazio colonna di x per costruzione. Poi l'idea è, dato un vettore y, che è in principio, sappiamo che non è nello spazio colonna di x, Qual è il vettore xw che crea, lasciatemi chiamare, questo y cappello, che è nello spazio colonna di x, ed è il più vicino possibile a y.

### 5.2 y_hat come migliore approssimazione {#migliore-approssimazione}

`00:53:57` 
Okay? È chiaro? Quindi, stiamo cercando w cappello, tale che xw cappello, che è un vettore che chiameremo y cappello, e appartiene allo spazio colonna di x, è il più vicino possibile a y, che è il vettore originale delle etichette. Questa è l'immagine.

`00:54:27` 
Quindi, avete qui y. Il vettore originale delle etichette y, e la freccia più lunga originale. eh freccia poi questo piano blu è lo spazio colonna di x qui è chiamato a ma è lo stesso, e quindi se avete un punto nello spazio e il sottospazio qual è il.

`00:55:01` 
punto più vicino che avete sul sottospazio al punto fuori dello spazio è la proiezione, okay è la proiezione ortogonale di y sul sottospazio okay per definizione la proiezione ortogonale vi dà la distanza più breve dal punto y al sottospazio okay quindi.

`00:55:37` 
Quello che essenzialmente stiamo sostenendo è che il vettore y cappello che stiamo che vogliamo costruire è la proiezione ortogonale di y sullo spazio colonna di a o x. Okay. E questa è l'interpretazione geometrica di questa operazione. Okay.

`00:56:13` 
Qui nell'immagine non è enfatizzato, ma proverò a fare un'immagine. Ma quindi supponiamo che abbiate il sottospazio. Qui avete il vettore y. E quella qui è la proiezione. Quindi questo è y e questo è y cappello. Quindi questo angolo è il vettore di x.

`00:56:55` 
OK, quindi l'idea è che se prendete qualsiasi altro vettore nello spazio colonna di x. Quindi supponiamo che stiate prendendo il vettore qui, questo che sto chiamando y tilde, per esempio. È chiaro che la distanza da y a.

`00:57:35` 
Y tilde è più grande di questo, semplicemente ispezionando il fatto che avete un triangolo qui, okay, questo è un triangolo rettangolo, quindi qui avete i due lati, e questo è più lungo di questo, okay?

`00:58:06` 
Quindi, che non è nient'altro che un'immagine geometrica che vi mostra che l'ortogonale, questo vettore, vi sta dando la migliore approssimazione che potete ottenere, okay? L'approssimazione più vicina a Y nello spazio colonna di X. Ora, quindi, data questa intuizione geometrica, ora cerchiamo di formalizzare questa idea.

### 5.3 Derivazione delle equazioni normali {#equazioni-normali}

`00:58:47` 
## Derivazione Algebrica: Equazioni Normali

**Condizione chiave:** Il residuo $r = y - X\hat{w}$ è **ortogonale** a $\text{span}(X)$:

$$
X^T r = X^T (y - X\hat{w}) = 0
$$

**Perché?** Se $r \perp \text{span}(X)$, allora $r$ è ortogonale a **ogni colonna** di $X$:

$$
\langle x^{(j)}, r \rangle = 0 \quad \text{per } j = 1, \ldots, p
$$

In forma matriciale: $X^T r = 0$

`00:59:53` 
**Espandendo:**

$$
X^T (y - X\hat{w}) = 0 \quad \Rightarrow \quad X^T y = X^T X \hat{w}
$$

Queste sono le **equazioni normali** (normal equations).

`01:00:30` 
**Soluzione:** Se $X$ ha rango colonna pieno ($\text{rank}(X) = p$), allora $X^T X$ è invertibile:

$$
\hat{w} = (X^T X)^{-1} X^T y
$$

**Nota:** Questa è la stessa formula vista nel corso di Analisi Numerica per calcolare i coefficienti $m$ e $q$ della retta di regressione!

`01:01:07` 
OK, o come abbiamo qui, X trasposta Y deve essere uguale a X trasposta X W cappello. Quindi questo sistema. È chiamato il sistema delle equazioni normali, che è il nome che probabilmente avete visto anche nel corso di analisi numerica quando avete scritto esempi semplici, per esempio, per la regressione lineare.

`01:01:47` 
Quando dovete scrivere tutti i vincoli per calcolare la M e Q per la regressione lineare, dovete scrivere le equazioni normali, che non sono nient'altro che questa rappresentazione matriciale. E voglio solo enfatizzare il fatto che qui, di nuovo, entra in gioco la matrice X trasposta X.

### 5.4 Soluzione analitica: w_hat {#soluzione-analitica}

`01:02:17` 
Quindi, ovviamente, ora quello che vogliamo calcolare è il vettore W, quindi dato che questo è simmetrico e definito positivo può essere invertito, quindi potete ottenere il vettore W uguale a X trasposta X inversa X trasposta Y, okay? Quindi il vettore di pesi che risolve il problema dei minimi quadrati è dato da questa rappresentazione, okay?

`01:02:55` 
E abbiamo ottenuto questo risultato solo ispezionando la geometria del problema, capendo, almeno in un caso semplice, e generalizzando a qualsiasi definizione. Ma se ricordate, quando abbiamo introdotto il problema, e abbiamo impostato il problema, abbiamo detto che il problema dei minimi quadrati equivale a trovare il minimo del quadrato della norma, trovare w, tale che minimizza il quadrato della norma del residuo.

`01:03:38` 
Okay, quindi ora quello che vogliamo fare è considerare questa formulazione del problema. Quindi considerare un vero problema di minimizzazione. Okay. E in particolare, per un momento, possiamo immaginare di non aver, In effetti, la geometria del problema, non abbiamo intuizione su quale sia la configurazione speciale dei nostri dati, ma vogliamo solo andare alla cieca e scrivere questo funzionale e minimizzare questo funzionale rispetto a W.

`01:04:17` 
Okay, quindi questo è il secondo modo di risolvere il problema. Prima di passare ai dati, un paio di considerazioni. Quindi abbiamo detto che W cappello è ottenuto usando questa espressione.

`01:04:49` 
Questa matrice, X trasposta X, è sicuramente inversa. È invertibile, è simmetrica e definita positiva. L'unica cosa che, in realtà, potrebbe essere anche semi-positiva, perché può avere, in principio, alcuni autovalori zero. Ma quello che vogliamo mostrare è che è di rango p. In realtà, abbiamo già mostrato quando abbiamo dimostrato la SVD, ma lo vedremo di nuovo.

`01:05:28` 
Quindi, qui, stiamo solo rivedendo quel risultato, quindi se la matrice originale ha colonne linearmente indipendenti, e questa era una delle assunzioni che abbiamo fatto all'inizio, ricordate che nel blocco rosso abbiamo assunto che sia di rango pieno di colonna.

`01:05:59` 
Allora, x trasposta x è invertibile, quindi non è semi-definita positiva, ma è definita positiva, quindi non ci sono autovalori zero o valori singolari zero di quel valore. Quindi, questa è la dimostrazione che abbiamo già visto. È solo una questione di scrivere queste espressioni, quindi moltiplicando l'espressione x trasposta x, dove v è un vettore nello spazio nullo.

`01:06:39` 
E poi sfruttando la trasposta, avete questa relazione, che è la norma al quadrato. E quindi la norma al quadrato uguale a zero significa che xv dovrebbe essere uguale a zero. Ma dato che v è nello spazio nullo di x, significa che, scusate, dato che xv è uguale a zero, v è nello spazio nullo di x. Ma x ha colonne linearmente indipendenti, quindi contiene solo il vettore zero, e quindi v è uguale a zero.

`01:07:22` 
Quindi non abbiamo nessun altro, quindi sappiamo che una caratterizzazione della matrice non singolare è il fatto che l'unico vettore nello spazio nullo è il vettore nullo. Quindi, in altri termini, se avete un sistema lineare AX uguale a B, questo è risolvibile solo se lo spazio nullo di A è dato dall'unico elemento dello spazio nullo di A è dato dal valore zero.

`01:08:06` 
Quindi, se X è di rango pieno, allora X trasposta X è invertibile. E questa espressione ha senso, possiamo calcolare W. Okay, qui ho solo riportato un paio di esempi, ma forse possiamo, lascerò.

`01:08:39` 
Quindi ora dobbiamo formalizzare quello che abbiamo detto prima, ma prima di ciò voglio dire qualcosa sulla matrice di proiezione. Quindi sappiamo che se abbiamo il vettore y cappello costruito come xw cappello, questo vettore y cappello è essenzialmente una proiezione del w originale.

### 6. Matrici di Proiezione e Proprietà {#matrice-proiezione}

### 6.1 Matrice di proiezione P {#definizione-p}

`01:09:33` 
ma la soluzione dove quando quel w cappello è calcolato in modo corretto è la proiezione di y sullo spazio colonna di x okay quindi essenzialmente se qui inseriamo questa espressione possiamo ottenere che y cappello è dato da x x trasposta x inversa x trasposta y okay ho solo preso questa espressione e l'ho inserita qui okay.

`01:10:23` 
Quindi qui sto dicendo che y cappello, che è la proiezione di y secondo l'intuizione geometrica che abbiamo, l'immagine geometrica che abbiamo visto, è la proiezione ortogonale di y sullo spazio colonna di x, è data da questa espressione. Qui abbiamo y e qui abbiamo questo strano oggetto. In realtà questo strano oggetto è quello che è chiamato la proiezione, è un'istanza.

`01:10:59` 
di una famiglia di matrici chiamate matrici di proiezione. Quindi se considerate questa matrice, e prendete qualsiasi vettore dello spazio, l'effetto di questa matrice su quel vettore, è che proietta, anche, quel vettore sullo spazio colonna di X, okay? Potete verificare che P è simmetrica.

### 6.2 Proprietà: simmetria e idempotenza {#proprieta-p}

`01:11:31` 
e P al quadrato è uguale a P, okay? Quindi una volta che avete proiettato l'elemento sullo spazio colonna, se riapplicate la matrice, essenzialmente non vi state muovendo da quel punto, okay? Okay, quindi questa è un'osservazione importante. Okay, ora veniamo al problema di minimizzazione,

### 7. Minimizzazione come Problema di Ottimizzazione {#minimizzazione-ottimizzazione}

### 7.1 Formulazione funzionale J(w) {#formulazione-funzionale}

`01:12:02` 
il problema che abbiamo enunciato prima. Prima di tutto, qui sto usando questa funzione argmin.

`01:12:33` 
Avete mai visto questa funzione prima? Sì, quindi chi non ha mai visto questa funzione? Alzate la mano. Siete tutti consapevoli di ciò. Okay, quindi sto formulando il problema come w cappello è il vettore che minimizza, il funzionale j di w, dove il funzionale j di w è quello che abbiamo visto prima.

### 7.2 Espansione del residuo quadratico {#espansione-residuo}

`01:13:04` 
Quindi è la norma del residuo al quadrato. Ora quello che dobbiamo fare è esprimere esplicitamente il quadrato della norma del residuo. E questo non è nient'altro che quello che abbiamo fatto molte volte per una singola situazione, ma qui avete il residuo trasposto per il residuo, poi applicate la proprietà della trasposta, e poi dovete eseguire tutte le operazioni.

`01:13:44` 
Poi questo è uno scalare, quindi potete dire che è uguale alla sua trasposta. Quindi alla fine, quindi essenzialmente questi due sono uguali, e quindi il funzionale è questo, che è un funzionale quadratico in W.

### 7.3 Calcolo del gradiente {#calcolo-gradiente}

`01:14:16` 
E quello che devo fare è minimizzare questa funzione, e il W cappello che abbiamo ottenuto da considerazioni geometriche, sperabilmente sarà lo stesso che troveremo minimizzando questa funzione. Quindi, dobbiamo calcolare essenzialmente il gradiente di J rispetto a W.

`01:15:03` 
e qui abbiamo i tre contributi. Il primo non dipende da w, per il secondo abbiamo, meno due x trasposta y, e qui abbiamo due x trasposta x w. Poi dobbiamo impostare, questa quantità uguale a zero, e quello con cui finiamo è questo insieme di equazioni,

`01:15:34` 
che è esattamente l'insieme delle equazioni normali, che è lo stesso che abbiamo ottenuto dall'approccio geometrico. Quindi ovviamente i due approcci stanno dando lo stesso risultato, e poi da qui posso calcolare w cappello.

### 7.4 Equivalenza con approccio geometrico {#equivalenza-approcci}

`01:16:08` 
Quindi, in pratica, i due approcci sono totalmente equivalenti, e in qualche modo siamo stati in grado di risolvere il problema che avevamo in mente. Abbiamo trovato questo vettore di pesi w cappello, che ci sta fornendo un modo di creare quello che possiamo chiamare un modello dei nostri dati, x w cappello, con cui possiamo fare previsioni.

`01:16:43` 
Possiamo usare per l'inferenza, se volete. Ora, in pratica... Quando vogliamo risolvere il problema e ottenere W cappello, dobbiamo risolvere questo sistema lineare.

### 8. Soluzione via SVD {#soluzione-svd}

### 8.1 Problemi computazionali di X^TX {#problemi-computazionali}

`01:17:21` 
Se ricordate, quando stavamo parlando della PCA, abbiamo detto che, OK, il calcolo di X trasposta X è instabile. Può essere molto grande. Qui abbiamo esattamente gli stessi problemi. Perché dobbiamo calcolare X trasposta X. Quindi, anche, in questo contesto, dobbiamo ottenere qualcosa che possa aiutarci a risolvere quel problema efficientemente e in modo stabile.

`01:17:59` 
Qual è l'idea? Usare la SVD. Esattamente come prima. Quindi quello che vedremo è come posso usare la SVD per risolvere il problema dei minimi quadrati? Ricordate che. OK, quindi considerate la SVD ridotta.

### 8.2 SVD ridotta (economy) {#svd-ridotta}

`01:18:49` 
Quindi non è la troncata. È la ridotta significa che a seconda della dimensione o del rango della matrice, state solo tenendo, quindi u e v non saranno quadrate, e sigma sarà una matrice quadrata invece di una pseudo-diagonale, okay? Quindi avete ur, sigma r, vr trasposta. Poi abbiamo la soluzione per il nostro problema, w cappello è x trasposta x inversa x trasposta y.

### 8.3 Derivazione di w_hat via SVD {#derivazione-svd}

`01:19:26` 
Quello che dobbiamo fare è sostituire la SVD, la SVD ridotta che abbiamo calcolato nella formula della soluzione. Prima, calcoliamo x trasposta x, ma questo è qualcosa che abbiamo già fatto anche precedentemente, e qui avete esattamente lo stesso risultato a parte il fattore n meno 1, il fattore di scala della PCA.

`01:20:02` 
Se ricordate, nella PCA, qui avevamo su n meno 1. Quindi, x trasposta x è qualcosa che abbiamo calcolato. Poi, nella soluzione, abbiamo x trasposta x alla meno 1.

`01:20:40` 
Quindi, dato questo, dobbiamo calcolare l'inversa, quindi abbiamo Vr trasposta inversa, l'inversa di sigma al quadrato, e poi Vr inversa. Quindi, qui abbiamo il primo fattore è calcolato, che è questo. Qui stiamo solo inserendo la SVD.

`01:21:15` 
Vr trasposta Vr è l'identità. Quindi, W alla fine è Vr, sigma r alla meno uno, U r trasposta y. Quindi, di nuovo, abbiamo calcolato la soluzione del nostro problema in termini della SVD dell'originale. Qui.

### 8.4 Pseudo-inversa di Moore-Penrose {#pseudo-inversa}

`01:21:55` 
dice che questo termine è chiamato la pseudo-inversa della matrice x. In realtà, formalmente la pseudo-inversa è quella che abbiamo laggiù, è questa. E cos'è la pseudo-inversa? Ricordate che x nella nostra applicazione è una matrice rettangolare. Quindi per.

`01:22:30` 
una matrice rettangolare, non ha alcun senso definire l'inversa. Non è definita. La pseudo inversa sta generalizzando, è chiamata anche la matrice di Moore-Penrose. È una generalizzazione del concetto di inversa per qualsiasi matrice. Okay, quindi qui ho solo, quindi questa è la soluzione che abbiamo calcolato.

`01:23:35` 
OK, quindi questa è la diapositiva importante. Poi le metterò insieme. Questa matrice è solitamente denotata da X più. E le proprietà delle proprietà importanti di questa matrice è che data una matrice quadrata rettangolare X,

`01:24:07` 
la pseudo-inversa X più è l'unica matrice che soddisfa queste quattro proprietà. La prima è che X più X è uguale a X, lo stesso con l'ordine inverso. E poi, x, x più, trasposta è uguale a x, x più, e anche l'altro modo.

`01:24:42` 
Se la matrice x è quadrata e invertibile, la pseudo-inversa è uguale all'inversa classica. Quindi, questa matrice è in realtà una generalizzazione dell'inversa di una matrice. E ricordate che quello che abbiamo considerato nel nostro problema era questo, okay?

`01:25:14` 
Quindi, in pratica, avevamo x trasposta, x, w cappello, uguale a X trasposta Y okay quindi questo era il normale l'insieme delle equazioni normali e questa è una matrice quadrata e vogliamo risolvere questo problema per risolvere.

`01:25:47` 
questo problema stiamo moltiplicando per l'inversa di questa matrice quadrata otteniamo questa matrice X trasposta X inversa X trasposta che è X più ed è una sorta di inversa di una matrice rettangolare okay.

`01:26:58` 
Quindi abbiamo visto la definizione della pseudo-inversa, qui c'è la rappresentazione della pseudo-inversa, la SVD della matrice originale e come potete vedere esattamente come abbiamo visto nell'altra presentazione la pseudo inversa di x è v sigma più dove sigma più è essenzialmente la pseudo inversa.

`01:27:38` 
di sigma ma se usate la SVD ridotta o economy dato che sigma è quadrata non è nient'altro che l'inversa di sigma che è una matrice diagonale e quindi è una matrice diagonale con elementi sulla diagonale uguali a uno su sigma i okay quindi esattamente questo e quello U trasposta okay.

`01:28:11` 
Che è esattamente quello che abbiamo qui. Qual è il problema? Quindi, siamo arrivati con una rappresentazione di W in termini della SVD ridotta o economy della matrice X originale.

### 8.5 Problema dei valori singolari piccoli {#valori-singolari-piccoli}

`01:28:42` 
In questa formula abbiamo sigma, sappiamo che è quadrata, sigma inversa. E quindi significa che sulla diagonale di quella matrice abbiamo uno su sigma I. Qual è il problema qui? Il problema è se avete un sigma I che è piccolo.

`01:29:14` 
forse molto, molto piccolo, avete questo termine, che può essere molto, molto grande. Quindi quali sono i problemi relativi a questo fatto? Prima di tutto, supponiamo che siate, ricordate che la matrice qui è applicata a y, okay?

`01:29:45` 
Quindi sappiamo che u è una matrice ortogonale, quindi applicata a y, non cambia la lunghezza, è solo una rotazione del vettore. Quindi l'applicazione di sigma inversa a questo vettore, è uno scaling, se nel vettore y, avete, Abbiamo detto etichette, ma potrebbero essere anche, come abbiamo detto prima, immaginate che nella matrice X, abbiate pressione, temperatura e concentrazione, e Y è il volume, o forse avete ancora più caratteristiche, e poi avete un valore corrispondente.

`01:30:35` 
Quindi non è un problema di classificazione, ma è un vero problema di regressione, okay? Poi i valori che avete in Y potrebbero essere affetti da qualche rumore, alcuni errori nella misurazione. Se qui avete un fattore di scala che è molto, molto grande, quello che può succedere è che state amplificando i possibili errori o rumore che è presente nel vettore Y originale.

`01:31:12` 
Okay, questo è il primo problema. In altri termini, significa che il problema è mal-condizionato, e dal punto di vista della scienza dei dati, la soluzione che avete,

### 9. Regolarizzazione Ridge (L2) {#regolarizzazione-ridge}

### 9.1 Problema del mal-condizionamento {#mal-condizionamento}

`01:31:43` 
la soluzione significa che il vettore w cappello è un vettore che può avere una norma molto grande. Questo corrisponde in pratica all'overfitting. Quindi state in qualche modo costruendo un modello che dato che sta in qualche modo seguendo troppo il rumore, i dati, sta facendo overfitting. E quindi non è molto buono per l'inferenza, per le previsioni.

`01:32:16` 
E questo è un grande problema se usate la versione semplice della tecnica dei minimi quadrati. Quindi quali sono i possibili metodi per curare questo problema? Ricordate quello che abbiamo scritto qui. Il risultato potrebbe essere un vettore w cappello.

`01:32:49` 
che è molto grande in norma quindi un'idea potrebbe essere è possibile in qualche modo modificare il nostro problema originale per imporre qualcosa sulla norma di W questo è uno dei punti chiave non ho commentato sull'essere di un sistema mal-condizionato ma suppongo che sapete molto bene cosa non va okay quindi qual è l'idea.

### 9.2 Termine di penalizzazione λ||w||² {#termine-penalizzazione}

`01:33:25` 
## Regolarizzazione Ridge (L2)

**Nuova funzione obiettivo:**

$$
J_{\text{Ridge}}(w) = \|y - Xw\|_2^2 + \lambda \|w\|_2^2
$$

Dove:
- **Primo termine:** Errore di fitting (loss originale)
- **Secondo termine:** **Penalizzazione** sulla norma di $w$
- $\lambda > 0$: **Parametro di regolarizzazione** (iperparametro)

`01:34:14` 
**Interpretazione:**

Vogliamo minimizzare **sia** l'errore di predizione **sia** la lunghezza di $w$.

- $\lambda$ grande: Priorità a $w$ piccolo (rischio underfitting)
- $\lambda$ piccolo: Priorità a fit accurato (rischio overfitting)
- $\lambda = 0$: Torniamo a OLS standard

`01:34:46` 
**Vantaggi:**
1. **Stabilità numerica:** Anche se $X^T X$ è mal condizionata
2. **Previene overfitting:** Pesi grandi sono penalizzati
3. **Soluzione chiusa:** Esiste formula analitica!

`01:35:20` 
**Soluzione analitica (Equazioni Normali modificate):**

Derivando $J_{\text{Ridge}}$ e ponendo uguale a zero:

$$
\hat{w}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T y
$$

Notare: $X^T X + \lambda I$ è **sempre invertibile** anche se $X^T X$ è singolare!

`01:36:40` 
**Interpretazione via SVD:**

Se $X = U \Sigma V^T$, allora:

$$
\hat{w}_{\text{Ridge}} = \sum_{i=1}^p \frac{\sigma_i}{\sigma_i^2 + \lambda} u_i^T y \cdot v_i
$$

`01:37:25` 
**Effetto su valori singolari piccoli:**

Nella soluzione OLS: coefficiente $\frac{1}{\sigma_i}$ **esplode** se $\sigma_i \to 0$

Con Ridge: coefficiente $\frac{\sigma_i}{\sigma_i^2 + \lambda}$ **rimane limitato**:

$$
\frac{\sigma_i}{\sigma_i^2 + \lambda} \leq \frac{1}{\lambda}
$$

Questo **stabilizza** la soluzione rispetto a perturbazioni!

`01:38:02` 
il problema è ben-condizionato. E quindi, la soluzione che avete qui per il W-ridge è una soluzione migliore in termini di robustezza rispetto ai possibili piccoli valori di piccoli valori singolari. E qual è l'idea dietro la regressione ridge è trovare un W, che sia il più piccolo possibile in norma due, nella norma due.

### 9.5 Interpretazione: lunghezza minima {#interpretazione-ridge}

`01:38:50` 
Okay, quindi la norma due del vettore W è minimizzata in qualche modo. State cercando un W con una piccola lunghezza. Okay, questa è una possibilità. Quindi regressione ridge significa che state cercando di imporre un W, che è di lunghezza minima.

### 10. Regolarizzazione LASSO (L1) {#regolarizzazione-lasso}

### 10.1 Norma L1 e sparsità {#norma-l1}

`01:39:21` 
## Regolarizzazione LASSO (L1) - Least Absolute Shrinkage and Selection Operator

**Funzione obiettivo:**

$$
J_{\text{LASSO}}(w) = \|y - Xw\|_2^2 + \lambda \|w\|_1
$$

Dove $\|w\|_1 = \sum_{i=1}^p |w_i|$ (norma L1)

`01:39:51` 
**Differenza chiave con Ridge:**

| **Ridge (L2)** | **LASSO (L1)** |
|---|---|
| $\lambda \|w\|_2^2 = \lambda \sum w_i^2$ | $\lambda \|w\|_1 = \lambda \sum |w_i|$ |
| Pesi **piccoli** ma **non zero** | Pesi **esattamente zero** (sparsità) |
| Soluzione **chiusa** | Soluzione **iterativa** (no formula chiusa) |
| Lunghezza minima | **Feature selection automatica** |

`01:40:26` 
**Sparsità = Riduzione dimensionalità automatica:**

Se $w$ ha 20 componenti ma LASSO restituisce $\hat{w}$ con solo 8 valori non-zero:
- Le **12 features** corrispondenti ai pesi zero sono **irrilevanti**
- Possiamo **scartarle** senza perdere performance
- **Selezione automatica** delle features più importanti!

`01:41:01` 
non sono così importanti in pratica, okay? O potete ottenere un modello che è ancora significativo, considerando solo otto su 20. Qual è... Il, questo tipo di regolarizzazione è chiamato LASSO, L-A-S-S-O, e quindi il, lasciatemi, prima, questa è un'immagine di cosa sta succedendo.

### 10.3 Visualizzazione geometrica: norma L2 vs L1 {#visualizzazione-geometrica}

`01:41:41` 
### Interpretazione Geometrica: Level Sets

**Formulazione equivalente (problema vincolato):**

Minimizzare $\|y - Xw\|_2^2$ soggetto a $\|w\| \leq t$ (con $t$ dipendente da $\lambda$)

`01:42:16` 
**Visualizzazione 2D:**

```
Ridge (L2):              LASSO (L1):

   w₂                      w₂
    │   ○○○                 │    /\
    │  ○ ● ○                │   /  \
    │   ○○○                 │  /____\
    └───── w₁              └───── w₁
  (cerchi)              (rombi/quadrati)
```

- **Ellissi concentriche:** Livelli di $\|y - Xw\|_2^2$
- **Vincolo L2:** $w_1^2 + w_2^2 \leq t$ (cerchio)
- **Vincolo L1:** $|w_1| + |w_2| \leq t$ (rombo)

`01:43:19` 
**Soluzione ottima:** Punto di **contatto** tra ellisse e regione vincolata

- **Ridge:** Contatto in punto **generico** → $w_1, w_2 \neq 0$
- **LASSO:** Contatto spesso su **spigoli** → $w_1 = 0$ o $w_2 = 0$ (**sparsità**!)

Gli **spigoli** del rombo L1 inducono naturalmente soluzioni sparse!

`01:43:51` 
Quindi è un problema di minimizzazione vincolata, essenzialmente. OK, e come potete vedere, il vettore di pesi con cui stiamo venendo fuori è un vettore in cui abbiamo entrambe le componenti. In questo caso, sono chiamate theta uno e theta due che sono diverse da zero. OK, è un vettore. Poi se andiamo alla norma L1. La rappresentazione della L1, l'insieme di livello della norma L1 di un vettore, invece di essere cerchi, come per la norma due, sono quadrati, okay?

### 10.4 Level sets: cerchi vs quadrati {#level-sets}

`01:44:35` 
Perché ricordate che la norma L1 è la somma dei valori assoluti. Quindi l'insieme di livello della norma sono quadrati centrati nell'origine. Quindi abbiamo questo tipo di caratteristica. Come potete vedere ora, il minimo è il punto verde. Qual è la peculiarità di quel punto? È che in quel caso, solo una componente del vettore theta è diversa da zero, che è theta2.

### 10.5 Soluzione sparsa {#soluzione-sparsa}

`01:45:09` 
Theta1 è zero. Questa è la sparsità, perché in quel caso, abbiamo ottenuto, abbiamo detto che vogliamo trovare il minimo del residuo soggetto al vincolo che la norma uno di W è in qualche modo minimizzata. Ma questo problema di minimizzazione vincolata equivale a trovare questa soluzione, che è una soluzione che essenzialmente scarta la prima componente di W di theta.

`01:45:44` 
Quindi invece di avere due componenti per W, avete solo una. In altri termini, significa che il vostro problema che inizialmente era caratterizzato da due caratteristiche, quando derivate il vostro modello, solo una. Una sarà importante. In questo caso, la seconda cosa in pratica, succede che. Questo è buono perché ottiene la sparsità.

### 11. Elastic Net {#elastic-net}

### 11.1 Combinazione di L1 e L2 {#combinazione-l1-l2}

`01:46:18` 
## Elastic Net: Combinazione L1 + L2

**Funzione obiettivo:**

$$
J_{\text{ElasticNet}}(w) = \|y - Xw\|_2^2 + \lambda \left( \alpha \|w\|_1 + (1-\alpha) \|w\|_2^2 \right)
$$

Dove:
- $\lambda > 0$: Intensità complessiva della regolarizzazione
- $\alpha \in [0,1]$: **Mixing parameter** tra L1 e L2

`01:46:54` 
**Casi speciali:**

| $\alpha$ | Metodo | Effetto |
|---------|--------|--------|
| $\alpha = 0$ | **Ridge puro** | Solo lunghezza minima |
| $\alpha = 1$ | **LASSO puro** | Solo sparsità |
| $0 < \alpha < 1$ | **Elastic Net** | Compromesso |
| $\lambda = 0$ | **OLS** | Nessuna regolarizzazione |

**Vantaggi:**
1. **Feature selection** (da L1) + **Stabilità** (da L2)
2. Gestisce **gruppi di features correlate** meglio di LASSO
3. **Flessibilità:** Due iperparametri per tuning fine

`01:47:27` 
E potete giocare con il parametro dell'elastic net per ottenere la vera elastic net se lambda e alpha. Quindi se alpha non è uno o zero, o potete recuperare o il lasso o il ridge se sono o uno o zero. E ovviamente, se scegliete lambda uguale a zero, state solo recuperando il metodo semplice originale.

`01:48:01` 
Okay, quindi in... In pratica, il metodo dei minimi quadrati è sempre risolto usando qualche tipo di regolarizzazione. E l'idea di regolarizzazione sarà di fondamentale importanza anche quando considereremo il problema di minimizzazione nel contesto della rete neurale. Anche, in quel caso, avete una funzione da minimizzare, che è la funzione di costo che volete considerare per la vostra rete neurale.

`01:48:40` 
E in quel caso pure, l'introduzione di qualche regolarizzazione è... Qualcosa che in, direi, il 90% dei casi permetterà di risolvere efficientemente il problema. Quindi questa è una prima istanza dell'uso di questa tecnica, che è in realtà diffusa in molti altri contesti del deep learning e del machine learning.

### 11.3 Confronto: Ridge vs LASSO vs Elastic Net {#confronto-metodi}

`01:49:16` 
## Tabella di Confronto: Ridge vs LASSO vs Elastic Net

| **Metodo** | **Penalizzazione** | **Obiettivo** | **Sparsità** | **Soluzione** |
|------------|-------------------|---------------|---------------|---------------|
| **OLS** | Nessuna | Fit esatto | No | $(X^T X)^{-1} X^T y$ |
| **Ridge (L2)** | $\lambda \|w\|_2^2$ | Lunghezza minima | No | $(X^T X + \lambda I)^{-1} X^T y$ |
| **LASSO (L1)** | $\lambda \|w\|_1$ | Sparsità | **Sì** | Iterativa (soft-thresholding) |
| **Elastic Net** | $\lambda(\alpha \|w\|_1 + (1-\alpha)\|w\|_2^2)$ | Compromesso | **Sì** | Iterativa |

`01:49:47` 
**Quando usare ciascun metodo:**

- **Ridge:** Features correlate, tutte potenzialmente importanti, problema mal condizionato
- **LASSO:** Feature selection, molte features irrilevanti, interpretabilità
- **Elastic Net:** Gruppi di features correlate + desiderio di sparsità
- **OLS:** Solo se $n >> p$, problema ben condizionato, no overfitting

`01:50:20` 
### Selezione degli Iperparametri

**Problema:** Come scegliere $\lambda$ (e $\alpha$ per Elastic Net)?

**Approcci:**

1. **Cross-Validation (CV):**
   - Griglia di valori: $\lambda \in \{10^{-4}, 10^{-3}, \ldots, 10^2\}$
   - k-fold CV (tipicamente k=5 o k=10)
   - Scegli $\lambda$ che minimizza errore di validazione

2. **Information Criteria:**
   - AIC (Akaike Information Criterion)
   - BIC (Bayesian Information Criterion)

3. **Path Algorithms:**
   - Calcola soluzioni per **tutti** i $\lambda$ efficientemente
   - Esempio: LARS (Least Angle Regression) per LASSO

`01:51:04` 
**Nota importante:**

Non esiste "ricetta universale"! La scelta dipende da:
- Struttura del dataset (correlazioni tra features)
- Obiettivo (predizione vs interpretabilità)
- Constraints computazionali

**Regola pratica:** Inizia con Elastic Net ($\alpha = 0.5$) e CV, poi specializza se necessario.

### 12. Riepilogo e Comunicazioni {#riepilogo}

`01:52:04` 
Okay, forse possiamo fermarci qui per oggi perché devo iniziare un altro grande argomento, quindi non ho tempo. Domande? Solo un messaggio. Durante la pausa, qualcuno mi ha detto che non era totalmente chiaro il fatto che venerdì avete lab. Questo sarà una costante per tutto il semestre. Venerdì, a meno di comunicazione specifica,

`01:52:38` 
avrete sempre lab. Okay.

---

## 13. Implementazione Completa: PCA in Python {#implementazione-pca}

### 13.1 Setup Dati Sintetici con Rotazione

**Codice Completo dal Notebook del Corso:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Configurazione figure
plt.rcParams['figure.figsize'] = [16, 8]

# Parametri cloud di punti 2D
xC = np.array([2, 1])          # Centro dei dati (media)
sig = np.array([2, 0.5])       # Assi principali (deviazioni standard)
theta = np.pi/3                # Rotazione di π/3 (60°)

# Matrice di rotazione R(θ)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Generazione 10,000 punti gaussiani
nPoints = 10000
X = (R @ np.diag(sig) @ np.random.randn(2, nPoints) + 
     np.diag(xC) @ np.ones((2, nPoints)))
```

**Spiegazione Matematica:**
1. Genera punti $\mathcal{N}(0, I)$ standard 2D
2. Scala con $\Sigma = \text{diag}(2, 0.5)$ (ellisse con assi 2 e 0.5)
3. Ruota con $R(\pi/3)$
4. Trasla al centro $(2, 1)$

**Risultato:** Cloud ellittico ruotato di 60°, centrato in (2,1)

---

### 13.2 Visualizzazione Dati Originali

```python
# Plot 1: Dati grezzi
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0,:], X[1,:], 'k.', markersize=1)
ax1.grid()
ax1.set_xlim((-6, 8))
ax1.set_ylim((-6, 8))
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('Raw Data')
ax1.set_aspect('equal')
```

**Interpretazione Visuale:**
- Cloud ellittico ruotato
- Centro NON in origine
- Asse maggiore: direzione di massima varianza
- Asse minore: direzione di minima varianza

---

### 13.3 Centramento Dati (Mean-Subtraction)

```python
# Step 1: Calcola media campionaria
Xavg = np.mean(X, axis=1)  # Media per ogni feature (riga)
print(f"Media: {Xavg}")    # ≈ [2, 1]

# Step 2: Centra dati (sottrai media)
B = X - np.tile(Xavg, (nPoints, 1)).T

# Metodo alternativo con centering matrix (meno efficiente)
# H = np.eye(nPoints) - (1/nPoints) * np.ones((nPoints, nPoints))
# B = X @ H

# Verifica: media di B deve essere ≈ 0
print(f"Media B: {np.mean(B, axis=1)}")  # ≈ [0, 0]
```

**Perché Centrare?**
- PCA cerca direzioni di **massima varianza**
- Varianza definita rispetto alla media
- Centramento trasla cloud nell'origine
- **Matematicamente:** $B = X - \bar{x}\mathbf{1}^T$ dove $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$

**Centering Matrix (Approccio Matriciale):**
$$
H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T
$$
$$
B = X H
$$

**Proprietà di $H$:**
- Simmetrica: $H^T = H$
- Idempotente: $H^2 = H$
- $H\mathbf{1} = \mathbf{0}$ (annulla componente costante)

---

### 13.4 SVD su Dati Centrati (PCA via SVD)

```python
# SVD su matrice centrata e normalizzata
# Divisione per √(n-1) → varianza campionaria unbias
U, S, VT = np.linalg.svd(B / np.sqrt(nPoints - 1), full_matrices=False)

print(f"U shape: {U.shape}")   # (2, 2) - Componenti principali
print(f"S shape: {S.shape}")   # (2,)   - Deviazioni standard
print(f"VT shape: {VT.shape}") # (2, 10000) - Coefficienti proiezione
```

**Interpretazione Fattori SVD per PCA:**

| Fattore | Dimensione | Significato PCA |
|---------|------------|-----------------|
| **U** | $d \times d$ | **Componenti principali** (eigenvectors di $C$) |
| **S** | $d$ | **Deviazioni standard** lungo assi principali |
| **VT** | $d \times n$ | **Scores** (coefficienti proiezione punti) |

**Relazione con Matrice di Covarianza:**
$$
C = \frac{1}{n-1}B B^T = U \Sigma^2 U^T
$$

Dove:
- $U$: eigenvectors di $C$ (direzioni assi principali)
- $\Sigma^2 = \text{diag}(s_1^2, s_2^2, \ldots)$: eigenvalues (varianze)

**Varianza Spiegata:**
$$
\text{Var explained by PC}_i = \frac{s_i^2}{\sum_{j=1}^d s_j^2}
$$

---

### 13.5 Visualizzazione PCA: Assi Principali e Ellissi

```python
# Plot 2: Dati + PCA overlay
ax2 = fig.add_subplot(122)
ax2.plot(X[0,:], X[1,:], 'k.', markersize=1, alpha=0.5)
ax2.grid()
ax2.set_xlim((-6, 8))
ax2.set_ylim((-6, 8))
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')
ax2.set_title('PCA: Principal Components & Confidence Intervals')
ax2.set_aspect('equal')

# Ellissi di confidenza (1σ, 2σ, 3σ)
theta_circle = 2 * np.pi * np.arange(0, 1, 0.01)

# Punti ellisse unitaria nel sistema PC
circle_PC = np.array([np.cos(theta_circle), np.sin(theta_circle)])

# Trasforma ellisse nel sistema originale
for k, color in zip([1, 2, 3], ['red', 'orange', 'yellow']):
    Xstd = U @ np.diag(S) @ circle_PC
    ax2.plot(Xavg[0] + k * Xstd[0,:], 
             Xavg[1] + k * Xstd[1,:], 
             '-', color=color, linewidth=2, 
             label=f'{k}σ confidence')

# Plot assi principali (vettori U[:,i] scalati per S[i])
# PC1: Asse maggiore
ax2.arrow(Xavg[0], Xavg[1], 
          U[0,0] * S[0], U[1,0] * S[0],
          head_width=0.3, head_length=0.4, 
          fc='cyan', ec='cyan', linewidth=3,
          label='PC1 (1st principal component)')

# PC2: Asse minore
ax2.arrow(Xavg[0], Xavg[1], 
          U[0,1] * S[1], U[1,1] * S[1],
          head_width=0.3, head_length=0.4, 
          fc='magenta', ec='magenta', linewidth=3,
          label='PC2 (2nd principal component)')

# Centro (media)
ax2.plot(Xavg[0], Xavg[1], 'ro', markersize=10, label='Mean')

ax2.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
```

**Elementi della Visualizzazione:**

1. **Punti neri**: Dati originali $X$
2. **Punto rosso**: Media $\bar{x} = (2, 1)$
3. **Frecce cyan/magenta**: Componenti principali $u_1 s_1$, $u_2 s_2$
4. **Ellissi colorate**: Intervalli di confidenza $k\sigma$ (k=1,2,3)

**Interpretazione Geometrica:**
- **PC1** (cyan): Direzione di **massima varianza** (lunghezza $s_1 \approx 2$)
- **PC2** (magenta): Direzione **ortogonale** a PC1 (lunghezza $s_2 \approx 0.5$)
- **Ellissi**: Contengono ~68%, ~95%, ~99.7% dei dati (regola 1-2-3 sigma)

---

### 13.6 Verifica Numerica: Recupero Parametri Originali

```python
# Parametri originali
print("=== PARAMETRI ORIGINALI ===")
print(f"Centro: {xC}")
print(f"Assi: {sig}")
print(f"Angolo rotazione: {theta:.4f} rad = {np.degrees(theta):.1f}°")

# Parametri recuperati da PCA
print("\n=== PARAMETRI RECUPERATI (PCA) ===")
print(f"Centro (media): {Xavg}")
print(f"Assi (std): {S}")

# Angolo PC1 rispetto a x-axis
angle_PC1 = np.arctan2(U[1,0], U[0,0])
print(f"Angolo PC1: {angle_PC1:.4f} rad = {np.degrees(angle_PC1):.1f}°")

# Differenze (dovute a rumore statistico)
print("\n=== ERRORI ===")
print(f"Errore centro: {np.linalg.norm(Xavg - xC):.4f}")
print(f"Errore assi: {np.linalg.norm(S - sig):.4f}")
print(f"Errore angolo: {np.abs(angle_PC1 - theta):.4f} rad")
```

**Output Atteso:**
```
=== PARAMETRI ORIGINALI ===
Centro: [2 1]
Assi: [2.  0.5]
Angolo rotazione: 1.0472 rad = 60.0°

=== PARAMETRI RECUPERATI (PCA) ===
Centro (media): [2.001  0.998]
Assi (std): [2.003  0.501]
Angolo PC1: 1.0485 rad = 60.1°

=== ERRORI ===
Errore centro: 0.0022
Errore assi: 0.0032
Errore angolo: 0.0013 rad
```

**Conclusione:** PCA recupera quasi perfettamente i parametri originali!

---

### 13.7 Riduzione Dimensionalità: Proiezione su PC1

```python
# Proiezione dati su primo componente principale
# Scores: coefficienti nella base PC
scores_PC1 = U[:, 0].T @ B  # (n,) - Coordinata lungo PC1

# Ricostruzione approssimata (rank-1)
X_approx = np.outer(U[:, 0] * S[0], scores_PC1) + Xavg[:, None]

# Plot confronto
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Dati originali
ax1.plot(X[0,:], X[1,:], 'k.', markersize=1, alpha=0.5)
ax1.plot(Xavg[0], Xavg[1], 'ro', markersize=10)
ax1.arrow(Xavg[0], Xavg[1], U[0,0]*S[0], U[1,0]*S[0],
          head_width=0.3, fc='cyan', ec='cyan', linewidth=3)
ax1.set_title('Original Data')
ax1.set_aspect('equal')
ax1.grid()

# Proiezione su PC1 (1D)
ax2.hist(scores_PC1, bins=50, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Score along PC1')
ax2.set_ylabel('Frequency')
ax2.set_title('Projection on PC1 (1D representation)')
ax2.grid()

# Ricostruzione approssimata
ax3.plot(X_approx[0,:], X_approx[1,:], 'b.', markersize=1, alpha=0.5)
ax3.plot(Xavg[0], Xavg[1], 'ro', markersize=10)
ax3.arrow(Xavg[0], Xavg[1], U[0,0]*S[0], U[1,0]*S[0],
          head_width=0.3, fc='cyan', ec='cyan', linewidth=3)
ax3.set_title('Rank-1 Approximation (PC1 only)')
ax3.set_aspect('equal')
ax3.grid()

plt.tight_layout()
plt.show()

# Errore ricostruzione
reconstruction_error = np.linalg.norm(X - X_approx, 'fro') / np.linalg.norm(X, 'fro')
print(f"Errore ricostruzione (relativo): {reconstruction_error:.4f}")

# Varianza spiegata da PC1
var_explained = S[0]**2 / np.sum(S**2)
print(f"Varianza spiegata da PC1: {var_explained:.2%}")
```

**Interpretazione:**
- **PC1** cattura ~94% della varianza (dato che $s_1 = 2 >> s_2 = 0.5$)
- **Errore ~25%**: Perdiamo info lungo PC2
- **1D representation**: Istogramma mostra distribuzione lungo asse principale

---

## 14. Formule Complete PCA {#formule-complete-pca}

### 14.1 Algoritmo PCA Classico (via Matrice Covarianza)

**Input:** Dataset $X \in \mathbb{R}^{d \times n}$ (d features, n samples)

**Step 1:** Centra dati
$$
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \quad B = X - \bar{x}\mathbf{1}^T
$$

**Step 2:** Calcola matrice covarianza
$$
C = \frac{1}{n-1}B B^T \in \mathbb{R}^{d \times d}
$$

**Step 3:** Diagonalizza $C$ (autovalori/autovettori)
$$
C = U \Lambda U^T, \quad \Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)
$$

Dove:
- $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ (autovalori ordinati)
- $U = [u_1 | u_2 | \cdots | u_d]$ (autovettori ortonormali)

**Step 4:** Componenti principali
$$
\text{PC}_i = u_i \quad (i = 1, 2, \ldots, d)
$$

**Step 5:** Proiezione su k componenti
$$
Z = U_k^T B \in \mathbb{R}^{k \times n}
$$

Dove $U_k = [u_1 | \cdots | u_k]$ (prime k colonne di U)

**Step 6:** Ricostruzione approssimata
$$
\tilde{X} = U_k Z + \bar{x}\mathbf{1}^T = U_k U_k^T B + \bar{x}\mathbf{1}^T
$$

---

### 14.2 PCA via SVD (Metodo Stabile)

**Alternativa:** Evitare calcolo esplicito di $C = BB^T$ (mal-condizionato!)

**Step 1-2:** Identici (centra dati)

**Step 3:** SVD su $B$ (thin SVD)
$$
\frac{B}{\sqrt{n-1}} = U S V^T
$$

**Relazioni:**
- Eigenvectors di $C$: $u_i$ (colonne di $U$)
- Eigenvalues di $C$: $\lambda_i = s_i^2$ (quadrati valori singolari)
- Varianza lungo PC$_i$: $\lambda_i = s_i^2$

**Step 4-6:** Identici, usando $U$ e $S$ da SVD

**Vantaggi SVD:**
- ✅ Stabile numericamente (no $X^T X$)
- ✅ Complessità $O(dn^2)$ invece di $O(d^3)$ per grandi $d$
- ✅ Fornisce direttamente scores: $Z = S V^T$

---

### 14.3 Varianza Spiegata e Scelta di k

**Varianza totale:**
$$
\text{Var}_{\text{tot}} = \sum_{i=1}^d \lambda_i = \sum_{i=1}^d s_i^2 = \text{tr}(C)
$$

**Varianza spiegata da k componenti:**
$$
\text{Var}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j} = \frac{\sum_{i=1}^k s_i^2}{\sum_{j=1}^d s_j^2}
$$

**Scelta k:** Fissare soglia (es. 95% varianza)
$$
k^* = \min\left\{k : \text{Var}_k \geq 0.95\right\}
$$

**Errore ricostruzione (norma Frobenius):**
$$
\|X - \tilde{X}\|_F = \sqrt{\sum_{i=k+1}^d s_i^2}
$$

**Teorema Eckart-Young:** $\tilde{X}$ è la migliore approssimazione rank-k di $X$

---

## 15. PCA vs Regressione: Differenze Chiave {#pca-vs-regressione-dettagli}

### 15.1 Obiettivi Diversi

| Aspetto | PCA | Regressione Lineare |
|---------|-----|---------------------|
| **Obiettivo** | Massimizzare **varianza** proiettata | Minimizzare **errore predizione** |
| **Distanza** | **Ortogonale** a PC | **Verticale** rispetto a $y$ |
| **Variabili** | **Simmetriche** (tutte features) | **Asimmetriche** ($y$ target, $X$ features) |
| **Output** | Sottospazio k-dimensionale | Funzione $f(X) = w^T X + b$ |
| **Uso** | Riduzione dim., visualizzazione | Predizione, inferenza |

### 15.2 Distanze: Ortogonale vs Verticale

**PCA (Distanza Ortogonale):**
- Minimizza: $\sum_{i=1}^n \|x_i - \pi_{U_k}(x_i)\|^2$
- Dove $\pi_{U_k}(x) = U_k U_k^T (x - \bar{x})$ è proiezione su sottospazio PC

**Regressione (Distanza Verticale):**
- Minimizza: $\sum_{i=1}^n (y_i - w^T x_i)^2$
- Solo errore nella direzione $y$ (target)

**Esempio Visuale 2D:**
```
      y
      |
    * | *  (dati)
   *  |  *
  *   |---* (punto)
 *    |    \
------+-----x
      |      \ (distanza ortogonale PCA)
      |(verticale regr.)
```

### 15.3 Simmetria: Quando PCA ≈ Regressione?

**Se:**
1. Dati "sottili" lungo una direzione (rank ≈ 1)
2. PC1 quasi allineato con asse $y$

**Allora:** Soluzione PCA ≈ Soluzione regressione

**Altrimenti:** Soluzioni MOLTO diverse!

---

## 16. Pseudo-Inversa di Moore-Penrose {#pseudo-inversa-dettagli}

### 16.1 Definizione Completa

**Pseudo-inversa di $A \in \mathbb{R}^{m \times n}$:**
$$
A^+ = V \Sigma^+ U^T
$$

Dove $A = U \Sigma V^T$ (SVD) e:
$$
\Sigma^+ = \begin{bmatrix}
\frac{1}{\sigma_1} & 0 & \cdots & 0 \\
0 & \frac{1}{\sigma_2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \frac{1}{\sigma_r} \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}_{n \times m}
$$

**Rango:** $r = \text{rank}(A)$ (numero valori singolari $\neq 0$)

### 16.2 Proprietà (Assiomi di Moore-Penrose)

1. **$A A^+ A = A$** (idempotenza a sinistra)
2. **$A^+ A A^+ = A^+$** (idempotenza a destra)
3. **$(A A^+)^T = A A^+$** (simmetria proiezione range)
4. **$(A^+ A)^T = A^+ A$** (simmetria proiezione null-space)

### 16.3 Casi Speciali

**Matrice invertibile ($m = n$, $\text{rank}(A) = n$):**
$$
A^+ = A^{-1}
$$

**Matrice tall ($m > n$, $\text{rank}(A) = n$):**
$$
A^+ = (A^T A)^{-1} A^T \quad \text{(left inverse)}
$$

**Matrice wide ($m < n$, $\text{rank}(A) = m$):**
$$
A^+ = A^T (A A^T)^{-1} \quad \text{(right inverse)}
$$

**Matrice rank-deficient:** Usa formula SVD completa

### 16.4 Soluzione Minimi Quadrati via Pseudo-Inversa

**Problema:** $\min_w \|Aw - b\|^2$

**Soluzione generale:**
$$
w^* = A^+ b
$$

**Proprietà:**
1. Se sistema compatibile ($b \in \text{range}(A)$): $w^*$ soluzione esatta
2. Se sistema incompatibile: $w^*$ soluzione ai minimi quadrati
3. $w^*$ ha **norma minima** tra tutte le soluzioni LS

**Interpretazione Geometrica:**
$$
w^* = V \Sigma^+ U^T b = \sum_{i=1}^r \frac{u_i^T b}{\sigma_i} v_i
$$

Combinazione lineare di $v_i$ (vettori singolari destri) pesati per $\frac{u_i^T b}{\sigma_i}$

---

## 17. Regolarizzazione: Analisi Completa {#regolarizzazione-analisi}

### 17.1 Problema del Mal-Condizionamento

**Numero di condizione:**
$$
\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \frac{\sigma_1}{\sigma_r}
$$

**Problema:**
- Se $\kappa(A) \gg 1$: matrice **mal-condizionata**
- Piccoli $\sigma_i$ → denominatori grandi in $A^+$
- Amplificano errori numerici in $b$

**Esempio:** $\sigma_r = 10^{-10}$
$$
w_r = \frac{u_r^T b}{\sigma_r} v_r \approx 10^{10} (u_r^T b) v_r
$$
Rumore in $b$ amplificato di $10^{10}$!

---

### 17.2 Ridge Regression (L2): Soluzione Completa

**Funzionale regolarizzato:**
$$
J_{\text{ridge}}(w) = \|Aw - b\|^2 + \lambda \|w\|^2
$$

**Gradiente:**
$$
\nabla J = 2A^T(Aw - b) + 2\lambda w = 0
$$

**Equazioni normali modificate:**
$$
(A^T A + \lambda I)w_{\text{ridge}} = A^T b
$$

**Soluzione esplicita:**
$$
w_{\text{ridge}} = (A^T A + \lambda I)^{-1} A^T b
$$

**Via SVD (forma stabile):**
$$
w_{\text{ridge}} = \sum_{i=1}^r \frac{\sigma_i}{\sigma_i^2 + \lambda} (u_i^T b) v_i
$$

**Confronto con OLS:**
$$
w_{\text{OLS}} = \sum_{i=1}^r \frac{1}{\sigma_i} (u_i^T b) v_i
$$

**Effetto di $\lambda$:**
- **$\lambda = 0$**: OLS classica ($\frac{\sigma_i}{\sigma_i^2} = \frac{1}{\sigma_i}$)
- **$\lambda > 0$**: Shrinkage ($\frac{\sigma_i}{\sigma_i^2 + \lambda} < \frac{1}{\sigma_i}$)
- **$\lambda \to \infty$**: $w_{\text{ridge}} \to 0$

**Shrinkage factors:**
$$
f_i(\lambda) = \frac{\sigma_i^2}{\sigma_i^2 + \lambda} \in [0, 1]
$$
- Grandi $\sigma_i$: $f_i \approx 1$ (poco shrinkage)
- Piccoli $\sigma_i$: $f_i \approx 0$ (forte shrinkage)

**Interpretazione:** Ridge "attenua" componenti con piccoli $\sigma_i$ (rumorosi)

---

### 17.3 LASSO (L1): Sparsità e Feature Selection

**Funzionale:**
$$
J_{\text{LASSO}}(w) = \|Aw - b\|^2 + \lambda \|w\|_1
$$

Dove $\|w\|_1 = \sum_{i=1}^n |w_i|$

**Caratteristiche:**
- ❌ **NON differenziabile** in $w_i = 0$ (gradiente subdifferenziale)
- ❌ **NO soluzione closed-form** (richiede algoritmi iterativi)
- ✅ **Soluzione sparsa** (molti $w_i^* = 0$)

**Algoritmi di Soluzione:**
1. **LARS** (Least Angle Regression)
2. **Coordinate Descent**
3. **Proximal Gradient Methods**

**Perché Sparsità?**

**Level sets della norma:**
- **L2**: Cerchi/sfere (regioni convesse lisce)
- **L1**: Rombi/iper-cubi (vertici sugli assi)

**Geometria:**
- Ellissi residuo (paraboloide) intersecano L1-ball nei **vertici**
- Vertici hanno coordinate $= 0$ → sparsità!

**Esempio 2D:**
```
      w₂
       |
   L1  |\  Livelli residuo (ellissi)
  ball | \
    /  |  o (ottimo LASSO, w₁=0)
   /__\|___w₁
       |
```

**Feature Selection Automatica:**
- $w_i^* = 0$ → Feature $i$ **non importante**
- Mantieni solo features con $w_i^* \neq 0$
- Riduzione dimensionalità **interpretabile**

---

### 17.4 Elastic Net: Il Meglio di Entrambi

**Funzionale:**
$$
J_{\text{EN}}(w) = \|Aw - b\|^2 + \lambda \left(\alpha \|w\|_1 + (1-\alpha) \|w\|^2\right)
$$

**Parametri:**
- $\lambda > 0$: Intensità regolarizzazione totale
- $\alpha \in [0, 1]$: Mixing parameter

**Casi speciali:**
- $\alpha = 0$: Ridge pura
- $\alpha = 1$: LASSO puro
- $\alpha \in (0,1)$: Elastic Net vera

**Vantaggi:**
1. **Sparsità** (da L1): Feature selection
2. **Stabilità** (da L2): Gruppo-selezione (features correlate insieme)
3. **Flessibilità**: Interpola tra Ridge e LASSO

**Quando Usare:**
- **Ridge** ($\alpha \approx 0$): Multicollinearità, tutte features importanti
- **LASSO** ($\alpha \approx 1$): Molte features irrilevanti, serve selezione
- **Elastic Net** ($\alpha \approx 0.5$): Mix di features rilevanti/irrilevanti

---

## 18. Materiali e Riferimenti {#materiali-lab2}

### 18.1 Notebook del Corso

**`PCA_EX2024.ipynb`** (Lecture October 7th):
- Generazione dati sintetici 2D ruotati
- PCA via SVD con visualizzazione
- Ellissi di confidenza (1σ, 2σ, 3σ)
- Plot componenti principali
- Proiezione e ricostruzione rank-k

**Codice Chiave:**
```python
# Centramento
Xavg = np.mean(X, axis=1)
B = X - np.tile(Xavg, (nPoints, 1)).T

# SVD (PCA stabile)
U, S, VT = np.linalg.svd(B / np.sqrt(nPoints - 1), full_matrices=False)

# Componenti principali: colonne di U
# Deviazioni standard: S
# Scores: VT (o S*VT)
```

### 18.2 PDF Teorici (Lecture October 7th)

**⚠️ NOTA:** I PDF nella cartella sembrano essere slides introduttive del corso (info generali). Le derivazioni teoriche sono nella trascrizione video.

**Contenuti Teorici Trattati:**
1. **EY_proofs_pres.pdf**: Teorema Eckart-Young (dimostrazioni)
2. **LeastSquares_New.pdf**: Minimi quadrati e regolarizzazione
3. **PseudoInverse.pdf**: Pseudo-inversa di Moore-Penrose

### 18.3 Riferimenti a Figure (Conceptuali)

Sebbene i PDF disponibili non contengano le slide esatte della lezione, i concetti discussi sono visualizzabili come segue:

**Figura 1 - PCA Cloud Rotato:**
- Riferimento: `PCA_EX2024.ipynb`, plot subplot(121)
- Cloud ellittico 2D ruotato di 60°
- Centro in (2, 1), assi 2.0 e 0.5

**Figura 2 - PCA con Assi Principali:**
- Riferimento: `PCA_EX2024.ipynb`, plot subplot(122)
- Dati + ellissi 1σ/2σ/3σ (rosse/arancioni/gialle)
- Frecce cyan/magenta: PC1 e PC2

**Figura 3 - PCA vs Regressione (Conceptuale):**
- Trascrizione timestamp `00:35:51`
- Distanze ortogonali (PCA) vs verticali (regression)
- Due rette diverse per stesso dataset

**Figura 4 - Level Sets L2 vs L1 (Conceptuale):**
- Trascrizione timestamp `01:41:41` - `01:44:35`
- Cerchi (L2) vs quadrati (L1)
- Ellissi residuo che intersecano vincoli
- Soluzione Ridge: punto interno cerchio
- Soluzione LASSO: vertice quadrato (sparsità!)

---

## 19. Checklist Completa Lab 2 {#checklist-lab2}

### Teorema Eckart-Young

- [ ] **Enunciato**: $A_k = U_k \Sigma_k V_k^T$ ottimale in norma Frobenius
- [ ] **Dimostrazione**: Disuguaglianza di Weyl + espansione $\|A - A_k\|_F^2$
- [ ] **Risultato**: $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$

### PCA - Teoria

- [ ] **Centramento**: $B = X - \bar{x}\mathbf{1}^T$ (media = 0)
- [ ] **Matrice covarianza**: $C = \frac{1}{n-1}BB^T$
- [ ] **Eigenvectors $C$** = Componenti principali
- [ ] **Eigenvalues $C$** = Varianze lungo PC
- [ ] **PCA via SVD**: $\frac{B}{\sqrt{n-1}} = U S V^T$, $C = U S^2 U^T$

### PCA - Implementazione

- [ ] **Genera dati test**: Cloud 2D ruotato con `np.random.randn`
- [ ] **Centra**: `Xavg = mean(X, axis=1)`, `B = X - Xavg`
- [ ] **SVD**: `U, S, VT = svd(B/sqrt(n-1))`
- [ ] **Plot assi**: Frecce $\bar{x} + u_i s_i$
- [ ] **Ellissi confidenza**: $\bar{x} + U \text{diag}(S) \cdot \text{circle}$

### Minimi Quadrati

- [ ] **Problema**: $\min_w \|Aw - b\|^2$
- [ ] **Equazioni normali**: $A^T A w = A^T b$
- [ ] **Soluzione OLS**: $w = (A^T A)^{-1} A^T b$
- [ ] **Via SVD (stabile)**: $w = V \Sigma^{-1} U^T b$
- [ ] **Pseudo-inversa**: $A^+ = V \Sigma^+ U^T$

### Regolarizzazione

- [ ] **Ridge (L2)**: $\min \|Aw-b\|^2 + \lambda \|w\|^2$
  - Soluzione: $(A^T A + \lambda I)^{-1} A^T b$
  - Via SVD: $\sum \frac{\sigma_i}{\sigma_i^2 + \lambda} (u_i^T b) v_i$
  - Effetto: Shrinkage uniforme

- [ ] **LASSO (L1)**: $\min \|Aw-b\|^2 + \lambda \|w\|_1$
  - No closed-form (algoritmi iterativi)
  - Effetto: Sparsità ($w_i = 0$)
  
- [ ] **Elastic Net**: $\alpha \|w\|_1 + (1-\alpha) \|w\|^2$
  - Interpolazione Ridge-LASSO
  - Feature selection + stabilità

### Differenze PCA vs Regressione

- [ ] **PCA**: Massimizza varianza, distanze ortogonali, simmetrico
- [ ] **Regressione**: Minimizza errore, distanze verticali, asimmetrico
- [ ] **Quando diversi**: Quasi sempre! (tranne casi degeneri)

---

## 20. Esercizi Avanzati {#esercizi-avanzati-lab2}

### Esercizio 1: PCA su Dataset Reale (Iris)

```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Carica Iris dataset
iris = load_iris()
X = iris.data.T  # (4, 150) - 4 features, 150 samples
y = iris.target

# PCA
Xavg = np.mean(X, axis=1)
B = X - Xavg[:, None]
U, S, VT = np.linalg.svd(B / np.sqrt(X.shape[1] - 1), full_matrices=False)

# Proiezione 2D
Z = U[:, :2].T @ B  # (2, 150)

# Plot con colori per specie
plt.figure(figsize=(10, 8))
for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
    mask = (y == i)
    plt.scatter(Z[0, mask], Z[1, mask], label=species, s=50, alpha=0.7)

plt.xlabel(f'PC1 ({100*S[0]**2/np.sum(S**2):.1f}% var)')
plt.ylabel(f'PC2 ({100*S[1]**2/np.sum(S**2):.1f}% var)')
plt.title('Iris Dataset - PCA 2D Projection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# Varianza spiegata
for i in range(4):
    var_exp = 100 * S[i]**2 / np.sum(S**2)
    print(f"PC{i+1}: {var_exp:.2f}% varianza")
```

**Domande:**
1. Quanti PC servono per 95% varianza?
2. Le specie sono separabili in 2D?
3. Quale feature originale contribuisce di più a PC1?

---

### Esercizio 2: Ridge vs LASSO su Dati Sintetici

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

# Genera dati con features correlate
np.random.seed(42)
n, d = 100, 20
X = np.random.randn(n, d)
# Features 0-4 rilevanti, resto rumore
w_true = np.zeros(d)
w_true[:5] = [3, -2, 1.5, -1, 0.5]

y = X @ w_true + 0.5 * np.random.randn(n)

# Fit Ridge e LASSO con vari λ
lambdas = np.logspace(-3, 2, 50)
coefs_ridge = []
coefs_lasso = []

for lam in lambdas:
    ridge = Ridge(alpha=lam).fit(X, y)
    lasso = Lasso(alpha=lam, max_iter=10000).fit(X, y)
    coefs_ridge.append(ridge.coef_)
    coefs_lasso.append(lasso.coef_)

coefs_ridge = np.array(coefs_ridge)
coefs_lasso = np.array(coefs_lasso)

# Plot coefficienti
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for i in range(d):
    ax1.plot(lambdas, coefs_ridge[:, i], label=f'w{i}' if i < 5 else None)
    ax2.plot(lambdas, coefs_lasso[:, i], label=f'w{i}' if i < 5 else None)

for ax, title in zip([ax1, ax2], ['Ridge (L2)', 'LASSO (L1)']):
    ax.set_xscale('log')
    ax.set_xlabel('λ (regularization)')
    ax.set_ylabel('Coefficient value')
    ax.set_title(f'{title}: Coefficient Paths')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# Sparsità vs λ
sparsity_lasso = [np.sum(coef == 0) for coef in coefs_lasso]
plt.figure(figsize=(10, 6))
plt.plot(lambdas, sparsity_lasso, 'b-', linewidth=2)
plt.xscale('log')
plt.xlabel('λ (regularization)')
plt.ylabel('Number of zero coefficients')
plt.title('LASSO: Sparsity vs Regularization')
plt.grid(True, alpha=0.3)
plt.show()
```

**Osservazioni:**
- **Ridge**: Shrinkage graduale, NO zeri esatti
- **LASSO**: Coefficienti → 0 per λ crescente (sparsità!)
- **Ottimale λ**: Cross-validation (prossimo lab)

---

### Esercizio 3: Confronto OLS vs Ridge su Dati Mal-Condizionati

```python
import numpy as np
import matplotlib.pyplot as plt

# Genera matrice mal-condizionata
n, d = 50, 10
U, _ = np.linalg.qr(np.random.randn(n, d))
V, _ = np.linalg.qr(np.random.randn(d, d))

# Valori singolari: geometricamente decrescenti
s = np.logspace(2, -8, d)  # κ(A) = 10^10 !
A = U @ np.diag(s) @ V.T

print(f"Condition number: {np.linalg.cond(A):.2e}")

# Vettore vero + rumore
w_true = np.random.randn(d)
y_clean = A @ w_true
y_noisy = y_clean + 0.01 * np.random.randn(n)

# Soluzioni OLS vs Ridge
w_ols = np.linalg.lstsq(A, y_noisy, rcond=None)[0]

lambdas_test = [0.01, 0.1, 1, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, lam in zip(axes.flatten(), lambdas_test):
    w_ridge = np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ y_noisy)
    
    ax.plot(w_true, 'go-', label='True', linewidth=2, markersize=8)
    ax.plot(w_ols, 'r^--', label='OLS', linewidth=2, markersize=6)
    ax.plot(w_ridge, 'bs-', label=f'Ridge (λ={lam})', linewidth=2, markersize=6)
    
    error_ols = np.linalg.norm(w_ols - w_true)
    error_ridge = np.linalg.norm(w_ridge - w_true)
    
    ax.set_title(f'λ = {lam}\nOLS error: {error_ols:.2f}, Ridge error: {error_ridge:.2f}')
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Conclusioni:**
- **OLS**: Oscillazioni enormi (instabile!)
- **Ridge con λ ottimale**: Vicino a vero coefficiente
- **Trade-off bias-variance**: λ troppo grande → underfitting

---

## 🎯 Punti Chiave Finali Lab 2

1. **Eckart-Young**: SVD troncata = approssimazione ottimale rank-k (norma F)

2. **PCA via SVD**: Metodo stabile, evita $X^T X$
   - $U$: Componenti principali (eigenvectors covarianza)
   - $S^2$: Varianze (eigenvalues covarianza)

3. **PCA ≠ Regressione**:
   - Distanze ortogonali vs verticali
   - Simmetria vs asimmetria variabili

4. **Pseudo-Inversa**: $A^+ = V \Sigma^+ U^T$
   - Soluzione LS a norma minima
   - Problemi se $\sigma_i \approx 0$

5. **Regolarizzazione essenziale**:
   - **Ridge (L2)**: Shrinkage, stabilità
   - **LASSO (L1)**: Sparsità, feature selection
   - **Elastic Net**: Meglio di entrambi!

6. **Selezione λ**: Cross-validation (prossime lezioni)

---

**Fine Lezione 10 - PCA e Regolarizzazione** 
