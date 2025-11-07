# Lez14-20ott - Matrix Completion e Netflix Problem

## 🎯 Obiettivi della Lezione

### Concetti Teorici
- Comprendere il problema del **Matrix Completion** e il caso Netflix
- Studiare la formulazione matematica con **osservazioni parziali**
- Analizzare il **rank** come vincolo strutturale per dati sparsi
- Apprendere il **rilassamento convesso** da rank minimization a nuclear norm minimization
- Studiare le **garanzie teoriche** (teorema di Candès-Recht)

### Algoritmi e Metodi
- Implementare l'algoritmo **Singular Value Thresholding (SVT)**
- Comprendere il **soft thresholding** sui valori singolari
- Analizzare la convergenza lineare dell'algoritmo SVT
- Introduzione ai **Support Vector Machines** (SVM) per classificazione e regressione

### Applicazioni Pratiche
- Sistemi di raccomandazione (Netflix, e-commerce)
- Video background subtraction
- Image inpainting
- System identification, genomics, finance

## 📚 Prerequisiti

### Matematica
- **Algebra Lineare**: SVD, valori singolari, rank di matrici, norme matriciali
- **Ottimizzazione Convessa**: funzioni convesse, rilassamento convesso, minimizzazione vincolata
- **Analisi**: norme (Frobenius, nucleare), convergenza di algoritmi iterativi

### Teoria
- **Singular Value Decomposition (SVD)**: da lezioni precedenti
- **Norme Matriciali**: norma di Frobenius, norma operatore
- **Teorema di Perron-Frobenius**: autovalori dominanti

## 📑 Indice Completo

### [Parte 1: Matrix Completion e Netflix Problem](#parte-1-matrix-completion-e-netflix-problem) (`00:00` - `19:43`)
1. [Introduzione al Matrix Completion](#introduzione-al-matrix-completion) - `00:00:32`
2. [Il Caso Netflix: Rating Matrix Sparsa](#il-caso-netflix-rating-matrix-sparsa) - `00:02:15`
3. [Low-Rank Structure e Fattori Latenti](#low-rank-structure-e-fattori-latenti) - `00:04:47`
4. [Gradi di Libertà: 2nR - R² vs n²](#gradi-di-libertà-2nr---r²-vs-n²) - `00:07:33`
5. [Formulazione del Problema](#formalizzazione-del-problema) - `00:10:23`
6. [Projection Operator P_Ω](#projection-operator-p_ω) - `00:12:58`
7. [Rank Minimization (NP-Hard)](#rank-minimization-np-hard) - `00:15:44`

### [Parte 2: Rilassamento Convesso e Nuclear Norm](#parte-2-rilassamento-convesso-e-nuclear-norm) (`19:43` - `44:02`)
8. [Rilassamento Convesso](#rilassamento-convesso-e-norma-nucleare) - `00:17:36`
9. [Nuclear Norm: ||M||_* = Σσᵢ](#nuclear-norm-m_--σσᵢ) - `00:20:14`
10. [Parallelismo: L1 Norm ↔ Nuclear Norm](#parallelismo-l1-norm--nuclear-norm) - `00:23:05`
11. [Teorema di Candès-Recht (2008)](#teoria-del-matrix-completion) - `00:31:22`
12. [Required Samples: m ≥ C·n^1.25·R·log(n)](#required-samples-m--cn125rlogn) - `00:34:09`

### [Parte 3: Algoritmo SVT](#parte-3-algoritmo-svt) (`44:02` - `01:07:36`)
13. [Singular Value Thresholding](#algoritmo-singular-value-thresholding) - `00:36:46`
14. [Soft Thresholding: S_τ(x)](#soft-thresholding-s_τx) - `00:39:22`
15. [Operator D_τ(Y): SVD + Threshold + Reconstruct](#operator-d_τy-svd--threshold--reconstruct) - `00:42:38`
16. [Equivalenza con Ottimizzazione](#equivalenza-con-ottimizzazione) - `00:46:11`
17. [Procedura Iterativa](#procedura-iterativa) - `00:49:27`
18. [Convergenza: δ ∈ (0,2)](#convergenza-δ--02) - `00:52:55`
19. [Complessità: O(k·r·n·d)](#complessità-okrnd) - `00:56:18`
20. [Applicazioni Pratiche](#applicazioni-pratiche) - `00:59:44`

### [Parte 4: Support Vector Machines (Introduzione)](#parte-4-support-vector-machines-introduzione) (`01:07:36` - `01:23:22`)
21. [Support Vector Machines](#support-vector-machines) - `01:07:36`
22. [Classificazione: Max Margin](#classificazione-max-margin) - `01:09:52`
23. [SVM per Regressione](#svm-per-regressione) - `01:13:38`
24. [Epsilon-Tube e Support Vectors](#epsilon-tube-e-support-vectors) - `01:16:36`
25. [Representer Theorem](#representer-theorem) - `01:19:51`

---

## Parte 1: Matrix Completion e Netflix Problem

---

## Parte 1: Matrix Completion e Netflix Problem

## Introduzione al Matrix Completion

`00:00:32` 
Okay, alla fine dell'ultima lezione, abbiamo introdotto un nuovo problema, che si chiama problema generale di completamento di matrici. E abbiamo introdotto il problema con un esempio, un esempio molto significativo, che è il cosiddetto problema Netflix.

**Context: Il Problema Netflix**

Nel 2006, Netflix lanciò la **Netflix Prize**, una competizione per migliorare il loro sistema di raccomandazione film del 10%. Il dataset consisteva in:
- **~480,000 utenti**
- **~17,000 film**
- **~100 milioni di rating** (scala 1-5)
- **Rating matrix** molto **sparsa**: solo ~1.2% di valori osservati!

**Obiettivo**: Predire i rating mancanti con alta accuratezza per suggerire film agli utenti.

**Sfida**: Con 480K × 17K = **8.16 miliardi** di posizioni potenziali, ma solo **100M osservate** (98.8% mancanti!), come possiamo predire i valori sconosciuti?

**Riferimento Slide**: MC1.pdf, Slide 2-3 - Visualizzazione matrice sparsa Netflix con rating osservati (puntini colorati) e celle vuote (valori mancanti)

### Formulazione Generale del Problema

`00:01:11` 
Quindi l'idea, in generale, è che avete una matrice, o in generale, un tensore, dove, date alcune dimensioni di questa matrice, quindi in questo caso possiamo immaginare di avere N utenti e D film, ogni utente ha valutato solo alcuni film.

**Setup Matematico:**

Sia $X \in \mathbb{R}^{n \times d}$ la **rating matrix**:
- **Righe** $(i = 1, \ldots, n)$: Utenti
- **Colonne** $(j = 1, \ldots, d)$: Film/items
- **Elementi** $X_{ij}$: Rating dell'utente $i$ per il film $j$

**Osservazioni Parziali:**

Definiamo $\Omega \subseteq [n] \times [d]$ come l'insieme delle **posizioni osservate**:
$$
\Omega = \{(i,j) : \text{rating } X_{ij} \text{ osservato}\}
$$

Dove:
- $|\Omega| = m$: Numero di osservazioni (es. Netflix: $m \approx 100$ milioni)
- **Sparsity**: $\rho = \frac{m}{n \cdot d}$ (Netflix: $\rho \approx 0.012 = 1.2\%$)

**Osservazioni** (valori dati):
$$
\mathcal{D} = \{X_{ij} : (i,j) \in \Omega\}
$$

`00:01:44` 
E quindi l'obiettivo è essere in grado di prevedere quale sarebbe il punteggio delle valutazioni mancanti per ogni utente. al fine di creare alcuni suggerimenti per l'utente. È correlato ai cosiddetti sistemi di raccomandazione, okay? È un'istanza dei cosiddetti.

**Obiettivo del Matrix Completion:**

Dato $\Omega$ e $\{X_{ij}\}_{(i,j) \in \Omega}$, trovare $\hat{X} \in \mathbb{R}^{n \times d}$ tale che:

1. **Consistenza**: $\hat{X}_{ij} = X_{ij}$ per ogni $(i,j) \in \Omega$
2. **Buona Generalizzazione**: $\hat{X}_{ij} \approx X_{ij}$ per $(i,j) \notin \Omega$

**Applicazioni oltre Netflix:**

- **E-commerce**: Raccomandazione prodotti (Amazon, Alibaba)
- **Social Networks**: Suggerimenti amicizie/connessioni
- **Computer Vision**: 
  * Video background subtraction (rimozione foreground)
  * Image inpainting (riempimento aree danneggiate)
  * Multi-view reconstruction
- **Genomics**: Predizione interazioni gene-gene da dati sparsi
- **Finance**: Stima correlazioni asset da dati incompleti
- **System Identification**: Ricostruzione modelli dinamici

**Riferimento Slide**: MC1.pdf, Slide 4 - Esempi di applicazioni (immagini: Netflix UI, video subtraction, inpainting)

### Low-Rank Structure e Fattori Latenti

`00:02:18` 
sistemi di raccomandazione. Quindi quello che vogliamo fare è, partendo da questo esempio, vogliamo capire come possiamo affrontare questo problema. Una delle caratteristiche importanti del, problema è che nelle applicazioni pratiche sia M che D sono abbastanza grandi. Quindi avete molti utenti, e molti film, serie, documentari, qualunque cosa, okay?

`00:02:57` 
E il punto chiave in questo problema è che vogliamo sfruttare il fatto che il dataset che ci viene dato è caratterizzato da un basso rango. Significa, in pratica, che ci sono alcuni fattori latenti o caratteristiche derivate importanti che possono essere usate per descrivere tutte le altre colonne o righe del dataset.

**Ipotesi Fondamentale: Low-Rank Structure**

**Assumiamo**: $\text{rank}(X) = r \ll \min(n, d)$

**Perché basso rango?** Esistono $r$ **fattori latenti** (features nascoste) che determinano i rating:

$$
X = U V^T
$$

Dove:
- $U \in \mathbb{R}^{n \times r}$: **User feature matrix**
  * Riga $U_i$ = "profilo gusti" utente $i$ nei $r$ fattori latenti
- $V \in \mathbb{R}^{d \times r}$: **Movie feature matrix**
  * Riga $V_j$ = "caratteristiche" film $j$ nei $r$ fattori latenti

**Rating Prediction**:
$$
X_{ij} = U_i \cdot V_j^T = \sum_{k=1}^{r} U_{ik} V_{jk}
$$

**Interpretazione**: Il rating è un **prodotto scalare** tra preferenze utente e caratteristiche film nello spazio latente.

`00:03:48` 
Ecco alcuni esempi. Quindi, ecco alcune possibili caratteristiche latenti come, Il genere, l'era, il pubblico, e il punto chiave è che questo numero di caratteristiche latenti, caratteristiche latenti significative, è dato da R, che è effettivamente il rango della matrice, e questo numero R è molto più piccolo del numero di utenti e del numero di film.

**Esempi di Fattori Latenti per Film:**

Supponiamo $r = 5$ fattori latenti:

1. **Action/Drama**: $[-1, +1]$ → -1 = pure action, +1 = pure drama
2. **Vintage/Modern**: Epoca del film (classico vs contemporaneo)
3. **Blockbuster/Art-house**: Budget e target audience
4. **Serious/Comedy**: Tono del film
5. **Romance Level**: Quantità di contenuto romantico

**User Profile Example**:
$$
U_{\text{Alice}} = [0.8, -0.6, 0.3, 0.5, -0.2]
$$
- Alice preferisce **drama** (0.8), **film vintage** (-0.6), **comedy** (0.5)
- Non le piace il **romance** (-0.2)

**Movie Profile Example**:
$$
V_{\text{Inception}} = [0.4, 0.7, 0.9, 0.8, 0.1]
$$
- Inception è **balanced action-drama** (0.4), **moderno** (0.7), **blockbuster** (0.9)

**Predicted Rating**:
$$
X_{\text{Alice,Inception}} = U_{\text{Alice}} \cdot V_{\text{Inception}}^T = 0.8 \cdot 0.4 + (-0.6) \cdot 0.7 + \ldots = 0.77
$$

**Normalizzato** a scala 1-5: $\hat{R} \approx 3.9$ ⭐

**Riferimento Slide**: MC1.pdf, Slide 5 - Diagramma fattorizzazione matriciale $X = UV^T$ con illustrazione fattori latenti (genere, era, audience)

`00:04:24` 
Quindi, in altre parole, ci viene dato, questa è un'ipotesi fondamentale, ci viene dato un dataset, che è a basso rango, che ha intrinsecamente una struttura a basso rango. Facciamo un esempio. Se abbiamo, per esempio, una matrice quadrata, supponiamo che M, che è un insieme theta, sia una matrice M per N, e il rango di questa matrice sia R, invece di rango pieno, invece di N.

## Gradi di Libertà e Struttura a Basso Rango

`00:05:16` 
E tenete a mente che assumiamo che R, scusate, che N, in questo caso, sia grande. Quindi se il rango è R, qual è il numero di gradi di libertà di cui abbiamo bisogno per descrivere completamente il dataset? Se tenete a mente la decomposizione SVD... della matrice M, abbiamo la matrice sigma che è caratterizzata.

`00:05:51` 
da R valori. Poi abbiamo le due matrici U e V. Che sono caratterizzate da r vettori, ciascuna è composta da r vettori, perché sappiamo che i vettori rimanenti sono totalmente insignificanti dal punto di vista pratico. Ma questi vettori, questi r vettori, sia in u che in v, sono correlati in qualche modo dal fatto che sono unitari e sono mutuamente ortogonali.

`00:06:33` 
Quindi in realtà, in linea di principio, si potrebbe dire che questi vettori sono caratterizzati da r per n, ma questo non è vero perché c'è una relazione tra le componenti di questi vettori. Quindi se, per esempio, se fate un calcolo molto semplice ricorrendo alla... procedura di ortogonalizzazione di Gram-Schmidt, potete facilmente vedere che ogni.

`00:07:09` 
fattore, quindi ogni u o v, è caratterizzato da un numero di gradi di libertà che è uguale a 2n meno i meno 1 per r su 2. Quindi questo è per u e questo è per v. Se li sommate, quello che ottenete è che il numero di gradi di libertà di cui avete bisogno è 2n meno r per r. Quindi è quel numero.

`00:07:45` 
E se tenete a mente che n in generale è grande, supponiamo che solo per fissare l'idea che n sia dell'ordine di 10 alla potenza 6, quindi abbiamo 1 milione di utenti e 1 milione di film, okay? E r sia 10 alla potenza 2. Quindi abbiamo un rango di 100. È chiaro che in linea di principio dovremmo avere 10 alla potenza 12 elementi per descrivere la matrice.

`00:08:29` 
In pratica, se il rango è r, abbiamo 2. Per n meno r, quindi è 2 per 10 alla potenza 6 meno 100 per... Quindi, è dell'ordine di 10 alla potenza di 8, che è ovviamente considerevolmente più piccolo di n al quadrato, okay?

`00:08:59` 
Quindi, questo è il punto chiave. Abbiamo una struttura a basso rango sottostante nel dataset. È chiaro che se, per caso, vi viene dato un dataset con valori mancanti e questo dataset non è caratterizzato da una struttura a basso rango, non avete possibilità di riempire i valori mancanti.

`00:09:31` 
In qualche modo significativo, a meno che non vogliate usare, non lo so, usate il valore più vicino per riga o colonne o fate la media dei valori circostanti, ma è qualcosa che non è molto significativo dal punto di vista pratico. Okay, quindi il fatto che abbia una struttura a basso rango è qualcosa che accade in pratica perché nelle applicazioni pratiche, se avete un dataset molto grande, è ragionevole assumere che non tutti i campioni siano totalmente non correlati l'uno con l'altro.

`00:10:23` 
E inoltre, è di importanza fondamentale perché se non avete queste ipotesi di struttura a basso rango, non avete speranza di venire con un algoritmo significativo per riempire il valore mancante. Okay?

## Formalizzazione del Problema

`00:10:23` 
Okay, quindi ora vogliamo formalizzare il problema. Abbiamo una matrice che è solo parzialmente osservata. Quindi significa che omega sarà un sottoinsieme di tutte le possibili coppie di indici, e in omega abbiamo solo la coppia di indici i, j, per cui abbiamo un valore.

**Setup Formale del Matrix Completion**

**Dati**:
- $X \in \mathbb{R}^{n \times d}$: **matrice target sconosciuta** con $\text{rank}(X) = r \ll \min(n,d)$
- $\Omega \subseteq [n] \times [d]$: **insieme di osservazioni**
  $$
  \Omega = \{(i,j) : \text{valore } X_{ij} \text{ osservato}\}
  $$
- $m = |\Omega|$: numero di osservazioni
- **Valori osservati**: $\{X_{ij}\}_{(i,j) \in \Omega}$

`00:11:14` 
Ci viene dato un valore. Grazie. Quindi queste sono quelle che possiamo chiamare osservazioni profonde, x di ij, dove ij appartiene a omega, sono valori dati, l'osservazione, e m, che è la cardinalità di omega, è il numero di elementi che ci vengono dati. L'obiettivo è recuperare la matrice completa x di rango r, dove r è molto più piccolo di m e b, il minimo, con l'assunzione che abbiamo una struttura a basso rango.

**Obiettivo**:
Stimare $\hat{X} \in \mathbb{R}^{n \times d}$ tale che:
1. **Coerenza con le osservazioni**: $\hat{X}_{ij} = X_{ij}$ per ogni $(i,j) \in \Omega$
2. **Basso rango**: $\text{rank}(\hat{X}) = r$
3. **Buona generalizzazione**: $\hat{X}_{ij} \approx X_{ij}$ per $(i,j) \notin \Omega$

**Sparsity Pattern Sampling**:

In generale, $\Omega$ può seguire diversi modelli:

**Sampling Uniforme** (assumiamo questo):
$$
\mathbb{P}[(i,j) \in \Omega] = p \quad \text{i.i.d. per ogni } (i,j)
$$
Quindi $m \approx p \cdot n \cdot d$ in aspettazione.

**Altri modelli**:
- **Row-wise**: Alcuni utenti hanno rating completi, altri nessuno
- **Structured**: Blocchi di osservazioni (es. utenti recenti × film popolari)

`00:12:01` 
Quindi questa è la grande immagine, okay? Dovete tenere a mente questa immagine del codice. Quindi, um... Abbiamo bisogno di alcune altre, in particolare, queste altre due definizioni, quindi p omega di x è un operatore di proiezione, è solo un modo di dire che data una matrice x, questo operatore di proiezione sull'insieme omega.

### Operatore di Proiezione $P_{\Omega}$

**Definizione Formale**:

$$
\boxed{[P_{\Omega}(M)]_{ij} = \begin{cases}
M_{ij} & \text{se } (i,j) \in \Omega \\
0 & \text{se } (i,j) \notin \Omega
\end{cases}}
$$

Per ogni matrice $M \in \mathbb{R}^{n \times d}$.

`00:12:46` 
dà xij, che è un valore dato perché gli indici i e j appartengono a omega, altrimenti avete zero. Quindi, in altre parole, data una matrice, se alimentate l'operatore di proiezione p con una matrice generica A, Cosa farà P è mettere i valori noti in A in quelle posizioni dove avete osservazioni, okay?

**Proprietà di $P_{\Omega}$**:

1. **Linearità**: $P_{\Omega}(\alpha A + \beta B) = \alpha P_{\Omega}(A) + \beta P_{\Omega}(B)$

2. **Idempotenza**: $P_{\Omega} \circ P_{\Omega} = P_{\Omega}$
   $$
   P_{\Omega}(P_{\Omega}(M)) = P_{\Omega}(M)
   $$

3. **Proiezione ortogonale** (rispetto a Frobenius inner product):
   $$
   \langle P_{\Omega}(A), P_{\Omega}(B) \rangle_F = \sum_{(i,j) \in \Omega} A_{ij} B_{ij}
   $$

4. **Autoadgiunto**: $P_{\Omega}^* = P_{\Omega}$

5. **Norma operator**: $\|P_{\Omega}\|_{op} = 1$

**Esempio Visivo**:

Matrice $3 \times 3$ con $\Omega = \{(1,1), (1,3), (2,2), (3,1), (3,2)\}$ (5 osservazioni):

$$
M = \begin{bmatrix}
7 & 2 & 4 \\
1 & 9 & 3 \\
5 & 8 & 6
\end{bmatrix}
\quad \Rightarrow \quad
P_{\Omega}(M) = \begin{bmatrix}
\mathbf{7} & 0 & \mathbf{4} \\
0 & \mathbf{9} & 0 \\
\mathbf{5} & \mathbf{8} & 0
\end{bmatrix}
$$

Valori in **bold** = osservati; $0$ = non osservati (mascherati).

**Matrice Complementare**: $P_{\Omega^c}(M) = M - P_{\Omega}(M)$ (valori mancanti)

`00:13:30` 
Questa è la proiezione. E metterà anche zero. E poi la frazione del soggetto è solo il rapporto tra il numero di valori osservati. Sul numero totale di elementi del metodo. Qual è il problema? In linea di principio, questo è un problema reale.

### Sampling Fraction

**Definizione**:
$$
\rho = \frac{m}{n \cdot d} = \frac{|\Omega|}{n \cdot d}
$$

Frazione di elementi osservati.

**Esempi**:
- Netflix: $\rho = \frac{10^8}{8.16 \times 10^9} \approx 0.012 = \mathbf{1.2\%}$ (very sparse!)
- Typical CV applications: $\rho = 0.05 \text{-} 0.2$ (5-20%)
- System identification: $\rho = 0.3 \text{-} 0.5$ (30-50%)

**Sample Complexity Bound**: Per recovery esatto serve almeno:
$$
\rho \geq \frac{C \cdot r \cdot \log^2(n)}{n}
$$

**Esempio**: $n = 10^4$, $r = 10$
$$
\rho \geq \frac{C \cdot 10 \cdot (\log 10^4)^2}{10^4} \approx \frac{1000 C}{10^4} = 0.1 C
$$

Con $C \approx 5$, serve $\rho \geq 0.5 = \mathbf{50\%}$ osservazioni!

`00:14:01` 
Perché se... non avete alcuna ipotesi aggiuntiva, questa formulazione del problema è proposta, il che significa che ci sono un numero infinito di matrici che soddisfano il vincolo che essenzialmente rispetta il vincolo che per ij appartenente a omega, avete il valore corretto.

**Problema: Infinite Soluzioni senza Low-Rank**

**Set di Soluzioni Ammissibili**:
$$
\mathcal{F} = \{M \in \mathbb{R}^{n \times d} : P_{\Omega}(M) = P_{\Omega}(X)\}
$$

Tutte le matrici che **concordano** con le osservazioni.

**Dimensione di $\mathcal{F}$**:

$\mathcal{F}$ è uno **spazio affine** di dimensione:
$$
\dim(\mathcal{F}) = n \cdot d - m
$$

**Esempio**: $n = d = 100$, $m = 5000$ osservazioni
$$
\dim(\mathcal{F}) = 10{,}000 - 5{,}000 = \mathbf{5{,}000 \text{ gradi di libertà liberi}}
$$

Ci sono $\mathbb{R}^{5000}$ soluzioni diverse!

**Perché servono vincoli aggiuntivi**:
Senza low-rank, il problema è **ill-posed** (mal posto):
- **Infinitely many solutions**: $|\mathcal{F}| = \infty$
- **No uniqueness**: Non possiamo scegliere una soluzione specifica
- **Arbitrary predictions**: I valori mancanti possono essere qualunque!

`00:14:36` 
Quindi, questo punto è la chiave. Dobbiamo sfruttare il fatto che la matrice X è bassa, giusto? Okay? Quindi, data la grande immagine, l'obiettivo, e questa definizione formale, il prossimo passo è aggiungere a questa grande immagine, il fatto che la matrice X dovrebbe essere più bassa.

**Soluzione: Imporre Low-Rank**

Se aggiungiamo il vincolo $\text{rank}(M) \leq r$, il problema diventa **ben posto**:

$$
\mathcal{F}_r = \{M \in \mathbb{R}^{n \times d} : P_{\Omega}(M) = P_{\Omega}(X), \; \text{rank}(M) \leq r\}
$$

Se $m \geq (n+d)r - r^2$ (sample complexity), tipicamente $|\mathcal{F}_r| = 1$ (soluzione unica)!

**Riferimento Slide**: MC1.pdf, Slide 11 - Diagramma: spazio affine $\mathcal{F}$ (infinito) vs. manifold low-rank $\mathcal{F}_r$ (finito)

`00:15:09` 
Quindi in pratica, cosa significa? Significa che tra tutte le matrici possibili, che soddisfano, che concordano con l'osservazione nelle posizioni, i, j appartenenti a omega, vogliamo selezionare quella, che ha rango minimo. Quindi formalmente, il problema può essere riformulato in questo modo.

### Formulazione del Rank Minimization

**Problema di Ottimizzazione**:

$$
\boxed{
\begin{aligned}
\min_{M \in \mathbb{R}^{n \times d}} \quad & \text{rank}(M) \\
\text{s.t.} \quad & P_{\Omega}(M) = P_{\Omega}(X)
\end{aligned}
}
$$

**Forma equivalente**:

`00:15:41` 
Vogliamo minimizzare il rango di M, dove la matrice M è di dimensione M per d, soggetto al vincolo che la proiezione di M, su omega, quindi l'operatore che abbiamo visto prima, è uguale alla proiezione di x su omega, okay? Quindi, in altre parole, che nella posizione ij appartenente a omega, la matrice M ha gli stessi valori della matrice originale sconosciuta X, okay?

$$
\boxed{
\begin{aligned}
\min_{M \in \mathbb{R}^{n \times d}} \quad & \text{rank}(M) \\
\text{s.t.} \quad & M_{ij} = X_{ij}, \quad \forall (i,j) \in \Omega
\end{aligned}
}
$$

**Interpretazione**: 
- **Obiettivo**: Trova la matrice di **rango minimo**
- **Vincolo**: Che sia **consistente** con le osservazioni

Principio di **Occam's Razor**: La soluzione più semplice (basso rango = pochi parametri) è preferibile.

**Analogia con Compressed Sensing**:

| Compressed Sensing | Matrix Completion |
|--------------------|-------------------|
| Segnale sparso: $\min \\|x\\|_0$ | Matrice low-rank: $\min \text{rank}(M)$ |
| Osservazioni: $Ax = b$ | Osservazioni: $P_{\Omega}(M) = P_{\Omega}(X)$ |
| $\\|x\\|_0 = $ # nonzeri | $\text{rank}(M) = $ # valori singolari $> 0$ |
| Rilassamento: $\\|x\\|_1$ | Rilassamento: $\\|M\\|_*$ (nuclear norm) |

`00:16:24` 
Oppure, se volete, invece di usare la proiezione, potete scrivere formalmente che volete minimizzare il rango di M tale che M di ij è uguale a x di ij quando il berretto di ij appartiene a omega. Questa è la formulazione. Qual è il problema? Questo problema è...

### NP-Hardness del Rank Minimization

`00:16:55` 
Molto difficile da risolvere. In realtà, appartiene al cosiddetto problema NP-hard. Quindi immagino che sappiate qual è la distinzione tra P, NP, e il problema hard. E questo problema è NP-hard. Quindi significa che è quasi impossibile da risolvere in pratica. Quindi in qualche modo dobbiamo venire con qualche idea per rendere questo problema gestibile.

**Teorema**: Il problema di rank minimization è **NP-hard**.

**Perché NP-Hard?**

1. **Funzione obiettivo non convessa**: $\text{rank}(M)$ è una funzione **discreta** (valori interi $0, 1, 2, \ldots$)

2. **Non differenziabile**: Non possiamo usare gradient descent!

3. **Combinatorial explosion**: Per verificare se $\text{rank}(M) = r$, serve controllare tutti i minori $(r+1) \times (r+1)$
   $$
   \text{# minori} = \binom{n}{r+1} \times \binom{d}{r+1} \sim \mathcal{O}\left(\frac{n^{r+1} d^{r+1}}{(r!)^2}\right)
   $$

4. **Riduzione da SAT**: È possibile ridurre il problema 3-SAT (NP-complete) a rank minimization

**Complessità**:
- **Tempo**: Nessun algoritmo polinomiale conosciuto
- **Worst-case**: Exponential in $\min(n,d)$

**Approcci Naive (Impraticabili)**:

**1. Exhaustive Search**: Prova tutti i ranghi $r = 0, 1, 2, \ldots$
   - Per ogni $r$: Cerca $M$ con $\text{rank}(M) = r$ che soddisfa vincoli
   - **Complessità**: $\mathcal{O}(\min(n,d) \cdot \text{poly}(n,d,m))$ → **intrattabile**

**2. Greedy Rank-1 Additions**: Aggiungi iterativamente componenti rank-1
   - Nessuna garanzia di ottimalità globale
   - Può rimanere bloccato in minimi locali

**Soluzione: Convex Relaxation**

Invece di minimizzare $\text{rank}(M)$ (non convesso), minimizziamo una **surrogate convessa**: la **Nuclear Norm** $\|M\|_*$.

$$
\text{rank}(M) \quad \longrightarrow \quad \|M\|_* = \sum_{i=1}^{\min(n,d)} \sigma_i(M)
$$

**Riferimento Slide**: MC1.pdf, Slide 13 - Grafico: rank function (step function) vs. nuclear norm (convex envelope)

---

## Parte 2: Rilassamento Convesso e Nuclear Norm

## Rilassamento Convesso e Norma Nucleare

`00:17:36` 
Quindi il prossimo passo è, una volta che abbiamo capito che vogliamo sfruttare la struttura a basso rango, abbiamo formalizzato il problema in termini di questo problema minimo. Dobbiamo, sfortunatamente, fare un passo avanti per rendere questo problema trattabile. L'idea è, ovviamente, sapete qual è il problema. Il problema è che la funzione rango, quindi la funzione che vogliamo minimizzare, è, in generale, non convessa.

**Problema: Non-Convessità del Rango**

**Definizione**: Una funzione $f: \mathbb{R}^n \to \mathbb{R}$ è **convessa** se per ogni $x, y \in \mathbb{R}^n$ e $\lambda \in [0,1]$:
$$
f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)
$$

**Il rango NON è convesso!**

**Controesempio**:
$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
B = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}
$$

- $\text{rank}(A) = 1$, $\text{rank}(B) = 1$
- $\frac{1}{2}A + \frac{1}{2}B = \frac{1}{2}I_2$ → $\text{rank}\left(\frac{1}{2}A + \frac{1}{2}B\right) = 2$

**Violazione convessità**:
$$
\text{rank}\left(\frac{1}{2}A + \frac{1}{2}B\right) = 2 > \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot 1 = 1
$$

**Proprietà Sbagliata del Rango**:
$$
\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B) \quad \text{(subadditività)}
$$

Ma per convessità servirebbe:
$$
\text{rank}(\lambda A + (1-\lambda)B) \leq \lambda \text{rank}(A) + (1-\lambda) \text{rank}(B)
$$

Che **non vale** in generale!

`00:18:14` 
Vi darò un esempio tra un momento. E, quindi, sapete tutti qual è la differenza tra una funzione convessa e non convessa. Quindi, quali sono i problemi con la funzione non convessa? Il fatto che convessa sia molto buona perché di solito ha un singolo minimo globale, è semplice da ottimizzare se pensate, per esempio, anche in 1D, se avete, se ricordate quello che avete fatto quando avete considerato il metodo del gradiente per risolvere un sistema lineare, iniziate con qualsiasi x uguale a b.

### Funzioni Convesse vs Non-Convesse

**Funzione Convessa** (esempio: $f(x) = x^2$):

```
     f(x)
      |
      |     ___
      |   /     \
      | /         \
      |/___________\_____
                       x
   Singolo minimo globale
```

**Proprietà**:
1. **Unico minimo globale** (se esiste)
2. **Ogni minimo locale è globale**
3. **Gradient descent converge** al minimo globale
4. **No saddle points** (punti di sella)

**Esempi**:
- $f(x) = \|x\|_2^2$ (quadratica)
- $f(x) = \|x\|_1$ (norma $\ell_1$)
- $f(X) = \|X\|_*$ (**nuclear norm** ← convessa!)

`00:19:03` 
Poi avete riscritto questo problema in termini di. in termini di questo funzionale che deve essere minimizzato e il minimo di questo funzionale, che è un funzionale quadratico, quindi un convesso, è in realtà la soluzione di questo problema. Su questo funzionale che potete ideare il metodo del gradiente gradiente coniugato e poi.

**Gradient Descent su Funzioni Convesse**:

Per risolvere $Ax = b$ (sistema lineare), minimizziamo:
$$
f(x) = \frac{1}{2}\|Ax - b\|^2 = \frac{1}{2}x^T A^T A x - x^T A^T b + \frac{1}{2}\|b\|^2
$$

**Funzionale quadratico convesso** (se $A^T A \succ 0$).

**Algoritmo Gradient Descent**:
$$
x^{(k+1)} = x^{(k)} - \alpha \nabla f(x^{(k)}) = x^{(k)} - \alpha (A^T A x^{(k)} - A^T b)
$$

**Convergenza garantita** a $x^* = (A^T A)^{-1} A^T b$ (unico minimo globale)!

**Metodi Efficienti**:
- **Conjugate Gradient**: $\mathcal{O}(n^2)$ iterazioni
- **Accelerated methods** (Nesterov): $\mathcal{O}(n)$
- **Teoria ben sviluppata**: Rate di convergenza, condizionamento, precondizionamento

`00:19:36` 
ideare tutta la famiglia di metodi del gradiente per questo tipo di problema. E questo è fortemente correlato al fatto che questo funzionale è convesso, okay? Se dovete affrontare un problema dove il funzionale è non convesso, allora avete alcuni problemi perché.

**Funzione Non-Convessa** (esempio: $f(x) = x^4 - 2x^2$):

```
     f(x)
      |  __         __
      | /  \   /   \
      |      \ /
      |       X  ← saddle point
      |_____________________
                       x
   Multipli minimi locali
```

`00:20:27` 
Okay, quindi d'altra parte, un problema o funzionale non convesso, che è quello sulla destra, è caratterizzato dal fatto che non avete un singolo minimo, potreste avere un insieme di punti, e quindi è un paesaggio totalmente diverso, e quindi se usate, per esempio, un metodo del gradiente, potete rimanere bloccati in un minimo locale o in un insieme di punti, quindi dipendendo.

**Problemi con Funzioni Non-Convesse**:

1. **Multipli minimi locali**: Non tutti sono minimi globali!
   $$
   \nabla f(x^*) = 0 \quad \not\Rightarrow \quad x^* \text{ è minimo globale}
   $$

2. **Dipendenza dall'inizializzazione**: Gradient descent può convergere a minimi locali diversi:
   $$
   x_0^{(1)} \to x_{\text{local}}^{(1)}, \quad x_0^{(2)} \to x_{\text{local}}^{(2)} \neq x_{\text{global}}
   $$

3. **Saddle points** (punti di sella): $\nabla f = 0$ ma non minimo (Hessian indefinito)

4. **Plateaus**: Regioni piatte dove $\nabla f \approx 0$ ma non ottimali

5. **Nessuna garanzia di convergenza**: Algoritmi possono oscillare o divergere

`00:21:01` 
per esempio, dal punto di partenza, potete trovare una soluzione o l'altra. È vero, d'altra parte, che la realtà è non convessa, nel senso che questo tipo di problemi, la funzione convessa... non sono in generale adatte per descrivere situazioni del mondo reale vedremo che quando scrivete.

**Non-Convessità nella Realtà**:

Molti problemi del mondo reale sono **intrinsecamente non-convessi**:

- **Neural Networks**: Loss function con milioni di minimi locali
  $$
  \mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(f_{\theta}(x_i), y_i) \quad \text{(non convesso in } \theta \text{)}
  $$

- **Matrix Factorization**: $\min_{U,V} \|X - UV^T\|_F^2$ (bilineare → non convesso)

- **Sparse Coding**: $\min_{\alpha} \|y - D\alpha\|^2 + \lambda\|\alpha\|_0$ (norma $\ell_0$ non convessa)

**Strategie Pratiche**:
1. **Multi-start**: Prova diverse inizializzazioni random
2. **Simulated Annealing**: Esplorazione stocastica del landscape
3. **Convex Relaxation**: Approssima con problema convesso (← matrix completion!)

`00:21:35` 
la funzione di perdita per una rete neurale quella funzione di perdita è non comune è caratterizzata da, la presenza di molti minimi locali quindi è un paesaggio molto complicato quindi qual è l'idea l'idea è sostituire questo problema che è la minimizzazione del rango.

### Convex Relaxation: L'Idea Chiave

**Problema Originale (NP-hard, non convesso)**:
$$
\min_{M} \; \text{rank}(M) \quad \text{s.t.} \quad P_{\Omega}(M) = P_{\Omega}(X)
$$

**Problema Rilassato (convesso, trattabile)**:
$$
\boxed{\min_{M} \; \|M\|_* \quad \text{s.t.} \quad P_{\Omega}(M) = P_{\Omega}(X)}
$$

Dove $\|M\|_*$ è la **Nuclear Norm** (definita tra poco).

**Perché funziona?**

**Analogia con Compressed Sensing** ($\ell_0 \to \ell_1$):

| Problema | Originale (Non-Convesso) | Rilassato (Convesso) |
|----------|--------------------------|----------------------|
| **Compressed Sensing** | $\min \\|x\\|_0$ | $\min \\|x\\|_1$ |
| **Matrix Completion** | $\min \text{rank}(M)$ | $\min \\|M\\|_*$ |

**$\ell_0$ norm**: $\|x\|_0 = \#\{i : x_i \neq 0\}$ (# nonzeri)
**$\ell_1$ norm**: $\|x\|_1 = \sum_{i} |x_i|$ (**convex envelope** di $\|x\|_0$)

**Rank**: $\text{rank}(M) = \#\{\sigma_i : \sigma_i > 0\}$ (# valori singolari $> 0$)
**Nuclear norm**: $\|M\|_* = \sum_{i} \sigma_i$ (**convex envelope** di rank!)

`00:22:09` 
con qualcosa che è strettamente correlato al rango ma è in qualche modo complesso, Quindi lasciatemi mostrare questa immagine. A sinistra, ho considerato la matrice che potete vedere là. È una matrice due per due che dipende da due parametri. Quindi è radice quadrata di quattro per x,

### Visualizzazione: Rank vs Nuclear Norm

`00:22:43` 
x più y, x più y, e radice quadrata di quattro per y. Quindi questa matrice dipende dai due parametri x e y, e se calcolate il rango di questo parametro come funzione di x e y, quello che potete ottenere è l'immagine qui a sinistra dello schermo.

**Esempio Parametrico**:

$$
M(x,y) = \begin{bmatrix}
\sqrt{4x} & x+y \\
x+y & \sqrt{4y}
\end{bmatrix}, \quad x, y \geq 0
$$

**Rank di $M(x,y)$**:

- Se $x = 0$ o $y = 0$: $\text{rank}(M) = 1$ (una riga/colonna zero)
- Se $x, y > 0$ e $\det(M) = 0$: $\text{rank}(M) = 1$
  $$
  \det(M) = 4xy - (x+y)^2 = 0 \iff (\sqrt{x} - \sqrt{y})^2 = 0 \iff x = y
  $$
- Altrimenti: $\text{rank}(M) = 2$

**Funzione Rank**:
$$
\text{rank}(M(x,y)) = \begin{cases}
1 & \text{se } x = y \text{ oppure } x = 0 \text{ oppure } y = 0 \\
2 & \text{altrimenti}
\end{cases}
$$

**Grafico 3D** (sinistra nella slide):
```
rank(M)
   2 ┤████████████████████
     │████████████████████
   1 ┤───┼────────┼───────  ← discontinuità!
     │   │        │
   0 └───┴────────┴───────
       0    x=y       y
```

**Non convessa**: Salti discontinui, regioni piatte → impossibile ottimizzare!

`00:23:19` 
E come potete vedere, quella funzione ovviamente non è complessa, non complessa, okay? A destra, qui, c'è il grafico della norma nucleare della matrice. Quindi torniamo per un momento a questa definizione. Cos'è la norma nucleare di una matrice? La norma nucleare di una matrice è essenzialmente una norma data dalla somma dei valori singolari.

**Nuclear Norm di $M(x,y)$**:

Calcoliamo $\|M(x,y)\|_* = \sigma_1 + \sigma_2$ (somma valori singolari).

**Valori singolari** di matrice simmetrica $2 \times 2$:
$$
\sigma_1, \sigma_2 = \frac{\text{tr}(M) \pm \sqrt{\text{tr}(M)^2 - 4\det(M)}}{2}
$$

Per $M(x,y)$:
- $\text{tr}(M) = \sqrt{4x} + \sqrt{4y} = 2(\sqrt{x} + \sqrt{y})$
- $\det(M) = 4xy - (x+y)^2$

**Nuclear Norm**:
$$
\|M(x,y)\|_* = \sigma_1 + \sigma_2 = \text{tr}(M) = 2(\sqrt{x} + \sqrt{y})
$$

**Grafico 3D** (destra nella slide):
```
||M||*
     ┤     ╱╲
     │   ╱    ╲
     │ ╱        ╲   ← smooth, convessa!
     │╱          ╲
   0 └─────────────
       0         y
```

**Convessa**: Superficie liscia, unico minimo (0,0), ottimizzabile con gradient descent!

**Riferimento Slide**: MC1.pdf, Slide 15 - Confronto 3D: rank function (step) vs. nuclear norm (smooth cone)

`00:23:58` 
della matrice, okay? Quindi la definizione è molto, molto semplice. È solo la somma dei valori singolari. Quali sono le buone proprietà di questa funzione in questo contesto? la somma dei valori singolari è la traccia della radice quadrata di m trasposta m che è qualcosa.

`00:24:29` 
che abbiamo già visto ma perché è importante perché essenzialmente se avete il se immaginate di creare un vettore potete chiamarlo sigma con i valori singolari della matrice la norma nucleare della matrice m è in realtà la norma l1 di questo particolare vettore okay ricordate la norma l1.

`00:25:02` 
del vettore è solo la somma della componente del vettore delle componenti del vettore, Okay, quindi la norma nucleare è equivalente alla norma L del vettore dei valori singolari. Okay, e come possiamo vedere da questa immagine a destra, è una funzione convessa. Okay, perché è interessante dal nostro punto di vista?

`00:25:37` 
Perché, facciamo un parallelismo con i vettori. Quindi, prima di tutto, cos'è la norma zero di un vettore? In realtà, immagino che siate tutti familiari con la norma uno, norma due, norma infinito. In linea di principio, potete definire anche la norma zero di un vettore, che non è altro che. Qualcosa che vi dice quanti elementi medi zero avete nel vettore, quante componenti sono diverse da zero.

`00:26:22` 
Quindi, in altre parole, se usate la norma zero per i vettori e cercate di minimizzare questa norma, questa norma, è facile vedere, è una norma non compatta perché dipende dal numero di componenti non zero del vettore. D'altra parte, se considerate la norma uno del vettore, che è la somma delle componenti del vettore, e minimizzate.

`00:27:01` 
la norma uno del vettore, state in realtà andando verso una. rappresentazione sparsa del vettore, ma usando una funzione convessa. Nel nostro caso, vogliamo sfruttare la stessa idea. Abbiamo una matrice e ora quello che vogliamo fare non è.

`00:27:32` 
andare verso la sparsità. Quello che cerchiamo è un basso rango. Basso rango in generale significa che, dovremmo in linea di principio rappresentare la funzione rango, ma abbiamo osservato che la funzione rango è non convessa, quindi dobbiamo in qualche modo trovare un surrogato per la funzione rango. che sia in grado di descrivere, in ogni caso, il fatto che stiamo cercando matrici a basso rango.

`00:28:13` 
E l'idea è usare, quindi, il rilassamento convesso della funzione rango nel contesto della matrice è la norma nucleare. Potete pensare in questo modo. Quindi, la norma L1, in cima a questa tabella, promuove la sparsità in un vettore. Abbiamo visto che la norma nucleare della matrice è equivalente alla norma L1 del vettore dei valori singolari. Ma il fatto che.

`00:28:55` 
Abbiamo questi equivalenti. Significa che essenzialmente il fatto che vogliamo minimizzare la norma nucleare, è equivalente a minimizzare la norma L1 del vettore dei valori singolari. Ma minimizzare il singolare, la norma L1 del vettore dei valori singolari significa che stiamo cercando un vettore dei valori singolari, che sia sparso. Cosa significa, sparso? Significa che ha molti zeri, okay? Ma molti zeri significa cosa? Basso rango, okay?

`00:29:34` 
Quindi questa è l'idea. E dato questo framework, ora possiamo effettivamente riformulare il problema come... Minimizzare. Invece di minimizzare il rango, minimizziamo il rilassamento convesso della funzione rango, che è la norma nucleare, e i vincoli sono sempre gli stessi. Soggetto alla proiezione su omega della matrice M deve essere uguale alla proiezione di X su omega t.

`00:30:17` 
I vantaggi di questa formulazione è che è convessa, quindi può essere risolta efficientemente usando metodi che sono sviluppati per problemi convessi, e poi c'è un framework teorico molto ben stabilito che garantisce la convergenza di questi metodi.

`00:30:48` 
E poi, questi sono risultati abbastanza recenti, avete il recupero esatto della matrice X sotto certe ipotesi, okay?

## Teoria del Matrix Completion

`00:30:48` 
Okay, quindi ora quello che vogliamo capire è, dal punto di vista pratico, come risolvere quel problema, quel problema convesso.

`00:31:22` 
Prima di andare all'algoritmo, voglio dire qualcosa sulla teoria che sta dietro questo problema di completamento di matrici. Questi risultati sono stati presentati per la prima volta in questo articolo, 2008, e l'idea è... Se partite da una matrice M, che è data da questa rappresentazione che ora conosciamo molto bene,

**Teorema di Candès-Recht (2009)**

Il risultato fondamentale che garantisce il **recovery esatto** per matrix completion.

`00:31:54` 
è somma di contributi di rango uno, dove i vettori uK e i vettori vK sono sono selezionati uniformemente, casualmente, uniformemente, tra vettori ortonormali. Quindi, in pratica, se considerate questa famiglia di matrici, che sono generate in questo modo,

**Setup del Teorema**:

Consideriamo una matrice $X \in \mathbb{R}^{n \times n}$ (per semplicità quadrata) con:

1. **Low-rank**: $\text{rank}(X) = r \ll n$

2. **Modello Ortogonale Random** (Random Orthogonal Model):
   $$
   X = \sum_{k=1}^{r} u_k v_k^T
   $$
   Dove $\{u_1, \ldots, u_r\}$ e $\{v_1, \ldots, v_r\}$ sono vettori ortonormali scelti **uniformemente a caso** dallo spazio di vettori ortonormali in $\mathbb{R}^n$.

3. **Sampling Uniforme**: Gli indici in $\Omega$ sono scelti **uniformemente i.i.d.** con probabilità $p = m/(n^2)$.

`00:32:28` 
dite che state considerando un modello ortogonale casuale, M. È chiaro che se la matrice M, in linea di principio, è data da dimensione N per E, come al solito, qui stiamo assumendo che R sia molto più piccolo del minimo tra M. Uno dei risultati importanti dell'articolo è questo.

**Incoerenza Condition**:

Definiamo il **parametro di incoerenza** $\mu \geq 1$:

$$
\mu(U) = \frac{n}{r} \max_{i=1,\ldots,n} \|U_{i,:}\|^2
$$

Dove $U_{i,:}$ è la $i$-esima riga della matrice $U$ nella fattorizzazione $X = UV^T$.

**Interpretazione**:
- $\mu = 1$: **Incoherent** (energia uniformemente distribuita, best case)
- $\mu = n/r$: **Coherent** (energia concentrata su poche righe, worst case)

**Esempi**:
- Fourier matrix: $\mu = 1$ (perfettamente incoherent)
- Spiky matrix (1 riga dominante): $\mu = n/r$ (maximally coherent)

`00:33:02` 
Quindi, supponiamo che abbiate una matrice, qui è M1 per M2, ed è una matrice che è un modello ortogonale casuale. Quindi, è stata generata in quel modo. N è il massimo tra M ed E, e supponiamo che osserviate M entrate in questa matrice, casuali.

**Teorema (Candès-Recht, 2009)**:

Sia $X \in \mathbb{R}^{n_1 \times n_2}$ con $\text{rank}(X) = r$ e parametro di incoerenza $\mu$. 

Sia $n = \max(n_1, n_2)$ e supponiamo di osservare $m$ elementi scelti **uniformemente a caso**.

`00:33:35` 
Allora, potete trovare due costanti, una C maiuscola e una c minuscola, tali che se il numero di campioni che avete scelto nella vostra matrice, il numero di osservazioni, è maggiore o uguale a questa espressione, allora il minimizzatore della norma nucleare è unico e, più importante, è uguale a M, quindi la vera matrice, con probabilità 1 meno C M alla meno 3.

**Statement**:

Se il numero di osservazioni soddisfa:
$$
\boxed{m \geq C \cdot \mu^2 \cdot n^{1.2} \cdot r \cdot \log^2(n)}
$$

Allora con probabilità almeno $1 - c n^{-3}$ (per costanti universali $C, c > 0$), il minimizzatore della nuclear norm:
$$
\hat{X} = \arg\min_{M} \|M\|_* \quad \text{s.t.} \quad P_{\Omega}(M) = P_{\Omega}(X)
$$

è **unico** e **uguale esattamente a** $X$:
$$
\hat{X} = X \quad \text{(exact recovery!)}
$$

**Versione Migliorata** (Candès-Tao, 2010): Con $\mu_0, \mu_1$ (parametri di incoerenza raffinati):
$$
m \geq C \cdot \max(\mu_0^2, \mu_1) \cdot n \cdot r \cdot \log^2(n)
$$

Questa versione migliora la dipendenza da $n^{1.2}$ a $n$ (lineare!).

`00:34:22` 
Quindi significa che... uh ricordate che n è di solito abbastanza grande quindi significa che state recuperando con probabilità molto alta le due metriche okay questa è uh l'idea o in generale uh in molti libri di testo dove, non è riportato esattamente il risultato si dice che se avete abbastanza osservazioni.

**Interpretazione del Teorema**:

**1. Sample Complexity**: 
   
Per $n = 10^4$, $r = 10$, $\mu = 2$:
$$
m \geq C \cdot 4 \cdot (10^4)^{1.2} \cdot 10 \cdot \log^2(10^4) \approx 10^7
$$

Circa $m/(n^2) \approx 10\%$ di elementi osservati!

**2. Probabilità di Success**:

$$
\mathbb{P}[\text{exact recovery}] \geq 1 - c n^{-3}
$$

Per $n = 1000$: $\mathbb{P} \geq 1 - 10^{-9}$ (quasi certo!)

**3. Ruolo dell'Incoerenza**:

Se $\mu$ grande (matrice coherent), servono **più osservazioni**:
$$
m \propto \mu^2
$$

Se $\mu = O(1)$ (incoherent), sample complexity ottimale!

**Riferimento Slide**: MC1.pdf, Slide 18 - Grafico: Sample complexity $m$ vs $n$ per diversi $r$ e $\mu$

`00:34:55` 
allora con alta probabilità siete in grado di recuperare le vere metriche okay. Quindi, in pratica, se le ipotesi sono verificate, c'è una matrice unica coerente con l'osservazione. Quindi, le proiezioni sono le stesse, e possiamo usare il problema di ottimizzazione convessa, perché nel teorema si menziona la norma nucleare, quindi stiamo considerando il problema convesso.

**Condizioni Chiave per Recovery Esatto**:

1. ✅ **Low-rank**: $r \ll n$
2. ✅ **Incoherence**: $\mu(X) = O(1)$ (non troppo spike)
3. ✅ **Uniform sampling**: $\Omega$ uniforme random
4. ✅ **Sufficient samples**: $m \geq C \mu^2 n^{1.2} r \log^2 n$

Se **tutte** soddisfatte → **Recovery esatto** con alta probabilità!

**Perché funziona?**

**Intuizione Geometrica**:
- Lo spazio delle matrici rank-$r$ è un **manifold** di dimensione $(n_1 + n_2 - r) \cdot r$
- Le osservazioni $P_{\Omega}$ forniscono $m$ vincoli lineari
- Se $m$ abbastanza grande e ben distribuito → manifold **interseca** univocamente lo spazio dei vincoli
- Nuclear norm minimization "trova" questa intersezione

**Proof Sketch** (step principali):

1. **Dual Certificate**: Costruire matrice $Y$ tale che:
   $$
   P_T(Y) = UV^T, \quad P_{T^\perp}(Y) \text{ piccolo}
   $$
   Dove $T = \text{span}(U, V)$ (tangent space al manifold).

2. **Golfing Scheme**: Iterativamente "aggiusta" $Y$ usando osservazioni random
   
3. **Concentration Inequalities**: Probabilità che costruzione fallisca $\leq c n^{-3}$

`00:35:38` 
E quindi, abbiamo trovato in qualche modo che il rilassamento della norma nucleare dà una soluzione che è formalmente equivalente, o molto molto vicina, a quella che sarebbe stata data dal non convesso NP-hard. In pratica, quindi, ora abbiamo formalizzato in modo gestibile il problema.

**Risultato Sorprendente**:

$$
\text{NP-hard } \text{rank}(M) \text{ minimization} \quad \xrightarrow{\text{convex relaxation}} \quad \text{Tractable } \|M\|_* \text{ minimization}
$$

**Con garanzie teoriche**: Sotto ipotesi ragionevoli, la soluzione convessa **coincide** con la soluzione NP-hard!

Questo è un risultato **non banale**: non è vero in generale che convex relaxation dia soluzione ottimale del problema non-convesso.

**Estensioni del Teorema**:

1. **Matrix Completion con Rumore** (Candès-Plan, 2010):
   $$
   P_{\Omega}(M) = P_{\Omega}(X) + \varepsilon
   $$
   Recovery stabile: $\|M - X\|_F \leq C \|\varepsilon\|_F$

2. **Rectangular Matrices**: Teorema si estende a $n_1 \neq n_2$

3. **Structured Sampling**: Alcuni pattern non-uniformi OK (es. row-column sampling)

4. **Rank Adaptivity**: Non serve conoscere $r$ a priori!

`00:36:13` 
Abbiamo visto dal punto di vista teorico che questo problema ha bisogno di altre ipotesi adatte, ha bisogno di una soluzione unica, e questa soluzione ha bisogno, con alta probabilità, della soluzione vera che stiamo cercando. Ora dobbiamo praticamente risolvere il problema. Anche se complesso e un buon problema, questo è un problema abbastanza grande da risolvere, quindi dobbiamo.

**Riassunto**:

| Aspetto | Dettaglio |
|---------|-----------|
| **Problema** | $\min \|\|M\|\|_*$ s.t. $P_{\Omega}(M) = P_{\Omega}(X)$ |
| **Requisiti** | Low-rank, incoherence, uniform sampling |
| **Sample Complexity** | $m \gtrsim \mu^2 n^{1.2} r \log^2 n$ |
| **Garanzia** | Exact recovery w.p. $\geq 1 - c n^{-3}$ |
| **Algoritmo** | SVT, proximal gradient, ADMM |
| **Complessità** | $\mathcal{O}(k \cdot r \cdot nd)$ per iterazione |

**Riferimento**: E. Candès and B. Recht, "Exact matrix completion via convex optimization," *Foundations of Computational Mathematics*, 2009.

---

## Parte 3: Algoritmo SVT

## Algoritmo Singular Value Thresholding

`00:36:46` 
ideare strategie adatte. L'algoritmo originale che è stato ideato per considerare questo problema è il cosiddetto algoritmo di soglia dei valori singolari. È un algoritmo iterativo, si basa fortemente sulla struttura a basso rango della matrice, richiede la SVD della matrice, ed è efficiente anche per problemi grandi.

`00:37:23` 
Quali sono gli ingredienti chiave di questo singolare valore per Schrödinger? La prima idea è introdurre questa funzione, che è la cosiddetta soglia morbida, quindi dato un numero x e una soglia di flusso, che è decisa a priori,

`00:37:53` 
avete questa funzione. Essenzialmente, se x è, maggiore della soglia, avete come output o x più tau o x meno tau. Se x è minore di tau in valore assoluto, allora state impostando l'output a zero.

`00:38:29` 
Quindi sarebbe qualcosa del genere. Ora questo è per un singolo numero, quello che vogliamo fare è applicare questa funzione, s di tau, alla matrice.

`00:39:00` 
In particolare, alla matrice dei valori singolari. Quindi, se avete la matrice M e la sua SVD, dove sigma è la matrice diagonale con i valori singolari, l'operatore di soglia dei valori singolari della matrice è definito come u s tome applicato a sigma,

`00:39:35` 
il che significa che è applicato a ciascuno degli elementi diagonali di sigma, per b trasposta. Okay, quindi in pratica, se usate la notazione usuale, invece di avere sigma qui, avete la funzione di sotto-soglia scalare. Okay. E potete applicarla ai vettori singolari. Quindi, in altre parole, state mettendo a zero i piccoli valori singolari, okay?

`00:40:13` 
Quindi questa è graficamente, l'idea della funzione di soglia morbida. Ora, qual è la connessione tra questi operatori, l'operatore che abbiamo qui, possiamo usare qui, il SVT, soglia dei valori singolari il numeratore parametrico,

`00:40:45` 
e la norma nucleare. L'operatore D di TAU può essere scritto anche come... in questa forma. Quindi avete la prima parte, che è un mezzo, poi avete il quadrato della norma proveniente di m meno y, più tau per la norma nucleare di m. In pratica, qui la matrice.

`00:41:24` 
y è la matrice con le osservazioni, e poi state calcolando l'argomento di quella funzione, l'argomento che minimizza la funzione. Quindi in altre parole, significa che la soglia morbida è equivalente a un problema dei minimi quadrati regolarizzato.

`00:41:56` 
d al quadrato nella norma di Frobenius per la matrice M. In questa formulazione, abbiamo qui il primo termine è essenzialmente la fedeltà a Y, quindi all'osservazione, e qui stiamo penalizzando la norma nucleare. Quindi la prima parte ci sta effettivamente dicendo che la matrice M è vicina a Y, vogliamo essere il più vicino possibile a Y,

`00:42:29` 
e il secondo termine è qualcosa che ci aiuta ad andare verso un basso rango. L'idea è che in futuro, quando avete il problema dei minimi quadrati, regolarizzato o no, potete risolvere usando la SVD. Grazie. Quindi, in pratica, calcolate la SVG di y, soglia la matrice sigma, e poi il veto risultante di y è u sigma primo v trasposta, dove sigma primo è ottenuto applicando la soglia morbida alla matrice sigma.

`00:43:21` 
In altre parole, se consideriamo questi equivalenti, possiamo dire che possiamo, per scopi pratici, possiamo considerare una di queste due formulazioni del problema. La prima è il problema di minimizzazione.

`00:43:53` 
Non vincolato, abbiamo volte totali la norma nucleare di n più, invece di avere m e y, abbiamo la proiezione di m su omega, e meno la proiezione di x su omega, dovrebbe essere il più minimo possibile. Quindi vogliamo avere le osservazioni che sono date, vogliamo che siano mantenute, okay?

`00:44:26` 
E ovviamente tau è un parametro positivo, altrimenti potete riformulare lo stesso problema in termini di ottimizzazione vincolata, quindi vogliamo minimizzare la norma nucleare soggetto all'equivalenza del progetto. l'algoritmo. Quindi l'idea è, come input abbiamo l'osservazione, l'insieme di osservazioni.

`00:44:59` 
Poi abbiamo il parametro, tau, che è la soglia che vogliamo applicare. Poi abbiamo quel parametro, la dimensione del passo, che di solito è nell'intervallo 0 delta. Tra un momento commenterò su quello. Inizializzate il problema usando y0 uguale, scusate, questo dovrebbe essere c0. C0 è una costante che può essere calcolata o potete anche scegliere 1, per esempio.

`00:45:40` 
Poi delta è la dimensione del passo che state considerando. E poi avete la proiezione di x. Quindi questo è y0. Poi calcolate mk. mk, ricordate, è questo, okay? Come potete vedere, mk, m, l'iterazione k, è il tau di yk.

`00:46:12` 
Quindi al primo passo, state calcolando, essenzialmente, significa che state calcolando la SVD di yk, e poi state applicando la protezione software ai valori singolari, okay? È un modo formale per rappresentare queste due operazioni. E poi aggiornate y, aggiornate y considerando il precedente y, più delta,

`00:46:44` 
per la proiezione del... una sorta di residuo x meno mk. Da dove viene questa formula? Ricordate che questo è un, problema convesso, quindi in linea di principio è un calcolo lungo, ma in pratica significa.

`00:47:15` 
formalmente che scrivete l'algoritmo di discesa del gradiente per questo problema, impostate a zero l'algoritmo del gradiente, e fate un passo per muovervi verso il minimo, okay? E essenzialmente questo delta è la dimensione del set e il p omega. x meno mk è la direzione in cui vi state muovendo, okay, quindi partendo da yk vi state muovendo nella direzione della proiezione del residuo con la dimensione del passo data da delta, okay, quindi quella regola di aggiornamento non è altro che quello che potete ottenere scrivendo formalmente il problema come problema minimo impostato per calcolare il gradiente.

`00:48:11` 
e impostandolo a zero, okay, e poi eseguite questa iterazione per un certo numero di passi, okay, quindi il primo passo restringe i valori singolari per soglia morbida, secondo passo state imponendo le osservazioni, okay.

`00:48:43` 
Quindi, state, come potete vedere, alternando tra imporre la minimizzazione del rango e la fedeltà ai dati. Questo fatto significa che l'algoritmo non sarà un algoritmo monotono. Perché? Perché nel primo passo, state andando verso la minimizzazione del rango.

`00:49:16` 
Ma una volta che modificate alcuni degli elementi della matrice, probabilmente state, sicuramente, modificando il rango e non è... Quindi, in generale, facendo quella regola di aggiornamento, modificherete il rango in modo tale che forse aumenterà, o non avete garanzia che il rango stia diminuendo monotonicamente, okay?

`00:49:51` 
Ma è come il metodo di bisezione per trovare lo zero di una puntura. Non è monotono, ma se fate abbastanza passi, sapete che state convergendo verso lo zero, okay? Okay, qui c'è solo una divisione del metodo.

`00:50:25` 
Torneremo ai parametri tra un momento. Sulla convergenza, stavo dicendo che non è monotonicamente convergente, ma potete, in un altro articolo dello stesso autore, se scegliete delta in 0-2, quell'algoritmo converge alla soluzione ottimale del problema di minimizzazione nucleare.

`00:50:59` 
La convergenza è lineare, e il tasso dipende da delta. Di solito delta è tra 1.2 e 1.5. Non è qualcosa che è stato ottenuto teoricamente, ma è solo una regola empirica.

`00:51:31` 
Dato che è una procedura iterativa, dobbiamo o decidere a priori quanti passi vogliamo eseguire, o ideare un criterio di arresto, ok? Il criterio di arresto proposto per il metodo è controllare la norma di provenienza della differenza tra la matrice proiettata e K.

`00:52:01` 
K e X, divisa per la norma di provenienza della matrice originale X, ok? Ed epsilon è la tolleranza data che volete imporre. La complessità di... del metodo è dell'ordine di k è il numero di iterazioni che avete eseguito,

`00:52:37` 
r è il rango, e n e d sono le dimensioni della matrice. Quindi è chiaro che se, anche se n e d sono grandi, se r è piccolo, allora ho la complessità complessiva dell'algoritmo è gestibile. Okay, ecco tutte le buone proprietà del metodo.

`00:53:16` 
Ci sono alcuni commenti da menzionare sui parametri. Per esempio, per la soglia, avete essenzialmente tre possibilità. La prima è una scelta teorica che, se ricordate, è simile alla soglia che abbiamo visto.

`00:53:49` 
quando abbiamo considerato l'algoritmo di soglia per le metriche casuali. E avete essenzialmente un tau che è dell'ordine della radice quadrata del prodotto della dimensione, e gamma è un parametro da 5 a 10, e questa è una possibilità.

`00:54:20` 
Oppure potete fare una convalida incrociata, quindi potete fare essenzialmente una ricerca a griglia. su una certa porzione dello spazio dei parametri di solito l'intervallo è da 1 a 10 volte la radice quadrata di nd e selezionate tau minimizzando l'errore di validazione o potete usare una.

`00:54:53` 
strategia adattiva queste tre ricette sono anche le ricette che vedremo e forse se avete familiarità con le reti neurali e per esempio con la discesa del gradiente stocastica per minimizzare per allenare la rete neurale anche in quel caso il tasso di apprendimento che è.

`00:55:24` 
un, voi. Direi più simile a delta, perché è correlato al passo, ma può essere scelto in questi tre modi, o regola fissa a priori, basata su considerazioni empiriche o teoriche. Potete scegliere una strategia adattiva, per esempio, potete scegliere un tasso di apprendimento grande, un passo grande all'inizio, e poi scegliere tassi di apprendimento o passi sempre più piccoli quando vi state muovendo, si spera, verso il minimo.

`00:56:08` 
Oppure per convalida incrociata, facendo qualche usando, quando avete il dataset, dividete il vostro dataset in training, validazione e test. State usando il dataset di validazione durante la fase di training per calcolare l'errore di validazione e trovare qual è il miglior passo, la migliore opzione di valore.

`00:56:39` 
Quindi queste sono tre tipi di strategie che, in questo contesto, sono applicate ai parametri di soglia, ma sono anche usate in altri contesti. Che dire di delta? Delta... Non ci sono, in generale, ricette o suggerimenti particolari, a parte il fatto che, come vi ho detto, è di solito scelto nell'intervallo tra 1 e 1.5.

`00:57:12` 
Ricordate che la convergenza è assicurata se delta è 0.02, okay? E dato che delta è la lunghezza del passo che state facendo, se scegliete un delta grande, state facendo un passo lungo lungo una data direzione, okay? Se scegliete un delta piccolo, ovviamente, quando avete un delta grande, quello che può succedere è che potete avere...

`00:57:51` 
Più grande. salti e a volte la convergenza può essere influenzata da questi grandi passi che state usando. Se state usando un delta piccolo, ovviamente dovrete fare un numero maggiore di passi per raggiungere la convergenza, ma l'algoritmo complessivo sarà più stabile. È più.

`00:58:24` 
probabile che non affronterete alcuna instabilità durante l'algoritmo, durante il calcolo. Okay qui c'è, ho solo registrato il. quindi il codice che sarà usato per l'implementazione di impulso. Qui ho anche riportato alcuni degli altri metodi che possono essere.

`00:59:05` 
usati per risolvere lo stesso problema. Ho riportato queste slide anche per i progetti. Quindi se siete interessati a questo problema, completamento di matrici in generale, forse una possibilità è esplorare e implementare alcuni degli altri metodi.

`00:59:35` 
che sono stati proposti in letteratura per risolvere questo tipo di problema. L'idea del completamento di matrici che abbiamo introdotto nel contesto di questo problema Netflix è in realtà più generale, e appare in molti altri contesti, per esempio, nella visione artificiale, quando avete, per esempio, durante il laboratorio, vedrete la rimozione dello sfondo in un video.

`01:00:20` 
O ovviamente completamento di video o pittura di immagini. Quando avete un'identificazione del sistema e avete, per esempio, volete fare affidamento su misurazioni. uh date da alcuni sensori e avete dati mancanti dai sensori quindi potete usare la stessa idea, in bioinformatica in genomica.

`01:00:51` 
un algoritmo che può essere usato nella ricostruzione di stati quantistici. il numero cinque è quello che abbiamo visto e poi anche nelle applicazioni finanziarie per esempio se avete, Se volete fare un'ottimizzazione del portafoglio con rendimenti mancanti, quindi non avete molte informazioni sui rendimenti di un portafoglio per tutte le possibilità, potete cercare di usare questo tipo di strategia.

`01:01:30` 
Quindi, anche se abbiamo visto per un problema molto specifico, il framework teorico che abbiamo ideato è in realtà adatto per affrontare tutti questi altri problemi che essenzialmente possono essere riformulati nello stesso framework. Bene, qui era solo l'esempio dell'invasione. E qui c'è...

---

## Parte 4: Support Vector Machines (Introduzione)

## Support Vector Machines

`01:02:02` 
Riassunto di quello che abbiamo visto e i punti importanti che abbiamo visto finora. Quindi, l'idea è... uguaglianze fondamentali nel basso rango, passando da un problema non convesso a uno convesso sfruttando la norma nucleare, e poi ideando una procedura iterativa basata su SVD, e in particolare restringendo i singoli valori.

`01:02:36` 
Okay? Okay, ora possiamo passare al secondo argomento di questa lezione. Se ricordate, quando avevamo considerato il...

`01:03:10` 
problema dei minimi quadrati, ho menzionato che, e in particolare i metodi di Kalman, una possibilità era sviluppare quell'idea, e quello sviluppo è essenzialmente ciò che crea il framework delle cosiddette macchine a vettori di supporto.

`01:03:42` 
In particolare, in questo contesto, poi entreremo più nei dettagli, ci sono due grandi famiglie di problemi. La classificazione a vettori di supporto e la regressione a vettori di supporto. Qual è la differenza? La differenza è che in un caso, classificazione a vettori di supporto, quello che vogliamo fare è venire con un confine di separazione lineare tra dati che sono sparsi nel piano o nello spazio.

`01:04:19` 
D'altra parte, nella regressione a vettori di supporto, volete in qualche modo adattare... L'iperpiano, fate attenzione che in questo secondo caso, quando parlo di iperpiano, intendo iperpiano, ma con l'albero del kernel. Quindi non significa necessariamente che il seme sia necessariamente lineare.

`01:04:56` 
Dato che state sfruttando l'albero del kernel qui, e sappiamo che in realtà l'albero del kernel sta facendo qualcosa sfruttando la funzione kernel, state facendo qualcosa che è intrinsecamente non lineare, lo state rendendo lineare passando a una dimensione superiore. Okay, quindi.

`01:05:29` 
Cosa è ottimale nei due casi? Per la classificazione, ottimale significa che ha i margini massimi dai punti dati più vicini di qualsiasi classe. Quindi, significa che quando avete il...

`01:06:18` 
Okay, ecco un esempio molto semplice, sto generando due cluster e poi sto usando il singolare, scusate, la classificazione a vettori di supporto per trovare, okay, questa è l'area, okay? Quindi vogliamo, come abbiamo visto qui, ottimale significa ottenere il margine massimo dai punti dati più vicini di qualsiasi cluster.

`01:06:56` 
Quindi significa che quello che stiamo facendo è considerare essenzialmente questa striscia, e questa striscia è una sorta di area, quel cubo. Che separa i due cluster, okay? Che dire della regressione?

`01:07:27` 
Ottenete questo ottimale significa che ha quanti più punti dati possibile all'interno di un cosiddetto tubo esodolucente, e questo tubo è...

`01:08:11` 
Quindi, essenzialmente, in... In blu avete i dati originali, e in rosso avete i cosiddetti vettori di supporto. Quindi l'area ombreggiata è quella che viene chiamata il tubo epsilon, qui epsilon è 0.1, e la linea rossa è il modello, la regressione che è stata ottenuta con questo metodo.

`01:08:47` 
L'idea è che questi elementi, questi dati che sono all'interno del tubo epsilon, non sono significativi dal punto di vista della regressione. Quindi solo quello rosso, che sono chiamati vettori di supporto, perché quei valori che sono usati per costruire la regressione, sono quelli importanti. Okay?

`01:09:29` 
Okay, quindi l'idea è simile a quella che abbiamo visto in precedenza. Avete una funzione di perdita U2P2, che ora è, Anche da questa espressione, che dipende da epsilon. Epsilon è la larghezza del tubo.

`01:10:00` 
Quindi ovviamente scegliendo epsilon più grande, state dicendo che volete ridurre il numero di vettori di supporto perché nella maggior parte dei casi, allargando il tubo epsilon, molti dei dati cadranno all'interno dell'area ombreggiata.

`01:10:31` 
Quindi, in pratica, qui, state dicendo che se siete nell'area tra meno epsilon e epsilon, se il reciproco è in quest'area, il contributo è zero. Se è al di fuori di quest'area, il contributo alla funzione ultima è significativo. Okay?

`01:11:01` 
Come possiamo formalizzare questo problema? Essenzialmente, è esattamente quello che abbiamo visto 10 minuti fa per il completamento di matrici. Anche in questo caso, vogliamo riscrivere il problema di ottimizzazione come... minimizzazione non vincolata dove abbiamo vogliamo minimizzare la norma del vettore w.

`01:11:38` 
e poi abbiamo qualche termine aggiuntivo che coinvolge quelle variabili c i e c i star che sono chiamate le variabili di slack uh che sono usate per uh per imporre queste condizioni okay quindi queste condizioni essenzialmente sono uh ovviamente il prodotto tra uh uh scusate il le due uh le.

`01:12:13` 
variabili selezionate dovrebbero essere positive o uguali a zero e qui state essenzialmente dicendo che, uh volete essere in grado di, discriminare tra oggetti che sono o sopra o sotto la linea che volete ottenere. Ricordate che i parametri w, b, c, i, c, i, star devono essere calcolati.

`01:12:46` 
Quindi come potete vedere dovete minimizzare qui rispetto a w, b, c, e c star. E c è il parametro di regolarizzazione che vi dice il bilanciamento tra la piattezza del risultante.

`01:13:25` 
filtro e la tolleranza per gli errori che state calcolando. Come risolvere questo problema? È il formalismo usuale. Potete calcolare il Lagrangiano di questa funzione e impostate a zero tutte le derivate. Okay, correggerò quello.

`01:13:55` 
Okay. E queste quattro condizioni sono quelle che vi permettono di venire con la soluzione del problema. Okay? Quindi essenzialmente, una volta che avete queste condizioni, potete tornare alla formulazione e potete calcolare la soluzione. Una cosa importante è la prima formula che abbiamo là.

`01:14:27` 
w uguale alla somma o i che varia da 1 a n di alpha i meno alpha i star per x i. Ricordate che il. Quella formula dovrebbe ricordarvi il teorema che abbiamo visto nella lezione precedente.

`01:15:04` 
In particolare, dato che X, I sono i dati, i punti dati originali, potete vedere che W, la soluzione è una combinazione lineare dei dati. Quindi anche in questo caso, possiamo vedere che la soluzione, omega, può essere scritta in termini dei dati originali.

`01:15:37` 
E se volete usare una rappresentazione non lineare dei dati, potete sempre applicare la K-metrica come abbiamo visto prima. Grazie. Una volta che avete W, beta, e V, allora se volete fare un'inferenza, una previsione, allora potete sfruttare l'espressione per W e per V, e potete venire con questa formula.

`01:16:16` 
E i coefficienti di questa formula sono i cosiddetti vettori di supporto. Lasciatemi solo mostrarvi nell'esempio precedente, qui sto usando l'implementazione che è disponibile in ScikitLearn per la regressione.

`01:16:53` 
Come potete vedere, dove trovate l'oggetto regressione a vettori di supporto, potete usare kernel diversi, e qui ho usato l'RBF, la funzione di base radiale, in altre parole, il kernel Gaussiano. È chiaro che se usate per gli stessi dati un kernel diverso, per esempio, un polinomio di grado 4, quello che vedrete è che il fitting che state generando è totalmente diverso.

`01:17:56` 
Okay, eccolo. Quindi come potete vedere, abbiamo un calore molto diverso dei dati. Ovviamente, se considerate qui grado uno, significa che state considerando un'approssimazione lineare dei dati.

`01:18:28` 
Okay, quindi nessun caramello, state solo usando polinomi lineari. E come potete vedere, in questo caso, l'idea è che se state usando questi tipi di kernel, avete molti vettori di supporto, il che significa che potete vedere tutti i, cinque o forse sei dati originali che riempiono il tubo epsilon.

`01:19:05` 
Questo è ragionevole perché i dati che abbiamo generato, non seguono la tendenza lineare, a meno che non abbiamo un numero molto grande di vettori di supporto. Ovviamente, l'idea della discussione sui vettori di supporto in generale è venire con una rappresentazione con un numero di vettori di supporto che sia il più piccolo possibile, come al solito, perché volete essere in grado di rappresentare la tendenza dei vostri dati.

`01:19:40` 
usando pochi elementi. Quindi, se torniamo all'RBF... Potete vedere che ci sono molti più punti che sono all'interno del cubo epsilon e il numero di vettori di supporto è molto più piccolo di quello che abbiamo visto prima.

`01:20:34` 
Okay, in pratica, per i vettori di supporto, quello che è importante è, l'idea principale è che dovete... Modificare la funzione di costo che state minimizzando, e di solito la funzione di costo che considerate è la cosiddetta funzione di perdita hinge.

`01:21:04` 
Nell'implementazione, per esempio, quella che vi ho mostrato, che è disponibile in machine learning, in particolare per la classificazione, quando usate l'SVC, quindi la funzione di classificazione a vettori di supporto da Scikit-Learn, state, automaticamente usando la funzione di perdita hinge. Quindi è qualcosa che è implicito nell'implementazione. Durante il laboratorio vedrete come modificare la.

`01:21:40` 
macchina che avete visto per i minimi quadrati per introdurre i vettori di supporto, in particolare la regressione ma anche la classificazione. E in pratica questo ammonta a considerare la funzione di perdita che abbiamo visto prima o questa per la precedente era per la regressione, questa è per.

`01:22:12` 
la classificazione. Ma l'idea è esattamente la stessa. Quindi a parte il nome, L'idea della regressione o classificazione a vettori di supporto è esattamente la stessa che abbiamo visto per i minimi quadrati, a parte il fatto che state cambiando la funzione di perdita che state usando. E con questi, dato che in quella funzione di perdita state introducendo un parametro che è correlato all'ampiezza,

`01:22:46` 
alla larghezza del tubo epsilon che state considerando, l'implicazione è che state selezionando solo alcuni punti che sono significativi per creare o il confine di separazione nella classificazione o per creare la regressione nel caso della regressione. Okay? Okay. Penso che per oggi possiamo fermarci qui.

`01:23:22` 
Grazie.
---

## Appendice: Formule Chiave e Riferimenti

### Matrix Completion: Formulazioni

**Problema NP-Hard (Rank Minimization)**:
`
minimize    rank(M)
subject to  P_O(M) = P_O(X)
`

**Problema Convesso (Nuclear Norm Minimization)**:
`
minimize    ||M||_* = S? s?(M)
subject to  P_O(M) = P_O(X)
`

### Gradi di Libert�

**Matrice Full-Rank**: DoF = n  d

**Matrice Rank-r**: DoF = (n + d)  r - r = r(n + d - r)

**Esempio Netflix** (n=480K, d=17K, r=100):
- Full: 8.16  10 parametri
- Rank-100: ~5  10 parametri
- **Riduzione**: 99.4% (164x compression)

### Nuclear Norm

**Definizione**: ||M||_* = S? s? = tr((M^T M))

**Propriet�**:
- Convex norm
- ||M||_* = ||s(M)|| (l norm dei valori singolari)
- Dual norm: ||M||_* = max_{||Z||_op1} tr(M^T Z)
- Promuove low-rank (analogamente a l che promuove sparsity)

### Teorema di Cand�s-Recht (2009)

**Sample Complexity**: Per exact recovery con probabilit�  1 - cn:

`
m  C  �  n^1.2  r  log(n)
`

Dove:
- m: numero di osservazioni
- �: parametro di incoerenza (�  1)
- n: max(n, n)
- r: rank
- C, c: costanti universali

**Incoerenza**: �(U) = (n/r)  max_i ||U_{i,:}||

- � = 1: perfectly incoherent (best case)
- � = n/r: maximally coherent (worst case)

### SVT Algorithm (Singular Value Thresholding)

**Soft-Thresholding Scalare**:
`
S_t(x) = sign(x)  max(|x| - t, 0)
       = { x - t    if x > t
         { 0        if |x|  t
         { x + t    if x < -t
`

**SVT Operator**:
`
D_t(Y) = U  S_t(S)  V^T
       = S? S_t(s?)  u?v?^T
`

Dove Y = USV^T (SVD)

**Proximal Interpretation**:
`
D_t(Y) = argmin_M { (1/2)||M - Y||_F + t||M||_* }
`

**Algoritmo Iterativo**:
`
1. Inizializza: Y = P_O(X)
2. Per k = 0, 1, 2, ...
   a. M_k = D_t(Y_k)              [SVT step: shrink singular values]
   b. Y_{k+1} = Y_k + d P_O(X - M_k)  [Data fidelity step]
3. Stop quando ||P_O(M_k - X)||_F / ||X||_F < e
`

**Parametri**:
- t: threshold (t ~ (nd), tuning via cross-validation)
- d: step size (d  (0, 2), tipicamente d  [1.2, 1.5])

**Convergenza**:
- Lineare: ||M_k - M*||  C  ?^k
- Rate dipende da d
- Complessit�: O(k  r  n  d) per iterazione

### Analogia Compressed Sensing  Matrix Completion

| Compressed Sensing | Matrix Completion |
|--------------------|-------------------|
| Vettore x  Rn | Matrice M  Rn?? |
| Sparsity: ||x|| | Low-rank: rank(M) |
| Convex relaxation: ||x|| | Convex relaxation: ||M||_* |
| Misurazioni: Ax = b | Osservazioni: P_O(M) = P_O(X) |
| RIP condition | Incoherence condition |
| Recovery: m  k log(n/k) | Recovery: m  �nr log(n) |

### Norme Matriciali

| Norma | Definizione | Interpretazione |
|-------|-------------|-----------------|
| Frobenius | ||M||_F = (S?? M??) = (S? s?) | l norm of entries |
| Nuclear | ||M||_* = S? s? | l norm of sing. vals |
| Spectral | ||M||_op = max s? = s | l norm of sing. vals |
| l | ||M|| = S?? |M??| | Sum of absolute values |
| l | ||M||_ = max?? |M??| | Max entry magnitude |

**Relazioni**:
- ||M||_op  ||M||_F  r  ||M||_op
- ||M||_*  r  ||M||_F
- ||M||_op  ||M||_*  r  ||M||_op

### Applicazioni del Matrix Completion

**1. Recommender Systems**:
- Netflix Prize (rating prediction)
- E-commerce (product recommendation)
- Social networks (link prediction)

**2. Computer Vision**:
- Video background subtraction
- Image inpainting
- Structure from motion
- Photometric stereo

**3. System Identification**:
- Dynamic system modeling
- Control theory applications

**4. Genomics**:
- Gene-gene interaction networks
- Drug-target interaction prediction

**5. Finance**:
- Missing data imputation
- Covariance matrix estimation

### Complessit� Computazionale

**SVT Iteration**:
- SVD parziale (top-r): O(r  n  d)
- Soft-thresholding: O(r)
- Projection: O(m)
- **Totale per iterazione**: O(k  r  nd)

**Convergence**:
- Tipicamente k ~ 10-100 iterazioni
- Per Netflix (n=480K, d=17K, r=100, k=50):
  * Per-iteration: ~4  10 ops
  * Total: ~2  10 ops (tractable!)

### SVM - Introduzione

**e-insensitive Loss** (Regression):
`
L_e(y, f(x)) = max(0, |y - f(x)| - e)
                = { 0           if |y - f(x)|  e
                  { |y - f(x)| - e   otherwise
`

**Hinge Loss** (Classification):
`
L_hinge(y, f(x)) = max(0, 1 - y  f(x))
`

**Kernel Trick**:
`
f(x) = S? a? K(x?, x) + b
`

Kernels comuni:
- Linear: K(x, x') = x^T x'
- RBF (Gaussian): K(x, x') = exp(-?||x - x'||)
- Polynomial: K(x, x') = (x^T x' + c)^d

**Representer Theorem**: La soluzione ottima ha forma
`
f*(x) = S?n a? K(x?, x)
`

Dove solo **support vectors** (a? > 0) contribuiscono.

---

## Riferimenti Bibliografici

1. **Cand�s, E. J., & Recht, B. (2009)**. "Exact matrix completion via convex optimization." *Foundations of Computational Mathematics*, 9(6), 717-772.

2. **Cand�s, E. J., & Tao, T. (2010)**. "The power of convex relaxation: Near-optimal matrix completion." *IEEE Transactions on Information Theory*, 56(5), 2053-2080.

3. **Cai, J. F., Cand�s, E. J., & Shen, Z. (2010)**. "A singular value thresholding algorithm for matrix completion." *SIAM Journal on Optimization*, 20(4), 1956-1982.

4. **Recht, B., Fazel, M., & Parrilo, P. A. (2010)**. "Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization." *SIAM review*, 52(3), 471-501.

5. **Keshavan, R. H., Montanari, A., & Oh, S. (2010)**. "Matrix completion from a few entries." *IEEE transactions on information theory*, 56(6), 2980-2998.

---

**Fine Lezione 14 - Matrix Completion & SVT**

*Prossima lezione: Lab 4 - Implementazione SVT e applicazioni*
