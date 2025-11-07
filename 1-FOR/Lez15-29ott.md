# Lezione 15 - 29 Ottobre: Introduzione alla Programmazione Lineare e Geometria degli Insiemi Convessi

---

## 📑 Indice

1. **[Esempio Introduttivo: Produzione di Porte e Finestre](#1-esempio-introduttivo-produzione-di-porte-e-finestre)** `00:00:14 - 00:08:29`
   - 1.1 [Descrizione del problema](#11-descrizione-del-problema)
   - 1.2 [Formulazione del modello](#12-formulazione-del-modello)
   - 1.3 [Notazione matriciale](#13-notazione-matriciale)

2. **[Metodo Geometrico di Risoluzione](#2-metodo-geometrico-di-risoluzione)** `00:08:29 - 00:18:09`
   - 2.1 [Rappresentazione grafica](#21-rappresentazione-grafica)
   - 2.2 [Gradiente e regione ammissibile](#22-gradiente-e-regione-ammissibile)
   - 2.3 [Linee di livello e soluzione ottima](#23-linee-di-livello-e-soluzione-ottima)
   - 2.4 [Caratterizzazione dell'ottimalità](#24-caratterizzazione-dellottimalità)

3. **[Concetti di Base: Convessità](#3-concetti-di-base-convessità)** `00:20:35 - 00:24:47`
   - 3.1 [Insieme convesso](#31-insieme-convesso)
   - 3.2 [Combinazione convessa](#32-combinazione-convessa)
   - 3.3 [Poliedro convesso](#33-poliedro-convesso)

4. **[Coni e Coni Poliedrici](#4-coni-e-coni-poliedrici)** `00:24:47 - 00:36:21`
   - 4.1 [Definizione di cono](#41-definizione-di-cono)
   - 4.2 [Cono convesso](#42-cono-convesso)
   - 4.3 [Cono poliedrico](#43-cono-poliedrico)
   - 4.4 [Cono finitamente generato](#44-cono-finitamente-generato)

5. **[Caratterizzazione dei Vertici](#5-caratterizzazione-dei-vertici)** `00:36:58 - 00:55:32`
   - 5.1 [Definizione 1: Punto estremo](#51-definizione-1-punto-estremo)
   - 5.2 [Definizione 2: Vertice](#52-definizione-2-vertice)
   - 5.3 [Definizione 3: Soluzione di base](#53-definizione-3-soluzione-di-base)
   - 5.4 [Equivalenza delle definizioni](#54-equivalenza-delle-definizioni)

6. **[Decomposizione Poliedro = Politopo + Cono](#6-decomposizione-poliedro--politopo--cono)** `00:55:32 - 01:05:35`
   - 6.1 [Teorema di decomposizione](#61-teorema-di-decomposizione)
   - 6.2 [Esempio grafico](#62-esempio-grafico)
   - 6.3 [Conseguenza: ottimo finito sui vertici](#63-conseguenza-ottimo-finito-sui-vertici)

7. **[Teorema dell'Iperpiano Separatore](#7-teorema-delliperpiano-separatore)** `01:08:52 - 01:11:05`
   - 7.1 [Enunciato del teorema](#71-enunciato-del-teorema)
   - 7.2 [Importanza per la convergenza dell'algoritmo](#72-importanza-per-la-convergenza-dellalgoritmo)

8. **[Esercizio: Algoritmo di Edmonds-Karp](#8-esercizio-algoritmo-di-edmonds-karp)** `01:11:05 - 01:26:08`
   - 8.1 [Setup iniziale](#81-setup-iniziale)
   - 8.2 [Iterazione 1](#82-iterazione-1)
   - 8.3 [Iterazione 2](#83-iterazione-2)
   - 8.4 [Iterazione 3](#84-iterazione-3)
   - 8.5 [Iterazione finale e taglio](#85-iterazione-finale-e-taglio)
   - 8.6 [Analisi post-ottimalità](#86-analisi-post-ottimalità)

---

## 1. Esempio Introduttivo: Produzione di Porte e Finestre

### 1.1 Descrizione del problema

`00:00:14`

🏭 **Problema di production mix:**

Simile all'esempio dei telefoni cellulari visto nella prima lezione.

**Prodotti:**
- **Porte in alluminio** (aluminum doors)
- **Finestre in legno** (wooden windows)

`00:00:46`

**Ricavi:**
- Porte: **30 €** per unità
- Finestre: **50 €** per unità

`00:01:19`

**Risorse disponibili:**

Abbiamo tre lavoratori con disponibilità limitate:

| Lavoratore | Disponibilità | Tempo per porta | Tempo per finestra |
|-----------|--------------|-----------------|-------------------|
| **Fabbro (Smith)** | 4 ore | 1 ora | 0 ore |
| **Falegname (Carpenter)** | 12 ore | 0 ore | 2 ore |
| **Assemblatore (Assembler)** | 18 ore | 3 ore | 2 ore |

`00:01:55`

🎯 **Obiettivo:** Pianificare la produzione per **massimizzare il ricavo totale**.

### 1.2 Formulazione del modello

`00:02:29`

**Variabili decisionali:**

`00:03:01`

$$
\begin{aligned}
x_D &= \text{numero di porte da produrre} \geq 0 \\
x_W &= \text{numero di finestre da produrre} \geq 0
\end{aligned}
$$

`00:03:33`

**Funzione obiettivo:**

`00:04:07`

$$
\max \; Z = 30 \cdot x_D + 50 \cdot x_W
$$

Moltiplichiamo il ricavo per porta × numero porte + ricavo per finestra × numero finestre.

`00:04:42`

**Vincoli di disponibilità risorse:**

`00:05:19`

Non possiamo eccedere la disponibilità per ogni risorsa.

**Vincolo 1 - Fabbro:**
$$
1 \cdot x_D + 0 \cdot x_W \leq 4
$$

`00:05:19`

**Vincolo 2 - Falegname:**
$$
0 \cdot x_D + 2 \cdot x_W \leq 12
$$

**Vincolo 3 - Assemblatore:**
$$
3 \cdot x_D + 2 \cdot x_W \leq 18
$$

`00:06:02`

**Vincoli di non negatività:**

$$
x_D \geq 0, \quad x_W \geq 0
$$

`00:06:33`

✅ **Formulazione completa:**

$$
\begin{aligned}
\max \quad & 30 x_D + 50 x_W \\
\text{s.t.} \quad & x_D \leq 4 \\
& 2 x_W \leq 12 \\
& 3 x_D + 2 x_W \leq 18 \\
& x_D, x_W \geq 0
\end{aligned}
$$

### 1.3 Notazione matriciale

`00:06:33`

📐 **Forma standard:**

Ricordiamo che useremo la notazione matriciale:

$$
\begin{aligned}
\max \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A} \mathbf{x} \leq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{aligned}
$$

`00:07:18`

**Identificazione delle matrici:**

**Vettore dei coefficienti obiettivo (riga):**
$$
\mathbf{c} = \begin{pmatrix} 30 & 50 \end{pmatrix}
$$

**Matrice dei coefficienti tecnologici (3 righe × 2 colonne):**
$$
\mathbf{A} = \begin{pmatrix}
1 & 0 \\
0 & 2 \\
3 & 2
\end{pmatrix}
$$

Ogni **riga** corrisponde a una risorsa, ogni **colonna** a un prodotto.

`00:07:56`

**Vettore termini noti (colonna):**
$$
\mathbf{b} = \begin{pmatrix} 4 \\ 12 \\ 18 \end{pmatrix}
$$

**Vettore variabili (colonna):**
$$
\mathbf{x} = \begin{pmatrix} x_D \\ x_W \end{pmatrix}
$$

---

## 2. Metodo Geometrico di Risoluzione

### 2.1 Rappresentazione grafica

`00:08:29`

Nella prima lezione abbiamo visto che con **due variabili** possiamo fare un disegno della regione ammissibile e della funzione obiettivo.

**Assi:**
- Asse orizzontale: $x_D$ (porte)
- Asse verticale: $x_W$ (finestre)

### 2.2 Gradiente e regione ammissibile

`00:09:13`

**Vincolo 1:** $x_D \leq 4$

**Bordo:** $x_D = 4$ (linea verticale)

`00:09:49`

💡 **Come identificare la parte ammissibile?**

Usiamo il **gradiente**. Il gradiente di una funzione lineare è dato dai coefficienti della prima riga di A.

$$
\nabla A_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

Vettore orizzontale che punta a destra.

`00:10:22`

📌 **Regola:**

Poiché il vincolo è **≤**, sul bordo siamo esattamente al limite. Se andiamo oltre il bordo, la funzione **cresce** (nella direzione del gradiente), quindi è **vietato**.

La parte ammissibile è **opposta** al gradiente.

Poiché tutti i vincoli sono ≤, la regione ammissibile è sempre **opposta al gradiente**.

**Vincolo 2:** $2x_W \leq 12$ → $x_W \leq 6$

**Bordo:** $x_W = 6$ (linea orizzontale)

$$
\nabla A_2 = \begin{pmatrix} 0 \\ 2 \end{pmatrix}
$$

Vettore verticale che punta verso l'alto.

Regione ammissibile: **sotto** la linea.

`00:11:04`

**Vincolo 3:** $3x_D + 2x_W \leq 18$

**Bordo:** $3x_D + 2x_W = 18$

Punti di intersezione con gli assi:
- Se $x_D = 0$: $x_W = 9$
- Se $x_W = 0$: $x_D = 6$

$$
\nabla A_3 = \begin{pmatrix} 3 \\ 2 \end{pmatrix}
$$

Vettore sempre **ortogonale** al bordo, regione ammissibile opposta.

`00:11:36`

**Vincoli di segno:**

Possono essere visti come vincoli "dummy":

$$
-x_D \leq 0 \quad \text{(equivalente a } x_D \geq 0\text{)}
$$

$$
-x_W \leq 0 \quad \text{(equivalente a } x_W \geq 0\text{)}
$$

`00:12:07`

Gradienti:
- $\nabla(-x_D) = (-1, 0)$ → punta a sinistra
- $\nabla(-x_W) = (0, -1)$ → punta in basso

`00:12:48`

**Regione ammissibile finale:**

L'**intersezione** di tutte le parti ammissibili è la **regione blu** evidenziata.

Qualsiasi punto (anche frazionario) all'interno di questa regione è una **soluzione ammissibile**:
- Dà un certo ricavo
- Soddisfa tutti i vincoli di risorsa
- È implementabile in pratica

### 2.3 Linee di livello e soluzione ottima

`00:13:20`

Abbiamo **infinite soluzioni**. Dobbiamo trovare la migliore, quella che massimizza $30x_D + 50x_W$.

`00:13:54`

💡 **Metodo delle linee di livello:**

Prendiamo una soluzione qualsiasi, es. $(x_D, x_W) = (1, 1)$.

**Ricavo:** $30(1) + 50(1) = 80$ €

`00:14:26`

**Domanda:** Quali altre soluzioni danno lo stesso ricavo?

**Risposta:** Tutte le soluzioni sulla **linea di livello**:

$$
30 x_D + 50 x_W = 80
$$

Qualsiasi punto su questa linea vale 80 €.

**Possiamo fare di meglio?**

`00:14:56`

Sì! Se spostiamo questa linea **verso l'alto** seguendo il gradiente $\nabla c = (30, 50)$, abbiamo ancora intersezione con la regione ammissibile e possiamo migliorare la soluzione.

`00:15:44`

**Esempio:** Spostiamo fino al punto (2, 2).

**Ricavo:** $30(2) + 50(2) = 60 + 100 = 160$ €

Abbiamo **raddoppiato** entrambe le variabili e il ricavo è raddoppiato.

📌 **Proprietà importante della PL:** **Proporzionalità** nella funzione obiettivo.

`00:16:21`

**Continuiamo a spostare la linea...**

Possiamo spostare la linea di livello verso l'alto fino a quando abbiamo **un solo punto** di intersezione.

Qual è quel punto?

**Il vertice in alto a destra della regione ammissibile!**

Se spostiamo ancora un po' la linea lungo il gradiente, finiamo senza intersezione.

`00:16:53`

✅ **Soluzione ottima:**

**Coordinate:** $(x_D^*, x_W^*) = (2, 6)$

**Ricavo ottimo:**
$$
Z^* = 30(2) + 50(6) = 60 + 300 = 360 \text{ €}
$$

### 2.4 Caratterizzazione dell'ottimalità

`00:17:33`

📊 **Metodo geometrico:**

Questo metodo può essere applicato solo quando possiamo vedere il disegno:
- **R²** (2 variabili): facile
- **R³** (3 variabili): se siamo bravi a fare i grafici
- **R⁴ o più**: impossibile visualizzare

`00:18:09`

💡 **Idea per l'algoritmo:**

Possiamo sfruttare le **proprietà geometriche** per:
1. **Caratterizzare** la soluzione ottima
2. Se siamo nella soluzione ottima, **certificare l'ottimalità**
3. Altrimenti, trovare un modo per **spostarci** verso un punto migliore

Questa sarà l'**essenza della programmazione lineare**:
- Riconoscere l'ottimalità
- Usare le proprietà per muoversi da un punto non ottimo verso uno migliore

---

## 3. Concetti di Base: Convessità

### 3.1 Insieme convesso

`00:20:35`

📚 **Definizione di insieme convesso:**

`00:21:49`

Un insieme S è **convesso** se:

> Per ogni coppia di punti x, y ∈ S, il segmento che li congiunge appartiene interamente a S.

**Esempi:**

```
CONVESSO (patata):        NON CONVESSO (fagiolo):
    ●─────●                   ●─────●
   /       \                 /   ╱   \
  /         \               /  ╱      \
 ●───────────●             ●─╱─────────●
                            ╱ (parte fuori)
```

`00:22:27`

**Definizione matematica:**

$$
S \text{ è convesso} \iff \forall x, y \in S: \; \lambda x + (1-\lambda)y \in S \quad \forall \lambda \in [0,1]
$$

Il segmento è l'insieme delle **combinazioni convesse** di x e y.

### 3.2 Combinazione convessa

`00:23:03`

**Combinazione convessa di due punti:**

$$
z = \lambda x + (1-\lambda)y, \quad \lambda \in [0,1]
$$

- Se $\lambda = 0$: siamo in y
- Se $\lambda = 1$: siamo in x
- Se $0 < \lambda < 1$: siamo sul segmento tra x e y

`00:23:34`

**Generalizzazione a n punti:**

$$
z = \sum_{i=1}^n \lambda_i x_i
$$

con:
$$
\lambda_i \geq 0 \quad \text{e} \quad \sum_{i=1}^n \lambda_i = 1
$$

📌 **Importante:** La somma dei coefficienti deve essere **1**.

Qui: $\lambda + (1-\lambda) = 1$ ✓

### 3.3 Poliedro convesso

`00:24:06`

**Concerto oggi alle 18:00!** (annuncio del professore)

`00:24:47`

📚 **Definizione di poliedro convesso:**

Un **poliedro convesso** P è un insieme convesso definito da un **numero finito** di disequazioni lineari:

$$
P = \{x \in \mathbb{R}^n : \mathbf{A}x \leq \mathbf{b}\}
$$

Dove A ha **m righe** (numero finito di vincoli).

`00:25:22`

⚠️ **Perché "numero finito" è importante?**

Se avessimo **infiniti** vincoli, potremmo ottenere, ad esempio, un **cerchio** usando tutti i piani tangenti (infiniti).

Ma un cerchio **non è** un poliedro convesso perché richiede infiniti vincoli.

---

## 4. Coni e Coni Poliedrici

### 4.1 Definizione di cono

`00:25:55`

💡 **Idea di cono:**

Generalizzazione del concetto di **semiretta** (half-line).

`00:26:32`

**Semiretta:** Se ho una semiretta con origine O e un punto x sulla semiretta, allora **qualsiasi multiplo** di x appartiene alla semiretta.

$$
x \in \text{semiretta} \implies \alpha x \in \text{semiretta} \quad \forall \alpha \geq 0
$$

- Se $\alpha = 0$: siamo nell'origine
- Se $\alpha > 1$: ci allontaniamo da O
- Se $0 < \alpha < 1$: siamo tra O e x

`00:27:05`

**Cono:** Generalizzazione a più semirette.

📚 **Definizione di cono:**

C è un **cono** se:

$$
x \in C \implies \lambda x \in C \quad \forall \lambda \geq 0
$$

`00:27:44`

**Esempio:** Insieme di semirette con origine comune.

Prendiamo tre punti: $x_1, x_2, x_3$ nel cono.

Le semirette generate sono:
- $\{\lambda x_1 : \lambda \geq 0\}$
- $\{\lambda x_2 : \lambda \geq 0\}$
- $\{\lambda x_3 : \lambda \geq 0\}$

`00:28:15`

❓ **È un insieme convesso?**

Prendiamo due punti sul bordo. Il segmento che li congiunge appartiene al cono?

**Risposta:** Non necessariamente!

Quindi la definizione di cono **non implica** convessità. Dobbiamo imporla.

### 4.2 Cono convesso

`00:28:47`

📚 **Definizione di cono convesso:**

C è un **cono convesso** se soddisfa **entrambe** le proprietà:
1. Proprietà di cono
2. Convessità

Formalmente:

$$
\forall x, y \in C, \; \forall \lambda, \mu \geq 0: \quad \lambda x + \mu y \in C
$$

È una **combinazione lineare non negativa** di x e y.

`00:29:56`

**Rappresentazione grafica:**

```
        y (su una semiretta)
       /
      /
     / λx (su x)
    /   \
   /     \ μy
  /       \
 O────────x  λx + μy (interno al cono)
```

`00:30:27`

- x è su una semiretta
- y è su un'altra semiretta
- $\lambda x$ (con $\lambda > 1$ nell'esempio) si allontana da O lungo x
- $\mu y$ (con $0 < \mu < 1$) è tra O e y

`00:31:18`

Se sommiamo $\lambda x + \mu y$, otteniamo un punto **interno** al cono.

Variando opportunamente $\lambda$ e $\mu$, possiamo costruire **tutti i punti** nell'angolo definito dalle due semirette.

`00:31:50`

⚠️ **Limitazione in R²:**

In R², il massimo numero di semirette che possiamo avere per definire un cono convesso è finito.

`00:32:25`

**In R³:** Cono gelato infinito!

Possiamo avere **infinite semirette** giacenti sulla superficie circolare del cono.

`00:33:35`

Quindi la definizione di cono convesso è **troppo generale** per i nostri scopi.

Dobbiamo occuparci di poliedri convessi, quindi vogliamo una versione "poliedrale" del cono.

### 4.3 Cono poliedrico

`00:33:35`

📚 **Definizione di cono poliedrico:**

Un **cono poliedrico** C è un poliedro convesso che è anche un cono:
- Deve partire dall'origine
- Ha **un solo vertice**: l'origine

`00:34:10`

**Formulazione con disequazioni:**

Poiché tutti i bordi del cono passano per l'origine, la definizione è:

$$
C = \{x \in \mathbb{R}^n : \mathbf{A}x \leq 0\}
$$

con **numero finito** di disequazioni.

`00:34:44`

📌 **Proprietà importante:**

Il cono poliedrico ammette una **rappresentazione alternativa**.

### 4.4 Cono finitamente generato

`00:35:18`

**Rappresentazione alternativa:** Cono generato da un numero finito di **raggi**.

I raggi sono le nostre semirette, come x e y nell'esempio precedente.

$$
C = \left\{ x \in \mathbb{R}^n : x = \sum_{i=1}^r \lambda_i y_i, \; \lambda_i \geq 0 \right\}
$$

Dove $y_1, y_2, \ldots, y_r$ sono i **generatori** (raggi) del cono.

`00:35:49`

**Esempio con 2 generatori:**

Nell'esempio grafico con x e y:

$$
C = \{\lambda_1 x + \lambda_2 y : \lambda_1, \lambda_2 \geq 0\}
$$

`00:36:21`

📌 **Importante:**

I moltiplicatori di ogni raggio devono essere **non negativi**.

Altrimenti andremmo nella regione opposta, che è vietata.

**Definizioni da ricordare:**
1. ✅ Poliedro convesso
2. ✅ Cono poliedrico
3. ✅ Cono finitamente generato

**Teorema:** Esiste un teorema che permette di passare dalla rappresentazione per disequazioni a quella per generatori e viceversa.

---

## 5. Caratterizzazione dei Vertici

### 5.1 Definizione 1: Punto estremo

`00:36:58`

🎯 **Primo obiettivo:** Caratterizzare i vertici del poliedro.

**Setup:**

```
Regione ammissibile (porte-finestre):
        │
    A₂  │  gradiente
        │
  ──────┼──────  A₁
        │╲
        │ ╲ A₃
        │  ╲
```

`00:38:15`

Prendiamo un vertice, ad esempio questo (in basso a sinistra).

📚 **Definizione di punto estremo:**

Consideriamo un poliedro P definito da un insieme finito di disequazioni.

Un punto x ∈ P è un **punto estremo** se:

> Non esistono due punti y, z ∈ P, entrambi diversi da x, tali che x possa essere scritto come combinazione convessa di y e z.

`00:39:28`

**Formalmente:**

$$
\nexists y, z \in P, \; y \neq x, \; z \neq x: \quad x = \lambda y + (1-\lambda)z \text{ per qualche } \lambda \in (0,1)
$$

`00:40:03`

**Interpretazione grafica:**

Se prendiamo due punti y e z interni a P, il segmento che li congiunge sta tutto dentro P.

Ma x **non appartiene** al segmento.

`00:40:37`

Per fare in modo che x appartenga al segmento, almeno una delle seguenti deve essere vera:
- x è uno degli estremi (y = x o z = x)
- Almeno uno dei due punti è fuori da P

`00:41:39`

❌ **Problema con questa definizione:**

È **corretta** (caratterizza effettivamente un vertice), ma è **non adatta** per un algoritmo.

Come si fa a provare che "non esistono due punti"? Troppo computazionalmente costoso!

### 5.2 Definizione 2: Vertice

`00:42:10`

💡 **Seconda definizione:** Basata sul metodo che abbiamo usato per trovare la soluzione ottima.

📚 **Definizione di vertice:**

x è un **vertice** se esiste un vettore c tale che:

$$
\mathbf{c}^T x > \mathbf{c}^T y \quad \forall y \in P, \; y \neq x
$$

`00:42:51`

**Interpretazione:**

x è il **punto ottimo** per una certa funzione obiettivo $\mathbf{c}^T x$.

Ricordate il metodo: abbiamo trovato il vertice usando c come gradiente e spostando la linea di livello.

`00:43:30`

✅ **Migliore della prima:** Almeno abbiamo un metodo per verificare se è ottimo.

❌ **Problema:** La definizione di c **non è unica**!

`00:44:07`

Possiamo usare:
- Questo gradiente c
- Quest'altro c'
- Quest'altro ancora c''

Ci sono **infinite scelte** di c che soddisfano la proprietà.

Questa variabilità **non è buona** per un algoritmo.

### 5.3 Definizione 3: Soluzione di base

`00:44:50`

💡 **Osservazione chiave:**

Guardiamo questo vertice. Come lo caratterizziamo oltre alle coordinate?

```
   │ A₂
   │
───┼───  A₁
   │╲
   │ ╲ A₃
   │  ╲
   └───►
```

`00:45:22`

È l'**intersezione** di:
- 1 piano (in R²: 1 retta)
- 2 piani (in R²: 2 rette)
- 3 piani (in R²: non possibile visualizzare)

In R² questo vertice è all'intersezione di **due bordi**.

`00:45:55`

📚 **Definizione di soluzione di base:**

Consideriamo un poliedro P e un punto $x^* \in P$.

Definiamo:
$$
I(x^*) = \{i : a_i x^* = b_i\}
$$

L'insieme degli **indici dei vincoli attivi** (che valgono come uguaglianze strette) in $x^*$.

`00:46:53`

**Esempio nel nostro problema:**

Prendiamo il punto $(x_D^*, x_W^*) = (4, 3)$.

Quali vincoli sono attivi?

`00:47:52`

Sostituendo nelle disequazioni:

1. $x_D \leq 4$: $4 = 4$ ✓ **Attivo**
2. $2x_W \leq 12$: $2(3) = 6 < 12$ ✗ Non attivo
3. $3x_D + 2x_W \leq 18$: $3(4) + 2(3) = 12 + 6 = 18$ ✓ **Attivo**

$$
I(x^*) = \{1, 3\}
$$

`00:48:23`

**Sottomatrice dei vincoli attivi:**

$$
A_{I(x^*)} = \begin{pmatrix} 1 & 0 \\ 3 & 2 \end{pmatrix}
$$

`00:49:28`

**$x^*$ è una soluzione di base se:**

Tra tutti i vincoli attivi, ce ne sono **n linearmente indipendenti** (dove n = numero variabili).

**Come verificare l'indipendenza lineare?**

Calcoliamo il **determinante** di $A_{I(x^*)}$:

$$
\det(A_{I(x^*)}) = 1 \cdot 2 - 3 \cdot 0 = 2 \neq 0 \quad ✓
$$

`00:50:12`

Poiché il determinante è diverso da zero, le due disequazioni sono **linearmente indipendenti**.

Il sistema:
$$
A_{I(x^*)} x = b_{I(x^*)}
$$

ha una **soluzione unica**.

`00:50:48`

Possiamo calcolare:
$$
x = A_{I(x^*)}^{-1} b_{I(x^*)}
$$

**Verifica:** Risolvendo il sistema:
$$
\begin{cases}
x_D = 4 \\
3x_D + 2x_W = 18
\end{cases}
$$

Otteniamo effettivamente $x^* = (4, 3)$ ✓

`00:51:18`

❌ **Controesempio:** $I = \{2, 5\}$

Se consideriamo i vincoli 2 e 5 (segno su $x_W$):

$$
A_{\{2,5\}} = \begin{pmatrix} 0 & 2 \\ 0 & -1 \end{pmatrix}
$$

`00:51:51`

$$
\det(A_{\{2,5\}}) = 0 \cdot (-1) - 0 \cdot 2 = 0
$$

Il sistema **non ha soluzione** (o ne ha infinite).

`00:52:58`

**Interpretazione geometrica:**

```
       │ A₂
       │
       │
       │ (vincoli paralleli!)
  ─────┼─────
       │
       │ A₅
```

I bordi di A₂ e A₅ sono **paralleli** → non si incontrano mai!

`00:53:38`

✅ **Questa definizione è gestibile algoritmicamente!**

Invece di fornire le coordinate, possiamo caratterizzare un vertice tramite un **sottoinsieme di indici**.

### 5.4 Equivalenza delle definizioni

`00:54:17`

📌 **Teorema:** Le tre definizioni sono **equivalenti**:

1. Punto estremo
2. Vertice
3. Soluzione di base

`00:54:58`

La dimostrazione è nelle note del corso, ma è puramente teorica.

`00:55:32`

**Utilizzo nell'algoritmo:**

Useremo la **definizione di soluzione di base** perché più maneggevole.

Ogni vertice è identificato da un sottoinsieme di indici invece delle coordinate.

---

## 6. Decomposizione Poliedro = Politopo + Cono

### 6.1 Teorema di decomposizione

`00:55:32`

🔑 **Proprietà importante:**

Questa proprietà sarà usata per dimostrare l'intuizione iniziale:

> Se il problema di ottimizzazione ha un ottimo finito, l'ottimo si trova su un vertice.

`00:56:02`

📚 **Teorema:**

Ogni poliedro convesso P può essere scritto come **somma** di:
- Un **politopo** Q (poliedro convesso finito)
- Un **cono** C (parte infinita)

$$
P = Q + C
$$

`00:56:44`

**Politopo:** Generalizzazione della combinazione convessa di punti (poliedro convesso finito).

**Cono:** Parte che va all'infinito.

### 6.2 Esempio grafico

`00:56:44`

**Grafico del poliedro:**

```
       A₄
        │╲
        │ ╲
    x₃  │  ╲     (va all'infinito)
    ●───┼───●
    │   │    ╲ A₁ (va all'infinito)
x₁●─┼───┼─────●x₂
    │   │
    │  A₂
    │
   A₃
```

`00:57:44`

Questo è P, l'intersezione di 4 disequazioni.

Il poliedro **non è finito** perché possiamo andare all'infinito in alcune direzioni.

Ma ha alcuni **vertici**: $x_1, x_2, x_3$.

`00:58:23`

**Decomposizione:**

**Politopo Q (triangolo rosso):**

`00:58:58`

Se facciamo la combinazione convessa dei tre vertici:

$$
Q = \left\{ x : x = \sum_{i=1}^3 \lambda_i x_i, \; \lambda_i \geq 0, \; \sum_{i=1}^3 \lambda_i = 1 \right\}
$$

`01:00:13`

Otteniamo il **triangolo rosso** con vertici $x_1, x_2, x_3$.

`00:59:29`

Facendo combinazioni convesse:
- $x_1$ e $x_2$ → lato inferiore
- $x_2$ e $x_3$ → lato destro
- $x_1$ e $x_3$ → lato sinistro

Combinando tutti e tre i punti → interno del triangolo

`01:00:46`

**Cono C:**

`01:01:39`

I bordi A₁ e A₄ vanno all'infinito.

Se ci muoviamo in queste direzioni:
- $y_1$ (direzione lungo A₁)
- $y_2$ (direzione lungo A₄)

da qualsiasi punto possiamo andare all'infinito rimanendo dentro P.

$$
C = \{\lambda_1 y_1 + \lambda_2 y_2 : \lambda_1, \lambda_2 \geq 0\}
$$

`01:02:18`

**Insieme P completo:**

Qualsiasi punto x ∈ P può essere scritto come:

$$
x = z + y
$$

dove:
- $z \in Q$ (punto nel triangolo)
- $y \in C$ (punto nel cono)

`01:03:31`

**Esempio:**

```
    C (cono)
     ╱│
    ╱ │
   ╱  │
  ╱   │y
 ╱    │
●─────●───→ x
  z (nel triangolo Q)
```

Partendo da z nel triangolo e aggiungendo y nel cono, raggiungiamo x.

### 6.3 Conseguenza: ottimo finito sui vertici

`01:04:14`

📌 **Risultato importante:**

Questo semplice risultato (che graficamente è ovvio) ci aiuta a dimostrare che:

> Se la soluzione ottima del problema di ottimizzazione è finita, allora esiste un vertice ottimo.

Non può essere strettamente all'interno o su un bordo.

`01:04:54`

Il professore metterà un **breve video** con la dimostrazione formale di questo risultato.

---

## 7. Teorema dell'Iperpiano Separatore

### 7.1 Enunciato del teorema

`01:08:52`

C'è un'altra proprietà importante, almeno dal punto di vista teorico, che è il **motore del nostro algoritmo**.

📚 **Teorema dell'iperpiano separatore:**

Dato un insieme convesso S e un punto C **non appartenente** a S:

$$
C \notin S
$$

Esiste sempre un **iperpiano** che separa C da S.

`01:09:38`

**Rappresentazione grafica:**

```
     S (insieme convesso)
    ╱  ╲
   │    │
   │  ● │ punto in S
    ╲  ╱
     ╲╱
      │ iperpiano separatore
  ────┼──── (gradiente c)
      │
    ● C (fuori da S)
```

Il gradiente dell'iperpiano è **c**.

$$
\mathbf{c}^T C > \mathbf{c}^T y \quad \forall y \in S
$$

`01:10:30`

**Interpretazione:** Interpretando c come gradiente di una funzione obiettivo, C ha sempre un valore maggiore di qualsiasi punto in S.

### 7.2 Importanza per la convergenza dell'algoritmo

`01:10:30`

Questo teorema è alla **base della convergenza** dell'algoritmo che proporremo per risolvere la programmazione lineare.

La dimostrazione richiede un po' più di matematica e si trova nelle note.

`01:11:05`

❌ **Non vale per insiemi non convessi:**

```
S (non convesso, forma a "C")
  ●─╮
  │ │
  ╰─●
    │ iperpiano?
  ──┼──
    │
  ● C
```

Non c'è modo di separare C da S con un iperpiano: taglieremmo sempre parti di S.

---

## 8. Esercizio: Algoritmo di Edmonds-Karp

### 8.1 Setup iniziale

`01:11:05`

Applichiamo l'**algoritmo di Edmonds-Karp** da zero.

**Regola:** Selezioniamo il cammino con il **numero minimo di archi**.

**Grafo iniziale:**

```
      3       2       2       8
S ───→ 1 ───→ 2 ───→ 3 ───→ T
│      ↓     ↑       ↑      ↑
│      6     5       11     10
│      ↓     ↑       ↑      ↑
└───→ 4 ───→ 5 ───→ 6 ───→ 7
   10      5       5       3
```

(Nota: le capacità sono indicate sui numeri)

`01:11:54`

⚠️ **Correzione:** L'arco va da S a 4 (non S a 2).

**Soluzione iniziale:** Flusso = 0 ovunque.

**Grafo residuale iniziale:**

Poiché tutti i flussi sono 0, abbiamo solo **archi verdi** (forward), corrispondenti agli archi originali.

Nessun arco rosso (backward).

### 8.2 Iterazione 1

`01:12:55`

**Cammino aumentante (minimo numero archi):**

$$
P_1: S \to 1 \to 2 \to 3 \to T
$$

4 archi (il minimo possibile).

`01:13:42`

**Capacità residua θ₁:**

Tutti archi verdi, quindi:

$$
\theta_1 = \min\{3-0, 2-0, 2-0, 8-0\} = \min\{3, 2, 2, 8\} = 2
$$

`01:14:21`

**Flusso aggiornato:**

$$
\begin{aligned}
x(S,1) &= 0 + 2 = 2 \\
x(1,2) &= 0 + 2 = 2 \\
x(2,3) &= 0 + 2 = 2 \\
x(3,T) &= 0 + 2 = 2
\end{aligned}
$$

**Flusso totale:** F = 2

`01:14:51`

**Aggiornamento grafo residuale:**

Modifichiamo solo gli archi nel cammino. Gli altri rimangono invariati.

**Archi rossi (backward):**
- 1 ← S (capacità residua = 2)
- 2 ← 1 (capacità residua = 2)
- 3 ← 2 (capacità residua = 2)
- T ← 3 (capacità residua = 2)

`01:15:21`

**Archi verdi (forward ancora con capacità):**
- S → 1 (capacità residua = 3-2 = 1)
- 3 → T (capacità residua = 8-2 = 6)

### 8.3 Iterazione 2

`01:16:09`

**Nuovo cammino (minimo numero archi):**

Non è più possibile trovare un cammino con 4 archi.

Il cammino minimo ha ora **5 archi**:

$$
P_2: S \to 1 \to 6 \to 7 \to 3 \to T
$$

`01:16:50`

**Capacità residua θ₂:**

Tutti archi verdi:

$$
\theta_2 = \min\{3-2, 6-0, 5-0, 11-0, 8-2\} = \min\{1, 6, 5, 11, 6\} = 1
$$

**Flusso aggiornato:**

$$
\begin{aligned}
x(S,1) &= 2 + 1 = 3 \\
x(1,6) &= 0 + 1 = 1 \\
x(6,7) &= 0 + 1 = 1 \\
x(7,3) &= 0 + 1 = 1 \\
x(3,T) &= 2 + 1 = 3
\end{aligned}
$$

**Flusso totale:** F = 3

`01:17:25`

**Aggiornamento grafo residuale:**

**Archi rossi aggiunti:**
- Tutti gli archi del cammino P₂ hanno arco backward

**Archi verdi:**
- S → 1: **saturato** (non più presente come verde)
- 1 → 6, 6 → 7, 7 → 3, 3 → T: ancora presenti

### 8.4 Iterazione 3

`01:18:30`

**Nuovo cammino:**

Molto contorto! Ha **8 archi**:

$$
P_3: S \to 4 \to 5 \to 2 \to 1 \to 6 \to 7 \to 3 \to T
$$

`01:19:37`

**Capacità residua θ₃:**

Attenzione: l'arco 2 → 1 è **rosso** (backward)!

$$
\theta_3 = \min\{
\underbrace{3-0}_{S \to 4},
\underbrace{10-0}_{4 \to 5},
\underbrace{5-0}_{5 \to 2},
\underbrace{2}_{\text{rosso } 2 \to 1},
\underbrace{6-1}_{1 \to 6},
\underbrace{5-1}_{6 \to 7},
\underbrace{11-1}_{7 \to 3},
\underbrace{8-3}_{3 \to T}
\}
$$

$$
= \min\{3, 10, 5, 2, 5, 4, 10, 5\} = 2
$$

`01:20:40`

**Flusso aggiornato:**

Per gli archi verdi: +2

Per l'arco rosso 2→1 (che corrisponde a 1→2 nell'originale): -2

$$
\begin{aligned}
x(S,4) &= 0 + 2 = 2 \\
x(4,5) &= 0 + 2 = 2 \\
x(5,2) &= 0 + 2 = 2 \\
x(1,2) &= 2 - 2 = 0 \quad \text{(backward)} \\
x(1,6) &= 1 + 2 = 3 \\
x(6,7) &= 1 + 2 = 3 \\
x(7,3) &= 1 + 2 = 3 \\
x(3,T) &= 3 + 2 = 5
\end{aligned}
$$

**Flusso totale:** F = 5

### 8.5 Iterazione finale e taglio

`01:21:20`

**Aggiornamento grafo residuale:**

`01:21:53`

⚠️ **Arco mancante:** C'è un arco verde 5 → 1 che era stato dimenticato nella slide precedente.

Dopo l'aggiornamento:

**Da S possiamo raggiungere:**
- 4 (verde)
- 5 (da 4, verde)
- 2 (da 5, verde)

E poi? **Nessun altro nodo!**

`01:22:32`

✅ **Non esiste più un cammino da S a T nel grafo residuale!**

**Identificazione del taglio:**

$$
\begin{aligned}
N_S &= \{S, 4, 5, 2\} \\
N_T &= \{1, 6, 7, 3, T\}
\end{aligned}
$$

`01:23:19`

Nel grafo residuale, tutti gli archi che attraversano il taglio vanno da $N_T$ a $N_S$.

`01:23:52`

**Verifica nel grafo originale:**

Il taglio è:

```
N_S: {S, 4, 5, 2}  |  N_T: {1, 6, 7, 3, T}
```

**Archi che attraversano il taglio da $N_S$ a $N_T$:**

| Arco | Capacità | Flusso |
|------|----------|--------|
| S → 1 | 3 | 3 |
| 5 → 6 | 5 | 0 |
| 2 → 3 | 2 | 0 |

**Capacità del taglio:**
$$
U(N_S, N_T) = 3 + 5 + 2 = 10
$$

Ma aspetta... il flusso è 5, non 10!

`01:24:26`

Ah, c'è un errore nella soluzione. Controlliamo meglio...

**Archi da $N_S$ a $N_T$ con flusso = capacità:**
- S → 1: flusso 3, capacità 3 ✓

**Archi da $N_T$ a $N_S$ con flusso = 0:**
- Devono essere verificati

Il flusso effettivo è **F = 5**, quindi il taglio ha capacità 5 e tutti gli archi da $N_S$ a $N_T$ sono saturi.

### 8.6 Analisi post-ottimalità

`01:24:26`

❓ **Domanda:** Come cambia il flusso massimo se aumentiamo la capacità dell'arco (1,2)?

**Risposta:** **Non cambia affatto!**

`01:24:59`

**Ragione:**

L'arco (1,2) va da $N_T$ a $N_S$ (da 1 a 2).

Anche se aumentiamo molto la sua capacità, **non abbiamo interesse** ad usarla.

Gli archi che vanno da $N_T$ a $N_S$ non sono nel taglio limitante.

`01:25:32`

❓ **Domanda:** E se aumentiamo la capacità di un arco che va da $N_S$ a $N_T$?

**Risposta:** **Sì, possiamo migliorare!**

In questo caso aumenta la capacità del taglio, quindi possiamo inviare più flusso.

---

## 📊 Tabelle Riassuntive

### Notazione matriciale PL

**Forma standard:**

$$
\begin{aligned}
\max \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A} \mathbf{x} \leq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{aligned}
$$

| Elemento | Dimensione | Descrizione |
|----------|-----------|-------------|
| $\mathbf{c}$ | 1 × n (riga) | Coefficienti funzione obiettivo |
| $\mathbf{x}$ | n × 1 (colonna) | Variabili decisionali |
| $\mathbf{A}$ | m × n | Matrice coefficienti tecnologici |
| $\mathbf{b}$ | m × 1 (colonna) | Termini noti |

### Definizioni geometriche

| Concetto | Definizione | Formula | Note |
|----------|-------------|---------|------|
| **Insieme convesso** | Segmento tra 2 punti ∈ S | $\lambda x + (1-\lambda)y \in S$ | $\lambda \in [0,1]$ |
| **Poliedro convesso** | Definito da finite disequazioni | $\{x: \mathbf{A}x \leq \mathbf{b}\}$ | m vincoli finiti |
| **Cono** | Multiplo di punto ∈ C | $\lambda x \in C$ | $\lambda \geq 0$ |
| **Cono convesso** | Combinazione lineare non neg. | $\lambda x + \mu y \in C$ | $\lambda, \mu \geq 0$ |
| **Cono poliedrico** | Poliedro + cono dall'origine | $\{x: \mathbf{A}x \leq 0\}$ | Un solo vertice (O) |
| **Cono fin. generato** | Generato da r raggi | $\sum_{i=1}^r \lambda_i y_i$ | $\lambda_i \geq 0$ |

### Caratterizzazione vertici

| Definizione | Formulazione | Pro | Contro |
|-------------|-------------|-----|--------|
| **Punto estremo** | $\nexists y,z \in P: x = \lambda y + (1-\lambda)z$ | Intuitiva | Non computazionale |
| **Vertice** | $\exists c: \mathbf{c}^T x > \mathbf{c}^T y \; \forall y \in P$ | Collegata a ottimalità | c non unico |
| **Soluzione di base** | n vincoli attivi lin. indip. | Computazionale | Richiede det ≠ 0 |

**Teorema:** Le tre definizioni sono **equivalenti**.

### Algoritmo Edmonds-Karp

**Regola selezione cammino:** Minimo numero di archi (BFS)

**Complessità:** O(m²n)

| Iterazione | Cammino | Lunghezza | θ | Flusso |
|-----------|---------|-----------|---|--------|
| 1 | S→1→2→3→T | 4 archi | 2 | F=2 |
| 2 | S→1→6→7→3→T | 5 archi | 1 | F=3 |
| 3 | S→4→5→2→1→6→7→3→T | 8 archi | 2 | F=5 |
| - | Nessun cammino | - | - | **STOP** |

**Taglio ottimo:** $N_S = \{S,4,5,2\}$, $N_T = \{1,6,7,3,T\}$

**Certificato ottimalità:** F = U(N_S, N_T) = 5

