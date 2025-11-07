# Lezione 14 - 28 Ottobre: Ford-Fulkerson, Minimum Cost Flow e Introduzione alla Programmazione Lineare

---

## 📑 Indice

1. **[Puzzle del Ponte con Dynamic Programming](#1-puzzle-del-ponte-con-dynamic-programming)** `00:01:31 - 00:11:36`
   - 1.1 [Descrizione del problema](#11-descrizione-del-problema)
   - 1.2 [Soluzione con grafo di stati](#12-soluzione-con-grafo-di-stati)
   - 1.3 [Rappresentazione degli stati](#13-rappresentazione-degli-stati)
   - 1.4 [Soluzione ottima](#14-soluzione-ottima)

2. **[Esercizio di Modellazione: Organizzazione di Feste](#2-esercizio-di-modellazione-organizzazione-di-feste)** `00:11:36 - 00:20:00`
   - 2.1 [Descrizione del problema](#21-descrizione-del-problema)
   - 2.2 [Riduzione al problema di scheduling](#22-riduzione-al-problema-di-scheduling)
   - 2.3 [Costruzione del grafo](#23-costruzione-del-grafo)
   - 2.4 [Assegnazione delle capacità](#24-assegnazione-delle-capacità)

3. **[Problema di Assignment (Matching)](#3-problema-di-assignment-matching)** `00:20:00 - 00:32:29`
   - 3.1 [Maximum Cardinality Matching](#31-maximum-cardinality-matching)
   - 3.2 [Riduzione a Max Flow](#32-riduzione-a-max-flow)
   - 3.3 [Esempio di esecuzione dell'algoritmo](#33-esempio-di-esecuzione-dellalgoritmo)
   - 3.4 [Uso degli archi backward](#34-uso-degli-archi-backward)

4. **[Algoritmo Ford-Fulkerson: Complessità](#4-algoritmo-ford-fulkerson-complessità)** `00:31:21 - 00:45:58`
   - 4.1 [Sommario dell'algoritmo](#41-sommario-dellalgoritmo)
   - 4.2 [Esempio patologico](#42-esempio-patologico)
   - 4.3 [Algoritmo di Edmonds-Karp](#43-algoritmo-di-edmonds-karp)
   - 4.4 [Complessità computazionale](#44-complessità-computazionale)

5. **[Minimum Cost Flow Problem](#5-minimum-cost-flow-problem)** `00:45:58 - 01:11:01`
   - 5.1 [Motivazione e esempi](#51-motivazione-e-esempi)
   - 5.2 [Formulazione del problema](#52-formulazione-del-problema)
   - 5.3 [Esempio 1: Costruzione di una strada](#53-esempio-1-costruzione-di-una-strada)
   - 5.4 [Esempio 2: Raccolta dei rifiuti](#54-esempio-2-raccolta-dei-rifiuti)

6. **[Riduzione di Shortest Path e Max Flow a Min Cost Flow](#6-riduzione-di-shortest-path-e-max-flow-a-min-cost-flow)** `01:11:01 - 01:23:26`
   - 6.1 [Max Flow come Min Cost Flow](#61-max-flow-come-min-cost-flow)
   - 6.2 [Shortest Path come Min Cost Flow](#62-shortest-path-come-min-cost-flow)
   - 6.3 [Shortest Path Tree](#63-shortest-path-tree)

7. **[Introduzione alla Programmazione Lineare](#7-introduzione-alla-programmazione-lineare)** `01:24:11 - 01:29:01`
   - 7.1 [Argomenti del nuovo capitolo](#71-argomenti-del-nuovo-capitolo)
   - 7.2 [Conoscenze preliminari richieste](#72-conoscenze-preliminari-richieste)
   - 7.3 [Notazione matriciale](#73-notazione-matriciale)

---

## 1. Puzzle del Ponte con Dynamic Programming

### 1.1 Descrizione del problema

`00:01:31`

🧩 **Problema del ponte:**

Ci sono **quattro persone** che devono attraversare un ponte. Vincoli:
- Al massimo **due persone alla volta** possono attraversare
- C'è una sola **torcia elettrica**
- Qualcuno deve riportare indietro la torcia
- Trovare la **sequenza ottima** di attraversamenti

`00:02:04`

❓ **Domanda:** Come possiamo modellare questo problema con un grafo in modo tale che il **cammino minimo** tra origine e destinazione corrisponda alla sequenza ottima?

### 1.2 Soluzione con grafo di stati

`00:02:36`

💡 **Tecnica: Dynamic Programming**

L'idea è rappresentare lo **stato del sistema** con nodi.

### 1.3 Rappresentazione degli stati

`00:03:13`

**Notazione dello stato:**

Uno stato è rappresentato come: **IJ / HK ★**

Dove:
- **IJ** sono le persone sul lato **sinistro** del ponte
- **HK** sono le persone sul lato **destro** del ponte
- **★** indica dove si trova la **torcia**

`00:03:51`

**Stati estremi:**
- **Stato iniziale:** `1234 / ∅ ★` → tutti a sinistra con la torcia
- **Stato finale:** `∅ / 1234 ★` → tutti a destra con la torcia

`00:04:31`

**Struttura del grafo:**

**Livello 1 - Partenza:** `1234 / ∅ ★`

**Livello 2 - Prima coppia attraversa:**
- `34 / 12 ★` (coppia 1,2 attraversa)
- `24 / 13 ★` (coppia 1,3 attraversa)
- `23 / 14 ★` (coppia 1,4 attraversa)
- `14 / 23 ★` (coppia 2,3 attraversa)
- `13 / 24 ★` (coppia 2,4 attraversa)
- `12 / 34 ★` (coppia 3,4 attraversa)

`00:05:07`

⚠️ **Ottimizzazione importante:**

Non si introducono stati dove **una sola persona** attraversa da sinistra a destra all'inizio, perché sarebbe **inutile**:
- Quella persona sarebbe sola dall'altra parte
- Dovrebbe tornare indietro per riportare la torcia
- Sarebbe una perdita di tempo

📌 Quindi si limitano gli stati a:
- **Coppie** che attraversano da sinistra a destra
- **Singole persone** che tornano indietro da destra a sinistra

`00:05:50`

**Livello 3 - Una persona torna:**

Da ciascuno stato precedente, una persona torna indietro con la torcia.

Esempio: da `34 / 12 ★` possiamo avere:
- `134 / 2` (persona 1 torna)
- `234 / 1` (persona 2 torna)

In generale: **tre persone a sinistra**, **una a destra senza torcia**.

`00:06:27`

**Livello 4 - Seconda coppia attraversa:**

Si selezionano altre due persone che attraversano.

Rimane: **una persona a sinistra**, **tre persone con torcia a destra**.

`00:07:02`

**Livello 5 - Una persona torna (di nuovo):**

Una persona torna indietro, risultando in: **due a sinistra**, **due a destra**.

Ci sono altri 6 possibili stati (coppie).

**Livello 6 - Arrivo finale:**

Le ultime due persone attraversano il ponte → `∅ / 1234 ★`

`00:07:34`

⚠️ **Errore nella slide:** La stella dovrebbe essere a sinistra in uno stato intermedio.

### 1.4 Soluzione ottima

`00:08:07`

**Assegnazione dei costi agli archi:**

Ad ogni arco si associa il **tempo necessario** per attraversare il ponte.

Se due persone attraversano, il tempo è dato dal **più lento**.

**Tempi individuali (esempio):**
- Persona 1: 1 minuto
- Persona 2: 2 minuti
- Persona 3: 5 minuti
- Persona 4: 10 minuti

**Esempi di costi:**
- Arco (1,2): tempo = 2 (il più lento è 2)
- Arco (1,3): tempo = 5 (il più lento è 3)
- Arco (3,4): tempo = 10 (il più lento è 4)

`00:08:40`

✅ **Percorso minimo ottimale:**

1. **1,2 attraversano** → tempo: 2
2. **1 torna indietro** → tempo: 1
3. **3,4 attraversano** → tempo: 10
4. **2 torna indietro** → tempo: 2
5. **1,2 attraversano di nuovo** → tempo: 2

**Tempo totale:** 2 + 1 + 10 + 2 + 2 = **17 minuti**

`00:09:14`

📌 **Semplificazione possibile:**

Quando si è in uno stato intermedio, si può assumere implicitamente che la **persona più veloce** torni indietro, saltando uno stato intermedio.

`00:10:24`

✅ **Descrizione formale valida:**

Anche senza produrre il grafo completo, la descrizione formale con:
- Stati
- Transizioni
- Costi delle transizioni

è sufficiente.

❌ **Attenzione:** Se il grafo ha **cicli**, è **sbagliato**.

---

## 2. Esercizio di Modellazione: Organizzazione di Feste

### 2.1 Descrizione del problema

`00:11:36`

🎉 **Problema delle feste:**

`00:12:56`

C'è un insieme di **feste** (F) che vogliamo organizzare.

Per avere una festa di successo, dobbiamo invitare una **persona importante**.

📌 **Contesto storico:** Ai tempi in cui l'Italia vinse l'ultimo Mondiale di calcio, le persone importanti erano i giocatori.

`00:13:27`

**Elementi del problema:**
- Insieme di **serate** (S) disponibili per ospitare le feste in una sede
- Per ogni festa i, c'è un sottoinsieme **S(i) ⊆ S** di serate in cui la festa può svolgersi
- Ci sono **N giocatori** (persone importanti) sempre disponibili ogni sera

`00:14:33`

⚠️ **Vincolo cruciale:**

I giocatori **non possono essere presenti contemporaneamente** in due sedi diverse.

Se assegniamo un giocatore a una festa, quel giocatore **non può essere assegnato** a un'altra festa che si svolge nella stessa sera.

### 2.2 Riduzione al problema di scheduling

`00:14:33`

💡 **Mappatura concettuale:**

Vogliamo modellare questo come un problema di **max flow**.

L'idea è ridurlo al **problema di scheduling** visto in precedenza:

| Scheduling | Problema feste |
|-----------|---------------|
| Job | Festa |
| Istante di tempo | Serata |
| Macchina | Giocatore |

### 2.3 Costruzione del grafo

`00:15:17`

**Struttura del grafo:**

`00:15:49`

```
S → [Feste] → [Serate] → T
```

1. **Nodi S e T:** Obbligatori per il max flow

2. **Colonna delle feste:**
   - Un nodo per ogni festa i ∈ F
   - Cardinalità: |F|

`00:16:21`

3. **Colonna delle serate:**
   - Un nodo per ogni serata j ∈ S
   - Cardinalità: |S|
   - Nodi: 1, 2, 3, ..., |S|

`00:16:54`

4. **Archi di compatibilità:**

Tra le feste e le serate, gli archi sono dati dall'insieme S(i).

Se la festa i può svolgersi nelle serate definite da S(i), si creano archi da i a tutte le serate in S(i).

**Esempio:**
```
Festa i → serate in S(i)
```

Se S(i) = {2, 5, 7}, allora ci sono archi:
- i → 2
- i → 5
- i → 7

`00:17:32`

5. **Altri archi:**
   - Archi da **S** a tutte le **feste**
   - Archi da tutte le **serate** a **T**

### 2.4 Assegnazione delle capacità

`00:18:09`

**Capacità sugli archi:**

1. **S → Festa i:** Capacità = **1**
   - Una festa può essere organizzata **al massimo una volta**

`00:18:47`

2. **Festa i → Serata j (archi di compatibilità):** Capacità = **1** (o +∞)
   - Irrilevante perché entra al massimo 1 dalla sorgente
   - Si può mettere 1, ma anche +∞ non cambierebbe la soluzione

`00:19:19`

3. **Serata j → T:** Capacità = **N** (numero di giocatori)
   - Rappresenta il **massimo numero di feste** che possiamo organizzare in parallelo
   - Limitato dal numero di giocatori disponibili
   - Ogni festa richiede almeno 1 giocatore

**Logica:** Non possiamo organizzare più di N feste contemporaneamente perché abbiamo solo N giocatori.

`00:19:19`

✅ **Risultato:** Questo è un modo di modellare un problema apparentemente non correlato ai flussi come un problema di max flow.

---

## 3. Problema di Assignment (Matching)

### 3.1 Maximum Cardinality Matching

`00:20:38`

**Problema di assegnamento:**

Abbiamo:
- Un insieme di **lavoratori** con certe **competenze** (skills)
- Un insieme di **lavori** (jobs): 1, 2, 3, 4, 5, 6
- Ogni lavoro richiede una **skill specifica**

`00:21:10`

**Tabella competenze-lavori:**

| Lavoratore | Skills | Lavori compatibili |
|-----------|--------|-------------------|
| A | Spray, Drill | 1, 2, 3, 4 |
| B | Skill per job 1 e 5 | 1, 5 |
| C | ... | ... |
| ... | ... | ... |

Esempio: A può fare i lavori 1, 2, 3, 4 ma non può fare il lavoro 5 perché non ha la skill richiesta.

`00:22:00`

🎯 **Obiettivo:**

Trovare un'assegnazione **uno-a-uno**:
- Al massimo **un lavoro per lavoratore**
- Al massimo **un lavoratore per lavoro**
- **Massimizzare** il numero di lavori eseguiti

### 3.2 Riduzione a Max Flow

`00:22:34`

**Formulazione come Maximum Cardinality Matching:**

Dato un **grafo bipartito** G = (S ∪ T, A) dove:
- S = insieme lavoratori
- T = insieme lavori
- A ⊆ S × T = archi di compatibilità

`00:23:05`

📚 **Definizione di Matching:**

Un sottoinsieme M ⊆ A è un **matching** quando **nessun coppia di archi in M** è incidente nello stesso nodo.

**Rappresentazione grafica:**

```
S:   ●     ●     ●
      |\   / \   /|
      | \ /   \ / |
      |  X     X  |
      | / \   / \ |
      |/   \ /   \|
T:   ●     ●     ●
```

Un matching valido (esempio):
```
S:   ●     ●     ●
      |         / 
      |        /  
      |       /   
      |      /    
      |     /     
T:   ●     ●     ●
```

Tre archi, ognuno partendo da un nodo diverso in S e arrivando in un nodo diverso in T.

`00:23:35`

Questo è chiamato **perfect matching** o **maximum cardinality matching**.

`00:24:13`

**Formulazione matematica:**

**Variabile decisionale:**
$$
x_{ij} = \begin{cases}
1 & \text{se } (i,j) \in M \\
0 & \text{altrimenti}
\end{cases}
$$

**Funzione obiettivo:**
$$
\max \sum_{(i,j) \in A} x_{ij}
$$

Massimizza il numero di archi nel matching.

`00:24:44`

**Vincoli:**

Per ogni nodo i ∈ S (lavoratori):
$$
\sum_{(i,j) \in \delta^+(i)} x_{ij} \leq 1
$$

Al massimo **un arco esce** da ogni lavoratore.

Per ogni nodo j ∈ T (lavori):
$$
\sum_{(i,j) \in \delta^-(j)} x_{ij} \leq 1
$$

Al massimo **un arco entra** in ogni lavoro.

`00:25:14`

Questa è la formulazione canonica del problema di matching.

Si può risolvere con un solver, ma questo problema **può essere ridotto** a un max flow.

`00:25:46`

**Costruzione del grafo per max flow:**

**Passo 1:** Partire dal grafo bipartito originale.

```
S: A   B   C          (lavoratori)
   |\ /|\ /|
   | X | X |
   |/ \|/ \|
T: 1   2   3   4   5   6   (lavori)
```

Gli archi rappresentano la compatibilità.

`00:26:24`

**Passo 2:** Aggiungere S e T, dare direzione agli archi.

```
      S (source)
     /|\
    / | \
   A  B  C
   |\ | /|
   | \|/ |
   1  2  3  4  5  6
    \ | | | | /
     \| | | |/
      T (sink)
```

**Passo 3:** Assegnare capacità = **1** a:
- Archi da S a ogni nodo in S (lavoratori)
- Archi da ogni nodo in T (lavori) a T

`00:26:56`

💡 **Trucco importante:**

Il vincolo aggregato $\sum x_{ij} \leq 1$ (che non è un vincolo di flusso standard) viene **tradotto** in un vincolo di capacità sull'**unico arco entrante/uscente**.

Poiché c'è solo un arco che arriva/parte da ogni nodo con capacità 1, questo **induce** il vincolo aggregato.

`00:27:27`

**Logica:**
- Capacità 1 su S→lavoratore: al massimo 1 unità arriva al lavoratore
- Per conservazione del flusso: 1 unità deve uscire lungo uno degli archi uscenti
- Capacità 1 su lavoro→T: al massimo 1 unità può lasciare il lavoro
- Quindi al massimo 1 unità può entrare nel lavoro

### 3.3 Esempio di esecuzione dell'algoritmo

`00:28:29`

**Iterazione 1:**

Soluzione iniziale: flusso = 0 ovunque.

**Grafo residuale:** Tutti gli archi sono **verdi** (forward), nessun arco rosso (backward).

`00:29:01`

**Cammino aumentante:** S → 1 → B → T

Aumento il flusso di 1 lungo il cammino.

**Aggiornamento grafo residuale:**
- Arco B→1: diventa **rosso** (backward) perché il flusso = 1
- Altri archi da S a 1 e da B a T: scompaiono o cambiano

`00:29:40`

**Iterazione 2:**

**Cammino aumentante:** S → 2 → A → T

Aumento il flusso di 1.

**Continuazione...**

**Iterazione k:**

**Cammino aumentante:** S → 6 → E → T

Aumento il flusso di 1.

### 3.4 Uso degli archi backward

`00:29:40`

**Situazione:** 
- Lavoratore 4 è **inattivo**
- Lavoro F **non è eseguito** da nessuno

❓ **Domanda:** Possiamo cambiare la soluzione in modo che 4 lavori e F venga eseguito?

`00:30:13`

✅ **Risposta:** Sì, usando il **grafo residuale**.

**Nel grafo residuale, esiste un cammino da S a T?**

Sì, anche se non immediato:

`00:30:49`

**Cammino aumentante con archi backward:**

S → 4 → A (arco verde) → 2 (arco **rosso** backward) → F → T

**Interpretazione:**
1. Assegno 4 al lavoro A (arco verde)
2. Cancello l'assegnamento di 2 ad A (arco rosso = sottraggo flusso)
3. Assegno 2 al lavoro F

`00:31:21`

**Effetto complessivo:**
- ❌ Cancello: 2 → A
- ✅ Aggiungo: 4 → A
- ✅ Aggiungo: 2 → F

Questo esempio mostra chiaramente il **vantaggio degli archi backward**: permettono di **riorganizzare** assegnamenti esistenti.

---

## 4. Algoritmo Ford-Fulkerson: Complessità

### 4.1 Sommario dell'algoritmo

`00:31:53`

📋 **Algoritmo Ford-Fulkerson:**

```
Input: Grafo G=(N,A), capacità u, sorgente S, sink T
Output: Flusso massimo x

1. Inizializzazione:
   x = flusso feasible (tipicamente x = 0)

2. REPEAT:
   a. Costruisci grafo residuale G_R(x)
   
   b. Esegui graph search da S a T in G_R
      (es. BFS, DFS)
   
   c. IF pred(T) ≠ NULL:  // esiste cammino S→T
      
      i. Calcola capacità residua:
         θ = min{r(i,j) : (i,j) ∈ P}
         
      ii. Augmenta il flusso:
          - Per (i,j) ∈ P ∩ A⁺: x(i,j) += θ
          - Per (i,j) ∈ P ∩ A⁻: x(j,i) -= θ
   
   UNTIL pred(T) = NULL  // nessun cammino aumentante

3. Return x
```

`00:32:26`

**Calcolo della capacità residua θ:**

$$
\theta = \min_{(i,j) \in P} r(i,j)
$$

Dove:
$$
r(i,j) = \begin{cases}
u(i,j) - x(i,j) & \text{se } (i,j) \in A^+ \text{ (verde, aumenta flusso)} \\
x(j,i) & \text{se } (i,j) \in A^- \text{ (rosso, diminuisci flusso)}
\end{cases}
$$

`00:33:01`

**Ragionamento:**
- **Archi verdi (A⁺):** Aumentiamo il flusso → vincolo: non superare capacità superiore u(i,j)
- **Archi rossi (A⁻):** Diminuiamo il flusso → vincolo: non scendere sotto 0

### 4.2 Esempio patologico

`00:33:32`

⚠️ **Problema:** A seconda del risultato della graph search, il comportamento dell'algoritmo può essere **imprevedibile**.

`00:34:20`

**Esempio patologico:**

```
        1000
    S -----→ 1
    |        |
1000|        | 1  (capacità molto piccola)
    |        |
    2 -----→ T
        1000
```

- Archi S→1, 1→T, S→2, 2→T: capacità = 1000
- Arco 1→2 (interno): capacità = 1

`00:34:54`

**Soluzione ottima (ovvia):**

Flusso massimo = **2000**

- Cammino superiore S→1→T: 1000 unità
- Cammino inferiore S→2→T: 1000 unità
- Arco 1→2 è **inutile**

`00:35:26`

Molto banale, non serve nemmeno il grafo residuale.

`00:35:59`

**Ma se applichiamo l'algoritmo in modo miope...**

**Iterazione 1:**

Soluzione iniziale: x = 0

Grafo residuale: tutti archi **verdi** (stesso verso degli originali).

`00:36:49`

**Cammino (stupido) scelto:** S → 1 → 2 → T

**Capacità residua:**

$$
\theta = \min\{1000-0, 1-0, 1000-0\} = \min\{1000, 1, 1000\} = 1
$$

`00:37:19`

**Flusso aggiornato:**
- x(S,1) = 1
- x(1,2) = 1
- x(2,T) = 1

**Flusso totale dopo iterazione 1:** F = 1

`00:37:51`

**Iterazione 2:**

**Aggiornamento grafo residuale:**

Solo gli archi lungo il cammino cambiano:
- S→1: ancora verde (flusso < capacità)
- 1→2: **non più verde** (saturato), ma appare arco **rosso** 2→1
- 2→T: ancora verde + arco rosso T→2

`00:38:54`

**Nuovo cammino (ancora stupido):** S → 2 → 1 → T

`00:39:24`

**Capacità residua:**

$$
\theta = \min\{1000-0, 1, 1000-0\} = 1
$$

(L'arco 2→1 è rosso, capacità residua = flusso corrente su 1→2 = 1)

`00:39:55`

**Flusso aggiornato:**
- x(S,2) = 1
- x(1,2) = 0 (sottratto 1)
- x(1,T) = 1

**Flusso totale dopo iterazione 2:** F = 2

`00:40:32`

**Continuando così...**

Se il cammino selezionato contiene sempre l'arco 1↔2, ogni iterazione aumenta il flusso di **solo 1 unità**.

Sono necessarie **2000 iterazioni** per raggiungere la soluzione ottima!

`00:41:14`

**Se invece usassimo i cammini diretti:**
- Iterazione 1: S→1→T con θ=1000
- Iterazione 2: S→2→T con θ=1000
- **Solo 2 iterazioni!**

📌 **Conclusione:** La **scelta del cammino** influenza pesantemente la complessità dell'algoritmo.

### 4.3 Algoritmo di Edmonds-Karp

`00:41:14`

📚 **Algoritmo di Edmonds-Karp:**

La regola più semplice per garantire **complessità polinomiale**:

> Selezionare ogni volta il cammino con il **numero minimo di archi**

Si usa **BFS (Breadth-First Search)** per trovare il cammino più corto in termini di numero di archi.

`00:41:45`

**Giustificazione (intuitiva):**

Ogni volta che troviamo un cammino aumentante:
- A causa del calcolo di θ, **almeno un arco scompare** dal grafo residuale
- Quell'arco può **riapparire** successivamente
- **SE** la selezione è consistente con la regola (minimo numero archi), quell'arco può riapparire solo se il cammino si è **allungato di almeno 2 archi**

`00:42:19`

Poiché:
- Il grafo ha n nodi
- Il cammino più corto ha al massimo n-1 archi
- Ogni arco può essere cancellato al massimo **n/2 volte** (approssimativamente n volte)

### 4.4 Complessità computazionale

`00:43:04`

**Complessità di Edmonds-Karp:**

$$
O(m^2 \cdot n)
$$

Dove:
- **m** = numero di archi (costo graph search)
- **m** = numero massimo di volte che un arco viene cancellato
- **n** = lunghezza massima del cammino

`00:43:35`

Questa non è la migliore complessità possibile.

`00:44:05`

**Migliore complessità teorica:** $O(n^3)$

Per raggiungere $O(n^3)$ serve una **filosofia completamente diversa**: non si possono usare cammini aumentanti, ma altri metodi (es. push-relabel).

`00:44:49`

📌 **Per grafi bipartiti:**

La complessità di Edmonds-Karp è anche **O(n³)**

Questo perché:
- Capacità unitarie
- Ogni iterazione aumenta il flusso di esattamente 1
- Struttura speciale del grafo bipartito

---

## 5. Minimum Cost Flow Problem

### 5.1 Motivazione e esempi

`00:45:58`

**Confronto tra i problemi visti:**

| Problema | Focus | Caratteristiche |
|----------|-------|----------------|
| **Shortest Path** | Costo degli archi | Non ci interessa la quantità di flusso |
| **Max Flow** | Flusso e capacità | Non ci interessa il costo |
| **Min Cost Flow** | **Entrambi** | Flusso + Costo combinati |

`00:47:02`

💡 **Idea:** Combinare i due aspetti:
- La **quantità di flusso** che inviamo sugli archi
- Il **costo** di inviare flusso sugli archi

`00:47:36`

### 5.2 Esempio 1: Raccolta dei rifiuti

`00:47:36`

**Problema:**

Dobbiamo raccogliere rifiuti da un paio di siti:
- **Sito 1:** 10 camion di rifiuti
- **Sito 2:** 55 camion di rifiuti

`00:48:09`

**Destinazioni possibili:**
1. **Discarica (landfill)**
2. **Inceneritore (burner)**
3. **Centro di riciclaggio (recycling center)**

⚠️ **Vincolo:** Per inviare rifiuti al centro di riciclaggio, i rifiuti devono essere **processati**.

`00:48:41`

💭 **Analogia cinematografica:** Nel film Toy Story, i giocattoli finiscono in una linea dove vengono separati (metallo/non-metallo) e poi bruciati → questo è il centro di processamento.

`00:49:46`

**Informazioni sulle connessioni:**

Ogni connessione tra due siti ha:
- **Costo:** costo di inviare un camion da un sito all'altro
- **Capacità:** limite sul numero di camion (la popolazione non vuole vedere troppi camion)

`00:50:19`

**Natura del problema:**

- Non c'è una sorgente S e un sink T come nel max flow
- Abbiamo **quantità fisse** di flusso: 10 e 55 camion
- Dobbiamo inviare questo flusso minimizzando il costo
- Rispettando le capacità

Senza capacità, invieremmo tutto lungo i cammini più economici. Con le capacità, diventa più complesso perché dobbiamo bilanciare diverse rotte.

### 5.3 Esempio 2: Costruzione di una strada

`00:50:50`

**Problema:**

Dobbiamo costruire una **strada o ferrovia**.

`00:51:21`

**Orografia del terreno:**
- **Colline:** eccesso di materiale (da rimuovere)
- **Depressioni:** bisogno di materiale (da riempire)

**Quantità:**
- Numeri **verdi:** migliaia di camion da **rimuovere** dalle colline
- Numeri **rossi:** migliaia di camion da **portare** alle depressioni

`00:52:03`

📊 **Esempio reale: Pedemontana:**

Movimento di terra stimato: **35 milioni di metri cubi**

`00:52:34`

Per dare un'idea della dimensione:
- Usare camion industriali grandi
- Metterli uno dopo l'altro
- Formerebbero una linea da **Milano a Mosca** senza interruzioni!

`00:53:04`

⚠️ **Semplificazione:**

Nel problema semplificato, tutta la terra rimossa è utilizzabile.

Nella realtà:
- Qualche terra è **inquinata**
- Qualche terra è **utilizzabile**
- Qualche terra è troppo **fangosa**
- Bisogna distribuire il materiale in diversi luoghi a seconda della qualità

`00:53:38`

**Natura del problema:**

Anche questo è un **minimum cost flow** perché:
- Flusso da spostare lungo la rete
- Rete = strade
- Possibili capacità
- Costo per spostare il materiale

Obiettivo: spostare il materiale a **costo minimo**.

### 5.4 Formulazione del problema

`00:54:14`

**Dati del problema:**

Dato un grafo G = (N, A) con:

`00:54:45`

1. **Bilanci dei nodi (node balances):** $b_i$ per ogni nodo i ∈ N

**Convenzione:**
$$
b_i = \begin{cases}
< 0 & \text{se il nodo offre materiale (eccesso)} \\
> 0 & \text{se il nodo richiede materiale (domanda)} \\
= 0 & \text{se è un nodo di transito}
\end{cases}
$$

`00:55:21`

**Assunzione importante:**
$$
\sum_{i \in N} b_i = 0
$$

Il sistema è **chiuso**: non serve materiale dall'esterno, non c'è eccesso da inviare fuori.

📌 Nota: Sappiamo come ridurre al caso con $\sum b_i = 0$.

`00:55:56`

2. **Costi unitari:** $c_{ij}$ per ogni arco (i,j) ∈ A
   - Per ogni unità di flusso, si paga $c_{ij}$

3. **Capacità:** $u_{ij}$ per ogni arco (i,j) ∈ A
   - Quantità massima di flusso sull'arco

`00:56:33`

🎯 **Problema:**

Inviare il flusso richiesto (specificato dai bilanci) a **costo minimo**, rispettando i vincoli di capacità.

### 5.5 Formulazione matematica

`00:56:33`

**Variabili decisionali:**

$$
x_{ij} = \text{flusso sull'arco } (i,j)
$$

`00:57:22`

**Funzione obiettivo:**

$$
\min \sum_{(i,j) \in A} c_{ij} \cdot x_{ij}
$$

Minimizza il costo totale (costo unitario × flusso).

`00:57:56`

**Vincoli di capacità:**

$$
0 \leq x_{ij} \leq u_{ij} \quad \forall (i,j) \in A
$$

`00:58:27`

**Vincoli di bilancio del flusso:**

Per ogni nodo i ∈ N:

$$
\sum_{(j,i) \in A} x_{ji} - \sum_{(i,j) \in A} x_{ij} = b_i
$$

Flusso entrante - Flusso uscente = Bilancio

`00:59:06`

**Interpretazione del bilancio:**

- Se $b_i > 0$ (domanda): flusso entrante > flusso uscente di esattamente $b_i$ unità
- Se $b_i < 0$ (offerta): flusso uscente > flusso entrante di esattamente $|b_i|$ unità
- Se $b_i = 0$ (transito): flusso entrante = flusso uscente

---

## 6. Riduzione di Shortest Path e Max Flow a Min Cost Flow

### 6.1 Esempio 1: Costruzione di una strada (riduzione)

`00:59:44`

**Costruzione del grafo:**

`01:00:15`

**Nodi:**

Per ogni sezione della strada, introduciamo un nodo:
- Nodo 1, 2, 3, 4, 5, 6

`01:00:48`

**Archi:**

Movimento possibile tra sezioni adiacenti (assumendo di usare solo la strada in costruzione):
- 1↔2, 2↔3, 3↔4, 4↔5, 5↔6

**Bilanci dei nodi:**

`01:01:20`

- **Nodo 1** (collina, 3 unità da rimuovere): $b_1 = -3$
- **Nodo 3** (collina, 4 unità da rimuovere): $b_3 = -4$
- **Nodo 5** (collina, 2 unità da rimuovere): $b_5 = -2$

`01:01:50`

Negativo perché sono nodi che **offrono** materiale → il flusso uscente deve dominare.

`01:02:21`

- **Nodo 2** (depressione, 3 unità richieste): $b_2 = +3$
- **Nodo 4** (depressione, 3 unità richieste): $b_4 = +3$
- **Nodo 6** (depressione, 3 unità richieste): $b_6 = +3$

Positivo perché richiedono materiale.

**Costi e capacità:**

Ad ogni arco si associano:
- $c_{12}$, $u_{12}$ per l'arco (1,2)
- $c_{21}$, $u_{21}$ per l'arco (2,1)
- E così via...

`01:03:25`

✅ **Soluzione:** Inserire tutti questi numeri (bilanci, costi, capacità) nel framework min cost flow e risolvere.

### 6.2 Esempio 2: Raccolta rifiuti (riduzione)

`01:03:25`

**Nodi:**

- **Nodo 1:** Sito piccolo (10 unità di rifiuti)
- **Nodo 2:** Sito grande (55 unità di rifiuti)
- **Nodo 3:** Centro di processamento
- **Nodo 4:** Discarica
- **Nodo 5:** Inceneritore
- **Nodo 6:** Centro di riciclaggio

`01:04:31`

**Archi:**

Come mostrato nell'immagine originale: 1→4, 2→3, 3→6, ecc.

**Bilanci:**

`01:05:02`

- $b_1 = -10$ (offre 10 unità)
- $b_2 = -55$ (offre 55 unità)
- $b_3 = 0$ (centro di processamento: tutto ciò che entra esce)

`01:05:33`

**Bilanci per nodi 4, 5, 6?**

Dovrebbero essere positivi, ma... **non conosciamo i valori esatti!**

Non sappiamo quanto bruciare, quanto depositare in discarica, quanto riciclare.

Sono **incognite** → **variabili decisionali**.

`01:06:06`

**Soluzione: Introdurre un nodo dummy T**

`01:06:42`

```
4 (discarica) ----→ T
5 (inceneritore) --→ T
6 (riciclaggio) ---→ T
```

**Bilanci aggiornati:**
- $b_4 = 0$ (non significa nessun rifiuto, ma che tutto va su 4→T)
- $b_5 = 0$
- $b_6 = 0$
- $b_T = +65$ (totale rifiuti: 10+55)

`01:07:17`

💡 **Ragionamento:**

Mettendo $b_4 = 0$, non stiamo dicendo di non mandare nulla alla discarica.

Stiamo dicendo che per contare la quantità che arriva in 4, guardiamo l'**unico arco uscente**: 4→T.

Lo stesso per 5 e 6.

`01:07:48`

**Caratteristiche aggiuntive:**

Con gli archi 4→T, 5→T, 6→T possiamo modellare ulteriori vincoli:

`01:08:24`

**Capacità:**
- $u_{4T}$ = capacità della discarica
- $u_{5T}$ = capacità dell'inceneritore
- $u_{6T}$ = capacità del centro di riciclaggio

`01:08:56`

**Costi (guadagni):**

Possiamo avere **benefici**:
- Energia prodotta dall'inceneritore: $-g_5$ (guadagno G5)
- Riciclaggio di plastica/vetro/metallo: $-g_6$ (guadagno G6)

`01:09:37`

Usiamo il **segno negativo** perché stiamo minimizzando i costi.

Per **massimizzare** i guadagni, mettiamo $-g$ nel costo e minimiziamo.

$$
\min (\text{costi} - \text{guadagni})
$$

### 6.3 Max Flow come Min Cost Flow

`01:11:01`

**Max Flow originale:**

```
      u₁₂     u₂₃
S ----→ 1 ----→ 2 ----→ T
  \                    /
   \__________________/
         u_ST (capacità archi)
```

- Grafo con capacità
- Nessun bilancio
- Obiettivo: massimizzare flusso da S a T

`01:11:37`

**Riduzione a Min Cost Flow:**

**Passo 1:** La quantità di flusso da S a T è **sconosciuta**.

**Passo 2:** Quantità sconosciuta → variabile decisionale → arco!

`01:13:33`

**Grafo modificato:**

```
      0       0       0
S ----→ 1 ----→ 2 ----→ T
  \                    /
   \__________________/
       -1, +∞
```

Aggiungiamo un **arco dummy T→S**.

`01:14:04`

**Caratteristiche:**

- **Tutti i bilanci = 0**
- **Costi originali = 0**
- **Costo arco T→S = -1**
- **Capacità arco T→S = +∞**

`01:14:45`

💡 **Logica:**

L'unico arco con costo negativo è T→S.

Vogliamo inviare il **massimo flusso possibile** su T→S per minimizzare il costo totale.

Ma poiché tutti i bilanci sono 0, tutto il flusso che va da T a S deve **tornare** da S a T attraverso la rete, rispettando le capacità.

Questa è chiamata **formulazione a circolazione**.

`01:15:19`

📌 **Nota:** È principalmente un esercizio matematico di riduzione, per mostrare che max flow è un caso speciale di min cost flow.

### 6.4 Shortest Path come Min Cost Flow

`01:15:51`

**Shortest Path originale:**

```
        c₁₂     c₂₃
S ----→ 1 ----→ 2 ----→ T
```

- Costi sugli archi
- Nessuna capacità
- Obiettivo: trovare cammino minimo da S a T

`01:16:54`

**Riduzione a Min Cost Flow:**

**Idea:** Inviare **1 unità** di flusso da S a T.

**Bilanci:**
- $b_S = -1$ (1 unità esce da S)
- $b_T = +1$ (1 unità arriva in T)
- $b_i = 0$ per tutti gli altri nodi

`01:17:33`

**Formulazione:**

$$
\min \sum_{(i,j) \in A} c_{ij} \cdot x_{ij}
$$

Soggetto a:

$$
\sum_{(j,i) \in \delta^-(i)} x_{ji} - \sum_{(i,j) \in \delta^+(i)} x_{ij} = b_i
$$

Dove:

$$
b_i = \begin{cases}
-1 & \text{se } i = S \\
+1 & \text{se } i = T \\
0 & \text{altrimenti}
\end{cases}
$$

`01:18:56`

**Interpretazione:**

- Da S: flusso uscente - flusso entrante = -1 → esce 1 unità
- A T: flusso entrante - flusso uscente = +1 → arriva 1 unità
- Nodi interni: flusso entrante = flusso uscente

Questa è **esattamente** la formulazione vista per lo shortest path!

`01:19:32`

📌 **Differenza con max flow:** Qui **non ci sono capacità**.

### 6.5 Shortest Path Tree

`01:19:32`

**Estensione: Shortest Path Tree Problem**

**Obiettivo:** Trovare la collezione di **tutti i cammini minimi** da S a tutti gli altri nodi.

`01:20:02`

**Riduzione a Min Cost Flow:**

Se vogliamo cammini minimi da S a T e da S a T':

```
        c_ij
S ----→ ... ----→ T
  \             /
   \-----------→ T'
```

- Da S devono uscire **2 unità** (una per T, una per T')
- In T deve arrivare **1 unità**
- In T' deve arrivare **1 unità**

`01:20:35`

**Generalizzazione a tutti i nodi:**

Se ci sono n nodi (incluso S):

**Bilanci:**
$$
b_i = \begin{cases}
-(n-1) & \text{se } i = S \\
+1 & \text{per tutti gli altri nodi}
\end{cases}
$$

`01:21:24`

**Formulazione:**

$$
\min \sum_{(i,j) \in A} c_{ij} \cdot x_{ij}
$$

Soggetto a:
$$
\sum_{(j,i) \in A} x_{ji} - \sum_{(i,j) \in A} x_{ij} = b_i
$$

`01:22:19`

**Interpretazione del flusso sugli archi:**

Le variabili $x_{ij}$ non sono più binarie (0/1).

Rappresentano la **quantità di flusso** (numero di cammini che usano quell'arco).

**Esempio:**

```
        2       1
S ----→ 1 ----→ T'
  \  3      1
   ----→ 2 ----→ T
```

- Arco S→1: flusso = 2 (due cammini passano di lì)
- Arco S→2: flusso = 3 (tre cammini passano di lì)
- Arco 1→T': flusso = 1
- Ecc.

`01:22:51`

Se alcuni archi sono **condivisi** tra diversi shortest path, la quantità di condivisione è rappresentata dal flusso su quegli archi.

`01:23:26`

✅ **Risultato:** Risolvendo il min cost flow con questi bilanci e senza capacità, otteniamo lo shortest path tree.

---

## 7. Introduzione alla Programmazione Lineare

### 7.1 Argomenti del nuovo capitolo

`01:24:11`

📚 **Nuovo argomento: Programmazione Lineare (Linear Programming)**

**Perché?**

Vogliamo **generalizzare** l'approccio ai problemi di ottimizzazione a una **classe più ampia** di problemi.

`01:25:26`

📌 **Osservazione importante:**

Tutti i problemi su grafi visti finora possono essere **facilmente inclusi** nella programmazione lineare.

Infatti, tutte le formulazioni date erano **formulazioni lineari**.

**Cosa faremo:**

1. **Caratterizzare l'ottimalità** della soluzione
   - Dare la **visione duale** del problema (come i tagli con il max flow)

`01:26:04`

2. **Interpretazione geometrica**
   - Aspetto cruciale per **capire cosa stiamo facendo**
   - Senza interpretazione geometrica: difficile comprensione
   - Richiede di rinfrescare conoscenze su algebra lineare e geometria

3. **Algoritmo Simplex** (cenni)
   - Solo il "sapore" dell'algoritmo di soluzione

`01:26:40`

4. **Analisi di post-ottimalità**
   - Questo sarà più importante

### 7.2 Conoscenze preliminari richieste

`01:27:19`

📋 **Conoscenze necessarie:**

1. **Notazione matriciale**
2. **Soluzione di piccoli sistemi di disequazioni**
3. **Interpretazione geometrica di funzioni lineari**
4. **Interpretazione geometrica di vincoli lineari (disequazioni)**
5. **Concetto di gradiente**
6. **Visione duale dei problemi**

### 7.3 Formulazione generale e notazione matriciale

`01:27:53`

**Problema di ottimizzazione generale:**

$$
\max f(x)
$$

Soggetto a:
$$
g_i(x) \leq b_i \quad i = 1, \ldots, m
$$

Dove:
- **f** = funzione obiettivo
- **g_i** = funzioni nei vincoli
- **x** = variabili **reali** (non discrete!)

`01:28:28`

⚠️ **Importante:** Variabili 0-1 o discrete sono **vietate** (per ora).

**Programmazione Lineare:**

Quando **f** e **g_i** sono tutte **funzioni lineari**, abbiamo un problema di programmazione lineare.

**Notazione matriciale:**

$$
\max \mathbf{c}^T \mathbf{x}
$$

Soggetto a:
$$
\mathbf{A} \mathbf{x} \leq \mathbf{b}
$$

Dove:
- $\mathbf{c}$ = vettore riga (o colonna trasposto)
- $\mathbf{x}$ = vettore colonna
- $\mathbf{A}$ = matrice m × n (m righe come $\mathbf{b}$, n colonne come $\mathbf{x}$)
- $\mathbf{b}$ = vettore colonna di termini noti

`01:29:01`

✅ **Prossima lezione:** Primo metodo (molto intuitivo) per risolvere un problema lineare.

---

## 📊 Tabelle Riassuntive

### Confronto tra problemi su grafi

| Problema | Costo | Flusso | Capacità | Bilanci |
|----------|-------|--------|----------|---------|
| **Shortest Path** | ✅ Sì | ❌ No (implicitamente 1) | ❌ No | S: -1, T: +1 |
| **Max Flow** | ❌ No | ✅ Sì | ✅ Sì | Tutti 0 |
| **Min Cost Flow** | ✅ Sì | ✅ Sì | ✅ Sì | $\sum b_i = 0$ |

### Riduzioni a Min Cost Flow

| Problema originale | Bilanci | Costi | Capacità | Archi aggiuntivi |
|-------------------|---------|-------|----------|------------------|
| **Max Flow** | Tutti = 0 | c=0 tranne T→S=-1 | Originali + T→S=+∞ | T→S (circolazione) |
| **Shortest Path** | S: -1, T: +1, altri: 0 | Originali | Nessuna | Nessuno |
| **Shortest Path Tree** | S: -(n-1), altri: +1 | Originali | Nessuna | Nessuno |

### Complessità degli algoritmi

| Algoritmo | Tipo grafo | Complessità | Note |
|-----------|-----------|-------------|------|
| **Ford-Fulkerson (naïve)** | Generale | Pseudo-polinomiale | Può richiedere O(U·m) iterazioni |
| **Edmonds-Karp** | Generale | O(m²n) | Usa BFS (cammino con min archi) |
| **Edmonds-Karp** | Bipartito | O(n³) | Capacità unitarie |
| **Migliore teorico** | Generale | O(n³) | Metodi diversi (push-relabel) |

### Formulazione Min Cost Flow

**Variabili:**
$$
x_{ij} = \text{flusso sull'arco } (i,j)
$$

**Obiettivo:**
$$
\min \sum_{(i,j) \in A} c_{ij} \cdot x_{ij}
$$

**Vincoli:**

| Tipo | Formula | Significato |
|------|---------|-------------|
| Capacità | $0 \leq x_{ij} \leq u_{ij}$ | Flusso limitato dalla capacità |
| Bilancio | $\sum x_{ji} - \sum x_{ij} = b_i$ | Conservazione flusso con bilancio |

**Interpretazione bilanci:**

$$
b_i = \begin{cases}
< 0 & \text{Nodo offre flusso (sorgente)} \\
> 0 & \text{Nodo richiede flusso (sink)} \\
= 0 & \text{Nodo di transito}
\end{cases}
$$

