# Lezione 13 - 27 Ottobre: Laboratorio su Problemi di Ottimizzazione su Grafi (Shortest Path, Minimum Spanning Tree, TSP)

---

## 📑 Indice

1. **[Introduzione al Laboratorio](#1-introduzione-al-laboratorio)** `00:04:27 - 00:09:10`
   - 1.1 [Problemi tecnici e registrazione peer review](#11-problemi-tecnici-e-registrazione-peer-review)
   - 1.2 [Panoramica dei tre problemi](#12-panoramica-dei-tre-problemi)

2. **[Problema 1: Shortest Path con Formulazione a Flusso](#2-problema-1-shortest-path-con-formulazione-a-flusso)** `00:09:10 - 00:33:30`
   - 2.1 [Descrizione del problema](#21-descrizione-del-problema)
   - 2.2 [Generazione dei punti casuali](#22-generazione-dei-punti-casuali)
   - 2.3 [Creazione degli archi e filtraggio distanze](#23-creazione-degli-archi-e-filtraggio-distanze)
   - 2.4 [Formulazione a flusso](#24-formulazione-a-flusso)
   - 2.5 [Implementazione del modello](#25-implementazione-del-modello)
   - 2.6 [Estrazione della soluzione](#26-estrazione-della-soluzione)

3. **[Problema 2: Minimum Spanning Tree (MST)](#3-problema-2-minimum-spanning-tree-mst)** `00:33:30 - 00:58:17`
   - 3.1 [Definizione di Spanning Tree](#31-definizione-di-spanning-tree)
   - 3.2 [Formulazione basata su Cut Sets](#32-formulazione-basata-su-cut-sets)
   - 3.3 [Generazione di tutte le combinazioni di sottoinsiemi](#33-generazione-di-tutte-le-combinazioni-di-sottoinsiemi)
   - 3.4 [Implementazione del modello MST](#34-implementazione-del-modello-mst)
   - 3.5 [Visualizzazione della soluzione](#35-visualizzazione-della-soluzione)

4. **[Problema 3: Travelling Salesman Problem (TSP)](#4-problema-3-travelling-salesman-problem-tsp)** `00:58:58 - 01:26:00`
   - 4.1 [Introduzione al TSP](#41-introduzione-al-tsp)
   - 4.2 [Vincoli di conservazione del flusso](#42-vincoli-di-conservazione-del-flusso)
   - 4.3 [Problema dei subtour](#43-problema-dei-subtour)
   - 4.4 [Subtour Elimination Constraints](#44-subtour-elimination-constraints)
   - 4.5 [Approccio iterativo](#45-approccio-iterativo)
   - 4.6 [Implementazione del modello TSP](#46-implementazione-del-modello-tsp)
   - 4.7 [Aggiunta iterativa dei vincoli](#47-aggiunta-iterativa-dei-vincoli)

5. **[Conclusioni e Domande](#5-conclusioni-e-domande)** `01:21:21 - 01:26:00`

---

## 1. Introduzione al Laboratorio

### 1.1 Problemi tecnici e registrazione peer review

`00:04:27`

Il laboratorio inizia con alcuni problemi tecnici relativi al microfono. Una volta risolti, l'assistente spiega il processo di registrazione per l'attività di peer review:

📢 **Processo di registrazione peer review:**
- Gli studenti riceveranno un annuncio su Webex con un link
- Cliccare sul link porta a un modulo di registrazione
- Inserire nome e registrarsi
- Dopo la registrazione, si riceveranno le soluzioni di altri tre studenti da valutare
- Rispondere a domande sulle soluzioni ricevute
- Ricevere le recensioni alla fine dell'attività
- Tutto è anonimo
- Le date sono indicate nell'annuncio

`00:06:25`

### 1.2 Panoramica dei tre problemi

`00:06:55`

Il laboratorio prevede tre notebook su problemi di ottimizzazione su grafi:
1. **Shortest Path** (problema del cammino minimo)
2. **Minimum Spanning Tree** (albero ricoprente minimo)
3. **Travelling Salesman Problem** (problema del commesso viaggiatore)

Tutti e tre i problemi saranno risolti utilizzando **formulazioni a flusso** invece degli algoritmi euristici classici.

---

## 2. Problema 1: Shortest Path con Formulazione a Flusso

### 2.1 Descrizione del problema

`00:08:00`

🎯 **Obiettivo:** Trovare il cammino minimo che connette due nodi su un grafo generato casualmente con k nodi, dove due nodi sono connessi se la distanza è inferiore a un massimo D_max.

**Passi del problema:**
1. Generare casualmente k punti in un intervallo tra 0 e 100 per X e Y
2. Creare gli archi connettendo ogni coppia di nodi e filtrare quelli con distanza > D_max
3. Creare un modello di ottimizzazione per trovare il cammino minimo con una formulazione basata sul flusso
4. Assumere che il percorso vada dal nodo 0 al nodo 12
5. Determinare il cammino minimo

### 2.2 Generazione dei punti casuali

`00:09:43`

**Librerie importate:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mip import Model, xsum, minimize, BINARY
```

**Parametri:**
```python
k = 15              # numero di punti
grid_side = 100     # dimensione della griglia
d_max = 0.35 * grid_side  # distanza massima (35% del lato della griglia)
start_point = 0     # punto di partenza
destination = 12    # destinazione
seed = ...          # seed per la generazione casuale
```

`00:11:00`

**Generazione dei punti:**
```python
points = np.random.random((k, 2)) * 100
```

Questa formula:
- Genera numeri casuali tra 0 e 1 con `np.random.random()`
- Moltiplica per 100 per ottenere coordinate tra 0 e 100
- Crea k=15 punti con coordinate (x, y)

### 2.3 Creazione degli archi e filtraggio distanze

`00:12:07`

**Matrice delle distanze:**
```python
distance = np.array([
    [np.sqrt((points[i] - points[j])**2).sum() 
     for j in range(k)] 
    for i in range(k)
])
```

Calcola la distanza euclidea tra ogni coppia di punti: $d(i,j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$

`00:13:52`

**Insieme di nodi e archi:**
```python
N = [i for i in range(k)]

A = [(i, j) for i in range(k) 
           for j in range(k) 
           if distance[i][j] <= d_max and i != j]
```

⚠️ **Nota importante:** 
- Condizione `i != j` esclude gli archi da un nodo a se stesso
- Condizione `distance[i][j] <= d_max` filtra archi troppo lunghi
- Gli archi sono **direzionali** (i,j) ≠ (j,i)

`00:14:43`

Se d_max è troppo alto (es. 75%), il grafo è troppo denso. Con d_max al 35-45% si ottiene un grafo più gestibile.

### 2.4 Formulazione a flusso

`00:17:04`

💡 **Idea della formulazione a flusso:**

In un network ci sono tre tipi di nodi:
1. **Nodo sorgente (S)**: da cui parte il flusso
2. **Nodo destinazione (T)**: dove arriva il flusso  
3. **Nodi intermedi**: attraverso cui passa il flusso

`00:18:11`

**Vincoli di conservazione del flusso per ogni tipo di nodo:**

| Tipo nodo | Vincolo | Significato |
|-----------|---------|-------------|
| Sorgente S | `OUT - IN = 1` | Esce esattamente 1 unità di flusso |
| Destinazione T | `OUT - IN = -1` | Entra esattamente 1 unità di flusso |
| Nodi intermedi | `OUT - IN = 0` | Tutto ciò che entra esce |

`00:21:27`

**Parametro ausiliario b_i:**

Per unificare i vincoli, si definisce un parametro $b_i$ per nodo:

$$
b_i = \begin{cases}
1 & \text{se } i = S \text{ (sorgente)} \\
-1 & \text{se } i = T \text{ (destinazione)} \\
0 & \text{altrimenti (nodi intermedi)}
\end{cases}
$$

**Vincolo unificato:**
$$
\sum_{j: (j,i) \in A} f_{ji} - \sum_{j: (i,j) \in A} f_{ij} = b_i \quad \forall i \in N
$$

Dove $f_{ij}$ è il flusso sull'arco (i,j).

**Funzione obiettivo:**
$$
\min \sum_{(i,j) \in A} d_{ij} \cdot f_{ij}
$$

Minimizza la distanza totale del percorso.

### 2.5 Implementazione del modello

`00:22:03`

**1. Inizializzazione del modello:**
```python
m = Model("shortest_path")
```

`00:22:46`

**2. Creazione delle variabili:**
```python
f = {}
for i, j in A:
    f[i, j] = m.add_var(name=f'f_{i}_{j}')
```

Le variabili $f_{ij}$ rappresentano il flusso sull'arco (i,j), sono continue ≥ 0.

`00:23:21`

**3. Creazione del parametro b_i:**
```python
b = {i: 0 for i in N}  # inizializza tutti a 0
b[0] = 1               # nodo sorgente
b[12] = -1             # nodo destinazione
```

`00:24:46`

**4. Vincoli di conservazione del flusso:**
```python
for i in N:
    m.add_constr(
        xsum(f[j, i] for j in N if (j, i) in A) -   # flusso IN
        xsum(f[i, j] for j in N if (i, j) in A) ==  # flusso OUT
        b[i]
    )
```

Traduzione del vincolo matematico:
- `xsum(f[j,i] for j in N if (j,i) in A)`: somma il flusso **entrante** nel nodo i
- `xsum(f[i,j] for j in N if (i,j) in A)`: somma il flusso **uscente** dal nodo i
- La differenza deve essere uguale a `b[i]`

`00:27:31`

⚠️ **Nota sulla direzionalità degli archi:**
- Gli archi sono **direzionali** (arcs, non edges)
- (i,j) indica flusso da i verso j
- (j,i) indica flusso da j verso i

`00:28:56`

**5. Funzione obiettivo:**
```python
m.objective = minimize(
    xsum(distance[i][j] * f[i, j] for (i, j) in A)
)
```

`00:29:37`

**6. Ottimizzazione:**
```python
m.optimize()
```

### 2.6 Estrazione della soluzione

`00:31:05`

**Valore della funzione obiettivo:**
```python
total_distance = m.objective_value
print(f"Distanza totale: {total_distance}")
```

`00:31:49`

**Estrazione degli archi nella soluzione:**
```python
solution_arcs = []
for (i, j) in A:
    if f[i, j].x > 0.9:  # o > 0
        print(f"Arco ({i}, {j}) nel percorso")
        solution_arcs.append((i, j))
```

📌 **Spiegazione:**
- `f[i,j].x` estrae il **valore** della variabile decisionale
- Se il valore è > 0.9 (essenzialmente 1), l'arco è nella soluzione
- Si usa 0.9 invece di == 1 per evitare problemi di precisione numerica

`00:32:21`

Il notebook contiene codice per visualizzare graficamente il percorso ottimo sulla griglia.

---

## 3. Problema 2: Minimum Spanning Tree (MST)

### 3.1 Definizione di Spanning Tree

`00:33:30`

📚 **Definizione di Minimum Spanning Tree:**

Dato un grafo G=(N,E):
- Uno **spanning tree** è un sottografo che:
  - Connette **tutti** i nodi
  - **Non forma cicli** (è un albero)
- Il **Minimum Spanning Tree** è lo spanning tree con il **costo totale minimo**

`00:35:20`

**Proprietà:**
- In un grafo possono esistere molti spanning tree diversi
- Il MST è quello con la somma minima dei pesi degli archi

### 3.2 Formulazione basata su Cut Sets

`00:36:32`

💡 **Concetto di Cut Set:**

`00:37:23`

Dato un grafo, se prendiamo un sottoinsieme S di nodi:

```
Esempio:
    1 --- 3
    |     |
    2     4
    |
    7
```

Se S = {1, 2}, il **cut set δ(S)** è l'insieme degli archi che "tagliano" il confine tra S e i nodi non in S.

`00:37:58`

Nel nostro esempio:
- S = {1, 2}
- Nodi non in S = {3, 4, 7}
- Cut set δ(S) = {(1,3), (1,4), (2,7)}

`00:39:00`

🔑 **Proprietà fondamentale:**

In un spanning tree valido, **almeno un arco del cut set deve essere nella soluzione**, altrimenti il grafo si disconnette.

**Formulazione matematica:**

$$
\min \sum_{e \in E} c_e \cdot x_e
$$

Soggetto a:
$$
\sum_{e \in \delta(S)} x_e \geq 1 \quad \forall S \subseteq N, S \neq \emptyset, S \neq N
$$

Dove:
- $x_e \in \{0,1\}$ indica se l'arco e è nello spanning tree
- $c_e$ è il costo dell'arco e
- $\delta(S)$ è il cut set del sottoinsieme S

`00:40:14`

⚠️ **Problema computazionale:**

Il numero di vincoli è **esponenziale** nel numero di nodi:

$$
|\text{Sottoinsiemi}| = 2^{|N|} - 2
$$

(si escludono l'insieme vuoto e l'insieme completo)

`00:41:32`

Per k=15 nodi: $2^{15} - 2 = 32766$ vincoli!

### 3.3 Generazione di tutte le combinazioni di sottoinsiemi

`00:42:52`

**Variabili decisionali:**

Le variabili $x_{ij}$ sono ora **binarie** (non più continue come nel flusso):

$$
x_{ij} = \begin{cases}
1 & \text{se l'arco (i,j) è nello spanning tree} \\
0 & \text{altrimenti}
\end{cases}
$$

`00:44:01`

**Generazione del power set (insieme delle parti):**

```python
from itertools import combinations

power_set = []
for size in range(1, k):  # da 1 a k-1 (escluso k)
    for subset in combinations(N, size):
        power_set.append(subset)
```

Questo genera:
- Tutti i sottoinsiemi di dimensione 1: {1}, {2}, {3}, ...
- Tutti i sottoinsiemi di dimensione 2: {1,2}, {1,3}, ...
- ...
- Tutti i sottoinsiemi di dimensione k-1

### 3.4 Implementazione del modello MST

`00:49:05`

**1. Generazione del grafo:**
```python
k = 15
points = np.random.random((k, 2)) * 100
distance = np.array([
    [np.sqrt((points[i] - points[j])**2).sum() 
     for j in range(k)] 
    for i in range(k)
])
```

`00:50:27`

**2. Definizione degli archi (edges, non arcs):**
```python
E = [(i, j) for i in range(k) 
           for j in range(k) 
           if distance[i][j] <= d_max and i < j]
```

⚠️ **Importante:** Ora si usa `i < j` perché gli **edges** sono non direzionali. Si considera solo una direzione per evitare duplicati.

`00:51:01`

**3. Inizializzazione del modello:**
```python
m = Model("MST")
```

**4. Variabili binarie:**
```python
x = {}
for i, j in E:
    x[i, j] = m.add_var(var_type=BINARY, name=f'x_{i}_{j}')
```

`00:51:34`

📌 Differenza chiave: `var_type=BINARY` specifica che la variabile può essere solo 0 o 1.

`00:52:05`

**5. Vincoli di cut set:**
```python
for S in power_set:
    if len(S) < k:  # esclude l'insieme completo
        m.add_constr(
            xsum(x[i, j] for (i, j) in E 
                 if (i in S and j not in S) or (i not in S and j in S)) 
            >= 1
        )
```

`00:54:06`

Spiegazione della condizione:
- `(i in S and j not in S)`: arco da S verso fuori
- `(i not in S and j in S)`: arco da fuori verso S
- Insieme: tutti gli archi che attraversano il confine di S

`00:56:00`

**6. Funzione obiettivo:**
```python
m.objective = minimize(
    xsum(distance[i][j] * x[i, j] for (i, j) in E)
)
```

**7. Ottimizzazione:**
```python
m.optimize()
```

### 3.5 Visualizzazione della soluzione

`00:57:08`

```python
# Estrazione degli archi nella soluzione
mst_edges = [(i, j) for (i, j) in E if x[i, j].x > 0.5]
```

`00:57:38`

✅ **Risultato:** Un albero che:
- Connette tutti i nodi
- Non ha cicli
- Ha costo minimo

---

## 4. Problema 3: Travelling Salesman Problem (TSP)

### 4.1 Introduzione al TSP

`00:58:58`

🎯 **Problema del Commesso Viaggiatore (TSP):**

Dato un insieme di n nodi dove la distanza tra ogni coppia è nota, trovare il **circuito hamiltoniano più breve** che:
- Visita **ogni nodo esattamente una volta**
- Ritorna al nodo di partenza

`00:59:34`

**Differenza con Shortest Path:**
- Shortest Path: connette due nodi specifici
- TSP: tour chiuso che visita tutti i nodi

### 4.2 Vincoli di conservazione del flusso

`01:00:10`

Per il TSP, il vincolo di flusso è diverso:

**Per ogni nodo:**
$$
\sum_{j: (i,j) \in E} x_{ij} + \sum_{j: (j,i) \in E} x_{ji} = 2
$$

Cioè: **IN + OUT = 2** per ogni nodo.

`01:00:53`

💡 **Ragionamento:**
- Ogni nodo deve essere **visitato** → 1 arco entrante
- Ogni nodo deve essere **lasciato** → 1 arco uscente
- Totale: 2 archi incidenti su ogni nodo

### 4.3 Problema dei subtour

`01:01:24`

❌ **Problema:** I vincoli di conservazione del flusso da soli **non garantiscono** un tour unico.

**Esempio di soluzione non valida:**

```
Subtour 1: 0 → 2 → 4 → 0
Subtour 2: 1 → 3 → 5 → 6 → 1
```

Questa soluzione soddisfa i vincoli di flusso (ogni nodo ha IN+OUT=2), ma **non è un tour unico**.

`01:01:58`

### 4.4 Subtour Elimination Constraints

`01:02:28`

**Vincoli di eliminazione dei subtour:**

Per ogni sottoinsieme proprio S ⊂ N:
$$
\sum_{i,j \in S} x_{ij} \leq |S| - 1
$$

💡 **Intuizione:**
- In un sottociclo di k nodi ci sono k archi
- Il vincolo impone al massimo k-1 archi
- Questo **rompe il sottociclo**

`01:03:43`

**Esempio:** Se abbiamo il subtour 0 → 2 → 4 → 0:

$$
x_{0,2} + x_{2,4} + x_{4,0} \leq 3 - 1 = 2
$$

Nella soluzione con il subtour: $x_{0,2} + x_{2,4} + x_{4,0} = 3$, che **viola** il vincolo.

`01:14:20`

### 4.5 Approccio iterativo

`01:14:54`

⚠️ **Problema:** Aggiungere tutti i vincoli di eliminazione subtour in anticipo è **impraticabile**:
- Numero di vincoli: $2^n - 2$ (esponenziale)
- Con 10-20 nodi già diventa intrattabile

`01:15:25`

✅ **Soluzione: Aggiunta iterativa (Lazy Constraints)**

**Algoritmo:**
1. Risolvere il problema **senza** vincoli di eliminazione subtour
2. Se la soluzione contiene subtour:
   - Identificare i subtour
   - Aggiungere vincoli per romperli
3. Risolvere di nuovo
4. Ripetere fino a ottenere un tour valido

`01:15:55`

📌 **Vantaggio:** Si aggiungono solo i vincoli **necessari**, evitando l'esplosione combinatoria.

### 4.6 Implementazione del modello TSP

`01:07:34`

**1. Generazione del grafo:**
```python
k = 15
points = np.random.random((k, 2)) * 100
distance = np.array([...])  # come prima
```

`01:08:23`

**2. Definizione degli archi:**
```python
E = [(i, j) for i in range(k) 
           for j in range(k) 
           if i < j and distance[i][j] <= d_max]
```

**3. Inizializzazione del modello:**
```python
m = Model("TSP")
```

`01:08:54`

**4. Variabili binarie:**
```python
x = {}
for i, j in E:
    x[i, j] = m.add_var(var_type=BINARY, name=f'x_{i}_{j}')
```

`01:10:02`

**5. Vincoli di conservazione del flusso:**
```python
for i in N:
    m.add_constr(
        xsum(x[j, i] for j in N if j < i and (j, i) in E) +    # IN
        xsum(x[i, j] for j in N if i < j and (i, j) in E)      # OUT
        == 2
    )
```

📌 **Nota:** Poiché gli archi sono non direzionali ma memorizzati come (i,j) con i<j, bisogna considerare entrambi gli orientamenti:
- `x[j,i]` con j<i rappresenta un arco entrante in i
- `x[i,j]` con i<j rappresenta un arco uscente da i

`01:11:31`

**6. Funzione obiettivo:**
```python
m.objective = minimize(
    xsum(distance[i][j] * x[i, j] for (i, j) in E)
)
```

`01:12:15`

**7. Prima ottimizzazione (senza subtour elimination):**
```python
m.optimize()
```

### 4.7 Aggiunta iterativa dei vincoli

`01:12:52`

**Esempio di risultato con subtour:**
```
Subtour 1: 0 → 2 → 4 → 0
Subtour 2: 6 → 3 → 1 → 5 → 6
```

`01:13:45`

**Aggiunta manuale dei vincoli:**

Per il primo subtour {0, 2, 4}:
```python
m.add_constr(x[0, 2] + x[2, 4] + x[4, 0] <= 2)
```

`01:14:20`

Per il secondo subtour {6, 5, 1, 3}:
```python
m.add_constr(x[6, 5] + x[5, 1] + x[1, 3] + x[3, 6] <= 3)
```

`01:16:40`

⚠️ **Attenzione alla direzione:** Se gli archi sono memorizzati come (i,j) con i<j, bisogna usare la rappresentazione corretta:
- Se 6 > 5, l'arco è `x[5, 6]`
- Se 3 > 1, l'arco è `x[1, 3]`
- Ecc.

`01:19:08`

**Risoluzione dopo aggiunta vincoli:**
```python
m.optimize()
```

`01:20:27`

**Implementazione automatica:**

Il notebook completo contiene funzioni per:
1. **Identificare automaticamente i subtour** nella soluzione corrente
2. **Aggiungere i vincoli** corrispondenti
3. **Iterare** fino a convergenza

```python
def find_subtours(solution_edges):
    # identifica i cicli nella soluzione
    # ritorna lista di subtour
    ...

def add_subtour_constraints(m, subtours, x):
    # aggiunge vincoli per ogni subtour
    for subtour in subtours:
        m.add_constr(
            xsum(x[i, j] for (i, j) in subtour) 
            <= len(subtour) - 1
        )

# Loop principale
while True:
    m.optimize()
    subtours = find_subtours(current_solution)
    if len(subtours) == 1:  # tour unico trovato
        break
    add_subtour_constraints(m, subtours, x)
```

`01:21:21`

✅ **Soluzione finale:** Un tour hamiltoniano valido che visita tutti i nodi esattamente una volta con distanza minima.

---

## 5. Conclusioni e Domande

`01:21:21`

**Riepilogo del laboratorio:**

| Problema | Tipo variabili | Vincoli principali | Difficoltà |
|----------|---------------|-------------------|-----------|
| **Shortest Path** | Continue (flusso) | Conservazione flusso (IN-OUT=b_i) | ⭐⭐ |
| **Minimum Spanning Tree** | Binarie | Cut set constraints ($\geq 1$ per ogni taglio) | ⭐⭐⭐⭐ |
| **TSP** | Binarie | Conservazione flusso + Subtour elimination | ⭐⭐⭐⭐⭐ |

`01:21:53`

📢 **Promemoria attività peer review:**
- Gli studenti che vogliono partecipare devono registrarsi entro fine settimana
- Riceveranno il problema corrispondente
- Valutare le soluzioni di altri studenti
- Ricevere feedback

`01:23:42`

**Domande degli studenti:**

❓ **È obbligatorio partecipare al peer review?**
- Non è chiaro se ci siano punti bonus
- Chiedere al professore Malucelli per i dettagli

`01:24:24`

❓ **Consigli per imparare Python per il corso?**
- Per questo corso serve conoscere **molto poco** Python
- Le parti principali sono tre funzioni del package MIP:
  - `add_var()` - definire variabili
  - `add_constr()` - definire vincoli
  - `objective` - definire funzione obiettivo
- Il resto è sintassi base Python (dizionari, liste)
- Non servono algoritmi complessi
- La parte importante è **saper modellare** il problema matematicamente
- Tradurre il modello in codice è semplice

`01:26:00`

✅ **Fine del laboratorio**

---

## 📊 Tabelle Riassuntive

### Confronto tra le tre formulazioni

| Aspetto | Shortest Path | MST | TSP |
|---------|--------------|-----|-----|
| **Tipo grafo** | Diretto (arcs) | Non diretto (edges) | Non diretto (edges) |
| **Variabili** | Continue $f_{ij} \geq 0$ | Binarie $x_{ij} \in \{0,1\}$ | Binarie $x_{ij} \in \{0,1\}$ |
| **Obiettivo** | Min distanza S→T | Min costo totale albero | Min distanza tour |
| **Vincoli** | Conservazione flusso | Cut set constraints | Flusso + Subtour elim. |
| **Complessità vincoli** | O(n) | O($2^n$) | O($2^n$) |
| **Approccio** | Diretto | Tutti i vincoli upfront | Iterativo (lazy) |

### Sintassi Python-MIP essenziale

| Operazione | Codice | Note |
|-----------|--------|------|
| Creare modello | `m = Model()` | |
| Variabile continua | `m.add_var()` | Default ≥ 0 |
| Variabile binaria | `m.add_var(var_type=BINARY)` | Solo 0 o 1 |
| Aggiungere vincolo | `m.add_constr(expr)` | expr con `==`, `<=`, `>=` |
| Funzione obiettivo | `m.objective = minimize(expr)` | O `maximize` |
| Sommatoria | `xsum(expr for ... if ...)` | Come comprehension |
| Risolvere | `m.optimize()` | |
| Valore obiettivo | `m.objective_value` | Dopo optimize |
| Valore variabile | `x[i,j].x` | Dopo optimize |

### Parametri tipici usati

```python
k = 15                      # numero di nodi
grid_side = 100            # dimensione griglia
d_max = 0.35 * grid_side   # distanza massima per arco
start_point = 0            # nodo sorgente
destination = 12           # nodo destinazione
seed = 42                  # seed per riproducibilità
```

