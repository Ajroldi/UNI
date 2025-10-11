# Lezione 7 - 7 Ottobre: Problemi Multi-periodo, MST e TSP

## 📑 Indice

1. [Introduzione](#introduzione) `00:02:52`
2. [Problema di Acquisto Gas Multi-periodo](#problema-gas) `00:02:52 - 00:22:00`
   - [Formulazione del Problema](#formulazione-gas)
   - [Insiemi, Parametri e Variabili](#insiemi-parametri-gas)
   - [Funzione Obiettivo e Vincoli](#funzione-obiettivo-gas)
3. [Minimum Spanning Tree (MST)](#minimum-spanning-tree) `00:22:00 - 00:38:08`
   - [Definizione e Proprietà](#definizione-mst)
   - [Network Design e Applicazioni](#network-design)
   - [Formulazione Matematica](#formulazione-mst)
4. [Algoritmo di Kruskal](#algoritmo-kruskal) `00:29:38 - 00:35:13`
   - [Approccio Greedy](#approccio-greedy-kruskal)
   - [Struttura Dati Find-Union](#find-union)
   - [Complessità Computazionale](#complessita-kruskal)
5. [Algoritmo di Prim](#algoritmo-prim) `00:35:13 - 00:36:00`
6. [Traveling Salesman Problem (TSP)](#traveling-salesman-problem) `00:38:08 - fine`
   - [Motivazione e Formulazione](#motivazione-tsp)
   - [Vincoli di Grado](#vincoli-grado-tsp)
   - [Eliminazione Sottocicli](#eliminazione-sottocicli)
   - [Ciclo Hamiltoniano](#ciclo-hamiltoniano)
7. [Riepilogo e CAHOOT](#riepilogo-cahoot) `00:49:31 - 01:09:00`

---

## <a name="introduzione"></a>1. Introduzione

`00:02:52`

Buongiorno. Oggi affronteremo:
- Un **esercizio di modellazione** su un problema multi-periodo
- Il **ripasso dell'algoritmo** del minimum spanning tree
- La **generalizzazione** del problema come network design
- L'introduzione ai **grafi orientati**

Faremo anche il **recap count** sull'attività blended riguardante le definizioni sui grafi.

---

## <a name="problema-gas"></a>2. Problema di Acquisto Gas Multi-periodo

### <a name="formulazione-gas"></a>2.1 Formulazione del Problema

`00:02:52 - 00:05:04`

> **Contesto del Problema**
> 
> Dobbiamo acquistare gas naturale per i prossimi tre giorni, con le seguenti caratteristiche:
> - **Prezzi variabili** giorno per giorno
> - **Deposito** con capacità limitata Q per lo stoccaggio
> - **Espansione della capacità** possibile a costo fisso CT per giorno
> - **Domanda di mercato** da soddisfare esattamente
> - **Incertezza nella domanda** che richiede gas di riserva
> - **Soglia minima** q di sicurezza nel deposito
> - **Penalità** per scendere sotto la soglia minima

**Obiettivo**: Minimizzare il costo totale di acquisto, espansione e penalità, gestendo l'incertezza della domanda.

### <a name="insiemi-parametri-gas"></a>2.2 Insiemi, Parametri e Variabili

`00:05:04 - 00:10:00`

#### Insiemi

- **D**: Insieme dei giorni nell'orizzonte di pianificazione
  - Indice: **d** ∈ D

#### Parametri

- **Q**: Capacità nominale del deposito
- **Q⁺**: Capacità massima espansa del deposito
- **P_d**: Prezzo del gas al giorno d
- **C_d**: Costo di espansione della capacità al giorno d
- **D_d**: Domanda di mercato al giorno d
- **q**: Soglia minima di sicurezza nel deposito
- **π_d**: Penalità per unità di gas mancante sotto la soglia al giorno d

#### Variabili Decisionali

- **x_d**: Quantità di gas acquistata al giorno d
- **y_d**: Variabile binaria di decisione espansione (1 se espandiamo, 0 altrimenti)
- **s_d**: Livello di scorta (stock) nel deposito al giorno d
- **z_d**: Quantità sotto la soglia minima (shortage) al giorno d

> **📝 Nota**: Le variabili di stock **s_d** rappresentano l'inventario alla fine di ogni giorno, mentre le variabili di shortage **z_d** quantificano il rischio accettato scendendo sotto la soglia di sicurezza.

### <a name="funzione-obiettivo-gas"></a>2.3 Funzione Obiettivo e Vincoli

`00:10:00 - 00:22:00`

#### Funzione Obiettivo

```
min Σ_{d∈D} (P_d · x_d + C_d · y_d + π_d · z_d)
```

La funzione obiettivo minimizza la somma di tre componenti di costo:
1. **Costo di acquisto**: P_d · x_d
2. **Costo di espansione**: C_d · y_d  
3. **Penalità per rischio**: π_d · z_d

#### Vincoli del Problema

**1. Bilancio di Inventario Multi-periodo**

```
s_d = s_{d-1} + x_d - D_d    ∀d ∈ D, d > 1
s_1 = s_0 + x_1 - D_1         (primo giorno, con s_0 dato)
```

> Questo vincolo rappresenta la **conservazione del materiale**: lo stock alla fine del giorno d è uguale allo stock del giorno precedente, più l'acquisto del giorno, meno la domanda soddisfatta.

**2. Vincoli di Capacità**

```
s_d ≤ Q + y_d · (Q⁺ - Q)    ∀d ∈ D
```

Se y_d = 0 (nessuna espansione): s_d ≤ Q (capacità nominale)
Se y_d = 1 (espansione attivata): s_d ≤ Q⁺ (capacità espansa)

**3. Vincoli di Soglia Minima e Shortage**

```
s_d + z_d ≥ q    ∀d ∈ D
```

Questo vincolo garantisce che la somma dello stock effettivo e della quantità mancante (shortage) soddisfi sempre la soglia di sicurezza.

**4. Vincoli di Non-negatività**

```
x_d, s_d, z_d ≥ 0    ∀d ∈ D
y_d ∈ {0, 1}         ∀d ∈ D
```

> **💡 Insight**: Questo è un tipico problema di **programmazione lineare intera mista** (MILP) perché combina variabili continue (x_d, s_d, z_d) e variabili binarie (y_d).

---

## <a name="minimum-spanning-tree"></a>3. Minimum Spanning Tree (MST)

### <a name="definizione-mst"></a>3.1 Definizione e Proprietà

`00:22:00 - 00:29:38`

### <a name="network-design"></a>3.2 Network Design e Applicazioni

`00:22:00`

> **Esempio Applicativo: Campus Universitario**
>
> Immaginiamo di dover **connettere gli edifici di un campus universitario** con una rete di comunicazione (fibra ottica, cavi elettrici, tubi, ecc.).
> 
> - Ogni edificio è un **nodo**
> - Ogni possibile connessione è un **arco** con costo di installazione
> - Obiettivo: **Minimizzare il costo totale** garantendo che tutti gli edifici siano connessi

Questo è un classico problema di **network design ridondante**, dove vogliamo la rete meno costosa che garantisca la connettività.

#### Proprietà di un Minimum Spanning Tree

Un **albero ricoprente minimo** (MST) su un grafo non orientato G = (N, A) con n nodi ha le seguenti proprietà fondamentali:

1. **n-1 archi**: Un MST con n nodi contiene esattamente n-1 archi
   - Meno archi → grafo non connesso
   - Più archi → presenza di cicli

2. **Nessun ciclo**: È un albero (grafo aciclico connesso)

3. **Connesso**: Esiste un percorso tra ogni coppia di nodi

4. **Sottografo ricoprente**: Include tutti i nodi del grafo originale

5. **Peso minimo**: Tra tutti gli alberi ricoprenti possibili, ha la somma minima dei costi degli archi

> **📌 Definizione Formale**
> 
> Un albero ricoprente T di G è un sottografo T = (N, A_T) dove:
> - A_T ⊆ A (subset degli archi)
> - |A_T| = n - 1
> - T è connesso e aciclico
> - T minimizza Σ_{e∈A_T} c_e

### <a name="formulazione-mst"></a>3.3 Formulazione Matematica

`00:27:00 - 00:29:38`

#### Formulazione con Vincoli di Connettività

**Variabili**:
- x_ij ∈ {0,1}: uguale a 1 se l'arco (i,j) è selezionato nell'MST

**Funzione Obiettivo**:
```
min Σ_{(i,j)∈A} c_ij · x_ij
```

**Vincoli**:
```
Σ_{(i,j)∈A} x_ij = n - 1                     (esattamente n-1 archi)

Σ_{(i,j)∈δ(S)} x_ij ≥ 1    ∀S ⊂ N, S ≠ ∅, S ≠ N    (connettività)

x_ij ∈ {0,1}    ∀(i,j) ∈ A
```

Dove δ(S) (= insieme di archi) rappresenta il **taglio** (cut) definito dal sottoinsieme S di nodi.

> **⚠️ Complessità**
> 
> Il numero di vincoli di connettività è **esponenziale** nel numero di nodi (2^n - 2 vincoli possibili = tutti i possibili sottoinsiemi non banali di archi, cioè i possibili tagli).
> 
> Tuttavia, il problema MST è risolvibile in **tempo polinomiale** grazie al **problema di separazione**, che può essere risolto trovando il taglio minimo in tempo polinomiale.

**Separazione dei Vincoli (problema di separazione)**:

Dato una soluzione x* candidata, per verificare se viola qualche vincolo di connettività (sono troppi per controllarli ciascuno):
1. Costruiamo un grafo con gli archi dove x*_ij > 0 (archi presenti nella soluzione x*)
2. Cerchiamo un **taglio minimo** (questa ricerca si risolve in tempo polinomiale)
3. Se il taglio minimo ha capacità < 1, abbiamo trovato un vincolo violato (c’è meno di un collegamento “completo” che collega il gruppo al resto, quindi il grafo rischia di spezzarsi in due parti separate).
4. Altrimenti, la soluzione è ammissibile

Questo significa che, anche con vincoli esponenziali, possiamo risolvere il problema in tempo polinomiale.

---

## <a name="algoritmo-kruskal"></a>4. Algoritmo di Kruskal

### <a name="approccio-greedy-kruskal"></a>4.1 Approccio Greedy

`00:29:38 - 00:31:00`

L'**algoritmo di Kruskal** è un algoritmo **greedy** (goloso) che costruisce l'MST selezionando archi in ordine crescente di costo, evitando la formazione di cicli.

#### Pseudocodice dell'Algoritmo

```
KRUSKAL(G, c):
    1. Ordina gli archi A in ordine crescente di costo c
    2. Inizializza T = ∅ (insieme archi dell'MST)
    3. Inizializza n componenti connesse (ogni nodo è una componente)
    
    4. Per ogni arco (i,j) in ordine crescente di costo:
        5. Se i e j appartengono a componenti diverse:
            6. Aggiungi (i,j) a T
            7. Unisci le due componenti connesse
    
    8. Ritorna T
```

> **💡 Idea Chiave**
> 
> Ad ogni passo, scegliamo l'arco di **costo minimo** che non crea cicli. Un arco crea un ciclo se e solo se i due estremi appartengono già alla stessa componente connessa.

#### Esempio di Esecuzione

Consideriamo un grafo con archi ordinati:
- (1,2): costo 1
- (3,4): costo 2
- (2,3): costo 3
- (1,4): costo 5
- (2,4): costo 6

**Passo 1**: Aggiungiamo (1,2) → Componenti: {1,2}, {3}, {4}
**Passo 2**: Aggiungiamo (3,4) → Componenti: {1,2}, {3,4}
**Passo 3**: Aggiungiamo (2,3) → Componenti: {1,2,3,4}
**Passo 4**: Saltiamo (1,4) perché creerebbe un ciclo
**Passo 5**: Saltiamo (2,4) perché creerebbe un ciclo

**MST**: {(1,2), (3,4), (2,3)} con costo totale = 6

### <a name="find-union"></a>4.2 Struttura Dati Find-Union

`00:31:00 - 00:33:30`

Per implementare efficientemente l'algoritmo di Kruskal, utilizziamo la struttura dati **Find-Union** (o **Union-Find** o **Disjoint Set**).

#### Operazioni Fondamentali

1. **FIND(x)**: Restituisce l'identificatore della componente connessa contenente x
2. **UNION(x, y)**: Unisce le componenti connesse contenenti x e y

#### Implementazione con Path Compression

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Ogni nodo è genitore di se stesso
        self.rank = [0] * n           # Altezza dell'albero
    
    def find(self, x):
        # Path compression: collega x direttamente alla radice
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Già nella stessa componente
        
        # Union by rank: attacca l'albero più piccolo al più grande
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True  # Unione avvenuta
```

> **📝 Contributo di Robert Tarjan**
> 
> La struttura dati Union-Find con **path compression** e **union by rank** fu analizzata e ottimizzata da **Robert Tarjan**, vincitore del Turing Award 1986 per i suoi contributi fondamentali agli algoritmi su grafi.

### <a name="complessita-kruskal"></a>4.3 Complessità Computazionale

`00:33:30 - 00:35:13`

#### Analisi della Complessità

Sia n = |N| il numero di nodi e m = |A| il numero di archi.

1. **Ordinamento archi**: O(m log m)
2. **Inizializzazione Union-Find**: O(n)
3. **Ciclo principale**: 
   - m iterazioni
   - Ogni iterazione: 2 operazioni FIND + 1 UNION
   - Costo ammortizzato per operazione: O(α(n))
   - Dove α(n) è la funzione **inversa di Ackermann**

#### Complessità Totale

```
T(n,m) = O(m log m) + O(n) + O(m · α(n))
       = O(m log m)
```

Poiché m ≤ n(n-1)/2, abbiamo log m ≤ log(n²) = 2 log n, quindi:

```
T(n,m) = O(m log n)
```

> **🚀 Efficienza Quasi-Lineare**
> 
> La funzione α(n) cresce **estremamente lentamente**:
> - α(n) < 5 per ogni valore pratico di n (anche n = 2^65536)
> - In pratica, O(m · α(n)) ≈ O(m)
> 
> Quindi l'algoritmo di Kruskal ha complessità **quasi-lineare** nel numero di archi!

#### Confronto con Altri Algoritmi MST

| Algoritmo | Complessità | Note |
|-----------|-------------|------|
| Kruskal | O(m log n) | Ottimo per grafi sparsi |
| Prim | O(m + n log n) | Con heap di Fibonacci |
| Borůvka | O(m log n) | Parallelizzabile |

---

## <a name="algoritmo-prim"></a>5. Algoritmo di Prim

`00:35:13 - 00:36:00`

L'**algoritmo di Prim** è un altro algoritmo greedy per trovare l'MST, ma con un approccio diverso da Kruskal.

#### Idea dell'Algoritmo

Invece di ordinare tutti gli archi, Prim:
1. Inizia da un **singolo nodo** arbitrario
2. Mantiene un **insieme S** di nodi già inclusi nell'MST
3. Ad ogni passo, aggiunge l'arco di **costo minimo** che connette un nodo in S con un nodo fuori da S
4. Continua fino a includere tutti i nodi

#### Pseudocodice

```
PRIM(G, c, s):
    1. Inizializza S = {s} (componente iniziale)
    2. Inizializza T = ∅ (archi MST)
    
    3. Finché S ≠ N:
        4. Trova l'arco (i,j) di costo minimo con i ∈ S, j ∉ S
        5. Aggiungi (i,j) a T
        6. Aggiungi j a S
    
    7. Ritorna T
```

#### Differenze con Kruskal

| Aspetto | Kruskal | Prim |
|---------|---------|------|
| **Approccio** | Ordina tutti gli archi globalmente | Cresce componente localmente |
| **Struttura dati** | Union-Find | Heap/Priority Queue |
| **Migliore per** | Grafi sparsi | Grafi densi |
| **Parallelizzazione** | Difficile | Difficile |

> **📚 Nota Storica**
> 
> Entrambi gli algoritmi furono sviluppati indipendentemente:
> - **Kruskal**: Joseph Kruskal, 1956
> - **Prim**: Robert Prim, 1957 (ma già scoperto da Jarník nel 1930)

---

## <a name="traveling-salesman-problem"></a>6. Traveling Salesman Problem (TSP)

### <a name="motivazione-tsp"></a>6.1 Motivazione e Formulazione

`00:38:08 - 00:42:00`

Il **Problema del Commesso Viaggiatore** (Traveling Salesman Problem, TSP) è una generalizzazione del problema MST dove vogliamo una rete **ridondante**.

#### Dal MST al TSP

Nel problema MST:
- Obiettivo: Connettere tutti i nodi con costo minimo
- Soluzione: n-1 archi (albero)
- Proprietà: Nessun ciclo, connettività minima

Nel TSP:
- Obiettivo: Visitare tutti i nodi e tornare al punto di partenza con costo minimo
- Soluzione: n archi (ciclo)
- Proprietà: Un ciclo che passa per tutti i nodi esattamente una volta

> **💡 Intuizione**
> 
> Il TSP può essere visto come un problema di **network design con ridondanza**: vogliamo una rete dove ogni nodo è connesso con **esattamente 2 archi**, garantendo un percorso ciclico completo.

#### Formulazione Matematica del TSP

**Variabili**:
- x_ij ∈ {0,1}: uguale a 1 se l'arco (i,j) è nel tour

**Funzione Obiettivo**:
```
min Σ_{(i,j)∈A} c_ij · x_ij
```

### <a name="vincoli-grado-tsp"></a>6.2 Vincoli di Grado

`00:42:00 - 00:44:00`

**Vincoli di Grado** (Degree Constraints):

```
Σ_{j:(i,j)∈A} x_ij + Σ_{j:(j,i)∈A} x_ji = 2    ∀i ∈ N
```

Ogni nodo deve avere **esattamente 2 archi** incidenti:
- Un arco "entrante" (incoming)
- Un arco "uscente" (outgoing)

In un grafo non orientato, questo significa che ogni nodo deve avere grado esattamente 2.

> **⚠️ Problema**: I vincoli di grado da soli NON garantiscono un unico ciclo! Potrebbero formarsi **sottocicli** (subtours) disgiunti.

#### Esempio di Soluzione con Sottocicli

Consideriamo 6 nodi {1, 2, 3, 4, 5, 6}:

**Soluzione con sottocicli**:
- Ciclo 1: 1 → 2 → 3 → 1 (grado 2 per nodi 1,2,3 ✓)
- Ciclo 2: 4 → 5 → 6 → 4 (grado 2 per nodi 4,5,6 ✓)

Questa soluzione soddisfa i vincoli di grado ma **non è un tour valido** perché abbiamo due cicli separati invece di uno solo!

### <a name="eliminazione-sottocicli"></a>6.3 Eliminazione Sottocicli

`00:44:00 - 00:46:00`

Per eliminare i sottocicli, aggiungiamo **vincoli di connettività**:

```
Σ_{(i,j)∈δ(S)} x_ij ≥ 2    ∀S ⊂ N, 2 ≤ |S| ≤ n-2
```

Dove δ(S) è il taglio definito da S.

Questi vincoli richiedono che per ogni sottoinsieme proprio di nodi, **almeno 2 archi** attraversino il taglio, garantendo connettività forte e impedendo sottocicli isolati.

> **🔍 Osservazione Chiave**
> 
> Confrontiamo MST e TSP:
> 
> | Aspetto | MST | TSP |
> |---------|-----|-----|
> | **Numero archi** | n-1 | n |
> | **Vincoli taglio** | ≥ 1 arco | ≥ 2 archi |
> | **Grado nodi** | Variabile | Esattamente 2 |
> | **Complessità** | **Polinomiale** | **NP-hard** |
> 
> Una differenza apparentemente piccola (1 vs 2 archi per taglio) porta a una **drammatica differenza di complessità**!

### <a name="ciclo-hamiltoniano"></a>6.4 Ciclo Hamiltoniano

`00:46:00 - 00:49:01`

Il TSP cerca un **ciclo Hamiltoniano**: un ciclo che visita ogni nodo esattamente una volta.

#### Storia: Il Gioco Dicosiano di Lord Hamilton

`00:46:30`

Il concetto di ciclo Hamiltoniano prende il nome da **Sir William Rowan Hamilton**, matematico irlandese del XIX secolo.

Nel 1857, Hamilton inventò il **Gioco Dicosiano** (Icosian Game):
- Basato su un dodecaedro (12 facce pentagonali, 20 vertici)
- Ogni vertice rappresenta una città famosa
- Obiettivo: Trovare un percorso che visiti ogni città esattamente una volta e torni all'inizio
- Hamilton brevettò e vendette il gioco per £25 (circa $50 dell'epoca)

> **📚 Curiosità Storiche**
> 
> - Il gioco non ebbe grande successo commerciale
> - Hamilton era più famoso per i **quaternioni** (estensione dei numeri complessi)
> - Il concetto di ciclo Hamiltoniano divenne fondamentale nella teoria dei grafi
> - Oggi il TSP è uno dei problemi più studiati in ricerca operativa

#### TSP nella Cultura Popolare

`00:48:21 - 00:49:01`

Il TSP è apparso in vari contesti culturali:

1. **Software Concord TSP**: Solver gratuito per TSP disponibile per smartphone

2. **Film**: Esiste un thriller/detective basato sulla soluzione di un TSP (difficile da trovare online)

3. **Letteratura**: Un libro di narrativa (fiction) basato sul TSP, disponibile solo in **italiano** e **tedesco**

4. **Competizioni**: Esistono istanze TSP famose con migliaia di città (es. USA 48 stati, tour mondiale)

> **🎮 Provalo Tu!**
> 
> Cerca "Concord TSP" sul tuo smartphone per risolvere istanze TSP interattivamente!

---

## <a name="riepilogo-cahoot"></a>7. Riepilogo e CAHOOT

### 7.1 Riepilogo della Lezione

`00:49:31 - 00:54:19`

Ricapitoliamo ciò che abbiamo visto:

#### Grafi Non Orientati

1. **Notazione dei grafi**:
   - Nodi (N) e archi (A)
   - Incidenza, adiacenza
   - Gradi dei nodi

2. **Minimum Spanning Tree**:
   - Definizione e proprietà (n-1 archi, nessun ciclo, connesso)
   - **Algoritmo di Kruskal**: approccio greedy con Union-Find, O(m log n)
   - **Algoritmo di Prim**: crescita della componente, O(m + n log n)
   - Caratterizzazione dell'ottimalità

3. **Traveling Salesman Problem**:
   - Generalizzazione con ridondanza (n archi, grado 2)
   - Vincoli di grado e eliminazione sottocicli
   - Ciclo Hamiltoniano
   - **Problema difficile** (NP-hard)

#### Riflessione Proposta

`00:51:10 - 00:52:18`

Due esercizi di riflessione:

1. **Confrontare MST e TSP**: 
   - Analizzare le differenze nelle formulazioni
   - Capire perché TSP è così difficile nonostante la somiglianza con MST

2. **Estendere Kruskal per TSP**:
   - Provare a modificare l'algoritmo di Kruskal per fornire una soluzione (anche ammissibile) per il TSP
   - Sperimentare approcci euristici

> **💭 Osservazione del Professore**
> 
> "La cosa bella del TSP è che è molto facile **giocare** con il problema e fornire soluzioni o algoritmi, anche se il problema è difficile. Provate!"

### 7.2 CAHOOT - Verifica delle Conoscenze

`00:56:07 - 01:09:00`

#### Domanda 1: Riconoscimento Strutture

`00:56:07`

**Grafo Dato**: Insieme di archi colorati

**Domande**:
- Rosso: Catena ❌
- Blu: Spanning tree ❌
- Giallo: Percorso orientato aperto ❌
- Verde: **Ciclo ✓**

**Risposta Corretta**: Verde (ciclo)

**Spiegazione**: 
- Da 2 → 1 → 4 → 3 → 2
- È un percorso **chiuso** che ritorna al nodo di partenza
- Forma un ciclo

#### Domanda 2: Proprietà dei Tagli

`00:57:18`

**Domanda**: Se rimuoviamo gli archi di un taglio, cosa succede al grafo?

**Opzioni**:
- Rosso: Grafo partizionato in **esattamente** 2 componenti connesse ❌
- Blu: Grafo partizionato in **almeno** 2 componenti connesse ✓
- Giallo: Grafo rimane connesso ❌
- Verde: Non c'è percorso che attraversa il taglio ✓

**Risposte Corrette**: Blu e Verde

**Spiegazione**:
- Un taglio può separare il grafo in **2 o più** componenti (non necessariamente esattamente 2)
- Per definizione, rimuovendo un taglio, non esistono archi tra le componenti

#### Domanda 3: Matrice di Incidenza Nodo-Arco

`00:59:43 - 01:03:26`

**Rappresentazione**: Matrice dove:
- **Righe**: Nodi
- **Colonne**: Archi
- **Valori**:
  - `-1`: Coda dell'arco (tail)
  - `+1`: Testa dell'arco (head)
  - `0`: Nodo non incidente con l'arco

**Esempio**: Arco (i,j)

```
Nodo i: -1  (arco esce da i)
Nodo j: +1  (arco entra in j)
Altri:   0
```

**Moltiplicazione Matrice × Vettore**:

Quando moltiplichiamo la matrice di incidenza per un vettore x (variabili sugli archi), otteniamo:

```
(A · x)_i = Σ_{j∈FS(i)} x_ij - Σ_{j∈BS(i)} x_ji
```

Cioè: **Flusso uscente - Flusso entrante** nel nodo i

> **🔗 Connessione con Vincoli di Flusso**
> 
> Questa rappresentazione è alla base dei **vincoli di conservazione del flusso** nei problemi di flusso su rete!

#### Domanda 4: Backward Star

`01:03:26 - 01:04:00`

**Domanda**: Cos'è l'insieme BS(i)?

**Risposta Corretta**: L'insieme di tutti gli **archi entranti** nel nodo i

**Notazione**:
```
BS(i) = {(j,i) ∈ A : j ∈ N}
```

- **Backward**: Guarda "indietro", gli archi che arrivano
- **Star**: Stella di archi incidenti
- Si riferisce ad **archi**, non a nodi

#### Domanda 5: Forward Star

`01:04:00 - 01:04:37`

**Domanda**: Cos'è la forward star?

**Risposta Corretta**: L'insieme di tutti gli **archi uscenti** dal nodo i

**Notazione**:
```
FS(i) = {(i,j) ∈ A : j ∈ N}
```

- **Forward**: Guarda "avanti", gli archi che partono
- Duale della backward star

#### Domanda 6: Grafi Bipartiti

`01:05:07 - 01:09:00`

**Domanda**: Un grafo non orientato è bipartito se e solo se...

**Opzioni**:
- Non ci sono cicli con numero pari di archi ❌
- **Non ci sono cicli con numero dispari di archi ✓**
- Non ci sono cicli ❌
- Nessuna delle precedenti ❌

**Risposta Corretta**: Non ci sono cicli con numero dispari di archi

**Spiegazione**:

Un **grafo bipartito** ha nodi partizionati in due insiemi S e T tali che:
- Tutti gli archi vanno da S a T o da T a S
- **Nessun arco** connette nodi dello stesso insieme

```
S: ●     ●     ●
    \   / \   /
     \ /   \ /
T:    ●     ●     ●
```

Perché solo cicli pari?

- Partiamo da un nodo in S
- Attraversiamo un arco → arriviamo in T
- Attraversiamo un altro arco → torniamo in S
- Per tornare al nodo iniziale, dobbiamo alternare S↔T
- Questo richiede un **numero pari** di archi

Un ciclo dispari (3, 5, 7, ... archi) è **impossibile** in un grafo bipartito!

> **📊 Applicazioni dei Grafi Bipartiti**
> 
> - **Matching problems**: Assegnamento lavoratori-mansioni
> - **Recommendation systems**: Utenti-prodotti
> - **Network flow**: Sorgenti-destinazioni
> - **Scheduling**: Risorse-task

### 7.3 Tabella Bonus Point

`00:54:19 - 00:55:28`

Il professore ha pubblicato su **WeBeep** la tabella con i punti bonus assegnati finora.

**Richiesta agli studenti**:
- Controllare i propri punti bonus
- Segnalare eventuali errori o mancanze
- Chi è salito sul podio 2 volte nelle prime 3 attività ha avuto un punto cancellato (per dare opportunità a tutti)

---

## 📝 Note Finali e Prossimi Passi

### Completamento Notazione Grafi

`00:49:31`

Completeremo la notazione dei grafi con il CAHOOT nei prossimi minuti della lezione.

### Transizione ai Grafi Orientati

`01:09:00 - fine`

**Prossimo argomento**: Inizieremo il nostro viaggio sui **percorsi nei grafi orientati**.

**Contenuti previsti**:
1. **Nuove sfide** sui grafi orientati
2. Una **proprietà molto importante** alla base di tutti i problemi di ottimizzazione facili
3. Il segreto per capire se un problema di ottimizzazione è **facile o difficile**
4. Applicazioni del **percorso più breve** (shortest path) oltre le mappe
5. **Formulazioni alternative** del problema
6. **Almeno 3 algoritmi** diversi per il problema del percorso più breve
7. Come **selezionare il migliore** in base alle caratteristiche del grafo

### Prerequisiti per la Prossima Lezione

**Dalla lezione corrente**:
- Notazione di base dei grafi (nodi, archi, forward star, backward star)
- Concetto di percorso diretto
- Concetto di taglio

**Competenze generali**:
- Principi della modellazione
- Basi di analisi della complessità degli algoritmi

---

## 🎯 Concetti Chiave da Ricordare

### Problemi Multi-periodo
- Bilancio di inventario
- Gestione dell'incertezza
- Trade-off costo-rischio

### Minimum Spanning Tree
- n-1 archi per connettere n nodi
- Algoritmi polinomiali (Kruskal O(m log n), Prim O(m + n log n))
- Problema di separazione per vincoli esponenziali

### Traveling Salesman Problem
- n archi, grado 2 per ogni nodo
- Problema NP-hard
- Differenza cruciale con MST nonostante formulazioni simili

### Strutture Dati
- Union-Find per Kruskal
- Matrice di incidenza per vincoli di flusso
- Forward/Backward star per grafi orientati

### Grafi Bipartiti
- Nessun ciclo dispari
- Due partizioni di nodi
- Applicazioni in matching e assegnamento

---

## 📚 Risorse e Approfondimenti

### Software e Tool
- **Concord TSP**: Solver gratuito per smartphone
- Istanze TSP famose da risolvere

### Letture Consigliate
- Libro di narrativa sul TSP (italiano/tedesco)
- Storia del Gioco Dicosiano di Hamilton

### Esercizi Proposti
1. Estendere l'algoritmo di Kruskal per TSP
2. Confrontare formulazioni MST vs TSP
3. Sperimentare con grafi bipartiti

---

**Fine Lezione 7**

*Prossima lezione: Grafi Orientati e Percorsi più Brevi*
