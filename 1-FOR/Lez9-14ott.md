# Lezione 9 - 14 Ottobre: Shortest Path - Algoritmi Avanzati e Applicazioni

## 📑 Indice

1. [Modello Production Mix](#production-mix) `00:02:28 - 00:09:53`
   - [Formulazione del Problema](#formulazione-production)
   - [Insiemi, Parametri e Variabili](#sets-params-production)
   - [Vincoli di Capacità e Linking](#vincoli-production)
2. [Esercizio Satelliti: Soluzione](#esercizio-satelliti) `00:09:53 - 00:17:02`
   - [Modellazione come Grafo Completo](#grafo-satelliti)
   - [Definizione Costi e Riduzione a TSP](#riduzione-tsp-satelliti)
3. [Formulazione Alternativa Shortest Path](#formulazione-alternativa) `00:17:02 - 00:44:46`
   - [Forma Duale con Variabili sui Nodi](#forma-duale)
   - [Interpretazione Fisica: Corde e Tensione](#interpretazione-fisica)
   - [Relazione Primale-Duale](#primale-duale)
   - [Corrispondenza Labels-Variabili Duali](#labels-variabili)
4. [Project Scheduling con Grafo Aciclico](#project-scheduling) `00:44:46 - 01:14:34`
   - [Costruzione Grafo delle Precedenze](#grafo-precedenze)
   - [Rappresentazione Node-Based](#node-based)
   - [Formulazione Matematica e Duale](#formulazione-scheduling)
   - [Critical Path Method](#critical-path-method)
   - [Algoritmo Longest Path](#longest-path)
5. [Algoritmo di Dijkstra](#algoritmo-dijkstra) `01:14:34 - 01:31:46`
   - [Motivazione e Setup](#motivazione-dijkstra)
   - [Struttura dell'Algoritmo](#struttura-dijkstra)
   - [Dimostrazione Pratica con Corde](#demo-pratica)
   - [Pseudocodice Completo](#pseudocodice-dijkstra)
   - [Applet Interattivo](#applet-dijkstra)

---

## <a name="production-mix"></a>1. Modello Production Mix

### <a name="formulazione-production"></a>1.1 Formulazione del Problema

`00:02:28 - 00:04:08`

> **Contesto Reale: Produzione Multi-Prodotto**
> 
> Un'azienda manifatturiera deve decidere:
> - **Quanti prodotti** di ciascun tipo produrre
> - **Come allocare** le risorse (robot) tra i prodotti
> - **Considerare** i costi fissi di setup quando un robot viene dedicato a un prodotto
> 
> **Obiettivo**: Massimizzare il profitto (ricavi - costi setup - costi produzione)

#### Elementi del Problema

**N prodotti diversi**: Ogni prodotto genera un ricavo quando venduto

**M robot**: Ogni robot ha una capacità limitata di tempo

**Relazione prodotto-robot**: 
- Per produrre **1 unità** del prodotto i serve **a_ij unità di tempo** del robot j
- Ogni robot j ha disponibilità massima **b_j**

**Costi fissi di setup**: 
- Se il robot j viene assegnato al prodotto i, si paga **c_ij** (indipendente dalla quantità)
- Questo simula i costi di configurazione iniziale

### <a name="sets-params-production"></a>1.2 Insiemi, Parametri e Variabili

`00:04:08 - 00:08:35`

#### Insiemi

```
P = {1, 2, ..., n}  Prodotti (indice: i)
R = {1, 2, ..., m}  Robot (indice: j)
```

#### Parametri

```
a_ij  Tempo (minuti) richiesto dal robot j per produrre 1 unità di prodotto i
b_j   Disponibilità totale (minuti) del robot j
r_i   Ricavo per unità di prodotto i venduta
c_ij  Costo fisso setup quando robot j produce prodotto i
```

#### Variabili Decisionali

**Variabili Continue**:
```
X_ij ≥ 0  ∀i ∈ P, j ∈ R
```
- X_ij = Quantità di prodotto i prodotta con robot j

**Variabili Binarie** (per i costi fissi):
```
Y_ij ∈ {0,1}  ∀i ∈ P, j ∈ R
```
- Y_ij = 1 se il robot j produce (anche solo una unità di) prodotto i
- Y_ij = 0 altrimenti

### <a name="vincoli-production"></a>1.3 Vincoli di Capacità e Linking

`00:08:35 - 00:09:53`

#### Funzione Obiettivo

```
max Σ_{i∈P} r_i · (Σ_{j∈R} X_ij) - Σ_{i∈P} Σ_{j∈R} c_ij · Y_ij
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Ricavi totali                  Costi setup totali
```

**Interpretazione**:
- Primo termine: Ricavi = ricavo unitario × quantità totale prodotta di i (su tutti i robot)
- Secondo termine: Costi fissi totali di setup

#### Vincoli di Capacità

```
Σ_{i∈P} a_ij · X_ij ≤ b_j    ∀j ∈ R
```

**Significato**: 
- Per ogni robot j, la somma dei tempi di produzione per tutti i prodotti non può superare b_j
- Risorsa limitata: tempo del robot

#### Vincoli di Linking

```
X_ij ≤ M · Y_ij    ∀i ∈ P, j ∈ R
```

**Dove**: M è un "big number" (upper bound su X_ij, ad es. M = b_j / a_ij)

**Logica**:
- Se Y_ij = 0 → X_ij ≤ 0 → X_ij = 0 (non produciamo i con j)
- Se Y_ij = 1 → X_ij ≤ M (possiamo produrre fino a M unità)

**Effetto**: Paghiamo c_ij solo se effettivamente produciamo i con j!

> **💡 Tecnica Generale: Big-M Constraints**
> 
> I vincoli di tipo `x ≤ M · y` dove y ∈ {0,1} sono molto comuni in programmazione lineare intera per:
> - Attivare/disattivare vincoli
> - Collegare decisioni binarie e continue
> - Modellare costi fissi

#### Formulazione Completa

```
max Σ_{i∈P} r_i · (Σ_{j∈R} X_ij) - Σ_{i∈P} Σ_{j∈R} c_ij · Y_ij

s.t.:
    Σ_{i∈P} a_ij · X_ij ≤ b_j    ∀j ∈ R     (capacità robot)
    X_ij ≤ M · Y_ij               ∀i,j       (linking setup)
    X_ij ≥ 0                      ∀i,j
    Y_ij ∈ {0,1}                  ∀i,j
```

---

## <a name="esercizio-satelliti"></a>2. Esercizio Satelliti: Soluzione

### <a name="grafo-satelliti"></a>2.1 Modellazione come Grafo Completo

`00:09:53 - 00:14:12`

#### Richiamo del Problema

- **n richieste** di foto satellitari da diverse posizioni
- **Processing time** p_j per ogni foto j
- **Setup time** s_ij per orientare la camera da posizione i a posizione j
- **Obiettivo**: Sequenza che minimizza tempo totale

#### Costruzione del Grafo

**Nodi**: 
```
N = {0, 1, 2, ..., n}
```
- Nodo 0: Posizione iniziale del satellite
- Nodi 1...n: Richieste di foto

**Archi**:
```
A = {(i,j) : i,j ∈ N, i ≠ j}
```
- Grafo **completo**: esiste un arco tra ogni coppia di nodi
- Simbolo: K_{n+1} (complete graph con n+1 nodi)

> **📐 Definizione: Grafo Completo**
> 
> Un grafo orientato è **completo** se contiene tutti i possibili archi tra ogni coppia di nodi distinti.
> 
> Numero di archi in K_n: |A| = n(n-1)

### <a name="riduzione-tsp-satelliti"></a>2.2 Definizione Costi e Riduzione a TSP

`00:12:32 - 00:17:02`

#### Assegnazione Costi agli Archi

```
c_ij = s_ij + p_j    ∀(i,j) ∈ A
```

**Con l'assunzione**: p_0 = 0 (nessun processing time nella posizione iniziale)

**Interpretazione**:
- s_ij: Tempo per orientare da i a j
- p_j: Tempo per scattare la foto j una volta posizionati

#### Perché questa Definizione?

Se troviamo un **ciclo hamiltoniano** (tour che visita ogni nodo esattamente una volta):

```
Tour: 0 → i_1 → i_2 → ... → i_n → 0

Costo totale = c_{0,i_1} + c_{i_1,i_2} + ... + c_{i_n,0}
             = (s_{0,i_1} + p_{i_1}) + (s_{i_1,i_2} + p_{i_2}) + ... + (s_{i_n,0} + p_0)
             = Σ s + Σ p_j    (ogni p_j compare esattamente una volta)
```

#### Osservazione Cruciale

`00:14:40 - 00:15:40`

Il termine **Σ p_j** è **costante** e **indipendente dalla sequenza**!

Quindi minimizzare:
```
Σ s + Σ p_j
```

È equivalente a minimizzare:
```
Σ s_ij    (solo i tempi di setup)
```

**Implicazione**: Possiamo **omettere** i p_j dai costi e usare:
```
c_ij = s_ij    ∀(i,j) ∈ A
```

Il tour ottimale rimane lo stesso!

> **🎯 Riduzione a TSP**
> 
> Il problema del sequenziamento di foto satellitari si riduce esattamente a un **Traveling Salesman Problem** (TSP) con:
> - Nodi = posizioni (inclusa quella iniziale)
> - Costi = tempi di setup tra posizioni
> - Soluzione = ciclo hamiltoniano di costo minimo

#### Completezza della Soluzione

`00:15:40 - 00:17:02`

**Domanda**: È sufficiente fornire solo il grafo?

**Risposta**: **NO**, se la richiesta è "formulare come grafo".

Una formulazione completa richiede:
1. ✅ Definizione dell'insieme dei nodi N
2. ✅ Definizione dell'insieme degli archi A
3. ✅ Definizione dei costi c_ij
4. ❓ Formulazione matematica (variabili + vincoli) **solo se richiesta**

Nel nostro caso, descrivere il TSP come problema di ottimizzazione sarebbe:
```
min Σ_{(i,j)∈A} c_ij · x_ij

s.t.: Vincoli di ciclo hamiltoniano
```

Ma questo non era richiesto dall'esercizio!

---

## <a name="formulazione-alternativa"></a>3. Formulazione Alternativa Shortest Path

### <a name="forma-duale"></a>3.1 Forma Duale con Variabili sui Nodi

`00:17:02 - 00:21:45`

Ricordiamo la formulazione classica (primale) dello shortest path:

#### Formulazione Primale (con variabili sugli archi)

```
min Σ_{(i,j)∈A} c_ij · X_ij

s.t.:
    Σ_{j:(s,j)∈FS(s)} X_sj - Σ_{i:(i,s)∈BS(s)} X_is = 1      (sorgente)
    
    Σ_{j:(t,j)∈FS(t)} X_tj - Σ_{i:(i,t)∈BS(t)} X_it = -1     (destinazione)
    
    Σ_{j:(i,j)∈FS(i)} X_ij - Σ_{j:(j,i)∈BS(i)} X_ji = 0       ∀i ∈ N\{s,t}
    
    X_ij ≥ 0                                                   ∀(i,j) ∈ A
```

**Caratteristiche**:
- **m variabili** (una per arco)
- **n vincoli** (uno per nodo)
- Coefficienti: matrice di incidenza nodo-arco

#### Formulazione Duale (con variabili sui nodi)

`00:21:45 - 00:24:08`

Introduciamo variabili **π_i** per ogni nodo:

```
max π_t - π_s

s.t.:
    π_j - π_i ≤ c_ij    ∀(i,j) ∈ A
    π_i ∈ ℝ             ∀i ∈ N
```

**Caratteristiche**:
- **n variabili** (una per nodo)
- **m vincoli** (uno per arco)
- Ruoli di nodi e archi sono **invertiti** rispetto al primale!

> **🔄 Dualità Primale-Duale**
> 
> | Aspetto | Primale | Duale |
> |---------|---------|-------|
> | **Variabili** | m (archi) | n (nodi) |
> | **Vincoli** | n (nodi) | m (archi) |
> | **Obiettivo** | min (costi archi) | max (differenza livelli) |
> | **Coefficienti RHS** | ±1, 0 | c_ij |
> | **Coefficienti obj** | c_ij | ±1 |
> 
> La **matrice dei coefficienti** del duale è la **trasposta** di quella del primale!

### <a name="interpretazione-fisica"></a>3.2 Interpretazione Fisica: Corde e Tensione

`00:24:08 - 00:39:03`

#### Metafora con le Corde

`00:24:08 - 00:34:30`

Immaginiamo il grafo fatto di:
- **Nodi** = clip/mollette
- **Archi** = corde di lunghezza c_ij

**Setup**:
1. Fissiamo il nodo s al pavimento (π_s = 0)
2. Teniamo il nodo t in mano
3. Tiriamo verso l'alto t cercando di **massimizzare π_t**

**Vincoli fisici**: 
```
π_j - π_i ≤ c_ij
```

Significato: La differenza di altezza tra i e j non può superare la lunghezza della corda!

**Se tiriamo troppo**: Le corde si spezzano → violiamo i vincoli

#### Dimostrazione Pratica

`00:25:33 - 00:34:00`

Con il grafo della scorsa lezione (nodi A, B, C, D, E, F):

1. **Fissiamo A al pavimento**: π_A = 0
2. **Tiriamo F verso l'alto**: Massimizziamo π_F - π_A
3. **Le corde si tendono**: 
   - Alcune diventano **tese** (vincolo attivo: π_j - π_i = c_ij)
   - Altre restano **lente** (vincolo slack: π_j - π_i < c_ij)
4. **Percorso più breve**: Gli archi tesi formano il percorso ottimo!

**Osservazione Chiave**: 
- Gli archi **tesi** → X_ij = 1 nella soluzione primale
- Gli archi **lenti** → X_ij = 0 nella soluzione primale

> **💡 Interpretazione Fisica**
> 
> - **Problema primale** (minimizzazione): Trovare il percorso di costo minimo
> - **Problema duale** (massimizzazione): Massimizzare la "tensione" del grafo mantenendo intatte le corde
> 
> **Soluzione ottima**: Il valore obiettivo è lo stesso!

#### Indipendenza dal Livello di Partenza

`00:38:30 - 00:39:03`

Le variabili π_i non sono uniche! Possiamo:
- Fissare π_s = 0 (convenzione)
- Oppure π_s = 100, π_s = -50, ...

**Libertà**: 1 grado di libertà nella scelta del livello di riferimento

**Invariante**: Le **differenze** π_j - π_i (che compaiono nei vincoli e nell'obiettivo)

Quindi il problema ha:
- n variabili
- Ma solo n-1 "gradi di libertà" effettivi

### <a name="primale-duale"></a>3.3 Relazione Primale-Duale

`00:39:34 - 00:44:46`

#### Corrispondenza dei Coefficienti

**Coefficienti Obiettivo ↔ Termini Noti**:

Primale:
```
min Σ c_ij · X_ij
```
I coefficienti c_ij appaiono nell'obiettivo.

Duale:
```
π_j - π_i ≤ c_ij
```
Gli stessi c_ij appaiono nei termini noti (RHS) dei vincoli!

**Termini Noti ↔ Coefficienti Obiettivo**:

Primale (RHS):
```
... = 1   (per s)
... = -1  (per t)
... = 0   (altri)
```

Duale (obiettivo):
```
max π_t - π_s = max (+1)·π_t + (-1)·π_s
```

Gli stessi +1, -1 appaiono come coefficienti!

#### Matrice Incidenza Trasposta

`00:40:41 - 00:41:48`

**Matrice nodo-arco** (Primale):
- Righe: nodi
- Colonne: archi
- Entrata (i, (j,k)): 
  - +1 se i = k (arco entra in i)
  - -1 se i = j (arco esce da i)
  - 0 altrimenti

**Matrice duale**:
- Righe: archi
- Colonne: nodi
- È la **trasposta** della matrice primale!

> **🔗 Teorema di Dualità in PL**
> 
> Per ogni problema di Programmazione Lineare (LP) primale:
> ```
> min c^T x
> s.t. Ax = b, x ≥ 0
> ```
> 
> Esiste un problema **duale**:
> ```
> max b^T π
> s.t. A^T π ≤ c
> ```
> 
> Proprietà:
> - **Weak Duality**: Valore duale ≤ Valore primale (per min)
> - **Strong Duality**: All'ottimo, i valori coincidono
> - **Complementary Slackness**: x_j > 0 ⟹ vincolo duale j tight

### <a name="labels-variabili"></a>3.4 Corrispondenza Labels-Variabili Duali

`00:42:09 - 00:44:46`

#### Le Etichette d_i sono le π_i!

Nell'algoritmo per grafi aciclici, calcoliamo:

```
d_s = 0
d_i = min{d_j + c_ji : (j,i) ∈ BS(i)}
```

**Claim**: Le etichette d_i sono esattamente le variabili duali π_i!

#### Dimostrazione Euristica

**Passo base**: d_s = 0 corrisponde a π_s = 0 (fissato)

**Passo induttivo**: Supponiamo d_j siano già le π_j per tutti i predecessori di i.

Per la regola dell'algoritmo:
```
d_i = min{d_j + c_ji}
```

Questo significa che stiamo cercando il massimo "livello" compatibile con i vincoli:
```
π_i ≤ π_j + c_ji    per ogni j predecessore
```

Riscrivendo:
```
π_i - π_j ≤ c_ji
```

Ma questo è esattamente il vincolo duale per l'arco (j,i)!

**Conclusione**: L'algoritmo shortest path sta **costruendo la soluzione duale ottima**!

> **🎓 Insight Profondo**
> 
> Gli algoritmi di shortest path operano **implicitamente** sul problema duale:
> - Calcolano le variabili duali π_i (etichette d_i)
> - Identificano i vincoli **tight** (archi nel percorso)
> - Soddisfano le condizioni di **complementary slackness**
> 
> Questo spiega perché le etichette hanno significato di "distanza da s"!

---

## <a name="project-scheduling"></a>4. Project Scheduling con Grafo Aciclico

### <a name="grafo-precedenze"></a>4.1 Costruzione Grafo delle Precedenze

`00:44:46 - 00:53:23`

#### Contesto del Problema

> **Scenario: Costruzione di una Casa**
> 
> Un progetto complesso composto da **attività** con:
> - **Durate** fisse p_i
> - **Precedenze**: alcune attività devono finire prima che altre inizino
> 
> **Obiettivo**: Pianificare l'esecuzione minimizzando il **completion time** totale.

#### Attività e Precedenze - Esempio

**Attività** (lettera = nome, numero blu = durata):
```
A (7): Costruire muri
B (2): Preparare fondamenta  
C (15): Scavi
E (10): Installare impianto elettrico
D (8): Posare tubi idraulici
G (5): Costruire tetto
F (2): Installare infissi
H (8): Verniciare esterno
I (2): Verniciare interno
J (3): Ispezione finale
```

**Precedenze** (frecce nell'esempio):
```
E precede D e G    (impianto elettrico prima di tubi e tetto)
A precede E        (muri prima di elettricità)
B precede E        (fondamenta prima di elettricità)
D precede F        (tubi prima di infissi)
G precede F e H    (tetto prima di infissi e verniciatura esterna)
C precede I        (scavi prima di verniciatura interna)
F precede I        (infissi prima di verniciatura interna)
I precede J        (verniciatura interna prima di ispezione)
H precede J        (verniciatura esterna prima di ispezione)
```

#### Costruzione del Grafo

`00:47:00 - 00:53:23`

**Passo 1**: Nodo per ogni attività
```
Nodi: {A, B, C, E, D, G, F, H, I, J}
```

**Passo 2**: Arco per ogni precedenza
```
Archi: {(A,E), (B,E), (E,D), (E,G), (D,F), (G,F), (G,H), (C,I), (F,I), (I,J), (H,J)}
```

**Passo 3**: Aggiungere nodi dummy per **Beginning** e **End**
```
Beginning → A, B, C    (attività senza predecessori)
H, J → End             (attività senza successori)
```

**Grafo Finale**:
```
                    Beginning
                    ↙   ↓   ↘
                   A    B    C
                    ↘   ↓   ↙
                      E
                    ↙   ↘
                   D     G
                    ↘   ↙ ↘
                      F    H
                      ↓    ↓
                      I    ↓
                      ↓    ↓
                      J    ↓
                       ↘  ↙
                        End
```

> **📐 Proprietà: Grafo Aciclico (DAG)**
> 
> Il grafo delle precedenze di un progetto ben definito è **sempre aciclico**.
> 
> Perché? Se esistesse un ciclo A → B → C → A, avremmo:
> - A deve precedere B
> - B deve precedere C
> - C deve precedere A → **Contraddizione!**

### <a name="node-based"></a>4.2 Rappresentazione Node-Based

`00:51:13 - 00:56:16`

#### Dove Mettere le Durate?

**Problema**: Le durate p_i sono attributi delle **attività** (nodi), ma nei problemi su grafi i costi sono tipicamente sugli **archi**.

**Soluzione**: **Spostare** le durate sui **archi uscenti**.

#### Regola di Conversione

Per ogni attività i con durata p_i:
```
Per ogni arco (i, j):  c_ij ← p_i
```

**Significato**: L'arco (i,j) rappresenta "attività i deve finire (dopo p_i giorni) prima che j inizi".

#### Esempio di Assegnazione Costi

```
Attività E (durata 10):
  Archi uscenti: (E,D), (E,G)
  Costi: c_{E,D} = 10, c_{E,G} = 10

Attività A (durata 7):
  Arco uscente: (A,E)
  Costo: c_{A,E} = 7

Archi da Beginning:
  (Beginning, A), (Beginning, B), (Beginning, C)
  Costi: Tutti 0 (attività dummy senza durata)
```

**Grafo con Costi**:
```
               Begin
              0↙ 0↓ 0↘
               A   B   C
              7↘ 2↓ 15↙
                  E
               10↙ 10↘
                D     G
               8↘   5↙ 5↘
                  F     H
                 2↓     8↓
                  I     ↓
                 2↓     ↓
                  J     ↓
                 3↘    ↙
                   End
```

### <a name="formulazione-scheduling"></a>4.3 Formulazione Matematica e Duale

`00:56:16 - 01:04:00`

#### Formulazione Naturale (con Starting Times)

**Variabili**: t_i = tempo di inizio dell'attività i

**Obiettivo**: Minimizzare la durata del progetto
```
min t_{End} - t_{Begin}
```

**Vincoli di Precedenza**: Se i precede j:
```
t_i + p_i ≤ t_j
```

Riscrivendo:
```
t_j - t_i ≥ p_i
```

**Formulazione Completa**:
```
min t_{End} - t_{Begin}

s.t.:
    t_j - t_i ≥ d_ij    ∀(i,j) ∈ A (precedenze)
    t_i ≥ 0             ∀i ∈ N
```

Dove d_ij = durata dell'attività i per l'arco (i,j).

#### Confronto con Shortest Path Duale

`01:00:13 - 01:04:00`

**Shortest Path Duale**:
```
max π_t - π_s
s.t. π_j - π_i ≤ c_ij
```

**Project Scheduling**:
```
min t_{End} - t_{Begin}
s.t. t_j - t_i ≥ d_ij
```

**Differenze**:
- Max ↔ Min
- ≤ ↔ ≥

**Trasformazione**: Se definiamo π_i = -t_i:
```
min t_{End} - t_{Begin} = min -(-t_{End} + t_{Begin})
                        = min -(π_{Begin} - π_{End})
                        = max π_{End} - π_{Begin}

t_j - t_i ≥ d_ij
-π_j + π_i ≥ d_ij
π_i - π_j ≥ d_ij    (oppure π_j - π_i ≤ -d_ij)
```

> **🔄 Dualità Scheduling-Shortest Path**
> 
> Il problema di project scheduling è **duele** al problema di:
> - Trovare il **longest path** (percorso più lungo)
> - In un grafo con costi = durate
> 
> **Interpretazione fisica**: Stiamo "schiacciando" il grafo invece che "tirarlo"!

### <a name="critical-path-method"></a>4.4 Critical Path Method

`01:07:08 - 01:14:05`

#### Algoritmo per Longest Path su DAG

`01:07:08 - 01:09:27`

Modifica dell'algoritmo shortest path per calcolare longest path:

**Cambiamenti necessari**:
1. Inizializzazione: d_i ← **-∞** (invece di +∞)
2. Regola di aggiornamento: **max** invece di min
   ```
   d_j ← max{d_j, d_i + c_ij}
   ```

**Pseudocodice**:
```
LONGEST_PATH_DAG(G, c, s):
    1. Calcola ordinamento topologico di G
    
    2. Inizializzazione:
       d_i ← -∞    ∀i ∈ N
       d_s ← 0
       p_i ← 0     ∀i ∈ N
       p_s ← s
    
    3. FOR i = 1 TO n (ordine topologico):
        4. FOR each (i,j) ∈ FS(i):
            5. IF d_i + c_ij > d_j:
                6. d_j ← d_i + c_ij
                7. p_j ← i
    
    8. RETURN d, p
```

**Complessità**: O(n + m) = O(m) (identica a shortest path)

#### Esempio di Esecuzione

`01:09:27 - 01:13:02`

**Ordinamento Topologico**: Begin, A, B, C, E, G, D, F, H, I, J, End

**Inizializzazione**:
```
d = [0, -∞, -∞, -∞, -∞, -∞, -∞, -∞, -∞, -∞, -∞, -∞]
     B   A   B   C   E   G   D   F   H   I   J  End
```

**i = Begin**: Visita FS(Begin) = {(B,A), (B,B), (B,C)}
```
d_A = max(-∞, 0+0) = 0
d_B = max(-∞, 0+0) = 0
d_C = max(-∞, 0+0) = 0
```

**i = A**: Visita FS(A) = {(A,E)}
```
d_E = max(-∞, 0+7) = 7
```

**i = B**: Visita FS(B) = {(B,E)}
```
d_E = max(7, 0+2) = 7    (no change)
```

**i = C**: Visita FS(C) = {(C,I)}
```
d_I = max(-∞, 0+15) = 15
```

**i = E**: Visita FS(E) = {(E,D), (E,G)}
```
d_D = max(-∞, 7+10) = 17
d_G = max(-∞, 7+10) = 17
```

**i = G**: Visita FS(G) = {(G,F), (G,H)}
```
d_F = max(-∞, 17+5) = 22
d_H = max(-∞, 17+5) = 22
```

**i = D**: Visita FS(D) = {(D,F)}
```
d_F = max(22, 17+8) = max(22, 25) = 25    (aggiornato!)
```

**i = F**: Visita FS(F) = {(F,I)}
```
d_I = max(15, 25+2) = max(15, 27) = 27    (aggiornato!)
```

**i = H**: Visita FS(H) = {(H,End)}
```
d_{End} = max(-∞, 22+8) = 30
```

**i = I**: Visita FS(I) = {(I,J)}
```
d_J = max(-∞, 27+2) = 29
```

**i = J**: Visita FS(J) = {(J,End)}
```
d_{End} = max(30, 29+3) = max(30, 32) = 32    (aggiornato!)
```

**Soluzione Ottima**:
```
Completion Time = d_{End} = 32 giorni
```

#### Ricostruzione del Percorso Critico

`01:11:00 - 01:13:02`

**Percorso**: Begin → B → E → D → F → I → J → End

Verificando:
```
0 + 0 = 0 (B)
0 + 2 = 2 ... no, era 0+7 da A!
```

Correzione: Begin → A → E → D → F → I → J → End

```
0 + 0 = 0 (A)
0 + 7 = 7 (E)
7 + 10 = 17 (D)
17 + 8 = 25 (F)
25 + 2 = 27 (I)
27 + 2 = 29 (J)
29 + 3 = 32 (End) ✓
```

> **📍 Critical Path**
> 
> **Attività sul percorso critico**: A, E, D, F, I, J
> 
> **Proprietà**: Ritardare **qualsiasi** di queste attività ritarda l'intero progetto!

#### Slack Time

`01:12:19 - 01:14:05`

Per attività **non** sul percorso critico (es. H):

**Earliest Start**: d_H = 22
**Latest Start** (senza ritardare progetto): 
```
d_{End} - p_H - (tempo da H a End) = 32 - 8 - 0 = 24
```

**Slack**: 24 - 22 = **2 giorni**

Possiamo ritardare H di max 2 giorni senza impatto sul progetto!

> **⏱️ Critical Path Method (CPM)**
> 
> Tecnica di project management che:
> 1. Identifica il **percorso critico** (longest path)
> 2. Calcola **slack times** per attività non critiche
> 3. Permette di **allocare risorse** prioritizzando attività critiche
> 
> Usato in grandi progetti (costruzioni, software, eventi) fin dagli anni '50!

---

## <a name="algoritmo-dijkstra"></a>5. Algoritmo di Dijkstra

### <a name="motivazione-dijkstra"></a>5.1 Motivazione e Setup

`01:14:34 - 01:19:46`

#### Limitazioni dell'Algoritmo per DAG

L'algoritmo visto funziona benissimo per grafi aciclici, ma:

**Cosa succede se il grafo ha cicli?**
- Non possiamo fare ordinamento topologico!
- Non esiste una "numerazione che va avanti"

**Idea**: Cerchiamo di **simulare** l'ordinamento topologico anche con cicli!

#### Grafo con Cicli - Esempio

`01:17:28 - 01:19:46`

```
      5        3
  1 ──→ 3 ──→ 6
  ↓      ↓    ↓
  2      4    1
  ↓      ↓
  4 ──→ 5 ──→ 6
     1    7
```

Cicli presenti:
- 1 → 3 → 6 → 1
- 1 → 3 → 4 → 1
- ...

**Problema**: Non possiamo dire "3 viene prima di 6" perché c'è un arco all'indietro!

#### Struttura dell'Algoritmo con Cicli

`01:19:07 - 01:19:46`

**Idea Chiave**: Usare una coda Q come nel graph visit, ma:
- Selezionare sempre il nodo con **etichetta minima**
- Garantire che l'etichetta non possa più migliorare dopo l'estrazione

**Assunzione Cruciale**: **Costi non-negativi**
```
c_ij ≥ 0    ∀(i,j) ∈ A
```

Senza questa assunzione, l'algoritmo può **non terminare** o dare risultati errati!

### <a name="struttura-dijkstra"></a>5.2 Struttura dell'Algoritmo

`01:19:46 - 01:22:25`

#### Componenti Principali

**1. Etichette**: d_i = lunghezza tentativa del percorso più breve da s a i

**2. Predecessori**: p_i = nodo da cui abbiamo raggiunto i

**3. Coda Q**: Insieme di nodi **candidati** per l'esplorazione

**4. Regola di Selezione**: Estrai sempre il nodo con **d minima** da Q

#### Differenze con Graph Visit

| Aspetto | Graph Visit | Dijkstra |
|---------|-------------|----------|
| **Estrazione da Q** | Qualsiasi ordine | Minima etichetta |
| **Costi** | Non considerati | Fondamentali |
| **Re-inserimenti** | Impossibili | Possibili (se miglioriamo) |
| **Garanzia** | Connettività | Shortest path |

### <a name="demo-pratica"></a>5.3 Dimostrazione Pratica con Corde

`00:24:00 - 00:34:30`

> **🧪 Esperimento Fisico**
> 
> Usando il grafo fatto di corde e clip:
> 
> 1. **Fissiamo A** (nodo sorgente) al pavimento
> 2. **Teniamo F** (nodo destinazione) in mano
> 3. **Tiriamo verso l'alto** cercando di massimizzare la distanza

#### Procedura Step-by-Step

`00:25:33 - 00:34:00`

**Setup Iniziale**:
- Nodo A sul pavimento (livello 0)
- Tutti gli altri nodi "al soffitto" (livello +∞)

**Step 1**: Rilasciamo i nodi connessi direttamente ad A
- Nodi B e C cadono alle loro "altezze naturali" rispetto ad A
- B a livello 5 (lunghezza corda A-B)
- C a livello 5 (lunghezza corda A-C)

**Step 2**: Selezioniamo il nodo con altezza minima (B o C indifferente)
- Scegliamo B (livello 5)
- **Fissiamo** B a livello 5 (etichetta definitiva!)

**Step 3**: Rilasciamo i nodi connessi a B
- D scende a livello 5 + 7 = 12
- F scende a livello 5 + 10 = 15

**Step 4**: Consideriamo anche C (livello 5)
- **Fissiamo** C a livello 5
- Rilasciamo nodi connessi:
  - D scende a livello 5 + 3 = 8 (migliore di 12!)
  - E scende a livello 5 + 6 = 11

**Step 5**: Minimo ora è D (livello 8)
- **Fissiamo** D
- Rilasciamo:
  - F scende a 8 + 3 = 11 (migliore di 15!)

**Step 6**: Minimo è E (livello 11)
- **Fissiamo** E
- Rilasciamo:
  - F ora a 11 + 3 = 14 (peggio di 11, no change)

**Step 7**: Minimo è F (livello 11)
- **Algoritmo terminato!**

**Soluzione**:
```
Shortest path A → F: A → C → D → F
Costo: 5 + 3 + 3 = 11
```

#### Osservazioni dalla Dimostrazione

`00:32:44 - 00:34:30`

**Archi Tesi**: Quelli nel percorso più breve
- A → C (teso)
- C → D (teso)
- D → F (teso)

**Archi Lenti**: Quelli non nel percorso
- A → B (lento)
- B → D (lento)
- B → F (lento)
- E → F (lento)

**Domanda**: Se tiriamo il grafo, perché otteniamo il percorso più **breve** (minimizzazione) quando stiamo **massimizzando** la distanza?

**Risposta**: I **vincoli** sono le lunghezze fisiche delle corde!
- Non possiamo estendere più di c_ij
- Quindi stiamo risolvendo: max π_F s.t. π_j - π_i ≤ c_ij (problema duale)

### <a name="pseudocodice-dijkstra"></a>5.4 Pseudocodice Completo

`01:22:25 - 01:26:48`

#### Algoritmo di Dijkstra

```
DIJKSTRA(G, c, s):
    Input: Grafo orientato G = (N, A), costi non-negativi c, sorgente s
    Output: Vettori d (etichette) e p (predecessori)
    
    Prerequisito: c_ij ≥ 0  ∀(i,j) ∈ A
    
    1. Inizializzazione:
       d_i ← +∞       ∀i ∈ N \ {s}
       d_s ← 0
       p_i ← 0        ∀i ∈ N
       p_s ← s
       Q ← {s}
    
    2. REPEAT UNTIL Q = ∅:
        
        3. Estrai da Q il nodo i con d_i minima     ← CHIAVE!
        
        4. FOR each (i,j) ∈ FS(i):
            
            5. IF d_i + c_ij < d_j:                 // Percorso migliore?
                
                6. d_j ← d_i + c_ij                 // Aggiorna etichetta
                7. p_j ← i                          // Aggiorna predecessore
                
                8. IF j ∉ Q:
                    9. Q ← Q ∪ {j}                  // Inserisci in coda
    
    10. RETURN d, p
```

#### Dettagli Chiave

**Riga 3**: Selezione del **minimo**
- Questa è la differenza fondamentale rispetto al graph visit
- Garantisce che d_i è definitivo quando estraiamo i

**Righe 5-7**: Aggiornamento etichetta
- "Relaxation" dell'arco (i,j)
- Se troviamo percorso migliore verso j passando per i, aggiorniamo

**Righe 8-9**: Re-inserimento in Q
- Se j era già stato estratto, NON viene reinserito (con costi ≥ 0)
- Se j è nuovo o in Q, potrebbe essere aggiornato più volte

#### Invariante dell'Algoritmo

> **🔒 Invariante di Dijkstra**
> 
> Una volta che un nodo i viene **estratto** da Q:
> - La sua etichetta d_i è **definitiva**
> - Nessun altro percorso può migliorarla
> - i **non verrà mai reinserito** in Q
> 
> **Perché?** Con costi ≥ 0, qualsiasi percorso futuro che passa per nodi ancora in Q sarà ≥ d_i.

#### Complessità (Anteprima)

`01:26:48 - 01:27:42`

Vedremo nella prossima lezione:
- **Con lista non ordinata**: O(n²)
- **Con binary heap**: O((n + m) log n) = O(m log n)
- Dipende dall'implementazione di Q!

### <a name="applet-dijkstra"></a>5.5 Applet Interattivo

`01:27:32 - 01:31:46`

#### Dimostrazione Online

Link fornito su Webeep per applet interattivo.

**Funzionalità**:
- Grafo visualizzato con nodi e archi
- Etichette mostrate dentro i nodi
- Nodi colorati per stato:
  - **Bianco**: Non ancora visitato (d = +∞)
  - **Giallo**: In coda Q (tentativo)
  - **Rosso**: Estratto da Q (definitivo)
- Archi colorati:
  - **Verde**: Nel shortest path tree
  - **Nero**: Non usati

#### Esecuzione Step-by-Step

`01:28:06 - 01:31:46`

**Grafo dell'Applet**:
```
Nodi: A, B, C, D, E
Archi con costi visualizzati
```

**Step 1**: Inizializza
```
d_A = 0 (rosso)
Altri nodi: d = ∞ (bianchi)
```

**Step 2**: Visita FS(A)
```
Aggiorna B, C, D (gialli)
```

**Step 3**: Estrai nodo con d minima (es. B)
```
B diventa rosso
Visita FS(B), aggiorna etichette
```

**Step 4-N**: Continua fino a coda vuota
```
Ogni nodo diventa rosso quando estratto
Archi verdi formano l'albero shortest path
```

**Risultato Finale**:
- Tutti i nodi rossi
- Albero verde mostra tutti i shortest paths da A
- Etichette d_i sono le distanze minime

> **🎮 Esercizio Consigliato**
> 
> Usare l'applet per:
> 1. Eseguire Dijkstra passo-passo su vari grafi
> 2. Osservare come cambiano le etichette
> 3. Verificare che i nodi rossi non cambiano più
> 4. Ricostruire i percorsi usando i predecessori (archi verdi)

---

## 📝 Riepilogo e Prossima Lezione

### Concetti Chiave Lezione 9

`01:31:46 - fine`

**Modellazione**:
- Production mix: Linking constraints per costi fissi
- Satelliti: TSP con processing + setup times

**Dualità**:
- Shortest path: Primale (variabili archi) ↔ Duale (variabili nodi)
- Interpretazione fisica: Corde tese vs lente
- Etichette = variabili duali

**Project Scheduling**:
- DAG con precedenze → Longest path
- Critical Path Method (CPM)
- Slack times per attività non critiche

**Dijkstra**:
- Estensione a grafi con cicli
- Selezione nodo con etichetta minima
- Dimostrazione con modello fisico

### Prossima Lezione

**Argomenti**:
1. **Analisi di complessità** di Dijkstra
   - Implementazione con lista non ordinata: O(n²)
   - Implementazione con binary heap: O(m log n)
2. **Grafi con costi negativi**
   - Problemi di Dijkstra con costi < 0
   - Algoritmo di Bellman-Ford (label-correcting)
3. **Esercizi applicativi**
   - Route 66 (Obama campaign)
   - Car rental con refueling

### Materiali

> **📚 Risorse su Webeep**:
> - Esercizi sui grafi (da leggere prima della lezione)
> - Link applet Dijkstra
> - Tutorial Dynamic Programming (opzionale, utile per esame)
> - Lecture notes sezione "Shortest Path"

> **🎒 Portare a Lezione**:
> - Materiale per grafi (corde, clip)
> - Esercizi letti e preparati

---

**Fine Lezione 9**

*Prossima lezione: Complessità Dijkstra, Bellman-Ford, Applicazioni Dynamic Programming*
