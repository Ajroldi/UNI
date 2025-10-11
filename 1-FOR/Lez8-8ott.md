# Lezione 8 - 8 Ottobre: Algoritmi su Grafi e Percorsi più Brevi

## 📑 Indice

1. [Introduzione](#introduzione) `00:02:00`
2. [Problema OD Matrix Matching](#od-matrix-matching) `00:02:00 - 00:21:23`
   - [Contesto e Formulazione](#contesto-od)
   - [Grafo Bipartito e Variabili](#grafo-bipartito-od)
   - [Linearizzazione Funzione Bottleneck](#linearizzazione-bottleneck)
3. [Esercizio: Sequenziamento Immagini Satellitari](#esercizio-satelliti) `00:17:34 - 00:21:23`
4. [Algoritmo di Visita del Grafo - Ripasso](#algoritmo-visita) `00:22:00 - 00:38:00`
   - [Strutture Dati](#strutture-dati-visita)
   - [Pseudocodice e Funzionamento](#pseudocodice-visita)
   - [Esempio di Esecuzione](#esempio-esecuzione-visita)
   - [Analisi della Complessità](#complessita-visita)
   - [Proprietà Percorso vs Taglio](#proprieta-percorso-taglio)
5. [Formulazione Shortest Path](#formulazione-shortest-path) `00:40:23 - 00:55:32`
   - [Variabili e Funzione Obiettivo](#variabili-shortest-path)
   - [Vincoli di Flusso](#vincoli-flusso)
   - [Problema dei Cicli a Costo Negativo](#cicli-negativi)
6. [Problema di Pianificazione Rinnovo](#problema-rinnovo) `00:55:32 - 01:07:08`
   - [Contesto del Problema](#contesto-rinnovo)
   - [Modellazione come Grafo](#modellazione-grafo-rinnovo)
   - [Calcolo dei Costi degli Archi](#calcolo-costi-rinnovo)
7. [Shortest Path per Grafi Aciclici](#shortest-path-aciclico) `01:07:08 - fine`
   - [Ordinamento Topologico](#ordinamento-topologico)
   - [Algoritmo con Regola Induttiva](#algoritmo-induttivo)
   - [Algoritmo Push-Forward](#algoritmo-push-forward)
   - [Esempio Dettagliato](#esempio-dettagliato-aciclico)

---

## <a name="introduzione"></a>1. Introduzione

`00:02:00`

Nella lezione di oggi affronteremo:
- **Ripasso dell'algoritmo di visita del grafo** visto alla fine della lezione precedente
- **Formulazione del problema del percorso più breve** (shortest path)
- **Algoritmo per grafi aciclici** con ordinamento topologico
- Applicazioni pratiche: matching OD matrix, pianificazione rinnovo

---

## <a name="od-matrix-matching"></a>2. Problema OD Matrix Matching

### <a name="contesto-od"></a>2.1 Contesto e Formulazione

`00:02:00 - 00:10:00`

> **Contesto Reale: Analisi Dati di Trasporto**
> 
> Nel sistema di pedaggio autostradale, ogni veicolo genera due "beep":
> - Un **beep di entrata** (Origin) quando entra in autostrada
> - Un **beep di uscita** (Destination) quando esce
> 
> **Problema**: A volte i beep sono "orfani" - non matchano correttamente. Dobbiamo **ricostruire gli accoppiamenti** (O,D) più plausibili basandoci sui timestamp.

#### Formulazione del Problema

**Dati**:
- **Insieme I**: Beep di ingresso orfani con timestamp t_i
- **Insieme O**: Beep di uscita orfani con timestamp t_j
- Assunzione: |I| = |O| = n (stesso numero di beep orfani)

**Obiettivo**: 
Accoppiare ogni ingresso con un'uscita minimizzando il **massimo scarto temporale** (funzione bottleneck).

### <a name="grafo-bipartito-od"></a>2.2 Grafo Bipartito e Variabili

`00:10:00 - 00:14:00`

Modelliamo il problema con un **grafo bipartito**:

```
Ingressi (I):  ●     ●     ●
                \   / \   / \
                 \ /   \ /   \
Uscite (O):      ●     ●     ●
```

#### Variabili Decisionali

```
X_ij ∈ {0,1}  ∀i ∈ I, j ∈ O
```

- X_ij = 1 se accoppiamo l'ingresso i con l'uscita j
- X_ij = 0 altrimenti

**Parametro**:
- Δ_ij = |t_j - t_i| (differenza temporale tra ingresso i e uscita j)

### <a name="linearizzazione-bottleneck"></a>2.3 Linearizzazione Funzione Bottleneck

`00:14:00 - 00:17:34`

#### Funzione Obiettivo Originale (Non Lineare)

Vogliamo minimizzare il **massimo scarto**:

```
min max{Δ_ij · X_ij : i ∈ I, j ∈ O}
```

Questa è una funzione **bottleneck** (minimax) - non lineare!

#### Linearizzazione con Variabile Ausiliaria

Introduciamo una variabile **dummy D** che rappresenta il massimo scarto:

```
min D

soggetto a:
    D ≥ Δ_ij · X_ij    ∀i ∈ I, j ∈ O
```

Quando X_ij = 1: Il vincolo diventa D ≥ Δ_ij
Quando X_ij = 0: Il vincolo diventa D ≥ 0 (sempre soddisfatto)

Quindi D cattura automaticamente il **massimo** tra tutti gli accoppiamenti selezionati!

> **💡 Tecnica Generale**
> 
> Questa tecnica di **linearizzazione con variabile dummy** è applicabile a molte funzioni minimax:
> ```
> min max_i f_i(x)  →  min D  s.t.  D ≥ f_i(x)  ∀i
> ```

#### Vincoli di Matching

**Vincoli "All-Different"**: Ogni ingresso accoppiato con un'unica uscita e viceversa

```
Σ_{j∈O} X_ij = 1    ∀i ∈ I  (ogni ingresso usato esattamente una volta)

Σ_{i∈I} X_ij = 1    ∀j ∈ O  (ogni uscita usata esattamente una volta)
```

Questi sono i classici **vincoli di assegnamento** (assignment constraints).

#### Formulazione Completa

```
min D

s.t.:
    D ≥ Δ_ij · X_ij              ∀i ∈ I, j ∈ O
    Σ_{j∈O} X_ij = 1              ∀i ∈ I
    Σ_{i∈I} X_ij = 1              ∀j ∈ O
    X_ij ∈ {0,1}                  ∀i ∈ I, j ∈ O
    D ≥ 0
```

> **📊 Tipo di Problema**
> 
> Questo è un **problema di assegnamento** (assignment problem) con funzione obiettivo minimax.
> - Con costi lineari: Risolubile in O(n³) con l'algoritmo ungherese
> - Con minimax: Risolubile con programmazione lineare intera

---

## <a name="esercizio-satelliti"></a>3. Esercizio: Sequenziamento Immagini Satellitari

`00:17:34 - 00:21:23`

> **Esercizio Settimanale - Contesto**
> 
> Un satellite orbita attorno alla Terra con le seguenti caratteristiche:
> - **Periodo orbitale**: 1 settimana (7 giorni)
> - **Fotocamera** con capacità di inclinazione
> - **Richieste** di immagini di diverse zone della Terra
> - **Tempi di setup** S_ij per orientare la camera tra richiesta i e richiesta j

**Obiettivo**: Determinare la sequenza ottimale di acquisizione immagini minimizzando il tempo totale di setup.

#### Riduzione a TSP

Questo problema può essere **ridotto a un TSP**:

1. **Nodi**: Ogni richiesta di immagine è un nodo
2. **Archi**: Ogni possibile transizione tra richieste
3. **Costi**: S_ij (tempo di setup da richiesta i a richiesta j)
4. **Tour**: Sequenza che visita tutte le richieste una volta

#### Costruzione del Grafo

**Passo 1**: Creare un nodo per ogni richiesta di immagine

**Passo 2**: Per ogni coppia di richieste (i,j):
- Calcolare il tempo di setup S_ij
- Questo dipende dalla distanza angolare tra le zone

**Passo 3**: Risolvere il TSP risultante

> **🔍 Esercizio Proposto**
> 
> Costruire esplicitamente il grafo per un insieme dato di richieste e applicare un algoritmo TSP (euristico o esatto).

---

## <a name="algoritmo-visita"></a>4. Algoritmo di Visita del Grafo - Ripasso

`00:22:00 - 00:38:00`

### <a name="strutture-dati-visita"></a>4.1 Strutture Dati

`00:22:00 - 00:24:21`

L'algoritmo di visita del grafo utilizza due strutture dati fondamentali:

#### 1. Vettore dei Predecessori **p**

```
p_i = {
    0           se il nodo i non è stato ancora raggiunto
    j           se abbiamo raggiunto i da j
    s           se i = s (nodo di partenza)
}
```

**Scopo**: Tenere traccia del cammino seguito per raggiungere ogni nodo

#### 2. Coda **Q** (Queue)

**Contenuto**: Insieme di nodi da cui dobbiamo continuare la ricerca

**Operazioni**:
- **Inserimento**: Quando scopriamo un nuovo nodo
- **Estrazione**: Per selezionare il prossimo nodo da visitare

> **📝 Nota Importante**
> 
> L'ordine di estrazione da Q **non influenza il risultato** (raggiungibilità), ma può influenzare:
> - Il percorso specifico trovato
> - L'ordine di esplorazione
> 
> Possibili politiche:
> - **FIFO** (Queue): Breadth-First Search (ricerca in ampiezza) = visito prima tutti i nodi vicini alla sorgente, poi quelli più lontani
> - **LIFO** (Stack): Depth-First Search (Ricerca in Profondità) = esploro un grafo andando il più lontano possibile lungo un percorso prima di tornare indietro per esplorare gli altri rami
> - **Priority Queue**: Dijkstra (per shortest path)

### <a name="pseudocodice-visita"></a>4.2 Pseudocodice e Funzionamento

`00:24:21 - 00:27:00`

#### Pseudocodice dell'Algoritmo

```
GRAPH_VISIT(G, s):
    Input: Grafo G = (N, A), nodo sorgente s
    Output: Vettore predecessori p
    
    1. Inizializzazione:
       p_i ← 0    ∀i ∈ N        // Nessun nodo raggiunto
       p_s ← s                   // Punto di partenza
       Q ← {s}                   // Inizia da s
    
    2. Ciclo principale:
       REPEAT until Q = ∅:
           3. Seleziona e rimuovi un nodo i da Q
           
           4. FOR each arco (i,j) ∈ FS(i):  // Forward star di i
               5. IF p_j = 0:               // j non ancora visitato
                   6. p_j ← i               // Raggiungiamo j da i
                   7. Q ← Q ∪ {j}           // Aggiungi j alla coda
    
    8. RETURN p
```

#### Spiegazione Passo-Passo

**Inizializzazione** (righe 1-2):
- Tutti i predecessori a 0 (nodi non visitati)
- Predecessore di s = s (convenzione per il punto di partenza)
- Q contiene solo s inizialmente

**Ciclo Principale** (righe 2-7):
- **Riga 3**: Estrai un nodo i dalla coda
- **Riga 4**: Esamina tutti gli archi uscenti da i
- **Riga 5**: Se j non è stato visitato (p_j = 0)...
- **Riga 6**: ...segna i come predecessore di j
- **Riga 7**: ...aggiungi j alla coda per esplorarlo dopo

**Terminazione**: Quando Q è vuota, abbiamo esplorato tutti i nodi raggiungibili da s

### <a name="esempio-esecuzione-visita"></a>4.3 Esempio di Esecuzione

`00:28:02 - 00:36:19`

Consideriamo un grafo con 10 nodi, partendo dal nodo 1.

#### Grafo di Esempio

```
    2 ← 1 → 3
    ↓   ↓   ↓
    5   4   7
    ↓
    6
```

#### Esecuzione Step-by-Step

**Inizializzazione**:
```
Q = {1}
p = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Iterazione 1**: Estrai 1
- Visita FS(1) = {(1,2), (1,3)}
- p_2 = 0 → p_2 ← 1, Q ← Q ∪ {2}
- p_3 = 0 → p_3 ← 1, Q ← Q ∪ {3}

```
Q = {2, 3}
p = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
```

**Iterazione 2**: Estrai 2
- Visita FS(2) = {(2,1), (2,3), (2,4), (2,5)}
- p_1 ≠ 0 (già visitato) → skip
- p_3 ≠ 0 (già visitato) → skip
- p_4 = 0 → p_4 ← 2, Q ← Q ∪ {4}
- p_5 = 0 → p_5 ← 2, Q ← Q ∪ {5}

```
Q = {3, 4, 5}
p = [1, 1, 1, 2, 2, 0, 0, 0, 0, 0]
```

**Iterazione 3**: Estrai 3
- Visita FS(3) = {(3,1), (3,2), (3,4), (3,7)}
- p_1, p_2, p_4 ≠ 0 → skip
- p_7 = 0 → p_7 ← 3, Q ← Q ∪ {7}

```
Q = {4, 5, 7}
p = [1, 1, 1, 2, 2, 0, 3, 0, 0, 0]
```

**Iterazione 4**: Estrai 4
- Visita FS(4) = {(4,7)}
- p_7 ≠ 0 → skip

```
Q = {5, 7}
p = [1, 1, 1, 2, 2, 0, 3, 0, 0, 0]
```

**Iterazione 5**: Estrai 5
- Visita FS(5) = {(5,2), (5,4), (5,6)}
- p_2, p_4 ≠ 0 → skip
- p_6 = 0 → p_6 ← 5, Q ← Q ∪ {6}

```
Q = {7, 6}
p = [1, 1, 1, 2, 2, 5, 3, 0, 0, 0]
```

**Iterazione 6**: Estrai 7
- Visita FS(7) = {(7,4), (7,6)}
- p_4, p_6 ≠ 0 → skip

```
Q = {6}
p = [1, 1, 1, 2, 2, 5, 3, 0, 0, 0]
```

**Iterazione 7**: Estrai 6
- Visita FS(6) = {(6,5), (6,7)}
- p_5, p_7 ≠ 0 → skip

```
Q = ∅
p = [1, 1, 1, 2, 2, 5, 3, 0, 0, 0]
```

**Algoritmo Terminato!**

#### Interpretazione dei Risultati

**Domanda 1**: Esiste un percorso da 1 a 6?
- Controlla p_6 = 5 ≠ 0 → **SÌ**
- Ricostruisci il percorso all'indietro:
  - 6 ← 5 ← 2 ← 1
  - **Percorso**: 1 → 2 → 5 → 6

**Domanda 2**: Esiste un percorso da 1 a 10?
- Controlla p_10 = 0 → **NO**
- **Taglio**: N_1 = {1,2,3,4,5,6,7}, N_10 = {8,9,10}
- Tutti gli archi nel taglio vanno da N_10 a N_1

### <a name="complessita-visita"></a>4.4 Analisi della Complessità

`00:27:00 - 00:28:02`

#### Complessità per Componenti

**1. Inizializzazione** (righe 1-2):
```
T_init = O(n)
```
- Imposta p_i = 0 per ogni nodo: O(n)
- Imposta p_s = s: O(1)
- Inizializza Q: O(1)

**2. Ciclo Repeat-Until**:

Quante volte eseguiamo il ciclo?
- Al massimo **n volte** (una per ogni nodo)
- Ogni nodo entra in Q al massimo una volta
- Perché? Una volta che p_j ≠ 0, non può più essere re-inserito

**3. Operazioni all'interno del ciclo**:
- Estrazione da Q: O(1)
- For each arco in FS(i): dipende da |FS(i)|

#### Analisi Aggregata

**Osservazione Chiave**: Se "srotolamento" il repeat-until, stiamo visitando la forward star di ogni nodo **esattamente una volta**.

Costo totale del for:
```
Σ_{i∈N} |FS(i)| = |A| = m
```

Perché? Le forward star sono **disgiunte** e la loro unione è A!

#### Complessità Totale

```
T(n,m) = O(n) + O(n) · O(1) + O(m)
       = O(n + m)
       = O(m)  se il grafo è connesso (m ≥ n-1)
```

> **🚀 Efficienza**
> 
> L'algoritmo di visita del grafo è **lineare** nella dimensione dell'input!
> - Tempo: O(m)
> - Spazio: O(n) per memorizzare p e Q
> 
> Non possiamo fare meglio di così per un problema su grafi (dobbiamo almeno "guardare" tutti gli archi).

### <a name="proprieta-percorso-taglio"></a>4.5 Proprietà Percorso vs Taglio

`00:37:28 - 00:38:40`

#### Teorema Fondamentale

> **📐 Proprietà Dualità Percorso-Taglio**
> 
> Dato un grafo orientato G = (N, A) e due nodi s, t:
> 
> **O** esiste un percorso da s a t
> 
> **OPPURE**
> 
> esiste un taglio (N_s, N_t) tale che:
> - s ∈ N_s, t ∈ N_t
> - Tutti gli archi del taglio vanno da N_t a N_s
> 
> **Non ci sono altre possibilità!**

#### Certificati Polinomiali

**Certificato per "SÌ esiste percorso"**:
- Il percorso stesso
- Lunghezza: O(n)
- Verificabile in tempo O(n)

**Certificato per "NO non esiste percorso"**:
- Il taglio (N_s, N_t)
- Dimensione: O(n)
- Verificabile in tempo O(m)

> **💡 Implicazione Importante**
> 
> Entrambe le risposte (SÌ e NO) hanno certificati **concisi** (dimensione polinomiale) verificabili in tempo polinomiale.
> 
> Questa è una caratteristica dei **problemi facili** (classe P).

#### Collegamenti con Altre Proprietà

**Lemma di Farkas per Grafi**: un sistema di equazioni lineari ha **una soluzione** oppure esiste un **certificato** (una combinazione lineare) che dimostra che nessuna soluzione è possibile — ma non entrambe le cose

**Lemma di Colorazione di Menger**: Versione con colorazione casuale degli archi:
- O trovi un percorso fatto di archi di 2 colori
- O trovi un taglio fatto di archi degli altri 2 colori

**Spazi Duali**: Nello spazio vettoriale:
- Spazio dei percorsi e spazio dei tagli sono **complementari**
- Sono **duali** l'uno dell'altro

---

## <a name="formulazione-shortest-path"></a>5. Formulazione Shortest Path

`00:40:23 - 00:55:32`

### <a name="variabili-shortest-path"></a>5.1 Variabili e Funzione Obiettivo

`00:40:23 - 00:42:31`

Passiamo ora dal problema di **esistenza di un percorso** al problema di trovare il **percorso più breve**.

#### Input del Problema

- **Grafo orientato** G = (N, A)
- **Costi** c_ij associati ad ogni arco (i,j) ∈ A
- **Nodo sorgente** s
- **Nodo destinazione** t

#### Variabili Decisionali

```
X_ij ∈ {0,1}  ∀(i,j) ∈ A
```

- X_ij = 1 se l'arco (i,j) è selezionato nel percorso più breve
- X_ij = 0 altrimenti

#### Funzione Obiettivo

```
min Σ_{(i,j)∈A} c_ij · X_ij
```

Minimizziamo la **somma dei costi** degli archi selezionati.

### <a name="vincoli-flusso"></a>5.2 Vincoli di Flusso

`00:42:31 - 00:50:00`

I vincoli garantiscono che gli archi selezionati formino un percorso da s a t.

#### Vincoli alla Sorgente s

```
Σ_{j:(s,j)∈FS(s)} X_sj = 1
```

**Significato**: Dalla sorgente s, deve uscire esattamente 1 arco.

> **Interpretazione come Flusso**: Iniettiamo 1 unità di flusso dalla sorgente.

#### Vincoli alla Destinazione t

```
Σ_{i:(i,t)∈BS(t)} X_it = 1
```

**Significato**: Nella destinazione t, deve entrare esattamente 1 arco.

> **Interpretazione come Flusso**: Estraiamo 1 unità di flusso alla destinazione.

#### Vincoli di Conservazione del Flusso (Nodi Interni)

`00:47:28 - 00:50:00`

Per ogni nodo i ≠ s, i ≠ t:

```
Σ_{j:(j,i)∈BS(i)} X_ji = Σ_{j:(i,j)∈FS(i)} X_ij
```

**Forma alternativa** (più comune):

```
Σ_{j:(j,i)∈BS(i)} X_ji - Σ_{j:(i,j)∈FS(i)} X_ij = 0
```

**Significato**: 
- Flusso entrante = Flusso uscente
- Se un arco entra nel nodo, un arco deve uscire
- Se nessun arco entra, nessun arco esce (il nodo non è sul percorso)

#### Configurazioni Possibili per un Nodo Interno

**Caso 1**: Nodo sul percorso
```
Entrante: 1 arco
Uscente:  1 arco
Bilancio: 1 - 1 = 0 ✓
```

**Caso 2**: Nodo non sul percorso
```
Entrante: 0 archi
Uscente:  0 archi
Bilancio: 0 - 0 = 0 ✓
```

> **💡 Insight**
> 
> Il vincolo di conservazione del flusso **unifica** due casi in un'unica equazione elegante!

#### Formulazione Completa

```
min Σ_{(i,j)∈A} c_ij · X_ij

s.t.:
    Σ_{j:(s,j)∈FS(s)} X_sj = 1                              (sorgente)
    
    Σ_{i:(i,t)∈BS(t)} X_it = 1                              (destinazione)
    
    Σ_{j:(j,i)∈BS(i)} X_ji - Σ_{j:(i,j)∈FS(i)} X_ij = 0     ∀i ∈ N \ {s,t}
    
    X_ij ∈ {0,1}                                             ∀(i,j) ∈ A
```

#### Struttura dei Vincoli

`00:54:19 - 00:54:54`

Osservazione importante:

- **1 variabile per ogni arco**: |X| = m
- **1 vincolo per ogni nodo**: |constraints| = n

Questa è la struttura tipica dei **problemi di flusso su rete**.

> **🔗 Connessione**: Riconoscete questa struttura? Sono **vincoli di flusso** analoghi a quelli visti nel problema multi-periodo dell'inventario!

### <a name="cicli-negativi"></a>5.3 Problema dei Cicli a Costo Negativo

`00:51:32 - 00:54:19`

#### Limitazione della Formulazione

**Domanda**: Cosa succede se ci sono **costi negativi**?

**Risposta**: Dobbiamo fare attenzione ai **cicli a costo negativo**!

#### Esempio di Ciclo Negativo

```
Ciclo: 1 → 2 → 3 → 1
Costi: c_12 = -2, c_23 = -1, c_31 = -1
Costo totale: -2 + (-1) + (-1) = -4 < 0
```

Se esiste un ciclo a costo negativo raggiungibile da s:
- Possiamo percorrere il ciclo infinite volte
- Ogni volta riduciamo il costo di -4
- Il costo ottimo tende a -∞!

> **⚠️ Problema Mal Definito**
> 
> Con cicli a costo negativo, il problema shortest path **non ha soluzione finita**.

#### Assunzioni per Risolvibilità

Per garantire una soluzione finita, assumiamo una delle seguenti:

**Opzione 1**: **Tutti i costi sono non-negativi**
```
c_ij ≥ 0  ∀(i,j) ∈ A
```

**Opzione 2**: **Il grafo è aciclico** (DAG)
```
Non esistono cicli (né positivi né negativi)
```

Con queste assunzioni, il problema è ben definito e risolvibile!

---

## <a name="problema-rinnovo"></a>6. Problema di Pianificazione Rinnovo

`00:55:32 - 01:07:08`

### <a name="contesto-rinnovo"></a>6.1 Contesto del Problema

`00:55:32 - 00:57:45`

> **Scenario Reale: Piano di Rinnovo Auto**
> 
> Dopo la laurea, ottenete un lavoro che richiede un'auto. Dovete pianificare:
> - **Quando comprare** una nuova auto
> - **Quando vendere** l'auto corrente
> - **Bilancio** tra costi di acquisto, manutenzione e ricavi dalla vendita
> 
> **Orizzonte di pianificazione**: 5 anni

#### Dati del Problema

**Costi di Acquisto** (inflazione):
```
Anno 0: $12,000
Anno 1: $13,000
Anno 2: $14,000
Anno 3: $15,000
Anno 4: $16,000
```

**Tabella Manutenzione e Rivendita**:

| Anni possesso | Manutenzione | Rivendita |
|---------------|--------------|-----------|
| 1 anno        | $2,000       | $7,000    |
| 2 anni        | $4,000       | $6,000    |
| 3 anni        | $7,000       | $4,000    |
| 4 anni        | $11,000      | $2,000    |
| 5 anni        | $16,000      | $1,000    |

**Obiettivo**: Minimizzare il costo netto totale sull'orizzonte di 5 anni.

### <a name="modellazione-grafo-rinnovo"></a>6.2 Modellazione come Grafo

`00:57:45 - 01:03:00`

#### Costruzione del Grafo

**Nodi**: Istanti di tempo quando possiamo prendere decisioni
```
Nodi: 0, 1, 2, 3, 4, 5
- 0: Inizio (ora)
- 1: Fine anno 1
- 2: Fine anno 2
- ...
- 5: Fine orizzonte
```

**Archi**: Decisioni di "tenere l'auto per k anni"
```
Arco (i, j): "Compro un'auto all'anno i e la rivendo all'anno j"
            Dove j - i = numero di anni di possesso
```

#### Esempio di Archi

```
Da nodo 0:
- (0, 1): Tengo auto 1 anno
- (0, 2): Tengo auto 2 anni
- (0, 3): Tengo auto 3 anni
- (0, 4): Tengo auto 4 anni
- (0, 5): Tengo auto 5 anni

Da nodo 1:
- (1, 2): Compro anno 1, vendo anno 2 (1 anno)
- (1, 3): Compro anno 1, vendo anno 3 (2 anni)
- (1, 4): Compro anno 1, vendo anno 4 (3 anni)
- (1, 5): Compro anno 1, vendo anno 5 (4 anni)

Da nodo 2:
- (2, 3): Compro anno 2, vendo anno 3 (1 anno)
- (2, 4): Compro anno 2, vendo anno 4 (2 anni)
- (2, 5): Compro anno 2, vendo anno 5 (3 anni)

...
```

#### Grafo Risultante

```
0 ──→ 1 ──→ 2 ──→ 3 ──→ 4 ──→ 5
 \     \     \     \     \
  \     \     \     \     └────→ 5
   \     \     \     └─────────→ 5
    \     \     └──────────────→ 5
     \     └───────────────────→ 5
      └────────────────────────→ 5
```

> **💡 Osservazione Chiave**
> 
> - Ogni **percorso** da 0 a 5 rappresenta un **piano di rinnovo ammissibile**
> - Il **costo del percorso** è il **costo totale del piano**
> - Il **percorso più breve** è il **piano ottimo**!

#### Esempio di Percorsi

**Percorso 1**: 0 → 1 → 4 → 5
- Compro anno 0, vendo anno 1 (1 anno)
- Compro anno 1, vendo anno 4 (3 anni)
- Compro anno 4, vendo anno 5 (1 anno)

**Percorso 2**: 0 → 2 → 5
- Compro anno 0, vendo anno 2 (2 anni)
- Compro anno 2, vendo anno 5 (3 anni)

**Percorso 3**: 0 → 5
- Compro anno 0, vendo anno 5 (5 anni - nessun cambio)

### <a name="calcolo-costi-rinnovo"></a>6.3 Calcolo dei Costi degli Archi

`01:03:00 - 01:06:36`

#### Formula del Costo di un Arco

Per l'arco (i, j) dove teniamo l'auto per k = j - i anni:

```
c_ij = Costo_acquisto_i + Σ Costi_manutenzione - Ricavo_vendita_j
```

#### Esempi di Calcolo

**Arco (0, 1)**: Compro anno 0, vendo anno 1 (1 anno)
```
c_01 = 12,000 + 2,000 - 7,000 = 7,000
```

**Arco (0, 2)**: Compro anno 0, vendo anno 2 (2 anni)
```
c_02 = 12,000 + (2,000 + 4,000) - 6,000 = 12,000
```

**Arco (1, 2)**: Compro anno 1, vendo anno 2 (1 anno)
```
c_12 = 13,000 + 2,000 - 7,000 = 8,000
```

**Arco (1, 4)**: Compro anno 1, vendo anno 4 (3 anni)
```
c_14 = 13,000 + (2,000 + 4,000 + 7,000) - 4,000 = 22,000
```

#### Grafo con Costi

```
     7K      8K      9K     10K     11K
0 ──→ 1 ──→ 2 ──→ 3 ──→ 4 ──→ 5
 \ 12K \ 13K \ 14K \ 15K
  \     \     \     \
   \     \     \     └──────→ 5 (25K)
    \     \     └────────────→ 5 (22K)
     \     └─────────────────→ 5 (22K)
      └───────────────────────→ 5 (24K)
```

(Nota: Valori approssimativi per illustrazione)

> **🎯 Problema Risolto**
> 
> Ora basta applicare un **algoritmo di shortest path** su questo grafo per trovare il piano di rinnovo ottimale!

---

## <a name="shortest-path-aciclico"></a>7. Shortest Path per Grafi Aciclici

`01:07:08 - fine`

### <a name="ordinamento-topologico"></a>7.1 Ordinamento Topologico

`01:07:08 - 01:11:27`

#### Definizione

Un grafo è **aciclico** (DAG - Directed Acyclic Graph) se non contiene cicli.

**Ordinamento Topologico**: Una numerazione dei nodi tale che:
```
Se (i,j) ∈ A  ⟹  i < j
```

In altre parole: tutti gli archi "vanno avanti" nella numerazione.

> **⚙️ Proprietà del Problema di Rinnovo**
> 
> Il grafo del problema di rinnovo è **automaticamente aciclico** perché le decisioni sono sul tempo, e il tempo non torna indietro!

#### Algoritmo di Renumerazione

**Idea**: Ripetutamente seleziona un nodo senza archi entranti.

```
TOPOLOGICAL_SORT(G):
    1. numero ← 1
    2. S ← insieme dei nodi senza archi entranti
    
    3. WHILE S ≠ ∅:
        4. Seleziona un nodo i da S
        5. Assegna i ← numero
        6. numero ← numero + 1
        7. Rimuovi i e tutti gli archi uscenti da i
        8. Aggiorna S con nuovi nodi senza archi entranti
    
    9. IF numero ≤ n:
        10. RETURN "Grafo contiene cicli"
    11. ELSE:
        12. RETURN numerazione
```

#### Esempio di Esecuzione

Grafo iniziale:
```
C → B → A → E → D → F → G
    ↓       ↓
    A       D
```

**Step 1**: C non ha archi entranti → C = 1

**Step 2**: Rimuovi C → B non ha archi entranti → B = 2

**Step 3**: Rimuovi B → A non ha archi entranti → A = 3

**Step 4**: Rimuovi A → E non ha archi entranti → E = 4

**Step 5**: Rimuovi E → D non ha archi entranti → D = 5

**Step 6**: Rimuovi D → F non ha archi entranti → F = 6

**Step 7**: Rimuovi F → G non ha archi entranti → G = 7

**Ordinamento finale**: C(1), B(2), A(3), E(4), D(5), F(6), G(7)

> **🚀 Complessità**
> 
> L'ordinamento topologico può essere implementato in **O(n + m)** tempo lineare usando una coda dei nodi senza archi entranti.

### <a name="algoritmo-induttivo"></a>7.2 Algoritmo con Regola Induttiva

`01:11:27 - 01:14:59`

#### Idea dell'Algoritmo

Sfruttiamo l'ordinamento topologico per calcolare le **etichette** (lunghezza del percorso più breve) in ordine.

#### Etichetta di un Nodo

```
d_i = Lunghezza del percorso più breve da s a i
```

**Base dell'induzione**: 
```
d_s = 0  (distanza da s a s è zero)
```

**Passo induttivo**: Per un nodo i (in ordine topologico):
```
d_i = min{d_j + c_ji : (j,i) ∈ BS(i)}
```

Dove BS(i) è la **backward star** di i (archi entranti).

> **💡 Perché Funziona?**
> 
> Grazie all'ordinamento topologico, quando calcoliamo d_i:
> - Tutti i nodi j nella backward star hanno j < i
> - Quindi d_j è già stato calcolato!
> - Possiamo usare il principio di **programmazione dinamica**

#### Regola Induttiva Spiegata

Per trovare il percorso più breve verso i:
1. Guarda tutti gli archi che **entrano** in i
2. Per ogni arco (j,i):
   - Il percorso passerebbe per j e poi userebbe l'arco (j,i)
   - Costo totale: d_j + c_ji
3. Prendi il **minimo** tra tutte queste opzioni

#### Esempio Numerico

Grafo con ordinamento topologico 1,2,3,4,5,6,7:

```
d_1 = 0  (nodo sorgente)

d_2 = d_1 + c_12 = 0 + 1 = 1

d_3 = min{d_1 + c_13, d_2 + c_23}
    = min{0 + 4, 1 + 1}
    = min{4, 2} = 2

d_4 = d_3 + c_34 = 2 + 2 = 4

d_5 = min{d_2 + c_25, d_3 + c_35, d_4 + c_45}
    = min{1 + 6, 2 + 3, 4 + 0}
    = min{7, 5, 4} = 4

d_6 = min{d_2 + c_26, d_5 + c_56}
    = min{1 + 10, 4 + 3}
    = min{11, 7} = 7

d_7 = min{d_4 + c_47, d_5 + c_57, d_6 + c_67}
    = min{4 + 4, 4 + 2, 7 + 10}
    = min{8, 6, 17} = 6
```

**Percorso più breve da 1 a 7**: Lunghezza 6

### <a name="algoritmo-push-forward"></a>7.3 Algoritmo Push-Forward

`01:17:33 - 01:22:00`

Invece di "tirare" informazioni dalla backward star, "spingiamo" lungo la forward star.

#### Pseudocodice

```
SHORTEST_PATH_DAG(G, c, s):
    Input: DAG G = (N,A) in ordine topologico, costi c, sorgente s
    Output: Vettore predecessori p, vettore etichette d
    
    1. Inizializzazione:
       p_i ← 0           ∀i ∈ N
       d_i ← +∞          ∀i ∈ N
       p_s ← s
       d_s ← 0
    
    2. FOR i = 1 TO n-1:  // Ordine topologico
        3. FOR each (i,j) ∈ FS(i):  // Forward star
            4. IF d_i + c_ij < d_j:  // Trovato percorso migliore
                5. d_j ← d_i + c_ij  // Aggiorna etichetta
                6. p_j ← i            // Aggiorna predecessore
    
    7. RETURN p, d
```

#### Differenze con Algoritmo Pull

| Aspetto | Pull (Backward) | Push (Forward) |
|---------|----------------|----------------|
| **Direzione** | Guarda indietro (BS) | Guarda avanti (FS) |
| **Ordine** | Topologico diretto | Topologico diretto |
| **Logica** | "Da dove arrivo?" | "Dove posso andare?" |
| **Implementazione** | Min su BS | Aggiornamenti su FS |

Entrambi gli approcci sono **equivalenti** e hanno la stessa complessità!

#### Analisi della Complessità

**Inizializzazione**: O(n)

**Ciclo esterno**: n-1 iterazioni (tutti i nodi tranne l'ultimo)

**Ciclo interno**: Visita forward star di ogni nodo

**Complessità totale**:
```
Σ_{i=1}^{n-1} |FS(i)| = |A| = m
```

**Risultato**: O(n + m) = **O(m)** (lineare negli archi!)

> **🎉 Efficienza Ottimale**
> 
> Per grafi aciclici, possiamo risolvere shortest path in **tempo lineare** senza bisogno di strutture dati complesse come gli heap!

### <a name="esempio-dettagliato-aciclico"></a>7.4 Esempio Dettagliato

`01:22:00 - 01:34:00`

#### Grafo di Esempio

```
Nodi: 1, 2, 3, 4, 5, 6, 7 (già in ordine topologico)

Archi e costi:
(1,2): 1    (1,3): 4
(2,3): 1    (2,4): 3    (2,5): 6    (2,6): 10
(3,4): 2    (3,5): 3
(4,5): 0    (4,7): 4
(5,6): 3    (5,7): 2
(6,7): 10
```

#### Esecuzione Algoritmo Push-Forward

**Inizializzazione**:
```
Etichette: [0, ∞, ∞, ∞, ∞, ∞, ∞]
Predecessori: [1, 0, 0, 0, 0, 0, 0]
```

**i = 1**: Visita FS(1) = {(1,2), (1,3)}
```
(1,2): d_2 = min(∞, 0+1) = 1, p_2 = 1
(1,3): d_3 = min(∞, 0+4) = 4, p_3 = 1

Etichette: [0, 1, 4, ∞, ∞, ∞, ∞]
Predecessori: [1, 1, 1, 0, 0, 0, 0]
```

**i = 2**: Visita FS(2) = {(2,3), (2,4), (2,5), (2,6)}
```
(2,3): d_3 = min(4, 1+1) = 2, p_3 = 2
(2,4): d_4 = min(∞, 1+3) = 4, p_4 = 2
(2,5): d_5 = min(∞, 1+6) = 7, p_5 = 2
(2,6): d_6 = min(∞, 1+10) = 11, p_6 = 2

Etichette: [0, 1, 2, 4, 7, 11, ∞]
Predecessori: [1, 1, 2, 2, 2, 2, 0]
```

**i = 3**: Visita FS(3) = {(3,4), (3,5)}
```
(3,4): d_4 = min(4, 2+2) = 4 (no change)
(3,5): d_5 = min(7, 2+3) = 5, p_5 = 3

Etichette: [0, 1, 2, 4, 5, 11, ∞]
Predecessori: [1, 1, 2, 2, 3, 2, 0]
```

**i = 4**: Visita FS(4) = {(4,5), (4,7)}
```
(4,5): d_5 = min(5, 4+0) = 4, p_5 = 4
(4,7): d_7 = min(∞, 4+4) = 8, p_7 = 4

Etichette: [0, 1, 2, 4, 4, 11, 8]
Predecessori: [1, 1, 2, 2, 4, 2, 4]
```

**i = 5**: Visita FS(5) = {(5,6), (5,7)}
```
(5,6): d_6 = min(11, 4+3) = 7, p_6 = 5
(5,7): d_7 = min(8, 4+2) = 6, p_7 = 5

Etichette: [0, 1, 2, 4, 4, 7, 6]
Predecessori: [1, 1, 2, 2, 4, 5, 5]
```

**i = 6**: Visita FS(6) = {(6,7)}
```
(6,7): d_7 = min(6, 7+10) = 6 (no change)

Etichette finali: [0, 1, 2, 4, 4, 7, 6]
Predecessori finali: [1, 1, 2, 2, 4, 5, 5]
```

#### Ricostruzione del Percorso

**Percorso da 1 a 7**:
- Partendo da 7: p_7 = 5
- Da 5: p_5 = 4
- Da 4: p_4 = 2
- Da 2: p_2 = 1
- Arrivati a 1!

**Percorso**: 1 → 2 → 4 → 5 → 7

**Costo**: 6

#### Albero dei Percorsi più Brevi

```
        1 (0)
        │
        ↓
        2 (1)
       ╱│
      ↙ │
    3   │
   (2)  ↓
        4 (4)
        │
        ↓
        5 (4)
       ╱ ↘
      ↙   ↘
    6      7
   (7)    (6)
```

> **🌳 Shortest Path Tree**
> 
> L'algoritmo non solo trova il percorso più breve verso il nodo di destinazione, ma costruisce un **albero dei percorsi più brevi** da s a **tutti** gli altri nodi!

---

## 📝 Note Finali e Preparazione Prossima Lezione

### Riepilogo Lezione 8

`01:34:00 - fine`

Abbiamo visto:

1. **Algoritmo di visita del grafo**: O(m) tempo, trova percorsi o tagli
2. **Formulazione shortest path**: Problema di flusso con conservazione
3. **Problemi applicativi**: OD matching, pianificazione rinnovo, satelliti
4. **Algoritmo per DAG**: O(m) tempo con ordinamento topologico

### Prossimi Argomenti

Nella prossima lezione vedremo:

1. **Formulazione alternativa** del shortest path
2. **Significato delle etichette** in termini duali
3. Algoritmi per grafi **non aciclici**:
   - Algoritmo di Dijkstra (costi non-negativi)
   - Algoritmo di Bellman-Ford (costi generali)
4. **Selezione dell'algoritmo** in base alle caratteristiche del grafo

### Materiale da Portare

`01:34:00`

> **📌 Importante**: Non dimenticate di portare il vostro materiale con:
> - **Corde** (rope/string)
> - **Clip/mollette**
> 
> Per fare attività pratiche sui grafi!

---

## 🎯 Concetti Chiave da Ricordare

### Algoritmi su Grafi
- **Visita del grafo**: O(m) lineare, certificati per percorso e taglio
- **Ordinamento topologico**: O(n+m) per DAG
- **Shortest path su DAG**: O(m) senza strutture dati complesse

### Modellazione
- **OD Matrix**: Grafo bipartito, linearizzazione minimax
- **Rinnovo auto**: Decisioni nel tempo come archi in DAG
- **Satelliti**: Riduzione a TSP

### Vincoli di Flusso
- Conservazione del flusso unifica casi diversi
- Struttura matrice incidenza collega flussi e grafi

### Proprietà Teoriche
- Dualità percorso-taglio
- Certificati polinomiali per problemi facili (P)
- Cicli negativi rendono shortest path mal definito

---

**Fine Lezione 8**

*Prossima lezione: Algoritmi Shortest Path per Grafi Generali (Dijkstra, Bellman-Ford)*
