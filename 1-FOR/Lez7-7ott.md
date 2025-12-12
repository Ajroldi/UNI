# Ricerca Operativa - Lezione 7

**Data:** 7 Ottobre | **Prof. Federico Malucelli** | *Politecnico di Milano*

## Argomenti della Lezione

1. Problema di Acquisto Gas Multi-periodo
2. Minimum Spanning Tree (MST)
3. Algoritmi di Kruskal e Prim
4. Traveling Salesman Problem (TSP)

## 1. Problema di Acquisto Gas Multi-periodo

**Contesto:** Acquistare gas per 3 giorni minimizzando costi con gestione dell'incertezza.

### Parametri
- **Q**: Capacità deposito
- **Q⁺**: Capacità espansa
- **Pₐ**: Prezzo gas giorno d
- **Cₐ**: Costo espansione
- **Dₐ**: Domanda mercato
- **q**: Soglia minima sicurezza
- **πₐ**: Penalità per shortage

### Variabili Decisionali
- **xₐ**: Quantità gas acquistata
- **yₐ ∈ {0,1}**: Decisione espansione
- **sₐ**: Stock nel deposito
- **zₐ**: Shortage (sotto soglia minima)

### Formulazione Matematica

**Funzione obiettivo:**
```
min Σₐ (Pₐ·xₐ + Cₐ·yₐ + πₐ·zₐ)
```

**Vincoli:**
1. Bilancio inventario: `sₐ = sₐ₋₁ + xₐ - Dₐ`
2. Capacità: `sₐ ≤ Q + yₐ·(Q⁺ - Q)`
3. Soglia minima: `sₐ + zₐ ≥ q`
4. Non-negatività: `xₐ, sₐ, zₐ ≥ 0`, `yₐ ∈ {0,1}`

**Tipo:** Programmazione Lineare Intera Mista (MILP)

## 2. Minimum Spanning Tree (MST)

### Definizione
**Albero ricoprente minimo** su grafo G = (N, A): sottografo connesso aciclico che include tutti i nodi con costo totale minimo.

### Proprietà Fondamentali
1. **n-1 archi** (n = numero nodi)
2. **Nessun ciclo** (è un albero)
3. **Connesso** (percorso tra ogni coppia di nodi)
4. **Peso minimo** tra tutti gli alberi ricoprenti

### Esempio Applicativo
**Campus universitario:** Connettere edifici con rete di comunicazione minimizzando costi di installazione.

### Formulazione Matematica

**Variabili:** `xᵢⱼ ∈ {0,1}` (1 se arco selezionato)

**Obiettivo:**
```
min Σ₍ᵢ,ⱼ₎∈A cᵢⱼ·xᵢⱼ
```

**Vincoli:**
```
Σ₍ᵢ,ⱼ₎∈A xᵢⱼ = n - 1                    (esattamente n-1 archi)
Σ₍ᵢ,ⱼ₎∈δ(S) xᵢⱼ ≥ 1  ∀S ⊂ N, S ≠ ∅    (connettività)
xᵢⱼ ∈ {0,1}
```

**Nota:** Vincoli di connettività sono esponenziali (2ⁿ - 2), ma risolvibile in tempo polinomiale tramite problema di separazione (taglio minimo).

## 3. Algoritmo di Kruskal

### Approccio Greedy
Costruisce MST selezionando archi in ordine crescente di costo, evitando cicli.

### Pseudocodice
```
1. Ordina archi A per costo crescente
2. Inizializza T = ∅ (MST vuoto)
3. Ogni nodo è una componente connessa

4. Per ogni arco (i,j) in ordine:
   5. Se i e j in componenti diverse:
      6. Aggiungi (i,j) a T
      7. Unisci le componenti
      
8. Ritorna T
```

### Struttura Dati Union-Find

**Operazioni:**
- **FIND(x)**: Identifica componente contenente x
- **UNION(x,y)**: Unisce componenti di x e y

**Ottimizzazioni:**
- **Path compression**: Collega nodi direttamente alla radice
- **Union by rank**: Attacca albero piccolo a quello grande

### Complessità
- **Ordinamento archi**: O(m log m)
- **Union-Find**: O(m·α(n)) dove α(n) è funzione inversa Ackermann (≈ costante)
- **Totale**: **O(m log n)**

**Efficienza:** Quasi-lineare nel numero di archi!

## 4. Algoritmo di Prim

### Idea
Cresce componente partendo da singolo nodo, aggiungendo arco minimo che connette interno-esterno.

### Pseudocodice
```
1. Inizializza S = {s} (nodo iniziale)
2. Inizializza T = ∅

3. Finché S ≠ N:
   4. Trova arco (i,j) di costo minimo con i ∈ S, j ∉ S
   5. Aggiungi (i,j) a T e j a S
   
6. Ritorna T
```

### Confronto Kruskal vs Prim

| Aspetto | Kruskal | Prim |
|---------|---------|------|
| Approccio | Globale (ordina tutti) | Locale (cresce componente) |
| Struttura dati | Union-Find | Heap/Priority Queue |
| Migliore per | Grafi sparsi | Grafi densi |
| Complessità | O(m log n) | O(m + n log n) |

## 5. Traveling Salesman Problem (TSP)

### Definizione
Trovare il **ciclo Hamiltoniano** di costo minimo che visita tutti i nodi esattamente una volta.

### Differenze con MST

| Aspetto | MST | TSP |
|---------|-----|-----|
| Archi | n-1 | n |
| Grado nodi | Variabile | Esattamente 2 |
| Vincoli taglio | ≥ 1 arco | ≥ 2 archi |
| Complessità | **Polinomiale** | **NP-hard** |

### Formulazione Matematica

**Variabili:** `xᵢⱼ ∈ {0,1}` (1 se arco nel tour)

**Obiettivo:**
```
min Σ₍ᵢ,ⱼ₎∈A cᵢⱼ·xᵢⱼ
```

**Vincoli:**
```
Σⱼ xᵢⱼ + Σⱼ xⱼᵢ = 2  ∀i ∈ N           (grado 2 per ogni nodo)
Σ₍ᵢ,ⱼ₎∈δ(S) xᵢⱼ ≥ 2  ∀S ⊂ N, 2 ≤ |S| ≤ n-2  (eliminazione sottocicli)
xᵢⱼ ∈ {0,1}
```

### Problema dei Sottocicli
I vincoli di grado da soli NON garantiscono un unico ciclo. Esempio:
- Ciclo 1: 1→2→3→1
- Ciclo 2: 4→5→6→4

Soluzione con grado 2 per tutti, ma **non tour valido**!

### Ciclo Hamiltoniano
**Storia:** Nome da Sir William Rowan Hamilton (1857), inventore del Gioco Dicosiano.
- Basato su dodecaedro (20 vertici = città)
- Obiettivo: Tour che visita ogni città una volta
- Brevettato e venduto per £25

## 6. Concetti Chiave

### Problemi Multi-periodo
- Bilancio inventario: stato oggi = stato ieri + acquisti - domanda
- Gestione incertezza con variabili slack/shortage
- Trade-off tra costi operativi e rischio

### MST - Proprietà Essenziali
- Risolvibile in tempo polinomiale
- Due algoritmi principali: Kruskal (sparso) e Prim (denso)
- Vincoli esponenziali risolvibili per problema separazione

### TSP - Complessità
- Problema NP-hard (difficile)
- Piccola differenza da MST (2 archi vs 1 per taglio) → grande differenza computazionale
- Eliminazione sottocicli cruciale per soluzione valida

### Strutture Dati Fondamentali
- **Union-Find**: Gestione componenti connesse (Kruskal)
- **Forward Star (FS)**: Archi uscenti da nodo
- **Backward Star (BS)**: Archi entranti in nodo
- **Matrice incidenza**: Rappresentazione vincoli flusso

### Grafi Bipartiti
**Proprietà caratterizzante:** Nessun ciclo con numero dispari di archi.
- Nodi partizionati in due insiemi S e T
- Archi solo tra S↔T, mai dentro stesso insieme
- Ciclo richiede alternanza S→T→S... → numero pari archi

## 7. Applicazioni Pratiche

### MST
- Reti di comunicazione (fibra ottica)
- Reti elettriche
- Reti idriche
- Pianificazione urbana

### TSP
- Logistica e routing
- Scheduling di macchine
- DNA sequencing
- Perforazione circuiti stampati

## Preparazione Prossima Lezione

**Argomento:** Grafi orientati e percorsi più brevi

**Contenuti previsti:**
- Percorsi nei grafi orientati
- Proprietà base problemi ottimizzazione "facili"
- Algoritmi shortest path
- Criteri selezione algoritmo ottimale

**Prerequisiti:**
- Notazione grafi (nodi, archi, FS, BS)
- Concetti di percorso e taglio
- Analisi complessità algoritmi