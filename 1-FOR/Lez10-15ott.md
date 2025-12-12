# Lezione 10 - Analisi Dijkstra, Bellman-Ford e Applicazioni

## 1. Dimostrazione Correttezza Dijkstra

**Modello fisico con corde**: Nodi = clip, archi = corde di lunghezza c_ij
- Fissiamo sorgente A al pavimento (π_A = 0)
- Altri nodi "cadono" vincolati dalle corde
- Selezioniamo sempre il nodo con altezza minima → lo "fissiamo" permanentemente
- **Archi tesi** (tight): π_j - π_i = c_ij → nel shortest path
- **Archi lenti** (slack): π_j - π_i < c_ij → non usati

**Intuizione correttezza**: Con costi ≥ 0, quando estraiamo nodo i con d_i minima:
- Tutti i nodi in Q hanno etichetta ≥ d_i
- Qualsiasi percorso futuro avrà lunghezza ≥ d_i
- Quindi d_i è **definitivo** (ottimo)

## 2. Analisi Complessità Dijkstra

**Decomposizione operazioni**:
- **INIT**: O(n) - eseguita 1 volta
- **SELECT & REMOVE**: estratto da Q - eseguita n volte
- **UPDATE & INSERT**: su archi - eseguita m volte totali

### Implementazione Lista Non Ordinata
```
SELECT: O(n) - scansiona lista per trovare minimo
UPDATE: O(1) - accesso diretto
INSERT: O(1) - aggiungi in coda

Totale: O(n²) - ottimo per grafi DENSI (m ≈ n²)
```

### Implementazione Binary Heap
**Binary heap**: albero binario completo con heap property (min alla radice)
```
FIND_MIN: O(1)
EXTRACT_MIN: O(log n) - bubble down
INSERT: O(log n) - bubble up
UPDATE: O(log n) - decrease key

Totale: O(m log n) - ottimo per grafi SPARSI (m << n²)
```

**Confronto**:
- Grafo denso (m ≈ n²): Lista O(n²) meglio di Heap O(n² log n)
- Grafo sparso (m ≈ n): Heap O(n log n) meglio di Lista O(n²)

## 3. Dijkstra con Costi Negativi

**Problema cicli negativi**: Se Σ c_ij < 0 per un ciclo C:
- Possiamo attraversare C infinite volte
- Costo percorso → -∞
- Shortest path **non definito**

**Dijkstra fallisce anche su DAG con costi negativi**:
- Esempio: può estrarre nodi "troppo presto"
- Percorsi migliori scoperti dopo, ma nodo già fissato
- Risultati **errati** o tempo **esponenziale**

**Morale**: MAI usare Dijkstra con costi negativi!

## 4. Algoritmo di Bellman-Ford

### Versione Naive
**Idea**: Scansiona tutti gli archi ripetutamente, fai relaxation
```
REPEAT fino a nessun cambiamento:
    FOR each arco (i,j):
        IF d[i] + c_ij < d[j]:
            d[j] ← d[i] + c_ij

Al più n-1 scansioni complete
Complessità: O(nm)
```

**Proprietà**: Dopo k scansioni, abbiamo trovato tutti i percorsi di ≤ k archi

### Versione Label-Correcting (FIFO)
**Ottimizzazione**: Scansiona solo FS(i) di nodi i il cui d è cambiato
```
Q ← {s} (coda FIFO)
WHILE Q ≠ ∅:
    i ← Estrai PRIMO da Q (FIFO policy)
    FOR (i,j) in FS(i):
        IF d[i] + c_ij < d[j]:
            d[j] ← d[i] + c_ij
            IF j ∉ Q: Inserisci j in Q

Worst-case: O(nm)
Average-case: molto migliore!
```

### Rilevamento Cicli Negativi
**Contatore**: count[i] = volte che i entra in Q
- Se count[i] ≥ n → ciclo negativo esiste
- Applicazioni: arbitraggio finanziario, verifica feasibility

## 5. Riepilogo Algoritmi Shortest Path

**Decision tree**:
```
Grafo ACICLICO?
├─ SÌ → Algoritmo TOPOLOGICO - O(m)
│       Costi qualsiasi, anche negativi
│
└─ NO → Costi TUTTI ≥ 0?
        ├─ SÌ → DIJKSTRA - O(m log n) con heap
        │       O(n²) con lista (grafi densi)
        │
        └─ NO → BELLMAN-FORD - O(nm)
                Rileva cicli negativi se necessario
```

**Tabella comparativa**:
| Algoritmo | Grafo | Costi | Complessità | Note |
|-----------|-------|-------|-------------|------|
| Topologico | DAG | Qualsiasi | O(m) | Ottimo per DAG |
| Dijkstra heap | Qualsiasi | ≥ 0 | O(m log n) | Standard |
| Dijkstra lista | Qualsiasi | ≥ 0 | O(n²) | Grafi densi |
| Bellman-Ford | Qualsiasi | Qualsiasi* | O(nm) | *No cicli neg. |

## 6. Puzzle: Scaling dei Costi

**Domanda**: Se aggiungiamo costante K a tutti i costi, shortest path cambia?
```
c'_ij = c_ij + K    ∀(i,j)
```

**Risposta**: **NO, cambia**! 
- Percorsi con diverso numero di archi vengono scalati diversamente
- P₁ con k₁ archi: costo' = costo + k₁·K
- P₂ con k₂ archi: costo' = costo + k₂·K
- Se k₁ ≠ k₂, l'ordine può invertirsi!

**Controesempio** (3 nodi):
```
s → t diretto: costo 10 (1 arco)
s → u → t: costo 5+4=9 (2 archi)

Dopo +K=5:
s → t: 15 (1 arco)
s → u → t: 10+9=19 (2 archi)
Ordine invertito!
```

## 7. Esercizio Route 66

**Problema**: Obama viaggia Chicago → LA lungo Route 66
- n città con guadagni v_i (dollari/voti)
- Distanze c_{i,i+1} tra città consecutive
- Vincoli: small-d ≤ lunghezza tappa ≤ CAPITAL-D
- **Obiettivo**: Massimizzare guadagno totale

**Modellazione come grafo**:
```
Nodi: N = {1, 2, ..., n}
Archi: (i,j) se i<j AND small-d ≤ Σ(c_k) ≤ CAPITAL-D
```

**Riduzione a shortest/longest path**:
- Opzione 1: Costi g_ij = -v_j → shortest path (costi negativi)
- Opzione 2: Costi g_ij = v_j → longest path (costi positivi)

**Quale algoritmo?**
- Grafo è **DAG** (archi solo i→j con i<j)
- **Algoritmo topologico** - O(m) lineare!
- Funziona anche con costi negativi

**Soluzione**:
```
1. Costruisci grafo con archi ammissibili
2. Applica longest path su DAG
3. Ricostruisci percorso da predecessori
4. Fermate ottimali = nodi nel percorso
```

## Prossima Lezione
- Dynamic Programming approfondito
- Car Rental con Refueling
- Introduzione Network Flows