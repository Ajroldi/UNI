# Lezione 9 - Shortest Path: Algoritmi Avanzati e Applicazioni

## 1. Production Mix e Modellazione

**Problema**: Azienda con N prodotti e M robot deve massimizzare profitti considerando:
- Ricavi per prodotto venduto
- Costi fissi di setup quando un robot produce un prodotto
- Capacità limitate dei robot

**Variabili**:
- `X_ij`: quantità di prodotto i prodotta con robot j
- `Y_ij ∈ {0,1}`: 1 se robot j produce prodotto i

**Vincoli chiave**:
- Capacità: `Σ a_ij·X_ij ≤ b_j` (tempo robot)
- Linking: `X_ij ≤ M·Y_ij` (attiva costi setup solo se produciamo)

## 2. Esercizio Satelliti → TSP

**Problema**: n foto satellitari con processing time p_j e setup time s_ij

**Soluzione**: Grafo completo K_{n+1} con:
- Costi archi: `c_ij = s_ij + p_j`
- Osservazione: `Σ p_j` è costante → equivalente usare solo `c_ij = s_ij`
- **Riduzione a TSP**: trovare ciclo hamiltoniano di costo minimo

## 3. Formulazione Duale Shortest Path

**Primale** (variabili sugli archi X_ij):
```
min Σ c_ij·X_ij
s.t. vincoli di flusso sui nodi
```

**Duale** (variabili sui nodi π_i):
```
max π_t - π_s
s.t. π_j - π_i ≤ c_ij  ∀(i,j)
```

**Interpretazione fisica**: Grafo fatto di corde di lunghezza c_ij
- Fissiamo nodo s al pavimento (π_s = 0)
- Tiriamo nodo t verso l'alto
- Archi **tesi**: nel shortest path (vincolo tight)
- Archi **lenti**: non usati (vincolo slack)

**Scoperta**: Le etichette d_i dell'algoritmo sono le variabili duali π_i!

## 4. Project Scheduling (CPM)

**Problema**: Attività con durate p_i e precedenze → minimizzare completion time

**Costruzione grafo**:
1. Nodo per ogni attività
2. Arco (i,j) se i precede j, con costo c_ij = p_i
3. Nodi dummy Begin e End

**Formulazione** (variabili t_i = tempo inizio):
```
min t_End - t_Begin
s.t. t_j - t_i ≥ p_i  ∀ precedenze
```

**Soluzione**: Algoritmo **Longest Path** su DAG (modifica shortest path con max invece di min)

**Critical Path Method**:
- Percorso critico = longest path
- Attività critiche: ritardarle ritarda l'intero progetto
- Slack time: margine per attività non critiche

## 5. Algoritmo di Dijkstra

**Estensione a grafi con cicli** (richiede costi ≥ 0)

**Idea chiave**: Simulare ordinamento topologico estraendo sempre il nodo con **etichetta minima**

**Pseudocodice**:
```
1. d_i ← +∞ ∀i, d_s ← 0, Q ← {s}
2. WHILE Q ≠ ∅:
   3. i ← nodo in Q con d_i minima    ← CHIAVE!
   4. FOR (i,j) ∈ FS(i):
      5. IF d_i + c_ij < d_j:
         6. d_j ← d_i + c_ij
         7. p_j ← i
         8. IF j ∉ Q: Q ← Q ∪ {j}
```

**Invariante**: Una volta estratto da Q, d_i è **definitivo** (con costi ≥ 0)

**Complessità**: O(n²) con lista semplice, O(m log n) con binary heap

## Prossima Lezione
- Analisi complessità Dijkstra
- Grafi con costi negativi (Bellman-Ford)
- Applicazioni pratiche (Route 66, Car rental)