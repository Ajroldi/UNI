# Lezione 11 - Esercizi Shortest Path e Spanning Tree

## 1. Puzzle: Scaling dei Costi

**Domanda**: Sommando una costante a tutti i costi, lo shortest path cambia?

**Risposta**: **SÌ, cambia!**

**Controesempio** (3 nodi):
```
Grafo originale:
1 → 2: costo 1
2 → 3: costo -2
1 → 3: costo 0

Shortest path: 1→2→3 (costo -1)

Dopo scaling +2:
1 → 2: costo 3
2 → 3: costo 0  
1 → 3: costo 2

Shortest path: 1→3 (costo 2) ← CAMBIATO!
```

**Perché non funziona**: Percorsi con **diverso numero di archi** vengono penalizzati diversamente
- Percorso con 2 archi: penalità 2M
- Percorso con 1 arco: penalità M
- **Ordine si inverte**!

**Quando funziona**: MST - tutti gli alberi hanno sempre n-1 archi → penalità costante → equivalente

## 2. Minimum Spanning Tree

**Algoritmo Kruskal** sul grafo dato:
```
Ordine archi: (1,2):2, (6,7):3, (1,5):4, (1,3):5, (3,5):6, (3,6):7, ...

Archi aggiunti: (1,2), (6,7), (1,5), (1,3), (3,6), (6,4)
Costo totale MST: 2+3+4+5+7+12 = 33
```

### Post-Ottimalità Arco (5,7)

**Arco NON nell'albero** - analisi per diminuzione:
- Aggiungendo (5,7) si crea **ciclo**: 5-1-3-6-7-5
- Archi nel ciclo: (1,5):4, (1,3):5, (3,6):7, (6,7):3
- **Condizione**: C(5,7) ≥ max{4,5,7,3} = **7**

**Range**: **[7, +∞)**

### Post-Ottimalità Arco (3,6)

**Arco nell'albero** - analisi per aumento:
- Rimuovendo (3,6) si crea **taglio**: {1,2,3,5} vs {4,6,7}
- Archi che attraversano: (2,4):16, (3,4):15, (5,6):10, (5,7):9, (3,7):13
- **Condizione**: C(3,6) ≤ min{16,15,10,9,13} = **9**

**Range**: **[0, 9]**

## 3. Viaggio con Rifornimenti

**Problema**: Minimizzare costo carburante da A a B con rifornimenti completi

**Modellazione**:
- Nodi: 0 (partenza), 1...n (stazioni), n+1 (arrivo)
- Archi (i,j): se Σ consumi ≤ capacità serbatoio Q
```
Condizione: Σ(h=i+1 to j) p_h ≤ Q
```

**Costo archi**:
```
C(i,j) = (Σ consumi da i a j) × (prezzo carburante in j)
       = [Σ(h=i+1 to j) p_h] × c_j
```

**Soluzione**: Shortest path 0 → n+1

## 4. Raccolta Rifiuti Industriali

**Problema**: Camion raccoglie rifiuti da clienti in ordine fisso, capacità Q limitata

**Grafo Logico**:
- Nodi: 0 (deposito inizio), 1...n (clienti), n+1 (deposito fine)
- Arco (i,j): servire clienti i, i+1, ..., j-1 consecutivamente

**Vincolo capacità**:
```
Σ(h=i to j-1) q_h ≤ Q
```

**Costo archi** (distanza fisica):
```
L(i,j) = s_i + Σ(h=i to j-2) d_h + s_{j-1}

dove:
- s_i = distanza Impianto → cliente i
- d_h = distanza cliente h → cliente h+1
```

**Soluzione**: Shortest path 0 → n+1 nel grafo logico

## 5. Quiz: Scelta Algoritmo

**Decision tree**:
```
Grafo ACICLICO?
├─ SÌ → SPT-Acyclic - O(m)
└─ NO → Costi TUTTI ≥ 0?
        ├─ SÌ → Dijkstra - O(m log n)
        └─ NO → SPT-LQ (Label-Correcting) - O(mn)
```

**Esempi**:
- Grafo con cicli + costi ≥0 → **Dijkstra**
- Grafo con cicli + costi negativi → **SPT-LQ**

## 6. Esercizio SPT con Radice 1

**Grafo**: Cicli presenti, costi negativi → **SPT-LQ**

**Esecuzione** (FIFO queue):
```
Init: D(1)=0, altri=∞, Q={1}

Step 1: Estrai 1 → Aggiorna 2,3
Step 2: Estrai 2 → Aggiorna 7,6,4
Step 3: Estrai 3 → Aggiorna 4,7
...

Risultato finale:
D(1)=0, D(2)=4, D(3)=2, D(4)=3, D(5)=2, D(6)=1, D(7)=-3

SPT archi: (1,2), (1,3), (2,6), (3,7), (6,4), (4,5)
```

## 7. Scaling per Cambio Radice

**Problema**: Calcolare SPT con radice 6, avendo già SPT con radice 1

**Tecnica di scaling**:
```
C'(i,j) = C(i,j) + D(i) - D(j)
```

**Proprietà**:
- Archi nello SPT → C'(i,j) = **0** (tesi)
- Archi fuori SPT → C'(i,j) **≥ 0**
- **Tutti i nuovi costi sono ≥ 0!**

**Dimostrazione equivalenza** (percorso i→...→j):
```
C_new = C_old + [D(i) - D(j)]
```

**Cancellazione telescopica**: Termini intermedi si elidono, resta solo differenza estremi!

**Vantaggio**:
- Prima computazione: SPT-LQ O(mn) - costosa
- Scaling: O(m)
- Successive computazioni: **Dijkstra O(m log n)** - veloce!

**Differenza dal puzzle iniziale**:
- Puzzle: scaling fisso → numero archi varia → NON equivalente
- Questa tecnica: scaling dipende da soluzione → cancellazione → equivalente ✅

## Prossima Lezione
Introduzione ai **Network Flows**