# Lezione 13 - Laboratorio: Shortest Path, MST e TSP

## 1. Introduzione

**Laboratorio pratico**: Tre problemi di ottimizzazione su grafi usando Python-MIP

**Problemi**:
1. Shortest Path (cammino minimo)
2. Minimum Spanning Tree (albero ricoprente minimo)
3. Travelling Salesman Problem (commesso viaggiatore)

**Approccio**: Formulazioni a flusso con solver MIP, non algoritmi euristici

**Librerie Python**:
```python
import numpy as np
import matplotlib.pyplot as plt
from mip import Model, xsum, minimize, BINARY
```

## 2. Shortest Path con Formulazione a Flusso

**Setup problema**:
- k=15 punti casuali in griglia 100×100
- Archi esistono se distanza ≤ d_max (35% lato griglia)
- Trovare cammino minimo da nodo 0 a nodo 12

**Generazione dati**:
```python
points = np.random.random((k, 2)) * 100  # coordinate casuali
distance = np.array([                     # matrice distanze
    [np.sqrt((points[i] - points[j])**2).sum() 
     for j in range(k)] 
    for i in range(k)
])

A = [(i,j) for i in range(k) for j in range(k)  # archi
     if distance[i][j] <= d_max and i != j]
```

**Formulazione a flusso**:

Parametro ausiliario:
```
b_i = { +1  se i = S (sorgente)
      { -1  se i = T (destinazione)
      {  0  altrimenti (nodi intermedi)
```

**Vincoli conservazione flusso**:
```
Σ f_ji - Σ f_ij = b_i  ∀i
(IN)     (OUT)
```

**Implementazione**:
```python
m = Model("shortest_path")

# Variabili (flusso continuo)
f = {(i,j): m.add_var(name=f'f_{i}_{j}') for (i,j) in A}

# Parametro b
b = {i: 0 for i in N}
b[0] = 1    # sorgente
b[12] = -1  # destinazione

# Vincoli
for i in N:
    m.add_constr(
        xsum(f[j,i] for j in N if (j,i) in A) -  # IN
        xsum(f[i,j] for j in N if (i,j) in A) == # OUT
        b[i]
    )

# Obiettivo
m.objective = minimize(xsum(distance[i][j] * f[i,j] for (i,j) in A))
m.optimize()
```

**Estrazione soluzione**:
```python
solution_arcs = [(i,j) for (i,j) in A if f[i,j].x > 0.9]
total_distance = m.objective_value
```

## 3. Minimum Spanning Tree

**Definizione**: Albero che connette tutti i nodi con costo minimo totale

**Formulazione Cut Set**:

Per ogni sottoinsieme S ⊂ N (S ≠ ∅, S ≠ N):
```
Σ x_e ≥ 1  dove e attraversa il confine di S
e∈δ(S)
```

**Significato**: Almeno un arco deve attraversare ogni possibile taglio

**Problema**: Numero vincoli esponenziale (2^n - 2)

**Implementazione**:
```python
# Grafo NON diretto (edges)
E = [(i,j) for i in range(k) for j in range(k) 
     if distance[i][j] <= d_max and i < j]  # i<j evita duplicati

# Power set (tutti i sottoinsiemi)
from itertools import combinations
power_set = []
for size in range(1, k):  # escludi ∅ e N
    for subset in combinations(N, size):
        power_set.append(subset)

m = Model("MST")

# Variabili BINARIE
x = {(i,j): m.add_var(var_type=BINARY, name=f'x_{i}_{j}') 
     for (i,j) in E}

# Vincoli cut set
for S in power_set:
    m.add_constr(
        xsum(x[i,j] for (i,j) in E 
             if (i in S and j not in S) or (i not in S and j in S)) 
        >= 1
    )

# Obiettivo
m.objective = minimize(xsum(distance[i][j] * x[i,j] for (i,j) in E))
m.optimize()
```

**Soluzione**: Albero senza cicli che connette tutti i nodi

## 4. Travelling Salesman Problem

**Problema**: Tour chiuso che visita ogni nodo esattamente una volta con distanza minima

**Vincoli base** (IN + OUT = 2):
```python
for i in N:
    m.add_constr(
        xsum(x[j,i] for j in N if j<i and (j,i) in E) +  # IN
        xsum(x[i,j] for j in N if i<j and (i,j) in E)    # OUT
        == 2
    )
```

**Problema**: Vincoli base permettono **subtour** (cicli separati)!

Esempio NON valido:
```
Subtour 1: 0 → 2 → 4 → 0
Subtour 2: 1 → 3 → 5 → 1
```

**Subtour Elimination Constraints**:

Per ogni sottoinsieme S ⊂ N:
```
Σ x_ij ≤ |S| - 1
i,j∈S
```

**Intuizione**: k nodi in ciclo hanno k archi, vincolo impone ≤ k-1 → rompe il ciclo

**Approccio iterativo** (Lazy Constraints):
```
1. Risolvi senza vincoli subtour
2. Se soluzione ha subtour:
   - Identifica i subtour
   - Aggiungi vincoli per romperli
3. Ripeti fino a tour unico
```

**Implementazione**:
```python
m = Model("TSP")
x = {(i,j): m.add_var(var_type=BINARY) for (i,j) in E}

# Vincoli flusso
for i in N:
    m.add_constr(xsum(...) == 2)  # come sopra

# Obiettivo
m.objective = minimize(xsum(distance[i][j] * x[i,j] for (i,j) in E))

# Loop iterativo
while True:
    m.optimize()
    subtours = find_subtours(solution)  # funzione per trovare cicli
    
    if len(subtours) == 1:  # tour unico trovato
        break
    
    # Aggiungi vincoli per rompere subtour
    for subtour in subtours:
        m.add_constr(
            xsum(x[i,j] for (i,j) in subtour_edges) 
            <= len(subtour) - 1
        )
```

**Esempio aggiunta manuale**:
```python
# Per subtour {0, 2, 4}
m.add_constr(x[0,2] + x[2,4] + x[4,0] <= 2)

# Per subtour {1, 3, 5, 6}
m.add_constr(x[1,3] + x[3,5] + x[5,6] + x[6,1] <= 3)
```

## 5. Confronto Formulazioni

| Problema | Variabili | Grafo | Vincoli | N° vincoli | Approccio |
|----------|-----------|-------|---------|------------|-----------|
| **Shortest Path** | Continue | Diretto | Flusso | O(n) | Diretto |
| **MST** | Binarie | Non dir. | Cut set | O(2^n) | Tutti upfront |
| **TSP** | Binarie | Non dir. | Flusso + Subtour | O(2^n) | Iterativo |

## 6. Sintassi MIP Essenziale

**Operazioni base**:
```python
m = Model()                              # crea modello
x = m.add_var()                          # variabile continua ≥0
y = m.add_var(var_type=BINARY)          # variabile binaria
m.add_constr(expr == value)             # vincolo
m.objective = minimize(expr)             # obiettivo
m.optimize()                             # risolve
value = m.objective_value                # valore ottimo
x_val = x.x                              # valore variabile
```

**Sommatorie**:
```python
xsum(x[i,j] for (i,j) in A if condition)  # come list comprehension
```

## 7. Note Implementative

**Archi direzionali vs non direzionali**:
```python
# Diretto (arcs): entrambe le direzioni separate
A = [(i,j) for i in range(k) for j in range(k) if i != j]

# Non diretto (edges): solo una direzione
E = [(i,j) for i in range(k) for j in range(k) if i < j]
```

**Estrazione soluzione**:
```python
# Soglia > 0.9 invece di == 1 (precisione numerica)
selected = [(i,j) for (i,j) in A if x[i,j].x > 0.9]
```

**Peer Review**: Attività opzionale di valutazione reciproca soluzioni