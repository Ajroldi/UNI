# Lezione 15 - Introduzione alla Programmazione Lineare

## Esempio Introduttivo: Produzione Porte e Finestre

**Problema**: Massimizzare il ricavo producendo porte (30€) e finestre (50€) con risorse limitate.

**Variabili**: $x_D$ = porte, $x_W$ = finestre

**Formulazione**:
```
max  30x_D + 50x_W
s.t. x_D ≤ 4         (Fabbro)
     2x_W ≤ 12       (Falegname)
     3x_D + 2x_W ≤ 18 (Assemblatore)
     x_D, x_W ≥ 0
```

**Forma matriciale**: $\max \mathbf{c}^T\mathbf{x}$ con $\mathbf{Ax} \leq \mathbf{b}$, $\mathbf{x} \geq 0$

## Metodo Geometrico di Risoluzione

1. **Disegnare la regione ammissibile** usando i gradienti (opposti ai vincoli ≤)
2. **Tracciare linee di livello** della funzione obiettivo
3. **Spostare la linea** lungo il gradiente fino all'ultimo punto di intersezione
4. **Soluzione ottima**: sul vertice $(x_D^*, x_W^*) = (2, 6)$ con $Z^* = 360€$

**Proprietà chiave**: L'ottimo si trova sempre su un **vertice** del poliedro.

## Concetti di Convessità

**Insieme convesso**: Il segmento tra due punti qualsiasi appartiene all'insieme
- Formula: $\lambda x + (1-\lambda)y \in S$ per $\lambda \in [0,1]$

**Poliedro convesso**: Definito da un numero **finito** di disequazioni lineari
- $P = \{x \in \mathbb{R}^n : \mathbf{Ax} \leq \mathbf{b}\}$

**Cono**: Se $x \in C$ allora $\lambda x \in C$ per ogni $\lambda \geq 0$

**Cono convesso**: $\lambda x + \mu y \in C$ per $\lambda, \mu \geq 0$

**Cono poliedrico**: 
- Rappresentazione per disequazioni: $\{x: \mathbf{Ax} \leq 0\}$
- Rappresentazione per generatori: $\{x: x = \sum_i \lambda_i y_i, \lambda_i \geq 0\}$

## Caratterizzazione dei Vertici (3 definizioni equivalenti)

### 1. Punto Estremo
Non può essere scritto come combinazione convessa di altri due punti distinti in P.
- ❌ Problema: non computazionale

### 2. Vertice
Esiste un vettore $\mathbf{c}$ tale che $\mathbf{c}^Tx > \mathbf{c}^Ty$ per ogni $y \in P$, $y \neq x$
- È il punto ottimo per qualche funzione obiettivo
- ❌ Problema: c non è unico

### 3. Soluzione di Base
Esistono **n vincoli attivi linearmente indipendenti** in x
- $I(x) = \{i: a_i x = b_i\}$ (indici vincoli attivi)
- Verifica: $\det(A_{I(x)}) \neq 0$
- ✅ **Usata negli algoritmi** perché computazionale

## Teorema di Decomposizione

Ogni poliedro convesso P si decompone come:
$$P = Q + C$$

dove:
- **Q** = politopo (combinazione convessa di vertici, parte finita)
- **C** = cono (direzioni che vanno all'infinito)

**Conseguenza**: Se l'ottimo è finito, esiste sempre un **vertice ottimo**.

## Teorema dell'Iperpiano Separatore

Dato un insieme convesso S e un punto C ∉ S, esiste sempre un iperpiano che li separa:
$$\mathbf{c}^TC > \mathbf{c}^Ty \quad \forall y \in S$$

Questo teorema è alla base della **convergenza** dell'algoritmo del simplesso.

## Esercizio: Edmonds-Karp

Applicazione dell'algoritmo con regola del **minimo numero di archi** (BFS).

**Iterazioni**:
1. Cammino S→1→2→3→T (4 archi), θ=2, F=2
2. Cammino S→1→6→7→3→T (5 archi), θ=1, F=3
3. Cammino S→4→5→2→1→6→7→3→T (8 archi, con arco backward 2→1), θ=2, F=5

**Taglio ottimo**: $N_S = \{S,4,5,2\}$, $N_T = \{1,6,7,3,T\}$
- Capacità taglio = Flusso massimo = 5

**Analisi post-ottimalità**:
- Aumentare capacità archi $N_T \to N_S$ (es. 1→2): **nessun effetto**
- Aumentare capacità archi $N_S \to N_T$: **migliora il flusso**