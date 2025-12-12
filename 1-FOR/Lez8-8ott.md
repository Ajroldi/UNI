# Ricerca Operativa - Lezione 8

**Data:** 8 Ottobre | **Prof. Federico Malucelli** | *Politecnico di Milano*

## Argomenti della Lezione

1. Problema OD Matrix Matching
2. Algoritmo di Visita del Grafo
3. Formulazione Shortest Path
4. Problema di Pianificazione Rinnovo
5. Shortest Path per Grafi Aciclici

## 1. Problema OD Matrix Matching

### Contesto
**Sistema di pedaggio autostradale:** Ricostruire accoppiamenti (Origine, Destinazione) per beep "orfani" che non matchano correttamente.

**Dati:**
- Insieme I: beep di ingresso con timestamp tᵢ
- Insieme O: beep di uscita con timestamp tⱼ
- |I| = |O| = n

**Obiettivo:** Minimizzare il massimo scarto temporale (funzione bottleneck).

### Modellazione con Grafo Bipartito

**Variabili:** `Xᵢⱼ ∈ {0,1}` (1 se accoppiamo ingresso i con uscita j)

**Parametro:** `Δᵢⱼ = |tⱼ - tᵢ|` (differenza temporale)

### Linearizzazione Funzione Bottleneck

**Obiettivo originale (non lineare):**
```
min max{Δᵢⱼ · Xᵢⱼ : i ∈ I, j ∈ O}
```

**Linearizzazione con variabile dummy D:**
```
min D
s.t. D ≥ Δᵢⱼ · Xᵢⱼ  ∀i ∈ I, j ∈ O
```

Quando Xᵢⱼ = 1: D ≥ Δᵢⱼ
Quando Xᵢⱼ = 0: D ≥ 0 (sempre soddisfatto)

### Formulazione Completa

```
min D

s.t.:
    D ≥ Δᵢⱼ · Xᵢⱼ              ∀i ∈ I, j ∈ O
    Σⱼ∈O Xᵢⱼ = 1                ∀i ∈ I (ogni ingresso usato una volta)
    Σᵢ∈I Xᵢⱼ = 1                ∀j ∈ O (ogni uscita usata una volta)
    Xᵢⱼ ∈ {0,1}
    D ≥ 0
```

**Tipo:** Problema di assegnamento con funzione minimax.

## 2. Algoritmo di Visita del Grafo

### Strutture Dati

**Vettore predecessori p:**
```
pᵢ = 0    se i non raggiunto
pᵢ = j    se raggiunto i da j
pᵢ = s    se i = s (sorgente)
```

**Coda Q:** Nodi da cui continuare la ricerca.

### Pseudocodice

```
1. Inizializzazione:
   pᵢ ← 0  ∀i ∈ N
   pₛ ← s
   Q ← {s}

2. REPEAT until Q = ∅:
   3. Estrai nodo i da Q
   4. FOR each (i,j) ∈ FS(i):
      5. IF pⱼ = 0:
         6. pⱼ ← i
         7. Q ← Q ∪ {j}

8. RETURN p
```

### Complessità

**Inizializzazione:** O(n)

**Ciclo principale:** Ogni nodo entra in Q al massimo una volta → O(n) iterazioni

**For interno:** Visita FS(i) per ogni nodo → Σᵢ |FS(i)| = m

**Totale:** O(n + m) = **O(m)** (lineare negli archi!)

### Proprietà Percorso vs Taglio

**Teorema fondamentale:** Dato grafo G = (N, A) e nodi s, t:

**O** esiste un percorso da s a t

**OPPURE**

esiste un taglio (Nₛ, Nₜ) con s ∈ Nₛ, t ∈ Nₜ e tutti gli archi vanno da Nₜ a Nₛ.

**Certificati polinomiali:**
- SÌ: il percorso (O(n))
- NO: il taglio (O(n))

Entrambi verificabili in tempo polinomiale → problema "facile" (classe P).

## 3. Formulazione Shortest Path

### Problema
Trovare il percorso di costo minimo da sorgente s a destinazione t.

**Input:** Grafo G = (N, A), costi cᵢⱼ, nodi s e t

**Variabili:** `Xᵢⱼ ∈ {0,1}` (1 se arco nel percorso)

### Formulazione Completa

```
min Σ₍ᵢ,ⱼ₎∈A cᵢⱼ · Xᵢⱼ

s.t.:
    Σⱼ:(s,j)∈FS(s) Xₛⱼ = 1                              (sorgente)
    
    Σᵢ:(i,t)∈BS(t) Xᵢₜ = 1                              (destinazione)
    
    Σⱼ:(j,i)∈BS(i) Xⱼᵢ - Σⱼ:(i,j)∈FS(i) Xᵢⱼ = 0         ∀i ∈ N\{s,t}
    
    Xᵢⱼ ∈ {0,1}
```

**Vincoli di flusso:**
- Dalla sorgente esce 1 unità
- Nella destinazione entra 1 unità
- Nei nodi interni: flusso entrante = flusso uscente

**Struttura:** 1 variabile per arco (m variabili), 1 vincolo per nodo (n vincoli).

### Problema dei Cicli Negativi

Se esistono cicli a costo negativo raggiungibili da s:
- Possiamo percorrerli infinite volte
- Costo ottimo → -∞
- Problema **mal definito**

**Assunzioni per risolvibilità:**
1. Tutti i costi non-negativi: `cᵢⱼ ≥ 0`
2. Grafo aciclico (DAG)

## 4. Problema di Pianificazione Rinnovo

### Contesto
Piano di rinnovo auto su 5 anni minimizzando costo netto totale.

**Dati:**
- Costi acquisto (inflazione): Anno 0: $12K, Anno 1: $13K, ...
- Manutenzione e rivendita per anni di possesso

### Modellazione come Grafo

**Nodi:** Istanti di decisione (0, 1, 2, 3, 4, 5)

**Archi:** Decisioni "tenere auto da anno i ad anno j"
```
Arco (i,j): Compro anno i, vendo anno j
Durata possesso: k = j - i anni
```

**Costo arco (i,j):**
```
cᵢⱼ = Costo_acquisto_i + Manutenzione_totale - Ricavo_vendita_j
```

### Esempio

**Arco (0,1):** Compro anno 0, vendo anno 1 (1 anno)
```
c₀₁ = 12,000 + 2,000 - 7,000 = 7,000
```

**Arco (0,2):** Compro anno 0, vendo anno 2 (2 anni)
```
c₀₂ = 12,000 + (2,000 + 4,000) - 6,000 = 12,000
```

### Soluzione

Ogni **percorso** da 0 a 5 = piano di rinnovo ammissibile

**Percorso più breve** = piano ottimo!

Applicare algoritmo shortest path su questo DAG.

## 5. Shortest Path per Grafi Aciclici (DAG)

### Ordinamento Topologico

**Definizione:** Numerazione dei nodi tale che:
```
Se (i,j) ∈ A  ⟹  i < j
```

Tutti gli archi "vanno avanti" nella numerazione.

**Algoritmo:**
```
1. numero ← 1
2. S ← nodi senza archi entranti

3. WHILE S ≠ ∅:
   4. Seleziona i da S
   5. Assegna i ← numero
   6. numero ← numero + 1
   7. Rimuovi i e archi uscenti
   8. Aggiorna S

9. IF numero ≤ n: "Grafo ciclico"
10. ELSE: return numerazione
```

**Complessità:** O(n + m)

### Algoritmo con Regola Induttiva

**Etichetta dᵢ:** Lunghezza percorso più breve da s a i

**Base:** `dₛ = 0`

**Passo induttivo:** Per nodo i (in ordine topologico):
```
dᵢ = min{dⱼ + cⱼᵢ : (j,i) ∈ BS(i)}
```

**Perché funziona:** Grazie all'ordinamento topologico, quando calcoliamo dᵢ, tutti i dⱼ con j < i sono già stati calcolati.

### Algoritmo Push-Forward

```
1. Inizializzazione:
   pᵢ ← 0, dᵢ ← +∞  ∀i ∈ N
   pₛ ← s, dₛ ← 0

2. FOR i = 1 TO n-1:  (ordine topologico)
   3. FOR each (i,j) ∈ FS(i):
      4. IF dᵢ + cᵢⱼ < dⱼ:
         5. dⱼ ← dᵢ + cᵢⱼ
         6. pⱼ ← i

7. RETURN p, d
```

**Differenze approccio Pull vs Push:**
- **Pull:** Guarda backward star (BS), "da dove arrivo?"
- **Push:** Guarda forward star (FS), "dove posso andare?"

Entrambi equivalenti, stessa complessità.

### Complessità

**Inizializzazione:** O(n)

**Ciclo esterno:** n-1 iterazioni

**Ciclo interno:** Σᵢ |FS(i)| = m

**Totale:** O(n + m) = **O(m)** (tempo lineare!)

### Esempio di Esecuzione

Grafo: 1 → 2 → 3 → 4 → 5 → 6 → 7 (con archi multipli)

**Etichette finali:** [0, 1, 2, 4, 4, 7, 6]

**Percorso 1→7:** 1 → 2 → 4 → 5 → 7 (costo 6)

**Risultato:** L'algoritmo costruisce un **albero dei percorsi più brevi** da s a tutti gli altri nodi.

## Concetti Chiave

### Tecniche di Modellazione

**Linearizzazione minimax:**
```
min max_i fᵢ(x)  →  min D  s.t.  D ≥ fᵢ(x)  ∀i
```

**Decisioni temporali come DAG:**
- Nodi = istanti di tempo
- Archi = decisioni che collegano istanti
- Percorso = sequenza di decisioni

### Algoritmi su Grafi

**Visita del grafo:**
- O(m) tempo lineare
- Certificati polinomiali per percorsi e tagli
- Dualità percorso-taglio

**Shortest path su DAG:**
- O(m) senza strutture dati complesse
- Richiede ordinamento topologico
- Programmazione dinamica con regola induttiva

### Vincoli di Flusso

**Struttura fondamentale:**
- Conservazione del flusso ai nodi interni
- Iniezione alla sorgente (+1)
- Estrazione alla destinazione (-1)
- Unifica casi "sul percorso" e "fuori dal percorso"

### Problemi Applicativi

**OD Matrix:** Assegnamento con bottleneck, grafo bipartito

**Rinnovo auto:** Decisioni multi-periodo, DAG naturale

**Satelliti:** Sequenziamento immagini, riduzione a TSP

## Preparazione Prossima Lezione

**Argomenti previsti:**
- Formulazione alternativa shortest path
- Significato duale delle etichette
- Algoritmo di Dijkstra (costi non-negativi)
- Algoritmo di Bellman-Ford (costi generali)
- Criteri di selezione algoritmo ottimale

**Prerequisiti:**
- Comprensione vincoli di flusso
- Ordinamento topologico
- Proprietà percorso-taglio