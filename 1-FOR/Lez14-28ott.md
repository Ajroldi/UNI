# Lezione 14 - Ford-Fulkerson, Min Cost Flow e Intro LP

## 1. Puzzle del Ponte (Dynamic Programming)

**Problema**: 4 persone attraversano ponte con 1 torcia
- Max 2 persone per volta
- Qualcuno deve riportare la torcia

**Soluzione**: Grafo di **stati**

**Notazione stato**: `IJ / HK ★`
- IJ = persone a sinistra
- HK = persone a destra  
- ★ = posizione torcia

**Stati**:
```
Inizio: 1234 / ∅ ★
Livello 1: Coppie attraversano → 34/12★, 24/13★, ...
Livello 2: Uno torna → 134/2, 234/1, ...
...
Fine: ∅ / 1234 ★
```

**Ottimizzazione**: Solo coppie vanno da sinistra, singoli tornano (evita mosse inutili)

**Costi archi**: Tempo attraversamento = max(tempo persone)

**Soluzione ottima** (tempi 1,2,5,10):
```
1,2 attraversano (2) → 1 torna (1) → 3,4 attraversano (10) 
→ 2 torna (2) → 1,2 attraversano (2)
Totale: 17 minuti
```

## 2. Organizzazione Feste

**Problema**: Organizzare feste con giocatori famosi
- Ogni festa i può svolgersi in serate S(i)
- N giocatori disponibili
- Max 1 giocatore per festa, non sovrapposizioni

**Riduzione a Max Flow**:

**Grafo bipartito**:
```
S → [Feste] → [Serate] → T
```

**Capacità**:
- S → Festa: **1** (ogni festa max 1 volta)
- Festa → Serata: **1** (o ∞, irrilevante)
- Serata → T: **N** (max N feste in parallelo = numero giocatori)

**Archi**: Festa i → Serata j se j ∈ S(i)

## 3. Assignment (Matching)

**Problema**: Assegnare lavoratori a lavori (1-a-1)

**NB**: 1 unità di flusso = 1 assegnamento valido

x_ij = 1 se e solo se il lavoratore i è assegnato al lavoro j,
x_ij = 0 altrimenti.

**Matching**: Sottoinsieme M di archi dove nessuna coppia condivide nodi

**Formulazione**:
```
max Σ x_ij

Vincoli:
Σj x_ij ≤ 1  ∀ lavoratore i (un lavoro per lavoratore)
Σi x_ij ≤ 1  ∀ lavoro j (un lavoratore per lavoro)
```

**Riduzione a Max Flow**:
```
        Capacità 1
S → [Lavoratori] → [Lavori] → T
```

**Trucco**: Vincolo aggregato (Σ ≤ 1) tradotto in capacità sull'arco S→i e j→T

**Archi backward**: Permettono **riorganizzazione** assegnamenti esistenti

Esempio: S → 4 → A(verde) → 2(rosso) → F → T
- Assegna 4 ad A
- Cancella assegnamento 2-A
- Assegna 2 a F

## 4. Ford-Fulkerson: Complessità

**Algoritmo**:
```
1. x = 0
2. REPEAT:
   - Costruisci G_R(x)
   - Graph search S → T
   - Se trovato cammino P:
     * θ = min{r(i,j) : (i,j) ∈ P}
     * Augmenta flusso
   UNTIL no cammino
```

**Capacità residua**:
```
r(i,j) = { u(i,j) - x(i,j)  se verde (aumenta)
         { x(j,i)           se rosso (diminuisci)
```

**Esempio patologico**:
```
     1000
S ───→ 1
│      │ 1 (piccola!)
│1000  │
2 ───→ T
  1000
```

- Ottimo: 2000 (due cammini diretti)
- Se si sceglie sempre S→1→2→T: **2000 iterazioni**! (θ=1 ogni volta)

**Edmonds-Karp**: Seleziona cammino con **min numero archi** (BFS)
- **Complessità**: O(m²n)
- Bipartito con capacità unitarie: O(n³)

**Migliore teorico**: O(n³) con metodi diversi (push-relabel)

## 5. Minimum Cost Flow

**Motivazione**: Combinare flusso + costo, cioè consiste nel spostare flusso, rispettando le capacità, spendendo il meno possibile. Ogni nodo deve ricevere o mandare la quantità richiesta e il costo totale del trasporto va minimizzato.

| Problema | Flusso | Costo |
|----------|--------|-------|
| Shortest Path | No (impl. 1) | Sì |
| Max Flow | Sì | No |
| **Min Cost Flow** | **Sì** | **Sì** |

**Esempi**:

### Raccolta Rifiuti
- Siti con quantità fisse da rimuovere
- Destinazioni (discarica, inceneritore, riciclaggio)
- Costi trasporto + capacità collegamenti

### Costruzione Strada
- Colline: eccesso materiale (b<0)
- Depressioni: bisogno materiale (b>0)
- Spostare materiale a costo minimo
- Esempio: Pedemontana = 35M m³ (camion Milano-Mosca!)

**Formulazione**:

**Dati**:
- Bilanci b_i (b_i < 0 offre, b_i > 0 richiede, b_i = 0 transito)
- Costi unitari c_ij
- Capacità u_ij
- **Assunzione**: Σ b_i = 0 (sistema chiuso)

**Problema**:
```
min Σ c_ij · x_ij

s.t.:
  Σ x_ji - Σ x_ij = b_i  ∀i (bilancio)
  0 ≤ x_ij ≤ u_ij        ∀(i,j) (capacità)
```

**Nodo dummy**: Se quantità destinazioni incognite, aggiungi T con archi da destinazioni

## 6. Riduzioni a Min Cost Flow

### Max Flow → Min Cost Flow

**Trucco**: Aggiungi arco **T→S**
```
Bilanci: tutti = 0
Costi: originali = 0, c_TS = -1
Capacità: originali + u_TS = +∞
```

**Logica**: Massimizzare flusso su T→S (costo negativo) = max flow

### Shortest Path → Min Cost Flow

**Bilanci**:
```
b_S = -1 (esce 1 unità)
b_T = +1 (arriva 1 unità)
altri = 0
```

**Costi**: originali, **no capacità**

### Shortest Path Tree → Min Cost Flow

**Per tutti i cammini da S**:
```
b_S = -(n-1) (escono n-1 unità)
b_i = +1 ∀i≠S (arriva 1 unità)
```

**Flusso x_ij**: Numero di cammini che usano arco (i,j)

## 7. Intro Programmazione Lineare

**Nuovo capitolo**: Generalizzazione problemi ottimizzazione

**Forma generale**:
```
max f(x)
s.t. g_i(x) ≤ b_i
```

**Programmazione Lineare**: f e g_i **lineari**

**Notazione matriciale**:
```
max c^T x
s.t. Ax ≤ b
```

Dove:
- c = vettore coefficienti obiettivo
- A = matrice vincoli (m×n)
- b = vettore termini noti
- x = variabili **continue** (no discrete!)

**Argomenti**:
1. Caratterizzazione ottimalità (dualità)
2. **Interpretazione geometrica** (cruciale!)
3. Algoritmo Simplex (cenni)
4. Analisi post-ottimalità

**Prerequisiti**:
- Notazione matriciale
- Sistemi disequazioni
- Interpretazione geometrica funzioni/vincoli lineari
- Gradiente
- Dualità

## Tabelle Riassuntive

**Confronto problemi**:
| Problema | Costo | Flusso | Capacità | Bilanci |
|----------|-------|--------|----------|---------|
| Shortest Path | Sì | Impl. 1 | No | -1,+1,0 |
| Max Flow | No | Sì | Sì | Tutti 0 |
| Min Cost Flow | Sì | Sì | Sì | Σ=0 |

**Complessità**:
| Algoritmo | Complessità | Note |
|-----------|-------------|------|
| Ford-Fulk. naive | Pseudo-pol. | Dipende da U |
| Edmonds-Karp | O(m²n) | BFS |
| E-K bipartito | O(n³) | Cap. unit. |
| Migliore | O(n³) | Push-relabel |