# Lezione 12 - Introduzione ai Problemi di Flusso su Grafi

## 1. Esercizio Shortest Path Tree

**Grafo**: Cicli presenti, costi tutti ≥ 0 → **Dijkstra**

**Esecuzione**: Via applet interattivo
- Estrazione nodi per etichetta crescente
- Archi verdi = SPT finale

**Post-ottimalità arco DF**:
- D(D) = 6, D(F) = 13
- Differenza: 13 - 6 = 7
- **Condizione**: C(D,F) ≥ 7 (arco fuori SPT)
- **Range**: [7, +∞)

**Regola generale**:
- Archi nello SPT: D(j) - D(i) = C(i,j) (tesi)
- Archi fuori SPT: D(j) - D(i) ≤ C(i,j) (condizione ottimalità)

## 2. Puzzle: Il Ponte di Notte

**Problema**: 4 persone attraversano ponte con una torcia
- Max 2 persone per volta
- Tempo = velocità del più lento
- Qualcuno deve riportare torcia

**Tempi**: 1, 2, 5, 10 minuti

**Richiesta**: Modellare come shortest path su grafo opportuno (usare programmazione dinamica)

## 3. Problemi di Flusso - Novità

**Differenze con Shortest Path/MST**:
| Prima | Ora |
|-------|-----|
| Solo costi | Costi + **Capacità** |
| Quantità irrilevante | **Quantità importa** |
| Minimizzare | **Massimizzare flusso** |

**Nuovo concetto**: Capacità u(i,j) = limite superiore al flusso sull'arco

**Applicazioni**:
- Piani di evacuazione
- Scheduling
- Yield management
- Job assignment

## 4. Maximum Flow Problem
è un problema globale (non locale), trasformando il problema da una scelta di percorsi a una distribuzione globale del flusso nella rete, con l’obiettivo di massimizzare il flusso totale.

**Input**:
- Grafo G = (N, A)
- Source S, Sink T
- Capacità u(i,j) ≥ 0 per ogni arco

**Obiettivo**: Massimizzare flusso da S a T

**Vincoli**:
```
1. Capacità: x(i,j) ≤ u(i,j)  ∀(i,j)

2. Conservazione flusso (nodi interni):
   Σ x(j,i) - Σ x(i,j) = 0  ∀i ∉ {S,T}
   (entrante = uscente)
```

**Funzione obiettivo** (equivalenti = sono uguali grazie alla conservazione del flusso):
```
max Σ x(j,T)    (flusso entrante in T)
max Σ x(S,j)    (flusso uscente da S)
```

## 5. Modellazione: Piano di Evacuazione

**Grafo**:
- Nodo S: sorgente fittizia
- Nodi interni: stanze, corridoi, scale, uscite
- Nodo T: punto di raccolta

**Archi dummy** da S alle stanze:
```
u(S, stanza_i) = numero persone nella stanza
```

**Capacità interne**: Capacità fisica delle sezioni

**Interpretazione soluzione**:
```
F = max flow totale
Persone totali = Σ capacità archi da S

Se F = Persone totali → ✅ Evacuazione possibile
Se F < Persone totali → ❌ Servono più uscite
```

## 6. Modellazione: Scheduling

**Problema**: M macchine identiche, N job con p_j, r_j, d_j

**Grafo bipartito**:
```
Lato sinistro: Job 1, 2, ..., N
Lato destro: Intervalli temporali [t, t+1]
```

**Archi**: Job j → [t,t+1] se [t,t+1] ⊆ [r_j, d_j]

**Capacità**:
```
S → Job j: u = p_j (tempo processamento)
Job → Intervallo: u = 1 (max 1 unità per intervallo)
Intervallo → T: u = M (M macchine disponibili)
```

**Interpretazione**:
```
F = Σ p_j → ✅ Scheduling fattibile
F < Σ p_j → ❌ Non fattibile
```

## 7. Minimum Cut Problem

**Scenario duale**: Terrorista vuole interrompere flusso

**Taglio (S,T)-cut**: Partizione N = N_S ∪ N_T
- S ∈ N_S, T ∈ N_T

**Capacità taglio**:
```
U(N_S, N_T) = Σ u(i,j)
              i∈N_S, j∈N_T
```

**Problema**: min U(N_S, N_T) - minimizzare capacità taglio

## 8. Proprietà Flusso attraverso Taglio

**Teorema fondamentale**: Per qualsiasi taglio:
```
F(N_S, N_T) = Σ x(i,j) - Σ x(j,i) = F
              →          ←
```

**Flusso invariante**: Attraverso ogni taglio passa sempre lo stesso flusso totale F!

**Disuguaglianza fondamentale**:
```
F ≤ U(N_S, N_T)  per ogni taglio
```

**Conseguenza**: Se F = U per qualche taglio → **soluzione ottimale**!

## 9. Grafo Residuo

**Definizione**: G_R(x) = (N, A⁺ ∪ A⁻)

**Archi verdi A⁺** (push forward):
```
(i,j) ∈ A⁺ se x(i,j) < u(i,j)
Posso aumentare il flusso
```

**Archi rossi A⁻** (pull back):
```
(j,i) ∈ A⁻ se x(i,j) > 0
Posso diminuire il flusso (arco inverso!)
```

**Capacità residua**:
```
Verde: r(i,j) = u(i,j) - x(i,j)
Rosso: r(j,i) = x(i,j)
```

**Cammino aumentante**: S → T nel grafo residuo
- Se esiste → posso migliorare il flusso
- Se NON esiste → **soluzione ottimale**!

**Esempio aumento**:
```
Cammino: S → 1 → 3 → 4 (rosso!) → T
θ = min{capacità residue} = 1

Aggiornamento:
- Archi verdi: x += θ
- Archi rossi: x -= θ (pull back)
```

## 10. Algoritmo Ford-Fulkerson

**Pseudocodice**:
```
1. x(i,j) = 0  ∀(i,j)

2. REPEAT:
   a) Costruisci G_R(x)
   b) Graph Search S → T in G_R
   c) IF percorso P trovato:
      - θ = min capacità residue su P
      - Aggiorna flussi lungo P
   UNTIL nessun percorso S → T

3. RETURN x
```

**Flow Augmentation**:
```
θ = min{r(i,j) : (i,j) ∈ P}

Per ogni (i,j) in P:
  Se verde: x(i,j) += θ
  Se rosso: x(j,i) -= θ (inverso!)
```

**Correttezza**: Quando termina (no percorso in G_R):
- Esiste taglio (N_S, N_T) con S ∈ N_S, T ∈ N_T
- Archi da N_S → N_T: tutti saturi (x = u)
- Archi da N_T → N_S: tutti vuoti (x = 0)
- **F = U** → ottimo certificato!

**Max-Flow Min-Cut Theorem** (anticipazione):
```
max F = min U(N_S, N_T)
```

## Prossima Lezione
- Analisi complessità Ford-Fulkerson
- Strategie selezione cammino aumentante
- Algoritmi più efficienti (Edmonds-Karp, Dinic)