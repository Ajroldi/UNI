# Lezione 11 - 21 Ottobre: Esercizi Shortest Path e Spanning Tree

## 📑 Indice

1. [Puzzle della Settimana: Scaling dei Costi](#puzzle-scaling) `00:01:03 - 00:07:51`
   - [Controesempio con 3 Nodi](#controesempio)
   - [Perché il Trick Non Funziona](#perche-non-funziona)
   - [Quando il Trick Funziona](#quando-funziona)
2. [Esercizio: Minimum Spanning Tree](#esercizio-mst) `00:07:51 - 00:26:08`
   - [Applicazione Algoritmo di Kruskal](#kruskal)
   - [Analisi Post-Ottimalità: Arco 5-7](#post-opt-57)
   - [Analisi Post-Ottimalità: Arco 3-6](#post-opt-36)
3. [Modellazione: Viaggio con Rifornimenti](#viaggio-rifornimenti) `00:26:08 - 00:32:14`
   - [Formulazione del Problema](#formulazione-viaggio)
   - [Costruzione del Grafo](#costruzione-grafo-viaggio)
   - [Costo degli Archi](#costo-archi-viaggio)
4. [Modellazione: Raccolta Rifiuti Industriali](#raccolta-rifiuti) `00:32:14 - 00:51:04`
   - [Descrizione del Problema](#descrizione-rifiuti)
   - [Grafo Logico vs Grafo Fisico](#grafo-logico)
   - [Definizione Archi e Vincoli](#archi-vincoli-rifiuti)
   - [Costo degli Archi](#costo-archi-rifiuti)
5. [Quiz: Scelta dell'Algoritmo](#quiz-algoritmi) `00:53:35 - 01:01:42`
   - [Grafo Aciclico](#quiz-aciclico)
   - [Grafo con Cicli e Costi Non Negativi](#quiz-dijkstra)
   - [Grafo con Cicli e Costi Negativi](#quiz-lq)
6. [Esercizio: SPT con Radice 1](#esercizio-spt) `01:02:17 - 01:12:39`
   - [Selezione dell'Algoritmo](#selezione-algoritmo)
   - [Esecuzione SPT-LQ](#esecuzione-lq)
   - [Risultato: Shortest Path Tree](#risultato-spt)
7. [Scaling dei Costi per Cambio Radice](#scaling-cambio-radice) `01:13:12 - 01:29:28`
   - [Problema: SPT con Radice 6](#problema-radice6)
   - [Trasformazione dei Costi](#trasformazione-costi)
   - [Dimostrazione Equivalenza](#dimostrazione-equivalenza)
   - [Vantaggi della Tecnica](#vantaggi-scaling)

---

## <a name="puzzle-scaling"></a>1. Puzzle della Settimana: Scaling dei Costi

### <a name="controesempio"></a>1.1 Controesempio con 3 Nodi

`00:01:03 - 00:03:38`

Buongiorno, oggi faremo principalmente esercizi sui problemi di shortest path e anche uno sugli spanning tree, e se abbiamo tempo introduciamo il prossimo argomento.

**Puzzle della scorsa settimana**: Dato che avere costi negativi in un grafo può creare problemi (cicli a costo negativo) e richiede algoritmi meno efficienti di Dijkstra, **è possibile scalare i costi del grafo in modo che tutti gli archi abbiano lunghezza non negativa, preservando l'equivalenza del problema?**

**Risposta: NO**

È sufficiente produrre un **controesempio**. Il più piccolo è fatto di **3 nodi**.

**Esempio**:

```
    1 -----(1)-----> 2
     \              /
      \            /
    (0)\          /(-2)
        \        /
         \      /
          v    v
            3
```

- **Shortest path da 1 a 3**: 1 → 2 → 3
- **Costo totale**: 1 + (-2) = **-1**

> **Nota**: Questo è uno dei triangoli della nostra istanza "Dijkstra-Killer".

### <a name="perche-non-funziona"></a>1.2 Perché il Trick Non Funziona

`00:03:38 - 00:06:17`

**Tentativo di scaling**: Sia M = |C_min| = 2 (valore assoluto del costo minimo).

Sommiamo M a tutti i costi degli archi:

```
Nuovi costi:
- Arco (1,2): 1 + 2 = 3
- Arco (2,3): -2 + 2 = 0
- Arco (1,3): 0 + 2 = 2
```

**Nuovo shortest path da 1 a 3**: 1 → 3 diretto
- **Costo**: 2

**Vecchio shortest path 1 → 2 → 3**:
- **Nuovo costo**: 3 + 0 = 3

**Problema**: Lo shortest path è cambiato! ❌

**Motivo**: Il cammino minimo può avere un **numero variabile di archi**.

- Cammino 1 → 2 → 3: **2 archi** → penalità doppia (2M)
- Cammino 1 → 3: **1 arco** → penalità singola (M)

I cammini con **meno archi** sono **meno penalizzati** rispetto a quelli con più archi.

### <a name="quando-funziona"></a>1.3 Quando il Trick Funziona

`00:07:21 - 00:09:41`

**Il trick funziona quando la soluzione contiene un numero FISSO di elementi**.

**Esempio: Minimum Spanning Tree**
- Numero di archi: sempre **n - 1**
- Scalando tutti i costi di una costante, la penalità è **uguale per tutte le soluzioni**
- **Problema equivalente** ✅

**Domanda**: E lo **Shortest Path Tree**?
- Numero di archi: sempre **n - 1**
- Perché il trick **non funziona**?

> **Risposta**: C'è qualcosa nella formulazione che non permette di dire che il numero di elementi nella soluzione è costante. (Vedremo più avanti)

---

## <a name="esercizio-mst"></a>2. Esercizio: Minimum Spanning Tree

### <a name="kruskal"></a>2.1 Applicazione Algoritmo di Kruskal

`00:09:41 - 00:12:49`

**Algoritmo di Kruskal**:
1. Ordinare gli archi in ordine crescente di costo
2. Aggiungere archi alla soluzione, **evitando cicli**

**Esecuzione**:

| Iterazione | Arco | Costo | Azione |
|------------|------|-------|--------|
| 1 | (1,2) | 2 | ✅ Aggiunto |
| 2 | (6,7) | 3 | ✅ Aggiunto |
| 3 | (1,5) | 4 | ✅ Aggiunto |
| 4 | (1,3) | 5 | ✅ Aggiunto |
| 5 | (3,5) | 6 | ❌ Crea ciclo (1-3-5-1) |
| 6 | (3,6) | 7 | ✅ Aggiunto (collega {1,2,3,5} con {6,7}) |
| 7 | (2,3) | 8 | ❌ Crea ciclo (1-2-3-1) |
| 8 | (5,7) | 9 | ❌ Crea ciclo |
| 9 | (5,6) | 10 | ❌ Crea ciclo |
| 10 | (6,4) | 12 | ✅ Aggiunto (4 non connesso) |

**Soluzione**: 6 archi (n = 7 nodi → n-1 = 6 ✅)

### <a name="post-opt-57"></a>2.2 Analisi Post-Ottimalità: Arco 5-7

`00:13:21 - 00:19:01`

**Domanda**: Qual è l'intervallo di costo per l'arco (5,7) che mantiene la soluzione ottimale?

**Arco (5,7) NON appartiene all'albero T**

#### Aumento del costo
Se **aumentiamo** C(5,7):
- Nessun cambiamento
- L'arco era già scartato, lo sarà ancora di più ✅

**Range superiore**: C(5,7) → +∞

#### Diminuzione del costo
Se **diminuiamo** C(5,7), dobbiamo essere attenti.

**Condizione di ottimalità (Cicli)**: Se aggiungiamo un arco fuori dalla soluzione, creiamo un ciclo. L'arco aggiunto deve avere costo ≥ di tutti gli altri archi nel ciclo.

**Ciclo creato da (5,7)**:
- Archi: (1,5), (1,3), (3,6), (6,7)
- Costi: 4, 5, 7, 3

**Condizione**: C(5,7) ≥ max{4, 5, 7, 3} = **7**

**Range finale**: **7 ≤ C(5,7) < +∞**

> **Nota**: Se C(5,7) = 7, abbiamo una soluzione alternativa equivalente (scambio con (3,6)).

### <a name="post-opt-36"></a>2.3 Analisi Post-Ottimalità: Arco 3-6

`00:19:01 - 00:26:08`

**Domanda**: Qual è l'intervallo di costo per l'arco (3,6) che mantiene la soluzione ottimale?

**Arco (3,6) APPARTIENE all'albero T**

#### Diminuzione del costo
Se **diminuiamo** C(3,6):
- Nessun problema
- La soluzione migliora o rimane ottimale ✅

**Range inferiore**: C(3,6) → 0

#### Aumento del costo
Se **aumentiamo** C(3,6), dobbiamo applicare la condizione di ottimalità sui **tagli**.

**Condizione di ottimalità (Tagli)**: Se rimuoviamo un arco dalla soluzione, creiamo un taglio. L'arco rimosso deve avere costo ≤ di tutti gli altri archi che attraversano il taglio.

**Taglio creato rimuovendo (3,6)**:
- Lato A: {1, 2, 3, 5}
- Lato B: {4, 6, 7}

**Archi che attraversano il taglio**:
- (2,4): costo 16
- (3,4): costo 15
- (3,6): costo 7 ✅
- (3,7): costo 13
- (5,6): costo 10
- (5,7): costo 9

**Condizione**: C(3,6) ≤ min{16, 15, 13, 10, 9} = **9**

**Range finale**: **0 ≤ C(3,6) ≤ 9**

> **Nota**: Se C(3,6) = 9, abbiamo una soluzione alternativa equivalente (scambio con (5,7)).

**⚠️ Importante**: Non serve riapplicare l'algoritmo! Basta analizzare le proprietà della soluzione ottimale.

---

## <a name="viaggio-rifornimenti"></a>3. Modellazione: Viaggio con Rifornimenti

### <a name="formulazione-viaggio"></a>3.1 Formulazione del Problema

`00:26:08 - 00:27:50`

**Problema**: Viaggiare da A a B facendo rifornimenti intermedi.

**Ipotesi**:
- Viaggio molto lungo (es. Lecce → Barcelona con camion)
- Costo carburante varia nelle diverse stazioni
- **Ad ogni fermata, facciamo rifornimento completo**

**Obiettivo**: Minimizzare il costo totale del viaggio decidendo:
- Dove fermarsi
- Quanto rifornire (sempre pieno per ipotesi)

### <a name="costruzione-grafo-viaggio"></a>3.2 Costruzione del Grafo

`00:27:50 - 00:29:41`

**Nodi**:
- Nodo 0: Punto di partenza A (serbatoio pieno)
- Nodo 1, 2, ..., n: Stazioni intermedie
- Nodo n+1: Destinazione B (serbatoio pieno)

**Archi**:
- Esiste arco (i, j) con i < j se:
  - Possiamo raggiungere j da i senza rifornimento
  - Consumo totale ≤ capacità serbatoio

**Condizione di fattibilità**:

```
∑(h=i+1 to j) p_h ≤ Q

dove:
- p_h = consumo carburante nel tratto h
- Q = capacità serbatoio
```

### <a name="costo-archi-viaggio"></a>3.3 Costo degli Archi

`00:29:41 - 00:32:14`

**Arco (i,j)** rappresenta:
- Tratta continua di guida da i a j
- Rifornimento completo in j

**Costo dell'arco**:

```
C(i,j) = (∑(h=i+1 to j) p_h) × c_j

dove:
- p_h = consumo carburante nel tratto h
- c_j = costo carburante alla stazione j
```

**Interpretazione**: Il costo è la spesa per il rifornimento in j, calcolata come:
- Consumo totale da i a j × Prezzo alla stazione j

**Soluzione**: Shortest path da 0 a n+1

---

## <a name="raccolta-rifiuti"></a>4. Modellazione: Raccolta Rifiuti Industriali

### <a name="descrizione-rifiuti"></a>4.1 Descrizione del Problema

`00:32:14 - 00:37:55`

**Contesto**: Azienda che raccoglie rifiuti speciali (olio, amianto, ecc.)

**Vincoli**:
- Un solo camion con capacità limitata Q
- Sequenza clienti già fissata (non modificabile)
- Raccolta integrale: non è permesso prelievo parziale
- Deposito periodico in impianto di smaltimento

**Elementi**:
- Deposito: punto di partenza e ritorno
- Clienti: 1, 2, ..., n (ordine fisso)
- Impianto: dove scaricare i rifiuti
- q_i: quantità di rifiuti dal cliente i

**Esempio di soluzione**:
```
Deposito → 1 → 2 → Impianto → 3 → ... → i → Impianto → i+1 → ... → n → Deposito

Vincoli:
- q_1 + q_2 ≤ Q
- q_3 + ... + q_i ≤ Q
- ecc.
```

### <a name="grafo-logico"></a>4.2 Grafo Logico vs Grafo Fisico

`00:37:55 - 00:40:49`

**Problema con grafo fisico**:
- L'impianto è visitato più volte
- Non è un cammino elementare
- Shortest path classico non applicabile

**Soluzione: Grafo Logico**

**Nodi**:
- Nodo 0: Deposito (inizio)
- Nodo 1, 2, ..., n: Clienti
- Nodo n+1: Deposito (fine)

**Rappresentazione**:
```
0 --- 1 --- 2 --- 3 --- ... --- i --- j --- ... --- n --- (n+1)
```

### <a name="archi-vincoli-rifiuti"></a>4.3 Definizione Archi e Vincoli

`00:40:49 - 00:43:22`

**Arco (i,j)** con i < j rappresenta:
- Servire tutti i clienti da i a j-1 consecutivamente
- Viaggio: Impianto → i → i+1 → ... → j-1 → Impianto

**Vincolo di capacità**:
```
∑(h=i to j-1) q_h ≤ Q
```

**Interpretazione**:
- Se esiste arco (0,3): serviamo clienti 1 e 2, poi scarichiamo
- Se esiste arco (3,j): serviamo clienti 3, 4, ..., j-1, poi scarichiamo

### <a name="costo-archi-rifiuti"></a>4.4 Costo degli Archi

`00:43:22 - 00:51:04`

**Viaggio fisico per arco (i,j)**:
```
     s_i      d_i    d_{i+1}       d_{j-2}      s_{j-1}
Impianto → i → i+1 → i+2 → ... → j-1 → Impianto

dove:
- s_i = distanza Impianto → cliente i
- d_h = distanza cliente h → cliente h+1
```

**Costo dell'arco**:
```
L(i,j) = s_i + ∑(h=i to j-2) d_h + s_{j-1}
```

**Correzione** (dopo discussione):
```
L(i,j) = s_i + ∑(h=i to j-2) d_h + s_{j-1}
```

**Soluzione**: Shortest path da 0 a n+1 nel grafo logico garantisce:
- Visita di tutti i clienti
- Rispetto vincolo capacità
- Minimizzazione distanza totale

---

## <a name="quiz-algoritmi"></a>5. Quiz: Scelta dell'Algoritmo

`00:53:35 - 01:01:42`

### <a name="quiz-aciclico"></a>5.1 Grafo Aciclico

`00:55:54 - 00:57:42`

**Domanda**: Qual è l'algoritmo più efficiente per trovare lo Shortest Path Tree con radice 4?

**Opzioni**:
- 🔴 SPT Acyclic
- 🔵 Dijkstra
- 🟡 LQ (Label-Correcting)
- 🟢 Graph Search

**Risposta**: 🔴 **SPT Acyclic**

**Motivo**: Non ci sono cicli → algoritmo più efficiente

**Complessità**: O(m)
- Visita tutti gli archi una sola volta
- Nessun costo computazionale per determinare l'ordine

### <a name="quiz-dijkstra"></a>5.2 Grafo con Cicli e Costi Non Negativi

`00:57:42 - 00:59:53`

**Domanda**: Stesso problema, grafo diverso con cicli. Qual è l'algoritmo più efficiente?

**Risposta**: 🔵 **Dijkstra**

**Motivo**: 
- Ci sono cicli → ❌ SPT Acyclic
- Costi non negativi → ✅ Dijkstra

**Complessità**:
- Con lista non ordinata: **O(n²)**
- Con priority queue (heap): **O(m log n)**

### <a name="quiz-lq"></a>5.3 Grafo con Cicli e Costi Negativi

`00:59:53 - 01:01:42`

**Domanda**: Grafo con cicli E costi negativi. Qual è l'algoritmo più efficiente?

**Opzioni**:
- SPT Acyclic: ❌ (ci sono cicli)
- Dijkstra: ❌ (costi negativi)
- Graph Search: ❌ (non considera i costi)
- LQ: ✅

**Risposta**: 🟡 **SPT-LQ** (Label-Correcting con FIFO)

**Complessità**: **O(mn)**
- Caso peggiore: O(n³) se m = O(n²)

**Vincitore del quiz**: Raoul 🏆

---

## <a name="esercizio-spt"></a>6. Esercizio: SPT con Radice 1

### <a name="selezione-algoritmo"></a>6.1 Selezione dell'Algoritmo

`01:02:17 - 01:03:22`

**Grafo dato**:
- Nodi: 1, 2, 3, 4, 5, 6, 7
- Radice: 1

**Decision Tree**:

1. ❓ Ci sono cicli?
   - Sì: Ciclo 1-2-4-1 ✅
   - → ❌ Non possiamo usare SPT Acyclic

2. ❓ Tutti i costi sono non negativi?
   - No: (2,6) = -3, (4,1) = -1, (4,5) = -1 ✅
   - → ❌ Non possiamo usare Dijkstra

3. **Algoritmo scelto**: **SPT-LQ** (Label-Correcting con FIFO)

### <a name="esecuzione-lq"></a>6.2 Esecuzione SPT-LQ

`01:03:22 - 01:12:39`

**Inizializzazione**:
- D(1) = 0, pred(1) = ∅
- D(i) = +∞, pred(i) = ∅ per i ≠ 1
- Q = {1}

**Iterazioni**:

| Step | Nodo estratto | Arco | Calcolo | Nuovo D | Pred | Q |
|------|---------------|------|---------|---------|------|---|
| 1 | 1 | (1,2) | 0+4 | D(2)=4 | 1 | {2,3} |
| | | (1,3) | 0+2 | D(3)=2 | 1 | |
| 2 | 2 | (2,7) | 4-1 | D(7)=3 | 2 | {3,7,6,4} |
| | | (2,6) | 4-3 | D(6)=1 | 2 | |
| | | (2,4) | 4+3 | D(4)=7 | 2 | |
| 3 | 3 | (3,4) | 2+2 | D(4)=4 | 3 | {7,6,4} |
| | | (3,7) | 2-5 | D(7)=-3 | 3 | |
| 4 | 7 | - | - | - | - | {6,4} |
| 5 | 6 | (6,7) | 1+1=2 | - | - | {4} |
| | | (6,4) | 1+2 | D(4)=3 | 6 | |
| 6 | 4 | (4,1) | 3-1=2 | - | - | {5} |
| | | (4,5) | 3-1 | D(5)=2 | 4 | |
| 7 | 5 | (5,6) | 2+1=3 | - | - | ∅ |
| | | (5,3) | 2+5=7 | - | - | |

**Etichette finali**:
- D(1) = 0
- D(2) = 4
- D(3) = 2
- D(4) = 3
- D(5) = 2
- D(6) = 1
- D(7) = -3

### <a name="risultato-spt"></a>6.3 Risultato: Shortest Path Tree

`01:12:39`

**Archi dello SPT**:
- (1,2) - costo 4
- (1,3) - costo 2
- (2,6) - costo -3
- (3,7) - costo -5
- (6,4) - costo 2
- (4,5) - costo -1

**Esempi di cammini minimi**:
- 1 → 6: 1 → 2 → 6 (costo 1)
- 1 → 7: 1 → 3 → 7 (costo -3)

---

## <a name="scaling-cambio-radice"></a>7. Scaling dei Costi per Cambio Radice

### <a name="problema-radice6"></a>7.1 Problema: SPT con Radice 6

`01:13:12 - 01:14:18`

**Nuovo problema**: Calcolare SPT con radice 6

**Approccio naive**:
- Riapplicare SPT-LQ da zero
- Grafo ha cicli e costi negativi
- Complessità: O(mn) 😞

**Problema**: Stiamo perdendo informazione utile dalla soluzione precedente!

**Informazione disponibile**: Le etichette D(i) della soluzione con radice 1

### <a name="trasformazione-costi"></a>7.2 Trasformazione dei Costi

`01:14:18 - 01:19:35`

**Osservazione sugli archi dello SPT**:
- Per archi (i,j) ∈ SPT: D(j) - D(i) = C(i,j) (arco "teso")
- Per archi (i,j) ∉ SPT: D(j) - D(i) < C(i,j) + D(i)

**Nuova definizione dei costi**:
```
C'(i,j) = C(i,j) + D(i) - D(j)
```

**Proprietà**:

1. **Per archi nello SPT** (i,j) ∈ SPT:
   ```
   C'(i,j) = C(i,j) + D(i) - D(j)
           = C(i,j) + D(i) - (D(i) + C(i,j))
           = 0
   ```

2. **Per archi fuori dallo SPT** (i,j) ∉ SPT:
   ```
   C(i,j) + D(i) ≥ D(j)
   ⟹ C'(i,j) = C(i,j) + D(i) - D(j) ≥ 0
   ```

**Risultato**: Tutti i nuovi costi sono **non negativi**! ✅

**Esempio di calcolo**:
- Arco (2,7): C(2,7) = -1, D(2) = 4, D(7) = -3
  - C'(2,7) = -1 + 4 - (-3) = **6** ✅
  
- Arco (5,3): C(5,3) = 5, D(5) = 2, D(3) = 2
  - C'(5,3) = 5 + 2 - 2 = **5** ✅

**Vantaggio**: Ora possiamo usare **Dijkstra** invece di SPT-LQ! 🚀

### <a name="dimostrazione-equivalenza"></a>7.3 Dimostrazione Equivalenza

`01:19:35 - 01:27:45`

**Domanda**: Il problema con i costi scalati è equivalente?

**Dimostrazione**:

Consideriamo un cammino generico da i a j:
```
i → i₁ → i₂ → ... → iᵣ → j
```

**Costo con costi originali**:
```
C_old = C(i,i₁) + C(i₁,i₂) + ... + C(iᵣ,j)
```

**Costo con costi scalati**:
```
C_new = C'(i,i₁) + C'(i₁,i₂) + ... + C'(iᵣ,j)

      = [C(i,i₁) + D(i) - D(i₁)]
      + [C(i₁,i₂) + D(i₁) - D(i₂)]
      + ...
      + [C(iᵣ,j) + D(iᵣ) - D(j)]

      = C(i,i₁) + C(i₁,i₂) + ... + C(iᵣ,j)
      + [D(i) - D(i₁) + D(i₁) - D(i₂) + ... + D(iᵣ) - D(j)]
```

**Cancellazione telescopica**:
```
D(i) - D(i₁) + D(i₁) - D(i₂) + ... + D(iᵣ) - D(j) = D(i) - D(j)
```

**Risultato**:
```
C_new = C_old + [D(i) - D(j)]
```

**⭐ Proprietà fondamentale**: La differenza tra i costi dipende **solo dagli estremi** (i, j), **non dai nodi interni**!

**Conseguenza**: Lo shortest path tra due nodi nel grafo scalato corrisponde allo shortest path nel grafo originale. ✅

### <a name="vantaggi-scaling"></a>7.4 Vantaggi della Tecnica

`01:27:45 - 01:29:28`

**Applicazione**:
1. Prima computazione (radice 1): Costosa O(mn) con SPT-LQ
2. Scaling dei costi: O(m)
3. Successive computazioni (altre radici): Veloci con Dijkstra O(m log n)

**Vantaggi**:
- ✅ Preserva equivalenza del problema
- ✅ Permette di usare algoritmo più efficiente
- ✅ Utile per calcolare SPT con radici diverse in sequenza

**Differenza con scaling del puzzle iniziale**:
- ❌ Puzzle: Scaling fisso → numero archi variabile → non equivalente
- ✅ Questa tecnica: Scaling dipende dalla soluzione → cancellazione telescopica → equivalente

**Conclusione**: Tecnica potente ma complessa, utile quando serve calcolare SPT da molte radici diverse.

---

## 📝 Note Finali

`01:29:28`

Fine della lezione. La prossima volta introdurremo i **problemi di flusso**.

Buona giornata! 👋
