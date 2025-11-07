# Lezione 12 - 22 Ottobre: Introduzione ai Problemi di Flusso su Grafi

## 📑 Indice

1. [Esercizio: Shortest Path Tree con Radice 1](#esercizio-spt) `00:00:36 - 00:06:53`
   - [Selezione dell'Algoritmo](#selezione-algoritmo)
   - [Esecuzione con Dijkstra](#esecuzione-dijkstra)
   - [Analisi Post-Ottimalità: Arco DF](#post-opt-df)
2. [Puzzle: Il Ponte di Notte](#puzzle-ponte) `00:06:53 - 00:10:56`
   - [Descrizione del Problema](#descrizione-ponte)
   - [Regole e Vincoli](#regole-ponte)
   - [Obiettivo del Puzzle](#obiettivo-ponte)
3. [Introduzione ai Problemi di Flusso](#intro-flussi) `00:10:56 - 00:14:42`
   - [Cosa c'è di Nuovo](#novita-flussi)
   - [Applicazioni](#applicazioni-flussi)
   - [Prerequisiti](#prerequisiti-flussi)
4. [Problema del Flusso Massimo](#max-flow) `00:14:42 - 00:22:56`
   - [Piano di Evacuazione](#piano-evacuazione)
   - [Formulazione del Problema](#formulazione-maxflow)
   - [Vincoli e Obiettivo](#vincoli-obiettivo)
5. [Modellazione: Piano di Evacuazione](#modellazione-evacuazione) `00:27:23 - 00:32:31`
   - [Costruzione del Grafo](#costruzione-grafo-evacuazione)
   - [Nodi e Archi Dummy](#nodi-archi-dummy)
   - [Interpretazione della Soluzione](#interpretazione-evacuazione)
6. [Modellazione: Problema di Scheduling](#scheduling) `00:32:31 - 00:48:53`
   - [Descrizione del Problema](#descrizione-scheduling)
   - [Parametri: pⱼ, rⱼ, dⱼ](#parametri-scheduling)
   - [Costruzione del Grafo Bipartito](#grafo-bipartito)
   - [Capacità e Interpretazione](#capacita-scheduling)
7. [Problema del Taglio di Capacità Minima](#min-cut) `00:49:26 - 00:54:40`
   - [Problema Duale](#problema-duale)
   - [Definizione di Taglio (Cut)](#definizione-cut)
   - [Capacità del Taglio](#capacita-cut)
8. [Proprietà del Flusso attraverso un Taglio](#flusso-taglio) `00:54:40 - 01:01:10`
   - [Teorema: Flusso Invariante](#flusso-invariante)
   - [Disuguaglianza Fondamentale](#disuguaglianza-fondamentale)
9. [Grafo Residuo e Cammino Aumentante](#grafo-residuo) `01:01:10 - 01:19:57`
   - [Definizione di Grafo Residuo](#definizione-residuo)
   - [Esempio di Verifica Ottimalità](#esempio-verifica)
   - [Archi Verdi (Push) e Rossi (Pull)](#archi-verdi-rossi)
   - [Certificato di Ottimalità](#certificato-ottimalita)
10. [Algoritmo di Ford-Fulkerson](#ford-fulkerson) `01:19:57 - 01:27:18`
    - [Pseudocodice](#pseudocodice-ff)
    - [Procedura Flow Augmentation](#flow-augmentation)
    - [Capacità Residua](#capacita-residua)
    - [Correttezza dell'Algoritmo](#correttezza-ff)

---

## <a name="esercizio-spt"></a>1. Esercizio: Shortest Path Tree con Radice 1

### <a name="selezione-algoritmo"></a>1.1 Selezione dell'Algoritmo

`00:00:36 - 00:01:43`

Buongiorno, prima di iniziare con il nuovo argomento sui flussi su grafi, facciamo un ultimo esercizio sugli shortest path.

**Problema**: Calcolare lo Shortest Path Tree con radice 1.

**Decision Tree**:

1. ❓ Ci sono cicli nel grafo?
   - Sì: Ciclo identificato ✅
   - → ❌ Non possiamo usare SPT Acyclic

2. ❓ Tutti i costi sono non negativi?
   - Sì: Tutti i costi ≥ 0 ✅
   - → ✅ Possiamo usare **Dijkstra**

**Algoritmo scelto**: **Dijkstra**

### <a name="esecuzione-dijkstra"></a>1.2 Esecuzione con Dijkstra

`00:01:43 - 00:04:15`

**Esecuzione tramite applet**:

| Step | Nodo | D(nodo) | Azione |
|------|------|---------|--------|
| 0 | A (1) | 0 | Inizializzazione |
| 1 | A | 0 | Visita forward star: C, B |
| 2 | C | min | Estratto (label minima) |
| 3 | E | 5 | Estratto |
| 4 | - | - | Visita forward star: 3 archi |
| 5 | D | 1 | Estratto (label minima tra D, G, F, B) |
| 6 | G | - | Estratto (nessun arco uscente) |
| 7 | F | - | Estratto |
| 8 | B | - | Estratto (ultimo) |

**Risultato**: Shortest Path Tree rappresentato dagli archi verdi nell'applet

### <a name="post-opt-df"></a>1.3 Analisi Post-Ottimalità: Arco DF

`00:04:15 - 00:06:53`

**Domanda**: Quali costi per l'arco DF mantengono la soluzione ottimale?

**Etichette**:
- D(D) = 6
- D(F) = 13

**Proprietà degli archi nello SPT**:
```
D(j) - D(i) = C(i,j)  (arco "teso")
```

**Proprietà degli archi fuori dallo SPT**:
```
D(j) - D(i) ≤ C(i,j)
```

**Calcolo**:
```
D(F) - D(D) = 13 - 6 = 7
```

**Condizione di ottimalità**:
```
C(D,F) ≥ 7
```

**Range ammissibile**: **C(D,F) ∈ [7, +∞)**

**Interpretazione**: Se C(D,F) < 7, l'arco sarebbe "rotto" nel modello fisico (corda allentata), e potremmo migliorare D(F) usando questo arco.

**⚠️ Regola generale**:
- **Archi nello SPT**: D(j) - D(i) = C(i,j) esattamente
- **Archi fuori dallo SPT**: D(j) - D(i) ≤ C(i,j)

Questa è la **condizione di ottimalità** per lo Shortest Path Tree.

---

## <a name="puzzle-ponte"></a>2. Puzzle: Il Ponte di Notte

### <a name="descrizione-ponte"></a>2.1 Descrizione del Problema

`00:06:53 - 00:08:06`

> **📚 Nota sul Materiale**: Su WeBeep è disponibile un modulo di auto-istruzione sulla **programmazione dinamica**. Solo 14 studenti l'hanno provato. La scadenza è stata posticipata! 

**Problema del Ponte**:
- 4 persone devono attraversare un ponte pericoloso
- È notte, serve una torcia
- Solo **una torcia** disponibile
- Distanza troppo lunga per lanciarla → qualcuno deve riportarla indietro

### <a name="regole-ponte"></a>2.2 Regole e Vincoli

`00:08:06 - 00:09:17`

**Vincoli**:
- Massimo **2 persone** per volta sul ponte
- Tempo di attraversamento per persona: diverso
- Con 2 persone: velocità del **più lento**

**Esempio**:
```
Persona 1: 1 minuto
Persona 2: 2 minuti
Persona 3: 5 minuti
Persona 4: 10 minuti

Se 1 e 4 attraversano insieme: 10 minuti
```

**Preemption**: Quando qualcuno torna indietro, viaggia alla **propria velocità** (non rallentato).

### <a name="obiettivo-ponte"></a>2.3 Obiettivo del Puzzle

`00:09:17 - 00:10:56`

**Obiettivo**: Trovare la sequenza ottimale di attraversamenti che minimizza il tempo totale.

**⚠️ Importante**: Non serve trovare la soluzione ottimale!

**Richiesta del puzzle**: **Modellare il problema come ricerca di shortest path** su un grafo opportuno.

**Suggerimenti**:
- Il grafo **non è** semplicemente A → B
- Serve applicare tecniche di **programmazione dinamica**
- Lo shortest path nel grafo corrisponde alla soluzione ottimale

**Deadline**: Prossima settimana

---

## <a name="intro-flussi"></a>3. Introduzione ai Problemi di Flusso

### <a name="novita-flussi"></a>3.1 Cosa c'è di Nuovo

`00:10:56 - 00:11:29`

**Differenza con problemi precedenti**:

| Shortest Path / MST | Problemi di Flusso |
|---------------------|-------------------|
| Costi sugli archi | Costi + **Capacità** |
| Quantità irrilevante | **Quantità importa** |
| Minimizzare costo | Massimizzare flusso |

**Nuovo concetto**: **Capacità u(i,j)**
- Limite superiore alla quantità di flusso su un arco
- Non possiamo inviare più di u(i,j) sull'arco (i,j)

**Tipi di problemi**:
1. Solo capacità (no costi) → **Maximum Flow**
2. Capacità + costi → **Minimum Cost Flow** (più avanti)

### <a name="applicazioni-flussi"></a>3.2 Applicazioni

`00:11:29 - 00:12:36`

**Applicazioni pratiche**:
- 🚨 Piani di evacuazione
- 📅 Problemi di scheduling
- ✈️ Yield management (gestione capacità)
- 👥 Job assignment (assegnazione lavori)

**Potenza dei flussi**: Framework di modellazione molto generale e potente.

**Caratterizzazione soluzione ottimale**: Come per shortest path e MST, sarà **cruciale** caratterizzare la soluzione ottimale per:
- Riconoscere quando l'abbiamo trovata
- Progettare algoritmi efficienti

### <a name="prerequisiti-flussi"></a>3.3 Prerequisiti

`00:12:36 - 00:14:42`

**Concetti da ricordare**:

1. **Graph Search**: Algoritmo per trovare un cammino qualsiasi (non shortest)
2. **Proprietà fondamentale**: Tra due nodi i e j:
   - Esiste un cammino, **OPPURE**
   - Esiste un taglio (cut)
3. **Visione duale**: Massimizzazione vs Minimizzazione
4. **Decisioni sugli archi**: Le variabili decisionali corrispondono agli archi
5. **Cicli a costo negativo**: Come rilevarli

---

## <a name="max-flow"></a>4. Problema del Flusso Massimo

### <a name="piano-evacuazione"></a>4.1 Piano di Evacuazione

`00:14:42 - 00:18:06`

**Contesto**: Ogni edificio ha un piano di evacuazione.

**Esempio**:
- Stima persone per stanza
- Uscite di emergenza identificate
- Dimensionamento uscite per evacuazione sicura

**Casi reali**:

| Scenario | Tempo Evacuazione | Metodo Verifica |
|----------|-------------------|-----------------|
| ✈️ Aeroplano | **90 secondi** | Modello 1:1 con persone |
| 🚢 Nave da crociera | **60 minuti** | Modelli matematici |

**Problema aereo**: 
- Costruiscono modello 1:1 della cabina
- Riempiono con passeggeri
- Testano evacuazione
- Se > 90s → riprogettano (aggiungono uscite)

**Problema nave**:
- Impossibile costruire modello 1:1
- Usano **modelli matematici** (max flow!)

**Elementi del piano**:
- Capacità delle sezioni dell'edificio
- Capacità scale e corridoi
- Capacità uscite di emergenza
- Verifica: tutti possono uscire in sicurezza?

### <a name="formulazione-maxflow"></a>4.2 Formulazione del Problema

`00:18:06 - 00:19:53`

**Problema del Flusso Massimo (Maximum Flow)**:

**Input**:
- Grafo G = (N, A)
- Due nodi speciali: **S** (source/sorgente), **T** (sink/pozzo)
- Capacità u(i,j) per ogni arco (i,j) ∈ A
- Capacità intere (ipotesi semplificativa)

**Obiettivo**: Inviare il **massimo flusso possibile** da S a T

**Analogia**: 
- S = sorgente del flusso (da cui tutto origina)
- T = destinazione (dove tutto deve arrivare)

### <a name="vincoli-obiettivo"></a>4.3 Vincoli e Obiettivo

`00:19:53 - 00:22:56`

**Vincoli**:

1. **Vincoli di capacità**:
   ```
   x(i,j) ≤ u(i,j)    ∀(i,j) ∈ A
   ```

2. **Vincoli di conservazione del flusso**:
   ```
   ∑ x(j,i) - ∑ x(i,j) = 0    ∀i ∈ N \ {S,T}
   (j,i)∈BS(i)  (i,j)∈FS(i)
   
   dove:
   - BS(i) = backward star (archi entranti in i)
   - FS(i) = forward star (archi uscenti da i)
   ```

**Obiettivo**: Due formulazioni equivalenti:

**Opzione 1** - Massimizzare flusso entrante in T:
```
max ∑ x(j,T)
    (j,T)∈BS(T)
```

**Opzione 2** - Massimizzare flusso uscente da S:
```
max ∑ x(S,j)
    (S,j)∈FS(S)
```

**⭐ Proprietà fondamentale**: Le due formulazioni sono **equivalenti**!

**Perché?** Grazie ai vincoli di conservazione: tutto ciò che esce da S deve arrivare a T (nei nodi interni non c'è perdita né guadagno).

**Nota**: Per ora ignoriamo i costi. Ci interessa solo inviare il massimo flusso possibile.

---

## <a name="modellazione-evacuazione"></a>5. Modellazione: Piano di Evacuazione

### <a name="costruzione-grafo-evacuazione"></a>5.1 Costruzione del Grafo

`00:27:23 - 00:29:38`

**Elementi del piano di evacuazione**:
- Stanze con numero di persone (es. 15, 20, 30)
- Corridoi e scale con capacità
- Uscite di emergenza
- Punto di raccolta (meeting point)

**Nodi del grafo**:

1. **Nodo T**: Punto di raccolta (sink)
2. **Nodo S**: Sorgente fittizia (mystery node!)
3. **Nodi interni**: 
   - Un nodo per ogni stanza
   - Un nodo per ogni sezione di corridoio
   - Un nodo per ogni scala
   - Un nodo per ogni uscita

**Archi interni**:
- Arco per ogni possibile movimento delle persone
- Capacità = capacità della sezione (es. 60, 50)

**Esempio capacità**:
```
Corridoio: 60 persone
Scala 1: 50 persone
Scala 2: 60 persone
Uscita 1: 50 persone
Uscita 2: 50 persone
Uscita 3: 50 persone
```

### <a name="nodi-archi-dummy"></a>5.2 Nodi e Archi Dummy

`00:29:38 - 00:31:29`

**Problema**: Nel max flow tutto il flusso origina da S, ma nell'evacuazione il flusso origina dalle stanze!

**Soluzione**: Archi dummy (fittizi) da S alle stanze

```
    ╔═══════════════════════════════════╗
    ║  Stanza 1: 15 persone             ║
    ║  Stanza 2: 20 persone             ║
    ║  Stanza 3: 30 persone             ║
    ╚═══════════════════════════════════╝
              ↑ ↑ ↑
             15 20 30 (capacità archi dummy)
              ║ ║ ║
            ══╝ ╚═════
            ║
            S (sorgente fittizia)
```

**Capacità archi dummy**: Esattamente il numero di persone in quella stanza
- Da S → Stanza 1: u = 15
- Da S → Stanza 2: u = 20
- Da S → Stanza 3: u = 30

**Interpretazione**: Le persone entrano nell'edificio attraverso le finestre! (ovviamente è una finzione matematica)

### <a name="interpretazione-evacuazione"></a>5.3 Interpretazione della Soluzione

`00:31:29 - 00:32:31`

**Valore del max flow**: Sia F il flusso massimo da S a T

**Interpretazione**:

| Condizione | Risposta | Azione |
|------------|----------|--------|
| F = ∑ capacità archi da S | ✅ Piano fattibile | Evacuazione possibile |
| F < ∑ capacità archi da S | ❌ Piano non fattibile | Serve aumentare capacità |

**Esempio**:
```
Totale persone: 15 + 20 + 30 = 65
Max flow F: 65 → ✅ OK!
Max flow F: 50 → ❌ Problema! Servono uscite aggiuntive
```

**Nota**: Il problema richiede una risposta **Sì/No**, non il valore numerico del flusso in sé.

---

## <a name="scheduling"></a>6. Modellazione: Problema di Scheduling

### <a name="descrizione-scheduling"></a>6.1 Descrizione del Problema

`00:32:31 - 00:33:34`

**Contesto**:
- M macchine identiche (tempo di processamento uguale)
- N job da schedulare

**Parametri per ogni job j**:

| Parametro | Simbolo | Significato |
|-----------|---------|-------------|
| Tempo processamento | p_j | Tempo necessario per completare il job |
| Release time | r_j | Quando il job diventa disponibile |
| Due date | d_j | Deadline del job |

**Vincolo temporale**:
```
d_j - r_j ≥ p_j
```
(altrimenti impossibile!)

**Caratteristiche**:
- Un job può essere eseguito su **una sola macchina**
- **Preemption** permessa: possiamo interrompere e riprendere senza penalità
- Domanda: Esiste uno scheduling fattibile? (Sì/No)

### <a name="parametri-scheduling"></a>6.2 Parametri: pⱼ, rⱼ, dⱼ

`00:33:34 - 00:37:02`

**Esempio con 4 job**:

| Job | p_j | r_j | d_j | Interpretazione |
|-----|-----|-----|-----|-----------------|
| 1 | 1.7 | 1 | 3 | Disponibile dal tempo 1, scadenza 3 |
| 2 | 3.7 | 1 | 5 | Disponibile dal tempo 1, scadenza 5 |
| 3 | 3.5 | 3 | 7 | Disponibile dal tempo 3, scadenza 7 |
| 4 | 2.8 | 5 | 8 | Disponibile dal tempo 5, scadenza 8 |

**Interpretazione intervalli**:
- Job 1: può essere processato in [1, 3]
- Job 2: può essere processato in [1, 5]
- Job 3: può essere processato in [3, 7]
- Job 4: può essere processato in [5, 8]

### <a name="grafo-bipartito"></a>6.3 Costruzione del Grafo Bipartito

`00:37:02 - 00:40:43`

**Nodi del grafo**:

1. **S e T**: source e sink (come sempre)
2. **Nodi job**: 1, 2, 3, 4
3. **Nodi intervalli temporali**: [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8]

**Struttura**:
```
        Job 1
        Job 2     [1,2]
S →     Job 3  →  [2,3]
        Job 4     [3,4]
                  [4,5]  → T
                  [5,6]
                  [6,7]
                  [7,8]
```

**Archi da Job a Intervalli**:
- Esiste arco (j, [t,t+1]) se e solo se **[t, t+1] ⊆ [r_j, d_j]**

**Esempio Job 1** (r_1=1, d_1=3):
- ✅ Arco a [1,2]
- ✅ Arco a [2,3]
- ❌ NO arco a [3,4] (deadline scaduta!)

**Esempio Job 3** (r_3=3, d_3=7):
- ✅ Archi a [3,4], [4,5], [5,6], [6,7]

### <a name="capacita-scheduling"></a>6.4 Capacità e Interpretazione

`00:40:43 - 00:48:53`

**Significato del flusso**: Quantità di lavoro svolto sul job j nell'intervallo [t,t+1]

**Capacità degli archi**:

1. **Da S ai Job**: u(S, j) = **p_j**
   - Capacità = tempo di processamento totale richiesto
   - Esempio: u(S,1) = 1.7, u(S,2) = 3.7, ecc.

2. **Da Job a Intervalli**: u(j, [t,t+1]) = **1**
   - In un intervallo unitario, posso lavorare al massimo 1 unità di tempo
   - Vale per tutti gli archi interni

3. **Da Intervalli a T**: u([t,t+1], T) = **M**
   - M = numero di macchine
   - In un intervallo, posso fare M unità di lavoro (una per macchina)

**⚠️ Domanda importante**: Posso avere x(S,j) > p_j?

**Risposta**: NO! ❌

**Motivo**: I vincoli di capacità devono essere soddisfatti **per ogni arco**. Non posso eccedere la capacità di nessun arco uscente da S.

**Domanda**: Posso avere F > ∑ p_j?

**Risposta**: NO! ❌ Per lo stesso motivo.

**Proprietà generale**: In **qualsiasi sezione** del grafo, il flusso non può eccedere la somma delle capacità in quella sezione.

**Interpretazione soluzione**:

```
F = valore max flow

Se F = ∑ p_j  → ✅ Scheduling fattibile
Se F < ∑ p_j  → ❌ Scheduling non fattibile
```

---

## <a name="min-cut"></a>7. Problema del Taglio di Capacità Minima

### <a name="problema-duale"></a>7.1 Problema Duale

`00:49:26 - 00:50:34`

**Cambio di prospettiva**: Visione da "terrorista" 💣

**Scenario**:
- S = pozzo petrolifero
- T = raffineria
- Archi = oleodotti
- Capacità = dimensione oleodotto

**Obiettivo (terrorista)**: Interrompere il flusso da S a T con **minimo costo**

**Costo distruzione arco**: Proporzionale alla capacità
- Oleodotto grande → serve molta dinamite
- Oleodotto piccolo → serve poca dinamite

### <a name="definizione-cut"></a>7.2 Definizione di Taglio (Cut)

`00:50:34 - 00:52:40`

**Definizione**: Un taglio (S,T)-cut è una **partizione** dei nodi in due insiemi:
- N_S: contiene S
- N_T: contiene T
- N_S ∪ N_T = N
- N_S ∩ N_T = ∅

**Esempi di tagli**:

```
Taglio 1 (estremi):
N_S = {S}
N_T = {tutti gli altri}

Taglio 2 (estremi):
N_S = {tutti tranne T}
N_T = {T}

Taglio 3 (interno):
N_S = {S, 1, 2}
N_T = {3, 4, T}
```

### <a name="capacita-cut"></a>7.3 Capacità del Taglio

`00:52:40 - 00:54:40`

**Capacità di un taglio**:
```
U(N_S, N_T) = ∑ u(i,j)
              i∈N_S, j∈N_T
              (i,j)∈A
```

**⚠️ Attenzione**: Solo archi da N_S a N_T! Gli archi inversi non contano.

**Problema del Minimum Capacity Cut**:
```
min U(N_S, N_T)
```

**Input**: Identico al max flow (grafo, S, T, capacità)

**Decisioni**: Quali nodi mettere in N_S e quali in N_T

**Esercizio per casa**: Formulare il problema usando variabili 0-1 per i nodi.

---

## <a name="flusso-taglio"></a>8. Proprietà del Flusso attraverso un Taglio

### <a name="flusso-invariante"></a>8.1 Teorema: Flusso Invariante

`00:54:40 - 00:59:22`

**Teorema fondamentale**:

Dato un flusso fattibile x, per **qualsiasi taglio** (N_S, N_T):

```
F(N_S, N_T) = ∑ x(i,j) - ∑ x(j,i) = F
              i∈N_S      j∈N_T
              j∈N_T      i∈N_S
              (i,j)∈A    (j,i)∈A
```

dove F = flusso totale da S a T.

**Interpretazione**: Il flusso attraverso **qualsiasi taglio** è sempre uguale al flusso totale!

**Dimostrazione tramite esempio**:

```
Grafo con flussi (numeri rossi = flusso, blu = capacità):

S → 1: 3/10    2 → 1: 2/∞
S → 2: 2/2     1 → 3: 2/3
              3 → 4: 2/2
              4 → T: 3/6
              etc.
```

**Verifica su diversi tagli**:

| Taglio | Archi S→T | Archi T→S | Flusso netto |
|--------|-----------|-----------|--------------|
| {S} vs resto | (S,1)=3, (S,2)=2 | - | 3+2 = **5** |
| {S,1,2} vs resto | ... | ... | **5** |
| {S,1,2,3} vs {4,T} | (3,4)=2, (1,4)=3, (2,4)=4 | (4,3)=2 | 7-2 = **5** |

**Conclusione**: Il flusso è **invariante** rispetto al taglio! ✅

### <a name="disuguaglianza-fondamentale"></a>8.2 Disuguaglianza Fondamentale

`00:59:22 - 01:01:10`

**Teorema**: Per qualsiasi flusso fattibile x e qualsiasi taglio (N_S, N_T):

```
F(N_S, N_T) ≤ U(N_S, N_T)
```

**Dimostrazione**:

Dalla definizione:
```
F(N_S, N_T) = ∑ x(i,j) - ∑ x(j,i)
              →         ←
```

Per massimizzare F:
- Termine positivo: x(i,j) ≤ u(i,j) → massimo = u(i,j)
- Termine negativo: minimizzare = mettere 0

Quindi:
```
max F(N_S, N_T) = ∑ u(i,j) - 0 = U(N_S, N_T)
```

**Conseguenze**:
1. Il flusso attraverso un taglio non può mai eccedere la capacità del taglio
2. Se troviamo F = U per qualche taglio → **soluzione ottimale**!

---

## <a name="grafo-residuo"></a>9. Grafo Residuo e Cammino Aumentante

### <a name="definizione-residuo"></a>9.1 Definizione di Grafo Residuo

`01:01:10 - 01:04:09`

**Domanda chiave**: Data una soluzione fattibile, posso migliorarla?

**Strumento**: **Grafo Residuo** G_R(x)

Dato il grafo originale G = (N, A) con flusso x:

**Grafo Residuo** G_R = (N, A⁺ ∪ A⁻)

**Archi A⁺ (verdi)** - Push forward:
```
(i,j) ∈ A⁺  ⟺  (i,j) ∈ A  AND  x(i,j) < u(i,j)
```
Posso **aumentare** il flusso

**Archi A⁻ (rossi)** - Pull back:
```
(j,i) ∈ A⁻  ⟺  (i,j) ∈ A  AND  x(i,j) > 0
```
Posso **diminuire** il flusso (arco inverso!)

### <a name="esempio-verifica"></a>9.2 Esempio di Verifica Ottimalità

`01:04:09 - 01:08:09`

**Grafo originale** con flusso:

```
S → 1: 3/10    2 → 1: 0/∞
S → 2: 2/2     1 → 3: 2/3
              1 → 4: 1/1
              3 → 2: 2/10
              2 → 4: 4/4
              4 → 3: 2/2
              4 → T: 3/6
              3 → T: 2/2
```

**Costruzione grafo residuo**:

**Archi verdi** (x < u):
- S → 1: ✅ (3 < 10)
- S → 2: ❌ (2 = 2, saturato!)
- 2 → 1: ✅ (0 < ∞)
- 1 → 3: ✅ (2 < 3)
- 3 → 2: ✅ (2 < 10)
- 4 → T: ✅ (3 < 6)
- 1 → 4: ❌ (saturato)
- 2 → 4: ❌ (saturato)
- 4 → 3: ❌ (saturato)
- 3 → T: ❌ (saturato)

**Archi rossi** (x > 0):
- 1 → S (x(S,1) = 3 > 0)
- 2 → S (x(S,2) = 2 > 0)
- 3 → 1 (x(1,3) = 2 > 0)
- 4 → 1 (x(1,4) = 1 > 0)
- 2 → 3 (x(3,2) = 2 > 0)
- 4 → 2 (x(2,4) = 4 > 0)
- 3 → 4 (x(4,3) = 2 > 0)
- T → 4 (x(4,T) = 3 > 0)
- T → 3 (x(3,T) = 2 > 0)

### <a name="archi-verdi-rossi"></a>9.3 Archi Verdi (Push) e Rossi (Pull)

`01:08:09 - 01:14:40`

**Ricerca cammino S → T nel grafo residuo**:

**Tentativo 1**:
```
S → 1 (verde)
```
Bloccato! Non posso andare oltre.

**Tentativo 2 - Cammino trovato**:
```
S → 1 (verde) → 3 (verde) → 4 (rosso!) → T (verde)
```

**Interpretazione**:
- S → 1: Aumento flusso ↑
- 1 → 3: Aumento flusso ↑
- 3 → 4: **Diminuisco** flusso su (4,3) ↓ (arco rosso!)
- 4 → T: Aumento flusso ↑

**Calcolo capacità residua**:

```
θ = min {
    u(S,1) - x(S,1) = 10 - 3 = 7,   (verde)
    u(1,3) - x(1,3) = 3 - 2 = 1,    (verde)
    x(4,3) - 0 = 2,                  (rosso)
    u(4,T) - x(4,T) = 6 - 3 = 3     (verde)
}
```

**θ = min{7, 1, 2, 3} = 1**

**Aggiornamento flussi**:
```
x(S,1): 3 → 4  (+1)
x(1,3): 2 → 3  (+1)
x(4,3): 2 → 1  (-1)  ← Pull back!
x(4,T): 3 → 4  (+1)
```

**Verifica conservazione flusso**:
- Nodo 1: +1 entrante, +1 uscente ✅
- Nodo 3: +1 entrante, -1 da (4,3) che entra di meno = +1 netto uscente ✅
- Nodo 4: -1 uscente verso 3, +1 uscente verso T ✅

**Nuovo flusso totale**: F = 5 + 1 = **6** ✅

### <a name="certificato-ottimalita"></a>9.4 Certificato di Ottimalità

`01:14:40 - 01:19:57`

**Aggiornamento grafo residuo** dopo l'aumento:

**Nuovi archi**:
- (3,4) verde: ora possibile! (flusso non più saturato)
- (1,3) scompare: saturato!

**Ricerca nuovo cammino**: 

```
Da S: posso andare solo a 1
Da 1: posso andare solo a S (back)
```

**Nessun cammino S → T** nel grafo residuo! 🛑

**Taglio identificato**:
```
N_S = {S, 1}
N_T = {2, 3, 4, T}
```

**Archi che attraversano il taglio** (N_S → N_T):
- (S,2): capacità 2, flusso 2
- (1,3): capacità 3, flusso 3
- (1,4): capacità 1, flusso 1

**Verifica**:
```
Flusso attraverso taglio = 2 + 3 + 1 = 6
Capacità taglio = 2 + 3 + 1 = 6
```

**F = U** → **Soluzione ottimale certificata!** ✅

**Teorema (anticipazione)**: 
```
max F = min U(N_S, N_T)
```

Questo è il famoso **Max-Flow Min-Cut Theorem**!

---

## <a name="ford-fulkerson"></a>10. Algoritmo di Ford-Fulkerson

### <a name="pseudocodice-ff"></a>10.1 Pseudocodice

`01:19:57 - 01:22:42`

**Algoritmo di Ford-Fulkerson**:

```
INPUT: G = (N,A), S, T, u(i,j)

1. Inizializzazione:
   x(i,j) = 0    ∀(i,j) ∈ A

2. REPEAT:
   a) Costruisci G_R(x)
   
   b) Esegui Graph Search da S in G_R
   
   c) IF pred(T) ≠ NULL:
        - Ricostruisci cammino P da S a T (backtrack su pred)
        - Flow_Augmentation(P, x)
        - Aggiorna x
   
   UNTIL pred(T) = NULL

3. RETURN x
```

**Criterio di stop**: pred(T) = NULL significa che non esiste cammino S → T nel grafo residuo.

**Certificato di ottimalità**: Quando ci fermiamo, esiste un taglio con F = U.

### <a name="flow-augmentation"></a>10.2 Procedura Flow Augmentation

`01:22:42 - 01:24:25`

**Flow_Augmentation(P, x)**:

```
INPUT: Cammino P, flusso corrente x

1. Calcola capacità residua:
   θ = min { r(i,j) : (i,j) ∈ P }
   
2. Per ogni arco (i,j) ∈ P:
   
   IF (i,j) ∈ A⁺:  // arco verde
      x(i,j) = x(i,j) + θ
   
   IF (i,j) ∈ A⁻:  // arco rosso
      // (i,j) è inverso di (j,i) in grafo originale
      x(j,i) = x(j,i) - θ

3. Per tutti gli altri archi: invariato
```

### <a name="capacita-residua"></a>10.3 Capacità Residua

`01:24:25 - 01:25:05`

**Capacità residua** r(i,j) di un arco nel grafo residuo:

```
Se (i,j) ∈ A⁺:  (arco verde - push)
   r(i,j) = u(i,j) - x(i,j)

Se (i,j) ∈ A⁻:  (arco rosso - pull)
   // (i,j) è inverso di (j,i)
   r(i,j) = x(j,i)
```

**Capacità residua del cammino**:
```
θ = min { r(i,j) : (i,j) ∈ P }
```

**Proprietà**: Scegliendo θ come minimo:
1. ✅ Conservazione del flusso nei nodi interni
2. ✅ Rispetto vincoli di capacità (0 ≤ x ≤ u)

### <a name="correttezza-ff"></a>10.4 Correttezza dell'Algoritmo

`01:25:05 - 01:27:18`

**Teorema di correttezza**: L'algoritmo termina sempre con la soluzione ottimale.

**Dimostrazione (sketch)**:

1. **Terminazione**: L'algoritmo si ferma quando pred(T) = NULL

2. **Condizione di stop**: Non esiste cammino S → T in G_R(x)

3. **Taglio**: Se non c'è cammino, esiste un taglio (N_S, N_T) con:
   - S ∈ N_S
   - T ∈ N_T
   - Nessun arco del grafo residuo attraversa da N_S a N_T

4. **Analisi archi del taglio**:
   
   Per (i,j) con i ∈ N_S, j ∈ N_T:
   
   - Se (i,j) ∉ G_R (verde): allora x(i,j) = u(i,j) (saturato!)
   - Se (j,i) ∉ G_R (rosso): allora x(i,j) = 0

5. **Calcolo flusso attraverso taglio**:
   ```
   F(N_S, N_T) = ∑ u(i,j) - ∑ 0 = U(N_S, N_T)
   ```

6. **Ottimalità**: F = U → max flow raggiunto! ✅

**Video supplementare**: Su WeBeep disponibile dimostrazione formale completa.

**Prossima lezione**: Analisi della complessità. La scelta del cammino in G_R influenza l'efficienza!

---

## 📝 Note Finali

`01:27:18`

**Argomenti prossima lezione**:
- Complessità algoritmo Ford-Fulkerson
- Strategie di selezione del cammino aumentante
- Algoritmi più efficienti (Edmonds-Karp, Dinic)

**Punti chiave da ricordare**:
1. Max Flow ≠ Shortest Path (quantità importa!)
2. Grafo residuo: strumento fondamentale
3. Archi verdi (push) e rossi (pull)
4. Certificato ottimalità: taglio con F = U
5. Conservazione flusso: chiave della correttezza

Buona giornata! 👋
