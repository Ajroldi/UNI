# Riepilogo Unificato
## Panoramica
Questa sessione combinata ha fornito un’esplorazione completa dell’algoritmo Branch and Bound. Si è iniziato con note amministrative sull’uso di strumenti AI e una revisione teorica dello schema generale dell’algoritmo, inclusi i concetti di rilassamento e pre-solving. Si è poi passati a un’applicazione pratica, mostrando l’intero processo Branch and Bound sia su un problema Knapsack 0/1 che su un problema generale di programmazione lineare intera.
## Contenuti trattati
### 1. Introduzione e note di amministrazione
Il docente ha affrontato consegne di studenti che sembravano usare strumenti AI come ChatGPT in modo acritico, impiegando concetti avanzati non ancora trattati. È stato sottolineato che lo scopo di questi esercizi è l’apprendimento e il feedback, non solo il bonus. Di conseguenza, questo tipo di esercizio bonus sarà probabilmente eliminato nelle prossime edizioni del corso.
### 2. Algoritmo Branch and Bound: teoria
- **Schema generale:** Il motore dell’algoritmo è il calcolo di un upper bound (B) per un problema di massimizzazione. Un problema viene ramificato ricorsivamente in sottoproblemi, che possono essere potati (fathomed) in tre casi:
    1. L’upper bound non è migliore della soluzione corrente (incumbent): `B <= V`.
    2. Il sottoproblema è inammissibile.
    3. Il calcolo dell’upper bound produce una soluzione intera ammissibile, che è ottima per quel sottoproblema.
- **Concetto di rilassamento:** Un rilassamento (P') di un problema (P) è un problema più semplice la cui soluzione ottima fornisce un upper bound per P. Si ottiene ampliando l’insieme ammissibile e/o sovrastimando la funzione obiettivo. Se la soluzione ottima del rilassamento è ammissibile per il problema originale, è anche ottima per l’originale.
- **Tipi di rilassamento:**
    - **Rilassamento LP:** Si rilassano i vincoli di integrità (es. `x ∈ {0,1}` diventa `0 ≤ x ≤ 1`). È il metodo più comune.
    - **Omissione di vincoli:** Si rimuovono vincoli “complicanti” per semplificare la struttura (es. da un problema di flusso complesso a uno standard).
    - **Rilassamento lagrangiano:** Si spostano vincoli nella funzione obiettivo con penalità per la violazione (menzionato come tema futuro).
### 3. Applicazione 1: Knapsack 0/1
- **Pre-solving (fissaggio per dominanza):** Un problema Knapsack a 7 variabili è stato prima semplificato.
    - Una variabile con profitto negativo e peso positivo (“lose-lose”) è stata fissata a 0.
    - Una variabile con profitto positivo e peso negativo (“win-win”) è stata fissata a 1.
    - Il problema si è ridotto a 5 variabili, con budget e valore obiettivo aggiornati.
- **Esecuzione Branch and Bound:**
    1.  Una soluzione incumbent iniziale (valore: 14) è stata trovata con un approccio greedy (ordinando per rapporto valore/peso).
    2.  Il rilassamento LP del problema radice (P0) ha dato upper bound 16. Poiché 16 > 14, si è ramificato.
    3.  Si è ramificato su una variabile frazionaria (x4), creando sottoproblemi P1 (x4=1) e P2 (x4=0).
    4.  Risolvendo P1 si è trovata una nuova soluzione intera incumbent migliore (valore: 15).
    5.  Risolvendo P2 si è ottenuta una soluzione frazionaria con upper bound 16, richiedendo ulteriore ramificazione.
    6.  I sottoproblemi successivi sono stati esplorati e potati in base ai bound o trovando soluzioni intere non migliorative. Il processo è stato riassunto con un albero di enumerazione.
### 4. Applicazione 2: Programmazione intera generale
- **Strategia di branching:** Per variabili intere generali (non solo 0/1), ramificare su un valore frazionario `f` crea due sottoproblemi: `x <= floor(f)` e `x >= ceil(f)`.
- **Walkthrough geometrico:**
    1.  Introdotto un nuovo problema intero a 2 variabili. Incumbent iniziale 0.
    2.  Il rilassamento LP è stato risolto geometricamente, dando soluzione frazionaria con upper bound 2.35.
    3.  Si è ramificato su `x1`, creando due nuove regioni ammissibili.
    4.  Risolvendo un ramo (P2: `x1 >= 2`) geometricamente si è trovata una nuova soluzione intera incumbent (2, 2) valore 1.8.
    5.  L’altro ramo (P1: `x1 <= 1`) aveva ancora soluzione frazionaria con upper bound (2.067) > incumbent, richiedendo ulteriore branching su `x2`.
    6.  Il processo è continuato, con regioni ammissibili sempre più piccole illustrate con marker colorati. Sottoproblemi potati per vuotezza, bound o trovando la soluzione intera ottima finale (1, 2) valore 1.9.
### 5. Logistica del corso
- La lezione della prossima settimana sarà solo online martedì. Gli studenti possono seguire lo streaming dall’aula o da casa.
## Domande degli studenti
Nessuna domanda è stata posta dagli studenti durante la sessione.