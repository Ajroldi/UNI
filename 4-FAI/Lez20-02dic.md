> [Data e Ora]: 2025-12-11 18:42:00
> [Luogo]: [Luogo tradotto]
> [Corso]: Intelligenza Artificiale
## Panoramica
Questa sessione ha concluso l'unità sulla pianificazione automatica introducendo tecniche efficienti come GraphPlan e la Pianificazione Gerarchica, e discutendo scenari reali che coinvolgono il non-determinismo. La lezione è poi passata a un modulo completo sulla Modellazione dei Problemi. Ciò ha incluso una metodologia per distinguere tra problemi di Ricerca, CSP e Ottimizzazione, seguita da un approfondimento sulla formalizzazione di esempi specifici (Sliding Tile Puzzle, Spesa Alimentare e Tiling con Domino) per illustrare la gerarchia e la convertibilità tra questi framework di IA.
## Contenuti Rimanenti
1. **Spesa Alimentare come Pianificazione:** Rappresentare esplicitamente il problema della Spesa come un problema di Pianificazione (saltato per mancanza di tempo).
2. **Scheduling di Processi:** Modellazione di questo tipo di problema (assegnato come compito).
3. **Incertezza:** Annunciato come argomento della prossima lezione.
## Contenuti Trattati
### 1. Algoritmo GraphPlan
- **Concetto:** Risolve problemi di pianificazione costruendo un grafo strutturato in livelli alternati di letterali (stati) e azioni.
- **Struttura:**
    - **Livello S0:** Letterali dello stato iniziale.
    - **Livello A_n:** Azioni potenzialmente applicabili dati i letterali in S_n.
    - **Livello S_{n+1}:** Letterali potenzialmente veri dopo l'esecuzione delle azioni in A_n.
- **Regole di Costruzione:** Il grafo cresce livello dopo livello, includendo "azioni di persistenza" (azioni identità). La costruzione si ferma quando un livello contiene tutti i letterali obiettivo (ed essi sono consistenti).
- **Uso come Euristica:** L'indice del livello in cui gli obiettivi compaiono per la prima volta funge da euristica ammissibile per algoritmi di ricerca A* (stima ottimistica).
### 2. Pianificazione Gerarchica
- **Idea di Base:** Decomporre compiti di alto livello (compiti radice) in sotto-compiti via via più specifici fino a raggiungere operatori primitivi.
- **Struttura:**
    - **Tasks:** Obiettivi generali da raggiungere.
    - **Methods:** Modi alternativi per decomporre un task (relazione OR).
    - **Operators:** Azioni primitive eseguibili (sequenza obbligatoria/relazione AND).
- **Esempio:** Il task "Viaggiare" decomposto in Aereo, Autobus o Auto (alternative). "Comprare un biglietto dell'autobus" decomposto in sequenza obbligatoria: andare allo sportello, richiedere, pagare.
### 3. Pianificazione nel Mondo Reale (Estensioni)
- **Ipotesi Classiche vs. Realtà:** Le ipotesi della pianificazione classica (ambienti pienamente osservabili, statici, deterministici) raramente tengono nel mondo reale.
- **Tecniche:**
    - **Sensor-less / Conformant Planning:** Pianificare senza feedback di stato (ad es., un robot cieco che "over-acting" urta volutamente un muro per assicurarsi la posizione).
    - **Pianificazione Condizionale:** Inserimento di istruzioni `if-then` basate sulla percezione.
    - **Esecuzione e Monitoraggio:** Verificare se le azioni hanno avuto l'effetto desiderato e ripianificare se necessario (da zero o tramite percorsi di riparazione).
    - **Pianificazione Continua / Online:** Intercalare pianificazione ed esecuzione (ad es., Orizzonte Recedente).
### 4. Metodologia di Modellazione dei Problemi
- **Quadro Decisionale:**
    1. **È richiesta una sequenza di azioni?**
        - **Sì:** Modella come **Problema di Ricerca** (o Pianificazione).
    2. **Se No (conta solo lo stato finale):**
        - **Tutti gli stati obiettivo hanno lo stesso valore?**
            - **Sì:** Modella come **Problema di Soddisfacimento di Vincoli (CSP)**.
        - **Gli stati hanno costi/utilità differenti?**
            - **No:** Modella come **Problema di Ottimizzazione** (Ricerca Locale).
- **Rappresentazione dello Stato:** Atomica (Ricerca), Fattorizzata (CSP), Strutturata (Pianificazione).
### 5. Esempio 1: Sliding Tile Puzzle (Pianificazione & Ricerca)
- **Problema:** Tavola a 7 celle, 3 tessere nere, 3 bianche, 1 spazio vuoto. Obiettivo: scambiare le posizioni.
- **Scelta di Modellazione:** Inizialmente identificato come Problema di Ricerca perché la *sequenza* conta.
- **Formalizzazione come Pianificazione:**
    - **Predicati:** `adjacent(x, y)` (statico) e `at(color, location)` (fluente).
    - **Formulazione del Goal:** Evidenziato un potenziale bug: le variabili devono essere distinte per evitare che mappino sulla stessa posizione.
    - **PDDL vs. STRIPS:** Il problema richiede PDDL a causa di precondizioni negative (verificare che una destinazione non sia occupata), non supportate da STRIPS.
### 6. Esempio 2: Spesa Alimentare (Ottimizzazione)
- **Problema:** Selezionare articoli alimentari entro un budget (300 yen) per massimizzare l'utilità, con limiti di quantità.
- **Ricerca Locale (Hill Climbing):** Modellato con Stato (insieme di articoli) e Mosse (aggiungi/sostituisci).
- **Analisi del Costo di Passo:** Dimostrato che convertire "Massimizza l'Utilità" in "Minimizza il Costo di Cammino" (ad es., usando `Costo = Costante - Punteggio`) fallisce in algoritmi come Uniform Cost Search perché l'algoritmo preferirà percorsi più "economici" che corrispondono a utilità inferiori.
### 7. Esempio 3: Tiling con Domino (CSP)
- **Problema:** Coprire una regione di 2n celle con n domino.
- **Scelta di Modellazione:** Identificato come CSP perché il percorso non conta e tutte le coperture complete sono equivalenti.
- **Formalizzazione:**
    - **Variabili:** I domino.
    - **Domini:** Coppie ordinate di celle adiacenti.
    - **Vincoli:** Nessuna sovrapposizione (celle condivise). Il controllo delle sovrapposizioni è sufficiente; vincoli espliciti di "copertura totale" non sono necessari se l'area coincide.
### 8. Gerarchia dei Modelli
- **Interconnessione:** I problemi di Ottimizzazione possono essere modellati come Ricerca/Pianificazione, ma i problemi di Pianificazione generalmente non possono essere modellati come CSP (la conversione verso il basso non è logica).
- **Rappresentazione Naturale:** Ogni problema ha una rappresentazione "più naturale" (ad es., il Tiling è naturalmente CSP, la Spesa è naturalmente Ottimizzazione).
> **Suggerimenti AI**
> *   **GraphPlan:** Definire esplicitamente i collegamenti "Mutex" per spiegare perché l'algoritmo non si ferma immediatamente quando i letterali compaiono ma sono tra loro inconsistenti.
> *   **Pianificazione Gerarchica:** Disegnare visivamente la struttura ad albero (Tasks -> Methods -> Operators) per rinforzare la differenza tra alternative OR e sequenze AND.
> *   **Mondo Reale:** Collegare esplicitamente la Pianificazione Continua al "Model Predictive Control" per creare un ponte tra IA e teoria del controllo.
> *   **Albero Decisionale di Modellazione:** Presentare un problema ambiguo e chiedere alla classe di votare il framework prima di rivelare la risposta per aumentare il coinvolgimento.
> *   **Tile Puzzle:** Discutendo PDDL vs. STRIPS, scrivere alla lavagna la sintassi specifica delle precondizioni negative accanto a un tentativo STRIPS fallito per visualizzare il limite.
> *   **Spesa Alimentare:** Disegnare due piccoli alberi di ricerca: uno che calcola la "vera utilità" e uno che calcola il "costo dell'algoritmo" fallito, per mostrare visivamente perché `11 - score` fallisce.
> *   **Gerarchia:** Fornire un diagramma visivo (cerchi annidati o flow chart) che mostri quali tipi di problema sono sottoinsiemi di altri.
## Domande degli Studenti
1. **[Studente]** "Se vai all'indietro, vedi le azioni che dobbiamo fare... ottenere i codici reali che ci mancano dal primo passo." (Riguardo GraphPlan).
    - **[Risposta]** No, GraphPlan fornisce principalmente un'euristica per A* stimando la lunghezza minima del piano.
2. **[Studente]** "Nel primo livello dobbiamo scegliere, e nell'ultimo livello dobbiamo decomporre." (Riguardo la Pianificazione Gerarchica).
    - **[Risposta]** Chiarito che le opzioni iniziali (Methods) sono alternative (OR), mentre i passi all'interno di un metodo sono sequenze obbligatorie (AND).
3. **[Studente]** Ha espresso scetticismo sull'esempio del robot cieco.
    - **[Risposta]** Spiegato che l'"over-acting" (muoversi più del necessario) garantisce che il robot finisca in una posizione nota (il muro) anche senza sensori.
4. **[Studente]** "Come puoi fare... solo cinque passi, se non sai che alla fine... ti avvicinerai davvero." (Riguardo la pianificazione continua).
    - **[Risposta]** Si usa un'euristica o una funzione di valutazione per stimare quale stato all'orizzonte sia più vicino all'obiettivo finale.
5. **[Studente]** Dato la rappresentazione iniziale dello stato obiettivo per il Tile puzzle, l'obiettivo risulta soddisfatto (anche se non dovrebbe)?
    - **[Risposta]** Sì, a meno che le variabili non siano esplicitamente definite come diverse, potrebbero tutte puntare alla stessa posizione, soddisfacendo tecnicamente la logica.
6. **[Studente]** È utile definire una mossa locale che rimuove un elemento dallo stato (nell'Hill Climbing per la spesa)?
    - **[Risposta]** No, rimuovere un elemento diminuisce l'utilità, quindi l'algoritmo sceglierebbe raramente tale mossa.
7. **[Studente]** Dobbiamo aggiungere un vincolo riguardo all'ordine delle coppie nel problema dei Domino?
    - **[Risposta]** No. Il vincolo di non sovrapposizione è sufficiente. Se nessun domino si sovrappone e l'area coincide, la copertura è garantita matematicamente.