## Panoramica: Dalla Ricerca Generale ai Problemi di Soddisfacimento Vincoli (CSP)
Gli algoritmi di ricerca standard (informati e non informati) sono general-purpose e non fanno assunzioni sulla struttura dello stato di un problema. Specializzando la rappresentazione del problema per una classe specifica di problemi, possono essere sviluppati algoritmi più efficienti e intelligenti. I Problemi di Soddisfacimento Vincoli (CSP) rappresentano tale classe, concentrandosi su problemi dove l'obiettivo è uno stato finale che soddisfa certe condizioni, piuttosto che la sequenza di azioni per raggiungerlo.
-   **Ricerca Generale:** Gli algoritmi funzionano su qualsiasi rappresentazione stato, necessitando solo azioni, risultati, test obiettivo e costi passo.
-   **Rappresentazione Specializzata (CSP):** Restringendo la rappresentazione a variabili, valori e vincoli, l'algoritmo può sfruttare la struttura del problema per ridurre lo spazio di ricerca e trovare soluzioni più efficientemente. Questo è particolarmente utile per "problemi di identificazione" dove la soluzione è un'assegnazione di valori, non un percorso.

### L'Inefficienza della Ricerca Standard per Problemi Vincoli
Applicare un algoritmo standard come Depth-First Search (DFS) a un CSP è possibile ma altamente inefficiente perché rileva i fallimenti troppo tardi.
-   **Esempio: Colorazione Mappa con DFS**
    -   **Problema:** Colorare 7 territori dell'Australia con 3 colori, assicurando che territori adiacenti non condividano un colore.
    -   **Processo Inefficiente:** Un DFS potrebbe assegnare 'rosso' al Western Australia e poi 'rosso' al Northern Territory adiacente. Invece di identificare immediatamente questo conflitto, l'algoritmo continuerebbe esplorando l'intero sotto-albero di possibilità derivante da questa assegnazione parziale non valida.
    -   **Conseguenza:** Una quantità significativa di tempo è sprecata esplorando un ramo dell'albero di ricerca garantito a fallire. La complessità temporale per questa ricerca futile è esponenziale.
    -   **Conclusione:** La chiave per risolvere CSP efficientemente è identificare e potare questi rami inconsistenti il prima possibile.

## Definizione Formale di un Problema Soddisfacimento Vincoli (CSP)
Un CSP è definito da una rappresentazione fattorizzata consistente di tre componenti principali:
1.  **Variabili (X):** Un insieme di `n` variabili, {X₁, ..., Xₙ}.
    -   *Esempio (Colorazione Mappa):* Le variabili sono i territori {WA, NT, SA, Q, NSW, V, T}.
2.  **Domini (D):** Un dominio di `d` valori possibili per ogni variabile.
    -   *Esempio (Colorazione Mappa):* Il dominio per ogni territorio è {rosso, verde, blu}.
3.  **Vincoli (C):** Un insieme di regole che specificano combinazioni valide di valori per sottoinsiemi di variabili.
    -   *Esempio (Colorazione Mappa):* WA ≠ NT; WA ≠ SA; ecc.
-   **Spazio di Ricerca:** Il numero totale di assegnazioni complete possibili è `dⁿ`.
-   **Soluzione:** Una soluzione a un CSP è un'assegnazione di valori a variabili che è sia:
    -   **Completa:** Ogni variabile è assegnata un valore.
    -   **Consistente:** L'assegnazione non viola alcun vincolo.

## Esempi di Problemi Modellati come CSP
### Colorazione Mappa (Australia)
-   **Variabili:** I 7 territori.
-   **Dominio:** {rosso, verde, blu}.
-   **Vincoli:** Territori adiacenti devono avere colori diversi.

### Problema 8-Regine
-   **Variabili:** Le 8 colonne della scacchiera (X₁ a X₈).
-   **Dominio:** Le 8 posizioni riga (1 a 8) per la regina in ogni colonna.
-   **Vincoli:** Per due regine qualsiasi a (Xᵢ, Vᵢ) e (Xⱼ, Vⱼ), Vᵢ ≠ Vⱼ (non nella stessa riga) e |Xᵢ - Xⱼ| ≠ |Vᵢ - Vⱼ| (non sulla stessa diagonale).

### Puzzle Cripto-aritmetici
-   **Esempio:** `DUE + DUE = QUATTRO`
-   **Variabili:** Le lettere uniche {D, U, E, Q, A, T, R, O}.
-   **Dominio:** Le cifre {0, 1, ..., 9}.
-   **Vincoli:**
    -   Ogni lettera deve essere una cifra unica.
    -   Le regole dell'aritmetica devono valere (es. E + E = O + 10 * C₁, dove C₁ è il riporto).
    -   Questi possono essere rappresentati visualmente usando un ipergrafo vincoli.

### Pianificazione Compiti
-   **Esempio:** Pianificare 4 compiti (T1, T2, T3, T4) con dipendenze temporali.
-   **Variabili:** I compiti stessi.
-   **Domini:** Possibili tempi inizio/fine.
-   **Vincoli:** Regole temporali, come "T2 deve essere completato prima che T1 inizi" o "T1 deve essere fatto durante l'ora 3".

## Risolvere CSP: Concetti Chiave e Strategie
### Vincoli Binari e Grafi Vincoli
-   Il focus sarà sui **vincoli binari**, che sono vincoli tra esattamente due variabili.
-   Qualsiasi problema con vincoli di ordine superiore (coinvolgenti più di due variabili) può essere convertito in un problema equivalente con solo vincoli binari introducendo variabili aggiuntive.
-   CSP con vincoli binari possono essere visualizzati come **grafo vincoli**:
    -   **Nodi:** Rappresentano le variabili.
    -   **Archi:** Rappresentano i vincoli tra coppie di variabili.

### Sfruttare Struttura Problema
-   La struttura del grafo vincoli può essere usata per semplificare il problema. Se un grafo può essere spezzato in componenti disconnessi multipli, il problema complessivo può essere diviso in sottoproblemi più piccoli e indipendenti, riducendo drammaticamente la complessità di ricerca.
-   **Esempio (Mappa Australia):** Tasmania (T) è un nodo disconnesso nel grafo. La colorazione di Tasmania è un sottoproblema indipendente che non influenza il resto della ricerca.
-   **Esempio (Decomposizione):** Un problema con 80 variabili binarie ha uno spazio di ricerca di 2⁸⁰. Se può essere decomposto in quattro sottoproblemi indipendenti di 20 variabili, la complessità diventa 4 * 2²⁰, che è computazionalmente fattibile.

### Strategia Centrale: Rilevazione Precoce Fallimenti
L'obiettivo primario degli algoritmi CSP è identificare fallimenti non appena un'assegnazione parziale viola un vincolo. Questo permette all'algoritmo di potare interi sotto-alberi dello spazio di ricerca, evitando il costo temporale esponenziale di esplorarli. Questo è raggiunto da:
1.  **Migliorare la funzione azione** per prevenire la generazione di stati inconsistenti.
2.  **Usare vincoli per propagare conoscenza** e proattivamente restringere i domini delle variabili rimanenti non assegnate.

## Ricerca Backtracking Base
Questo è un algoritmo fondamentale per risolvere CSP che applica una semplice ricerca depth-first (DFS).
- **Principi Centrali**:
    1.  **Assegnazione Incrementale**: Assegna un valore a una variabile alla volta.
    2.  **Controllo Vincoli**: Mentre le assegnazioni sono fatte, controllarle contro tutti i vincoli rilevanti. Considerare solo valori consistenti con l'assegnazione parziale esistente.
- **Caratteristiche**:
    - È lineare nell'uso memoria, sfruttando lo stack chiamate ricorsive.
    - È esponenziale nella complessità temporale.
- **Processo**: La ricerca evita di generare nodi figli che violano vincoli. Per esempio, nella colorazione mappa, se uno stato è colorato rosso, l'algoritmo non considererà rosso come opzione per stati adiacenti. Questa è una forma base di potatura dell'albero di ricerca.

## Migliorare Backtracking: Tecniche Filtraggio
Queste tecniche, anche conosciute come filtraggio o inferenza, migliorano il backtracking base eliminando proattivamente valori dai domini di variabili non assegnate. L'obiettivo è rilevare fallimenti inevitabili precocemente.

### Forward Checking
- **Meccanismo**: Quando una variabile `X` è assegnata un valore, l'algoritmo controlla tutte le variabili non assegnate `Y` connesse a `X` da un vincolo. Poi rimuove qualsiasi valore dal dominio di `Y` che è inconsistente con la nuova assegnazione di `X`.
- **Esempio (Colorazione Mappa Australia)**:
    1.  **Assegna WA = Rosso**: Il valore Rosso è rimosso dai domini di Northern Territory (NT) e South Australia (SA).
    2.  **Assegna QLD = Verde**: Il valore Verde è rimosso dai domini di NT, SA, e New South Wales (NSW).
    3.  **Assegna VIC = Blu**: Il valore Blu è rimosso dai domini di SA e NSW.
    4.  **Rilevazione Fallimento**: A questo punto, il dominio di SA diventa vuoto, poiché tutti i suoi colori possibili (Rosso, Verde, Blu) sono stati eliminati. Il percorso di ricerca è fallito, e l'algoritmo deve fare backtrack.
- **Complessità**: La complessità è O(n * s * t), dove `n` è il numero di variabili, `s` è il numero più grande di vincoli coinvolgenti una singola variabile, e `t` è la dimensione dominio.

### Consistenza Arco (AC-3)
AC-3 è un algoritmo filtraggio più potente e computazionalmente costoso che può rilevare più inconsistenze del forward checking.
- **Principio**: Un arco (Xi → Xj) è arco-consistente se per ogni valore `x` nel dominio di Xi, esiste qualche valore `y` nel dominio di Xj tale che l'assegnazione (Xi=x, Xj=y) è consistente. L'algoritmo impone questo per tutti gli archi nel problema.
- **Miglioramento su Forward Checking**: AC-3 può rilevare inconsistenze tra due variabili *non assegnate*. Nell'esempio colorazione mappa, dopo le prime due assegnazioni, sia NT che SA potrebbero avere solo {Blu} rimasti nei loro domini. Poiché NT e SA sono adiacenti, AC-3 identificherebbe questo come inconsistenza e attiverebbe un fallimento, anche prima che una terza variabile (come Victoria) sia assegnata.
- **Processo Algoritmo**:
    1.  Una coda è inizializzata con tutti gli archi (vincoli) nel CSP.
    2.  Un arco (Xi → Xj) è rimosso dalla coda e una procedura `revise` è eseguita.
    3.  **Revise**: Per ogni valore nel dominio di Xi, l'algoritmo controlla se esiste un valore compatibile nel dominio di Xj. Se nessun tale valore esiste, il valore è rimosso dal dominio di Xi.
    4.  **Propagazione**: Se la procedura `revise` rimuove qualsiasi valore dal dominio di Xi, tutti gli altri archi puntanti a Xi (es. Xk → Xi) sono aggiunti nuovamente alla coda per propagare il cambiamento.
    5.  Il processo continua finché la coda è vuota. Se il dominio di qualsiasi variabile diventa vuoto, il problema è inconsistente.
- **Complessità e Limitazioni**: La complessità di AC-3 è cubica nella dimensione del dominio (d), O(c * d³), dove `c` è il numero di archi. Non può rilevare tutte le inconsistenze. Esiste una generalizzazione, k-consistenza, ma è spesso troppo complessa. AC-3 è un buon compromesso tra tempo ricerca e tempo propagazione. Per alcuni problemi come semplici puzzle Sudoku, AC-3 da solo può risolvere l'intero problema senza alcuna ricerca.

## Euristiche per Ordinamento Variabili e Valori
L'ordine in cui variabili e valori sono scelti può impattare significativamente le prestazioni di ricerca.

### Selezione Variabile: Principio Fail-Fast
L'obiettivo è scegliere una variabile che probabilmente porterà a potare l'albero di ricerca o trovare un fallimento velocemente.
1.  **Minimum Remaining Values (MRV)**: Seleziona la variabile con meno valori legali rimanenti nel suo dominio. Questo è anche conosciuto come "variabile più vincolata".
2.  **Euristica Grado**: Come tie-breaker per MRV, seleziona la variabile coinvolta nel numero più grande di vincoli con altre variabili non assegnate.

### Selezione Valore: Massimizzare Opzioni Future
Una volta che una variabile è selezionata, la scelta di quale valore assegnare prima conta.
- **Least Constraining Value (LCV)**: Preferire il valore che esclude meno scelte per le variabili vicine nel grafo vincoli. Questo lascia il numero massimo di opzioni disponibili per assegnazioni successive.

## L'Algoritmo Backtracking Completo con Inferenza
L'algoritmo finale e ottimizzato integra backtracking, euristiche e inferenza.
- **Processo Ricorsivo**:
    1.  Se l'assegnazione è completa, ritorna la soluzione.
    2.  Seleziona una variabile non assegnata usando le euristiche **MRV** e **Grado**.
    3.  Per ogni valore nel dominio della variabile (ordinato dall'euristica **LCV**):
        a. Controlla se il valore è consistente con l'assegnazione corrente.
        b. Se consistente, aggiungi l'assegnazione (`var = valore`).
        c. Esegui un algoritmo inferenza (es. **AC-3**) per propagare vincoli.
        d. Se l'inferenza non risulta in un fallimento (cioè nessun dominio vuoto):
            i. Fai una chiamata ricorsiva con il nuovo stato.
            ii. Se la chiamata ricorsiva ha successo, ritorna la soluzione.
    4.  Se tutti i valori sono stati provati e falliti, ritorna fallimento per backtrack.

### Esempio: Problema 4-Regine con AC-3
Questo esempio dimostra la potenza dell'algoritmo completo.
- **Stato Iniziale**: 4 variabili (regine in colonne), ognuna con dominio {1, 2, 3, 4}.
- **Percorso 1: Assegna X1 = 1**
    1.  L'assegnazione è fatta.
    2.  **Inferenza (AC-3)** è eseguita. L'algoritmo propaga vincoli tra le variabili non assegnate (X2, X3, X4).
    3.  Attraverso una serie di revisioni, AC-3 trova che il dominio di X3 diventa vuoto.
    4.  **Risultato**: La ricerca fallisce immediatamente. L'intero ramo iniziante con X1=1 è potato senza fare altre assegnazioni. L'algoritmo fa backtrack.
- **Percorso 2: Assegna X1 = 2**
    1.  L'assegnazione è fatta.
    2.  **Inferenza (AC-3)** è eseguita di nuovo.
    3.  Forward checking elimina molti conflitti immediati.
    4.  AC-3 continua propagando e riduce i domini di X3 e X4 a valori singoli.
    5.  **Passo Successivo**: L'euristica MRV ora detta che la prossima variabile da assegnare deve essere X3 o X4, poiché hanno solo un valore rimasto.
    6.  L'algoritmo procede assegnando questi valori forzati, portando velocemente a una soluzione valida.

## Confronto Approcci Soddisfacimento Vincoli e Ottimizzazione
### Introduzione Ricerca Locale e Ottimizzazione
I metodi ricerca CSP standard iniziano da uno stato vuoto e costruiscono una soluzione passo-passo. Questi metodi non considerano intrinsecamente il **costo** di una soluzione. Se un problema richiede trovare una soluzione con il costo più basso, è necessario un approccio alternativo.
-   **Ricerca Locale e Ottimizzazione**: Questo approccio inizia con una configurazione completa, ma potenzialmente non fattibile o non ottimale. Poi fa piccole modifiche locali per migliorare iterativamente la configurazione finché una soluzione fattibile o ottimale è raggiunta. Questo metodo è capace di incorporare una funzione costo.

### L'Algoritmo Hill-Climbing
Hill-Climbing è un metodo ricerca locale usato per problemi ottimizzazione.
-   **Processo:**
    1.  **Inizializzazione:** Inizia con una configurazione completa casuale.
    2.  **Valutazione:** Valuta lo stato corrente usando una funzione obiettivo (es. costo o valore).
    3.  **Iterazione:** Genera stati vicini facendo piccole modifiche. Muovi al vicino con il valore migliore.
    4.  **Terminazione:** Se nessun vicino offre un miglioramento, lo stato corrente è considerato la soluzione (un ottimo locale). L'algoritmo termina.

### Esempio Applicazione: Posizionamento Ospedali
-   **Problema:** Posiziona ospedali in una città per minimizzare la distanza totale di viaggio da tutte le case al loro ospedale più vicino, con vincoli che gli ospedali non possono essere su case o uno sull'altro.
-   **Perché CSP è insufficiente:** Un CSP standard può imporre i vincoli posizionamento ma non può gestire l'obiettivo ottimizzazione di minimizzare distanza totale.
-   **Soluzione Ricerca Locale:**
    1.  Inizia posizionando ospedali casualmente.
    2.  Calcola il costo totale iniziale (somma distanze).
    3.  Esplora vicini muovendo un ospedale a uno spot vuoto adiacente.
    4.  Ricalcola il costo. Se il costo è più basso, accetta il nuovo stato.
    5.  Ripeti finché nessuna mossa singola può ridurre ulteriormente il costo.

### Sfide e Terminologia in Ottimizzazione
-   Una sfida primaria di Hill-Climbing è che può rimanere bloccato in un **massimo locale** (un picco che non è il più alto complessivo) e fallire nel trovare il **massimo globale** (la miglior soluzione possibile).
-   **Terminologia Paesaggio Spazio-Stati:**
    -   **Massimo Globale:** Il valore più alto nell'intero spazio stati.
    -   **Massimo Locale:** Uno stato migliore di tutti i suoi vicini ma non necessariamente il migliore complessivo.
    -   **Plateau (o Massimo Locale Piatto):** Un'area piatta dove tutti gli stati vicini hanno lo stesso valore.
    -   **Shoulder:** Un plateau da cui il progresso è possibile.

### Modellare CSP come Problema Ottimizzazione
Un Problema Soddisfacimento Vincoli (CSP) può essere riformulato come problema ottimizzazione.
-   **Esempio (N-Regine):**
    -   **Obiettivo:** Minimizza il numero di conflitti (coppie regine attaccanti). Una soluzione è trovata quando il numero di conflitti è zero.
    -   **Processo:**
        1.  Inizia con una configurazione di N regine sulla scacchiera.
        2.  Calcola il numero di conflitti.
        3.  Fai modifiche locali (es. muovi una regina dentro la sua colonna).
        4.  Seleziona la mossa che risulta nella riduzione più grande di conflitti.
        5.  Ripeti finché il numero di conflitti è 0.