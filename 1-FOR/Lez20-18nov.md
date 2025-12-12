Data e ora: 2025-12-11 12:14:10
Luogo: [Inserisci luogo]: [Inserisci luogo]
Corso: [Inserisci nome corso]: Programmazione Lineare
## Panoramica
Questa lezione ha trattato argomenti avanzati di programmazione lineare. Temi chiave: metodo delle due fasi per trovare una soluzione ammissibile iniziale, teoria e applicazione della slackness complementare per certificare l’ottimalità, interpretazione economica delle variabili duali come prezzi ombra. Si è inoltre esplorata l’analisi di sensitività sulla funzione obiettivo e svolto un esercizio dettagliato sulla ricerca di una direzione ammissibile crescente. È stato anche assegnato un quesito bonus per la prossima settimana.
## Contenuti rimanenti
1. Un’importante applicazione della slackness complementare che richiede più tempo.
2. Due concetti molto importanti per concludere la parte sulla programmazione lineare.
3. Altri esercizi.
## Contenuti trattati
### 1. Metodo delle due fasi per trovare una soluzione ammissibile
- **Ricapitolo:** L’algoritmo generale per risolvere problemi di PL richiede una soluzione ammissibile di partenza.
- **Caso 1 (Semplice):** Se tutti i termini noti (b) sono non negativi in un sistema `Ax <= b`, allora `x = 0` è una soluzione ammissibile banale.
- **Caso 2 (Complesso):** Se alcuni vincoli rendono `x = 0` non ammissibile, serve un problema ausiliario.
- **Metodo delle due fasi:**
    - **Fase 1: Trovare una soluzione ammissibile.**
        - Si dividono i vincoli in “buoni” (soddisfatti da x=0) e “cattivi” (non soddisfatti da x=0).
        - Si costruisce un problema ausiliario aggiungendo una variabile ausiliaria non negativa `u_i` a ciascun vincolo “cattivo”.
        - L’obiettivo del problema ausiliario è massimizzare la somma negativa di queste variabili ausiliarie (es. `max -Σu_i`), penalizzandone l’uso.
        - Una soluzione ammissibile iniziale per il problema ausiliario è `x=0` e `u_i = -b_i` per i vincoli cattivi.
        - Se la soluzione ottima del problema ausiliario ha tutti `u_i = 0`, il vettore `x` risultante è soluzione ammissibile per il problema originale.
        - Se l’ottimo `u` è maggiore di zero, il problema originale è inammissibile.
    - **Fase 2: Risolvere il problema di PL originale.** Si parte dalla soluzione ammissibile trovata in Fase 1.
- Questo processo è spesso visibile nei log dei solver, che mostrano le iterazioni per Fase 1 e Fase 2.
### 2. Condizioni di slackness complementare
- Tre affermazioni equivalenti per l’ottimalità di soluzioni ammissibili `x_bar` (primal) e `y_bar` (dual):
    1. `x_bar` e `y_bar` sono ottimali.
    2. I valori della funzione obiettivo sono uguali: `c*x_bar = y_bar*b`.
    3. Il prodotto tra il vettore delle variabili duali e il vettore degli slack primali è zero: `y_bar * (b - A*x_bar) = 0`.
- Questa terza affermazione porta alle condizioni di slackness complementare.
- **Per coppie asimmetriche:** `y_i * (b_i - a_i*x) = 0` per ogni vincolo `i`.
    - Se una variabile duale `y_i` è diversa da zero, il corrispondente vincolo primale `i` deve essere attivo (uguaglianza).
    - Se un vincolo primale `i` è slack (stretta disuguaglianza), la corrispondente variabile duale `y_i` deve essere zero.
- **Per coppie simmetriche:** Si applicano due insiemi di condizioni.
    - `y_i * (b_i - a_i*x) = 0` (come sopra).
    - `x_j * (c_j - y*A_j) = 0`: Se una variabile primale `x_j` è diversa da zero, il corrispondente vincolo duale `j` deve essere attivo. Se un vincolo duale `j` è slack, la corrispondente variabile primale `x_j` deve essere zero.
### 3. Applicazione della slackness complementare: certificare l’ottimalità
- Questo è un metodo per verificare se una soluzione primale ammissibile è ottima senza eseguire tutto il simplesso.
- **Procedura:**
    1. Data una soluzione primale ammissibile `x_bar`.
    2. Identificare quali vincoli primali sono attivi e quali slack.
    3. Usare la slackness complementare per dedurre che per ogni vincolo primale slack, la corrispondente variabile duale deve essere zero.
    4. Identificare quali variabili primali sono diverse da zero.
    5. Usare il secondo insieme di condizioni di slackness complementare: per ogni variabile primale diversa da zero, il corrispondente vincolo duale deve essere attivo (uguaglianza).
    6. Questo crea un sistema di equazioni lineari per le restanti variabili duali incognite. Risolverlo per trovare la soluzione duale `y_bar`.
    7. **Verifica di ammissibilità:** La soluzione `y_bar` così costruita è complementare per costruzione. Per certificare l’ottimalità, bisogna verificare che `y_bar` sia anche ammissibile per il duale (cioè soddisfi tutti i vincoli duali e le condizioni di segno).
    8. Se `y_bar` è ammissibile, la `x_bar` originale è certificata ottima.
- È stato svolto un esempio dettagliato con il “problema dei cellulari”.
### 4. Analisi di sensitività tramite slackness complementare
- Questa applicazione analizza come le variazioni dei coefficienti della funzione obiettivo influenzano l’ottimalità.
- L’analisi si fa un parametro alla volta.
- **Esempio:** Nel problema “porte e finestre” con soluzione ottima (2,6), il primo coefficiente obiettivo viene cambiato da 30 a un parametro `alpha`. Si cerca l’intervallo di `alpha` per cui (2,6) resta ottimo.
- **Procedura:**
    1. Con la soluzione (2,6), i vincoli 2 e 3 sono attivi. Per slackness complementare, le variabili duali y1, y4 e y5 sono zero.
    2. Si ottiene un sistema parametrico per le restanti variabili duali: `3y3 = alpha` e `2y2 + 2y3 = 50`.
    3. Risolvendo: `y3 = alpha/3` e `y2 = 25 - (1/3)alpha`.
    4. Per ammissibilità duale, le variabili duali devono essere non negative (`y2 >= 0`, `y3 >= 0`).
    5. Imporre queste condizioni dà l’intervallo per `alpha`: `0 <= alpha <= 75`.
- **Interpretazione geometrica:** Il vettore obiettivo `C` deve restare nel cono generato dai gradienti dei vincoli attivi.
### 5. Interpretazione economica delle variabili duali (prezzi ombra)
- Le variabili duali rappresentano il valore marginale o “prezzo ombra” di una risorsa.
- **Analisi:** Se il termine noto di un vincolo primale `i` viene perturbato di una piccola quantità `epsilon`, la variazione del valore ottimo è `y_i * epsilon`.
- **Conclusione:** La variabile duale `y_i` è la derivata del valore ottimo rispetto a una variazione unitaria della risorsa `b_i`.
    - Se un vincolo di risorsa è slack, il suo prezzo ombra `y_i` è 0 (c’è abbondanza, quindi un’unità in più non vale nulla).
    - Se un vincolo di risorsa è attivo, il suo prezzo ombra `y_i` è positivo, indicando il profitto marginale di un’unità in più.
- Questa interpretazione vale solo per piccole perturbazioni che non cambiano l’insieme dei vincoli attivi all’ottimo.
### 6. Esercizio: ricerca di una direzione ammissibile crescente
- Dato un problema di programmazione lineare e un punto x_bar = (2, 1).
- **Passo 1: Identificare i vincoli attivi.** Sostituire il punto nei vincoli per vedere quali sono in uguaglianza. Per (2,1), i vincoli attivi sono 2 e 4.
- **Passo 2: Formulare il primale ristretto.** Questo sistema caratterizza le direzioni ammissibili crescenti (xi) dove l’obiettivo migliora (`c^T * xi > 0`) e la fattibilità è mantenuta sui vincoli attivi (`a_i^T * xi <= 0`).
- **Passo 3: Formulare il duale ristretto.** Questo sistema verifica l’ottimalità cercando di scrivere `c` come combinazione lineare non negativa dei gradienti dei vincoli attivi. Se il duale ristretto non ha soluzione, esiste una direzione ammissibile crescente.
- **Passo 4: Verificare una direzione proposta.** Una direzione proposta, xi_bar = (1, 4), è stata verificata sostituendola nel sistema primale ristretto, confermando che è una direzione ammissibile crescente.
- **Passo 5: Calcolare la massima ampiezza del passo (lambda).** Considerando solo i vincoli non attivi, lambda è il minimo di `(b_i - a_i^T * x_bar) / (a_i^T * xi_bar)` per tutti gli `i` con denominatore positivo. Il calcolo ha dato lambda = 3/8.
- **Passo 6: Trovare il nuovo punto.** Il nuovo punto è `x_prime = x_bar + lambda * xi_bar` = (2,1) + 3/8 * (1,4). Si è previsto che i nuovi vincoli attivi in x_prime saranno il vincolo 3 (che ha determinato lambda) e il vincolo 2.
### 7. Quesito bonus: dimostrare l’ottimalità di un algoritmo di investimento
- È stato assegnato un quesito bonus per la prossima settimana.
- **Problema:** Dimostrare che l’algoritmo greedy per il problema dello zaino frazionario (problema di investimento) fornisce una soluzione ottima.
- **Metodo:** Usare la slackness complementare.
- **Passi per gli studenti:**
    1. Formalizzare il problema di investimento (primale).
    2. Scrivere il problema duale.
    3. Scrivere le equazioni di slackness complementare.
    4. Usare la struttura della soluzione greedy per costruire una soluzione duale complementare.
    5. Dimostrare che questa soluzione duale è ammissibile.
## Domande degli studenti
1. **Domanda sulla visualizzazione dell’ottimalità in alta dimensione:** Uno studente ha notato la difficoltà di visualizzare la condizione di ottimalità (il vettore obiettivo nel cono dei gradienti dei vincoli attivi) in spazi ad alta dimensione.
    - **Risposta:** Anche se la visualizzazione è difficile in alta dimensione (es. 1.000 disequazioni, 1 milione di variabili), i solver spesso forniscono l’intervallo per cui una soluzione resta ottima, che è l’equivalente analitico di questo concetto geometrico.
2. **Domanda su come iniziare l’esercizio:** Uno studente ha chiesto come iniziare l’esercizio di ricerca di una direzione ammissibile crescente dal punto (2,1).
    - **Risposta:** Il primo passo è determinare l’insieme dei vincoli attivi sostituendo il punto (2,1) nel sistema di disequazioni per vedere quali sono in uguaglianza.