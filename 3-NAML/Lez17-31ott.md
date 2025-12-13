Di seguito è riportata la traduzione e la rielaborazione strutturata della trascrizione fornita, in conformità con le direttive specificate.
## Introduzione al Laboratorio: Obiettivi e Strumenti
[00:00] Il programma odierno prevede due parti principali. Nella prima metà del laboratorio, verrà implementato l'algoritmo di *Singular Value Thresholding* (Soglia sul Valore Singolare), applicandolo a un caso di studio rilevante già affrontato durante le lezioni teoriche.
[00:13] Nella seconda parte, invece, si affronterà un argomento differente: JAX, una libreria per la differenziazione automatica. JAX è uno strumento molto utilizzato, versatile e potente; verrà analizzato attraverso una presentazione frontale, esaminando passo dopo passo i suoi oggetti caratteristici e le sue funzionalità tramite esempi pratici.
[00:31] Il materiale didattico è sostanzialmente lo stesso della settimana precedente, con solo alcuni piccoli dettagli aggiornati nei notebook. La nuova versione, leggermente modificata, è disponibile per chi volesse scaricarla.
## Contesto del Problema: Il Dataset MovieLens
[00:43] Per inquadrare il problema, si utilizzeranno le slide già viste durante la lezione. Il contesto è quello del dataset MovieLens, una raccolta di dati in cui degli utenti hanno assegnato una valutazione a determinati film. L'obiettivo è predire il gradimento di un utente per un film che non ha ancora visto.
[01:01] Questo problema è di grande interesse pratico, specialmente per grandi aziende come Netflix o Amazon. Per queste compagnie, anche un piccolo miglioramento percentuale nell'efficacia dei sistemi di raccomandazione può tradursi in milioni di euro di ricavi aggiuntivi. Si tratta quindi di un problema applicativo di notevole rilevanza.
## Caricamento e Analisi Preliminare dei Dati
[01:20] Il primo passo consiste nel caricare il dataset. Dopo aver stabilito la connessione alla macchina, si carica il file `MovieLens`, che si trova in formato CSV, nella sessione locale.
[01:38] Il dataset contiene 100.000 valutazioni, espresse con un punteggio da 1 a 5, fornite da circa 1.000 utenti su 1.600 film. Una caratteristica importante è che ogni utente ha valutato almeno 20 film.
[01:54] Il formato dei dati grezzi non è la matrice "utente-film" che ci si aspetta. Il primo compito sarà quindi convertire il file CSV in una matrice in cui le righe rappresentano gli utenti e le colonne i film.
[02:06] Il file CSV è, più precisamente, una lista di valori separati da tabulazioni (*tab-separated list*). Ogni riga contiene quattro informazioni:
1.  `user ID`: un identificatore univoco per l'utente.
2.  `item ID`: un identificatore univoco per il film.
3.  `rating`: la valutazione, un numero intero da 1 a 5.
4.  `timestamp`: il momento in cui l'utente ha inserito la valutazione. Quest'ultima informazione non verrà utilizzata nel modello, ma è presente nel dataset.
[02:29] Si inizia importando le librerie necessarie. Verranno utilizzate `pandas` per leggere il file CSV, `numpy` per le operazioni numeriche e alcuni moduli di `scipy` che si vedranno in seguito.
[02:41] La lettura del dataset è il primo passo operativo. In un contesto d'esame, questa fase è solitamente già fornita.
[02:49] Una volta caricato, il dataset si presenta come un `DataFrame` di `pandas`. Il notebook Jupyter offre una visualizzazione tabellare simile a un foglio di calcolo, con colonne nominate e righe indicizzate.
[03:00] Ad esempio, la prima riga mostra che l'utente con ID 196 ha assegnato al film con ID 242 una valutazione di 3 in un determinato istante temporale.
## Estrazione delle Dimensioni del Dataset
[03:08] Il passo successivo è verificare le dimensioni del problema: quanti utenti, film e valutazioni totali ci sono.
[03:14] Per contare il numero di utenti unici, si utilizza la funzione `numpy.unique` sulla colonna degli ID utente. Questa funzione restituisce un array contenente tutti gli ID univoci.
[03:22] Il numero di utenti (`n\_users`) è quindi la dimensione (`size`) di questo array di ID unici.
[03:29] Analogamente, per determinare il numero di film (`n\_movies`), si applica `numpy.unique` alla colonna degli ID dei film e se ne ottiene la dimensione.
[03:39] Il numero totale di valutazioni (`n\_ratings`) corrisponde semplicemente al numero di righe del `DataFrame`.
[03:48] I risultati ottenuti confermano le informazioni fornite nella descrizione del dataset: il numero di utenti, film e valutazioni corrisponde a quanto atteso. Eseguire questo doppio controllo è una buona pratica per assicurarsi che il caricamento dei dati sia avvenuto correttamente.
## Pre-elaborazione dei Dati: Mescolamento (Shuffling)
[04:05] Si introduce ora un passaggio fondamentale quando si lavora con dataset realistici, non visto nel laboratorio precedente: il mescolamento (*shuffling*) dei dati.
[04:13] La necessità di mescolare deriva dal fatto che i dati potrebbero avere un ordinamento intrinseco, ad esempio temporale, dato che il dataset include un `timestamp`. Le valutazioni sono probabilmente ordinate cronologicamente.
[04:24] Se si dovesse dividere il dataset in training (80%) e test (20%) senza mescolarlo, il set di training conterrebbe le valutazioni più vecchie e quello di test le più recenti. Questo potrebbe introdurre distorsioni legate alla temporalità.
[04:38] Ad esempio, un nuovo utente potrebbe registrarsi e valutare 20 film consecutivamente. Senza shuffling, tutte queste 20 valutazioni finirebbero nello stesso sottoinsieme (training o test).
[04:48] Per garantire che l'algoritmo venga addestrato e testato su un campione rappresentativo, si vuole che le valutazioni di uno stesso utente siano distribuite casualmente tra training e test. Questo aumenta l'entropia del dataset e rompe le dipendenze legate all'ordine di inserimento.
[05:06] Per eseguire lo shuffling, per prima cosa si imposta un seme casuale (*random seed*) per garantire la riproducibilità dei risultati. Successivamente, si crea un array di indici da 0 al numero totale di valutazioni (100.000).
[05:16] La funzione `numpy.random.shuffle` permuta casualmente questo array di indici.
[05:25] Si utilizza quindi questo array di indici permutati per riordinare le righe del `DataFrame`. L'indicizzazione tramite un array di indici (es. `data[shuffled\_indices]`) permette di riorganizzare le righe secondo l'ordine specificato, ottenendo così un mescolamento completo del dataset.
## Estrazione e Pulizia degli Indici
[05:41] Una volta mescolate le righe, si estraggono le colonne di interesse:
-   `rows`: gli ID degli utenti.
-   `cols`: gli ID dei film.
-   `values`: le valutazioni.
[05:54] A questo punto, si dispone di tre vettori che rappresentano le triplette (indice di riga, indice di colonna, valore) della futura matrice delle valutazioni.
[06:02] Tuttavia, il processo non è ancora terminato. Potrebbero esserci dei problemi legati agli indici. Si consideri un piccolo esempio per illustrare il problema.
[06:10] Immaginiamo un dataset con tre sole righe:
-   Utente 0 valuta film 1.
-   Utente 0 valuta film 2.
-   Utente 3 valuta film 2.
[06:26] Se si usassero direttamente gli ID `[0, 0, 3]` come indici di riga e `[1, 2, 2]` come indici di colonna per costruire la matrice, si otterrebbe una struttura con righe e colonne interamente nulle.
[06:38] Ad esempio, si avrebbe una riga per l'utente 0, ma anche per gli utenti 1 e 2 (che non esistono nel mini-dataset) e una per l'utente 3. Similmente, si avrebbe una colonna per il film 0 (non valutato), una per il film 1 e una per il film 2.
[06:47] Questo genera righe e colonne piene di zeri, che non portano alcuna informazione utile. Non ha senso fare inferenza su un film che nessuno ha visto o su un utente che non ha espresso valutazioni.
[07:06] Si vogliono quindi eliminare questi "buchi" negli indici, compattandoli.
[07:13] Per questo esercizio, è sufficiente sapere che esiste una funzione che esegue questa compattazione. Non si entrerà nei dettagli implementativi, ma la funzione `numpy.unique`, con l'opzione `return\_inverse=True`, permette di mappare gli indici originali (sparsi) a un nuovo insieme di indici contigui (compatti).
[07:30] In pratica, questo passaggio trasforma i vettori di indici `rows` e `cols` in modo da eliminare le righe e le colonne completamente vuote dalla matrice finale.
## Suddivisione in Training e Test Set
[07:44] Ora che gli indici sono stati puliti e compattati, si può procedere con la suddivisione del dataset in training e test.
[07:50] Si stabilisce che il training set conterrà l'80% delle valutazioni. Si calcola il numero di campioni per il training moltiplicando il numero totale di valutazioni per 0.8 e arrotondando il risultato.
[08:02] Poiché il dataset è stato mescolato, si può semplicemente prendere il primo 80% dei dati per il training e il restante 20% per il test.
[08:06] Utilizzando lo slicing degli array, si selezionano gli elementi da 0 fino al numero di campioni di training per il set di addestramento.
[08:15] Il set di test sarà composto da tutti gli elementi a partire dall'indice finale del training set fino alla fine dell'array.
[08:23] Al termine di questo passaggio, si ottengono sei vettori: una tripletta (righe, colonne, valori) per il training e una tripletta analoga per il test.
## Costruzione della Matrice delle Valutazioni
[08:32] Il passo successivo è trasformare queste triplette di dati in una matrice. La libreria `scipy` offre una funzione specifica per questo scopo, ottimizzata per matrici con molti zeri, note come matrici sparse.
[08:45] La funzione `scipy.sparse.csr\_matrix` prende in input la tripletta (valori, (indici di riga, indici di colonna)) e costruisce una matrice sparsa.
[08:56] In questa matrice, l'elemento $(i, j)$ conterrà la valutazione corrispondente se la coppia (utente $i$, film $j$) è presente nel dataset; altrimenti, l'elemento sarà zero.
[09:06] Infine, per le operazioni successive, si converte questa matrice sparsa (che memorizza implicitamente gli zeri) in una matrice densa (`full matrix`), dove gli zeri sono rappresentati esplicitamente.
[09:15] La matrice risultante, `X\_full`, conterrà molti zeri, poiché la maggior parte delle coppie utente-film non ha una valutazione associata. Tuttavia, grazie al passaggio di pulizia precedente, si è sicuri che non ci siano righe o colonne interamente nulle.
## Compito 1: Implementazione di un Sistema di Raccomandazione Banale (Baseline)
[09:27] Ora inizia la parte pratica da implementare. Il primo compito è creare un sistema di raccomandazione banale (*Trivial Recommender System*).
[09:33] Quando si sviluppa un nuovo modello di machine learning, è una buona pratica confrontarlo con una baseline, ovvero un modello molto semplice. Questo serve a verificare che il modello più sofisticato offra un reale vantaggio.
[09:48] La baseline si basa su un'idea semplice: la prossima valutazione di un utente sarà probabilmente simile alla media delle sue valutazioni passate.
[09:56] Se un utente ha sempre dato 5 come voto, è probabile che darà 5 anche al prossimo film. Se ha sempre dato 1, è probabile che darà 1. Sebbene semplice, questo approccio costituisce un punto di riferimento utile.
[10:13] Matematicamente, la predizione per la coppia (utente $i$, film $j$) è definita come la media delle valutazioni fornite dall'utente $i$ su tutti i film $k$ che ha già visto:
```math
\text{predizione}_{i,j} = \frac{\sum_{k \in \text{film visti da } i} \text{valutazione}_{i,k}}{\text{numero di film visti da } i}
```
[10:20] In pratica, per ogni utente (ogni riga della matrice), si sommano tutti gli elementi non nulli e si divide per il loro numero. Gli zeri non vengono considerati nel calcolo della media perché rappresentano valutazioni mancanti.
## Metriche di Valutazione
[10:40] Per valutare le prestazioni del modello, sono necessarie metriche quantitative. Si propongono due indici:
1.  **Root Mean Squared Error (RMSE)**: Misura la deviazione media tra le predizioni del modello e le valutazioni reali. Un valore più basso indica una migliore accuratezza.
2.  **Coefficiente di Correlazione di Pearson (ρ)**: Un indice statistico che misura la correlazione lineare tra due variabili, in questo caso le predizioni e i valori reali.
    -   Varia tra -1 e +1.
    -   `+1`: correlazione lineare positiva perfetta.
    -   `-1`: correlazione lineare negativa perfetta.
    -   `0`: assenza di correlazione lineare.
[11:18] Il primo compito consiste nell'implementare questo predittore banale. Il modello va costruito usando il training set e valutato sul test set.
[11:25] Si dovrà calcolare la valutazione media per ogni utente e usare questo valore per calcolare l'RMSE e il coefficiente di correlazione $\rho$ sui dati di test. Per il calcolo di $\rho$, si può utilizzare la funzione `pearsonr` di `scipy`.
## Compito 2: Implementazione dell'Algoritmo di Singular Value Thresholding (Hard)
[11:42] Il secondo compito è implementare l'algoritmo di *Singular Value Thresholding* (SVT), in particolare nella sua variante *hard*. Questa versione è più semplice da implementare rispetto alla variante *soft* vista a lezione.
[12:05] L'algoritmo *hard thresholding* si articola nei seguenti passaggi:
1.  **Decomposizione ai Valori Singolari (SVD)**: Si calcola la SVD della matrice delle valutazioni.
2.  **Soglia (Thresholding)**: Si mantengono solo le componenti singolari il cui valore singolare è superiore a una certa soglia.
3.  **Ricostruzione**: Si ricostruisce la matrice utilizzando solo le componenti selezionate.
4.  **Imposizione dei Dati di Training**: Si sovrascrivono le predizioni della matrice ricostruita con i valori reali noti del training set.
5.  **Iterazione**: Si calcola la differenza tra la matrice corrente e quella del passo precedente e si ripetono i passaggi.
[12:23] Questo metodo è chiamato *hard thresholding* perché "taglia" nettamente i valori singolari al di sotto della soglia, azzerandoli, mentre mantiene inalterati quelli al di sopra.
[12:31] All'interno di un ciclo iterativo, si dovranno implementare questi cinque passaggi:
1.  Calcolare la SVD.
2.  Mantenere solo i valori singolari "importanti".
3.  Sovrascrivere le entry della matrice con i valori di training.
4.  Calcolare la differenza rispetto all'iterazione precedente.
5.  Calcolare e salvare le metriche di valutazione sul test set.
[12:47] Infine, opzionalmente, si può visualizzare l'andamento dell'RMSE e del coefficiente di correlazione nel corso delle iterazioni.
[12:55] Per completare questi compiti sono previsti 20 minuti.
## Soluzione Compito 1: Predittore Banale
[13:26] Si analizza la soluzione del primo compito. L'obiettivo è produrre un vettore, `vals\_trivial`, contenente le predizioni per ogni campione del test set. Questo vettore avrà una dimensione pari al numero di campioni di test (20.000).
[13:37] Ogni elemento di questo vettore è una predizione per una specifica coppia (utente, film), definita dai vettori `row\_test` e `col\_test`.
[13:48] Il processo si svolge in due fasi. La prima è calcolare la valutazione media per ogni utente. Un modo diretto, ma computazionalmente meno efficiente, è usare un ciclo `for`.
[14:02] Si inizializza un vettore vuoto, `average\_rating`, con una dimensione pari al numero di utenti (`n\_people`).
[14:17] Successivamente, si itera su ogni utente `i` da 0 al numero totale di utenti.
[14:24] Per ogni utente `i`, si estrae la riga corrispondente dalla matrice `X\_full`.
[14:32] Si sommano tutti gli elementi di questa riga.
[14:36] Per calcolare la media, si deve dividere questa somma per il numero di valutazioni non nulle. Per contare queste valutazioni, si può creare una maschera booleana verificando dove gli elementi della riga sono maggiori di zero.
[14:45] Sommando questa maschera booleana (dove `True` vale 1 e `False` vale 0), si ottiene il numero di elementi non nulli.
[15:00] La valutazione media per l'utente `i` è quindi il rapporto tra la somma delle sue valutazioni e il numero di film che ha valutato.
[15:10] Un approccio alternativo e più pulito consiste nell'utilizzare direttamente i vettori delle triplette (righe, colonne, valori) del training set.
[15:20] Per calcolare la media per l'utente `i`, si filtra il vettore `vals\_train` usando una maschera booleana. La maschera è `True` dove l'ID utente in `row\_trains` è uguale a `i`.
[15:33] Applicando questa maschera a `vals\_train`, si estraggono solo le valutazioni dell'utente `i`.
[15:45] A questo punto, si può calcolare direttamente la media (`.mean()`) di questi valori.
[16:00] Verificando, si può notare che entrambi i metodi producono lo stesso risultato, confermando la correttezza di entrambi gli approcci.
[16:15] Una volta ottenuto il vettore `average\_rating` (con una valutazione media per ogni utente), lo si deve usare per generare le predizioni sul test set.
[16:22] Il modello banale predice un valore che dipende solo dall'utente, non dal film.
[16:35] Per ottenere il vettore `vals\_trivial`, si può usare un'indicizzazione avanzata. Il vettore `row\_test` contiene gli ID degli utenti per ogni campione di test. Usando `average\_rating[row\_test]`, si associa a ogni campione di test la valutazione media dell'utente corrispondente.
[16:55] In modo più esplicito (verboso), si potrebbe inizializzare un vettore vuoto `vals\_trivial` e iterare su ogni campione del test set.
[17:02] Per ogni campione `i`, si estrae l'ID dell'utente (`userId = row\_test[i]`) e l'ID del film (`movieId = col\_test[i]`).
[17:10] La predizione per questa coppia è semplicemente la valutazione media per `userId`, ignorando completamente `movieId`.
[17:20] Questo valore viene quindi assegnato all'i-esimo elemento di `vals\_trivial`. L'indicizzazione avanzata `average\_rating[row\_test]` è una versione vettorizzata e più efficiente di questo ciclo.
[17:50] Infine, per calcolare le metriche, si confrontano i valori predetti (`vals\_trivial`) con i valori reali del test set (`vals\_test`). L'errore è la discrepanza tra questi due vettori.
[18:00] Una domanda sorge sulla corrispondenza degli indici: l'utente nella posizione 0 del vettore `average\_rating` corrisponde all'utente con ID 0? Sì, grazie al passaggio di pulizia e compattazione degli indici eseguito in precedenza, c'è una corrispondenza diretta tra l'indice dell'array e l'ID dell'entità (utente o film).
## Confronto tra Sintassi: Groupby e Ciclo For
[00:00] L'operazione eseguita con la sintassi `groupby` è semanticamente identica a quella realizzata tramite un ciclo `for`. Sebbene la sintassi sia differente, l'azione sottostante è la medesima.
[00:05] Ad esempio, se si considera che la predizione sia un determinato valore e l'ID utente un altro, la riga di codice che utilizza `groupby` compie la stessa operazione di un ciclo `for` esplicito.
[00:10] In sostanza, il metodo `groupby` esegue internamente un'operazione equivalente a un ciclo. Una volta compreso questo, si capisce che si tratta solo di una diversa modalità sintattica per esprimere lo stesso concetto.
[00:15] I dati in questo contesto sono già mescolati (shuffled) e mantengono la coerenza tra di loro.
## Valutazione del Predittore Banale
[00:20] L'errore riscontrato in precedenza era dovuto a un'imprecisione nel codice: il ciclo stava iterando sulla dimensione del set di test (`test\_size`) invece che su quella del set di addestramento (`training\_size`).
[00:23] Dopo aver calcolato l'errore, definito come la discrepanza tra i valori reali del set di test e le previsioni del modello, si procede a calcolare le metriche di valutazione.
[00:27] La prima metrica è l'**Errore Quadratico Medio Radice (Root Mean Squared Error, RMSE)**.
*   **Definizione di RMSE**: L'RMSE è una metrica che misura la deviazione media delle previsioni rispetto ai valori reali. Si calcola elevando al quadrato ogni errore, calcolandone la media e infine estraendo la radice quadrata del risultato.
[00:31] La seconda metrica è il **coefficiente di correlazione di Pearson (R)**.
*   **Definizione di Correlazione di Pearson**: Questo coefficiente misura la forza e la direzione di una relazione lineare tra due variabili continue. Fornendo in input i due vettori (valori reali e predetti), restituisce un singolo numero compreso tra -1 e 1.
[00:34] I risultati per il predittore banale sono i seguenti:
*   **RMSE**: circa 1
*   **Correlazione**: 0.3
[00:38] Un valore di correlazione di 0.3 indica una correlazione debole ma positiva. Questo stesso indice era stato utilizzato anche per il PageRank, ottenendo in quel caso risultati migliori.
## Introduzione all'Algoritmo SVT (Singular Value Thresholding)
[00:43] Si passa ora all'implementazione dell'algoritmo SVT (Singular Value Thresholding).
[00:46] Per prima cosa, vengono impostati alcuni parametri fondamentali:
*   **Numero totale di iterazioni**: Definisce il numero massimo di cicli che l'algoritmo eseguirà.
*   **Soglia (threshold)**: Un valore critico per il filtraggio dei valori singolari.
*   **Tolleranza sull'incremento**: Un criterio per arrestare il ciclo quando la variazione tra un'iterazione e la successiva diventa sufficientemente piccola.
[00:51] La soglia, in particolare, è un iperparametro che richiede una calibrazione specifica per ogni problema (tuning). In questo caso, viene fornito un valore di 100, che è stato precedentemente testato e si è dimostrato efficace.
[00:58] In un contesto reale, sarebbe necessario variare leggermente i parametri per osservare quali combinazioni funzionano meglio.
## Implementazione dell'Algoritmo SVT: Passaggi Chiave
[01:02] L'implementazione dell'algoritmo inizia con il salvataggio di una copia della matrice `A`. Questa copia (`A\_old`) è necessaria per calcolare l'incremento, ovvero la differenza tra la matrice allo stato corrente e quella allo stato precedente, al fine di verificare la convergenza.
[01:10] Successivamente, si applica la Decomposizione a Valori Singolari (SVD) alla matrice `A`, utilizzando la funzione `np.linalg.svd` con il parametro `full\_matrices=False`.
[01:15] A questo punto, si applica il *thresholding*: i valori singolari inferiori alla soglia predefinita vengono impostati a zero.
[01:19] La matrice `A` viene quindi ricostruita. La ricostruzione avviene moltiplicando la matrice `U` per i valori singolari modificati (`S\_l`).
```math
A = U \cdot S_l
```
[01:22] Sebbene non sia il metodo più efficiente dal punto di vista computazionale, questa rappresentazione è la più chiara per comprendere il processo.
[01:25] Annullando i valori singolari inferiori alla soglia, si eliminano i contributi delle componenti principali associate a quelle direzioni, conservando solo le informazioni ritenute più significative.
[01:33] Un passaggio cruciale consiste nell'assicurarsi che, nelle posizioni in cui i dati sono noti (cioè nel set di addestramento), la matrice ricostruita corrisponda ai valori originali. Pertanto, per ogni componente $(i, j)$ nota, il valore corrispondente in `A` viene forzato ad essere uguale al valore corretto (`vals\_trained`).
[01:44] Si calcola poi l'incremento tra l'iterazione corrente e la precedente utilizzando la norma di Frobenius della differenza tra la matrice `A` e la sua copia `A\_old`.
```math
\text{incremento} = \| A - A_{\text{old}} \|_F
```
*   **Norma di Frobenius**: È una norma matriciale definita come la radice quadrata della somma dei quadrati dei suoi elementi. Misura la "grandezza" complessiva della matrice.
[01:50] Successivamente, si calcolano le predizioni del modello. I valori predetti sono estratti dalla matrice `A` ricostruita, in corrispondenza delle righe e delle colonne del set di test.
[01:57] La matrice `A` risulterà ora "piena" (dense), poiché il processo di SVD e ricostruzione, eliminando alcuni valori singolari, modifica la struttura originale, trasformando gli zeri in valori non nulli.
[02:04] L'errore viene calcolato come la differenza tra i valori reali del set di test (`vals\_test`) e i valori predetti.
[02:07] Infine, le metriche di valutazione vengono salvate in due liste:
*   `rmse\_list`: Viene aggiunto il valore dell'RMSE, calcolato come `np.sqrt(np.mean(errors**2))`.
*   `rho\_list`: Viene aggiunto il coefficiente di correlazione di Pearson tra i valori reali e quelli predetti.
## Analisi dei Risultati dell'Algoritmo SVT
[02:13] L'esecuzione dell'algoritmo richiede del tempo. L'obiettivo è ottenere un risultato migliore rispetto al predittore banale.
[02:17] Si ricorda che il predittore banale aveva un RMSE di circa 1 e una correlazione (rho) di circa 0.3.
[02:21] L'obiettivo è minimizzare l'RMSE (avvicinandolo a 0) e massimizzare la correlazione (avvicinandola a 1).
[02:26] All'inizio del processo iterativo, le performance del modello SVT sono peggiori di quelle del predittore banale. Tuttavia, migliorano progressivamente.
[02:30] Il coefficiente di correlazione `rho` supera rapidamente il valore del predittore banale, superando 0.5.
[02:33] L'RMSE, inizialmente più alto, diminuisce gradualmente ad ogni iterazione, con l'aspettativa che scenda al di sotto di 1.
## Visualizzazione e Confronto dei Risultati
[02:42] L'ultimo passaggio consiste nel visualizzare l'andamento storico dell'RMSE e del coefficiente di correlazione.
[02:46] Viene creato un grafico con due subplot affiancati (due colonne, una riga).
[02:50] Nel primo subplot (`ax[0]`), viene plottato l'andamento dell'RMSE nel tempo (numero di iterazioni).
[02:53] Nel secondo subplot (`ax[1]`), viene plottato l'andamento del coefficiente di correlazione `rho`.
[02:55] Per il confronto, i valori del predittore banale vengono rappresentati come una linea orizzontale, poiché si tratta di un singolo valore costante che non ha una cronologia di miglioramento.
[02:59] La funzione `ax.hlines` (horizontal line) viene utilizzata per tracciare una linea orizzontale parallela all'asse x, con un'ordinata `y` pari al valore della metrica del predittore banale (es. `rmse\_trivial`).
[03:09] Il risultato finale mostra che, per quanto riguarda il coefficiente di correlazione, l'algoritmo SVT diventa rapidamente migliore del predittore banale, superando il valore di 0.3438 in poche iterazioni.
[03:16] Tuttavia, per l'RMSE, il predittore banale si dimostra molto competitivo. L'algoritmo SVT riesce a scendere al di sotto del valore di riferimento del predittore banale solo verso la fine del processo e con un margine molto ridotto.
[03:22] Questo evidenzia l'importanza di utilizzare sempre un modello di base (baseline) come termine di paragone. Un modello molto semplice, come la media, può già fornire prestazioni notevoli, indicando che un algoritmo più complesso, pur migliorando il risultato, potrebbe non offrire un vantaggio così significativo.
## Applicazione alla Ricostruzione di Immagini: Algoritmo dalle Slide
[03:33] Viene ora presentata un'implementazione rapida di un altro algoritmo, identico a quello mostrato nelle slide del corso, applicato alla ricostruzione di immagini.
[03:39] Si tratta di una panoramica veloce, data la mancanza di tempo per approfondire tutti i dettagli. La logica è comunque molto simile a quella vista durante la lezione.
[03:45] Il codice completo con la soluzione è già disponibile sulla piattaforma WeBeep.
[03:48] Vengono importate le librerie standard e la libreria `PIL` (Pillow), utilizzata per la lettura e la manipolazione di immagini.
[03:52] Vengono fornite due funzioni ausiliarie:
1.  Una per ridimensionare l'immagine, utile per evitare di eseguire la SVD su matrici di dimensioni eccessive.
2.  Una per generare una maschera che simula una corruzione dell'immagine.
[04:00] Questa maschera, nell'algoritmo, corrisponde all'operatore di proiezione $P_\Omega$. Essa seleziona un insieme di pixel considerati corrotti, che l'algoritmo dovrà ricostruire.
[04:08] Si procede caricando un'immagine (`image.open`), ridimensionandola a una dimensione più piccola (es. 400 pixel) e trasformandola in un array NumPy.
[04:13] L'immagine viene convertita in scala di grigi calcolando la media lungo l'ultimo asse, ovvero mediando i canali RGB (Rosso, Verde, Blu).
[04:18] L'immagine utilizzata per questo esempio è il dipinto di Mondrian.
## Implementazione dell'Algoritmo di Ricostruzione
[04:21] Si definisce una percentuale di pixel corrotti, ad esempio il 50%. Viene generata una maschera di rumore casuale che copre il 50% dei pixel dell'immagine.
[04:27] Vengono impostati i parametri dell'algoritmo:
*   `max\_iter`: Numero massimo di iterazioni (es. 700).
*   `tolerance`: Tolleranza per il criterio di arresto (es. 0.01).
[04:31] I parametri `delta`, `tau` e `c0` sono quelli definiti nelle slide del corso.
*   `delta` è impostato a 1.2.
*   `tau` è calcolato come $\gamma \sqrt{n_1 n_2}$, dove $\gamma=5$ e $n_1, n_2$ sono le dimensioni dell'immagine.
*   `c0` è un valore standard scelto come da indicazioni presenti nel paper di riferimento.
[04:43] L'algoritmo utilizza tre matrici: `X`, `M` e `Y`.
*   `Y` è la variabile duale, inizializzata come $c_0 \delta P_\Omega(X)$, dove $P_\Omega(X)$ è l'applicazione della maschera alla matrice `X`.
[04:51] Il ciclo principale dell'algoritmo, pur avendo una struttura generale simile a quello visto in precedenza, esegue operazioni differenti.
[04:55] Viene applicato il *soft thresholding*.
*   **Soft Thresholding**: A differenza dell'hard thresholding che azzera i valori sotto soglia, il soft thresholding "restringe" (shrink) tutti i valori singolari verso lo zero di una quantità pari alla soglia $\tau$.
[05:00] La matrice `M` viene ricostruita, e la variabile duale `Y` viene aggiornata.
[05:03] I passaggi sono:
1.  Eseguire la SVD.
2.  Applicare il soft thresholding per spostare i valori verso zero.
3.  Ricostruire la matrice `M`.
4.  Calcolare il residuo `R`.
5.  Aggiornare la variabile duale `Y`.
[05:10] L'errore relativo viene calcolato ad ogni iterazione, e il ciclo continua finché la differenza tra iterazioni successive non scende al di sotto della tolleranza.
[05:14] Ogni `k` iterazioni (es. ogni 4), il risultato intermedio viene visualizzato.
## Risultati della Ricostruzione dell'Immagine
[05:16] Il punto di partenza è l'immagine originale con il 50% dei pixel corrotti dal rumore.
[05:20] Alla prima iterazione, l'immagine ricostruita è quasi completamente nera (piena di zeri), e l'errore (differenza tra l'immagine ricostruita e quella originale) è molto alto.
[05:26] L'algoritmo richiede molte iterazioni per convergere.
[05:31] Passo dopo passo, la ricostruzione si avvicina sempre di più all'immagine originale.
[05:35] Dopo un numero sufficiente di iterazioni, il risultato è notevolmente buono. L'algoritmo, partendo da un'immagine pesantemente corrotta, riesce a ricostruire la struttura originale con grande fedeltà.
[05:44] La SVD si dimostra particolarmente efficace in questo scenario, grazie alla natura geometrica e strutturata dell'immagine di Mondrian.
[05:49] Per testare l'algoritmo su un caso più complesso, viene utilizzata una foto di un paesaggio.
[05:54] Anche in questo caso, partendo da un'immagine con molto rumore, i risultati sono apprezzabili.
[06:02] Dopo 500 iterazioni, la ricostruzione è decente: sebbene presenti ancora rumore e sfocatura (blurriness), il risultato non è affatto male.
[06:07] Lasciando l'algoritmo in esecuzione per più tempo, il risultato continuerebbe a migliorare.
[06:11] Considerando che il punto di partenza era un'immagine quasi irriconoscibile, il risultato ottenuto è notevole.
## Pausa e Prossimi Argomenti
[06:17] Viene annunciata una pausa di 10 minuti.
[06:21] La lezione riprenderà 5 minuti prima delle 16:00 per trattare l'argomento successivo, JAX, che richiederà un'ora di spiegazione continua.
[06:26] Durante la pausa, è possibile porre domande.
## Introduzione a JAX
### Panoramica e contesto storico
[00:00] Questo documento illustra le funzionalità di JAX, una libreria per la differenziazione automatica. L'implementazione di librerie di questo tipo è un'operazione estremamente complessa, motivo per cui ci si affida a soluzioni già esistenti.
[00:13] In passato, le librerie di differenziazione automatica più rilevanti erano TensorFlow e PyTorch. Inizialmente, TensorFlow, sviluppato da Google, godeva di maggiore popolarità grazie alla sua velocità, ottenuta calcolando il grafo computazionale una sola volta.
[00:26] Successivamente, PyTorch (originariamente Torch) divenne più performante, grazie alla sua capacità di calcolare il grafo in modo dinamico. La sua API più semplice contribuì a superare la popolarità di TensorFlow.
[00:40] Attualmente, Google ha spostato la sua attenzione da TensorFlow a JAX, in quanto quest'ultimo è considerato più potente e dotato di un'API più intuitiva.
[00:49] La principale differenza tra JAX e librerie come TensorFlow o PyTorch risiede nel suo scopo: JAX è esclusivamente una libreria per la differenziazione automatica. Ciò significa che non fornisce strumenti nativi per la creazione di reti neurali, rappresentando uno svantaggio.
[01:03] Tuttavia, esistono librerie di terze parti che si basano su JAX per implementare reti neurali e ottimizzatori. Un esempio è Keras che, a partire dalla sua terza versione, supporta JAX, PyTorch e TensorFlow come backend per la differenziazione automatica nella costruzione di reti neurali.
### Configurazione dell'ambiente di esecuzione
[01:20] Per sfruttare appieno le capacità di JAX, è importante configurare correttamente l'ambiente di esecuzione. È necessario selezionare un runtime di tipo GPU.
[01:29] Questa scelta è fondamentale perché le operazioni di algebra lineare, tipiche delle reti neurali, traggono grande vantaggio dall'accelerazione hardware offerta dalle GPU. JAX semplifica notevolmente la connessione e l'utilizzo di queste risorse.
[01:44] Di norma, se JAX è stato installato con il supporto per GPU, dovrebbe rilevarla e utilizzarla automaticamente. Successivamente, verranno mostrate le procedure per verificare quali dispositivi JAX sta effettivamente utilizzando.
## Fondamenti di JAX
### Importazione delle librerie e API principali
[02:01] L'utilizzo di JAX inizia con l'importazione della libreria stessa. Una delle caratteristiche principali di JAX è la sua API, molto simile a quella di NumPy, che viene convenzionalmente importata come `jnp`.
[02:08] Si importa quindi `jax` e il suo modulo `jax.numpy` con l'alias `jnp`. Per confronto e per alcune operazioni, si importa anche la libreria standard `numpy`.
[02:14] Esiste anche un'API di livello inferiore, chiamata `LAX`, che è notevolmente più complessa. Sebbene l'API `jnp` verrà utilizzata per circa il 95% del tempo, alcuni aspetti di `LAX` verranno illustrati per dimostrare la potenza e la velocità che JAX può raggiungere grazie a questa interfaccia a basso livello.
[02:38] Infine, si importa la libreria `matplotlib` per la visualizzazione grafica dei dati.
### Creazione e utilizzo degli array JAX
[02:44] Per iniziare con un esempio pratico, si può utilizzare `jax.numpy` per creare un array. La funzione `linspace` viene usata per generare 1000 punti equispaziati tra 0 e 10.
[02:52] Successivamente, si definisce una funzione matematica, come il prodotto di un seno e un coseno. Tutti gli array utilizzati in queste operazioni non sono array NumPy, ma array specifici di JAX.
[03:10] L'utilizzo di questa API è molto intuitivo, poiché la sintassi è identica a quella di NumPy.
[03:17] Analizzando un array creato con `jnp`, si nota che la sua struttura è molto simile a quella di un array NumPy, sebbene venga identificato semplicemente come `array`.
### Immutabilità degli array JAX
[03:29] Una caratteristica peculiare e fondamentale di JAX è l'immutabilità dei suoi array. A differenza degli array NumPy, gli array JAX non possono essere modificati dopo la loro creazione.
[03:38] Per illustrare questa differenza, si può creare un array NumPy di dimensione 10. È possibile accedere a un elemento, ad esempio quello in posizione 0, e assegnargli un nuovo valore, come 23. Questa operazione modifica l'array originale "in-place".
[03:53] In JAX, questo tipo di modifica diretta è impossibile. Non si può selezionare un elemento di un array e assegnargli un nuovo valore. Sebbene possa sembrare una limitazione, questa è una scelta di progettazione deliberata i cui vantaggi diventeranno chiari in seguito.
[04:10] Per modificare un valore, è necessario creare un nuovo array. Questa operazione può apparire inefficiente, ma ha una sua logica nel contesto del funzionamento di JAX.
[04:18] Ad esempio, si inizializza una matrice di zeri di dimensioni 3x3 con JAX. Per modificare una riga, come la prima (indice 1), e impostare tutti i suoi valori a 1, non si modifica la matrice originale.
[04:28] Si utilizza una sintassi specifica che restituisce un nuovo array JAX. Questo nuovo array è una copia dell'originale con la modifica richiesta applicata.
[04:40] L'operazione consiste nel prendere la matrice originale, copiata, e applicare la modifica, restituendo una nuova matrice. Apparentemente, questa operazione è dispendiosa in termini di memoria e calcolo. Se usata in modo ingenuo, lo è. Tuttavia, JAX dispone di meccanismi interni per ottimizzare questi processi e renderli efficienti.
[04:59] L'espressività di NumPy è completamente preservata. Tutte le operazioni di slicing, selezione di righe e colonne, sono disponibili anche in JAX.
[05:12] L'unica differenza fondamentale da tenere a mente è la necessità di creare copie invece di modificare gli array "in-place", utilizzando metodi specifici che possono apparire non convenzionali.
[05:24] Ad esempio, è possibile impostare a 7 il valore degli elementi delle righe prima e ultima, ma solo per le colonne dall'indice 1 in poi, sempre generando un nuovo array.
### Gestione dei numeri casuali
[05:39] Un'altra differenza significativa rispetto a NumPy riguarda la gestione dei numeri casuali. In NumPy, si imposta un seme globale con `np.random.seed()` e tutte le successive chiamate a funzioni casuali dipenderanno da quel seme.
[05:50] In JAX, l'approccio è diverso e più esplicito. È necessario creare una "chiave" (`key`) a partire da un seme numerico. Questa chiave deve poi essere passata come argomento a ogni funzione che genera numeri casuali.
[06:03] Il processo prevede quindi un passaggio aggiuntivo:
1.  Si definisce un seme, ad esempio `seed = 0`.
2.  Si crea una chiave usando `jax.random.PRNGKey(seed)`.
3.  Quando si chiama una funzione di generazione casuale (es. `jax.random.normal`), si passano la chiave e le dimensioni dell'array desiderato.
[06:13] Anche questa particolarità, come l'immutabilità, troverà una giustificazione logica nel contesto delle ottimizzazioni di JAX.
## Ottimizzazione e Performance in JAX
### JAX e l'accelerazione hardware (CPU, GPU, TPU)
[06:26] Un grande vantaggio di JAX è la sua capacità di essere "agnostico" rispetto all'acceleratore hardware. Il codice JAX può essere eseguito su CPU, GPU e TPU senza modifiche sostanziali.
[06:35] È tuttavia cruciale essere consapevoli di dove risiedono i dati (cioè in quale memoria, della CPU o della GPU), poiché il trasferimento di dati tra questi dispositivi può essere un'operazione costosa in termini di tempo.
[06:46] Per dimostrarlo, si analizzano diversi scenari di calcolo.
[06:50] **Scenario 1: Tutto su GPU.** Si crea un array `x` con JAX. Di default, se una GPU è disponibile e configurata, questo array viene allocato sulla memoria della GPU. Si calcola il prodotto scalare tra l'array e la sua trasposta. Tutte le computazioni avvengono direttamente sulla GPU.
[07:08] **Scenario 2: Tutto su CPU.** Si esegue la stessa operazione utilizzando NumPy. Poiché NumPy non può utilizzare la GPU, sia i dati che i calcoli rimangono sulla CPU.
[07:20] **Scenario 3: Trasferimento implicito CPU -> GPU.** Si crea un array con NumPy (quindi sulla CPU). Successivamente, si utilizza una funzione JAX (come il prodotto scalare) su questo array. JAX, per eseguire il calcolo sulla GPU, deve prima copiare implicitamente i dati dalla memoria della CPU a quella della GPU. Questa operazione di copia introduce un overhead.
[07:38] **Scenario 4: Trasferimento esplicito CPU -> GPU.** È possibile gestire il trasferimento dei dati in modo esplicito. Si può forzare lo spostamento di un array dalla CPU alla GPU utilizzando la funzione `jax.device\_put`. Successivamente, si esegue il calcolo JAX sui dati che sono già stati trasferiti sulla GPU. Questo approccio è concettualmente simile allo scenario 1, ma rende esplicito il momento del trasferimento di memoria.
### Analisi delle performance
[08:00] I risultati dei test di performance mostrano differenze significative:
-   **Tutto su GPU:** L'operazione richiede circa 15 millisecondi.
-   **Tutto su CPU:** L'operazione richiede più di 10 volte tanto. Questo dimostra il vantaggio dell'utilizzo della GPU per questo tipo di calcoli.
-   **Trasferimento implicito:** Il costo computazionale raddoppia rispetto al calcolo interamente su GPU, passando da 15 a 35 millisecondi. Questo evidenzia il costo non trascurabile del trasferimento di dati.
-   **Trasferimento esplicito:** Le performance sono quasi equivalenti a quelle del primo scenario (circa 2 millisecondi in più), dimostrando che un controllo esplicito del trasferimento dati è efficiente.
[08:26] Il messaggio chiave è prestare attenzione alla localizzazione della memoria, poiché i trasferimenti possono avere un impatto notevole sulle performance.
### Compilazione Just-In-Time (JIT)
[08:34] Un altro componente fondamentale di JAX è la compilazione JIT (Just-In-Time), accessibile tramite la funzione `jax.jit`.
[08:39] Questa funzione ottimizza il codice Python. La prima volta che una funzione "jittata" viene eseguita, JAX analizza le operazioni che compie e la compila in una versione ottimizzata e molto più veloce per le esecuzioni successive.
[08:53] Si definisce una funzione di visualizzazione e una funzione `selu` (Scaled Exponential Linear Unit), una funzione di attivazione comune nelle reti neurali, la cui definizione è moderatamente complessa.
[09:05] Per ottimizzare una funzione con JIT, si passa la funzione stessa a `jax.jit`.
```math
\text{funzione\_ottimizzata} = \text{jax.jit}(\text{funzione\_originale})
```
[09:10] `jax.jit` è una funzione di ordine superiore: accetta una funzione come input e restituisce una nuova funzione (la versione ottimizzata) come output.
[09:25] La funzione restituita può essere utilizzata esattamente come l'originale. Si può quindi misurare il tempo di esecuzione della funzione normale e di quella compilata.
[09:35] La funzione `selu` ha un andamento quasi orizzontale per valori negativi e lineare per valori positivi, un comportamento tipico delle funzioni di attivazione.
[09:46] Confrontando i costi computazionali, la versione compilata della funzione risulta circa 10 volte più veloce di quella non compilata.
[09:56] La compilazione JIT è anche il meccanismo che permette a JAX di superare l'inefficienza apparente dell'immutabilità degli array. Quando si compila un'espressione che sembra richiedere una copia (come la modifica di un elemento), JAX è abbastanza intelligente da capire che, in molti casi, non è necessario creare una copia e può modificare l'array "in-place" a livello di codice compilato. Questa ottimizzazione è possibile solo dopo la compilazione.
## Differenziazione Automatica in JAX
### Calcolo del gradiente con `jax.grad`
[10:31] JAX è, prima di tutto, una libreria di differenziazione automatica. La funzione principale per questo scopo è `jax.grad`.
[10:39] Similmente a `jax.jit`, `jax.grad` è una funzione di ordine superiore: riceve una funzione come input e restituisce una nuova funzione che calcola il gradiente della funzione di input rispetto ai suoi argomenti.
```math
\text{funzione\_gradiente} = \text{jax.grad}(f)
```
[10:51] È possibile specificare rispetto a quale argomento calcolare il gradiente utilizzando il parametro opzionale `argnums`.
[10:57] Si consideri come esempio una funzione parabolica $f(x) = x^2$. Si può usare `jax.grad` per ottenere la sua derivata prima.
[11:06] Una delle grandi potenzialità di JAX è la possibilità di comporre `grad` ricorsivamente per calcolare derivate di ordine superiore (seconda, terza, ecc.) in modo efficiente e robusto.
```math
\begin{align*}
\text{grad\_f} &= \text{jax.grad}(f) \\
\text{grad\_grad\_f} &= \text{jax.grad}(\text{grad\_f}) \quad \text{(derivata seconda)} \\
\text{grad\_grad\_grad\_f} &= \text{jax.grad}(\text{grad\_grad\_f}) \quad \text{(derivata terza)}
\end{align*}
```
[11:13] Questa capacità di comporre l'operatore di gradiente ripetutamente è una delle caratteristiche che rendono JAX più potente di altre librerie.
[11:23] Se la funzione ha un solo argomento, `grad` calcola automaticamente la derivata rispetto a quell'unico input.
### Jacobiano e Hessiano per funzioni vettoriali
[11:30] Se una funzione ha più di un input, come una funzione $f(x, y)$, il concetto di gradiente si estende. Si possono calcolare le derivate parziali rispetto a ciascun componente.
[11:39] Il gradiente di una funzione a valori vettoriali non è più uno scalare, ma un vettore (il Jacobiano). Il gradiente del gradiente diventa una matrice (l'Hessiano), che contiene tutte le derivate parziali seconde.
[11:53] Per calcolare queste quantità, invece di `grad`, si utilizza la funzione `jax.jacobian`.
[12:02] JAX offre due implementazioni per il calcolo del Jacobiano: `jacfwd` (forward-mode) e `jacrev` (reverse-mode).
    - **Differenziazione automatica forward-mode (`jacfwd`)**: Più efficiente per matrici Jacobiane "alte" (più righe che colonne).
    - **Differenziazione automatica reverse-mode (`jacrev`)**: Più efficiente per matrici Jacobiane "larghe" (più colonne che righe).
[12:16] Per calcolare l'Hessiano, la combinazione più efficiente è solitamente applicare prima `jacrev` e poi `jacfwd`.
[12:24] Ad esempio, data una funzione `F` con due input, si può calcolare il Jacobiano rispetto a entrambi. Se `F` è una funzione a valori vettoriali, applicando nuovamente il Jacobiano si ottiene l'Hessiano.
[12:35] Il risultato non è un singolo valore, ma un vettore (Jacobiano) o una matrice (Hessiano).
[12:41] In sintesi: per funzioni con più input o output, si usa `jacobian`. Per ottenere le massime performance nel calcolo dell'Hessiano, la regola pratica è usare `jacfwd(jacrev(f))`.
[12:57] Naturalmente, gli input di una funzione possono essere vettori o array, non solo scalari. Si può calcolare il Jacobiano e l'Hessiano rispetto a un singolo argomento che è un array multidimensionale.
[13:10] Scrivere una funzione con più argomenti scalari o con un unico argomento vettoriale sono due modi equivalenti per rappresentare lo stesso problema.
### Gestione delle funzioni non differenziabili
[13:21] È interessante analizzare come JAX gestisce le funzioni non differenziabili, come la funzione valore assoluto, $f(x) = |x|$, che non è differenziabile in $x=0$.
[13:30] JAX adotta un approccio pragmatico. Invece di restituire un errore o un valore non utilizzabile (come `NaN`), prende una decisione definita per garantire la stabilità numerica del codice.
[13:38] Per verificare questo comportamento, si può definire la funzione valore assoluto e calcolarne il gradiente.
[13:45] Valutando il gradiente in punti vicini a zero, si osserva che:
-   Per un valore leggermente positivo, il gradiente è `1`.
-   Per un valore leggermente negativo, il gradiente è `-1`.
-   Esattamente in `0`, dove la funzione non è differenziabile, JAX restituisce `1`.
[14:02] Questa scelta convenzionale serve a evitare la propagazione di valori infiniti o non numerici (`NaN`) all'interno del codice.
## Vettorizzazione con `vmap`
### Il concetto di `vmap`
[14:12] Il terzo concetto fondamentale di JAX, insieme a JIT e `grad`, è `vmap` (vectorizing map). `vmap` è uno strumento per vettorizzare il codice in modo automatico.
[14:21] La scrittura di codice vettorizzato è cruciale per evitare i cicli `for` espliciti in Python, che sono notoriamente lenti. `vmap` trasforma un ciclo `for` in un'operazione che si comporta come codice compilato e vettorizzato, rendendolo estremamente efficiente.
### Esempio pratico di `vmap`
[14:38] Si definisce una funzione arbitraria, ad esempio un "prodotto scalare personalizzato" che calcola il prodotto scalare tra due vettori e ne eleva al quadrato il risultato.
[14:51] Si supponga di avere due matrici, `X` e `Y`, e di voler applicare questa funzione a ogni coppia di righe corrispondenti delle due matrici. Questo è un pattern comune, ad esempio, nell'elaborazione di un dataset, dove ogni riga rappresenta un campione di dati.
[15:07] **Approccio ingenuo:** Si utilizza un ciclo `for` in Python per iterare su ogni riga delle matrici e applicare la funzione. Questo approccio sarà lento.
[15:26] **Prima ottimizzazione (JIT):** Una prima idea è compilare la funzione che contiene il ciclo `for` con JIT. Questo si può fare usando la sintassi del decoratore `@jax.jit` sopra la definizione della funzione.
[15:38] Il decoratore `@` è una scorciatoia sintattica. Scrivere `@jax.jit` sopra una funzione `mia\_funzione` è equivalente a scrivere `mia\_funzione = jax.jit(mia\_funzione)` dopo la sua definizione.
[15:55] La compilazione JIT del ciclo migliorerà le performance rispetto alla versione ingenua.
[16:01] **Seconda ottimizzazione (`vmap`):** Si può fare di meglio usando `vmap`. `vmap` agisce come un ciclo `for` implicito e ottimizzato.
[16:06] A `vmap` si passano due argomenti principali:
1.  La funzione da applicare.
2.  Un argomento `in\_axes` che specifica su quali assi degli input la funzione deve essere "mappata" (iterata).
[16:14] Ad esempio, `in\_axes=(0, 0)` significa che si vuole applicare la funzione iternado lungo l'asse 0 (le righe) del primo argomento e l'asse 0 del secondo argomento. `vmap` restituisce una nuova funzione vettorizzata.
[16:40] **Terza ottimizzazione (JIT + `vmap`):** L'approccio migliore consiste nel combinare `vmap` e JIT. Si applica prima `vmap` per creare un ciclo for ottimizzato e poi si compila la funzione risultante con JIT. Questo garantisce le massime performance.
[17:00] L'argomento `in\_axes` è una sequenza di interi la cui lunghezza corrisponde al numero di argomenti della funzione. Ogni intero indica l'asse su cui iterare per l'argomento corrispondente.
[17:18] Questo meccanismo è un'estensione del concetto di `axis` presente in funzioni NumPy come `min` o `sum`, ma generalizzato a qualsiasi funzione definita dall'utente.
### Analisi delle performance con `vmap`
[17:34] I risultati del confronto tra i diversi approcci sono impressionanti:
-   **Approccio ingenuo (ciclo `for`):** Richiede circa mezzo secondo.
-   **Approccio vettorizzato (`vmap`):** È quasi 500 volte più veloce.
-   **Solo JIT (senza `vmap`):** Le performance sono paragonabili a quelle di `vmap`, leggermente inferiori a un millisecondo.
-   **`vmap` + JIT:** È altre 10 volte più veloce, risultando circa 5000 volte più rapido dell'approccio ingenuo.
[18:00] Questa capacità di trasformare funzioni semplici in codice estremamente performante è ciò che rende JAX così potente.
### Riepilogo e prossimi passi
[18:11] I concetti chiave di JAX finora discussi sono:
1.  **Compilazione Just-In-Time (`jit`)**: Per l'ottimizzazione della velocità.
2.  **Differenziazione automatica (`grad`, `jacobian`)**: Per il calcolo di gradienti e derivate di ordine superiore.
3.  **Vettorizzazione (`vmap`)**: Per trasformare cicli in operazioni efficienti.
[18:18] Questi costituiscono l'interfaccia di alto livello di JAX. Ora si procederà ad analizzare l'interfaccia di livello inferiore per comprendere meglio le scelte di progettazione più particolari, come l'immutabilità degli array e la gestione delle chiavi per i numeri casuali.
## Introduzione alle Funzioni Chiave di JAX
[00:00] Le funzioni fondamentali da comprendere approfonditamente sono `jit`, `vmap` e `grad`. Questa sezione si concentrerà sui dettagli del funzionamento interno di JAX, analizzando il meccanismo che si cela dietro l'interfaccia di alto livello.
## Interfaccia di Alto Livello (JMP) vs. Basso Livello (LAX)
### Flessibilità dell'Interfaccia NumPy (JMP)
[00:09] L'interfaccia di alto livello di JAX, che emula NumPy e viene comunemente importata come `jmp`, è caratterizzata da una notevole flessibilità. Ad esempio, è possibile sommare un numero intero e un numero in virgola mobile senza generare errori, poiché JAX gestisce automaticamente la conversione dei tipi.
### Rigidità dell'Interfaccia di Basso Livello (LAX)
[00:17] Al contrario, l'interfaccia di basso livello, nota come `lax`, impone regole più stringenti. Le operazioni, come la somma, sono consentite solo tra operandi che possiedono lo stesso tipo di dato.
[00:22] Ad esempio, è possibile sommare due numeri in virgola mobile (es. `1.0` e `1.0`), ma il tentativo di sommare un intero e un numero in virgola mobile provocherà un errore.
[00:28] L'interfaccia `lax` richiede una promozione esplicita dei tipi (`explicit type promotion`). L'errore generato in questo caso sarebbe del tipo: `LAX-add requires arguments to have the same types, int32 and float32`.
[00:35] Questa rigidità è motivata dalla necessità di evitare costi computazionali impliciti. La conversione automatica di un tipo di dato in un altro (`implicit cast`) è un'operazione che può avere un impatto sulle prestazioni.
[00:43] Lavorando a basso livello, è cruciale avere il controllo su ogni dettaglio che potrebbe influenzare il costo computazionale del risultato finale. L'interfaccia `lax` è quindi più potente ma meno intuitiva per l'utente.
### Esempio: Operazione di Convoluzione
[00:52] Un esempio pratico è l'operazione di convoluzione. Utilizzando l'interfaccia di alto livello, simile a NumPy, è sufficiente passare gli array `x` e `y` alla funzione `convolve` per ottenere il risultato.
[01:00] Se si utilizza l'interfaccia di basso livello, l'operazione richiede una configurazione più dettagliata. Innanzitutto, è necessario assicurarsi che entrambi gli array di input abbiano lo stesso tipo di dato, ad esempio `float`.
[01:05] Inoltre, è obbligatorio specificare argomenti aggiuntivi per definire il comportamento della convoluzione, come la dimensione della finestra (`window\_dimensions`) e il `padding` da applicare all'inizio e alla fine degli array.
[01:12] Questo dimostra che l'interfaccia di basso livello offre maggiore potenza e un numero superiore di opzioni, ma richiede una conoscenza approfondita della documentazione.
[01:19] È importante notare che il risultato finale delle due interfacce è identico, poiché l'interfaccia di alto livello di JAX agisce come un "wrapper", ovvero un involucro, per quella di basso livello.
## Limiti della Compilazione Just-In-Time (JIT)
### Dipendenza dalla Forma Statica dell'Output
[01:25] La compilazione *Just-In-Time* (JIT) è un meccanismo potente, ma presenta delle limitazioni significative.
[01:31] Si consideri una funzione che, dato un array `x`, restituisce un nuovo array contenente solo gli elementi negativi di `x`. Questo si ottiene applicando una maschera booleana che seleziona i valori per cui la condizione `x < 0` è vera.
[01:42] Questa operazione, sebbene comune, non può essere compilata con `jit`. JAX solleverà un errore.
[01:48] La ragione di questo comportamento risiede nel fatto che la dimensione (`shape`) dell'array risultante dipende dai valori contenuti nell'input `x`.
[01:53] Se `x` è `[-1, -1]`, l'output avrà dimensione 2. Se `x` è `[1, 1]`, l'output avrà dimensione 0.
[01:59] Poiché la forma dell'output è dinamica e non può essere determinata a priori, JAX non è in grado di compilare il codice *Just-In-Time*.
[02:04] Questa è una limitazione intrinseca di JAX che bisogna conoscere e accettare. La maggior parte dei problemi con `jit` emerge quando si ha a che fare con strutture dati dinamiche.
[02:11] La compilazione `jit` è applicabile solo a codice "statico", ovvero codice i cui risultati non cambiano né in forma (`shape`) né in tipo (`dtype`). JAX necessita di queste informazioni per poter ottimizzare il codice in modo efficace.
## Funzionamento Interno della Compilazione JIT
### Il Processo di Tracciamento (Tracing)
[02:21] Per comprendere meglio il funzionamento della compilazione JIT, si analizza il processo passo dopo passo. Si definisca una funzione `f` che calcola il prodotto scalare tra due vettori `x` e `y`.
[02:27] All'interno della funzione, si inseriscono delle istruzioni di stampa (`print`) per visualizzare gli argomenti `x` e `y` e il risultato dell'operazione.
[02:37] Si creano due vettori casuali e si invoca la funzione `f` due volte consecutive con questi argomenti.
[02:44] Si osservano due fenomeni importanti. Il primo è che, durante la prima esecuzione, gli argomenti `x` e `y` stampati non sono gli array numerici che ci si aspetterebbe.
[02:50] Sebbene la funzione sia stata chiamata con array NumPy, al suo interno `x` e `y` appaiono come oggetti speciali. Questo accade perché, la prima volta che una funzione viene eseguita dopo essere stata decorata con `@jit`, JAX non passa i dati reali.
[02:58] Invece, passa delle strutture dati speciali chiamate **tracer**.
> **Tracer**: Un oggetto segnaposto che non contiene i valori numerici di un array, ma ne registra le proprietà astratte come la forma (`shape`) e il tipo (`dtype`). I tracer attraversano il flusso di operazioni della funzione per registrarne la struttura computazionale.
[03:02] In questo modo, JAX "traccia" il percorso dei dati all'interno della funzione per comprenderne il funzionamento e poterla ottimizzare.
### Compilazione e Rimozione degli Effetti Collaterali
[03:09] Il secondo punto chiave è che, durante la seconda chiamata alla funzione, le istruzioni di stampa non vengono più eseguite e i tracer non appaiono.
[03:13] La compilazione JIT si svolge in due fasi:
1.  **Tracciamento (Tracing)**: Durante la prima chiamata, la funzione viene tracciata. Questa operazione avviene una sola volta per una data combinazione di forma e tipo degli input.
2.  **Compilazione e Ottimizzazione**: Una volta tracciata, la funzione viene compilata in una versione ottimizzata. Durante questo processo, tutti gli **effetti collaterali** vengono rimossi.
> **Effetto Collaterale (Side Effect)**: Qualsiasi interazione di una funzione con uno stato esterno al suo ambito, come la stampa a schermo, la modifica di variabili globali o la lettura/scrittura di file.
[03:21] Vengono conservate solo le operazioni strettamente necessarie al calcolo del risultato.
[03:26] Il messaggio fondamentale è che la prima chiamata a una funzione compilata con `jit` è costosa, poiché include il tempo necessario per il tracciamento e la compilazione.
[03:31] Dalla seconda chiamata in poi, l'esecuzione è estremamente rapida, poiché JAX utilizza la versione compilata e salvata in memoria.
### Analisi della Grammatica Astratta con `make\_jaxpr`
[03:37] È possibile ispezionare ciò che JAX fa internamente utilizzando la funzione `make\_jaxpr`. Questa funzione, data una funzione Python, restituisce la grammatica astratta (`jaxpr`) che JAX costruisce per rappresentare e ottimizzare il calcolo.
[03:47] Applicando `make\_jaxpr` a una funzione, si può osservare come JAX rappresenti gli input tramite i tracer.
[03:52] L'oggetto `jaxpr` risultante contiene informazioni cruciali, come il tipo e la dimensione degli array di input. Ad esempio, un input `A` potrebbe essere descritto come un array in virgola mobile a 32 bit (`float32`) di dimensione `3x4`, e un input `B` come un array di dimensione `4`.
[04:00] Questo conferma perché JAX richiede che la forma e il tipo degli array rimangano costanti: queste informazioni sono indispensabili per costruire la grammatica astratta necessaria all'ottimizzazione.
[04:10] La grammatica `jaxpr` descrive testualmente la sequenza di chiamate all'API di basso livello (`lax`) che implementano la funzione.
[04:15] Ad esempio, si possono vedere variabili intermedie (es. `a`, `b`, `c`) e le operazioni che le legano. Una riga potrebbe indicare che la variabile `c` (con un certo tipo e dimensione) è il risultato della funzione `lax.add` applicata alla variabile `a` e a una costante `1.0`.
[04:28] In sintesi, la compilazione JIT consiste nel tradurre la funzione Python in una grammatica astratta basata sull'API di basso livello (`lax`) e nel salvare questa rappresentazione ottimizzata.
[04:35] La rigidità dell'API `lax`, che richiede la conoscenza esatta di tipi e dimensioni, è proprio ciò che ne garantisce l'elevata efficienza e permette a JAX di ottimizzare il codice in modo aggressivo.
## Gestione degli Argomenti Dinamici in JIT
### Problemi con le Strutture di Controllo Condizionali
[04:44] Un altro problema comune con la compilazione JIT riguarda gli argomenti che modificano il flusso di esecuzione del codice.
[04:49] Si consideri una funzione `f` che accetta un array `x` e un valore booleano `neg`. A seconda che `neg` sia `True` o `False`, la funzione restituisce `-x` o `x`.
[04:58] Il tentativo di compilare questa funzione con `jit` genererà un errore.
[05:04] Il motivo è che il valore dell'argomento `neg` determina quale ramo del costrutto `if-else` viene eseguito, alterando la struttura computazionale (e quindi la grammatica astratta) della funzione.
[05:14] Poiché la grammatica non può essere fissata a priori se dipende dal valore di un input, JAX non può procedere con la compilazione.
### Soluzione 1: Riformulazione Matematica
[05:20] Esistono soluzioni alternative per aggirare questo limite. Una prima strategia consiste nel riscrivere la funzione eliminando le clausole `if`.
[05:24] Si può sfruttare il fatto che i valori booleani `True` e `False` possono essere trattati come `1` e `0` in operazioni aritmetiche.
[05:30] La funzione può essere riscritta utilizzando solo moltiplicazioni e sottrazioni. Ad esempio, l'espressione `x * (1 - 2 * neg)` produce lo stesso risultato della funzione con `if-else`.
[05:35] Se `neg` è `False` (cioè `0`), l'espressione diventa `x * (1 - 0) = x`.
[05:39] Se `neg` è `True` (cioè `1`), l'espressione diventa `x * (1 - 2) = -x`.
[05:46] Questa versione della funzione può essere compilata con `jit` perché la sua struttura computazionale è fissa e non dipende dal valore di `neg`.
### Soluzione 2: Argomenti Statici (`static\_argnums`)
[05:53] Un'altra soluzione, sebbene meno raccomandata, è l'uso degli argomenti statici.
[05:58] Utilizzando il decoratore `@partial` o specificando l'opzione `static\_argnums` in `@jit`, si può indicare a JAX di trattare un certo argomento come "statico".
[06:02] Questo approccio istruisce JAX a creare e compilare una versione specializzata della funzione per ogni valore unico che l'argomento statico assume.
[06:10] Nel nostro esempio, specificare `neg` come argomento statico (passando il suo indice, `1`, a `static\_argnums`) equivale a definire due funzioni separate, `f\_true` e `f\_false`, e a compilare entrambe con `jit`.
[06:20] Il meccanismo è automatico: JAX gestisce la creazione e la memorizzazione delle diverse versioni compilate.
[06:25] Di conseguenza, ogni volta che la funzione viene chiamata con un nuovo valore per l'argomento statico, JAX deve eseguire un nuovo tracciamento e una nuova compilazione.
[06:31] Ad esempio, la prima chiamata con `True` innescherà un tracciamento. La prima chiamata con `False` innescherà un altro tracciamento.
[06:37] Anche una chiamata con `1` (di tipo intero) innescherà un'ulteriore compilazione, poiché il tipo è diverso da quello booleano di `True`.
[06:45] Questa soluzione non è ideale perché nasconde il fatto che si stanno eseguendo multiple compilazioni, un'operazione costosa, ogni volta che il valore dell'argomento statico cambia.
## Purezza delle Funzioni e JAX
### Definizione di Funzione Pura
[06:56] Un concetto fondamentale per il corretto funzionamento di JAX è quello di **funzione pura**.
> **Funzione Pura (Pure Function)**: Una funzione che soddisfa due condizioni:
> 1.  Tutti i dati di cui ha bisogno le vengono passati esclusivamente tramite i suoi argomenti.
> 2.  Restituisce sempre lo stesso output se invocata con gli stessi input. Non ha effetti collaterali.
### Problemi con le Funzioni Impure e JIT
[07:07] Si consideri una variabile globale `g` inizializzata a `0` e una funzione che somma il suo argomento `x` a `g`.
[07:12] Questa funzione è **impura** perché il suo risultato dipende da `g`, una variabile esterna al suo scope (ambito).
[07:21] Quando si compila una funzione impura con `jit`, JAX "congela" il valore delle variabili esterne al momento della prima compilazione.
[07:27] Se si compila la funzione e la si chiama con `x = 4`, il risultato sarà `4 + 0 = 4`.
[07:32] Se successivamente si modifica la variabile globale, ad esempio `g = 10`, e si richiama la funzione compilata con `x = 5`, il risultato non sarà `5 + 10 = 15`.
[07:38] Il risultato sarà invece `5`, perché JAX ha memorizzato (`cached`) il valore di `g` (`0`) durante la prima compilazione e continuerà a usare quello.
[07:45] È quindi fondamentale prestare la massima attenzione: quando si usa `jit`, tutti i dati necessari alla funzione devono essere passati esplicitamente come argomenti.
[07:51] In caso contrario, `jit` ottimizzerà il codice utilizzando il valore che la variabile esterna aveva al momento del tracciamento, portando a risultati errati e difficili da diagnosticare.
[08:04] È possibile che il valore della variabile globale venga "aggiornato" solo se si forza una nuova compilazione, ad esempio cambiando la forma o il tipo di un argomento di input, costringendo così il tracer a rieseguire l'intero processo.
### Comportamenti Inattesi e Debug
[08:12] Un'altra conseguenza della natura di JAX è che alcuni errori, come l'indicizzazione fuori dai limiti (`out-of-bounds indexing`), potrebbero non essere segnalati come ci si aspetterebbe.
[08:18] Ad esempio, se si ha un array di 10 elementi e si tenta di modificare l'undicesimo elemento, JAX potrebbe non sollevare un errore e semplicemente non eseguire alcuna operazione.
[08:25] Questo comportamento, simile a quello che si può riscontrare in linguaggi come il C++, richiede particolare attenzione durante il debug, poiché si potrebbe accedere a aree di memoria non allocate senza ricevere avvisi.
## Gestione dei Numeri Casuali in JAX
### Il Modello a Stato di NumPy
[08:34] Per comprendere l'approccio di JAX alla generazione di numeri casuali, è utile analizzare prima il funzionamento di NumPy.
[08:39] In NumPy, si imposta un `seed` (seme) iniziale. Dietro le quinte, NumPy mantiene uno **stato** globale, un numero che tiene traccia del punto corrente nella sequenza del generatore di numeri casuali.
[08:49] Ogni volta che si chiama una funzione per generare un numero casuale (es. `numpy.random.rand`), NumPy aggiorna implicitamente questo stato interno per produrre un nuovo valore.
[08:58] È possibile ispezionare questo stato con la funzione `get\_state()`. Dopo aver impostato il seed, lo stato ha un certo valore. Dopo aver generato un numero casuale, si può osservare che lo stato è cambiato.
### L'Approccio Funzionale Puro di JAX
[09:08] Il problema del modello di NumPy è che le sue funzioni di generazione casuale sono **impure**: dipendono e modificano uno stato globale esterno, proprio come la variabile `g` nell'esempio precedente.
[09:17] JAX, basandosi interamente sul paradigma delle funzioni pure, non può adottare questo tipo di implementazione.
[09:24] Per questo motivo, JAX utilizza una sintassi apparentemente più complessa, in cui la gestione dello stato del generatore di numeri casuali è esplicita e manuale.
[09:28] La `key` (chiave) che viene passata alle funzioni `jax.random` è l'equivalente dello stato del generatore di NumPy.
[09:33] Per mantenere la purezza funzionale, questo stato deve essere gestito manualmente dall'utente.
[09:40] Il processo in JAX è il seguente:
1.  Si parte da un `seed` per ottenere una `key` iniziale, che rappresenta lo stato del generatore.
2.  Si passa questa `key` a una funzione di generazione (es. `jax.random.normal`) per ottenere un numero o un array casuale.
3.  Se si desidera un nuovo numero casuale, è necessario aggiornare esplicitamente la `key`. Se si riutilizza la stessa `key`, si otterrà sempre lo stesso numero casuale.
[09:53] Per ottenere una nuova `key` (ovvero, per aggiornare lo stato), si utilizza la funzione `jax.random.split()`.
[09:57] Questa funzione prende in input una `key` e restituisce due nuove `key`, che possono essere usate per successive generazioni di numeri casuali.
[10:03] In sintesi, JAX richiede di gestire manualmente lo stato del generatore, che in NumPy è invece gestito implicitamente, per garantire che tutte le operazioni rimangano funzionalmente pure.
### Riproducibilità in Ambienti Paralleli
[10:11] Esiste un'altra ragione fondamentale per cui JAX adotta questo approccio: la **riproducibilità in ambienti paralleli**.
[10:16] Con il modello a stato globale di NumPy, se si addestra un modello di machine learning su più GPU, si possono verificare delle `race conditions`.
> **Race Condition**: Una condizione di errore che si verifica quando il risultato di un sistema dipende dalla sequenza o dalla tempistica di eventi incontrollabili, come l'ordine di esecuzione di processi paralleli che accedono a una risorsa condivisa.
[10:21] Le diverse GPU, avendo carichi computazionali differenti, potrebbero tentare di aggiornare lo stato casuale condiviso (che risiede sulla CPU) in momenti diversi e imprevedibili, rendendo il codice non riproducibile.
[10:31] Un esempio può simulare questo scenario. Si definiscono due "worker" (che rappresentano due processi su hardware distinti) che generano numeri casuali. Una pausa (`sleep`) simula carichi di lavoro differenti.
[10:38] Questo scenario è analogo a un aggiornamento dei gradienti in un ciclo di addestramento, dove a ogni iterazione si estrae un nuovo batch di dati in modo casuale.
[10:47] Eseguendo il codice in thread paralleli, si può osservare che, anche reinizializzando il `seed` allo stesso valore ad ogni esecuzione, i risultati ottenuti possono essere diversi.
[10:59] Questo dimostra che l'approccio di NumPy non è affidabile in contesti paralleli a causa delle `race conditions` sullo stato condiviso.
[11:05] Al contrario, l'approccio di JAX, che consiste nel dividere esplicitamente lo stato (`key`) e assegnare a ogni processo (o GPU) la propria `key` da gestire autonomamente, garantisce una perfetta riproducibilità del codice.
## Interazione tra `grad`, `jit` e Strutture di Controllo
### `grad` e le Clausole `if-else`
[11:16] Se si desidera calcolare il gradiente di una funzione (`grad`) che contiene clausole `if-else`, non ci sono problemi. JAX è in grado di gestire correttamente la differenziazione attraverso i rami condizionali.
### Composizione di `jit` e `grad`
[11:21] Tuttavia, se si compongono `jit` e `grad` per ottenere un calcolo del gradiente più veloce, le clausole `if-else` tornano a essere un problema.
[11:27] Come spiegato in precedenza, `jit` non può gestire strutture di controllo che dipendono dai valori degli input, a causa della necessità di un tracciamento statico.
[11:32] Il tentativo di applicare `jit` a una funzione con `if-else` (anche se destinata a `grad`) genererà un errore.
### Soluzione: `jax.numpy.where`
[11:36] La soluzione consiste nell'utilizzare la funzione `jax.numpy.where` al posto delle clausole `if-else`.
[11:40] `where` è una versione vettorizzata del costrutto `if`. La sua sintassi è `where(condizione, x, y)`.
[11:42] La funzione restituisce un array in cui, per ogni elemento, viene scelto il valore da `x` se la `condizione` è vera, altrimenti viene scelto il valore da `y`.
[11:47] Questo approccio è compatibile con `jit` perché la forma (`shape`) dell'array risultante è sempre la stessa, indipendentemente dalla maschera booleana della condizione.
[11:51] Ad esempio, si può definire una funzione che restituisce una parabola per $x \le 3$ e una retta per $x > 3$. Usando `where`, l'output avrà sempre la stessa dimensione dell'input `x`, ma i suoi valori saranno presi da due calcoli diversi a seconda della condizione.
[12:00] Poiché la forma del risultato è statica, è possibile comporre `jit` e `grad` senza problemi.
## Altri Concetti Avanzati di JAX
### `fori\_loop`: Alternativa a Basso Livello per `vmap`
[12:03] La funzione `vmap` ha un equivalente nell'API di basso livello chiamato `fori\_loop`.
[12:07] `fori\_loop` permette di implementare un ciclo `for` in modo vettorizzato, specificando l'indice iniziale, l'indice finale, la funzione da applicare a ogni iterazione e un valore iniziale su cui operare. È un altro strumento per la vettorizzazione dei cicli in JAX.
### Debug di Valori `NaN`
[12:15] Quando si lavora con codice complesso, può capitare che i calcoli producano valori `NaN` (Not a Number), rendendo difficile individuare l'origine del problema.
[12:20] JAX offre un'opzione di configurazione per abilitare il tracciamento dei `NaN`. Impostando `jax.config.update("jax\_debug\_nans", True)`, JAX solleverà un errore non appena un `NaN` viene generato, facilitando il debug.
### Precisione Computazionale: 32-bit vs. 64-bit
[12:28] Per impostazione predefinita, JAX esegue tutti i calcoli in precisione a 32 bit (`float32`). Questa scelta è motivata dal fatto che le GPU sono altamente ottimizzate per operazioni a 32 bit.
[12:34] Al contrario, NumPy e molti contesti matematici utilizzano di default la precisione a 64 bit (`float64` o "double precision").
[12:42] È possibile forzare JAX a utilizzare la precisione a 64 bit, ma è importante essere consapevoli che ciò potrebbe ridurre le prestazioni a livello hardware, poiché tale precisione è generalmente meno ottimizzata sulle GPU.