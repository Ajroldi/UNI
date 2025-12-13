# Capitolo 1: Introduzione e Correzione di Errori Precedenti
## Panoramica della Lezione
[00:00] Il piano per la lezione di oggi prevede due parti principali. Inizieremo dedicando circa dieci minuti all'analisi di un problema riscontrato alla fine del laboratorio precedente. Durante una sessione di codifica dal vivo, si è verificata una difficoltà nel replicare il risultato della soluzione proposta, che ha richiesto di copiare e incollare una porzione di codice. Dopo un'analisi successiva, è stato identificato un errore che potrebbe verificarsi comunemente. Si tratta di un bug difficile da individuare, ed è quindi utile esaminare cosa è successo e perché.
[00:20] Comprendere la causa di questo errore e come prevenirlo è un'ottima opportunità di apprendimento. Successivamente, dedicheremo il resto del tempo all'implementazione da zero di una rete neurale artificiale. Questo processo permetterà di osservare ogni singolo passo, partendo dalle basi fino alla costruzione completa della rete. La prima applicazione sarà l'apprendimento della funzione XOR, un problema semplice ma non banale, poiché la sua struttura non può essere appresa da un modello lineare semplice.
[00:48] Se il tempo a disposizione lo consentirà, verrà mostrato anche come implementare una rete neurale per un problema di classificazione non lineare. In particolare, verranno generati punti appartenenti a due cerchi concentrici, uno più piccolo e uno più grande. L'obiettivo sarà classificare i punti in base alla loro appartenenza a uno dei due cerchi. Questo è un problema i cui risultati possono essere visualizzati in modo molto efficace, pur non essendo banale dal punto di vista della soluzione.
## Analisi dell'Errore nella Funzione di Perdita (Hinge Loss)
[01:13] Iniziamo analizzando l'errore commesso nella lezione precedente. Il notebook relativo non è stato caricato, poiché si tratta di una piccola modifica a quello già esistente. Il problema si è verificato durante l'implementazione della funzione di perdita (*loss function*). Per ottenere il risultato corretto, è stato necessario modificare un termine specifico che era stato implementato in modo errato.
[01:35] L'errore risiedeva nella parte di regolarizzazione della funzione di perdita, in particolare nel termine che calcola la media della funzione *hinge loss*. La formula include un termine di regolarizzazione $ \lambda w^2 $ e la media $ \frac{1}{n} $ della *hinge loss*. Il problema specifico si trovava nel calcolo di questo termine.
[01:51] Analizzando il termine problematico, si osserva che per ogni singolo campione $ i $, si calcola uno scalare $ \zeta_i $. In questa espressione, $ x $ è un vettore che rappresenta un campione, mentre $ w $ e $ b $ (il bias) sono i parametri del modello, concatenati in un unico vettore.
[02:07] L'operazione da eseguire per ogni campione $ i $ consiste nel calcolare il prodotto scalare tra il campione $ x_i $ e la parte $ w $ dei parametri, aggiungere il bias $ b $ (che è uno scalare) e moltiplicare il tutto per l'etichetta $ y_i $ (anch'essa uno scalare). Il risultato è lo scalare $ \zeta_i $.
## Vettorizzazione e il Problema del Broadcasting
[02:24] Come di consueto, queste operazioni vengono eseguite in modalità *batched*, ovvero tramite un'unica espressione che opera su tutti i campioni contemporaneamente. Invece di un singolo vettore $ x_i $, si utilizza una matrice $ X $, dove ogni riga corrisponde a un campione. Allo stesso modo, si usa un vettore $ y $ che contiene tutte le etichette $ y_i $.
[02:41] Per isolare il problema, importiamo NumPy e impostiamo un seme per la riproducibilità. Definiamo un piccolo set di dati con 10 campioni e 2 feature. Vengono generati una matrice $ X $, un vettore di etichette $ y $ e un vettore di parametri `params` in modo casuale, solo per avere dei valori concreti su cui lavorare.
[03:02] La matrice `x` ha 10 righe (i campioni) e 2 colonne (le feature $ x_1 $ e $ x_2 $), come previsto per un problema di classificazione bidimensionale.
[03:17] Il vettore `y` contiene 10 componenti. Anche se in un problema di classificazione reale le etichette potrebbero essere 0 e 1, ciò che conta qui è la sua forma: un vettore monodimensionale con 10 elementi. Infine, `params` è un vettore con 3 componenti: le prime due costituiscono la parte $ W $ e l'ultima è il bias $ b $. L'obiettivo è moltiplicare ogni riga di $ X $ per la parte $ W $ dei parametri e aggiungere il bias.
[03:41] L'implementazione corretta prevede di prendere la matrice $ X $, calcolare il prodotto con i primi due parametri (la parte $ W $) e sommare il bias. Successivamente, si calcola il massimo tra questo risultato e $ 1 - y $, e infine si moltiplica ogni elemento di questo vettore (chiamato `decision`) per il corrispondente elemento di $ y $. Il valore medio corretto di questa operazione è 0.44.
[04:06] Il punto cruciale, dove si è verificato l'errore, riguarda la forma del vettore `decision`. Nell'implementazione corretta, `decision` è un vettore con 10 componenti, risultato di questa parte del calcolo. L'errore commesso nella lezione precedente è stato aggiungere un'operazione di `reshape(-1, 1)`.
[04:21] Con questa modifica, `decision` non è più un vettore monodimensionale, ma diventa un vettore colonna. Dal punto di vista matematico, questa scelta può sembrare più coerente: avendo una matrice $ X $ con molte righe e due colonne, ha senso pensare ai parametri come a un vettore colonna per eseguire un prodotto matrice-vettore.
[04:42] Tuttavia, questa modifica porta a un risultato errato: 0.3. La causa risiede nel comportamento di NumPy. Il vettore `decision` ora ha una forma (*shape*) di (10, 1), rendendolo un oggetto bidimensionale. Il problema sorge quando lo si moltiplica per `y`.
[05:00] Il vettore `y` è monodimensionale. Quando NumPy tenta di moltiplicare un vettore colonna (bidimensionale) per un vettore riga (monodimensionale), attiva un meccanismo chiamato *broadcasting*. Invece di eseguire un prodotto elemento per elemento, il broadcasting espande le dimensioni dei due vettori per renderle compatibili.
[05:16] Il broadcasting, in questo caso, costruisce una matrice in cui ogni elemento $(i, j)$ è il prodotto dell'elemento $i$-esimo del primo vettore e dell'elemento $j$-esimo del secondo. Questo corrisponde a un prodotto esterno (*outer product*).
[05:30] Il problema nasce perché i due oggetti, `decision` e `y`, hanno dimensioni diverse. Se stampiamo le loro forme (`shape`), vediamo che `y.shape` è `(10,)`, indicando un oggetto 1D, mentre la forma dell'altro oggetto è `(10, 1)`, che indica un oggetto 2D.
[05:49] NumPy interpreta questa differenza come una richiesta implicita di broadcasting. Di conseguenza, l'operazione eseguita non è quella desiderata. Poiché alla fine viene chiamata la funzione `mean()`, che calcola la media di tutti gli elementi dell'oggetto risultante (in questo caso, una matrice), si ottiene comunque un singolo valore scalare. La funzione sembra quindi funzionare come una funzione di perdita, ma l'operazione sottostante è completamente diversa.
[06:07] È fondamentale prestare molta attenzione alla distinzione tra oggetti monodimensionali e bidimensionali, poiché il broadcasting implicito può avvenire inaspettatamente. Questo tipo di problema è comune nell'uso quotidiano di librerie come NumPy, e la capacità di individuare rapidamente questi bug è un'abilità importante.
[06:29] Se non ci sono domande, possiamo procedere con il notebook della lezione odierna.
# Capitolo 2: Implementazione di una Rete Neurale da Zero per la Funzione XOR
## Obiettivo: Apprendere la Funzione XOR
[06:40] Il notebook è disponibile su Wibip. Dopo averlo aperto in Google Colab e caricato, l'obiettivo è costruire da zero una rete neurale in grado di apprendere la funzione XOR.
[06:58] Per sfruttare appieno la potenza di JAX, è necessario modificare il runtime per utilizzare la GPU. Iniziamo con la connessione e l'importazione delle librerie necessarie, principalmente NumPy, JAX e Matplotlib per la visualizzazione. L'obiettivo è apprendere la funzione XOR.
[07:18] Il dataset per questo problema è molto piccolo, composto da soli quattro campioni. Per questo motivo, implementeremo un *full gradient descent*, ovvero un algoritmo di discesa del gradiente che utilizza l'intero dataset a ogni passo. L'uso di mini-batch non avrebbe senso con un dataset così ridotto. Gli input hanno due feature (i due bit di input della XOR) e l'output è un singolo valore che rappresenta il risultato della funzione logica.
[07:42] La rete neurale che intendiamo costruire ha la seguente architettura: due neuroni di input, seguiti da due strati nascosti (*hidden layers*), uno con quattro neuroni e l'altro con tre. Infine, lo strato di output ha un singolo neurone. L'output della rete rappresenta una probabilità o una verosimiglianza che il risultato sia vero o falso, quindi il suo valore deve essere compreso tra 0 e 1.
[08:09] Per garantire che l'output sia confinato in questo intervallo, utilizzeremo la funzione sigmoide.
## Definizione degli Iperparametri e Inizializzazione dei Parametri
[08:15] Il primo passo consiste nel definire gli iperparametri.
- **Definizione di Iperparametri**: Sono i parametri che non vengono appresi direttamente durante l'addestramento della rete (come pesi e bias), ma che ne definiscono l'architettura e il processo di training.
In questo caso, definiamo il numero di strati nascosti e la loro dimensione. Le dimensioni dei vari strati (input, hidden, output) sono definite dalla lista `[2, 4, 3, 1]`.
[08:35] Il nostro primo compito è inizializzare lo stato dei parametri della rete.
[08:45] La rete neurale è una funzione $ f(x, \theta) $, dove $ x $ è l'input (un vettore 2D) e $ \theta $ rappresenta i parametri del modello. Questi parametri $ \theta $ sono costituiti da una lista di pesi $ W $ (matrici) e bias $ b $ (vettori) per ogni strato.
[09:06] L'obiettivo è trovare la funzione $ f $ ottimale che, dato un input $ x_i $, predica correttamente l'output $ y_i $. Il nostro dataset è una lista di coppie $ (x_i, y_i) $, con $ i $ che va da 1 a 4.
[09:21] Per raggiungere questo obiettivo, minimizzeremo una funzione di perdita (*loss function*), che misura la distanza tra la predizione della rete $ f(x_i, \theta) $ e il valore reale $ y_i $.
[09:37] La minimizzazione avviene modificando i parametri $ \theta $. Per farlo, utilizzeremo l'algoritmo della discesa del gradiente (*gradient descent*), che richiede il calcolo del gradiente della funzione di perdita rispetto a $ \theta $. La discesa del gradiente è un processo iterativo che, partendo da un punto iniziale $ \theta_0 $, converge verso un punto che minimizza la funzione.
[10:00] Pertanto, il nostro primo compito pratico è definire un punto di partenza $ \theta_0 $ per i parametri. Questo sarà il punto iniziale per l'algoritmo di discesa del gradiente.
[10:13] Iniziamo impostando un seme per la riproducibilità, utilizzando NumPy per semplicità, per poi convertire tutto in formato JAX. Il primo passo è creare la matrice dei pesi $ W_1 $.
[10:29] I pesi verranno inizializzati campionando da una distribuzione normale standard. Esistono tecniche di inizializzazione più avanzate, studiate empiricamente, che possono accelerare la convergenza del training. La scelta del punto iniziale è cruciale per la velocità e l'efficacia dell'addestramento. Per ora, useremo una semplice inizializzazione con numeri casuali da una distribuzione normale.
[10:52] Dobbiamo definire una matrice che trasformi un input 2D in un output 4D, poiché il primo strato nascosto ha 4 neuroni. La matrice $ W_1 $ avrà quindi dimensioni `(n2, n1)`, ovvero (4, 2).
[11:08] L'operazione eseguita da uno strato della rete è una trasformazione affine seguita da una funzione di attivazione non lineare. La formula generale per uno strato è: $ \sigma(W \cdot x + b) $, dove $ \sigma $ è la funzione di attivazione.
[11:24] Per quanto riguarda le dimensioni, se organizziamo i campioni di input $ x $ come colonne di una matrice, la moltiplicazione avviene tra la matrice dei pesi $ W_1 $ e la matrice degli input.
[11:38] L'output di questa operazione avrà un numero di righe pari a quello di $ W_1 $ (4) e un numero di colonne pari al numero di campioni. Questa è la dimensione corretta per l'output del primo strato. Il vettore dei bias $ B_1 $ avrà una dimensione pari alla dimensione di output dello strato, quindi 4.
[12:00] I bias vengono inizializzati a zero. È fondamentale avere sempre chiare le dimensioni delle matrici e dei vettori per garantire che le moltiplicazioni siano coerenti. Sebbene questa non sia l'unica convenzione possibile (ad esempio, si potrebbe usare $ X \cdot W^T $), è quella che adotteremo in questa implementazione.
[12:28] Procediamo definendo i pesi e i bias per gli altri strati in modo analogo.
- $ W_2 $: avrà dimensioni `(n3, n2)`, ovvero (3, 4).
- $ B_2 $: avrà dimensione `(n3,)`, ovvero (3,).
- $ W_3 $: avrà dimensioni `(n4, n3)`, ovvero (1, 3).
- $ B_3 $: avrà dimensione `(n4,)`, ovvero (1,).
[12:47] Tutti i parametri ($ W_1, B_1, W_2, B_2, W_3, B_3 $) vengono raggruppati in una lista e convertiti in formato JAX.
[12:58] Questo approccio dovrebbe essere chiaro. Ora, il prossimo passo è l'implementazione della funzione della rete neurale.
## Implementazione della Funzione della Rete Neurale
[13:04] La funzione da implementare, che rappresenta la rete neurale, accetta come input la matrice dei dati $ x $ e la lista dei parametri `params` (pesi e bias). Deve restituire la predizione della rete.
[13:21] Come funzione di attivazione per gli strati nascosti, si utilizzerà la tangente iperbolica, disponibile in JAX come `jmp.tanh`. Per lo strato di output, si userà la funzione sigmoide.
- **Funzione Sigmoide**: È una funzione che mappa qualsiasi valore reale nell'intervallo (0, 1). Può essere derivata dalla tangente iperbolica (`tanh`) con la formula: $ \text{sigmoid}(x) = \frac{\tanh(x) + 1}{2} $. Poiché `tanh` restituisce valori tra -1 e 1, aggiungendo 1 si ottiene un intervallo tra 0 e 2; dividendo per 2, si normalizza l'output tra 0 e 1.
[13:49] L'obiettivo è implementare una funzione che, dati i pesi, i bias e l'input, calcoli la predizione finale. Questo richiede di far passare l'input attraverso ogni strato della rete, applicando la trasformazione `W * input + b` e la funzione di attivazione `tanh`.
[14:06] L'output di ogni strato diventa l'input per quello successivo. Questa operazione va ripetuta per i tre strati che compongono la nostra rete. Verranno concessi cinque minuti per completare questa implementazione.
[14:28] (Pausa per l'esercizio)
# Capitolo 3: Implementazione del Forward Pass e Gestione delle Dimensioni
## Impostazione della Convenzione per la Moltiplicazione Matriciale
[00:00] La scelta della convenzione per le operazioni matriciali è una questione di preferenza. Si potrebbe mantenere una convenzione coerente, ma ciò richiederebbe di trasporre tutti gli elementi e gestire le matrici in modo diverso. Ad esempio, per calcolare il prodotto, si dovrebbe scrivere `x` per la matrice prodotto per `w`. Questa notazione è leggermente diversa da quella solitamente presentata nei corsi, dove non viene trattata come esempio standard.
[00:07] Per evitare di modificare la convenzione del prodotto matriciale, si considera `w` come la matrice per il prodotto. Tuttavia, questa scelta può portare a delle discontinuità nel codice. Per chiarire il processo, analizzeremo la soluzione passo dopo passo. Invece di definire immediatamente una funzione, costruiremo la logica all'interno di una cella di codice per comprendere meglio ogni passaggio e solo successivamente la trasformeremo in una funzione.
[00:15] L'input del nostro modello è una matrice in cui ogni riga rappresenta un campione diverso. Questa convenzione differisce da quella che useremo internamente per i layer della rete. Pertanto, il primo passo consiste nell'adattare i dati per il primo layer.
## Unpacking dei Parametri e Preparazione dei Dati
[00:22] Iniziamo con l'estrarre i parametri, che sono stati raggruppati in un unico oggetto `params`. Per maggiore chiarezza, è preferibile averli separati. Li "spacchettiamo" assegnandoli a variabili singole.
[00:28] Nel primo layer, dobbiamo eseguire la moltiplicazione matriciale tra `W1` e la matrice di input. Per seguire la convenzione desiderata, chiamiamo l'input `x` e procediamo con la sua trasposizione. Questo approccio semplificherà la successiva trasformazione del codice in una funzione.
[00:36] La nostra matrice `x` trasposta (`x\_transpose`) avrà quattro righe, una per ciascuna feature, e quattro colonne, una per ciascun campione. Successivamente, aggiungiamo il vettore dei bias `b1`. Questa operazione viene eseguita colonna per colonna.
## Gestione del Broadcasting per i Vettori di Bias
[00:43] È fondamentale verificare che l'operazione di broadcasting, ovvero l'estensione automatica delle dimensioni di un array per renderlo compatibile con un altro, avvenga correttamente. Dobbiamo assicurarci che il vettore `b1` venga sommato a ciascuna colonna della matrice. Per verificarlo, inizializziamo temporaneamente `b1` con una sequenza di numeri (es. 1, 2, 3, 4) invece che con zeri.
[00:52] Osservando il risultato, notiamo che la somma viene eseguita riga per riga, il che è errato per il nostro scopo. Per correggere questo comportamento, dobbiamo sommare colonna per colonna. Questo si ottiene modificando la forma del vettore di bias tramite un'operazione di `reshape`.
[00:57] Se trasformiamo `b1` in un vettore colonna, il broadcasting funzionerà come desiderato. Utilizzando `reshape(-1, 1)`, forziamo l'array a diventare una matrice con una sola colonna. Ora la somma avviene correttamente colonna per colonna, come si può vedere dal fatto che la prima colonna della matrice risultante è [0, 1, 2, 3].
[01:05] Per evitare di dover eseguire il `reshape` all'interno del calcolo del layer, modifichiamo direttamente l'inizializzazione dei parametri. Torniamo alla cella di definizione dei parametri, ripristiniamo i valori di `b1` a zero e definiamo tutti i vettori di bias come vettori colonna.
[01:10] Secondo la nostra convenzione, i bias non sono più vettori monodimensionali (1D), ma matrici bidimensionali (2D) con una sola colonna. Questa struttura garantisce che il broadcasting avvenga correttamente durante la somma.
## Calcolo dell'Output dei Layer Intermedi
[01:16] L'output del primo calcolo lineare, che chiameremo `layer2` (poiché il primo è il layer di input), deve essere processato da una funzione di attivazione. Senza di essa, una composizione di strati lineari si ridurrebbe a un'unica trasformazione lineare, limitando la capacità espressiva della rete. Applichiamo quindi la funzione di attivazione tangente iperbolica (`tanh`).
[01:23] Il secondo layer segue la stessa logica del primo. L'operazione è `w2 * layer2 + b2`.
[01:27] Successivamente, calcoliamo l'output del terzo layer, che chiamiamo `layer4` (seguendo una numerazione progressiva). Questo è dato dall'applicazione della tangente iperbolica al risultato di `w3 * layer3 + b3`.
## Normalizzazione dell'Output Finale
[01:30] Per l'ultimo layer, desideriamo un output che rappresenti una probabilità, quindi i suoi valori devono essere compresi nell'intervallo `[0, 1]`. La funzione di attivazione `tanh` produce valori nell'intervallo `[-1, 1]`.
[01:36] Per mappare questo intervallo a `[0, 1]`, eseguiamo una semplice trasformazione. Prima sommiamo 1 all'output del layer, `layer4 + 1`, ottenendo valori nell'intervallo `[0, 2]`. Successivamente, dividiamo il risultato per 2. In questo modo, l'output finale sarà correttamente normalizzato tra 0 e 1.
[01:43] La trasformazione `(tanh(z) + 1) / 2` è equivalente alla funzione sigmoide. Per il layer finale, non usiamo la `tanh` direttamente, ma questa sua variante per ottenere l'output desiderato. L'ultima riga di codice nella cella serve semplicemente a visualizzare il valore della variabile finale.
[01:51] Se si verificano errori relativi alle dimensioni (`shape`), è probabile che la cella di definizione dei parametri non sia stata eseguita nuovamente dopo le modifiche. Definendo i bias come vettori colonna (con `shape` `(n, 1)`) e rieseguendo il codice, l'output del `layer4` sarà un vettore riga bidimensionale, come indicato dalle doppie parentesi quadre.
## Coerenza della Convenzione di Input e Output
[02:00] Questa convenzione, con i campioni disposti sulle colonne, differisce da quella che abbiamo stabilito per l'input, dove ogni riga rappresenta un campione. Per rendere l'output coerente con l'input, applichiamo una trasposizione finale. In questo modo, sia la matrice di input che quella di output seguono la stessa convenzione.
[02:07] La convenzione interna ai layer della rete neurale prevede che ogni campione sia rappresentato da una colonna. Per questo motivo, abbiamo trasposto l'input all'inizio del processo.
[02:11] Per scrivere ogni layer come una moltiplicazione matrice-matrice del tipo `W * X`, esistono due approcci:
1.  Utilizzare la convenzione in cui i campioni sono disposti sulle colonne di `X`.
2.  Invertire l'ordine della moltiplicazione e trasporre le matrici. Invece di `W * X`, si calcolerebbe `X * W\_transpose`.
[02:19] Non è possibile mantenere la convenzione con i campioni sulle righe di `X` e contemporaneamente usare la formula `W * X` senza trasporre `X`. La notazione `X * W\_transpose` è matematicamente valida ma meno comune nei contesti didattici. Per questo motivo, si è preferito utilizzare la convenzione `W * X` con la trasposizione dell'input. Matematicamente, `(A * B)^T = B^T * A^T`.
[02:26] L'aspetto cruciale nell'implementazione di reti neurali è mantenere la coerenza con la convenzione scelta e assicurarsi di seguirla in ogni passaggio. L'ambiente dei notebook è particolarmente utile in questo, poiché permette di ispezionare l'output e la forma (`shape`) di ogni oggetto ad ogni passo.
## Trasformazione del Codice in Funzione
[02:33] Il codice sviluppato nella cella può essere facilmente trasformato in una funzione. Definiamo una funzione `artificial\_neural\_network` che accetta come argomenti la matrice di input `X` e i parametri `params`. La funzione restituirà l'output finale, `layer4`.
[02:39] Per verificare la correttezza della funzione, la invochiamo con gli input e i parametri definiti in precedenza e stampiamo la predizione. Questo processo mostra come, tenendo traccia della forma di ogni oggetto, sia semplice incapsulare una logica complessa all'interno di una funzione riutilizzabile.
## Chiarimenti sulla Gestione degli Input
[02:47] La trasformazione dell'input all'interno del ciclo di visualizzazione è stata fatta solo a scopo dimostrativo. La convenzione standard prevede che alla rete venga sempre passata una matrice, dove ogni riga corrisponde a un campione.
[02:52] L'input `[0, 0]` è un vettore 1D, ma poiché la nostra convenzione richiede matrici, viene eseguito un `reshape` per trasformarlo in una matrice `(1, 2)`. Questa operazione è necessaria per la visualizzazione che mostra l'operazione XOR tra i due valori.
[02:59] Un modo alternativo per ottenere lo stesso risultato di visualizzazione consiste nell'iterare sui quattro campioni. Per ogni indice `i` da 0 a 3, si può estrarre la riga `i`-esima dalla matrice di input e associarla alla predizione `i`-esima.
[03:08] Le due sintassi producono lo stesso risultato. Nel primo caso, si itera su ogni riga della matrice di input, la si estrae, la si rimodella come matrice e la si passa alla funzione. Nel secondo caso, si itera sugli indici dei campioni, si estraggono i valori dalla matrice di input e li si associa alla predizione corrispondente.
[03:15] L'aspetto fondamentale è che l'input della funzione `ANN` deve sempre rispettare la convenzione stabilita, ovvero una matrice in cui ogni riga rappresenta un campione.
# Capitolo 4: Addestramento della Rete Neurale
## Valutazione delle Prestazioni Iniziali
[03:20] Con i parametri iniziali scelti casualmente, le prestazioni della rete neurale sono molto scarse. Analizzando le predizioni, si nota che sono quasi sempre errate.
-   Input `[0, 0]`: XOR dovrebbe essere `0`, la predizione è `0` (corretto).
-   Input `[0, 1]`: XOR dovrebbe essere `1`, la predizione è `0` (errato).
-   Input `[1, 0]`: XOR dovrebbe essere `1`, la predizione è `1` (corretto).
-   Input `[1, 1]`: XOR dovrebbe essere `0`, la predizione è `1` (errato).
[03:28] Inizialmente, la rete si comporta in modo casuale. Per migliorare le sue prestazioni, è necessario modificare i parametri in modo che le predizioni diventino corrette.
## Introduzione al Gradient Descent e alle Funzioni di Costo
[03:36] Il metodo utilizzato per ottimizzare i parametri è il **Gradient Descent** (discesa del gradiente). L'obiettivo è modificare i parametri per migliorare le predizioni della rete.
[03:40] Per fare ciò, dobbiamo definire una **funzione di costo** (o *loss function*), ovvero una funzione che misura l'errore della rete e che vogliamo minimizzare. Implementeremo il Gradient Descent su questa funzione. Vengono proposte due diverse funzioni di costo.
## Funzione di Costo Quadratica (Mean Squared Error)
[03:48] La prima funzione di costo è la `loss\_quadratic`. Dati l'input `x`, il target `y` (il valore corretto) e i parametri della rete, questa funzione calcola l'**errore quadratico medio** (*Mean Squared Error*, MSE) tra la predizione e il valore reale.
```math
L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{true, i})^2
```
Questa è la funzione di costo standard, già utilizzata in altre occasioni. Per implementarla, si utilizza la funzione `artificial\_neural\_network` per ottenere la predizione e si calcola l'MSE rispetto al target `y`.
## Funzione di Costo Cross-Entropy
[03:58] La seconda funzione di costo è la **Cross-Entropy** (entropia incrociata). Poiché il nostro è un compito di classificazione e gli output sono interpretati come probabilità, l'uso della cross-entropy è più appropriato.
[04:04] La ragione matematica è che la cross-entropy è una misura della distanza tra distribuzioni di probabilità. Dal punto di vista pratico, questa funzione di costo è efficace perché penalizza pesantemente le predizioni errate fatte con alta confidenza.
[04:10] Consideriamo un esempio: supponiamo che il valore corretto `yi` sia `1`. Se la rete predice con alta confidenza un valore errato, ad esempio `0.01`, la funzione di costo reagisce in modo significativo. La formula della cross-entropy per la classificazione binaria è:
```math
L_{CE} = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
```
-   Se `yi = 1`, il secondo termine `(1 - yi)` si annulla. Rimane `yi * log(y\_hat\_i)`, che diventa `log(0.01)`.
-   Il logaritmo di un numero molto vicino a zero (come `0.01`) è un numero negativo molto grande.
[04:20] Il logaritmo di `0.01` è un valore molto negativo. A causa del segno meno davanti alla sommatoria nella formula della cross-entropy, il contributo alla loss diventa un numero positivo molto grande.
[04:28] Questo valore è molto più grande di quello che si otterrebbe con l'errore quadratico medio, penalizzando così in modo severo le predizioni sbagliate e sicure. Questa è la ragione pratica per cui la cross-entropy funziona bene in compiti di classificazione.
[04:34] L'implementazione di queste funzioni è relativamente semplice. Si calcola la predizione `y\_hat` usando la funzione `ANN` e poi si applica la formula corrispondente. Verranno concessi cinque minuti per implementare sia la funzione di costo basata sull'errore quadratico medio sia quella basata sulla cross-entropy.
# Capitolo 5: Implementazione delle Funzioni di Perdita e Calcolo dei Gradienti
## Errore Quadratico Medio (MSE)
[00:00] Iniziamo con l'implementazione dell'errore quadratico medio (Mean Squared Error, MSE). Il processo verrà sviluppato passo dopo passo all'interno di una cella di codice per verificarne il funzionamento, per poi essere incapsulato in una funzione. Il primo passo consiste nel calcolare la predizione della rete neurale. La predizione si ottiene applicando la funzione della rete, `ANN`, ai dati di input `x` e ai parametri correnti del modello.
[00:11] La variabile `x` corrisponde agli `inputs`. Una volta calcolata la predizione, è possibile visualizzarla. Successivamente, si calcola l'errore quadratico medio tra le predizioni e i valori target. I valori target, denominati `outputs`, vengono assegnati a una variabile `y` per chiarezza.
[00:24] Confrontando le dimensioni (`shape`) della predizione e di `y`, si può notare che sono identiche. Questa corrispondenza dimensionale è fondamentale per poter eseguire operazioni aritmetiche tra di esse, come la sottrazione `predizione - y`, che rappresenta l'errore. L'errore viene quindi elevato al quadrato e, infine, se ne calcola la media utilizzando la funzione `jmp.mean`. Il risultato è un singolo valore scalare che rappresenta l'errore quadratico medio.
[00:41] L'errore quadratico medio è definito come la media delle differenze al quadrato tra il valore target `y` e la predizione. La funzione che implementa questo calcolo accetta come argomenti `x`, `y` e i parametri del modello (`params`).
[00:52] Per riassumere, la funzione MSE prima calcola la predizione utilizzando la rete neurale, poi calcola l'errore come differenza tra la predizione e il target `y`, eleva al quadrato questo errore e infine ne calcola la media. Il risultato è un valore scalare, la funzione di perdita (loss function), che può essere minimizzato durante l'addestramento.
## Entropia Incrociata (Cross-Entropy)
[01:03] L'implementazione della funzione di perdita basata sull'entropia incrociata (cross-entropy) segue un approccio simile. Anche in questo caso, il punto di partenza è il calcolo della predizione, che si ottiene applicando la rete neurale `ann` agli input `x` e ai parametri `params`.
[01:10] La formula dell'entropia incrociata per la classificazione binaria è:
```math
L(\theta) = - \sum_i [y_i \log(p(x_i; \theta)) + (1 - y_i) \log(1 - p(x_i; \theta))]
```
dove $p(x_i; \theta)$ è la predizione della rete. L'implementazione in codice traduce questa formula come segue: `y * jmp.log(predizione) + (1 - y) * jmp.log(1 - predizione)`.
[01:20] È importante osservare che, per ogni campione, solo uno dei due termini della somma è attivo. Poiché `y` può assumere solo i valori 0 o 1, se `y` è 1, il secondo termine si annulla e rimane solo il primo; se `y` è 0, si annulla il primo termine e rimane solo il secondo.
[01:36] Successivamente, si applica la somma (`jmp.sum`) su tutti i campioni per calcolare la similarità tra le distribuzioni di probabilità. Poiché la funzione di perdita deve misurare una discrepanza (un errore), si antepone un segno negativo al risultato della somma, ottenendo così l'implementazione completa della formula dell'entropia incrociata.
## Definizione delle Funzioni Gradiente
[01:48] In questo contesto, l'uso della media (`mean`) al posto della somma (`sum`) nella funzione di perdita è preferibile. Sebbene in questo specifico caso non cambi molto, poiché la dimensione del campione è costante, l'uso della media rende la funzione di perdita robusta a eventuali variazioni della dimensione del batch (batch size).
[01:57] Il passo successivo consiste nel calcolare i gradienti delle funzioni di perdita. Per ottimizzare le prestazioni, si utilizza la compilazione just-in-time (JIT) di JAX. Si definiscono quindi le versioni JIT sia della funzione di perdita quadratica (`loss\_quadratic`) sia di quella basata sull'entropia incrociata.
[02:10] Il calcolo del gradiente richiede particolare attenzione. Si definisce una funzione `grad\_msc\_jit` utilizzando `jacks.jit(jacks.grad(loss\_quadratic))`. Tuttavia, questa implementazione è errata.
[02:21] L'errore risiede nel fatto che la funzione `jacks.grad` calcola di default il gradiente rispetto al primo argomento della funzione. La nostra funzione di perdita, `loss\_quadratic(x, y, params)`, ha tre argomenti, e noi siamo interessati al gradiente rispetto al terzo, ovvero `params`. Per specificare ciò, si utilizza l'argomento `argnums` nella chiamata a `jacks.grad`.
[02:34] L'argomento `argnums` può essere un intero o una sequenza di interi. In questo caso, si imposta `argnums=2` per indicare che il gradiente deve essere calcolato rispetto al terzo argomento (indicizzato con 2).
[02:45] La stessa procedura viene applicata per calcolare il gradiente della funzione di perdita basata sull'entropia incrociata.
## Addestramento della Rete Neurale con Discesa del Gradiente
[02:54] A questo punto, si dispone di tutti gli elementi necessari per addestrare la rete neurale. Il caso in esame è semplificato: il set di addestramento (training set) e il set di test (test set) coincidono, e non si utilizzano mini-batch, ma si esegue l'addestramento su batch completi (full batch).
[03:07] L'addestramento avviene implementando l'algoritmo della discesa del gradiente (gradient descent), utilizzando una delle due funzioni di perdita definite (MSE o entropia incrociata).
[03:14] Lo pseudo-codice per il ciclo di addestramento è il seguente: si itera per un numero totale di epoche. In ogni epoca, essendo un approccio full-batch, si calcola il gradiente sull'intero dataset.
[03:23] Successivamente, per ogni componente dei parametri del modello (matrici di pesi e vettori di bias), si aggiorna il suo valore muovendosi nella direzione opposta a quella del gradiente, con un passo di aggiornamento (step size) pari al tasso di apprendimento (learning rate).
[03:34] Alla fine di ogni epoca, si salva il valore corrente della funzione di perdita per monitorarne l'andamento. Infine, si visualizza (plot) l'evoluzione della perdita nel tempo. Il gradiente viene calcolato per ogni parametro, ovvero per ogni matrice di pesi `W` e ogni bias `B`.
[03:49] L'aggiornamento dei parametri avviene quindi sottraendo il gradiente moltiplicato per il learning rate. Dopo l'aggiornamento, si calcola e si salva il valore corrente di entrambe le funzioni di perdita (MSE e entropia incrociata) per monitorare il processo. Infine, si verifica che la perdita sia effettivamente diminuita. Se l'implementazione è corretta, la rete dovrebbe imparare a classificare correttamente tutti gli esempi, dato che il dataset è molto semplice.
# Capitolo 6: Implementazione del Ciclo di Addestramento e Analisi dei Risultati
## Debug e Sviluppo del Codice
[04:26] Per implementare la soluzione, si parte dallo pseudo-codice e si procede passo dopo passo. Inizialmente, il ciclo di addestramento viene eseguito per una sola epoca (`for epoch in range(1)`) per finalità di debug, assicurandosi che ogni parte del codice funzioni correttamente prima di aumentare il numero di iterazioni.
[04:43] Si sceglie di utilizzare l'errore quadratico medio come funzione per il calcolo del gradiente, assegnando `GradMC\_JIT` a una variabile `gradient\_function`. Il gradiente viene quindi calcolato chiamando questa funzione con gli stessi argomenti della funzione di perdita: `inputs`, `outputs` (corrispondenti a `x` e `y`) e i `params`.
[05:00] Per ispezionare il risultato, si stampa il gradiente calcolato (`grad`). L'output è un oggetto (una lista di array) che ha la stessa struttura e le stesse dimensioni dei parametri del modello. I parametri sono una lista contenente matrici di pesi e vettori di bias.
[05:11] Analogamente, `grad` è una lista di array che contiene i gradienti della funzione di perdita calcolati rispetto a ciascun parametro corrispondente. Esiste una corrispondenza uno-a-uno tra ogni parametro e il suo gradiente.
[05:26] L'aggiornamento dei parametri può essere eseguito esplicitamente con un ciclo. Si itera sulla lista dei parametri (`for i in range(len(params))`). Ogni elemento `params[i]` è un oggetto JAX (matrice o vettore).
[05:35] L'aggiornamento avviene sottraendo al parametro corrente il gradiente corrispondente `grads[i]` moltiplicato per il `learning\_rate`. Il learning rate viene definito in precedenza, ad esempio con un valore di 0.1.
[05:54] Dopo l'aggiornamento dei parametri, si calcola il valore di entrambe le funzioni di perdita (MSE e entropia incrociata) e si memorizza il loro andamento in due liste, `history\_mnc` e `history\_cross\_entropy`.
[06:05] È importante notare che, sebbene si possano monitorare più metriche di perdita, l'aggiornamento dei parametri può essere basato solo su un gradiente alla volta, poiché i gradienti calcolati da funzioni di perdita diverse punteranno in direzioni differenti.
[06:18] Eseguendo il codice, si può procedere a visualizzare l'andamento delle perdite nel tempo utilizzando `plt.plot`.
## Analisi dei Risultati dell'Addestramento
[06:27] Aumentando il numero di epoche a 2000 (un iperparametro del modello), si può osservare l'evoluzione dell'addestramento. Il monitoraggio di più metriche, anche se non usate per l'aggiornamento, è utile. Ad esempio, l'entropia incrociata funge sempre da metrica valida: un suo valore elevato può indicare un problema, anche se l'ottimizzazione è basata su un'altra funzione di perdita.
[06:42] Inoltre, confrontare diverse funzioni di perdita permette di valutare l'impatto di eventuali modifiche. In scenari reali, si potrebbe modificare una funzione di perdita (ad esempio, regolandone alcuni parametri interni) e monitorare l'effetto su metriche di benchmark standard, come l'entropia incrociata, per verificare se la modifica porta a un miglioramento generale.
[07:01] Nel grafico risultante, la curva blu rappresenta l'errore quadratico medio (MSE), mentre quella arancione rappresenta l'entropia incrociata. Entrambe le perdite scendono sotto il valore di 0.1.
[07:12] Per analizzare meglio l'andamento, è utile visualizzare il grafico della perdita con una scala logaritmica sull'asse y (`plt.yscale('log')`).
[07:21] Con la scala logaritmica, si osserva che l'errore quadratico medio scende al di sotto di 0.001, indicando un buon andamento dell'addestramento. Anche l'entropia incrociata scende sotto 0.1, confermando che la rete neurale sta apprendendo correttamente.
[07:36] Il test finale consiste nel verificare le predizioni della rete sulle coppie input-output. Dato l'andamento della perdita, è molto probabile che i risultati siano corretti.
[07:44] Infatti, per input che dovrebbero dare output 0, la rete predice una probabilità molto bassa (es. 0.01), mentre per input che dovrebbero dare 1, la probabilità predetta è molto alta (es. 0.98).
## Confronto tra Funzioni di Perdita e Valutazione del Modello
[08:00] Si procede ora a modificare l'esperimento utilizzando l'entropia incrociata come funzione di perdita per il calcolo del gradiente, al posto dell'errore quadratico medio (MSE).
[08:08] Eseguendo nuovamente l'addestramento, si osserva un risultato interessante: sebbene si stia ottimizzando l'entropia incrociata, il valore finale dell'MSE è molto più basso rispetto a prima.
[08:16] L'MSE scende al di sotto di $10^{-5}$. Confrontando con l'addestramento precedente, dove dopo lo stesso numero di epoche l'MSE era appena sotto $10^{-3}$, ora si raggiunge un valore ben al di sotto di $10^{-6}$.
[08:26] Questo dimostra che, per questo problema di classificazione, l'entropia incrociata è più efficace dell'MSE nel ridurre l'MSE stesso. Questo è uno dei motivi pratici per cui l'entropia incrociata è la scelta standard per i problemi di classificazione.
[08:37] La ragione di questa superiorità risiede nel fatto che l'entropia incrociata penalizza molto severamente le predizioni errate fatte con alta confidenza. Da un punto di vista matematico, è strettamente legata al concetto di "vicinanza" tra distribuzioni di probabilità.
[08:47] Per verificare ulteriormente questo comportamento, si riesegue l'intero processo dall'inizio, reinizializzando i parametri.
[08:55] Anche partendo da un valore di perdita iniziale molto più alto, l'addestramento con l'entropia incrociata converge a un valore di MSE estremamente basso, confermando la sua efficacia.
## Calcolo dell'Accuratezza e della Matrice di Confusione
[09:10] Il passo finale è calcolare l'accuratezza (accuracy) del modello. Per fare ciò, le probabilità continue prodotte dalla rete devono essere convertite in predizioni binarie (0 o 1).
[09:17] Si calcolano le predizioni finali applicando una soglia di 0.5: se la probabilità predetta è maggiore di 0.5, la classe predetta è 1 (True), altrimenti è 0 (False).
[09:29] Confrontando le predizioni binarie (`pred`) con i valori target reali (`y`), si osserva che sono sempre uguali. L'accuratezza, calcolata come il numero di predizioni corrette diviso per il numero totale di campioni, risulta quindi del 100%.
[09:44] Nei problemi di classificazione, un altro strumento di valutazione molto utile è la matrice di confusione. Questa matrice mostra il numero di veri positivi, veri negativi, falsi positivi e falsi negativi.
[09:56] Per calcolarla, si può utilizzare la funzione `confusion\_matrix` dalla libreria Scikit-learn, evitando di scrivere codice verboso.
[10:04] Alla funzione si passano i valori reali (`y`) e le predizioni binarie.
[10:10] La matrice di confusione risultante mostra valori non nulli solo sulla diagonale principale e zeri sulla anti-diagonale. Questo indica che tutte le predizioni sono corrette e il modello ha una performance perfetta su questo dataset.
# Capitolo 7: Generalizzazione a un Problema di Classificazione Non Lineare
## Introduzione a un Nuovo Dataset
[10:24] Il passo successivo consiste nel rendere il problema più complesso e realistico. Sebbene non sia ancora uno scenario di utilizzo del tutto realistico, si aggiungono gradualmente elementi di complessità.
[10:34] Si utilizza un dataset generato da Scikit-learn noto come "make\_circles". Questo dataset è composto da due cerchi concentrici: uno più piccolo al centro e uno più grande che lo circonda.
[10:44] Si tratta di un problema di classificazione binaria in cui l'obiettivo è distinguere i punti appartenenti al cerchio interno da quelli appartenenti all'anello esterno. L'input per la rete neurale sarà costituito dalle coordinate (x, y) di ogni punto, mentre l'output sarà 0 o 1 a seconda dell'anello di appartenenza.
[10:57] Il codice per generare i dati è già fornito. Vengono generate le coordinate X e Y, assicurandosi che abbiano le dimensioni corrette.
[11:04] Successivamente, si utilizza la funzione `train\_test\_split` per suddividere il dataset in un 80% per l'addestramento e un 20% per il test. Questa suddivisione rende lo scenario più realistico. I dati vengono poi visualizzati: un grafico a dispersione (`scatter plot`) mostra i punti, colorati in base alla loro classe (Y). I punti di test sono contrassegnati con una 'X', mentre quelli di addestramento con un cerchio.
[11:20] Le variabili fondamentali sono quattro matrici: `X\_train`, `Y\_train`, `X\_test`, `Y\_test`. Ogni riga rappresenta un campione; le matrici X hanno due colonne (le coordinate), mentre le matrici Y hanno una colonna (la classe).
## Struttura del Nuovo Compito di Programmazione
[11:27] Il compito consiste nell'implementare nuovamente tutti gli ingredienti necessari, ma in modo più generale e scalabile. Invece di definire manualmente un numero fisso di strati, come fatto in precedenza, si vuole creare una struttura più flessibile.
[11:38] L'obiettivo è evitare di scrivere codice ripetitivo per ogni strato (layer1, layer2, ...), utilizzando invece un ciclo `for`. Per fare ciò, si definiscono delle funzioni di supporto.
[11:47] La prima funzione, `initLayerParameters`, riceve una chiave casuale di JAX, la dimensione di input e la dimensione di output di un singolo strato, e restituisce una coppia di pesi e bias per quello strato.
[11:56] La seconda funzione, `initializeMLPParameters`, riceve una chiave e una lista contenente le dimensioni di ogni strato della rete. Questa funzione genera automaticamente la lista completa dei parametri (pesi e bias) per l'intera rete neurale multistrato (MLP).
[12:07] L'idea è che la lista `layer\_sizes` possa avere qualsiasi lunghezza e contenere qualsiasi dimensione, e la funzione creerà dinamicamente i parametri corrispondenti. L'architettura della rete rimane simile a quella precedente, con funzioni di attivazione a tangente iperbolica (`tanh`) negli strati intermedi e una sigmoide nell'ultimo strato.
[12:20] La funzione di predizione, ora chiamata `forward`, riceve i parametri e l'input `x` e restituisce la predizione. La differenza principale rispetto a prima è che, invece di scrivere ogni strato manualmente, si utilizzerà un ciclo `for` per propagare l'output di uno strato come input del successivo.
[12:30] Si utilizzerà nuovamente la funzione di perdita basata sull'entropia incrociata, che può essere copiata dall'implementazione precedente. Infine, si implementerà la discesa del gradiente con mini-batch, anziché full-batch.
[12:38] Il dataset è più grande (300 punti), quindi si definisce una funzione `update` che aggiorna i parametri utilizzando solo un mini-batch di dati (`x` e `y`) estratto dal dataset completo. L'estrazione del mini-batch avverrà in un secondo momento; per ora, l'obiettivo è creare una funzione di aggiornamento che incapsuli questa logica.
[12:56] Sebbene sembri un compito esteso, è molto simile a quanto già fatto, ma con un approccio più generalizzato. L'obiettivo è prendere il codice precedente e renderlo più flessibile e scalabile.
# Capitolo 8: Implementazione di una Rete Neurale Generalizzata con JAX
## Introduzione alla Soluzione Guidata
[00:01] In questa sezione, verrà illustrata la soluzione passo dopo passo, con l'obiettivo di procedere lentamente e mostrare alcuni accorgimenti per rendere il codice più generale. L'approccio seguito è pensato per essere didattico e dettagliato.
## Inizializzazione dei Parametri della Rete
[00:08] Si inizia definendo la funzione per inizializzare i parametri di un singolo strato (`layer`). La funzione, denominata `init\_layer\_params`, accetta tre argomenti: una chiave (`key`) di JAX per la generazione di numeri casuali, la dimensione di input (`in\_dimension`) e la dimensione di output (`out\_dimension`).
[00:16] All'interno della funzione, il primo passo è inizializzare la matrice dei pesi `w`. Per questa operazione si utilizza la funzione `jax.random.normal`, che genera numeri casuali da una distribuzione normale. Per il momento, una distribuzione normale o uniforme è considerata adeguata.
[00:23] La forma (`shape`) della matrice dei pesi viene definita seguendo una convenzione specifica. In precedenza, si era discussa una convenzione in cui i campioni (`samples`) erano disposti sulle colonne.
[00:32] Ora, si adotta una convenzione alternativa in cui i campioni sono disposti sulle righe. Essere flessibili rispetto a queste convenzioni è una competenza importante nella programmazione, poiché non si può sapere a priori quale approccio si incontrerà in un codice esistente.
[00:44] Adottando la convenzione con i campioni per riga, è necessario scambiare le dimensioni della matrice dei pesi `w`. La forma sarà quindi `(in\_dimension, out\_dimension)`.
[00:52] Questa scelta è motivata dalla regola della moltiplicazione tra matrici: il numero di colonne della matrice di input `x` deve essere uguale al numero di righe della matrice dei pesi `w`.
[00:57] Il vettore dei bias `b` viene inizializzato a zero utilizzando la funzione `jnp.zeros`. La sua dimensione deve corrispondere alla dimensione di output (`out\_dimension`).
[01:02] La forma del vettore dei bias sarà `(out\_dimension,)`. Non è `(out\_dimension, 1)` perché, con la nuova convenzione, il *broadcasting* (l'adattamento automatico delle dimensioni durante le operazioni) non avviene più sulle colonne, ma sulle righe.
[01:08] Infine, la funzione restituisce i pesi `w` e i bias `b`. La corretta gestione delle dimensioni sarà cruciale nella funzione di `forward propagation`.
[01:14] Successivamente, si definisce la funzione `init\_mlp\_params` per inizializzare i parametri di un intero Multi-Layer Perceptron (MLP). Questa funzione accetta una chiave (`key`) di JAX e una lista `layer\_sizes` che specifica il numero di neuroni per ogni strato, inclusi quello di input e di output.
[01:20] Per garantire che ogni strato abbia pesi e bias inizializzati in modo indipendente, è necessario generare una chiave casuale diversa per ciascuno di essi. Questo si ottiene con `jax.random.split(key, num\_keys)`.
[01:27] Il numero di chiavi necessarie (`num\_keys`) è pari al numero di strati della rete, che corrisponde a `len(layer\_sizes) - 1`.
[01:33] Ad esempio, se `layer\_sizes` è una lista di 4 elementi (es. `[input, hidden1, hidden2, output]`), la rete avrà 3 strati di connessioni (input -> hidden1, hidden1 -> hidden2, hidden2 -> output), e quindi richiederà 3 coppie di pesi e bias.
[01:43] Il motivo di `len(layer\_sizes) - 1` diventa evidente nel ciclo `for` che segue.
[01:47] Si inizializza una lista vuota `params` che conterrà i parametri di tutti gli strati.
[01:50] Si itera su un range che va da 0 a `len(layer\_sizes) - 2`. All'interno del ciclo, per ogni strato `i`, si chiama la funzione `init\_layer\_params` passandole la chiave specifica per quello strato (`keys[i]`), la dimensione di input `layer\_sizes[i]` e la dimensione di output `layer\_sizes[i+1]`.
[02:00] In questo modo, si scorrono tutti gli strati tranne l'ultimo. La dimensione di input di uno strato corrisponde alla dimensione di output dello strato precedente.
[02:06] Ad esempio, con `layer\_sizes = [2, 4, 1]`, il primo strato avrà dimensione di input 2 e output 4, mentre il secondo strato avrà input 4 e output 1.
[02:11] Viene mostrato un esempio pratico. Si definisce una struttura di rete con `layer\_sizes = [2, 4, 1]`, che corrisponde a uno strato di input con 2 neuroni, uno strato nascosto con 4 neuroni e uno strato di output con 1 neurone.
[02:20] La chiave per la generazione casuale deve essere creata tramite il generatore di chiavi di JAX, non può essere un semplice numero.
[02:25] Esaminando i parametri generati (`params`), si osserva la struttura: una prima matrice di pesi con forma `(2, 4)`, un vettore di bias di dimensione 4, una seconda matrice di pesi con forma `(4, 1)` e un bias finale di dimensione 1.
[02:38] Questa struttura è coerente con le dimensioni definite e conferma il corretto funzionamento della funzione di inizializzazione.
## Implementazione della Propagazione in Avanti (Forward Propagation)
[02:48] Viene definita la funzione di attivazione sigmoide. Si può implementare in due modi equivalenti: utilizzando la relazione con la tangente iperbolica `jnp.tanh(x) / 2` o la sua definizione standard `1 / (1 + jnp.exp(-x))`.
[02:57] Entrambe le implementazioni producono lo stesso risultato.
[02:59] La funzione `forward` è il cuore della propagazione in avanti, dove è richiesta particolare attenzione alle dimensioni delle matrici.
[03:04] L'obiettivo è calcolare l'output della rete a partire da un input `x\_train` e dai parametri `params`.
[03:08] Per un singolo strato, l'operazione consiste in una moltiplicazione matrice-matrice tra l'input e i pesi, seguita dalla somma del bias.
[03:12] I parametri di uno strato sono memorizzati in una tupla. Ad esempio, `params[0]` contiene la tupla `(pesi, bias)` del primo strato. I pesi sono il primo elemento (`params[0][0]`) e i bias il secondo (`params[0][1]`).
[03:17] Il prodotto matrice-matrice si esegue con `x\_train @ params[0][0]`.
[03:22] Successivamente, si somma il vettore dei bias `params[0][1]`. Il broadcasting di JAX gestisce correttamente la somma tra la matrice risultante e il vettore dei bias, dato che le dimensioni sono compatibili.
[03:28] Per generalizzare il processo a una rete con più strati, si definisce la funzione `forward(params, x)`.
[03:31] Si utilizza una sintassi Python per ciclare sui parametri di tutti gli strati, tranne l'ultimo: `for w, b in params[:-1]:`.
[03:35] Questa sintassi permette di "spacchettare" direttamente la tupla `(w, b)` contenuta in ogni elemento della lista `params`. Il ciclo si ferma prima dell'ultimo strato perché quest'ultimo richiede una funzione di attivazione diversa (la sigmoide, per problemi di classificazione binaria).
[03:46] All'interno del ciclo, l'output dello strato corrente viene calcolato e riassegnato alla variabile `x`: `x = jnp.tanh(x @ w + b)`.
[03:53] La funzione di attivazione per gli strati nascosti è la tangente iperbolica (`tanh`).
[03:56] Dopo il ciclo, si gestisce l'ultimo strato. I pesi `W` e i bias `B` dell'ultimo strato vengono estratti da `params[-1]`.
[03:59] L'output finale della rete è calcolato applicando la funzione sigmoide: `return sigmoid(x @ W + B)`.
[04:03] Punti chiave di questa implementazione: il broadcasting dei bias `B` avviene per riga e la moltiplicazione `x @ W` segue la convenzione in cui ogni riga di `x` è un campione.
[04:10] La gestione delle dimensioni è una delle parti più delicate e una fonte comune di errori.
## Funzione di Perdita e Aggiornamento dei Parametri
[04:15] Si implementa la funzione di perdita, chiamata `binary\_cross\_entropy`, che accetta i parametri `params`, l'input `X` e le etichette reali `Y`.
[04:19] Per prima cosa, si calcolano le predizioni `y\_pred` della rete tramite la funzione `forward(params, X)`.
[04:22] La formula della binary cross-entropy viene implementata calcolando la media su tutti i campioni:
```math
\text{loss} = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
```
Il codice corrispondente è `jnp.mean(-(Y * jnp.log(y\_pred) + (1 - Y) * jnp.log(1 - y\_pred)))`.
[04:35] L'ultimo componente fondamentale è la funzione `update`, che aggiorna i parametri della rete.
[04:39] Questa funzione riceve i parametri correnti `params`, un minibatch di dati (`x`, `y`) e il `learning\_rate`.
[04:43] Si utilizza il decoratore `@jax.jit` per compilare la funzione *just-in-time*, ottimizzandone l'esecuzione.
[04:48] Per calcolare i gradienti, si usa `jax.grad`, che crea una funzione gradiente a partire dalla funzione di perdita. Poiché i parametri sono il primo argomento della `binary\_cross\_entropy`, non sono necessarie modifiche.
[04:54] I gradienti vengono calcolati valutando la funzione gradiente sui parametri e sul minibatch correnti: `grads = grad\_fn(params, x, y)`.
[05:00] Un approccio per aggiornare i parametri sarebbe iterare su ogni peso e bias con un doppio ciclo `for`.
[05:10] Tuttavia, questo approccio non è ideale. Man mano che le architetture delle reti neurali diventano più complesse (es. Reti Convoluzionali, Transformer), i parametri sono organizzati in strutture dati complesse come dizionari annidati.
[05:19] Scrivere cicli `for` annidati per far corrispondere parametri e gradienti diventa macchinoso e soggetto a errori.
[05:25] JAX offre una soluzione più elegante e generale: la funzione `tree\_map` dal sottomodulo `jax.tree\_util`.
[05:29] Un "albero" (`tree`) in questo contesto si riferisce alla struttura dati annidata che contiene i parametri (liste di tuple, dizionari, ecc.).
[05:37] La funzione `tree\_map` permette di applicare una funzione a ogni "foglia" (elemento finale) di uno o più alberi contemporaneamente.
[05:41] La sintassi è la seguente: `updated\_params = jax.tree\_util.tree\_map(lambda p, g: p - learning\_rate * g, params, grads)`.
[05:48] Questa singola riga sostituisce l'intero doppio ciclo `for`. `tree\_map` applica la funzione `lambda` a ogni coppia di elementi corrispondenti `(p, g)` presi dagli alberi `params` e `grads`. La funzione `lambda p, g: p - learning\_rate * g` definisce la regola di aggiornamento del gradient descent.
[06:01] `p` rappresenta un parametro (un tensore di pesi o un vettore di bias) e `g` il suo gradiente corrispondente.
[06:10] Gli ultimi due argomenti di `tree\_map` sono le strutture dati su cui operare (`params` e `grads`).
[06:18] Questo approccio è estremamente potente perché si adatta a qualsiasi struttura dati, a condizione che `params` e `grads` abbiano la stessa struttura "ad albero".
[06:23] Ad esempio, se si utilizzano metodi di ottimizzazione più avanzati, la logica di aggiornamento può essere modificata all'interno della funzione `lambda` senza cambiare il resto del codice.
[06:38] La funzione `tree\_map` restituisce una nuova struttura dati con i parametri aggiornati, che viene quindi ritornata dalla funzione `update`.
[06:44] L'uso di `tree\_map` è una pratica standard e consigliata per scrivere codice JAX robusto e scalabile, specialmente con architetture complesse.
# Capitolo 9: Addestramento, Valutazione e Visualizzazione del Modello Generalizzato
## Impostazione degli Iperparametri e Inizializzazione
[07:05] Si procede con la fase di addestramento. Vengono definiti gli iperparametri: `layer\_sizes` = `[2, 16, 1]` (input a 2 dimensioni, uno strato nascosto da 16 neuroni, output a 1 dimensione), `learning\_rate` = `0.01`, `epochs` = `5000`, `batch\_size` = `64`.
[07:15] Questi valori sono forniti come iperparametri che garantiscono buoni risultati per questo specifico problema. In un caso reale, la loro ricerca (tuning) è una parte fondamentale del processo.
[07:21] I parametri della rete vengono inizializzati usando la funzione `init\_mlp\_params`.
## Gestione dei Minibatch e Permutazione dei Dati
[07:24] L'addestramento avviene tramite minibatch. Per prima cosa, si calcola il numero di batch per epoca: `num\_batches = x\_train.shape[0] // batch\_size`.
[07:34] Si entra nel ciclo di addestramento principale, che itera per il numero di epoche specificato.
[07:38] All'inizio di ogni epoca, è fondamentale mescolare (`shuffle`) il dataset di addestramento per evitare che la rete impari l'ordine dei dati e per migliorare la convergenza.
[07:42] Per fare ciò, si genera una nuova chiave casuale ad ogni epoca.
[07:44] Successivamente, si crea una permutazione degli indici del dataset di addestramento: `permutation = jax.random.permutation(key, x\_train.shape[0])`.
[07:51] `permutation` è un array contenente gli indici da 0 al numero di campioni, disposti in ordine casuale.
[08:03] Si utilizzano questi indici per riordinare sia i dati di input (`x\_train`) che le etichette (`y\_train`), creando `x\_shuffled` e `y\_shuffled`.
## Ciclo sui Minibatch e Aggiornamento dei Parametri
[08:14] Ora che il dataset è mescolato, si può iterare sui minibatch.
[08:17] Si avvia un ciclo interno che itera `num\_batches` volte. In ogni iterazione `i`, si calcolano gli indici di inizio (`start = i * batch\_size`) e fine (`end = start + batch\_size`) del minibatch corrente.
[08:24] Ad esempio, per `i=0`, il batch va da 0 a 64. Per `i=1`, da 64 a 128, e così via.
[08:32] Si estraggono i dati del minibatch (`x\_batch`, `y\_batch`) dagli array `x\_shuffled` e `y\_shuffled` usando lo slicing.
[08:45] Con il minibatch pronto, si chiama la funzione `update` per aggiornare i parametri della rete.
[08:50] Alla funzione `update` vengono passati i parametri correnti, `x\_batch`, `y\_batch` e il `learning\_rate`.
## Monitoraggio della Perdita
[08:55] Per monitorare l'andamento dell'addestramento, si calcola e si stampa la funzione di perdita a intervalli regolari (es. ogni 100 epoche).
[09:00] È importante notare che la perdita viene calcolata sull'intero dataset di test (`x\_test`, `y\_test`), non sul minibatch di addestramento. Questo fornisce una stima più stabile e imparziale delle prestazioni del modello su dati mai visti.
[09:10] Il codice esegue l'addestramento e si osserva che il valore della funzione di perdita diminuisce progressivamente, indicando che la rete sta imparando.
## Calcolo delle Predizioni e dell'Accuratezza
[09:22] Una volta terminato l'addestramento, si valutano le prestazioni del modello sul dataset di test.
[09:24] Le predizioni (`predictions`) vengono ottenute eseguendo la `forward propagation` con i parametri addestrati e i dati di test (`x\_test`).
[09:28] L'output della rete è una probabilità (un valore tra 0 e 1). Per ottenere una classificazione binaria (vero/falso o 1/0), si applica una soglia di 0.5. Le predizioni diventano `True` se la probabilità è maggiore di 0.5, altrimenti `False`.
[09:32] L'accuratezza (`accuracy`) viene calcolata come la media dei casi in cui le predizioni binarie corrispondono alle etichette reali (`y\_test`). Questo funziona perché `(predictions == y\_test)` produce un array di `True` (1) e `False` (0), e la media di questo array è la frazione di predizioni corrette.
## Matrice di Confusione
[09:42] Si stampa l'accuratezza e si calcola la matrice di confusione per analizzare più in dettaglio gli errori.
[09:45] La matrice di confusione viene calcolata passando le etichette reali del test (`y\_test`) e le predizioni del modello.
[09:49] Nel caso specifico, si ottiene un'accuratezza del 98% sul dataset di test. La matrice di confusione mostra che c'è stato un solo errore di classificazione.
[09:55] Gli elementi sulla diagonale principale rappresentano le predizioni corrette: 34 campioni della prima classe classificati correttamente e 25 della seconda.
[09:59] L'elemento fuori diagonale indica un campione classificato erroneamente.
## Visualizzazione del Confine Decisionale
[10:03] Infine, i risultati vengono visualizzati graficamente. Il grafico mostra i dati di addestramento (cerchi), i dati di test (croci) e il confine decisionale appreso dalla rete neurale.
[10:08] Per tracciare il confine decisionale, si crea una griglia di punti molto fitta (`meshgrid`) che copre l'intero spazio bidimensionale.
[10:15] La rete neurale viene valutata su ogni punto di questa griglia.
[10:18] A ogni punto viene assegnato un colore (rosa o azzurro) in base alla classe predetta dalla rete.
[10:21] Il risultato è una mappa a colori che mostra come la rete suddivide lo spazio. Il confine tra le due regioni colorate è il confine decisionale. Si può osservare che la rete ha imparato una separazione non lineare per distinguere le due classi.