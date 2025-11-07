# Lez17-31ott - Lab 4 NAML: SVT e JAX

## 🎯 Obiettivi del Laboratorio

### Competenze Pratiche
- Lavorare con dataset reali: **MovieLens 100k** (1000 utenti, 1600 film)
- Implementare **preprocessing** per matrix completion (shuffling, compacting)
- Costruire **baseline** con trivial recommender (media utente)
- Implementare **SVT hard thresholding** per matrix completion
- Introduzione a **JAX**: libreria Google per AD e GPU computing
- Utilizzare **JIT compilation** per accelerazione 10x
- Applicare **jax.grad** per differenziazione automatica

### Concetti Teorici
- **Data shuffling**: prevenire temporal bias nel train/test split
- **Index compaction**: rimuovere righe/colonne vuote con np.unique
- **Sparse matrices**: formato CSR per efficienza memoria
- **Baseline importance**: sempre confrontare metodi sofisticati con alternative semplici
- **Convergenza iterativa**: SVT migliora gradualmente RMSE e ρ
- **JAX immutability**: array immutabili per ottimizzazioni JIT

### Applicazioni
- **Recommender systems**: predire rating mancanti (Netflix problem)
- **Image inpainting**: ricostruzione pixel corrotti (Mondrian painting)
- **JAX for ML**: successor di TensorFlow, GPU/TPU support

## 📚 Prerequisiti

### Python & Librerie
- **NumPy**: operazioni matriciali, SVD
- **Pandas**: caricamento TSV, manipolazione DataFrame
- **SciPy**: sparse matrices (csr_matrix)
- **JAX**: installazione, jax.numpy, jax.random
- **Matplotlib**: visualizzazione risultati

### Matematica (da lezioni precedenti)
- **SVD**: Singular Value Decomposition (Lez9)
- **Matrix Completion**: teoria (Lez14), algoritmo SVT
- **Hard thresholding**: σ[σ < τ] = 0
- **Metriche**: RMSE, Pearson correlation ρ

### Teoria
- **Soft vs Hard Thresholding**: S_τ(x) vs threshold binario
- **Nuclear Norm Minimization**: rilassamento convesso del rank
- **Low-rank assumption**: R ≪ min(M,D)

## 📑 Indice Completo

### [Parte 1: MovieLens Dataset](#parte-1-movielens-dataset) (`00:00` - `21:16`)
1. [Introduzione: MovieLens Dataset](#introduzione-movielens-dataset) - `00:00:02`
2. [Dataset Structure: 100k Ratings, TSV Format](#dataset-structure) - `00:03:24`
3. [Preparazione dei Dati](#preparazione-dei-dati) - `00:03:24`
4. [Load con Pandas: pd.read_csv(sep='\\t')](#load-con-pandas) - `00:05:47`
5. [**Data Shuffling**: Rimuovere Temporal Bias](#data-shuffling) - `00:08:19`
6. [**Index Compaction**: np.unique(return_inverse=True)](#index-compaction) - `00:11:32`
7. [Train/Test Split: 80/20](#traintest-split-8020) - `00:14:12`
8. [Sparse→Dense: scipy.sparse.csr_matrix](#sparsedense) - `00:16:58`
9. [Trivial Recommender System](#trivial-recommender-system) - `00:14:59`
10. [Baseline: Media Rating Utente](#baseline-media-rating-utente) - `00:17:31`
11. [Metriche: RMSE ≈ 1.0, ρ ≈ 0.3](#metriche-rmse--10-ρ--03) - `00:19:44`

### [Parte 2: SVT Implementation](#parte-2-svt-implementation) (`21:16` - `01:06:57`)
12. [Implementazione SVT](#implementazione-svt) - `00:21:16`
13. [Hard Thresholding Variant](#hard-thresholding-variant) - `00:23:13`
14. [Soluzione ed Analisi](#soluzione-ed-analisi) - `00:23:13`
15. [Algorithm Steps](#algorithm-steps) - `00:26:48`
16. [1. SVD: U, s, Vt = np.linalg.svd()](#svd-u-s-vt) - `00:29:21`
17. [2. Threshold: s[s < τ] = 0](#threshold-ss--τ--0) - `00:32:15`
18. [3. Reconstruct: A = U @ diag(s) @ Vt](#reconstruct-a) - `00:35:08`
19. [4. Impose Known Values: A[train] = vals_train](#impose-known-values) - `00:38:42`
20. [Convergenza: RMSE→0.95, ρ→0.55](#convergenza-rmse095-ρ055) - `00:42:18`
21. [Beats Baseline dopo ~50 Iterazioni](#beats-baseline) - `00:45:53`
22. [Image Inpainting: Mondrian Example](#image-inpainting) - `00:49:27`
23. [50% Pixel Corruption, Soft Thresholding](#50-pixel-corruption) - `00:52:41`

### [Parte 3: JAX Introduction](#parte-3-jax-introduction) (`01:06:57` - `01:33:46`)
24. [JAX Introduction](#jax-introduction) - `01:06:57`
25. [Google's AD Library, TensorFlow Successor](#googles-ad-library) - `01:09:24`
26. [Differenze da NumPy](#differenze-da-numpy) - `01:12:08`
27. [1. **Immutable Arrays**: x.at[i].set(val)](#immutable-arrays) - `01:14:51`
28. [2. **Explicit Random Keys**: jax.random.PRNGKey(seed)](#explicit-random-keys) - `01:18:26`
29. [3. **Device-Agnostic**: CPU/GPU/TPU](#device-agnostic) - `01:21:39`
30. [**JIT Compilation**: @jax.jit, 10x Speedup](#jit-compilation) - `01:25:14`
31. [**Automatic Differentiation**: jax.grad(f)](#automatic-differentiation) - `01:28:57`
32. [Composable: jax.grad(jax.grad(f)) = 2nd Derivative](#composable-grad) - `01:31:42`

### [Parte 4: JAX Deep Dive](#parte-4-jax-deep-dive) (`01:33:46` - `02:10:54`)
33. [Composable Transformations](#composable-transformations) - `01:33:46`
34. [Multiple Inputs: jax.jacrev, jax.jacfwd](#multiple-inputs) - `01:37:27`
35. [Hessian: jacrev(jacfwd(f))](#hessian-jacrevjacfwdf) - `01:40:13`
36. [Non-Differentiable Functions: |x| in x=0](#non-differentiable-functions) - `01:43:06`
37. [**VMAP**: Vettorizzazione Automatica](#vmap-vettorizzazione) - `01:46:41`
38. [500x Speedup con vmap+jit](#500x-speedup) - `01:50:28`
39. [Low-Level Interface: LAX](#low-level-interface-lax) - `01:54:12`
40. [JIT Limitations: Dynamic Shapes](#jit-limitations) - `01:57:45`
41. [Pure Functions Required](#pure-functions-required) - `02:01:23`
42. [Out-of-Bounds Indexing](#out-of-bounds-indexing) - `02:04:56`
43. [Random Number Generation](#random-number-generation) - `02:08:31`

---

## Parte 1: MovieLens Dataset

## Introduzione: MovieLens Dataset

---

## Introduzione: MovieLens Dataset

`00:00:02` 
Okay, quindi buon pomeriggio a tutti. Quindi il programma per oggi è il seguente. Nella prima metà del lab, implementeremo insieme l'algoritmo di singular value thresholding applicato a un'importante applicazione che dovreste aver visto durante la lezione. Nel frattempo, nella seconda metà, vedremo insieme qualcosa che è piuttosto diverso, che è JAX, che è una libreria di differenziazione automatica. JAX è molto usato, è molto mobile e molto potente, e lo vedremo insieme in modo frontale.

`00:00:41` 
Passeremo attraverso alcuni esempi e vedremo passo dopo passo quali sono gli oggetti caratteristici. Okay? Quindi, il materiale è praticamente lo stesso della settimana scorsa, ho aggiornato solo pochissimi dettagli minori dei notebook, quindi se volete potete trovare la nuova versione, che è molto leggermente diversa. Quindi solo per fissare le idee, lasciate che usi la slide che dovreste aver visto durante la lezione.

`00:01:13` 
Quindi il problema qui è che questo dataset MovieLens, che è un dataset composto da utenti che hanno valutato certi film, e vogliamo predire quanto un utente apprezzerebbe un altro film. Questo è un problema molto interessante perché per grandi compagnie come Netflix o Amazon, solo pochi punti percentuali di migliori sistemi di raccomandazione significano milioni in termini di ricavi.

`00:01:43` 
Quindi, questo è un problema molto interessante in un'applicazione reale. Grazie mille per la vostra attenzione. Quindi il dataset qui, che dobbiamo caricare, quindi lasciatemi connettere la macchina, e poi possiamo caricare il dataset che è sul BIP. Quindi andate qui nella cartella, caricate nella sessione locale il dataset MovieLens, che è un file CSV.

`00:02:18` 
Okay, quindi vi sta dicendo che qui abbiamo 10.000 valutazioni da 1 a 5 da più o meno 1.000 utenti su 1.600 film. E ogni utente ha valutato almeno 20 film. Il dataset qui non è dato nello stesso formato che vedete qui in queste belle matrici, quindi il nostro primo compito è convertire il nostro dataset, che è un file CSV, nelle matrici, dove sulle varie righe abbiamo i vari utenti e sulle varie colonne abbiamo i vari film.

`00:02:53` 
In particolare qui, vi sta dicendo che questo file CSV è in realtà una lista separata da tabulazioni, dove ogni riga è un ID utente che identifica univocamente l'utente, l'ID dell'item, che è l'ID del film, la valutazione, che è il numero che va da 1 a 5, e il timestamp, che è quando l'utente ha inserito la sua valutazione. Non useremo questo, ma è un'informazione che abbiamo. Quindi iniziamo importando le librerie di area. Stiamo usando Pandas per leggere il file CSV.

`00:03:24` 
E poi abbiamo NumPy e alcune cose da SciPy che vedremo passo dopo passo. Quindi qui, prima di tutto, leggiamo il dataset. Di solito, potete pensare che durante l'esame, questo passo vi è dato. Ma una volta che abbiamo il dataset, allora abbiamo... Quindi questo è il nostro dataset. Questo è un dataframe Pandas. E avete nel notebook Python questa bella visualizzazione. Vedete questo più o meno come un foglio Excel. Vedete le colonne dei valori, il loro nome, e le righe dei valori.

## Preparazione dei Dati

`00:03:56` 
Quindi, per esempio, qui abbiamo l'utente 196 ha valutato il film 242. Abbiamo una valutazione di 3 a questo timestamp, e così via. Quindi il primo compito è, okay, controlliamo quanti film abbiamo, quante persone, e quante valutazioni. Quindi per controllare quante persone abbiamo, possiamo usare la funzione np.unique per ottenere tutti gli ID unici che abbiamo.

`00:04:26` 
Quindi il numero di utenti è il numero di ID utente diversi che abbiamo. Questo sarà nel nostro modo con tutti gli ID utenti unici. Prendiamo l'attributo size per ottenere il numero di utenti diversi. Similarmente, il numero di film è okay. Usiamo NumPy unique per estrarre tutti gli ID film unici da questa colonna. E usiamo l'attributo size per ottenere il numero di film. Invece, il numero di valutazioni è il numero di righe nei nostri dati.

`00:04:56` 
Quindi il numero di righe nel nostro, diciamo, foglio Excel. Infatti, otteniamo quello che vi è stato detto sul dataset, sul numero di persone, il numero di film, e il numero di valutazioni. Fare un doppio controllo è sempre una buona pratica per assicurarsi che le informazioni fornite sul dataset siano le stesse che stiamo ottenendo. Solo per assicurarsi che abbiate letto il dataset correttamente. Poi qui stiamo introducendo un nuovo passo che non abbiamo visto nel lab precedente, ma questo è molto, molto importante da quando usate un dataset realistico. Quindi dobbiamo mescolarli. Perché dobbiamo mescolare il dataset? Perché probabilmente c'è un timestamp e quindi gli utenti hanno inserito le loro valutazioni nel tempo.

`00:05:48` 
Quindi molto realisticamente, la tabella è ordinata nel tempo e a un certo punto volete dividere il dataset nella parte di training, validazione e testing. E se lo dividiamo come il primo 80% e l'ultimo 20%, quello che abbiamo è che all'inizio abbiamo tutte le vecchie valutazioni e alla fine abbiamo tutte le nuove valutazioni. Quindi ci sono alcune idiosincrasie del dataset che sono connesse alla divisione. Pensate per esempio, un utente si registra per la prima volta nel sito web e poi valuta 20 film.

`00:06:28` 
Se non mescoliamo, tutti i 20 film valutati da questo utente saranno in un certo punto del dataset. Invece, vogliamo testare il nostro algoritmo in alcuni posti dove questo utente ha aggiunto questi film e alcuni li vogliamo usare per l'addestramento. Quindi per assicurarci che ci sia un po' di entropia e per lo stesso utente possiamo avere alcuni dati sia nel training che nel test, vogliamo mescolare il dataset in modo che le caratteristiche temporali di come il dataset è stato costruito.

`00:07:04` 
non siano prese in considerazione quando lo dividiamo in test e dataset. Quindi quello che facciamo è che impostiamo un seed casuale per la riproducibilità, poi creiamo un indexes che è solo un array con tutti i numeri da 0 a 100.000.

`00:07:43` 
Poi abbiamo una funzione chiamata shuffle che permuta casualmente questo array e quindi dopo lo shuffle quello che abbiamo è che i numeri da 0 a 10.000 sono mescolati in questo array. e poi quello che facciamo qui con le parentesi quadre stiamo dicendo che se mettete zero otteniamo la prima riga se mettiamo due otteniamo la seconda riga e così via quindi se mettiamo un array di indici mescolati.

`00:08:16` 
quello che stiamo facendo è che stiamo permutando le righe seguendo questi indici permutati e quindi stiamo permutando le righe in questo modo una volta che abbiamo permutato le righe possiamo estrarre le colonne che ci interessano quindi abbiamo le righe che sono gli ID utente seguendo questa convenzione, poi le colonne che sono gli ID film e i valori che sono le valutazioni che.

`00:08:47` 
sono gli elementi nelle matrici okay quindi a questo punto abbiamo tre vettori diversi, come triplette di indice riga indice colonna e valore che abbiamo nelle matrici. Tuttavia, ancora non abbiamo finito perché ci possono essere alcuni problemi. Quindi prendete, per esempio, questo esempio qui. E questo è un dataset molto piccolo con tre righe. Ma questo è qualcosa che potrebbe succedere perché non siamo sicuri di come è fatto questo dataset. Quindi per esempio, abbiamo l'utente zero che ha valutato il film uno. Lo valuteremo nel nucleo. Abbiamo l'utente zero che ha valutato il film due.

`00:09:34` 
Abbiamo una valutazione di cinque e l'utente numero tre, che ha valutato il film due. Abbiamo una valutazione di cinque. Se usate questo valore zero zero tre e questi valori uno e due come l'indice delle righe e delle colonne, quando usiamo le matrici, abbiamo qualcosa del genere. Okay, quindi per abbiamo l'utente zero che ha valutato il film numero uno e numero due, e l'utente numero tre che ha valutato solo il film numero due. Quello che abbiamo qui, abbiamo righe.

`00:10:10` 
con molti zeri e colonne con molti zeri. Okay, non vogliamo questo perché queste, righe non sono connesse a nessun utente che sia significativo per noi, e la prima colonna non è connessa a un film che sia significativo per noi. Quindi vogliamo rimuovere queste righe e queste colonne vuote perché non possiamo fare alcuna apparenza su un film che nessuno ha visto o sull'utente che, non ha fatto alcuna valutazione. Non ha senso. Quindi vogliamo rimuovere diciamo intervalli vuoti.

`00:10:46` 
in questi insiemi. Come facciamo questo? Per il bene di questo esercizio potete pensare che questa funzione fa esattamente quello che volete. Non voglio entrare nei dettagli di come funziona ma se volete potete controllare la documentazione ma sappiate che la funzione np-unique che abbiamo usato prima ha un'opzione speciale, che è chiamata return inverse che mappa questo indice di vettori all'indice del.

`00:11:20` 
vettore indice che vogliamo. Quindi prima avevamo righe con righe zero duplicate queste output righe senza questi intervalli extra. Quindi questo passo è solo per evitare di avere queste righe zero e colonne zero. Poi.

`00:11:50` 
Siamo tutti pronti, perché ora i nostri array sono a posto, e possiamo dividerli in training e testing. Quindi i dati di training sono l'80% di tutte le valutazioni, quindi otteniamo il numero di valutazioni, lo moltiplichiamo per 0.8, e arrotondiamo questo numero. E questo è il numero di campioni che usiamo per le valutazioni. Dato che il dataset è mescolato, possiamo prendere come dati di training il primo 80%, quindi, come vi ho detto, abbiamo la colonna, poi i dati di training, questo significa che stiamo prendendo tutti gli elementi da 0 a training data,

`00:12:31` 
e poi la parte di testing è tutti gli elementi che vanno da training data alla fine. Quindi la colonna senza nulla sulla destra significa fino alla fine. Così, abbiamo sei vettori, la prima tripletta di righe, colonne, e valori, che è quella di training, e la seconda tripletta, che è quella che usiamo per il testing.

`00:13:05` 
Il passo successivo è, abbiamo questa tripletta, c'è una funzione dentro SciPy che trasforma questa tripletta esattamente nella matrice che vogliamo. In particolare, questa è chiamata matrice sparsa, perché è una matrice con molti zeri, molti punti interrogativi, e dato che questa è una cosa che viene fatta spesso, quello che facciamo è che chiamiamo questa funzione CRS matrix e anche la tripletta di valori righe.

`00:13:40` 
indici e indici colonne vi riferisce una matrice sparsa costruita in questo modo. Quindi in particolare l'entrata ij della nostra matrice sarà la valutazione ij se ij è un indice che abbiamo altrimenti sarà zero. Infine questa matrice sparsa vogliamo effettivamente avere gli zeri e.

`00:14:12` 
non solo memorizzarli implicitamente quindi trasformiamo questa matrice sparsa in una matrice piena effettiva. Quindi se controllate x full. è questa matrice abbiamo molti zeri perché molti film e molti utenti non hanno, creato quella combinazione di film e utente tuttavia siamo sicuri che non ci sono righe zero e nessuna colonna zero okay ora viene quello che dovete implementare quindi questo era uh tutta questa roba.

## Trivial Recommender System

`00:14:59` 
ora viene quello che ora vi chiedo di implementare quindi prima di tutto quello che abbiamo è un sistema di raccomandazione banale quindi quando implementate un nuovo modello di machine learning quello che dovete quello che dovete fare o quella che è la best practice è sempre avere una baseline quindi un modello molto semplice, a volte anche solo numeri casuali che usate come baseline per controllare se il vostro modello sofisticato batte questa baseline.

`00:15:33` 
La nostra baseline, in questo caso, è il Trivial Recommender System. Il Trivial Recommender System dice, okay, abbiamo un utente, ha valutato alcuni film, probabilmente la prossima valutazione sarà la media delle sue valutazioni precedenti. Quindi, un utente ha tutti cinque, probabilmente il prossimo film sarà valutato cinque. Un utente ha tutti uno, probabilmente il prossimo film sarà valutato uno. Okay? Questo è molto semplice, in qualche modo anche stupido.

`00:16:05` 
Tuttavia, è una baseline semplice e possiamo usarla per controllare se il nostro singular value thresholding sta effettivamente funzionando. Quindi questa è solo la definizione matematica. Quindi il sistema di raccomandazione banale, la predizione per ij è uguale alla media per quell'utente di tutti i film. Quindi tutti i j, tutti i film che ha visto. In pratica, quello che state facendo, state fissando l'utente. Quindi state prendendo una riga. Stiamo sommando insieme tutti gli elementi delle righe e state dividendo per il numero di elementi che non sono zero perché se è zero.

`00:16:50` 
Non è lì. E quindi stiamo prendendo solo la media dei film che stiamo considerando. Poi il secondo ingrediente di cui abbiamo bisogno è una metrica. Quindi abbiamo un modello e vogliamo controllare se questo modello è meglio dell'altro. E per valutare quanto bene sta facendo il nostro modello, abbiamo bisogno di un indice, un numero scalare che ci dice quanto bene stiamo facendo. In questo caso, vi propongo due metriche diverse.

`00:17:25` 
La radice dell'errore quadratico medio, quindi quanto è lontana la nostra predizione dalla valutazione vera, e il coefficiente di correlazione di Pearson. Questo è un numero che sta tra meno uno e uno. È molto usato in statistica, e quello che dice è quanto due quantità stanno correlando. Se è uno, abbiamo una perfetta correlazione lineare. Se è meno uno, abbiamo non solo una perfetta correlazione inversa.

`00:18:00` 
E se è zero, significa che probabilmente queste due quantità non stanno correlando affatto. Okay, quindi il primo compito è implementare questo predittore banale, quindi avete la matrice con tutti gli utenti e tutti i film, e volete implementare questo predittore banale e capire quali sono le sue metriche. Il predittore banale è costruito usando il dataset di training e valutato usando il dataset di test.

`00:18:34` 
Quindi qui, data la matrice X, vi chiedo di implementare qual è la media della valutazione di ogni utente, e usare questa quantità per calcolare la radice dell'errore quadratico medio e il coefficiente di correlazione rho. Il coefficiente di correlazione rho, potete calcolarlo usando questa funzione, pearson r, che viene da SciPy.

## Implementazione SVT

`00:19:05` 
Secondo passo, implementare l'algoritmo di singular value thresholding. Qui vi chiedo di implementare la variazione hard, dove il troncamento è hard. La variazione hard è più semplice da implementare della variazione soft. Quindi durante la lezione, avete visto questo algoritmo. Questo è più complesso e vi mostrerò un'implementazione di questo dopo questo.

`00:19:38` 
Per iniziare, vi chiedo di implementare quest'altra versione, che è molto più semplice. Quindi quello che fate è, prima di tutto, calcolate l'SVD della vostra matrice. Mantenete tutte le componenti singolari, che hanno un valore singolare che è maggiore della soglia. E poi ricostruite la matrice usando queste componenti.

`00:20:11` 
Dopo questo, imponete che nella componente ij, avete i veri valori del dataset di training, e poi calcolate la differenza dell'ij del passo temporale corrente, del passo corrente con quello precedente, e fate alcune iterazioni. Questo è chiamato hard thresholding perché state tagliando, a un certo punto, i valori singolari e state mantenendo solo alcuni.

`00:20:43` 
Qui abbiamo il loop, e qui dobbiamo implementare questi cinque passi. Quindi, l'SVD, mantenere solo i valori singolari importanti, assicurarsi che nell'entrata che abbiamo i valori corretti sovrascrivendo la matrice, calcoliamo la differenza. e poi calcolate e salvate la matrice di test. Quindi la valutazione della matrice sul test.

`00:21:16` 
Infine, se volete, potete anche plottare una storia della radice dell'errore quadratico medio e del coefficiente di correlazione. Per questo compito, vi do 20 minuti. Cercate di fare quanto più potete. Se avete domande, sarò in giro. Avete domande iniziali sul vostro compito? Okay.

`00:23:13` 
Grazie.

`00:28:23` 
Oh.

`00:29:16` 
Grazie.

`00:39:27` 
Sì.

## Soluzione ed Analisi

`00:40:44` 
okay controlliamo la soluzione quindi prima di tutto fissiamo solo un po' le idee, Quindi, vogliamo avere, come risultato, questo VALS trivial, quindi questo è un vettore che ha un numero di elementi uguale al numero di campioni nel dataset di test, quindi questo sarà 20.000, e in particolare questo è connesso a cosa?

`00:41:20` 
A row test e col test. Quindi, data una coppia, l'ID della persona e l'ID del film, vogliamo connetterlo a un valore che è la predizione per quel film e quell'utente, okay? Quindi, abbiamo due passi. Il primo è che per ogni utente, vogliamo calcolare la sua media.

`00:41:55` 
Quindi per fare questo direttamente senza il for loop e usando solo la vettorizzazione, direi che è un compito difficile. Nella soluzione, vi mostro come fare questo. Voglio mostrarvi un modo più ingenuo, che è completamente a posto. Quindi cosa facciamo? Diciamo, okay, la valutazione media sarà np empty vettore con dimensione x dot shape zero.

`00:42:28` 
Quindi abbiamo un vettore vuoto con la stessa dimensione del numero di utenti. Era chiamato x, era chiamato x full. Ma questo è anche m people. Questo è più chiaro, okay? Poi cosa facciamo? Iteriamo su ogni persona, utente e calcoliamo la sua media. Quindi per i in range del numero di persone, cosa facciamo? Quindi penso che il modo migliore per vedere questo sia usando la matrice xfull.

`00:43:07` 
Quindi vogliamo prendere cosa? La riga i-esima. Quindi stiamo facendo i, doppio punto. Questa è la riga i-esima. E poi qui cosa facciamo? Calcoliamo la somma. Sommiamo tutto insieme. E poi vogliamo dividere per il numero di film che non sono zero. Quindi come può essere fatto questo? Beh, prendiamo questo.

`00:43:45` 
Prendiamo quando questo è maggiore di zero, questo sarà o vero o falso, quindi lasciatemi farlo così, diciamo che per esempio prendiamo questo, che è la riga zero, possiamo fare maggiore di zero, questo sarà un vettore di vero e falso, quando la valutazione è maggiore di zero, i falsi sono automaticamente zero, i veri sono automaticamente uno, quindi possiamo fare una somma, questo è il numero di elementi non zero di questa riga.

`00:44:30` 
Quindi la valutazione media per l'utente i-esimo è questa. Grazie. Grazie. Okay, un altro modo che voglio mostrarvi che è un po' più pulito direi ma.

`00:45:01` 
ovviamente questo era okay è dire okay la valutazione per l'utente è cosa? Beh possiamo prendere, vals train, questo è un vettore dove ogni elemento è connesso a un certo, utente e un certo film e vogliamo filtrare solo le valutazioni di questo utente. Come possiamo fare questo? Usiamo una maschera booleana e diciamo row train uguale uguale a i.

`00:45:37` 
Okay quindi queste sono solo, le valutazioni di questo utente perché questo è un vettore booleano di vero e falso, dove è vero solo per gli elementi di questo utente poi passiamo questa maschera booleana a questo vettore e quindi stiamo estraendo solo le valutazioni di questo utente e a questo punto possiamo fare la media.

`00:46:07` 
perché ha anche la dimensione corretta quindi chiamiamo questo v2 e se stampiamo. controlliamo che questi strumenti siano uguali okay abbiamo tutti zeri.

`00:46:37` 
Okay, quindi abbiamo anche controllato che i due modi che abbiamo usato sono effettivamente gli stessi. Okay, quindi sono entrambi modi validi. Voglio solo mostrarvi due modi per fare la stessa cosa mentre usate la matrice piena, quella con tutti gli zeri e invece usate solo le triplette, la colonna i-esima, la riga i-esima e il valore. Okay, poi non abbiamo finito perché questo è un vettore che ha una dimensione uguale al numero di persone. Poi vogliamo usare il nostro modello.

`00:47:13` 
Quindi il nostro modello è qualcosa che anche l'ID riga e l'ID colonna, quindi l'ID utente e l'ID film, ci restituisce la valutazione del film. Tuttavia, questo è un, modello particolare perché non siamo interessati al film quindi dato che questa è la media per l'utente questo è indipendente dal film okay quindi come otteniamo questo valore trivial.

`00:47:46` 
beh quello che facciamo è che otteniamo la valutazione media e la prendiamo a row test cosa significa questo è un vettore pieno di indici utente come zero uno due tre, quindi indipendentemente dal film che stiamo considerando stiamo prendendo la stessa predizione che è la media per quell'utente.

`00:48:22` 
come facciamo questo beh prendiamo la predizione media per quell'utente e la iteriamo. Se volete questo in modo più pulito, quindi valuesTrivial è uguale a np.empty, questo ha la stessa dimensione di valsTrain, e poi cosa facciamo, per i in range di valsTrivial.size, cosa stiamo facendo, qui avremmo che il nostro movieId è uguale a test di i.

`00:49:10` 
Il nostro userId è uguale a rowTest di i. Tuttavia, per quell'utente in quel film, non siamo interessati al movieId, perché stiamo prendendo la media su tutti i film. Quindi la nostra predizione è qual è la valutazione media per questo userId e poi come predittore banale per questa coppia è questo.

`00:49:50` 
Okay, questo è il modo verboso dove fate ogni passo. Questo è il modo veloce. Quindi qual è la predizione per ogni valore nel trade. Quindi iteriamo su ogni elemento al registratore, che è l'ID film. Stiamo fissando l'ID utente che stiamo fissando e poi la nostra predizione è la media per questo utente. Quindi non siamo interessati a questo film.

`00:50:21` 
La valutazione media per l'utente è questo valore scalare. E quindi la predizione, la predizione banale in questo punto è questa. Cosa devo fare da qui?

`00:51:17` 
Oh, non è train, scusate. Qui è test. Okay. E queste sono le due metriche. Quindi per calcolare le metriche, okay, l'errore è la discrepanza tra i veri valori di test e il predittore banale. Sì. Non sono sicuro di capire questo direttamente, perché significa che l'utente alla posizione zero nella valutazione media è l'utente zero.

`00:51:51` 
Dici questo? Sì. Questo è esattamente lo stesso di questo. È una sintassi diversa, ma quello che state facendo è lo stesso. Quindi se compattate questa cosa, okay, diciamo che questo, la predizione è questa cosa, okay, e user ID è questa cosa, okay?

`00:52:22` 
Questa riga nel gruppo for è semanticamente esattamente la stessa cosa di questo. Quindi questa cosa sta facendo dietro le quinte esattamente questo. Quindi se questo è chiaro, questo è solo una questione di sintassi diversa per scrivere la stessa cosa. Sì, questi sono già mescolati. Qui tutto è mescolato ed è coerente uno con l'altro.

`00:53:05` 
Forse possiamo parlare di questo dopo. L'errore che ho avuto un minuto fa. Qui ho messo la dimensione del training. Sta iterando sulla dimensione del test. Okay, quindi abbiamo ottenuto l'errore, che è la discrepanza tra i veri valori di test e il nostro modello, e poi mettiamo in alcune matrici.

`00:53:40` 
Quindi radice dell'errore quadratico medio, prendiamo ogni valore, lo eleviamo al quadrato, prendiamo la media, prendiamo la radice quadrata. L'altra metrica è la pearson R, quindi passiamo questi due vettori, ci dà un numero. Quindi questa è la nostra matrice, quindi radice dell'errore quadratico medio circa 1, correlazione 0.3. Questo significa che è leggermente correlato, diciamo. Abbiamo usato questo indice anche per il page rank, e abbiamo ottenuto qualcosa di meglio in quel caso.

`00:54:20` 
Okay, ora passiamo al nostro algoritmo, che è l'SVT. Quindi quello che stiamo facendo qui, quindi prima di tutto, impostiamo alcuni parametri, il numero di iterazioni totali, la soglia, e la tolleranza sull'incremento per fermare il loop. Questa soglia in particolare è un parametro che dovrebbe essere regolato problema per problema. Per ora, vi ho già dato un valore che è abbastanza buono, che ho testato, ed è 100.

`00:54:55` 
Nella vita reale, dovete solo cambiare un po' i parametri e vedere cosa funziona e cosa no. Okay, quindi implementiamo l'algoritmo. Prima di tutto, salvo una copia di A. Perché? Perché devo calcolare l'incremento tra il passo precedente e quello corrente.

`00:55:29` 
Per controllare qual è la differenza e quindi salvo una copia poi applichiamo l'SVD come al solito e np dot linear algebra dot SVD di a full matrix è uguale a false poi prendiamo solo i valori singolari che sono maggiori della soglia quindi dove s è più piccolo della soglia li impostiamo a zero e poi calcoliamo a.

`00:56:09` 
Quindi a è u moltiplicato per s trasposta di v. Okay, quindi questo non è il modo più efficiente di fare questo, ma questo è ciò che lo rende più chiaro. Quindi prendiamo i valori singolari, tutti i valori singolari che sono più piccoli della soglia, li mettiamo a zero, quindi non abbiamo contributo dalle componenti principali in quelle direzioni.

`00:56:41` 
Quindi in un certo modo, stiamo prendendo solo quello che pensiamo sia importante, e poi dobbiamo assicurarci che dove abbiamo dati, la matrice sia effettivamente quello che sappiamo essere vero. Quindi in A, nelle righe trained, righe, scusate, colonne trained, questo è uguale a vals trained.

`00:57:17` 
okay per ogni componente ij dove conosciamo la predizione corretta la impostiamo uguale alla predizione corretta poi abbiamo l'incremento che è uguale a np linear algebra norm.

`00:57:49` 
di a meno a old norma di Frobenius poi. calcoliamo effettivamente la predizione quindi abbiamo questo è il nostro modello quali sono le nostre predizioni quindi.

`00:58:20` 
i valori predetti è a preso alle righe test righe colonne queste sono le nostre predizioni quindi questa matrice ora sarà piena perché quando facciamo SVD e la ricostruiamo eliminando alcuni valori singolari allora avremo una struttura molto diversa dove.

`00:58:51` 
gli zeri sono diventati qualcos'altro e gli errori sono vals test meno il predetto. Infine, salviamo la matrice, quindi abbiamo una lista, root mean squared error list, alla quale appendiamo np dot square root di np dot mean di errors al quadrato, e nella row list, appendiamo la pearson R di vals test vals predicted.

`00:59:57` 
Okay, quindi questo richiede un po'. Se tutto è corretto, alla fine avremo qualcosa che è un po' meglio del predittore banale. Quindi ricordo che prima la radice dell'errore quadratico medio era circa 1 e il rho era più o meno 0.3.

`01:00:28` 
Vogliamo che la radice dell'errore quadratico medio sia il più piccola possibile e il rho sia il più vicino possibile a 1. Quindi all'inizio siamo molto peggio, ma poco a poco stiamo migliorando. Quindi il rho è già meglio, è oltre 0.5. La radice dell'errore quadratico medio per ora è più grande, ma passo dopo passo sta diventando più piccola.

`01:01:00` 
E sperabilmente scende sotto 1. che dovrebbe essere il caso se ricordo correttamente uh.

`01:02:04` 
okay quando questo gira. L'ultimo passo che facciamo è plottare la storia della radice dell'errore quadratico medio, e del coefficiente di correlazione. Quindi quello che facciamo è, okay, facciamo un nuovo plot, fig-axe uguale a plt dot subplots, due colonne, una riga. Quello che facciamo, sul primo asse,

`01:02:35` 
plottiamo la storia della radice dell'errore quadratico medio. Sul secondo, plottiamo la storia di rho, e poi lo confrontiamo con quello banale. Quindi, axe zero dot, uh,

`01:03:05` 
È H-line. Quindi il predittore banale è solo un valore. Non ha alcuna storia. Quindi plottiamo solo una linea orizzontale. Quindi plottiamo H sta per horizontal line parallela all'asse. E questo è RMSE trivial. E sull'asse uno, plottiamo H-line.

`01:03:44` 
Quindi questo è sul predittore precedente, abbiamo un numero. Non abbiamo alcuna storia. Vogliamo una linea orizzontale. Quindi questa funzione, dato un numero, stampa una linea orizzontale. con y uguale a questo numero, e questo è il risultato.

`01:04:25` 
Quindi cosa sta succedendo qui è che, se vedete, riguardo al coefficiente di correlazione, il nostro, algoritmo SVT è rapidamente meglio del predittore banale. Andiamo oltre 0.3438 abbastanza rapidamente, tuttavia, se andiamo avanti, la radice dell'errore quadratico medio il predittore banale stava facendo abbastanza bene e andiamo sotto la soglia solo alla fine, e a malapena.

`01:04:59` 
Quindi questo vi mostra anche l'importanza di usare un modello baseline. Un modello molto semplice qui stava già performando abbastanza bene, e vi sta dicendo, okay, state facendo un po' meglio, ma non state facendo così tanto meglio che usare qualcosa che è così semplice come la media. Avete domande? Okay, se no, voglio mostrare.

`01:05:46` 
vi molto rapidamente un'implementazione di quest'altro algoritmo, che è esattamente quello che trovate nelle slide, e con un'applicazione alla ricostruzione di immagini. Okay, questa sarà una panoramica molto rapida, perché non ho il tempo di mostrarvi i dettagli. Tuttavia, è molto simile a quello che avete visto durante la lezione. Quindi prima di tutto, quello con la soluzione è già su WeBit.

`01:06:23` 
Quindi prima di tutto, importiamo le librerie usuali, più questa libreria PIL, che è per leggere immagini. E poi abbiamo due funzioni che potete pensare che durante l'esame queste siano date. Una per ridimensionare l'immagine, così che se ha molti pixel, non facciamo l'SVD per una matrice molto, molto grande. E un'altra per generare una matrice che simula una corruzione.

`01:06:57` 
dell'immagine. In questo algoritmo, questo è il p-omega, è la proiezione ed è alcuni pixel che consideriamo come sbagliati e vogliamo ricostruire l'immagine in questi punti perché sono corrotti. Poi quello che facciamo è che carichiamo la nostra immagine, quindi usiamo image.open per leggere l'immagine, la ridimensioniamo a una dimensione molto più piccola di 400 e la trasformiamo in un array.

`01:07:29` 
In particolare, in scala di grigi, facendo la media sull'ultimo asse, quindi stiamo facendo la media sui canali RGB. Abbiamo fatto questo molte volte. Prima di tutto, voglio mostrarvi questo, abbiamo visto questo molte volte, è il dipinto di Mondrian. E poi implementiamo il nostro algoritmo. Quindi cosa diciamo? Diciamo che la percentuale di pixel sbagliati è del 50%.

`01:07:59` 
Generiamo una maschera casuale di rumore sul 50% dei pixel. E poi eseguiamo il nostro algoritmo. Il numero totale di iterazioni, massimo uno è 700. E la tolleranza qui è 0.01. Questi parametri, delta, tau, e c0, sono quelli che trovate sulle slide. Quindi abbiamo che delta è 1.2. Come vedete qui, abbiamo che tau è esattamente questo con gamma uguale a 5.

`01:08:39` 
Quindi la radice quadrata del prodotto delle dimensioni dell'immagine. E c0. Non so se è scritto sulle slide ma lo trovate nel paper. Questo è di solito quello che è scelto come c0. Poi quello che facciamo è che abbiamo tre matrici.

`01:09:11` 
x, m e y. y è c0 delta moltiplicato per p di x che è questo. c0 delta p è la maschera quindi moltiplichiamo la maschera per x e quindi questo è p omega di x. E poi facciamo il loop. Questo è molto simile a prima, come struttura complessiva ma le operazioni effettive sono un po' diverse. Quindi applichiamo questo SVD. Questo è il soft thresholding. Invece di tagliare.

`01:09:46` 
duramente la soglia, riduciamo tutto più verso zero. della soglia tau, ricostruiamo m, e poi abbiamo la variabile duale y, dove la aggiorniamo. Quindi questo è l'SVD, il soft thresholding, dove spostiamo tutto verso zero, ricostruiamo m, calcoliamo r, che è il residuo, che è questa parte qui,

`01:10:19` 
e aggiorniamo y. Poi calcoliamo l'errore, l'errore relativo, e iteriamo finché la differenza è più piccola di una certa soglia. Infine, ogni quarta iterazione plottiamo il risultato. Okay, quindi per esempio, questo è il nostro punto di partenza, questa è la nostra matrice originale, quella con il rumore, con i dipinti, il 50% dei pixel sono corrotti, questa è l'immagine ricostruita alla prima iterazione, quindi è piena di zeri, e questo è l'errore, cioè la differenza tra l'immagine ricostruita e l'immagine originale.

`01:11:10` 
Questo richiederà molte iterazioni, sto eseguendo questo sul mio laptop perché su Google Chrome ci vorrà del tempo, e vedete che passo dopo passo ci stiamo avvicinando sempre di più all'immagine originale, se eseguite questo per molte iterazioni, alla fine ottenete qualcosa che è davvero buono, quindi sembra che questo fosse quello che l'algoritmo conosce, e questo è quello che ha ricostruito, e questo era l'originale, quindi sta facendo un lavoro abbastanza buono.

`01:11:49` 
Quindi abbiamo visto che questo è uno scenario molto bello per l'SVT perché gli piacciono queste immagini molto geometriche. Vi fornisco anche qualcosa che è un po' più impegnativo che è una foto di un paesaggio e voglio mostrarvi che questo funziona bene anche quando l'immagine è un po' più complessa. Quindi questa è l'immagine originale, quella con molto rumore,

`01:12:27` 
quella con molto rumore, quella con molto rumore. quindi qui dopo 500 iterazioni vedete che le ricostruzioni sono state abbastanza decenti come c'è molto rumore e c'è molta sfocatura tuttavia non è male e se lasciate girare questo per molto tempo questo diventerà sempre meglio quindi questo è praticamente il risultato finale.

`01:13:01` 
e se pensate che siete partiti da questo che è qualcosa che non potete più o meno nemmeno riconoscere e avete ottenuto questo non è così male okay ora farò 10 minuti di pausa perché poi devo parlare per un'ora di JAX senza sosta quindi per favore lasciatemi avere.

## JAX: Introduzione

`01:13:36` 
Solo 10 minuti di pausa, quindi non finiamo troppo tardi, e a 5 minuti, prima delle 4, ricominciamo. Se avete domande, sarò in giro, potete venire qui, ma lasciatemi solo 10 minuti. Grazie. Okay, quindi ricominciamo. Quindi, abbiamo ora un notebook chiamato JAX. Mentre lo caricate, lasciatemi dirvi alcune cose su JAX.

`01:14:22` 
Quindi, prima di tutto, JAX è una libreria di differenziazione automatica. Quindi avete visto durante la lezione la differenziazione automatica. Implementare questo tipo di libreria è davvero, davvero complesso. E quindi di solito è qualcosa su cui fate affidamento. Quindi un po' di storia. Alcuni anni fa, le librerie di differenziazione automatica più importanti erano TensorFlow e PyTorch.

`01:14:58` 
All'inizio, TensorFlow, sviluppato da Google, era il più popolare perché era il più veloce e raggiungeva questo calcolando il grafo del calcolo. Solo una volta. Tuttavia. Dopo questo, PyTorch, che calcolava il grafo dinamicamente, è diventato un po' più veloce, e ha raggiunto un punto dove il fatto che aveva un'API più semplice ha superato l'uso di TensorFlow.

`01:15:40` 
Oggigiorno, Google si è spostato da TensorFlow a JAX, perché JAX è molto più potente, e ha un'API molto bella. La differenza tra JAX e TensorFlow o PyTorch è che JAX è solo una libreria di differenziazione automatica, il che significa che non vi fornisce alcuno strumento per creare reti neurali. Quindi questo è lo svantaggio di JAX. Tuttavia, ci sono librerie di terze parti che usano JAX per implementare reti neurali e ottimizzatori.

`01:16:17` 
Per esempio, Keras, dalla terza release maggiore, supporta sia JAX, PyTorch, che TensorFlow come backend per la differenziazione automatica per costruire reti neurali. Okay, quindi questa è solo una panoramica molto breve, e qualcosa che è un po' diverso ora dalle altre volte è che, per favore andate sulla freccia sul lato di Connect e selezionate Change Runtime Type, e selezionate GPU.

`01:16:54` 
Questo è importante perché, come probabilmente molti di voi sanno, le reti neurali e questo tipo di algebra lineare sfruttano davvero le GPU, e voglio mostrarvi che con JAX, è davvero facile connettersi. Okay, quindi connettiamoci. uh sì penso che di default uh dovrebbe solo dovrebbe vedere la vostra gpu se l'avete installato con.

`01:17:31` 
il supporto gpu e automaticamente dovrebbe vedere la gpu e usarla dopo vi dico come controllare quali dispositivi jax sta effettivamente usando okay quindi prima di tutto le buone notizie quindi possiamo importare jax e importiamo jax inoltre ha un'api che è molto molto simile a numpy che è chiamata jax dot numpy e di solito selezioniamo questo chiamandolo jnp jax numpy poi importiamo numpy.

`01:18:11` 
E poi le cattive notizie sono che c'è anche questa API LAX, che è di livello molto più basso, e questa è molto più complessa. Vi mostrerò alcuni punti di questo, e questo è anche ciò che rende, fa JAX così potente. Ha un'API di basso livello, che è molto, molto veloce e un po' complessa. Useremo il 90% del tempo, diciamo il 95% del tempo, questo jnp. Tuttavia, ora vi mostrerò anche cosa rende JAX così potente nell'API di basso livello. Infine, abbiamo Matplotlib prima di fare i plot.

`01:18:52` 
Quindi, iniziamo dalle cose semplici e facendo un plot. Quindi, possiamo usare JaxNumPy per avere un linspace tra 0 e 10 con 1000 punti e possiamo creare una funzione che è il prodotto di un seno per un coseno. E ora tutto quello che stiamo usando non è un array NumPy ma un array Jax. Sì, non ho provato questo. Okay? Quindi, è molto, molto facile usare questo tipo di API perché è esattamente lo stesso di NumPy.

`01:19:27` 
Tuttavia, se controlliamo questo x jnp, vedrete che è molto simile a un array NumPy. Tuttavia, qui ora è solo chiamato array. Okay, quindi ora iniziamo con qualcosa che è un po' più peculiare e diverso da NumPy. Quindi gli array JAX sono immutabili. Cosa significa questo? Significa che non potete cambiare un array JAX. Invece, potete cambiare un array NumPy.

`01:20:10` 
Quindi per esempio, okay, iniziamo creando un array NumPy di dimensione 10 e poi all'indice 10 mettiamo un valore di, scusate, all'indice 0 mettiamo un valore di 23. Quindi quello che abbiamo fatto è che avevamo un array NumPy, abbiamo acceduto all'elemento 0 e abbiamo cambiato il suo valore. In JAX, tutto è immutabile. Significa che non potete prendere un array e assegnare un valore a un certo punto.

`01:20:45` 
Questo sembra molto strano. Tuttavia, è una scelta molto ben pensata che avrà senso nel seguito. Per ora, sappiate che è impossibile cambiare gli array Jax. Invece, quello che potete fare è il seguente. Questo può sembrare dispendioso, tuttavia, questo ha effettivamente un senso. Quindi, per esempio, quello che possiamo fare è,

`01:21:17` 
okay, inizializziamo una matrice con Jax, una matrice di dimensione tre per tre. E poi quello che possiamo dire, è che alla riga numero uno, impostiamo il valore uguale a uno. E quello che restituiamo è un altro array Jax, che è diverso dall'originale che è stato costruito con questa proprietà. Okay, quindi questo era l'originale, una matrice di 3 per 3, chiamiamo zeri. Poi abbiamo preso la prima riga e l'abbiamo impostata uguale a 1. Non potevamo cambiarla in place, e usando la sintassi, JAX vi dà una nuova matrice copiando l'originale e applicando questa operazione.

`01:22:12` 
Sulla carta, questo è dispendioso, e se lo usate così com'è, è dispendioso. Tuttavia, vedrete che JAX poi ha diversi meccanismi che sono in atto per rendere questo non dispendioso. L'espressività di NumPy è qui, nel senso che tutte le operazioni che abbiamo fatto con NumPy possono essere fatte anche qui. Quindi potete usare lo slicing con le colonne su entrambe le righe e colonne come abbiamo fatto con NumPy.

`01:22:49` 
Quindi praticamente, potete fare tutto quello che avete fatto con NumPy, anche con JAX. Tenete a mente che dovete sempre fare copie. E quindi a volte dovete usare questi metodi, che sono un po' strani, dove non potete cambiare l'array in place, ma dovete usare qualcosa che è un po' diverso. Okay, per esempio, qui stiamo impostando a 7 la prima e l'ultima riga, solo colonne da 1 in poi.

`01:23:25` 
Okay, siamo chiari per ora? Perfetto. Un altro fatto strano è che JAX gestisce i numeri casuali diversamente da NumPy. Quindi, in NumPy, quello che fate è che avete il vostro np.random.seed, e impostate il seed qui, e siete a posto. Tuttavia, in JAX, le cose sono diverse, nel senso che quello che fate è che dovete chiamare questa funzione, che vi restituisce una chiave, dato un certo seed, e ogni volta che chiamate un numero casuale, dovete passare questa chiave.

`01:24:16` 
Okay, quindi un passo extra. In particolare, qui dite, okay, seed è zero, creo una chiave con seed zero, e quando chiamo il generatore casuale, passo la dimensione. che voglio creare e e questo è il risultato okay questa è un'altra cosa strana, ma a un certo punto penso che tutto avrà senso quindi abbiate solo pazienza con me per qualche.

`01:24:49` 
minuto infine questo è un fatto molto bello ed è che jax è agnostico dell'acceleratore nel senso che gira senza alcun problema particolare di solito su cpu gpu e tpu dovete solo essere un po' consapevoli di dove sono i vostri dati perché spostare i vostri dati dalla cpu alla gpu può essere costoso.

`01:25:22` 
okay e ora stiamo vedendo esattamente questo. Quindi, prima di tutto, quello che facciamo è che, okay, creiamo con JAX il nostro array x, e questo, di default, è sulla GPU. Poi, usando la funzione dot, vogliamo calcolare il prodotto scalare di x con la sua trasposta.

`01:25:54` 
E quindi, quello che stiamo facendo qui è che, di default, questo array è già sulla GPU, e JAX fa tutti i calcoli direttamente sulla GPU. L'altra possibilità è fare tutto sulla CPU, e qui facciamo questo usando NumPy, perché NumPy non può usare la vostra GPU. E quindi, quello che stiamo facendo è l'esatta stessa operazione, ma sulla CPU, usando l'array che è sulla CPU, perché è creato con NumPy.

`01:26:27` 
Qui la terza opzione è spostare l'array che era sulla CPU sulla GPU e questo è fatto implicitamente quindi dovete stare attenti qui perché questo resta sulla CPU. Chiamate la funzione JAX e quindi implicitamente JAX sta prendendo la memoria che è sulla CPU e deve copiarla sulla GPU e questo ha un overhead. Infine quello che potete fare è anche esplicitamente spostare questo indietro alla CPU se volete.

`01:27:03` 
Il primo modo è che all'inizio forzate questi dati CPU a essere spostati sulla GPU in questo modo con device put e ora potete usare, il dot product sui dati che sono stati esplicitamente spostati sulla GPU. Quindi questo è esattamente lo stesso di uno, ma con la forzatura dello spostamento esplicito della memoria e non quello implicito.

`01:27:39` 
Quindi questi sono i risultati. Tutto sulla GPU, ci vogliono 15 millisecondi. Tutto sulla CPU, ci vuole più di 10 volte di più. Quindi vedete che con l'hardware che avete, se usate GPU sulla GPU, ci vuole una quantità molto diversa di tempo. Qui vedete che lo spostamento dei dati è in realtà non banale, nel senso che se non spostate i dati correttamente, il vostro costo computazionale raddoppia.

`01:28:16` 
Invece di 15 millisecondi, avete 35 millisecondi. E infine, se usate il trasferimento esplicito dei dati, questo è più o meno equivalente a uno. Questo ci mette due millisecondi in più, ma è più o meno equivalente. Okay? Quindi messaggio da portare a casa, state attenti a dove vive la vostra memoria, perché spostarla può essere costoso. Okay. Ora, introduciamo un nuovo componente. Questo componente è chiamato JIT, just in time compiled. Quello che fa è che sotto il cofano, fa un po' di magia nera così che la vostra funzione è ottimizzata. In particolare,

`01:29:18` 
quello che fa è che controlla cosa fa la vostra funzione la prima volta che viene eseguita e poi vi dà un modo ottimizzato di eseguire questa funzione quindi facciamo un esempio qui abbiamo una funzione per visualizzare una funzione fn in un certo range con un certo numero di punti poi definiamo funzioni qui questa è una selu che è una funzione molto ben nota nelle reti neurali che è definita in questo modo che è un po' complessa e poi quello che possiamo fare è che data una.

`01:29:58` 
funzione possiamo chiamare jit su questa funzione e ci restituisce una funzione che è ottimizzata okay, quindi questa è una sintassi molto particolare perché è una funzione che prende in input una funzione e dà come output una funzione. Di solito, abbiamo funzioni che prendono come input un numero e danno come output un numero. In questo caso, abbiamo una funzione che prende come input una funzione e restituisce una funzione. Questa funzione può essere usata esattamente come qualsiasi altra funzione.

`01:30:32` 
E in particolare, quello che possiamo fare è cronometrare il tempo di esecuzione della funzione normale e quella compilata. Quindi, questa è un'attivazione SLU. Quindi, vedete che è più o meno orizzontale prima di zero e poi è lineare dopo zero.

`01:31:03` 
Questo è un comportamento classico della funzione di attivazione. Tuttavia, se vedete il costo computazionale, la versione compilata della funzione è più o meno 10 volte più veloce di quella che non è compilata. Questo è molto importante perché questo vi dà la possibilità anche di evitare calcoli dispendiosi. Quindi prima abbiamo visto che gli array erano immutabili. Se compilate just in time questo tipo di espressione dove dovete avere una copia, JAX è abbastanza intelligente da sapere che in realtà non ha bisogno di fare copie e in realtà cambia l'array in place.

`01:31:49` 
Ma può fare questo solo dopo che avete compilato la funzione, non prima. Okay? Qualche domanda fino a qui? Perché altrimenti passiamo alla parte del gradiente, che è un'altra cosa. Quindi voglio che tutto sia chiaro fino ad ora. Okay. Qualche domanda? Okay. Quindi ora introduciamo davvero la differenziazione automatica. Vi ho detto che JAX ora è una libreria di differenziazione automatica. E quindi ora vi mostro come usare effettivamente la differenziazione automatica.

`01:32:29` 
Quindi la funzione principale qui è JAX.grad. Come la compilazione just-in-time, è una funzione che riceve come input una funzione e dà come output una funzione, che è il gradiente della funzione che avete dato come input. Okay. Quindi se avete f e chiamate grad di f, avete il gradiente di f rispetto al suo input. Quale input potete specificare rispetto a quale input con questo argomento opzionale argnums.

`01:33:10` 
Okay, quindi facciamo un esempio. Abbiamo x, un punto, poi abbiamo una funzione, questa è una parabola, la visualizziamo, e quello che possiamo fare è calcolare il gradiente, in questo caso è una derivata 1D di f rispetto a x, e comporla di nuovo, quindi possiamo calcolare la seconda derivata e la terza derivata, di questa funzione, okay?

`01:33:46` 
Quindi una delle cose che rende JAX molto più potente di, per esempio, PyTorch è che potete applicare ricorsivamente al gradiente senza molte preoccupazioni è così potente che potete comporre ancora e ancora. Gradiente. Di solito funziona abbastanza bene e velocemente. Qui, dato che prende solo un argomento, sa che deve differenziare rispetto solo al primo argomento. Quindi stiamo facendo una derivata rispetto a x. Potrebbe succedere che possiate avere più di un input.

`01:34:25` 
Per esempio, invece di una funzione 1D, potete avere una funzione 2D, che ha due argomenti, x e y. Quindi in questo caso, avete altra roba come. Il gradiente rispetto a ciascuna di queste componenti, e poi calcolate il gradiente del gradiente che abbiamo l'Hessiano non è più solo uno scalare, ma ora abbiamo una matrice perché abbiamo tutte le derivate parziali, la derivata di questa funzione rispetto a x e y, la derivata di questa funzione rispetto a x due volte, e così via.

`01:35:02` 
Quindi per fare questo, invece di usare gradient, qui usate la funzione per calcolare lo Jacobiano, okay, perché abbiamo più di un input, e quindi invece di calcolare un gradiente per una funzione scalare, ora stiamo calcolando lo Jacobiano per una funzione che ha valori vettoriali. Avete effettivamente due funzioni diverse, JacRev e JacForward.

`01:35:35` 
Queste sono diverse perché sono implementate in due modi diversi, una usa la differenziazione automatica in modalità forward e una usa la differenziazione automatica in modalità reverse. Quello che dovete sapere è che una è più efficiente per matrici alte e una è più efficiente per matrici larghe. Quindi di solito per come lo Jacobiano è calcolato, quello che volete fare è che volete usare prima il reverse e poi il forward.

`01:36:15` 
Questo è il modo più efficiente per calcolare gli Hessiani. Quindi per esempio qui quello che stiamo facendo è che, okay, abbiamo f, calcoliamo lo Jacobiano, rispetto al primo e secondo input. Qui. abbiamo una funzione vettoriale e poi calcoliamo di nuovo lo Jacobiano rispetto al primo e secondo output e questo è il risultato. Quindi invece di avere un valore abbiamo o un vettore che.

`01:36:49` 
è lo Jacobiano o l'Hessiano che è la matrice. Il messaggio da portare a casa qui è se la vostra funzione ha più di un input dovete usare la funzione che invece di calcolare il gradiente calcola lo Jacobiano e se volete l'Hessiano come regola per ragioni di implementazione dovete usare prima la versione reverse e poi quella forward se volete spremere le massime prestazioni dal vostro codice.

`01:37:27` 
Naturalmente, potete passare come parametri a una funzione vettori, quindi per esempio qui x non è più uno scalare, ma è un array Jax, quindi potete effettivamente calcolare lo Jacobiano e l'Hessiano rispetto solo al primo input, che in questo caso non è più uno scalare, ma è un array di dimensioni arbitrarie. E i risultati sono esattamente gli stessi di prima. Questi sono solo due modi diversi di scrivere la stessa cosa. Invece di avere parametri diversi nella funzione, ne avete solo uno, che tuttavia è un array, invece di essere due scalari.

`01:38:26` 
Infine, per esempio, possiamo anche controllare manualmente cosa succede se abbiamo una funzione che non è differenziabile. Quindi abbiamo il valore assoluto, e sapete che in 0, questa funzione non è differenziabile. JAX è molto pragmatico, e non usa direttamente qualcosa che non è utilizzabile, ma prende alcune decisioni. E in particolare, qui, possiamo testarlo. Se non siete sicuri di cosa fa il codice, quello che potete fare è solo scrivere un esempio semplice. Quindi abbiamo la nostra funzione, lambda, il valore assoluto di x, e calcoliamo il gradiente di f rispetto a x.

`01:39:06` 
E poi lo valutiamo in 0, molto piccolo, in un punto molto vicino a 0, ma maggiore di 0, e in un punto che è, diciamo, molto vicino a 0, ma più piccolo di 0. Quindi potete vedere che in questi due punti abbiamo il risultato corretto che è meno uno e uno, e in zero, dove la funzione è effettivamente non differenziabile, JAX prende la decisione di dire okay è uno. Questo è per evitare la propagazione di.

`01:39:41` 
infiniti o not-a-number nel vostro codice. Il terzo concetto molto importante di JAX che introduciamo oggi è VMAP. Quindi se dovete portare qualcosa a casa da questa lezione su JAX, i tre concetti più importanti sono la compilazione just-in-time,

`01:40:13` 
il gradiente con lo Jacobiano e il VMAP. Questi sono ciò che rende JAX molto potente. Il VMAP è un modo per vettorizzare il codice. Quindi vi ho parlato molte volte di quanto sia importante scrivere codice vettorizzato per evitare i for loop. VMAP è qualcosa che trasforma il vostro for loop in qualcosa che è molto vicino.

`01:40:46` 
a codice compilato. Quindi potete trasformare un for loop che è inefficiente in un for loop che è efficiente, proprio come se fosse codice vettorizzato. E qui vi mostro un esempio. Quindi prima di tutto definisco una funzione strana di cui non mi interessa cosa sia. Questo è un prodotto scalare personalizzato dove faccio il prodotto scalare dei due vettori e poi elevo al quadrato il valore. Questo non ha un significato particolare, è solo una funzione.

`01:41:21` 
Poi pensate che x e y siano due matrici, e voglio applicare questa funzione a ogni riga di queste due matrici. Questo è qualcosa che succede abbastanza spesso nelle reti neurali, perché quello che fate è che avete un dataset, ogni punto nel dataset è una riga, e poi applicate una funzione a quel punto nel dataset. Quindi se facciamo questo in modo ingenuo, quello che possiamo fare è che facciamo un loop su ogni riga. Quindi in questa sintassi, potete pensare che v1 e v2 siano due campioni diversi, vettore 1 e vettore 2,

`01:41:57` 
e poi vogliamo applicare la nostra funzione a ogni riga di queste due matrici. Quindi vogliamo applicare una sorta di funzione a ogni riga di due matrici diverse, iterando su ogni riga. E questo sarà lento, perché stiamo usando un loop Python. La prima idea che potete avere è okay, facciamo solo la compilazione just in time. Quindi abbiamo visto che la compilazione just in time può fare meraviglie. E quindi quello che possiamo fare è che, okay, abbiamo questa funzione, e la compiliamo. Quindi questa sintassi con il simbolo at che vedete sopra la funzione è un particolare pezzo di codice chiamato decorator. E quello che fa è l'esatta stessa cosa di dire che.

`01:43:02` 
è uguale a jax.jit di questo. Quindi questo pezzo di codice e quello che vedete sopra sono gli stessi. Questa sintassi significa solo che stiamo applicando questa funzione a questa funzione. Quindi quello che stiamo facendo è che stiamo compilando just-in-time questa funzione, e questo è già.

`01:43:35` 
qualcosa di buono. Quindi vedrete che la versione compilata just-in-time sarà meglio di quella ingenua. Tuttavia, possiamo fare anche meglio. Possiamo chiamare vmap, vmap è come un for loop implicito. Quindi quello che fate è che a vmap passate la funzione che volete applicare. Aggiungete alcuni argomenti extra che dicono in quale direzione volete applicare questa funzione.

`01:44:11` 
e in particolare qui in axis zero zero significa che volete applicare questa funzione in modo riga per riga su entrambe le matrici e automaticamente vmap fa un for loop nelle direzioni indicate, da questa variabile okay quindi state applicando vmap data una funzione e un modo in cui applicate la funzione vi restituisce un modo ottimizzato per applicare questa funzione ai vostri argomenti.

`01:44:52` 
il modo finale e migliore di fare questo è usare sia vmap che la compilazione just-in-time se, unite questi due vedrete i migliori risultati e in particolare qui prima applicate il for loop ottimizzato e poi state anche compilandolo e questo vi darà le migliori prestazioni. Okay, quindi controlliamo la documentazione. Quindi questi sono una sequenza di interi che sono uguali al numero di argomenti che avete.

`01:45:33` 
che dice a questa funzione su quale asse dell'input state iterando. Quindi abbiamo due matrici come input, e vogliamo applicare questa funzione su base riga per riga per entrambe le matrici. Quindi in questo modo stiamo dicendo che vogliamo applicare questa funzione sul primo asse, che è l'asse numero zero, che è quello delle righe. Quindi significa che vogliamo applicare.

`01:46:04` 
la funzione che stiamo passando a vmap, che è il nostro prodotto scalare personalizzato, sul primo asse del primo argomento e il primo asse del secondo argomento. Quindi significa che stiamo iterando su ogni riga di entrambi i modelli. Questo è molto simile a quando abbiamo fatto il numpy min e dovevamo dire su quale asse vogliamo fare, applicare il min sull'array. È esattamente la stessa cosa, e questo è un modo per estendere quello che è stato fatto solo per due funzioni, come il min.

`01:46:36` 
o la sum in numpy, a tutte le funzioni. Questi sono i risultati. Vedrete che la differenza è abbastanza impressionante nel senso che nel modo ingenuo con il for loop, ci vuole mezzo secondo. Se fate il modo vettorizzato, quello con il VMAP, questo è quasi 500 volte più veloce.

`01:47:07` 
Se compilate just in time senza vettorizzazione, è più o meno paragonabile a quello vettorizzato. È un po' meno di un millisecondo. Se usate vettorizzazione e compilazione just in time, è altre 10 volte più veloce. In totale, è come 5.000 volte più veloce. Questo è ciò che rende JAX così potente. Potete usare questa funzione per rendere le vostre funzioni semplici molto potenti, molto, molto veloci.

`01:47:50` 
Okay, qualche domanda? Okay, quindi questi erano gli argomenti principali di cui voglio parlare, su JAX. Questa è l'interfaccia di alto livello. Ora stiamo andando nell'interfaccia di livello più basso e cercando di capire un po' meglio le parti strane, come il fatto che non potete cambiare un array, il fatto che la generazione di numeri casuali.

`01:48:24` 
è un po' particolare. Okay? Okay, quindi davvero la parte importante sono le funzioni JIT, vmap, e grad. Questo è davvero quello che voglio che sappiate molto bene. Ora andiamo nei dettagli, come si dice in inglese, e cerchiamo di capire un po' meglio come funziona JAX dietro le quinte. Quindi, l'interfaccia NumPy di alto livello, il jnp-add, è molto flessibile.

`01:48:58` 
Per esempio, qui potete passare questo, che è un intero, e questo, che è un numero in virgola mobile, e JAX li aggiungerà insieme volentieri. Tuttavia, se usate il JAX di basso livello, l'interfaccia LAX, le cose sono più difficili. Per esempio, potete aggiungere insieme solo numeri che hanno lo stesso tipo. Quindi, qui potete aggiungere 1 e 1, dove entrambi sono numeri in virgola mobile, ma se aggiungete un intero e un numero in virgola mobile,

`01:49:31` 
l'interfaccia JAX di basso livello vi darà un errore, perché richiede promozione esplicita del tipo. In particolare, qui c'è l'errore. LAX-add richiede che gli argomenti abbiano gli stessi tipi, int32 e float32. Questo è importante perché altrimenti quello che state facendo è un cast implicito, il che significa che state trasformando un numero da un tipo a un altro, e questa operazione può essere costosa.

`01:50:04` 
E se state facendo cose a basso livello, è importante avere controllo su ogni piccola parte, che può avere un costo sul risultato finale. Infatti, l'interfaccia di basso livello è più potente, ma è meno user-friendly. Quindi prendete per esempio questa funzione che fa una convoluzione, una convoluzione generale, proprio come in NumPy, passate solo x e y e basta, vi dà una convoluzione.

`01:50:45` 
Invece, se andate all'interfaccia di basso livello, quello che dovete fare è che dovete avere che entrambi i tipi abbiano lo stesso, abbiano i tipi corretti, quindi numeri in virgola mobile, e poi dovete aggiungere argomenti extra per assicurarvi che tutto sia fatto correttamente. Per esempio, la dimensione della finestra della convoluzione, e il padding che abbiamo all'inizio e alla fine. Cosa significa questo? Significa che potete fare cose in modo più potente, nel senso che avete più opzioni, ma dovete anche leggere molta documentazione.

`01:51:30` 
In questo, il risultato è esattamente lo stesso, perché l'interfaccia JAX di alto livello è solo un wrapper per l'interfaccia JAX di basso livello. Okay. Ora andiamo a cose che sono importanti riguardo a JIT e sono un po' strane. Quindi la compilazione just-in-time è magia nera, ma ha molte limitazioni. Per esempio, qui abbiamo questa funzione, che dato x vi restituisce un array dove tutti, con tutti gli elementi negativi di x.

`01:52:13` 
Quindi questa è una maschera booleana con vero o falso a seconda se x è più piccolo di zero. E quindi in questo caso, quello che stiamo facendo è che stiamo estraendo da x tutti i valori che sono più piccoli di zero, che è qualcosa che abbiamo fatto anche oggi. Tuttavia, non potete compilare just-in-time questa funzione. Jax vi darà un errore. Perché?

`01:52:44` 
La ragione è che la dimensione del risultato di questa operazione dipende da x stesso. Se x è meno 1, meno 1, il risultato ha dimensione 2. Se x è 1, 1, la dimensione e il risultato è 0. Dato che x ha una forma dinamica, JAX non può compilare questo codice just in time. Questa è una limitazione di JAX, e dobbiamo venire a un accordo con questo.

`01:53:18` 
Questo è come è fatto JAX, e dovete sapere questo. Quindi, la maggior parte dei problemi che avrete con JIT è con cose che sono dinamiche. Potete usare JIT solo con codice che è in un certo senso statico, che ha risultati che non cambiano in forma e tipo, perché JAX deve sapere queste cose per ottimizzarlo. Controlliamo un po' più in dettaglio come funziona la compilazione just-in-time per cercare di capire cosa sta succedendo.

`01:53:55` 
Quindi abbiamo questa funzione f, e poi quello che facciamo è che aggiungiamo alcune stampe alcune istruzioni print per capire cosa è stato passato alla funzione. Quindi abbiamo print, funzione f, print x, print y. Facciamo qualche operazione, che è il prodotto scalare, e poi stampiamo il risultato. Creiamo due vettori casuali, e chiamiamo la funzione due volte, con il primo argomento e il secondo.

`01:54:30` 
Quindi quello che notiamo qui, due cose importanti. La prima è che x e y non sono array qui. Quando stiamo eseguendo questo codice, x non è un array e y non è un array, ma stiamo chiamando questa funzione con argomenti array. Perché? Perché JAX, per compilare la funzione, passa strutture dati molto speciali, che sono chiamate JIT tracers, la prima volta che la funzione è chiamata.

`01:55:01` 
E in questo modo, è in grado di capire come funziona la funzione e ottimizzarla. Quindi la prima volta che chiamate questa funzione, non è effettivamente con i vettori, ma abbiamo strutture dati molto speciali che sono chiamate dietro le quinte che capiscono il flusso di dati del... Il secondo punto qui è che durante la seconda chiamata, non vedete più questi JIT tracers, perché la compilazione just-in-time funziona in due passi.

`01:55:35` 
Nel primo passo, tracciate. Tracciate la funzione, e questo è fatto solo una volta. la seconda volta la funzione è stata tracciata è compilata è ottimizzata e tutti gli effetti collaterali come il print sono rimossi mantenete solo ciò che è strettamente necessario e ottimizzato okay quindi il messaggio da portare a casa qui è che la prima volta che chiamate jit è costoso ed è chiamato con speciali.

`01:56:09` 
strutture dati che lo rendono possibile ottimizzare poi dalla seconda chiamata in poi è veloce ed è salvato in qualche parte della memoria potete controllare cosa sta facendo JAX usando questa funzione make jax expression, Questa è una funzione che, data una funzione, vi restituisce la grammatica astratta che JAX.

`01:56:40` 
sta usando per creare la funzione ottimizzata. Quindi, per esempio, qui vedete che, data questa funzione, JAX passa i tracers e costruisce questo oggetto. In questo oggetto quello che avete è il tipo che state usando, quindi questo è un floating point a 32 bit tipo di dimensione tre per quattro, e b è un array di dimensione quattro. Quindi qui vedete che JAX, per ottimizzare il codice,

`01:57:12` 
deve conoscere la dimensione, che è tre per quattro e quattro, e il tipo dell'array, e questo è il motivo per cui non potete cambiare il tipo dell'array, perché altrimenti non potete creare questa grammatica astratta, che JAX ha bisogno per ottimizzare la funzione. E qui in parole, avete tutta l'API di basso livello che JAX chiama con questi argomenti. Quindi avete una variabile a, una variabile b, una variabile c, una variabile d, una variabile a. E vedete che, per esempio, la variabile c, che ha questo tipo e questa dimensione, è uguale alla funzione add dell'interfaccia JAX di basso livello chiamata sulla variabile a.

`01:57:55` 
E secondo argomento, 1.0, che è un array a 32 bit con dimensione 0. Okay? Quindi questo è come JAX sta facendo questa compilazione just-in-time. Sta creando una grammatica astratta per tradurre la vostra funzione nella sua API di basso livello. E poi la sta salvando. Tuttavia, dovete sapere molto bene, dato che questo viene dall'API di basso livello, quali sono i tipi e quali sono le dimensioni, perché altrimenti l'API LAX non funziona.

`01:58:30` 
La rigidità dell'API di basso livello è ciò che la rende così veloce, perché conoscendo esattamente dimensioni e tipi, può ottimizzare un po' le cose. Altri problemi saranno solo nella compilazione in tempo. Quindi, prima abbiamo visto un problema dove la dimensione di x cambia, ora quello che cambia è il tipo dell'argomento. Per esempio, qui abbiamo una funzione f, che dato un vettore x e un booleano, che può essere o vero o falso, restituisce o meno x o x.

`01:59:18` 
Qui, la funzione vi dà un errore. Dà solo un errore quando cercate di compilare just in time questa funzione. Perché? Perché qui state passando, scusate, qui, quando state chiamando questo, qui state passando.

`01:59:48` 
questo valore, che cambia la struttura e la grammatica di questa funzione, a seconda del suo valore. Quindi, neg può essere vero o falso, e a seconda se è vero o falso, andate in un ramo o un altro dell'input. Questo significa che non potete creare questo tipo di grammatica quando il vostro input cambia, e quindi solleva un errore.

`02:00:19` 
Tuttavia, ci sono soluzioni intelligenti, per esempio, potete usare quest'altra versione di questa funzione dove non ci sono clausole if, quindi per esempio, qui neg è ancora un booleano, ma non usate istruzioni if, invece qui sfruttate il fatto che vero o falso è in realtà uno o zero, e fate tutto usando moltiplicazione e sottrazione. Per esempio, se qui avete zero, moltiplicate zero per meno due, e poi avete uno. Invece, se qui abbiamo due, che è uno, qui abbiamo due, e poi avete uno meno due, che è meno uno.

`02:01:13` 
E quindi questo è un modo per aggirare questo problema. Perché qui non state cambiando la struttura if-else della vostra funzione, e quindi queste possono essere compilate just in time. Un altro modo per aggirare questo problema è usare argomenti statici. Non mi piace questo tipo di soluzioni, tuttavia dovete sapere che esistono.

`02:01:47` 
Quindi quello che potete fare è usare questo decorator chiamato partial, e quello che sta facendo è che sta creando una nuova funzione per ogni argomento che state passando qui. Quindi quello che questo tipo di soluzione sta facendo è creare una nuova funzione per ogni tipo di argomento che passate come secondo argomento. Quindi qui questo sarebbe equivalente ad avere due funzioni, ftrue e ffalse.

`02:02:18` 
E poi compilare, just in time, entrambe le funzioni, la ftrue e ffalse. E questo è fatto automaticamente con partial. In particolare qui, a static arguments, stiamo passando l'id dell'argomento, quindi il secondo, quello con id 1, che deve essere trattato come una funzione diversa. Quindi per esempio qui, vedete che ogni volta che il secondo argomento cambia,

`02:02:50` 
abbiamo un nuovo jit tracer, perché ogni volta che questo argomento cambia, stiamo creando una nuova funzione, e quindi il tracer deve essere eseguito ancora una volta. quindi dobbiamo eseguire il tracer qui la prima volta che chiamiamo true dobbiamo chiamare il tracer qui la prima volta che chiamiamo false e dobbiamo eseguire il tracer qui la prima volta che chiamiamo questa funzione con uno perché uno è tipo intero che è diverso dal tipo di true che è.

`02:03:20` 
di tipo booleano quindi ogni volta che chiamate questa funzione con un argomento diverso dovete ricompilarla just in time questo è il motivo per cui non mi piace perché nasconde dietro il trucco il fatto che state effettivamente compilando just in time ancora e ancora e la prima volta che eseguite il tracer è effettivamente costoso okay stiamo arrivando in fondo a questo e.

`02:03:55` 
un altro punto molto importante è che jax è progettato per funzionare solo con funzioni pure, cosa sono le funzioni pure le funzioni pure sono funzioni dove tutti i dati di input sono passati attraverso parametri e restituiscono sempre lo stesso valore se invocate con gli stessi input questo può sembrare più strano ma è davvero semplice in pratica quindi abbiamo una variabile globale g.

`02:04:25` 
uguale a zero possiamo definire questa funzione che somma x e g questa non è una funzione pura ma perché stiamo usando g che è un parametro che sta che sta al di fuori della funzione e quindi questo è impuro perché stiamo catturando una variabile che è al di fuori dello scope di questa funzione, per creare il suo risultato questo è importante perché quando fate la compilazione just-in-time.

`02:05:01` 
Effettivamente congelate il valore della variabile esterna. Okay, quindi quello che sta succedendo è che stiamo prendendo questa funzione, la stiamo compilando just in time, e chiamandola con parametro 4. Quindi quello che stiamo facendo qui è che poi stiamo facendo 4 più 0, che è 4. Poi aggiorniamo g, che è la variabile che rende la funzione impura, e se chiamiamo questo di nuovo, anche compilandolo di nuovo,

`02:05:38` 
il risultato non è 15, ma è 5, perché ha memorizzato in cache il valore della variabile esterna, e quindi vi sta dando il risultato sbagliato. Okay, quindi questa è un'altra parte a cui dovete stare davvero attenti. Quando compilate just-in-time una funzione, tutto quello che usate dentro la funzione deve essere passato come argomento.

`02:06:09` 
Altrimenti, la compilazione just-in-time lo ottimizzerà via e salverà il suo valore una volta per tutte. Questo è qualcosa che abbiamo fatto molte volte durante gli esercizi passati. Se volete, potete tornare indietro nel lab precedente e controllare ogni volta che abbiamo fatto qualcosa del genere. E se compilate just-in-time la funzione che abbiamo scritto, può benissimo essere che non funzionino correttamente a causa di questa esatta ragione.

`02:06:44` 
Se state cambiando quello che state passando alla funzione, quindi forse cambia la dimensione e dovete ritracciarla, allora e solo allora potete catturare di nuovo. la variabile globale perché state forzando il tracer a passare attraverso tutti i passi di nuovo. State davvero attenti che una volta che fate la compilazione just-in-time e fate questo tipo di cose,

`02:07:19` 
JAX non vi dà a volte errori ragionevoli nel senso che l'indicizzazione fuori dai limiti, a volte non funziona correttamente. Per esempio qui abbiamo un array con 10 elementi e all'11esimo stiamo aggiungendo 23. 11 è fuori da questo range e quindi il suo tipo non sta facendo nulla ma non vi sta dando un errore.

`02:07:50` 
Quindi, un po' come in C++, a volte potete accedere a memoria che non è vostra e non vi dice nulla. Infine, lasciatemi parlare un po' di più dei numeri casuali. Questo è molto importante e dovrebbe chiudere tutto riguardo alla compilazione just-in-time e alle funzioni pure. Quindi, torniamo un po' indietro a NumPy e capiamo un po' meglio come funziona il seeding.

`02:08:23` 
Quindi, impostiamo il seed a zero, poi chiamiamo NumPy random seed e impostiamo il seed. Quello che NumPy sta facendo dietro le quinte è che ha uno stato, che è un numero, che vi dice a che punto del generatore di numeri casuali siete. Quindi, il modo che NumPy usa... per generare nuovi numeri casuali è che ogni volta che chiamate una funzione che fa qualcosa a caso.

`02:08:57` 
aggiorna questo stato che sta dietro le quinte per generare nuovi numeri casuali e potete ottenere questo stato con la funzione get state quindi per esempio all'inizio quando impostiamo il seed abbiamo questo stato poi generiamo un numero casuale 0.2054 e potete vedere che lo stato è stato cambiato quindi ogni volta che chiedete un nuovo numero casuale dietro le quinte numpy cambia.

`02:09:31` 
una variabile che è lo stato per fornirvi nuovi numeri casuali qual è il problema, qui è che queste sono funzioni impure è esattamente quello che stava succedendo qui con la g la g è come lo stato esterno del generatore di numeri casuali e quindi per generare numeri casuali sta usando funzioni impure e quindi Jax non può usare questo tipo di comportamento questo tipo di implementazione per implementare la sua casualità perché Jax si basa sul fatto che tutto è basato sulla casualità.

`02:10:18` 
E quindi non avrebbe senso implementare la casualità con funzioni impure e questo è il motivo per cui avete quella sintassi molto strana dove dovete recuperare una chiave quella chiave che stavamo usando prima è lo stato del generatore di numeri casuali e dovete aggiornarla manualmente perché dobbiamo trattarla in modo puro. e non in modo impuro dove lo stato è questa è l'idea infatti questo è esattamente lo stesso di prima.

`02:10:54` 
ma nel modo di jax quindi quello che facciamo è che abbiamo il nostro seed otteniamo la nostra chiave che è la stessa lo stato del generatore di numeri casuali poi chiamiamo jax random normal con lo stato, e la forma che vogliamo avere e poi dobbiamo aggiornare la chiave se facciamo se vogliamo un numero diverso perché se manteniamo la stessa chiave abbiamo lo stesso numero casuale di prima.

`02:11:29` 
come otteniamo una nuova chiave significando aggiorniamo lo stato usiamo la funzione jax random split il jax random split aggiorna lo stato casuale dandovi. due nuovi stati casuali, quello vecchio e quello nuovo.

`02:12:06` 
Okay, quindi questo è il modo per generare nuovi numeri casuali. Dobbiamo aggiornare manualmente lo stato che in NumPy era fatto implicitamente e dietro le quinte, perché in NumPy tutto è fatto con, scusate, in JAX tutto è fatto con funzioni pure. C'è un'altra ragione molto importante per cui la casualità è implementata in questo modo. Con il modo NumPy, non è possibile avere un codice facilmente riproducibile in un ambiente parallelo perché se state addestrando il vostro modello di machine learning su due GPU diverse, potreste avere le condizioni di rischio nell'aggiornare lo stato casuale in un modo che non è facilmente riproducibile.

`02:13:00` 
Perché le due GPU hanno probabilmente carichi computazionali diversi e in tempi diversi potrebbero chiedere allo stato unico che sta nella CPU di essere aggiornato. E qui c'è un esempio del codice per mostrarvi che effettivamente questo è il modo. Quindi abbiamo due funzioni qui che chiamo workers e potete immaginare queste come due hardware diversi che stiamo usando.

`02:13:32` 
come due CPU diverse, e qui quello che stiamo facendo è che stiamo generando nuovi numeri, e qui dormo per un po' di tempo per simulare carichi di lavoro diversi. Questo è molto simile a un aggiornamento del gradiente, quindi durante la lezione forse avete visto una discesa del gradiente. L'idea è che se state aggiornando con discesa del gradiente in batch in un loop, ogni volta volete estrarre un nuovo batch a caso,

`02:14:05` 
e questo potrebbe simulare questo tipo di processo. Quindi poi quello che stiamo facendo è che stiamo inizializzando due thread diversi che chiamano questi due workers. Quindi qui è solo per simulare il fatto che abbiamo due diversi, lo stesso codice che sta girando su due GPU diverse che sono diverse. la parte importante è questa okay se eseguite il codice diverse volte potete potete ottenere risultati diversi anche se il seed è esattamente lo stesso quindi sto eseguendo questa funzione che sta facendo.

`02:14:40` 
due lavori paralleli ogni volta che inizializzo nel loop re-inizializzo il seed esattamente allo stesso punto e anche inizializzando il seed nello stesso punto ottengo risultati diversi okay questo è il motivo per cui numpy non è affidabile in ambienti paralleli perché ci sono condizioni di rischio allo stato comune invece se dividete manualmente lo stato e ogni cpu gestisce il suo stato potete avere un codice riproducibile infatti ogni volta che eseguite questa funzione potete avere risultati diversi.

`02:15:32` 
Lasciatemi vedere cosa voglio dirvi ora su questo. Okay, alcune cose in più. Se volete usare il gradiente, calcolare il gradiente della funzione, se avete if o else, tutto è a posto. Tutto funziona bene. Ma se componete insieme jit e gradient,

`02:16:05` 
se avete if else, le cose potrebbero andare male. In particolare, ricordate che con gradient, if ed else non sono un problema, ma se volete usare jit e gradient, abbiamo un codice più veloce, dovete cercare di evitare di usare if ed else, per la ragione che vi ho detto prima riguardo al tracciamento e al fatto che le cose cambiano. In questo qui, avete un errore se cercate di usare un JIT su questa funzione che ha una clausola if-else.

`02:16:42` 
La soluzione qui è che invece dell'if-else, potete usare il where. Avete visto questi negli esercizi precedenti che abbiamo compilato. Questo è un modo per vettorizzare l'if. Quindi dove l'x è più o meno uguale a 3, avete questo ramo dell'if, altrimenti avete l'altro. E questo va bene perché il risultato non cambia. Anche se qui avete una maschera booleana, il risultato è sempre della stessa dimensione perché in alcune parti degli array mettete questa espressione.

`02:17:15` 
e sull'altra parte dell'array avete l'altra. Non state solo prendendo alcuni indici. Qui state dicendo, okay, dove x è più piccolo di 3, mettete la parabola, dove x è maggiore di 3, mettete la parte lineare. Quindi, il risultato è sempre della stessa dimensione, e potete usare il JIT e il GRAD, okay? Il VMAP ha più o meno un equivalente in un'API di basso livello, che è chiamata il for loop,

`02:17:57` 
dove potete mettere l'indice iniziale, l'indice finale, la funzione che volete impiegare in questo for-loop, e un valore iniziale dove volete applicare questa funzione. Questo è un altro modo per vettorizzare i for-loop in JAX. Ultime cose finali. Se non avete i numeri intorno, ci sono opzioni.

`02:18:30` 
uh per tracciare dove non i numeri uh appaiono mettendo questo a true e sollevando errori ogni volta che avete not a number questo è molto utile quando fate avete codice complesso a un certo punto tutto è not a number non sapete da dove viene uh ultimo dito di default jax fa tutto in precisione a 32 bit perché le gpu sono ottimizzate per la precisione a 32 bit.

`02:19:02` 
quindi uh tuttavia quando usate i doppi e quando usate numpy tutto di solito è in precisione a 64 bit in particolare se venite da un background di matematica come matematico di solito usate sempre la precisione a 64 bit e potete forzare jax a usare questa precisione ma sappiate che questo è meno ottimizzato a livello hardware di solito okay. Questo è tutto. Se avete domande, sono qui. Altrimenti, buon fine settimana.
---

## Appendice: Formule Chiave Lab 4 - SVT e JAX

### Matrix Completion con SVT: Formule Pratiche

**Dataset MovieLens 100k**:
- **Utenti**: M  1000
- **Film**: D  1600
- **Rating**: ~100,000 (6.25% densit�)
- **Range**: r  {1, 2, 3, 4, 5}

**Preprocessing Pipeline**:
```
1. Load:  df = pd.read_csv('movielens.csv', sep='\t')
2. Shuffle:  df = df.iloc[np.random.permutation(len(df))]   rimuove temporal bias!
3. Compact:  rows, _ = np.unique(user_ids, return_inverse=True)   rimuove righe/colonne vuote
4. Split:  train (80%), test (20%)
5. SparseDense:  X = csr_matrix((vals, (rows, cols))).toarray()
```

**Perch� Shuffling**:
- Dataset ordinato per timestamp  bias temporale
- User registra  valuta 20 film in sequenza
- Senza shuffle: **tutti** i 20 film stesso utente in train OR test
- Con shuffle: distribuzione uniforme train+test per ogni utente

**Index Compaction**:
```
Problema: IDs non contigui  righe/colonne vuote
Esempio: user_ids = [0, 0, 3]  matrice 4D (riga 1-2 vuote!)

Soluzione: np.unique(return_inverse=True)
compact_ids, inverse_map = np.unique(user_ids, return_inverse=True)
 compact_ids = [0, 3]
 inverse_map = [0, 0, 1]  (mappa originale  compatto)

Risultato: Matrice 2D (no righe vuote)
```

**Trivial Recommender (Baseline)**:
```
Predizione: r_ij = (1/|N_i|) S_{kN_i} r_ik

dove N_i = {film valutati da utente i}

Codice NumPy:
avg_rating = np.empty(n_users)
for i in range(n_users):
    mask = (X_train[i, :] > 0)  # film valutati
    avg_rating[i] = X_train[i, mask].mean()

Predizione test: pred_trivial = avg_rating[row_test]
```

**Metriche Evaluation**:
```
1. RMSE (Root Mean Squared Error):
   RMSE = ( (1/n) S (r_true - r_pred) )
    Pi� basso = meglio
    Baseline: ~1.0

2. Pearson Correlation ?:
   ? = cov(r_true, r_pred) / (s_true  s_pred)
    Range: [-1, 1]
    ? = 1: perfetta correlazione
    ? = 0: no correlazione
    Baseline: ~0.3
```

### SVT Hard Thresholding Algorithm

**Algoritmo** (versione hard threshold):
```
Input: 
  - X_train: matrice con known values + zeros
  - t: threshold (es. 100)
  - max_iter: 200
  - tol: 1e-4

Initialize: A = X_train

For iter = 1 to max_iter:
    1. SVD: U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    2. Hard Threshold:
       s[s < t] = 0   taglia valori singolari piccoli!
    
    3. Reconstruct:
       A = U @ np.diag(s) @ Vt
    
    4. Impose Known Values:
       A[rows_train, cols_train] = vals_train
        Forza A_ij = r_ij per posizioni note!
    
    5. Check Convergence:
       delta = ||A - A_old||_F
       if delta < tol: break
    
    6. Evaluate:
       pred_test = A[rows_test, cols_test]
       RMSE = (mean((vals_test - pred_test)))
       ? = pearson_r(vals_test, pred_test)

Output: Completed matrix A
```

**Differenza Hard vs Soft Thresholding**:
| Aspetto | Hard Threshold | Soft Threshold (SVT originale) |
|---------|----------------|-------------------------------|
| Formula | s_i  0 se s_i < t | s_i  max(0, s_i - t) |
| Geometria | Taglio netto | Shrinkage graduale |
| Implementazione | s[s < t] = 0 | s = np.maximum(0, s - t) |
| Convergenza | Pi� veloce ma meno smooth | Pi� lenta ma pi� stabile |
| Uso | Esercitazione didattica | Paper originale Cai et al. |

**Convergenza SVT**:
```
Iterazione 0:   RMSE  1.5,  ?  0.1  (peggio di baseline!)
Iterazione 50:  RMSE  1.0,  ?  0.44 (supera baseline)
Iterazione 200: RMSE  0.95, ?  0.55 (best!)

Messaggio: Metodi sofisticati richiedono iterazioni,
           ma vincono su baseline semplici!
```

**Complessit� Computazionale**:
```
Per ogni iterazione:
- SVD: O(min(M,D)  max(M,D))   bottleneck!
- Threshold: O(min(M,D))
- Reconstruct: O(MDmin(M,D))
- Impose: O(nnz(train))

MovieLens 100k: ~5 sec/iter su CPU, ~200 iterazioni  15 min totale
```

### Image Inpainting con SVT

**Setup**:
```
Input: Immagine corrotta (50% pixel = noise)
Goal: Ricostruire pixel corrotti usando SVT

Rappresentazione: I  R^(HW) (grayscale)
Corruzione: P_O(I) dove O = {pixel non corrotti}

SVT applica:
- Known: pixel non corrotti
- Unknown: pixel corrotti  da ricostruire
```

**Soft Thresholding Completo (da paper)**:
```
Input:
  - X: immagine corrotta
  - P: maschera (1 = noto, 0 = corrotto)
  - t: threshold
  - d: step size (es. 1.2)
  - c0: initial dual variable scale

Initialize:
  - Y = c0  d  P(X)
  - M = 0
  - max_iter = 700

For iter = 1 to max_iter:
    1. U, s, Vt = svd(X + dM)
    
    2. Soft Threshold:
       s_new = max(0, s - t)
    
    3. M = U @ diag(s_new) @ Vt
    
    4. Residual:
       R = X - P(M)
    
    5. Update Dual:
       Y = Y + dR
    
    6. Update X:
       X = P(X) + (1-P)(M)
    
    7. Check Convergence:
       err = ||M - M_old||_F / ||M_old||_F
       if err < tol: break

Output: M (immagine ricostruita)
```

**Parametri Tipici** (da Cai et al. 2010):
```
t = 5  (max(H, W))   threshold adattivo alle dimensioni
d = 1.2                 step size (accelerazione)
c0 = min(||X||_F / ||P(X)||_F, 100)   dual initialization
```

**Risultati Mondrian Painting**:
- Input: 50% pixel corrotti (random noise)
- Output: Ricostruzione quasi perfetta dopo 500 iter
- Perch� funziona: Immagine geometrica  **rank molto basso**!

**Risultati Landscape Photo**:
- Input: 50% pixel corrotti
- Output: Ricostruzione con sfocatura, ma riconoscibile
- Limite: Foto naturali  rank pi� alto  SVT meno efficace

---

## JAX: API Moderna per Differenziazione Automatica

### Storia e Motivazione

**Timeline ML Frameworks**:
```
2015-2017: TensorFlow (Google) domina
           - Grafo statico  veloce
           - API complessa

2017-2020: PyTorch (Facebook) sorpassa
           - Grafo dinamico  flessibile
           - API intuitiva
           - Debugging facile

2020-oggi: JAX (Google) emerge
           - Best of both: veloce + flessibile
           - Pure AD library (no NN built-in)
           - Composable transformations
           - GPU/TPU first-class
```

**JAX vs PyTorch/TensorFlow**:
| Feature | JAX | PyTorch | TensorFlow 2 |
|---------|-----|---------|--------------|
| AD |  Core |  Autograd |  GradientTape |
| JIT |  @jax.jit |  torch.jit |  @tf.function |
| GPU/TPU |  Excellent |  Good |  Good |
| NN layers |  (use Flax/Haiku) |  torch.nn |  tf.keras |
| API style | NumPy-like | PyTorch-like | Keras-like |
| Immutability |  Enforced |  Mutable |  Mutable |
| Functional |  Pure functions |  OOP |  OOP |

**Keras 3.0** (2023): Supporta JAX + PyTorch + TensorFlow come backend!

### JAX Core Concepts (i 3 pilastri)

**1. JIT Compilation** (@jax.jit):
```python
import jax
import jax.numpy as jnp

# Funzione normale
def f(x):
    return jnp.dot(x, x.T)

# Funzione JIT-compilata
f_jit = jax.jit(f)

# Equivalente con decorator
@jax.jit
def f_fast(x):
    return jnp.dot(x, x.T)
```

**Speedup JIT**:
```
Normal:     15 ms  (baseline)
JIT (1st):  50 ms  (tracing overhead)
JIT (2nd+): 1.5 ms (10 faster!)
```

**Come funziona JIT**:
1. **Prima chiamata** (tracing):
   - JAX passa "tracers" (non array reali!)
   - Costruisce Abstract Syntax Tree (AST)
   - Ottimizza: rimuove branch non necessari, fonde operazioni
   - Compila a XLA (Accelerated Linear Algebra)
   
2. **Chiamate successive**:
   - Usa versione compilata (cached)
   - Nessun Python overhead
   - GPU/TPU kernels ottimizzati

**Limitazioni JIT**:
-  Dynamic shapes: x[x < 0] (dimensione dipende da x)
-  Dynamic control flow con dati: if x > 0: ... (traccia entrambi i branch!)
-  Static control flow: if CONSTANT > 0: ...

**2. Automatic Differentiation** (jax.grad):
```python
# Funzione scalare  scalare
def f(x):
    return x**3 - 2*x**2 + x

df_dx = jax.grad(f)  # f'(x) = 3x - 4x + 1

# Esempio
x = 2.0
print(df_dx(x))  # 5.0 = 34 - 42 + 1

# Composizione: seconda derivata
d2f_dx2 = jax.grad(jax.grad(f))
print(d2f_dx2(x))  # 8.0 = 62 - 4

# Terza derivata
d3f_dx3 = jax.grad(jax.grad(jax.grad(f)))
print(d3f_dx3(x))  # 6.0 = 6
```

**Multiple inputs** (Jacobiano):
```python
def f(x, y):
    return x**2 + x*y

# Gradiente rispetto a primo argomento
df_dx = jax.grad(f, argnums=0)  # 2x + y

# Rispetto a entrambi
df = jax.grad(f, argnums=(0, 1))  # (2x+y, x)

# Jacobiano (funzioni vettoriali)
def g(x):
    return jnp.array([x[0]**2, x[0]*x[1]])

J = jax.jacrev(g)  # Jacobiano via reverse mode
# oppure
J = jax.jacfwd(g)  # via forward mode

# Hessiano (matrice 2nd derivatives)
H = jax.jacrev(jax.jacfwd(f))   ordine ottimale!
```

**Forward vs Reverse Mode** (come Lez15):
| Mode | Uso ottimale | JAX function |
|------|--------------|--------------|
| Forward | n  m (pochi input) | jax.jacfwd |
| Reverse | n  m (molti input) | jax.jacrev |

**3. VMAP** (Vectorization):
```python
# Funzione per singolo vettore
def dot_squared(v1, v2):
    return jnp.dot(v1, v2)**2

# Dati: matrici (batch di vettori)
X = jnp.random.randn(1000, 100)  # 1000 samples, 100 features
Y = jnp.random.randn(1000, 100)

#  For loop ingenuo (LENTO)
result = np.empty(1000)
for i in range(1000):
    result[i] = dot_squared(X[i], Y[i])
# Tempo: ~500 ms

#  VMAP (VELOCE)
batched_dot_squared = jax.vmap(dot_squared, in_axes=(0, 0))
result = batched_dot_squared(X, Y)
# Tempo: ~1 ms (500 faster!)

#  VMAP + JIT (FASTEST)
@jax.jit
def batched_dot_squared_jit(X, Y):
    return jax.vmap(dot_squared, in_axes=(0, 0))(X, Y)

result = batched_dot_squared_jit(X, Y)
# Tempo: ~0.1 ms (5000 faster!)
```

**VMAP Parameters**:
```
in_axes: tuple di int o None
  - 0: applica lungo primo asse (righe)
  - 1: applica lungo secondo asse (colonne)
  - None: broadcast (non iterare)

Esempio:
vmap(f, in_axes=(0, None))(X, y)
 applica f(X[i], y) per ogni riga i di X
 y � condiviso (broadcast)
```

### JAX Peculiarit� (Important!)

**1. Immutable Arrays**:
```python
#  NumPy style (mutable)
x = np.array([1, 2, 3])
x[0] = 10  # OK

#  JAX non permette
x = jnp.array([1, 2, 3])
x[0] = 10  # TypeError: JAX arrays are immutable

#  JAX style (funzionale)
x = jnp.array([1, 2, 3])
x_new = x.at[0].set(10)  # crea nuovo array!
# x rimane [1,2,3], x_new � [10,2,3]

# Operazioni avanzate
x_new = x.at[0:2].add(100)   # [101, 102, 3]
x_new = x.at[:].multiply(2)  # [2, 4, 6]
```

**Perch� immutabilit�?**
- JIT pu� ottimizzare: se vede che x non cambia, pu� riusare memoria
- No side effects  pure functions  parallelizzabile
- Functional programming style  composable

**2. Explicit Random Keys**:
```python
#  NumPy style (global state)
np.random.seed(42)
x = np.random.randn(10)
y = np.random.randn(10)  # diverso da x

#  JAX style (explicit state)
key = jax.random.PRNGKey(42)  # inizializza
x = jax.random.normal(key, shape=(10,))

#  ERRORE: riusare stessa key  stesso random!
y = jax.random.normal(key, shape=(10,))  # y == x !!

#  CORRETTO: split key
key, subkey = jax.random.split(key)
y = jax.random.normal(subkey, shape=(10,))  # y  x

# Pattern comune: split multipli
key, *subkeys = jax.random.split(key, num=5)
# subkeys = [key1, key2, key3, key4]
```

**Perch� explicit keys?**
- **Riproducibilit�**: stesso key  stesso output (sempre!)
- **Parallelismo**: no race conditions su global state
- **Pure functions**: randomness come input esplicito

**3. Device Management** (CPU/GPU/TPU):
```python
# Check device
print(jax.devices())  # [GpuDevice(id=0)]

# Array creato su default device (GPU se disponibile)
x = jnp.array([1, 2, 3])  # su GPU automaticamente

# NumPy array (sempre CPU)
x_np = np.array([1, 2, 3])

#  Mixing CPU/GPU (implicit transfer)
result = jnp.dot(x, x.T)  # GPU  GPU: fast (15 ms)
result = jnp.dot(x_np, x_np.T)  # CPU: slow (150 ms)
result = jnp.dot(x_np, x.T)  # CPUGPU transfer: slowest (35 ms)

#  Explicit transfer
x_gpu = jax.device_put(x_np)  # CPU  GPU
result = jnp.dot(x_gpu, x.T)  # GPU  GPU: fast!

# Bring back to CPU (for NumPy operations)
x_cpu = np.array(x)  # GPU  CPU
```

### JAX Advanced Topics

**Pure Functions Requirement**:
```python
#  IMPURE: usa variabile globale
g = 10
def impure(x):
    return x + g

impure_jit = jax.jit(impure)
print(impure_jit(5))  # 15
g = 20  # cambia g
print(impure_jit(5))  # 15 (NON 25!)  cached!

#  PURE: tutto via parametri
def pure(x, g):
    return x + g

pure_jit = jax.jit(pure)
print(pure_jit(5, 10))  # 15
print(pure_jit(5, 20))  # 25 
```

**Control Flow**:
```python
#  Data-dependent if (non traceable)
@jax.jit
def bad(x):
    if x > 0:  # x � array!
        return x**2
    else:
        return -x
# ConcretizationTypeError

#  Soluzione 1: jnp.where
@jax.jit
def good(x):
    return jnp.where(x > 0, x**2, -x)

#  Soluzione 2: lax.cond (pi� efficiente)
@jax.jit
def better(x):
    return jax.lax.cond(
        x > 0,
        lambda x: x**2,  # true branch
        lambda x: -x,    # false branch
        x
    )
```

**LAX (Low-Level API)**:
```python
# High-level (user-friendly)
result = jnp.add(1.0, 2.0)  # type promotion automatica

# Low-level (performant ma strict)
result = jax.lax.add(1.0, 2.0)  # OK
result = jax.lax.add(1, 2.0)    # ERROR: same type required!

# LAX per control flow
# lax.fori_loop: vettorizza for loop
def body_fn(i, val):
    return val + i

result = jax.lax.fori_loop(0, 10, body_fn, init_val=0)
# Equivalente: sum(range(10))

# lax.scan: accumula con state
def scan_fn(carry, x):
    new_carry = carry + x
    output = carry * x
    return new_carry, output

final_carry, outputs = jax.lax.scan(scan_fn, 0, jnp.arange(5))
```

**Non-Differentiable Functions**:
```python
f = lambda x: jnp.abs(x)
df_dx = jax.grad(f)

print(df_dx(1.0))   # 1.0 
print(df_dx(-1.0))  # -1.0 
print(df_dx(0.0))   # 1.0 (pragmatic choice, non -inf/+inf!)
```

### JAX Best Practices

**Performance Tips**:
```
1.  JIT everything: @jax.jit su funzioni hot
2.  VMAP batches: sostituisci for loops
3.  Combine JIT + VMAP: speedup moltiplicativo
4.  Keep data on device: minimizza CPUGPU transfer
5.  Avoid dynamic shapes: JIT recompila ogni volta
6.  Watch memory: immutability  molte copie (JIT ottimizza)
```

**Common Pitfalls**:
```
1.  Riusare random keys  same output!
    Sempre split: key, subkey = jax.random.split(key)

2.  Modificare array in-place  TypeError
    Usa .at[].set(): x_new = x.at[0].set(10)

3.  Global variables in JIT  cached!
    Pass as arguments: f(x, param)

4.  Data-dependent if in JIT  error
    Use jnp.where or lax.cond

5.  Mixing NumPy/JAX senza pensare  slow transfer
    Explicit jax.device_put quando needed
```

**Debugging JAX**:
```python
# Enable NaN/inf checks
jax.config.update('jax_debug_nans', True)

# Disable JIT for debugging
with jax.disable_jit():
    result = my_jitted_function(x)  # run eagerly

# Check compilation
jax.make_jaxpr(f)(x)  # mostra AST
```

**Installation**:
```bash
# CPU only
pip install jax jaxlib

# GPU (CUDA 12)
pip install jax[cuda12]

# TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## Codice Completo: SVT per MovieLens

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

# 1. LOAD & PREPROCESS
df = pd.read_csv('movielens.csv', sep='\\t', 
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

# 2. SHUFFLE (remove temporal bias)
np.random.seed(42)
indices = np.random.permutation(len(df))
df = df.iloc[indices].reset_index(drop=True)

# 3. EXTRACT & COMPACT
rows_orig = df['user_id'].values
cols_orig = df['item_id'].values
vals = df['rating'].values

# Compact indices (remove empty rows/cols)
rows, _ = np.unique(rows_orig, return_inverse=True)
cols, _ = np.unique(cols_orig, return_inverse=True)

n_users = len(np.unique(rows))
n_items = len(np.unique(cols))

# 4. TRAIN/TEST SPLIT (80/20)
n_train = int(0.8 * len(vals))
rows_train, rows_test = rows[:n_train], rows[n_train:]
cols_train, cols_test = cols[:n_train], cols[n_train:]
vals_train, vals_test = vals[:n_train], vals[n_train:]

# 5. CREATE SPARSE  DENSE
X_train = csr_matrix((vals_train, (rows_train, cols_train)), 
                     shape=(n_users, n_items)).toarray()

# 6. BASELINE: TRIVIAL RECOMMENDER
avg_rating = np.empty(n_users)
for i in range(n_users):
    mask = (X_train[i, :] > 0)
    avg_rating[i] = X_train[i, mask].mean() if mask.any() else 0

pred_trivial = avg_rating[rows_test]
rmse_trivial = np.sqrt(np.mean((vals_test - pred_trivial)**2))
rho_trivial, _ = pearsonr(vals_test, pred_trivial)

print(f"Baseline RMSE: {rmse_trivial:.3f}, ?: {rho_trivial:.3f}")

# 7. SVT HARD THRESHOLDING
A = X_train.copy()
tau = 100
max_iter = 200
tol = 1e-4

rmse_history = []
rho_history = []

for iteration in range(max_iter):
    A_old = A.copy()
    
    # SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Hard threshold
    s[s < tau] = 0
    
    # Reconstruct
    A = U @ np.diag(s) @ Vt
    
    # Impose known values
    A[rows_train, cols_train] = vals_train
    
    # Convergence check
    delta = np.linalg.norm(A - A_old, 'fro')
    
    # Evaluate
    pred_test = A[rows_test, cols_test]
    rmse = np.sqrt(np.mean((vals_test - pred_test)**2))
    rho, _ = pearsonr(vals_test, pred_test)
    
    rmse_history.append(rmse)
    rho_history.append(rho)
    
    if iteration % 20 == 0:
        print(f"Iter {iteration}: RMSE={rmse:.3f}, ?={rho:.3f}, ?={delta:.2e}")
    
    if delta < tol:
        print(f"Converged at iteration {iteration}")
        break

print(f"\\nFinal: RMSE={rmse:.3f}, ?={rho:.3f}")
print(f"Improvement over baseline: RMSE {rmse_trivial-rmse:.3f}, ? +{rho-rho_trivial:.3f}")
```

---

## Codice Completo: JAX Speedup Demo

```python
import jax
import jax.numpy as jnp
import numpy as np
import time

# Custom function
def custom_dot_squared(v1, v2):
    return jnp.dot(v1, v2)**2

# Data
X = jnp.random.randn(1000, 100)
Y = jnp.random.randn(1000, 100)

#  Naive for loop
start = time.time()
result = np.empty(1000)
for i in range(1000):
    result[i] = custom_dot_squared(X[i], Y[i])
time_naive = time.time() - start

#  JIT only
@jax.jit
def batched_jit(X, Y):
    result = []
    for i in range(len(X)):
        result.append(custom_dot_squared(X[i], Y[i]))
    return jnp.array(result)

_ = batched_jit(X, Y)  # warmup
start = time.time()
result = batched_jit(X, Y)
time_jit = time.time() - start

#  VMAP only
batched_vmap = jax.vmap(custom_dot_squared, in_axes=(0, 0))
_ = batched_vmap(X, Y)  # warmup
start = time.time()
result = batched_vmap(X, Y)
time_vmap = time.time() - start

#  VMAP + JIT
@jax.jit
def batched_vmap_jit(X, Y):
    return jax.vmap(custom_dot_squared, in_axes=(0, 0))(X, Y)

_ = batched_vmap_jit(X, Y)  # warmup
start = time.time()
result = batched_vmap_jit(X, Y)
time_vmap_jit = time.time() - start

print(f"Naive:        {time_naive*1000:.1f} ms")
print(f"JIT:          {time_jit*1000:.1f} ms ({time_naive/time_jit:.0f} faster)")
print(f"VMAP:         {time_vmap*1000:.1f} ms ({time_naive/time_vmap:.0f} faster)")
print(f"VMAP+JIT:     {time_vmap_jit*1000:.1f} ms ({time_naive/time_vmap_jit:.0f} faster)")
```

---

## Riferimenti Bibliografici

1. **Cai, J. F., Cand�s, E. J., & Shen, Z. (2010)**. "A singular value thresholding algorithm for matrix completion." *SIAM Journal on Optimization*, 20(4), 1956-1982.
   - Paper originale SVT (soft thresholding, dual variables)

2. **Cand�s, E. J., & Recht, B. (2009)**. "Exact matrix completion via convex optimization." *Foundations of Computational mathematics*, 9(6), 717-772.
   - Teoria: quando matrix completion � possibile

3. **Harper, F. M., & Konstan, J. A. (2015)**. "The MovieLens Datasets: History and Context." *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1-19.
   - MovieLens dataset paper

4. **Bradbury, J., Frostig, R., Hawkins, P., et al. (2018)**. "JAX: composable transformations of Python+NumPy programs." 
   - JAX documentation: https://jax.readthedocs.io/

5. **Frostig, R., Johnson, M. J., & Leary, C. (2018)**. "Compiling machine learning programs via high-level tracing." *MLSys*.
   - JAX JIT compilation internals

6. **Kingma, D. P., & Ba, J. (2014)**. "Adam: A method for stochastic optimization." *arXiv preprint arXiv:1412.6980*.
   - Adam optimizer (usato con JAX)

7. **Koren, Y., Bell, R., & Volinsky, C. (2009)**. "Matrix factorization techniques for recommender systems." *Computer*, 42(8), 30-37.
   - Recommender systems overview

---

**Fine Lezione 17 - Lab 4: SVT e JAX**

*Prossimo lab: Neural Networks con JAX/Flax, Training Loop, Optimizers*
