# Capitolo 1: Riepilogo e Confronto dei Metodi di Ottimizzazione
## Riepilogo dei Metodi di Ottimizzazione
[00:00] Questa sezione conclude il capitolo dedicato ai metodi di ottimizzazione e minimizzazione. Riassumendo i concetti analizzati finora, il percorso è iniziato con il metodo della discesa del gradiente (*steepest descent*), esaminato sia nella sua versione classica che in quella stocastica. Per entrambi questi approcci, si è osservata una convergenza di tipo lineare.
[00:07] Dal punto di vista del costo computazionale, il numero di operazioni richieste è dell'ordine di $O(n)$, dove $n$ rappresenta il numero di incognite del problema. Successivamente, è stato introdotto il metodo di Newton come esempio di metodo del secondo ordine. È stato dimostrato, in modo analogo a quanto avviene nel caso unidimensionale dove la prova è relativamente semplice, che tale metodo presenta una convergenza di tipo quadratico.
[00:15] Questo miglioramento nell'ordine di convergenza comporta, tuttavia, un costo computazionale significativo. Ogni passo dell'algoritmo richiede un numero di operazioni che ammonta a $O(n^3)$. Tra questi due estremi, rappresentati dalla discesa del gradiente e dal metodo di Newton, si collocano i metodi quasi-Newton. Come analizzato, questi metodi si basano sull'approssimazione dell'azione della matrice Hessiana.
[00:25] Con i metodi quasi-Newton non è possibile raggiungere la convergenza quadratica tipica del metodo di Newton, ma si ottiene una convergenza superlineare, che si posiziona a un livello intermedio. Il numero di operazioni richieste a ogni iterazione è dell'ordine di $O(n^2)$. Nell'ambito dei metodi quasi-Newton, tra le diverse strategie per approssimare la matrice Hessiana, è stato analizzato in dettaglio il metodo BFGS.
[00:36] Si è constatato che, nella sua versione standard, il metodo BFGS presenta un costo computazionale, in termini di numero di operazioni, dell'ordine di $O(n^2)$. Questo permette di raggiungere l'obiettivo di ridurre il numero di operazioni rispetto al metodo di Newton.
[00:44] Tuttavia, questo vantaggio ha un costo in termini di requisiti di memoria. In particolare, è necessario memorizzare l'azione della matrice, il che richiede di archiviare $n^2$ elementi. Per superare questo limite, l'idea successiva è stata quella di evolvere dal classico metodo BFGS al metodo L-BFGS (*Low-memory BFGS*), che si basa sull'idea di non memorizzare l'intera matrice Hessiana.
[00:56] Invece di conservare l'intera cronologia dei vettori utilizzati per calcolare l'Hessiana, come avviene nel metodo BFGS, si memorizza solo un sottoinsieme di essi, ad esempio quelli relativi alle ultime 10 o 12 iterazioni. Utilizzando questi vettori, si ricostruisce l'Hessiana a ogni passo, implementando una sorta di "finestra mobile". Se si fissa una finestra di dimensione 10, si mantengono in memoria solo i vettori relativi alle ultime 10 iterazioni.
[01:09] Sfruttando questa idea, si ottiene una drastica riduzione dei requisiti di memoria. Invece di $O(n^2)$, il fabbisogno di memoria diventa dell'ordine di $O(n \times m)$, dove $m$ è la dimensione della finestra, ovvero il numero di vettori memorizzati. Per fornire un esempio concreto, se la dimensione del problema $n$ è dell'ordine di un milione ($10^6$), invece di dover memorizzare $10^{12}$ elementi, supponendo di utilizzare una finestra $m=10$, si arriva a circa $10^7$ elementi. Si tratta di una riduzione molto significativa dei requisiti di memoria. Lo stesso principio si applica al numero di operazioni da eseguire a ogni passo.
[01:25] Questo vantaggio ha però un costo in termini di velocità di convergenza: invece di essere superlineare, la convergenza torna a essere di tipo lineare. In modo approssimativo ma efficace, si possono riassumere i contesti di utilizzo dei metodi BFGS e L-BFGS.
[01:33] Essenzialmente, quando la dimensione del problema $N$ è grande, si predilige l'utilizzo del metodo L-BFGS. Al contrario, quando $N$ è di dimensioni contenute, ad esempio durante la fase di prototipazione di una soluzione o per testare nuove idee, si può ricorrere al metodo BFGS. Quando si passa alla fase di produzione, è molto probabile che sia necessario utilizzare un metodo meno esigente in termini di requisiti di memoria.
# Capitolo 2: L'Algoritmo di Levenberg-Marquardt
## Introduzione all'Algoritmo di Levenberg-Marquardt
[01:43] Esiste un altro metodo che ha guadagnato grande popolarità ed è molto comune nelle applicazioni pratiche: l'algoritmo di Levenberg-Marquardt. In questa sezione verranno analizzate le modalità di utilizzo e le idee fondamentali alla base di questo metodo. L'idea verrà presentata partendo dal problema dei minimi quadrati non lineari.
[01:55] Come già visto in precedenza, il problema consiste nel trovare un insieme di vettori di pesi $w$ che permetta di creare un modello non lineare per un dato insieme di punti $(x_i, y_i)$. Questo problema è stato formalizzato in diverse occasioni.
[02:05] Per ogni campione di dati, è possibile calcolare il residuo. Una volta ottenuto il vettore dei residui, la funzione di costo è definita come la somma dei quadrati dei residui. Questa può essere scritta in forma matriciale come $R^T R$, che corrisponde alla norma al quadrato del vettore dei residui.
[02:14] Spesso, davanti a questo termine si inserisce un fattore $\frac{1}{2}$ per semplificare i calcoli durante il processo di derivazione, ma la sua presenza non ha un'importanza concettuale particolare. L'obiettivo finale è trovare il vettore di pesi $w$ che minimizza questa funzione di costo, indicata con $f(w)$.
## Confronto tra Discesa del Gradiente e Metodo di Newton
[02:22] Una possibile strategia per minimizzare la funzione di costo è utilizzare la discesa del gradiente. In questo caso, l'aggiornamento avviene muovendosi nella direzione di massima discesa, ovvero lungo la direzione opposta al gradiente. La formula di aggiornamento per il vettore dei pesi è quella usuale:
```math
w_{k+1} = w_k - \gamma \nabla f(w_k)
```
dove $\gamma$ è il tasso di apprendimento (*learning rate*).
[02:29] Come già osservato, questo metodo è piuttosto stabile e converge, ma risulta essere molto lento, poiché utilizza esclusivamente informazioni del primo ordine. D'altra parte, è stato analizzato il metodo di Newton, che implica l'uso della matrice Hessiana, con tutti gli svantaggi e i problemi già discussi.
[02:39] Tra questi, i principali sono il calcolo della matrice Hessiana e della sua inversa. In sintesi, lo svantaggio principale di questo metodo è la sua elevata richiesta computazionale.
## Obiettivo dell'Algoritmo di Levenberg-Marquardt
[02:44] L'obiettivo dell'algoritmo di Levenberg-Marquardt è quello di sviluppare un metodo che combini la velocità di convergenza del metodo di Newton con la stabilità della discesa del gradiente, evitando l'uso diretto della matrice Hessiana.
[02:51] Questi sono i due scopi che si intendono raggiungere. Rivediamo alcuni calcoli già visti in precedenza. Le componenti del gradiente della funzione di costo $J$ (qui indicata come $f$) sono date da:
```math
\nabla f = 2 J^T R
```
dove $J$ è la matrice Jacobiana dei residui e $R$ è il vettore dei residui.
[02:57] A questo punto, si può notare il motivo per cui spesso si introduce il fattore $\frac{1}{2}$ nella funzione di costo: in tal caso, il fattore 2 in questa espressione scompare. Questa è l'espressione del gradiente della funzione di costo in forma vettoriale.
[03:06] Successivamente, è necessario calcolare la matrice Hessiana. Derivando la relazione precedente, si ottiene che l'Hessiana $H$ è data da:
```math
H = 2 J^T J + 2 \sum_i R_i \nabla^2 R_i
```
[03:13] L'Hessiana può quindi essere scritta come la somma di due componenti. Il primo termine, $2 J^T J$, può essere considerato un termine del primo ordine. Il secondo termine, $2 \sum_i R_i \nabla^2 R_i$, è un termine del secondo ordine, poiché coinvolge le derivate seconde dei residui $R_i$.
## Il Metodo di Gauss-Newton
[03:22] A questo punto, viene introdotta l'idea del cosiddetto metodo di Gauss-Newton, che rappresenta il primo passo verso lo sviluppo dell'algoritmo di Levenberg-Marquardt.
[03:28] L'idea fondamentale consiste nel chiedersi cosa accade se si trascura il secondo termine nell'espressione dell'Hessiana. In questo modo, si ottiene un'approssimazione dell'Hessiana scartando il termine del secondo ordine. Questa è l'idea centrale del metodo.
[03:35] Questa approssimazione, che consiste nel trascurare il termine del secondo ordine, è particolarmente significativa in due situazioni principali. La prima si verifica quando i residui $R_i$ sono piccoli. In pratica, se i residui sono piccoli, è ragionevole supporre di essere vicini al minimo della funzione di costo, ovvero alla soluzione del problema.
[03:46] La seconda situazione si ha quando il modello è quasi lineare. Se i dati forniti sono vicini a un modello lineare, è ragionevole che le derivate seconde dei residui siano prossime a zero. In questi contesti, l'approssimazione dell'Hessiana ottenuta trascurando il termine del secondo ordine risulta sensata.
[03:57] L'idea del metodo di Gauss-Newton è partire dal classico passo di aggiornamento del metodo di Newton. L'incremento $\Delta$ che si vuole calcolare si ottiene risolvendo il seguente sistema lineare:
```math
H \Delta = - \nabla f
```
Si ricorda che nell'aggiornamento di Newton compare il termine $H^{-1} \nabla f$. Come più volte sottolineato, non si calcola mai esplicitamente l'inversa di una matrice.
[04:08] Si preferisce, invece, risolvere un sistema lineare. Questo è il sistema lineare che deve essere risolto. Poiché il gradiente è già stato calcolato come $\nabla f = 2 J^T R$, è possibile sostituire l'espressione del gradiente e l'approssimazione dell'Hessiana ($H \approx 2 J^T J$) nel passo di Newton.
[04:17] Semplificando il fattore 2, si ottiene la seguente relazione, che può essere vista come la controparte delle equazioni normali (viste nel contesto dei minimi quadrati) per il metodo di Gauss-Newton:
```math
(J^T J) \Delta = - J^T R
```
[04:25] Una volta calcolato $\Delta$ da questa equazione, si può procedere con l'aggiornamento del vettore dei pesi. È possibile introdurre un tasso di apprendimento per modulare questo passo, ma per ora l'attenzione è focalizzata sulla direzione di aggiornamento.
## Vantaggi e Svantaggi del Metodo di Gauss-Newton
[04:30] I vantaggi di questo metodo sono i seguenti: se il metodo converge, la convergenza è quadratica quando si è vicini alla soluzione. Questo avviene perché, in tale situazione, l'approssimazione dell'Hessiana che è stata fatta è ragionevole. È quindi lecito aspettarsi che la convergenza di questa approssimazione del metodo di Newton si comporti come il metodo originale.
[04:41] Inoltre, non è necessario calcolare la matrice Hessiana completa. In pratica, calcolare la matrice $J^T J$ è più semplice dal punto di vista computazionale rispetto al calcolo dell'Hessiana. Per comprendere il motivo, si può ricordare quanto visto sulla differenziazione automatica (*automatic differentiation*). Impostando correttamente il vettore iniziale del metodo all'indietro (*backward mode*), è possibile calcolare ogni colonna della matrice Jacobiana in un solo passaggio.
[04:54] Ciò significa che calcolare la matrice $J$ è relativamente semplice sfruttando, ad esempio, algoritmi di differenziazione automatica all'indietro. D'altra parte, per calcolare l'Hessiana, come già accennato, è necessario applicare il metodo due volte.
[05:03] Per calcolare l'Hessiana, è necessario utilizzare una procedura che combina sia il modo in avanti (*forward*) che quello all'indietro (*backward*), risultando molto più costosa di un singolo passaggio all'indietro.
[05:07] Gli svantaggi di questo metodo sono i seguenti: può divergere, specialmente se la stima iniziale non è sufficientemente buona, poiché l'approssimazione di $H$ è valida solo in prossimità della soluzione.
[05:13] Se la stima iniziale è lontana dal minimo, ci si possono aspettare problemi e persino la divergenza dell'algoritmo. Inoltre, può accadere che la matrice $J^T J$ sia mal condizionata.
[05:20] Un cattivo condizionamento implicherebbe che la soluzione del sistema lineare potrebbe essere instabile, e di conseguenza il valore di $\Delta$ calcolato potrebbe non essere affidabile come direzione di discesa.
## L'Idea Fondamentale dell'Algoritmo di Levenberg-Marquardt
[05:26] L'idea alla base dell'algoritmo di Levenberg-Marquardt è quella di combinare la stabilità della discesa del gradiente con la velocità del metodo di Gauss-Newton. Questo è l'obiettivo finale che si vuole raggiungere.
[05:34] L'idea consiste essenzialmente nel regolarizzare la soluzione del sistema lineare. In pratica, invece di considerare il sistema $J^T J \Delta = -J^T R$ per calcolare $\Delta$, si introduce un termine aggiuntivo, che può essere definito di rilassamento, smorzamento (*damping*) o regolarizzazione.
[05:46] Questo termine aggiuntivo può essere interpretato anche in un altro modo. Se si trascura il termine $J^T J$, si ottiene $\lambda I \Delta = -J^T R$. Questa espressione corrisponde esattamente a un passo di discesa del gradiente. Se, invece, si pone $\lambda = 0$, si ottiene il metodo di Gauss-Newton. Di conseguenza, $\lambda$ può essere visto come un parametro che determina il "peso" di ciascuno dei due metodi.
[06:00] Si tratta di una sorta di interpolazione tra i due metodi. Il punto chiave è che questa interpolazione non è statica: il parametro $\lambda$ non viene scelto una volta per tutte, ma viene modificato dinamicamente durante l'esecuzione dell'algoritmo. La strategia di implementazione verrà discussa a breve.
[06:10] L'aggiornamento si basa sulla seguente equazione, dove $\lambda$ è il parametro di smorzamento e $I$ è la matrice identità:
```math
(J^T J + \lambda I) \Delta = - J^T R
```
Una nota importante riguarda uno degli svantaggi del metodo di Gauss-Newton: la possibilità che la matrice $J^T J$ fosse singolare o mal condizionata.
[06:19] Introducendo il termine aggiuntivo $\lambda I$, si garantisce che la matrice $(J^T J + \lambda I)$ sia sempre invertibile (per $\lambda > 0$). Questo rappresenta un primo punto a favore di questa idea di interpolazione. In altri termini, l'incremento $\Delta$ è dato dalla seguente espressione:
```math
\Delta = -(J^T J + \lambda I)^{-1} J^T R
```
dove, invece di avere solo l'inversa di $J^T J$, compare anche il termine di regolarizzazione $\lambda I$.
## Casi Limite e Interpretazione di $\lambda$
[06:30] Vengono ora analizzati i due casi limite. Il primo si ha quando il parametro $\lambda$ tende a zero. Come già osservato, se $\lambda \to 0$, l'incremento $\Delta$ è dato da:
```math
\Delta \approx -(J^T J)^{-1} J^T R
```
Questa espressione corrisponde esattamente all'aggiornamento del metodo di Gauss-Newton. Tale aggiornamento è veloce ma potenzialmente instabile.
[06:39] Si desidera utilizzare questo tipo di aggiornamento quando si è sicuri di essere vicini a una buona soluzione. Il secondo caso limite si ha quando $\lambda$ tende a infinito. In questa situazione, come osservato, l'incremento $\Delta$ è dato da:
```math
\Delta \approx -\frac{1}{\lambda} J^T R
```
Questa espressione corrisponde a un passo di discesa del gradiente.
[06:47] È noto che la discesa del gradiente è stabile ma lenta. Grazie alla sua stabilità, può essere preferibile all'inizio del processo iterativo, quando la stima iniziale può essere arbitraria. L'uso di questo approccio può migliorare il comportamento globale dell'algoritmo.
## Struttura dell'Algoritmo di Levenberg-Marquardt
[06:56] Viene ora presentata la struttura dell'algoritmo che può essere implementato sfruttando le idee appena discusse. Si inizia con una stima iniziale per i pesi, $w_0$, e per il parametro di smorzamento, $\lambda_0$. Come detto, $\lambda$ non è un parametro costante, ma viene modificato durante la procedura.
[07:06] A ogni iterazione $k$, si calcolano il vettore dei residui $R_k$ e la matrice Jacobiana $J_k$. Successivamente, si valuta la funzione di costo sulla soluzione corrente, $f(w_k)$. Si risolve poi il sistema lineare per calcolare il vettore di incremento $\Delta_k$ utilizzando l'aggiornamento di Levenberg-Marquardt.
[07:16] Si utilizza il vettore di aggiornamento $\Delta_k$ appena calcolato per determinare un nuovo vettore di pesi candidato, che viene chiamato $w_{new}$:
```math
w_{new} = w_k + \Delta_k
```
Questo non è ancora il vettore finale $w_{k+1}$, poiché è necessario eseguire alcuni controlli aggiuntivi. In particolare, una volta calcolato $w_{new}$, si può valutare la funzione di costo su questo nuovo vettore.
[07:26] A questo punto, si deve definire la strategia di aggiornamento per il parametro $\lambda$. Questa strategia determinerà se l'algoritmo si sta muovendo nella direzione della discesa del gradiente o del metodo di Gauss-Newton. Se il nuovo vettore di pesi è tale che la funzione di costo valutata su di esso è inferiore a quella dell'iterazione precedente, ovvero $f(w_{new}) < f(w_k)$, allora si è trovata una direzione di aggiornamento che ha prodotto una riduzione della funzione di costo.
[07:33] Questo è un passo che può essere accettato. Di conseguenza, si può affermare che il nuovo vettore di pesi all'iterazione $k+1$ è $w_{new}$.
[07:43] Inoltre, poiché si è ottenuta una riduzione del costo, è probabile trovarsi in una buona regione dello spazio dei parametri. È quindi possibile ridurre il valore di $\lambda$. Si ricorda che ridurre $\lambda$ significa spostarsi verso il metodo di Gauss-Newton, accelerando così la convergenza. La riduzione può avvenire, ad esempio, tramite un fattore $\mu > 1$: $\lambda_{k+1} = \lambda_k / \mu$.
[07:54] Se, al contrario, il nuovo vettore di pesi è tale che la funzione di costo è uguale o maggiore di quella dell'iterazione precedente, $f(w_{new}) \ge f(w_k)$, allora il passo viene rifiutato.
[08:01] In questo caso, si conclude che il valore di $\lambda$ deve essere aumentato. Questa situazione indica che ci si trova in una regione in cui è preferibile un metodo più stabile. Pertanto, si aumenta $\lambda$, ad esempio $\lambda_{k+1} = \lambda_k \times \mu$, e il passo viene ripetuto con il nuovo valore di $\lambda$.
[08:10] Come per ogni metodo iterativo, è necessario introdurre un criterio di arresto. Ad esempio, si può monitorare la norma dell'incremento $\|\Delta_k\|$ o la differenza tra i valori della funzione di costo in due iterazioni consecutive.
[08:18] Se si raggiunge la convergenza, ovvero se una di queste quantità scende al di sotto di una tolleranza predefinita, l'algoritmo si ferma.
## Considerazioni Pratiche e Costo Computazionale
[08:21] L'algoritmo di Levenberg-Marquardt è utilizzato molto spesso perché, come accennato, in un'applicazione pratica (ad esempio, analizzando notebook su piattaforme come Kaggle per l'addestramento di reti neurali), una strategia comune consiste nell'eseguire alcuni passi di discesa del gradiente (ad esempio, 100 passi) e poi, partendo dalla soluzione ottenuta, passare a un metodo più raffinato come BFGS, L-BFGS o altri.
[08:31] In sostanza, questa procedura manuale e predefinita realizza ciò che l'algoritmo di Levenberg-Marquardt cerca di fare in modo adattivo. Questa è la ragione della sua grande diffusione.
[08:48] Per quanto riguarda il costo computazionale, uno degli svantaggi è che bisogna ancora risolvere un sistema lineare $n \times n$ a ogni iterazione, il che richiede $O(n^3)$ operazioni. Esistono alcune varianti che permettono di ridurre questo numero di operazioni, ma sono tecnicamente complesse. L'obiettivo di questa sezione era presentare l'idea di base del metodo.
[09:01] Come per gli altri metodi di minimizzazione analizzati, è importante tenere presente che non vi è alcuna garanzia che uno qualsiasi di questi metodi possa raggiungere un minimo globale. Essi sono in grado di trovare un minimo locale, ma non è garantito che trovino quello globale.
[09:11] Se la "superficie" della funzione di costo presenta più di un minimo locale, il minimo raggiunto dal metodo dipende fortemente dalla stima iniziale utilizzata. Questo è stato mostrato anche in precedenza, durante l'analisi del comportamento del metodo del momento.
[09:20] Ci sono anche altri due parametri da impostare, che rientrano nella grande famiglia degli iperparametri del metodo: il valore iniziale di $\lambda$ e il fattore di aggiornamento $\mu$. Anche questi parametri possono influenzare notevolmente la convergenza. Tuttavia, per applicazioni "classiche", esistono valori suggeriti in letteratura o intervalli di valori ragionevoli che possono essere utilizzati in modo sicuro.
[09:33] In pratica, per problemi di minimi quadrati non lineari, questo è il metodo solitamente adottato.
## Riepilogo Finale sui Metodi di Minimizzazione
[09:37] Dal punto di vista degli algoritmi di minimizzazione utilizzati nel contesto del machine learning, sono stati introdotti i metodi classici, alcuni loro miglioramenti e anche alcuni sviluppi più recenti. In realtà, consultando la letteratura scientifica o le librerie software, si possono trovare molti altri metodi.
[09:48] La maggior parte di essi sono variazioni di quelli che sono stati analizzati, sviluppate magari per applicazioni specifiche con peculiarità che richiedono un'ottimizzazione mirata di questi metodi. Avendo una chiara comprensione dei metodi analizzati, si è in grado di comprendere anche queste varianti.
# Capitolo 3: Fondamenti Matematici della Convoluzione
## Introduzione alla Convoluzione
[00:00] Si procede ora all'introduzione del concetto di convoluzione. Sebbene alcuni possano già avere familiarità con questo argomento, le seguenti sezioni forniranno una base concettuale. La convoluzione riveste un'importanza fondamentale in specifiche tipologie di reti neurali, in particolare quelle progettate per compiti come il riconoscimento di immagini. L'obiettivo è fornire un'idea generale dei fondamenti matematici che ne sono alla base.
[00:23] In senso matematico, la convoluzione è un'operazione che permette di analizzare l'effetto di una funzione su un'altra.
[00:31] Nello specifico, se si dispone di un segnale, indicato con $F$, e di un filtro, indicato con $G$, la convoluzione, solitamente denotata con il simbolo dell'asterisco ($*$), descrive come il filtro agisce sul segnale.
[00:41] Ad esempio, dato un segnale $F$ e un filtro $G$, il risultato della loro convoluzione rappresenta l'effetto del filtro sul segnale.
[00:51] La convoluzione è un concetto cruciale nell'elaborazione dei segnali e delle immagini. In questi contesti, i filtri sono strumenti progettati per rilevare caratteristiche specifiche (in inglese, *features*) all'interno delle immagini di input, come angoli, bordi o altri pattern.
[01:06] I filtri possono anche essere utilizzati per modificare l'immagine originale. Un esempio comune è l'applicazione di un effetto di sfocatura (*blur*), che si ottiene tramite un filtro specifico. Nelle applicazioni pratiche e, in particolare, nel machine learning, l'interesse non è tanto per la convoluzione in senso generale, quanto per la sua controparte discreta.
## Matrici di Toeplitz e Circolanti
[01:22] La versione discreta della convoluzione viene realizzata attraverso l'uso di una classe particolare di matrici. Alla base di questo approccio si trovano le cosiddette matrici di Toeplitz.
[01:32] La struttura generica di una matrice di Toeplitz è caratterizzata dal fatto che, una volta definita la prima riga (ad esempio, con elementi $A, B, C, D$), le righe successive sono ottenute tramite uno spostamento (*shift*) di questa lungo la diagonale. La matrice risultante è costante lungo ogni diagonale.
[01:48] Una sottoclasse ancora più importante delle matrici di Toeplitz, poiché sono quelle effettivamente utilizzate nella convoluzione, è rappresentata dalle matrici circolanti. La differenza fondamentale tra una matrice di Toeplitz e una circolante risiede nel modo in cui avviene lo spostamento. In una matrice circolante, l'elemento che "scompare" a destra durante lo spostamento della prima riga (ad esempio, l'elemento $D$ nell'esempio precedente) "rientra" all'inizio della riga successiva.
[02:08] Una matrice circolante è quindi definita completamente dal vettore che costituisce la sua prima riga. La dimensione del vettore determina la dimensione della matrice quadrata, e la struttura circolare definisce tutti gli altri elementi.
[02:24] Si può dimostrare facilmente che il prodotto di due matrici circolanti, $C$ e $D$, è ancora una matrice circolante. Inoltre, il prodotto è commutativo, ovvero $C \cdot D = D \cdot C$, una proprietà non comune per il prodotto tra matrici.
## Matrici Circolanti e Polinomi
[02:37] Un'altra proprietà importante di queste matrici riguarda la loro rappresentazione. Come detto, una matrice circolante è definita dalla sua prima riga.
[02:43] Viene ora introdotta una matrice speciale, $P$, che può essere definita matrice di permutazione o matrice di spostamento (*shift matrix*).
[02:51] Questa matrice potrebbe essere già nota da altri contesti, come l'analisi numerica.
[03:01] Ad esempio, durante lo studio della fattorizzazione LU di una matrice $A$, si può incontrare una situazione in cui l'algoritmo non può procedere.
[03:08] Nella creazione dei fattori $L$ e $U$, l'algoritmo richiede di dividere per alcuni elementi, detti *pivot*, che appaiono sulla diagonale delle matrici intermedie. Se uno di questi elementi pivot è nullo, l'algoritmo si interrompe. Per rendere l'algoritmo più robusto, si può introdurre la fattorizzazione LU applicata a una versione della matrice in cui alcune righe o colonne sono state permutate o spostate.
[03:26] Supponiamo di avere una matrice $A$ e di voler scambiare due delle sue righe. Se si moltiplica la matrice identità per $A$, il risultato è $A$ stessa.
[03:36] Tuttavia, se si modifica la matrice identità scambiando la sua prima e seconda riga, e poi si moltiplica questa nuova matrice per $A$, il risultato sarà una versione di $A$ con la prima e la seconda riga scambiate.
[03:46] Ad esempio, la moltiplicazione della prima riga della matrice di permutazione per la prima colonna di $A$ darà come risultato il primo elemento della nuova prima riga. Questo processo si ripete per tutti gli elementi.
[03:59] Questa matrice modificata è una matrice di permutazione. Se si applica questa permutazione al sistema lineare $Ax=b$, si ottiene $PAx = Pb$. A questo punto, è possibile eseguire la fattorizzazione $PA = LU$. La permutazione delle righe consente, nella maggior parte dei casi, di evitare la presenza di elementi pivot nulli e di completare la fattorizzazione.
[04:14] È anche possibile pre-moltiplicare e post-moltiplicare la matrice per matrici di permutazione, $P$ e $Q$, per permutare sia le righe che le colonne. Se la matrice di permutazione pre-moltiplica la matrice $A$, si permutano le righe.
[04:24] Se invece post-moltiplica $A$, si spostano le colonne. La matrice $P$ specifica che stiamo considerando sposta la prima riga della matrice identità all'ultima posizione, causando uno spostamento ciclico di tutte le righe. Calcolando le potenze successive di $P$, come $P^2$, si ottiene uno spostamento ulteriore.
[04:38] In particolare, $P^2$ sposta ogni colonna di una posizione a destra, con rientro da sinistra. $P^3$ produce un altro spostamento ciclico. Per una matrice $4 \times 4$, la potenza $P^4$ sarà uguale alla matrice identità, poiché dopo quattro spostamenti ciclici si ritorna alla configurazione iniziale.
[04:54] Dato un vettore $c = [c_0, c_1, \dots, c_{n-1}]$ che definisce la prima riga di una matrice circolante $C$, questa può essere espressa come una combinazione lineare delle potenze della matrice di spostamento $P$:
```math
C = c_0 I + c_1 P + c_2 P^2 + \dots + c_{n-1} P^{n-1}
```
dove $I$ è la matrice identità.
[05:07] Questa espressione è essenzialmente un polinomio valutato nella matrice $P$.
[05:14] Si può quindi affermare che ogni matrice circolante $C$ è il risultato della valutazione di un polinomio, i cui coefficienti sono gli elementi della prima riga di $C$, nella matrice di spostamento ciclico elementare $P$ di dimensione appropriata.
## Prodotto Ciclico e Convoluzione
[05:26] Si considera ora il prodotto di due matrici circolanti. Se si ha una matrice circolante $C$, definita dal vettore $c$, e una matrice $D$, definita dal vettore $d$, queste possono essere viste come due polinomi, $p(x)$ e $q(x)$, valutati nella matrice $P$. I coefficienti di $p(x)$ sono dati dal vettore $c$, mentre quelli di $q(x)$ sono dati dal vettore $d$.
[05:43] Il prodotto delle due matrici, $C \cdot D$, corrisponde al prodotto dei due polinomi valutato in $P$, ovvero $(p \cdot q)(P)$. Questo prodotto polinomiale viene calcolato modulo $x^n - 1$.
[05:52] Analizziamo questo concetto più in dettaglio. Supponiamo di avere due polinomi rappresentati dai vettori di coefficienti $[1, 2, 3]$ e $[4, 5, 0]$. Il prodotto standard tra questi due polinomi genera un polinomio di grado 3.
[06:04] Il prodotto polinomiale classico dà come risultato un nuovo vettore di coefficienti $[4, 13, 22, 15]$. Tuttavia, quando si lavora con matrici circolanti, si deve considerare il prodotto ciclico.
[06:13] Il prodotto ciclico sfrutta la proprietà $P^n = I$, dove $n$ è la dimensione della matrice. Nel caso di matrici $3 \times 3$, si ha $P^3 = I$. Ciò significa che il coefficiente del termine di grado 3, $x^3$, viene sommato al coefficiente del termine di grado 0 (la costante).
[06:27] Nell'esempio, il coefficiente 15 (corrispondente a $x^3$) viene sommato al coefficiente 4 (corrispondente a $x^0$), risultando in 19. Il vettore dei coefficienti del prodotto ciclico diventa quindi $[19, 13, 22]$.
[06:38] Questo significa che il prodotto delle due matrici circolanti definite dai vettori $[1, 2, 3]$ e $[4, 5, 0]$ è una nuova matrice circolante la cui prima riga è data dal vettore $[19, 13, 22]$.
[06:47] L'operazione di prodotto tra due matrici circolanti è stata ridotta a un prodotto ciclico tra i polinomi i cui coefficienti sono i vettori che definiscono le prime righe delle matrici.
[07:01] Dal punto di vista polinomiale, l'espressione "modulo $x^3 - 1$" implica l'assunzione che $x^3 = 1$, che è esattamente la proprietà utilizzata per calcolare il prodotto ciclico.
## Autovettori e Autovalori delle Matrici Circolanti
[07:10] Esiste una proprietà fondamentale delle matrici circolanti: tutte le matrici circolanti di una data dimensione $n \times n$ condividono gli stessi autovettori.
[07:20] In particolare, questi autovettori sono le colonne della cosiddetta matrice di Fourier. La matrice di Fourier, $F$, è definita dalla seguente espressione:
```math
F_{jk} = \omega^{jk}
```
dove $\omega = e^{2\pi i / n}$ è la radice $n$-esima primitiva dell'unità.
[07:34] Ad esempio, per una matrice $3 \times 3$, le radici terze dell'unità si trovano sul cerchio unitario nel piano complesso. Per una matrice $4 \times 4$, le radici sono $1, i, -1, -i$.
[07:43] In generale, per una matrice di ordine $n$, si calcolano le $n$ radici dell'unità. Queste definiscono gli autovettori.
[07:50] L'autovalore corrispondente a ciascun autovettore è legato alla trasformata di Fourier discreta (DFT) del vettore $c$ che definisce la prima riga della matrice.
[07:58] Ricordando che una matrice circolante è identificata dal vettore $c$ della sua prima riga, la trasformata di Fourier discreta di $c$ è data da:
```math
\hat{c}_k = \sum_{j=0}^{n-1} c_j \omega^{-kj}
```
dove $k$ varia da $0$ a $n-1$. I valori $\hat{c}_k$ sono gli autovalori della matrice circolante.
[08:12] In sintesi, per una matrice circolante di ordine $n$, gli autovettori sono sempre le colonne della matrice di Fourier di ordine $n$, mentre gli autovalori si ottengono calcolando la DFT del vettore che definisce la prima riga della matrice.
[08:24] Vediamo un esempio per $n=4$. La radice dell'unità è $\omega = e^{2\pi i / 4} = i$.
[08:30] La matrice di Fourier di ordine 4 è quindi costruita usando le potenze di $i$. Se si considera una matrice circolante $C$ definita dal vettore $[c_0, c_1, c_2, c_3]$, i suoi autovettori sono le quattro colonne della matrice di Fourier, mentre i suoi autovalori sono dati dalle combinazioni lineari che definiscono la DFT di tale vettore.
## La Regola della Convoluzione
[08:46] Un'altra proprietà fondamentale è la cosiddetta regola della convoluzione, ampiamente utilizzata nelle implementazioni pratiche per ridurre il costo computazionale della moltiplicazione.
[08:56] Poiché gli autovettori di una matrice circolante $C$ sono le colonne della matrice di Fourier $F$ e i suoi autovalori sono la DFT del vettore $c$, la matrice $C$ può essere diagonalizzata come segue:
```math
C = F^{-1} \Lambda_c F
```
dove $\Lambda_c$ è la matrice diagonale contenente gli autovalori di $C$.
[09:09] La convoluzione di due vettori, $c$ e $d$, che nel caso discreto corrisponde al prodotto ciclico dei polinomi associati, può essere scritta come il prodotto della matrice circolante $C$ per il vettore $d$.
[09:20] Sostituendo la forma diagonalizzata di $C$, si ottiene:
```math
c * d = C d = (F^{-1} \Lambda_c F) d
```
Questo significa che per calcolare la convoluzione tra $c$ e $d$, si possono seguire questi passaggi:
1.  Calcolare la DFT di $d$: $\hat{d} = Fd$.
2.  Calcolare la DFT di $c$ per ottenere gli autovalori: $\hat{c}$.
3.  Moltiplicare i due vettori risultanti, $\hat{c}$ e $\hat{d}$, elemento per elemento (prodotto di Hadamard).
4.  Calcolare la trasformata di Fourier discreta inversa (IDFT) del risultato per tornare al dominio originale: $r = F^{-1} (\hat{c} \odot \hat{d})$.
[09:49] In altre parole, la convoluzione nel dominio del tempo (o dello spazio) è equivalente a una moltiplicazione elemento per elemento nel dominio della frequenza. Il dominio della frequenza è quello in cui si opera dopo aver applicato la trasformata di Fourier discreta.
[10:02] Una volta ottenuto il prodotto $\hat{r}$ nel dominio della frequenza, si applica la IDFT per ottenere il vettore $r$, che rappresenta il risultato della convoluzione.
[10:09] Questa connessione tra la convoluzione (o prodotto di matrici circolanti) e la trasformata di Fourier discreta è di grande importanza.
## Vantaggi Computazionali della Regola della Convoluzione
[10:14] Supponiamo di avere due matrici circolanti $C$ e $D$ di dimensione $n \times n$.
[10:20] Se si calcola il prodotto $C \cdot D$ direttamente, è necessario calcolare la prima riga della matrice risultante, che è la convoluzione ciclica delle prime righe di $C$ e $D$. Questo richiede $n$ prodotti per ciascuno degli $n$ elementi della prima riga, portando a un totale di $O(n^2)$ operazioni.
[10:33] Vediamo cosa accade applicando la regola della convoluzione. Per calcolare gli autovalori di $C$ e $D$, è necessario calcolare le loro DFT.
[10:39] Sfruttando l'algoritmo della Trasformata Veloce di Fourier (FFT), queste operazioni possono essere eseguite in tempo $O(n \log n)$. Successivamente, si esegue la moltiplicazione elemento per elemento dei due vettori di autovalori, che richiede $O(n)$ operazioni. Infine, si calcola la Trasformata Veloce di Fourier Inversa (IFFT) per ottenere la prima riga della matrice risultante, un'altra operazione da $O(n \log n)$.
[10:54] Per valori di $n$ sufficientemente grandi, l'approccio basato sulla FFT, con un costo di $O(n \log n)$, è significativamente più efficiente dell'approccio diretto, che ha un costo di $O(n^2)$.
[11:01] Questa relazione tra convoluzione, prodotto di matrici e prodotto polinomiale ciclico è di fondamentale importanza perché permette di eseguire prodotti di matrici in modo molto rapido.
# Capitolo 4: Applicazioni della Convoluzione nelle Reti Neurali
## Applicazione della Convoluzione alle Immagini
[11:10] Questo è rilevante perché, quando si applicano filtri a un'immagine, si sta essenzialmente eseguendo un'operazione di convoluzione. Ad esempio, data un'immagine di $1000 \times 1000$ pixel, si può scegliere un filtro, come un filtro di sfocatura di $20 \times 20$ pixel.
[11:23] Questo filtro può essere rappresentato da una piccola matrice che viene spostata lungo l'immagine per osservarne l'effetto su diverse porzioni. Questo processo richiede di eseguire molteplici moltiplicazioni di matrici, che possono essere accelerate utilizzando l'approccio basato sulla FFT.
[11:34] Finora è stato discusso il caso monodimensionale (1D). L'estensione al caso bidimensionale (2D), come quello delle immagini, segue principi analoghi. L'idea di un filtro è applicare una trasformazione a un'immagine di input.
[11:42] L'immagine di input può essere rappresentata da una matrice di valori, ad esempio in scala di grigi (da 0 a 255). Si applica un filtro, detto anche *kernel*, ad esempio di dimensione $3 \times 3$.
[11:52] L'obiettivo è produrre un nuovo valore (un pixel nell'immagine di output) che rappresenti l'effetto del filtro su una porzione specifica dell'immagine originale. Questa operazione riduce un blocco di pixel a un singolo pixel che contiene informazioni aggregate sulla regione originale.
[12:05] Ad esempio, supponiamo che il kernel sia progettato per rilevare i bordi verticali. Il valore del pixel risultante nell'immagine elaborata dovrebbe contenere informazioni utili per inferire la presenza o meno di un bordo verticale nella porzione corrispondente dell'immagine originale.
[12:16] Le reti neurali convoluzionali (CNN) applicano ricorsivamente questo tipo di filtri per trasformare un'immagine complessa in una rappresentazione più semplice, come un vettore, che può essere utilizzata per classificare o rilevare oggetti nell'immagine.
[12:30] Questo vettore finale deve conservare le informazioni salienti dell'immagine originale. La progettazione dei filtri è quindi di fondamentale importanza per ottenere una rappresentazione significativa.
## Esempi di Filtri (Kernel)
[12:38] Un esempio pratico è l'applicazione di un filtro per ammorbidire (*soften*) o sfocare (*blur*) un'immagine.
[12:42] Utilizzando un kernel specifico su un'immagine con valori di pixel 10 e 90, l'applicazione del filtro può smussare il valore centrale da 90 a 19, producendo il classico effetto di sfocatura.
[12:55] Un altro esempio è il rilevamento dei bordi (*edge detection*). Un kernel specifico può essere utilizzato per rilevare i bordi verticali. Applicando questo filtro a una matrice di pixel, si può ottenere un valore molto alto (es. 320), indicando una probabilità elevata della presenza di un bordo verticale.
[13:07] In un'immagine dove i valori dei pixel cambiano bruscamente in direzione verticale, questo filtro evidenzierà la posizione del bordo.
## Esempio di Rete Neurale Convoluzionale (CNN)
[13:14] Si considera ora un esempio pratico di una rete neurale convoluzionale. In un'interfaccia grafica, è possibile disegnare una cifra numerica.
[13:21] La rete analizza l'immagine per riconoscere la cifra disegnata.
[13:28] L'immagine disegnata costituisce il livello di input (*input layer*). Segue un primo livello convoluzionale, dove ogni pixel dell'immagine elaborata è ottenuto considerando un sottoinsieme dell'immagine originale e applicando un filtro.
[13:40] Analizzando il processo, si può osservare l'immagine di input, il filtro applicato e i calcoli eseguiti. Diversi filtri vengono applicati all'immagine originale, generando più mappe di caratteristiche (in questo caso, sei).
[13:52] Successivamente, si applica un livello di *downsampling* (sottocampionamento).
[13:57] Il *downsampling* riduce la dimensione delle mappe di caratteristiche. Un metodo comune è il *max pooling*. Se si considera una regione di pixel (ad esempio, $3 \times 3$), il *max pooling* la riduce a un singolo pixel, il cui valore è il massimo tra quelli della regione.
[14:13] Ad esempio, da una regione di 9 pixel, il *max pooling* estrae il valore 25, che è il massimo. Sebbene la dimensione delle immagini si riduca, le caratteristiche essenziali vengono preservate.
[14:22] Dopo il primo *downsampling*, può seguire un altro livello convoluzionale. Il *max pooling* è solo una delle possibili operazioni di *downsampling*; si potrebbe anche usare la media (*average pooling*). Non esiste una ricetta unica.
[14:36] Si esegue quindi un'altra convoluzione, applicando nuovi filtri.
[14:42] Segue un altro livello di *downsampling*, ancora con *max pooling*. Infine, i dati vengono processati da strati completamente connessi (*fully connected layers*). In questi strati, ogni neurone è connesso a tutti gli output dello strato precedente, spesso utilizzando funzioni di attivazione come la tangente iperbolica.
[14:57] L'ultimo strato è il livello di output (*output layer*), che fornisce i valori finali per la classificazione. In questo caso, il valore più alto (1.01) corrisponde alla classe "3", indicando che la rete ha riconosciuto la cifra come un tre.
[15:09] Se si disegna una cifra diversa, la rete potrebbe interpretarla in modo errato. Ad esempio, un "3" disegnato male potrebbe essere riconosciuto come un "8".
[15:15] In questo caso, il valore massimo è 1.0, corrispondente alla classe "8", mentre il valore per la classe "3" è -1.0.
[15:20] L'idea di base è che la convoluzione viene applicata nei primi strati della rete, sfruttando i concetti matematici descritti in precedenza per elaborare le immagini in modo efficiente.
## Architettura di una CNN per Immagini a Colori
[15:28] In situazioni più complesse, come l'analisi di immagini a colori, l'immagine di input viene suddivisa nei suoi tre canali: rosso, verde e blu (RGB).
[15:34] La rete può includere diversi livelli convoluzionali, intervallati da livelli di attivazione (come ReLU) e di *max pooling*. I livelli convoluzionali mantengono la dimensione dell'immagine, mentre i livelli di *max pooling* la riducono.
[15:44] Man mano che si procede attraverso la rete, le rappresentazioni diventano sempre più astratte e meno simili all'immagine originale.
[15:50] Dopo diversi strati di convoluzione e *pooling*, i dati vengono inviati a strati completamente connessi che eseguono la classificazione finale. In un esempio, la rete è in grado di riconoscere un oggetto (una pizza) anche se la sua forma non è quella classica, dimostrando la capacità di astrazione del modello.
[16:04] La lezione si interrompe.
# Capitolo 5: Il Teorema di Approssimazione Universale
## Introduzione alle Reti Neurali: Riepilogo dei Concetti
[00:00] Finora, nel contesto delle reti neurali, è stata definita la loro struttura, ovvero la topologia composta da un livello di input, livelli nascosti e un livello di output. Sono stati anche analizzati gli elementi costitutivi di una rete neurale, come le funzioni di attivazione e le funzioni di costo.
[00:13] Successivamente, si è compreso che l'addestramento di una rete neurale consiste nel trovare l'insieme di pesi e bias che minimizzano la funzione di costo. Per raggiungere questo obiettivo, sono stati esaminati diversi metodi di ottimizzazione, sia del primo ordine che del secondo ordine. Tali metodi richiedono il calcolo delle derivate prime (gradienti) o delle derivate seconde (hessiani).
[00:28] È stato mostrato come calcolare queste derivate utilizzando la differenziazione automatica. In pratica, sono stati analizzati tutti gli elementi necessari per scrivere e addestrare una rete neurale. Ora, tuttavia, si vuole affrontare un aspetto più teorico.
## Il Teorema di Approssimazione Universale: La Domanda Fondamentale
[00:40] La questione teorica che si intende esplorare riguarda il motivo per cui si è scelto di utilizzare una rete neurale. L'obiettivo pratico è il seguente: dati un insieme di input $X$ e i corrispondenti output, come nel caso della classificazione delle cifre manoscritte, si dispone di un certo numero di campioni della forma $(x_i, y_i)$.
[00:57] In questo contesto, i vettori $x_i$ rappresentano, ad esempio, le matrici delle immagini delle cifre, trasformate in vettori (processo di *flattening*), mentre gli $y_i$ sono le etichette corrispondenti (0, 1, 2, 3, ecc.). Ciò che si cerca è una funzione $f$ che, prendendo in input il vettore delle caratteristiche $x$, restituisca l'output $y$.
[01:11] Questa è la formalizzazione del problema. La natura di questa funzione $f$ è data dalla rete neurale, che non è altro che una complessa combinazione di funzioni. Specificamente, è una composizione di funzioni di attivazione applicate a una combinazione lineare degli input provenienti dal livello precedente, pesati e sommati ai bias.
[01:28] La struttura della rete neurale definisce quindi una funzione composita. Ora, la domanda cruciale è: se questi dati provenissero da una funzione ideale ed esatta, che chiameremo $f$, la nostra funzione approssimata dalla rete neurale, che indicheremo con $f_{NN}$ per sottolinearne l'origine, è in grado di rappresentare qualsiasi tipo di funzione $f$?
[01:48] In altre parole, si vuole capire se questo formalismo sia capace di rappresentare qualunque funzione $f$. Una formulazione più rigorosa della stessa domanda è: questa tecnica è un **approssimatore universale**? Per rispondere a questa domanda, verranno seguiti due approcci.
## Approccio alla Dimostrazione: Visivo e Formale
[02:03] Il primo approccio, che verrà analizzato in questa sezione, è di natura più intuitiva e grafica. Successivamente, si affronterà un approccio formale. L'idea di base è supporre che la funzione originale $f$ sia una funzione continua.
[02:16] Per la dimostrazione visiva, si inizierà considerando il caso di una funzione $f$ monodimensionale (1D), ovvero una curva nel piano. È necessario essere consapevoli di due aspetti importanti del teorema di approssimazione universale, specialmente nella sua parte teorica.
[02:28] In primo luogo, il teorema è un risultato di esistenza: afferma che è sempre possibile trovare una rete neurale di un certo tipo che approssimi una data funzione. Tuttavia, il teorema non fornisce alcuna indicazione su come costruire concretamente tale rete.
[02:43] Si tratta quindi solo di una prova di esistenza, senza suggerimenti algoritmici o costruttivi su come procedere. Il secondo punto, che verrà ripreso in seguito, riguarda l'accuratezza desiderata. Quando si vuole approssimare una funzione $f$ con un'altra funzione, è necessario specificare anche il grado di precisione richiesto.
[03:00] Se si chiama $\tilde{f}$ la funzione approssimata, è necessario misurare la differenza tra la funzione originale e quella approssimata, ad esempio tramite una norma, e assicurarsi che sia inferiore a una data tolleranza $\epsilon$:
```math
\| f - \tilde{f} \| < \epsilon
```
La capacità di costruire una rete dipende quindi da quanto fine deve essere l'accuratezza desiderata.
[03:16] Il secondo punto implica che, a seconda della precisione che si vuole raggiungere, la costruzione della rete potrebbe essere fattibile o, al contrario, computazionalmente irrealizzabile. Questo aspetto è strettamente legato alla cosiddetta **complessità** di una rete neurale.
[03:28] In altri termini, l'idea è: data una funzione $f$ e una tolleranza $\epsilon$, si vuole avere un'idea di quanti parametri (pesi e bias) sono necessari per progettare una rete neurale in grado di rappresentare $f$ con tale accuratezza. In alcune situazioni, il numero di parametri richiesto può essere enorme.
## Dimostrazione Visiva con la Funzione Sigmoide
[03:47] L'idea di base della dimostrazione visiva è costruttiva, il che significa che alla fine si otterrà una sorta di algoritmo che indicherà come procedere per creare la rete neurale.
[04:01] L'idea è partire da una delle funzioni di attivazione già viste: la **funzione sigmoide**. Si analizzerà l'effetto della modifica dei suoi parametri. Se $\sigma$ è una funzione sigmoide, il suo argomento è $wx + b$. Si vuole osservare cosa accade alla forma della funzione quando si modificano i valori di $w$ e $b$.
[04:16] Si vedrà che, scegliendo opportunamente $w$ e $b$, si è in grado di trasformare una funzione sigmoide in una funzione a gradino (*step function*), o in un'approssimazione di essa. Successivamente, componendo due funzioni a gradino, si riuscirà a costruire una sorta di funzione rettangolare.
[04:32] Infine, componendo molte di queste funzioni rettangolari, si sarà in grado di approssimare qualsiasi funzione data.
### Manipolazione dei Parametri della Sigmoide
[04:41] Viene rappresentata la funzione sigmoide con parametri $w=1$ e $b=0$. Si vuole ora osservare l'effetto di $w$ e $b$. In particolare, interessa vedere cosa succede quando $w$ diventa sempre più grande.
[04:55] Aumentando il valore di $w$, si può notare che la funzione sigmoide si avvicina sempre di più a una funzione a gradino. La pendenza della curva diventa sempre più ripida.
[05:08] Utilizzando un valore elevato, ad esempio $w=50$, e modificando il valore di $b$, ad esempio impostando $b=25$, si osserva che la posizione del "salto" si è spostata.
[05:25] Per vederlo più chiaramente, si usa $w=100$. Con $b=200$ (implicito dal grafico mostrato), il salto si verifica in $x=-2$. La posizione del salto, che chiamiamo $x_{salto}$, è data dalla formula:
```math
x_{salto} = -\frac{b}{w}
```
[05:37] Giocando con i parametri $w$ e $b$, si è passati da una funzione sigmoide a una buona approssimazione di una funzione a gradino.
[05:46] Successivamente, mantenendo $w$ sufficientemente grande e variando $b$, è possibile spostare a piacimento la posizione del salto, secondo la relazione $x_{salto} = -b/w$.
[05:55] L'input della funzione è $x$, che proviene dal livello precedente della rete, mentre $w$ e $b$ sono i pesi e il bias. Ciò che si sta dimostrando è che, manipolando $w$ e $b$, è possibile modificare la forma della funzione di attivazione.
[06:07] La funzione rimane una sigmoide, ma la sua pendenza è cambiata drasticamente. L'input è $x$, mentre $w$ e $b$ sono parametri che, in una vera rete neurale, vengono fissati e poi modificati durante l'addestramento per trovare il minimo della funzione di costo.
[06:22] Questi sono esattamente i parametri che vengono ottimizzati per minimizzare la funzione di costo. In questo contesto, l'operazione viene eseguita "a mano" o graficamente, per dimostrare che, data una funzione, è possibile trovare un insieme adeguato di pesi e bias che ne generi un'approssimazione.
[06:40] Questo processo è analogo a ciò che avviene durante la procedura di addestramento. Qui si sta considerando una singola funzione (un singolo neurone).
### Costruzione di una Funzione Rettangolare
[06:49] Si è visto che per un valore di $w$ sufficientemente grande, la sigmoide approssima una funzione a gradino, e la posizione del salto è in $x = -b/w$. Ora, l'obiettivo è creare una funzione a forma di onda quadra, o rettangolo.
[07:04] Per fare ciò, non è sufficiente un solo neurone, ma ne servono due. Come indicato, si dovranno usare due pesi per i due neuroni, che in generale devono avere segno opposto (ad esempio, $H$ e $-H$).
[07:17] Tornando all'implementazione, si ha la possibilità di comporre due funzioni del tipo appena visto. In questa simulazione, si assume già che i pesi $w$ siano sufficientemente grandi (ad esempio, 500), garantendo che la sigmoide sia una buona approssimazione di una funzione a gradino.
[07:32] Negli slider sono presenti i parametri $S_1$ e $S_2$, che rappresentano le posizioni dei salti. Conoscendo $w$ e $S$, è possibile calcolare i bias $b_1$ e $b_2$ per le due funzioni. Infine, sono presenti $W_{01}$ e $W_{02}$, che sono i due pesi della combinazione lineare (indicati come $-H$ e $H$ in precedenza).
[07:51] Si inizia impostando i due pesi, ad esempio a $1$ e $-1$. Successivamente, si impostano le posizioni dei salti $S_1$ e $S_2$.
[08:01] Ad esempio, se si vuole un "impulso" tra 0 e 1, si possono impostare i salti in queste posizioni. Componendo queste due funzioni a gradino, si è riusciti a ottenere questo tipo di funzione rettangolare.
### Analogia con l'Integrazione di Riemann
[08:14] Questo tipo di approssimazione richiama un concetto già noto: l'integrazione. Quando si definisce l'integrale di Riemann, si utilizzano le somme superiori e inferiori, costruite suddividendo il dominio in piccoli rettangoli.
[08:29] Qui, in pratica, si sta facendo qualcosa di molto simile. L'idea è che, invece di definire graficamente i rettangoli, si sta sfruttando la capacità della rete neurale, e in particolare della funzione di attivazione, di costruirli.
[08:41] Nel caso dell'integrale di Riemann, si suddivide l'asse $x$ in intervalli di uguale ampiezza. Per definire le somme inferiori e superiori, in ogni intervallo si sceglie il valore minimo e massimo della funzione per determinare l'altezza dei rettangoli.
[08:56] Le aree dei rettangoli costruiti con i valori minimi vengono sommate per ottenere la somma inferiore; lo stesso si fa con i valori massimi per la somma superiore. Se, al tendere a zero dell'ampiezza degli intervalli ($\Delta x \to 0$), i valori della somma inferiore e superiore convergono allo stesso limite, quel valore è l'integrale di Riemann della funzione.
[09:14] Qui si sta seguendo un approccio simile, ma invece di usare direttamente i valori della funzione, si sfrutta la funzione di attivazione per costruire i rettangoli. C'è un altro dettaglio importante: l'altezza del rettangolo costruito è pari a 1.
[09:28] Questo valore, in generale, non è modificabile direttamente. Tuttavia, osservando l'espressione utilizzata, si ha una combinazione lineare $W_{01}y_1 + W_{02}y_2$, dove $y_1$ e $y_2$ sono gli output delle funzioni sigmoide.
[09:44] Ciò che si può fare è introdurre un parametro aggiuntivo al di fuori di questa combinazione lineare, ad esempio nel livello di output finale della rete, che agisca come un fattore di scala. Se, ad esempio, si moltiplica il risultato per 2.2, si ottiene un rettangolo con altezza 2.2. Si possono anche usare valori negativi. In questo modo, è possibile regolare l'altezza del rettangolo per farla corrispondere esattamente al valore desiderato della funzione in una certa regione.
### Approssimazione di una Funzione Generica
[10:04] Sfruttando questo fattore di scala, è possibile generalizzare l'idea. Sono stati costruiti molti rettangoli di questo tipo, ciascuno con un'altezza specifica, scelta ad esempio come il valore della funzione all'estremo sinistro dell'intervallo (ma potrebbe essere il punto medio o qualsiasi altro criterio).
[10:17] L'aspetto fondamentale è che, utilizzando rettangoli con una base sempre più piccola, è possibile migliorare l'approssimazione. In termini di rete neurale, ridurre la base dei rettangoli significa aumentare il numero di neuroni nel livello nascosto.
[10:27] Ad esempio, per questa approssimazione sono stati usati circa 10 neuroni. Per costruire ogni rettangolo, infatti, ne servono due. Avendo cinque rettangoli, il totale è di 10 neuroni (più eventualmente uno per il fattore di scala). Con 15 rettangoli, servono circa 30 neuroni; con 50 rettangoli, ne servono circa 100 per ottenere quella rappresentazione.
[10:48] In ogni caso, questo dimostra che, sfruttando questa idea, è possibile approssimare una funzione con accuratezza arbitraria semplicemente aumentando il numero di neuroni. Questa è una rappresentazione visiva della proprietà di approssimazione universale.
### Estensione al Caso Bidimensionale (2D)
[11:01] Questa idea può essere estesa al caso bidimensionale (2D). Invece di approssimare una curva con rettangoli, si approssima una superficie con dei parallelepipedi. Per costruire ogni parallelepipedo, sono necessarie quattro funzioni a gradino.
[11:15] In questo modo, si può ottenere una rappresentazione "a scala" della superficie, in modo analogo al caso 1D.
## Dimostrazione Visiva con la Funzione ReLU
[11:21] Un punto importante da considerare è che, nella pratica, le reti neurali spesso non utilizzano la funzione sigmoide. Sebbene sia usata, una delle funzioni di attivazione più comuni è la **ReLU (Rectified Linear Unit)**.
[11:33] È possibile dimostrare, e lo si vedrà formalmente, che il teorema di approssimazione universale è valido per qualsiasi funzione di attivazione non lineare che sia "a forma di S". In particolare, è valido per qualsiasi funzione **sigmoidale**.
[11:45] Una funzione sigmoidale è una generalizzazione della sigmoide: è una funzione che tende a 0 per $x \to -\infty$ e a 1 per $x \to +\infty$. Non deve essere necessariamente la funzione sigmoide classica.
[11:58] Tuttavia, in molte reti neurali si utilizza la funzione ReLU. La ReLU è stata introdotta come la funzione definita da $\max(0, z)$. Il suo grafico è una rampa: è zero per valori negativi e cresce linearmente per valori positivi.
[12:13] Come si può fornire una dimostrazione visiva della proprietà di approssimazione universale se si usa la ReLU invece della sigmoide? La ReLU, infatti, non è una funzione sigmoidale, poiché è illimitata a destra.
[12:28] L'idea è simile a quella vista per la funzione sigmoide, ma verrà presentata in modo leggermente diverso.
### Costruzione di una Funzione "a Cappello" (Hat Function)
[12:34] In pratica, partendo dalla funzione ReLU, e combinando un numero adeguato di queste funzioni, si costruirà una funzione con una forma a "cappello" (o triangolare). Questa funzione sarà, ad esempio, nulla al di fuori dell'intervallo $[-1, 1]$ e avrà un picco di valore 1 in $x=0$.
[12:52] Questa particolare forma di funzione dovrebbe ricordare gli elementi finiti P1. Nel metodo degli elementi finiti in 1D, questa è la tipica **funzione a cappello** (*hat function*) utilizzata per implementare elementi finiti lineari.
[13:05] È una funzione lineare a tratti (*piecewise linear*) e, in particolare, ha un supporto compatto, ovvero è diversa da zero solo in un intervallo limitato (in questo caso, $[-1, 1]$).
[13:17] Attraverso una combinazione lineare di funzioni di questo tipo, è possibile rappresentare un'altra funzione. Se si chiama questa funzione $\phi(x)$, e si suddivide un dominio (es. $[0, 1]$) in nodi, è possibile definire una funzione a cappello centrata su ciascun nodo.
[13:34] Assegnando un indice a ciascuna di queste funzioni base ($\phi_0, \phi_1, \dots, \phi_6$), e sapendo che il loro valore massimo è 1, qualsiasi funzione $f(x)$ può essere approssimata come una somma pesata:
```math
\tilde{f}(x) = \sum_{i=0}^{6} c_i \phi_i(x)
```
[13:44] dove $\phi_i(x)$ è una delle funzioni a cappello e $c_i$ è un coefficiente. Ad esempio, per approssimare una data funzione, il coefficiente $c_i$ può essere scelto come il valore della funzione nel nodo $i$. In questo modo, quando si valuta la combinazione lineare in un nodo specifico, si ottiene il valore esatto della funzione in quel punto, poiché tutte le altre funzioni base sono nulle.
[14:04] Questa è un'approssimazione $\tilde{f}(x)$ della funzione $f(x)$. Aumentando il numero di nodi, si aumenta l'accuratezza dell'approssimazione. L'idea, quindi, è sfruttare questa analogia con gli elementi finiti e vedere se è possibile ottenere la funzione a cappello partendo dalla funzione ReLU.
### Dalla ReLU alla Funzione a Cappello
[14:21] Esattamente come fatto in precedenza, si analizza l'effetto dei parametri $w$ e $b$ sulla funzione ReLU. Mantenendo $w=1$ e variando $b$, si osserva che si sposta la posizione del punto di "svolta" (*king point*).
[14:32] In particolare, se $b$ è positivo, il punto di svolta si trova nella regione negativa dell'asse $x$, e viceversa. Ora, fissando $b$ (ad esempio a 6) e aumentando $w$, si modifica la pendenza della rampa e si sposta anche la posizione del punto di svolta.
[14:48] Come si può vedere, la posizione del punto di svolta è data, anche in questo caso, dalla formula:
```math
x_{svolta} = -\frac{b}{w}
```
[14:56] Vediamo ora come si possono combinare le funzioni ReLU per ottenere la funzione a cappello. Una volta dimostrato che è possibile costruire questa funzione base utilizzando un numero adeguato di ReLU, si può concludere, per analogia con gli elementi finiti, che una combinazione di ReLU può approssimare qualsiasi funzione 1D.
[15:11] In questo caso, si combineranno tre funzioni ReLU.
[15:18] Partendo con una certa configurazione, si può ottenere una funzione simile a una sigmoide. Qui, ad esempio, ne sono state combinate solo due, poiché il peso della terza ($W_{03}$) è zero.
[15:25] Combinando due funzioni ReLU, si è stati in grado di ottenere una funzione sigmoidale. Quindi, una prima possibilità con la ReLU è sfruttare il fatto che la combinazione di due ReLU può generare una sigmoide. Avvicinando i punti di svolta, si può far sì che questa sigmoide approssimi una funzione a gradino.
[15:40] Una volta ottenuta la funzione a gradino, se ne può usare un'altra (costruita con altre due ReLU) e combinarle per creare il rettangolo visto in precedenza. Questa è una possibilità, ma richiederebbe quattro funzioni ReLU per ogni rettangolo (due per il primo gradino e due per il secondo).
[15:57] Si vuole esplorare un approccio diverso per ridurre il numero di funzioni necessarie e, inoltre, ottenere un risultato migliore: un'approssimazione lineare a tratti invece che costante a tratti, quindi più accurata.
[16:09] L'idea è di far entrare in gioco anche la terza funzione ReLU. Si mantengono i bias delle tre funzioni a 1, 0 e -1, e si usano i pesi 1, 1 e -2. Il peso -2 fa sì che la terza componente "scenda". L'obiettivo è rendere piatta la regione a destra del picco.
[16:30] In questo modo, è stata costruita esattamente la funzione a cappello. È una funzione che è zero al di fuori di un intervallo, lineare a tratti su un supporto compatto. Con il fattore di scala $h$, è possibile regolare l'altezza della funzione a cappello a piacimento.
[16:50] Quindi, anche con le funzioni ReLU, che non sono sigmoidali (un elemento chiave nella dimostrazione formale che si vedrà), si è comunque in grado di fornire una prova visiva del fatto che una rete neurale può approssimare qualsiasi funzione in 1D.
[17:06] Nel caso 2D, l'analogia con gli elementi finiti è meno diretta, poiché le funzioni a cappello 2D sono definite su una base di triangoli, rendendo la costruzione più complessa. Ad esempio, su una griglia di triangoli, la funzione base è 1 in un nodo centrale e 0 in tutti gli altri nodi.
[17:23] Per il caso 2D, l'analogia visiva è più semplice utilizzando le funzioni a gradino. Tuttavia, si è già visto che la funzione a gradino può essere ottenuta anche a partire dalla ReLU. Di conseguenza, anche con la ReLU è possibile dimostrare l'approssimazione universale nel caso 2D.
[17:36] Nella prossima sezione si vedrà come passare da questa dimostrazione empirica e visiva a una dimostrazione formale.
[17:44] Grazie per l'attenzione.