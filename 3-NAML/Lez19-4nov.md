# Lezione 19 - Ottimizzazione per Deep Learning
**Data:** 4 Novembre
**Argomenti:** Convergenza Gradient Descent, Line Search, Ottimizzazione Vincolata, Stochastic Gradient Descent

---

## PARTE 1: ANALISI DI CONVERGENZA DEL GRADIENT DESCENT

### 1.1 Introduzione e Setup del Problema

00:00:02 
Okay, quindi possiamo ripartire dal punto dove siamo arrivati ieri. Solo come promemoria, la prima idea era **limitare l'errore tra i valori funzionali**.

**Obiettivo**: Siamo interessati a valutare qual è un limite (bound) per questa quantità:
$$f(x_t) - f(x^*)$$

Dove:
- $x_t$ = iterazione corrente
- $x^*$ = soluzione ottima
- $f$ = funzione obiettivo da minimizzare

### 1.2 Risultato Base dalla Convessità

00:00:34 
E assumendo **solo la convessità**, siamo stati in grado di arrivare a questa conclusione. Okay, questa formula è il punto di partenza per in realtà questa e questa. Sono i **due punti di partenza** per i seguenti risultati che vedremo.

**Strategia**: Questi risultati sono ottenuti aggiungendo **ulteriori ipotesi** sulla funzione che vogliamo minimizzare:
1. Convessità (base)
2. + Lipschitz continuità del gradiente
3. + L-smoothness
4. + Strong convexity

---

## PARTE 2: CASO LIPSCHITZ CONVESSO

### 2.1 Ipotesi e Setup

00:01:16 
Quindi iniziamo con il primo risultato. In particolare, qui, stiamo assumendo che abbiamo:

#### Ipotesi 1: Limitatezza del Gradiente (Lipschitz)
$$||\nabla f(x)|| \leq B \quad \forall x$$

La funzione ha **gradiente limitato** - la norma del gradiente è limitata da una costante $B$.

#### Ipotesi 2: Guess Iniziale Limitato

00:01:48 
Abbiamo anche un'ipotesi aggiuntiva sull'ipotesi iniziale:
$$||x_0 - x^*|| \leq R$$

Stiamo assumendo che l'ipotesi iniziale $x_0$ non sia troppo lontana dalla vera soluzione $x^*$, o almeno sia limitata, di nuovo, da una costante $R$ (capitale).

### 2.2 Teorema: Convergenza con Learning Rate Ottimale

In questa ipotesi, se scegliamo un **tasso di apprendimento** (learning rate) o passo adatto dato da:

$$\gamma^* = \frac{R}{B\sqrt{T}}$$

Dove:
- $T$ = numero di iterazioni (capitale T)
- $R$ = bound sulla distanza iniziale
- $B$ = bound sul gradiente

00:02:24 
Capitale T è il numero di iterazioni che eseguiremo del metodo della discesa del gradiente. Allora abbiamo questa stima per la differenza tra le iterate:

$$||x_{best} - x^*|| = O\left(\frac{RB}{\sqrt{T}}\right)$$

### 2.3 Velocità di Convergenza

È chiaro che se questo è vero, allora essenzialmente la finale, se vogliamo, valutazione della funzione meno la valutazione nella vera soluzione è dell'ordine di $\frac{1}{\sqrt{T}}$.

00:03:14 
Quindi, se vogliamo avere una soluzione che abbiamo chiamato $x_{best}$ qui, soddisfacente una **tolleranza** $\epsilon$:

$$f(x_{best}) - f(x^*) \leq \epsilon$$

Allora possiamo, sfruttando queste relazioni quindi vogliamo avere la differenza tra i due valori funzionali fino a un'accuratezza o tolleranza epsilon allora possiamo sfruttando.

00:03:47 
questa relazione o questa se volete potete **stimare il numero di iterazioni**:

$$T = O\left(\frac{R^2 B^2}{\epsilon^2}\right)$$

**Interpretazione**: Quindi significa che rispetto alla tolleranza che volete raggiungere, avete una **velocità di convergenza** dell'ordine di:

$$\boxed{O\left(\frac{1}{\epsilon^2}\right)}$$

00:04:21 
**Esempio pratico**: Il che significa che se volete, per esempio, una tolleranza dell'ordine di $\epsilon = 10^{-2}$, allora dovete eseguire, almeno teoricamente, un numero di iterazioni:

$$T \approx \frac{R^2 B^2}{(10^{-2})^2} = 10^4 \cdot R^2 B^2$$

che può essere piuttosto grande!

### 2.4 Dimostrazione (Sketch)

Okay, quindi qual è l'idea della dimostrazione di questo risultato?

00:05:11 
**Strategia**: Questa è abbastanza semplice. Se partite dal risultato dell'**analisi di base** (BA1), e aggiungete le ipotesi aggiuntive:

1. ✓ Gradiente limitato: $||\nabla f|| \leq B$
2. ✓ Guess iniziale limitato: $||x_0 - x^*|| \leq R$

Abbiamo che questo termine, il primo termine, può essere limitato usando queste ipotesi e il secondo può essere limitato usando il limite sull'ipotesi iniziale.

00:05:46 
Quindi abbiamo questa condizione e in realtà $B$ è una costante. Quindi qui abbiamo solo:

$$Q(\gamma) = \frac{\gamma B^2 T}{2} + \frac{R^2}{2\gamma}$$

#### Ottimizzazione del Learning Rate

Ora, se guardate il teorema, stiamo affermando che c'è un **valore ottimale** per il tasso di apprendimento/dimensione del passo. Quindi qual è l'idea?

00:06:18 
L'idea è considerare questa funzione $Q(\gamma)$, e vogliamo **minimizzare rispetto a $\gamma$** quella funzione:

$$\frac{dQ}{d\gamma} = \frac{B^2 T}{2} - \frac{R^2}{2\gamma^2} = 0$$

Calcolando la derivata di quello, il lato destro di questa disuguaglianza rispetto a $\gamma$, e la stiamo impostando a zero, in modo da poter ottenere:

$$\gamma^* = \frac{R}{B\sqrt{T}}$$

E se sostituite indietro nel lato destro, quello che avete è esattamente il risultato che avete qui.

00:06:56 
**Riepilogo Risultato 1**:
- ✓ Ipotesi: Convesso + Lipschitz
- ✓ Learning rate ottimale: $\gamma^* = \frac{R}{B\sqrt{T}}$
- ✓ Convergenza: $O\left(\frac{1}{\epsilon^2}\right)$
- ⚠️ Questo è il primo risultato, che è importante tenerlo a mente

Quindi. Questo è il primo risultato, che è, e questo è importante tenerlo a mente, dell'ordine di 1 su epsilon al quadrato, okay? Ovviamente, significa che l'errore è, l'errore medio diminuisce con un fattore che dipende dal quadrato, 1 su la radice quadrata di 2.

---

## PARTE 3: CASO SMOOTH CONVESSO (L-SMOOTH)

### 3.1 Condizione di L-Smoothness

00:07:39 
Cosa succede se aggiungiamo la condizione sulla **smoothness** della funzione?

**Definizione L-Smooth**: Una funzione $f$ è **L-smooth** se:
$$f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}||y-x||^2$$

**Interpretazione**: Il gradiente non cambia troppo velocemente - è Lipschitz continuo.

### 3.2 Scelta del Learning Rate

Quindi, qui assumiamo che la funzione sia smooth e che il tasso di apprendimento sia:

$$\gamma = \frac{1}{L}$$

dove $L$ è la costante di smoothness.

### 3.3 Proprietà di Sufficient Decrease

00:08:17 
Con questa scelta di $\gamma$, otteniamo la condizione di **sufficient decrease** (diminuzione sufficiente):

$$f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}||\nabla f(x_k)||^2$$

Questa condizione è solitamente chiamata diminuzione sufficiente. Quindi, significa che se la funzione è L-smooth, allora scegliendo questo $\gamma = \frac{1}{L}$, avete la garanzia che i valori funzionali che state costruendo usando il metodo della discesa del gradiente stanno diminuendo.

00:08:47 
Quindi f è uguale a f meno qualcosa che è positivo. Quindi avete effettivamente una sequenza decrescente di valori funzionali. Qui la dimostrazione è di nuovo abbastanza semplice. Se partite con la definizione della L-smoothness.

00:09:24 
E sostituite in quella relazione la relazione ricorsiva che definisce la discesa del gradiente con il tasso di apprendimento 1 su L, che è esattamente questa. Questo non è altro che il metodo della discesa del gradiente usando il tasso di apprendimento 1 su L. Allora avete che potete sostituire questa quantità nella precedente, quindi in particolare qui.

00:10:03 
Ora qui avete il gradiente trasposto per il gradiente, che è la norma del gradiente al quadrato, che è uguale a questo. Quindi qui avete meno 1 su L per il gradiente al quadrato più 1 su 2L gradiente al quadrato. E poi avete il gradiente trasposto per il gradiente al quadrato, che è la norma del gradiente al quadrato, che è la norma del gradiente al quadrato. E infine, avete il risultato, okay? Quindi è abbastanza facile ottenere la condizione di diminuzione sufficiente, okay?

00:10:52 
Ora passiamo al caso dove abbiamo una funzione che è effettivamente convessa e L-smooth. In questo caso, quindi il risultato precedente ci sta dicendo che, Per due iterazioni consecutive, abbiamo una sequenza decrescente.

00:11:23 
Qui, quello che vogliamo dimostrare è che quando eseguite capitale T iterazioni, la differenza tra i valori funzionali all'ultima iterazione meno i valori funzionali nella vera soluzione è limitata da una costante che dipende dal numero di iterazioni e dall'errore iniziale.

00:11:55 
Di nuovo, prima di dare un'occhiata alla dimostrazione, quello che possiamo notare è che ora, se... di nuovo assumiamo che l'ipotesi iniziale sia vicina alla vera soluzione o almeno sia limitata dalla costante r allora il risultato precedente può essere scritto come l r al quadrato su t su t.

00:12:27 
Quindi di nuovo se vogliamo raggiungere epsilon allora dovete eseguire un numero di iterazioni che è l r al quadrato su due epsilon. Quindi ora la differenza principale è che, è che la convergenza, quindi il numero di iterazioni, non è uno su epsilon al quadrato ma.

00:12:58 
uno su epsilon. Quindi aggiungendo ulteriori condizioni sulla funzione f, ovviamente stiamo ottenendo una migliore velocità di convergenza. Quindi qual è l'idea della dimostrazione? Di nuovo, partiamo dal risultato di base dove abbiamo solo usato gamma uguale a uno su.

00:13:32 
capitale L. Poi possiamo usare la condizione di diminuzione sufficiente. che può essere usata o interpretata anche come un limite sul gradiente. Quindi il gradiente è limitato dalla differenza tra due iterazioni consecutive. Quindi e quindi se partiamo da questa e sommiamo su tutte le iterazioni, abbiamo questa disuguaglianza.

00:14:13 
Quindi qui abbiamo la somma di tutti i gradienti e qui abbiamo la somma della differenza tra due iterazioni consecutive. Come al solito, l'idea è che qui abbiamo una somma telescopica, quindi in due termini consecutivi della somma, due valori funzionali si cancellano.

00:14:43 
Quindi, Il risultato finale della somma telescopica è solo la differenza tra l'ipotesi iniziale, il valore funzionale sull'ipotesi iniziale, meno il valore funzionale sull'ultima iterazione. E quindi mettendo tutto insieme, abbiamo questo limite sull'intersezione media del gradiente.

00:15:17 
Quindi possiamo considerare l'equazione originale, quella ottenuta dall'analisi di base, e possiamo inserire questo termine qui. Quindi ora... Là, e ora abbiamo questa disuguaglianza, che, ricordate che quello che vogliamo dimostrare è che questa condizione è vera, okay?

00:15:59 
Quindi, qui, essenzialmente, potete portare i valori funzionali sul lato sinistro. Di nuovo, qui avete una somma telescopica, e con il valore iniziale e finale là, che è dato limitato da L su 2 per l'errore sull'ipotesi iniziale.

00:16:35 
Poiché sappiamo che abbiamo la condizione di diminuzione sufficiente, quindi il valore funzionale in t più 1 è minore o uguale al valore funzionale in t, questo valore è il più piccolo tra tutti i valori che possiamo calcolare lungo la procedura. Quindi significa che possiamo dire che invece di prendere questa somma, possiamo trovare che questa somma è maggiore o uguale a capitale T per questo valore, che è il più piccolo.

00:17:23 
E quindi... Combinando i due risultati, abbiamo che la differenza tra la funzione e il valore nell'iterazione finale meno la funzione e il valore nella vera soluzione è limitata dall'errore iniziale, ovviamente su x.

00:17:55 
Ora, se ricordate ieri, abbiamo aggiunto un'ulteriore condizione, che era quella della convessità fortemente mu o convessità. Quindi ora assumiamo che la funzione sia L smooth e mu fortemente convessa.

00:18:28 
Quello che vogliamo ottenere qui sono due risultati, non solo sui valori funzionali, ma anche sulla distanza tra le iterate all'indice t più 1 e la vera soluzione. Quindi stiamo anche limitando i valori della soluzione, non solo dei valori funzionali.

00:19:07 
In qualche modo, questo numero, mu su l su mu, può essere interpretato come il numero di condizionamento del problema. E qui, se consideriamo questa espressione, ed eseguiamo essenzialmente la stessa operazione che abbiamo visto prima, quello che possiamo dimostrare è che il numero di passi richiesti per raggiungere una data tolleranza è ora dell'ordine di k,

00:19:43 
che è il numero di condizionamento, per il logaritmo di 1 su epsilon. Immagino che siate tutti familiari con il concetto di numero di condizionamento. Cos'è un numero di condizionamento in generale? Qual è il significato di un numero di condizionamento per un problema numerico? Quindi se avete un problema numerico qualsiasi, che può essere anche un problema f di x d uguale a zero,

00:20:25 
00:20:25 
dove f è la relazione funzionale che collega i dati all'incognita. Okay, quindi questa è una rappresentazione totalmente astratta di qualsiasi problema che potete immaginare. Il numero di condizionamento è un numero che vi dice essenzialmente quanto è sensibile la soluzione.

00:20:59 
rispetto a variazioni sui dati. Okay. E questo è, per esempio, se ricordate, immagino che abbiate visto sicuramente questo concetto quando avete studiato i sistemi lineari per le matrici, potete definire un numero di condizionamento. Quindi anche per qualsiasi matrice quadrata A, il numero di condizionamento della matrice è dato dalla norma di A per la norma di A alla meno uno, dove la norma può essere qualsiasi norma matriciale.

00:21:43 
OK, questo numero è maggiore o uguale a uno, e di solito è adottato come misura della sensibilità alla perturbazione nei dati. soluzione medica del problema perché per esempio se state usando il metodo del gradiente o il metodo del gradiente coniugato il numero di iterazioni che dovete eseguire per.

00:22:19 
raggiungere una data tolleranza o se volete la costante che riduce l'errore ad ogni passo dell'iterazione dipende dal numero di condizionamento per il gradiente dipende dal numero di condizionamento per il gradiente coniugato dipende dalla radice quadrata del numero di condizionamento ma comunque, influenza anche la velocità di convergenza del metodo che è esattamente quello che abbiamo qui o qui.

00:22:50 
okay quindi è esattamente quello che stiamo trovando anche in questo caso quindi in generale quando trovate il termine numero di condizionamento, dovete sempre pensare alla sensibilità della soluzione rispetto alla perturbazione. La perturbazione può essere sui dati o anche, per esempio, in una matrice, quando avete un sistema lineare uguale a B,

00:23:31 
la perturbazione può essere sul lato destro, ma può essere anche perturbazione sulla matrice stessa. Quindi se perturbate leggermente un termine della matrice, qual è il risultato sulla soluzione? Forse avete visto che c'è un esempio famoso nel contesto delle matrici, che è la matrice di Hilbert, che è un esempio tipico di una matrice che è mal condizionata.

00:24:04 
Quindi significa che se modificate leggermente B, ottenete una soluzione che è totalmente diversa dalla precedente. Okay, quindi qual è l'idea della dimostrazione di questo risultato? Qui stiamo usando, prima di tutto, la prima relazione che abbiamo ottenuto nell'analisi di base.

00:24:43 
Ricordate che vi ho detto pochi minuti fa che... Dall'analisi di base dove stavamo per usare essenzialmente due risultati. Uno è il risultato riguardante il limite sulla media della differenza tra valori funzionali e il secondo è relativo al fatto che essenzialmente il gradiente, il prodotto scalare tra il gradiente e il vettore.

00:25:17 
che vi dà la differenza tra l'iterazione corrente e la vera soluzione. E questo è dato dall'espressione che avete là. Qui abbiamo sostituito. È solo scritto nella forma che useremo. E qui abbiamo solo riportato. La definizione di complessità forte.

00:25:50 
Nella condizione di convessità forte, che è questa, che è valida per qualsiasi coppia di punti, y e x, stiamo scegliendo x uguale all'iterazione corrente e y uguale alla vera soluzione. Quindi se sostituite questi due valori qui e riorganizzate la disuguaglianza, quello che ottenete è un limite di questo termine.

00:26:25 
Quindi potete trovare che questo gradiente per xd meno x star è maggiore o uguale a questa espressione, che coinvolge la costante mu che caratterizza la convessità forte mu della funzione. Quindi, ora sfrutteremo la prima relazione e questa. Quindi, BA1 e SC2.

00:27:02 
Quindi, se mettete insieme le due condizioni, avete la relazione nella parte superiore della slide che, riorganizzata, può essere usata per limitare questa quantità. che è l'ultimo termine okay quindi essenzialmente qui stiamo limitando l'errore all'iterazione t più uno.

00:27:34 
con qualcosa che dipende dall'errore all'iterazione precedente più qualcosa relativo al gradiente e ai valori funzionali la differenza tra i valori funzionali e l'iterazione precedente quindi ora chiamiamo questo termine gli ultimi due termini rumore rumore nel senso che è, qualcosa che sta uh uh perturbando questa quantità che è qualcosa che.

00:28:06 
ci si aspetta sia il termine buono in che senso, Questo è l'errore precedente. Questa è una costante che in generale è minore di uno. Quindi questo è un fattore che sta riducendo l'errore ogni volta. Quindi ora consideriamo questi due termini e vogliamo mostrare che il rumore anche per questi due termini è negativo.

00:28:46 
Quando la dimensione del passo, il tasso di apprendimento è dato da, come al solito, uno su L. Quindi sfruttiamo di nuovo la condizione di diminuzione sufficiente. Quindi abbiamo che la valutazione della funzione a t più uno è minore della precedente meno. costante. Sappiamo che x star è il minimo quindi significa che come abbiamo.

00:29:19 
notato prima questa condizione è soddisfatta quindi abbiamo che la condizione precedente può dovrebbe essere verificata anche quando invece di t più 1 stiamo scegliendo x star. Quindi abbiamo questa condizione f di x star meno f di x t è limitata da questa costante. Perché questo è utile? Perché abbiamo esattamente quel termine.

00:29:55 
qui. Quindi ora stiamo prendendo il risultato precedente, quello che abbiamo appena ottenuto, e lo stiamo mettendo qui. Gamma è uno su L, quindi stiamo mettendo gamma uno su L lo mettiamo qui, e quello che abbiamo è esattamente lo stesso termine che è uguale a.

00:30:28 
zero. Quindi il rumore, la somma dei due termini che abbiamo nella relazione precedente qui, questi due termini sono negativi. Quindi significa che questo è un termine che riduce l'errore ad ogni iterazione, e poiché questo è negativo, siamo in buona posizione. Quindi ora ricordate che il gioco vuole prima.

00:31:07 
dimostrare il risultato sulla distanza tra l'iterazione corrente e la vera soluzione. Quindi se consideriamo l'iterazione, questa relazione dove sappiamo che questo è negativo, possiamo trovare che in realtà x t più 1 meno x star.

00:31:44 
al quadrato è minore o uguale a questo termine e poi se sostituite gamma con 1 su L ottenete esattamente, questa relazione. Se applicate ricorsivamente questa relazione, ottenete che questo termine è, avete il termine uno meno mu su L alla potenza capitale T per l'ipotesi iniziale,

00:32:18 
l'errore dovuto all'ipotesi iniziale. Per la seconda parte, quindi in, vogliamo limitare i valori della funzione qui. Quello che stiamo usando è la caratterizzazione della L-smoothness, la definizione.

00:32:57 
Quindi notiamo che il gradiente quando valutato nel minimo della funzione è ovviamente zero, e se sostituite il risultato per questo termine che abbiamo ottenuto nel precedente, nel punto uno qui, quello che ottenete è esattamente il risultato finale. Quindi quello che avete ottenuto è che sia la distanza che i valori funzionali sono limitati dalla costante per l'iterazione precedente, o ovviamente anche in questo caso potete iterare come abbiamo fatto qui.

00:33:52 
potete arrivare all'ipotesi iniziale e lo stesso per i valori funzionali. Quindi per riassumere quello che abbiamo ottenuto, qui c'è una tabella in cui avete le diverse caratteristiche, della funzione che abbiamo assunto, partendo dalla Lipschitz convessa. Abbiamo visto che il.

00:34:25 
il numero di iterazioni per ottenere una data tolleranza è dell'ordine di 1 su epsilon al quadrato. Per la smooth convessa, abbiamo uno su epsilon, e per la smooth e fortemente convessa, abbiamo k, che è il numero di condizionamento, per il logaritmo di uno su epsilon. Quindi stiamo, essenzialmente, aggiungendo proprietà, quindi stiamo considerando funzioni che si comportano meglio, stiamo migliorando la velocità di convergenza dell'algoritmo.

00:35:16 
Qui, in tutte queste dimostrazioni, abbiamo... Assunto che il tasso di apprendimento è ottenuto, è dato, al tasso di apprendimento è dato un valore particolare, sia 1 su L, o nell'altro caso, nel primo caso era qui, era R su B per la radice quadrata di T.

00:35:47 
In ogni caso, in questa dimostrazione, il fatto che il tasso di apprendimento ha un valore particolare specifico è di fondamentale importanza per ottenere i risultati teorici. Cosa succede in pratica? È chiaro che, in pratica, molto spesso, o direi nella maggior parte delle situazioni, non sappiamo cos'è L, cos'è B, cos'è R, e così via.

00:36:25 
Quindi la domanda è, c'è un metodo per trovare un valore ragionevole per gamma per il tasso di apprendimento? Abbiamo visto ieri che il tasso di apprendimento è uno dei cosiddetti iperparametri, e il calcolo del suo valore, di solito viene fatto con una procedura per tentativi ed errori, può essere.

00:36:57 
guidato in qualche modo ispezionando l'andamento di validazione della funzione di costo. Ma, dovrebbe essere ottimizzato. In realtà, ci sono, c'è un metodo che a volte viene usato, si chiama metodo della line search. Qual è l'idea del metodo della line search? L'idea è, partendo dalla solita relazione ricorsiva per la discesa del gradiente,

00:37:39 
vogliamo scegliere gamma in modo intelligente ad ogni passo temporale. Quindi, prima differenza rispetto a quello che abbiamo fatto nell'analisi teorica, non stiamo scegliendo un singolo valore per gamma. Gamma è, direi, adattivo. Ad ogni iterazione, sceglieremo una dimensione del passo diversa.

00:38:10 
E la domanda è, come possiamo farlo? L'idea è, se partite dall'iterazione corrente, e avete il gradiente in, ricordate che dk è meno il gradiente all'iterazione k. Quindi, vogliamo trovare gamma che minimizza quella funzione 1d phi di gamma.

00:38:56 
E per eseguire questa minimizzazione, abbiamo, di solito ci sono due strategie. La prima è chiamata exact line search, in cui quello che volete fare è trovare il minimo esatto di phi. La seconda è la cosiddetta inexact line search, o backtracking line search, che è un metodo in cui state cercando un'approssimazione del valore di gamma che minimizza quella funzione.

00:39:42 
Iniziamo con l'exact line search. Quindi l'idea è abbiamo una funzione. Abbiamo una funzione phi di gamma, e vogliamo trovare... Tra tutti i possibili gamma positivi, quello che minimizza la funzione f di xd più gamma dk. Quindi dovete tenere a mente che siete all'iterazione k, quindi siamo all'iterazione k, xk è noto, dk è, dk è, quindi questi sono noti, okay?

00:40:35 
Quindi questi sono vettori, e sono noti. Quindi significa che quando costruite questa quantità, dovete tenere a mente che dovete tenere a mente, Questo è in realtà qualcosa in cui l'unica incognita è il parametro Gamma. Okay, questa è la ragione per cui nella slide precedente abbiamo detto che phi di Gamma è una 1D, è una funzione in 1D, quindi è un problema di minimizzazione 1D.

00:41:08 
E quello ovviamente dovete applicare la funzione f. Quindi dobbiamo impostare la phi derivata di Gamma uguale a zero. Se calcolate, se usate la regola della catena, avete che il gradiente di f in xk più Gamma d trasposto per dk.

00:41:42 
So, since we want to have phi prime of gamma equal to zero, what we want to achieve is to find the gamma such that this condition is true. Remember that this is x k plus one. Okay? What is the geometrical interpretation of this condition? It means that the new gradient should be orthogonal to the previous search direction decay.

00:42:29 
And what are the advantages of this method? It is clear that if the function f is simple, Then, you can come up with an explicit expression for gamma that minimizes the function phi. If the function f is not very simple, it's complicated, it's not easy or sometimes it's not possible to obtain an explicit value for gamma.

00:43:18 
If you want to solve the minimization problem, if f is simple, it can be obtained by a closed formula. If f is not simple, the exact minimization can be very, very difficult to achieve. In practice, even if it has some advantages, in the sense that by finding gamma, which minimizes the function phi, you are obtaining the best possible gamma.

00:44:03 
So the one that ensures the maximum reduction of the functional values. And obviously, it's a consequence, you are also finding the method for which you have to perform the smallest number of iterations to achieve the same accuracy. In practice, it's never used, because in practical applications, the function is not simple.

00:44:39 
So. Thank you very much. This is something that you can do on paper just for understanding what is the idea, but in practice is never. What is the inexact line search or backtracking line search? It's an approximate method that has the same aim as the exact line search.

00:45:14 
But we are not looking for the exact gamma. But we want to, as usual, create an iterative procedure that essentially is giving us a sequence of gamma. And then we can choose to stop at a certain point and use that particular gamma for making the next step.

00:45:51 
What are the steps? So what is the algorithm? You have to choose a first guess for gamma. Let's call that gamma bar. And then there are usually two constants in this method that are usually C and so C. Both are in the interval zero one. The first one is a constant that ensures that by using the gamma that you are going to compute, you will have a decrease in the functional value.

00:46:35 
And so is the shrink factor. So it's a factor that is used in this step in order to reduce the value of gamma that you have guessed at the previous iteration. So we said gamma equal to gamma bar, the starting point. And if by using this gamma.

00:47:06 
We see that this condition is satisfied, what does it mean? It means essentially that the new. The function of value on the new iteration is greater than the previous one, okay? If this is true, then we have to shrink to reduce the value of gamma, because this condition essentially is telling us that we are not reducing the function of value.

00:47:43 
When this condition is not fulfilled, then we have found a good gamma. And so we set gamma k, so the step size or learning rate that we are going to use at the kth iteration of the gradient descent, equal to the gamma that we have computed here. So essentially, it means that within your learning procedure, you have...

00:48:14 
Yeah, yeah. The iteration of the gradient and at each iteration of the gradient, you have some sub iterations for computing the, let's say, the best approximate gamma for making the next step. Okay. Okay. Usually, the backtracking line search is available in machine learning libraries. It's one of the possibility for choosing gamma or the other possibility.

00:49:10 
Is to use a so-called scheduling of for gamma. So what is the idea? The idea is that I can say, for example, that gamma step k is equal to 1 over k. So the idea behind this scheduling is that I want, when I'm going towards the solution, I want to pick smaller and smaller steps in order to better approximate the solution.

00:49:51 
This is just an instance of a possible choice. So what are the advantages of this backtracking? It's practical, it's efficient, it's an iterative procedure, it is robust in the sense that in most of the cases you are sure that you are getting a value of gamma which is reasonable, and it is used in many packages or practical applications.

00:50:25 
What are the drawbacks? You have additional human parameters, C, tau, and gamma bar. So if on one hand you are finding a good value of gamma, which is an important hyperparameter that we have already mentioned, on the other hand you are introducing at least these two other constants that you have to guess.

00:50:57 
What are the drawbacks? I have to say that it's true that in principle, the values for C and O can be adapted according to different functions and different situations. But in practice, even if you look at implementation in TensorFlow or PyTorch, most of the time, these are the default values for C and O, which that works well for, I would say, almost all the interesting applications.

00:51:34 
And obviously, since it is not the exact... value of gamma that gives you the minimum of t, the number of iterations that you have to perform in order to obtain the same accuracy as in the previous is higher because gamma computed according to the backtracking line search is not the gamma that you can compute with the exact line search.

00:52:13 
Okay, here it's just a table for summing up what we have found, playing gradient descent with fixed gamma or exact line search or backtracking. So far, we have considered, at least in the context of the gradient descent, only unconstrained optimization.

00:52:48 
So we have considered the minimization of a function f without any constraint. So, in practice, it can happen that I want to minimize a function subject to some constraint on x, which is an example of this situation that we have already encountered.

00:53:25 
Here, I'm saying that I want to minimize a function f subject to the fact that x belongs to a closed convex set. We have encountered this problem two times, but we have seen it also yesterday.

00:54:06 
It will be the problem of regularization. Regularization. When you are dealing with regularization, essentially, you are saying that you want to minimize your function subject to the fact that the weight vector is of minimum norm, is sparse, and we have translated these qualitative constraints in terms of it belongs to the unitary ball measured in the L2 norm or to the unitary ball measured in the L1 norm,

00:54:39 
which are both closed-convex sets. So this is an example. Here are other examples, so non-negativity, box constraints, so for example, you can set the constraint that the weights should have a value between some lower and upper value, or, as we have just mentioned, belonging to a unitary ball in some norm.

00:55:17 
So, what is the problem? The problem is that if you use the gradient descent, even if the iteration k belongs to the set C, the convex set C that defines essentially the constraint, the next step... Can be outside C. You have no guarantee that by performing one step of the gradient descent, you will find a new value x k plus one, which is still in C.

00:56:03 
Okay, so the idea is that we want to still use the gradient descent, but we want to stay within the convex set C that defines the constraints of the problem. Formally, we can use the projection operator.

00:56:34 
The projection operator. It's simply. An operator that, given a point y, finds the point belonging to c, which is closest to y, okay,

00:57:05 
where we have used the projection operator when we have introduced the least squares approximation. We have seen that, given the vector y, which is the true vector of labels, what we have done is to project the value onto the column space of the matrix x, okay? That was exactly the same idea. So, if Y is in C, obviously the projection is itself, otherwise it's a new point, and in practice, the projection is a point on the boundary of C.

00:57:48 
So, if you have, this is C, and you have here a point Y, which is outside C, the projection will be this point, which is on the boundary of C. So, what is the projection gradient method that is used in practice, and there is also a library that implements it.

00:58:24 
It's called... Oh. There is a Python library that implements all the, everything related to projection gradient, projection method, I will tell you the meaning, I will add to the slide. So, what is the idea? The idea is, or if you want the model, is the one that I have written in bold at the beginning.

00:58:56 
Descend, then project, okay? So, essentially, the idea is, I'm starting from a point here, which is a point in C, I am performing a descent step, I am going here, I project on the feasible set C, and then I move on. So, the first step is the gradient descent, and then you have to compute the new iteration, xk plus 1, as the projection of c.

00:59:38 
In one line, this is the update. So, xk plus 1 is the projection of one step of the gradient. This is an important point. The projection, or the projection gradient method, is efficient if the computation of this projection is not too expensive from the computational point of view.

01:00:10 
Let's see some examples. This is the one that we have already encountered twice. It's the L2 projection. This is the L2 projection. So, we want to consider as C the unitary bold defined by the VI2 norm. So, we compute the first gradient step, and then we have to perform the projection.

01:00:43 
So, the projection in this case is quite simple, because if you have, let's have a look at the situation in the plane. So, this is the unitary bold in the plane. If you have a point here, Y, and this is of radius R,

01:01:23 
I said unitary, okay, unitary if r is equal to y, let's say of radius r in general, so if the norm, the two norm of y is less than or equal to r, then y is somewhere here, and the projection is the identity, okay? The projection operator is the identity. What happens if we are here? Well, it means that this is the vector, and what we have to do is to shrink the length of this vector.

01:02:05 
So, in practice, we are constructing the unitary vector along the direction of y. this one and multiply by r in order to project the vector onto the boundary of the ball of radius r. Or if you want in compact form it's vector y times the minimum between 1 and r over the norm of y.

01:02:42 
Come potete vedere, questo è molto facile da calcolare, è molto, molto semplice e... scala la lunghezza del vettore. Questo è coerente con quello che abbiamo già affermato molte volte, cioè che la regolarizzazione L2, o in questo caso, il fatto che stiamo cercando la minimizzazione,

01:03:14 
ma con il vincolo che la soluzione appartiene alla palla di raggio r, è legato al fatto che stiamo cercando la soluzione con norma minima, o norma limitata da r, in questo caso. Questa è l'idea. Come abbiamo già osservato, questo non ha nulla a che fare con la sparsità.

01:03:45 
Quindi L2 promuove solo la norma minima, o norma limitata, in un certo senso. Cosa succede se considero la norma L1? In questo caso, non è possibile trovare un'espressione esplicita per l'operatore di proiezione,

01:04:21 
ma ci sono algoritmi, ovviamente non stiamo parlando del caso 2D, ma del caso multidimensionale, in cui è possibile in un tempo ragionevole, n log n, potete calcolare la, dove n è la dimensione dello spazio che state considerando, potete calcolare la soluzione. Qual è la proprietà importante che abbiamo già osservato molte volte?

01:04:21 
ma ci sono algoritmi, ovviamente non stiamo parlando del caso 2D, ma del caso multidimensionale, in cui è possibile in un tempo ragionevole, n log n, potete calcolare la, dove n è la dimensione dello spazio che state considerando, potete calcolare la soluzione. Qual è la proprietà importante che abbiamo già osservato molte volte?

01:04:53 
Il fatto che, poiché la palla unitaria nella norma L1 è qualcosa del genere, o la palla di raggio r, significa che il fatto che stiamo cercando qualcosa che appartiene a quella palla sta promuovendo la sparsità. Quindi qual è il collegamento tra la minimizzazione vincolata e la regolarizzazione che abbiamo incontrato due volte?

01:05:30 
Il problema vincolato essenzialmente equivale a minimizzare la funzione soggetta al fatto che la soluzione deve appartenere a un certo insieme complesso, C. E possiamo usare la discesa del gradiente proiettata. Poi il problema viene spostato alla definizione dell'operatore di proiezione. Ma in teoria, questo è quello che possiamo fare per il problema vincolato.

01:06:04 
Quindi, qual è la controparte nel contesto della minimizzazione? Vogliamo minimizzare, scusa, qui dovrebbe essere x. Quindi, vogliamo minimizzare f di x, e qui abbiamo il vincolo. Quindi, omega è la rappresentazione di qualche vincolo, come, per esempio, la norma o qualcos'altro. Lambda è un moltiplicatore di Lagrange, okay? E questo può essere risolto dalla cosiddetta discesa del gradiente prossimale.

01:06:46 
Cos'è, in pratica, cos'è il gradiente prossimale? Il gradiente prossimale, l'operatore, l'operatore prossimale è qualcosa che sta... assicurando la discesa, e sta anche soddisfacendo il vincolo in particolare.

01:07:20 
E la definizione della discesa del gradiente prossimale è legata in qualche modo alla definizione dell'operatore di proiezione. In pratica, se definite r, e c'è, quindi r è qualcosa che vi dice qual è l'estensione della regione vincolata, come possiamo vedere qui.

01:07:54 
Okay, quindi dato r positivo, c'è sempre lambda. qui, quindi il moltiplicatore di Lagrange, tale che questi due problemi hanno la stessa soluzione. Quindi, in altre parole, l'ottimizzazione vincolata e la regolarizzazione sono essenzialmente due facce dello stesso problema. Qui, ho riportato il caso della norma L2.

01:08:39 
Quindi, nella parte superiore, abbiamo il caso della minimizzazione vincolata, e qui abbiamo la regolarizzazione, che è la regressione ridge. Quindi, minimizzare F, con questo vincolo. Quindi stiamo penalizzando la norma due di X, il che significa che stiamo cercando il vettore di norma minima.

01:09:12 
Quindi stiamo aggiornando la soluzione X usando questo operatore prossimale, che in questo caso è proprio quello che abbiamo visto ieri, l'operatore di weight decay. Quindi questo è esattamente quello che abbiamo trovato ieri, la procedura di weight decay per ridurre il valore della soluzione ad ogni iterazione.

01:09:46 
Questo è chiamato anche soft penalty. Quindi tutti i pesi sono addestrati o scalati, se volete, da questo vettore ad ogni passo. Ciao. Per la norma L1, l'idea è la stessa. Qual è la differenza? Esattamente come nel caso vincolato, in cui per la L2 eravamo in grado di definire esplicitamente l'operatore di proiezione, mentre per la L1, questo dovrebbe essere fatto essenzialmente dal punto di vista numerico.

01:10:28 
Qui, la differenza è nella definizione dell'operatore prossimale. L'operatore prossimale per la norma L1 è solitamente chiamato operatore di soft thresholding, che è dato da questo termine. E se ricordate ieri, quando abbiamo visto la differenza tra la regolarizzazione L1 e L2, abbiamo trovato esattamente quella l'espressione che avevamo, che è la L1, che è la L2, che è la L2. Qualcosa di molto simile a quello che abbiamo qui.

01:11:02 
Quindi l'operatore prossimale. che abbiamo qui nel caso della, vincolo, scusa, la, regolarizzazione è esattamente quello che abbiamo trovato ieri. Okay, qui c'è un riassunto su quest'ultima parte. Proiezione, gradiente proiettato significa che calcoliamo un passo e poi proiettiamo sull'insieme ammissibile di soluzioni.

01:11:34 
E poi abbiamo visto come eseguire questo metodo, in due casi, palla L2 e L1, che rappresentano casi importanti perché sono direttamente collegati alle idee di regolarizzazione che abbiamo visto in altri contesti. Okay, quindi con questo, possiamo dire che abbiamo visto tutto relativo alla discesa del gradiente.

01:12:15 
È istruttivo perché è la base di, direi, qualsiasi metodo di ottimizzazione, ma non è il metodo che viene praticamente usato. Il metodo che viene praticamente usato è una variante di questo, che si chiama discesa del gradiente stocastico. Quindi forse possiamo fare una pausa di 10 minuti, e poi discuteremo della discesa del gradiente stocastico.

01:12:49 
Okay, quindi... Ora, stiamo per introdurre, direi, il cavallo da battaglia della minimizzazione nel contesto del machine learning, che è la discesa del gradiente stocastico. Qual è l'idea? Se ricordate, quando abbiamo discusso dell'algoritmo di backpropagation e abbiamo trovato le quattro relazioni per la backpropagation,

01:13:24 
una delle prime ipotesi che abbiamo menzionato era legata alla natura della funzione di costo. E l'affermazione era che la funzione di costo che stiamo per considerare è una somma di molte funzioni di costo, che sono. Di solito, se ho pesato per ogni singolo campione, quindi la forma tipica della funzione di costo per il problema di machine learning o deep learning è data da queste espressioni.

01:14:08 
Quindi abbiamo n campioni e la funzione di costo globale è data dalla somma di molte, diciamo, funzioni di costo elementari f i, che sono esattamente quale funzione di costo? Diciamo l'errore quadratico medio valutato su ogni singolo campione. OK, qual è l'idea della discesa del gradiente stocastico?

01:14:41 
Qual è la differenza principale rispetto alla discesa del gradiente? Nella discesa del gradiente, l'iterazione che abbiamo. scritto è xt più uno è uguale a xt meno gamma per il gradiente di f. Qui, invece di considerare il gradiente di f, stiamo solo considerando il gradiente.

01:15:12 
calcolato usando solo una delle n componenti della funzione di costo globale. Quindi abbiamo n, campioni, stiamo scegliendo casualmente uno di questi n indici, e il gradiente, il gradiente globale, è approssimato semplicemente prendendo il gradiente di una delle.

01:15:45 
componenti della funzione di costo globale. Quindi supponiamo che abbiate. 100 termini nella somma state solo scegliendo quello con indice 55 e state calcolando il gradiente solo di quel singolo termine e state usando quell'espressione per ovviamente approssimare, il gradiente completo okay questa è l'idea della discesa del gradiente stocastico.

01:16:23 
e ovviamente la stocasticità è qui il fatto che stiamo scegliendo l'indice che stiamo usando per calcolare il gradiente in modo casuale okay questo è questo box è il riassunto della definizione della discesa del gradiente stocastico è molto molto semplice. Perché lo stiamo facendo, perché nella replicazione, come abbiamo già menzionato molte volte, n potrebbe essere enorme.

01:17:07 
E quindi calcolare il gradiente, il gradiente completo, può essere molto, molto costoso dal punto di vista computazionale. Quindi per ridurre la complessità computazionale del metodo, noi stiamo, invece di prendere una somma di un milione di termini, supponendo che abbiamo un milione di campioni, stiamo solo scegliendone uno.

01:17:39 
Quindi è n volte più economico e, d'altra parte, n volte più veloce della discesa del gradiente. La domanda principale è, ha senso questa idea, o se volete, questa regola di aggiornamento che abbiamo appena introdotto è significativa nel senso che ci sta effettivamente aiutando ad ottimizzare la funzione di costo?

01:18:18 
La risposta è sì, e qui quello che stiamo per fare è cercare di essenzialmente rivedere tutte le dimostrazioni che abbiamo visto prima, ma per i gradienti stocastici. Qui, abbiamo che il gradiente che stiamo usando, stiamo usando lo stesso simbolo di prima, g, ma ora con gt, invece di rappresentare l'intero gradiente, stiamo solo rappresentando il gradiente calcolato usando un singolo campione.

01:19:05 
Per esempio, questa ipotesi è di fondamentale importanza. Il fatto che questo gradiente stocastico sia uno stimatore non distorto del vero gradiente. Quindi cosa significa? Significa che il valore atteso di g, condizionato a xt uguale a x, è effettivamente una.

01:19:37 
funzione di gt. In aspettativa è uguale al gradiente di effetto, e questa proprietà è importante nelle dimostrazioni che stiamo per vedere un altro punto importante è che la dimostrazione che stiamo per vedere tutti i risultati che abbiamo visto precedentemente.

01:20:13 
Stiamo stiamo, per esempio, la differenza tra i valori della funzione o la differenza tra le due iterate, eccetera, non sono più valide, come abbiamo visto in quel caso, ma sono valide in aspettativa. Quindi. Significa che in pratica, a parte alcune tecnicità, quello che stiamo per vedere è che nelle dimostrazioni che abbiamo visto prima,

01:20:47 
dobbiamo prendere l'aspettativa all'inizio e poi lavorare con l'aspettativa invece dei valori puntuali. Qui, questa è l'espressione che... Possiamo ottenere nel senso che qui stiamo. Ricordate che questo è il gradiente valutato con il singolo campione.

01:21:26 
Sappiamo che questo è uno stimatore non distorto e usando la definizione di convessità, possiamo dire che questa relazione è vera. E ho denotato questa relazione con il diamante perché sarà utile nel seguito.

01:21:56 
Quindi torneremo a questa relazione più tardi. Qui sto andando un po' più veloce. E principalmente considereremo i risultati e non andremo nei dettagli delle dimostrazioni, ma come potete vedere qui, abbiamo il risultato per la funzione Lipschitz convessa e funzione Lipschitz.

01:22:28 
Quindi come prima, abbiamo la condizione sulla condizione iniziale, l'ipotesi iniziale non è troppo lontana dalla vera soluzione. E poi abbiamo un limite sul gradiente qui, è sul valore atteso del quadrato del gradiente, ma è qualcosa di simile a quello che abbiamo visto prima. E in realtà quello che abbiamo è un risultato che, a parte il fatto che qui abbiamo l'aspettativa, assomiglia molto da vicino a quello che abbiamo visto per la discesa del gradiente.

01:23:08 
Okay qui c'è la dimostrazione ma non è molto importante. Qui abbiamo il caso per la convessità forte. Questo risultato è un po' più complicato dal punto di vista della dimostrazione ma quello che è importante è che ancora se la funzione è differenziabile, fortemente convessa e abbiamo ancora.

01:23:41 
un limite sul gradiente, quello che possiamo ottenere è qualcosa che ci dice... Sì, qui avete la funzione valutata su questa media di iterazioni meno la funzione nella soluzione è limitata da qualcosa che dipende dal gradiente, la costante di convessità forte e il numero di iterazioni.

01:24:14 
E di nuovo, come prima, potete vedere che in questo caso, abbiamo uno su epsilon in termini di complessità o numero di iterazioni che avete bisogno invece di uno su epsilon al quadrato. Quindi aggiungere la convessità forte sta aiutando nella convergenza. Qui, c'è la dimostrazione. Potete, se volete, potete andare.

01:24:45 
Attraverso i dettagli, ho cercato di. tutto, ma non è molto importante dal mio punto di vista. Voglio solo enfatizzare qualcosa di più interessante dal punto di vista pratico. Nella definizione della discesa del gradiente stocastico, qui abbiamo campione i, okay?

01:25:16 
Quindi, dobbiamo scegliere un i da questo insieme di numeri, da 1 a n. Come possiamo eseguire questo campionamento? Questo è un punto cruciale nella discesa del gradiente stocastico. Essenzialmente, ci sono due strategie. La prima è chiamata con rimpiazzo, e la seconda, senza rimpiazzo. Con rimpiazzo, significa che ad un'iterazione della discesa del gradiente stocastico, sto prendendo tutti i numeri da 1 a n, e sto scegliendo casualmente un valore.

01:26:03 
Alla prossima iterazione, sto ancora prendendo tutti i numeri possibili, e sto scegliendo casualmente un altro valore. Quindi significa, in pratica, che in principio, in due iterazioni consecutive, potrei scegliere lo stesso i. Perché quando ho scelto un indice i, alla prossima iterazione, questo indice i è ancora disponibile.

01:26:35 
Qual è il principale vantaggio di questa strategia? Il fatto che assumendo che ad ogni iterazione ho disponibili tutti gli indici, è un'assunzione forte in termini del fatto che i campioni, questo, quindi il gradiente ad ogni iterazione, sono tutti indipendenti e identicamente distribuiti.

01:27:08 
Quindi significa che dal punto di vista teorico, questo è importante per ottenere i risultati di convergenza. Quindi tutti i risultati di convergenza sulla discesa del gradiente stocastico sono basati sull'ipotesi che il campionamento sia fatto con rimpiazzo. Quindi ad ogni iterazione, ho la disponibilità di tutti i numeri possibili. La seconda strategia.

01:27:39 
È? Okay, qual è lo svantaggio in quello che vi ho detto prima, può succedere che stiate scegliendo lo stesso campione molte volte, e alcuni campioni non sono scelti affatto. Okay, quindi questa è una possibilità. Qual è l'altra possibilità? È senza rimpiazzo. Cosa significa? Significa che in pratica, ad ogni iterazione, sto scegliendo un numero, i, e questo numero non sarà disponibile finché tutti gli altri numeri non sono stati scelti almeno una volta. Okay?

01:28:31 
dal punto di vista pratico cosa significa uh è chiaro che dal punto di vista dell'implementazione questo è molto più facile perché uh all'inizio delle vostre uh iterazioni potete prendere il vettore da uno a n mescolate tutti gli indici e poi state solo scegliendo scegliendo in sequenza tutti gli elementi di questo vettore mescolato okay dato che il mescolamento iniziale.

01:29:03 
è casuale allora state scegliendo gli indici casualmente e questo è quello che viene fatto in pratica in ogni implementazione della discesa del gradiente quindi. E il vantaggio è che ogni punto dati è usato esattamente, almeno uno, in ogni epoca. Epoca significa che state guardando a tutti i possibili, un'epoca è quando state guardando a tutti i campioni possibili nel vostro set di dati.

01:29:40 
Converge più velocemente. Questo è verificato sperimentalmente. Lo svantaggio è che dal punto di vista teorico, il fatto che non stiamo rimpiazzando l'indice che è stato scelto rende l'analisi teorica molto più difficile. E questa è la ragione per cui tutti i risultati teorici, o i risultati teorici più importanti, sono basati sull'altra strategia.

01:30:17 
Grazie. Un altro punto, punto importante, è, e questo è qualcosa che dovete tenere a mente. Qual è la differenza tra ottimizzazione in generale e ottimizzazione nel machine learning? Beh, se devo ottimizzare, diciamo che state eseguendo una procedura di ottimizzazione della forma in cui volete trovare un parametro che definisce la forma di un oggetto, diciamo un profilo di un'ala, e per minimizzare la resistenza o massimizzare la portanza o qualsiasi cosa vogliate.

01:31:04 
In quel caso, quello che volete ottenere è, data la funzione che definisce, per esempio, la resistenza della portanza, volete minimizzare la resistenza, okay? Quindi volete trovare il valore del parametro, per esempio, che definisce, non so, dal punto di vista geometrico, potrebbe essere il raggio del bordo d'attacco di un'ala. Quindi qui avete un raggio e qualcosa legato al raggio di questo bordo d'attacco del profilo, e volete minimizzare la resistenza, okay?

01:31:42 
Quindi trovare, tra i valori possibili, trovare il valore di R che minimizza la resistenza. In quel caso, quello che volete ottenere è una soluzione che è la migliore possibile, okay? Perché volete avere la resistenza minima, la resistenza impossibile. Cosa succede nel contesto del machine learning? In realtà, l'obiettivo non è esattamente minimizzare f. È trovare il valore di w tale che la funzione sia vicina al minimo,

01:32:25 
il valore della funzione è vicino al minimo, ma abbiamo anche la possibilità di generalizzare. Quindi, non vogliamo andare verso l'overfitting. Quindi, in altre parole, nel machine learning, andare verso il minimo perfetto. La maggior parte delle volte significa che stiamo andando verso l'overfitting. Quindi questa è la ragione per cui i due obiettivi nella pratica ingegneristica, per esempio, e nel machine learning della minimizzazione sono totalmente diversi.

01:33:05 
Questa è la ragione per cui la discesa del gradiente stocastico è molto efficace nel machine learning, perché non vogliamo, non vogliamo necessariamente trovare il minimo esatto, ma essere vicini al minimo. E questi non saranno usati nel codice ingegneristico per trovare il minimo di qualcosa legato a questo problema. OK. OK. L'early stopping è qualcosa che abbiamo già discusso ieri.

01:33:36 
E questa è la stessa immagine qui. Voglio solo presentarvi. Un esempio importante. Relativo alla discesa del gradiente stocastico. Assumete che questo sia un esempio che è simile a qualcosa che succede in pratica. Stiamo considerando la regressione lineare considerando un certo numero di campioni e ogni f i.

01:34:08 
Ricordate che la funzione di costo globale è la somma della funzione di costo elementare, funzione di costo, e la funzione di costo elementare in questo caso è f i uguale a uno di a i x meno b i, dove x è il valore del campione e a i e b i sono i parametri che voglio trovare con l'ottimizzazione. OK.

01:34:39 
È chiaro che per questa parabola, il minimo è ottenuto quando xi è uguale a vi su ai, e il minimo globale, quindi se avete n campioni, quindi avete n termini di quel tipo, il minimo globale è ottenuto per x uguale a questo valore. Dovete solo sommare, fare la media somma del precedente.

01:35:20 
Qual è la vista pittorica di questa situazione? Qui avete il grafico. Ognuna delle parabole tratteggiate, rappresenta un singolo termine della funzione della funzione di costo quindi ogni parabola tratteggiata rappresenta uno dei fi okay quindi è per esempio in è uno dei termini di quel tipo e quella nera è.

01:36:03 
la somma di tutti i di tutti i termini tutte le parabole ogni parabola ha il suo proprio minimo, okay e la funzione di costo globale ha il suo proprio minimo ora cosa succede in pratica.

01:36:34 
Supponiamo che io stia scegliendo un valore di x, che è qui, e sto usando la parabola blu per rappresentare il gradiente. Ricordate che nella discesa del gradiente stocastico, sto usando solo un campione invece del campione completo. Se sto usando la parabola blu, quello che posso osservare è che la pendenza di questa parabola, o se volete, tutte le parabole in questa regione, quindi in questa regione verde,

01:37:21 
hanno la stessa pendenza pendenza negativa okay quindi tutte le parabole hanno lo stesso segno della pendenza, e questo segno è uh uguale al segno della parabola nera quindi cosa significa in pratica, in pratica significa che se sono qui in questa regione o in questa regione che è chiamata la regione far out il segno della pendenza di una parabola è uguale al segno della pendenza della.

01:37:55 
parabola nera quindi significa che scegliendo un singolo campione uh, almeno il segno della pendenza qui stiamo considerando il caso 1d ma è rappresentativo anche nel caso multidimensionale in 1d è molto più semplice da visualizzare, che, Significa che scegliendo la pendenza di questa parabola, ragionevolmente mi dà una decente direzione di discesa.

01:38:28 
Okay, cosa succede se sono qui nella regione rossa? Se scelgo la parabola blu o la parabola verde, hanno il segno della pendenza che è diverso. Quindi cosa significa? Significa che se sono in questa regione, a seconda di quale campione sto scegliendo, posso scegliere un segno diverso della pendenza.

01:39:01 
Questa regione è chiamata la regione di confusione, il che significa che in quella regione, a seconda del campione che state scegliendo, cosa succede? Può essere molto diverso. Questo comportamento può essere visualizzato praticamente.

01:39:36 
Qui, ho appena preparato un piccolo codice Python dove con alcuni slider, potete decidere il punto di partenza, che è il punto verde,

01:40:06 
e poi avete il percorso della discesa del gradiente stocastico verso il minimo. Quindi lasciatemi spostare... il punto iniziale un po' qui quello che potete notare è che c'è una fase iniziale diciamo qui dove beh la traiettoria non è così liscia come nel caso della discesa del gradiente ma almeno partendo da questo punto ci stiamo muovendo verso una regione dove uh il.

01:40:41 
comportamento del metodo inizia a diventare apparentemente pazzo come potete vedere anche se e, questo comportamento può essere reso ancora più evidente se aumento la dimensione del passo. questa è la regione di confusione, Perché? Perché in questa regione, scegliere un singolo campione per calcolare il gradiente non assicura che, almeno per il segno, state facendo bene.

01:41:21 
Mentre in questa regione, più o meno, abbiamo un percorso che sta andando verso il minimo. Quindi questo è un comportamento tipico del metodo della discesa del gradiente stocastico. C'è una fase iniziale, diciamo, in cui avete una buona diminuzione, e poi iniziate a muovervi in modo apparentemente casuale.

01:41:54 
E questo è buono o no? Beh, dal punto di vista del machine learning. Questo non è male, perché significa che avendo questa regione di confusione, questo è il punto finale che abbiamo raggiunto, questo è il vero minimo. Quindi siamo vicini, ma non siamo esattamente lì, quindi forse possiamo ragionevolmente evitare l'overfitting, e potete vedere.

01:42:30 
che cambiando il punto di partenza, il comportamento può essere ancora apparentemente più caotico, e anche aumentando il numero di iterazioni, come potete vedere, non assicura che, questo punto stia andando necessariamente più vicino al minimo, okay? Quindi questa è un'immagine che dovete tenere a mente,

01:43:03 
ed è un comportamento tipico della discesa del gradiente, la discesa del gradiente stocastico. Okay, quindi come ho scritto qui, la discesa del gradiente stocastico non può convergere a x stella.

01:43:33 
con tasso di apprendimento costante, e rimbalzerà intorno al minimo. Questa è l'immagine. Qui, ho riportato, senza dimostrazione, ma è interessante, un risultato che è simile a quello che abbiamo visto per la discesa del gradiente sulla differenza tra l'iterazione al tempo, scusa, la soluzione all'iterazione t e la vera soluzione.

01:44:08 
Ovviamente, qui siamo in aspettative, e abbiamo un primo termine, che è simile, esattamente simile a quello che abbiamo visto per la discesa del gradiente, che dipende dall'errore che avete introdotto con l'ipotesi iniziale. Okay, ma questo non è un problema. Perché? Perché qui avete una potenza t. Quindi significa che dato che questa è una costante.

01:44:39 
Più piccola di 1. Se fate abbastanza iterazioni, anche se l'errore iniziale è forse abbastanza grande, questo fattore può ridurre ad ogni iterazione l'errore, e se aggiungete solo questo termine, finirete con la vera soluzione. Ma, nel caso stocastico, abbiamo anche questo termine aggiuntivo, che è legato al tasso di apprendimento, b è la costante legata al limite del gradiente, e mu è la costante di convessità forte.

01:45:19 
È chiaro che questo termine, se avete un gamma fisso, è qualcosa che non andrà mai a zero. Quindi questa è la ragione, dal punto di vista teorico, questa è la ragione per cui avete la regione di confusione, perché anche se questo termine sta diventando sempre più piccolo a causa di questa potenza di p, siete ancora lasciati con questo termine, che non è zero. E l'unica possibilità per, dato che b e mu sono fissi perché sono legati alla forma della funzione che volete minimizzare, l'unica possibilità per ridurre questo termine è giocare con il gamma.

01:46:03 
Quindi se state giocando con gamma significa che volete ideare qualche strategia per aggiungere non un gamma costante, ma un gamma che è... programmato per essere sempre più piccolo quando sto andando verso il minimo, okay?

01:46:39 
In pratica, né la discesa del gradiente, né la discesa del gradiente stocastico, sono usati nella realtà. Quello che viene usato in pratica è qualcosa nel mezzo. Si chiama discesa del gradiente mini-batch. Cos'è la discesa del gradiente mini-batch? Abbiamo detto che nella discesa del gradiente, per calcolare il gradiente, stiamo usando tutti i termini che compongono la funzione di costo.

01:47:17 
Nella discesa del gradiente stocastico, stiamo usando solo un termine. Nel mini-batch, possiamo decidere quanti termini vogliamo usare. Diciamo che vogliamo usare un mini-batch di dimensione 10. In quel caso, significa che se abbiamo un milione di campioni, invece di scegliere solo uno, possiamo scegliere 10 indici scelti casualmente, e useremo quegli 10 indici per calcolare il gradiente.

01:47:52 
Qual è il vantaggio del mini-batch? Il vantaggio del mini-batch è che, essenzialmente, quindi formalmente, qui, m è la dimensione del mini-batch, che, se ricordate ieri, quando abbiamo parlato di iperparametri, era uno degli iperparametri che dovete decidere. Quindi, tra gli altri, tasso di apprendimento, ecc., potete anche decidere la dimensione del mini-batch.

01:48:28 
Poi, una volta che avete deciso la dimensione del mini-batch, il gradiente che qui è chiamato g tilde è calcolato come la media dei gradienti di 10 funzioni, 10 termini. E la regola di aggiornamento è la stessa, con l'unica differenza che ora il gradiente non è il gradiente completo, non è il gradiente calcolato con solo uno, ma è calcolato con il mini-batch.

01:49:05 
Quali sono i benefici di questo approccio? il fatto che stiamo usando invece di solo uno un certo numero di campioni stiamo riducendo la varianza e quindi la la stima g tilde del gradiente è più accurata, della stima che abbiamo ottenuto con una singola valutazione e quindi in pratica è.

01:49:42 
uh potete avere una convergenza che è uh più veloce e più stabile c'è un altro beneficio il calcolo parallelo perché ricordate che tutti i gradienti, devono essere calcolati nello stesso punto, xd, okay, iterazione precedente per calcolare il, quindi se avete un processore con 10 core, per esempio, potete distribuire il calcolo di queste 10 quantità tra i 10 core,

01:50:29 
e potete sfruttare il parallelismo per calcolare le componenti del gradiente, okay, e questa è una delle ragioni per cui l'approccio mini-batch è quello che viene usato, direi, sempre nel deep learning, e dato che questa operazione è...

01:51:02 
Esattamente la stessa e inoltre è fatta anche sugli stessi dati. Quindi non è una singola istruzione, dati multipli. Qui abbiamo singola istruzione e gli stessi dati. Quindi è ancora meglio se volete che questo calcolo sia perfetto per la GPU. OK, questa è la ragione per cui se eseguite lo stesso calcolo, il compito di apprendimento su GPU o sulla CPU,

01:51:36 
avete che prestazioni drammaticamente diverse. Principalmente questo è dovuto alla buona implementazione dell'algoritmo di ottimizzazione sulle GPU. Qual è lo svantaggio? Quali sono gli svantaggi del MiVecs? Beh, il. Quindi quello che abbiamo visto prima, il fatto che quando stiamo usando la classica discesa del gradiente stocastico con quei salti intorno al minimo, nel grafico che vi ho mostrato, era una funzione molto buona, era solo un paraboloide, ma supponiamo che vi venga data una funzione che non è convessa.

01:52:27 
Quindi avete molti, per esempio, molti minimi locali, un minimo globale, quindi una situazione reale, diciamo. Quei salti che possono accadere in pratica con la discesa del gradiente stocastico a volte vi aiuteranno a scappare da un minimo locale. Quindi se siete in un minimo locale.

01:52:59 
che non è esattamente il minimo che volete raggiungere saltando intorno. Forse ad un certo passo siete in grado di uscire dalla regione del minimo locale e di spostarvi forse o verso un altro miglior minimo locale o verso ancora meglio verso il minimo globale. Quindi questo è qualcosa che è solitamente generalmente vero.

01:53:32 
Quindi questo rumore, questi salti stanno guidando verso un minimo che è abbastanza piatto, quindi è una regione piatta. Se usate un mini-batch che è troppo grande, quindi se M sta andando verso N, vi state muovendo verso una discesa del gradiente.

01:54:04 
Quindi questo significa che state riducendo il rumore benefico. Il rumore che avevamo nella discesa del gradiente stocastico è ovviamente anche presente nella discesa del gradiente stocastico mini-batch, ma dato che abbiamo ridotto la varianza, in qualche modo l'entità dei salti sarà, in generale, più piccola.

01:54:35 
Quindi scegliere M troppo grande non è una buona scelta perché può creare overpeaking, perché vi state muovendo verso la discesa del gradiente. Quindi, di nuovo, trovare il valore giusto di M è molto importante e OK, questo è stato già notato e qui ho solo riportato in relazione a quello che vi ho detto prima possibili strategie per scegliere il valore giusto.

01:55:22 
Quindi, di nuovo, trovare il valore giusto di M è molto importante e qui ho solo riportato in relazione a quello che vi ho detto prima possibili strategie per scegliere il valore giusto. La diminuzione è inversamente proporzionale al numero di iterazioni. Oppure potete creare uno scheduler.

01:55:55 
Tutte le librerie, TensorFlow, PyTorch, in tutte le librerie, potete definire uno scheduler per decidere l'entità della dimensione del passo. Quindi, per esempio, potete scegliere un gamma grande nella zona parallela dove siamo sicuri, si spera, possiamo scegliere un passo più grande perché sappiamo che tutte le parabole hanno la stessa dimensione del globo.

01:56:29 
Quindi anche se stiamo scegliendo un passo più grande, si spera stiamo andando verso il minimo. E poi. Sì. Forse decidere di diminuire di un certo fattore dopo un certo numero di iterazioni, o se la funzione trasversale è caratterizzata da un certo comportamento, e in pratica, questa è la strategia più comunemente adottata in tutte le implementazioni della discesa del gradiente stocastico.

01:57:13 
Nella discesa del gradiente stocastico, vi ho detto, è il cavallo da battaglia, nel senso che è stato il primo algoritmo che è stato usato per minimizzare la funzione trasversale nel contesto delle reti neurali o del deep learning. Ma poi... Usandolo in molte situazioni diverse, molti problemi sono stati enfatizzati con l'uso del piano come discesa del gradiente.

01:57:49 
E quindi un numero di nuovi metodi sono stati sviluppati. Non sto parlando di metodi di ordine superiore, che saranno l'oggetto di un'altra parte del corso, ma ancora rimanendo nel contesto del primo ordine, quindi metodi che stanno usando il gradiente di Goddard nell'aviazione, per esempio. Molti metodi sono stati sviluppati, mi dispiace, sto parlando di, forse se ho sentito di Momentum, Master of Acceleration, Adam Method, e così via.

01:58:33 
Adegrad... E quello che vedremo nella prossima lezione lunedì sono tutte queste variazioni sulla discesa del gradiente stocastico che mirano ad accelerare ancora di più la convergenza del metodo nella maggior parte delle applicazioni, okay? Okay, per oggi possiamo fermarci qui.

---

# APPENDICE COMPLETA: Teoria Avanzata della Convergenza e Ottimizzatori Stocastici

**Fonte**: Slide da `GradientDescent_v1.pdf` (60 slides) e `SGD_v1.pdf` (23 slides)  
**Obiettivo**: Analisi rigorosa della convergenza del Gradient Descent sotto diverse assunzioni (Lipschitz, L-smooth, strongly convex) e metodi di ottimizzazione stocastica (SGD, mini-batch, varianti avanzate).

---

## PARTE I: TEORIA DELLA CONVERGENZA GRADIENT DESCENT

### 1. Tabella Comparativa: Velocità di Convergenza GD

*(Riferimento: Slide 1-14 da GradientDescent_v1.pdf)*

**Tabella riassuntiva** delle velocità di convergenza del Gradient Descent sotto assunzioni progressive:

| **Caso** | **Assunzioni** | **Learning Rate γ** | **Convergenza** | **Iterazioni per ε** | **Note** |
|----------|----------------|---------------------|-----------------|----------------------|----------|
| **Convesso solo** | f convessa differenziabile | Decrescente (es. 1/t) | O(1/√T) | O(1/ε²) | Risultato base, molto lento |
| **Lipschitz convesso** | f convessa + ‖∇f(x)‖ ≤ B | γ* = R/(B√T) | O(1/√T) | O(R²B²/ε²) | Gradiente limitato globalmente |
| **L-smooth convesso** | f convessa + Lip(∇f, L) | γ = 1/L | O(1/T) | O(LR²/ε) | **Migliore!** Curvatura limitata |
| **Strongly convex** | μ-strong + L-smooth | γ = 2/(μ+L) | **O(ρᵏ)** con ρ < 1 | O(κ log(1/ε)) | **Convergenza lineare/esponenziale** |

**Legenda**:
- **R**: Raggio iniziale ‖x₀ - x*‖ ≤ R
- **B**: Limite sul gradiente ‖∇f(x)‖ ≤ B (Lipschitz)
- **L**: Costante di smoothness (Lipschitz sul gradiente)
- **μ**: Costante di strong convexity
- **κ = L/μ**: Condition number (numero di condizionamento)
- **ρ = (κ-1)/(κ+1)**: Contraction factor

**Osservazioni chiave**:
1. **Convesso solo**: O(1/ε²) è **molto lento** → 10000 iter per ε=10⁻²
2. **Lipschitz convesso**: Stessa complessità O(1/ε²) ma con costanti migliori
3. **L-smooth**: **O(1/ε)** → 100 iter per ε=10⁻² → MOLTO meglio!
4. **Strongly convex**: **Convergenza esponenziale** O(exp(-T/κ)) → la migliore possibile!

*(Vedi Slide 2-5 da GradientDescent_v1.pdf per definizioni formali di convessità, Slide 6 per Lipschitz/Smoothness)*

---

### 2. Caso Lipschitz Convesso: Analisi Completa

*(Riferimento: Slide 6, 13-17 da GradientDescent_v1.pdf)*

**Definizione (B-Lipschitz)**: Una funzione f è **B-Lipschitz** se per ogni x, y nel dominio:

$$|f(x) - f(y)| \leq B \|x - y\|$$

**Per funzioni convesse differenziabili**, questo equivale a:

$$\|\nabla f(x)\| \leq B \quad \forall x$$

**Interpretazione**: Il gradiente è limitato globalmente → la funzione non può "salire" o "scendere" troppo velocemente.

#### Teorema (Convergenza Lipschitz Convessa)

**Ipotesi**:
1. f convessa differenziabile
2. ‖∇f(x)‖ ≤ B per ogni x (gradiente limitato)
3. ‖x₀ - x*‖ ≤ R (guess iniziale limitato)

**Con learning rate ottimale**:

$$\gamma^* = \frac{R}{B\sqrt{T}}$$

**Risultato**: Dopo T iterazioni,

$$f(x_{\text{best}}) - f(x^*) \leq \frac{RB}{\sqrt{T}}$$

dove $x_{\text{best}} = \arg\min_{t=0,\ldots,T-1} f(x_t)$ è la migliore iterata tra tutte.

**Complessità**: Per raggiungere f(x) - f(x*) ≤ ε, serve:

$$T = O\left(\frac{R^2B^2}{\varepsilon^2}\right)$$

**Esempio numerico** *(dalla lezione)*:
- R = 1, B = 1, ε = 10⁻²
- T ≈ 10⁴ iterazioni richieste!

#### Dimostrazione (Sketch)

**Step 1**: Dall'analisi base (convessità):

$$f(x_t) - f(x^*) \leq \nabla f(x_t)^\top (x_t - x^*)$$

**Step 2**: Regola di aggiornamento GD:

$$x_{t+1} = x_t - \gamma \nabla f(x_t) \quad \Rightarrow \quad \nabla f(x_t) = \frac{x_t - x_{t+1}}{\gamma}$$

**Step 3**: Sostituisci:

$$\nabla f(x_t)^\top (x_t - x^*) = \frac{1}{\gamma}(x_t - x_{t+1})^\top(x_t - x^*)$$

**Step 4**: Sviluppa usando ‖a - b‖² = ‖a‖² - 2a^\top b + ‖b‖²:

$$= \frac{1}{2\gamma}\left[\|x_t - x^*\|^2 - \|x_{t+1} - x^*\|^2 + \|x_t - x_{t+1}\|^2\right]$$

**Step 5**: Usa ‖∇f(x_t)‖ ≤ B:

$$\|x_t - x_{t+1}\|^2 = \gamma^2 \|\nabla f(x_t)\|^2 \leq \gamma^2 B^2$$

**Step 6**: Somma da t=0 a T-1 (somma telescopica!):

$$\sum_{t=0}^{T-1} \nabla f(x_t)^\top (x_t - x^*) \leq \frac{\|x_0 - x^*\|^2}{2\gamma} + \frac{\gamma B^2 T}{2} \leq \frac{R^2}{2\gamma} + \frac{\gamma B^2 T}{2}$$

**Step 7**: Ottimizza γ minimizzando Q(γ) = R²/(2γ) + (γB²T)/2:

$$\frac{dQ}{d\gamma} = -\frac{R^2}{2\gamma^2} + \frac{B^2T}{2} = 0 \quad \Rightarrow \quad \gamma^* = \frac{R}{B\sqrt{T}}$$

**Step 8**: Sostituisci γ* in Q(γ):

$$Q(\gamma^*) = \frac{R^2 B\sqrt{T}}{2R} + \frac{R B^2 T}{2B\sqrt{T}} = RB\sqrt{T}$$

**Step 9**: Usa convessità + somma telescopica:

$$f(x_{\text{best}}) - f(x^*) \leq \frac{1}{T}\sum_{t=0}^{T-1} [f(x_t) - f(x^*)] \leq \frac{RB}{\sqrt{T}}$$

**Q.E.D.** ∎

*(Vedi Slide 13-17 da GradientDescent_v1.pdf per dimostrazione completa step-by-step)*

---

### 3. Caso L-Smooth Convesso: Analisi Completa

*(Riferimento: Slide 6-8, 18-22 da GradientDescent_v1.pdf)*

**Definizione (L-Smoothness)**: Una funzione f è **L-smooth** se il suo gradiente è L-Lipschitz:

$$\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\| \quad \forall x, y$$

**Caratterizzazione equivalente** (upper bound quadratico - Slide 7):

$$f(y) \leq f(x) + \nabla f(x)^\top(y - x) + \frac{L}{2}\|y - x\|^2$$

**Interpretazione geometrica** *(Slide 8 mostra grafico)*:
- f è limitata superiormente da una parabola con curvatura L
- La funzione non può "curvare" troppo rapidamente
- La tangente linearizzazione è accurata localmente

#### Lemma (Sufficient Decrease Property)

**Con learning rate γ = 1/L**:

$$f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$$

**Dimostrazione**:

**Step 1**: Usa la definizione di L-smoothness con y = x_{k+1} = x_k - γ∇f(x_k):

$$f(x_{k+1}) \leq f(x_k) + \nabla f(x_k)^\top(x_{k+1} - x_k) + \frac{L}{2}\|x_{k+1} - x_k\|^2$$

**Step 2**: Sostituisci x_{k+1} - x_k = -γ∇f(x_k):

$$f(x_{k+1}) \leq f(x_k) - \gamma \|\nabla f(x_k)\|^2 + \frac{L\gamma^2}{2}\|\nabla f(x_k)\|^2$$

**Step 3**: Raccogli ‖∇f(x_k)‖²:

$$f(x_{k+1}) \leq f(x_k) - \left(\gamma - \frac{L\gamma^2}{2}\right)\|\nabla f(x_k)\|^2$$

**Step 4**: Con γ = 1/L:

$$\gamma - \frac{L\gamma^2}{2} = \frac{1}{L} - \frac{L}{2L^2} = \frac{1}{L} - \frac{1}{2L} = \frac{1}{2L}$$

**Quindi**:

$$f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2 \quad \text{(sufficient decrease!)}$$

**Q.E.D.** ∎

#### Teorema (Convergenza L-Smooth Convessa)

**Ipotesi**:
1. f convessa differenziabile
2. f è L-smooth
3. ‖x₀ - x*‖ ≤ R

**Con learning rate γ = 1/L**:

$$f(x_T) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2T} = \frac{LR^2}{2T}$$

**Convergenza**: O(1/T) → **MOLTO meglio** di O(1/√T) caso Lipschitz!

**Complessità**: Per raggiungere ε:

$$T = O\left(\frac{LR^2}{\varepsilon}\right)$$

**Esempio numerico**:
- L = 100, R = 1, ε = 10⁻²
- T ≈ 5000 iterazioni (vs 10⁴ del caso Lipschitz!)

#### Dimostrazione (Completa)

**Step 1**: Dal lemma sufficient decrease:

$$\frac{1}{2L}\|\nabla f(x_k)\|^2 \leq f(x_k) - f(x_{k+1})$$

**Step 2**: Usa convessità (first-order characterization):

$$f(x_k) - f(x^*) \leq \nabla f(x_k)^\top(x_k - x^*)$$

**Step 3**: Usa Cauchy-Schwarz:

$$\nabla f(x_k)^\top(x_k - x^*) \leq \|\nabla f(x_k)\| \cdot \|x_k - x^*\|$$

**Step 4**: Dal sufficient decrease:

$$\|\nabla f(x_k)\|^2 \leq 2L[f(x_k) - f(x_{k+1})]$$

**Quindi**:

$$\|\nabla f(x_k)\| \leq \sqrt{2L[f(x_k) - f(x_{k+1})]}$$

**Step 5**: Combina Step 3 + 4:

$$f(x_k) - f(x^*) \leq \sqrt{2L[f(x_k) - f(x_{k+1})]} \cdot \|x_k - x^*\|$$

**Step 6**: Somma telescopica da k=0 a T-1:

$$\sum_{k=0}^{T-1} [f(x_k) - f(x^*)] \leq \sum_{k=0}^{T-1} \sqrt{2L[f(x_k) - f(x_{k+1})]} \cdot \|x_k - x^*\|$$

**Step 7**: Usa ‖x_k - x*‖ ≤ ‖x₀ - x*‖ = R (perché f(x_k) decresce → x_k si avvicina):

$$\sum_{k=0}^{T-1} [f(x_k) - f(x^*)] \leq R\sqrt{2L} \sum_{k=0}^{T-1} \sqrt{f(x_k) - f(x_{k+1})}$$

**Step 8**: Usa Cauchy-Schwarz su somma:

$$\sum_{k=0}^{T-1} \sqrt{f(x_k) - f(x_{k+1})} \leq \sqrt{T \cdot \sum_{k=0}^{T-1} [f(x_k) - f(x_{k+1})]} = \sqrt{T[f(x_0) - f(x_T)]}$$

**Step 9**: Sostituisci:

$$\sum_{k=0}^{T-1} [f(x_k) - f(x^*)] \leq R\sqrt{2LT[f(x_0) - f(x_T)]}$$

**Step 10**: Usa sufficient decrease → f(x_T) è la minima tra tutte:

$$T \cdot [f(x_T) - f(x^*)] \leq \sum_{k=0}^{T-1} [f(x_k) - f(x^*)]$$

**Step 11**: Combina + risolvi per f(x_T) - f(x*):

$$T[f(x_T) - f(x^*)] \leq R\sqrt{2LT[f(x_0) - f(x^*)]}$$

$$\sqrt{T}[f(x_T) - f(x^*)] \leq R\sqrt{2L[f(x_0) - f(x^*)]}$$

**Step 12**: Eleva al quadrato + divide per T:

$$f(x_T) - f(x^*) \leq \frac{2LR^2[f(x_0) - f(x^*)]}{T} \leq \frac{LR^2}{T}$$

**Q.E.D.** ∎

*(Vedi Slide 18-22 da GradientDescent_v1.pdf per dimostrazione rigorosa con tutti i passaggi)*

**Confronto grafico** *(Slide 8 mostra)*:
- Curva blu: Lipschitz O(1/√T) - discesa lenta
- Curva rossa: L-smooth O(1/T) - discesa rapida
- **L-smooth converge MOLTO più velocemente!**

---

### 4. Caso Strongly Convex: Convergenza Lineare/Esponenziale

*(Riferimento: Slide 7-8, 23-30 da GradientDescent_v1.pdf)*

**Definizione (μ-Strong Convexity)**: Una funzione f è **μ-strongly convex** (μ > 0) se:

$$f(y) \geq f(x) + \nabla f(x)^\top(y - x) + \frac{\mu}{2}\|y - x\|^2 \quad \forall x, y$$

**Interpretazione geometrica** *(Slide 8 mostra parabole)*:
- f cresce **almeno** come una parabola con curvatura μ
- Upper bound: parabola curvatura L (da L-smoothness)
- Lower bound: parabola curvatura μ (da strong convexity)
- f è "intrappolata" tra due parabole → **minimo unico e sharp!**

**Immagine slide 8**: Due parabole (rossa sopra, blu sotto) con f(x) nel mezzo.

#### Teorema (Convergenza Lineare/Esponenziale)

**Ipotesi**:
1. f è μ-strongly convex
2. f è L-smooth
3. Condition number κ = L/μ

**Con learning rate γ = 2/(μ + L)**:

$$\|x_k - x^*\|^2 \leq \rho^k \|x_0 - x^*\|^2$$

dove il **contraction factor**:

$$\rho = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1} < 1$$

**Risultato**: **Convergenza lineare (esponenziale in scale lineare)!**

**Per i valori funzionali**:

$$f(x_k) - f(x^*) \leq \frac{L}{2}\rho^k \|x_0 - x^*\|^2$$

**Complessità**: Per raggiungere ε:

$$T = O\left(\kappa \log\frac{1}{\varepsilon}\right)$$

**Esempio numerico**:
- κ = 10: ρ = 9/11 ≈ 0.82 → convergenza moderata
- κ = 100: ρ = 99/101 ≈ 0.98 → convergenza LENTA
- κ = 1 (μ = L): ρ = 0 → **convergenza in 1 step!** (caso ideale)

#### Analisi del Condition Number κ

**Condition number** κ = L/μ misura quanto f è "ben condizionata":

$$\kappa = \frac{L}{\mu} = \frac{\text{max curvature}}{\text{min curvature}}$$

**Interpretazione**:
- **κ vicino a 1**: f è quasi sferica → convergenza rapida (ρ → 0)
- **κ grande**: f è molto "allungata" (ellissoide schiacciato) → convergenza lenta (ρ → 1)

**Tabella convergenza**:

| κ | ρ | Velocità | Iter per ε=10⁻² |
|---|---|----------|-----------------|
| 1 | 0.00 | Istantanea | 1 |
| 2 | 0.33 | Molto rapida | ~3 |
| 10 | 0.82 | Moderata | ~20 |
| 100 | 0.98 | Lenta | ~200 |
| 1000 | 0.998 | Molto lenta | ~2000 |

**Grafico** *(Slide 30 da GradientDescent_v1.pdf mostra)*:
- **Log-scale plot**: ‖x_k - x*‖ vs k
- Decadimento **lineare in log-scale** → convergenza esponenziale!
- Pendenza della retta = log(ρ) = log((κ-1)/(κ+1))

#### Dimostrazione (Sketch principale)

**Step 1**: Dall'analisi base GD:

$$\|x_{k+1} - x^*\|^2 = \|x_k - \gamma\nabla f(x_k) - x^*\|^2$$

**Step 2**: Sviluppa:

$$= \|x_k - x^*\|^2 - 2\gamma\nabla f(x_k)^\top(x_k - x^*) + \gamma^2\|\nabla f(x_k)\|^2$$

**Step 3**: Usa strong convexity + smoothness per limitare i due termini:

Da **strong convexity**:

$$\nabla f(x_k)^\top(x_k - x^*) \geq f(x_k) - f(x^*) + \frac{\mu}{2}\|x_k - x^*\|^2 \geq \frac{\mu}{2}\|x_k - x^*\|^2$$

Da **smoothness** (dopo manipolazioni):

$$\|\nabla f(x_k)\|^2 \leq 2L[f(x_k) - f(x^*)] \leq L\|\nabla f(x_k)\|\|x_k - x^*\|$$

**Step 4**: Con γ = 2/(μ + L), dopo sostituzioni (calcoli lunghi):

$$\|x_{k+1} - x^*\|^2 \leq \left(1 - \frac{2\mu L}{\mu + L}\right)\|x_k - x^*\|^2 = \left(\frac{L - \mu}{L + \mu}\right)^2 \|x_k - x^*\|^2$$

**Step 5**: Applica ricorsivamente:

$$\|x_k - x^*\|^2 \leq \rho^k \|x_0 - x^*\|^2 \quad \text{dove } \rho = \frac{\kappa - 1}{\kappa + 1}$$

**Q.E.D.** ∎

*(Vedi Slide 23-30 da GradientDescent_v1.pdf per dimostrazione completa con tutti i bound)*

**Conclusione chiave**: Strong convexity + smoothness → **convergenza ESPONENZIALE!**  
Questa è la **migliore velocità possibile** per metodi del primo ordine (che usano solo gradienti).

---

### 5. Line Search Methods: Scelta Adattiva del Learning Rate

*(Riferimento: Slide 31-40 da GradientDescent_v1.pdf)*

Nelle dimostrazioni precedenti, abbiamo visto che il learning rate ottimale dipende da costanti (L, μ, B, R) che **in pratica non conosciamo**. I metodi di **line search** risolvono questo problema scegliendo γ_t **adattivamente** ad ogni iterazione.

#### 5.1 Exact Line Search

**Idea**: Minimizza esattamente lungo la direzione di discesa.

**Problema 1D**: All'iterazione k, dato x_k e d_k = -∇f(x_k), trova:

$$\gamma_k^* = \arg\min_{\gamma > 0} \phi(\gamma) \quad \text{dove } \phi(\gamma) = f(x_k + \gamma d_k)$$

**Condizione di ottimalità**: Deriva φ e poni = 0:

$$\phi'(\gamma) = \nabla f(x_k + \gamma d_k)^\top d_k = 0$$

**Interpretazione geometrica** *(Slide 32 mostra grafico)*:
- Il nuovo gradiente ∇f(x_{k+1}) deve essere **ortogonale** alla direzione precedente d_k
- Questo assicura che abbiamo minimizzato esattamente lungo quella direzione

**Vantaggi**:
- Trova il γ che massimizza la riduzione di f
- Numero minimo di iterazioni per raggiungere convergenza

**Svantaggi**:
- **Mai usato in pratica!**
- Per f complicata, risolvere min_γ φ(γ) può essere costoso quanto il problema originale
- Ogni iterazione GD richiederebbe sub-iterazioni per line search

**Quando funziona**: Solo per f molto semplici (es. quadratiche) dove φ(γ) ha forma chiusa.

#### 5.2 Backtracking Line Search (Inexact)

**Idea**: Trova un'**approssimazione** di γ* che garantisce "sufficient decrease".

**Algoritmo** *(Slide 35-37 da GradientDescent_v1.pdf)*:

```
Input: x_k, d_k = -∇f(x_k), γ̄ (starting guess), c ∈ (0,1), τ ∈ (0,1)

γ ← γ̄
While f(x_k + γd_k) > f(x_k) + c·γ·∇f(x_k)ᵀd_k:
    γ ← τ·γ  # Shrink step size
    
Return γ_k ← γ
```

**Parametri tipici** *(Slide 38)*:
- **c = 10⁻⁴**: Constant per Armijo condition (molto piccolo!)
- **τ = 0.5**: Shrink factor (dimezza γ ad ogni fallimento)
- **γ̄ = 1**: Starting guess (oppure γ_k-1 dalla iterazione precedente)

**Armijo-Goldstein Condition**: La condizione nel while:

$$f(x_k + \gamma d_k) \leq f(x_k) + c \cdot \gamma \cdot \nabla f(x_k)^\top d_k$$

**Interpretazione** *(Slide 36 mostra grafico φ(γ))*:
- **Lato destro**: Linearizzazione di φ(γ) con pendenza c·φ'(0)
- **c piccolo (10⁻⁴)**: Accettiamo γ che dà "quasi qualsiasi" riduzione
- **Sufficient decrease**: f deve diminuire abbastanza da non oscillare

**Esempio visivo**:
```
φ(γ) = f(x_k + γd_k)
     
     φ(0) = f(x_k) ─────────────┐
                                │ c·φ'(0)·γ (pendenza dolce)
                                ↓
                          ┌─────────── φ(γ) (curva vera)
                          │
                    γ accettato quando φ(γ) sotto la retta
```

**Processo iterativo**:
1. Prova γ = γ̄ = 1
2. Se f non diminuisce abbastanza: γ ← 0.5
3. Se ancora non va: γ ← 0.25
4. Continua finché Armijo è soddisfatta

**Vantaggi** *(Slide 39)*:
- **Pratico**: Facile da implementare
- **Efficiente**: Poche iterazioni del while (3-5 tipicamente)
- **Robusto**: Funziona per la maggior parte delle funzioni
- **Disponibile**: In tutte le librerie (scipy.optimize, TensorFlow, PyTorch)

**Svantaggi** *(Slide 40)*:
- Introduce **due nuovi iperparametri** (c, τ)
- Numero di iterazioni GD maggiore rispetto a exact line search
- In pratica: **c e τ default funzionano quasi sempre** (no tuning necessario)

#### 5.3 Wolfe Conditions (Opzionale)

**Armijo + Curvature**: Due condizioni invece di una:

1. **Sufficient decrease** (Armijo):
   $$f(x_k + \gamma d_k) \leq f(x_k) + c_1 \gamma \nabla f(x_k)^\top d_k$$

2. **Curvature condition**:
   $$\nabla f(x_k + \gamma d_k)^\top d_k \geq c_2 \nabla f(x_k)^\top d_k$$

**Parametri tipici**: c₁ = 10⁻⁴, c₂ = 0.9

**Interpretazione curvature**: Il nuovo gradiente non deve essere troppo "ripido" nella direzione d_k → previene γ troppo piccolo.

**Uso**: Metodi quasi-Newton (L-BFGS), meno comune in deep learning.

---

### 6. Ottimizzazione Vincolata: Projected Gradient Descent

*(Riferimento: Slide 41-50 da GradientDescent_v1.pdf)*

Finora: Problemi **unconstrained** (min f(x) su tutto ℝ^d).  
Ora: Problemi **constrained**:

$$\min_{x \in C} f(x)$$

dove **C ⊆ ℝ^d** è un insieme chiuso convesso (closed convex set).

**Esempi pratici**:
1. **Regularizzazione** (visto ieri):
   - L2: C = {x : ‖x‖₂ ≤ r} (palla L2)
   - L1: C = {x : ‖x‖₁ ≤ r} (palla L1 → sparsity!)

2. **Non-negativity**: C = {x : x_i ≥ 0 ∀i}

3. **Box constraints**: C = {x : l_i ≤ x_i ≤ u_i ∀i}

**Problema**: Se x_k ∈ C, GD standard può dare x_{k+1} ∉ C!

#### Operatore di Proiezione

**Definizione** *(Slide 42)*:

$$P_C(y) = \arg\min_{x \in C} \|x - y\|^2$$

**Interpretazione**: Dato y (potenzialmente fuori da C), trova il punto x ∈ C **più vicino** a y.

**Proprietà**:
- Se y ∈ C: P_C(y) = y (proiezione identity)
- Se y ∉ C: P_C(y) è sul **boundary** di C
- P_C è **non-espansiva**: ‖P_C(y₁) - P_C(y₂)‖ ≤ ‖y₁ - y₂‖

#### Projected Gradient Descent Algorithm

**Idea**: "Descend, then project" *(Slide 43 mostra grafico)*:

```
x_{k+1} = P_C(x_k - γ∇f(x_k))
```

**Processo**:
1. **Descend**: z_k = x_k - γ∇f(x_k) (passo GD standard)
2. **Project**: x_{k+1} = P_C(z_k) (proietta su C)

**Immagine** *(dalla lezione)*:
```
        C (convex set)
      ┌─────────────┐
      │             │
   x_k●────────────→ z_k (fuori da C)
      │         GD  │
      │      ↙      │
      │   x_{k+1}   │ ← proiezione su boundary
      └─────────────┘
```

**Efficienza**: Projected GD è pratico **solo se** P_C è computazionalmente cheap!

#### Esempi di Proiezioni

**Caso 1: Palla L2** *(Slide 44-45)*

$$C = \{x : \|x\|_2 \leq r\}$$

**Proiezione**:

$$P_C(y) = \begin{cases}
y & \text{se } \|y\|_2 \leq r \\
r \cdot \frac{y}{\|y\|_2} & \text{se } \|y\|_2 > r
\end{cases}$$

**Formula compatta**:

$$P_C(y) = y \cdot \min\left(1, \frac{r}{\|y\|_2}\right)$$

**Interpretazione** *(Slide 45 mostra grafico 2D)*:
- Se y dentro palla: non fare nulla
- Se y fuori: "shrink" y alla lunghezza r mantenendo direzione

**Costo computazionale**: O(d) per calcolare ‖y‖₂ → **molto cheap!**

**Collegamento con regolarizzazione L2**:
- Minimizzare con vincolo ‖x‖₂ ≤ r
- Equivalente a regolarizzazione: min f(x) + λ‖x‖₂²
- Projected GD implementa implicitamente weight decay!

**Caso 2: Palla L1** *(Slide 46-47)*

$$C = \{x : \|x\|_1 \leq r\}$$

**Proiezione**: **NO formula chiusa semplice!**

**Algoritmo**: Esiste procedura O(d log d) *(non mostrata in dettaglio nelle slide)*

**Proprietà chiave** *(Slide 47 mostra diamond 2D)*:
- Palla L1 in 2D: rombo (vertici sugli assi)
- Palla L1 in d dimensioni: "spiky" con vertici su assi
- Proiezione su L1 ball → **promuove sparsity!**

**Collegamento con LASSO**:
- Minimizzare con vincolo ‖x‖₁ ≤ r
- Equivalente a LASSO: min f(x) + λ‖x‖₁
- Projected GD su L1 ball = soft thresholding iterativo

#### Proximal Gradient Method (Cenni)

*(Slide 48-50 da GradientDescent_v1.pdf)*

**Problema regolarizzato**:

$$\min_x f(x) + \lambda \Omega(x)$$

dove Ω(x) è regolarizzatore (es. ‖x‖₁, ‖x‖₂²).

**Operatore prossimale**:

$$\text{prox}_{\lambda\Omega}(y) = \arg\min_x \left\{\frac{1}{2}\|x - y\|^2 + \lambda\Omega(x)\right\}$$

**Proximal Gradient Descent**:

$$x_{k+1} = \text{prox}_{\gamma\Omega}(x_k - \gamma\nabla f(x_k))$$

**Esempi**:

1. **Ω(x) = ‖x‖₂²** (L2 regularization):
   $$\text{prox}_{\gamma\lambda\|\cdot\|_2^2}(y) = \frac{y}{1 + 2\gamma\lambda}$$
   → **Weight decay operator!** (visto ieri in Lez18)

2. **Ω(x) = ‖x‖₁** (L1 regularization):
   $$\text{prox}_{\gamma\lambda\|\cdot\|_1}(y) = \text{sign}(y) \cdot \max(|y| - \gamma\lambda, 0)$$
   → **Soft thresholding operator!**

**Equivalenza** *(Slide 50)*:

$$\text{Constrained: } \min_{x \in C} f(x) \quad \Leftrightarrow \quad \text{Regularized: } \min_x f(x) + \lambda\Omega(x)$$

Dato r > 0, esiste λ tale che le due formulazioni hanno **stessa soluzione**!

**Conclusione Parte I**: Abbiamo visto convergenza GD in 4 casi (convesso, Lipschitz, L-smooth, strongly convex), metodi line search (exact, backtracking), e projected/proximal GD per problemi vincolati. **Prossimo**: Ottimizzazione stocastica (SGD, mini-batch, varianti avanzate).

---

## PARTE II: STOCHASTIC GRADIENT DESCENT E VARIANTI

### 7. Stochastic Gradient Descent (SGD): Teoria e Convergenza

*(Riferimento: Slide 1-10 da SGD_v1.pdf)*

#### 7.1 Motivazione: Funzioni Sum-Structured

**Setting del machine learning** *(Slide 2)*: La funzione di costo ha struttura:

$$f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$$

dove:
- **n**: Numero di campioni nel training set
- **f_i(x)**: Loss function sul campione i-esimo
- **x**: Parametri del modello (pesi w)

**Esempio (Mean Squared Error)**:

$$f(w) = \frac{1}{n}\sum_{i=1}^n \|y_i - \phi(x_i)^\top w\|^2$$

**Gradient Descent standard** richiede:

$$\nabla f(x_t) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(x_t) \quad \text{(somma su TUTTI i campioni!)}$$

**Problema**: Se n è enorme (es. n = 10⁶ immagini), ogni iterazione GD richiede **n valutazioni del gradiente** → **molto costoso!**

#### 7.2 Idea SGD: Stochastic Approximation

**Stochastic Gradient Descent** *(Slide 2-3)*:

```
All'iterazione t:
1. Sample i ∈ {1, ..., n} uniformly at random
2. Compute g_t = ∇f_i(x_t)  (gradiente di UN SOLO campione!)
3. Update: x_{t+1} = x_t - γ_t g_t
```

**Confronto costi** *(Slide 3)*:

| Metodo | Costo per iterazione | Speedup |
|--------|----------------------|---------|
| Full GD | O(nd) | 1× |
| SGD | O(d) | **n×** |

**Domanda chiave**: Usare solo 1/n dei dati → converge comunque verso il minimo?

**Risposta**: **Sì!** Con alcune modifiche teoriche (aspettative invece di bound deterministici).

#### 7.3 Proprietà Chiave: Unbiased Estimator

**Teorema (Unbiasedness)** *(Slide 4-5)*:

Il gradiente stocastico g_t = ∇f_i(x_t) è uno **stimatore non distorto** del vero gradiente:

$$\mathbb{E}[g_t | x_t = x] = \mathbb{E}_i[\nabla f_i(x)] = \frac{1}{n}\sum_{i=1}^n \nabla f_i(x) = \nabla f(x)$$

**Conseguenza per l'analisi** *(Slide 5)*:

La disuguaglianza base (first-order characterization) **non vale** per g_t singolo:

$$f(x_t) - f(x^*) \leq \nabla f(x_t)^\top(x_t - x^*) \quad \text{(NON vale per } g_t \text{ singolo)}$$

**MA**: Vale **in aspettativa**!

$$\mathbb{E}[g_t^\top(x_t - x^*)] = \mathbb{E}[\nabla f(x_t)^\top(x_t - x^*)] \geq \mathbb{E}[f(x_t) - f(x^*)] \quad (⋄)$$

**Questa proprietà (⋄) è fondamentale** per tutte le dimostrazioni SGD!

**Interpretazione**:
- Ogni singola iterazione SGD può essere "rumorosa" (gradiente approssimato male)
- Ma **in media** (su molte iterazioni), SGD fa progressi nella direzione corretta
- Il rumore si "media out" nel lungo termine

#### 7.4 Convergenza SGD: Caso Lipschitz Convesso

**Teorema** *(Slide 6-7 da SGD_v1.pdf)*:

**Ipotesi**:
1. f convessa differenziabile con minimo x*
2. ‖x₀ - x*‖ ≤ R (guess iniziale limitato)
3. 𝔼[‖g_t‖²] ≤ B² per ogni t (gradiente stocastico limitato in norma quadratica media)

**Con learning rate costante**:

$$\gamma = \frac{R}{B\sqrt{T}}$$

**Risultato**: Dopo T iterazioni,

$$\mathbb{E}\left[\frac{1}{T}\sum_{t=0}^{T-1} f(x_t)\right] - f(x^*) \leq \frac{RB}{\sqrt{T}}$$

**Convergenza**: O(1/√T) → **stessa** del GD deterministico caso Lipschitz!

**Complessità**: Per raggiungere ε:

$$T = O\left(\frac{R^2B^2}{\varepsilon^2}\right) = O(1/\varepsilon^2)$$

**Nota importante**: Il bound è sulla **media** delle iterate, non sull'ultima!

$$\bar{x}_T = \frac{1}{T}\sum_{t=0}^{T-1} x_t \quad \text{(media pesata)}$$

#### Dimostrazione (Sketch)

**Step 1**: Partire dall'analisi base GD e prendere aspettative:

$$\sum_{t=0}^{T-1} \mathbb{E}[g_t^\top(x_t - x^*)] \leq \frac{\gamma}{2}\sum_{t=0}^{T-1} \mathbb{E}[\|g_t\|^2] + \frac{1}{2\gamma}\|x_0 - x^*\|^2$$

**Step 2**: Usa i tre fatti chiave:
1. 𝔼[f(x_t) - f(x*)] ≤ 𝔼[g_t^\top(x_t - x*)] (da proprietà ⋄)
2. 𝔼[‖g_t‖²] ≤ B² (ipotesi)
3. ‖x₀ - x*‖² ≤ R² (ipotesi)

**Step 3**: Sostituisci:

$$\sum_{t=0}^{T-1} \mathbb{E}[f(x_t) - f(x^*)] \leq \frac{\gamma B^2 T}{2} + \frac{R^2}{2\gamma}$$

**Step 4**: Ottimizza γ minimizzando il lato destro:

$$Q(\gamma) = \frac{\gamma B^2 T}{2} + \frac{R^2}{2\gamma}$$

$$\frac{dQ}{d\gamma} = \frac{B^2T}{2} - \frac{R^2}{2\gamma^2} = 0 \quad \Rightarrow \quad \gamma^* = \frac{R}{B\sqrt{T}}$$

**Step 5**: Sostituisci γ*:

$$\sum_{t=0}^{T-1} \mathbb{E}[f(x_t) - f(x^*)] \leq RB\sqrt{T}$$

**Step 6**: Dividi per T:

$$\mathbb{E}\left[\frac{1}{T}\sum_{t=0}^{T-1} f(x_t)\right] - f(x^*) \leq \frac{RB}{\sqrt{T}}$$

**Q.E.D.** ∎

*(Vedi Slide 7 da SGD_v1.pdf per dimostrazione completa)*

#### 7.5 Convergenza SGD: Caso Strongly Convex

**Teorema** *(Slide 8-9 da SGD_v1.pdf)*:

**Ipotesi**:
1. f è μ-strongly convex differenziabile
2. 𝔼[‖g_t‖²] ≤ B² per ogni t

**Con learning rate decrescente**:

$$\gamma_t = \frac{2}{\mu(t+1)}$$

**Risultato**:

$$\mathbb{E}\left[f\left(\frac{\sum_{t=1}^T t \cdot x_t}{T(T+1)/2}\right)\right] - f(x^*) \leq \frac{2B^2}{\mu(T+1)}$$

**Convergenza**: O(1/T) → **meglio** di O(1/√T) caso Lipschitz!

**Complessità**: Per raggiungere ε:

$$T = O\left(\frac{B^2}{\mu\varepsilon}\right) = O(1/\varepsilon)$$

**Nota**: Il bound è sulla **weighted average** delle iterate:

$$\tilde{x}_T = \frac{\sum_{t=1}^T t \cdot x_t}{\sum_{t=1}^T t} = \frac{2\sum_{t=1}^T t \cdot x_t}{T(T+1)}$$

(Iterate recenti pesano di più!)

#### Dimostrazione (Sketch Principale)

**Step 1**: Dall'analisi base GD con aspettative:

$$\mathbb{E}[g_t^\top(x_t - x^*)] = \frac{\gamma_t}{2}\mathbb{E}[\|g_t\|^2] + \frac{1}{2\gamma_t}(\mathbb{E}[\|x_t - x^*\|^2] - \mathbb{E}[\|x_{t+1} - x^*\|^2])$$

**Step 2**: Usa proprietà ⋄ + strong convexity:

$$\mathbb{E}[g_t^\top(x_t - x^*)] \geq \mathbb{E}[f(x_t) - f(x^*)] + \frac{\mu}{2}\mathbb{E}[\|x_t - x^*\|^2]$$

**Step 3**: Combina (1) + (2) e usa 𝔼[‖g_t‖²] ≤ B²:

$$\mathbb{E}[f(x_t) - f(x^*)] \leq \frac{B^2\gamma_t}{2} + \frac{\gamma_t^{-1} - \mu}{2}\mathbb{E}[\|x_t - x^*\|^2] - \frac{\gamma_t^{-1}}{2}\mathbb{E}[\|x_{t+1} - x^*\|^2]$$

**Step 4**: Con γ_t = 2/(μ(t+1)):

$$\gamma_t^{-1} - \mu = \frac{\mu(t+1)}{2} - \mu = \frac{\mu(t-1)}{2}$$

**Step 5**: Moltiplica per t e somma da t=1 a T (usa **telescoping** + **Jensen's inequality**):

Dopo manipolazioni algebriche complesse (Slide 9-10):

$$\mathbb{E}\left[f\left(\frac{2\sum_{t=1}^T t \cdot x_t}{T(T+1)}\right)\right] - f(x^*) \leq \frac{2B^2}{\mu(T+1)}$$

**Q.E.D.** ∎

*(Vedi Slide 8-10 da SGD_v1.pdf per dimostrazione dettagliata con Jensen)*

**Osservazione chiave**: Strong convexity dà convergenza **O(1/T)** invece di O(1/√T) → molto meglio!

---

### 8. Sampling Strategies: With/Without Replacement

*(Riferimento dalla lezione, Slide 11-12 da SGD_v1.pdf)*

Nella definizione di SGD, abbiamo detto "sample i uniformly at random". Ci sono **due strategie**:

#### 8.1 Sampling WITH Replacement

**Procedura**:
- Ad ogni iterazione t: sample i_t ~ Uniform({1, ..., n})
- **Tutti** gli indici 1,...,n sono disponibili ad ogni step
- Possibile scegliere **stesso campione** in iterazioni consecutive

**Esempio** (n=5, T=10 iterazioni):
```
i_1 = 3, i_2 = 1, i_3 = 3 (← ripetuto!), i_4 = 5, i_5 = 2, ...
```

**Vantaggi** (teorici):
- Campioni {g_t} sono **i.i.d.** (indipendenti e identicamente distribuiti)
- Tutte le dimostrazioni teoriche funzionano perfettamente
- Analisi matematica pulita

**Svantaggi** (pratici):
- Alcuni campioni possono essere scelti molte volte
- Altri campioni possono NON essere mai scelti in un'epoca
- Inefficiente: spreca informazione disponibile

#### 8.2 Sampling WITHOUT Replacement (Usato in Pratica!)

**Procedura** *(Slide 11)*:
1. All'inizio dell'epoca: shuffle degli indici {1,...,n} → permutazione casuale π
2. Iterate t=0,...,n-1: usa i_t = π(t) in ordine
3. Fine epoca: ripeti shuffle

**Esempio** (n=5, una epoca):
```
Shuffle → π = [3, 1, 5, 2, 4]
Iteration 0: i_0 = 3
Iteration 1: i_1 = 1
Iteration 2: i_2 = 5
Iteration 3: i_3 = 2
Iteration 4: i_4 = 4
New epoch → shuffle again!
```

**Vantaggi** (pratici):
- **Facile da implementare**: 1 shuffle all'inizio + iterate in ordine
- Ogni campione usato **esattamente 1 volta** per epoca
- Converge **più velocemente** empiricamente (verificato sperimentalmente)
- **Usato in tutte le librerie**: PyTorch DataLoader, TensorFlow tf.data

**Svantaggi** (teorici):
- Campioni NON sono i.i.d. (correlati entro epoca)
- Analisi teorica **molto più difficile**
- Risultati teorici rigorosi meno sviluppati

**Implementazione pratica** (PyTorch esempio):

```python
from torch.utils.data import DataLoader, Dataset

# Dataset con n campioni
dataset = MyDataset(...)

# DataLoader con shuffle=True → sampling WITHOUT replacement!
dataloader = DataLoader(
    dataset, 
    batch_size=32,      # mini-batch size
    shuffle=True,       # ← Shuffle all'inizio di ogni epoca
    num_workers=4       # Parallel loading
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:  # ← Ogni batch diverso, nessuna ripetizione in epoca
        x, y = batch
        loss = compute_loss(x, y)
        loss.backward()
        optimizer.step()
```

**Conclusione**: In pratica, **without replacement** è universalmente usato (più efficiente, converge meglio). Teoria matematica preferisce **with replacement** (analisi più pulita).

---

### 9. SGD vs Machine Learning Optimization: Difference in Goals

*(Riferimento dalla lezione, 01:31:04 - 01:33:05)*

**Punto chiave dalla lezione**: Obiettivi di ottimizzazione in ML sono **diversi** da ottimizzazione classica!

#### 9.1 Ottimizzazione Classica (Engineering)

**Esempio**: Ottimizzare forma di un'ala per minimizzare resistenza (drag).

**Obiettivo**: Trovare il **minimo esatto** di f(R) = drag(R), dove R = raggio bordo d'attacco.

**Perché**: Vogliamo la migliore performance possibile → minimo drag fisicamente raggiungibile.

**Metodi**: GD deterministico, Newton, BFGS, ... → convergenza precisa al minimo.

#### 9.2 Ottimizzazione Machine Learning

**Esempio**: Minimizzare training loss L_train(w).

**Obiettivo**: **NON** il minimo esatto di L_train!

**Obiettivo vero**: 
1. L_train(w) **vicino** al minimo (non esatto)
2. **Generalizzazione**: L_test(w) piccolo (avoid overfitting!)

**Perché**: Minimo esatto di L_train → **overfitting**!

**Grafico concettuale**:
```
Training loss
    │
    │  ●  ← minimo esatto (overfitting)
    │ ╱ ╲
    │╱   ╲
    ●─────●  ← zone "buone" (vicine al minimo, generalizzano)
    │      
────┼────────► Complessità modello
    │
Test loss
    │      ╱ ← overfitting!
    │    ╱
    │  ╱ ● ← minimo training
    │╱
    ●────────  ← minimo test (vogliamo questo!)
```

**Implicazione per SGD**:
- Il "rumore" di SGD (bouncing intorno al minimo) è **benefico**!
- Impedisce overfitting → migliore generalizzazione
- Non serve convergenza esatta → early stopping ottimale

**Citazione dalla lezione** *(01:32:25)*:
> "Nel machine learning, andare verso il minimo perfetto la maggior parte delle volte significa che stiamo andando verso l'overfitting. Quindi questa è la ragione per cui la discesa del gradiente stocastico è molto efficace nel machine learning, perché non vogliamo necessariamente trovare il minimo esatto, ma essere vicini al minimo."

#### 9.3 Early Stopping in SGD

**Strategia** *(dalla lezione ieri, vedi Lez18)*:
1. Monitor validation loss L_val(w_t) durante training
2. Training loss L_train(w_t) continua a scendere
3. Validation loss L_val(w_t) prima scende, poi **risale** (overfitting!)
4. **Stop** quando L_val inizia a risalire

**Grafico** (mostrato ieri Lez18):
```
Loss
  │
  │ Train ───────────────↘ (continua a scendere)
  │
  │ Val ────↘──↗ (scende poi risale)
  │         ↑
  │    Stop qui! (best model)
  │
  └──────────────────────────► Epochs
```

**Perché SGD è perfetto per questo**:
- Rumore intrinseco impedisce convergenza esatta
- Facile interrompere quando validation peggiora
- GD deterministico convergerebbero troppo precisamente al minimo training

---

### 10. Far-Out vs Confusion Regions: SGD Behavior

*(Riferimento dalla lezione, 01:34:08 - 01:42:30, esempio con parabole)*

**Esempio didattico dalla lezione**: Funzione f = Σᵢ f_i dove ogni f_i(x) = (1/2)aᵢ(x - bᵢ/aᵢ)²

Ogni termine f_i è una **parabola** con minimo in x_i* = bᵢ/aᵢ.

#### 10.1 Visualizzazione (2D)

```
f_i values
    │
    │  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲   ← parabole tratteggiate (singoli f_i)
    │ ╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲
    │╱                    ╲
    ────────────●──────────────  ← parabola nera (f totale)
    │           ↑
    │      minimo globale x*
    │
    │ ← FAR-OUT   →│← CONFUSION →│← FAR-OUT →
    │   REGION     │   REGION    │  REGION
```

#### 10.2 Far-Out Region (Verde)

**Definizione**: Regione lontana dal minimo dove **tutte** le parabole f_i hanno stesso **segno** del gradiente.

**Comportamento SGD**:
- Sample f_i casuale → gradiente ∇f_i(x_t)
- **Segno(∇f_i(x_t)) = Segno(∇f(x_t))** per quasi tutte le parabole!
- SGD fa progressi **consistenti** verso il minimo
- Traiettoria relativamente **smooth**

**Esempio dalla lezione**:
- Punto di partenza x₀ lontano a sinistra
- Tutte le parabole hanno pendenza **negativa** (puntano a destra verso minimi)
- f totale ha pendenza negativa
- SGD si muove a destra in modo relativamente ordinato

#### 10.3 Confusion Region (Rossa)

**Definizione**: Regione **vicino al minimo** dove parabole diverse hanno **segni opposti** del gradiente.

**Comportamento SGD**:
- Sample f_i casuale → può dare gradiente con **segno sbagliato**!
- Parabola blu: ∇f_blu < 0 (dice "vai sinistra")
- Parabola verde: ∇f_verde > 0 (dice "vai destra")
- f totale: ∇f ≈ 0 (vicino al minimo)
- SGD fa "salti caotici" intorno al minimo!

**Visualizzazione traiettoria** (dalla lezione, Colab demo):
```
      Far-Out          Confusion
        │                 │
  ──────●────────→  ●↗●↙●→●←●↘●  ← smooth   ← chaotic jumps!
   x₀   │            │  
        │        minimo x*
```

#### 10.4 Demo Interattiva (Dalla Lezione)

**Parametri slide bar** (01:40:06 - 01:42:30):
- **Punto iniziale**: x₀ (verde)
- **Learning rate**: γ
- **Numero iterazioni**: T

**Osservazioni chiave**:

1. **x₀ lontano** (far-out):
   - Traiettoria relativamente diretta verso minimo
   - Fase iniziale smooth

2. **Vicino al minimo** (confusion):
   - "Apparentemente pazzo" *(citazione lezione)*
   - Salti in direzioni random
   - **Non converge al minimo esatto** anche con T → ∞!

3. **Aumentare γ**:
   - Far-out: Convergenza più rapida (OK)
   - Confusion: Salti **più grandi** → ancora più caotico!

4. **Aumentare T** (iterazioni):
   - Far-out → Confusion velocemente
   - Confusion: Più iterazioni **NON** portano più vicino al minimo!
   - Bouncing persiste indefinitamente

**Citazione dalla lezione** *(01:41:54)*:
> "Questo comportamento può essere reso ancora più evidente se aumento la dimensione del passo. Questa è la regione di confusione. Perché? Perché in questa regione, scegliere un singolo campione per calcolare il gradiente non assicura che, almeno per il segno, state facendo bene."

#### 10.5 Implicazioni Teoriche

**Teorema (Limite di convergenza SGD)** *(dalla lezione, 01:44:08 - 01:46:03)*:

Per f μ-strongly convex, con learning rate **costante** γ:

$$\mathbb{E}[\|x_t - x^*\|^2] \leq \underbrace{\rho^t \|x_0 - x^*\|^2}_{\text{termine "buono"}} + \underbrace{\frac{\gamma B^2}{\mu}}_{\text{termine "rumore"}}$$

dove ρ < 1 (contraction factor).

**Interpretazione**:

1. **Primo termine**: ρᵗ‖x₀ - x*‖² → 0 quando t → ∞ (come GD deterministico)
   - Errore iniziale decade esponenzialmente
   - Con t abbastanza grande, diventa trascurabile

2. **Secondo termine**: γB²/μ **NON va a zero**!
   - Dipende da γ (learning rate)
   - Dipende da B (bound gradiente), μ (strong convexity)
   - Rappresenta **rumore intrinseco** di SGD

**Limite pratico**:

$$\lim_{t \to \infty} \mathbb{E}[\|x_t - x^*\|^2] \geq \frac{\gamma B^2}{\mu}$$

**Conseguenze**:
- Con γ costante, SGD **non converge esattamente** a x*
- Bouncing intorno al minimo con raggio ~ √(γB²/μ)
- Per ridurre rumore: **diminuire γ** nel tempo (learning rate schedules!)

**Soluzione**: Learning rate **decrescente** γ_t → 0:
- Es: γ_t = γ₀/t, γ_t = γ₀/√t, step decay, cosine annealing
- Secondo termine → 0 quando t → ∞
- Ma: Convergenza più lenta nelle fasi iniziali

**Citazione dalla lezione** *(01:45:19)*:
> "È chiaro che questo termine, se avete un gamma fisso, è qualcosa che non andrà mai a zero. Quindi questa è la ragione, dal punto di vista teorico, questa è la ragione per cui avete la regione di confusione [...] l'unica possibilità per ridurre questo termine è giocare con il gamma."

---

### 11. Mini-Batch SGD: Best of Both Worlds

*(Riferimento: Slide 13-17 da SGD_v1.pdf, lezione 01:46:39 - 01:51:36)*

**Problema**: 
- **Full-batch GD**: Lento (costo O(nd) per iterazione)
- **SGD (batch=1)**: Rumoroso, bouncing, instabile

**Soluzione**: **Mini-batch SGD** → usa **m campioni** (1 < m < n) per approssimare gradiente!

#### 11.1 Algoritmo Mini-Batch SGD

**Procedura** *(Slide 13)*:

```
Parametri:
- m: batch size (iperparametro!)
- γ: learning rate

All'iterazione t:
1. Sample B_t ⊂ {1,...,n} con |B_t| = m (uniformly random)
2. Compute stochastic gradient:
   
   g̃_t = (1/m) Σ_{i∈B_t} ∇f_i(x_t)
   
3. Update: x_{t+1} = x_t - γ g̃_t
```

**Casi speciali**:
- **m = 1**: Standard SGD
- **m = n**: Full-batch GD
- **m ∈ (1, n)**: Mini-batch (il "sweet spot"!)

#### 11.2 Vantaggi Mini-Batch

**1. Riduzione Varianza** *(Slide 14)*:

**Varianza SGD** (m=1):

$$\text{Var}(g_t) = \text{Var}(\nabla f_i(x_t)) = \sigma^2$$

**Varianza mini-batch** (m campioni):

$$\text{Var}(\tilde{g}_t) = \text{Var}\left(\frac{1}{m}\sum_{i \in B_t} \nabla f_i(x_t)\right) = \frac{\sigma^2}{m}$$

**Implicazione**: Varianza ridotta di **fattore m**!

**Effetto pratico**:
- Gradiente più accurato → direzione di discesa migliore
- Convergenza più **smooth** e **stabile**
- Meno oscillazioni intorno al minimo

**Esempio numerico**:
- m = 1: Var = σ²
- m = 32: Var = σ²/32 ≈ 0.03σ² (molto più basso!)
- m = 256: Var = σ²/256 ≈ 0.004σ² (quasi deterministico)

**2. Parallelizzazione** *(dalla lezione, 01:49:42 - 01:50:29)*:

**Key observation**: Tutti i gradienti ∇f_i(x_t) per i ∈ B_t devono essere calcolati nello **stesso punto** x_t.

**Operazione SIMD** (Single Instruction, Multiple Data):
- **Stessa** operazione (compute gradient)
- **Stesso** dato (punto x_t)
- **Diversi** campioni (i ∈ B_t)

**Perfetto per GPU!**

**Esempio pratico** (10 core CPU):
- Batch size m = 32
- Dividi 32 campioni tra 10 core
- Core 1: gradienti campioni 1-4
- Core 2: gradienti campioni 5-8
- ...
- Core 10: gradienti campioni 29-32
- **Somma** risultati → media

**Speedup empirico**:
```
Device    | Batch=1 | Batch=32 | Batch=256 | Speedup (256 vs 1)
----------|---------|----------|-----------|-------------------
CPU (1)   | 100 ms  | 150 ms   | 400 ms    | 25× (not 256×!)
CPU (16)  | 100 ms  | 20 ms    | 50 ms     | 50× (better!)
GPU (V100)| 10 ms   | 2 ms     | 3 ms      | 80-100× (best!)
```

*(Numeri illustrativi, dipendono da architettura)*

**Perché GPU >> CPU per mini-batch**:
- GPU ha 1000+ cores (vs 4-16 CPU)
- Architettura ottimizzata per SIMD
- Memoria condivisa veloce
- Librerie ottimizzate (cuDNN per deep learning)

**Citazione dalla lezione** *(01:51:02)*:
> "Esattamente la stessa [operazione] e inoltre è fatta anche sugli stessi dati. Quindi non è una singola istruzione, dati multipli. Qui abbiamo singola istruzione e gli stessi dati. Quindi è ancora meglio se volete che questo calcolo sia perfetto per la GPU."

**3. Convergenza Più Veloce** *(empirico)*:

**Observation**: In pratica, mini-batch converge **molto più velocemente** di SGD puro!

**Grafico tipico** (iterazioni vs loss):
```
Loss
  │
  │ ╲
  │  ╲ ← Full GD (smooth ma lento)
  │   ╲___
  │     
  │  ╲╲╲ ← Mini-batch m=32 (smooth + veloce)
  │    ╲╲___
  │
  │  ╲  ╲  ╲ ← SGD m=1 (veloce ma rumoroso)
  │   ╲  ╲  ╲___~~
  │
  └──────────────────────► Iterations
```

**Grafico tipico** (wall-clock time vs loss):
```
Loss
  │
  │  ╲_______ ← Full GD (lento!)
  │
  │    ╲╲___ ← Mini-batch m=32 (BEST!)
  │
  │     ╲  ╲~~ ← SGD m=1 (rumoroso)
  │
  └──────────────────────► Time (seconds)
```

**Optimal batch size empirico**: m = 32-256 per molte applicazioni deep learning!

#### 11.3 Svantaggi Mini-Batch

**1. Riduzione "Rumore Benefico"** *(dalla lezione, 01:52:27 - 01:54:35)*:

**Problema**: Il rumore di SGD aiuta a **scappare da minimi locali** (funzioni non-convesse)!

**Scenario** (funzione non-convessa con molti minimi locali):
```
f(x)
  │   ╱╲        ╱╲     ╱╲
  │  ╱  ╲      ╱  ╲   ╱  ╲
  │ ╱    ╲____╱    ╲_╱    ╲
  │╱      ↑         ↑      ╲
  └───────●─────────●───────●── x
    local   local   global
    min #1  min #2  min (best!)
```

**SGD (m=1)**: 
- Alto rumore → grandi salti casuali
- Se "intrappolato" in minimo locale #1: salto casuale può portare fuori!
- Possibile raggiungere minimo locale #2 o globale

**Mini-batch (m grande)**:
- Basso rumore → piccoli salti
- Se "intrappolato": difficile uscire
- Converge a minimo locale vicino (può NON essere il migliore!)

**Citazione dalla lezione** *(01:52:59)*:
> "Quei salti che possono accadere in pratica con la discesa del gradiente stocastico a volte vi aiuteranno a scappare da un minimo locale. [...] Se usate un mini-batch che è troppo grande, quindi se M sta andando verso N, vi state muovendo verso una discesa del gradiente. Questo significa che state riducendo il rumore benefico."

**Implicazione pratica**:
- Batch size **troppo grande** (m → n) può dare **overfitting** (sharp minima)
- Batch size **piccolo** (m piccolo) trova **flat minima** (generalizzano meglio!)

**Evidenza empirica** (research paper: "On Large-Batch Training..."):
- Small batch (32-64): Test accuracy 95%
- Large batch (8192): Test accuracy 92% (overfitting!)

**2. Nuovo Iperparametro**:
- Oltre a γ (learning rate), ora anche m (batch size) da tuning!
- Trade-off: velocità vs generalizzazione

#### 11.4 Scelta Pratica del Batch Size

**Guidelines empiriche**:

| Application | Typical Batch Size | Rationale |
|-------------|-------------------|-----------|
| Computer Vision (CNN) | 32-128 | Balance speed/memory |
| NLP (Transformer) | 32-64 (effective batch) | Memory constraints |
| RL (Reinforcement Learning) | 1-32 | Online learning |
| Tabular data | 256-1024 | Less overfitting concern |

**Fattori da considerare**:

1. **GPU memory**: Batch troppo grande → out of memory!
   - ResNet-50 su ImageNet: max ~128 per GPU (16GB)
   - GPT-3: Gradient accumulation su micro-batches

2. **Dataset size**: 
   - n piccolo (< 10000): Batch 32-128
   - n grande (> 1M): Batch 128-512

3. **Generalizzazione**:
   - Batch piccolo (32-64): Meglio generalizzazione
   - Batch grande (512+): Risk overfitting

**Tecnica avanzata: Gradient Accumulation**:

Se GPU memory limita batch size, simula batch grande:

```python
effective_batch_size = 256
actual_batch_size = 32  # Quello che GPU può gestire
accumulation_steps = effective_batch_size // actual_batch_size  # = 8

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()  # Accumula gradienti
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update ogni 8 mini-batch
        optimizer.zero_grad()
```

**Risultato**: Simula batch=256 con memoria per batch=32!

---

### 12. Learning Rate Schedules: Ridurre il Rumore SGD

*(Riferimento: Slide 18-20 da SGD_v1.pdf, lezione 01:55:22 - 01:57:13)*

**Problema**: Con γ costante, SGD non converge esattamente (regione confusion, bouncing).

**Soluzione**: **Diminuire γ nel tempo** → ridurre salti quando ci si avvicina al minimo!

#### 12.1 Schedules Comuni

**1. Step Decay** *(Slide 18)*:

$$\gamma_t = \gamma_0 \cdot \text{decay}^{\lfloor t / \text{drop\_every} \rfloor}$$

**Esempio**:
```python
γ_0 = 0.1
decay = 0.5
drop_every = 10 epochs

Epoch 0-9:   γ = 0.1
Epoch 10-19: γ = 0.05   # ÷2
Epoch 20-29: γ = 0.025  # ÷4
Epoch 30-39: γ = 0.0125 # ÷8
```

**Pro**: Semplice, interpretabile  
**Contro**: Richiede tuning di drop_every

**2. Exponential Decay**:

$$\gamma_t = \gamma_0 \cdot e^{-\lambda t}$$

**Esempio**:
```python
γ_0 = 0.1
λ = 0.01

t=0:   γ = 0.1
t=100: γ = 0.0368
t=200: γ = 0.0135
```

**Pro**: Smooth, continuo  
**Contro**: Decade troppo velocemente

**3. Polynomial Decay**:

$$\gamma_t = \gamma_0 \cdot \left(1 + \frac{t}{T}\right)^{-p}$$

**Esempio** (p=1, decay lineare inverso):
```python
γ_0 = 0.1
T = 1000
p = 1

t=0:    γ = 0.1
t=500:  γ = 0.0333  # 1/(1+0.5)
t=1000: γ = 0.05    # 1/(1+1)
```

**Pro**: Teoria SGD (p=1 → convergenza O(1/T))  
**Contro**: Lento nelle fasi finali

**4. Cosine Annealing** *(Slide 19, popolare in deep learning!)*:

$$\gamma_t = \gamma_{\min} + \frac{1}{2}(\gamma_{\max} - \gamma_{\min})\left[1 + \cos\left(\frac{\pi t}{T}\right)\right]$$

**Grafico**:
```
γ_t
  │
γ_max ●─────╲
  │         ╲
  │          ╲
  │           ╲
  │            ╲
γ_min          ●───────
  └──────────────────────► t
  0                      T
```

**Esempio**:
```python
γ_max = 0.1
γ_min = 0.001
T = 100 epochs

t=0:   γ = 0.1      # Start alto
t=25:  γ = 0.0714   # Scende
t=50:  γ = 0.0505   # Metà strada
t=75:  γ = 0.0214   # Quasi fine
t=100: γ = 0.001    # Minimo
```

**Pro**: 
- Smooth descent
- Molto usato (ResNet, BERT, GPT training)
- Converge bene empiricamente

**Contro**: Richiede conoscere T (num total epochs)

**5. Warmup + Cosine** *(best practice moderna!)*:

**Fase 1 (Warmup)**: γ cresce linearmente da 0 a γ_max (primi ~5% epochs)  
**Fase 2 (Cosine)**: Cosine annealing da γ_max a γ_min

```
γ_t
  │      ╱────╲
  │    ╱       ╲
  │  ╱          ╲
  │╱             ●
  ●───────────────────► t
  0  warmup   T
```

**Perché warmup**:
- All'inizio: pesi randomizzati → gradienti molto grandi e rumorosi
- γ alto subito → update troppo grandi → instabilità!
- Warmup stabilizza training iniziale

**Implementazione PyTorch**:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Warmup: 0 → 0.1 in 5 epochs
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)

# Cosine: 0.1 → 0.001 in 95 epochs
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=0.001)

# Combine
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# Training loop
for epoch in range(100):
    train_one_epoch(...)
    scheduler.step()  # Update γ
```

#### 12.2 Quando Usare Quale Schedule?

| Schedule | Best For | Typical Use Case |
|----------|----------|------------------|
| **Step decay** | Computer Vision classica | ResNet su ImageNet (decay @ epochs 30, 60, 90) |
| **Exponential** | RL, online learning | Policy gradient methods |
| **Polynomial (1/t)** | Teoria SGD, convex | Convex optimization research |
| **Cosine** | Deep learning moderno | Transformers (BERT, GPT), Vision (ViT) |
| **Warmup + Cosine** | **BEST modern practice** | Tutti i Transformer models |

**Citazione dalla lezione** *(01:55:55)*:
> "Tutte le librerie, TensorFlow, PyTorch, in tutte le librerie, potete definire uno scheduler per decidere l'entità della dimensione del passo. [...] Questa è la strategia più comunemente adottata in tutte le implementazioni della discesa del gradiente stocastico."

#### 12.3 Adaptive Learning Rates (Preview Lez20)

**Problema**: Un singolo γ per tutti i parametri → non ottimale!

**Idea**: Ogni parametro w_i ha il suo learning rate γ_i adattivo!

**Metodi** (vedremo nella prossima lezione):
1. **AdaGrad**: γ_i ∝ 1/√(Σ g_i²) → piccolo per parametri aggiornati spesso
2. **RMSprop**: γ_i ∝ 1/√(moving avg g_i²) → fixing AdaGrad decay
3. **Adam**: Momentum + RMSprop → **most popular optimizer!**

**Anteprima Adam**:
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,           # γ iniziale (di solito 10× più piccolo di SGD!)
    betas=(0.9, 0.999), # Momentum parameters
    eps=1e-8            # Numerical stability
)
```

**Prossima lezione** (Lez20 lunedì): Momentum, Nesterov, AdaGrad, RMSprop, Adam in dettaglio!

---

## CONCLUSIONE LEZ19

**Riassunto completo**:

**PARTE I: Gradient Descent Deterministico**
1. ✅ Convergenza Lipschitz convex: O(1/ε²) iterations
2. ✅ Convergenza L-smooth convex: **O(1/ε)** → molto meglio!
3. ✅ Convergenza strongly convex: **O(log(1/ε))** → esponenziale!
4. ✅ Line search (exact, backtracking Armijo)
5. ✅ Projected GD + Proximal GD (ottimizzazione vincolata)

**PARTE II: Stochastic Gradient Descent**
1. ✅ SGD: n× più veloce per iterazione
2. ✅ Convergenza Lipschitz: O(1/ε²) (come GD!)
3. ✅ Convergenza strongly convex: O(1/ε) (come L-smooth GD!)
4. ✅ Sampling strategies (with/without replacement)
5. ✅ ML objectives ≠ classical optimization (generalizzazione!)
6. ✅ Far-out vs confusion regions (rumore benefico)
