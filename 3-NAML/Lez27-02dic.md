# Capitolo 1: Introduzione al Teorema di Approssimazione Universale
## Riepilogo della Lezione Precedente: Prova Visiva
[00:00] Nella lezione precedente è stato introdotto il problema di determinare se una rete neurale sia in grado di rappresentare, o più precisamente di approssimare, una qualsiasi funzione. È stata fornita una "prova visiva" della veridicità di questa affermazione.
[00:11] Attraverso l'uso di alcuni artifici, partendo da una funzione sigmoide o da una funzione ReLU, è stata illustrata l'idea di fondo. Dal punto di vista visivo, il concetto chiave consiste nel costruire una sorta di base per uno specifico spazio funzionale.
[00:24] Nel primo caso, con la funzione sigmoide, sono state utilizzate come base le cosiddette "column function". Nel secondo caso, con la funzione ReLU, sono state costruite le "hat function" (funzioni a cappello). Sfruttando queste basi, è possibile rappresentare qualsiasi funzione con un'accuratezza arbitraria, semplicemente attraverso una combinazione lineare di tali funzioni di base. L'accuratezza della rappresentazione dipende direttamente dal numero di funzioni di base impiegate.
[00:47] In sostanza, il problema è stato riformulato in termini di un problema di approssimazione e di un teorema di interpolazione. Per la dimostrazione teorica, il percorso da seguire è leggermente diverso. In particolare, verrà analizzata la dimostrazione valida solo per le reti "shallow", ovvero reti neurali caratterizzate da un unico strato nascosto (hidden layer).
[01:07] La dimostrazione per reti più generali, le cosiddette "deep networks", è molto più complessa. Tuttavia, è fondamentale comprendere l'idea di base nel caso delle reti shallow, che è sufficiente per affermare che le reti neurali possono effettivamente approssimare qualsiasi funzione. Per comprendere le idee principali della dimostrazione, sono necessari alcuni concetti di analisi funzionale.
## Introduzione all'Analisi Funzionale e Motivazioni
[01:29] In questa lezione si inizierà con concetti molto basilari, alcuni dei quali potrebbero risultare già noti. Il materiale è stato riassunto in questa presentazione per fornire un riferimento unico e completo per lo studio e il ripasso.
[01:48] La motivazione principale per cui si ricorre all'analisi funzionale, almeno nel contesto delle reti neurali, è stata appena menzionata. Nel deep learning e nel machine learning in generale, si lavora con vettori a dimensione finita, i cosiddetti "feature vectors".
[02:06] L'obiettivo è estendere concetti come lunghezza, angolo e convergenza, tipici degli spazi $R^n$, a spazi a dimensione infinita. Il teorema di Cybenko, che verrà menzionato, è stato il primo risultato formale proposto riguardo al teorema di approssimazione universale per le reti shallow.
[02:24] A partire da questo risultato fondamentale, ne sono stati proposti molti altri, che condividono le stesse idee di base pur introducendo alcune tecnicalità o generalizzazioni. Pertanto, ci si concentrerà principalmente su questo teorema. L'articolo originale, di circa 15 pagine, è piuttosto chiaro e può essere consultato per approfondimenti.
# Capitolo 2: Concetti Fondamentali di Analisi Funzionale
## Spazi Vettoriali
[02:46] Si inizia con il concetto di spazio vettoriale. Uno spazio vettoriale reale, come già visto nella prima lezione, è un insieme dotato di alcune operazioni che devono soddisfare determinate proprietà.
[02:59] Gli spazi vettoriali sono importanti perché rappresentano un framework molto generale. Esiste un concetto ancora più generale, quello di spazio topologico, ma per scopi pratici gli spazi vettoriali costituiscono un punto di partenza adeguato. Essi forniscono il contesto ideale per trattare funzioni e oggetti a dimensione infinita.
[03:15] Di seguito sono riportati alcuni esempi di spazi vettoriali:
- Lo spazio $R^n$.
- L'insieme dei polinomi di grado minore o uguale a $k$ su un intervallo.
[03:26] - Lo spazio delle funzioni continue su un intervallo.
- Lo spazio $L^2$ delle funzioni a quadrato integrabile su un dominio generico $\Omega$. Si ritornerà su quest'ultimo spazio più avanti.
## Basi di uno Spazio Vettoriale
[03:37] Successivamente, viene introdotto il concetto di base.
*   **Base**: L'insieme minimo di funzioni linearmente indipendenti che permette di rappresentare qualsiasi altro vettore dello spazio come combinazione lineare delle funzioni della base stessa.
[03:50] Questo concetto è di fondamentale importanza, ad esempio, nel metodo degli elementi finiti. Una volta scelto il tipo di elemento finito (P1, P2, Q1, ecc.), è necessario decidere quale base utilizzare per quello specifico spazio.
[04:07] Le costanti $c_1, \dots, c_n$ sono i coefficienti che permettono di rappresentare una funzione. Ricordando l'animazione grafica della scorsa lezione, l'ultimo slider permetteva di regolare l'altezza della funzione. Quell'altezza corrisponde a questi coefficienti, assumendo che le funzioni di base abbiano un'altezza massima pari a 1, una convenzione comune.
[04:29] A seconda dello spazio considerato, le funzioni o i vettori della base sono oggetti diversi. Per $R^n$, le funzioni $\phi_1, \dots, \phi_n$ sono i vettori canonici (ad esempio, $0, 0, 1, \dots$).
## Prodotto Interno e Norma
### Definizione di Prodotto Interno
[04:40] Una volta definito uno spazio vettoriale, si possono introdurre altre due operazioni importanti: il prodotto interno e la norma.
*   **Prodotto Interno**: Una funzione che prende due elementi dello spazio e restituisce un numero.
[04:52] Questo numero deve soddisfare le seguenti proprietà:
1.  **Positività**: È positivo o nullo se calcolato sullo stesso argomento. È nullo se e solo se l'argomento è l'elemento nullo dello spazio ($u=0$).
2.  **Simmetria**: È simmetrico rispetto ai suoi argomenti.
3.  **Linearità**: È lineare rispetto a entrambi gli argomenti.
[05:06] Il prodotto interno è un concetto fondamentale perché generalizza la nozione di angolo e ortogonalità. In $R^n$ si è abituati a questo concetto, ma esso si estende a spazi più generali.
[05:15] Ad esempio, quando sono state trattate le matrici ortogonali, l'obiettivo era trovare una matrice in cui lo stesso spazio delle colonne fosse rappresentato da vettori unitari e ortogonali tra loro.
[05:29] L'ortogonalità può essere verificata anche quando gli elementi della base, le funzioni $\phi$, non sono vettori ma funzioni generiche.
### Esempi di Prodotto Interno
[05:36] Ecco alcuni esempi di prodotti interni:
- In $R^n$, è il prodotto scalare standard.
[05:43] - Per le funzioni continue su un intervallo, è la controparte continua del precedente: l'integrale del prodotto delle funzioni sull'intervallo (es. $\int_0^1 u(x)v(x)dx$).
- Nello spazio $C^1$, il prodotto interno include non solo il prodotto dei valori delle funzioni, ma anche il prodotto delle loro derivate prime.
### Definizione di Norma
[06:00] Ogni prodotto interno induce naturalmente una norma, definita come la radice quadrata del prodotto interno di un elemento con se stesso: $\|u\| = \sqrt{\langle u, u \rangle}$. Poiché il prodotto interno $\langle u, u \rangle$ è sempre non negativo, questa operazione è ben definita.
[06:09] Una norma è un'operazione che associa a un vettore un numero reale non negativo. Questo numero è zero se e solo se il vettore è l'elemento nullo.
[06:20] La norma soddisfa anche le seguenti proprietà:
1.  **Omogeneità**: $\|\alpha u\| = |\alpha| \|u\|$. Questa è una conseguenza diretta della linearità del prodotto interno.
2.  **Disuguaglianza triangolare**: $\|u+v\| \le \|u\| + \|v\|$.
[06:29] Il concetto di norma generalizza quello di lunghezza o distanza, a cui si è abituati in $R^n$, a qualsiasi spazio vettoriale. La sua importanza è cruciale, poiché la norma è utilizzata per verificare la convergenza e l'approssimazione.
[06:44] Come visto più volte, in un processo iterativo è necessario definire un criterio di arresto. Questo criterio spesso si basa sulla misura della norma della differenza tra due iterati consecutivi o sulla norma del residuo. In ogni caso, si deve misurare qualcosa che sia legato alla convergenza dell'algoritmo.
[07:03] In questo contesto, il concetto di convergenza si lega anche a quello di approssimazione. Un esempio è l'approssimazione di una funzione tramite una combinazione lineare di funzioni di base, come nel metodo degli elementi finiti.
[07:13] Negli elementi finiti, si possono derivare le cosiddette "stime dell'errore", che misurano l'errore commesso approssimando la soluzione vera di un'equazione differenziale alle derivate parziali (PDE) con una soluzione agli elementi finiti. Questo errore viene misurato tramite una norma.
[07:27] Tipicamente, si ottiene una disuguaglianza del tipo:
$$
\|U - U_h\| \le C \cdot h^p
$$
dove $U$ è la soluzione vera, $U_h$ è la soluzione approssimata, $C$ è una costante, $h$ è il parametro di discretizzazione (legato al numero di funzioni di base) e $p$ è l'ordine di convergenza del metodo.
[07:49] L'elemento chiave in questa stima è la norma utilizzata per misurare l'errore di approssimazione.
### Esempi di Norme
[07:54] Le norme comuni per i vettori, già incontrate, sono la norma euclidea (o norma 2), la norma massima (o norma infinito) e la norma 1.
[08:02] Per le funzioni, esistono le controparti continue. Ad esempio, la norma $L^2$ è definita come:
$$
\|f\|_{L^2} = \sqrt{\int |f(x)|^2 dx}
$$
Questi concetti sono già stati visti per le matrici, introducendo la norma 2 e la norma di Frobenius.
[08:18] Norme diverse forniscono misure diverse della "grandezza" di un oggetto. La scelta della norma da utilizzare dipende specificamente dal contesto applicativo.
[08:26] Ad esempio, nel contesto degli elementi finiti, la norma comunemente adottata è la norma $H^1$. Se si è interessati a controllare sia la funzione che il suo gradiente, si possono usare anche norme $H^2$ o $H^p$ per avere un controllo maggiore sul comportamento della funzione.
### Disuguaglianza di Cauchy-Schwarz
[08:42] Esiste una relazione fondamentale tra prodotto interno e norma, nota come disuguaglianza di Cauchy-Schwarz. Essa afferma che il valore assoluto del prodotto interno tra due elementi $u$ e $v$ è minore o uguale al prodotto delle loro norme:
$$
|\langle u, v \rangle| \le \|u\| \cdot \|v\|
$$
[08:52] Questa disuguaglianza è una generalizzazione della nota formula per il prodotto scalare tra vettori in $R^n$, che può essere definito come il prodotto delle norme per il coseno dell'angolo compreso tra i due vettori.
[09:06] La disuguaglianza di Cauchy-Schwarz generalizza questo concetto a spazi più ampi. Un esempio di applicazione per la norma $C^0$ di due funzioni è il seguente, dove a sinistra si ha il prodotto scalare in $C^0$ e a destra il prodotto delle norme delle due funzioni.
# Capitolo 3: Convergenza, Completezza e Spazi Funzionali
## Successioni di Cauchy e Successioni Convergenti
[09:20] Un punto importante riguarda la convergenza e la completezza di uno spazio. È necessario definire due concetti: la successione di Cauchy e la successione convergente.
[09:26] Una successione di elementi $\{v_i\}$ è detta **successione di Cauchy** se, per ogni $\epsilon > 0$, esiste un indice $N$ tale che per ogni coppia di indici $i, j > N$, la norma della differenza tra i termini $v_i$ e $v_j$ è minore di $\epsilon$:
$$
\|v_i - v_j\| < \epsilon
$$
[09:44] Il concetto di **convergenza** è leggermente diverso. Una successione $\{v_i\}$ è detta convergente a un limite $v$ se, per ogni $\epsilon > 0$, esiste un indice $N$ tale che per ogni $i > N$, la norma della differenza tra il termine $v_i$ e il limite $v$ è minore o uguale a $\epsilon$:
$$
\|v_i - v\| \le \epsilon
$$
[10:00] La domanda fondamentale è: qual è la relazione tra una successione di Cauchy e una successione convergente? Consideriamo un esempio classico. Prendiamo l'insieme $Q$ dei numeri razionali con la norma data dal valore assoluto.
[10:10] Consideriamo la successione definita da:
$$
\nu_n = \frac{1}{0!} + \frac{1}{1!} + \dots + \frac{1}{n!}
$$
[10:18] Questa è una successione di Cauchy in $Q$, ma converge al numero $e$ (il numero di Nepero), che non appartiene a $Q$ poiché è un numero irrazionale.
[10:26] In questo caso, si ha una successione che è di Cauchy ma non è convergente all'interno dello spazio $Q$.
## Completezza di uno Spazio
[10:32] Qui entra in gioco il concetto di **completezza**.
*   **Spazio Completo**: Uno spazio si dice completo se ogni successione di Cauchy al suo interno è anche una successione convergente (e il suo limite appartiene allo spazio stesso).
[10:39] La completezza è una proprietà fondamentale perché garantisce che, durante l'analisi di un algoritmo o la dimostrazione di un teorema, si rimanga sempre all'interno dello spazio di partenza.
[10:47] Ogni successione che si costruisce, ad esempio per una dimostrazione, avrà un limite che è ancora un elemento dello stesso spazio. Questo è cruciale per dimostrare la convergenza di un algoritmo.
## Spazi di Banach e di Hilbert
[10:58] Una volta definito il concetto di completezza, si possono introdurre ulteriori definizioni. Finora sono stati visti lo spazio vettoriale, lo spazio con prodotto interno e lo spazio normato. Si ricorda che uno spazio con prodotto interno è sempre anche uno spazio normato, poiché il prodotto interno induce una norma.
[11:13] Si definiscono ora:
- **Spazio di Banach**: È uno spazio vettoriale normato e completo. È importante notare che uno spazio di Banach ha una norma, ma non necessariamente un prodotto interno.
- **Spazio di Hilbert**: È uno spazio con prodotto interno e completo.
[11:26] Da queste definizioni, è chiaro che ogni spazio di Hilbert è anche uno spazio di Banach, poiché un prodotto interno induce sempre una norma. Il viceversa non è vero.
[11:34] L'importanza degli spazi di Hilbert e di Banach risiede proprio nella loro completezza: essi sono "chiusi" rispetto all'operazione di limite di successioni. Il limite di una successione di elementi dello spazio è ancora un elemento dello spazio stesso.
### Esempi di Spazi di Banach e di Hilbert
[11:46] Esempi di spazi di Banach:
- Lo spazio $R^n$ con una qualsiasi p-norma ($p \ge 1$).
[11:52] - Lo spazio $L^p$ e lo spazio $C^0$ (funzioni continue) con la norma infinito.
[11:59] Esempi di spazi di Hilbert:
- Lo spazio $R^n$ con il prodotto scalare standard (che induce la norma 2).
- Lo spazio $L^2$ e lo spazio $H^1$.
### Gerarchia degli Spazi Funzionali
[12:06] È possibile visualizzare una gerarchia di questi spazi:
1.  Spazi Vettoriali (la categoria più generale)
2.  Spazi Lineari Normati
3.  Spazi di Banach (spazi normati completi)
4.  Spazi di Hilbert (spazi con prodotto interno completi)
# Capitolo 4: Limiti dell'Integrale di Riemann e Introduzione all'Integrale di Lebesgue
## L'Integrale di Riemann
[12:13] Nei corsi di Analisi 1 e 2 si introduce il concetto di integrale di Riemann. Sebbene sia adatto a molte applicazioni, presenta alcuni svantaggi e, per la maggior parte delle applicazioni avanzate, non è sufficiente.
[12:26] Si richiama brevemente la sua definizione. Data una funzione su un intervallo $[a, b]$, si possono definire una somma inferiore e una somma superiore.
[12:34] Si partiziona l'asse delle ascisse in sottointervalli di ampiezza $\Delta x$. Su ciascuno di questi intervalli, la funzione raggiunge un minimo e un massimo.
[12:49] Si definiscono due tipi di rettangoli:
- Uno la cui altezza è il valore minimo della funzione nell'intervallo.
- Uno la cui altezza è il valore massimo della funzione nell'intervallo.
La somma delle aree di tutti i rettangoli costruiti usando i minimi è chiamata **somma inferiore**.
[13:02] Analogamente, la somma delle aree dei rettangoli costruiti usando i massimi è chiamata **somma superiore**.
[13:08] Se, al tendere a zero dell'ampiezza degli intervalli ($\Delta x \to 0$), il supremo delle somme inferiori e l'infimo delle somme superiori coincidono, allora questo valore comune è definito come l'**integrale di Riemann** della funzione, e rappresenta l'area sottesa dal suo grafico.
[13:21] Questo approccio è semplice, intuitivo e funziona per molte funzioni.
## Un Esempio Problematico: la Funzione di Dirichlet
[13:26] Viene ora presentato un esempio classico in cui l'integrale di Riemann fallisce. Si considera la **funzione di Dirichlet**, definita come:
$$
f(x) = \begin{cases} 1 & \text{se } x \in \mathbb{Q} \text{ (numeri razionali)} \\ 0 & \text{se } x \notin \mathbb{Q} \text{ (numeri irrazionali)} \end{cases}
$$
[13:35] In questo caso, per qualsiasi partizione dell'intervallo, la somma inferiore sarà sempre 0 (poiché in ogni intervallo esistono numeri irrazionali) e la somma superiore sarà sempre 1 (poiché in ogni intervallo esistono numeri razionali).
[13:44] Di conseguenza, la somma inferiore e la somma superiore rimangono sempre diverse, indipendentemente dalla finezza della partizione. Si conclude che la funzione di Dirichlet non è integrabile secondo Riemann.
[13:54] Sebbene questa sia una funzione piuttosto patologica, funzioni simili possono emergere in applicazioni pratiche. Avere una teoria dell'integrazione che non può gestire tali situazioni è limitante.
[14:07] Questo non è l'unico motivo per cui questo esempio costituisce una motivazione per cercare una teoria dell'integrazione più generale.
# Capitolo 5: Introduzione all'Integrazione di Lebesgue
## Le motivazioni dietro una nuova teoria dell'integrazione
[00:00] L'introduzione di un nuovo concetto di integrale non è motivata unicamente dalla necessità di trattare funzioni "strane", come la funzione di Dirichlet. Esistono altre ragioni di pari importanza che giustificano questa estensione. La prima motivazione è, appunto, la capacità di gestire funzioni con comportamenti non convenzionali. Una seconda ragione fondamentale riguarda l'integrazione multidimensionale. Ad esempio, quando si desidera calcolare un integrale di superficie in una dimensione superiore, come l'integrale di una funzione $f(x, y)$ su un dominio $\Omega$:
$$
\int_{\Omega} f(x, y) \, dx \, dy
$$
[00:15] È necessario sviluppare un concetto di integrale che semplifichi la dimostrazione del teorema di Fubini.
*   **Teorema di Fubini**: Questo teorema permette di scomporre un integrale multiplo in una sequenza di integrali semplici, calcolati uno dopo l'altro. In questo caso, consentirebbe di scrivere l'integrale doppio come:
    $$
    \int \left( \int f(x, y) \, dy \right) \, dx
    $$
[00:30] Questa scomposizione dell'integrazione è un'operazione cruciale in molte applicazioni. Un'altra motivazione di grande rilievo è la possibilità di scambiare l'operatore di integrale con l'operatore di limite. Questo scambio, sebbene intuitivo, risulta problematico e non sempre valido quando si utilizza l'integrale di Riemann. L'estensione che verrà introdotta, invece, permette di eseguire questa operazione in modo più robusto e generale.
## L'idea intuitiva dell'integrale di Lebesgue
[00:48] La generalizzazione che si andrà a considerare è nota come **integrale di Lebesgue**. L'idea di base può essere introdotta in modo intuitivo, poiché un approccio formale richiederebbe molto più tempo e dettagli tecnici. A sinistra, si ha il classico integrale di Riemann, basato sul concetto di somme inferiori e somme superiori.
[01:01] Se, al tendere a zero dell'ampiezza degli intervalli $\Delta x$, le somme inferiori e superiori convergono allo stesso valore, la funzione è definita Riemann-integrabile. Questo processo si basa sulla suddivisione dell'asse delle ascisse (l'asse $x$) in piccoli intervalli. L'idea fondamentale dell'integrale di Lebesgue è invertire questo approccio: invece di partizionare l'asse $x$, si partiziona l'asse delle ordinate (l'asse $y$).
[01:18] Una volta suddiviso l'asse $y$ in intervalli, per ciascuno di questi "strati" orizzontali, si deve considerare l'insieme di punti sull'asse $x$ la cui immagine tramite la funzione $f(x)$ cade all'interno di quello strato. Il passo successivo è calcolare la **misura** di questi insiemi di punti sull'asse $x$. Ad esempio, dopo aver identificato le intersezioni tra uno strato sull'asse $y$ e il grafico della funzione, si deve dare un significato alla misura (in questo caso, la lunghezza) dell'insieme risultante sull'asse $x$.
[01:36] Per un dato strato sull'asse $y$, l'insieme corrispondente sull'asse $x$ potrebbe essere l'unione di più intervalli disgiunti. In questo caso specifico, si hanno due insiemi separati da misurare.
[01:46] Si considerano quindi le intersezioni e si determina l'intervallo o l'insieme di intervalli corrispondente sull'asse $x$, per poi misurarne l'ampiezza totale, che si può chiamare $\Delta x$. Il punto cruciale di questo cambiamento di prospettiva è il passaggio da una somma basata sui valori minimo ($m$) e massimo ($M$) della funzione in un intervallo, moltiplicati per $\Delta x$, come nell'integrale di Riemann:
$$
\sum m_i \Delta x_i \quad \text{e} \quad \sum M_i \Delta x_i
$$
[02:04] a una situazione in cui l'integrale della funzione è dato da una somma di un altro tipo. L'integrale di Lebesgue è approssimato dalla somma dei valori $\alpha_i$ (che rappresentano i livelli sull'asse $y$) moltiplicati per la misura dell'insieme $S_i$:
$$
\int f \, d\mu \approx \sum_i \alpha_i \cdot \text{misura}(S_i)
$$
dove $S_i$ è l'insieme dei punti $x$ tali per cui il valore della funzione $f(x)$ appartiene all'intervallo $[\alpha_i, \alpha_{i+1})$:
$$
S_i = \{ x \mid f(x) \in [\alpha_i, \alpha_{i+1}) \}
$$
[02:21] Come si può notare dalla notazione $\int f \, d\mu$, il concetto centrale diventa la capacità di **misurare insiemi di punti**. Questo introduce il campo dell'analisi matematica noto come **teoria della misura**, che sviluppa metodi rigorosi per assegnare una misura a insiemi di punti.
# Capitolo 6: Teoria della Misura e sue Applicazioni
## Il concetto di misura e gli insiemi di misura nulla
[02:37] Al di là degli aspetti tecnici, un risultato fondamentale della teoria della misura è che **ogni insieme numerabile ha misura zero**.
*   **Insieme numerabile**: Un insieme i cui elementi possono essere messi in corrispondenza biunivoca con i numeri naturali (ad esempio, l'insieme dei numeri razionali $\mathbb{Q}$).
[02:44] In generale, una **misura** è una generalizzazione del concetto intuitivo di "lunghezza", "area" o "volume". Le sue proprietà fondamentali sono:
1.  La misura dell'insieme vuoto è zero.
2.  Possiede la proprietà di **additività**: la misura dell'unione di insiemi disgiunti $A_i$ è uguale alla somma delle loro singole misure.
    $$
    \text{misura}\left(\bigcup_i A_i\right) = \sum_i \text{misura}(A_i) \quad \text{se } A_i \cap A_j = \emptyset \text{ per } i \neq j
    $$
## Applicazione alla funzione di Dirichlet
[02:58] Tornando all'esempio della funzione di Dirichlet, l'insieme dei numeri razionali $\mathbb{Q}$ è un insieme numerabile, e quindi ha misura zero. Ciò significa che, nel calcolo dell'integrale di Lebesgue, il contributo dei punti razionali è nullo.
[03:09] Secondo questa nuova definizione di integrale, il calcolo procede come segue. La funzione assume solo due valori: 1 (sui razionali) e 0 (sugli irrazionali).
[03:18] Quando la funzione vale 1, ciò accade sull'insieme dei numeri razionali, che ha misura zero. Per tutti gli altri punti (gli irrazionali), la funzione vale 0, e la loro misura è "qualunque cosa rimanga". In questo modo, l'integrale di Lebesgue riesce a dare un significato ben definito all'integrale di questa funzione patologica.
[03:32] Il fatto che un insieme numerabile abbia misura zero è il concetto chiave che permette di trattare queste situazioni altrimenti problematiche.
[03:42] Si procede ora con il calcolo esplicito per la funzione di Dirichlet, $f(x) = \chi_{\mathbb{Q}}(x)$, definita sull'intervallo $[0, 1]$. L'integrale è dato dalla somma dei contributi dei due livelli di valore della funzione:
1.  Il valore della funzione è 1 sull'insieme dei razionali $\mathbb{Q} \cap [0, 1]$.
2.  Il valore della funzione è 0 sull'insieme degli irrazionali in $[0, 1]$.
L'integrale si calcola come:
$$
\int_{[0,1]} f(x) \, d\mu = 1 \cdot \text{misura}(\mathbb{Q} \cap [0,1]) + 0 \cdot \text{misura}([0,1] \setminus \mathbb{Q})
$$
[03:53] Poiché l'insieme dei razionali è numerabile, la sua misura è zero. L'insieme degli irrazionali in $[0,1]$ ha misura 1. Sostituendo questi valori, si ottiene:
$$
1 \cdot 0 + 0 \cdot 1 = 0
$$
Si è quindi riusciti a calcolare un valore significativo per questo integrale.
# Capitolo 7: Proprietà Fondamentali dell'Integrale di Lebesgue
## Vantaggi rispetto all'integrale di Riemann
[04:05] È stato dimostrato che, con la definizione di integrale di Lebesgue, si è in grado di calcolare l'integrale per funzioni complesse come quella di Dirichlet. Inoltre, si può dimostrare che la validità del teorema di Fubini (lo scambio degli ordini di integrazione) è molto più semplice ed elegante da provare nel contesto dell'integrale di Lebesgue.
[04:19] Un'altra proprietà fondamentale è il **Teorema della Convergenza Monotona**.
*   **Teorema della Convergenza Monotona**: Se $\{f_n\}$ è una successione di funzioni misurabili non negative che converge monotonicamente (cioè $f_1 \le f_2 \le \dots \le f_n \le \dots$) a una funzione $f$, allora è possibile scambiare il limite con l'integrale:
    $$
    \lim_{n \to \infty} \int f_n \, d\mu = \int \left( \lim_{n \to \infty} f_n \right) d\mu = \int f \, d\mu
    $$
[04:34] Un altro risultato importante è il **Teorema della Convergenza Dominata**.
*   **Teorema della Convergenza Dominata**: Se $\{f_n\}$ è una successione di funzioni che converge a una funzione $f$ "quasi ovunque" (cioè, converge ovunque tranne che su un insieme di misura zero), e se esiste una funzione integrabile $g$ tale che $|f_n(x)| \le g(x)$ per ogni $n$, allora si può nuovamente scambiare il limite con l'integrale:
    $$
    \lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu
    $$
[04:46] Questi due teoremi sono particolarmente importanti per gli argomenti che verranno trattati riguardo alle reti neurali.
# Capitolo 8: Spazi Funzionali $L^p$
## Definizione degli spazi $L^p$
[04:51] Una volta definito l'integrale di Lebesgue, è possibile definire gli **spazi $L^p$**.
[04:57] Lo spazio generico $L^p(\Omega)$ è l'insieme di tutte le funzioni definite su un dominio $\Omega$ a valori in $\mathbb{R}$, tali che la loro **norma $L^p$** sia finita. La norma è definita come:
- Per $1 \le p < \infty$:
  $$
  \|f\|_{L^p} = \left( \int_{\Omega} |f(x)|^p \, d\mu \right)^{1/p} < \infty
  $$
- Per $p = \infty$:
  $$
  \|f\|_{L^\infty} = \sup_{x \in \Omega} |f(x)| < \infty
  $$
  dove `sup` indica l'estremo superiore.
## Esempi di spazi $L^p$
[05:12] Si analizzano ora alcuni casi specifici:
- **Spazio $L^1$**: Impostando $p=1$, si ottiene lo spazio delle funzioni **assolutamente integrabili**, ovvero quelle per cui l'integrale del loro valore assoluto è finito.
- **Spazio $L^2$**: Impostando $p=2$, si ottiene lo spazio delle funzioni a **quadrato integrabile**. Queste funzioni sono associate a un concetto di "energia finita".
[05:21] Ad esempio, nel metodo degli elementi finiti, se si considera una corda elastica fissata ai suoi estremi e si definisce $u$ come lo spostamento verticale, la norma $L^2$ di $u$ è legata all'energia elastica interna della corda.
[05:41] - **Spazio $L^\infty$**: Questo è lo spazio delle **funzioni limitate**, poiché se l'estremo superiore del loro valore assoluto è un numero finito, significa che la funzione non può assumere valori infiniti.
## Proprietà degli spazi $L^p$
[05:49] Si può dimostrare che per $p$ compreso tra 1 e $\infty$, gli spazi $L^p$ sono **spazi di Banach**.
*   **Spazio di Banach**: Uno spazio vettoriale normato e completo (ogni successione di Cauchy converge a un elemento dello spazio stesso).
Inoltre, per il caso specifico $p=2$, lo spazio $L^2$ è anche uno **spazio di Hilbert**.
*   **Spazio di Hilbert**: Uno spazio di Banach dotato di un prodotto interno, che a sua volta induce una norma.
[05:57] Il prodotto interno in $L^2$ è definito come:
$$
\langle f, g \rangle = \int_{\Omega} f(x) g(x) \, d\mu
$$
La norma corrispondente è:
$$
\|f\|_{L^2} = \sqrt{\langle f, f \rangle} = \left( \int_{\Omega} |f(x)|^2 \, d\mu \right)^{1/2}
$$
[06:04] La struttura di spazio di Hilbert dello spazio $L^2$ è estremamente importante perché, avendo a disposizione sia una norma che un prodotto interno, è possibile definire tutte le proprietà geometriche classiche a cui si è abituati, come l'**ortogonalità**, le **proiezioni** e altro ancora.
[06:14] Questi concetti geometrici sono fondamentali quando si vuole analizzare la convergenza di approssimazioni o di metodi numerici.
## Relazioni di inclusione tra gli spazi $L^p$
[06:18] Se $\Omega$ è un dominio con misura finita, esiste una relazione di inclusione tra questi spazi.
[06:23] L'inclusione è la seguente:
$$
L^\infty(\Omega) \subset \dots \subset L^2(\Omega) \subset L^1(\Omega)
$$
Questo significa che una funzione in $L^\infty$ è anche in $L^2$, e una funzione in $L^2$ è anche in $L^1$. Inoltre, lo spazio delle funzioni continue su un dominio compatto, $C(\Omega)$, è un sottoinsieme di tutti questi spazi $L^p$.
# Capitolo 9: Introduzione alla Derivazione Debole
## Necessità di Generalizzare il Concetto di Derivata
[00:00] Analogamente a quanto discusso per il concetto di integrale, è necessario sviluppare una strategia per trattare le derivate di funzioni, in particolare di quelle che non sono derivabili in senso classico. In realtà, questa esigenza non si limita a funzioni specifiche, ma emerge in contesti fisici molto comuni.
[00:08] Si consideri, ad esempio, una corda elastica fissata ai suoi estremi, definita nell'intervallo $[0, L]$. Se si applica una forza verticale verso il basso esattamente nel punto medio della corda, la configurazione di equilibrio che ci si aspetta intuitivamente è quella di un triangolo.
[00:25] Immaginando un esperimento ideale, se si prende una corda elastica e si applica una pressione con un dito al centro, la forma che la corda assume è quella di un triangolo. Allo stesso modo, se si posiziona una palla molto pesante ma di dimensioni trascurabili (una massa puntiforme) al centro della corda, la configurazione risultante sarà triangolare.
[00:46] Supponendo per semplicità che tutte le costanti elastiche e le proprietà del materiale siano uguali a uno, l'equazione che descrive lo spostamento verticale $u(x)$ in funzione della forza applicata $F$ è l'equazione dell'elasticità.
[00:57] Per derivare questa equazione, si combinano l'equazione di equilibrio e un'equazione costitutiva. Trattandosi di una corda elastica, l'equazione costitutiva sarà una generalizzazione della legge di Hook per un materiale elastico. Questo processo è simile a quello usato per l'equazione del calore, dove si combinano l'equilibrio termico e la legge di Fourier.
[01:12] L'equazione risultante ha la forma:
$$
-u''(x) = F(x)
$$
Questa è analoga all'equazione del calore, dove $-T''(x) = R(x)$, con $T$ che rappresenta la temperatura e $R$ il termine di sorgente.
## Il Problema della Derivabilità Classica
[01:21] Si analizza ora il problema che sorge quando si applica questa equazione al caso della forza concentrata in un punto. L'equazione è stata derivata considerando le relazioni di equilibrio e l'equazione costitutiva della corda elastica, mentre $F$ rappresenta il termine di forzante.
[01:34] Se la forza è concentrata in un singolo punto, si sa intuitivamente che la soluzione $u(x)$ ha una forma a triangolo. Il problema diventa evidente quando si prova a calcolare le derivate di questa funzione.
[01:48] La funzione $u(x)$ che descrive la configurazione a triangolo è continua ma non derivabile nel punto di applicazione della forza. La sua derivata prima sarà una funzione a gradino (discontinua). Ad esempio, se la pendenza della prima metà è -1 e quella della seconda è +1, la derivata prima sarà una funzione che vale -1 in un intervallo e +1 nell'altro.
[02:04] Quando si tenta di calcolare la derivata seconda, ci si trova di fronte a un ostacolo insormontabile: la derivata prima è una funzione discontinua e quindi non è derivabile in senso classico nel punto di discontinuità. Questo evidenzia un'incoerenza tra il modello fisico, che prevede una soluzione di questo tipo, e gli strumenti matematici classici.
[02:18] Questa è la ragione per cui è necessario generalizzare non solo il concetto di integrale, ma anche quello di derivata. L'obiettivo è dare un significato alla derivata anche per funzioni come quella a triangolo, in ogni punto del loro dominio.
# Capitolo 10: La Derivata Debole e gli Spazi di Sobolev
## Definizione e Idea Concettuale della Derivata Debole
[02:32] L'idea alla base della derivata debole è concettualmente semplice e si fonda sull'introduzione di una classe di funzioni ausiliarie e sull'applicazione della formula di integrazione per parti.
[02:39] Queste funzioni ausiliarie, qui indicate con $\phi$, sono funzioni a supporto compatto.
*   **Funzione a supporto compatto**: Una funzione si dice a supporto compatto su un intervallo se è non nulla solo all'interno di un sottoinsieme chiuso e limitato (compatto) dell'intervallo, e si annulla al di fuori di esso, in particolare agli estremi.
[02:45] Supponiamo di voler dare un senso alla derivata di una funzione come $u'$, che presenta una discontinuità. L'approccio consiste nel moltiplicare $u'$ per una funzione test $\phi$, che è molto regolare (ad esempio, di classe $C^1$ o addirittura $C^\infty$).
[02:59] Poiché la funzione test $\phi$ ha supporto compatto, essa si annulla agli estremi dell'intervallo di integrazione. Questo permette di applicare la formula di integrazione per parti senza che compaiano termini di bordo. L'operatore di derivata viene così "trasferito" dalla funzione originale $u$ alla funzione test regolare $\phi$. L'uguaglianza che si ottiene è:
$$
\int u' \phi \, dx = - \int u \phi' \, dx
$$
[03:10] Questa relazione, derivata dall'integrazione per parti, costituisce la definizione stessa di derivata debole. Si dice che una funzione $g$ è la derivata debole di una funzione $u$ se vale la seguente uguaglianza per ogni funzione test $\phi$:
$$
\int g \phi \, dx = - \int u \phi' \, dx
$$
[03:19] Le funzioni test $\phi$ appartengono a uno spazio di funzioni molto regolari, tipicamente $C^\infty$ (infinitamente derivabili), e a supporto compatto. Questo garantisce che non ci siano termini al contorno nell'integrazione per parti.
[03:33] Sfruttando questa idea, si può definire il concetto di derivata debole. È importante notare che gli integrali in questa definizione devono essere intesi in senso di Lebesgue, per poter gestire funzioni meno regolari.
## Esempio e Proprietà della Derivata Debole
[03:40] Si consideri la funzione $u(x) = 3 - |x|$, che ha una forma a "V" rovesciata, simile all'esempio della corda elastica. La sua derivata debole è una funzione a gradino, che vale +1 per $x < 0$ e -1 per $x > 0$. Questa definizione è valida anche nel punto $x=0$, dove la funzione $u$ non è derivabile in senso classico.
[04:00] Le derivate deboli condividono molte proprietà con le derivate classiche, tra cui:
*   **Linearità**: La derivata debole di una combinazione lineare di funzioni è la combinazione lineare delle loro derivate deboli.
*   **Regola del prodotto (Leibniz rule)**.
*   **Regola della catena (Chain rule)**.
Inoltre, se una funzione è derivabile in senso classico, la sua derivata debole coincide con la derivata classica.
## Gli Spazi di Sobolev: Definizione e Struttura
[04:13] L'introduzione delle derivate deboli permette di definire una nuova classe di spazi funzionali: gli spazi di Sobolev.
[04:19] Lo spazio di Sobolev $W^{k,p}(\Omega)$ è definito come l'insieme delle funzioni $u$ che appartengono allo spazio $L^p(\Omega)$ e le cui derivate deboli, fino all'ordine $k$, appartengono anch'esse a $L^p(\Omega)$.
*   **Spazio $L^p(\Omega)$**: È lo spazio delle funzioni il cui valore assoluto elevato alla potenza $p$ è integrabile secondo Lebesgue sull'insieme $\Omega$.
*   L'indice $p$ è legato allo spazio di integrabilità della funzione.
*   L'indice $k$ è legato alla regolarità della funzione, indicando l'ordine massimo di derivabilità (in senso debole) richiesto.
[04:40] La norma definita su questo spazio è data dalla somma delle norme $L^p$ di tutte le derivate deboli, dall'ordine 0 (la funzione stessa) fino all'ordine $k$:
$$
\|u\|_{W^{k,p}} = \left( \sum_{|\alpha| \le k} \|D^\alpha u\|_{L^p}^p \right)^{1/p}
$$
Questa struttura è analoga a quella della norma per gli spazi di funzioni continue $C^k$, dove la norma include la funzione e le sue derivate.
## Gli Spazi di Hilbert $H^k$
[05:00] Un caso particolare di grande importanza si ha quando $p=2$. Gli spazi di Sobolev $W^{k,2}$ sono indicati anche come $H^k$ e sono fondamentali nelle applicazioni pratiche.
[05:12] Per $k=1$, lo spazio $H^1$ è l'insieme delle funzioni che sono in $L^2$ e le cui derivate prime (il gradiente) sono anch'esse in $L^2$.
[05:20] Essendo uno spazio di Hilbert, $H^1$ è dotato di un prodotto scalare, definito come:
$$
(u, v)_{H^1} = \int (uv + \nabla u \cdot \nabla v) \, dx
$$
La norma associata è la radice quadrata di questo prodotto scalare applicato a una funzione con se stessa, e può essere scritta come:
$$
\|u\|_{H^1}^2 = \|u\|_{L^2}^2 + \|\nabla u\|_{L^2}^2
$$
[05:36] Questa espressione mostra che la norma in $H^1$ è composta dalla norma in $L^2$ della funzione e dalla norma in $L^2$ del suo gradiente. Per $k > 1$, si definiscono analogamente gli spazi $H^2, H^3, \dots$, che sono altrettanto importanti, sebbene $H^1$ e $H^2$ siano i più comuni nelle applicazioni.
[05:55] In generale, lo spazio $W^{k,p}$ è uno spazio di Banach per $1 \le p < \infty$, mentre lo spazio $H^k$ (con $p=2$) è uno spazio di Hilbert. L'introduzione delle derivate deboli e degli spazi di Sobolev permette di lavorare con funzioni non regolari, come quelle che presentano "spigoli" (kinks).
[06:09] Questi spazi, in particolare gli spazi $H^k$, sono lo strumento standard utilizzato in analisi funzionale per dimostrare l'esistenza e l'unicità delle soluzioni di equazioni differenziali alle derivate parziali.
## Relazione tra $H^1$ e Funzioni Continue
[06:20] La relazione tra lo spazio $H^1$ e lo spazio delle funzioni continue ($C^0$) dipende dalla dimensione $d$ del dominio.
*   **In una dimensione ($d=1$)**: Le funzioni in $H^1$ sono continue. Lo spazio $H^1$ è equivalente allo spazio delle funzioni continue.
*   **In due dimensioni ($d=2$)**: Le funzioni in $H^1$ possono presentare discontinuità isolate in punti. Un insieme di punti, anche infinito, ha misura nulla, quindi la funzione rimane in $H^1$.
*   **In tre dimensioni ($d=3$)**: Le funzioni in $H^1$ possono essere discontinue lungo linee o curve nello spazio.
[06:47] Pertanto, l'equivalenza tra $H^1$ e lo spazio delle funzioni continue è valida solo nel caso monodimensionale.
# Capitolo 11: Operatori e Funzionali Lineari
## Operatori Lineari e Limitatezza
[06:54] Il concetto di operatore lineare generalizza quello di matrice a spazi vettoriali generici. Un operatore lineare $T$ è un'applicazione tra due spazi vettoriali $X$ e $Y$ che soddisfa la proprietà di linearità.
[07:04] Dati due scalari $\alpha, \beta$ e due vettori $u, v \in X$, un operatore $T: X \to Y$ è lineare se:
$$
T(\alpha u + \beta v) = \alpha T(u) + \beta T(v)
$$
[07:12] Un operatore lineare $T$ si dice **limitato** se esiste una costante $C$ tale che la norma dell'immagine $T(u)$ è maggiorata dalla norma dell'argomento $u$, moltiplicata per $C$:
$$
\|T(u)\|_Y \le C \|u\|_X
$$
Qui, $\| \cdot \|_Y$ è la norma nello spazio di arrivo $Y$, e $\| \cdot \|_X$ è la norma nello spazio di partenza $X$.
[07:30] La norma di un operatore lineare $T$ è definita come il più piccolo valore di $C$ per cui la disuguaglianza di limitatezza è soddisfatta. Questa definizione è analoga a quella della norma-p di una matrice:
$$
\|T\| = \sup_{u \neq 0} \frac{\|T(u)\|_Y}{\|u\|_X}
$$
La condizione $u \neq 0$ è necessaria per evitare la divisione per zero.
[07:55] Gli operatori limitati sono importanti perché garantiscono che l'applicazione dell'operatore non produca un "blow-up", ovvero che l'output rimanga controllato. Questo è analogo a quanto accade con le matrici, che hanno sempre norma finita.
## Funzionali Lineari e Spazio Duale
[08:07] Un **funzionale lineare** è un caso particolare di operatore lineare in cui lo spazio di arrivo è l'insieme dei numeri reali (o complessi). È un'applicazione $L: X \to \mathbb{R}$ che associa a ogni elemento dello spazio vettoriale $X$ un numero.
[08:18] L'introduzione dei funzionali lineari permette di definire lo **spazio duale** $X'$, che è l'insieme di tutti i funzionali lineari e limitati definiti su $X$.
[08:28] La norma di un funzionale lineare $L \in X'$ è definita in modo analogo alla norma di un operatore:
$$
\|L\|_{X'} = \sup_{u \neq 0} \frac{|L(u)|}{\|u\|_X}
$$
Poiché $L(u)$ è un numero, al numeratore si usa il valore assoluto invece della norma. Lo spazio duale $X'$ è uno spazio di Banach.
[08:41] Intuitivamente, lo spazio duale può essere visto come lo spazio che rappresenta tutte le possibili "misure" che si possono effettuare sugli elementi dello spazio $X$.
# Capitolo 12: Teorema di Rappresentazione di Riesz e Teoria dell'Approssimazione
## Teorema di Rappresentazione di Riesz
[08:52] I funzionali lineari sono oggetti piuttosto astratti. Il **Teorema di Rappresentazione di Riesz** stabilisce una connessione fondamentale tra i funzionali lineari e i prodotti scalari, rendendoli più concreti e maneggevoli.
[09:05] Il teorema afferma che, se $H$ è uno spazio di Hilbert e $L$ è un funzionale lineare e limitato su $H$, allora esiste un unico elemento $u \in H$ tale che, per ogni $v \in H$:
$$
L(v) = (v, u)_H
$$
In altre parole, l'applicazione del funzionale $L$ a un elemento $v$ è equivalente al prodotto scalare tra $v$ e un elemento fisso $u$.
[09:18] Inoltre, la norma del funzionale $L$ nello spazio duale è uguale alla norma dell'elemento $u$ nello spazio di Hilbert:
$$
\|L\|_{H'} = \|u\|_H
$$
Questo significa che, in uno spazio di Hilbert, ogni funzionale lineare può essere identificato con un prodotto scalare.
## Esempi di Applicazione del Teorema di Riesz
[09:26] Vediamo alcuni esempi concreti:
1.  **Spazio $\mathbb{R}^n$**: Si consideri un funzionale lineare $L$ su $\mathbb{R}^n$. Fissato un vettore $y \in \mathbb{R}^n$, il funzionale $L(x) = x \cdot y$ (prodotto scalare) è un funzionale lineare. Ad esempio, il funzionale che calcola la somma delle componenti di un vettore $x$, $L(x) = \sum_i x_i$, può essere rappresentato come il prodotto scalare tra $x$ e il vettore $u = (1, 1, \dots, 1)$. In questo caso, il vettore $u$ è l'elemento unico garantito dal teorema di Riesz.
2.  **Spazio $L^2([0,1])$**: Si consideri il funzionale $L(f) = \int_0^1 f(x) \, dx$. Secondo il teorema di Riesz, esiste una funzione unica $u \in L^2([0,1])$ tale che questo integrale possa essere scritto come un prodotto scalare in $L^2$:
    $$
    \int_0^1 f(x) \, dx = \int_0^1 f(x) u(x) \, dx
    $$
    In questo caso, la funzione $u(x)$ è la funzione che vale costantemente 1 sull'intervallo $[0,1]$. Se il funzionale fosse stato, ad esempio, l'integrale di $f(x)$ solo su $[0, 1/2]$, la funzione $u(x)$ corrispondente sarebbe stata 1 su $[0, 1/2]$ e 0 altrove.
## Il Problema dell'Approssimazione di Funzioni
[10:30] Una domanda fondamentale, che condurrà al Teorema di Cybenko, è: è possibile approssimare una funzione qualsiasi con una combinazione di funzioni più semplici, come le funzioni ReLU, le tangenti iperboliche o altre?
[10:42] Nelle reti neurali, l'obiettivo è proprio questo: approssimare la relazione (una funzione) tra input e output. Questa approssimazione viene costruita attraverso la composizione di funzioni semplici, le funzioni di attivazione, presenti nei vari strati nascosti della rete.
[11:02] Nella teoria classica dell'approssimazione, si utilizzano strumenti come i polinomi di Lagrange o le basi di Fourier per approssimare una funzione. Altri metodi, come quelli basati su PCA o SVD, sfruttano strutture a basso rango nascoste nei dati per ottenere un'approssimazione.
## Il Teorema di Weierstrass e il Concetto di Densità
[11:18] Un teorema chiave in questo contesto è il **Teorema di Approssimazione di Weierstrass**, il quale afferma che ogni funzione continua definita su un intervallo chiuso e limitato può essere approssimata uniformemente da un polinomio.
[11:25] Formalmente, per ogni funzione $f \in C([a,b])$ e per ogni tolleranza $\epsilon > 0$, esiste un polinomio $P$ tale che:
$$
\|f - P\|_{\infty} < \epsilon
$$
dove $\| \cdot \|_{\infty}$ è la norma del massimo (o norma uniforme).
[11:35] Questo risultato è un'applicazione di un concetto più generale e fondamentale: la **densità**.
[11:44] Un insieme $S$ si dice **denso** in uno spazio normato $X$ (dove $S$ è un sottoinsieme di $X$) se per ogni elemento $u \in X$ e per ogni tolleranza $\epsilon > 0$, è sempre possibile trovare un elemento $s \in S$ tale che la distanza tra $u$ e $s$ sia minore di $\epsilon$:
$$
\|u - s\|_X < \epsilon
$$
[12:05] In altre parole, un insieme $S$ è denso in $X$ se i suoi elementi possono "avvicinarsi" arbitrariamente a qualsiasi elemento di $X$. Indipendentemente dal punto di destinazione $u$ e da quanto piccola sia la tolleranza $\epsilon$, si troverà sempre un elemento $s \in S$ sufficientemente vicino a $u$.
## Esempi di Insiemi Densi
[12:25]
1.  I **polinomi** sono densi nello spazio delle funzioni continue $C([a,b])$. Questo è esattamente ciò che afferma il Teorema di Weierstrass.
2.  I **polinomi trigonometrici** sono densi nello spazio $L^2([0, 2\pi])$. Questo risultato è alla base dello sviluppo in serie di Fourier.
3.  Le **funzioni continue** sono dense negli spazi $L^p$.
## Densità e Teorema di Cybenko
[12:42] Il **Teorema di Cybenko** (o Teorema di Approssimazione Universale) stabilisce che, dato uno spazio di funzioni target $X$, l'insieme delle funzioni rappresentabili da una rete neurale con un singolo strato nascosto è denso in $X$.
[12:54] Più precisamente, il teorema dimostra che le funzioni della forma $\sum_i c_i \sigma(w_i \cdot x + b_i)$, dove $\sigma$ è una funzione di attivazione che soddisfa certi requisiti, formano un insieme denso nello spazio delle funzioni che si vogliono approssimare.
[13:12] Questo è il motivo per cui il concetto di densità è così cruciale per la teoria delle reti neurali: garantisce che, in linea di principio, una rete neurale sufficientemente grande possa approssimare qualsiasi funzione continua con un grado di precisione arbitrario.
[13:21] La norma utilizzata nella definizione di densità dipende ovviamente dallo spazio $X$ in cui si sta lavorando. Poiché $S$ è un sottoinsieme di $X$, la norma è la stessa per entrambi.
## Concetti Chiave per la Dimostrazione
[13:32] Per dimostrare il Teorema di Cybenko, i concetti fondamentali che verranno utilizzati sono:
*   **Densità**
*   **Compattezza**
*   **Convergenza**
*   Proprietà degli **spazi di Hilbert**
Questi strumenti matematici saranno impiegati per analizzare le proprietà delle funzioni generate dalle reti neurali.
[13:55] La parte teorica della lezione si conclude qui. Seguirà una pausa di dieci minuti prima di passare alla discussione del progetto.