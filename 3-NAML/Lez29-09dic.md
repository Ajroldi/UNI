# Introduzione alla Teoria dell'Approssimazione Universale
## Contesto e Obiettivo della Lezione
[00:00] In questa lezione, l'analisi riprende dall'esame della proprietà di approssimazione universale delle reti neurali. Dopo aver fornito una spiegazione intuitiva per il caso unidimensionale (1D) e aver accennato a come estenderla al caso bidimensionale (2D), si procederà con una trattazione più formale.
[00:11] A tale scopo, verranno utilizzati alcuni strumenti di analisi funzionale introdotti in precedenza. L'obiettivo è presentare un risultato formale che giustifichi la capacità delle reti neurali di approssimare funzioni complesse.
## Impostazione del Problema di Approssimazione
[00:20] Si definisce la notazione che verrà utilizzata: $X$ rappresenta il vettore di input (il *feature vector*), mentre $Z$ è il target, ovvero l'output desiderato della rete neurale. L'obiettivo è trovare una funzione $z = f(x)$ che rappresenti in modo adeguato una rete neurale.
[00:34] Una rete neurale, infatti, può essere vista come una funzione costruita secondo una struttura specifica, determinata dalla sua architettura. Di conseguenza, il problema consiste nel trovare una funzione $z = f(x)$ capace di approssimare una data serie di dati di input e i corrispondenti output.
## Collegamento con la Teoria dell'Approssimazione
[00:48] Questo problema può essere inquadrato nel contesto più generale della teoria dell'approssimazione. In tale teoria, l'obiettivo è, data una funzione $f$ e alcuni punti campionati da essa, costruire una funzione approssimante che descriva il comportamento della funzione originale entro una certa tolleranza prestabilita.
[01:04] Esistono diverse tecniche per raggiungere questo scopo, come l'interpolazione di Lagrange (semplice o composita) o l'uso delle serie di Fourier, metodi già noti dai corsi di analisi numerica.
[01:16] Nel caso specifico delle reti neurali, la domanda è se, data una qualsiasi funzione, sia sempre possibile progettare una rete neurale in grado di approssimarla con una tolleranza definita dall'utente.
[01:26] La funzione da approssimare appartiene a un dato spazio di funzioni $S$, nel quale è definita una distanza $d$.
# Il Concetto di Approssimatore Universale
## Definizione di Densità e Approssimatore Universale
[01:31] Sulla base di quanto visto in precedenza, è possibile definire formalmente cosa si intende per approssimatore universale. Una rete neurale è un approssimatore universale per uno spazio di funzioni $S$ con una data distanza $D$ se lo spazio di tutti i possibili output della rete, indicato con $U$, è *D-denso* in $S$.
[01:45] Si richiama il concetto di densità: uno spazio $U$ è denso in uno spazio più grande $S$ se, per ogni elemento scelto nello spazio grande, è sempre possibile trovare un elemento dello spazio più piccolo che lo approssima con la precisione desiderata.
[01:58] Formalmente, ciò significa che per ogni funzione $f \in S$ e per ogni tolleranza $\varepsilon > 0$, è possibile trovare una funzione $g \in U$ (che rappresenta una rete neurale) tale che la distanza tra $f$ e $g$ sia arbitrariamente piccola.
[02:07] In altre parole, la distanza $D(f, g)$ deve essere minore di $\varepsilon$:
```math
\forall f \in S, \forall \varepsilon > 0, \exists g \in U \text{ tale che } D(f, g) < \varepsilon
```
Questo implica che si è in grado di approssimare qualsiasi funzione $f$ con una precisione arbitraria utilizzando una funzione $g$ appartenente allo spazio delle reti neurali.
[02:13] Il risultato che verrà dimostrato è che questa proprietà è vera sotto specifiche ipotesi.
## Il Ruolo della Funzione di Attivazione
[02:21] Un ingrediente chiave nella definizione di una rete neurale, oltre all'architettura (numero di strati, numero di neuroni per strato), è la funzione di attivazione.
[02:34] In questo contesto, l'analisi si concentra sulle funzioni *sigmoidali*. È importante distinguere tra la funzione sigmoide e la funzione sigmoidale: quest'ultima è una generalizzazione della prima.
[02:42] Una funzione $\sigma: \mathbb{R} \to [0, 1]$ è definita sigmoidale se soddisfa le seguenti condizioni sui limiti:
```math
\lim_{x \to -\infty} \sigma(x) = 0
```
```math
\lim_{x \to +\infty} \sigma(x) = 1
```
[02:56] Qualsiasi funzione che presenti questo comportamento asintotico è una funzione sigmoidale. Esempi includono la classica funzione a "S", ma anche funzioni con andamenti diversi che rispettino i limiti, come la tangente iperbolica opportunamente riscalata e traslata.
# La Proprietà Discriminatoria
## Definizione di Funzione Discriminatoria
[03:08] Una proprietà fondamentale per la dimostrazione del teorema di approssimazione universale è la cosiddetta *proprietà discriminatoria* della funzione di attivazione.
[03:15] Dato un intero $m$, una funzione di attivazione $f: \mathbb{R} \to \mathbb{R}$ è detta *$m$-discriminatoria* se l'unica misura $\mu$ tale che il seguente integrale sia nullo è la misura nulla stessa:
```math
\int_{\mathbb{R}^m} f(w^T x + \theta) \, d\mu(x) = 0
```
[03:26] È importante notare che questi integrali sono definiti secondo Lebesgue, motivo per cui si utilizza la notazione $d\mu$ invece di $dx$. Una delle differenze principali tra l'integrale di Riemann e quello di Lebesgue risiede nella capacità di quest'ultimo di definire una misura per gli insiemi nel dominio della funzione.
[03:42] In generale, una funzione di attivazione è detta *discriminatoria* se la proprietà sopra descritta vale per qualsiasi dimensione $m$.
## Significato Intuitivo della Proprietà Discriminatoria
[03:50] Il significato pratico di questa proprietà è che, se una funzione è discriminatoria, l'integrale visto sopra si annulla solo quando la misura è nulla. Questo implica che, per un dato input $x$, il risultato dell'operazione integrale è quasi sempre diverso da zero.
[04:02] Si dice comunemente che una funzione discriminatoria è *non distruttiva* rispetto all'input. In altre parole, data un'informazione in ingresso, la funzione è in grado di restituire un'informazione in uscita, senza "appiattire" o annullare l'input.
[04:11] Questa caratteristica è un punto cruciale nella dimostrazione del teorema di approssimazione universale.
# Strumenti Matematici per la Dimostrazione
## Richiami di Analisi Funzionale
[04:16] Per procedere con la dimostrazione, è necessario richiamare alcuni concetti di analisi funzionale.
[04:20] Si considera un insieme compatto $K \subset \mathbb{R}^n$. Con $I_n$ si indica l'ipercubo unitario in $n$ dimensioni.
- Per $n=1$, $I_1$ è l'intervallo $[0, 1]$.
- Per $n=2$, $I_2$ è il quadrato di lato unitario.
- Per $n=3$, $I_3$ è il cubo di lato unitario, e così via.
[04:31] Si definiscono inoltre i seguenti spazi:
- $C(K)$: lo spazio delle funzioni continue definite sull'insieme compatto $K$.
- $M(I_n)$: lo spazio delle misure regolari definite sull'ipercubo $I_n$. Una misura è detta *regolare* quando, pur potendo fornire valori numerici diversi per una stessa funzione rispetto ad altre misure, cattura in modo consistente l'idea di "spazio occupato" dalla funzione rispetto allo spazio totale.
## Teorema di Rappresentazione di Riesz
[04:47] Un risultato fondamentale è il Teorema di Rappresentazione di Riesz, già menzionato in precedenza.
[04:53] Questo teorema afferma che, dato un funzionale lineare $L$, è sempre possibile trovare una misura $\mu$ tale che il funzionale possa essere espresso come un integrale. La potenza del teorema risiede nel garantire l'esistenza di una misura $\mu$ che permette di scrivere:
```math
L(f) = \int f \, d\mu
```
dove $f$ è l'argomento del funzionale.
## Teorema di Hahn-Banach
[05:07] Un altro strumento essenziale è il Teorema di Hahn-Banach.
[05:12] La versione generale del teorema è complessa, ma ai fini della dimostrazione è sufficiente una sua riformulazione più pratica.
[05:17] Si considera uno spazio normato $X$ e un suo sottospazio lineare $U$. Se $U$ è *non denso* in $X$ (cioè non soddisfa la proprietà di densità vista prima), allora è sempre possibile trovare un funzionale lineare $L$ definito su tutto lo spazio $X$ con le seguenti proprietà:
- $L$ è non nullo su $X$.
- $L$ è identicamente nullo sul sottospazio $U$.
[05:30] Visivamente, si può immaginare uno spazio grande $X$ contenente un sottospazio più piccolo $U$.
[05:38] Se $U$ non è denso in $X$, esiste un funzionale lineare $L$ tale che $L \neq 0$ per gli elementi di $X$ che non appartengono a $U$, mentre $L(h) = 0$ per ogni $h \in U$. Questa è la formulazione che sarà utile.
## Applicazione Combinata dei Teoremi
[05:48] Questi concetti vengono ora applicati allo spazio $C(I_n)$, ovvero lo spazio delle funzioni continue definite sull'ipercubo n-dimensionale.
[05:53] Si suppone che $U$ sia un sottospazio lineare non denso di $C(I_n)$. Combinando i due teoremi precedenti, si può dedurre quanto segue:
1.  Dal **Teorema di Hahn-Banach**: esiste un funzionale lineare $L$ non nullo su $C(I_n)$ ma nullo su $U$.
2.  Dal **Teorema di Rappresentazione di Riesz**: questo funzionale $L$ può essere rappresentato tramite un integrale.
[06:06] Di conseguenza, si può affermare che esiste una misura $\mu$ tale che il funzionale può essere scritto come $\int h \, d\mu$, dove $h$ è una funzione.
[06:13] Poiché il funzionale è nullo su $U$, si conclude che per ogni funzione $h \in U$, l'integrale corrispondente deve essere zero:
```math
\int_{I_n} h(x) \, d\mu(x) = 0, \quad \forall h \in U
```
[06:23] Questo risultato, che unisce il teorema di Hahn-Banach e il teorema di rappresentazione di Riesz, è fondamentale per la dimostrazione che seguirà.
# Teorema di Approssimazione Universale per Reti a Singolo Strato
## Enunciato del Teorema
[06:33] Si è ora pronti per enunciare il teorema. Si considera una funzione di attivazione $\sigma$ che sia *discriminatoria*.
[06:38] Si analizza una classe di funzioni $g(x)$ definite come segue:
```math
g(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^T x + \theta_j)
```
Questa espressione rappresenta una rete neurale con un singolo strato nascosto e $N$ neuroni.
[06:46] In questa formula:
- $w_j \in \mathbb{R}^n$ è il vettore dei pesi sinaptici del neurone $j$.
- $\theta_j \in \mathbb{R}$ è il bias del neurone $j$.
- $\alpha_j \in \mathbb{R}$ è il peso di output associato al neurone $j$.
[06:53] Questa struttura richiama l'approccio intuitivo visto in precedenza: si creano funzioni a gradino (usando la sigmoide), le si combinano per formare blocchi e si scala l'altezza di ciascun blocco con un coefficiente $\alpha_j$ per far corrispondere l'output della rete al valore della funzione target in un punto specifico.
[07:10] L'idea è una combinazione lineare di funzioni di attivazione traslate e riscalate.
[07:18] Il teorema afferma che, se la funzione di attivazione $\sigma$ è discriminatoria, l'insieme di tutte le funzioni $g(x)$ di questo tipo è *denso* in $C(I_n)$.
## Significato Pratico del Teorema
[07:23] In pratica, questo significa che lo spazio $U$ delle funzioni $g(x)$ è denso nello spazio $S = C(I_n)$ delle funzioni continue sull'ipercubo.
[07:34] Di conseguenza, per qualsiasi funzione continua definita su $I_n$, è possibile trovare una rete neurale con un singolo strato nascosto (con un numero sufficiente di neuroni $N$) che la approssima con una precisione arbitrariamente alta.
[07:47] Il teorema garantisce quindi che una rete neurale con un singolo strato e $N$ neuroni può approssimare qualsiasi funzione continua.
## Dimostrazione del Teorema (per Assurdo)
[07:55] La dimostrazione si basa su una strategia per contraddizione.
[08:00] Si indica con $U$ l'insieme delle funzioni $g(x)$ definite come sopra.
[08:06] **Ipotesi per assurdo**: si assume che $U$ *non* sia denso in $C(I_n)$.
[08:11] Se $U$ non è denso, allora, per il risultato combinato di Hahn-Banach e Riesz derivato in precedenza, deve esistere una misura $\mu$ non nulla tale che l'integrale di ogni funzione $h \in U$ rispetto a questa misura sia zero.
[08:16] Formalmente:
```math
\exists \mu \neq 0 \text{ tale che } \int_{I_n} h(x) \, d\mu(x) = 0, \quad \forall h \in U
```
[08:26] Si sa che le funzioni $h \in U$ hanno la forma specifica di $g(x)$. Si sostituisce quindi l'espressione di $g(x)$ nell'integrale:
```math
\int_{I_n} \left( \sum_{j=1}^{N} \alpha_j \sigma(w_j^T x + \theta_j) \right) d\mu(x) = 0
```
[08:37] Questa uguaglianza deve valere per qualsiasi scelta dei parametri $\alpha_j$, $\theta_j$, $w_j$ e per qualsiasi numero di neuroni $N$.
[08:42] Si sceglie un caso molto semplice: una rete con un solo neurone ($N=1$) e un peso di output unitario ($\alpha_1=1$). L'espressione si semplifica notevolmente:
```math
\int_{I_n} \sigma(w^T x + \theta) \, d\mu(x) = 0
```
[08:50] A questo punto, la dimostrazione è quasi conclusa.
[08:53] L'ipotesi fondamentale del teorema è che la funzione di attivazione $\sigma$ sia *discriminatoria*. Per definizione, una funzione discriminatoria è tale che l'integrale $\int \sigma(\dots) \, d\mu$ si annulla se e solo se la misura $\mu$ è la misura nulla.
[09:01] Tuttavia, partendo dall'assunzione che $U$ non fosse denso, si è giunti alla conclusione che l'integrale $\int \sigma(w^T x + \theta) \, d\mu(x)$ è uguale a zero per una misura $\mu$ che era stata stabilita essere *non nulla*.
[09:08] Questa è una **contraddizione**. L'ipotesi che $\sigma$ sia discriminatoria è incompatibile con la conclusione a cui si è arrivati.
[09:16] Pertanto, l'assunzione iniziale che $U$ non sia denso in $C(I_n)$ deve essere falsa. Ne consegue che $U$ è denso in $C(I_n)$.
[09:20] Si è così dimostrato che l'insieme delle funzioni $g(x)$, rappresentanti reti neurali a singolo strato, è denso nello spazio delle funzioni continue. Questo significa che, scegliendo opportunamente il numero di neuroni $N$ e i parametri, è possibile approssimare con precisione arbitraria qualsiasi funzione continua.
# Verifica della Proprietà Discriminatoria
## L'Importanza di Verificare le Funzioni di Attivazione
[09:33] La dimostrazione si basa sull'ipotesi cruciale che la funzione di attivazione $\sigma$ sia discriminatoria.
[09:39] A questo punto, il problema si sposta sulla verifica di questa proprietà per le funzioni di attivazione comunemente utilizzate. Se si riesce a dimostrare che queste funzioni sono effettivamente discriminatorie, il risultato del teorema di approssimazione universale diventa pienamente applicabile.
## Dimostrazione per le Funzioni Sigmoidali
[09:50] Si inizia analizzando le funzioni sigmoidali, ovvero quelle funzioni il cui grafico ha la tipica forma a "S" e che tendono a 0 per $x \to -\infty$ e a 1 per $x \to +\infty$.
[09:55] Si vuole dimostrare che *qualsiasi funzione sigmoidale è discriminatoria*.
[09:59] Per avviare la dimostrazione, viene introdotta una notazione geometrica.
[10:03] Si considera lo spazio tridimensionale (3D). L'equazione $w^T x + \theta = 0$ rappresenta un piano.
[10:08] Dati un vettore $w$ e uno scalare $\theta$, l'equazione $w_1 x_1 + w_2 x_2 + \theta = 0$ (nel caso 2D per l'input $x$) o, più in generale, $a x + b y + c z + d = 0$ (nel caso 3D), definisce un iperpiano (un piano in 3D).
## Introduzione agli Iperpiani e Semispazi
[00:00] In questo contesto, le componenti $a, b, c$ rappresentano gli elementi di un vettore $w$, mentre $d$ corrisponde a un parametro $\theta$. L'equazione $P_{w\theta}$ definisce un piano nello spazio tridimensionale. A partire da questo piano, si definiscono due sottospazi. Il primo, indicato come $H^+_{w\theta}$, è l'insieme dei punti $X$ dello spazio tali per cui la quantità $w^T X + \theta$ è maggiore di zero.
[00:10] Questo sottospazio corrisponde a una delle due regioni delimitate dal piano. Analogamente, il sottospazio $H^-$ è definito come la regione in cui la stessa espressione, $w^T X + \theta$, risulta minore di zero. Successivamente, viene introdotto un risultato fondamentale riguardante una misura $\mu$.
[00:21] Se una misura $\mu$ si annulla su tutti gli iperpiani, ovvero per ogni scelta di $w$ e $\theta$ la misura del piano $P$ è zero, e si annulla anche su tutti i semispazi (sia $H^+$ che $H^-$), allora tale misura è identicamente nulla. In sintesi, se una misura si annulla su un piano, sul semispazio positivo $H^+$ e su quello negativo $H^-$, deve necessariamente essere la misura nulla.
## Dimostrazione della Proprietà Discriminatoria delle Funzioni Sigmoidali
[00:37] L'obiettivo è dimostrare che una funzione sigmoidale è *discriminatoria*.
- **Funzione Discriminatoria**: Una funzione $\sigma$ è detta discriminatoria se l'annullarsi dell'integrale $\int \sigma(w^T x + \theta) d\mu(x)$ per ogni $w$ e $\theta$ implica necessariamente che la misura $\mu$ sia la misura nulla.
[00:43] Si assume che, per una data misura $\mu$, l'integrale della funzione sigmoidale $\sigma$ esteso al dominio di interesse sia nullo:
```math
\int \sigma(w^T x + \theta) d\mu(x) = 0
```
Si vuole dimostrare che questa condizione implica $\mu = 0$. Poiché $\sigma$ è discriminatoria, se l'integrale è zero, non esistono altre possibilità se non che la misura $\mu$ sia nulla.
[00:56] Per procedere, si costruisce una funzione scalata, indicata come $\sigma_\lambda$. Invece di considerare semplicemente l'argomento $w^T x + \theta$, lo si moltiplica per un fattore $\lambda$ e si aggiunge un valore $t$:
```math
\sigma_\lambda(x) = \sigma(\lambda(w^T x + \theta) + t)
```
Si analizza quindi il comportamento di questa funzione $\sigma_\lambda$ al tendere di $\lambda$ all'infinito.
[01:08] Quando $\lambda \to \infty$, l'elemento cruciale da considerare è il segno dell'espressione $w^T x + \theta$.
[01:13] Se il punto $x$ appartiene al semispazio positivo $H^+$, per definizione si ha $w^T x + \theta > 0$. Di conseguenza, quando $\lambda$ tende a infinito, l'argomento della funzione $\sigma$ tende a $+\infty$. Poiché il limite di una funzione sigmoidale per l'argomento che tende a $+\infty$ è 1, si ottiene:
```math
\lim_{\lambda \to \infty} \sigma_\lambda(x) = 1 \quad \text{per } x \in H^+
```
[01:26] Analogamente, se il punto $x$ appartiene al semispazio negativo $H^-$, si ha $w^T x + \theta < 0$. In questo caso, l'argomento della funzione tende a $-\infty$, e il limite della funzione sigmoidale è 0:
```math
\lim_{\lambda \to \infty} \sigma_\lambda(x) = 0 \quad \text{per } x \in H^-
```
Questa funzione limite, che chiamiamo $\gamma(x)$, assume quindi valori distinti nei due semispazi.
[01:34] Infine, se il punto $x$ si trova esattamente sull'iperpiano $P$, la prima parte dell'argomento, $\lambda(w^T x + \theta)$, è zero. L'argomento si riduce quindi a $t$, che qui viene indicato con $\phi$. La funzione limite sull'iperpiano è:
```math
\lim_{\lambda \to \infty} \sigma_\lambda(x) = \sigma(\phi) \quad \text{per } x \in P
```
In sintesi, la funzione limite $\gamma(x)$ è una funzione a gradino che vale 1 nel semispazio positivo, 0 in quello negativo e $\sigma(\phi)$ sull'iperpiano.
## Applicazione del Teorema della Convergenza Dominata
[01:43] A questo punto si applica il **Teorema della Convergenza Dominata di Lebesgue**. Questo teorema, una delle motivazioni principali per l'introduzione dell'integrale di Lebesgue, stabilisce le condizioni sotto cui è possibile scambiare l'operatore di limite con l'operatore di integrale.
[02:01] Grazie a questo teorema, è possibile portare il limite per $\lambda \to \infty$ all'interno dell'integrale:
```math
\lim_{\lambda \to \infty} \int \sigma_\lambda(x) d\mu(x) = \int \left( \lim_{\lambda \to \infty} \sigma_\lambda(x) \right) d\mu(x) = \int \gamma(x) d\mu(x)
```
[02:09] L'integrale di $\gamma(x)$ può essere scomposto in base alle regioni dello spazio:
- Per $x \in H^+$, $\gamma(x) = 1$. Il contributo all'integrale è $\int_{H^+} 1 \cdot d\mu(x)$, che per definizione è la misura del semispazio positivo, $\mu(H^+)$.
- Per $x \in H^-$, $\gamma(x) = 0$. Il contributo è nullo.
- Per $x \in P$, $\gamma(x) = \sigma(\phi)$, che è una costante. Il contributo è $\sigma(\phi) \int_P d\mu(x)$, ovvero $\sigma(\phi) \cdot \mu(P)$.
[02:21] Sommando i contributi, si ottiene l'espressione finale. Poiché l'integrale di partenza era nullo per ipotesi, anche il suo limite lo è. Si arriva quindi alla relazione:
```math
\mu(H^+) + \sigma(\phi) \cdot \mu(P) = 0
```
Questa equazione è valida per qualsiasi valore di $\phi$.
[02:32] Ora si può analizzare cosa accade al variare di $\phi$. Si considerano i limiti per $\phi \to +\infty$ e $\phi \to -\infty$.
[02:37] Se $\phi \to +\infty$, la funzione sigmoidale $\sigma(\phi)$ tende a 1. Sostituendo questo valore nella relazione precedente, si ottiene:
```math
\mu(H^+) + \mu(P) = 0
```
[02:45] Se $\phi \to -\infty$, la funzione sigmoidale $\sigma(\phi)$ tende a 0. La relazione diventa:
```math
\mu(H^+) = 0
```
[02:51] Poiché le misure sono quantità non negative, dall'equazione $\mu(H^+) + \mu(P) = 0$ e sapendo che $\mu(H^+) = 0$, si deduce immediatamente che anche la misura dell'iperpiano $\mu(P)$ deve essere zero.
[02:57] Ripetendo lo stesso ragionamento per simmetria (ad esempio, considerando $-\sigma$ invece di $\sigma$), si può dimostrare che anche la misura del semispazio negativo, $\mu(H^-)$, è nulla.
[03:02] Si è quindi dimostrato che la misura $\mu$ si annulla sia sull'iperpiano $P$ che sui due semispazi $H^+$ e $H^-$.
[03:07] Richiamando il lemma introdotto all'inizio, se una misura si annulla su tutti gli iperpiani e su tutti i semispazi, essa deve essere la misura identicamente nulla. Pertanto, si conclude che $\mu = 0$.
[03:13] Questo completa la dimostrazione: partendo dall'ipotesi che l'integrale della funzione sigmoidale fosse nullo, si è dimostrato che la misura $\mu$ deve essere zero. Ciò conferma che la funzione sigmoidale è discriminatoria.
## Conseguenze per il Teorema di Approssimazione Universale
[03:20] La dimostrazione che le funzioni sigmoidali sono discriminatorie ha una conseguenza diretta sul **Teorema di Approssimazione Universale**. Se si considera un insieme di funzioni $g(x)$ definite come combinazioni lineari di funzioni sigmoidali $\sigma$ con coefficienti appropriati, questo insieme risulta denso nello spazio delle funzioni continue sull'ipercubo, $C(I^n)$.
- **Insieme Denso**: Un insieme di funzioni $G$ è denso in uno spazio di funzioni $F$ se, per ogni funzione $f \in F$ e per ogni livello di precisione $\epsilon > 0$, esiste una funzione $g \in G$ tale che la distanza tra $f$ e $g$ sia minore di $\epsilon$.
[03:28] Avendo provato che le funzioni sigmoidali sono discriminatorie, il teorema di approssimazione universale è verificato per questo caso specifico.
[03:35] In conclusione, per una rete neurale *shallow* (con un solo strato nascosto) che utilizza funzioni di attivazione discriminatorie, come le funzioni sigmoidali, la proprietà di approssimazione universale è valida.
# Analisi di Altre Funzioni di Attivazione e Limiti del Teorema
## Analisi della Funzione di Attivazione ReLU
[03:42] Un'osservazione importante riguarda l'uso pratico delle reti neurali. Spesso, al posto delle funzioni sigmoidali, si utilizzano le funzioni **ReLU (Rectified Linear Unit)**. La funzione ReLU non è sigmoidale, poiché il suo limite per $x \to +\infty$ è $+\infty$, non 1.
[03:57] La domanda che sorge è se anche la funzione ReLU sia discriminatoria.
### Caso Unidimensionale (1D)
[04:03] È possibile dimostrare facilmente che la ReLU è discriminatoria nel caso unidimensionale (1D). Successivamente, verrà fornito un accenno su come estendere la dimostrazione al caso n-dimensionale.
[04:11] Nel caso 1D, le variabili $y$ (corrispondente a $w$) e $\theta$ sono scalari reali. Si parte, come prima, dall'ipotesi che l'integrale della funzione ReLU sia nullo per una certa misura $\mu$:
```math
\int \text{ReLU}(yx + \theta) d\mu(x) = 0
```
L'obiettivo è dimostrare che questa condizione implica $\mu = 0$.
[04:21] L'idea di base è la stessa utilizzata in precedenza: partire dall'integrale nullo e provare che la misura deve essere nulla. Il trucco consiste nel costruire una funzione sigmoidale combinando due funzioni ReLU, un approccio già visto graficamente.
[04:34] In una lezione precedente, si era mostrato come, partendo dalla funzione ReLU, fosse possibile costruire una funzione simile a una "gobba" (o "bump function"), che è un tipo di funzione sigmoidale.
[04:50] L'idea è quindi quella di costruire una funzione sigmoidale $f(x)$ come combinazione lineare di due funzioni ReLU.
[04:56] Si considera una funzione a "gobba" con pendenza unitaria. Per $x < 0$ la funzione è zero, per $x$ tra 0 e 1 la funzione è $x$, e per $x > 1$ torna a zero.
[05:05] Come visto in precedenza, è possibile ottenere una funzione $g(x)$ di questo tipo come differenza di due funzioni ReLU con valori opportuni di $\theta_1$ e $\theta_2$. Il parametro chiave che si può manipolare è il rapporto tra il bias e il peso, $b/w$ (qui $\theta/y$).
[05:20] Scegliendo opportunamente $\theta_1$ e $\theta_2$, si può costruire una funzione $f(x)$ che è effettivamente una funzione sigmoidale. Si deve prestare attenzione al caso in cui $y=0$, che è un caso banale.
[05:31] Si considera quindi l'integrale della funzione $f(x)$ rispetto alla misura $\mu$. Poiché $f(x)$ è costruita come differenza di due ReLU, si ha:
```math
\int f(x) d\mu(x) = \int (\text{ReLU}_1 - \text{ReLU}_2) d\mu(x)
```
[05:38] L'integrale della differenza può essere scomposto nella differenza degli integrali. Per l'ipotesi iniziale, ciascuno di questi integrali è nullo.
```math
\int f(x) d\mu(x) = \int \text{ReLU}_1 d\mu(x) - \int \text{ReLU}_2 d\mu(x) = 0 - 0 = 0
```
[05:44] Quindi, l'integrale della funzione $f(x)$ è zero. Ma $f(x)$ è una funzione sigmoidale. Poiché le funzioni sigmoidali sono discriminatorie, l'unica possibilità è che la misura $\mu$ sia nulla.
[05:55] Si è così dimostrato che, nel caso 1D, anche la funzione ReLU è discriminatoria. Il trucco è stato ricondurre il problema a quello già risolto, costruendo una funzione sigmoidale a partire da due ReLU.
### Estensione al Caso Multidimensionale
[06:06] La generalizzazione di questo risultato al caso multidimensionale è più complessa, ma segue un principio simile. Se si dimostra che la funzione ReLU è discriminatoria in 1D, è possibile provare che l'insieme delle funzioni costruite con attivazioni ReLU è denso nello spazio $C(I^n)$.
[06:16] In sostanza, la funzione ReLU risulta essere discriminatoria anche per dimensioni $m > 1$.
## Limiti del Teorema di Approssimazione Universale
[06:21] È importante fare alcune considerazioni sui limiti pratici del teorema di approssimazione universale.
[06:26] 1. **Teorema di Esistenza non Costruttivo**: Il teorema garantisce che, data una funzione $f$, esiste una rete neurale in grado di approssimarla con precisione arbitraria. Tuttavia, non fornisce alcuna ricetta o metodo per costruire tale rete. È un teorema di esistenza, non un algoritmo costruttivo.
[06:36] 2. **Complessità nel Caso Shallow**: Il risultato è stato ottenuto per reti *shallow* (un solo strato nascosto). Come si vedrà, in alcune situazioni, la costruzione effettiva di una rete shallow adeguata può essere computazionalmente proibitiva.
[06:50] Finora si è stabilito che, data una funzione $f$, esiste una funzione $\hat{f}$ (corrispondente alla $g(x)$ vista prima) nello spazio delle reti neurali che la approssima con accuratezza arbitraria. L'esistenza è garantita, almeno per il caso shallow. Se esiste per il caso shallow, esiste anche per il caso *deep* (profondo), sebbene la dimostrazione sia più complessa.
[07:08] Ora si affrontano due questioni legate alla complessità della rete:
- Quanti neuroni sono necessari in una rete shallow per raggiungere una data precisione?
- Qual è la situazione nel caso di reti deep?
# Complessità delle Reti Neurali: Shallow vs. Deep
## La Maledizione della Dimensionalità nelle Reti Shallow
[07:20] Si introduce il concetto di **maledizione della dimensionalità** (*curse of dimensionality*) nel contesto delle reti neurali. Questo fenomeno descrive come, all'aumentare della dimensione del vettore di input (il numero di feature), il numero di neuroni richiesto per ottenere una certa precisione in una rete shallow possa diventare proibitivo.
[07:35] Una rete shallow tipica ha $N$ neuroni nello strato nascosto. È stato dimostrato che, se si utilizza una funzione di attivazione infinitamente differenziabile ma non polinomiale (come una sigmoide o una tangente iperbolica), l'errore di approssimazione si comporta in un modo specifico.
[07:47] Se si vuole approssimare una funzione $f$ con una certa regolarità (cioè differenziabile fino all'ordine $r$), l'errore di approssimazione $\epsilon$ è legato al numero di neuroni $N$ dalla seguente relazione:
```math
\epsilon \propto N^{-r/d}
```
dove:
- $N$ è il numero di neuroni nello strato nascosto.
- $r$ è l'ordine di regolarità (differenziabilità) della funzione da approssimare.
- $d$ è la dimensione del vettore di input (il numero di feature).
[08:03] Invertendo questa relazione per esprimere $N$ in funzione dell'errore desiderato $\epsilon$, si ottiene:
```math
N \propto \epsilon^{-d/r}
```
[08:12] Da questa formula è evidente che se la dimensione $d$ del vettore di feature è grande, il numero di neuroni $N$ richiesto cresce esponenzialmente al diminuire della tolleranza $\epsilon$.
[08:20] Questo significa che le reti shallow sono soggette alla maledizione della dimensionalità. All'aumentare della dimensione dell'input, il numero di neuroni necessario per mantenere una data precisione diventa esponenzialmente grande, rendendo l'approccio impraticabile.
## Vantaggi delle Reti Deep: l'Ipotesi Composizionale
[08:30] La situazione è diversa per le reti *deep*. Per analizzare questo caso, si introduce un'ipotesi fondamentale: si assume che la funzione da approssimare sia **composizionale**, ovvero che abbia una struttura gerarchica.
[08:37] Una funzione composizionale può essere scomposta in una gerarchia di funzioni più semplici. Ad esempio, una funzione con 8 variabili di input ($x_1, \dots, x_8$) può essere scritta come:
- Al livello più basso, funzioni $h_{1,j}$ operano su coppie di input: $h_{1,1}(x_1, x_2)$, $h_{1,2}(x_3, x_4)$, ecc.
- Al livello successivo, funzioni $h_{2,k}$ operano sugli output del livello precedente: $h_{2,1}(h_{1,1}, h_{1,2})$, ecc.
- Questo processo continua fino al livello più alto, che produce l'output finale.
[09:05] Si assume quindi che la funzione target possa essere rappresentata in questa forma gerarchica.
[09:11] Se si vuole approssimare una funzione con questa struttura, utilizzando una rete deep e una funzione di attivazione infinitamente differenziabile e non polinomiale, il numero di neuroni richiesto cambia drasticamente.
[09:18] Per una rete deep, il numero di neuroni $N$ necessario per raggiungere una tolleranza $\epsilon$ è dato da:
```math
N \propto (d-1) \cdot \epsilon^{-\tilde{d}/r}
```
dove:
- $d$ è la dimensione totale dell'input.
- $r$ è la regolarità della funzione.
- $\tilde{d}$ è il numero di variabili richieste da ciascuna funzione nella rappresentazione gerarchica.
[09:27] Nell'esempio precedente, poiché ogni funzione $h$ prende due argomenti, $\tilde{d}=2$. Questa è chiamata rappresentazione composizionale binaria.
[09:35] Confrontando le due formule, si nota che nel caso deep l'esponente di $\epsilon$ è $-\tilde{d}/r$, dove $\tilde{d}$ è tipicamente un numero piccolo (es. 2), indipendente dalla dimensione totale $d$ dell'input. Nel caso shallow, l'esponente era $-d/r$. Questo significa che le reti deep possono evitare la crescita esponenziale del numero di neuroni rispetto alla dimensione dell'input, a condizione che la funzione target abbia una struttura composizionale.
## Approssimazione di Funzioni con Reti Binarie
[00:00] Nel caso specifico in analisi, il parametro $\tilde{D}$ assume il valore 2. In precedenza, si è accennato al fatto che l'adozione di funzioni composizionali, e in particolare di funzioni rappresentabili tramite una rappresentazione binaria, potrebbe apparire come un vincolo restrittivo.
[00:09] Tuttavia, è possibile dimostrare matematicamente che qualsiasi funzione $f$ può essere approssimata con un grado di accuratezza arbitrario utilizzando una rappresentazione binaria. Di conseguenza, considerare questa classe di funzioni non costituisce una limitazione significativa.
## Estensione dei Teoremi a Funzioni Non Lisce (ReLU)
[00:19] Un'altra assunzione fatta in entrambi i teoremi discussi è che la funzione di attivazione $\sigma$ sia infinitamente differenziabile. Questo solleva una questione riguardo alla funzione ReLU (Rectified Linear Unit), la quale non è differenziabile nell'origine.
[00:26] In realtà, esistono risultati teorici simili anche per funzioni non lisce, come la ReLU. L'idea fondamentale per estendere la teoria consiste nel considerare la funzione ReLU, che ha un andamento caratteristico.
[00:32] È possibile selezionare un intervallo arbitrariamente piccolo attorno all'origine e, all'interno di questo intervallo, creare una connessione liscia (o "regolarizzata") tra i due rami della funzione. Applicando questa tecnica di regolarizzazione locale, si può dimostrare che anche per la funzione ReLU valgono risultati analoghi a quelli visti per le funzioni lisce.
## Complessità delle Reti Shallow e Deep con Funzioni Lipschitziane
[00:43] In particolare, per il caso di una rete *shallow* (a un solo strato nascosto) che approssima funzioni L-Lipschitziane, il numero di neuroni richiesti è dell'ordine di:
```math
\left(\frac{\epsilon}{L}\right)^{-D}
```
dove:
- $\epsilon$ è l'errore di approssimazione desiderato.
- $L$ è la costante di Lipschitz della funzione.
- $D$ è la dimensione del vettore di input.
Per il caso di una rete *deep* (profonda), la complessità è invece:
```math
(D-1) \left(\frac{\epsilon}{L}\right)^{-2}
```
[00:52] L'esponente $-2$ in questa formula deriva dal fatto che stiamo considerando il caso specifico di una rappresentazione binaria. In un contesto più generale, in cui si utilizza una rappresentazione gerarchica dove ogni funzione componente dipende da $\tilde{D}$ argomenti, il numero 2 deve essere sostituito con il valore specifico di $\tilde{D}$.
[01:03] L'aspetto cruciale da osservare, valido sia per il caso di funzioni di attivazione continue e lisce sia per quelle non lisce come la ReLU, è come la dimensione dell'input $D$ influisce sulla complessità.
- Nelle reti **shallow**, la dimensione $D$ appare come **esponente**.
- Nelle reti **deep**, la dimensione $D$ appare come **costante moltiplicativa**.
## Esempio Pratico: Confronto del Numero di Neuroni
[01:14] Per illustrare la differenza in modo concreto, si considera un esempio numerico. Supponiamo di avere:
- Un vettore di feature con $D = 1000$ dimensioni.
- Una regolarità della funzione pari a 10.
- Funzioni binarie, quindi $\tilde{D} = 2$.
- Un errore di approssimazione desiderato $\epsilon$ dell'ordine di $10^{-2}$.
[01:23] Calcoliamo il numero di neuroni necessari nei due scenari:
- **Rete Shallow**: Il numero di neuroni richiesto è dell'ordine di $10^{200}$. Questa è una quantità di neuroni computazionalmente irrealizzabile o estremamente difficile da gestire.
- **Rete Deep**: Il numero di neuroni richiesto è di circa 2500.
[01:31] La differenza nel numero di neuroni è enorme. Questo spiega perché, in quasi tutte le applicazioni pratiche, si utilizzano reti neurali profonde (deep) anziché reti a singolo strato (shallow), relegate ormai a semplici esempi didattici per mostrare i meccanismi di base.
## Composizione di Funzioni e Complessità Esponenziale
[01:40] Tornando al concetto di composizione, si è visto in precedenza come, partendo dalla funzione ReLU, sia possibile costruire una funzione "a cappello" (*hat function*).
[01:46] Nello specifico, una funzione a cappello definita sull'intervallo $[0, 1]$ può essere descritta dall'espressione:
```math
g(x) = \max(0, \min(2x, 2(1-x)))
```
Questa funzione è composta da un ramo $2x$ e un ramo $2(1-x)$, formando un singolo picco.
[01:54] La rappresentazione gerarchica delle reti profonde è essenzialmente una composizione di funzioni. Si analizza cosa accade componendo la funzione $g(x)$ con se stessa, ovvero calcolando $g(g(x))$.
- La funzione originale $g(x)$ presenta un singolo picco.
- La funzione composta $g(g(x))$ presenta due picchi.
[02:03] Generalizzando, si può osservare un andamento ricorsivo:
- **Profondità 1** (una sola applicazione di $g$): $2^0 = 1$ picco.
- **Profondità 2** (due applicazioni, $g \circ g$): $2^1 = 2$ picchi.
- **Profondità 4**: si ottengono $2^3 = 8$ picchi.
In generale, una composizione di profondità $k$ genera una funzione con un numero esponenziale di oscillazioni (picchi).
[02:13] Esiste un risultato teorico che stabilisce la complessità necessaria per approssimare una tale funzione con $2^{k-1}$ picchi:
- Una **rete shallow** richiede un numero di neuroni dell'ordine di $2^k$, ovvero circa un neurone per ogni picco da rappresentare.
- Una **rete deep** richiede solamente $k$ neuroni.
[02:21] Questo esempio dimostra nuovamente il vantaggio delle reti profonde. Per le reti shallow, un parametro che descrive la complessità della funzione (in questo caso, il numero di picchi) appare all'**esponente** nel calcolo dei neuroni necessari. Per le reti deep, lo stesso parametro agisce come una **costante moltiplicativa** (o, in questo caso specifico, è direttamente proporzionale al numero di neuroni).
[02:30] Anche da questa analisi emerge chiaramente che le reti neurali profonde rappresentano la soluzione più efficiente.
## Riepilogo Comparativo: Reti Shallow vs. Deep
[02:33] Per riassumere le differenze chiave tra i due tipi di architetture:
**Rete Shallow (a singolo strato)**
[02:41] - **Numero di Neuroni**: Richiede un numero molto elevato di neuroni.
[02:46] - **Computazione**: L'elaborazione avviene in modo parallelo. Poiché lo strato è completamente connesso, tutti i neuroni possono processare gli stessi parametri di input simultaneamente.
[02:52] - **Struttura Matematica**: Realizza essenzialmente una combinazione lineare di funzioni di attivazione.
[02:58] - **Funzioni ad Alta Frequenza**: Per rappresentare funzioni con $N$ oscillazioni, il numero di neuroni è dell'ordine di $O(N)$.
[03:09] - **Simmetrie**: Fatica a rappresentare le simmetrie presenti nella funzione target. Per addestrare una rete shallow a riconoscere una funzione non simmetrica è necessario un numero molto elevato di neuroni.
[03:15] - **Ottimizzazione**: Se il numero di strati è fissato (a uno) e si utilizza una funzione di costo come l'errore quadratico medio (MSE), il problema di ottimizzazione è **convesso**.
**Rete Deep (profonda)**
[02:44] - **Numero di Neuroni**: Richiede un numero significativamente inferiore di neuroni.
[02:49] - **Computazione**: La computazione è seriale tra gli strati, ma parallela all'interno di ogni singolo strato.
[02:55] - **Struttura Matematica**: Realizza una composizione di funzioni.
[03:06] - **Funzioni ad Alta Frequenza**: Per rappresentare funzioni con $N$ oscillazioni, il numero di neuroni è dell'ordine di $O(\log N)$.
[03:15] - **Simmetrie**: È molto più efficiente nel catturare e rappresentare le simmetrie della funzione.
[03:15] - **Ottimizzazione**: Il problema di ottimizzazione è **non-convesso**. Il paesaggio della funzione di costo è molto più complesso, caratterizzato tipicamente da un minimo globale e numerosi minimi locali. Questo rende più difficile per gli algoritmi di ottimizzazione trovare una soluzione ottimale.
## Conclusioni sulla Complessità
[03:25] In sintesi, le reti shallow presentano una complessità **esponenziale** rispetto alla dimensione del vettore di feature in input, mentre le reti deep hanno una complessità **moltiplicativa**.
[03:33] Nonostante questa differenza in termini di efficienza, i teoremi di esistenza garantiscono che, in linea di principio, entrambe le architetture sono in grado di approssimare le funzioni desiderate.
[03:36] A questo punto, la trattazione teorica viene sospesa per una pausa, prima di procedere con la discussione relativa al progetto.
# Discussione Organizzativa e Logistica
## Collaborazioni e Appuntamenti
[03:43] Si passa a discutere di questioni organizzative relative a collaborazioni e appuntamenti.
[03:46] Vengono menzionate collaborazioni passate con Eni e RSE.
[03:49] Viene fissato un appuntamento per mercoledì successivo alle 15:30 presso l'ufficio del docente.
[03:54] L'incontro ha lo scopo di introdurre lo studente al Dott. Anda, che lavora sul problema di interesse, per focalizzare il lavoro. L'ufficio si trova al Dipartimento di Matematica, al settimo piano dell'edificio "Nave".
[04:02] Si concorda per le 15:15.
[04:05] Verrà inviata un'email con il numero di cellulare per facilitare l'incontro, nel caso ci si trovi in una sala riunioni.
[04:11] In alternativa al cellulare, è possibile contattare il numero dell'ufficio: 022-399-4517.
## Gestione degli Appelli d'Esame
[04:18] Si chiariscono dettagli su un corso e un appello d'esame.
[04:21] Si cerca di capire se ci sono sovrapposizioni con altri esami per il secondo appello.
[04:23] Si chiede se qualcuno dei corsi di Ingegneria Informatica (IT) o High Performance Computing (HPC) avesse segnalato una sovrapposizione di esami.
[04:27] Si valuta la possibilità di spostare un esame, ma sembra difficile a causa di un effetto a catena che creerebbe nuove sovrapposizioni.
[04:30] L'esame in questione si sovrappone con "Algoritmi e Parallel Computing" e "Ricerca Operativa".
[04:34] Si chiede se la sovrapposizione si verifica anche nel secondo appello, fissato per il 6 febbraio.
[04:41] Si conferma che nel secondo appello non ci sono altre sovrapposizioni oltre a quelle già note.
[04:47] Si discute della fattibilità di sostenere due esami nello stesso giorno, considerando gli orari. Un esame ha una durata di due ore e mezza.
[04:53] Si considera l'ipotesi di posticipare l'orario di inizio dell'esame alle 16:00, ma questo comporterebbe la fine della prova intorno alle 19:00.
[04:57] Spostare la data dell'esame è molto difficile. Un altro ostacolo è la disponibilità delle aule, che sono state richieste in anticipo. Si farà comunque un tentativo e verranno forniti aggiornamenti.
[05:01] Si chiarisce che la sovrapposizione di orario è totale, rendendo impossibile la partecipazione a entrambi gli esami.
[05:03] Uno studente ha già segnalato il problema alla Presidenza, su indicazione della segreteria didattica. Si attende una risposta.
[05:06] Si discute delle tempistiche per la presentazione dei progetti. Solitamente, questa avviene circa 10-12 giorni dopo la prova scritta, dopo aver sostenuto anche la prova orale.
[05:09] Questo vale sia per la prima che per la seconda sessione di gennaio.
[05:12] È possibile organizzare anche una sessione informale aggiuntiva, ad esempio ad aprile. In questo caso, la registrazione del voto non potrà essere immediata, poiché non ci sono sessioni d'esame aperte in quel periodo. Il voto verrebbe quindi registrato ufficialmente nella sessione di giugno-luglio.
[05:15] Si attende una risposta dalla segreteria della presidenza riguardo alla sovrapposizione. Nel frattempo, si cercherà di contattare altre persone per trovare una soluzione.
[05:20] Si riprende il problema della sovrapposizione degli esami nella prima e seconda data.
[05:25] Si era discussa in precedenza la possibilità di creare due sessioni d'esame distinte: una per Ingegneria Matematica (che non ha problemi di sovrapposizione) e un'altra per HPC e Ingegneria Informatica.
[05:31] È fondamentale chiarire un punto: se si procederà in questo modo, i testi delle due prove d'esame saranno necessariamente diversi.
[05:36] Si sottolinea con fermezza che non verranno accettati reclami successivi riguardo a presunte differenze nel livello di difficoltà tra le due prove.
[05:43] La registrazione viene interrotta.
