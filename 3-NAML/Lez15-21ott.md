## Discesa del gradiente e motivazione generale
[00:00] L’argomento introdotto è distinto rispetto al filone principale finora considerato, ma è cruciale per i temi che seguiranno. Si collega ai problemi di minimizzazione di funzioni di costo. Nel caso dei minimi quadrati, si definisce una funzione di costo, la si minimizza e si ottiene un vettore dei pesi da impiegare per costruire un modello lineare, eventualmente reso non lineare tramite metriche kernel. Per “funzione di costo” si intende una funzione che misura l’errore del modello; il suo minimo corrisponde alla soluzione ottimale.
[00:30] In tale contesto, la soluzione del problema di minimizzazione è disponibile in forma chiusa, anche con decomposizione ai valori singolari, fornendo un’espressione esplicita per il vettore dei pesi $W$. Questo rende l’applicazione diretta. Nelle applicazioni pratiche, in particolare per reti neurali, una soluzione analitica o in forma chiusa per pesi e varianze che minimizzino la funzione di costo non è disponibile.
[01:00] Diventa necessario adottare algoritmi di minimizzazione numerica, che includono quasi sempre varianti della discesa del gradiente. L’idea generale è una procedura iterativa in cui il vettore incognito, indicato con $\theta$ (o $W$), si aggiorna effettuando passi nella direzione del gradiente della funzione di costo, con passo $\eta$ detto tasso di apprendimento. Numericamente, $\eta$ è la lunghezza del passo lungo la direzione scelta. Il gradiente è il vettore delle derivate parziali rispetto ai parametri e indica la direzione di massima variazione della funzione.
[01:30] Con un’interpretazione su una parabola, partendo da un punto iniziale, il gradiente indica la direzione in cui spostarsi per ridurre il valore della funzione, e si procede con un passo fissato. L’aggiornamento si ripete. Questa è la forma più semplice; esistono varianti che aggiungono termini per rendere la convergenza più robusta o più rapida. Dalla formula di aggiornamento è evidente che serve il gradiente della funzione di costo.
[02:00] Data una funzione di perdita $L$, qualunque sia la sua forma, bisogna calcolare il gradiente, cioè le derivate di $L$ rispetto ai parametri. Il gradiente è un metodo del primo ordine: la velocità di convergenza è tipicamente lineare. Per ottenere convergenza più rapida si può usare il metodo di Newton, che richiede la seconda derivata; nel caso multidimensionale si introduce la matrice Hessiana, cioè la matrice delle derivate seconde.
[02:30] In più dimensioni, si calcola il gradiente; quando la funzione ha più componenti, si considera il Jacobiano, la matrice delle derivate parziali delle componenti rispetto agli ingressi. Il punto essenziale è che per qualunque metodo di minimizzazione volto a trovare i parametri che minimizzano la funzione di costo, è necessario calcolare derivate: di primo ordine (gradiente) o di ordine superiore (Hessiana).
## Derivate in reti neurali e sensibilità dei parametri
[03:00] È essenziale capire come calcolare queste derivate. Nel caso di una rete neurale per classificazione di immagini (gatti e cani), si dispone di migliaia di immagini con etichette binarie (1 per gatto, 0 per cane). Si usa un sottoinsieme (ad esempio 800 immagini) per l’addestramento: si forniscono le immagini alla rete e si ottiene un valore previsto in uscita. La “rete neurale” è un modello composto da parametri (pesi) che trasformano gli ingressi in uscite tramite operazioni elementari.
[03:30] La funzione di costo misura la differenza tra uscita della rete ed etichetta reale. Dopo l’addestramento, si testa il modello su 200 immagini restanti. La funzione di costo dipende dalla discrepanza tra etichette vere e predette; bisogna calcolare la derivata rispetto a tutti i parametri della rete. In linea di principio si considera la derivata dell’uscita rispetto all’ingresso, ma operativamente interessa la sensibilità dell’uscita rispetto ai parametri del modello: come varia l’uscita al variare dei pesi.
[04:00] Per ogni parametro, si deve quantificare l’effetto di una sua variazione sull’uscita: una misura di sensibilità. Questo rende il calcolo delle derivate un passaggio decisivo. Le domande riguardano i metodi disponibili per calcolare tali derivate e la loro fattibilità.
## Quattro approcci al calcolo delle derivate
[04:30] Esistono quattro modalità: calcolo manuale, differenziazione numerica (differenze finite), differenziazione simbolica (Maple, MATLAB, Mathematica, pacchetti simbolici in Python), e differenziazione automatica. Tra queste, la differenziazione automatica è il metodo utilizzato nella pratica. La “differenziazione automatica” (AD) calcola valori di derivate numeriche esatte basandosi sulla composizione di operazioni elementari e sulle regole della catena.
[05:00] Il calcolo manuale consente, se la formula viene semplificata, espressioni eleganti; è utile per dimostrazioni di sensibilità e convergenza grazie a derivate esplicitate. Tuttavia, per espressioni complesse è soggetto a errori ed è dispendioso in tempo.
[05:30] La differenziazione numerica tramite differenze finite è semplice, funziona come scatola nera su qualunque funzione, richiede solo poche valutazioni della funzione e una divisione. La formula di differenza in avanti per la derivata prima di $f$ in $x_0$ è:
```math
f'(x_0) \approx \frac{f(x_0 + h) - f(x_0)}{h}.
```
Qui $h$ è l’incremento scelto. È un’approssimazione e presenta rischi numerici dovuti alla sottrazione tra quantità vicine.
[06:00] Per stimare $f'(x_0)$ con $h$ piccolo, si sommano quantità di ordini di grandezza diversi (ad esempio $x_0 \approx 1$ e $h \approx 10^{-4}$), causando problemi di rappresentazione in virgola mobile. Inoltre, la sottrazione tra valori molto vicini introduce cancellazioni numeriche. In generale, le formule numeriche presentano due errori: errore di troncamento (dovuto all’approssimazione) ed errore di arrotondamento (dovuto alla rappresentazione in virgola mobile).
[06:30] L’errore di arrotondamento tipicamente si comporta come $1/h$, indipendentemente dal metodo, mentre l’errore di troncamento dipende dalla formula adottata e dalla potenza di $h$. La formula in avanti ha errore di troncamento di primo ordine:
```math
f'(x_0) = \frac{f(x_0 + h) - f(x_0)}{h} + \mathcal{O}(h).
```
La formula centrata ha ordine secondo:
```math
f'(x_0) \approx \frac{f(x_0 + h) - f(x_0 - h)}{2h} + \mathcal{O}(h^2).
```
La “$\mathcal{O}(\cdot)$” indica il termine d’errore asintotico.
[07:00] La somma dei due errori produce un andamento con un valore ottimo di $h$ che minimizza l’errore totale. Non è vero, in generale, che riducendo indefinitamente $h$ si migliora la stima: l’errore di troncamento diminuisce con $h$, ma l’errore di arrotondamento cresce come $1/h$.
[07:30] Applicando le differenze finite a funzioni specifiche come il seno, si osservano tratti in cui la pendenza dell’errore indica l’ordine (più ripida per l’ordine 2 della centrata, meno per l’ordine 1 della avanzata). Per $h$ molto piccoli, gli effetti di arrotondamento e cancellazioni dominano, causando comportamento oscillante e stime inaffidabili.
[08:00] La differenziazione simbolica, tramite strumenti come Maple, Mathcad, Mathematica e toolbox simbolici, offre gli stessi vantaggi del calcolo manuale riducendo gli errori umani. Ha però svantaggi: costo computazionale elevato per espressioni complesse e “swelling” dell’espressione, cioè derivate estremamente lunghe che richiedono semplificazioni significative. Per casi semplici, le semplificazioni sono automatiche; per espressioni complesse serve intervento manuale.
[08:30] La differenziazione automatica (AD) si distingue nettamente. Nel calcolo manuale e simbolico, il risultato è un’espressione esplicita della derivata. Nel metodo numerico, si ottiene un valore della derivata in un punto. Nelle procedure di ottimizzazione basate su aggiornamenti come la discesa del gradiente, serve il valore del gradiente nel punto corrente dei parametri $\theta$, non necessariamente la sua espressione chiusa.
[09:00] L’AD è il motore operativo dei moderni framework di machine learning e deep learning: TensorFlow, PyTorch, scikit-learn e altre librerie integrano l’AD. In laboratorio si incontra JAX, una libreria per AD usata in contesti collegati a TensorFlow. Gli svantaggi dell’AD includono consumo di memoria elevato e implementazioni non banali; è comunque possibile implementare una semplice libreria AD per comprenderne gli aspetti progettuali.
## Costi computazionali della differenziazione numerica
[09:30] Oltre ai problemi numerici, la differenziazione numerica ha costi computazionali significativi. Nella “regione buona” di $h$, per una funzione $f:\mathbb{R}^n \to \mathbb{R}$, il calcolo del gradiente tramite differenze finite richiede una valutazione per ciascuna variabile d’ingresso. Il “gradiente” è il vettore delle derivate parziali; ogni componente richiede una valutazione differenziata.
[10:00] Ad esempio, un’immagine $1000 \times 1000$ ha $n=10^6$ ingressi: calcolare $10^6$ derivate dell’uscita rispetto a ciascun ingresso è impraticabile. Se si ha $f:\mathbb{R}^n \to \mathbb{R}^m$ (ad esempio $n=10^6$ e $m=100$ classi), il calcolo del Jacobiano richiede $n \times m$ valutazioni di funzione, con costo esorbitante. Il “Jacobiano” è la matrice $m \times n$ delle derivate parziali delle $m$ uscite rispetto ai $n$ ingressi.
[10:30] Questo rafforza la necessità di ricorrere all’AD per valutare efficientemente derivate e gradienti senza il costo quadratico o lineare in $n \times m$ tipico delle differenze finite.
## Cosa non è la differenziazione automatica
[11:00] L’AD non è differenziazione numerica: non usa formule approssimate con un parametro $h$ e non introduce gli errori di troncamento e arrotondamento tipici di quelle formule nella stima della derivata. L’AD fornisce il valore esatto della derivata (rispetto all’aritmetica macchina e alla composizione di operatori elementari) nel punto.
[11:30] L’AD non è differenziazione simbolica: non produce un’espressione esplicita della derivata come farebbe un sistema simbolico. Ad esempio, se in un sistema simbolico si definisce $x$ simbolico e $f=\sin(x)$, la derivata $df/dx$ è $ \cos(x)$, ossia un’espressione. L’AD, invece, fornisce valori numerici della derivata calcolati tramite la decomposizione della funzione in blocchi elementari.
[12:00] L’idea chiave dell’AD è scomporre qualunque funzione complessa nella composizione di funzioni elementari: somme, sottrazioni, moltiplicazioni, divisioni, funzioni trigonometriche, esponenziali e logaritmi. Rappresentando la funzione come composizione di tali elementi, si propagano derivate con regole locali esatte, ottenendo valori di derivate per l’uscita rispetto agli ingressi o ai parametri.
[12:30] In sintesi, l’AD opera sulla struttura computazionale della funzione valutata come grafo di operazioni elementari, permettendo il calcolo efficiente di gradienti e, quando necessario, di Jacobiani o prodotti Hessiano-vettore, senza ricorrere né a differenze finite né a manipolazioni simboliche.
## Introduzione al metodo di scomposizione tramite lista di Wengert
[00:00] L’obiettivo operativo è costruire una scomposizione di una funzione in passi elementari per calcolarne il valore in termini di funzioni elementari. Il primo concetto è la lista di Wengert, che descrive in sequenza le operazioni necessarie per ottenere il valore di una funzione come composizione di funzioni elementari. Data una funzione di due variabili, si crea una lista introducendo variabili ausiliarie $v$, iniziando dagli input: se si hanno due ingressi, si pone $v_{-1} = x_1$ e $v_0 = x_2$, con $x_1$ e $x_2$ i valori al punto di valutazione. In questa fase, l’obiettivo è la valutazione della funzione.
[00:45] La lista prosegue con variabili intermedie, cioè i passi indispensabili per determinare gli output, che sono le ultime variabili calcolate nella lista di Wengert. In termini formali, la lista ha tre blocchi: input (prime variabili $v$), variabili intermedie (trasformazioni elementari), e output (ultime variabili $v$ che forniscono il valore della funzione). Un esempio rende evidente il meccanismo.
## Esempio di funzione e costruzione della lista di Wengert
[01:20] Si consideri la funzione $f(x_1, x_2) = \ln(x_1) + x_1 x_2$. La lista di Wengert inizia con $v_{-1} = x_1$ e $v_0 = x_2$. Si possono considerare in seguito valori specifici, ad esempio $x_1 = 1$ e $x_2 = 2$, ma per ora si definisce la struttura generale.
[02:00] La parte intermedia introduce $v_1$, $v_2$, $v_3$:
- $v_1 = \ln(v_{-1})$, calcolo del logaritmo naturale di $x_1$.
- $v_2 = v_{-1} \cdot v_0$, prodotto $x_1 x_2$.
- $v_3 = v_1 + v_2$, somma dei due contributi.
L’output finale è $y = v_3$. Gli input sono $v_{-1}$ e $v_0$, le variabili intermedie $v_1$ e $v_2$, e l’output $v_3$. La scomposizione è utile perché di funzioni elementari si conoscono facilmente le derivate: ad esempio, la derivata di $\ln$ è $1/x$.
[02:35] La disponibilità di un “dizionario” di funzioni elementari e derivate, insieme alle regole di somma e prodotto, consente di calcolare derivate di funzioni composte applicando sistematicamente la regola della catena. La regola della catena stabilisce come derivare funzioni composte in base alle derivate dei singoli passi.
## Grafi computazionali e derivate sui lati
[03:20] Un’altra rappresentazione è il grafo computazionale, equivalente alla lista di Wengert, con tre parti: input, calcolo delle variabili intermedie, output. Qui $x_1$ entra nel logaritmo e nel prodotto con $x_2$; i risultati vengono sommati per ottenere l’output. I “nodi” rappresentano variabili o operazioni; gli “archi” rappresentano dipendenze.
[03:55] Nei nodi si collocano le variabili: un nodo per il prodotto $x_1 x_2$, uno per $\ln(x_1)$, uno per la somma, e gli archi esprimono le dipendenze. Una volta definito il grafo, si introducono le derivate sugli archi: a ogni arco si assegna la derivata parziale del nodo di arrivo rispetto al nodo di partenza, coerentemente con le definizioni. Questo consente la propagazione delle derivate.
[04:30] Le derivate sugli archi sono:
- Derivata di $v_2$ rispetto a $v_{-1}$: dato $v_2 = v_{-1} \cdot v_0$, si ha $\partial v_2 / \partial v_{-1} = v_0$.
- Derivata di $v_2$ rispetto a $v_0$: $\partial v_2 / \partial v_0 = v_{-1}$.
- Derivata di $v_3$ rispetto a $v_2$: con $v_3 = v_1 + v_2$, si ha $\partial v_3 / \partial v_2 = 1$, e analogamente $\partial v_3 / \partial v_1 = 1$.
[05:05] Queste derivate sui lati sono cruciali: per calcolare la derivata dell’output rispetto a un input, ad esempio $dy/dv_0$ con $y = v_3$, basta moltiplicare lungo il percorso rilevante le derivate:
```math
\frac{dv_3}{dv_0} = \frac{dv_3}{dv_2}\cdot \frac{dv_2}{dv_0}.
```
Questa è un’applicazione della regola della catena: la derivata di una composizione si ottiene moltiplicando le derivate parziali lungo il percorso.
[05:40] Se si vuole $\frac{dv_3}{dv_{-1}}$, esistono due percorsi dal nodo $v_{-1}$ a $v_3$, passando tramite $v_1$ e tramite $v_2$. In tal caso, si sommano i contributi dei percorsi:
```math
\frac{dv_3}{dv_{-1}} = \frac{dv_3}{dv_1}\cdot \frac{dv_1}{dv_{-1}} + \frac{dv_3}{dv_2}\cdot \frac{dv_2}{dv_{-1}}.
```
La somma riflette la presenza di rami multipli nel grafo.
## Valutazione avanti e introduzione del modo tangente (forward mode)
[06:20] Nella pratica si procede con una valutazione avanti (forward) partendo dai valori degli input. Assegnando $x_1 = 2$ e $x_2 = 5$, si calcolano intermedie e output $y = v_3$:
- $v_{-1} = 2$,
- $v_0 = 5$,
- $v_1 = \ln(2)$,
- $v_2 = 2 \cdot 5 = 10$,
- $v_3 = \ln(2) + 10$.
La “valutazione avanti” è la computazione dei valori delle variabili lungo il grafo in direzione input→output.
[06:55] Per calcolare la derivata di $f$ rispetto a $x_1$ si introduce un nuovo insieme di variabili, spesso denotate con il punto, chiamate variabili tangenti. Ogni $v_i^{\cdot}$ rappresenta la derivata di $v_i$ rispetto alla variabile d’interesse. Nel caso corrente:
- $v_{-1}^{\cdot} = \frac{d v_{-1}}{d x_1} = 1$, poiché $v_{-1} = x_1$,
- $v_0^{\cdot} = \frac{d v_0}{d x_1} = 0$, poiché $v_0 = x_2$ non dipende da $x_1$.
Le “variabili tangenti” accompagnano ogni nodo e codificano la sensibilità rispetto a un input specifico.
[07:30] In generale, se $f: \mathbb{R}^n \to \mathbb{R}$, si possono ottenere tutte le derivate parziali $\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}$. Nel modo tangente, l’inizializzazione si fa sul vettore degli input: si impostano tutti i $v^{\cdot}$ a zero tranne quello corrispondente all’input rispetto al quale si sta derivando, che si pone a uno. In tal modo, $v_i^{\cdot} = \frac{\partial v_i}{\partial x_j}$, con $j$ fissato.
[08:05] Applicando la regola della catena a ciascun passo elementare, si calcolano le variabili tangenti intermedie. Per $v_1 = \ln(v_{-1})$, la derivata rispetto a $x_1$ è:
```math
v_1^{\cdot} = \frac{\partial \ln(v_{-1})}{\partial v_{-1}} \cdot v_{-1}^{\cdot} = \frac{1}{v_{-1}} \cdot v_{-1}^{\cdot}.
```
Si usa la derivata del logaritmo $d(\ln u)/du = 1/u$ e la regola della catena.
[08:40] Per $v_2 = v_{-1} \cdot v_0$ (prodotto), la derivata rispetto a $x_1$ è:
```math
v_2^{\cdot} = v_{-1}^{\cdot}\cdot v_0 + v_{-1} \cdot v_0^{\cdot}.
```
Qui $v_0^{\cdot} = 0$, quindi il secondo termine è nullo e si ha $v_2^{\cdot} = v_{-1}^{\cdot} \cdot v_0$.
[09:10] Per $v_3 = v_1 + v_2$ (somma), la derivata è:
```math
v_3^{\cdot} = v_1^{\cdot} + v_2^{\cdot}.
```
Sostituendo i valori numerici ($v_{-1} = 2$, $v_0 = 5$, $v_{-1}^{\cdot} = 1$, $v_0^{\cdot} = 0$), si ottiene:
- $v_1^{\cdot} = \frac{1}{2} \cdot 1 = \frac{1}{2}$,
- $v_2^{\cdot} = 1 \cdot 5 = 5$,
- $v_3^{\cdot} = \frac{1}{2} + 5 = 5{,}5$.
La somma finale è la derivata della funzione rispetto all’input considerato.
[09:45] Il valore $v_3^{\cdot}$ coincide con $\frac{\partial f}{\partial x_1}$ calcolata nel punto $(2, 5)$, poiché $y = v_3$. Il procedimento: scomporre la funzione in passi, applicare la regola della catena e le derivate note delle funzioni elementari.
[10:15] Se si desidera anche $\frac{\partial f}{\partial x_2}$, occorre ripetere l’intera procedura con inizializzazione diversa: $v_{-1}^{\cdot} = 0$, $v_0^{\cdot} = 1$, e ricalcolare tutte le variabili tangenti. Se il numero di input è molto grande (ad esempio un milione) e l’output è scalare, il calcolo del gradiente nel modo tangente richiede di ripetere la procedura per ciascun input, risultando costoso.
## Modo inverso (reverse mode) e backpropagazione
[10:50] Per ridurre il costo quando si hanno molti input e pochi output, si introduce il modo inverso dell’autodifferenziazione, alla base dell’algoritmo di backpropagation nelle reti neurali. La “backpropagation” è la reverse mode applicata a un problema specifico di apprendimento di pesi tramite propagazione delle derivate dall’uscita agli ingressi.
[11:20] Il problema del modo tangente: per ottenere le derivate rispetto a ciascun input, bisogna cambiare l’inizializzazione e ripetere la procedura. L’idea del modo inverso è opposta: si parte dall’output e si risale verso gli input. Si definiscono variabili $v_i^{\bar{}}$ come
```math
v_i^{\bar{}} = \frac{dy}{dv_i},
```
con $y$ l’output, e si imposta $y^{\bar{}} = \frac{dy}{dy} = 1$. Con una sola passata dall’output verso gli input, si ottengono tutte le derivate dell’output rispetto agli input.
[11:55] In generale, si può avere una funzione con $n$ input e $m$ output. Nelle reti neurali tipicamente $n \gg m$. In tali casi, la reverse mode è fondamentale: consente di ottenere le sensibilità degli output rispetto a tutti gli input con un costo molto inferiore rispetto al ripetere $n$ volte un calcolo in avanti. Se invece $m \gg n$, il modo tangente può risultare più conveniente.
[12:30] In sintesi: il modo tangente è preferibile quando $n \ll m$, ma nei casi comuni in machine learning, con funzioni di perdita scalari ($m=1$) e molti input, il modo inverso è più adatto.
## Costi computazionali e considerazioni pratiche
[13:00] Per il calcolo della Jacobiana di una funzione con $n$ input e $m$ output, i costi sono valutati in relazione al costo della valutazione di $f$. Se la valutazione di $f$ richiede, ad esempio, 5 operazioni floating point, nel modo tangente occorre ripeterla $n$ volte, con costo complessivo proporzionale a $n \cdot 5$. Nel modo inverso, il costo è proporzionale a $m \cdot 5$, molto più piccolo quando $m \ll n$.
[13:35] Un aspetto critico del modo inverso è il fabbisogno di memoria: è necessario memorizzare molti valori intermedi del forward pass per calcolare le derivate nel backward pass. Se la funzione è complessa, la memoria richiesta per conservare le variabili intermedie può essere significativa.
## Esempio dettagliato del modo inverso sulla funzione data
[14:05] Si applica la reverse mode alla funzione $f(x_1, x_2) = \ln(x_1) + x_1 x_2$. Si esegue prima un passo forward per calcolare $v_{-1}$, $v_0$, $v_1$, $v_2$, $v_3$:
- $v_{-1} = x_1$,
- $v_0 = x_2$,
- $v_1 = \ln(v_{-1})$,
- $v_2 = v_{-1} \cdot v_0$,
- $v_3 = v_1 + v_2 = y$.
Questa fase prepara i valori necessari per la propagazione all’indietro.
[14:40] Si inizializza il backward con $v_3^{\bar{}} = y^{\bar{}} = 1$. Dato $v_3 = v_1 + v_2$, si propagano le derivate a $v_1$ e $v_2$:
```math
v_1^{\bar{}} = v_3^{\bar{}} \cdot \frac{\partial v_3}{\partial v_1} = 1 \cdot 1 = 1,\qquad
v_2^{\bar{}} = v_3^{\bar{}} \cdot \frac{\partial v_3}{\partial v_2} = 1 \cdot 1 = 1.
```
Si assegnano a ciascun arco le derivate corrispondenti e si accumulano i contributi.
[15:15] Si propaga poi verso $v_{-1}$ e $v_0$ attraverso $v_2$. Poiché $v_2 = v_{-1} \cdot v_0$, le derivate parziali sono:
```math
\frac{\partial v_2}{\partial v_{-1}} = v_0,\quad \frac{\partial v_2}{\partial v_0} = v_{-1}.
```
Ne consegue:
```math
v_{-1}^{\bar{}} \text{ (contributo da } v_2) = v_2^{\bar{}} \cdot \frac{\partial v_2}{\partial v_{-1}} = 1 \cdot v_0,\qquad
v_0^{\bar{}} = v_2^{\bar{}} \cdot \frac{\partial v_2}{\partial v_0} = 1 \cdot v_{-1}.
```
[15:50] Si aggiunge il contributo da $v_1$ verso $v_{-1}$. Poiché $v_1 = \ln(v_{-1})$, si ha:
```math
\frac{\partial v_1}{\partial v_{-1}} = \frac{1}{v_{-1}},
```
quindi:
```math
v_{-1}^{\bar{}} \text{ (contributo da } v_1) = v_1^{\bar{}} \cdot \frac{\partial v_1}{\partial v_{-1}} = 1 \cdot \frac{1}{v_{-1}}.
```
I contributi si sommano su $v_{-1}^{\bar{}}$ perché ci sono due percorsi (tramite $v_1$ e $v_2$) che incidono su $v_{-1}$.
[16:25] Usando i valori $x_1 = 2$ e $x_2 = 5$:
- $v_0 = 5$, contributo da $v_2$ a $v_{-1}^{\bar{}}$ è $5$,
- $v_{-1} = 2$, contributo da $v_1$ a $v_{-1}^{\bar{}}$ è $1/2$,
- $v_0^{\bar{}} = v_{-1} = 2$.
[16:55] Pertanto:
```math
\frac{\partial f}{\partial x_1} = v_{-1}^{\bar{}} = 5 + \frac{1}{2} = 5{,}5,\qquad
\frac{\partial f}{\partial x_2} = v_0^{\bar{}} = 2.
```
Con un solo passaggio all’indietro si ottengono entrambe le derivate rispetto agli input.
[17:25] La stessa procedura si visualizza sul grafo computazionale impostando $\frac{dv_3}{dv_3} = 1$ e propagando all’indietro lungo gli archi: da $v_3$ a $v_2$ e $v_1$, poi da $v_2$ a $v_{-1}$ e $v_0$, e da $v_1$ a $v_{-1}$, accumulando i contributi.
## Gradiente, Jacobiana e prodotti con vettori: approccio matrix-free
[17:55] In molte situazioni è richiesto il gradiente di $f$ o la Jacobiana $J$ quando la funzione ha più input e più output. Talvolta interessa il prodotto scalare tra $\nabla f$ e un vettore, oppure il prodotto tra la Jacobiana $J$ e un vettore. Queste operazioni emergono in metodi iterativi per risolvere sistemi lineari, dove si vuole l’effetto di un operatore su un vettore.
[18:30] In tali casi si sfrutta l’approccio matrix-free: non si costruisce esplicitamente la matrice $J$, ma si calcola il prodotto $Jv$ per un vettore $v$. I metodi matrix-free non richiedono la materializzazione della matrice; puntano al risultato del prodotto. Si usa la struttura della lista di Wengert o del grafo computazionale insieme alle regole di differenziazione (forward o reverse) per ottenere l’effetto dell’operatore sul vettore senza costruire la matrice.
## Modalità forward: derivate direzionali e prodotti con vettori
[00:00] Nella modalità forward, si imposta $\dot{x}$. Nel contesto considerato, $\dot{x}$ è un vettore come $(1, 0)$ o $(0, 1)$. Il vettore $(1, 0)$ è la ricetta per calcolare la derivata di $f$ rispetto a $x_1$, mentre $(0, 1)$ è la ricetta per la derivata rispetto a $x_2$. La “derivata direzionale” lungo una direzione è la variazione di $f$ lungo il vettore impostato.
[00:30] Se si vuole il prodotto scalare tra il gradiente di $f$ e un vettore $r = (1, 2)$, si inizializza $\dot{x}$ con $(1, 2)$. Avviando la procedura con $(1, 2)$, l’output finale fornisce esattamente $\nabla f \cdot r$, dove $r$ è il vettore di inizializzazione. In questo modo si ottiene il valore desiderato senza calcolare esplicitamente il gradiente.
[01:00] Lo stesso principio vale se il sistema ha più uscite. Se si imposta il vettore iniziale $\dot{x}$ uguale a $b$, l’output $y$ coincide con il prodotto della Jacobiana di $f$ per $b$:
```math
y = J b.
```
Qui $J$ è la matrice delle derivate parziali della funzione vettoriale rispetto agli ingressi; il prodotto $Jb$ è la derivata direzionale della funzione lungo la direzione $b$.
## Modalità reverse: prodotti Jacobiana-vettore impostando l’uscita
[01:30] In modalità reverse, l’inizializzazione avviene sullo spazio delle uscite. Se l’uscita è caratterizzata da $m$ variabili, si imposta un vettore nel dominio delle uscite. La reverse mode calcola simultaneamente tutte le derivate rispetto agli input in un singolo backward.
[01:45] Supponendo $M = 5$ e considerando il vettore $(1, 2, 3, 4, 5)$: inizializzando l’uscita con questo vettore e svolgendo i passi della reverse mode, si ottiene un vettore pari al prodotto della Jacobiana trasposta per il vettore $B$:
```math
J^\top B.
```
Se $f: \mathbb{R}^n \to \mathbb{R}^m$ e si inizializza con $B \in \mathbb{R}^m$, il risultato finale è un vettore in $\mathbb{R}^n$, coerente con il fatto che la reverse mode propaga dalle uscite agli ingressi.
[02:15] Queste quantità (prodotti Jacobiana-vettore o gradiente-vettore) si calcolano senza costruire esplicitamente la matrice $J$ né il vettore gradiente di $f$, sfruttando la struttura del grafo computazionale.
## Metodi di ordine superiore: Newton e azione dell’Hessiano
[02:30] In alcune situazioni è utile impiegare metodi di ordine superiore, come il metodo di Newton. In 1D, il metodo di Newton coinvolge la seconda derivata della funzione; in multidimensione occorre l’estensione corrispondente. L’“Hessiano” $H(x)$ è la matrice delle derivate seconde della funzione.
[02:45] Per il metodo di Newton multidimensionale è richiesto il calcolo del prodotto tra l’azione di un operatore e un incremento $\Delta x$. “Azione” si riferisce all’applicazione dell’Hessiano a un vettore. Se $H(x)$ è l’Hessiano di $f$ in $x$, si richiede spesso $H(x)\, \Delta x$, cioè il prodotto matrice-vettore. L’obiettivo è calcolare efficientemente questa quantità, analogamente a quanto fatto per il prodotto Jacobiana-vettore.
[03:15] Se si ha una funzione con $n$ ingressi e una sola uscita, si può applicare la reverse mode per ottenere il gradiente con una sola passata backward, ottenendo le $n$ componenti di $\nabla f$. Il gradiente è usato come base per costrutti di ordine superiore.
[03:30] Successivamente, per ciascuna componente del gradiente, si dovrebbe calcolare una colonna dell’“azione” (ossia dell’Hessiano). Ciò richiede di ripetere $n$ passate reverse per ciascuna componente, portando a $n^2$ passate complessive. Se $n$ è grande, questo approccio non è praticabile.
## Approccio reverse+forward: funzione ausiliaria g(x)
[03:45] L’idea alternativa è l’approccio combinato reverse e forward. Si introduce una funzione $g(x)$ definita come gradiente di $f$ applicato a un vettore $b$, cioè la derivata direzionale di $f$ lungo $b$:
```math
g(x) = \nabla f(x) \cdot b = J(x)\, b,
```
dove $J(x)$ è la Jacobiana di $f$ nel punto $x$. La derivata direzionale lungo $b$ misura la variazione di $f$ nella direzione $b$.
[04:15] Se si calcola il gradiente di $g$, si ottiene l’azione dell’Hessiano di $f$ sul vettore $b$:
```math
\nabla g(x) = H(x)\, b,
```
dove $H(x)$ è l’Hessiano di $f$. Il gradiente di $g$ fornisce dunque il prodotto Hessiano-vettore, senza costruire esplicitamente l’Hessiano.
[04:30] In modalità reverse, il primo passo è una passata forward per calcolare i valori intermedi. Poiché si considera la nuova funzione $g$, si esegue la passata forward su $g$ per ottenere tutte le variabili intermedie, inclusa la combinazione $J(x)\, b$.
[04:45] Poi si calcola il gradiente di $g$ con una singola passata reverse. Il risultato è $H(x)\, b$, ossia l’azione dell’Hessiano su $b$. Il procedimento complessivo è: calcolare $g$ con una passata forward, quindi calcolare $\nabla g$ con una passata reverse, ottenendo $H(x)\, b$. Questo è coerente con l’inizializzazione in forward con $\dot{x} = b$ quando si considera la direzione $b$.
## Scelta tra forward e reverse e operazioni elementari
[05:15] Anche se la reverse mode è spesso preferita quando il numero di ingressi è molto maggiore del numero di uscite, occorre saper calcolare derivate di operazioni elementari, come $\log(v_1)$. Servono regole locali di derivazione per ciascuna operazione. Queste regole sono le basi della propagazione delle derivate nel grafo computazionale.
[05:30] Uno strumento importante, componente fondamentale di molte librerie di differenziazione automatica, è quello dei numeri duali. I “numeri duali” permettono di ottenere simultaneamente il valore della funzione e della sua derivata prima tramite valutazioni strutturate.
## Numeri duali: definizione e proprietà di base
[05:45] I numeri duali sono simili ai numeri complessi. I complessi hanno parte reale e parte immaginaria, con unità immaginaria $i$ tale che $i^2 = -1$. Un numero complesso generico è $a + i b$. Nei numeri duali, un elemento ha la forma $a + \varepsilon\, d$, dove $\varepsilon$ è un’unità duale con proprietà diversa: $\varepsilon$ è nilpotente, cioè $\varepsilon \neq 0$ ma:
```math
\varepsilon^2 = 0.
```
Questa proprietà consente di troncare naturalmente le espansioni al primo ordine.
[06:00] Un numero duale ha la forma $a + \varepsilon\, d$. La nilpotenza di $\varepsilon$ significa che prodotti di secondo ordine si annullano, semplificando le espansioni di Taylor. Questa struttura è utile per estrarre derivate prime come coefficienti della parte duale.
[06:15] Una rappresentazione matriciale utile è:
```math
a + \varepsilon b \;\;\leftrightarrow\;\;
\begin{pmatrix}
a & 0 \\
b & a
\end{pmatrix}
= a I + b E,
```
dove $I$ è l’identità ed $E = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$. Questa matrice $E$ è diversa da zero ma soddisfa:
```math
E^2 = 0,
```
poiché è nulla al quadrato, a riflesso della nilpotenza di $\varepsilon$.
[06:45] Formalmente, un numero duale si scrive $a + \varepsilon b$. Analogamente ai complessi, $a$ è la parte reale e $b$ è la parte duale. La parte duale codifica l’informazione sulla derivata, quando la funzione viene valutata su un input perturbato dualmente.
## Numeri duali e derivate tramite espansione di Taylor
[07:00] Considerando una funzione $f$ e valutandola su un numero duale con parte duale unitaria, $x + \varepsilon$, si usa l’espansione di Taylor:
```math
f(x + \varepsilon) = f(x) + f'(x)\, \varepsilon + \frac{f''(x)}{2!}\, \varepsilon^2 + \cdots.
```
Poiché $\varepsilon^2 = 0$, tutti i termini di ordine superiore svaniscono, lasciando:
```math
f(x + \varepsilon) = f(x) + f'(x)\, \varepsilon.
```
La parte reale è $f(x)$, la parte duale è $f'(x)$: si ottiene il valore della funzione e la sua derivata prima in un’unica valutazione.
[07:30] Con una sola valutazione si ottiene simultaneamente il valore della funzione e il valore della derivata prima. Questa proprietà rende i numeri duali uno strumento semplice per calcolare derivate direzionali nel modo tangente.
## Esempio: funzione razionale con coseno
[07:45] Si consideri $f(x) = \dfrac{x^2}{\cos x}$ e si ponga $x = \pi + \varepsilon$. Il numeratore è:
```math
(\pi + \varepsilon)^2 = \pi^2 + 2\pi\, \varepsilon.
```
Il denominatore è:
```math
\cos(\pi + \varepsilon).
```
La valutazione della funzione si riduce a semplificazioni al primo ordine.
[08:00] Usando la formula di addizione per il coseno, $\cos(\pi + \varepsilon) = -\cos(\varepsilon)$. Siccome $\varepsilon$ è nilpotente, $\cos(\varepsilon) = 1$ al primo ordine. Il denominatore diventa quindi $-1$, poiché le componenti di ordine superiore si annullano.
[08:15] La divisione produce:
```math
\frac{\pi^2 + 2\pi\, \varepsilon}{-1} = -\pi^2 - 2\pi\, \varepsilon.
```
La parte reale $-\pi^2$ è $f(\pi)$, e la parte duale $-2\pi$ è $f'(\pi)$, in accordo con il calcolo esplicito della derivata.
## Operazioni con numeri duali: somma, prodotto, composizione
[08:30] Proprietà utili:
- Somma:
```math
(a + \varepsilon b) + (c + \varepsilon d) = (a + c) + \varepsilon(b + d).
```
Si sommano separatamente le parti reali e duali.
- Prodotto:
```math
(a + \varepsilon b)(c + \varepsilon d) = ac + \varepsilon(ad + bc),
```
poiché $\varepsilon^2 = 0$. La parte duale riflette la regola del prodotto.
[08:45] La regola del prodotto corrisponde alla derivata di $(fg)$:
```math
(fg)'(x) = f'(x)\, g(x) + f(x)\, g'(x),
```
che nella notazione duale si riflette nella parte duale $ad + bc$, con $a=f(x)$, $b=f'(x)$, $c=g(x)$, $d=g'(x)$.
[09:00] Per una funzione composta $h(x) = g(f(x))$, valutando su $x + \varepsilon$:
```math
f(x + \varepsilon) = f(x) + f'(x)\, \varepsilon,
```
e poi:
```math
g\big(f(x) + f'(x)\, \varepsilon\big) = g\big(f(x)\big) + g'\big(f(x)\big)\, f'(x)\, \varepsilon.
```
La parte duale è $g'(f(x)) f'(x)$, cioè la regola della catena:
```math
h'(x) = g'(f(x))\, f'(x).
```
## Estensione ai numeri duali di ordine superiore
[09:15] Per calcolare derivate di ordine superiore, si modifica la definizione chiedendo:
```math
\varepsilon \neq 0,\quad \varepsilon^2 \neq 0,\quad \varepsilon^3 = 0.
```
In tal caso, l’espansione di Taylor al secondo ordine diventa:
```math
f(x + \varepsilon) = f(x) + f'(x)\, \varepsilon + \frac{f''(x)}{2}\, \varepsilon^2,
```
perché i termini di ordine $\ge 3$ si annullano.
[09:30] Con una sola valutazione si ottengono: il valore della funzione, la derivata prima e la derivata seconda (con fattore $1/2$ associato a $\varepsilon^2$). Questo estende il calcolo duale a informazioni di secondo ordine.
[09:45] Per funzioni di più variabili, si applica lo stesso trucco per ciascuna variabile. Per calcolare la derivata rispetto a $x_k$, si imposta $x_k \mapsto x_k + \varepsilon$ e le altre componenti $x_j$ restano invariati. Il risultato restituisce la derivata parziale rispetto a $x_k$.
## Esempio di seconda derivata: potenza cubica
[10:00] Consideriamo $f(x) = x^3$. Si imposta l’ingresso $x$ con componenti di ordine fino al secondo: $(2, 1, 0)$, dove la parte reale è $2$, la parte duale è $1$, e la componente di secondo ordine è $0$. Questa rappresentazione estesa consente di calcolare derivate fino al secondo ordine.
[10:15] Si calcola $x^3$ ricavando $x^2$ e poi moltiplicando di nuovo per $x$. Il risultato è un numero duale esteso che contiene:
- il valore della funzione in $x=2$, ossia $8$,
- la derivata prima,
- la derivata seconda,
con i coefficienti coerenti con:
```math
x^3 = 8 + 12\, \varepsilon + 6\, \varepsilon^2,
```
dove i coefficienti corrispondono a $f'(2) = 12$ e $f''(2) = 12$, con il fattore $1/2$ incorporato nella componente di $\varepsilon^2$.
## Uso pratico dei numeri duali: calcolo numerico delle derivate
[10:30] I numeri duali, come l’AD, non sono pensati per dedurre formule simboliche generali, ad esempio per $f(x) = \dfrac{1}{x^2}$. Formalmente:
```math
(x + \varepsilon)^{-2} = \frac{1}{(x + \varepsilon)^2} = \frac{1}{x^2 + 2x\, \varepsilon + \varepsilon^2}.
```
Poiché $\varepsilon^2 = 0$, si ha:
```math
\frac{1}{x^2 + 2x\, \varepsilon} = \frac{1}{x^2}\, \frac{1}{1 + \frac{2\varepsilon}{x}}.
```
[10:45] Espandendo al primo ordine:
```math
\frac{1}{1 + \frac{2\varepsilon}{x}} \approx 1 - \frac{2\varepsilon}{x},
```
quindi:
```math
\frac{1}{(x + \varepsilon)^2} \approx \frac{1}{x^2} - \frac{2}{x^3}\, \varepsilon.
```
La parte reale è $x^{-2}$, la parte duale è $-2 x^{-3}$, ossia la derivata. L’interesse principale sta nel calcolo numerico dei valori, non nella costruzione di formule simboliche.
## Gradiente di una funzione bivariata con numeri duali
[11:00] Si vuole il gradiente di $f(x_1, x_2) = x_1 \cos(x_2)$ nel punto $(2, \pi)$. Si considerano due valutazioni:
- Derivata rispetto a $x_1$: si imposta $x_1 = 2 + \varepsilon$, $x_2 = \pi + 0$. La valutazione produce la componente relativa a $x_1$.
- Derivata rispetto a $x_2$: si imposta $x_1 = 2 + 0$, $x_2 = \pi + \varepsilon$. La valutazione produce la componente relativa a $x_2$.
Il “gradiente” è il vettore delle due derivate parziali.
[11:30] I risultati sono le due componenti del gradiente nel punto $(2, \pi)$:
```math
\nabla f(2, \pi) = \big(-1,\, 0\big).
```
La prima componente $-1$ deriva dalla valutazione con $x_1 = 2 + \varepsilon$, $x_2 = \pi$, mentre la seconda componente $0$ deriva dalla valutazione con $x_2 = \pi + \varepsilon$.
[11:45] Il metodo consiste nell’impostare ciascuna variabile con una perturbazione duale quando si desidera la derivata parziale corrispondente, mantenendo le altre variabili senza perturbazione. Si ottengono, con valutazioni puntuali, le componenti del gradiente desiderato.
## Conclusione operativa
[12:00] Il procedimento mostra come, tramite modalità forward e reverse e l’uso dei numeri duali, si calcolino prodotti gradiente-vettore, Jacobiana-vettore e Hessiano-vettore senza costruire esplicitamente matrici di derivate. L’approccio combinato reverse+forward consente di ottenere efficientemente l’azione dell’Hessiano su un vettore. I numeri duali forniscono un meccanismo per ottenere, con una singola valutazione, il valore della funzione e delle sue derivate, includendo l’estensione a derivate di ordine superiore tramite unità duali nilpotenti di grado maggiore.