## Metodi di Newton e compromessi computazionali
[00:00] Si prosegue l’analisi dei Metodi di Newton mettendo in evidenza un compromesso centrale. Da un lato, il metodo di Newton ha ordine di convergenza due, cioè una convergenza teoricamente quadratica. Dall’altro lato, richiede il calcolo dell’“azione”, intesa come matrice Hessiana, e in particolare la soluzione di un sistema lineare in cui l’azione compare come parametro. La convergenza quadratica significa che l’errore si riduce approssimativamente come il quadrato dell’errore precedente, ma il costo per risolvere sistemi lineari con matrici dense può essere elevato.
[00:20] Nel metodo della discesa del gradiente, a ogni iterazione il numero di operazioni è dell’ordine di $n$. In pratica si eseguono prodotti scalari e operazioni vettoriali poco costose dal punto di vista computazionale, pur non garantendo una velocità di convergenza elevata. La complessità per iterazione è bassa, ma la convergenza è tipicamente lineare, dunque lenta su molti problemi.
[00:40] Il metodo di Newton è promettente per la sua convergenza quadratica in teoria; tuttavia, la necessità di risolvere un sistema lineare comporta un costo computazionale dell’ordine di $m^3$, dove $m$ rappresenta la dimensione del sistema lineare da risolvere (in genere $m=n$). L’obiettivo dei metodi quasi-Newton è mantenere per quanto possibile le buone proprietà di convergenza del metodo di Newton riducendone i costi computazionali, evitando la costruzione e l’inversione completa della Hessiana.
## Richiamo del problema di ottimizzazione e approssimazione dell’azione
[01:00] Si consideri il problema di minimizzazione di una funzione $f:\mathbb{R}^n\to\mathbb{R}$. La regola di aggiornamento nel metodo di Newton applica un’operazione al gradiente, utilizzando l’azione come matrice chiave. L’azione è assunta definita positiva per garantire l’unicità del minimizzatore della funzione quadratica di approssimazione locale.
[01:20] L’“azione” è la matrice Hessiana $H(x)$ di $f$ nel punto $x$, definita come la matrice delle derivate seconde. Una Hessiana definita positiva implica che la funzione di approssimazione quadratica locale sia strettamente convessa, assicurando un unico minimizzatore. La regola di aggiornamento del Newton classico è:
```math
x_{k+1} = x_k - H(x_k)^{-1}\,\nabla f(x_k),
```
dove $H(x_k)$ è la Hessiana valutata in $x_k$ e $\nabla f(x_k)$ è il gradiente di $f$ in $x_k$. Questa formula indica che si muove nella direzione che annulla la derivata del modello quadratico.
[01:40] L’idea quasi-Newton consiste nel sostituire la vera azione con un’approssimazione. Si introduce una matrice $v_k$ o $B_k$ come approssimazione della Hessiana al passo $k$. In questo modo, si conserva la struttura dell’aggiornamento riducendo il costo computazionale associato alla risoluzione del sistema lineare con l’Hessiana esatta.
[02:00] Sostituendo l’azione nella formula di Newton con $v_k$ e supponendo che $v_k$ sia definita positiva, si ottiene una regola di aggiornamento che ricalca il Newton classico, ma con la matrice approssimata al posto della Hessiana esatta. La direzione di aggiornamento è costruita sulla base di $v_k$, mirando a una direzione di discesa.
## Direzione di ricerca e passo lungo la direzione
[02:20] In questa formulazione, la quantità calcolata a partire da $v_k$ è interpretata come una direzione di ricerca, non come aggiornamento immediato. Analogamente al metodo del gradiente, si individua una direzione lungo cui procedere verso il minimo e si decide quanto avanzare lungo tale direzione.
[02:40] Nella terminologia dei metodi quasi-Newton, si indica con $d_k$ la direzione di ricerca:
```math
d_k = - B_k^{-1}\,\nabla f(x_k),
```
dove $B_k$ approssima la Hessiana (o la sua inversa). La direzione è definita applicando l’inversa approssimata di $B_k$ al gradiente e inserendo un segno negativo, così da puntare verso la diminuzione della funzione.
[03:00] Come in ogni metodo basato su una direzione di ricerca, è necessario stabilire la lunghezza del passo da compiere lungo $d_k$. Si introduce quindi un parametro di passo $\alpha_k$ che determina l’entità dell’aggiornamento lungo $d_k$. La regola di aggiornamento diventa:
```math
x_{k+1} = x_k + \alpha_k\, d_k,
```
dove $\alpha_k$ è scelto tramite un criterio di ricerca in linea per garantire una riduzione adeguata del valore di $f$.
[03:20] Si può ridurre il problema a una dimensione lungo $d_k$, implementando un algoritmo di line search per determinare $\alpha_k$. Le condizioni di Wolfe costituiscono una modalità efficace: esse controllano simultaneamente la riduzione della funzione e la curvatura lungo la direzione. In sintesi, l’aggiornamento prevede una direzione $d_k$ e un passo $\alpha_k$ determinato da una ricerca in linea.
## Strategia di aggiornamento dell’approssimazione dell’azione
[03:40] La scelta del passo $\alpha_k$ è affrontata separatamente; la questione centrale riguarda come calcolare le matrici $v_k$ e $v_{k+1}$. Dato $v_k$ (o $B_k$), occorre definire una strategia per ottenere $v_{k+1}$ e continuare il processo iterativo in modo efficiente e stabile.
[04:00] Si delineano le idee per progettare una strategia di calcolo di $v_k$. Si tratta di un algoritmo iterativo: si parte da un’ipotesi iniziale $x_0$ per la soluzione e da un’approssimazione iniziale dell’azione. Un’opzione comune è la matrice identità, che è definita positiva e fornisce un punto di partenza neutro.
[04:20] È definita una tolleranza positiva che stabilisce il criterio di arresto. Poiché l’obiettivo è trovare il minimo, si misura la norma del gradiente $\|\nabla f(x_k)\|$; se scende sotto la soglia prefissata, l’iterazione si interrompe. Una norma piccola del gradiente indica prossimità a un punto stazionario.
[04:40] Si calcola la direzione di ricerca sfruttando le informazioni disponibili al passo $k$, tra cui $B_k$ e il gradiente $\nabla f(x_k)$. La direzione segue la relazione già esposta. Successivamente, mediante un algoritmo di line search, si determina $\alpha_k$ secondo condizioni che garantiscono il progresso.
[05:00] Infine, si aggiorna la soluzione:
```math
x_{k+1} = x_k + \alpha_k\, d_k,
```
si costruisce la nuova approssimazione dell’azione e si incrementa l’indice $k \leftarrow k+1$, ripetendo il ciclo fino al soddisfacimento del criterio di arresto.
## Dettagli formali e definizioni operative
[05:20] L’azione è la matrice Hessiana $H(x)$ di $f$ nel punto $x$, cioè la matrice delle derivate seconde. Quando $H(x)$ è definita positiva, la funzione di Taylor di secondo ordine:
```math
q(\Delta x) = f(x) + \nabla f(x)^\top \Delta x + \tfrac{1}{2}\,\Delta x^\top H(x)\,\Delta x
```
è strettamente convessa in $\Delta x$, e il suo minimizzatore è unico. Questa approssimazione guida il metodo di Newton.
[05:40] Nel Newton classico, la direzione di Newton $p_k$ è data dalla soluzione del sistema lineare:
```math
H(x_k)\, p_k = -\nabla f(x_k),
```
equivalente a $p_k = -H(x_k)^{-1}\,\nabla f(x_k)$. La risoluzione di tale sistema implica un costo tipico $O(n^3)$ per fattorizzazioni dirette su matrici dense, motivando l’uso di approssimazioni meno onerose.
[06:00] Nei metodi quasi-Newton, si sostituisce $H(x_k)$ con un’approssimazione $B_k$ o si approssima direttamente l’inversa $B_k^{-1}$. La direzione di ricerca è:
```math
d_k = -B_k^{-1}\,\nabla f(x_k),
```
e l’aggiornamento della soluzione segue $x_{k+1} = x_k + \alpha_k\, d_k$, con $\alpha_k$ ottenuto tramite line search.
[06:20] La definizione di $\alpha_k$ tramite ricerca in linea consiste nel considerare la funzione monodimensionale:
```math
\phi(\alpha) = f(x_k + \alpha\, d_k),
```
e nel selezionare un valore di $\alpha$ che soddisfi criteri di riduzione del valore di $f$ e di adeguata curvatura lungo $d_k$ (ad esempio condizioni di Wolfe).
[06:40] Il criterio di arresto basato sulla norma del gradiente è:
```math
\|\nabla f(x_k)\| \le \text{tolleranza},
```
dove la tolleranza è un parametro positivo scelto a priori. Quando questa condizione è verificata, si assume di essere sufficientemente vicini a un minimo.
## Schema algoritmico iterativo
[07:00] Lo schema generale di un algoritmo quasi-Newton si riassume in passi:
- Scegliere $x_0$ (stima iniziale) e $B_0$ (approssimazione iniziale dell’azione), ad esempio $B_0 = I$.
- Fissare una tolleranza positiva per l’arresto.
[07:20] Iterare per $k = 0,1,2,\dots$:
- Calcolare il gradiente $\nabla f(x_k)$.
- Determinare la direzione $d_k = -B_k^{-1}\,\nabla f(x_k)$.
[07:40] - Eseguire una ricerca in linea per trovare $\alpha_k$ appropriato.
- Aggiornare la soluzione: $x_{k+1} = x_k + \alpha_k\, d_k$.
- Aggiornare l’approssimazione dell’azione: costruire $B_{k+1}$ con le informazioni correnti.
- Verificare il criterio di arresto $\|\nabla f(x_{k+1})\| \le \text{tolleranza}$; se vero, terminare.
[08:00] Questo schema mantiene l’ordine logico del processo: ipotesi iniziali, determinazione delle direzioni, scelta del passo con ricerca in linea, aggiornamento della soluzione e della matrice che approssima l’azione, controllo della condizione di arresto.
## Considerazioni sulla definita positività e unicità del minimizzatore
[08:20] L’assunzione che $B_k$ sia definita positiva è cruciale. Una matrice definita positiva garantisce che la direzione $d_k$ sia di discesa e che l’approssimazione quadratica locale sia ben condizionata. In assenza di questa proprietà, la direzione potrebbe non ridurre la funzione.
[08:40] La definita positività si riflette nella forma quadratica:
```math
\Delta x^\top B_k\, \Delta x > 0 \quad \text{per ogni } \Delta x \neq 0,
```
che assicura convessità locale dell’approssimazione e unicità del minimizzatore della funzione quadratica associata.
## Riepilogo operativo della direzione e del passo
[09:00] La direzione di ricerca $d_k$ è definita da:
```math
d_k = -B_k^{-1}\,\nabla f(x_k),
```
e rappresenta la direzione lungo cui si effettua la discesa del valore della funzione.
[09:20] Il passo $\alpha_k$ è scelto tramite un algoritmo di line search, valutando la funzione lungo $d_k$:
```math
\phi(\alpha) = f(x_k + \alpha\, d_k).
```
La selezione di $\alpha_k$ mira a ridurre $f$ in modo efficace e robusto.
[09:40] L’aggiornamento della soluzione segue:
```math
x_{k+1} = x_k + \alpha_k\, d_k,
```
e prepara il terreno per l’aggiornamento della matrice $B_{k+1}$, che approssima la Hessiana al nuovo punto.
## Parametri iniziali e ciclo iterativo
[10:00] La scelta di $x_0$ influenza l’andamento iniziale. Un $B_0$ impostato come identità, $I$, consente di partire con una direzione analoga a quella della discesa del gradiente, poiché $B_0^{-1} = I$ e quindi $d_0 = -\nabla f(x_0)$.
[10:20] La tolleranza determina il livello di accuratezza richiesto. Una tolleranza molto piccola comporta più iterazioni prima di soddisfare il criterio di arresto; una tolleranza più grande accelera la terminazione con minore precisione.
[10:40] La costruzione di $B_{k+1}$ avviene dopo l’aggiornamento della soluzione. È necessario definire una regola coerente per ottenere l’approssimazione dell’azione al passo successivo usando le informazioni computate a $x_{k+1}$, mantenendo stabilità e efficienza.
## Controllo del gradiente e condizione di arresto
[11:00] Il controllo del gradiente si effettua misurando la sua norma. Il criterio:
```math
\|\nabla f(x_k)\| \le \epsilon,
```
con $\epsilon>0$ prefissato, stabilisce la terminazione del processo quando la pendenza locale è sufficientemente piccola da indicare vicinanza a un punto stazionario.
[11:20] La scelta della norma (ad esempio Euclidea) misura l’ampiezza del gradiente per stabilire la prossimità al minimo. Una norma piccola suggerisce che ulteriori riduzioni significative di $f$ potrebbero essere limitate.
## Ricerca in linea e condizioni di Wolfe
[11:40] La ricerca in linea proietta il problema lungo la direzione $d_k$. La funzione monodimensionale:
```math
\phi(\alpha) = f(x_k + \alpha\, d_k)
```
viene analizzata per trovare un $\alpha_k$ che garantisca una diminuzione accettabile del valore di $f$.
[12:00] Le condizioni di Wolfe sono criteri che regolano la scelta del passo, assicurando che $\phi(\alpha)$ diminuisca e che la curvatura lungo $d_k$ sia appropriata. La loro adozione migliora l’efficacia rispetto a una ricerca in linea non strutturata.
## Sintesi del flusso dell’algoritmo
[12:20] La sequenza operativa è:
- Avvio con $x_0$, $B_0$ (ad esempio $I$), e tolleranza $\epsilon$.
- Per il passo $k$:
  - Calcolo di $\nabla f(x_k)$.
  - Determinazione di $d_k = -B_k^{-1}\,\nabla f(x_k)$.
  - Scelta di $\alpha_k$ tramite line search.
  - Aggiornamento $x_{k+1} = x_k + \alpha_k\, d_k$.
  - Costruzione di $B_{k+1}$.
  - Verifica $\|\nabla f(x_{k+1})\| \le \epsilon$.
[12:40] Il ciclo continua incrementando $k$ fino alla soddisfazione del criterio di arresto, perseguendo la minimizzazione di $f$ con un compromesso tra accuratezza di Newton ed efficienza computazionale.
## Schema dell’algoritmo e punti chiave
[00:00] Si presenta una rappresentazione schematica dell’algoritmo, evidenziando due punti chiave. Uno è classico nella formulazione (direzione e passo), l’altro riguarda la progettazione di aggiornamenti per $B_k$ più complessi. L’obiettivo è concentrare l’attenzione su entrambi per delineare le proprietà desiderate delle matrici approssimanti.
## Proprietà richieste per la matrice Bk
[00:13] Si definiscono le proprietà da imporre a $B_k$. In primo luogo, $B_k$ deve essere non singolare, cioè invertibile, condizione necessaria per calcolare la direzione di discesa.
[00:26] In secondo luogo, $B_k$ deve generare una direzione di discesa. La direzione $d_k$ calcolata deve ridurre il valore della funzione obiettivo lungo quella direzione, cioè $\nabla f(x_k)^\top d_k<0$.
[00:38] In terzo luogo, $B_k$ deve essere simmetrica, in coerenza con la simmetria della Hessiana. Queste sono le prime tre proprietà fondamentali richieste per $B_k$.
## Matrici simmetriche definite positive e implicazioni
[00:52] Se $B_k$ è simmetrica e definita positiva a ogni iterazione, allora è automaticamente non singolare: tutti gli autovalori sono positivi e il determinante è non nullo. Resta da assicurare la discesa. Sostituendo $d_k=-B_k^{-1}\,\nabla f(x_k)$ si ha:
```math
\nabla f(x_k)^\top d_k = -\,\nabla f(x_k)^\top B_k^{-1}\,\nabla f(x_k).
```
Poiché $B_k^{-1}$ è definita positiva, la forma $x^\top B_k^{-1} x$ è positiva per ogni $x\neq 0$, quindi l’espressione è negativa e garantisce direzione di discesa.
[01:12] La negatività di $\nabla f(x_k)^\top d_k$ assicura che $f$ diminuisca per piccoli passi positivi lungo $d_k$. In sintesi, scegliendo $B_k$ simmetrica e definita positiva, si soddisfano non singolarità, simmetria e generazione di una direzione di discesa.
[01:36] Questa scelta non è sufficiente: è necessaria un’ulteriore proprietà cruciale per passare da Newton a quasi-Newton riducendo la complessità. Occorre costruire aggiornamenti economici che sfruttino informazioni locali e rispettino un vincolo strutturale.
## Efficienza computazionale e aggiornamenti di Bk
[01:52] È essenziale che le matrici $B_k$ siano economiche da calcolare. Si desidera poter computare $B_{k+1}$ partendo da quantità già disponibili. Si introducono due vettori:
```math
\delta_k = x_{k+1} - x_k = \alpha_k\, d_k,
```
incremento nella posizione, e
```math
\gamma_k = \nabla f(x_{k+1}) - \nabla f(x_k),
```
variazione nel gradiente.
[02:04] Si usa uno sviluppo di Taylor del primo ordine per approssimare la variazione del gradiente:
```math
\gamma_k \approx \nabla^2 f(x_k)\,\delta_k,
```
cioè la Hessiana applicata allo spostamento $\delta_k$. Questa relazione motiva la progettazione di aggiornamenti coerenti.
[02:19] L’approssimazione $\gamma_k \approx H(x_k)\,\delta_k$ indica che le variazioni di gradiente sono spiegate dall’azione della Hessiana lungo lo spostamento. Sostituendo la Hessiana con la sua approssimazione, si introduce un vincolo noto come condizione di secante.
## Condizione di secante (second condition)
[02:35] Il vincolo fondamentale è la condizione di secante:
```math
B_{k+1}\,\delta_k = \gamma_k.
```
Essa impone che l’azione approssimata $B_{k+1}$, applicata allo spostamento $\delta_k$, predica esattamente la variazione osservata del gradiente $\gamma_k$.
[02:53] La condizione di secante è centrale nei metodi quasi-Newton. Il termine “secante” richiama l’analogia con differenze finite per derivare informazioni sulle derivate nel caso monodimensionale.
## Interpretazione 1D della condizione di secante
[03:07] Nel caso monodimensionale, i vettori diventano scalari e il gradiente coincide con la derivata prima. La controparte della condizione di secante è:
```math
b_{k+1}\,\delta_k = \gamma_k,
```
da cui si ottiene un’approssimazione della derivata seconda:
```math
b_{k+1} \approx \frac{\gamma_k}{\delta_k} = \frac{f'(x_{k+1}) - f'(x_k)}{x_{k+1} - x_k}.
```
Questo è il rapporto di differenze finite tra variazione della derivata prima e variazione della variabile.
[03:18] In sintesi, la condizione di secante in 1D approssima la derivata seconda mediante una differenza finita. Nel caso multidimensionale, la condizione $B_{k+1}\,\delta_k = \gamma_k$ generalizza tale concetto.
[03:35] La generalizzazione richiede aggiornamenti matriciali che rispettino simmetria, definita positività e vincoli di basso rango, per limitare la variazione tra iterazioni.
## Regolarità degli aggiornamenti e vicinanza tra iterazioni
[03:49] In $n$ dimensioni, $B_{k+1}$ dovrebbe essere “vicina” a $B_k$, cioè non discostarsi eccessivamente. Variazioni regolari semplificano l’analisi e favoriscono la stabilità. Quando la sequenza $x_k$ converge a $x^\star$, si desidera che $B_k$ tenda alla vera Hessiana in $x^\star$.
[04:02] “Essere vicini” si traduce nell’uso di aggiornamenti a basso rango, in particolare di rango uno o due, che modificano la matrice in modo controllato e con costo contenuto.
[04:19] Si impone inoltre che le operazioni per iterazione siano dell’ordine $n^2$, riducendo il costo rispetto al metodo di Newton, che richiede tipicamente $n^3$ operazioni per la risoluzione diretta del sistema lineare.
[04:31] Riassumendo, si vogliono sei proprietà per $B_k$: simmetria, definita positività (che implica non singolarità e direzione di discesa), condizione di secante, vicinanza degli aggiornamenti e costo computazionale $O(n^2)$.
## Aggiornamenti di rango uno: forma generale
[04:46] Si introduce l’aggiornamento di rango uno. Un possibile aggiornamento per $B_k$ è:
```math
B_{k+1} = B_k + u\,u^\top,
```
dove $u$ è un vettore. La matrice $u\,u^\top$ è di rango uno e simmetrica, quindi preserva la simmetria se $B_k$ è simmetrica.
[05:05] Si determina $u$ tramite la condizione di secante:
```math
B_{k+1}\,\delta_k = \gamma_k.
```
Sostituendo $B_{k+1} = B_k + u\,u^\top$:
```math
(B_k + u\,u^\top)\,\delta_k = \gamma_k \quad \Rightarrow \quad B_k\,\delta_k + u\,(u^\top \delta_k) = \gamma_k.
```
Si impone che il termine $u\,(u^\top \delta_k)$ compensi la discrepanza $\gamma_k - B_k\,\delta_k$.
[05:22] Dalla relazione si ricava:
```math
u\,(u^\top \delta_k) = \gamma_k - B_k\,\delta_k.
```
Poiché $u^\top \delta_k$ è uno scalare, si costruisce $u$ proporzionale a $\gamma_k - B_k\,\delta_k$, ottenendo l’aggiornamento simmetrico di rango uno (SR1):
```math
B_{k+1} = B_k + \frac{(\gamma_k - B_k\,\delta_k)\,(\gamma_k - B_k\,\delta_k)^\top}{(\gamma_k - B_k\,\delta_k)^\top \delta_k}.
```
[05:40] Questa forma soddisfa la condizione di secante forzando $B_{k+1}\,\delta_k = \gamma_k$. Inoltre, impiega solo quantità note all’iterazione corrente: $\delta_k$, $\gamma_k$ e il prodotto $B_k\,\delta_k$.
[05:57] L’SR1 è un aggiornamento semplice e di costo contenuto, ma presenta criticità che vanno considerate per garantire robustezza.
## Proprietà e criticità dell’aggiornamento SR1
[06:12] Un limite dell’SR1 è che, anche se $B_k$ è definita positiva, $B_{k+1}$ potrebbe non esserlo. Ne consegue che la direzione calcolata con $B_{k+1}$ potrebbe non essere di discesa, compromettendo una delle proprietà desiderate.
[06:24] Inoltre, il denominatore $(\gamma_k - B_k\,\delta_k)^\top \delta_k$ può essere molto piccolo o prossimo a zero, rendendo l’aggiornamento numericamente instabile. In tali casi, l’SR1 può produrre correzioni non affidabili.
[06:36] Dal punto di vista del costo computazionale, il calcolo di $B_{k+1}$ tramite SR1 coinvolge somme, prodotti matrice-vettore e prodotti scalari, dunque operazioni $O(n^2)$, soddisfacendo il requisito per iterazione limitatamente all’aggiornamento di $B_k$.
[06:49] Rimane però il problema del calcolo della direzione $d_k = -B_k^{-1}\,\nabla f(x_k)$: risolvere il sistema lineare in generale richiede $O(n^3)$. Per evitare questo costo, si può aggiornare direttamente l’inversa dell’Hessiana approssimata usando formule specifiche.
## Formula di Sherman–Morrison–Woodbury e aggiornamento dell’inversa
[07:03] La formula di Sherman–Morrison–Woodbury consente di calcolare l’inversa di matrici con aggiornamenti a basso rango:
```math
(A + U\,C\,V)^{-1} = A^{-1} - A^{-1}U\,(C^{-1} + V\,A^{-1}U)^{-1}\,V\,A^{-1}.
```
Nel caso di rango uno ($C=1$) e aggiornamento $A + u\,u^\top$, si ottiene:
```math
(A + u\,u^\top)^{-1} = A^{-1} - \frac{A^{-1}u\,u^\top A^{-1}}{1 + u^\top A^{-1}u}.
```
Questa relazione permette di aggiornare l’inversa senza risolvere un sistema completo.
[07:25] Applicando SMW a $B_{k+1} = B_k + u\,u^\top$ e mantenendo $H_k = B_k^{-1}$, con inizializzazione $B_0=I$ e $H_0=I$, si ottiene un aggiornamento efficiente dell’inversa a ogni passo.
[07:36] Definendo $A = B_k$ e scegliendo $u$ coerente con SR1, si ricava l’aggiornamento di rango uno per l’inversa:
```math
H_{k+1} = H_k - \frac{H_k\,u\,u^\top H_k}{1 + u^\top H_k\,u}.
```
Questo evita fattorizzazioni complete e mantiene il costo per iterazione contenuto.
[07:52] In pratica, aggiornando $H_k$ con operazioni vettoriali e scalari, si evita di risolvere sistemi lineari densi. La direzione di ricerca diventa:
```math
d_k = -H_k\,\nabla f(x_k),
```
calcolabile con un prodotto matrice-vettore $O(n^2)$, coerente con l’obiettivo di efficienza.
[08:06] L’aggiornamento su $H_k$ è di rango uno, poiché la correzione è un prodotto esterno di vettori. Questo mantiene semplicità e costo ridotto per il calcolo di $d_k$ e l’aggiornamento dell’inversa.
## Algoritmo quasi-Newton con inversa aggiornata
[08:21] Schema algoritmico:
- Inizializzazione: $H_0$ come congettura iniziale dell’inversa dell’azione (ad esempio identità).
- Controllo di convergenza sul gradiente.
- Calcolo della direzione di ricerca:
```math
d_k = -H_k\,\nabla f(x_k).
```
- Ricerca del passo $\alpha_k$ e aggiornamento di $x$:
```math
x_{k+1} = x_k + \alpha_k\, d_k.
```
- Aggiornamento di $H$ con SMW o con la forma indotta da SR1, basato su prodotti scalari e vettoriali.
[08:40] Ogni iterazione richiede $O(n^2)$ operazioni: prodotti scalari, prodotti matrice-vettore e aggiornamenti di basso rango. Questo soddisfa il requisito di costo per iterazione.
## Convergenza del metodo quasi-Newton con SR1
[08:54] Riguardo alla convergenza, il metodo di Newton ha ordine quadratico. Con un’approssimazione dell’azione, il metodo quasi-Newton perde quadraticità ma conserva un ordine di convergenza superlineare, compreso tra 1 e 2, in condizioni ideali e su problemi regolari.
[09:05] L’aggiornamento SR1:
- soddisfa la condizione di secante,
- opera con costo $O(n^2)$ per iterazione,
- non garantisce la definita positività di $B_{k+1}$, e quindi non assicura sempre una direzione di discesa.
[09:16] SR1 è utile ma talvolta non affidabile. Si passa quindi verso metodi più robusti che preservino la definita positività e stabilità numerica.
## Verso metodi più affidabili: BFGS
[09:30] Un metodo molto utilizzato è BFGS, che mira a soddisfare tutte le sei proprietà:
- simmetria,
- non singolarità,
- direzione di discesa (garantita dalla definita positività),
- condizione di secante,
- vicinanza/regolarità dell’aggiornamento (basso rango),
- costo computazionale $O(n^2)$ per iterazione.
[09:42] Il passaggio da SR1 a BFGS preserva la definita positività e la stabilità numerica, senza rinunciare alla condizione di secante e all’efficienza, tramite aggiornamenti di rango due strutturati.
## Aggiornamenti a bassa dimensione di rango: passaggio da rango 1 a rango 2
[00:00] Si continua con aggiornamenti a basso rango passando da rango 1 a rango 2. A partire da $V_k$ si introducono due vettori $u$ e $v$ e si aggiungono a $V_k$ due termini $u\,u^\top$ e $v\,v^\top$. L’obiettivo è mantenere il costo per iterazione nell’ordine di $n^2$, garantendo modifiche controllate.
[00:15] La formula risultante coinvolge i vettori $\delta_k$ e $\gamma_k$ già introdotti. Anche con l’aggiornamento di rango 2, resta necessario determinare la direzione $d_k$ risolvendo un sistema lineare, operazione tipicamente $O(n^3)$ se affrontata in modo diretto. Sarebbe possibile usare ancora Sherman–Morrison–Woodbury, ma esiste un approccio più stabile che preserva anche la definita positività.
[00:35] L’idea è di aggiornare non direttamente $B_k$ o $H_k$, bensì il fattore di Cholesky di $B_k$. La fattorizzazione di Cholesky è una specializzazione della LU per matrici simmetriche definite positive. L’operatore “chol” verifica anche la definita positività: se “chol(A)” fallisce, $A$ non è definita positiva.
[00:58] La fattorizzazione di Cholesky consiste nel trovare una matrice triangolare inferiore $L$ tale che $B = L\,L^\top$ per $B$ simmetrica definita positiva. Sulla diagonale di $L$ compaiono elementi positivi. Il teorema fondamentale è: esiste la fattorizzazione di Cholesky se e solo se la matrice è definita positiva. Costruendo una regola di aggiornamento che da $L_k$ porti a $L_{k+1}$ con $B_{k+1} = L_{k+1}\,L_{k+1}^\top$, si mantiene la definita positività.
## Uso della fattorizzazione di Cholesky per la direzione di ricerca
[01:25] Si supponga $L_k$ noto. La matrice $B_k$ si scrive come $L_k\,L_k^\top$ e la direzione $d_k$ si determina risolvendo:
```math
B_k\, d_k = -\nabla f(x_k).
```
Si scompone in due sistemi triangolari:
```math
L_k\, y_k = -\nabla f(x_k), \quad L_k^\top\, d_k = y_k.
```
Questi sistemi si risolvono con sostituzioni in avanti e all’indietro.
[01:45] Grazie al fattore di Cholesky, la direzione $d_k$ si calcola con costo $O(n^2)$, evitando il costo cubico della risoluzione diretta del sistema. Si necessita quindi di un metodo per aggiornare $L_k$ in $L_{k+1}$ in modo efficiente e stabile.
[02:08] L’uso del fattore $L_k$ consente sia stabilità numerica sia verifica della definita positività. L’aggiornamento di $L$ deve rispettare la struttura triangolare e mantenere basso costo computazionale.
## Obiettivo: aggiornare il fattore di Cholesky mantenendo la definitezza positiva
[02:25] Si parte da un fattore generico $L$ (indicato come $L^+$ per enfatizzare la nuova iterazione) che approssima $B$. Nella notazione: $L^+ \equiv L_{k+1}$, $L \equiv L_k$, $B^+ \equiv B_{k+1}$ e $B \equiv B_k$.
[02:40] Si richiede che $B^+$ sia rappresentabile come $L^+(L^+)^\top$, e che soddisfi la seconda condizione (condizione di secante). Inoltre, $L^+$ deve essere vicino a $L$ per regolarità.
[02:58] In generale, si considera un fattore $J$ tale che $V = J\,J^\top$, dove $J$ non è necessariamente triangolare. Si cerca un nuovo fattore $J^+$ vicino a $J$, misurando la vicinanza con la norma di Frobenius:
```math
\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}.
```
Si vuole minimizzare $\|J^+ - J\|_F$ soggetto al vincolo che $J^+(J^+)^\top$ soddisfi l’aggiornamento richiesto.
[03:20] Il problema completo è complesso; in pratica si risolve una versione gestibile: dato un vettore $g$, si minimizza una quantità sotto un vincolo affine, ottenendo un aggiornamento esplicito per $J$ con struttura rank-one.
[03:40] Si cerca il punto $J^+$ più vicino a $J$ in uno spazio affine, ricavando una regola di aggiornamento di forma rango 1. Questo produce una formula concreta per $J$ che rispetta il vincolo e mantiene la vicinanza.
## Scelta del vettore g e coefficiente di proporzionalità
[03:58] Occorre scegliere un vettore $g$ adatto. Il vincolo porta a richiedere $g$ proporzionale a $J^\top \delta$, dove $\delta$ è la quantità nota dal passo corrente. Si introduce un fattore di proporzionalità $\beta$ tale che:
```math
g = \beta\, J^\top \delta.
```
[04:15] Il fattore $\beta$ è espresso con quantità disponibili: $\gamma$, $\delta$ e $B$ (con $B=B_k$). Si verifica che $\gamma^\top \delta > 0$ e che $V_k$ è definita positiva, quindi $\beta$ è ben definito. Una forma utile è:
```math
\beta = \sqrt{\frac{\gamma^\top B\, \gamma}{\delta^\top B\, \delta}},
```
dove la positività dei termini garantisce la realtà di $\beta$.
[04:40] Determinato $J^+$, si costruisce $V^+ = J^+(J^+)^\top$. Sostituendo $g = \beta\, J^\top \delta$ con $\beta$ dato, si ottiene una formula di aggiornamento che coincide con la regola VLGS (aggiornamento di rango 2):
```math
B_{k+1} = B_k + u\,u^\top + v\,v^\top,
```
ossia la somma di due aggiornamenti di rango 1 che producono un aggiornamento di rango 2.
[05:05] Una proprietà importante è l’indipendenza dalla scelta di $J$. Si può scegliere $J = L_k$, cioè il precedente fattore di Cholesky, per convenienza e stabilità computazionale.
## Triangolarità e riduzione QR: mantenere la forma triangolare
[05:20] Con $J = L_k$, l’aggiornamento per $J$ si esprime nella forma derivata, e si considera anche la trasposta. Poiché $L_k^\top$ è triangolare superiore, l’aggiunta di un aggiornamento di rango 1 non garantisce di conservarne la triangolarità: può introdurre elementi fuori dalla struttura triangolare.
[05:40] Di conseguenza, $J^{+\,\top}$ risulta “quasi triangolare”: una matrice triangolare superiore più un aggiornamento di rango 1. Per ripristinare la triangolarità, si ricorre alla fattorizzazione QR.
[05:52] La fattorizzazione QR di una matrice quadrata $X$ è:
```math
X = Q\,R,
```
dove $Q$ è ortogonale ($Q^\top Q = I$) e $R$ è triangolare superiore, con diagonale positiva mediante convenzioni standard.
[06:10] Applicando la QR a $J^{+\,\top}$, si ottengono $Q$ e $R$. Dato che $J^{+\,\top}$ è prossimo a una matrice triangolare, la QR può essere calcolata in $O(n^2)$ operazioni, sfruttando rotazioni di Givens o riflessioni di Householder.
[06:35] Considerando $V^+ = J^+(J^+)^\top$ e usando $J^{+\,\top} = Q\,R$, dall’ortogonalità di $Q$ segue:
```math
V^+ = J^+(J^+)^\top = (R^\top Q^\top)(Q\,R) = R^\top R.
```
Quindi $R$ fornisce un nuovo fattore triangolare coerente, mantenendo la definita positività e il costo computazionale contenuto.
## Schema algoritmico con aggiornamento del fattore di Cholesky e line search
[06:58] L’algoritmo richiede di scegliere:
- Un vettore iniziale $x_0$.
- Un fattore di Cholesky iniziale $L_0$, ad esempio l’identità.
- Una soglia o tolleranza per l’arresto.
[07:10] Se la tolleranza non è soddisfatta, si procede:
- Calcolare la direzione $d_k$ usando $L_k$:
  ```math
L_k\, y_k = -\nabla f(x_k), \quad L_k^\top\, d_k = y_k.
```
- Eseguire una line search per determinare $\alpha_k$.
- Calcolare gli aggiornamenti:
  ```math
\delta_k = x_{k+1} - x_k, \quad \gamma_k = \nabla f(x_{k+1}) - \nabla f(x_k).
```
- Aggiornare il fattore $L$ sfruttando la regola di rango 2 e la riduzione QR per mantenere la triangolarità.
[07:35] L’algoritmo combina aggiornamenti di basso rango, fattorizzazioni QR e Cholesky, condizione di secante e line search, fornendo una sintesi operativa dei concetti fondamentali.
## Proprietà dell’algoritmo e confronto tra metodi
[07:52] Proprietà salienti:
- Costo per iterazione $O(n^2)$ per risoluzione dei sistemi triangolari e QR “quasi triangolare”.
- L’aggiornamento preserva la definita positività.
- La regola soddisfa la condizione di secante ed è a basso rango (rango 2), limitando la variazione tra iterazioni.
- Convergenza locale superlineare su funzioni quadratiche strettamente convesse.
- Su una funzione quadratica, con line search esatto, individua il minimo esatto in al più $n$ iterazioni.
[08:18] La clausola “con line search esatto” è significativa: nel caso quadratico, il problema unidimensionale lungo la direzione di ricerca ha soluzione in forma chiusa, consentendo di determinare $\alpha^\star$ esplicitamente.
[08:30] Sintesi operativa:
- Discesa del gradiente: costo per iterazione proporzionale a prodotti vettoriali; convergenza lineare.
- Metodo di Newton: convergenza quadratica, costo per iterazione $O(n^3)$ per la risoluzione del sistema di Newton.
- Metodi quasi-Newton: convergenza superlineare, costo intermedio con aggiornamenti a basso rango e fattorizzazioni efficienti.
## Line search approssimato e condizioni di Wolfe
[08:52] La line search mira a minimizzare lungo $d_k$ la funzione:
```math
\phi(\alpha) = f(x_k + \alpha\, d_k),
```
ottenendo un $\alpha$ che riduca efficacemente $f$. Se $f$ non è quadratica semplice, si usa una line search approssimata con un ciclo interno.
[09:10] In assenza di $\alpha^\star$ esplicito, si cerca un $\alpha_k$ “buono”:
- non troppo piccolo,
- sufficiente a garantire progresso nella minimizzazione.
[09:20] Il semplice requisito $f(x_k + \alpha\, d_k) < f(x_k)$ non è sufficiente a garantire convergenza. Si impongono condizioni di Wolfe:
- Regola di Armijo (sufficient decrease):
  ```math
f(x_k + \alpha\, d_k) \le f(x_k) + c_1\, \alpha\, \nabla f(x_k)^\top d_k,
```
  con $c_1\in(0,1)$, per evitare passi troppo lunghi e assicurare riduzione sufficiente.
[09:42] - Condizione di curvatura:
  ```math
\nabla f(x_k + \alpha\, d_k)^\top d_k \ge c_2\, \nabla f(x_k)^\top d_k,
```
  con $c_2\in(0,1)$, per controllare la pendenza al nuovo punto e evitare passi eccessivamente piccoli.
[09:58] Si ricorda che $\phi'(\alpha) = \nabla f(x_k + \alpha\, d_k)^\top d_k$. Dato che $\phi'(0) = \nabla f(x_k)^\top d_k < 0$, la condizione di curvatura richiede che la pendenza al nuovo punto non sia troppo negativa rispetto a quella iniziale.
[10:15] Schema semplificato di line search:
- Si parte da $\alpha=1$.
- Si fissa $\rho$ (tipicamente $\rho=0{.}5$) e $c_1\approx 10^{-4}$.
- Si verifica Armijo; se non soddisfatta, si riduce $\alpha \leftarrow \rho\,\alpha$.
- Quando la condizione è soddisfatta, si accetta $\alpha_k = \alpha$.
- Per includere la curvatura, si aggiunge un controllo ulteriore.
[10:40] Nel contesto complessivo, la ricerca del passo è un sottoprocesso essenziale per garantire stabilità e riduzione del valore della funzione.
## Considerazioni finali e limiti
[10:55] Tra gli svantaggi principali dei metodi quasi-Newton con aggiornamenti a basso rango, si evidenziano i requisiti di memoria: in problemi di grandi dimensioni, la memorizzazione e l’aggiornamento dei fattori possono incidere significativamente, richiedendo strategie a memoria limitata per mantenere l’efficienza.
## Gestione della memoria nelle metodologie BFGS e L-BFGS
[00:00] La memorizzazione richiesta riguarda l’archiviazione di fattori come $B_k$, $H_k$ o $D_l$. Indipendentemente dall’oggetto scelto, l’entità dei dati da conservare è dell’ordine di $n^2$, pari al numero di elementi di una matrice quadrata $n\times n$.
[00:20] Per $n=100{,}000$, la matrice contiene circa $10^{10}$ elementi. In doppia precisione, ogni elemento occupa 8 byte; la memoria necessaria cresce come:
```math
\text{Memoria} \approx n^2 \times 8 \text{ byte}.
```
[00:40] Per $n=1{,}000{,}000$, la matrice ha $10^{12}$ elementi. La memoria in doppia precisione diventa dell’ordine dei terabyte:
```math
\text{Memoria} \approx 10^{12} \times 8 \text{ byte} = 8 \times 10^{12} \text{ byte} = 8 \text{ TB}.
```
Questa quantità è proibitiva nella pratica.
[01:00] Sebbene BFGS sia promettente, presenta limiti pratici: richiede la memorizzazione completa di una matrice $n\times n$. L’ostacolo principale è l’aggiornamento ricorsivo di una matrice densa che accumula informazioni sulle direzioni e sulle variazioni di gradiente.
## Versione a memoria limitata: L-BFGS
[01:20] Esiste una variante pratica, L-BFGS (“low memory”), progettata per ridurre i costi di memorizzazione e di calcolo, evitando la costruzione esplicita di matrici $n\times n$.
[01:40] L’idea centrale è approssimare l’effetto di $H_k$ senza memorizzarlo esplicitamente. Si conserva in memoria solo un sottoinsieme di $m$ coppie recenti $(\Delta,\Gamma)$.
[02:00] Qui, $m$ è tipicamente piccolo (10–20). Le coppie $(\Delta,\Gamma)$ rappresentano:
- $\Delta$: l’incremento di posizione tra iterazioni consecutive,
- $\Gamma$: la variazione del gradiente tra iterazioni consecutive.
[02:20] Per calcolare la direzione $d_k$, cioè il prodotto $H_k\, g_k$ con $g_k=\nabla f(x_k)$, si approssima il prodotto senza formare $H_k$:
```math
d_k = H_k\, g_k,
```
dove $H_k$ è l’approssimazione dell’inversa della Hessiana e $g_k$ il gradiente al passo $k$.
[02:40] $H_k$ viene aggiornato attraverso una relazione ricorsiva che usa le ultime $m$ coppie $(\Delta,\Gamma)$. La struttura incorpora le informazioni recenti mantenendo basso il costo.
[03:00] Iterando la relazione, si ottiene un’espressione di $H_k$ al passo $k$ in termini di una matrice iniziale $H_0$ e di soli $m$ passi precedenti:
```math
H_k \approx \Phi\!\left(H_0; \{(\Delta_i,\Gamma_i)\}_{i=k-m}^{k-1}\right),
```
dove $\Phi$ indica l’applicazione sequenziale degli aggiornamenti BFGS limitati alle ultime $m$ coppie.
[03:20] Operativamente, si eseguono passaggi che percorrono la sequenza delle coppie dall’esterno all’interno, aggiungendo contributi di rango 1. Questo consente un aggiornamento computazionalmente leggero e “matrix-free”.
[03:40] Il costo computazionale per iterazione diventa dell’ordine di $m\,n$:
```math
\text{Costo computazionale} \approx m\,n,
```
poiché si applicano operazioni con $m$ coppie di vettori di dimensione $n$.
[04:00] Se $n\approx 10^6$ e $m=10$, invece di $n^2\approx 10^{12}$ operazioni, il costo diventa circa $10^7$ operazioni:
```math
n^2 \approx 10^{12} \quad \text{vs} \quad m\,n \approx 10 \times 10^6 = 10^7.
```
Questa riduzione rende L-BFGS praticabile su larga scala.
[04:20] Anche il costo di memorizzazione si riduce drasticamente, passando da $O(n^2)$ a $O(m\,n)$:
```math
\text{Memoria (L-BFGS)} \approx m\,n,
```
poiché si conservano solo $m$ coppie di vettori di lunghezza $n$.
## Idea chiave dell’aggiornamento limitato
[04:40] L-BFGS nasce per risolvere il problema di memoria di BFGS standard. Per aggiornare l’approssimazione dell’inversa dell’Hessiana, non è necessario considerare tutte le coppie accumulate; bastano le ultime $m$ coppie recenti, che catturano l’informazione più rilevante.
[05:00] Per costruire la direzione di discesa o il prodotto con $H_k$, si selezionano le ultime $m$ coppie, limitando memoria e costo.
[05:20] Nella versione limitata, si applicano cicli di aggiornamento che percorrono la sequenza dall’esterno all’interno, includendo aggiornamenti di rango 1. Così si evita la formazione esplicita della matrice e si ottiene l’effetto di $H_k$ su un vettore mediante operazioni vettoriali.
## Confronto tra BFGS e L-BFGS
[05:40] In termini di memoria:
```math
\text{Memoria (BFGS)} \approx n^2, \quad \text{Memoria (L-BFGS)} \approx m\,n,
```
con $m\ll n$. Il costo per iterazione segue la stessa relazione:
```math
\text{Costo per iterazione (BFGS)} \approx n^2, \quad \text{Costo per iterazione (L-BFGS)} \approx m\,n.
```
[06:00] Per la convergenza, BFGS ottiene superlinearità, mentre L-BFGS ha convergenza lineare ma veloce rispetto alla discesa del gradiente, che è lineare e spesso lenta.
[06:20] In sintesi: BFGS è adatto a problemi piccoli per prototipazione e verifica rapida; L-BFGS è preferibile su problemi di grande scala ($n\gtrsim 1000$) per evitare limiti di memoria e ridurre i tempi.
[06:40] L’uso dipende dalla dimensione del problema e dalle risorse disponibili. L-BFGS riduce la memorizzazione da una matrice $n\times n$ a un insieme di $m$ coppie di vettori, preservando buone proprietà di convergenza e grande efficienza.
## Considerazioni pratiche e conclusione
[07:00] La scelta tra BFGS e L-BFGS dipende da $n$ e dalle risorse. Limitare la memoria a $m\,n$ mantiene l’algoritmo praticabile anche con $n$ molto grande. Pur rinunciando alla superlinearità di BFGS, L-BFGS raggiunge una convergenza lineare molto veloce rispetto al gradiente standard.
[07:20] Fattori determinanti:
- La dimensione $n$ influenza drasticamente sia il costo di memoria sia il costo computazionale dei metodi che manipolano matrici dense.
- La capacità di limitare la memoria mantiene l’algoritmo praticabile su scale elevate.
- La qualità della convergenza: BFGS per piccoli problemi; L-BFGS per grande scala.
[07:40] Per prototipazione su piccola scala, BFGS offre prestazioni elevate. Per grande scala, L-BFGS è la scelta consigliabile per evitare limiti di memoria e ridurre i costi computazionali.
[08:00] L’adozione della versione a memoria limitata consente di affrontare problemi altrimenti proibitivi quando la memorizzazione di una matrice $n\times n$ non è possibile. L’uso di $m$ coppie $(\Delta,\Gamma)$ recenti è un compromesso efficace tra memoria, costo e velocità di convergenza.
[08:20] Si interrompe l’esposizione avendo delineato gli aspetti fondamentali: criticità della memorizzazione in BFGS, idea di L-BFGS come metodo “matrix-free”, riduzione dei costi da $n^2$ a $m\,n$, e confronto sulle proprietà di convergenza e sugli ambiti di utilizzo pratico.