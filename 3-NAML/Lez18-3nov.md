## Introduzione: problemi di costo e tecniche di apprendimento
[00:00] Nella seconda parte dell’introduzione alle reti neurali vengono affrontati due temi principali. Il primo riguarda la scelta della funzione di costo: nel problema di classificazione è possibile modificare la funzione di costo per ridurre i problemi legati al vanishing gradient. Il secondo tema concerne tecniche che migliorano la qualità dell’apprendimento delle reti neurali, intervenendo sia sulla struttura del modello sia sui meccanismi di ottimizzazione.
[00:20] Si richiamano le funzioni di costo classiche. Si assume una funzione di costo quadratica, che misura la differenza tra l’uscita predetta e l’etichetta vera mediante l’errore al quadrato. Questa scelta è tradizionale in vari contesti, ma presenta limiti in presenza di funzioni di attivazione con regioni piatte.
[00:40] Si calcolano le derivate della funzione di costo rispetto ai pesi e ai bias della rete. Per l’ultimo strato, nelle formule del gradiente compare in modo cruciale la derivata della funzione di attivazione, indicata con $\sigma'(z)$. Quando $\sigma'(z)$ è piccola, l’attivazione risulta piatta e le derivate diventano piccole; il neurone contribuisce poco all’aggiornamento dei parametri.
[01:10] Anche se l’errore è grande, se $\sigma'(z)$ è piccola, l’aggiornamento resta quasi bloccato a causa della piattezza dell’attivazione. Questa condizione ostacola la capacità di apprendimento della rete proprio dove sarebbe più necessario. Il fenomeno è noto come vanishing gradient e rallenta o impedisce la convergenza verso soluzioni utili.
## Funzione di costo quadratica e vanishing gradient
[01:30] Si osserva graficamente il comportamento della funzione di costo quadratica rispetto all’uscita $a \in [0,1]$ quando il target è $y=1$. La funzione di costo è nulla quando $a=1$ e cresce man mano che $a$ si allontana da 1 verso 0, in accordo con l’idea di penalizzare errori maggiori.
[01:55] Considerando la derivata rispetto all’uscita e assumendo una sigmoide come funzione di attivazione, anche se si è lontani dalla soluzione vera, la derivata può essere prossima a zero. Questo causa problemi nel calcolo del gradiente perché l’aggiornamento dei parametri diventa trascurabile, rallentando l’apprendimento.
[02:15] Viene introdotta un’alternativa: la funzione di costo di tipo cross-entropy, rappresentata graficamente da curve rosse. Si rimanda il confronto dettagliato dopo aver derivato le proprietà analitiche, mostrando come la cross-entropy influenzi favorevolmente i gradienti.
## Cross-entropy: definizione e proprietà di base
[02:35] Per la classificazione binaria si definisce la funzione di costo di tipo cross-entropy:
```math
J \;=\; -\frac{1}{n}\sum_{i=1}^n \Big[\, y_i \log(a_i) + (1-y_i)\log(1-a_i) \,\Big]
```
dove:
- $n$ è il numero di campioni,
- $y_i \in \{0,1\}$ è l’etichetta vera del campione $i$,
- $a_i \in (0,1)$ è l’attivazione (uscita) dell’ultimo strato per il campione $i$.
[02:55] La funzione $J$ è non negativa perché $-\log(a)$ e $-\log(1-a)$ sono non negativi per $a \in (0,1)$ quando pesati con $y \in \{0,1\}$. Quando $a$ è vicino a $y$, $J$ tende a zero: se $y=1$ e $a \to 1$, oppure se $y=0$ e $a \to 0$, i termini logaritmici si annullano. Questa proprietà rispetta il requisito di una funzione di costo che si riduce quando l’uscita coincide con il target.
[03:20] In un problema binario con $y \in \{0,1\}$, le proprietà fondamentali sono: positività e annullamento sulla soluzione attesa, ossia $J$ diventa nullo quando l’uscita $a$ coincide con l’etichetta $y$.
## Derivata della cross-entropy rispetto ai pesi: catena del calcolo
[03:40] Si calcola la derivata di $J$ rispetto ai pesi $w$ dell’ultimo strato usando la regola della catena. Si introducono:
- $z = w^\top x + b$, input dell’attivazione,
- $a = \sigma(z)$, attivazione dell’ultimo strato.
[04:00] Per la derivata si scrive:
```math
\frac{\partial J}{\partial w} \;=\; \frac{\partial J}{\partial a}\,\frac{\partial a}{\partial z}\,\frac{\partial z}{\partial w}
```
che decompone il gradiente nei fattori elementari secondo la catena.
[04:15] Il termine più semplice è:
```math
\frac{\partial z}{\partial w} \;=\; x
```
poiché $z = w^\top x + b$ è lineare in $w$; la derivata rispetto a $w$ coincide con l’input $x$.
[04:25] La derivata di $J$ rispetto ad $a$ è:
```math
\frac{\partial J}{\partial a} \;=\; -\frac{1}{n}\Big(\frac{y}{a} - \frac{1-y}{1-a}\Big)
```
trascurando il fattore $1/n$ per un ragionamento locale su un singolo campione. Riorganizzando algebricamente:
```math
\frac{\partial J}{\partial a} \;=\; \frac{a - y}{a(1-a)}
```
che unifica i termini razionali in una forma compatta.
[04:55] La derivata di $a$ rispetto a $z$ per la sigmoide $\sigma(z)=\frac{1}{1+e^{-z}}$ è:
```math
\sigma'(z) \;=\; \sigma(z)\,(1 - \sigma(z)) \;=\; a(1-a)
```
cioè la derivata della sigmoide è il prodotto tra l’attivazione e uno meno l’attivazione.
[05:20] Componendo i tre fattori:
```math
\frac{\partial J}{\partial w} \;=\; \Big(\frac{a - y}{a(1-a)}\Big)\,\big(a(1-a)\big)\,x \;=\; (a-y)\,x
```
si osserva la semplificazione di $a(1-a)$. La derivata rispetto al bias $b$ è:
```math
\frac{\partial J}{\partial b} \;=\; (a - y)
```
poiché $\frac{\partial z}{\partial b}=1$ e gli altri fattori restano invariati.
[05:45] Conseguenza cruciale: $\sigma'(z)$ scompare dall’espressione finale del gradiente per l’ultimo strato. Adottando la cross-entropy con sigmoide, la sensibilità dell’ultimo strato è proporzionale all’errore $(a-y)$, senza il fattore attenuante $\sigma'(z)$ che causa vanishing gradient in regime di saturazione della sigmoide.
## Implicazioni pratiche: evitare il vanishing gradient
[06:10] L’assenza di $\sigma'(z)$ nel gradiente dell’ultimo strato rende l’aggiornamento sensibile all’errore anche in regime saturo. Se l’errore è grande, il gradiente rimane grande e la rete può apprendere efficacemente, evitando blocchi dovuti a derivate troppo piccole.
[06:25] La scelta della funzione di costo, in relazione al tipo di problema e alla funzione di attivazione, influenza la velocità e la qualità dell’apprendimento. Non esiste una funzione di costo universalmente ottimale; la selezione dipende dall’applicazione e può richiedere tentativi ed errori per bilanciare stabilità e efficienza.
## Overfitting: definizione e manifestazioni
[06:45] L’overfitting si verifica quando il modello si adatta eccessivamente ai dati di training perdendo capacità di generalizzazione su dati nuovi. In termini intuitivi, invece di cogliere una relazione semplice attesa, il modello apprende un andamento troppo complesso che descrive fedelmente le fluttuazioni del training, risultando fuorviante sulle nuove osservazioni.
[07:10] La stima su input non visti può discostarsi dal comportamento atteso. L’overfitting limita la generalizzazione del modello, ossia la sua applicabilità a esempi non utilizzati durante l’addestramento, riducendone l’utilità pratica.
## Rilevazione dell’overfitting tramite training e validation
[07:30] Nella pratica si monitorano i valori della funzione di costo per epoca sia sul set di training sia su un set di validazione. Tipicamente il dataset è diviso in due sottoinsiemi: circa 70–80% per il training e 20–30% per la validazione, in modo da valutare il comportamento su dati non visti.
[07:50] A ogni passo dell’addestramento si calcola la funzione di costo sul training set e, con gli stessi parametri correnti, si valuta la funzione di costo sul validation set. Un andamento tipico mostra che:
- la curva di training decresce progressivamente;
- la curva di validazione decresce inizialmente, raggiunge un minimo e poi ricomincia a crescere.
[08:15] Dal punto in cui la curva di validazione inizia a crescere (ad esempio attorno alla 115-esima epoca), si assume l’insorgere dell’overfitting: la capacità di generalizzazione peggiora nonostante il miglioramento sul training. Da quel momento, la prosecuzione dell’addestramento può deteriorare le prestazioni su dati nuovi.
[08:30] Una contromisura operativa è l’early stopping: si interrompe l’addestramento quando la curva di validazione mostra un’inversione verso l’aumento, evitando di proseguire nel regime di overfitting e preservando la generalizzazione.
## Strategie per mitigare l’overfitting
[08:50] Le misure principali includono:
- aumentare i dati di training, quando possibile;
- utilizzare tecniche di data augmentation o generazione di dati sintetici coerenti con il dataset originale;
- applicare regolarizzazione alla funzione di costo.
[09:10] La regolarizzazione aggiunge un termine di penalità alla funzione di costo per scoraggiare soluzioni con parametri eccessivamente grandi o complessi, riducendo la tendenza all’overfitting. Si considerano due approcci classici: L2 e L1, che agiscono in modo diverso sulla geometria delle soluzioni.
## Regolarizzazione L2 (weight decay): formulazione e effetto
[09:30] La regolarizzazione L2 aggiunge una penalità proporzionale al quadrato della norma dei pesi:
```math
C \;=\; J + \frac{\lambda}{2n}\,\|w\|_2^2
```
dove:
- $J$ è la funzione di costo originale,
- $\lambda>0$ è il parametro di regolarizzazione,
- $n$ è il numero di campioni,
- $\|w\|_2^2 = \sum_j w_j^2$ è la somma dei quadrati dei pesi.
[09:55] In genere i bias non vengono regolarizzati, poiché incidono meno sulla complessità del modello in ottica di overfitting. La penalità agisce sui pesi rendendo sfavorevoli soluzioni con norma elevata, orientando la ricerca verso parametri più contenuti.
[10:10] Considerando l’aggiornamento per discesa del gradiente:
```math
w^{(k+1)} \;=\; w^{(k)} - \eta\,\nabla C\big(w^{(k)}\big)
```
con $\eta>0$ learning rate, il gradiente della penalità L2 è:
```math
\nabla_w \Big(\frac{\lambda}{2n}\|w\|_2^2\Big) \;=\; \frac{\lambda}{n}\,w
```
che aggiunge un contributo proporzionale ai pesi correnti.
[10:30] Raccolti i termini, l’aggiornamento diventa:
```math
w^{(k+1)} \;=\; \big(1 - \eta\,\tfrac{\lambda}{n}\big)\,w^{(k)} - \eta\,\nabla J\big(w^{(k)}\big)
```
dove $\eta\,\tfrac{\lambda}{n}$ è positivo e tipicamente piccolo. Il fattore $(1 - \eta\,\tfrac{\lambda}{n})$ è minore di 1 e produce uno “shrinking” dei pesi a ogni iterazione.
[10:55] Questo effetto giustifica il nome weight decay: i pesi si riducono progressivamente in magnitudine, evitando soluzioni con valori ampi. L’interpretazione è coerente con la ricerca di un vettore dei pesi a norma minima compatibile con i dati.
## Regolarizzazione L1: formulazione e effetto di sparsità
[11:15] La regolarizzazione L1 aggiunge una penalità proporzionale alla norma L1:
```math
C \;=\; J + \frac{\lambda}{n}\,\|w\|_1
```
dove $\|w\|_1 = \sum_j |w_j|$ è la somma dei valori assoluti dei pesi.
[11:30] L’aggiornamento per discesa del gradiente incorpora la penalità L1 con un termine che spinge i pesi verso zero in modo non uniforme, favorendo la sparsità:
```math
w^{(k+1)} \;\approx\; w^{(k)} - \eta\,\nabla J\big(w^{(k)}\big) - \eta\,\frac{\lambda}{n}\,\mathrm{sgn}\big(w^{(k)}\big)
```
dove $\mathrm{sgn}(w_j)$ è il segno di $w_j$; in $w_j=0$ si adotta un subgradiente. Questo meccanismo induce molti pesi a diventare esattamente nulli.
[11:55] Rispetto all’L2, l’L1 tende a selezionare un sottoinsieme di parametri rilevanti e ad annullare gli altri, portando a soluzioni più parsimoniose e con vettori dei pesi più sparsi.
## Interpretazione geometrica: palle unitarie L2 vs L1
[12:15] Si considerano le curve di livello della funzione di costo nello spazio dei pesi e si confrontano le palle unitarie in $\mathbb{R}^2$:
- palla unitaria L2 (cerchio),
- palla unitaria L1 (rombo).
[12:30] La geometria della penalità influisce sul punto di contatto con le curve di livello:
- L2 privilegia soluzioni con pesi distribuiti e magnitudini contenute;
- L1 privilegia soluzioni ai vertici del rombo, dove uno o più pesi sono esattamente nulli, favorendo la sparsità.
## Riepilogo della scelta della funzione di costo e regolarizzazione
[12:50] La scelta della funzione di costo, in relazione alla funzione di attivazione e al tipo di problema, è determinante per evitare il vanishing gradient e migliorare l’apprendimento. Per classificazione con sigmoide, la cross-entropy produce gradienti dell’ultimo strato del tipo $(a-y)$, eliminando il fattore $\sigma'(z)$ che può annullarsi.
[13:05] L’overfitting si individua tramite l’andamento divergente tra training e validation loss, ed è mitigabile con early stopping, incremento dei dati (o data augmentation) e regolarizzazione. L2 induce weight decay e controlla la norma dei pesi; L1 promuove la sparsità selezionando i parametri più rilevanti.
[13:20] In contesti reali non esiste una ricetta universale; spesso serve una combinazione di scelte mirate e sperimentazione per ottimizzare velocità e qualità dell’apprendimento, mantenendo la capacità di generalizzazione del modello.
## Regolarizzazione con vincoli L1 e L2 – Problema vincolato e sparsità
[00:00] L’obiettivo è trasformare l’ottimizzazione in un problema vincolato: si minimizza la funzione di costo imponendo che la soluzione appartenga alla palla unitaria in norma L2 oppure alla palla unitaria in norma L1. Il vincolo definisce l’insieme ammissibile delle soluzioni entro una regione geometrica specifica.
[00:20] Per la palla unitaria L1, l’intersezione tra le curve di livello minime della funzione di costo e la regione vincolata tende a collocarsi in uno degli spigoli del “quadrato” (rombo) che rappresenta la palla L1. Questo induce sparsità, ossia molte componenti della soluzione diventano esattamente nulle. Per la palla unitaria L2, l’intersezione può trovarsi in qualunque punto del cerchio e non favorisce direttamente soluzioni sparse.
[00:45] Il vincolo L1 promuove soluzioni con pochi coefficienti non nulli, mentre il vincolo L2 distribuisce la penalizzazione in modo uniforme su tutte le componenti, mantenendo la soluzione più “densa”. La differenza geometrica tra palla L1 (poliedrica, con spigoli) e palla L2 (sferica) guida la diversa natura delle soluzioni.
## Dropout – Modifica strutturale della rete e media d’insieme
[01:10] Il dropout è una tecnica pratica che introduce una forma di regolarizzazione modificando la rete neurale durante l’addestramento. Invece di intervenire sulla funzione di costo con penalizzazioni L1 o L2, si altera la struttura della rete in modo stocastico.
[01:30] L’operazione consiste nel considerare tutti gli strati nascosti e annullare casualmente l’uscita di una percentuale di neuroni in ciascuno strato. Con probabilità $P=0{.}5$, si annulla l’uscita della metà dei neuroni dello strato. Si eseguono i passi di forward e backward e si aggiornano pesi e bias, ripetendo per tutte le iterazioni previste.
[01:55] Al termine dell’addestramento, in valutazione si utilizza la rete completa senza dropout, ma le uscite degli strati nascosti sono moltiplicate per la probabilità usata in training. L’idea è assimilabile a una media su un insieme di modelli: si addestra come se si avessero molte reti diverse, ottenute disattivando neuroni con probabilità $P$, e in valutazione si usa una rete “media”.
[02:20] Questa strategia è concettualmente simile a un ensemble: il training coinvolge molte configurazioni (topologie) dovute a neuroni disattivati, e l’uscita finale corrisponde a una media rispetto a tutte le configurazioni stocastiche. In librerie come TensorFlow si introducono layer di dropout che specificano la probabilità con cui uno strato nascosto è soggetto a dropout.
[02:45] Il modello finale riflette un ensemble implicito ottenuto tramite campionamento casuale di neuroni. Questa modifica strutturale produce una regolarizzazione che riduce l’overfitting, impedendo co-adattamenti eccessivi tra neuroni e migliorando la generalizzazione.
## Inizializzazione dei pesi – Distribuzione, varianza e gradiente vaniscente
[03:10] L’inizializzazione dei parametri è cruciale, soprattutto per i pesi. Si consideri un’inizializzazione gaussiana con media zero e deviazione standard pari a uno per pesi e bias, assumendo input normalizzati e un neurone con $n$ ingressi e relativo bias.
[03:35] La variabile $z$ è la somma pesata degli input più il bias. Se i pesi sono inizializzati in modo indipendente e gaussiano, $z$ risulta approssimativamente gaussiana come somma di variabili indipendenti. La deviazione standard di $z$ è proporzionale alla radice del numero di ingressi del neurone, scalata dalla deviazione standard dei pesi:
```math
\sigma_z \;\approx\; \sqrt{n}\,\sigma_w
```
dove $n$ è il numero di ingressi e $\sigma_w$ la deviazione standard dei pesi iniziali.
[04:05] Se $n$ è grande (ad esempio dell’ordine di mille), $\sigma_z$ risulta elevata (circa 30 con $\sigma_w=1$) e $z$ può assumere valori molto grandi, positivi o negativi. Questo porta la sigmoide nella regione di saturazione, dove la derivata dell’attivazione è quasi nulla e l’aggiornamento dei pesi diventa molto lento: si verifica il gradiente vaniscente.
[04:35] Se la distribuzione iniziale dei pesi induce una $z$ con varianza troppo alta, l’apprendimento parte con velocità molto bassa. Per mitigare il problema si modifica la deviazione standard dei pesi iniziali rendendola inversamente proporzionale al numero di ingressi del neurone, ridimensionando la varianza di $z$.
[04:55] Si sceglie una distribuzione gaussiana a media zero per i pesi, con deviazione standard scalata in funzione del numero di input. Ripetendo il calcolo, si ottiene che la deviazione standard di $z$ può essere mantenuta circa unitaria:
```math
\sigma_z \;\approx\; 1
```
così $z$ tende a rimanere in una regione della funzione di attivazione non satura, attenuando il rischio di gradiente vaniscente.
[05:20] Le librerie standard (TensorFlow, PyTorch, scikit-learn) offrono metodi di inizializzazione dei pesi che fissano la varianza iniziale in modo coerente con il numero di input, riducendo la probabilità di saturazione precoce.
## Iperparametri – Definizione e tuning operativo
[05:45] Gli iperparametri sono parametri non ottimizzati durante l’addestramento e scelti a priori. Tra i principali: learning rate (spesso indicato con $\eta$ o $\gamma$), parametro di regolarizzazione $\lambda$, dimensione del mini-batch, numero di epoche e architettura della rete (numero di layer e neuroni per layer). Anche la scelta della funzione di attivazione influisce sensibilmente.
[06:10] Non esiste una ricetta generale che garantisca valori ottimali; la selezione dipende dal problema e si effettua empiricamente. È utile una strategia progressiva per individuare combinazioni di iperparametri che garantiscano stabilità e buona convergenza della funzione di costo.
[06:30] Strategia tipica:
- disattivare ogni regolarizzazione per scegliere un learning rate iniziale che produca una decrescita della funzione di costo (ad esempio partire da $0{.}01$);
- monitorare l’andamento della funzione di costo durante l’addestramento: se aumenta, ridurre il learning rate; se diminuisce troppo lentamente, valutare un incremento moderato;
- tarato il learning rate, introdurre la regolarizzazione e scegliere $\lambda$ partendo da $\lambda = 1$, esplorando altri valori su scala logaritmica, osservando la funzione di costo di validazione;
- se la curva di validazione inizia ad aumentare, si è probabilmente in una zona di $\lambda$ adeguata per controllare l’overfitting, e si può ritoccare $\eta$ con la regolarizzazione attiva;
- scegliere la dimensione del mini-batch, che impatta sulla stabilità e sulla varianza dell’aggiornamento.
[07:05] Il learning rate è tra gli iperparametri più critici:
- se $\eta$ è troppo alto, la funzione di costo può oscillare o divergere;
- se $\eta$ è troppo basso, l’addestramento procede lentamente;
- esiste un intervallo “buono” di $\eta$ che garantisce decrescita regolare della funzione di costo a velocità ragionevole.
[07:30] È fondamentale monitorare l’accuratezza e soprattutto la funzione di costo in validazione. Un pattern qualitativo:
- in rosso: $\eta$ troppo alto, iniziale discesa seguita da oscillazioni e divergenza;
- in blu: $\eta$ moderato, senza divergenza ma con convergenza molto lenta;
- in verde: scelta ottimale, decrescita regolare e apprendimento adeguato.
## Riepilogo operativo – Funzione di costo, regolarizzazione e inizializzazione
[07:55] La cross-entropy è una funzione di costo efficace se combinata con regolarizzazione per favorire la generalizzazione. Le scelte di inizializzazione influenzano la dinamica del gradiente e la probabilità di saturazione delle attivazioni. Le librerie standard offrono metodi consolidati di inizializzazione tra cui selezionare.
[08:15] La dimostrazione dei risultati sulla varianza di $z$ con diverse inizializzazioni è lasciata come esercizio. L’intuizione operativa è che una corretta scalatura della deviazione standard dei pesi in funzione del numero di ingressi riduce significativamente i problemi di gradiente vaniscente nella fase iniziale dell’addestramento.
## Discesa del gradiente – Problema di minimizzazione e metodo iterativo
[08:35] Il compito centrale è minimizzare la funzione di costo per determinare il miglior insieme di pesi e bias. La discesa del gradiente è un metodo iterativo che costituisce la base della discesa del gradiente stocastica, ampiamente impiegata nelle reti neurali.
[08:55] Definizioni utili:
- Convessità: una funzione $f$ è convessa se per qualunque $\lambda \in [0,1]$ vale
```math
f(\lambda x + (1-\lambda) y) \;\le\; \lambda f(x) + (1-\lambda) f(y).
```
La funzione sta sotto la corda tra due punti del suo grafico.
[09:20] Caratterizzazione del primo ordine (funzione differenziabile): la funzione giace sopra la propria tangente in ogni punto. Espressa tramite il gradiente:
```math
f(y) \;\ge\; f(x) + \nabla f(x)^\top (y - x).
```
Il piano tangente in $x$ fornisce un sottostimatore locale della funzione.
[09:40] Proprietà B-Lipschitz: $f$ è B-Lipschitz se
```math
|f(x) - f(y)| \;\le\; B \,\|x - y\|.
```
Se $f$ è differenziabile, ciò implica un vincolo sulla grandezza del gradiente; la funzione non varia più rapidamente di un limite proporzionale alla distanza.
[10:00] L-smoothness (gradiente Lipschitz): la variazione del gradiente è limitata da $L$:
```math
\|\nabla f(x) - \nabla f(y)\| \;\le\; L \,\|x - y\|.
```
Questa condizione limita la curvatura: il gradiente non cambia più velocemente di una quantità proporzionale alla distanza.
[10:20] Equivalenza a maggiorazione quadratica: l’L-smoothness è equivalente al fatto che $f$ ammette un limite superiore quadratico intorno a ogni punto:
```math
f(y) \;\le\; f(x) + \nabla f(x)^\top (y - x) + \frac{L}{2}\,\|y - x\|^2.
```
La funzione è al di sotto di una parabola che approssima localmente $f$ usando il gradiente in $x$ e un termine quadratico con coefficiente $L/2$.
[10:45] Forte convessità con parametro $\mu$: fornisce un limite inferiore quadratico:
```math
f(y) \;\ge\; f(x) + \nabla f(x)^\top (y - x) + \frac{\mu}{2}\,\|y - x\|^2.
```
La funzione cresce almeno quanto una parabola con curvatura $\mu/2$, garantendo unicità del minimo e proprietà di convergenza più forti.
[11:10] Sintesi grafica: con L-smoothness si ottiene una parabola superiore che maggiora la funzione; con forte convessità si ottiene una parabola inferiore che la minora. La funzione reale resta confinata nella regione compresa tra queste due approssimazioni quadratiche.
## Problema di ottimizzazione – Definizione e metodo di aggiornamento
[11:30] Si considera la minimizzazione di $f(x)$ su $\mathbb{R}^d$ senza vincoli, con $x$ che rappresenta il vettore dei parametri. La discesa del gradiente genera una sequenza $x_0, x_1, x_2, \dots$ tramite una regola di aggiornamento.
[11:50] Partendo da una stima iniziale $x_0$, si definisce l’iterazione:
```math
x_{t+1} \;=\; x_t + d_t
```
dove $d_t$ è il vettore di spostamento che indica direzione e ampiezza della modifica alla soluzione corrente.
[12:10] L’obiettivo è ottenere $f(x_{t+1}) < f(x_t)$. Usando uno sviluppo di Taylor al primo ordine:
```math
f(x_t + d_t) \;\approx\; f(x_t) + \nabla f(x_t)^\top d_t,
```
per avere una riduzione è necessario che
```math
\nabla f(x_t)^\top d_t \;<\; 0.
```
[12:30] La direzione che massimizza la diminuzione al primo ordine è opposta al gradiente:
```math
d_t \;=\; - \gamma \,\nabla f(x_t),
```
con $\gamma > 0$ step size o learning rate. L’aggiornamento diventa:
```math
x_{t+1} \;=\; x_t - \gamma \,\nabla f(x_t).
```
[12:55] La regola garantisce, per $\gamma$ sufficientemente piccolo e sotto ipotesi di regolarità, che $f(x_{t+1}) < f(x_t)$. La scelta di $\gamma$ determina velocità e stabilità: valori troppo grandi possono causare divergenza, valori troppo piccoli rendono la convergenza lenta.
[13:15] La scelta della direzione del gradiente e della lunghezza del passo richiama metodi classici di ottimizzazione; si cerca un $\gamma$ che riduca $f$ lungo la direzione selezionata in modo efficace, bilanciando rapidità e monotonia della convergenza.
[13:35] Rappresentazione qualitativa: si può immaginare la funzione in blu e gli spostamenti successivi come segmenti rossi che descrivono una sequenza di riduzioni della funzione di costo, indicando visivamente l’efficacia della discesa lungo il gradiente.
## Comportamento dell’algoritmo di discesa del gradiente su paraboloidi
[00:00] Si consideri un paraboloide e i suoi insiemi di livello. Si parte da un punto iniziale con dimensione del passo impostata a $0{.}2$. Si osserva il percorso seguito fino al minimo della funzione, evidenziando la relazione tra forma della funzione e dinamica dell’algoritmo.
[00:20] Modificando la posizione iniziale o la dimensione del passo, il comportamento del metodo cambia. Se il passo diventa troppo grande, la convergenza non è più monotona: compaiono oscillazioni. Il metodo è sensibile alla forma della funzione di costo.
[00:40] Nel primo caso, la funzione è un paraboloide con assi quasi uguali: gli insiemi di livello sono ellissi non troppo schiacciate. Quando gli insiemi di livello sono molto schiacciati, il paraboloide presenta assi con magnitudini molto diverse, generando anisotropia.
[01:00] In presenza di anisotropia, l’affidabilità del metodo cambia nettamente. Con passo circa $0{.}2$ si ottiene convergenza monotona nel caso quasi isotropo; nel caso molto anisotropo, si deve usare un passo di un ordine di grandezza più piccolo per ottenere un comportamento accettabile. Aumentando il passo, la situazione peggiora sensibilmente.
[01:20] Il passo che garantiva convergenza monotona nel caso isotropo genera ora un decadimento fortemente oscillante e persino instabile. In una situazione non convessa rappresentata pittoricamente, per evitare oscillazioni si deve utilizzare un passo molto piccolo, ad esempio $0{.}02$; se si imposta anche solo $0{.}05$, il comportamento può diventare sfavorevole, soprattutto all’inizio.
## Visualizzazione semplificata su una parabola unidimensionale
[02:00] Si consideri una parabola e si parta da un punto iniziale con l’obiettivo di raggiungere il minimo. Si analizzano quattro diversi valori del parametro di apprendimento $\gamma$ per evidenziare effetti sulla convergenza.
[02:15] Con il tracciato blu, il comportamento è regolare ma lento: anche dopo 10 passi, il punto è lontano dal minimo. Con il tracciato rosso, $\gamma$ è troppo grande: il metodo converge ma con oscillazioni, alternando da un lato all’altro della parabola.
[02:30] Con il tracciato magenta si ottiene divergenza. Con il tracciato verde si ha un buon compromesso tra velocità e monotonicità della convergenza. Questa visualizzazione intuitiva aiuta a comprendere le diverse dinamiche variando $\gamma$.
## Regola di aggiornamento di base e obiettivo dell’analisi
[03:00] Si considera la regola di aggiornamento di base dell’algoritmo per analizzarne le prestazioni in termini di convergenza. L’analisi di complessità mira a trovare un limite superiore per una quantità d’interesse, dove $x_t$ è la soluzione all’iterazione $t$ e $x^\star$ è la soluzione ottima.
[03:20] Si vuole limitare la differenza tra il valore della funzione nel minimo reale e il valore della funzione nel punto calcolato. Si utilizza la caratterizzazione di primo ordine della convessità per scrivere una relazione utile che leghi subottimalità e gradiente al punto corrente.
[03:35] La caratterizzazione di primo ordine della convessità afferma:
```math
f(y) \;\ge\; f(x) + \nabla f(x)^\top (y - x).
```
Sostituendo $x = x_t$ e $y = x^\star$, si ottiene:
```math
f(x^\star) \;\ge\; f(x_t) + \nabla f(x_t)^\top (x^\star - x_t),
```
equivalente a:
```math
f(x_t) - f(x^\star) \;\le\; \nabla f(x_t)^\top (x_t - x^\star).
```
Si indica $g_t = \nabla f(x_t)$.
## Quantità da limitare e uso della regola di aggiornamento
[04:20] La quantità da limitare è $f(x_t) - f(x^\star)$, che è minore o uguale a $g_t^\top (x_t - x^\star)$. Si lavora su $g_t^\top (x_t - x^\star)$ sfruttando la regola di aggiornamento:
```math
x_{t+1} \;=\; x_t - \gamma \nabla f(x_t).
```
[04:40] Dalla regola di aggiornamento si ricava il gradiente:
```math
\nabla f(x_t) \;=\; \frac{x_t - x_{t+1}}{\gamma}.
```
Sostituendo in $g_t^\top (x_t - x^\star)$:
```math
g_t^\top (x_t - x^\star) \;=\; \frac{1}{\gamma} (x_t - x_{t+1})^\top (x_t - x^\star).
```
## Identità algebrica sui prodotti scalari e sulle norme
[05:10] Si utilizza l’identità, per due vettori $b$ e $w$:
```math
2\, b^\top w \;=\; \|b\|^2 + \|w\|^2 - \|b - w\|^2,
```
che deriva da $\|b - w\|^2 = \|b\|^2 + \|w\|^2 - 2 b^\top w$.
[05:25] Si pone $b = x_t - x^\star$ e $w = x_t - x_{t+1}$. Allora:
```math
2 (x_t - x^\star)^\top (x_t - x_{t+1}) \;=\; \|x_t - x^\star\|^2 + \|x_t - x_{t+1}\|^2 - \|(x_t - x^\star) - (x_t - x_{t+1})\|^2.
```
[05:45] La differenza $(x_t - x^\star) - (x_t - x_{t+1})$ semplifica in $x_{t+1} - x^\star$:
```math
(x_t - x^\star) - (x_t - x_{t+1}) \;=\; x_{t+1} - x^\star.
```
Quindi:
```math
2 (x_t - x^\star)^\top (x_t - x_{t+1}) \;=\; \|x_t - x^\star\|^2 + \|x_t - x_{t+1}\|^2 - \|x_{t+1} - x^\star\|^2.
```
[06:10] Dividendo per $2\gamma$:
```math
\frac{1}{\gamma} (x_t - x_{t+1})^\top (x_t - x^\star) \;=\; \frac{1}{2\gamma}\Big( \|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2 + \|x_t - x_{t+1}\|^2 \Big).
```
Si ottiene così un’espressione del termine $g_t^\top (x_t - x^\star)$ in funzione di differenze di norme tra iterati consecutivi.
## Somma telescopica sugli iterati
[06:40] Sommando sui passi da $t = 0$ fino a $T$:
```math
\sum_{t=0}^{T} g_t^\top (x_t - x^\star) \;=\; \sum_{t=0}^{T} \frac{1}{2\gamma}\Big( \|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2 + \|x_t - x_{t+1}\|^2 \Big).
```
[06:55] La somma delle differenze $\|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2$ è telescopica: si cancellano tutti i termini intermedi e rimangono solo il primo e l’ultimo:
```math
\sum_{t=0}^{T} \Big( \|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2 \Big)
\;=\; \|x_0 - x^\star\|^2 - \|x_{T+1} - x^\star\|^2.
```
[07:15] Poiché le norme sono non negative, si introduce una disuguaglianza verso l’alto eliminando il termine $-\|x_{T+1} - x^\star\|^2$:
```math
\sum_{t=0}^{T} \Big( \|x_t - x^\star\|^2 - \|x_{t+1} - x^\star\|^2 \Big)
\;\le\; \|x_0 - x^\star\|^2.
```
[07:30] Pertanto:
```math
\sum_{t=0}^{T} g_t^\top (x_t - x^\star)
\;\le\; \frac{1}{2\gamma}\|x_0 - x^\star\|^2 + \frac{1}{2\gamma}\sum_{t=0}^{T} \|x_t - x_{t+1}\|^2.
```
Questo limite superiore dipende dalla distanza iniziale dal minimo vero e dai movimenti tra iterazioni.
## Dalla convessità al limite sulla subottimalità
[08:00] Dalla convessità:
```math
f(x_t) - f(x^\star) \;\le\; g_t^\top (x_t - x^\star).
```
Sommando per $t = 0, \dots, T$:
```math
\sum_{t=0}^{T} \big(f(x_t) - f(x^\star)\big)
\;\le\; \sum_{t=0}^{T} g_t^\top (x_t - x^\star).
```
[08:20] Combinando con il risultato precedente:
```math
\sum_{t=0}^{T} \big(f(x_t) - f(x^\star)\big)
\;\le\; \frac{1}{2\gamma}\|x_0 - x^\star\|^2 + \frac{1}{2\gamma}\sum_{t=0}^{T} \|x_t - x_{t+1}\|^2.
```
Il limite esprime la somma delle subottimalità in funzione del passo $\gamma$, della distanza iniziale dal minimo reale e delle differenze tra iterati.
[08:40] Il primo termine dipende dalla scelta del punto iniziale e del passo; il secondo raccoglie i contributi dei gradienti durante le iterazioni, poiché $\|x_t - x_{t+1}\| = \gamma \|\nabla f(x_t)\|$.
## Ipotesi utilizzate e prospettive di affinamento
[09:00] Il risultato si basa unicamente sull’assunzione che $f$ sia convessa. Non sono state aggiunte ulteriori ipotesi. Questo costituisce un risultato di base su cui si possono costruire analisi più raffinate per diverse categorie di funzioni.
[09:15] A partire da questa relazione, si possono ottenere risultati più specifici per funzioni convesso-lipschitziane e per funzioni fortemente convesse, sviluppando limiti più stretti e tassi di convergenza più informativi.
## Considerazioni finali
[09:30] I risultati ricavati consentono di discutere casi particolari per specifiche classi di funzioni. La combinazione tra scelta della funzione di costo, tecniche di regolarizzazione (L1, L2, dropout), inizializzazione dei pesi e tuning degli iperparametri orienta l’apprendimento verso soluzioni stabili e generalizzabili, riducendo fenomeni come il vanishing gradient e l’overfitting.