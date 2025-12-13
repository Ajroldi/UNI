# Capitolo 1: Limiti della Discesa Stocastica del Gradiente e Nuove Sfide
## Introduzione: Oltre la Discesa Stocastica del Gradiente
[00:00] In questa sezione, si analizzeranno i motivi per cui è necessario migliorare l'algoritmo della discesa stocastica del gradiente (SGD). L'esperienza pratica con le librerie di apprendimento automatico e apprendimento profondo dimostra che la scelta del tasso di apprendimento (*learning rate*), uno degli iperparametri più importanti, è di fondamentale importanza per le prestazioni dell'algoritmo.
[00:13] Il concetto chiave, già sottolineato in precedenza, riguarda l'impatto della scelta del tasso di apprendimento sulla convergenza del modello.
[00:21] Se il tasso di apprendimento è troppo piccolo, si ha una quasi certezza di evitare oscillazioni, ma al contempo la convergenza può risultare estremamente lenta. Al contrario, un tasso di apprendimento elevato può accelerare la convergenza, ma introduce il rischio di oscillazioni e, in alcuni casi, di divergenza dell'algoritmo.
[00:40] Una possibile soluzione a questo problema consiste nell'utilizzare una schedulazione del tasso di apprendimento, ovvero una strategia che prevede la sua diminuzione progressiva con l'aumentare del numero di iterazioni.
[00:50] Tuttavia, questo approccio presenta uno svantaggio significativo: la scelta del tasso di apprendimento rimane "cieca" rispetto alle caratteristiche specifiche del problema o del dataset in esame.
## La Sfida dei Dati Sparsi
[00:59] Un'altra questione rilevante riguarda la gestione dei dati sparsi, una condizione che si verifica in circa l'80% delle applicazioni pratiche.
[01:06] Un dataset è definito **sparso** quando molte delle sue caratteristiche (o *features*) sono nulle o assenti. Un esempio tipico è il sistema di valutazione dei film, in cui un utente potrebbe aver valutato solo un numero limitato di film (ad esempio, 20) tra le migliaia disponibili su una piattaforma.
[01:21] In questo contesto, alcune *features* sono più rare di altre. Dal punto di vista dei film, quelli più famosi ricevono un gran numero di valutazioni, mentre film più di nicchia ne ricevono pochissime.
[01:36] In presenza di un dataset sparso, è cruciale che il metodo di ottimizzazione sia in grado di gestire questa caratteristica. Il tasso di apprendimento non dovrebbe essere unico per tutti i parametri della rete.
[01:49] Esso dovrebbe essere adattato in modo specifico per ciascun parametro. In particolare, i parametri associati a *features* più rare dovrebbero avere un tasso di apprendimento più elevato, poiché vengono "attivati" solo in un numero limitato di casi.
## Complessità della Funzione di Costo
[02:02] La forma reale delle funzioni di costo utilizzate nella pratica è molto diversa da quella di una funzione convessa con le proprietà ideali discusse in precedenza. Anche in problemi a bassa dimensionalità, come nel caso bidimensionale, la visualizzazione dello spazio dei parametri rivela una funzione di costo complessa.
[02:16] Questa è tipicamente caratterizzata da un minimo globale, numerosi minimi locali e punti di sella.
- **Minimo Globale**: Il punto in cui la funzione di costo raggiunge il suo valore più basso in assoluto.
- **Minimo Locale**: Un punto in cui la funzione di costo ha un valore più basso rispetto ai punti immediatamente circostanti, ma non necessariamente il più basso in assoluto.
- **Punto di Sella**: Un punto in cui il gradiente della funzione è nullo, ma che non è né un minimo né un massimo locale.
[02:22] Queste caratteristiche creano un "paesaggio" (*landscape*) molto più difficile da esplorare rispetto al caso convesso.
# Capitolo 2: Analisi Visiva dei Problemi di Ottimizzazione
## Il Paesaggio della Funzione di Costo
[02:29] Per illustrare queste difficoltà, si considera un esempio visivo del paesaggio di una funzione di costo. Immaginiamo una superficie simile a un paraboloide da cui sono stati "sottratti" due campi gaussiani in punti diversi.
[02:40] Se uno dei due campi gaussiani è più profondo dell'altro, si creano un minimo globale e un minimo locale. Si analizza ora il comportamento della discesa del gradiente (stocastica o classica) su questa superficie.
[02:50] Partendo da un certo punto iniziale, l'algoritmo segue un percorso che, in questo caso, lo conduce a convergere in un minimo locale. Questo dimostra come il metodo sia estremamente sensibile alla scelta del punto di partenza.
[03:01] Con un punto di partenza diverso, l'algoritmo riesce invece a raggiungere il minimo globale.
[03:07] Se si parte da un punto situato nella zona centrale, vicino a un punto di sella, si osserva un altro problema. Un punto di sella è caratterizzato da un gradiente nullo.
[03:13] Quando l'algoritmo si avvicina a questa regione, i passi di avanzamento diventano estremamente piccoli, poiché il gradiente è quasi zero. In linea di principio, l'algoritmo potrebbe persino rimanere bloccato nel punto di sella.
[03:24] Come si può osservare, l'algoritmo è quasi fermo. Una piccola perturbazione potrebbe consentirgli di uscire da questa situazione e raggiungere uno dei minimi. Modificando leggermente il punto iniziale, si può passare dal convergere al minimo globale al convergere a quello locale.
[03:36] In questa simulazione, l'algoritmo è quasi bloccato. È possibile intervenire sul tasso di apprendimento: un valore più alto potrebbe aiutare a superare l'impasse.
[03:44] Tuttavia, un tasso di apprendimento molto elevato può introdurre un comportamento oscillatorio, con l'algoritmo che si muove da un lato all'altro della regione di minimo senza stabilizzarsi.
[03:54] Questo esempio evidenzia chiaramente i limiti della discesa del gradiente semplice e della sua variante stocastica.
## Comportamento Vicino ai Punti di Sella
[04:00] Il primo tentativo di miglioramento mira a superare i problemi illustrati, in particolare il comportamento problematico vicino ai punti di sella.
[04:05] Se l'algoritmo parte da un punto che lo porta ad avvicinarsi al punto di sella, le sue prestazioni diventano insoddisfacenti e la convergenza risulta estremamente lenta.
[04:14] L'obiettivo è quindi migliorare il comportamento della discesa del gradiente per superare queste difficoltà. Verranno presentati quattro metodi, in ordine cronologico di sviluppo, che cercano di risolvere questi problemi.
# Capitolo 3: Metodi di Ottimizzazione del Primo Ordine Avanzati
## Metodo del Momento (Momentum)
### Idea Fondamentale e Formulazione
[04:25] Il primo approccio per migliorare la discesa del gradiente è il **metodo del momento** (*Momentum*).
[04:31] L'idea centrale è modificare il modo in cui viene calcolata la direzione di aggiornamento dei parametri. Invece di basarsi unicamente sul gradiente calcolato al passo precedente, come nella regola di aggiornamento classica, si introduce un concetto di "memoria".
[04:40] La regola di aggiornamento per i parametri $w$ al passo $t+1$ è:
```math
w_{t+1} = w_t - v_{t+1}
```
dove $v_{t+1}$ è il vettore di aggiornamento.
[04:44] Questo vettore $v_{t+1}$ non è semplicemente proporzionale al gradiente, ma include anche una componente che tiene conto del vettore di aggiornamento precedente, $v_t$. La formula è:
```math
v_{t+1} = \mu v_t + \gamma \nabla J(w_t)
```
dove:
- $\mu$ è il **termine di momento** (*momentum term*), un iperparametro che controlla l'inerzia del movimento.
- $\gamma$ è il **tasso di apprendimento** (*learning rate*).
- $\nabla J(w_t)$ è il gradiente della funzione di costo $J$ calcolato rispetto ai parametri $w$ al passo $t$.
[04:52] In questo modo, l'aggiornamento ha una memoria della sua storia passata, non dipendendo solo dall'istante precedente.
[05:00] Il nome "momento" deriva dal comportamento pratico dell'algoritmo, che assomiglia a quello di una palla che rotola in una conca, accumulando inerzia.
[05:08] Nell'implementazione di questo metodo, appare un nuovo iperparametro, $\mu$, il termine di momento. Generalmente, il suo valore è impostato a 0.9. Il vettore $v_t$ è chiamato **vettore di aggiornamento** (*update vector*).
[05:18] A differenza della discesa del gradiente standard, che ha solo il tasso di apprendimento $\gamma$, il metodo del momento ne ha due: $\gamma$ e $\mu$. Sebbene $\mu$ sia spesso fissato a 0.9, può essere regolato in base alle necessità specifiche del problema.
### Vantaggi e Svantaggi del Metodo del Momento
[05:29] I principali vantaggi del metodo del momento sono:
1.  **Riduzione delle oscillazioni**: L'inerzia aiuta a smorzare i movimenti oscillatori.
2.  **Convergenza più rapida**: In molte situazioni, converge più velocemente rispetto alla discesa del gradiente classica.
[05:35] Tuttavia, c'è uno svantaggio: la "palla" è "cieca". A causa del momento accumulato, può superare un minimo e continuare la sua corsa, potenzialmente "sfuggendo" dalla regione di interesse.
### Confronto Visivo: Momentum vs. Discesa del Gradiente
[05:44] Tornando all'esempio visivo, si confronta la discesa del gradiente (in azzurro) con il metodo del momento (in magenta).
[05:51] Partendo entrambi dalla stessa posizione, si osserva che il metodo del momento raggiunge il minimo locale molto più rapidamente rispetto all'algoritmo standard. La scelta del minimo (locale o globale) dipende sempre dal punto di partenza.
[06:05] Partendo da un'altra posizione, si nota che, sebbene anche il metodo del momento possa incontrare difficoltà vicino al punto di sella, è in grado di superarlo più velocemente.
[06:17] In questa simulazione, il parametro $\mu$ (indicato come *decay rate*) è impostato a 0.8. Si osserva cosa accade con un valore di 0.9.
[06:22] Con $\mu = 0.9$, le prestazioni sono generalmente ancora migliori.
[06:27] Si può concludere che il metodo del momento sembra funzionare bene nel risolvere alcuni dei problemi evidenziati. Il confronto diretto tra i due metodi mostra un chiaro miglioramento.
## Gradiente Accelerato di Nesterov (NAG)
### Migliorare il Momento: l'Idea del "Look-Ahead"
[06:38] Un ulteriore miglioramento rispetto al metodo del momento classico è il **Gradiente Accelerato di Nesterov** (*Nesterov Accelerated Gradient*, NAG).
[06:46] Confrontando la formula del NAG con quella del momento, l'unica differenza risiede nell'argomento della funzione di costo $J$ all'interno del calcolo del gradiente.
[06:53] Nel metodo del momento classico, il gradiente è calcolato nella posizione corrente dei parametri, $w_t$.
[07:00] L'idea di Nesterov è di non usare la posizione corrente, ma di effettuare una sorta di "estrapolazione" o "previsione" della posizione futura.
[07:04] In pratica, si calcola una posizione approssimata futura, $w_{approx}$, utilizzando il passo di momento precedente:
```math
w_{approx} = w_t - \mu v_t
```
Il gradiente viene quindi calcolato in questo punto "guardando avanti" ($w_{approx}$) invece che nel punto corrente ($w_t$). La regola di aggiornamento diventa:
```math
v_{t+1} = \mu v_t + \gamma \nabla J(w_t - \mu v_t)
```
```math
w_{t+1} = w_t - v_{t+1}
```
[07:16] Tutti i termini in questa formula sono noti: $\mu$ è il termine di momento, $w_t$ è la posizione al passo precedente e $v_t$ è il vettore di aggiornamento precedente. Si esegue un'estrapolazione e si utilizza il valore estrapolato per calcolare il gradiente.
[07:25] L'idea è di avere un'informazione sul gradiente che non sia legata solo al passato, ma che sia più "informata" su ciò che sta per accadere. È una sorta di *look-ahead*, un tentativo di guardare nel futuro.
[07:36] In pratica, il NAG è più robusto del metodo del momento classico. È importante notare che non introduce nuovi iperparametri, quindi la sua complessità è identica a quella del momento, ma le sue prestazioni sono significativamente migliori.
### Rappresentazione Grafica: Momentum vs. NAG
[07:49] Una visualizzazione grafica può chiarire la differenza. A sinistra è rappresentato il metodo del momento classico.
[07:54] Il calcolo del gradiente (freccia rossa) avviene nella posizione corrente. Il vettore di aggiornamento del momento (freccia verde) viene sommato a questo gradiente per determinare il passo effettivo (freccia blu).
[08:03] Nel metodo di Nesterov, invece:
1.  Si calcola prima il passo basato sul momento precedente (freccia verde), che porta a un punto intermedio.
2.  Il gradiente (freccia rossa) viene calcolato in *questo punto intermedio*.
3.  Il passo effettivo (freccia blu) è la somma del passo di momento e del nuovo gradiente calcolato "in avanti".
[08:14] Sebbene la figura possa non evidenziare una differenza drastica, in pratica i due metodi si comportano in modo molto diverso.
[08:21] L'elemento chiave da ricordare è che il gradiente viene calcolato in due punti differenti: nel punto corrente per il Momentum, e in un punto futuro approssimato per il NAG.
## Adagrad (Adaptive Gradient)
### Adattare il Tasso di Apprendimento per Dati Sparsi
[08:27] Si torna ora al problema dei dati sparsi, dove è necessario utilizzare un tasso di apprendimento più elevato per le *features* più rare.
[08:36] Il primo metodo proposto per affrontare questo problema è **Adagrad** (*Adaptive Gradient*).
[08:42] L'idea di base è relativamente semplice. Invece di definire una regola di aggiornamento vettoriale, la si analizza parametro per parametro per una migliore comprensione.
[08:51] Per ogni singolo parametro $w_i$, la regola di aggiornamento al passo $t+1$ è:
```math
w_{i, t+1} = w_{i, t} - \frac{\gamma}{\sqrt{G_{ii, t} + \epsilon}} g_{i, t}
```
dove:
- $w_{i, t}$ è il valore del parametro $i$ all'iterazione $t$.
- $g_{i, t}$ è il gradiente della funzione di costo rispetto al parametro $i$ all'iterazione $t$.
- $\gamma$ è il tasso di apprendimento globale.
- $\epsilon$ è un piccolo termine di smussamento (*smoothing term*), solitamente dell'ordine di $10^{-8}$, aggiunto per evitare divisioni per zero.
- $G_{ii, t}$ è l'elemento sulla diagonale di una matrice diagonale $G_t$.
[09:08] Questo termine $G_{ii, t}$ accumula la somma dei quadrati dei gradienti passati per il parametro $i$:
```math
G_{ii, t} = \sum_{k=1}^{t} g_{i, k}^2
```
[09:20] L'idea è la seguente: se un parametro è associato a una *feature* che appare frequentemente, la somma dei quadrati dei suoi gradienti ($G_{ii, t}$) sarà grande. Di conseguenza, il tasso di apprendimento effettivo per quel parametro ($\frac{\gamma}{\sqrt{G_{ii, t} + \epsilon}}$) diventerà progressivamente più piccolo.
[09:34] Al contrario, per parametri associati a *features* rare, la somma $G_{ii, t}$ crescerà lentamente. Il denominatore rimarrà piccolo, e il tasso di apprendimento effettivo per quel parametro sarà più elevato.
[09:47] In forma vettoriale, la regola di aggiornamento si scrive come:
```math
w_{t+1} = w_t - \frac{\gamma}{\sqrt{G_t + \epsilon I}} \odot g_t
```
dove $G_t$ è la matrice diagonale contenente le somme dei quadrati dei gradienti, $I$ è la matrice identità e $\odot$ rappresenta il prodotto di Hadamard (prodotto elemento per elemento).
### Vantaggi e Svantaggi di Adagrad
[10:00] I vantaggi di Adagrad sono:
1.  **Tasso di apprendimento adattivo**: Il tasso di apprendimento viene adattato per ogni singolo parametro.
2.  **Adatto a dati sparsi**: È particolarmente efficace in presenza di *features* con frequenze diverse.
3.  **Minore necessità di sintonizzazione manuale**: In teoria, riduce la necessità di scegliere attentamente il tasso di apprendimento globale $\gamma$. Una volta impostato un valore iniziale (es. 0.01), l'algoritmo adatta automaticamente i tassi specifici per ciascun parametro.
[10:20] Tuttavia, Adagrad ha un importante svantaggio.
[10:23] Il problema risiede nell'accumulo continuo dei quadrati dei gradienti nel denominatore. Per parametri che vengono aggiornati frequentemente, questo termine può diventare molto grande.
[10:32] Di conseguenza, il tasso di apprendimento effettivo può diventare estremamente piccolo, portando a un arresto prematuro dell'apprendimento. Questo fenomeno è simile al problema del *vanishing gradient* (gradiente che svanisce), ma in questo caso è il tasso di apprendimento adattivo a diventare troppo piccolo, impedendo ulteriori progressi.
[10:47] Questo è il principale difetto del metodo e ha motivato la ricerca di algoritmi successivi che evitassero l'accumulo indefinito dei gradienti.
### Comportamento Pratico di Adagrad
[10:56] Analizzando il comportamento di Adagrad nell'esempio visivo, si nota che l'algoritmo sembra quasi fermo.
[11:03] Anche provando con un tasso di apprendimento più elevato, si conferma che, sebbene l'idea di base sia valida per i dati sparsi, le prestazioni pratiche in termini di velocità di convergenza non sono ottimali.
[11:15] Questo comportamento non è legato alla specifica funzione di costo utilizzata; in altri scenari, le sue prestazioni possono essere paragonabili ad altri metodi, ma il problema dell'accumulo dei gradienti rimane un limite intrinseco.
# Capitolo 4: Metodi di Ottimizzazione Adattivi
## Introduzione ai Metodi di Ottimizzazione Adattivi
[00:00] Nonostante i miglioramenti, il comportamento del metodo del gradiente non è ancora ottimale. Se confrontato con il metodo del momento, quest'ultimo risulta più performante di entrambi in questa specifica situazione. Il metodo AdaGrad è comunque rilevante perché introduce un approccio efficace per la gestione di dataset sparsi.
[00:06] Tuttavia, nella pratica, i due metodi più utilizzati sono ADAM e RMSprop. Si analizza ora come sono definiti. Prima di esaminare questi due algoritmi, è utile introdurre un passaggio intermedio.
[00:13] Questo passaggio intermedio è rappresentato dal metodo cosiddetto AdaDelta. Tutti questi metodi di ottimizzazione sono disponibili nelle principali librerie di calcolo come PyTorch o TensorFlow, permettendo di sperimentare con diversi algoritmi durante l'addestramento di una rete neurale.
[00:22] È possibile quindi testare l'efficacia di questi e di molti altri metodi disponibili. L'idea fondamentale di AdaDelta è quella di utilizzare una media mobile pesata (*decaying average*) dei gradienti passati al quadrato.
[00:29] Al di là delle formule specifiche, il concetto chiave è non utilizzare tutti i gradienti precedenti, ma una loro media mobile.
[00:34] Ad esempio, si può decidere di considerare solo le ultime 10 iterazioni, calcolando la media e la somma su questa finestra temporale specifica. Vengono poi introdotte alcune tecnicalità relative al riscalamento del vettore di aggiornamento, ma questo aspetto è meno cruciale.
[00:44] L'elemento fondamentale da comprendere è che l'idea centrale consiste nell'uso di una media mobile. Il passaggio evolutivo da AdaGrad ad AdaDelta risiede proprio nell'introduzione di questo tipo di media.
[00:52] A livello pratico, un'implementazione di AdaDelta non richiede l'introduzione esplicita di un tasso di apprendimento (*learning rate*). L'aggiornamento è determinato dalla radice quadrata della media dei quadrati (*root mean square*) di questa media mobile dei gradienti precedenti al quadrato. Di conseguenza, uno dei vantaggi principali è l'assenza della necessità di impostare un *learning rate*.
## Il Metodo RMSprop
[01:02] Il metodo RMSprop è stato sviluppato quasi contemporaneamente ad AdaDelta e la sua formulazione è molto simile.
[01:07] La differenza principale consiste nel fatto che, invece di eliminare il *learning rate* e usare la radice quadrata della media mobile, RMSprop mantiene il *learning rate* al numeratore e pone la media mobile al denominatore.
[01:15] Essendo stati sviluppati nello stesso periodo da gruppi di ricerca diversi, AdaDelta e RMSprop si basano essenzialmente sulla stessa idea di fondo.
## Il Metodo ADAM: Combinazione di Momento e RMSprop
[01:20] Successivamente, è stato introdotto il metodo ADAM, che è oggi uno dei più utilizzati. ADAM può essere considerato un approccio intermedio, in quanto combina l'idea del momento con quella di RMSprop.
[01:26] L'aggiornamento dei pesi viene calcolato considerando lo stato precedente e un termine di aggiornamento che dipende dal *learning rate*. Questo termine include un vettore $m_t$, chiamato primo momento.
[01:32] Il primo momento, $m_t$, è definito in modo simile a quanto visto nel metodo del momento. Il secondo termine, $v_t$, chiamato secondo momento, che appare nella formula, è invece legato al quadrato del gradiente, in modo analogo a quanto utilizzato in RMSprop o AdaDelta.
[01:41] ADAM rappresenta quindi una sintesi dei due approcci. Dal punto di vista pratico, questo è il metodo utilizzato nel 90% dei casi. Uno degli svantaggi è l'introduzione di due nuovi iperparametri, $\beta_1$ e $\beta_2$, che vengono usati per scalare i vettori $m_t$ e $v_t$.
[01:51] Questi iperparametri servono anche a calcolare la combinazione lineare tra il momento precedente e il secondo momento precedente con, rispettivamente, il gradiente e il quadrato del gradiente. Sebbene l'introduzione di nuovi parametri sia uno svantaggio, in pratica per $\beta_1$ e $\beta_2$ si utilizzano valori standard.
[01:59] In particolare, per $\beta_2$ si possono sperimentare valori compresi tra 0.9 e 0.999, ma l'intervallo di scelta non è molto ampio. Il metodo è più complesso e può richiedere una maggiore quantità di memoria, poiché è necessario salvare due vettori ($m_t$ e $v_t$). In scenari con un numero elevato di variabili, questi vettori possono diventare molto grandi.
[02:09] Inoltre, in alcune situazioni, il metodo del momento può comunque risultare più performante. Non esiste una regola aurea: sebbene ADAM sia il cavallo di battaglia per l'ottimizzazione, non è sempre la scelta migliore.
## Confronto Visivo dei Metodi di Ottimizzazione
[02:13] Si torna ora ai confronti visivi. Qui si può osservare il comportamento di RMSprop. Rispetto al gradiente semplice, è migliore. Tuttavia, come già notato, in questa particolare situazione il metodo del momento si dimostra superiore.
[02:22] Si analizza ora una situazione complessa con due "colline". Qui è presente un minimo globale, ma anche un altro punto difficile da raggiungere a seconda del punto di partenza.
[02:30] La difficoltà è dovuta alla presenza di un minimo locale dal quale è molto difficile uscire. Si osserva il comportamento dei diversi metodi. ADAM, rappresentato dalla linea blu, in questo caso si trova in difficoltà. La capacità di raggiungere il minimo globale dipende fortemente non tanto dal metodo, quanto dalla posizione iniziale.
[02:42] Ruotando la visualizzazione, si osserva che, partendo da un'altra posizione, il gradiente e il momento si dirigono direttamente verso il minimo locale, mentre ADAM riesce a trovare il minimo globale. Da questo punto di vista, ADAM è spesso il metodo consigliato.
[02:52] Tuttavia, come visto nel caso precedente, non è sempre così. Si considera ora una situazione caratterizzata da un'area di plateau. Per raggiungere il minimo globale, è ovviamente necessario agire sul *learning rate*.
[03:02] Ad esempio, utilizzando un *learning rate* più alto per ADAM e mantenendolo uguale per gli altri due metodi, si può osservare che ADAM raggiunge il minimo, il gradiente si blocca sul plateau e il momento inizia a oscillare.
[03:10] Questo dimostra che, in questo senso, ADAM è un metodo più robusto.
## Riepilogo sui Metodi di Ottimizzazione
[03:12] In sintesi, se il dataset è sparso, è ragionevole utilizzare uno dei metodi progettati specificamente per questa caratteristica, come AdaGrad, AdaDelta, RMSprop o ADAM.
[03:19] In generale, ADAM è la scelta migliore, ma esistono situazioni in cui la discesa del gradiente stocastico (SGD), abbinata a una schedulazione adeguata del *learning rate*, può superare anche i metodi più complessi.
[03:26] Questa analisi ha solo scalfito la superficie dei metodi di ottimizzazione ottenuti introducendo variazioni sulla discesa del gradiente o sulle sue versioni stocastiche.
[03:32] La letteratura scientifica propone molte altre soluzioni per affrontare situazioni particolari. L'obiettivo di questa parte era fornire i due concetti più importanti: l'idea del momento e l'idea del *learning rate* adattivo per dataset sparsi.
[03:41] Questi sono i due messaggi chiave da portare a casa da questa sezione.
# Capitolo 5: Informazioni Organizzative sui Progetti del Corso
## Introduzione ai Progetti del Corso
[03:45] Si dedicano ora alcuni minuti alla discussione dei progetti per il corso. Successivamente, se ci sarà tempo, si inizierà un nuovo argomento.
[03:51] Verrà condiviso un documento che elenca alcune idee per i progetti. Sono stati rimossi alcuni argomenti proposti negli anni passati perché trattati troppe volte. Il documento presenta una serie di progetti fino al punto 0.7 e, a seguire, alcune idee per approfondimenti teorici.
[04:02] Come anticipato all'inizio del corso, esistono due alternative principali per i progetti. La prima è quella definita "progetto", per la quale ci si aspetta che lo studente consideri un articolo scientifico o un argomento tra quelli elencati.
[04:11] Per l'argomento scelto, si dovrà utilizzare uno dei metodi proposti in letteratura per risolvere quel problema specifico o un suo aspetto. Questi progetti possono essere svolti da una, due o, in casi speciali, anche tre persone. L'approfondimento teorico, invece, è pensato per essere svolto individualmente.
[04:22] Le aspettative per le due tipologie sono diverse. Per il progetto pratico, è richiesta la consegna del codice (un notebook Python o script Python), accompagnato da una relazione che descriva i dataset, il metodo utilizzato, l'implementazione e una discussione dei risultati ottenuti.
[04:33] Per la presentazione finale, valida per entrambe le tipologie di progetto, è possibile utilizzare la relazione stessa, oppure preparare una presentazione in formato PowerPoint o Beamer, a discrezione dello studente, senza che sia un requisito stringente. Per l'approfondimento teorico, l'aspettativa è diversa.
[04:43] L'obiettivo dell'approfondimento teorico è analizzare uno degli argomenti che sono stati solo accennati o non trattati in dettaglio durante il corso. Si dovrà preparare una breve relazione, di circa 10-15 pagine, in cui viene presentata la teoria relativa all'argomento.
[04:52] La presentazione consisterà in una breve lezione di circa 20 minuti su quell'argomento. Nel documento, che verrà formattato meglio, sono stati aggiunti alla fine ulteriori progetti con riferimenti già inclusi a dataset o articoli scientifici.
## Modulo di Registrazione per i Progetti
[05:02] È stato preparato un modulo, che sarà pubblicato in serata, dove gli studenti dovranno inserire il proprio nome, l'indirizzo email e il tipo di corso (da 8 o 10 crediti). Questa informazione è importante. Anche gli studenti del corso da 8 crediti possono svolgere il progetto, se lo desiderano, per migliorare il proprio voto.
[05:13] Nel modulo si dovrà specificare il tipo di progetto e se l'argomento scelto è presente nella lista fornita. È accaduto, negli anni passati, che alcuni studenti avessero una propria idea di progetto. In tali casi, hanno proposto l'idea con alcuni riferimenti e hanno lavorato su quel progetto specifico, magari perché era un argomento che stavano già considerando per la tesi.
[05:25] È possibile discutere anche questo tipo di situazioni. Pertanto, il modulo chiede di indicare se il progetto scelto è nella lista o meno, il tipo di progetto e l'argomento. Si dovrà inoltre specificare se si lavora da soli o in gruppo, e in tal caso, i nomi degli altri membri del gruppo e le loro email. All'inizio del modulo, va indicato un referente per il gruppo.
[05:41] Un'altra informazione richiesta è la data di consegna prevista. Non è necessaria una data esatta, ma un'indicazione di massima (es. gennaio, febbraio, giugno, luglio, settembre) per avere un'idea generale delle tempistiche.
[05:51] Se uno studente indica gennaio ma poi decide di consegnare a settembre, non ci sono problemi. L'informazione serve solo a organizzare le sessioni di presentazione.
[05:59] Il modulo chiede anche se si necessita di materiale da parte del docente. Ad esempio, se si sceglie di lavorare sull'interpretabilità, come i modelli additivi generalizzati (Generalized Additive Models), ma non si ha idea di una possibile applicazione pratica.
[06:10] In questo caso, è possibile richiederlo e verranno forniti un articolo scientifico e un dataset (sintetico o reale) da utilizzare per il progetto. Se, per qualche motivo, si dispone già del materiale (ad esempio, perché si sta lavorando alla tesi o a un progetto congiunto con altri corsi),
[06:21] come un progetto tra questo corso e quello del professor Formaggia, è possibile discuterne. Se nel modulo si è indicato che l'argomento non è nella lista,
[06:29] è necessario fornire una descrizione dettagliata del tema con riferimenti, come link a PDF di articoli scientifici, in modo da poter valutare se il progetto è adeguato, troppo complesso o troppo semplice.
[06:37] Infine, anche se si è scelto un argomento dalla lista ma si dispone già di materiale, è richiesto di fornire un'idea del materiale in possesso.
## Tempistiche e Organizzazione
[06:44] La data odierna è il 10 novembre. Idealmente, il modulo dovrebbe essere compilato entro una settimana o dieci giorni, in modo da avere un quadro generale della situazione.
[06:54] Chiaramente, se qualcuno non è sicuro su cosa fare e vuole posticipare la decisione, può compilare il modulo anche a gennaio. Questo implica che si prevede di iniziare a lavorare sul progetto in quel periodo. L'idea è di iniziare leggendo l'articolo fornito o quello già in possesso.
[07:05] Si può iniziare a lavorare durante il corso, da ora fino alla fine, o anche dopo. È possibile fissare incontri, sia di persona che online, per discutere eventuali problemi riscontrati durante la realizzazione del progetto. In caso di problemi, è importante inviare
[07:15] una email. Si ricorda di includere nell'oggetto dell'email il tag `[NANL 2025]` seguito dall'argomento, e di inviare l'email sia al docente che a Matteo, in modo che almeno uno dei due possa rispondere.
[07:23] Le sessioni di presentazione dei progetti per la sessione di gennaio e febbraio vengono solitamente organizzate insieme. L'anno scorso, è stata pianificata un'unica sessione dopo la seconda prova scritta. Quindi, ci sarà la seconda prova scritta, la seconda prova orale e, una settimana dopo, la sessione di presentazione dei progetti.
[07:35] Indicativamente, per la prima sessione (gennaio-febbraio), le presentazioni si terranno verso metà o fine febbraio. Per la sessione di giugno-luglio, si terranno verso la fine di luglio, e per quella di settembre, a metà settembre. In caso di esigenze particolari, ad esempio presentare il progetto a metà del prossimo semestre (aprile),
[07:49] è possibile trovare un accordo. Ovviamente, non sarà possibile registrare immediatamente il voto. La registrazione avverrà durante una delle sessioni ufficiali di giugno, luglio o settembre, quando verrà verbalizzato il voto finale.
[07:58] Si possono quindi gestire esigenze particolari, come periodi di studio all'estero o la necessità di suddividere il carico di studio. In passato, si è sempre trovata una soluzione.
[08:07] Solitamente non si organizzano presentazioni singole, ma si cerca di raggruppare almeno due o tre persone. Sono già state ricevute due richieste per progetti congiunti con altri corsi.
[08:14] Se ci sono altre richieste di questo tipo, si prega di inviare una email per discuterne.
## Domande e Risposte sui Progetti
[08:18] Ci sono domande, anche da casa, riguardo ai progetti? Verrà condiviso il documento con le proposte, dopo averlo sistemato. Successivamente, si dovrà compilare il modulo. Verso la fine di novembre, quando un certo numero di risposte sarà stato raccolto,
[08:28] si inizierà a inviare il materiale. Ragionevolmente, il materiale verrà inviato entro 10-15 giorni da oggi. Se qualcuno ha l'esigenza di iniziare subito, ad esempio domani,
[08:36] può, dopo aver compilato il modulo, inviare anche una email specificando la necessità di ricevere il materiale immediatamente. Si farà il possibile per inviarlo quanto prima.
[08:44] Domanda: "Tutte e tre le persone del gruppo devono compilare il modulo?" No, è sufficiente che una persona di riferimento compili il modulo. Al punto 8 del modulo, si devono inserire i nomi degli altri membri. Uno studente sarà il "leader" del gruppo e gli altri saranno i membri.
[08:55] Esempio: se un gruppo è composto da tre persone, e Antonio è il leader, sarà lui a compilare il modulo per tutti, inserendo i nomi e le email degli altri membri.
[09:05] Antonio si occuperà della compilazione, e gli altri membri non dovranno farlo. Una sola persona per gruppo deve essere responsabile della compilazione del modulo.
[09:11] Un'altra questione: può capitare che, per vari motivi, un membro di un gruppo non sia disponibile in un giorno specifico. L'idea è che tutti i membri del gruppo siano presenti durante la presentazione.
[09:18] Tuttavia, è successo che, ad esempio, Antonio abbia superato l'esame scritto e orale e sia pronto a registrare anche il voto del progetto per ottenere il risultato finale, mentre Francesco non ha ancora superato l'esame scritto. Entrambi vogliono presentare il progetto insieme, ma Antonio ha bisogno di presentarlo in quella sessione per ottenere il voto.
[09:29] In questo caso, il progetto viene presentato da entrambi. Il voto di Francesco viene "congelato" e, quando in futuro supererà l'esame, verrà calcolata la media.
[09:36] Domanda sui linguaggi di programmazione: in linea di principio, si è liberi di usare qualsiasi linguaggio, con una precisazione. Se si usa Python, è possibile ricevere supporto. Se si usa C++, il supporto non è garantito con la stessa sicurezza.
[09:46] I linguaggi con cui c'è familiarità sono Python, MATLAB, Fortran e C (anche se è improbabile che qualcuno lo usi). C++ è conosciuto, ma non è tra i preferiti.
[09:54] Domanda sui progetti congiunti: "Presenteremo lo stesso progetto per due esami diversi?" L'anno scorso, due gruppi hanno svolto progetti congiunti e hanno tenuto due presentazioni separate, una per ciascun corso. Si può provare a organizzare in modo più semplice, se necessario. Il motivo delle presentazioni separate è che, sebbene i risultati siano gli stessi, il focus delle due presentazioni può essere leggermente diverso.
[10:11] Un aspetto importante dei progetti congiunti è che, essendo validi per due corsi, devono avere una certa rilevanza e complessità. Questo è l'unico requisito implicito.
[10:20] Domanda sul calcolo del voto finale: per il corso da 10 crediti (8+2), il voto finale sarà una media ponderata. Il voto della parte scritta e orale peserà per l'80%, mentre il progetto peserà per il 20%.
[10:29] Per gli studenti del corso da 8 crediti, il calcolo del voto è essenzialmente lo stesso. Per loro è ancora più importante compilare il modulo, poiché in linea di principio non sono tenuti a svolgere il progetto.
# Capitolo 6: Metodi di Ottimizzazione del Secondo Ordine
## Introduzione ai Metodi del Secondo Ordine
[00:00] Fino a questo momento sono stati analizzati esclusivamente i cosiddetti metodi del primo ordine. Questi metodi sono definiti "del primo ordine" perché la regola di aggiornamento per le iterazioni successive si basa unicamente su informazioni derivanti dal gradiente della funzione di costo.
[00:13] Esiste tuttavia una famiglia di metodi che sfrutta informazioni di ordine superiore, in particolare la matrice Hessiana.
[00:22] Prima di approfondire questi metodi specifici, è utile dedicare del tempo a un ripasso del metodo di Newton in un contesto generale.
## Il Metodo di Newton per la Ricerca di Zeri (Root Finding)
[00:28] Il metodo di Newton è comunemente introdotto nel contesto della ricerca di zeri (o radici) di una funzione. Data una funzione non lineare $f$, l'obiettivo è trovare uno dei suoi zeri, ovvero un punto $x$ tale che $f(x) = 0$.
[00:43] L'idea fondamentale del metodo consiste nell'utilizzare le rette tangenti alla funzione di cui si cercano gli zeri. Si calcola la tangente in un punto dato e si determina l'intersezione di tale retta con l'asse delle ascisse (l'asse x).
[00:54] Questo punto di intersezione diventa la nuova iterazione del metodo. Da qui, si proietta il punto sulla funzione, si calcola una nuova tangente e si ripete il processo. In sostanza, per trovare lo zero, il metodo si basa su un'approssimazione lineare locale della funzione.
### Formulazione Matematica del Metodo di Newton per il Root Finding
[01:02] L'equazione della retta tangente al punto $x_k$ può essere espressa come l'espansione di Taylor del primo ordine della funzione $f$ attorno a $x_k$:
```math
y(x) \approx f(x_k) + f'(x_k)(x - x_k)
```
dove:
- $y(x)$ è l'approssimazione lineare della funzione.
- $f(x_k)$ è il valore della funzione nel punto $x_k$.
- $f'(x_k)$ è la derivata prima della funzione calcolata in $x_k$.
[01:12] Una volta ottenuta questa espressione, l'obiettivo è trovare l'intersezione della retta tangente con l'asse delle ascisse, il che si ottiene ponendo l'equazione uguale a zero:
```math
f(x_k) + f'(x_k)(x - x_k) = 0
```
[01:17] Valutando questa espressione per $x = x_{k+1}$, si ottiene la regola di aggiornamento per il metodo di Newton per la ricerca degli zeri:
```math
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
```
Questa è la formula classica per trovare lo zero di una funzione.
### Interpretazione Geometrica
[01:26] L'interpretazione geometrica del processo è la seguente:
1.  Si parte da un punto iniziale $x_0$.
2.  Si calcola la retta tangente alla funzione in $x_0$.
3.  Si trova l'intersezione di questa retta con l'asse x, ottenendo il punto $x_1$.
4.  Si ripete il processo calcolando la tangente in $x_1$ per trovare $x_2$, e così via, avvicinandosi progressivamente allo zero della funzione.
## Adattamento del Metodo di Newton per l'Ottimizzazione
[01:38] L'interesse primario non è la ricerca di zeri, ma la minimizzazione di una funzione. Data una funzione $f(x)$, si vuole trovare un punto di minimo.
[01:46] L'osservazione cruciale è che in un punto di minimo, la pendenza della funzione è nulla. Ciò significa che la derivata prima della funzione si annulla in corrispondenza del minimo.
[01:53] L'idea, quindi, è applicare il metodo di Newton per la ricerca di zeri non alla funzione $f$, ma al suo gradiente (o alla sua derivata prima, nel caso monodimensionale). Si cerca il punto in cui la derivata prima, $f'$, è uguale a zero.
### Derivazione della Formula di Aggiornamento per la Minimizzazione
[02:02] Mantenendo la struttura della formula di Newton per il root finding, si sostituisce la funzione $f$ con la sua derivata prima $f'$.
[02:10] Di conseguenza, la derivata prima $f'$ nella formula originale viene sostituita dalla derivata seconda $f''$. Se si considera il gradiente, si ottiene il metodo di Newton per il gradiente. Sostituendo $g = f'$ e $g' = f''$ nella formula di aggiornamento, si ottiene la regola per la minimizzazione:
```math
x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}
```
## Prospettiva Geometrica della Minimizzazione: Approssimazione Quadratica
[02:20] Questa formula può essere derivata anche da una prospettiva geometrica differente, utilizzando l'espansione di Taylor.
[02:23] Nel caso del root finding, si utilizzava un'espansione di Taylor del primo ordine, approssimando localmente la funzione con una retta.
[02:31] Per la minimizzazione, si considera invece l'espansione di Taylor del secondo ordine della funzione $f$ attorno al punto $x_k$:
```math
q(x) = f(x_k) + f'(x_k)(x - x_k) + \frac{1}{2}f''(x_k)(x - x_k)^2
```
[02:40] Utilizzare un'espansione di Taylor del secondo ordine significa adottare un'approssimazione quadratica locale della funzione. In altre parole, si approssima la funzione $f$ con una parabola nelle vicinanze di $x_k$.
[02:50] Questa espressione, $q(x)$, è un'approssimazione locale della funzione $f$. Poiché siamo interessati al minimo, dobbiamo trovare il punto in cui la derivata di questa approssimazione si annulla.
[02:56] Si calcola quindi la derivata di $q(x)$ rispetto a $x$ e la si pone uguale a zero:
```math
q'(x) = f'(x_k) + f''(x_k)(x - x_k) = 0
```
[03:05] Esattamente come nel caso precedente, valutando questa espressione per $x = x_{k+1}$, si ottiene la regola di aggiornamento:
```math
f'(x_k) + f''(x_k)(x_{k+1} - x_k) = 0
```
Risolvendo per $x_{k+1}$, si ritrova la stessa formula ottenuta applicando il metodo di Newton al gradiente.
### Interpretazione Grafica dell'Approssimazione Quadratica
[03:13] La rappresentazione grafica di questo approccio è la seguente:
1.  Si parte da un punto $x_0$.
2.  Si calcola la parabola che approssima localmente la funzione in $x_0$.
3.  Si calcola il minimo di questa parabola, che diventa il nuovo punto $x_1$.
4.  Si ripete il processo costruendo una nuova parabola in $x_1$, calcolandone il minimo, e così via.
[03:24] L'idea è che, ad ogni passo, si minimizza un modello quadratico locale della funzione obiettivo.
### Condizioni di Applicabilità
[03:32] Sia nel caso della ricerca di zeri che in quello della minimizzazione, esistono delle condizioni da rispettare per garantire il corretto funzionamento del metodo.
-   **Root Finding**: È necessario che la derivata prima $f'(x_k)$ sia diversa da zero. Se $f'(x_k)$ fosse vicina a zero, potrebbero sorgere problemi numerici a causa della divisione.
-   **Minimizzazione**: Analogamente, è necessario che la derivata seconda $f''(x_k)$ sia diversa da zero.
## Estensione del Metodo di Newton al Caso Multidimensionale
[03:44] Il passo successivo consiste nell'estendere i concetti visti per il caso monodimensionale (1D) al caso multidimensionale (N-D).
[03:52] Si considera una funzione $f: \mathbb{R}^n \to \mathbb{R}$ che si assume essere due volte differenziabile. Questa assunzione è necessaria perché, analogamente alla comparsa della derivata seconda nel caso 1D, nel caso multidimensionale comparirà la matrice Hessiana.
[04:03] L'obiettivo è trovare un minimo della funzione $f$.
### Notazione e Condizione di Ottimalità
[04:06] Si introduce la notazione per la matrice Hessiana, indicata con $H_f(x)$ o semplicemente $H$. Questa è una matrice quadrata $n \times n$ contenente tutte le derivate parziali seconde della funzione, dove $n$ è il numero di variabili.
[04:12] La condizione necessaria per un punto di minimo è che il gradiente della funzione sia nullo:
```math
\nabla f(x) = 0
```
[04:17] Trovare il minimo di $f$ è quindi equivalente a trovare gli zeri del suo gradiente, $\nabla f$.
[04:21] Di conseguenza, è possibile applicare il metodo di Newton per la ricerca di zeri alla funzione vettoriale $g(x) = \nabla f(x)$, in modo del tutto analogo a quanto fatto nel caso 1D.
### Derivazione della Formula nel Caso Multidimensionale
[04:30] Si parte dalla formula del metodo di Newton per il caso 1D:
```math
x_{k+1} = x_k - [g'(x_k)]^{-1} g(x_k)
```
[04:33] Si analizzano ora le controparti N-dimensionali di ciascun termine:
-   $g(x_k)$ diventa il **gradiente** della funzione $f$ calcolato in $x_k$, ovvero $\nabla f(x_k)$.
-   $g'(x_k)$ diventa la **matrice Jacobiana** della funzione vettoriale $g(x) = \nabla f(x)$. La Jacobiana del gradiente è, per definizione, la **matrice Hessiana** $H_f(x_k)$.
[04:44] Nella formula 1D, il termine $[g'(x_k)]^{-1}$ rappresenta l'inverso della derivata. Nel caso multidimensionale, questo termine corrisponde all'**inversa della matrice Hessiana**, $H_f(x_k)^{-1}$.
[04:53] Sostituendo questi termini, la regola di aggiornamento per il metodo di Newton nel contesto dell'ottimizzazione multidimensionale diventa:
```math
x_{k+1} = x_k - H_f(x_k)^{-1} \nabla f(x_k)
```
[05:04] Questa espressione definisce l'aggiornamento ad ogni iterazione. Il vettore di aggiornamento, $\Delta x_k$, è dato da:
```math
\Delta x_k = - H_f(x_k)^{-1} \nabla f(x_k)
```
e la regola di aggiornamento è $x_{k+1} = x_k + \Delta x_k$.
### Risoluzione Pratica: il Sistema Lineare
[05:14] Una regola fondamentale dell'analisi numerica è che non si calcola mai esplicitamente l'inversa di una matrice, a meno che non sia strettamente necessario.
[05:20] Chiamando il vettore di aggiornamento $\Delta x$, invece di calcolare l'inversa dell'Hessiana, si risolve il seguente sistema di equazioni lineari:
```math
H_f(x_k) \Delta x = - \nabla f(x_k)
```
[05:29] In questo sistema lineare:
-   La matrice dei coefficienti è la matrice Hessiana $H_f(x_k)$.
-   Il vettore dei termini noti è il gradiente $-\nabla f(x_k)$.
-   La soluzione del sistema è il vettore di aggiornamento $\Delta x$.
## Approccio Geometrico Multidimensionale
[05:43] Anche nel caso multidimensionale, è possibile derivare la stessa formula partendo da un ragionamento geometrico, come fatto per il caso 1D.
[05:50] Si utilizza l'espansione di Taylor del secondo ordine per una funzione di più variabili, che fornisce un'approssimazione quadratica locale $q(x)$ della funzione $f(x)$ attorno a $x_k$:
```math
q(x) = f(x_k) + \nabla f(x_k)^T (x - x_k) + \frac{1}{2} (x - x_k)^T H_f(x_k) (x - x_k)
```
[06:00] Per trovare il minimo di questa approssimazione quadratica, se ne calcola il gradiente e lo si pone uguale a zero:
```math
\nabla q(x) = \nabla f(x_k) + H_f(x_k) (x - x_k) = 0
```
[06:07] Valutando questa espressione in $x = x_{k+1}$, si ottiene nuovamente la relazione che porta alla formula di aggiornamento del metodo di Newton.
## Sfide Computazionali del Metodo di Newton Multidimensionale
[06:15] Sebbene l'estensione dal caso 1D al caso N-D sia concettualmente diretta, emergono differenze significative dal punto di vista computazionale.
[06:22] La principale differenza è che i calcoli nel caso N-D sono molto più onerosi.
[06:27] Nel caso 1D, il calcolo dell'aggiornamento richiede una semplice divisione. Nel caso multidimensionale, è necessario risolvere un sistema lineare, che può essere di grandi dimensioni.
[06:35] Il costo computazionale per risolvere un sistema lineare $n \times n$ con metodi classici (come l'eliminazione di Gauss) è dell'ordine di $O(n^3)$ operazioni in virgola mobile.
[06:44] Se la dimensione $n$ è grande, come accade tipicamente nelle applicazioni pratiche (ad esempio, nel machine learning), questo costo rende il calcolo proibitivo.
[06:52] Un'altra condizione è che la matrice Hessiana sia invertibile, il che è necessario per poter risolvere il sistema lineare. In pratica, questo requisito è legato al fatto che, se la funzione ha un minimo, la sua Hessiana è definita positiva, almeno localmente.
[07:04] Tra i problemi pratici, il più rilevante è l'elevato costo computazionale legato alla formazione e alla soluzione del sistema lineare.
## Riepilogo Comparativo: Caso 1D vs. Caso N-D
[07:12] La tabella seguente riassume il parallelismo tra il caso monodimensionale e quello multidimensionale.
| Concetto | Caso 1D (Monodimensionale) | Caso N-D (Multidimensionale) |
| :--- | :--- | :--- |
| **Regola di Aggiornamento** | $x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$ | $x_{k+1} = x_k - H_f(x_k)^{-1} \nabla f(x_k)$ |
| **Costo Computazionale** | $O(1)$ (una divisione) | $O(n^3)$ (soluzione di un sistema lineare) |
[07:18] La sfida principale che i primi utilizzatori del metodo di Newton hanno dovuto affrontare è stata la soluzione di questo sistema lineare per valori elevati di $n$.
## Introduzione ai Metodi Quasi-Newton
[07:26] Per superare questi limiti computazionali, sono stati sviluppati i cosiddetti **metodi Quasi-Newton**.
[07:30] L'idea di base di questi metodi è quella di non utilizzare la matrice Hessiana esatta, $H$, ma una sua approssimazione, $B_k$, che sia più facile da calcolare e da invertire.
[07:37] In linea di principio, si potrebbe calcolare l'Hessiana tramite differenze finite, valutando il gradiente in punti vicini.
[07:45] Tuttavia, questo approccio soffre degli stessi inconvenienti visti per la differenziazione automatica con differenze finite, ovvero problemi di precisione legati alle operazioni in virgola mobile.
[07:54] Di conseguenza, questo approccio non è considerato praticabile e la differenziazione numerica tramite differenze finite non viene quasi mai utilizzata in questo contesto.
[07:59] Sono state invece sviluppate altre strategie per costruire approssimazioni della matrice Hessiana che siano efficienti dal punto di vista computazionale.
## Prossimi Argomenti: Metodi Quasi-Newton e BFGS
[08:03] Nelle prossime lezioni verranno analizzati in dettaglio i metodi Quasi-Newton.
[08:07] In particolare, si studierà l'algoritmo **BFGS (Broyden–Fletcher–Goldfarb–Shanno)**, che rappresenta un altro metodo di ottimizzazione fondamentale, ampiamente utilizzato nelle librerie di deep learning.
[08:14] Il prossimo argomento sarà quindi dedicato ai metodi Quasi-Newton.