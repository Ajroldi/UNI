## Capitolo 1: Il Percettrone, Origini e Formulazione
### Introduzione Storica: Il Percettrone
[00:00] In questa lezione introduttiva verranno presentati alcuni elementi fondamentali che saranno necessari nel prosieguo del corso. Si analizzeranno inoltre le connessioni tra i concetti di differenziazione automatica, discussi precedentemente, e le reti neurali.
[00:13] Da una prospettiva storica, il primo tentativo di sviluppare una struttura teorica che assomigliasse a un neurone risale agli anni '50. Il modello che ne derivò fu chiamato **percettrone**.
[00:22] Un percettrone è un elemento computazionale caratterizzato da un certo numero di input binari, i quali possono assumere valori come 0 o 1. Il suo scopo è produrre un singolo output, anch'esso binario.
[00:35] La rappresentazione formale di un percettrone con tre input è la seguente: un cerchio che riceve tre segnali in ingresso e produce un unico segnale in uscita. L'analisi si concentrerà su ciò che avviene all'interno di questo cerchio.
[00:45] L'idea fondamentale è associare a ciascun input un parametro chiamato **peso** (in inglese, *weight*). Questi pesi quantificano l'importanza di ogni singolo input nel calcolo dell'output finale.
[00:56] Successivamente, si calcola la somma ponderata di tutti gli input. L'output viene determinato confrontando questa somma con una soglia prestabilita.
[01:04] Formalmente, l'output del percettrone è definito come:
- **0** se la somma ponderata è minore o uguale alla soglia.
- **1** se la somma ponderata è maggiore della soglia.
### Formulazione Moderna con Pesi e Bias
[01:16] Questa è la formulazione originale. Attualmente, l'uso di una soglia esplicita è stato abbandonato. Il percettrone viene descritto in termini di pesi e di un altro parametro, il **bias**. Questo termine sarà ricorrente anche nel contesto di reti neurali più generali.
[01:31] Dal punto di vista pratico, la somma ponderata viene espressa come il prodotto scalare tra un vettore di pesi $w$ e il vettore di input $x$.
[01:38] Si introduce il bias, indicato con $b$, che è definito come l'opposto della soglia originale.
[01:43] Di conseguenza, la formula precedente può essere riscritta nel seguente modo:
$$
\text{output} =
\begin{cases}
0 & \text{se } w \cdot x + b \le 0 \\
1 & \text{se } w \cdot x + b > 0
\end{cases}
$$
Questa è la formulazione comunemente utilizzata nella pratica.
[01:53] Il significato del bias è complementare a quello dei pesi. Mentre i pesi determinano l'importanza relativa di ciascun input, il bias indica quanto sia facile per il percettrone produrre un output pari a 1.
[02:01] L'espressione "attivarsi" (in inglese, *to fire*) viene utilizzata in questo contesto perché, storicamente, il percettrone e l'intera architettura delle reti neurali sono stati sviluppati cercando di mimare il comportamento dei neuroni biologici.
[02:10] I neuroni nel cervello, a livello chimico, si attivano o meno a seconda del superamento di una certa soglia. Il bias, quindi, è una misura di quanto sia facile per il neurone artificiale raggiungere lo stato di attivazione (output 1).
[02:20] È chiaro che se il bias $b$ è un valore positivo e grande, sarà più facile ottenere un output pari a 1. Al contrario, se è un valore negativo e grande in valore assoluto, sarà molto difficile che l'output sia 1. Questa conclusione sarà ripresa più avanti.
## Capitolo 2: Universalità e Limiti del Percettrone
### Il Percettrone come Porta Logica Universale
[02:35] È opportuno fare un commento sulla nomenclatura. In letteratura, è comune l'acronimo **MLP**, che sta per *Multi-Layer Perceptron* (Percettrone Multistrato).
[02:44] Questa terminologia può essere fuorviante. Per definizione, un percettrone è esattamente la struttura descritta finora.
[02:53] Di conseguenza, un "percettrone multistrato" dovrebbe indicare una serie di queste strutture impilate una dopo l'altra.
[02:59] In pratica, tuttavia, l'acronimo MLP viene spesso usato per denotare le reti neurali in generale. Dal punto di vista strettamente formale, ciò non è del tutto corretto, poiché il percettrone è un tipo molto specifico di neurone, non un neurone generico. È importante essere consapevoli di questo uso esteso del termine.
[03:13] Viene qui presentata una tabella relativa a una delle porte logiche più note, la porta **NAND** (NOT-AND). Questa si ottiene calcolando la funzione logica AND e negandone il risultato.
[03:26] La porta NAND è importante perché è un esempio di **porta logica universale**.
- **Definizione di Porta Logica Universale**: Una porta logica è detta universale se può essere utilizzata, tramite opportune combinazioni, per costruire qualsiasi altra porta logica (come NOT, AND, OR, ecc.).
[03:37] Combinando più porte NAND, è possibile creare qualsiasi circuito logico. Ad esempio, la porta NOT si ottiene collegando entrambi gli input di una porta NAND a un unico segnale: se l'input è 0, l'output è 1; se l'input è 1, l'output è 0.
[03:53] Per creare una porta AND, si può utilizzare una porta NAND seguita da una porta NOT (a sua volta costruita con una porta NAND). Impilando due porte NAND in questo modo, si può ottenere il comportamento desiderato.
[04:07] Il motivo per cui si richiama questa tabella è che un percettrone può essere visto come una porta logica. In particolare, scegliendo in modo appropriato i pesi e il bias, è possibile replicare esattamente il comportamento della porta NAND.
[04:19] Se i pesi degli input sono entrambi impostati a -2 e il bias è impostato a 3, il percettrone si comporta esattamente come una porta NAND.
[04:26] Da ciò si potrebbe dedurre che, essendo la porta NAND universale e potendo il percettrone implementarla, in linea di principio si possa creare qualsiasi circuito logico.
[04:36] Sebbene questo sia vero, il punto cruciale è che finora si è assunto che i pesi e il bias siano dati e scelti appositamente per rappresentare la porta NAND.
[04:46] In pratica, l'obiettivo è **apprendere** (*to learn*) qualcosa. Apprendere significa non decidere a priori i valori dei pesi e del bias, ma lasciare che il sistema li determini autonomamente per modellare comportamenti più generali che legano l'input all'output.
### Limiti del Percettrone e Introduzione alla Funzione Sigmoide
[04:59] Esiste un'altra importante problematica legata al percettrone. L'obiettivo è l'apprendimento, ma la funzione che descrive il percettrone produce un output che può essere solo 0 o 1.
[05:08] Non esiste una transizione graduale tra i due stati; si tratta di un salto netto.
[05:14] Questo implica che una piccola variazione nel vettore di input $x$, quando la somma ponderata è vicina a zero, può causare un salto improvviso dell'output da 0 a 1 (o viceversa).
[05:21] In sostanza, la funzione di output del percettrone non è continua.
[05:26] Come discusso nella lezione precedente, per addestrare una rete neurale e farle apprendere un compito, è necessario calcolare le **sensitività**, ovvero le derivate della funzione di costo rispetto a tutti i parametri della rete (pesi e bias).
[05:42] È evidente che se la funzione di costo, o le funzioni in essa coinvolte, non sono continue, il calcolo delle derivate diventa problematico. Un output che salta bruscamente da 0 a 1 rende difficile ottenere un comportamento stabile e prevedibile.
[05:52] L'utilizzo di un algoritmo come la discesa del gradiente (*gradient descent*) con una funzione di questo tipo sarebbe estremamente complicato.
[05:59] L'idea è quindi quella di "rilassare" questo salto, introducendo una transizione più graduale tra i due stati.
[06:06] Questo permetterebbe di modificare gradualmente i pesi e il bias in base ai risultati del processo di minimizzazione della funzione di costo.
[06:12] Si vuole passare dalla descrizione discontinua del percettrone a una funzione più pratica e gestibile.
[06:20] Un'altra ragione per cui il percettrone non è utilizzato nella pratica è che il suo comportamento è molto limitato. Si desidera avere la possibilità di descrivere comportamenti più complessi della funzione che lega input e output.
## Capitolo 3: La Funzione Sigmoide e la Struttura delle Reti Neurali
### La Funzione Sigmoide
[06:32] La prima possibilità per superare il salto del percettrone è utilizzare la cosiddetta **funzione sigmoide**.
[06:37] La funzione sigmoide è un'istanza specifica di una classe più ampia di funzioni, chiamate **funzioni sigmoidali**.
- **Definizione di Funzione Sigmoidale**: Una funzione $\sigma(x)$ è detta sigmoidale se soddisfa le seguenti proprietà:
$$ \lim_{x \to -\infty} \sigma(x) = 0 $$
$$ \lim_{x \to +\infty} \sigma(x) = 1 $$
[06:58] Una funzione con un andamento a "S" è un esempio di funzione sigmoidale. Esistono molte funzioni con questo comportamento, e questa definizione sarà ripresa nella parte finale del corso, quando si tratterà il teorema di approssimazione universale per le reti neurali.
[07:12] Per ora, ci concentriamo sulla funzione sigmoide. È un'istanza particolare di questa classe di funzioni, e l'idea è di avere una funzione che si comporti in modo simile al percettrone.
[07:21] Si mantiene la stessa struttura di base: un certo numero di input $x_1, \dots, x_n$, i pesi corrispondenti e il bias.
[07:27] Per prima cosa, si costruisce la variabile ausiliaria $z$, che è la somma ponderata degli input più il bias, esattamente come prima: $z = w \cdot x + b$.
[07:35] Le differenze rispetto al percettrone sono due:
1.  Gli input e l'output non sono più vincolati a essere binari, ma possono assumere qualsiasi valore reale. Ad esempio, i valori dei pixel di un'immagine in scala di grigi possono essere un input valido.
2.  L'output è calcolato tramite la funzione sigmoide.
[07:52] L'espressione analitica della funzione sigmoide è:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
dove $z$ è la variabile definita in precedenza.
[07:59] Come si può osservare dal suo grafico, questa funzione ha esattamente il comportamento sigmoidale desiderato. In particolare, per $z=0$, la funzione assume il valore 0.5.
[08:06] La funzione sigmoide è il primo esempio di **funzione di attivazione**. È stata la prima funzione introdotta per risolvere i problemi legati al percettrone.
### Vantaggi della Funzione Sigmoide
[08:16] La proprietà di continuità e derivabilità (*smoothness*) che si cercava è garantita dalla funzione sigmoide.
[08:22] Questo assicura che una piccola variazione nei pesi o nel bias produca una variazione controllabile nell'output.
[08:29] Questa caratteristica è fondamentale per l'apprendimento. Durante l'addestramento, si minimizza una funzione di costo per trovare l'insieme di pesi e bias che ne individua il minimo.
[08:40] Per eseguire questa minimizzazione, si utilizzano solitamente algoritmi basati sulla discesa del gradiente, che richiedono il calcolo delle derivate.
[08:48] Grazie alla continuità della funzione sigmoide, l'applicazione di tali algoritmi diventa possibile e affidabile.
### Struttura Generale di una Rete Neurale
[08:56] Verrà ora descritta la struttura generica di una rete neurale. In un'architettura classica, si distinguono tre tipi di strati (*layer*).
[09:04] 1.  **Strato di Input (Input Layer)**: È il primo strato della rete. È composto da un numero di neuroni pari al numero di caratteristiche (*features*) dell'input.
[09:12] 2.  **Strato di Output (Output Layer)**: È l'ultimo strato e produce l'output finale della rete. Può essere composto da un solo neurone o da più neuroni.
[09:19] 3.  **Strati Nascosti (Hidden Layers)**: Si trovano tra lo strato di input e quello di output. Una rete può avere uno o più strati nascosti.
[09:25] - **Rete Shallow**: Una rete neurale con un solo strato nascosto è detta *shallow*.
- **Rete Deep**: Una rete con più di uno strato nascosto è detta *deep*.
[09:34] Queste definizioni saranno riprese nella parte finale del corso, quando si discuterà la complessità delle reti neurali e i risultati teorici correlati, a seconda che siano di tipo *shallow* o *deep*.
[09:45] Il numero di strati nascosti e il numero di neuroni in ciascuno di essi dipendono fortemente dal problema da risolvere. Possono variare da poche decine a milioni.
[09:55] Graficamente, l'idea è quella di una successione di strati. In questa rappresentazione, emerge un'altra caratteristica importante: per il momento, si assume che la rete sia **completamente connessa** (*fully connected*).
- **Definizione di Rete Completamente Connessa**: Significa che ogni neurone in uno strato $L$ è connesso a ogni neurone dello strato successivo $L+1$.
[10:09] Un'altra osservazione riguarda il flusso delle informazioni. Questa rete è detta **feedforward**.
- **Definizione di Rete Feedforward**: L'informazione si propaga in una sola direzione, dallo strato di input a quello di output, come indicato dalle frecce nel diagramma.
[10:17] Per ora, non si considerano architetture più complesse, come quelle con cicli chiusi, note come **reti neurali ricorrenti** (*recurrent neural networks*), o altre tipologie. L'attenzione è focalizzata sull'architettura più classica.
[10:31] In sintesi, il percettrone non è adatto per l'apprendimento; questo limite è stato superato introducendo la funzione sigmoide. È stata poi fornita un'introduzione generale alla terminologia usata per descrivere l'architettura di una rete neurale.
## Capitolo 4: Formalizzazione Matematica e Backpropagation
### Notazione per Pesi, Bias e Attivazioni
[10:41] L'obiettivo ora è descrivere formalmente come calcolare le sensitività necessarie per determinare i valori dei pesi e dei bias che minimizzano una data funzione di costo.
[10:55] Per fare ciò, è necessario introdurre una notazione specifica.
- **Peso $W_{jk}^{(L)}$**: Rappresenta il peso della connessione tra il neurone $k$ nello strato $L-1$ e il neurone $j$ nello strato $L$.
[11:06] A prima vista, l'ordine degli indici può sembrare strano (con $k$ nello strato $L-1$ e $j$ nello strato $L$), ma la ragione di questa scelta diventerà chiara in seguito.
[11:13] È importante ricordare che i pesi sono associati alle connessioni (gli "archi" del grafo), non ai neuroni.
[11:25] - **Bias $b_j^{(L)}$**: Rappresenta il bias del neurone $j$ nello strato $L$. A differenza dei pesi, i bias sono associati a ciascun singolo neurone.
[11:35] - **Attivazione $a_j^{(L)}$**: È una variabile associata a ciascun neurone, che rappresenta il suo output.
[11:46] La sua definizione è la seguente:
$$ a_j^{(L)} = \sigma \left( \sum_k W_{jk}^{(L)} a_k^{(L-1)} + b_j^{(L)} \right) $$
In pratica, l'attivazione è il risultato dell'applicazione della funzione di attivazione $\sigma$ (per ora, la sigmoide) alla somma ponderata degli input provenienti dallo strato precedente, più il bias.
[12:01] La variabile $\sigma$ è la **funzione di attivazione**.
### Rappresentazione Vettoriale
[12:04] Per i calcoli, è più conveniente rappresentare queste quantità in forma vettoriale.
- **Matrice dei Pesi $W^{(L)}$**: È la matrice che contiene tutti i pesi che connettono lo strato $L-1$ allo strato $L$. Ogni coppia di strati adiacenti ha la sua matrice dei pesi.
- **Vettore dei Bias $b^{(L)}$**: È il vettore contenente i bias di tutti i neuroni dello strato $L$.
- **Vettore delle Attivazioni $a^{(L)}$**: È il vettore contenente le attivazioni di tutti i neuroni dello strato $L$.
[12:22] L'argomento della funzione di attivazione è solitamente denotato con $z^{(L)}$, un vettore definito come:
$$ z^{(L)} = W^{(L)} a^{(L-1)} + b^{(L)} $$
Questa espressione rappresenta la somma ponderata più il bias in forma matriciale. È il prodotto matrice-vettore tra la matrice dei pesi $W^{(L)}$ e il vettore delle attivazioni dello strato precedente $a^{(L-1)}$, a cui si somma il vettore dei bias $b^{(L)}$.
[12:38] Di conseguenza, il vettore delle attivazioni $a^{(L)}$ può essere scritto come:
$$ a^{(L)} = \sigma(z^{(L)}) $$
dove la funzione di attivazione $\sigma$ si intende applicata a ciascun elemento del vettore $z^{(L)}$ (operazione *element-wise*).
[12:47] Riassumendo visivamente: la rete è composta da strati di neuroni. A ogni neurone sono associati un'attivazione e un bias. A ogni connessione tra neuroni è associato un peso. Tra due strati adiacenti, ad esempio il primo e il secondo, è definita una matrice dei pesi, come $W^{(2)}$.
### Calcolo delle Sensitività: Backpropagation
[13:01] Una volta definito il framework, l'obiettivo è calcolare le sensitività della funzione di costo rispetto a tutti i parametri della rete.
[13:10] Prima di entrare nei dettagli, è necessario formulare due ipotesi importanti sulla funzione di costo, che indicheremo con $J$.
[13:17] **Prima Ipotesi**: La funzione di costo può essere scritta come una media dei costi calcolati su singoli campioni:
$$ J = \frac{1}{n} \sum_{x} J_x $$
dove $J_x$ è il costo associato a un singolo campione di addestramento $x$.
[13:22] Per comprendere meglio, si consideri un esempio: l'addestramento di una rete neurale per la classificazione di immagini. Le immagini vengono pre-elaborate (ridimensionate, trasformate in vettori, ecc.) e ogni caratteristica viene normalizzata, ad esempio, in un valore tra 0 e 1.
[13:38] Il *training set* è composto da migliaia di immagini, ciascuna con la sua etichetta (*label*) corretta.
[13:44] In questo contesto, la sommatoria si estende su tutti i campioni (le immagini). Il termine $J_x$ rappresenta una misura dell'errore per un singolo campione, come la differenza tra l'etichetta predetta dalla rete e quella vera.
[13:54] Si sta quindi calcolando una media dei costi su tutto il *training set*. Questa è un'assunzione comune sia per problemi di classificazione che di regressione.
[14:08] Nel caso di un problema di regressione, si calcolerebbe la differenza (ad esempio, la norma) tra l'output atteso e quello calcolato dalla rete per ciascun campione del *training set*.
[14:16] In sintesi, la prima ipotesi è che la funzione di costo sia la somma (o la media) di $n$ contributi, ciascuno relativo a un campione del *training set*.
[14:23] **Seconda Ipotesi**: La funzione di costo può essere scritta come una funzione che dipende unicamente dalle attivazioni dello strato di output, $a^{(L)}$.
[14:31] Ovviamente, ciò che accade nell'ultimo strato è fortemente influenzato da tutti gli strati precedenti. Tuttavia, l'idea è che per calcolare il valore della funzione di costo sia sufficiente conoscere l'output finale della rete.
[14:39] Ad esempio, si supponga che $y$ sia il vettore degli output veri (le etichette) e $a^{(L)}$ sia il vettore degli output calcolati dalla rete. Una possibile funzione di costo potrebbe essere la distanza tra questi due vettori:
$$ J = \| y - a^{(L)} \|^2 $$
[14:52] Queste due ipotesi sono abbastanza naturali e non troppo restrittive, valide nel 99% delle applicazioni comuni.
[15:00] Sarà necessaria un'ulteriore notazione: il **prodotto di Hadamard** (o prodotto *element-wise*), indicato con $\odot$.
- **Definizione di Prodotto di Hadamard**: Dati due vettori della stessa dimensione, il loro prodotto di Hadamard è un nuovo vettore i cui elementi sono il prodotto degli elementi corrispondenti dei vettori originali.
[15:09] In MATLAB, questa operazione corrisponde a `.*` (*dot star*).
[15:14] In Python, con la libreria NumPy, l'operatore `*` applicato a due vettori NumPy esegue il prodotto di Hadamard. Per calcolare il prodotto scalare, invece, si deve usare la funzione `np.dot()`.
[15:25] In pratica, dati due vettori $u = [u_1, u_2, \dots, u_n]$ e $v = [v_1, v_2, \dots, v_n]$, il loro prodotto di Hadamard è:
$$ u \odot v = [u_1 v_1, u_2 v_2, \dots, u_n v_n] $$
[15:33] Si procederà ora a derivare quello che in letteratura è noto come l'insieme delle **quattro equazioni fondamentali** per il calcolo delle sensitività (algoritmo di *backpropagation*).
[15:41] Il punto più importante è l'introduzione di una quantità, chiamata in alcune referenze "errore" e in altre "sensitività". Questa quantità, che indicheremo con $\delta_j^{(L)}$, è associata a un dato neurone $j$ nello strato $L$.
[15:52] Essa fornisce la sensitività della funzione di costo $J$ rispetto alla variabile $z_j^{(L)}$ associata a quel particolare neurone. Si ricorda che $z$ è la somma ponderata degli input più il bias.
$$ \delta_j^{(L)} = \frac{\partial J}{\partial z_j^{(L)}} $$
[16:03] L'obiettivo finale è ottenere formule che permettano di calcolare le derivate della funzione di costo rispetto a tutti i pesi ($W$) e a tutti i bias ($b$) della rete.
[16:11] Queste quantità sono esattamente quelle utilizzate in un algoritmo di ottimizzazione come la discesa del gradiente. In una procedura di discesa del gradiente, è necessario calcolare il gradiente della funzione di costo rispetto alle incognite, che in questo caso sono appunto i pesi e i bias.
[16:24] Si inizierà la derivazione partendo dallo strato di output.
## Capitolo 5: Le Equazioni Fondamentali della Backpropagation
### L'Errore nello Strato di Output
[00:00] L'errore, o sensibilità, indicato dal vettore $\delta^L$ (con $L$ maiuscolo per denotare lo strato di output), è definito dalla seguente relazione: è dato dal gradiente della funzione di costo rispetto all'attivazione, moltiplicato per la derivata prima della funzione di attivazione, $\sigma'$, calcolata in $z^L$.
$$
\delta^L = \nabla_a J \odot \sigma'(z^L)
$$
È importante ricordare che $z^L$ rappresenta la somma pesata degli input più il bias, mentre $\sigma$ è la funzione di attivazione. Di conseguenza, $\sigma'$ è la derivata di $\sigma$ rispetto al suo argomento, ovvero $z^L$.
[00:15] Analizzando i termini di questo prodotto, si osserva che il primo termine, il gradiente di $J$ rispetto ad $a$ ($\nabla_a J$), misura la rapidità con cui la funzione di costo varia al variare dell'attivazione $a$. Il secondo termine, $\sigma'$, misura invece la rapidità con cui la funzione di attivazione $\sigma$ varia in risposta a variazioni del suo input $z$.
[00:32] Esprimendo questa relazione in forma di componenti, si ottiene che la componente $j$-esima del vettore di errore $\delta$ nello strato di output $L$ è data da:
$$
\delta_j^L = \frac{\partial J}{\partial a_j^L} \cdot \sigma'(z_j^L)
$$
dove:
- $\delta_j^L$ è la sensibilità del neurone $j$ nello strato di output $L$.
- $\frac{\partial J}{\partial a_j^L}$ è la derivata parziale della funzione di costo $J$ rispetto all'attivazione $a_j$ del neurone $j$ nello strato $L$.
- $\sigma'(z_j^L)$ è la derivata della funzione di attivazione calcolata nella somma pesata $z_j$ del neurone $j$ nello strato $L$.
### Il Problema del Gradiente Evanescente (Vanishing Gradient)
[00:44] Un aspetto di fondamentale importanza pratica in questa formula è la presenza del termine $\sigma'$. Per comprendere le sue implicazioni, è utile ricordare la forma grafica tipica della funzione di attivazione sigmoide, che presenta una curva a "S".
[00:56] Dall'analisi del grafico, è evidente che nelle regioni estreme, dove la funzione si appiattisce (sia per valori di input molto negativi che molto positivi), la derivata $\sigma'$ è quasi nulla.
[01:07] Questo ha una conseguenza diretta: se $\sigma'$ è zero o molto vicino a zero, anche il valore di $\delta$ sarà nullo o trascurabile.
[01:15] Dal punto di vista pratico, ciò significa che un peso sinaptico (*weight*) nell'ultimo strato apprenderà molto lentamente, ovvero si modificherà in modo quasi impercettibile, se il neurone di output corrispondente si trova in uno stato di attivazione molto basso o molto alto.
[01:27] Un neurone la cui attivazione si trova in una di queste regioni di appiattimento è definito "neurone saturo". Per un neurone saturo, non si verifica quasi alcuna variazione nei pesi e nei bias ad esso associati, proprio perché il termine $\sigma'$ è essenzialmente zero.
[01:42] Questo fenomeno è noto come **problema del gradiente evanescente** (vanishing gradient problem), un concetto chiave nel contesto del processo di apprendimento delle reti neurali. Il problema nasce proprio dalla presenza del termine $\sigma'$ nelle equazioni della sensibilità.
[01:54] In sintesi, se $\sigma'$ è prossimo a zero, il vettore di errore (o sensibilità) sarà composto da componenti prossime a zero. Di conseguenza, il neurone non aggiornerà in modo significativo i valori dei suoi pesi e bias, rallentando o bloccando di fatto l'apprendimento.
### Propagazione dell'Errore all'Indietro
[02:08] Dopo aver calcolato la sensibilità $\delta$ per lo strato di output, il passo successivo è propagare questo errore all'indietro, procedendo dallo strato finale ($L$) verso gli strati precedenti ($L-1$, $L-2$, e così via, fino al primo strato).
[02:22] L'obiettivo è calcolare il vettore di sensibilità $\delta$ per un generico strato $l$, dato il vettore di sensibilità $\delta$ dello strato successivo $l+1$.
[02:32] La formula per ottenere il vettore di sensibilità $\delta$ per lo strato $l$ è la seguente:
$$
\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)
$$
dove:
- $\delta^l$ è il vettore di sensibilità dello strato $l$.
- $(W^{l+1})^T$ è la trasposta della matrice dei pesi dello strato $l+1$.
- $\delta^{l+1}$ è il vettore di sensibilità dello strato $l+1$, già calcolato.
- $\sigma'(z^l)$ è il vettore delle derivate della funzione di attivazione calcolate nelle somme pesate $z$ dello strato $l$.
- $\odot$ indica il prodotto di Hadamard (prodotto elemento per elemento).
[02:44] A breve verrà fornita la dimostrazione di queste equazioni. Per il momento, l'attenzione è rivolta alla presentazione e al commento dei risultati.
[02:51] Un'osservazione importante riguarda la convenzione utilizzata per gli indici della matrice dei pesi $W$. L'uso della matrice trasposta $(W^{l+1})^T$ chiarisce il motivo di tale scelta.
[03:01] Se si osserva la struttura della rete, la trasposizione della matrice dei pesi permette di collegare i neuroni di uno strato con quelli dello strato precedente.
[03:10] Poiché l'algoritmo di backpropagation è una procedura che si muove all'indietro (da destra a sinistra, dallo strato di output a quello di input), l'adozione di questa notazione con la trasposta risulta naturale e conveniente. In alternativa, si sarebbe potuto definire fin dall'inizio la matrice dei pesi con indici invertiti ($kj$ invece di $jk$) per evitare l'uso della trasposta, ma la convenzione presentata è quella comunemente adottata.
### Calcolo dei Gradienti per l'Aggiornamento
[03:24] Sfruttando l'equazione per lo strato di output e quella per la propagazione ricorsiva, è possibile calcolare i vettori di sensibilità $\delta$ per tutti gli strati della rete.
[03:37] Tuttavia, il calcolo non è ancora completo. L'obiettivo finale è determinare i gradienti della funzione di costo rispetto a tutti i parametri della rete, ovvero $\frac{\partial J}{\partial w}$ e $\frac{\partial J}{\partial b}$, che sono gli ingredienti necessari per l'algoritmo di minimizzazione (come la discesa del gradiente).
[03:49] È quindi necessario collegare le quantità $\delta$ appena calcolate con i gradienti che si desidera ottenere.
[04:00] Si inizia considerando il gradiente rispetto ai bias. È possibile dimostrare che la derivata della funzione di costo $J$ rispetto a un generico bias della rete è esattamente uguale alla componente corrispondente del vettore di errore (o sensibilità).
$$
\frac{\partial J}{\partial b_j^l} = \delta_j^l
$$
[04:07] Questo significa che, una volta calcolato $\delta_j^l$, si è ottenuto direttamente il valore della derivata parziale di $J$ rispetto al bias $b_j^l$ di quel particolare neurone.
[04:16] Per quanto riguarda i pesi, la relazione è leggermente diversa. La derivata parziale della funzione di costo $J$ rispetto al peso $w_{jk}^l$, che connette il neurone $k$ dello strato $l-1$ con il neurone $j$ dello strato $l$, è data da:
$$
\frac{\partial J}{\partial w_{jk}^l} = a_k^{l-1} \cdot \delta_j^l
$$
[04:28] Questa derivata è il prodotto di due quantità: l'attivazione $a_k^{l-1}$ del neurone $k$ nello strato precedente ($l-1$) e la sensibilità $\delta_j^l$ del neurone $j$ nello strato corrente ($l$).
[04:37] In sintesi, sono state presentate quattro equazioni fondamentali che, sfruttando il calcolo delle sensibilità $\delta$, permettono di ottenere i gradienti necessari per l'algoritmo di minimizzazione.
## Capitolo 6: Dimostrazioni delle Equazioni di Backpropagation
### Dimostrazione 1: Errore nello Strato di Output
[04:47] Si procede ora con la dimostrazione delle relazioni introdotte, partendo dalla prima: l'equazione per l'errore $\delta_j^L$ nello strato di output.
[04:53] Si parte dalla definizione di sensibilità, che è la derivata parziale della funzione di costo $J$ rispetto alla somma pesata $z$ di un particolare neurone:
$$
\delta_j^L = \frac{\partial J}{\partial z_j^L}
$$
[05:01] Per calcolare questa derivata, si applica la regola della catena (chain rule), un principio già utilizzato in contesti come la differenziazione automatica. Un neurone è connesso a tutti i neuroni dello strato precedente, ma in questo caso si analizza la relazione tra $J$ e $z_j^L$ attraverso l'attivazione $a_k^L$.
[05:10] La derivata può essere espressa come una somma estesa a tutti i neuroni $k$ dello strato di output $L$:
$$
\frac{\partial J}{\partial z_j^L} = \sum_k \frac{\partial J}{\partial a_k^L} \frac{\partial a_k^L}{\partial z_j^L}
$$
[05:27] Si ricorda che l'attivazione $a_k^L$ è ottenuta applicando la funzione di attivazione $\sigma$ alla somma pesata $z_k^L$. La derivata $\frac{\partial a_k^L}{\partial z_j^L}$ è quindi diversa da zero solo quando gli indici $k$ e $j$ coincidono.
[05:36] In dettaglio:
- Se $k \neq j$, la derivata $\frac{\partial a_k^L}{\partial z_j^L}$ è zero, poiché l'attivazione del neurone $k$ non dipende direttamente dalla somma pesata del neurone $j$.
- Se $k = j$, la derivata $\frac{\partial a_j^L}{\partial z_j^L}$ è $\sigma'(z_j^L)$.
[05:53] Di conseguenza, nella sommatoria sopravvive unicamente il termine per cui $k = j$.
[06:02] Sostituendo questo risultato nella sommatoria, si ottiene:
$$
\delta_j^L = \frac{\partial J}{\partial z_j^L} = \frac{\partial J}{\partial a_j^L} \cdot \frac{\partial a_j^L}{\partial z_j^L} = \frac{\partial J}{\partial a_j^L} \cdot \sigma'(z_j^L)
$$
[06:09] Questo completa la dimostrazione della prima equazione. In formato vettoriale, questa relazione può essere espressa utilizzando il prodotto di Hadamard, come già menzionato.
### Dimostrazione 2: Propagazione dell'Errore
[06:20] Si passa alla seconda equazione, quella che descrive la propagazione dell'errore all'indietro. L'obiettivo è trovare un'espressione per la sensibilità $\delta_j^l$ di un generico neurone $j$ in un generico strato $l$.
[06:30] Si parte nuovamente dalla definizione di sensibilità:
$$
\delta_j^l = \frac{\partial J}{\partial z_j^l}
$$
[06:35] Applicando la regola della catena, si può esprimere questa derivata attraverso le somme pesate dello strato successivo, $l+1$:
$$
\delta_j^l = \frac{\partial J}{\partial z_j^l} = \sum_k \frac{\partial J}{\partial z_k^{l+1}} \frac{\partial z_k^{l+1}}{\partial z_j^l}
$$
[06:42] In questa espressione, si riconosce che il termine $\frac{\partial J}{\partial z_k^{l+1}}$ è per definizione la sensibilità $\delta_k^{l+1}$ del neurone $k$ nello strato $l+1$. Poiché l'algoritmo procede all'indietro, questo valore è già stato calcolato.
[06:54] Ora è necessario calcolare il secondo termine della produttoria: $\frac{\partial z_k^{l+1}}{\partial z_j^l}$. Si scrive l'espressione per $z_k^{l+1}$:
$$
z_k^{l+1} = \sum_i w_{ki}^{l+1} a_i^l + b_k^{l+1}
$$
[07:06] Poiché l'attivazione $a_i^l$ è $\sigma(z_i^l)$, si può sostituire questa espressione:
$$
z_k^{l+1} = \sum_i w_{ki}^{l+1} \sigma(z_i^l) + b_k^{l+1}
$$
[07:14] Calcolando la derivata di $z_k^{l+1}$ rispetto a $z_j^l$, si osserva che:
- Il bias $b_k^{l+1}$ non dipende da $z_j^l$, quindi la sua derivata è zero.
- Nella sommatoria, l'unico termine che dipende da $z_j^l$ è quello per cui $i = j$.
[07:23] Pertanto, la derivata risulta essere:
$$
\frac{\partial z_k^{l+1}}{\partial z_j^l} = w_{kj}^{l+1} \cdot \sigma'(z_j^l)
$$
[07:40] Sostituendo questo risultato nell'espressione iniziale per $\delta_j^l$, si ottiene:
$$
\delta_j^l = \sum_k \delta_k^{l+1} \cdot w_{kj}^{l+1} \cdot \sigma'(z_j^l)
$$
[07:50] Il termine $\sigma'(z_j^l)$ non dipende dall'indice di sommatoria $k$, quindi può essere portato fuori dalla somma:
$$
\delta_j^l = \left( \sum_k w_{kj}^{l+1} \delta_k^{l+1} \right) \cdot \sigma'(z_j^l)
$$
[07:58] La sommatoria tra parentesi corrisponde alla componente $j$-esima del prodotto matrice-vettore $(W^{l+1})^T \delta^{l+1}$. Questo conferma la seconda equazione, che permette di propagare le sensibilità all'indietro.
### Dimostrazione 3: Gradiente rispetto ai Bias
[08:12] La terza dimostrazione riguarda il gradiente della funzione di costo rispetto ai bias. Si vuole calcolare $\frac{\partial J}{\partial b_j^l}$.
[08:18] Utilizzando la regola della catena, si scrive:
$$
\frac{\partial J}{\partial b_j^l} = \frac{\partial J}{\partial z_j^l} \frac{\partial z_j^l}{\partial b_j^l}
$$
[08:23] Per definizione, il primo termine $\frac{\partial J}{\partial z_j^l}$ è la sensibilità $\delta_j^l$.
[08:27] Per il secondo termine, si ricorda che $z_j^l$ è la somma pesata più il bias:
$$
z_j^l = \sum_k w_{jk}^l a_k^{l-1} + b_j^l
$$
[08:33] La derivata di $z_j^l$ rispetto a $b_j^l$ è semplicemente 1.
[08:37] Sostituendo, si ottiene la relazione cercata:
$$
\frac{\partial J}{\partial b_j^l} = \delta_j^l \cdot 1 = \delta_j^l
$$
### Dimostrazione 4: Gradiente rispetto ai Pesi
[08:46] Infine, si dimostra la relazione per il gradiente rispetto ai pesi, $\frac{\partial J}{\partial w_{jk}^l}$.
[08:50] Anche in questo caso, si applica la regola della catena:
$$
\frac{\partial J}{\partial w_{jk}^l} = \frac{\partial J}{\partial z_j^l} \frac{\partial z_j^l}{\partial w_{jk}^l}
$$
[08:54] Il primo termine è, ancora una volta, la sensibilità $\delta_j^l$.
[08:58] Per il secondo termine, si considera l'espressione di $z_j^l$:
$$
z_j^l = \sum_i w_{ji}^l a_i^{l-1} + b_j^l
$$
[09:04] Quando si calcola la derivata rispetto a un peso specifico $w_{jk}^l$, l'unico termine della sommatoria che contribuisce è quello per cui l'indice $i$ è uguale a $k$.
[09:08] La derivata è quindi:
$$
\frac{\partial z_j^l}{\partial w_{jk}^l} = a_k^{l-1}
$$
[09:12] Sostituendo questo risultato nell'equazione del gradiente, si ottiene:
$$
\frac{\partial J}{\partial w_{jk}^l} = \delta_j^l \cdot a_k^{l-1}
$$
[09:22] Questa è l'ultima delle quattro relazioni fondamentali, che conclude le dimostrazioni.
### Riepilogo dell'Algoritmo di Backpropagation
[09:25] L'algoritmo di backpropagation può essere riassunto nei seguenti passaggi:
[09:50] 1.  **Inizializzazione**: Prima di iniziare, è necessario inizializzare i pesi e i bias della rete. Si possono impostare tutti a zero o utilizzare metodi più sofisticati per trovare una buona inizializzazione.
[09:58] Il valore iniziale dei bias è generalmente meno critico di quello dei pesi.
[10:03] Librerie pratiche come TensorFlow o PyTorch implementano metodi specifici per l'inizializzazione dei pesi. Solitamente, i bias vengono inizializzati a zero, poiché hanno un impatto minore sul corretto funzionamento dell'algoritmo di discesa del gradiente.
[09:30] 2.  **Forward Pass (Propagazione in Avanti)**: Si fornisce un vettore di input (ad esempio, le feature di un'immagine) e si esegue una propagazione in avanti attraverso la rete.
[09:33] Questo passaggio è necessario per calcolare e memorizzare i valori di tutte le attivazioni ($a^l$) e di tutte le somme pesate ($z^l$) per ogni strato.
[09:39] Questa fase è analoga a quanto visto nel calcolo delle derivate con la differenziazione automatica, dove era necessario calcolare tutti i passaggi intermedi in avanti.
[09:47] 3.  **Backward Pass (Propagazione all'Indietro)**: Una volta ottenuti i valori dal forward pass, si avvia la procedura di backpropagation.
[10:18] Partendo dall'ultimo strato e procedendo a ritroso fino al primo, si applicano le quattro equazioni dimostrate in precedenza per calcolare prima le sensibilità $\delta^l$ e poi i gradienti.
[10:25] 4.  **Output**: L'output dell'algoritmo è l'insieme dei gradienti della funzione di costo rispetto a tutti i pesi e a tutti i bias della rete ($\frac{\partial J}{\partial w}$ e $\frac{\partial J}{\partial b}$). Questi gradienti verranno poi utilizzati da un algoritmo di ottimizzazione (es. discesa del gradiente) per aggiornare i parametri della rete e minimizzare la funzione di costo.
[10:32] A titolo di confronto, si può considerare l'alternativa di calcolare i gradienti tramite differenziazione numerica.
[10:39] Senza entrare nei dettagli, che sono analoghi a quelli già discussi in precedenza, e tralasciando le considerazioni sull'aritmetica in virgola mobile, è sufficiente notare che il numero di operazioni richieste dalla differenziazione numerica è significativamente superiore a quello richiesto dall'algoritmo di backpropagation.
[10:52] La backpropagation rappresenta quindi un metodo molto più efficiente per calcolare i gradienti in una rete neurale.
## Capitolo 7: Panoramica delle Funzioni di Attivazione
### Introduzione alle Funzioni di Attivazione
[00:00] Le quattro relazioni fondamentali ottenute per l'algoritmo di retropropagazione dell'errore (backpropagation) coinvolgono una funzione di attivazione, indicata con $\sigma$. Tuttavia, $\sigma$ non deve essere necessariamente la funzione sigmoide. Può essere una qualsiasi funzione di attivazione.
[00:11] Verranno ora introdotte diverse funzioni di attivazione comunemente utilizzate in letteratura, analizzandone i rispettivi vantaggi e svantaggi.
### Funzioni di Attivazione Storiche e i Loro Limiti
[00:20] La prima funzione analizzata è quella utilizzata nel percettrone, che è essenzialmente una funzione a gradino (step function) con valori di output 0 e 1.
[00:25] L'utilizzo di questa funzione nell'algoritmo della discesa del gradiente rappresenta una sfida significativa. La sua derivata è una delta di Dirac nell'origine e zero in tutti gli altri punti.
[00:33] Di conseguenza, ci si trova quasi sempre in una situazione di "vanishing gradient" (gradiente che svanisce), un fenomeno in cui il gradiente diventa così piccolo da rendere l'apprendimento della rete estremamente lento o nullo.
[00:40] Subito dopo il percettrone, un primo tentativo per superare i limiti della funzione a gradino è stato l'impiego di una funzione di attivazione lineare.
[00:46] La funzione lineare è semplice e la sua derivata è costante, il che elimina il problema del gradiente che svanisce.
[00:51] Tuttavia, presenta un limite importante: essendo lineare, la composizione di più funzioni lineari rimane sempre una funzione lineare.
[00:57] Questo significa che, indipendentemente dal numero di strati (layer) o di neuroni utilizzati, la rete neurale nel suo complesso si comporterà sempre come un modello lineare.
[01:04] Pertanto, se si desidera modellare un fenomeno con una relazione input-output fortemente non lineare, l'uso di una funzione di attivazione lineare non consentirà di raggiungere l'obiettivo.
### Funzioni di Attivazione Sigmoidali
[01:11] La funzione sigmoide, già considerata in precedenza, presenta alcuni svantaggi. Il primo è il problema del "vanishing gradient", che si manifesta quando i valori di input sono molto grandi o molto piccoli, portando la derivata a zero.
[01:16] Un altro svantaggio è che non è centrata sullo zero ("not zero-centered"). Il suo valore per un input nullo è 0.5, non 0.
[01:20] Questo può introdurre una distorsione (bias) verso un particolare segno dei pesi durante l'addestramento, rendendola non sempre la scelta ottimale.
[01:26] D'altra parte, poiché il suo output è compreso nell'intervallo $[0, 1]$, può essere interpretato come una probabilità.
[01:30] Ad esempio, se utilizzata nello strato di output con un singolo neurone, può fornire la probabilità che l'output appartenga a una determinata classe (es. "gatto" o "non gatto").
[01:37] Una funzione simile alla sigmoide è la tangente iperbolica (`tanh`). Non è una funzione sigmoide in senso stretto, poiché per $x$ che tende a meno infinito, la funzione tende a -1.
[01:44] Il suo vantaggio principale è di essere centrata sullo zero ("zero-centered"), il che significa che durante la procedura di minimizzazione non si introduce una distorsione sistematica verso un segno specifico per i pesi.
[01:52] Questo aspetto può essere vantaggioso in determinate situazioni.
[01:55] Poiché il suo andamento è molto simile a quello della sigmoide, soffre anch'essa del problema del "vanishing gradient" per valori di input molto alti o molto bassi.
[02:01] La tangente iperbolica è ampiamente utilizzata negli strati nascosti (hidden layers) e, molto spesso, nelle reti neurali ricorrenti (Recurrent Neural Networks, RNN).
[02:05] Le RNN sono architetture particolari che includono cicli (loop) e, per i loro strati nascosti, la tangente iperbolica è generalmente preferita alla sigmoide.
### La Famiglia delle Funzioni ReLU
[02:11] Una delle funzioni di attivazione più note e utilizzate è la ReLU (Rectified Linear Unit).
[02:15] La sua forma è definita come:
$$
f(x) = \max(0, x)
$$
Questa funzione è composta da due regioni: per input negativi, l'output è zero; per input positivi, l'output è uguale all'input stesso.
[02:18] In ciascuna regione la funzione è lineare, ma il suo comportamento complessivo è non lineare.
[02:23] A differenza della funzione lineare semplice, la composizione di più funzioni ReLU consente alla rete di apprendere comportamenti non lineari, rendendola molto più flessibile.
[02:29] Tuttavia, il problema del "vanishing gradient" persiste, in particolare quando l'input è negativo. In questa parte del dominio, la funzione è identicamente nulla e, di conseguenza, anche la sua derivata è zero. Questo fenomeno è noto come "dying ReLU problem".
[02:38] La ReLU è la funzione di attivazione più comune nelle reti neurali convoluzionali (Convolutional Neural Network, CNN) e, in generale, nelle reti neurali profonde (Deep Neural Network).
[02:44] Dal punto di vista computazionale, è molto efficiente ("cheap"), poiché il calcolo della sua derivata è immediato, a differenza di funzioni più complesse come la tangente iperbolica.
[02:51] La Leaky ReLU è stata introdotta per mitigare il problema della "dying ReLU".
[02:55] La sua caratteristica è che, nella porzione negativa del dominio, invece di avere un output esattamente pari a zero, presenta una pendenza molto piccola ma non nulla. Un valore comune per questa pendenza è 0.01.
[03:00] La sua espressione è:
$$
f(x) = \begin{cases} x & \text{se } x > 0 \\ 0.01x & \text{se } x \le 0 \end{cases}
$$
[03:02] L'idea è che la presenza di una pendenza, seppur minima, anche nella regione negativa, consente alla rete di continuare ad apprendere (anche se molto lentamente) qualora i neuroni si trovino ad operare in questa regione.
[03:10] Non esiste una regola precisa che indichi quando usare la ReLU o la Leaky ReLU. La scelta della funzione migliore è spesso il risultato di un processo empirico di tipo "trial and error" (tentativi ed errori).
[03:20] È possibile estendere ulteriormente il concetto della Leaky ReLU. Invece di scegliere a priori il valore della pendenza per gli input negativi, si può trattare questo valore come un parametro da apprendere.
[03:25] Si ottiene così la Parametric ReLU (PReLU), la cui forma è:
$$
f(x) = \begin{cases} x & \text{se } x > 0 \\ \alpha x & \text{se } x \le 0 \end{cases}
$$
[03:28] In questo caso, il coefficiente $\alpha$ diventa un parametro libero che viene ottimizzato durante il processo di minimizzazione, insieme ai pesi della rete.
[03:34] In altre parole, si aggiungono ulteriori parametri al modello con la speranza di migliorarne le prestazioni.
[03:39] Un potenziale svantaggio è che l'aumento del numero di parametri incrementa la complessità complessiva del modello.
[03:44] L'aggiunta di nuovi parametri può anche aumentare il rischio di overfitting, specialmente se il dataset di addestramento non è sufficientemente grande.
- **Overfitting**: fenomeno che si verifica quando un modello di machine learning apprende troppo bene i dati di addestramento, inclusi il rumore e i dettagli casuali, a scapito della sua capacità di generalizzare a nuovi dati non visti.
### La Famiglia delle Funzioni Esponenziali
[03:50] Esiste una classe di funzioni di attivazione esponenziali. La prima è la ELU (Exponential Linear Unit).
[03:54] Per valori di input positivi, si comporta esattamente come la ReLU ($f(x) = x$). Per valori negativi, assume un andamento esponenziale.
[03:58] La sua forma è:
$$
f(x) = \begin{cases} x & \text{se } x > 0 \\ \alpha (e^x - 1) & \text{se } x \le 0 \end{cases}
$$
dove $\alpha$ è un parametro che controlla la pendenza per gli input negativi.
[04:01] Anche in questo caso, $\alpha$ è un iperparametro che può essere sintonizzato prima dell'addestramento o ottimizzato durante il processo stesso.
[04:06] Aggiungendo un ulteriore parametro, $\lambda$, si ottiene la SELU (Scaled Exponential Linear Unit).
[04:10] La sua forma è:
$$
f(x) = \lambda \begin{cases} x & \text{se } x > 0 \\ \alpha (e^x - 1) & \text{se } x \le 0 \end{cases}
$$
In questo caso, i valori di $\lambda$ e $\alpha$ sono scelti a priori.
[04:14] Questa funzione si dimostra particolarmente efficace se abbinata a una specifica inizializzazione della distribuzione dei pesi della rete.
[04:21] Nella pratica, tuttavia, la SELU non è molto utilizzata.
### Funzioni di Attivazione per Scopi Specifici
[04:24] La funzione Softmax è una funzione di attivazione utilizzata esclusivamente nello strato di output (output layer).
[04:28] La sua definizione è la seguente: dato un vettore di input $x$ con componenti $x_i$, la componente $i$-esima del vettore di output $y$ è calcolata come:
$$
y_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$
[04:30] L'idea è trasformare un vettore di valori reali (ad esempio, le attivazioni dell'ultimo strato) in un vettore di probabilità.
[04:34] Come si può notare dalla formula, al denominatore c'è la somma degli esponenziali di tutte le componenti del vettore di input, mentre al numeratore c'è l'esponenziale di una singola componente.
[04:40] Ad esempio, un vettore di input $[2, 1, 0.1]$ viene trasformato in un vettore di output come $[0.7, 0.2, 0.1]$, i cui elementi sommano a 1 e possono essere interpretati come probabilità.
[04:48] Se si affronta un problema di classificazione con 10 classi, si può utilizzare la funzione Softmax nello strato di output per convertire i valori finali della rete in probabilità di appartenenza a ciascuna classe.
[05:00] La funzione Swish è definita come il prodotto tra la funzione lineare ($x$) e la funzione sigmoide ($\sigma(x)$):
$$
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$
[05:03] I vantaggi di questa funzione sono stati dimostrati a livello pratico: le sue prestazioni sono spesso superiori a quelle della ReLU, in particolare nel caso di reti molto profonde (very deep network).
[05:10] Questa funzione di attivazione è stata sviluppata da Google ed è utilizzata nell'architettura chiamata "EfficientNet".
[05:18] Lo svantaggio principale è che, coinvolgendo la funzione sigmoide, può essere computazionalmente più costosa rispetto ad altre funzioni.
[05:22] D'altra parte, possiede proprietà interessanti, tra cui la capacità di evitare il problema della "dying ReLU".
[05:26] Un'altra variante è la funzione Mish.
[05:28] È importante sottolineare che non esistono prove teoriche definitive che dimostrino la superiorità di una funzione rispetto a un'altra. La scelta si basa principalmente sull'esperienza e sulle applicazioni pratiche.
[05:34] Ad esempio, in compiti di "object detection" (rilevamento di oggetti) in tempo reale, la funzione Mish è comunemente utilizzata e si è dimostrata superiore ad altre funzioni disponibili.
[05:42] Il motivo di questa superiorità rimane una questione aperta.
### Criteri di Scelta della Funzione di Attivazione
[05:45] Le librerie di machine learning più diffuse, come TensorFlow o PyTorch, implementano tutte le funzioni di attivazione menzionate e spesso anche altre. Quelle presentate sono le più comuni.
[05:56] Come scegliere la funzione di attivazione più adatta? Un approccio comune è iniziare con la ReLU.
[06:00] Se si riscontrano problemi con la ReLU (come il "dying ReLU problem"), si possono provare le sue varianti: Leaky ReLU, Parametric ReLU o ELU.
[06:06] Per applicazioni specifiche, come le reti neurali ricorrenti o il rilevamento di oggetti, si possono utilizzare funzioni di attivazione dedicate, come la tangente iperbolica o la Mish.
[06:12] Un suggerimento pratico per la maggior parte delle applicazioni è seguire questi due punti:
1.  Iniziare con la ReLU negli strati intermedi. In alternativa, si può provare la tangente iperbolica.
2.  Passare da queste scelte standard a opzioni più complesse solo se strettamente necessario, poiché ciò aumenta la complessità del modello.