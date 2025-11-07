# Lezione 18 - Reti Neurali: Funzioni di Costo e Tecniche di Regolarizzazione
**Data:** 3 Novembre
**Argomenti:** Cross-Entropy, Overfitting, Regolarizzazione (L1/L2), Dropout, Gradient Descent

---

## PARTE 1: FUNZIONI DI COSTO PER RETI NEURALI

### 1.1 Introduzione e Motivazione

00:00:00 
Okay, quindi passiamo alla seconda parte dell'introduzione alle reti neurali, e in particolare considereremo due problemi:

1. **Scelta della funzione di costo**: Per problemi di classificazione, è possibile modificare la funzione di costo per minimizzare il problema del gradiente che svanisce
2. **Tecniche di regolarizzazione**: Metodi per migliorare la qualità dell'apprendimento delle reti neurali

00:00:45 
E poi passeremo ad altre tecniche che sono volte a migliorare la qualità dell'apprendimento delle reti neurali. Grazie mille. Quindi, ricordiamo un po' ciò che abbiamo visto la volta scorsa nel contesto della funzione di costo, la funzione di costo classica.

### 1.2 Richiami: Funzione di Costo Quadratica (MSE)

### 1.2 Richiami: Funzione di Costo Quadratica (MSE)

00:01:15 
**Definizione**: Supponiamo di avere una funzione di costo quadratica (Mean Squared Error), che misura essenzialmente la differenza tra l'etichetta predetta e quella vera, in termini di norma L²:

$$C = \frac{1}{2}||y_{pred} - y_{true}||^2$$

Quindi noi, come abbiamo fatto la volta scorsa, possiamo calcolare le derivate della funzione di costo rispetto ai pesi e ai bias della rete.

00:01:57 
**Derivate per l'ultimo layer**: Quello che abbiamo trovato la volta scorsa era che per l'ultimo layer in particolare, le espressioni per queste due derivate sono date da:

$$\frac{\partial C}{\partial w} = (a - y) \cdot \sigma'(z) \cdot x$$
$$\frac{\partial C}{\partial b} = (a - y) \cdot \sigma'(z)$$

Dove:
- $a$ = output della rete (attivazione)
- $y$ = etichetta vera
- $\sigma'$ = derivata della funzione di attivazione
- $z$ = input pre-attivazione (wx + b)

**⚠️ PROBLEMA CRITICO**: Il punto più importante è la presenza di $\sigma'$, la derivata della funzione di attivazione. E abbiamo già osservato il fatto che la presenza di $\sigma'$ implica che quando $\sigma'$ è piccolo,

00:02:36 
Quindi la funzione di attivazione è piatta, le derivate sono piccole, e quindi il neurone particolare non è appreso adeguatamente, okay?

**Domanda chiave**: Come possiamo affrontare questo problema e in qualche modo evitarlo?

**Domanda chiave**: Come possiamo affrontare questo problema e in qualche modo evitarlo?

### 1.3 Il Problema del Gradiente Saturato

00:03:08 
**Analisi del problema**: Potete vedere che nell'equazione precedente, anche se abbiamo $(a - y)$, che rappresenta l'errore ed è abbastanza grande, se $\sigma'$ è piccolo, allora significa che:

- Nonostante la presenza di un **grande errore** (che richiederebbe un'alta capacità di apprendimento della rete)
- La rete è quasi **bloccata** a causa della piattezza della funzione di attivazione
- L'apprendimento procede molto lentamente o si ferma

**Conseguenza**: Errore grande → Derivata piccola → Apprendimento lento/bloccato

### 1.4 Visualizzazione del Problema con MSE

00:03:48 
Quindi, qui sto solo tracciando il comportamento. Diciamo che ora dobbiamo concentrarci sulla **linea blu tratteggiata**:

**Grafico in alto** (Funzione di Costo vs Output predetto):
- Asse X: output predetto $a$ (da 0 a 1)
- Asse Y: valore della funzione di costo
- **Caso considerato**: target vero $y = 1$

Queste sono... In alto, è essenzialmente la funzione di costo è una funzione del valore dell'etichetta predetta o output. Quindi, da zero fino a uno, okay?

00:04:34 
**Comportamento della funzione di costo** (curva blu):
- Quando $a = 1$ (output corretto): $C = 0$ ✓ **BUONO**
- Quando $a \to 0$ (output errato): $C$ aumenta ✓ **RAGIONEVOLE**

D, forse è piccola, ma qui, il vero target è y uguale a uno, okay? Quindi, la funzione di costo, scusate, la funzione di costo, essenzialmente, per il caso blu, è zero quando abbiamo l'etichetta vera, il che è buono, okay? E aumenta quando ci spostiamo da, uno a zero.

00:05:07 
Supponendo di avere un **problema di classificazione binaria**, questo comportamento è ragionevole: se ci spostiamo da 1 a 0, significa che stiamo andando sempre più lontano dall'etichetta vera, e quindi è significativo che la funzione di costo stia aumentando il suo valore.

**Grafico in basso** (Derivata della funzione di costo):

D'altra parte, qui abbiamo una derivata della funzione stessa, e quello che abbiamo di nuovo, come funzione della funzione, abbiamo una derivata della funzione stessa, e quello che abbiamo di nuovo, come funzione della funzione stessa, e quello che abbiamo di nuovo,

00:05:47 
Qui stiamo assumendo che la funzione di attivazione sottostante sia la **sigmoid**.

**⚠️ PROBLEMA**: Quello che abbiamo è che anche se siamo molto lontani dalla soluzione vera (grande errore), la derivata della funzione è **vicina a zero** e quindi abbiamo problemi con l'apprendimento.

**🔴 Le curve rosse**: Stanno rappresentando una funzione di costo alternativa che si chiama **cross-entropy**,

**🔴 Le curve rosse**: Stanno rappresentando una funzione di costo alternativa che si chiama **cross-entropy**,

00:06:27 
che vedremo tra un momento e torneremo su questa immagine più tardi.

---

## PARTE 2: CROSS-ENTROPY COME SOLUZIONE

### 2.1 Definizione della Cross-Entropy

Quindi l'idea della funzione cross-entropy è di definire una funzione di costo che è data da questa espressione:

$$J = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(a_i) + (1-y_i)\log(1-a_i)]$$

Dove:
- $n$ = numero di campioni
- $y$ = etichetta vera (ground truth)
- $a$ = output della rete (attivazione dell'ultimo layer)

### 2.2 Proprietà della Cross-Entropy

00:07:05 
È chiaro che, e potete verificarlo abbastanza facilmente, questa funzione soddisfa le proprietà essenziali di una funzione di costo:

#### ✓ Proprietà 1: Non-negatività
$$J \geq 0 \quad \forall a, y$$

#### ✓ Proprietà 2: Minimo sulla soluzione vera
Se l'output calcolato è vicino a quello reale (quindi se $a \approx y$), allora la funzione di costo va a zero, e questo è effettivamente qualcosa di significativo.

00:07:45 
**Analisi per problemi binari** (dove $y \in \{0,1\}$):

**Caso 1**: Se $y = 1$ e $a \to 1$:
$$J = -\log(a) \to 0$$

**Caso 2**: Se $y = 0$ e $a \to 0$:
$$J = -\log(1-a) \to 0$$

Qui, stiamo assumendo di avere un problema binario, dove l'etichetta vera come prima è $y = 1$, e l'altra possibilità è $y = 0$. Oppure qui, per esempio, se $y$ è uguale a zero, e $a$ sta andando a zero, allora effettivamente la funzione sta andando a zero anche lei.

00:08:26 
**Conclusione**: Dal punto di vista delle caratteristiche principali che una funzione di costo dovrebbe avere, la cross-entropy funziona perfettamente:
- ✓ È positiva
- ✓ Dà zero quando valutata sulla soluzione vera o attesa

### 2.3 Derivazione: La Magia della Cross-Entropy

Okay, è positiva e ci sta dando zero quando valutata sulla soluzione vera o attesa. Cerchiamo di vedere quali sono le caratteristiche, le caratteristiche più importanti di questa funzione. Vediamo, come abbiamo fatto per il caso precedente, calcoliamo la derivata di $J$ rispetto ai pesi.

#### Step 1: Applicare la Regola della Catena

00:09:07 
E come abbiamo già visto molte volte, possiamo calcolare questa derivata applicando la **chain rule**:

$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Dove:
- $z = wx + b$ (input pre-attivazione)
- $a = \sigma(z)$ (output post-attivazione)
- $\sigma$ = funzione di attivazione (sigmoid)

Quindi, stiamo scrivendo la regola della catena. Quindi, stiamo scrivendo $\frac{\partial J}{\partial w}$ come $\frac{\partial J}{\partial a}$ per $\frac{\partial a}{\partial z}$, z non è scritto qui, ma è la quantità che abbiamo definito come $wx + b$, okay? Quindi abbiamo $\frac{\partial a}{\partial z}$, ed è effettivamente l'argomento della funzione di attivazione, okay?

00:09:41 
Quindi abbiamo $\frac{\partial a}{\partial z}$, e poi $\frac{\partial z}{\partial w}$. Quindi calcoliamo l'espressione in cui abbiamo i tre fattori.

#### Step 2: Calcolare i Tre Termini Il più semplice è dz su bw, che è x, okay? Poi abbiamo... Il primo è bj su ba.

00:10:12 
Ricordate che j ora è la funzione cross. Quindi stiamo differenziando j rispetto ad a. Quindi abbiamo y per 1 su a. E 1 meno y per 1 meno a con un meno 2. Quindi questa è la derivata che può essere riorganizzata in questo modo.

00:10:50 
Poi abbiamo, infine, l'ultimo è da su dz. Ricordate che a è questa quantità, è l'attivazione. Quindi è la valutazione della funzione di attivazione sull'argomento z. Quindi qui stiamo calcolando essenzialmente sigma primo e quello che potete verificare per la funzione sigmoid è che sigma primo è effettivamente uguale a sigma per 1 su 9 sigma.

00:11:31 
Quindi questa è una proprietà che può essere facilmente quantificata per la funzione sigmoid. Qui nelle slide avrete anche la dimostrazione di questa proprietà se siete interessati. Quindi ora avete questi tre componenti. che possiamo mettere insieme per arrivare alla derivata di j rispetto a w.

00:12:08 
Voglio solo sottolineare una cosa, che in generale, quando dobbiamo calcolare le derivate per una rete neurale, non siamo interessati solo alle derivate di j rispetto a w, ma anche alle derivate di j rispetto a b. Ma questo è più facile, perché essenzialmente quello che avete qui è esattamente la stessa espressione,

00:12:39 
a parte il fatto che qui quello che avrete è dz su db, che è 1. Quindi essenzialmente non avremo quest'ultimo fattore per dj su db. Quindi, raccogliendo tutti i termini. Il primo termine è a-y diviso per a per a1-j moltiplicato per lo stesso termine e poi per x.

00:13:14 
Quindi, alla fine, dj su w è l'errore per x. Quindi, potete vedere che ora, modificando la funzione di costo, sigma primo è scomparso. Quindi, significa che se usiamo, in un problema di classificazione, una funzione di costo adatta che è in questo caso la funzione cross entropy quello con cui possiamo venire.

00:13:51 
è una relazione uh per le sensibilità le quantità che stiamo chiamando sensibilità che sono indipendenti da sigma primo quindi possiamo evitare il problema del gradiente che svanisce okay che è un risultato molto buono per scopi pratici okay e uh quindi significa che uh se ora, l'errore è più grande allora qui questa è la forza trainante nella derivata okay uh quindi uh questo è solo.

00:14:33 
un esempio del fatto che uh e questo è un metodo generale, um, Che, esattamente come abbiamo visto la volta scorsa, quando abbiamo dato una lista di possibili funzioni di costo, non c'è una funzione di costo che è adatta per tutte le applicazioni.

00:15:05 
A seconda dell'applicazione, e a seconda anche della funzione di attivazione che volete accoppiare con una funzione di costo adatta, allora i risultati che potete avere in termini di velocità di apprendimento e capacità di apprendimento possono essere grandemente influenzati. Quindi questo è un fatto che probabilmente è stato giocato con molte persone, ma è un fatto che.

00:15:35 
non lo è. Rete neurale, avete visto quello, ed è, a parte situazioni molto semplici, situazioni molto ben stabilite che sono state studiate nei dettagli, non c'è una ricetta, una ricetta generale per dire che se avete un problema particolare, quella è la funzione di attivazione e di costo adatta. È, sfortunatamente, una sorta di procedura per tentativi ed errori che avete.

00:16:10 
Another problem that you can encounter and that is related to some of the elements that we have seen is the so-called overfitting. What is the overfitting? Overfitting essentially is, if you want to give a pictorial view of overfitting, is the fact that if you have some measurement, let's say something like this, and your expectation is that you,

00:16:53 
---

## PARTE 3: OVERFITTING E SUE CONSEGUENZE

### 3.1 Che cos'è l'Overfitting?

00:16:10 
Un altro problema che potete incontrare e che è legato ad alcuni degli elementi che abbiamo visto è il cosiddetto **overfitting** (sovra-adattamento).

**Definizione**: Overfitting è il fenomeno in cui il modello si adatta troppo ai dati di training, perdendo la capacità di generalizzare.

#### Esempio Visuale: Regressione Lineare vs Polinomiale

00:16:53 
Se avete alcune misurazioni, diciamo qualcosa del genere:
- **Asse X**: Temperatura
- **Asse Y**: Pressione

Per un fenomeno specifico, vi aspettate una **relazione lineare** tra pressione e temperatura.

**❌ Overfitting**: Se state facendo overfitting, forse state creando qualcosa di simile a

00:17:25 
un polinomio di grado alto, okay, che è molto lontano dall'essere lineare.

**Problema critico**: Qual è il problema di questo fitting polinomiale?
- La verità fisica potrebbe essere qualcosa come una retta
- Se uso questa approssimazione overfittata per valutare qualcosa
- **Esempio**: Voglio valutare la pressione a una temperatura sconosciuta

00:17:56 
Quello con cui posso venire è un valore che è **molto diverso** da quello atteso in termini di significato fisico.

**Definizione formale**: In altri termini, l'overfitting è un fenomeno che **limita la capacità del vostro modello/rete neurale di generalizzare**, okay, di essere applicato ad elementi che non sono stati usati per la fase di training.

### 3.2 Come Identificare l'Overfitting: Training vs Validation Curves

00:18:34 
Okay, quali sono i **sintomi** che potete controllare per avere qualche, diciamo, indizio sul fatto che il vostro metodo sta facendo overfitting?

#### Setup: Splitting del Dataset

00:19:07 
Sapete che quando addestrate una rete neurale, o se create qualsiasi procedura di apprendimento, potete tracciare un grafico del valore della **funzione di costo** in termini di quello che è chiamato **epochs** nella terminologia delle reti neurali.

**Epochs** = iterazioni, essenzialmente, le iterazioni che state eseguendo per minimizzare la vostra funzione di costo.

00:19:44 
E l'idea è che questi sono **due grafici**:

**🔵 Curva BLU**: Training cost function
- Funzione di costo valutata sul **training set**

00:20:16 
Immagino che sappiate che quando vi viene dato un particolare set di dati, di solito dovete, una delle prime cose che dovete fare è **dividere il vostro dataset** in due o tre subset:

1. **Training set** (70-80% dei dati)
   - Usato per addestrare il modello
   - Aggiorna i pesi
   
2. **Validation set** (20-30% dei dati)
   - Usato per monitorare l'overfitting
   - NON usato per il training

00:20:50 
Validation, sorry, per la validation. Cosa significa?

#### Procedura di Monitoraggio

Significa essenzialmente che durante il training, ad ogni passo del training:

00:21:20 
1. Valuti la funzione di costo sul **training set**
2. Con i parametri correnti che hai calcolato a quella particolare iterazione
3. Valuti ANCHE il **validation set**
   - Set di campioni che NON sono stati usati per il training
   - Sono **dati mai visti** dal punto di vista del modello

#### Pattern Tipico di Overfitting

E un possibile pattern tipico è che:

- ✓ **Training cost**: Continua a decrescere
- ⚠️ **Validation cost**: Ha un minimo, poi **inizia ad aumentare**!

Come potete vedere qui.

00:22:11 
Questo significa che da questo punto, diciamo intorno alle **115 epoche**, è ragionevole supporre che abbiamo iniziato una **fase di overfitting**, perché la capacità di generalizzazione del modello sta peggiorando, okay?

**🚨 Sintomo chiave**:
```
Training cost ↓↓↓ (continua a scendere)
Validation cost ↑↑↑ (inizia a salire)
→ OVERFITTING!
```

Quindi questo è un pattern tipico che è un allarme

00:22:45 
riguardo al fatto che potreste aver trovato un'area di overfitting, okay?

#### Soluzione Immediata: Early Stopping

Quindi, per esempio, se vedete questo pattern tipico, potete eseguire quella che viene chiamata **early stopping**. Quindi potete **fermare la vostra procedura di apprendimento** qui quando vedete la curva relativa al costo di validazione

00:23:15 
che sta aumentando, okay?

### 3.3 Strategie per Combattere l'Overfitting

Quali sono le possibili, diciamo, **misure per curare l'overfitting**?

#### Strategia 1: Ottenere Più Dati

00:23:48 
La prima è **ottenere più dati di addestramento**. Questa è la più naturale, ma non è sempre possibile, ovviamente.

**Data Augmentation**: Ci sono metodi, come ho già menzionato, per eseguire un aumento del dataset:
- Se avete **immagini**: potete eseguire operazioni geometriche (rotazione, flip, crop, zoom) per creare nuove immagini
- Dal punto di vista della rete, sono **nuovi dati**
- Oppure potete usare altre tecniche volte a creare nuovi **dati sintetici**

00:24:21 
che sono coerenti con il dataset originale.

#### Strategia 2: Data Augmentation (vedi sopra)

E questa è la seconda strategia.

#### Strategia 3: Regolarizzazione

00:24:52 
E la terza è usare la **regolarizzazione**. Questo è qualcosa che abbiamo già incontrato nel contesto dei minimi quadrati.

---

## PARTE 4: REGOLARIZZAZIONE L1 E L2

### 4.1 Idea della Regolarizzazione

E qui, essenzialmente, quello che faremo è applicare più o meno le stesse tecniche. Quindi, l'idea è:

**Obiettivo**: Aggiungere una **penalità** alla funzione di costo per rendere la funzione di costo meno propensa ad andare verso l'overfitting.

00:25:29 
E le regolarizzazioni classiche dal punto di vista matematico sono la **L2** e **L1** che abbiamo già visto.

### 4.2 Regolarizzazione L2 (Weight Decay)

#### Definizione

Quindi, nel contesto della rete neurale, la regolarizzazione L2 equivale a qualcosa che è legato al **quadrato del vettore dei pesi**:

$$C_{reg} = C_{original} + \frac{\lambda}{2n}\sum_{i} ||w_i||^2$$

Dove:
- $\lambda$ = parametro di regolarizzazione (hyperparameter)
- $n$ = numero di campioni
- $w_i$ = pesi (weights)

Okay? Quindi, in qualche modo simile a questo. E lambda è il cosiddetto parametro di regolarizzazione.

#### Perché si chiama "Weight Decay"?

00:26:38 
Questa strategia è anche chiamata **weight decay** perché se calcolate, di nuovo, la derivata della funzione di costo rispetto ai pesi e ai bias:

**⚠️ Nota importante**: In realtà, i **bias** di solito NON sono regolarizzati. Come potete vedere qui nella funzione di costo, abbiamo solo un termine di penalità per i pesi perché i bias non sono così importanti in termini di overfitting, okay?

#### Derivazione: Gradient Descent con L2

00:27:11 
Quindi, l'idea è, ora, se calcolate la derivata della funzione di costo e... Non abbiamo ancora introdotto la discesa del gradiente stocastico, ma lo vedremo domani, forse. Ma l'idea è, diciamo, la **discesa del gradiente**, il metodo della discesa del gradiente consiste nell'aggiornare i pesi:

$$w_{k+1} = w_k - \gamma \cdot \frac{\partial C}{\partial w}|_{w_k}$$

Dove:
- $\gamma$ = learning rate (tasso di apprendimento)
- $k$ = iterazione corrente

Quindi questa è la cosiddetta discesa del gradiente.

00:27:54 
Stocastico, sì, per stocastico vedremo cosa significa domani. Ma, quindi, se guardate la regola di aggiornamento, essenzialmente potete inserire $\frac{\partial C}{\partial w}$ che avete calcolato qui per il gradiente:

$$\frac{\partial C_{reg}}{\partial w} = \frac{\partial C_{original}}{\partial w} + \frac{\lambda}{n}w$$

E se raccogliete i termini, avete:

$$w_{k+1} = w_k - \gamma\frac{\partial C_{original}}{\partial w} - \gamma\frac{\lambda}{n}w_k$$

00:28:38 
Riorganizzando:

$$w_{k+1} = \left(1 - \frac{\gamma\lambda}{n}\right)w_k - \gamma\frac{\partial C_{original}}{\partial w}$$

Quindi significa che questa quantità $(1 - \frac{\gamma\lambda}{n})$ è minore di uno (ma ancora positiva, di solito).

**Interpretazione**: Quindi, questa quantità sarà $< 1$ ma ancora positiva. Essenzialmente, quello che state facendo introducendo questo coefficiente è una **riduzione di w**.

00:29:14 
Quindi, l'iterazione dal precedente $w$ al nuovo sta dando una **riduzione** dell'iterazione precedente.

**Effetto "Decay"**:
```
w_new = (1 - ε)·w_old - γ·∇C
        ↑
    decay factor < 1
```

Quindi, passare da $w_k$ a $w_{k+1}$ equivale a eseguire una riduzione. E questo è quello che avreste avuto, a parte la presenza del tasso di apprendimento,

00:29:46 
è esattamente quello che avreste avuto senza la regolarizzazione. Quindi, questa è la ragione per cui si chiama **weight decay**, perché applicando ripetutamente, iterativamente questa ricetta, quello che state facendo è **ridurre il valore di W**.

00:30:22 
Quindi, si spera, non state andando verso grandi valori di $W$, che è, se ricordate, esattamente la stessa affermazione che abbiamo ottenuto quando abbiamo introdotto la regolarizzazione L2 nel contesto dei minimi quadrati:

**L2 → Norma minima**: $\min ||w||_2^2$

Significa trovare il vettore dei pesi di **norma minima**. Okay? Qui, è detto in un modo diverso, nel contesto generale, ma il messaggio è essenzialmente lo stesso: State riducendo, state andando verso $w$ piccoli.

### 4.3 Regolarizzazione L1 (Sparsity)

00:30:54 
Cosa dire della **L1**? L1 era volta alla **sparsità**, se ricordate.

#### Regola di Aggiornamento con L1

E qui, essenzialmente, quello che abbiamo è, se eseguite esattamente la stessa operazione di prima, qui abbiamo un'espressione diversa per la prima parte:

$$w_{k+1} = w_k - \frac{\eta\lambda}{n}\text{sign}(w_k) - \eta\nabla C_{original}$$

Quindi abbiamo, di nuovo, questo vettore $\frac{\eta\lambda}{n}$,

00:31:29 
e abbiamo $w$ meno questa quantità.

**Effetto**: Quindi, in pratica, di nuovo, in questo caso, quello che stiamo facendo è andare verso un vettore $w$ **sparso** (molti zeri).

### 4.4 Visualizzazione Geometrica: L1 vs L2

00:32:13 
Qui c'è un'immagine di cosa sta succedendo. Quindi:

- **Nero tratteggiato**: Curve di livello di una possibile funzione di costo
- **🔵 Blu**: Palla unitaria nella norma L2 (cerchio)
- **🔴 Rosso**: Palla unitaria nella norma L1 (diamante)

Quindi, in nero tratteggiato avete le curve di livello di una possibile funzione di costo, okay?

00:32:44
E poi avete, in blu e in rosso, le **due palle unitarie**, in questo caso in $\mathbb{R}^2$, rispettivamente usando:
- Per il blu: la norma **L2**
- Per il rosso: la norma **L1**

Quindi:
- Questa è la palla unitaria per la norma L1 (forma di diamante)
- Questa è per la norma L2 (forma circolare)

Okay? E,

potete vedere che in pratica quello che succede. i due punti almeno, quindi correggerò le immagini, quindi l'idea è che quello che volete ottenere è che essenzialmente avete un **problema vincolato**, volete

00:33:16 
minimizzare la funzione, la funzione di costo, soggetto al fatto che la soluzione appartiene, per esempio, alla palla unitaria in L2 o alla palla unitaria in L1. Quindi, in altre parole significa che per la palla unitaria in l1 potremmo essere finiti qui probabilmente e questa è.

00:33:54 
una visione pittorica del fatto che stiamo andando verso la sparsità perché molto probabilmente quando state usando l1 l'intersezione tra le curve di livello minimo e la palla l1 è in uno degli angoli del quadrato mentre per la l2 potete avere qualsiasi punto sul cerchio okay.

00:34:33 
C'è un'altra tecnica che viene usata, devo dire che questa tecnica non è, come dire, molto matematicamente rigorosa, ma è molto usata, quindi voglio solo menzionarla perché sono sicuro che incontrerete questo termine. Si chiama dropout. Qual è l'idea? Nel caso precedente, abbiamo modificato la funzione di costo introducendo un termine di penalità, sia L2 che L1.

00:35:17 
Qui, quello che stiamo facendo è, stiamo modificando la rete. In che senso? Durante la rete... Considerate tutti i livelli nascosti della vostra rete e casualmente, in ogni livello, andrete a eliminare o cancellare o impostare l'output a zero una certa percentuale di neuroni.

00:35:54 
Per esempio, per P uguale a 0.5, significa che state cancellando metà dei neuroni di una rete. Eseguite un passo forward e backward e aggiornate i pesi e i bias e ripetete questo processo per tutte le iterazioni che dovete eseguire secondo.

00:36:25 
la dimensione del vostro dataset. Alla fine, l'idea è che durante il testing o la valutazione, quindi dopo l'addestramento, usate le reti complete, quindi ora non state eliminando alcun neurone, ma i pesi degli strati nascosti sono moltiplicati per la probabilità che avete usato per eseguire il dropout.

00:37:11 
Qual è l'idea? Come eseguire quello che viene chiamato, per esempio, nelle previsioni meteorologiche, in cui avete alcuni modelli che devono essere alimentati da alcuni dati, per esempio, le condizioni iniziali, quindi temperatura, pressione, umidità, e così via, ora. E volete eseguire una previsione per i prossimi tre giorni o una settimana.

00:37:42 
Poiché questi modelli sono molto sensibili alle condizioni iniziali, quello che eseguono di solito non è una singola esecuzione, ma è un ensemble di simulazioni. E poi, modificando gli input con una certa distribuzione di probabilità, il risultato sarà qualche...

00:38:12 
Ogni salto. di tutte le simulazioni che avete eseguito. Qui, più o meno, stiamo facendo lo stesso. Abbiamo eseguito l'addestramento usando molte reti diverse, essenzialmente, con architettura diversa, con topologia diversa, se volete, perché abbiamo eliminato alcuni neuroni. E il risultato che stiamo considerando è una media di tutti i neuroni,

00:38:46 
tutti i modelli che avete considerato. Per esempio, in TensorFlow, quando definite la topologia della rete, potete introdurre quelli che vengono chiamati dropout layers, che non sono altro che un modo di specificare il fatto che un certo livello nascosto deve essere soggetto al dropout con una certa probabilità, okay?

00:39:18 
E alla fine, il modello finale che otterrete sarà creato secondo questa idea di ensemble. Un altro problema che dobbiamo affrontare è l'inizializzazione dei parametri. Questo è un problema che ho anche affrontato l'ultima volta.

00:39:51 
Di nuovo, non è molto critico per i bias, ma può esserlo per i pesi. Perché? Di solito, supponiamo che inizializziate i pesi e i bias usando una distribuzione gaussiana con media zero e deviazione standard uguale a uno. E poi supponiamo che abbiate il neurone, che ha n valori di input, quindi avrete n pesi di input e il corrispondente bias.

00:40:36 
Se assumete che gli input siano normalizzati, quindi per esempio, sono zero o uno in qualche modo, avrete n valori di input, quindi avrete n valori di input. La somma pesata che otteniamo sarà anche la distribuzione gaussiana e questo è il punto chiave, la deviazione standard di z, che è la distribuzione pesata, è data approssimativamente dalla radice quadrata del numero degli input per la deviazione standard della distribuzione iniziale, al quadrato.

00:41:18 
Quindi, proviamo questo, è approssimativamente uguale alla radice quadrata del numero di input del neurone. È chiaro che se avete una situazione reale dove un singolo neurone può avere fino a mille o anche più input, la deviazione standard può essere piuttosto alta, dell'ordine di 30 in questo esempio.

00:41:50 
Questo significa che Z può essere alto sia nella regione positiva che in quella negativa. qual è lo svantaggio di ovviamente qui stiamo parlando stiamo considerando il primo passo okay quindi nel primo passo abbiamo inizializzato i nostri pesi con una distribuzione gaussiana e quello che abbiamo scoperto è che approssimativamente la distribuzione della variabile z è una distribuzione gaussiana come.

00:42:26 
bene con una deviazione standard che può essere piuttosto alta significando che z può avere un valore piuttosto alto qual è il problema di avere uno z con valore alto significa che siamo nella regione dove per esempio la funzione sigmoid è piatta e quindi possiamo avere un problema di gradiente che svanisce okay quindi se la distribuzione della variabile z è maggiore o uguale alla distribuzione della variabile z del peso iniziale non è scelta correttamente possiamo iniziare.

00:43:04 
il nostro processo con la velocità di apprendimento molto bassa okay quindi questo è un problema che, è stato osservato in pratica nei primi esperimenti e molte ricette sono state proposte per usare una distribuzione dei pesi iniziali che è meno sensibile al gradiente che svanisce.

00:43:37 
problema quindi qui è solo la visualizzazione del problema e una delle possibili idee, è usare di nuovo una, uh distribuzione gaussiana con media zero ma cambiare la deviazione standard, La deviazione standard ora dovrebbe essere proporzionale al numero, inversamente proporzionale al numero del parametro di input della rete che, il neurone che state considerando.

00:44:21 
E ricordate che se avete ovviamente ER, quello che potete fare in una rete reale è prendere il n in, il più grande n in per inizializzare la vostra distribuzione. E se eseguiamo esattamente lo stesso calcolo che abbiamo visto prima, quello che possiamo ottenere è che in questo caso,

00:44:52 
la deviazione standard è uno. della della variabile z. Quindi significa che con questa particolare scelta per la distribuzione iniziale dei pesi, siamo meno propensi ad avere una situazione dove possiamo andare in regione del gradiente che svanisce. Poi, qui sto affrontando alcuni problemi pratici che sono.

00:45:38 
in realtà relativi alla numerica che sta dietro alla procedura di apprendimento della rete neurale. Quali sono i principali iperparametri? Prima di tutto, iperparametri è un termine che viene usato per denotare tutti i parametri che non sono ottimizzati durante la procedura di apprendimento, che sono scelti a priori, okay? E tipicamente abbiamo il tasso di apprendimento,

00:46:12 
quindi questo parametro, eta o gamma, il parametro di regolarizzazione, vedremo più tardi cos'è la dimensione del mini batch, il numero di epoche, e l'architettura della rete, quanti livelli, quanti neuroni per livello, e così via. Poi, se volete, questo ovviamente non è un iperparametro, ma anche la scelta.

00:46:42 
di quanti livelli della funzione di attivazione può fare la differenza. Ma se rimaniamo nel regno dei numeri, questi sono gli iperparametri classici. Come vi dicevo prima, non c'è una ricetta generale per dire, data quell'applicazione, il miglior tasso di apprendimento è il parametro di regolarizzazione dovrebbe essere, e così via.

00:47:14 
Quindi dipende. Quindi qual è la strategia tipica? Prima di tutto, per esempio, per trovare il tasso di apprendimento, dovete eliminare qualsiasi tecnica di regolarizzazione. Poi. Trovate un tasso di apprendimento iniziale che almeno vi dia un valore decrescente della funzione di costo.

00:47:55 
Per esempio, può essere 0.01. Poi dovete controllare se durante il corso, durante l'addestramento, vedrete che avrete un aumento nei valori della vostra funzione di costo. Allora dovrete diminuire il valore, altrimenti potete provare ad aumentarlo un po'.

00:48:28 
Una volta che avete ottimizzato il tasso di apprendimento, potete passare al parametro di regolarizzazione. Potete iniziare con lambda uguale a uno, e poi di solito qui potete usare una scala logaritmica per scegliere gli altri possibili valori. E qual è il miglior valore che potete scegliere?

00:49:00 
Forse dovete guardare il pattern della funzione di costo di validazione. Quindi, ricordate che abbiamo l'addestramento e poi la validazione. Quindi, forse se iniziate a vedere che a un certo punto la validazione sta aumentando, allora probabilmente avete trovato un buon lambda.

00:49:30 
Poi, possibilmente, potete iniziare di nuovo con quel lambda, potete riottimizzare eta, e così via. Poi dovete ottimizzare il mini-batch. Mini-batch è qualcosa che è legato alla discesa del gradiente stocastico, è un altro parametro che è di fondamentale importanza, e torneremo più tardi su questo argomento. In pratica, possiamo dire, come regola empirica, che il parametro più critico tra gli iperparametri è il tasso di apprendimento.

00:50:18 
Se il tasso di apprendimento è troppo alto, significa che la funzione di costo oscillerà o esploderà. Se è troppo basso, forse le cose sono diverse. Se è troppo alto, forse le cose sono diverse. La procedura di apprendimento sarà molto, molto lenta, e poi avete un buon valore o un buon range di tasso di apprendimento che assicura il fatto che la funzione di costo, il valore della funzione di costo sta diminuendo in modo regolare.

00:51:00 
00:51:00 
L'idea è che dovete sempre controllare l'accuratezza di validazione. Questo è il, direi, grafico più importante che dovete guardare durante l'addestramento, la validazione, il pattern della funzione di costo di validazione. Qui c'è una rappresentazione tipica di, in rosso, il valore del tasso di apprendimento è troppo alto.

00:51:40 
Abbiamo una fase iniziale dove stiamo diminuendo il valore della funzione di costo, ma poi inizia a oscillare e diverge. In blu, beh, non è male, nel senso che non stiamo divergendo, ma la convergenza è molto, molto lenta. In verde, questa è la scelta migliore. Abbiamo una diminuzione regolare della funzione di costo e una velocità di apprendimento ragionevole.

00:52:21 
OK, quindi questa è un'immagine che qualitativamente rappresenta le tre situazioni che abbiamo menzionato. OK, qui c'è solo il riassunto di quello che abbiamo visto in questa parte della cross entropy.

00:52:52 
È un'istanza di una funzione di costo alternativa regolarizzazione. per avere capacità di generalizzazione e inizializzazione dei pesi. Ho dimenticato di menzionare il fatto che anche nelle librerie tipiche, TensorFlow, PyTorch, avete la possibilità di scegliere tra i metodi più classici per inizializzare i pesi. Quindi se guardate weight initialization.

00:53:27 
nella documentazione di queste due librerie, o Scikit-learn o qualsiasi altra, troverete almeno, cinque o quattro metodi per inizializzare. E poi qui c'è solo la dimostrazione che è stata menzionata prima, ma la lascio a voi come esercizio. Okay, quindi.

00:54:05 
Abbiamo menzionato molte volte la discesa del gradiente. Perché? Perché in pratica, a parte il fatto che viene da un problema particolare, il problema più importante che dobbiamo affrontare è la minimizzazione della funzione di costo.

00:54:37 
Per trovare il miglior insieme di pesi e bias. Questo è il problema. Quindi ora stiamo iniziando una parte del corso relativa ai metodi di minimizzazione. Ovviamente siamo interessati a un'applicazione particolare, ma inizieremo con il metodo classico e poi ideeremo e studieremo altri metodi più focalizzati sulle reti neurali.

00:55:13 
Il primo metodo che considereremo è la discesa del gradiente, perché è alla base del direi primo e più importante metodo che è stato usato nel contesto delle reti neurali, che è la discesa del gradiente stocastico. Quindi prima di trattare lo stocastico, voglio solo darvi un'idea della classica discesa del gradiente.

00:55:45 
Quindi, prima di... Trattare la discesa del gradiente, voglio solo rivedere alcuni concetti che useremo in alcuni dei risultati che vedremo. Prima di tutto, la definizione di convessità. Questa è la definizione formale, e in pratica, significa che se avete...

00:56:16 
Questo è un caso bidimensionale, caso unidimensionale. A sinistra, avete una funzione convessa. A destra, avete una funzione non convessa. Significa che, essenzialmente, se prendete due punti qualsiasi sul grafico della funzione, il segmento non interseca. Nessun altro punto della definizione? Come in questo caso. Oppure, se volete, il segmento è totalmente incluso nell'epigrafo della funzione, quindi la porzione del piano contenuta.

00:57:00 
Questa è la definizione formale di convessità, che è esattamente quello che abbiamo visto. Quindi, dato lambda tra 0 e 1, questo è esattamente il segmento che unisce due punti sul ramo. Ma la convessità può essere anche caratterizzata in altri modi.

00:57:34 
C'è, per esempio, una caratterizzazione del primo ordine della convessità, che è valida quando la funzione f è anche differenziabile. E, in pratica, questa caratterizzazione significa che il valore della funzione y è sopra, o se volete, la funzione f è sopra la tangente in quel particolare punto.

00:58:13 
Quindi questa è l'immagine. Okay? Questa è una rappresentazione della caratterizzazione del primo ordine della convessità. Poi abbiamo altre due proprietà che sono importanti. La prima è la proprietà di Lipschitz-B della funzione. Diciamo che la funzione è Lipschitz-B se il valore assoluto della differenza di due valutazioni della funzione.

00:58:50 
è limitato dalla norma del vettore che collega X e Y per la costante B. In pratica, se la funzione è anche differenziabile, significa che se dividete per la norma di X meno Y, significa in pratica che il gradiente.

00:59:21 
i gradienti della funzione sono limitati. Poi abbiamo la definizione di L-smoothness se il gradiente è Lipschitz, quindi ora non stiamo considerando qui il valore della funzione, ma il valore del gradiente è di nuovo limitato dalla costante L per la norma di x meno y.

01:00:01 
Qual è il significato? Qui in quel caso abbiamo la limitatezza sul gradiente, qui abbiamo la limitatezza sulla curvatura della funzione. In realtà, questa proprietà, quindi la Lipschitz, può essere anche, e questo è un altro esercizio che vi lascio, può essere scritta anche come un limite superiore quadratico della funzione.

01:00:44 
Quindi significa essenzialmente che, quindi questa definizione è equivalente a questa, okay? In pratica, se confrontate questa con questa, qui abbiamo che la funzione giace sempre sotto il limite quadratico, okay? E infine, abbiamo un'altra proprietà, che è la cosiddetta convessità forte mu, che è, d'altra parte, se guardate queste due relazioni, qui abbiamo il segno opposto, nel senso che qui abbiamo, nel box superiore, abbiamo un limite superiore, qui abbiamo un limite inferiore.

01:01:42 
Quindi, graficamente, possiamo guardare le due proprietà in questo modo. Quindi, in nero avete la funzione, e se considerate il punto, la curva rossa vi sta dando il limite superiore, quindi avete un limite superiore quadratico della funzione, e il blu, vi sta dando un limite inferiore della funzione, un limite inferiore quadratico, okay, e la regione ombreggiata è la regione dove la funzione f appartiene in termini delle due parabole che limitano la funzione in un particolare punto.

01:02:41 
Okay, quindi teniamo a mente questa definizione. Quindi abbiamo convessità, smoothness, e convessità forte. Queste sono le tre proprietà che NL insegna, le quattro proprietà che richiameremo anche più tardi. Ma prima di trattare l'analisi dell'algoritmo del gradiente, definiamo cos'è il gradiente.

01:03:15 
L'obiettivo è, data una funzione f, idealmente la nostra funzione di costo, vogliamo minimizzare per trovare il valore x star che minimizza la funzione di costo. E per il momento, stiamo considerando x appartenente allo spazio rd completo. Non stiamo mettendo alcun vincolo sul possibile valore di x.

01:03:45 
In pratica, x sarà w, quindi il vettore dei pesi o il vettore dei bias. Quindi questo è il problema che vogliamo risolvere. La discesa del gradiente è un metodo iterativo, quindi non è un metodo che finirà in un, numero fisso di iterazioni, ma è un metodo iterativo che crea una sequenza di soluzioni,

01:04:19 
quindi x0, x1, x2 e così via, usando una regola di aggiornamento. L'idea è iniziare da un'ipotesi iniziale, x0, che è essenzialmente l'inizializzazione del vettore dei pesi che abbiamo discusso prima, e poi iterare. L'idea generale dell'iterazione è questa, quindi x t più 1 è uguale a x t più v t, dove v t è un vettore che mi dice quanto devo.

01:05:00 
in quale direzione e quanto devo modificare l'iterazione precedente. Quindi ora il punto è, devo decidere la direzione e la direzione del vettore dt. Okay, quindi qual è l'idea per la direzione? Il nostro obiettivo è trovare un metodo tale che la nuova iterazione, x t più uno o w t più uno,

01:05:38 
sta dando un valore della funzione f che è minore del precedente. Okay, quindi se... Prendete l'espansione di Taylor, potete vedere che il valore della funzione in xt più vt è uguale a questa espressione in primo ordine, e in pratica, significa che affinché questa quantità sia minore di questa, questa quantità dovrebbe essere negativa, okay?

01:06:15 
Okay, quindi dobbiamo trovare una direzione tale che il gradiente valutato in xt trasposto per vt sia minore di zero. È chiaro che una possibile scelta, e in realtà quella che massimizza, è se prendete vt uguale a meno il gradiente okay quindi la direzione è meno il gradiente di f in xt.

01:06:59 
e poi introduciamo la lunghezza del passo che è gamma è la dimensione del passo o il tasso di apprendimento se ricordate per esempio quando avete studiato il metodo del gradiente per risolvere un sistema lineare, o il metodo del gradiente coniugato. In quel caso, avete fatto esattamente la stessa cosa.

01:07:31 
Avete calcolato la direzione del gradiente come la direzione del movimento. E poi avete deciso la lunghezza del passo. In quel caso, è possibile trovare una lunghezza del passo ottimale per ridurre il valore del punto.

01:08:03 
Quindi, la regola di aggiornamento generale per la discesa del gradiente nel nostro contesto è xt più 1 uguale a xt meno gamma per il gradiente. Questa regola di aggiornamento assicura che la nuova iterazione Xt più uno sia tale che f valutata in Xt più uno sarà minore di f in Xt, okay?

01:08:34 
Ovviamente, gamma è supposto essere positivo, giusto? Qui c'è un esempio del... Non è veramente visibile. In blu avete la funzione, e poi avete alcuni segmenti rossi che vi stanno dando la diminuzione.

01:09:05 
Lasciate che... Okay. Qui, per esempio, è il paraboloide dato. Potete vedere l'insieme di livello della funzione. Iniziate in questo punto qui. La dimensione del passo è impostata a 0.2. E in quel caso, potete vedere il percorso che seguite dall'ipotesi iniziale fino al minimo. Okay.

01:09:43 
Ovviamente, se cambiate la posizione iniziale, l'ipotesi iniziale, o se cambiate la dimensione del passo. Avete un comportamento diverso e come potete vedere, se la dimensione del passo diventa troppo grande, il metodo non sta convergendo monotonicamente più alla soluzione. State iniziando ad avere oscillazioni e il metodo è anche sensibile alla forma della funzione di costo che state considerando.

01:10:29 
Qui avete, nel primo caso, avete una funzione di costo che è un paraboloide, direi con i due assi quasi uguali. OK. Gli insiemi di livello non sono veramente circolari, ma sono ellissi non troppo allungate. Se considerate una situazione dove sono, Avete un insieme di livello molto più allungato, quindi il paraboloide è caratterizzato dall'avere due assi con magnitudini diverse.

01:11:07 
La capacità del metodo è molto diversa, e in particolare, come potete vedere, qui stavamo usando più o meno 0.2, ed eravamo in grado di ottenere una convergenza monotona verso l'ottimo. In questo caso, come potete vedere qui, abbiamo un ordine di grandezza di dimensione del passo più piccolo, e se aumentiamo, in questo caso,

01:11:46 
la dimensione del passo come potete vedere qui la situazione può essere molto peggiore perché il fatto che questo è un paraboloide con due assi di diciamo ogni insieme di livello è una nuova ellisse con i due assi molto diversi vi dà un problema di condizionamento unico quindi è molto sensibile al.

01:12:17 
in particolare alla dimensione del passo come potete vedere qui con la dimensione del passo che ci stava dando convergenza, convergenza monotona nel caso precedente stiamo avendo un decadimento oscillante molto cattivo e persino. In una situazione dove non avete il problema convesso, ma questa è una visione pittorica del.

01:12:49 
problema non convesso, potete vedere che per evitare l'oscillazione, dovete usare, qui è 0.02, ed è molto più piccolo, e quello che potete vedere è che in questo caso, se usate.

01:13:29 
Diciamo, anche forse qui siamo 0.05, possiamo avere un comportamento molto cattivo del metodo, specialmente all'inizio. Okay, quindi tornando alla presentazione, questa è un'altra immagine in cui vi ho mostrato su una situazione molto più facile. Quindi abbiamo una parabola, e iniziamo dal punto, diciamo qui, e vogliamo raggiungere il minimo, giusto?

01:14:11 
E quello che abbiamo considerato sono quattro valori diversi del parametro di apprendimento di gamma. Con il blu, le cose stanno andando bene, ma il processo di apprendimento è molto, molto lento. Con forse 10 passi, siamo ancora qui, molto lontani dal minimo. Poi abbiamo il rosso. È troppo grande nel senso che il metodo sta convergendo,

01:14:48 
ma con comportamento oscillante. Quindi sta andando da un lato all'altro della parabola. E con il magenta, potete ottenere una situazione dove state divergendo. Poi avete il verde, che è buono in termini di sia velocità che monotonicità.

01:15:18 
la convergenza, okay? Quindi questa immagine è forse più intuitiva, ed è importante tenerla a mente, per visualizzare tutte le possibili situazioni. Okay, quindi data la regola di aggiornamento di base, questa, ora quello che vogliamo fare è vogliamo analizzare.

01:15:52 
in realtà la performance di questa rete in termini di convergenza. Quindi il di base, la prima regola di base che considereremo è basata sull'uso dell'ipotesi di convessità. E quello che vogliamo fare è trovare il limite per questo, dove xd è la soluzione all'iterazione t, e x star è la vera soluzione.

01:16:37 
Okay, quindi vogliamo trovare la differenza tra f valutata nel minimo reale e f valutata nel minimo calcolato. Usando la caratterizzazione del primo ordine della convessità, possiamo scrivere questa equazione. Quindi abbiamo solo usato la precedente dove avevamo x e y. Sostituendo xd e y, possiamo scrivere questa equazione.

01:17:08 
t e x star. Poi useremo... useremo in tutto questo capitolo sul gradiente e la discesa del gradiente, gt sarà il gradiente della funzione valutato all'iterazione t, quindi valutato su x t. Quello che dobbiamo limitare essenzialmente è questa quantità, okay? Questa è la quantità che vogliamo limitare, e poiché.

01:17:46 
questo è minore o uguale a questo, stiamo lavorando su questa quantità. Ora qui quello che possiamo sfruttare è la regola di aggiornamento che definisce il gradiente. Quindi abbiamo impostato che x t più uno è uguale a x t meno gamma gradiente di f in x. Questa è la regola di aggiornamento del gradiente, e da quella regola, possiamo calcolare il gradiente come la differenza tra le due iterate divisa per il tasso di apprendimento.

01:18:28 
Quindi possiamo inserire questa quantità nell'espressione precedente. Quindi abbiamo il gradiente trasposto all'iterazione t per x t meno x star è uguale a 1 su gamma per la differenza tra due iterate, le iterate consecutive, per x t meno x star.

01:18:59 
Ora è una questione di fare alcune semplificazioni. Quindi abbiamo che. Se considerate, che, la norma della differenza di due vettori al quadrato potete arrivare a questa relazione quindi, due volte b trasposto w è uguale alla norma di b al quadrato più la norma di w al quadrato meno.

01:19:30 
la norma della differenza dei due vettori vettori ora in questa relazione vogliamo usare che ricordate cosa abbiamo qui quindi questo vettore e questo quindi questo sarà d e questo sarà. Quindi possiamo, quindi la differenza è in realtà t si cancella, e abbiamo che la differenza è solo x-star meno x e più uno.

01:20:06 
Quindi due volte questo, il prodotto dei due vettori è v, la norma di v al quadrato, la norma di w al quadrato, e poi la differenza, che è esattamente questa quantità. Okay, quindi ora possiamo sostituire nella relazione precedente, quindi ricordate vogliamo sostituire qui, e quello che otteniamo è questa espressione per, se riorganizzate questa.

01:20:54 
Ora, quello che vogliamo fare ora è sommare su tutte le iterate dal punto di partenza, da zero, fino all'iterazione capitale T. Fate attenzione. Questa è T maiuscola per denotare l'ultima iterazione, e questa è trasposta. Sono diverse. Una è un po' inclinata.

01:21:42 
Dovete sommare questi contributi. Okay, il punto importante è qui. Qui abbiamo una somma telescopica perché abbiamo termini. Quindi qui iniziate da zero e x star fino a t meno x star. Tutti i termini intermedi si stanno cancellando perché quando t è uguale a 1,

01:22:12 
qui avrete un termine del tipo x1 meno x star, e qui 2 meno x star. E poi quello successivo, avrete un termine simile. Quindi si stanno cancellando. E quello che ci rimane sono solo il primo e l'ultimo termine. quindi il termine relativo alla valutazione della differenza tra l'ipotesi iniziale e la vera soluzione e la soluzione finale uh x in capitale t meno la vera soluzione okay quindi questa.

01:22:52 
somma collassa a questa quantità quindi significa che la somma di questa quantità è cosa ora qui abbiamo meno questa quantità ma poiché è una norma è positiva e quindi significa che se cancelliamo, i termini relativi a.

01:23:25 
la soluzione finale, Quello che abbiamo è che non abbiamo più un'uguaglianza, ma abbiamo che questa somma è minore o uguale a questa, okay? Perché abbiamo cancellato una sottrazione di un termine positivo, okay? Quindi, finalmente, ricordate che il termine iniziale che volevamo limitare era questo.

01:24:05 
E quindi, se torniamo qui, l'idea era di considerare questo, okay? E poiché abbiamo usato la somma sul lato destro, abbiamo qualcosa anche qui. E questo è il risultato finale, quindi siamo stati in grado, essenzialmente, di limitare la differenza tra le iterate della valutazione della funzione f sulle iterate meno la f sulla vera soluzione.

01:24:55 
in termini del gradiente e dell'algoritmo iniziale, okay? Il secondo termine è relativo all'algoritmo iniziale, okay? E qui ci sono i gradienti che abbiamo usato durante tutte le iterazioni. Questo risultato... che è basato solo sull'assunzione che la funzione sia convessa.

01:25:28 
Non abbiamo aggiunto nessun'altra assunzione. È il risultato di base che useremo per ottenere risultati più raffinati per diverse categorie di funzioni, principalmente funzioni convesse e Lipschitz o funzioni fortemente convesse. E quindi domani, partendo da questi risultati,

01:26:04 
considereremo alcuni risultati più particolari e più interessanti per categorie specifiche di funzioni. Grazie. Solo una cosa, dato che sto per finire.

---

## Appendice: Neural Networks II - Formule e Approfondimenti Teorici

*Riferimenti slide: NeuralNetworks2.pdf (Lecture November 3rd, Slide 1-34)*

### 1. Cross-Entropy: Soluzione al Learning Slowdown

#### 1.1 Il Problema con MSE (Mean Squared Error)

**Funzione di Costo Quadratica** *(Slide 2/34)*:
```
J = (1/2n) Σ ||y(x) - a^L(x)||
```

**Gradienti nell'ultimo layer**:
```
J/w^L_jk = (a^L_j - y_j)  σ'(z^L_j)  a^L-1_k
J/b^L_j = (a^L_j - y_j)  σ'(z^L_j)
```

** PROBLEMA CRITICO** *(Slide 3/34)*: Il termine **σ'(z^L_j)**

Quando il neurone è **saturato** (output a^L_j vicino a 0 o 1):
- Funzione sigmoid diventa molto piatta
- σ'(z^L_j)  0
- **Conseguenza**: Anche con errore (a^L_j - y_j) grande, gradienti diventano tiny  **learning slowdown**!

**Visualizzazione del problema**:
```
MSE con sigmoid:
- Quando a  0 (errore grande):  J/w  0   BLOCCATO!
- Quando a  1 (errore piccolo): J/w  0   OK

Curva blu (MSE): 
  Cost alto quando a=0, ma derivata  0  learning lento
```

#### 1.2 Cross-Entropy: Definizione e Proprietà

**Definizione** *(Slide 4-5/34)*:
```
J_CE = -(1/n) Σ [ylog(a) + (1-y)log(1-a)]
```

Dove:
- n = numero di campioni
- y  {0,1} = etichetta vera (classificazione binaria)
- a = output della rete (probabilità predetta)

** Proprietà Fondamentali**:

1. **Non-negatività**: J_CE  0 sempre
   - Per y=1: J = -log(a), minimo quando a1
   - Per y=0: J = -log(1-a), minimo quando a0

2. **Minimo sulla soluzione vera**:
   - Se a  y  J_CE  0

3. **No saturazione**: J/w NON contiene σ'!

#### 1.3 Derivazione: La Magia dell Cross-Entropy

**Chain Rule** *(Slide 6-7/34)*:
```
J_CE/w = (J_CE/a)  (a/z)  (z/w)
```

**Passo 1**: Calcolo J_CE/a
```
J_CE/a = -(y/a) + (1-y)/(1-a)
         = -(y - a) / [a(1-a)]
```

**Passo 2**: Derivata sigmoid
```
a/z = σ'(z) = σ(z)[1 - σ(z)] = a(1-a)
```

**Passo 3**: Derivata di z rispetto a w
```
z/w = x  (input del neurone)
```

** RISULTATO MAGICO** *(Slide 8/34)*:
```
J_CE/w = [-(y-a) / a(1-a)]  [a(1-a)]  x
         = -(y - a)  x
         = (a - y)  x

J_CE/b = (a - y)
```

** SPARISCE σ'(z)!**  No learning slowdown!

**Confronto visuale** *(riferimento: curve rosse vs blu nel transcript)*:
```
Cross-Entropy (curva rossa):
- Quando a=0 (errore grande): J/w GRANDE  learning veloce! 
- Quando a=1 (no errore):     J/w  0     corretto 

Derivata proporzionale SOLO all'errore (a-y), non alla saturazione!
```

#### 1.4 Estensione: Multi-Class Cross-Entropy (Softmax)

**Classificazione multi-classe** (K classi):
```
J_CE = -(1/n) Σ_x Σ_k y_k  log(a_k)

dove:
- y_k = 1 se x appartiene a classe k, 0 altrimenti (one-hot encoding)
- a_k = softmax_k(z) = exp(z_k) / Σ_j exp(z_j)
```

**Gradiente con Softmax + Cross-Entropy**:
```
J/z_k = a_k - y_k  (stessa forma elegante!)
```

---

### 2. Overfitting: Diagnosi e Soluzioni

#### 2.1 Cos'è l'Overfitting?

**Definizione** *(Slide 9-10/34)*:
Overfitting = modello si adatta **troppo** ai dati di training  perde capacità di **generalizzazione**

**Esempio visuale**:
```
Dati: Temperatura vs Pressione (relazione lineare vera)

 Overfitting: Polinomio grado 10
   - Passa per TUTTI i punti training
   - Oscillazioni selvagge tra i punti
   - Predizione su nuovi dati: PESSIMA

 Buon fit: Linea retta
   - Approssima la tendenza
   - Errore piccolo su training
   - Generalizza bene su test
```

**Definizione formale**:
```
Overfitting limita la capacità del modello di essere applicato 
ad elementi NON usati per training (dati unseen)
```

#### 2.2 Come Identificare Overfitting: Training vs Validation Curves

**Setup: Dataset Splitting** *(Slide 11-12/34)*:

```
Dataset completo (100%)
  
   Training set (70-80%)     Usato per aggiornare pesi
   Validation set (15-20%)   Usato per monitorare overfitting
   Test set (5-10%)          Usato SOLO a fine training
```

**Procedura di Monitoraggio**:
```
Ad ogni epoca:
  1. Calcola J_train sul training set
  2. Calcola J_val sul validation set (NO gradient updates!)
  3. Plotta entrambe le curve
```

** Pattern Tipico di Overfitting** *(Slide 13/34)*:

```
Epoch  J_train  J_val
---------------------------
0      2.5      2.6     Entrambi alti
50     1.2      1.3     Entrambi scendono
100    0.8      0.9     OK
115    0.6      0.85    J_val inizia a salire! 
150    0.4      1.1     OVERFITTING conclamato
200    0.2      1.5     Training perfetto, validation pessimo

Pattern:
  J_train:  (continua a scendere)
  J_val:    poi  (minimo, poi risale)
          
      Punto di overfitting (epoca ~115)
```

**Early Stopping** *(Slide 14/34)*:
```
Soluzione immediata: FERMARE il training quando J_val inizia a salire

Salvare i pesi al minimo di J_val (non alla fine del training!)
```

#### 2.3 Cause dell'Overfitting

**Cause principali** *(Slide 15/34)*:
1. **Modello troppo complesso** (troppi parametri rispetto ai dati)
   - Rete troppo profonda
   - Troppi neuroni per layer
   
2. **Dati insufficienti** (dataset piccolo)
   - Modello memorizza invece di generalizzare
   
3. **Training troppo lungo** (troppe epoche)
   - Rete si "specializza" su training set

#### 2.4 Strategie Anti-Overfitting

**Strategia 1: Aumentare i Dati** *(Slide 16/34)*

Metodi:
- **Raccolta**: Ottenere più dati (spesso impossibile/costoso)
- **Data Augmentation** (per immagini):
  ```
  Immagine originale  trasformazioni geometriche:
    - Rotazione (15°)
    - Flip orizzontale/verticale
    - Crop random
    - Zoom (0.9-1.1)
    - Traslazione (10% dimensioni)
    - Gaussian noise aggiunto
    - Cambio luminosità/contrasto
  
  1 immagine  10-20 varianti sintetiche
  Dataset 1000 immagini  10,000-20,000 effettive!
  ```

**Strategia 2: Regolarizzazione** (vedi sezione 3)

**Strategia 3: Dropout** (vedi sezione 4)

**Strategia 4: Semplificare Architettura**
- Ridurre numero di layer
- Ridurre neuroni per layer
- Aumentare stride in CNN (più pooling)

---

### 3. Regolarizzazione L1 e L2

#### 3.1 Idea Generale della Regolarizzazione

**Obiettivo** *(Slide 17/34)*:
```
Aggiungere PENALITÀ alla funzione di costo per limitare 
la complessità del modello (peso piccolo  modello semplice)
```

**Funzione di costo regolarizzata**:
```
J_reg = J_original + λ  Ω(w)

dove:
- J_original = funzione di costo base (MSE, Cross-Entropy, etc.)
- λ = parametro di regolarizzazione (hyperparameter, λ  0)
- Ω(w) = termine di penalità sui pesi
```

#### 3.2 Regolarizzazione L2 (Weight Decay)

**Definizione** *(Slide 18-19/34)*:
```
J_L2 = J_original + (λ/2n) Σ_l Σ_i Σ_j (w^l_ij)

        
   Somma su tutti i pesi di tutti i layer
```

** IMPORTANTE**: I **bias NON sono regolarizzati**!
```
Penalità solo su w, non su b (bias meno influenti su overfitting)
```

**Gradiente con L2**:
```
J_L2/w = J_original/w + (λ/n)w
```

**Gradient Descent Update Rule** *(Slide 20/34)*:
```
w_{k+1} = w_k - γJ/w
        = w_k - γ[J_original/w + (λ/n)w_k]
        = w_k - γJ_original/w - (γλ/n)w_k
        = [1 - γλ/n]w_k - γJ_original/w
           
       decay factor < 1
```

**Perché si chiama "Weight Decay"** *(Slide 21/34)*:
```
Fattore (1 - γλ/n) < 1  ad ogni iterazione w viene "ridotto"

Esempio numerico:
  γ = 0.01, λ = 1, n = 1000
   decay factor = 1 - 0.011/1000 = 0.99999

w_{k+1} = 0.99999w_k - ...
          
     Riduzione dello 0.001% per iterazione

Dopo 10,000 iterazioni: w ridotto del ~10%
```

**Effetto geometrico**: L2 spinge verso **norma minima**
```
min ||w||  Vettore w con minima norma Euclidea
              Pesi piccoli, uniformi
              Modello più "smooth"
```

#### 3.3 Regolarizzazione L1 (Sparsity)

**Definizione** *(Slide 22/34)*:
```
J_L1 = J_original + (λ/n) Σ_l Σ_i Σ_j |w^l_ij|
```

**Gradiente con L1**:
```
J_L1/w = J_original/w + (λ/n)sign(w)

dove sign(w) = +1 se w > 0
             = -1 se w < 0
             =  0 se w = 0
```

**Update Rule** *(Slide 23/34)*:
```
w_{k+1} = w_k - γJ_original/w - (γλ/n)sign(w_k)
```

**Effetto**: Spinta verso **sparsità** (molti pesi = 0)
```
Ogni iterazione: w ridotto di costante (γλ/n), indipendente da w

Esempio:
  w = 0.001, γλ/n = 0.0005
   w_new = 0.001 - 0.0005 = 0.0005  (riduzione 50%)
  
  w = 1.0, γλ/n = 0.0005
   w_new = 1.0 - 0.0005 = 0.9995  (riduzione 0.05%)

Pesi piccoli  azzerati rapidamente
Pesi grandi  riduzione lenta

Risultato: Rete con molti w = 0 (SPARSE)
```

#### 3.4 Confronto Geometrico L1 vs L2

**Visualizzazione** *(Slide 24/34, riferimento transcript "immagine in nero tratteggiato")*:

```
Problema vincolato equivalente:
  min J(w)  s.t.  ||w||  C

Curve di livello J (ellissi nere):
  - Centro = minimo senza vincolo
  - Vogliamo trovare intersezione con palla unitaria

Palla L2 (cerchio blu):
  ||w||  1   {w: w + w  1}
  - Forma: CERCHIO
  - Intersezione: punto generico sul cerchio
  - Risultato: w  0, w  0 (entrambi piccoli ma non zero)

Palla L1 (diamante rosso):
  ||w||  1   {w: |w| + |w|  1}
  - Forma: DIAMANTE (quadrato ruotato 45°)
  - Vertici: (1,0), (0,1), (-1,0), (0,-1)
  - Intersezione: MOLTO probabilmente su un vertice!
  - Risultato: w = 0 O w = 0 (SPARSITÀ)

Perché L1  sparsità:
  Ellissi "toccano" diamante sui vertici (angoli acuti)
   Coordinate zero naturalmente
```

**Tabella riassuntiva**:
```
| Aspetto            | L2 (Ridge)           | L1 (Lasso)          |
|--------------------|----------------------|---------------------|
| Penalità           | Σ w                 | Σ |w|               |
| Effetto geometrico | Cerchio              | Diamante            |
| Risultato pesi     | Piccoli, uniformi    | Sparsi (molti = 0)  |
| Selezione features | NO                   | SÌ (azzera features)|
| Interpretabilità   | Bassa                | Alta (pochi w  0)  |
| Derivabilità       | Ovunque              | Non in w=0          |
```

#### 3.5 Elastic Net: Combinazione L1 + L2

**Definizione**:
```
J_elastic = J_original + λ||w|| + λ||w||

Combina vantaggi di entrambi:
  - L2: stabilità numerica
  - L1: sparsità
```

---

### 4. Dropout: Ensemble di Reti

#### 4.1 Idea del Dropout

**Definizione** *(Slide 25-26/34, transcript 00:34:33)*:
```
Durante il training: CASUALMENTE elimina una percentuale p 
di neuroni nei layer nascosti ad ogni forward/backward pass

Parametro: p = probabilità di dropout (es. p=0.5  50% neuroni eliminati)
```

**Procedura** *(Slide 27/34)*:

**Training Phase**:
```
Ad ogni mini-batch:
  1. Per ogni layer nascosto:
       - Genera maschera binaria random: m ~ Bernoulli(1-p)
       - Applica maschera: h_dropout = h  m
       - Effettivamente: imposta 50% neuroni a zero
  
  2. Forward pass con neuroni sopravvissuti
  
  3. Backward pass (solo neuroni attivi ricevono gradienti)
  
  4. Aggiorna pesi (solo pesi connessi a neuroni attivi)
  
  5. RIPETI con NUOVA maschera random al prossimo batch

Esempio con p=0.5:
  Layer con 1000 neuroni  ogni batch usa ~500 neuroni diversi
   Effetto: training su sottorete diversa ad ogni iterazione!
```

**Testing/Inference Phase** *(Slide 28/34)*:
```
NO dropout durante test!

Usa TUTTI i neuroni, ma:
  - Moltiplica output per (1-p) per compensare

Perché? Durante training, output medio era ridotto (50% neuroni off)
        Durante test, tutti neuroni on  output 2 più grande
        
Correzione: h_test = (1-p)  h_all_neurons

Esempio: p=0.5  h_test = 0.5  h
```

#### 4.2 Perché Dropout Funziona?

**Interpretazione 1: Ensemble Learning** *(Slide 29/34)*:
```
Training con dropout = training di MOLTE sottorete diverse

Esempio: Layer con 1000 neuroni, p=0.5
   Numero di possibili sottorete = C(1000, 500)  10

Durante training: Campioniamo ~1000-10000 sottorete diverse

Test: Predizione = MEDIA approssimata di tutte le sottorete
      (come ensemble in Random Forest)
```

**Analogia con Meteorologia** *(transcript 00:37:11)*:
```
Previsioni meteo: Ensemble di simulazioni
  - Condizioni iniziali: T=20°C  0.5°C (incertezza)
  - Esegui 50 simulazioni con valori random in range
  - Predizione finale: MEDIA delle 50 simulazioni
  
Dropout: Ensemble di architetture
  - Topologia: ~500/1000 neuroni (incertezza strutturale)
  - Esegui training con ~1000 configurazioni random
  - Predizione finale: MEDIA implicita delle configurazioni
```

**Interpretazione 2: Feature Redundancy Reduction**:
```
Senza dropout:
  - Neuroni possono "co-adattarsi" (correggere errori di altri)
  - Alcuni neuroni diventano "lazy" (dipendono da altri)
  
Con dropout:
  - Ogni neurone deve essere "robusto" da solo
  - Non può dipendere da specifici altri neuroni (potrebbero essere off!)
  - Forza apprendimento di feature INDIPENDENTI
```

#### 4.3 Dropout in TensorFlow/Keras

**Codice esempio** *(Slide 30/34)*:
```python
from tensorflow.keras.layers import Dropout, Dense

model = Sequential([
    Dense(1000, activation='relu', input_shape=(784,)),
    Dropout(0.5),  #  50% neuroni droppati
    
    Dense(500, activation='relu'),
    Dropout(0.3),  #  30% neuroni droppati
    
    Dense(10, activation='softmax')  #  NO dropout su output!
])

# Durante training: dropout automaticamente applicato
model.fit(X_train, y_train, epochs=50)

# Durante test: dropout automaticamente disabilitato
predictions = model.predict(X_test)
```

**Best Practices**:
```
1. Dropout rate tipici: 0.2-0.5 (20-50%)
   - Layer grandi: p=0.5
   - Layer piccoli: p=0.2-0.3

2. NO dropout su:
   - Layer di output
   - Layer convoluzionali iniziali (CNN)
   - Batch normalization (ridondante)

3. Dropout + L2 regolarization: Possono coesistere!
   - Effetti complementari
   - L2 controlla magnitudine pesi
   - Dropout controlla co-adattamento
```

#### 4.4 Inverted Dropout (Implementazione Moderna)

**Problema**: Scaling durante test è scomodo

**Soluzione**: Inverted Dropout *(Slide 31/34)*:
```
Training Phase:
  h_dropout = (h  m) / (1-p)   Scala UP durante training!
  
Testing Phase:
  h_test = h   Nessun scaling necessario!

Esempio: p=0.5
  - Training: output medio rimane uguale (divisione per 0.5  2)
  - Test: usa output diretto (già nella scala giusta)

Vantaggi:
   Test più semplice (no moltiplicazione)
   Implementazione TensorFlow/PyTorch usa questo metodo
```

---

### 5. Weight Initialization: Evitare Vanishing/Exploding Gradients

#### 5.1 Il Problema dell'Inizializzazione

**Setup** *(Slide 32/34, transcript 00:39:51)*:
```
Inizializzazione ingenua:
  w ~ N(0, 1)  (Gaussiana con mean=0, std=1)
  b ~ N(0, 1)

Neurone con n input:
  z = Σ(w_i  x_i) + b

Assunzione: input normalizzati (x_i ~ N(0,1) o x_i  [0,1])
```

**Analisi statistica**:
```
z = wx + wx + ... + wx + b

Se w_i ~ N(0, 1) e x_i ~ N(0, 1) indipendenti:
   Prodotto w_ix_i ~ N(0, 1)
   Somma di n termini: z ~ N(0, n)

Esempio: n=1000 input
   σ_z  1000  31.6

z può avere valori [-100, +100]!
```

** PROBLEMA** *(Slide 33/34)*:
```
z molto grande  sigmoid saturo  σ'(z)  0
                              
                    VANISHING GRADIENT al primo step!

Prima iterazione: rete già "bloccata" in regione piatta
```

#### 5.2 Xavier/Glorot Initialization

**Soluzione** *(Slide 34/34, transcript 00:43:37)*:
```
Idea: Adattare varianza pesi al numero di connessioni

Xavier Init:
  w ~ N(0, σ)  dove σ = 1/n_in

dove n_in = numero di input del neurone
```

**Analisi**:
```
Con w ~ N(0, 1/n_in):
  z = Σ(w_i  x_i)  σ_z = (n  σ_w) = (n  1/n) = 1

Risultato: σ_z = 1 indipendentemente da n!

Esempio: n=1000
  Vecchio: σ_z  31.6  sigmoid saturo
  Xavier:  σ_z = 1     sigmoid in range lineare 
```

**Variante per ReLU** (He Initialization):
```
w ~ N(0, 2/n_in)

Perché 2 più grande?
  ReLU taglia 50% neuroni (output = 0 per z < 0)
   Compensare con varianza doppia
```

#### 5.3 Altri Metodi di Inizializzazione

**LeCun Initialization** (reti deep):
```
w ~ N(0, 1/n_in)  (come Xavier)
Usato per: SELU, ELU activation functions
```

**Uniform Xavier**:
```
w ~ U(-(6/(n_in + n_out)), +(6/(n_in + n_out)))

Distribuzione uniforme invece di Gaussiana
```

**Orthogonal Initialization** (RNN):
```
W = matrice ortogonale random (W^T W = I)

Vantaggi per RNN:
  - Preserva norma gradiente attraverso timesteps
  - Previene vanishing/exploding in sequenze lunghe
```

**Tabella riassuntiva**:
```
| Metodo      | Distribuzione     | Uso ottimale        |
|-------------|-------------------|---------------------|
| Xavier      | N(0, 1/n_in)     | Sigmoid, Tanh       |
| He          | N(0, 2/n_in)     | ReLU, LeakyReLU     |
| LeCun       | N(0, 1/n_in)     | SELU                |
| Orthogonal  | Matrice Q da QR   | RNN, LSTM           |
```

#### 5.4 Implementazione in TensorFlow/Keras

```python
from tensorflow.keras.initializers import GlorotUniform, HeNormal

model = Sequential([
    Dense(128, 
          activation='relu',
          kernel_initializer=HeNormal(),  #  He per ReLU
          bias_initializer='zeros'),
    
    Dense(64, 
          activation='tanh',
          kernel_initializer=GlorotUniform(),  #  Xavier per tanh
          bias_initializer='zeros'),
    
    Dense(10, activation='softmax')
])
```

---

### 6. Hyperparameter Tuning: Strategie Pratiche

#### 6.1 Principali Hyperparameters

**Lista completa** *(transcript 00:45:38)*:
```
1. Learning rate (γ, η)            PIÙ CRITICO!
2. Regolarizzazione (λ)
3. Mini-batch size
4. Numero epoche
5. Architettura:
   - Numero layer
   - Neuroni per layer
   - Funzioni di attivazione
6. Dropout rate (p)
7. Optimizer (SGD, Adam, RMSprop)
8. Weight initialization
```

#### 6.2 Strategia di Tuning (Order Matters!)

**Procedura standard** *(Slide transcript 00:46:12)*:

**Step 1: Learning Rate** (PRIORITÀ MASSIMA)
```
1. Disabilita regolarizzazione (λ=0)
2. Prova learning rate iniziale: γ = 0.01
3. Osserva andamento J_train:
   
   Caso A: J_train ESPLODE (aumenta)
      γ troppo grande, riduci: γ = 0.001
   
   Caso B: J_train scende MOLTO lentamente
      γ troppo piccolo, aumenta: γ = 0.1
   
   Caso C: J_train scende smooth, poi oscilla
      γ leggermente grande, riduci 2: γ = 0.005
   
   Caso D: J_train scende smooth fino a plateau
      γ ottimale! 

4. Range tipici da esplorare:
   [0.0001, 0.001, 0.01, 0.1, 1.0]
   Usa scala LOGARITMICA!
```

**Step 2: Regolarizzazione λ**
```
Prerequisito: γ ottimale già trovato

1. Inizia con λ = 1.0
2. Osserva pattern J_train vs J_val:
   
   Caso A: J_val aumenta presto (epoca < 50)
      Overfitting forte, aumenta λ: λ = 10
   
   Caso B: J_val e J_train quasi uguali per tutte epoche
      Underfitting, riduci λ: λ = 0.1
   
   Caso C: J_val ha minimo, poi sale dopo epoca 100
      λ buono! Oppure prova aumentare leggermente

3. Range tipici: [0.0001, 0.001, 0.01, 0.1, 1, 10]
   Scala logaritmica anche qui!
```

**Step 3: Re-optimize Learning Rate**
```
Con λ ottimale trovato:
  - Ritorna a Step 1
  - Riottimizza γ (può essere cambiato)
  - Itera 2-3 volte fino a convergenza
```

**Step 4: Mini-batch Size**
```
Range tipici: [32, 64, 128, 256, 512]

Considerazioni:
  - Batch piccolo (32):  aggiornamenti frequenti, rumorosi
  - Batch grande (512):  aggiornamenti rari, smooth
  
Trade-off:
   Batch grande  convergenza stabile, ma richiede memoria GPU
   Batch piccolo  esplora meglio, ma più lento

Regola empirica: batch_size = 32-128 per dataset < 100k samples
```

**Step 5: Architettura** (trial & error)
```
Inizia semplice, aumenta complessità se necessario:
  1. Rete shallow (2-3 layer)
  2. Se underfitting  aggiungi layer
  3. Se overfitting  riduci neuroni, aumenta λ, aggiungi dropout
```

#### 6.3 Visualizzazione: Effetto del Learning Rate

**Grafico qualitativo** *(transcript 01:13:29, riferimento immagine parabola)*:

```
J(w)
      
     γ troppo grande (magenta): DIVERGE
      
       γ grande (rosso): OSCILLA
      
          γ troppo piccolo (blu): LENTISSIMO
           
 ___________ γ ottimale (verde): SMOOTH & VELOCE
  Iterazioni

Messaggio chiave:
  - Troppo grande  instabilità/divergenza
  - Troppo piccolo  convergenza lenta (spreco computazionale)
  - Ottimale  sweet spot (convergenza veloce E stabile)
```

**Esempio numerico**:
```
Parabola: J(w) = w
Minimo: w* = 0

Update: w_{k+1} = w_k - γJ/w = w_k - γ2w_k = (1 - 2γ)w_k

Convergenza: |1 - 2γ| < 1
             0 < γ < 1

γ = 0.01:  w_k = 0.98^kw_0   lento (98 iter per 86% riduzione)
γ = 0.5:   w_k = 0^kw_0      immediato! (ma solo su parabola)
γ = 0.75:  w_k = (-0.5)^kw_0  OSCILLA (segno alterna)
γ = 1.5:   w_k = (-2)^kw_0    ESPLODE!
```

#### 6.4 Learning Rate Scheduling

**Problema**: γ fisso non è ottimale
```
Inizio training: Grande errore  γ grande OK
Fine training:  Piccolo errore  γ grande causa oscillazioni
```

**Soluzioni**:

**1. Step Decay**:
```
γ_k = γ_0  0.5^(floor(epoch / 10))

Esempio: γ_0 = 0.1
  Epoch 0-9:   γ = 0.1
  Epoch 10-19: γ = 0.05
  Epoch 20-29: γ = 0.025
  ...
```

**2. Exponential Decay**:
```
γ_k = γ_0  exp(-kdecay_rate)

Esempio: γ_0 = 0.1, decay_rate = 0.01
  Epoch 0:   γ = 0.1
  Epoch 50:  γ = 0.061
  Epoch 100: γ = 0.037
```

**3. Cosine Annealing**:
```
γ_k = γ_min + 0.5(γ_max - γ_min)(1 + cos(πk / K))

K = epoche totali
Warm restart dopo ogni ciclo (popolare in deep learning)
```

---

### 7. Gradient Descent: Fondamenti Teorici

#### 7.1 Setup del Problema

**Problema di Ottimizzazione** *(transcript 00:54:05)*:
```
min f(x)  dove x  ℝ^d

Nel nostro caso:
  - f = funzione di costo (J_CE, J_MSE, etc.)
  - x = vettore dei pesi w (e bias b)
  - d = dimensionalità (numero totale di parametri)
```

**Metodo iterativo**:
```
x_{k+1} = x_k + v_k

dove v_k = direzione di aggiornamento
```

**Obiettivo**: Trovare v_k tale che f(x_{k+1}) < f(x_k)

#### 7.2 Derivazione della Direzione di Discesa

**Espansione di Taylor al primo ordine** *(transcript 01:05:38)*:
```
f(x_k + v_k)  f(x_k) + f(x_k)^T  v_k

Per avere f(x_{k+1}) < f(x_k):
   f(x_k)^T  v_k < 0  (prodotto scalare negativo)
```

**Scelta ottimale di v_k**:
```
v_k = -γf(x_k)

Perché?
  f(x_k)^T  v_k = f(x_k)^T  (-γf(x_k))
                  = -γ||f(x_k)|| < 0  

Massimizza la riduzione al primo ordine!
```

**Regola di Gradient Descent** *(Slide GradientDescent_v1.pdf)*:
```
x_{k+1} = x_k - γf(x_k)

Parametri:
  - γ > 0: learning rate (step size)
  - f(x_k): gradiente valutato in x_k
```

#### 7.3 Proprietà Matematiche Necessarie

**Definizioni fondamentali** *(transcript 00:55:45 - 01:02:41)*:

**1. Convessità**:
```
f convessa  x,y, λ[0,1]:
  f(λx + (1-λ)y)  λf(x) + (1-λ)f(y)

Interpretazione geometrica:
  Segmento che unisce (x, f(x)) e (y, f(y)) 
  sta SOPRA il grafico di f
```

**2. First-Order Characterization** (f differenziabile):
```
f convessa  x,y:
  f(y)  f(x) + f(x)^T(y - x)

Interpretazione:
  f sta sempre SOPRA la sua tangente
  (piano tangente è lower bound)
```

**3. L-Smoothness** (Lipschitz continuous gradient):
```
f è L-smooth  x,y:
  ||f(x) - f(y)||  L||x - y||

Equivalentemente:
  f(y)  f(x) + f(x)^T(y-x) + (L/2)||y-x||
                                      
    valore      termine lineare    upper bound quadratico

Interpretazione:
  Curvatura di f limitata (Hessian eigenvalues  L)
  f sta SOTTO parabola con curvatura L
```

**4. μ-Strong Convexity**:
```
f è μ-strongly convex  x,y:
  f(y)  f(x) + f(x)^T(y-x) + (μ/2)||y-x||

Interpretazione:
  f sta SOPRA parabola con curvatura μ
  Crescita almeno quadratica lontano dal minimo
```

**Visualizzazione** *(riferimento transcript immagini)*:
```
          Upper bound (L-smooth)
                               
            ___f(x)___          
       ____          ____      
                           ____  
       Lower bound (μ-strong)    __
    

Regione ombreggiata = dove f può stare
  - Sopra: parabola con curvatura μ (lower)
  - Sotto: parabola con curvatura L (upper)
```

#### 7.4 Analisi di Convergenza: Caso Convesso

**Teorema Base** *(transcript 01:15:52 - 01:26:04)*:

**Assunzione**: f convessa

**Risultato**:
```
(1/T)Σ_{t=0}^{T-1} [f(x_t) - f(x*)]  ||x_0 - x*|| / (2γT)

dove:
  - x* = minimizer (f(x*) = min f)
  - T = numero iterazioni
  - γ = learning rate
```

**Interpretazione**:
```
Media dei gap f(x_t) - f(x*)  O(1/T)

 Convergenza SUBLINEARE
 Per ridurre errore di fattore 2, servono 2 iterazioni
```

**Dimostrazione (sketch)**:

**Step 1**: Caratterizzazione convessità
```
f(x*)  f(x_t) + f(x_t)^T(x* - x_t)

Riorganizzando:
  f(x_t) - f(x*)  f(x_t)^T(x_t - x*)   vogliamo limitare questo
```

**Step 2**: Update rule di GD
```
x_{t+1} = x_t - γf(x_t)

 f(x_t) = (x_t - x_{t+1}) / γ
```

**Step 3**: Sostituisci gradiente
```
f(x_t)^T(x_t - x*) = (1/γ)(x_t - x_{t+1})^T(x_t - x*)
```

**Step 4**: Sviluppo norma
```
2b^Tw = ||b|| + ||w|| - ||b - w||

Scegli b = x_t - x_{t+1}, w = x_t - x*:

2(x_t - x_{t+1})^T(x_t - x*) 
  = ||x_t - x_{t+1}|| + ||x_t - x*|| - ||x_{t+1} - x*||
```

**Step 5**: Combina e somma
```
f(x_t) - f(x*)  (1/2γ)[||x_t - x*|| - ||x_{t+1} - x*||]

Somma telescopica da t=0 a T-1:
  Σ [f(x_t) - f(x*)]  (1/2γ)[||x_0 - x*|| - ||x_T - x*||]
                      ||x_0 - x*|| / (2γ)

Dividi per T:
  (1/T)Σ [f(x_t) - f(x*)]  ||x_0 - x*|| / (2γT)  QED
```

**Implicazioni pratiche**:
```
- Convergenza garantita ma LENTA
- T = 10,000 iter  ε = O(1/10,000)
- Per ε/2 serve T = 20,000 (doppio!)
- OK per convesse semplici, LENTO per deep learning
```

#### 7.5 Estensioni (Accennate, dettagli Lez19)

**Caso L-smooth + convesso**:
```
Con γ = 1/L (learning rate ottimale):
  f(x_T) - f(x*)  (L/2T)||x_0 - x*||

Convergenza O(1/T) sul BEST iterate (non media)
```

**Caso μ-strongly convex + L-smooth**:
```
Con γ = 2/(μ + L):
  ||x_T - x*||  [(L-μ)/(L+μ)]^T  ||x_0 - x*||

Convergenza LINEARE! (esponenziale)
Condition number κ = L/μ determina velocità
```

---

### 8. Riferimenti Bibliografici e Immagini

**Slide PDF**: NeuralNetworks2.pdf (Lecture November 3rd, 34 slide)

**Slide chiave citate**:
- Slide 2-3/34: MSE learning slowdown problem
- Slide 4-8/34: Cross-entropy derivation
- Slide 9-14/34: Overfitting diagnosis (train/val curves)
- Slide 15-24/34: Regolarizzazione L1/L2 (geometria)
- Slide 25-31/34: Dropout mechanism
- Slide 32-34/34: Weight initialization (Xavier/He)

**Immagini presenti nelle slide** (non estratte ma descritte):
1. **Slide 3**: Grafico MSE cost + derivata (curve blu) vs Cross-Entropy (curve rosse)
2. **Slide 13**: Train/Validation curves con punto di overfitting (epoca 115)
3. **Slide 24**: Confronto geometrico L1 (diamante rosso) vs L2 (cerchio blu) con ellissi
4. **Slide parabol nel transcript**: Learning rate effects (blu lento, verde ottimale, rosso oscillante, magenta divergente)

**Paper References**:
1. **Rumelhart, Hinton, Williams (1986)**: "Learning representations by back-propagating errors" - Backpropagation originale
2. **Hinton et al. (2012)**: "Improving neural networks by preventing co-adaptation of feature detectors" - Dropout paper originale
3. **Glorot & Bengio (2010)**: "Understanding the difficulty of training deep feedforward neural networks" - Xavier initialization
4. **He et al. (2015)**: "Delving Deep into Rectifiers: Surpassing Human-Level Performance" - He initialization per ReLU
5. **Goodfellow, Bengio, Courville (2016)**: "Deep Learning" - Cap. 7 (Regularization), Cap. 8 (Optimization)

---

**Fine Lezione 18 - Neural Networks II**

*Prossima lezione: Stochastic Gradient Descent (SGD), Mini-batch, Momentum, Adam optimizer*

