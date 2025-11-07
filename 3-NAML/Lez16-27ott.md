# Lez16-27ott - Reti Neurali e Backpropagation

## 🎯 Obiettivi della Lezione

### Concetti Teorici
- Comprendere la storia del **perceptron** (1950s) e sue limitazioni
- Studiare la **funzione sigmoid** e sue proprietà matematiche
- Analizzare **architetture** di reti neurali (shallow vs deep, feedforward)
- Apprendere la **notazione matriciale** (W^L_jk, b^L_j, a^L_j, z^L_j)
- Derivare le **4 equazioni fondamentali** della backpropagation
- Comprendere il problema del **gradient vanishing**

### Algoritmi e Metodi
- **Backpropagation**: algoritmo efficiente per calcolare gradienti
- **Forward pass**: calcolare tutte le attivazioni a^l, z^l
- **Backward pass**: calcolare tutti i δ^l usando le 4 equazioni
- **Hadamard product (⊙)**: moltiplicazione element-wise
- **Gradient descent**: aggiornamento pesi W → W - η·∇_W C

### Derivazioni Matematiche
- **Equazione 1**: δ^L = ∇_a C ⊙ σ'(z^L) - errore output layer
- **Equazione 2**: δ^l = ((W^(l+1))^T·δ^(l+1)) ⊙ σ'(z^l) - propagazione backward
- **Equazione 3**: ∂C/∂b^l = δ^l - gradiente bias
- **Equazione 4**: ∂C/∂W^l_jk = a^(l-1)_k·δ^l_j - gradiente pesi

## 📚 Prerequisiti

### Matematica
- **Calcolo**: derivate parziali, regola della catena, gradiente
- **Algebra Lineare**: prodotti matrice-vettore, trasposizione
- **Probabilità**: funzioni di probabilità, interpretazione output

### Teoria (da lezioni precedenti)
- **Automatic Differentiation**: forward/reverse mode (Lez15)
- **Chain rule**: applicazione sistematica
- **Gradient descent**: ottimizzazione iterativa

## 📑 Indice Completo

### [Parte 1: Storia e Motivazione](#parte-1-storia-e-motivazione) (`00:00` - `22:42`)
1. [Prospettiva Storica: Il Percettrone](#prospettiva-storica-il-percettrone) - `00:00:09`
2. [Perceptron (1950s): Binary Input/Output](#perceptron-1950s) - `00:02:34`
3. [Step Function: w·x + b](#step-function) - `00:05:17`
4. [NAND Gate: Universal Gate](#nand-gate-universal-gate) - `00:08:42`
5. [Limiti del Percettrone](#limiti-del-percettrone-e-funzione-sigmoid) - `00:09:25`
6. [Step Function Non-Differenziabile](#step-function-non-differenziabile) - `00:11:48`
7. [Sigmoid: σ(z) = 1/(1+e^(-z))](#sigmoid-σz--11e-z) - `00:14:31`
8. [Proprietà: σ'(z) = σ(z)(1-σ(z))](#proprietà-σz--σz1-σz) - `00:17:55`
9. [Smooth Transition 0→1](#smooth-transition-01) - `00:20:18`

### [Parte 2: Architettura Neural Networks](#parte-2-architettura) (`22:42` - `40:27`)
10. [Architettura delle Reti Neurali](#architettura-delle-reti-neurali) - `00:22:42`
11. [Input Layer: #neurons = #features](#input-layer) - `00:24:29`
12. [Hidden Layer(s): 1→Shallow, 2+→Deep](#hidden-layers) - `00:26:53`
13. [Output Layer: #neurons = #outputs](#output-layer) - `00:29:17`
14. [Fully Connected Feedforward](#fully-connected-feedforward) - `00:31:42`
15. [Notazione: W^L_jk, b^L_j, a^L_j](#notazione-wl_jk-bl_j-al_j) - `00:34:11`
16. [Forma Vettorizzata: z^L = W^L·a^(L-1) + b^L](#forma-vettorizzata) - `00:36:58`
17. [Funzione di Costo e Assunzioni](#funzione-di-costo-e-assunzioni) - `00:33:48`
18. [C = (1/n)Σ_x C_x](#c--1nσ_x-c_x) - `00:38:33`

### [Parte 3: Quattro Equazioni Fondamentali](#parte-3-quattro-equazioni) (`40:27` - `01:00:15`)
19. [Equazioni Fondamentali della Backpropagation](#equazioni-fondamentali-della-backpropagation) - `00:40:27`
20. [Hadamard Product: s⊙t](#hadamard-product) - `00:42:54`
21. [Eq. 1: δ^L = ∇_a C ⊙ σ'(z^L)](#eq-1-δl--∇_a-c--σzl) - `00:45:18`
22. [Derivazione Eq. 1](#derivazione-eq-1) - `00:48:06`
23. [Eq. 2: δ^l = ((W^(l+1))^T·δ^(l+1)) ⊙ σ'(z^l)](#eq-2-δl) - `00:50:34`
24. [Derivazione Eq. 2](#derivazione-eq-2) - `00:53:29`
25. [Eq. 3: ∂C/∂b^l = δ^l](#eq-3-∂c∂bl--δl) - `00:56:17`
26. [Eq. 4: ∂C/∂W^l_jk = a^(l-1)_k·δ^l_j](#eq-4-∂c∂wl_jk) - `00:58:42`

### [Parte 4: Algoritmo e Problemi](#parte-4-algoritmo) (`01:00:15` - `01:26:04`)
27. [Algoritmo Backpropagation](#algoritmo-backpropagation) - `01:00:15`
28. [1. Initialize W, b](#initialize-w-b) - `01:02:38`
29. [2. Forward Pass: Compute a^l, z^l](#forward-pass) - `01:05:11`
30. [3. Compute δ^L (Eq. 1)](#compute-δl) - `01:07:44`
31. [4. Backpropagate δ^l (Eq. 2)](#backpropagate-δl) - `01:10:19`
32. [5. Compute Gradients (Eq. 3-4)](#compute-gradients) - `01:12:53`
33. [6. Update: W → W - η·∇C](#update-weights) - `01:15:28`
34. [Gradient Vanishing Problem](#gradient-vanishing-problem) - `01:18:05`
35. [σ' ≈ 0 quando Neurone Saturato](#σ-≈-0-quando-saturato) - `01:20:37`
36. [Weight Initialization Critica](#weight-initialization) - `01:23:11`

### [Parte 5: Funzioni di Attivazione](#parte-5-funzioni-attivazione) (`01:26:04` - `01:43:27`)
37. [Funzioni di Attivazione Alternative](#funzioni-attivazione) - `01:26:04`
38. [Sigmoid: Output come Probabilità](#sigmoid-output-probabilità) - `01:27:12`
39. [Tanh: Centrata su Zero](#tanh-centrata-zero) - `01:29:48`
40. [ReLU: max(0,x)](#relu-max0x) - `01:32:27`
41. [Leaky ReLU: Pendenza Piccola Negativa](#leaky-relu) - `01:35:53`
42. [Parametric ReLU (PReLU)](#parametric-relu) - `01:38:19`
43. [Softmax: Layer Output per Classificazione](#softmax-layer-output) - `01:40:46`
44. [Scelta Funzione di Attivazione](#scelta-funzione) - `01:43:27`

---

## Parte 1: Storia e Motivazione

## Prospettiva Storica: Il Percettrone

---

## Prospettiva Storica: Il Percettrone

`00:00:09` 
Okay, buon pomeriggio, oggi avremo una prima lezione, che è un'introduzione a, in questa lezione cercheremo di impostare alcuni elementi di cui avremo bisogno nel seguito, e discuteremo alcune delle connessioni tra ciò che abbiamo visto l'ultima volta, quindi la.

`00:00:44` 
differenziazione automatica, e neurale. Quindi iniziamo con una piccola prospettiva storica. Immagino che tutti voi sappiate che il primo tentativo di sviluppare una struttura, una struttura teorica che assomigliasse a un neurone risale agli anni '50.

`00:01:16` 
Il modello che è stato sviluppato è stato chiamato percettrone. Cos'è un percettrone? Un percettrone è essenzialmente un elemento che è caratterizzato da. un certo numero di input binari, quindi significa che possono essere o 0 o 1, per esempio,

`00:01:47` 
ed è supposto darvi solo un singolo numero binario. Quindi un'immagine formale di un percettrone con tre input è così. Quindi ora quello che vogliamo capire più in dettaglio è cosa c'è dentro questo, cerchio. L'idea è che assoceremo a ciascun input un parametro che è chiamato peso.

`00:02:27` 
E questi parametri essenzialmente vi dicono qual è l'importanza di ciascun input per calcolare l'output. E poi si sommano tutti questi input pesati, e l'output è calcolato in base al confronto tra questa somma pesata e una soglia che è data.

`00:03:07` 
Quindi, formalmente, possiamo dire che l'output è o zero. Se la somma pesata è inferiore o uguale alla soglia data, o è uguale a uno se la somma pesata è maggiore della soglia data. Questa è la formulazione originale. Oggigiorno, la formulazione in cui avete questa soglia non è più usata. E normalmente, il percettrone è scritto in termini di pesi e un altro parametro, che è chiamato bias.

`00:03:54` 
È un termine che useremo anche nel contesto di reti neurali più generali. E quindi, praticamente, la somma pesata è solitamente scritta come un prodotto scalare tra un vettore peso e il vettore di input, x. Si introduce il bias, che non è altro che l'opposto della soglia.

`00:04:30` 
E infine, potete riscrivere la formula che abbiamo visto prima come 0 se wx più b è minore o uguale a 0, o 1 se wx più b è maggiore di 0. Questa è la formulazione usuale che è usata nella pratica.

`00:05:06` 
Qual è il significato del bias? Il bias è che i pesi vi danno l'importanza relativa di ciascun input. E il bias è un parametro che vi dice quanto è facile ottenere uno. Quindi qui ho usato l'espressione, se il quanto è facile per il percettrone attivarsi.

`00:05:38` 
Perché? Perché originariamente il percettrone e tutta la macchina della rete neurale sono stati sviluppati cercando di imitare il più possibile i neuroni nel cervello. Quindi sappiamo che chimicamente i neuroni nel cervello o sparano il segnale o no, a seconda di una certa soglia che è data.

`00:06:09` 
Quindi il bias è una misura che vi dice quanto è facile ottenere uno. Quanto è facile ottenere uno. È chiaro che se b è positivo, grande e positivo, è facile ottenere uno. Se è grande e negativo, è molto difficile avere uno. Okay? Quindi, teniamo a mente questa conclusione perché torneremo su questo punto più tardi.

`00:06:49` 
Okay, qui, voglio solo fare un commento. Nella letteratura, c'è un nome comune, che è MLP. Immagino che molti di voi abbiano già incontrato questo acronimo, che sta per Multi-Layer Perceptron. Okay? Dal mio punto di vista questa affermazione è fuorviante. Perché? Perché percettrone, come definizione, è esattamente una struttura che è costruita in questa forma.

`00:07:29` 
OK, quindi questo è il percettrone. Quindi, multi-layer perceptron significa essenzialmente che state impilando uno dopo l'altro molte strutture come questa. In pratica, questo acronimo è usato anche per denotare reti neurali in generale, che, da un punto di vista strettamente teorico, non è totalmente corretto, perché un percettrone è un neurone molto particolare, non un neurone generico.

`00:08:06` 
Ma siate consapevoli che, potete trovare MLP per denotare le reti neurali generali. Qui, ho solo riportato una tabella riguardante una delle porte logiche ben note, che è la porta logica NOT-AND che si ottiene molto facilmente calcolando l'AND e poi negando l'AND, che è esattamente quello che avete qui.

`00:08:47` 
Perché questa porta è importante? Perché la porta NOT-AND è un esempio di una cosiddetta porta universale. Cos'è una porta universale? Una porta universale è una porta logica che può essere usata per costruire qualsiasi altra porta logica. Quindi, significa che combinando opportunamente molte... porte NOT-AND, potete creare NOT, per esempio, AND, o ecc.

`00:09:25` 
Okay, qualsiasi altra, per esempio, il NOT è abbastanza semplice. Se fate collassare entrambi gli input dell'AND in uno, per esempio, 0, 0, è solo uno, allora è chiaro che ottenete uno, o se sono entrambi uno, ottenete 0. Okay, quindi questo è il più semplice.

## Limiti del Percettrone e Funzione Sigmoid

`00:09:57` 
Poi, per esempio, per creare l'AND, potete usare il NOT-AND, quindi la porta stessa, e poi applicate il NOT costruito con un uso adeguato del NOT-AND, quindi state impilando una dopo l'altra due porte NOT-AND, e potete ottenere quello che volete. Perché sto richiamando questa tabella? Perché un percettrone può essere pensato come una porta logica. In particolare, se scegliete opportunamente i pesi dei bias,

`00:10:48` 
potete ottenere una descrizione esatta della porta NOT-AND. In particolare, se i pesi sono entrambi uguali a meno due e il bias è uguale a tre, ottenete esattamente la porta NOT-AND. Quindi, si potrebbe dire, se il NOT-AND è una porta universale e il percettrone può essere usato per descrivere.

`00:11:23` 
una porta NOT-AND scegliendo opportunamente i pesi e i bias, in principio posso creare qualsiasi circuito logico. Quindi, iniziamo esplorando questo fatto, che è ovviamente vero. Ma il punto è che qui quello che ho assunto è che i pesi e il bias sono dati e sono scelti esattamente per avere la rappresentazione NOT-AND.

`00:12:03` 
In pratica quello che volete fare è imparare qualcosa. Quindi imparare qualcosa significa che non volete decidere a priori quali sono i pesi e i bias. Volete avere la libertà di scegliere i pesi e i bias per imparare un comportamento più generale che collega l'input all'output.

`00:12:33` 
E c'è un altro... problema importante con il percettrone. L'idea è che vogliamo imparare qualcosa, okay? Ma abbiamo visto che la funzione che descrive il percettrone è questa.

`00:13:07` 
Quindi è o 0 o 1, okay? Quindi è tra valori, valore 0 e valore 1. Quindi non c'è una transizione fluida tra i due stati. È un salto. Quindi significa che può succedere che se siete vicini a zero, una piccola variazione nell'input nel vettore x potrebbe creare un salto da zero a uno, okay?

`00:13:42` 
Quindi essenzialmente, non avete un comportamento continuo di questa funzione. E ricordate cosa abbiamo detto nella lezione precedente. L'idea è che per calcolare i pesi e i bias di una rete neurale che deve imparare qualcosa, quello che voglio fare è calcolare le sensibilità, quindi le derivate, della funzione di costo.

`00:14:19` 
rispetto alla funzione di costo. Rispetto a uno qualsiasi dei parametri della rete. Ed è chiaro che se la funzione di costo che sto usando o le funzioni coinvolte non sono continue, questa derivata può, o il valore che ho, sta saltando da 0 a 1 o viceversa,

`00:14:50` 
ed è molto difficile avere un comportamento fluido. Quindi, per esempio, se state usando un algoritmo di discesa del gradiente per calcolare i pesi con questo tipo di funzione, potrebbe essere, è un incubo. Quindi, l'idea è che voglio creare una situazione in cui voglio rilassare questo salto. e venire con una transizione più fluida tra i due stati.

`00:15:28` 
Perché? Perché in questo modo posso modificare gradualmente i pesi e i bias in base ai risultati della procedura di minimizzazione. Quindi, ecco che voglio passare dalla descrizione precedente con un salto a una più.

`00:16:07` 
diciamo funzione pratica. C'è un'altra ragione per cui il percettrone in generale è, non praticamente usato. Questo è dovuto al fatto che essenzialmente il comportamento che ho con il percettrone è molto molto limitato e quindi voglio avere la possibilità di descrivere.

`00:16:50` 
un comportamento più complesso della funzione coinvolta nell'apprendimento. La prima possibilità è usare la cosiddetta funzione sigmoid invece del salto. Cos'è una funzione sigmoid? Una funzione sigmoid è un'istanza di una funzione appartenente a una classe più ampia di funzioni chiamate funzioni sigmoidali.

`00:17:23` 
Una funzione sigmoidale, in generale, è una funzione sigma, tale che il limite per x che va a meno infinito di sigma di x è uguale a zero, e il limite per x che va a più infinito di sigma di x è uguale a uno. Questa è la definizione di una funzione sigmoidale. Quindi, una sigmoid, per esempio,

`00:17:54` 
questa funzione. è una funzione sigmoidale, okay, ovviamente poi avete molte funzioni che si comportano in questo modo, e richiameremo questa definizione nell'ultima parte del corso quando tratteremo il.

`00:18:28` 
teorema di approssimazione universale per le reti neurali. Quindi restiamo con la sigmoid. Quindi la sigmoid è un'istanza particolare di questo tipo di funzione e l'idea è che voglio avere una funzione che si comporta più o meno come il percettrone. Quindi ho un certo numero di input x1 fino a xn, pesi corrispondenti e il bias.

`00:18:59` 
Prima di tutto, sto costruendo la, di solito è chiamata la variabile z, che è wx, quindi la somma pesata degli input, più i valori, esattamente come prima. Quali sono le differenze rispetto al percettrone? Prima di tutto, gli input e l'output corrispondente non sono ristretti a essere binari. Quindi possono essere qualsiasi cosa.

`00:19:30` 
E in particolare, per esempio, se considerate un'immagine dove avete valori del pixel o in bianco e scala di grigi, per esempio, quello potrebbe essere un esempio di un possibile input.

`00:20:04` 
L'espressione della funzione sigmoid è questa, 1 su 1 più e alla potenza meno z, dove z è la variabile che abbiamo visto prima. Come potete vedere, questa funzione ha esattamente questo comportamento, e in particolare, potete vedere che per x uguale a 0, la funzione prende il valore 1 mezzi.

`00:20:42` 
Quindi questa è la prima istanza delle cosiddette funzioni di attivazione. La funzione sigmoid è... La prima funzione che è stata... introdotta per risolvere il problema relativo al multi-layer perceptron, okay? La proprietà che stavamo cercando, la regolarità, è ovviamente garantita dalla funzione sigmoid.

`00:21:24` 
E quindi ora siamo sicuri che se abbiamo un piccolo cambiamento nei pesi o nei bias, l'output cambierà in una quantità che può essere controllata, okay?

`00:21:56` 
E questo è fondamentale per l'apprendimento, perché quando impariamo, essenzialmente quello che stiamo facendo è che stiamo minimizzando la funzione per trovare l'insieme di pesi e bias che permette di trovare il significato di quella particolare funzione. E per eseguire questa procedura di minimizzazione, di solito stiamo adottando una discesa del gradiente o alcune variazioni della discesa del gradiente, che richiede il calcolo delle derivate.

`00:22:42` 
E quindi avendo questa proprietà. Siamo sicuri che l'applicazione della discesa del gradiente, di cui abbiamo bisogno, di algoritmi è possibile. Okay, ora cerchiamo di descrivere la struttura generica di una rete neurale. Come sapete, essenzialmente in un'architettura tipica, diciamo architettura classica di una rete neurale,

## Architettura delle Reti Neurali

`00:23:21` 
abbiamo tre tipi di layer. Il primo è il cosiddetto layer di input. È il primo layer della rete ed è solitamente composto da un certo numero di neuroni che è uguale al numero di caratteristiche di input. Poi abbiamo, sul lato opposto, il layer di output che produce l'output della rete.

`00:23:53` 
Può essere solo un neurone o può essere composto da molti neuroni. E nel mezzo, abbiamo i cosiddetti layer nascosti. Potete avere un layer nascosto o più di uno. Di solito, la rete neurale con solo un layer nascosto è chiamata rete superficiale. Se avete più di un layer nascosto, la rete è chiamata rete profonda.

`00:24:28` 
E richiameremo questa definizione di nuovo nell'ultima parte del corso, quando parleremo della complessità di una rete neurale e dei risultati teorici sulla complessità di una rete neurale, a seconda del suo tipo, o superficiale o profonda. Quindi, ovviamente, il numero di layer e il numero di neuroni in ciascun layer dipende fortemente dal problema che volete risolvere.

`00:25:07` 
Possono essere solo 10 o possono essere milioni. Quindi, graficamente, questa è l'idea, e in questa immagine, c'è un'altra caratteristica che è importante menzionare, stiamo, almeno per il momento, stiamo assumendo che la rete sia completamente connessa, il che significa che un neurone nel layer L è connesso a ogni neurone nel layer L più uno.

`00:25:53` 
Quindi, questo sarà il layer 1, layer attivo, di solito è chiamato L maiuscolo per il layer di output. Un'altra osservazione è che questa rete è chiamata feedforward. Perché? Perché secondo la direzione della freccia, l'informazione si sta propagando dall'input all'output. Quindi qui, per il momento, forse alla fine del corso faremo alcuni riferimenti ad architetture più complicate,

`00:26:33` 
non stiamo considerando, per esempio, reti neurali in cui avete loop chiusi, che sono le cosiddette reti neurali ricorsive, o altri tipi di architettura. Quindi qui stiamo concentrando la nostra attenzione sull'architettura più classica.

`00:27:07` 
Quindi, per riassumere, il percettrone non è adatto, quindi abbiamo rilassato introducendo la sigmoid, e poi abbiamo dato l'introduzione generale sui nomi che sono comunemente usati nella descrizione dell'architettura di una rete neurale. Ora, quello che vogliamo fare è descrivere formalmente, date queste caratteristiche, descrivere formalmente come possiamo calcolare le sensibilità di cui abbiamo bisogno per venire con i valori dei pesi e dei bias.

`00:28:01` 
che minimizzano una funzione di costo adeguata che chiameremo funzione di perdita. Okay, quindi per questa ragione, dobbiamo introdurre alcuni nomi. E, in particolare, chiameremo W^L_jk il peso che connette il neurone k nel layer L-1 al neurone j nel layer L, okay?

`00:28:38` 
Quindi, ricordate, qui a prima vista può essere abbastanza strano il fatto che qui k è in L-1 e j è nel layer L. Ma la ragione per questa notazione sarà chiara più tardi. Tuttavia, questa è la notazione.

`00:29:09` 
Quindi, ricordate che i pesi. qui, i pesi sono associati ai collegamenti, okay? Quindi qui ho il peso che connette il, neurone j nel layer L con il neurone k nel layer L meno uno, quindi nel layer precedente, okay? Quindi i pesi sono associati alle frecce, il collegamento tra i neuroni.

`00:29:46` 
Mentre i bias sono associati a ciascun neurone, okay? E per questa ragione useremo la notazione b^L_j, quindi sarà il bias del neurone j nel layer L, okay? Infine, associata a ciascun neurone, avremo anche la cosiddetta attivazione. L'attivazione è, la definiremo tra un momento, di nuovo, una variabile che è relativa a un particolare neurone in un particolare layer,

`00:30:26` 
e la sua definizione è data in questo modo. Quindi, è la somma di, quindi essenzialmente, questa è la somma pesata degli input più il bias, alla quale applichiamo la funzione di attivazione sigma, la sigmoid per il momento. Abbiamo considerato solo la sigmoid. Quindi, questa sarà chiamata attivazione, e sigma è la funzione di attivazione.

`00:31:05` 
Grazie. Per il calcolo, è meglio rappresentare tutte queste variabili in forma vettoriale. Quindi, introdurremo la matrice W^L, che è la matrice dei pesi per il layer L. Quindi, essenzialmente, ogni coppia di layer ha la sua propria matrice.

`00:31:39` 
b^L è il vettore dei bias per un particolare layer, e a^L è l'attivazione corrispondente per un particolare layer. Ricordate l'argomento della... funzione di attivazione è solitamente denotato con z, è un altro vettore, che non è altro che la somma, la somma pesata, più il bias, scritto in formato matriciale.

`00:32:19` 
Quindi è il prodotto, prodotto matrice vettore, tra la matrice dei pesi e l'attivazione del layer precedente, più il bias. Okay? E quindi, potete vedere che a^L è sigma, applicata a z^L, dove la funzione di attivazione deve essere pensata come applicata al vettore z^L elemento per elemento, okay?

`00:33:00` 
Quindi, ecco l'immagine, abbiamo layer, quindi qui in ogni neurone aggiungiamo l'attivazione corrispondente, e avrei potuto anche aggiungere qui per ogni neurone il bias corrispondente, e per ogni arco della rete aggiungiamo i pesi. E, per esempio, tra questi due layer definiremo la matrice W^2, okay?

## Funzione di Costo e Assunzioni

`00:33:48` 
Okay, ora abbiamo impostato il... framework, e ora quello che vogliamo calcolare sono le sensibilità, le sensibilità di ciascuna, in particolare, della funzione di output, quindi la funzione di costo, rispetto a tutti i parametri.

`00:34:18` 
Per questo, noi, prima di entrare nei dettagli, dobbiamo descrivere due caratteristiche che sono importanti per la funzione di costo, le assunzioni che possiamo fare. La prima è che la funzione di costo può essere scritta come...

`00:34:55` 
La somma, la media, se volete, di J di X. Cos'è J di X? Teniamo a mente un esempio. Supponiamo che vogliate addestrare una rete neurale per classificare alcune immagini. E queste immagini sono descritte come vengono appiattite, vengono ridimensionate, vengono appiattite in un vettore, e in ogni caratteristica è un numero tra 0 e 1,

`00:35:32` 
normalizzato per descrivere l'immagine. E abbiamo il training set è composto da migliaia di immagini. Quindi, in pratica, poi abbiamo le etichette corrispondenti. Cosa significa qui? Significa che sto sommando su tutti i campioni e J è ciò che è, per esempio, la differenza tra l'etichetta calcolata e quella vera.

`00:36:08` 
Quindi ho una somma in quel caso di migliaia di termini e ogni termine si riferisce a un campione. OK. E poi sto facendo la media sul numero totale di campioni. E OK, quindi questa è un'assunzione comune nel contesto della rete neurale o. la forma della funzione di costo. È l'assunzione usuale. O se state considerando un problema di classificazione,

`00:36:40` 
dove misurate la differenza tra l'etichetta prevista e quella vera, o se avete un problema di regressione, avete l'output atteso e il calcolato. Okay, quindi state calcolando la norma per ciascuno degli input. Quindi per ogni campione appartenente al training set. Quindi questa è la prima assunzione.

`00:37:11` 
La funzione di costo può essere scritta come la somma di n contributi. Ognuno di essi è relativo a un campione del training set. Poi la, seconda assunzione è che la funzione di costo può essere scritta come una funzione dell'attivazione di output.

`00:37:41` 
Quindi, qualcosa che è relativo solo all'ultimo layer. Ovviamente, quello che succede nell'ultimo layer è fortemente influenzato da quello che sta succedendo prima. Quindi, l'idea è che posso usare solo l'attivazione dell'ultimo layer per misurare, per valutare la funzione di costo. Esempio, supponiamo che...

`00:38:12` 
Avete Y sono i risultati veri, e a^L sono i risultati dell'ultimo layer, del risultato calcolato dato dall'ultimo layer. E qui, state calcolando questa distanza. La distanza tra i risultati veri e calcolati.

`00:38:47` 
Quindi queste sono le due assunzioni, che sono, direi, abbastanza naturali e non troppo restrittive. Sono ragionevoli nel, direi, 99% delle applicazioni che possiamo ora considerare. Poi avremo bisogno solo di una notazione, qui questa notazione è il cosiddetto prodotto di Hadamard,

`00:39:25` 
quindi se avete due vettori e calcolate il prodotto di Hadamard di quei due vettori, è solo il prodotto elemento per elemento. Se siete familiari con MATLAB, questa è l'operazione dot star, okay, è esattamente quell'operazione. E se ricordate in Python,

`00:39:56` 
ricordate che l'operatore star applicato ai vettori NumPy sta restituendo esattamente il prodotto di Hadamard, mentre se volete calcolare il prodotto scalare dovete usare np dot, la funzione dot fornita da NumPy. Quindi in pratica se avete questi due vettori, il prodotto di Hadamard è semplicemente.

`00:40:27` 
il prodotto delle componenti corrispondenti. Quindi ora deriviamo quello che è chiamato nella letteratura l'insieme delle quattro relazioni o equazioni fondamentali per calcolare la sensibilità. Il punto importante, il punto più importante che introduciamo è in alcuni riferimenti potete trovare questa quantità chiamata errore, altri riferimenti la chiamano sensibilità.

## Equazioni Fondamentali della Backpropagation

`00:41:19` 
In ogni caso, l'idea è che questa quantità è relativa a un dato neurone, neurone j, nel layer L, e dà esattamente la sensibilità della funzione di output rispetto alla variabile z relativa a quel particolare neurone. Ricordate che z è la somma pesata più il bias.

`00:41:49` 
Okay? Questa è la quantità, la definizione, e in pratica quello che vogliamo fare è venire con qualcosa che ci permette di calcolare la sensibilità della funzione di costo rispetto a tutti i pesi e tutti i bias della rete.

`00:42:21` 
Perché queste sono le quantità che sono usate in, per esempio, un algoritmo di discesa del gradiente. Quando calcolate, quando avete una procedura di discesa del gradiente, dovete calcolare il gradiente della funzione di costo rispetto alle incognite. E le incognite in questo caso sono esattamente i pesi e i bias. OK, quindi iniziamo dall'output in questo caso.

`00:42:55` 
OK. L'errore o la sensibilità, il vettore delta^L, L maiuscolo, quindi è relativo al layer di output, è dato dal gradiente della funzione di costo rispetto all'attivazione moltiplicato per sigma primo di z^L.

`00:43:26` 
Ora, ricordate, z^L è la somma pesata più il bias, e sigma è la funzione di attivazione. Sigma primo è la derivata di sigma rispetto al suo argomento, quindi rispetto a z^L. Okay, quindi nel primo... termine del prodotto, quindi il gradiente di J rispetto ad a sta misurando come.

`00:44:03` 
velocemente la funzione di costo cambia rispetto ad a, quindi l'attivazione. E sigma primo sta misurando quanto velocemente la funzione di attivazione varia rispetto alle variazioni del suo input z. In forma per componenti, abbiamo che delta^L_j L maiuscolo, quindi siamo nell'ultimo layer,

`00:44:34` 
quindi stiamo calcolando queste quantità per l'ultimo layer, è solo la derivata parziale di J rispetto ad a^L_j moltiplicato per sigma primo di z^L_j. Okay? Punto importante, in questa formula abbiamo un sigma primo, questo è molto molto importante dal punto di vista pratico. Perché?

`00:45:10` 
Ricordate che quando abbiamo disegnato sigma, la forma era qualcosa del genere. Quindi, da questa immagine, è chiaro che in questa regione e in questa regione, sigma primo, la derivata di sigma rispetto al suo input, è quasi zero. È praticamente zero, molto molto vicino a zero.

`00:45:43` 
Quindi, cosa significa? Significa che se sigma primo è zero, questo delta è zero anche, o molto, molto piccolo. Dal punto di vista pratico, significa che poiché questo è vicino a zero, molto vicino a zero, e questo è anche quasi zero, significa che un peso nel layer finale imparerà lentamente,

`00:46:21` 
il che significa che cambierà molto, molto lentamente se il neurone di output è o basso o alto nella funzione di attivazione. Questo è il cosiddetto neurone saturato, quindi un neurone per il quale l'attivazione è o in questa regione o in questa regione, è chiamato neurone saturato.

`00:46:52` 
Significa che per quel neurone, essenzialmente, non c'è variazione nei pesi e nei bias dovuto al fatto che sigma primo è essenzialmente zero. In altri termini, questo è il cosiddetto problema del gradiente che svanisce che forse avete sentito nel contesto del processo di apprendimento della rete neurale. Il problema del gradiente che svanisce è esattamente dovuto al fatto che nelle sensibilità c'è l'apparizione di sigma primo.

`00:47:31` 
Quindi significa che se sigma primo è vicino a zero, allora il, vettore errore o vettore sensibilità è composto da componenti che sono vicine a zero e questo vi il il uh essenzialmente il neurone non sta cambiando i valori dei pesi e dei bias, uh okay ora abbiamo uh calcolato questa quantità delta uh per l'ultimo layer ora vogliamo andare.

`00:48:12` 
indietro okay vogliamo muoverci dall'ultimo layer all'ultimo meno uno ultimo meno due e così via fino all'inizio uh in particolare quello che vogliamo fare è calcolare uh il, vettore delta per il layer, il layer generico L, dato il vettore delta per il layer L più uno,

`00:48:45` 
dato che ci stiamo muovendo ora da destra a sinistra. Il vettore delta per il layer L è ottenuto prendendo la matrice W^(L+1) trasposta, moltiplicata per delta^(L+1),

`00:49:23` 
quindi quello che ho già calcolato, e poi, devo moltiplicare questo vettore per sigma primo di z^L, okay? Tra un momento, dimostreremo tutte queste equazioni, ma ora voglio solo presentarvi i risultati e commentarli. Poi vedremo le dimostrazioni, che sono abbastanza facili.

`00:49:56` 
L'osservazione che ho dimenticato di mettere qui, aggiungerò, è relativa a quello che stavo menzionando prima, il fatto che gli indici in W erano spostati, okay? Come potete vedere qui, ora potete capire perché, perché in... Se usate la trasposta per la matrice, state esattamente, lasciatemi tornare a questa immagine.

`00:50:34` 
Quindi, state esattamente, quando trasponete, avete un jk, state connettendo quello che è qui a quello che è nel layer precedente. Okay, quindi, dato che in pratica la procedura è una procedura all'indietro, quindi ci stiamo muovendo da qui a qui, questa è la ragione per cui di solito questo spostamento negli indici è adottato.

`01:01:04` 
Avreste potuto anche usare dall'inizio kj e poi non usare la trasposta, ma è comune adottare. Okay, quindi, quindi, per il momento, abbiamo calcolato, quindi usando, sfruttando quella formula, con la precedente, con quella dell'ultimo layer, siamo stati in grado di calcolare delta per l'ultimo layer.

`01:01:38` 
E poi, usando ricorsivamente questa formula, possiamo calcolare i vettori delta per tutti i layer nella rete. Okay? Ovviamente, non abbiamo finito, perché se ricordate, il nostro obiettivo è calcolare questo. Okay? Quindi, dJ su dW, e dJ su db, per tutti i parametri della rete.

`01:02:09` 
Quindi ora, in qualche modo, dobbiamo essere in grado, se possibile, di portare le quantità che abbiamo appena calcolato, i delta, a questa quantità a cui siamo interessati. Quindi prima è, quindi in pratica ora vogliamo calcolare i gradienti, che sono in realtà gli ingredienti che entrano nella procedura di minimizzazione.

`01:02:40` 
Prima di tutto, consideriamo il gradiente per i bias. È possibile, e vedremo tra un momento, dimostrare che la derivata di J rispetto al bias generico della rete è esattamente data dall'errore o sensibilità del, Okay, quindi una volta che avete calcolato delta^L_j, quello è esattamente la derivata parziale della funzione di costo rispetto al bias di quel particolare neurone.

`01:03:26` 
Okay, e i pesi? Per i pesi, abbiamo che. Se vogliamo considerare la derivata parziale del peso di J rispetto al peso che connette il neurone j nel layer L e il neurone k nel layer L-1, la relazione è.

`01:04:00` 
Questa. Quindi abbiamo l'attivazione nel neurone k nel layer L meno uno moltiplicata per l'errore o sensibilità del neurone j nel layer L. OK, è un prodotto di queste due quantità. OK, quindi ora abbiamo queste quattro equazioni che, come potete vedere, sfruttando il calcolo dei delta sono in grado di darci le derivate di cui abbiamo bisogno per la procedura di minimizzazione.

`01:04:54` 
Quindi ora andiamo una per una e cerchiamo di vedere come possiamo dimostrare queste relazioni è molto semplice. Quindi iniziamo con la prima. Quindi dobbiamo dimostrare che delta^L_j nel layer di output, è dato da delta la derivata parziale di J rispetto all'attivazione moltiplicato per sigma primo.

`01:05:25` 
Quindi iniziamo con la definizione che abbiamo dato precedentemente della sensibilità. La sensibilità è, definita come la derivata parziale della funzione di costo rispetto a z di un particolare neurone. Poi applichiamo, ricordate che un particolare neurone, se tornate all'immagine, okay, qui.

`01:05:58` 
Quindi supponiamo di considerare questo neurone è connesso a tutti i neuroni del layer precedente, okay? Quindi qui possiamo, questo è vero anche per l'ultimo layer, possiamo scrivere questa derivata come la somma su k, dove k sono i neuroni del layer L,

`01:06:37` 
di dJ su da moltiplicato per da su dz. Sto solo usando la regola della catena. che è esattamente quello che abbiamo fatto anche l'ultima volta quando stavamo trattando la, differenziazione automatica. Questa è esattamente la stessa storia. Stiamo sfruttando la regola della catena esattamente come abbiamo fatto l'ultima volta. Ora sappiamo che l'attivazione è data dall'applicazione.

`01:07:11` 
della funzione di attivazione a z wx più b. Ora da su dz è diverso da zero solo quando k è uguale a j. Ricordate che z è, come vi ho detto, solo wx più b.

`01:07:44` 
Quindi sto prendendo la derivata rispetto a z^L_j, e questo è chiaramente uguale a zero se k è diverso da j, ed è uguale a sigma primo valutato in z^L_j se k è uguale a j. Quindi se questi due indici sono diversi, allora la derivata è zero.

`01:08:17` 
Quindi in pratica, in questa sommatoria, rimaniamo con solo un termine, il termine relativo a k uguale a j. E in particolare, se voi... In questa sommatoria, prendete solo il caso k uguale a j, e ricordate che a è sigma, questo non è altro che sigma primo, il da su dz, quindi finiamo con la relazione che abbiamo visto prima.

`01:09:00` 
Quindi, l'errore o la sensibilità nel neurone j nell'ultimo layer è dato dalla derivata parziale della funzione di costo rispetto all'attivazione del neurone corrispondente nell'ultimo layer moltiplicato per sigma primo. O, se volete usare il formato matriciale, potete usare il prodotto di Hadamard, come abbiamo visto prima.

`01:09:34` 
Poi, um, il secondo, ricordate, il secondo è la propagazione, la backpropagation. Di nuovo, iniziamo con la definizione di, ora, nel caso precedente, stavamo considerando la sensibilità nell'ultimo layer. Qui, vogliamo considerare la sensibilità nel neurone generico j del layer generico L, che è, per definizione, dato dalla derivata parziale di J rispetto a z^L_j.

`01:10:13` 
Di nuovo, possiamo esplorare la regola della catena esattamente come prima, e qui abbiamo queste espressioni. Il punto interessante è che qui abbiamo, abbiamo la derivata parziale di J rispetto a z nel layer L più 1, che è quello che è delta nel layer L più 1.

`01:10:50` 
Quindi è il layer successivo, che è stato già calcolato, okay, perché ci stiamo muovendo all'indietro. Poi abbiamo questa quantità. Quindi queste sono le due z per il layer L e layer L più 1 valutate, possibilmente, in due neuroni diversi, k e j. Quindi esprimiamo la quantità che vogliamo derivare, che è data dalla somma pesata più il bias.

`01:11:30` 
Ricordate, la somma pesata è data dal peso per a^L_j più b^(L+1)_k. E ora possiamo sostituire l'attivazione con la sua espressione come l'applicazione di sigma a z del layer L. E ora possiamo calcolare questa derivata, okay?

`01:12:05` 
È chiaro che una volta quando deriviamo z^(L+1)_k rispetto a z^L_j, beh, il bias non contribuisce affatto. Quello che abbiamo qui è solo il contributo. relativo alla regola della catena della variabile.

`01:12:36` 
E di nuovo, in pratica, rimanete con solo questo termine, per questo particolare neurone in questo particolare layer. Ovviamente, questo è un numero, quindi la derivata sarà applicata a questa quantità. Quindi qui, avremo sigma primo, esattamente come abbiamo nella prima relazione.

`01:13:09` 
Poi, se ricordiamo l'espressione che avevamo qui, possiamo sostituire quello che abbiamo appena calcolato qui, e possiamo venire con... questa espressione, okay? Quindi ho solo sostituito qui il valore che abbiamo calcolato ora. Sigma primo.

`01:13:41` 
non dipende da k, quindi può essere portato fuori dalla sommatoria, e qui abbiamo il prodotto matrice vettore che abbiamo identificato prima, W moltiplicato per delta e poi abbiamo l'altro prodotto con sigma primo di z, okay? Quindi qui, questa è la seconda relazione che.

`01:14:16` 
permette di propagare le sensibilità all'indietro, partendo dall'ultimo layer fino al primo. Poi, dobbiamo dimostrare la prima relazione che permette di calcolare il gradiente rispetto ai bias. Questo è molto semplice. Quindi, se volete calcolare il gradiente di J rispetto al bias generico b^L_j,

`01:14:50` 
potete sfruttare la regola della catena. Potete scrivere la derivata come la derivata rispetto a z moltiplicata per la derivata di z rispetto a b. Ma, per definizione, questa è la sensibilità o l'errore. E quindi, dobbiamo capire cos'è questo termine. Ricordate che z^L_j è la somma pesata più il bias.

`01:15:23` 
E dobbiamo calcolare la derivata rispetto a b^L_j, che è 1, o semplicemente 1, quindi abbiamo esattamente la relazione che ci dice che il gradiente rispetto al bias è dato esattamente dalla sensibilità di quel particolare neurone.

`01:15:54` 
Infine, dobbiamo calcolare il gradiente dei pesi. Di nuovo, potete sfruttare la regola della catena, esattamente come prima. Possiamo osservare che questo è... la sensibilità. Quindi ora dobbiamo trattare questo termine, che è dove questo z^L_j è come al solito, la somma pesata più il bias, e dobbiamo calcolare la derivata rispetto a W^L_jk. Quindi.

`01:16:31` 
sarà il termine nella sommatoria, nella sommatoria per i uguale a k. Quindi questa derivata è esattamente uguale a a^(L-1)_k, L meno uno, scusate, e poi se sostituite quello nel primo, avete la sensibilità moltiplicata per l'attivazione del layer precedente, e poi avete la sensibilità moltiplicata per.

`01:17:04` 
l'attività del campione, e poi avete la sensibilità moltiplicata per l'attività del campione, che è esattamente quello che abbiamo ottenuto. abbiamo visto prima. Quindi, in pratica, l'idea è, per, la ben nota backpropagation, voi.

`01:17:34` 
avete un vettore di input come un'immagine con tutte le caratteristiche. Eseguite, una passata in avanti della rete. Perché avete bisogno della passata in avanti? Perché avete bisogno dei valori di tutte le attivazioni e di tutte le somme pesate. Quindi questo è fatto nel passo in avanti, che è esattamente quello che abbiamo fatto l'ultima volta quando stavamo calcolando la derivata di una funzione.

`01:18:06` 
Dovevamo calcolare tutti i passi in avanti per calcolare tutti i valori. Poi, una volta che abbiamo tutti i valori delle attivazioni e della somma pesata dati nel passo in avanti, possiamo iniziare la procedura all'indietro. Scusate, ho solo dimenticato cosa c'era qui. Ovviamente, qui manca un punto.

`01:18:43` 
Il fatto che, in questa procedura, dovete inizializzare i pesi e i bias in qualche modo, okay? Potete impostarli tutti a zero, o ci sono altri metodi per trovare una buona inizializzazione dei pesi e dei bias. Il valore iniziale per i bias non è molto critico. I valori iniziali per i pesi sono molto più importanti, e da questo punto di vista, se usate librerie pratiche come...

`01:19:29` 
o TensorFlow o PyTorch, ci sono metodi implementati per inizializzare i pesi. Di solito i bias sono inizializzati a zero perché sono meno importanti per far funzionare l'algoritmo di discesa del gradiente e progredire. Quindi, abbiamo inizializzato i pesi e i bias, abbiamo fatto un passo in avanti, ora eseguiamo il passo all'indietro, partendo dall'ultimo layer.

`01:20:21` 
il primo sfruttando esattamente le quattro relazioni che abbiamo visto prima e l'output sarà, i gradienti della funzione di costo con le derivate parziali della funzione di costo, rispetto a tutti i pesi e tutti i bias della rete. Qui ho solo.

`01:20:53` 
richiamato quello che in realtà abbiamo già visto l'ultima volta quando abbiamo fatto alcune osservazioni sulla possibilità di calcolare derivate usando la differenziazione numerica. Qui ho solo riportato, non entrerò nei dettagli, è esattamente a parte le considerazioni. Relative all'aritmetica in virgola mobile, qui solo un calcolo del numero di operazioni che richiedereste per usare una differenziazione numerica rispetto ai passi richiesti usando l'algoritmo di back-propagation.

`01:21:40` 
Quindi qui è solo questa osservazione. Okay, quindi ora...

## Funzioni di Attivazione

`01:22:20` 
Okay, nelle slide precedenti, abbiamo visto, come primo esempio, la funzione sigmoid. Le relazioni che abbiamo ottenuto, le quattro relazioni fondamentali riguardanti la backpropagation, coinvolgono sigma, ma in quella relazione, sigma non è necessariamente.

`01:22:57` 
la funzione sigmoid. Può essere qualsiasi funzione di attivazione. Okay, quindi ora voglio introdurre alcune funzioni di attivazione che sono comunemente usate in, quella letteratura. con i loro vantaggi e svantaggi. Questo è quello che abbiamo visto per il percettrone.

`01:23:33` 
È essenzialmente la funzione a gradino, 0, 1. Quali sono i problemi? È chiaro che questa funzione usata nell'algoritmo di discesa del gradiente è un incubo. Perché? Perché c'è una delta di Dirac vicino a 0, ma poi la derivata della funzione è 0 ovunque.

`01:24:05` 
Quindi, secondo quello che abbiamo visto prima, siamo sempre nella situazione del gradiente che svanisce. Poi abbiamo la funzione lineare, la funzione lineare è stata essenzialmente, forse esattamente dopo il percettrone, il primo tentativo per, diciamo, superare il problema con il percettrone è stata la funzione lineare.

`01:24:44` 
La funzione lineare è ovviamente semplice, la derivata è costante, quindi non abbiamo il problema del gradiente che svanisce, ma ha un problema importante. Dato che è lineare, e la composizione di funzioni lineari è lineare anche, se usate la funzione lineare come funzione di attivazione, non importa quanti layer state usando, non importa quanti neuroni state usando, la composizione di questa funzione sarà sempre lineare.

`01:25:29` 
Quindi, se volete rappresentare un comportamento altamente non lineare di un certo fenomeno, quindi la relazione tra l'input e l'output è altamente non lineare, allora usando questa funzione, non avete speranza di successo. Okay? Sigmoid è quella che abbiamo appena considerato.

`01:26:04` 
La... Abbiamo il problema del gradiente che svanisce. Poi non è centrata sullo zero. Cosa significa? Significa che lo zero è qui. Qui siamo a 0.5. Quindi può succedere che sia in qualche modo sbilanciata verso un certo segno dei pesi.

`01:26:36` 
E quindi in alcune situazioni, potrebbe non essere sempre, la scelta migliore. D'altra parte, dato che qui abbiamo 1, l'output può anche essere interpretato come una probabilità. Quindi per esempio, se usato in un layer di output con, per esempio, solo un singolo neurone, può darvi la probabilità che l'output sia un gatto o un cane, per esempio.

`01:27:12` 
Okay, un'altra funzione che è simile a quella che abbiamo visto ora, è la tangente iperbolica. Questa non è esattamente una funzione sigmoidale, perché per x che va a meno infinito, la funzione va a meno uno, ma è centrata sullo zero.

`01:27:48` 
Quindi, significa che ora, in una procedura di minimizzazione, non siamo sbilanciati verso un segno particolare per i pesi. Questo può essere qualcosa che può essere preso in considerazione in alcune situazioni. E dato che il comportamento è molto simile alla sigmoid, abbiamo ancora il problema del gradiente che svanisce per valori dell'input, che è molto alto o molto basso.

`01:28:27` 
buona per, è molto usata nei layer nascosti, ed è usata molto spesso nelle reti neurali ricorrenti, che è un'architettura particolare che include alcuni loop, e di solito per i layer nascosti, la tangente iperbolica è preferita alla sigmoid. Poi abbiamo una funzione molto ben nota, la ReLU, l'unità lineare rettificata, che è.

`01:29:05` 
data da questa forma. L'idea è che, volete avere in qualche modo qualcosa di molto semplice come potete vedere è una funzione che è composta da due regioni in ogni regione la funzione è lineare ma il comportamento complessivo della funzione è.

`01:29:38` 
non lineare quindi rispetto alla funzione x la composizione di ReLU vi permette di avere un comportamento non lineare della rete quindi è molto più flessibile e d'altra parte. è il problema del gradiente che svanisce è ancora lì in particolare.

`01:30:10` 
quando l'input è negativo perché in quella parte del dominio per l'input, la funzione è identicamente zero, e la derivata è zero anche. È la funzione di attivazione più comune per nelle reti neurali convoluzionali e reti neurali profonde in generale.

`01:30:47` 
D'altra parte, dal punto di vista computazionale, è molto economica perché calcolare la derivata di questa funzione è immediato rispetto, per esempio, alla tangente iperbolica. Qui abbiamo la cosiddetta leaky ReLU. Cos'è la Leaky ReLU? La Leaky ReLU è stata introdotta per rilassare un po' il problema della ReLU morente, quindi essenzialmente nella porzione negativa, invece di avere esattamente zero, abbiamo una pendenza molto molto piccola, per esempio il valore comune è 0.01.

`01:31:39` 
Qual è l'idea? L'idea è che il fatto che abbiamo anche piccola, abbiamo una pendenza anche nella regione negativa, permette di avere la possibilità di imparare anche, anche se molto lentamente, di imparare anche se cadiamo nelle regioni negative.

`01:32:17` 
Questa affermazione, lo so non è molto precisa, ma è un fatto nel senso che non c'è una ricetta esatta che può dirvi in questa situazione dovete usare la ReLU, in queste altre situazioni la leaky ReLU è molto meglio. È una procedura per tentativi ed errori che vi permette di scegliere qual è la migliore.

`01:32:51` 
Poi, in pratica, se guardate questa espressione qui, possiamo anche rilassarla ancora di più. Invece di scegliere a priori questa pendenza, possiamo lasciare questa pendenza. ReLU, libera. Quindi, abbiamo la cosiddetta ReLU parametrica o parametrizzata, in cui questo alpha è libero.

`01:33:25` 
In questo caso, alpha entra nella procedura di minimizzazione. Quindi, è un parametro aggiuntivo che può essere calcolato durante la procedura di minimizzazione. Okay? Quindi, in altre parole, state aggiungendo più parametri alla vostra rete per farla performare meglio, si spera.

`01:34:02` 
Ovviamente, un possibile svantaggio è che aumentando il numero di parametri, state aumentando la complessità complessiva del modello e dato che state aggiungendo nuovi parametri, potete aggiungere overfitting. Quindi, specialmente se il dataset non è molto grande, potete avere overfitting.

`01:34:38` 
Poi avete la classe di funzioni di attivazione esponenziali. La prima è l'unità lineare esponenziale, che è esattamente come la ReLU nella parte positiva. Ed è un esponenziale nella parte negativa. Qui avete un collegamento anche con la pendenza.

`01:35:11` 
E di nuovo qui avete un parametro che potete scegliere. Quindi alpha è anche in questo caso un altro parametro che potete o regolare prima o durante la procedura di ottimizzazione. Qui stiamo aggiungendo un altro parametro, lambda, che è...

`01:35:53` 
permette di avere la cosiddetta unità lineare esponenziale scalata. L'idea è che i valori lambda e alpha in questa scelta sono scelti a priori, e questa funzione funziona in particolare se,

`01:36:23` 
come stavo menzionando prima, state usando un'inizializzazione particolare per la distribuzione dei pesi nella rete. Devo dire che, in pratica, non è molto usata. Poi c'è una funzione di attivazione importante, la cosiddetta softmax. Softmax è una funzione di attivazione che è usata solo nel layer di output.

`01:36:57` 
Potete vedere la definizione. Essenzialmente, l'idea è che dato un vettore che è, per esempio, il vettore dell'attivazione dell'ultimo layer, crea un vettore di probabilità. OK, come potete vedere, stiamo prendendo dato il vettore di x con le componenti x_i.

`01:37:28` 
Al denominatore, avete la somma dell'esponenziale di tutte le componenti. E al numeratore, avete una componente. Quindi, per esempio, in questo caso, avete, 2, 1, 0.1 che sono convertiti a 0.7, 0.2, 0.1 che possono essere interpretati come probabilità. Quindi, per esempio, se avete un problema di classificazione con 10.

`01:38:01` 
classi, allora potete usare nel layer di output l'attivazione softmax per convertire l'ultimo valore che avete in probabilità. Poi avete quest'altra funzione chiamata swish.

`01:38:33` 
È il prodotto della lineare e della sigmoid. Quali sono i vantaggi? In generale, è stato dimostrato praticamente che il comportamento di questa funzione di attivazione è molto migliore della ReLU.

`01:39:05` 
In particolare, se avete una rete molto profonda. Questa funzione di attivazione è stata sviluppata da Google, ed è la funzione che è usata nell'architettura chiamata EfficientNet, che è un'architettura particolare chiamata EfficientNet.

`01:39:38` 
Che è stata proposta da Google. Lo svantaggio di questa funzione è che dato che coinvolge la sigmoid, può essere computazionalmente costosa. D'altra parte, ha alcune buone proprietà tra cui evita il problema della ReLU morente.

`01:40:14` 
Infine, un'altra variazione su questa è questa strana funzione chiamata la funzione mish. E di nuovo, non sto dicendo che c'è qualche prova teorica che una è meglio dell'altra. È solo basata sull'esperienza e sulle applicazioni pratiche.

`01:40:52` 
Per esempio, nel rilevamento di oggetti, quindi algoritmi per il rilevamento di oggetti in tempo reale, questo tipo di funzione è comunemente usato, ed è stato dimostrato superiore a qualsiasi altra funzione sul mercato. Perché? È una domanda aperta. Devo dire che se guardate nelle librerie, TensorFlow,

`01:41:31` 
PyTorch o qualsiasi altra libreria che previene algoritmi di machine learning o reti neurali, potete trovare direi tutte queste funzioni e forse anche di più. Qui ho solo riportato le più comuni. Quindi per riassumere, come scegliere la funzione di attivazione? La scelta comune è iniziare con.

`01:42:06` 
ReLU. Poi se avete problemi con la ReLU, potete provare a usare una qualsiasi delle variazioni della ReLU. La Leaky ReLU, ReLU Parametrica o Unità Lineari Esponenziali. Per applicazioni specifiche, diciamo, se rete neurale ricorrente, o ricorsiva, o come abbiamo visto qui, rilevamento di oggetti, potete usare funzioni di attivazione specifiche.

`01:42:48` 
Suggerimento da tenere a mente, in particolare, per la maggior parte delle applicazioni, i primi due punti, iniziate con la ReLU, forse nei layer intermedi, nei layer nascosti, potete anche provare con la tangente iperbolica, e poi passate da questa scelta standard ad altre scelte, scelte più complesse. Solo se veramente necessario, perché questo aumenta la complessità del modello.

`01:43:27` 
Grazie.
---

## Appendice: Formule Chiave di Neural Networks e Backpropagation

### Storia: Dal Percettrone alle Reti Profonde

**Percettrone (Rosenblatt, 1958)**:
- **Input**: x, x, ..., x  {0,1} (binari)
- **Output**: y  {0,1}
- **Funzione**: 
```
y = { 0  se wx + b  0
    { 1  se wx + b > 0
```

**Limitazioni del Percettrone**:
1. **Non differenziabile**: Salto discontinuo  no gradient descent
2. **XOR problem**: Non pu� rappresentare funzioni non linearmente separabili
3. **Step function**: f'(x) = 0 quasi ovunque (gradient vanishing estremo!)

**NAND Gate (Porta Universale)**:
- Con pesi w = w = -2, bias b = 3
- NAND(0,0) = 1, NAND(0,1) = 1, NAND(1,0) = 1, NAND(1,1) = 0
- Qualsiasi circuito logico costruibile combinando NAND gates!

### Notazione Matriciale delle Reti Neurali

**Layer L** (L = 1, 2, ..., L_max):
- **Pesi**: W^L  R^(n_L  n_(L-1)) dove W^L_jk connette neurone k (layer L-1)  neurone j (layer L)
- **Bias**: b^L  R^(n_L) dove b^L_j � bias del neurone j nel layer L
- **Weighted sum**: z^L = W^L  a^(L-1) + b^L  R^(n_L)
- **Activation**: a^L = s(z^L)  R^(n_L) (applicata element-wise)

**Forward Pass** (Input  Output):
```
Input:  a^0 = x  R^(n_0)
Layer 1:  z^1 = W^1a^0 + b^1,  a^1 = s(z^1)
Layer 2:  z^2 = W^2a^1 + b^2,  a^2 = s(z^2)
...
Output: z^L = W^La^(L-1) + b^L,  a^L = s(z^L)
```

**Architetture**:
- **Fully connected (feedforward)**: Ogni neurone connesso a TUTTI i neuroni del layer successivo
- **Shallow network**: 1 hidden layer
- **Deep network**: 2 hidden layers (Deep Learning!)

### Funzioni di Attivazione

| Funzione | Formula | s'(z) | Range | Pro | Contro |
|----------|---------|-------|-------|-----|--------|
| **Step** | 1 se z>0 else 0 | d(z) | {0,1} | Percettrone originale | Non differenziabile |
| **Linear** | s(z) = z | 1 | (-,) | Semplice | Composizione  lineare! |
| **Sigmoid** | 1/(1+e^(-z)) | s(z)(1-s(z)) | (0,1) | Smooth, output=prob | Gradient vanishing |
| **Tanh** | (e^z - e^(-z))/(e^z + e^(-z)) | 1 - s(z) | (-1,1) | Zero-centered | Gradient vanishing |
| **ReLU** | max(0,z) | 1 se z>0 else 0 | [0,) | Efficiente, no vanishing | "Dying ReLU" (z<0) |
| **Leaky ReLU** | max(az, z) | a se z<0 else 1 | (-,) | No dying ReLU | a fisso (es. 0.01) |
| **PReLU** | max(az, z) | a se z<0 else 1 | (-,) | a learnable | Pi� parametri |
| **ELU** | z se z>0 else a(e^z-1) | 1 se z>0 else s(z)+a | (-a,) | Smooth negatives | Costoso (exp) |
| **Swish** | zs(z) | s + zs'(z) | (-,) | Meglio di ReLU (deep) | Costoso (sigmoid) |
| **Softmax** | e^(z_i)/S_j e^(z_j) | (formula complessa) | [0,1], S=1 | Output = prob | Solo output layer |

**Propriet� Sigmoid**:
```
s(z) = 1/(1+e^(-z))
s'(z) = s(z)(1-s(z))   FONDAMENTALE per backprop!
lim_{z-} s(z) = 0
lim_{z+} s(z) = 1
s(0) = 0.5
```

**Quando s'(z)  0?**
- z  0 (neurone "off", output  0)
- z  0 (neurone "on", output  1)
- **Neurone saturato**  pesi/bias non si aggiornano  **gradient vanishing**!

**Scelta Pratica**:
1. **Hidden layers**: Inizia con **ReLU** (default moderno)
2. **Problemi ReLU**: Prova Leaky ReLU, PReLU, ELU
3. **Recurrent NN**: Tanh (zero-centered)
4. **Output layer**: 
   - Regressione  Linear
   - Classificazione binaria  Sigmoid
   - Classificazione multi-classe  Softmax

### Le 4 Equazioni Fondamentali della Backpropagation

**Hadamard Product** ():
```
(s  t)_j = s_j  t_j  (element-wise multiplication)
Esempio: [1,2,3]  [4,5,6] = [4,10,18]
```

**Errore/Sensibilit�** d^L_j:
```
d^L_j  C/z^L_j  (quanto C sensibile a z^L_j)
```

**Equazione 1** (Output layer error):
```
d^L = _a C  s'(z^L)

Componente j:  d^L_j = (C/a^L_j)  s'(z^L_j)
```

**Interpretazione**:
- _a C: quanto veloce cambia C rispetto ad attivazione output
- s'(z^L): quanto veloce cambia s rispetto a weighted sum
- **Se s'  0 (neurone saturato)  d^L  0  pesi non si aggiornano!**

**Equazione 2** (Backpropagate error):
```
d^l = ((W^(l+1))^T  d^(l+1))  s'(z^l)

Per ogni layer l = L-1, L-2, ..., 1 (backward!)
```

**Interpretazione**:
- Propaga errore dal layer l+1 al layer l
- (W^(l+1))^T: "reverse connections"
-  s'(z^l): modula per derivata attivazione
- **Ricorsivo**: d^L  d^(L-1)  d^(L-2)  ...  d^1

**Equazione 3** (Gradient rispetto bias):
```
C/b^l_j = d^l_j

Forma vettoriale:  C/b^l = d^l
```

**Interpretazione**:
- Gradiente bias = sensibilit� neurone!
- Diretto: una volta calcolato d^l, hai C/b^l gratis

**Equazione 4** (Gradient rispetto pesi):
```
C/W^l_jk = a^(l-1)_k  d^l_j

Forma matriciale:  C/W^l = d^l  (a^(l-1))^T
```

**Interpretazione**:
- Gradiente peso = attivazione input  errore output
- Se a^(l-1)_k piccolo  peso W^l_jk impara lentamente
- Se d^l_j piccolo (neurone saturato)  peso impara lentamente

### Dimostrazione delle 4 Equazioni

**Eq. 1** (d^L = _a C  s'(z^L)):

```
d^L_j = C/z^L_j  [definizione]
      = S_k (C/a^L_k)  (a^L_k/z^L_j)  [chain rule]
      = (C/a^L_j)  (a^L_j/z^L_j)  [k=j, altri termini = 0]
      = (C/a^L_j)  s'(z^L_j)  [a^L_j = s(z^L_j)]
```

**Eq. 2** (d^l = ((W^(l+1))^T  d^(l+1))  s'(z^l)):

```
d^l_j = C/z^l_j  [definizione]
      = S_k (C/z^(l+1)_k)  (z^(l+1)_k/z^l_j)  [chain rule]
      = S_k d^(l+1)_k  (z^(l+1)_k/z^l_j)  [def. d^(l+1)]

Ora: z^(l+1)_k = S_i W^(l+1)_ki  a^l_i + b^(l+1)_k
               = S_i W^(l+1)_ki  s(z^l_i) + b^(l+1)_k

z^(l+1)_k/z^l_j = W^(l+1)_kj  s'(z^l_j)

Quindi:
d^l_j = S_k d^(l+1)_k  W^(l+1)_kj  s'(z^l_j)
      = (S_k W^(l+1)_kj  d^(l+1)_k)  s'(z^l_j)
      = ((W^(l+1))^T  d^(l+1))_j  s'(z^l_j)  [prodotto matrice-vettore!]
```

**Eq. 3** (C/b^l_j = d^l_j):

```
C/b^l_j = (C/z^l_j)  (z^l_j/b^l_j)  [chain rule]
          = d^l_j  1  [z^l_j = W^l_ja^(l-1) + b^l_j]
          = d^l_j
```

**Eq. 4** (C/W^l_jk = a^(l-1)_k  d^l_j):

```
C/W^l_jk = (C/z^l_j)  (z^l_j/W^l_jk)  [chain rule]
           = d^l_j  (z^l_j/W^l_jk)

Ora: z^l_j = S_i W^l_ji  a^(l-1)_i + b^l_j

z^l_j/W^l_jk = a^(l-1)_k  [derivata rispetto W^l_jk]

Quindi:
C/W^l_jk = d^l_j  a^(l-1)_k
```

### Algoritmo Backpropagation (Step-by-Step)

**Input**: Training set {(x^(1), y^(1)), ..., (x^(m), y^(m))}

**Step 0**: **Inizializzazione**
```
Pesi W^l: Random Gaussian con media 0, std 1/(n_(l-1)) (Xavier init)
          oppure std (2/n_(l-1)) (He init per ReLU)
Bias b^l: Tutti a 0
```

**Step 1**: **Forward Pass** (per ogni campione x)
```
a^0 = x  (input)
For l = 1, 2, ..., L:
    z^l = W^l  a^(l-1) + b^l
    a^l = s(z^l)
```
**Output**: Tutti i valori z^l, a^l salvati (serve per backward!)

**Step 2**: **Compute Output Error** (Eq. 1)
```
d^L = _a C  s'(z^L)

Esempio (MSE):  C = (1/2)||a^L - y||
                _a C = a^L - y
                d^L = (a^L - y)  s'(z^L)
```

**Step 3**: **Backpropagate Error** (Eq. 2)
```
For l = L-1, L-2, ..., 1:  (backward!)
    d^l = ((W^(l+1))^T  d^(l+1))  s'(z^l)
```

**Step 4**: **Compute Gradients** (Eq. 3-4)
```
For l = 1, 2, ..., L:
    C/b^l = d^l
    C/W^l = d^l  (a^(l-1))^T  (outer product!)
```

**Step 5**: **Update Parameters** (Gradient Descent)
```
For l = 1, 2, ..., L:
    W^l  W^l - ?  (C/W^l)
    b^l  b^l - ?  (C/b^l)
```

**Step 6**: Ripeti Steps 1-5 per tutti i campioni (1 epoch), poi ripeti per N epochs

### Funzioni di Costo Comuni

**Mean Squared Error (MSE)** - Regressione:
```
C = (1/2n) S_x ||a^L(x) - y(x)||
_a C = a^L - y
```

**Cross-Entropy** - Classificazione binaria:
```
C = -(1/n) S_x [yln(a^L) + (1-y)ln(1-a^L)]
_a C = (a^L - y) / (a^L(1-a^L))
```

**Categorical Cross-Entropy** - Multi-classe (con Softmax):
```
C = -(1/n) S_x S_j y_jln(a^L_j)
dove: a^L_j = e^(z^L_j) / S_k e^(z^L_k)  (softmax)
_a C = a^L - y   SEMPLICE con softmax!
```

**Assunzioni per Backprop**:
1. **Additivit�**: C = (1/n) S_x C_x (media su campioni)
2. **Funzione di a^L**: C = C(a^L) dipende solo da output layer

### Gradient Vanishing Problem

**Causa**: s'(z)  0 quando neurone saturato

**Esempio Sigmoid**:
```
s'(z) = s(z)(1-s(z))
s'(0) = 0.25  (massimo!)
s'(5)  0.007  (quasi zero)
s'(-5)  0.007  (quasi zero)
```

**Impatto su Backprop**:
```
d^l = ((W^(l+1))^T  d^(l+1))  s'(z^l)
                                  
    Propagato dal layer    MOLTIPLICATO per s'!
       successivo
```

Se s'(z^l)  0  d^l  0  gradienti  0  pesi non si aggiornano!

**Propagazione nei Layer Profondi**:
```
d^1  s'(z^1)  s'(z^2)  ...  s'(z^L)
                                 
     0.1      0.1            0.1

Se 10 layers: d^1  (0.1)^10 = 10^(-10)  VANISHING!
```

**Soluzioni**:
1. **ReLU activation**: s'(z) = 1 per z>0 (no vanishing nella regione attiva)
2. **Residual connections** (ResNet): skip connections evitano moltiplicazioni ripetute
3. **Batch Normalization**: normalizza z^l  mantiene s'(z^l) lontano da 0
4. **Gradient clipping**: limita ||||  threshold
5. **LSTM/GRU**: per reti ricorrenti (gating mechanisms)

### Weight Initialization (Critica!)

**Problema**: Se W inizializzati male  training fallisce!

**Xavier/Glorot Initialization** (Tanh, Sigmoid):
```
W^l_jk ~ N(0, s) con s = 1/(n_(l-1))
oppure: W^l_jk ~ U[-a, a] con a = (6/(n_(l-1) + n_l))
```

**Motivazione**: Mantiene varianza attivazioni costante attraverso i layers

**He Initialization** (ReLU):
```
W^l_jk ~ N(0, s) con s = (2/n_(l-1))
```

**Motivazione**: ReLU "spegne" met� neuroni (z<0)  serve pi� varianza

**Bias Initialization**:
```
b^l_j = 0  (common default)
oppure: b^l_j = 0.01  (piccolo positivo per ReLU)
```

**ERRORI DA EVITARE**:
-  Tutti pesi = 0  simmetria mai rotta, tutti neuroni imparano stesso!
-  Pesi troppo grandi  saturazione immediata (vanishing)
-  Pesi troppo piccoli  attivazioni  0 (vanishing in forward!)

### Complessit� Computazionale

**Forward Pass**:
```
Layer l: z^l = W^l  a^(l-1) + b^l
                 
     n_ln_(l-1)  n_l

Moltiplicazioni: n_l  n_(l-1)
Addizioni: n_l  n_(l-1) + n_l

Totale per rete: O(S_l n_ln_(l-1))
```

**Backward Pass** (Backprop):
```
Step 1 (Eq. 1): d^L = _a C  s'(z^L)   O(n_L)
Step 2 (Eq. 2): d^l = ((W^(l+1))^Td^(l+1))  s'(z^l)   O(n_ln_(l+1))
Step 3 (Eq. 3): C/b^l = d^l   O(n_l)
Step 4 (Eq. 4): C/W^l = d^l(a^(l-1))^T   O(n_ln_(l-1))

Totale per layer: O(n_ln_(l-1) + n_ln_(l+1))
Totale backprop: O(S_l n_l(n_(l-1) + n_(l+1)))
```

**Backprop vs Finite Differences**:
| Metodo | Complessit� | Note |
|--------|-------------|------|
| Finite Diff. | O(P  cost(forward)) | P = # parametri |
| Backprop | O(cost(forward)) | **Lineare** nei parametri! |

**Esempio**: NN con P = 10 parametri
- Finite Diff: 10 forward passes
- Backprop: 1 forward + 1 backward  2 forward
- **Speedup**: ~500,000!

### Connessione con Automatic Differentiation

**Backpropagation = Reverse Mode AD applicato a NN loss!**

| AD Concetto | NN Equivalente |
|-------------|----------------|
| Computational graph | Neural network architecture |
| Forward pass | Compute activations a^l, z^l |
| Reverse pass | Compute sensitivities d^l |
| Adjoint v? = y/v? | Error d^l_j = C/z^l_j |
| Edge derivatives | s'(z), W^l |
| VJP (w^TJ) | Backprop gradients |

**Perch� Reverse Mode**:
- Input: n parametri (10 - 10 per NN moderne)
- Output: 1 scalar loss C
- Reverse mode: O(m  cost) = O(1  cost)  **OTTIMALE!**
- Forward mode: O(n  cost) = O(10  cost)  impraticabile

### Esempio Numerico Completo

**Network**: 2-2-1 (2 input, 2 hidden, 1 output)

**Parametri**:
```
W^1 = [[0.15, 0.20],    b^1 = [0.35]
       [0.25, 0.30]]            [0.35]

W^2 = [[0.40, 0.45]]    b^2 = [0.60]
```

**Input/Target**:
```
x = [0.05, 0.10]^T
y = 0.01  (target)
```

**Forward Pass**:
```
Layer 1:
z^1_1 = 0.150.05 + 0.200.10 + 0.35 = 0.3775
z^1_2 = 0.250.05 + 0.300.10 + 0.35 = 0.3925
a^1_1 = s(0.3775) = 0.593
a^1_2 = s(0.3925) = 0.597

Layer 2:
z^2 = 0.400.593 + 0.450.597 + 0.60 = 1.106
a^2 = s(1.106) = 0.751

Loss (MSE):
C = (1/2)(0.751 - 0.01) = 0.274
```

**Backward Pass**:
```
Output layer (Eq. 1):
_a C = 0.751 - 0.01 = 0.741
s'(z^2) = s(1.106)(1-s(1.106)) = 0.7510.249 = 0.187
d^2 = 0.741  0.187 = 0.139

Hidden layer (Eq. 2):
(W^2)^Td^2 = [0.40]0.139 = [0.056]
               [0.45]         [0.063]

s'(z^1_1) = 0.593(1-0.593) = 0.241
s'(z^1_2) = 0.597(1-0.597) = 0.241

d^1_1 = 0.056  0.241 = 0.014
d^1_2 = 0.063  0.241 = 0.015

Gradients (Eq. 3-4):
C/b^2 = 0.139
C/b^1_1 = 0.014,  C/b^1_2 = 0.015

C/W^2 = [0.1390.593, 0.1390.597] = [0.082, 0.083]

C/W^1 = [[0.0140.05, 0.0140.10],
           [0.0150.05, 0.0150.10]]
        = [[0.0007, 0.0014],
           [0.0007, 0.0015]]
```

**Update (? = 0.5)**:
```
W^2  [0.40, 0.45] - 0.5[0.082, 0.083] = [0.359, 0.408]
b^2  0.60 - 0.50.139 = 0.530

W^1  [[0.15, 0.20],  - 0.5[[0.0007, 0.0014],
       [0.25, 0.30]]          [0.0007, 0.0015]]
    = [[0.1496, 0.1993],
       [0.2496, 0.2992]]
```

**Verifica**: Dopo update, C diminuisce! (0.274  ~0.260)

---

## Riferimenti Bibliografici

1. **Nielsen, M. A. (2015)**. "Neural Networks and Deep Learning." *Determination Press*. 
   - Capitolo 2: "How the backpropagation algorithm works" (derivazione dettagliata)

2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. "Deep Learning." *MIT Press*.
   - Capitolo 6: "Deep Feedforward Networks" (backprop, activation functions, initialization)

3. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986)**. "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.
   - Paper originale backpropagation!

4. **Glorot, X., & Bengio, Y. (2010)**. "Understanding the difficulty of training deep feedforward neural networks." *AISTATS*.
   - Xavier initialization, gradient vanishing problem

5. **He, K., Zhang, X., Ren, S., & Sun, J. (2015)**. "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification." *ICCV*.
   - He initialization per ReLU, PReLU

6. **LeCun, Y., Bottou, L., Orr, G. B., & M�ller, K. R. (1998)**. "Efficient BackProp." *Neural Networks: Tricks of the Trade*.
   - Best practices: normalization, initialization, learning rate

7. **Nair, V., & Hinton, G. E. (2010)**. "Rectified linear units improve restricted Boltzmann machines." *ICML*.
   - Introduzione ReLU per deep learning

---

**Fine Lezione 16 - Neural Networks e Backpropagation**

*Prossima lezione: Optimization Methods (SGD, Momentum, Adam, Learning Rate Schedules)*
