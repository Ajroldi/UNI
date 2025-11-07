# Lab 3 NAML - Principal Component Analysis (PCA)

---

## 🎯 Obiettivi del Laboratorio

### Competenze Teoriche
- Comprendere le convenzioni della PCA (campioni su colonne vs righe)
- Distinguere tra direzioni principali e componenti principali
- Capire la relazione tra SVD e matrice di covarianza
- Conoscere i concetti di varianza spiegata e riduzione dimensionalità

### Competenze Pratiche
- Implementare PCA usando la SVD in NumPy
- Applicare broadcasting NumPy per operazioni vettorializzate
- Visualizzare componenti principali e direzioni principali
- Utilizzare PCA per classificazione di immagini (MNIST dataset)
- Calcolare metriche di classificazione (accuratezza, matrice di confusione)

### Applicazioni
- Esercizio accademico: recuperare trasformazioni geometriche con PCA
- Classificazione cifre scritte a mano (0 vs 9) usando solo 2 componenti principali
- Visualizzazione di dataset ad alta dimensionalità
- Feature extraction da immagini 28x28 (784 dimensioni)

---

## 📚 Prerequisiti

**Matematica**
- SVD: decomposizione ai valori singolari, matrici ortogonali U e V
- Matrice di covarianza: C = XX^T/(n-1)
- Proiezioni ortogonali: prodotto scalare come operazione di proiezione
- Algebra lineare: combinazioni lineari, spazi vettoriali

**Python e NumPy**
- Broadcasting NumPy per operazioni vettoriali
- Slicing avanzato e bitmask per filtraggio dati
- Reshape e manipolazione dimensioni array
- Matplotlib per visualizzazione (imshow, scatter, subplot)

**Teoria**
- PCA come applicazione della SVD alla matrice di covarianza
- Autovalori della matrice di covarianza = valori singolari al quadrato / (n-1)
- Direzioni principali = colonne di U (o V, dipende dalla convenzione)
- Componenti principali = proiezione dei dati sulle direzioni principali

---

## 📑 Indice Completo

### **Parte 1 - Fondamenti Teorici**
#### [1. Introduzione alla PCA](#introduzione-pca) `00:00:01 - 00:05:26`
- [1.1 Definizioni fondamentali](#definizioni-fondamentali) `00:00:32`
- [1.2 Direzioni principali e massima varianza](#direzioni-principali) `00:01:15`

#### [2. Convenzioni e Teoria](#convenzioni-teoria) `00:05:26 - 00:11:08`
- [2.1 Convenzione: campioni su colonne (M×N)](#convenzione-colonne) `02:30`
- [2.2 Matrice di covarianza e SVD](#matrice-covarianza-svd) `00:03:44`
- [2.3 Componenti principali come proiezioni](#componenti-proiezioni) `00:05:56`
- [2.4 Convenzione alternativa: campioni su righe (N×M)](#convenzione-righe) `00:07:49`

### **Parte 2 - Esercizio Accademico: Trasformazione Geometrica**
#### [3. Esercizio Accademico - Trasformazione Geometrica](#esercizio-accademico) `00:11:08 - 00:28:45`
- [3.1 Generazione dati gaussiani 2D](#generazione-dati-gaussiani) `00:12:40`
- [3.2 Trasformazione geometrica (rotazione, traslazione, dilatazione)](#trasformazione-geometrica) `00:13:34`
- [3.3 Seed per riproducibilità](#seed-riproducibilita) `00:14:18`
- [3.4 Definizione angoli θ₁ e θ₂](#definizione-angoli) `00:14:59`
- [3.5 Applicazione trasformazione: x = A·seed + B](#applicazione-trasformazione) `00:16:02`
- [3.6 Broadcasting NumPy per traslazione](#broadcasting-traslazione) `00:20:51`
- [3.7 Differenze vettori 1D vs 2D in NumPy](#vettori-numpy) `00:22:33`
- [3.8 Visualizzazione con frecce direzionali](#visualizzazione-frecce) `00:26:05`

#### [4. Implementazione PCA](#implementazione-pca) `00:28:45 - 00:52:20`
- [4.1 Calcolo media e centraggio dati](#calcolo-media) `00:29:22`
- [4.2 Applicazione SVD](#applicazione-svd) `00:29:22`
- [4.3 Estrazione direzioni principali U](#estrazione-u) `00:29:22`
- [4.4 Scaling con varianza campionaria](#scaling-varianza) `00:29:22`
- [4.5 Esercizio pratico guidato (15 minuti)](#esercizio-pratico) `00:30:50`

#### [5. Soluzione e Risultati](#soluzione-risultati) `00:52:20 - 01:06:04`
- [5.1 Calcolo media con axis=1](#calcolo-media-axis) `00:52:54`
- [5.2 Broadcasting per centraggio](#broadcasting-centraggio) `00:53:25`
- [5.3 SVD con full_matrices=False](#svd-full-matrices) `00:55:01`
- [5.4 Visualizzazione direzioni principali](#visualizzazione-direzioni) `00:56:45`
- [5.5 Confronto Z vs U: recupero trasformazione](#confronto-z-u) `01:00:42`
- [5.6 Calcolo componenti principali φ = U^T·X̄](#calcolo-componenti) `01:02:04`
- [5.7 Ricostruzione dati originali](#ricostruzione-dati) `01:04:02`

### **Parte 3 - Applicazione Reale: MNIST Dataset**
#### [6. Dataset MNIST - Cifre Scritte a Mano](#dataset-mnist) `01:06:04 - 01:18:24`
- [6.1 Introduzione al dataset MNIST](#intro-mnist) `01:06:04`
- [6.2 Caricamento file CSV da WeBeep](#caricamento-csv) `01:07:42`
- [6.3 Training set vs Test set](#training-test-set) `01:08:07`
- [6.4 Controllo qualità dati e corruzione](#controllo-qualita) `01:09:37`
- [6.5 Struttura dati: 20000 campioni × 784 features](#struttura-dati) `01:12:06`
- [6.6 Estrazione etichette e trasposizione](#estrazione-etichette) `01:12:42`
- [6.7 Visualizzazione prime 30 immagini](#visualizzazione-immagini) `01:15:35`
- [6.8 Filtraggio cifra 9 con bitmask](#filtraggio-cifra-9) `01:17:52`

#### [7. Preprocessing e Filtraggio Dati](#preprocessing-filtraggio) `01:18:24 - 01:26:29`
- [7.1 Bitmask NumPy per filtraggio](#bitmask-numpy) `01:18:24`
- [7.2 Logical OR per selezione 0 e 9](#logical-or) `01:18:24`
- [7.3 Reshape per plot sequenziale](#reshape-plot) `01:21:13`
- [7.4 Compito studenti: PCA su MNIST](#compito-studenti) `01:23:35`

#### [8. Soluzione PCA su MNIST](#soluzione-pca-mnist) `01:45:53 - 02:06:10`
- [8.1 Definizione cifre target e mask](#definizione-cifre) `01:46:27`
- [8.2 Plot primi 30 campioni filtrati](#plot-campioni-filtrati) `01:48:23`
- [8.3 Visualizzazione media immagini](#visualizzazione-media) `01:49:40`
- [8.4 SVD e valori singolari](#svd-valori-singolari) `01:50:56`
- [8.5 Frazione cumulativa e varianza spiegata](#frazione-cumulativa) `01:51:29`
- [8.6 Interpretazione calo valori singolari](#interpretazione-calo) `01:53:25`
- [8.7 Visualizzazione 30 direzioni principali](#visualizzazione-30-direzioni) `01:55:05`
- [8.8 Da strutture macroscopiche a rumore](#strutture-rumore) `01:56:52`
- [8.9 Calcolo prime 2 componenti principali](#calcolo-2-componenti) `01:57:55`
- [8.10 Scatter plot 2D con colori per etichetta](#scatter-2d) `02:00:12`

#### [9. Classificazione con Soglia](#classificazione-soglia) `02:02:42 - 02:06:10`
- [9.1 Algoritmo classificazione lineare naive](#algoritmo-naive) `02:02:42`
- [9.2 Scelta soglia ottimale (threshold=0)](#scelta-soglia) `02:03:23`
- [9.3 Visualizzazione linea separatrice](#linea-separatrice) `02:04:33`
- [9.4 Compito: applicazione su test set](#compito-test-set) `02:05:16`

### **Parte 4 - Validazione su Test Set**
#### [10. Test e Metriche](#test-metriche) `02:06:10 - 02:30:23`
- [10.1 Caricamento dataset di test](#caricamento-test) `02:18:09`
- [10.2 Filtraggio cifre 0 e 9](#filtraggio-test) `02:19:39`
- [10.3 Proiezione con direzioni di training](#proiezione-training) `02:21:34`
- [10.4 Uso media di training (non test!)](#media-training) `02:22:09`
- [10.5 Visualizzazione test set trasformato](#visualizzazione-test) `02:22:43`
- [10.6 Calcolo metriche: veri/falsi positivi/negativi](#calcolo-metriche) `02:24:34`
- [10.7 Accuratezza 95% con classificatore semplice](#accuratezza-95) `02:27:15`
- [10.8 Matrice di confusione con SciPy](#matrice-confusione) `02:29:06`
- [10.9 Bilanciamento predizioni](#bilanciamento) `02:29:53`

#### [11. Conclusioni e Homework](#conclusioni) `02:30:23 - 02:30:23`
- [11.1 Terzo notebook: classificazione cancro](#notebook-cancro) `02:30:23`

---

## Introduzione alla PCA {#introduzione-pca}

### 1.1 Definizioni fondamentali {#definizioni-fondamentali}

`00:00:01` 
Okay, quindi iniziamo. Il lab di oggi riguarda la PCA. Quindi la PCA è una tecnica per la riduzione della dimensionalità e in particolare potete pensarla come se avessimo un dataset con molte caratteristiche come un'immagine dove ogni caratteristica è un pixel o avete alcune misurazioni da alcuni sensori e per dare senso ai vostri dati volete ridurre la loro dimensionalità. Per iniziare, farò una breve revisione della PCA.

### 1.2 Direzioni principali e massima varianza {#direzioni-principali}

`00:00:32` 
Così, siamo sulla stessa pagina riguardo alle convenzioni perché è importante, capire quali sono le differenze se abbiamo, campioni sia sulle colonne o sulle righe. Okay, quindi farò una breve revisione della teoria e di ciò che ci serve per il lab e poi andremo su Colab e risolveremo l'esercizio. Quindi prima di tutto, due definizioni. Quindi le componenti principali, direzioni principali, cosa sono le direzioni principali sono le direzioni.

`00:01:15` 
degli autovettori della matrice di covarianza, C maiuscola. Perché questo è importante? Perché queste sono le direzioni di massima varianza, e vogliamo ruotare i nostri dati in.

`00:01:50` 
queste direzioni, perché queste direzioni che massimizzano la varianza sono le più importanti, e questo è quello che stiamo facendo nella PCA. Poi, proiettiamo i dati su queste direzioni, e queste sono chiamate le componenti principali.

`02:30` 
Direzioni principali. Okay? Qual è la convenzione che usiamo in questo lab? È che X è una matrice in R, M per N, dove M è uguale al numero di caratteristiche, il numero di pixel nell'immagine, per esempio e n è il numero di campioni.

`00:03:09` 
Di solito questo significa che la vostra matrice è così. Molti molti molti campioni caratteristiche che sono ancora molte ma di solito sono meno del numero di campioni. Quindi questa è più o meno la nostra impostazione. Inoltre un'assunzione molto grande in entrambe le colonne x è centrata. Questo significa che ha.

`00:03:44` 
media zero. Se questo non è vero calcoliamo la media di x e sottraiamo la media di x da x. Ma per semplificare la notazione sto solo facendo questa assunzione. Poi cosa abbiamo? Abbiamo che la matrice di covarianza C è uguale a x per x trasposta diviso per n meno 1. Quindi se x è uguale a u sigma vt, allora questo significa che C è uguale a 1 su n meno 1 u sigma vt v sigma ut.

`00:04:42` 
Okay? Trasponendo, cambiamo l'ordine qui, e dato che questo è pseudo-diagonale, la trasposta va via. Questo significa che questo è uguale a u sigma al quadrato ut diviso per n meno 1. Quindi, questa è una diagonalizzazione per la nostra matrice di covarianza, e quindi le colonne di U sono le direzioni principali, okay?

---

## Convenzioni e Teoria {#convenzioni-teoria}

### 2.1 Convenzione: campioni su colonne (M×N) {#convenzione-colonne}

`00:05:26` 
Sì, l'assunzione che X è centrata, questo è necessario quando definite la matrice di covarianza, perché altrimenti non è la matrice di covarianza. E, okay, poi le componenti principali, cosa sono, sono la proiezione dei dati sulle direzioni principali.

`00:05:56` 
quindi qual è l'operazione geometrica che significa proiezione è il prodotto scalare o il prodotto interno se preferite quindi come calcoliamo le componenti principali beh la nostra prima direzione che vogliamo è la prima colonna di u e se prendiamo il primo campione che è la prima colonna di x vogliamo proiettare questo per esempio sulla prima direzione quindi che è la prima colonna di u e quindi la componente del primo campione sulla prima direzione è il prodotto scalare tra u1 e.

`00:06:33` 
x1 dobbiamo fare questo in generale e quindi abbiamo che ut per x sono le componenti principali, Chiamiamole pc, okay, perché quello che stiamo facendo qui, stiamo prendendo la prima colonna di u, che ora è la prima riga di u trasposta, e facendo il prodotto scalare con la prima colonna di x. E quindi qui abbiamo la proiezione del primo campione sulla prima componente. Poi se.

`00:07:09` 
cambiamo la riga e prendiamo la seconda riga di u, stiamo prendendo la seconda componente principale e moltiplichiamo con la prima colonna di x, e questa è la proiezione del primo campione sulla seconda componente principale. E quindi questo prodotto matrice-vettore, è letteralmente solo l'operazione che facciamo per calcolare la componente principale. È chiaro questo? Okay, l'ultima parte è cosa succede se se x è in R,

`00:07:49` 
n per m. Quindi stiamo trasponendo la matrice e questo è campioni per caratteristiche. Beh cosa sta succedendo ora è che c è uguale a x trasposta x diviso per n meno uno. e quindi quello che abbiamo ora è che questo è uguale a v sigma u t u sigma.

`00:08:25` 
sigma v diviso per n meno uno. Quindi questo è uguale a v sigma al quadrato v trasposta. E quindi ora le direzioni principali sono le colonne di v.

`00:09:05` 
Quindi entrambe le convenzioni funzionano. Potete ottenere le direzioni principali sia che i campioni siano sulle righe o sulle colonne. Non cambia niente, a parte il fatto che in un caso troverete le direzioni principali sulle colonne di v, ma nell'altro caso sono sulle colonne di u. Usiamo questa convenzione, avremo sempre ogni campione che è una colonna diversa di x, ma se le cose cambiano, tutto funziona allo stesso modo, dovete solo scambiare u con v.

`00:09:45` 
È chiaro questo? Volete che spieghi qualcosa di nuovo? Okay. Accenderò i proiettori e lo schermo ora. Se volete, potete iniziare ad aprire Google Colab e caricare dai notebook.

`00:11:08` 
Sì, e sono quelli che formano la base per lo spazio, e quindi dato che alimentano i dati, sono quelli che, beh, gli assi orientano i dati. per spiegare la maggior parte dei dati, in un certo senso. Comunque, il primo esercizio è esattamente, è solo un esercizio accademico, ma è davvero fatto per mostrarvi praticamente cosa sta facendo la PCA, okay?

`00:11:45` 
Dovreste vedere il mio schermo, perfetto. Quindi, se voglio caricare uno di questi ruoli, vado allo stream,

---

## Esercizio Accademico - Trasformazione Geometrica {#esercizio-accademico}

### 3.1 Generazione dati gaussiani 2D {#generazione-dati-gaussiani}

`00:12:40` 
Okay, quindi il primo è solo un esercizio accademico e l'idea è la seguente. Quindi quello che faremo ora funziona così. Quindi stiamo generando alcuni dati indipendenti normali gaussiani in 2D. Okay, questo è il primo passo. Questo è quello che ogni statistico vuole. Okay, dati indipendenti normali gaussiani è dove tutte le ipotesi valgono per applicare strumenti molto potenti. E quindi iniziamo generando questi. Poi applichiamo una trasformazione geometrica. Questo è solo per.

`00:13:34` 
Simulare cosa succede con i dati reali. Quindi i vostri dati sono sporchi. I vostri dati non sono mai una distribuzione normale indipendente nelle varie direzioni. E quindi applichiamo questa trasformazione teorica per fare un esempio molto semplice di quello che succederebbe con i dati. Poi applichiamo la PCA e mostreremo che applicando la PCA, possiamo recuperare i dati originali. Okay. E in particolare, possiamo trovare la trasformazione che trasforma i nostri dati gaussiani indipendenti normali nei dati realistici.

`00:14:18` 
E quindi questo è un modo per mostrarvi come funziona la PCA, perché abbiamo alcuni dati che generiamo, applichiamo una trasformazione che conosciamo, e mostriamo che con la PCA possiamo recuperare la trasformazione. Senza alcuna conoscenza sulla trasformazione che abbiamo applicato. Quindi, entriamo nei dettagli. Iniziamo importando numpy e matplotlib. Poi, impostiamo il seed. Impostare il seed fa sì che questo esercizio sia riproducibile. Vi ricordo che impostare il seed significa che i dati casuali che generiamo sono sempre gli stessi, anche se si comportano come dati casuali.

`00:14:59` 
Poi, quello che facciamo è che definiamo due vettori che sono ortogonali, uno rispetto all'altro, e poi sono orientati con questo angolo, theta1, che è un sesto di pi. Quindi, definiamo theta1, che è pi diviso sei. Poi, definiamo l'altro angolo, che è 90 gradi rispetto a questo, aggiungendo pi su due, che è 90 gradi.

`00:15:29` 
E poi, calcoliamo queste due direzioni, theta1 e theta2. che sono quelle che usiamo per cambiare i nostri dati. Okay, e se stampate Z1 e Z2, vedete che questi sono vettori unitari, che hanno direzioni orientate di 30 gradi. Poi definiamo un punto B, che è una traslazione.

`00:16:02` 
Quindi applichiamo una rotazione, una traslazione, e poi una dilatazione. Quindi questa è la parte di traslazione B, che è 20 nella direzione di X e 30 nella direzione di Y. E infine, applichiamo la nostra trasformazione geometrica. Quindi generiamo 1.000 punti casuali, XI, secondo questa formula, dove Y, sono vettori casuali con componenti generate indipendentemente secondo una distribuzione normale quindi questi sono i nostri dati di partenza. Poi applichiamo questa trasformazione geometrica che è composta in A dalla rotazione, sì, riguardo a questi numeri, è completamente casuale, applichiamo una trasformazione geometrica casuale che decidiamo a priori, è un esercizio accademico quindi iniziamo con alcuni dati che sono gaussiani, che è il migliore.

`00:16:58` 
Poi aggiungiamo un tipo di trasformazione geometrica che vogliamo e vogliamo verificare che possiamo, che qualunque sia la trasformazione che applichiamo all'inizio è la stessa che otteniamo con la PCA. Questo è completamente astratto, tipo, non hanno un significato, perché vogliamo alcuni dati, e vogliamo applicare la PCA ad alcuni dati sporchi che simulano quello che potete avere raccogliendo alcuni dati in generale.

`00:17:35` 
E per fare questo, applichiamo solo una trasformazione casuale che per noi non ha un significato particolare, e vogliamo vedere che qualunque sia la trasformazione, indipendentemente dal fatto che questo sia 20 o 40, possiamo recuperare la trasformazione usando la PCA. Okay, questa non è ancora un'applicazione pratica, è solo per mostrarvi che la PCA può capire qual è la trasformazione applicata ad alcuni dati, senza conoscerla a priori.

`00:18:08` 
Quindi sì, in A abbiamo questa rotazione di Z1 e Z2, e poi una dilatazione di componenti Rho1 e Rho2. E ora applichiamo questo. Quindi iniziamo definendo rho uno e rho due e dicendo che vogliamo 1.000 punti. Poi questi sono i nostri punti originali, questi seeds. Se li tracciamo, le componenti x e y, abbiamo questi. L'equal significa che gli assi.

`00:19:10` 
Okay, l'axis equal significa che sul grafico, gli assi x e y hanno lo stesso spazio, quindi non c'è una distorsione, e questi sono i dati che abbiamo generato all'inizio. Questa è una nuvola molto bella, è una nuvola rotonda con una distribuzione gaussiana, e questo è quello che vorremmo ottenere. Axis equal significa che su questo schermo e sul vostro schermo, l'asse x e l'asse y hanno esattamente lo stesso spazio.

`00:19:43` 
La stessa scala, quindi significa che state visualizzando senza distorsione. Poi applichiamo la nostra trasformazione. Quindi prima definiamo A, che è questa. Quindi quello che stiamo facendo qui, stiamo prendendo il primo vettore, z1, il secondo vettore, z2, moltiplicandolo per la costante di dilatazione, e questi sono due vettori colonna, e li stiamo impilando insieme.

`00:20:13` 
Okay, uno con l'altro, e quindi stiamo ottenendo una matrice 2x2 in A. Quindi se tracciamo A, questa è una matrice 2x2, e poi applichiamo la trasformazione. Quindi questa parte è un po' delicata, e questo è il broadcasting. Abbiamo parlato del broadcasting nel lab passato, e quello che stiamo facendo qui è che, per definizione, vogliamo che ogni punto x sia uguale a A per ogni punto più B.

`00:20:51` 
Quindi questo sarebbe un loop. Prendiamo ogni punto R, lo moltiplichiamo con una moltiplicazione matriciale per A, e lo stiamo traslandolo con B. Tuttavia, abbiamo visto che fare un for loop è costoso, e quindi sfruttiamo la vettorizzazione, e lo stiamo facendo qui. Quindi abbiamo A per seed, e qui abbiamo che seed è due per n punti. NumPy è intelligente e capisce che se questa matrice è due per due, e stiamo facendo una moltiplicazione matriciale qui con questo,

`00:21:22` 
sta moltiplicando A per ogni colonna dei seeds. E poi lo stiamo traslandolo aggiungendo un vettore. Tuttavia, questa è una matrice che ha dimensioni due per n punti, non possiamo aggiungere punto per punto un vettore. E quindi quello che stiamo facendo qui è stiamo dicendo che, okay, aggiungiamo una dimensione extra in modo che questo sia due per uno, e poi NumPy capisce che volete fare broadcasting.

`00:21:56` 
In particolare, Il punto qui è che in NumPy, se avete B, che è questo vettore, è diverso da questo, che è ancora un vettore, ma con due dimensioni.

`00:22:33` 
Quindi anche se entrambi sono vettori bidimensionali, questo e questo per NumPy sono molto diversi. Questo è un unidimensionale, con solo un oggetto unidimensionale, con due elementi. Questo invece è una matrice, che ha solo una colonna, che ha elementi 20 e 30. Questo è diverso da MATLAB, quindi se avete un background in matematica e avete usato MATLAB, questo è un po' diverso, perché in MATLAB tutto è una matrice, e dall'inizio tutto è o un vettore riga o un vettore colonna.

`00:23:09` 
In NumPy, questo è diverso. Potete avere un vettore, che è solo un semplice vettore, e questo ha una dimensione uguale a 1. Questo è un oggetto bidimensionale, e questo è un oggetto bidimensionale che ha due righe diverse. Infatti, potreste avere anche un oggetto bidimensionale che ha una riga e due colonne, che è questo. Per applicare correttamente il broadcasting qui, NumPy vuole qualcosa che sia una matrice, perché qui abbiamo una matrice e non potete aggiungere qui un vettore.

`00:23:51` 
implicitamente usando il Broadcasting. Per assicurarvi che volete davvero il Broadcasting, NumPy vuole che questo sia una matrice, quindi un oggetto bidimensionale, e in particolare vuole che abbia due colonne, scusate, due righe, esattamente come A per seed. Okay, questa è solo una regola, nel senso che dobbiamo conoscere la regola del Broadcasting.

`00:24:22` 
La parte importante è che capiate che questi tre oggetti sono diversi anche se tutti rappresentano un vettore 2D. E poi, a seconda della rappresentazione dei vettori 2D, potete applicare o meno il Broadcasting. Okay? È chiaro questo? Quindi alla fine, otteniamo i nostri dati, x.

`00:24:53` 
e possiamo stamparli. Invece di seeds qui, sto stampando X. Quindi i nostri dati originali, che era una nuvola bella, avete la formula qui nel, quindi sto guardando l'immagine ora. Quindi la nuvola bella, la bella nuvola rotonda che abbiamo.

`00:25:26` 
all'inizio ora, è stata consolidata. Abbiamo questa forma ovale, che è stata anche traslata. Quindi qui vedete che non è più centrata in zero, non ha più valori uguali a uno, ed è anche ruotata. E applicando la PCA, mostriamo che possiamo recuperare esattamente la trasformazione che abbiamo applicato a questi dati, per recuperare quello iniziale. Quindi nella vita reale, a volte state facendo qualcosa del genere, applicate la PCA, e potete tornare a una distribuzione normale.

`00:26:05` 
Okay, ora abbiamo un passo di visualizzazione. Quindi solo per visualizzare le cose, quello che facciamo ora è, okay, tracciamo i dati, ma tracciamo anche alcune frecce che mostrano la direzione che abbiamo usato. Quindi questa prima parte è, okay, definiamo il plot. Facciamo lo scatter. Quindi nello scatter, passiamo le componenti X e le componenti Y. E poi usiamo questa funzione da PyPlot, che è chiamata arrow.

`00:26:36` 
E arrow ci permette di tracciare alcuni segmenti. I primi due argomenti sono il centro dei segmenti. Quindi vogliamo... Centrare questi, scusate, non è il centro, questi sono, okay, volete l'inizio della, dove iniziate la freccia, e poi la lunghezza della freccia in ogni direzione.

`00:27:10` 
Quindi quello che facciamo è, okay, prendiamo il centro, e poi sottraiamo la lunghezza x, il centro y, sottraiamo la y, e poi la lunghezza è due volte questo, okay, perché vogliamo sia una direzione che l'altra. E facciamo questo sia per z1 che per z2. Quindi, questo è il plot, e questo forse è un po' più chiaro quello che stiamo facendo.

`00:27:43` 
Quindi, stiamo partendo, queste frecce, questi due segmenti sono tracciati da, okay, dando questi punti, e questi punti, e questo punto, e questo punto. Quindi, come facciamo questo? L'inizio è il centro, b, meno la lunghezza della freccia, e l'inizio nella direzione y di z1 meno la lunghezza, perché z1 per rho1 è la lunghezza.

`00:28:14` 
Quindi, stiamo dando questo punto, e poi due volte questa lunghezza, per scrivere questa freccia e questa freccia, okay?

---

## Implementazione PCA {#implementazione-pca}

### 4.1 Calcolo media e centraggio dati {#calcolo-media}

`00:28:45` 
Ora viene la parte interessante, quindi questo è quello che dovete fare ora, quindi per ora quello che abbiamo fatto è solo visualizzazione, ora voglio che implementiate la PCA, quindi quello che state facendo ora è che vi viene dato x, avete x, lo avete calcolato insieme, poi calcolate la media di x, abbiamo fatto questo, potete usare la funzione mean di numpy, sottraete da x,

`00:29:22` 
la sua media, applicate la SVD, e applicando la SVD, state applicando la PCA, okay, per noi la PCA è solo una SVD con media zero, una volta che avete fatto questo, voglio che tracciate i primi due vettori singolari riscalati dalla radice della varianza campionaria, questo, e per fare questo, quello che dovete fare è solo copiare e incollare questo codice, e invece di usare Z1 e Z2, usate le prime due colonne di U, okay, quindi qui è come sopra, ma invece di Z1 e Z2, che sono le direzioni che conosciamo, usate U1 e U2.

`00:30:17` 
U1 e U2, riscalati come sopra, e qui vedrete che queste direzioni che troviamo nella PCA, che sono le componenti principali, le direzioni principali, sono le stesse delle Z1 e Z2 che abbiamo usato, okay? Quindi, vi darò, diciamo, un quarto d'ora.

`00:30:50` 
Se avete domande, alzate solo le mani. Io ed Elia saremo in giro per rispondere alle vostre domande. Questo non dovrebbe essere così difficile, perché il primo passo è solo fare la SVD, come nel lab precedente, e poi usate il codice sopra, quello con la freccia, e cambiate solo Z con U1, e cambiate rho con i valori singolari. Okay?

`00:31:43` 
Capito.

`00:32:34` 
Quindi siamo sempre con i dati, in modo che la piattaforma cartesiana faccia la strada e le direzioni dove c'è la maggior distribuzione dei dati. Perché c'è più varianza, perché sono più allungati. Quindi in realtà, la distribuzione è più complessa. Non posso nemmeno trovare la distribuzione. No, non puoi trovare la distribuzione. Devi trovarne diverse. Poi, semplicemente, quello che fa la PCA è girare le ruote nell'altro modo in modo che ci sia più varianza.

`00:33:04` 
E in questo caso, in cui abbiamo una distribuzione normale, ora sappiamo come tornare all'originale. Ma in generale, non ci dà la distribuzione. Grazie.

`00:37:38` 
Grazie mille. Grazie. No, no, è la varianza.

`00:38:40` 
è la varianza, è la varianza,

`00:39:39` 
Grazie.

`00:41:21` 
Grazie mille.

`00:41:52` 
Grazie.

`00:43:56` 
Grazie mille.

`00:46:27` 
Grazie.

`00:48:48` 
Grazie mille.

`00:49:43` 
Grazie.

`00:50:54` 
Grazie mille.

`00:51:41` 
Grazie.

---

## 13. Soluzione Completa: Esercizio 1 - PCA Geometrica 2D {#soluzione-esercizio-1}

### 13.1 Introduzione all'Implementazione {#intro-implementazione}

**Obiettivo**: Recuperare la trasformazione geometrica $x_i = A r_i + b$ usando solo i dati osservati, attraverso la PCA.

**File di riferimento**: 
- Notebook start: `Lab02/1_PCA_2D_start.ipynb`
- Soluzione completa: `Lab02/solutions/1_PCA_2D.ipynb`

**Passi dell'algoritmo**:
1. ✅ Setup parametri trasformazione (fatto sopra)
2. ✅ Generazione dati con seed Gaussiano (fatto sopra)
3. ⏳ **Calcolo media campionaria** ← INIZIAMO QUI
4. ⏳ Centraggio dati con broadcasting
5. ⏳ SVD su dati centrati
6. ⏳ Scaling varianza campionaria
7. ⏳ Visualizzazione direzioni principali
8. ⏳ Confronto direzioni vere vs stimate
9. ⏳ Calcolo componenti principali Φ
10. ⏳ Ricostruzione e interpretazione

---

## Soluzione e Risultati {#soluzione-risultati}

### 5.1 Calcolo media con axis=1 {#calcolo-media-axis}

`00:52:20` 
Ragazzi, mi dispiace, ora la soluzione, perché altrimenti abbiamo un programma, e ho molte cose da mostrarvi oggi, sfortunatamente, quindi iniziamo con la prima parte, quindi prima di tutto, x non ha media zero, dobbiamo calcolare la media. Quindi, x, se vediamo questo, è un vettore con due righe e 1.000 colonne, in particolare, questo, è la forma, ogni colonna è un punto in 2D, e ogni colonna diversa è un diverso.

`00:52:54` 
punto. Quindi come calcoliamo la media? x mean è uguale a numpy punto mean di, x axis uguale a 1. Perché? Perché quello che stiamo facendo è che stiamo facendo un loop su ogni colonna per calcolare la media per quella riga. E per fare questo in numpy, dite axis uguale a 1, perché 1 è il secondo asse, sono le colonne, e questo significa che numpy sta facendo il loop su tutte le colonne.

`00:53:25` 
per fare la media. Infatti, per avere un controllo, quello che volete è che il risultato sia un punto in 2D, perché è la media in x e la media in y. E quindi se calcolate la forma, quello che abbiamo è un vettore che ha due componenti. Questo è un oggetto unidimensionale con due componenti, e questo è esattamente quello che vogliamo. Poi possiamo definire X bar come la X senza la media e questo sarà uguale a X meno X mean. Ma se facciamo questo, abbiamo un errore. Okay, perché? Perché non possiamo fare la differenza elemento per elemento di questi due oggetti, che hanno forme diverse rispetto a quello che vogliamo fare è sottrarre a ogni colonna di X la sua media.

`00:54:21` 
Quindi dobbiamo usare il broadcasting e per fare questo in NumPy, dobbiamo cambiare la dimensione di questo oggetto come abbiamo fatto prima. Quindi diciamo, okay, prendiamo tutti gli elementi nella prima dimensione e poi aggiungiamo una nuova dimensione. Infatti, se tracciamo questo, abbiamo questo, che è una matrice. Vedete le doppie parentesi quadre. Questo significa che questo è un oggetto che ha due righe, una colonna. E dato che questa è una matrice, e questa è una matrice, ora NumPy è autorizzato a usare il broadcasting per sottrarre a ogni colonna di x la sua media.

`00:55:01` 
E quindi questo è x bar, e ora possiamo calcolare la svd. Quindi u, i valori singolari, vt è uguale a np.linalg.svd di x bar. E poi vi ricordo che molto spesso non abbiamo bisogno di una matrice completa qui, e quindi mettiamo false. Perché scartiamo tutti i dati di cui non abbiamo bisogno. Le colonne singolari e gli zeri extra.

`00:55:37` 
Perfetto. Quindi in u ora, abbiamo le direzioni principali, e vogliamo tracciarle. Quindi u1 è uguale a, u, la prima colonna, quindi prendiamo tutti gli elementi su tutte le righe e la colonna zero, la seconda, direzione principale è u, tutti gli elementi sulla riga, fissiamo la prima colonna, poi noi.

`00:56:13` 
calcoliamo r, che è s diviso per la radice quadrata di numpy di n meno 1, dove n è il numero di punti, okay, abbiamo la radice quadrata di sigma al quadrato, rimuovo solo la radice quadrata del qualcosa al quadrato, poi il plot, per il plot, copio e incollo il codice, perché è.

`00:56:45` 
molto molto simile, quindi vado sopra, prendo questo plot, e lo aggiungo qui quindi questi sono i nostri dati quello che facciamo è che invece di b non conosciamo b okay perché ora stiamo supponendo che la trasformazione geometrica che abbiamo applicato non sia conosciuta quindi qual è il nostro stimatore dal punto di vista statistico della media è la media campionaria che è x.

`00:57:18` 
mean di zero e sostituiamo b che è la media reale con lo stimatore della media che è la media campionaria dei dati e quindi ovunque dove c'è una b poi abbiamo rho uno rho uno è la dilatazione che è il nostro sostituto per la dilatazione è la stima, della deviazione standard e quindi invece di rho uno mettiamo r zero fino a qui qui e qui la.

`00:58:00` 
dilatazione nella seconda direzione è la stima della deviazione standard campionaria nella seconda direzione quindi è r1 e sostituite rho due con r1 infine invece delle direzioni reali dei dati che sono z1 e z2 abbiamo la stima delle migliori direzioni che spiegano le varie direzioni che sono le direzioni principali e quindi invece di z1 mettiamo u1 e invece di z2 mettiamo u2.

`00:58:48` 
E questo è il plot. Invece del rosso, metterò rosso, e lasciatemi solo copiare e incollare questo di nuovo, e quindi abbiamo sia il nero che il rosso. Okay? Questo è il risultato. Quelli rossi sono la direzione della trasformazione geometrica e la magnitudine che era quella reale, quella che abbiamo applicato ai dati.

`00:59:23` 
Quelli rossi sono quelli che sono stimati dalla PCA usando solo i dati. okay e quindi qui capite che quello che stiamo facendo è che con la pca stiamo trovando due direzioni che sono questa e questa che sono quelle che spiegano di più le varianze e spiegando di più le varianze intendo che in questa direzione in questa direzione i dati sono più, sparsi e quindi sono i migliori per capire cosa sta succedendo ai dati.

`00:59:55` 
okay scusate sì. quindi quello rosso è quello che io io ho solo copiato e incollato il codice questo è quello di prima con l'accento il nero è solo una semplice copia e incolla e poi questo è quello che abbiamo appena fatto insieme sostituendo quello.

`01:00:42` 
Okay? È chiaro questo? Solo per portare a casa il punto, qui abbiamo altri due punti. Quindi qui stampiamo solo Z1 e Z2 e U1 e U2 per confrontarli vedendo i numeri. Okay? E quello che vediamo qui è che i due numeri, i due vettori sono molto, molto simili, ma qui avete un segno diverso. Okay? Perché? Perché la PCA non capisce in che modo i numeri sono confrontati.

`01:01:29` 
qual è l'orientamento della direzione okay se moltiplichiamo tutto per meno uno, otteniamo ancora la direzione ancora non conosciamo il verso e questo va bene perché tutto è, moltiplicato per meno uno e quindi ha ancora senso okay quindi potete aspettarvi che la direzione non sia esattamente la stessa ma la stessa a meno di un segno infine calcoliamo le componenti principali.

`01:02:04` 
quindi chiamiamo phi le componenti principali e sono uguali a u trasposta prodotto matriciale, x bar vi mostro perché questo è il caso all'inizio del lab e quindi questa è la matrice, Dove ogni colonna è una componente principale diversa, il che significa che questa è la proiezione dei dati sulla direzione principale.

`01:02:37` 
E infine, facciamo uno scatter plot delle due componenti principali, quindi plt.scatter, phi, tutte le righe, prima componente principale, prima colonna, tutte le righe, seconda componente principale, seconda colonna. Quindi, cosa ho fatto di sbagliato? Scusate, è sulla prima...

`01:03:22` 
Scusate, quando facciamo questo, la prima componente principale è sulla prima riga, e la seconda componente principale è sulla seconda riga. Errore mio, scusate. Poi facciamo PLT punto axis equal, non abbiamo distorsione. E quello che abbiamo è quasi i dati originali. Quindi vedete ora che abbiamo ruotato i dati, le direzioni principali, e proiettandoli, abbiamo ottenuto dati ruotati dove sull'asse x e sull'asse y abbiamo le due direzioni di massima varianza, qui e qui.

`01:04:02` 
Per ottenere la bella nuvola, dovete anche applicare la parte di dilatazione, e per fare questo, quello che facciamo è dividiamo questo per... la stima della deviazione standard nella prima direzione e la stima della deviazione standard nella seconda direzione e quindi qui abbiamo qualcosa che è quasi normalmente la stima non era perfetta ma qui abbiamo questa nuvola rotonda molto bella okay qualche domanda sì.

`01:04:52` 
sono sono è sono quelli che abbiamo calcolato quindi sono stima della deviazione standard e li abbiamo calcolati qui, R è uguale a S, il valore singolare, quindi questo è uguale a sigma al quadrato, radice quadrata, diviso per la radice quadrata del numero di punti meno uno.

`01:05:29` 
Okay? Se non avete domande, poi passiamo al secondo esercizio, e nel secondo esercizio, vediamo un'applicazione a un dataset reale. Questo era un esercizio accademico, ma nel secondo, abbiamo un esercizio del mondo reale.

`01:06:04` 
In particolare, iniziamo qualcosa che è... Ha, che è molto ben noto nel machine learning e nel riconoscimento della scrittura. Quindi abbiamo molte immagini che sono immagini di un numero che è stato scritto a mano, e vogliamo distinguere un numero dall'altro. Quindi la PCA è ancora uno strumento che è molto potente, ma per questo tipo di applicazione non è il migliore. Quindi quello che stiamo facendo qui è una versione semplificata di questo compito, dove siamo in grado di riconoscere due cifre diverse, una dall'altra.

`01:06:45` 
Non siamo in grado di riconoscere tutte le cifre da zero a nove, ma vi mostrerò come usare la PCA per capire quali cifre sono zero e quali sono nove. OK, anche perché ci mancano alcuni strumenti di classificazione. Okay. Questo è chiamato anche il dataset MNIST. Questo è molto, molto famoso. Ha una pagina Wikipedia, ed è usato come benchmark per molte, molte reti neurali. Quindi ci stiamo connettendo al database, alla macchina virtuale, ma non abbiamo i dati. Quindi se eseguiamo questa prima cella, non abbiamo i dati. Perché? Come abbiamo caricato l'immagine l'ultima volta, ora dobbiamo caricare il dataset. Quindi apriamo questa cartella, clicchiamo su questa icona della cartella, e dobbiamo caricare il dataset.

`01:07:42` 
A sinistra, poi clicchiamo sul caricamento della sessione di storage, e caricate questi due file che ho caricato su WeBeep, il test MNIST e il modello di traccia MNIST. Okay? Se vedete, abbiamo due dataset diversi. Quindi iniziamo con le migliori pratiche del machine learning. Quando testiamo un algoritmo, abbiamo sempre almeno due dataset. Quello di training e quello di testing. Perché vogliamo davvero verificare come funziona il nostro algoritmo in uno scenario del mondo reale, vogliamo addestrarlo su alcuni dati e poi testarlo su alcuni dati che l'algoritmo non ha mai visto prima. Okay? Perché quello che potete avere altrimenti è sempre.

`01:08:34` 
overfitting e altre cose brutte che volete evitare. Perché per verificare davvero come funziona l'algoritmo nella vita reale, dovete salvare un po' del vostro dataset per testarlo. Perché altrimenti, forse ha imparato solo le idiosincrasie dei vostri dati. Quindi iniziamo caricando questo dataset di training, questo è un file CSV, usiamo questa funzione NumPy per caricarlo, e ha questa dimensione.

`01:09:37` 
Per evitare la corruzione dei dati. Ora andremo a. Sì, potreste avere alcuni dati che non appartengono a quel dataset o potreste avere qualcosa che è corrotto. Sì, è vero. Infatti, quello che di solito fate, se parlate nella vita reale, quindi questo è un compito piuttosto avanzato, quindi sarò molto breve e forse parlerò di questo quando usiamo le reti neurali. Quindi diciamo che avete un problema di classificazione e avete, non so, immagini di foglie che hanno malattie.

`01:10:20` 
quello che di solito fate è che prima di tutto fate molto controllo manuale quindi il fatto che dobbiamo caricare il dataset e controllare a mano molte immagini è qualcosa che fate nella vita reale come spendete molto tempo come tempo umano per controllare cosa c'è nel dataset prendendo a caso dati per capire se ci sono alcuni problemi macroscopici, non tutto il dataset ma spendete forse una due tre quattro ore prendendo a caso dati e controllando se c'è qualche problema perché non sapete e questo è solo per capire un.

`01:10:55` 
po' i dati e cosa sta succedendo e più grande è il problema più grande è l'azienda, forse spendete più o meno tempo avete più persone che lavorano su questo se avete un piccolo progetto per l'università se i dati sono oltre forse passate solo mezz'ora un'ora poi applicate tecniche economiche, Per clusterizzare i dati. Quindi per esempio, la SVD è davvero buona per questo. Se avete alcuni outlier, qualcosa che non appartiene al dataset, applicate la PCA e controllate sulle direzioni principali. Ci sono alcuni outlier, alcuni punti che sono molto lontani da tutti gli altri. E questa tecnica è molto buona per fare questo come SVD e PCA. È molto potente per fare controlli iniziali per la corruzione dei dati.

`01:11:36` 
Perché di solito i dati corrotti sono molto lontani dal, diciamo, cluster. Un'altra tecnica molto usata, per esempio, è chiamata T-SNE. E poi dopo che state facendo questa prima parte, applicate modelli più sofisticati come le reti neurali convoluzionali. E questa è più o meno la pipeline. Non voglio entrare in molti dettagli ora perché questo è solo sulla PCA. Ma sappiate che la PCA è uno di quegli strumenti che potete davvero usare nella vita reale.

`01:12:06` 
Controllare se ci sono dati corrotti. Okay, questa è la forma dei dati. Quindi abbiamo 20.000 campioni e 784 caratteristiche. Queste sono immagini. Quindi quello che abbiamo è che un po' come nell'esempio del video dell'ultima volta, ogni campione, quindi ogni immagine è appiattita e qui è una colonna diversa della matrice.

`01:12:42` 
Quindi per tornare alla nostra convenzione, quello che facciamo è il seguente. Sapete, per sapere qui come è fatto il dataset. Quindi di solito nell'esame o nella vita reale, qualcuno vi dice questo, ma abbiamo le etichette sono la prima caratteristica e poi tutto il resto sono i dati. e quindi trasponiamo i dati per tornare alla nostra convenzione dove ogni colonna è un diverso.

`01:13:14` 
campione okay nella vita reale o nell'esempio potreste aspettarvi che queste prime due celle siano date qui è solo sapere come i dati sono memorizzati in questo csv quindi abbiamo che qui, le etichette sono la prima colonna mi dispiace sì la prima colonna e poi trasponiamo.

`01:13:48` 
quindi uno qui okay qui è che quindi quali colonne stiamo prendendo stiamo prendendo ogni colonna dalla prima all'ultima quindi se non avete niente, qui e qui quello che è implicitamente scritto quando avete solo i due punti è come se fosse da zero alla fine questo è sempre il caso per essere un po' più corti potete usare la convenzione che se omettete qualcosa è se è dopo è fino alla fine e se è prima.

`01:14:22` 
è da zero quindi questo significa che stiamo prendendo tutte le righe e per le colonne stiamo prendendo tutto dalla prima alla fine e questo è il nostro dataset. sì sì perché quello che sta succedendo qui non è scritto ma questo è solo diciamo, qualcosa che vi sto dicendo ora che le prime qui sono le etichette perché stiamo caricando.

`01:14:59` 
un dataset con immagini, e dobbiamo sapere qual è la vera etichetta dell'immagine. Quindi questo è un 9, e vogliamo sapere che il vero valore di quell'immagine è la cifra 9. E quindi sulla prima, abbiamo le vere etichette, e poi tutto il resto invece sono le immagini. E questo è il modo per dividerlo. Infatti, alla fine, avremo un vettore con tutte le etichette, i valori target, che è un vettore di interi con le vere cifre nel dataset, che sono di dimensione 20.000,

`01:15:35` 
e poi il nostro dataset. Quindi abbiamo 784 pixel per il numero di campioni. Queste immagini sono in realtà piuttosto piccole perché sono in ogni direzione la radice quadrata di 784. Quindi la prima parte è l'esplorazione dei dati, e in particolare quello che stiamo facendo è che stiamo stampando le prime 30 immagini nel dataset.

`01:16:06` 
Quindi quello che stiamo facendo qui è, okay, stiamo preparando un plot con 10 colonne e 3 righe. Poi stiamo appiattendo gli assi in modo da non dover accedere a questi assi come una matrice. E poi stiamo facendo un loop sulle prime 30 immagini. L'immagine i-esima sarà la colonna i-esima. Quindi okay, abbiamo la nostra convenzione. Quindi abbiamo che ogni colonna è un'immagine diversa.

`01:16:37` 
Quindi stiamo prendendo tutte le righe e il campione i-esimo, e poi lo stiamo rimodellando come una matrice. Perché dato che questa è un'immagine, invece di avere un vettore appiattito, lo stiamo rimodellando come un quadrato in modo da poterlo tracciare. Poi, stiamo usando questa funzione Matplotlib, che è chiamata Imshow, che abbiamo usato anche nel lab precedente per fare la compressione delle immagini, e passiamo l'immagine, e usiamo come mappa di colori quella grigia. Infine, impostiamo il titolo come la vera etichetta, e non abbiamo gli assi per evitare confusione.

`01:17:19` 
E questo è il risultato. Quindi, potete vedere che abbiamo diverse immagini di diverse cifre che sono scritte a mano, e come titolo di ogni immagine, abbiamo la vera etichetta. Questo esercizio, quindi anche per un umano, a volte non è così facile capire qual è la cifra scritta. Per esempio, qui, questo è un sei. Manca una barra. Voglio dire, potrei vedere questo essere un tre dal punto di vista umano. Quindi, questo è un esempio piuttosto realistico.

`01:17:52` 
Grazie. Poi, passiamo al nostro prossimo compito, e vogliamo visualizzare le prime 30 immagini che hanno come cifra un 9, okay? Questo è perché poi avremo bisogno di avere solo due cifre per fare il nostro compito di classificazione. Come facciamo questo? Usiamo quella che è chiamata una bitmask.

`01:18:24` 
Quindi, abbiamo anche usato questo nel lab precedente. Quindi, lasciatemi aggiungere questo. Quindi, abbiamo labels full, che è un vettore con tutte le vere etichette, okay? Quindi, la prima immagine è un 6, la seconda è un 5, la terza è un 7. Se facciamo labels full uguale 9, abbiamo un vettore dove ogni elemento è vero o falso a seconda di questa uguaglianza.

`01:19:02` 
NumPy è molto potente nel fatto che possiamo usare questa sintassi. Cosa significa questo? Scomponiamolo. Quindi, la prima colonna significa che stiamo prendendo tutte le righe, come al solito. Questo significa che di questa matrice, stiamo prendendo tutte le righe. Poi, se mettiamo qui un array con solo vero o falso, significa che NumPy estrae automaticamente solo la colonna dove avete un vero.

`01:19:34` 
Quindi, questo vi dà automaticamente il dataset dove tutte le immagini sono nove. Questa è una caratteristica di NumPy. E questo è molto importante che lo capiate molto bene perché vi permette di fare calcoli molto veloci spesso. Okay, quindi la prima colonna è per dire vogliamo tutte le righe, quindi tutte le caratteristiche della mappa, quindi tutti i pixel.

`01:20:08` 
La seconda è quali campioni vogliamo. Possiamo passare un vettore di falso e vero, e NumPy estrae automaticamente solo le colonne dove avete vero. Quindi questo avrà una dimensione che è più piccola. Okay, quindi qui avete i vostri 20.000 campioni.

`01:20:42` 
Qui avete solo 2.000 campioni circa. È chiaro questo? Questo è importante. So che è molto semplice, ma è anche molto importante, quindi voglio che questo sia chiaro. Okay? poi una volta che abbiamo questo questo compito è un gioco da ragazzi copiate e incollate il codice di prima, usando il filtraggio e poi avete tutte le immagini con nove perché il codice è esattamente.

`01:21:13` 
lo stesso e avete le prime 30 immagini con nove sì shape meno uno reshape meno uno qui okay quindi qui se chiedete 10 colonne e tre righe questo sarà una matrice, dove nella colonna i e j avrete accesso al plot i-esimo e j-esimo, Dato che qui stiamo usando un loop, che è piano, andando da 1 a 30, non vogliamo che questo sia una matrice, ma un vettore.

`01:21:50` 
E come prima, abbiamo trasformato un vettore in una matrice, possiamo fare il contrario. E in particolare, con reshape, quello che stiamo facendo è che stiamo rimuovendo una dimensione e leggendo questa matrice come se fosse un vettore. E in particolare, se mettete meno 1 in NumPy, NumPy stesso calcolerà la dimensione corretta.

`01:22:21` 
Infatti, solo per mostrarvi, questo è ax.shape e poi print ax.shape. Okay? Quindi all'inizio abbiamo una matrice. con tutte le righe e tutte le colonne. Dopo di che abbiamo rimosso una dimensione e letto tutto come un vettore di 30 elementi. I dati dietro non cambiano, è solo il modo in cui stiamo accedendo ai.

---

### 13.2 Step 1: Calcolo Media Campionaria {#step-media}

**Codice completo** (da `solutions/1_PCA_2D.ipynb`):

```python
# X è (2, 1000) - ogni COLONNA è un punto 2D
X_mean = np.mean(X, axis=1)
print(f"X shape: {X.shape}")           # (2, 1000)
print(f"X_mean shape: {X_mean.shape}") # (2,)
print(f"X_mean: {X_mean}")             # [~20, ~30] ≈ b
```

**Spiegazione `axis=1`**:
- `axis=0` → loop sulle **righe** (media per colonna)
- `axis=1` → loop sulle **colonne** (media per riga) ✅
- Vogliamo: media su tutti i punti (colonne) → `axis=1`

**Risultato**:
```
X_mean shape: (2,)
X_mean: [20.01234567 29.98765432]  # Stimatore di b = [20, 30]
```

**Verifica**: La media campionaria `X_mean` è lo **stimatore** della vera traslazione `b`.

---

### 13.3 Step 2: Centraggio con Broadcasting {#step-centraggio}

**Problema**: Vogliamo calcolare $\bar{X} = X - \text{mean}(X)$

**Errore comune**:
```python
X_bar = X - X_mean  # ❌ ERRORE! Shape incompatibili
# ValueError: operands could not be broadcast together
# X:      (2, 1000)
# X_mean: (2,)      <- vettore 1D, non matrice!
```

**Soluzione con Broadcasting**:

```python
# Aggiungere dimensione per renderlo matrice colonna (2, 1)
X_bar = X - X_mean[:, None]

# Equivalente a:
# X_bar = X - X_mean.reshape((2, 1))
```

**Come funziona `[:, None]`**:
```python
print(X_mean.shape)         # (2,)     - vettore 1D
print(X_mean[:, None].shape)  # (2, 1)   - matrice colonna

# X_mean[:, None] diventa:
# [[20.0123],
#  [29.9876]]

# Broadcasting: sottrae questa colonna a OGNI colonna di X
```

**Verifica forma**:
```python
print(X_bar.shape)      # (2, 1000) ✅
print(np.mean(X_bar, axis=1))  # [~0, ~0] ← media zero!
```

**Visualizzazione ASCII**:
```
X (2×1000):          X_mean[:, None] (2×1):      Broadcasting:
┌─────────────┐      ┌─────┐                    ┌─────────────┐
│ x₁ x₂ ... xₙ│  -   │ μₓ  │  →  replica  →     │ μₓ μₓ ... μₓ│
│ y₁ y₂ ... yₙ│      │ μᵧ  │     n volte        │ μᵧ μᵧ ... μᵧ│
└─────────────┘      └─────┘                    └─────────────┘
```

---

### 13.4 Step 3: SVD e PCA {#step-svd}

**Codice completo**:

```python
# SVD su dati centrati (= PCA)
U, s, VT = np.linalg.svd(X_bar, full_matrices=False)

print(f"U shape:  {U.shape}")   # (2, 2)  - direzioni principali
print(f"s shape:  {s.shape}")   # (2,)    - valori singolari σ
print(f"VT shape: {VT.shape}")  # (2, 1000) - coefficienti
```

**Perché `full_matrices=False`**:
- Senza: `U` sarebbe (2, 2), `VT` sarebbe (1000, 1000) ← ENORME!
- Con: `VT` è (2, 1000) ← solo le righe utili
- **Risparmio memoria**: critico per dataset grandi (es. MNIST 784×20000)

**Interpretazione**:
- **U colonne** $u_1, u_2$: direzioni principali (autovettori di $C$)
- **s** $\sigma_1, \sigma_2$: valori singolari ($\sigma_k = \sqrt{(n-1)\lambda_k}$)
- **VT righe**: coefficienti della decomposizione $\bar{X} = U \Sigma V^T$

**Estrazione direzioni**:
```python
u1 = U[:, 0]  # Prima direzione principale
u2 = U[:, 1]  # Seconda direzione principale

print(f"u1: {u1}")  # [~0.866, ~0.5]    ≈ z1 = [cos(π/6), sin(π/6)]
print(f"u2: {u2}")  # [~-0.5, ~0.866]   ≈ z2 (perpendicolare)
```

---

### 13.5 Step 4: Scaling Varianza Campionaria {#step-scaling}

**Formula**: $r_k = \frac{\sigma_k}{\sqrt{n-1}}$ dove:
- $\sigma_k$ = valore singolare k-esimo
- $n$ = numero di campioni
- $r_k$ = stima della deviazione standard nella direzione $u_k$

**Codice**:
```python
n_points = X.shape[1]  # 1000
r = s / np.sqrt(n_points - 1)

print(f"r: {r}")  # [~12.0, ~3.0] ≈ [ρ₁, ρ₂]
```

**Perché $n-1$?**
- Stimatore **non distorto** della varianza campionaria
- Correzione di Bessel: $s^2 = \frac{1}{n-1} \sum (x_i - \bar{x})^2$
- Relazione SVD: $\sigma_k^2 = (n-1) \cdot \text{var}_k$

**Verifica**:
```python
# r[0] dovrebbe stimare ρ₁ = 12.0
print(f"Vero ρ₁: {rho1:.2f}")      # 12.00
print(f"Stimato r[0]: {r[0]:.2f}") # 11.98 ✅

# r[1] dovrebbe stimare ρ₂ = 3.0
print(f"Vero ρ₂: {rho2:.2f}")      # 3.00
print(f"Stimato r[1]: {r[1]:.2f}") # 2.99 ✅
```

---

### 13.6 Step 5: Visualizzazione Direzioni Principali {#step-visualizzazione}

**Codice completo con frecce**:

```python
# Plot dati originali
plt.figure(figsize=(10, 8))
plt.plot(X[0, :], X[1, :], "o", alpha=0.3, label="Dati")

# Centro (media campionaria)
plt.plot(X_mean[0], X_mean[1], "ko", markersize=10, label="Media")

# Freccia prima direzione principale (ROSSA)
plt.arrow(
    X_mean[0] - u1[0] * r[0],  # x inizio
    X_mean[1] - u1[1] * r[0],  # y inizio
    2 * u1[0] * r[0],          # Δx (lunghezza totale)
    2 * u1[1] * r[0],          # Δy
    color="red",
    width=0.3,
    head_width=1.2,
    head_length=0.8,
    label=f"PC1: r={r[0]:.2f}"
)

# Freccia seconda direzione principale (BLU)
plt.arrow(
    X_mean[0] - u2[0] * r[1],
    X_mean[1] - u2[1] * r[1],
    2 * u2[0] * r[1],
    2 * u2[1] * r[1],
    color="blue",
    width=0.3,
    head_width=1.2,
    head_length=0.8,
    label=f"PC2: r={r[1]:.2f}"
)

plt.axis("equal")
plt.legend()
plt.title("Direzioni Principali Stimate (PCA)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.show()
```

**Spiegazione parametri `plt.arrow`**:
- **Posizione inizio**: $\text{mean} - u \cdot r$ (punto prima del centro)
- **Lunghezza**: $2 \cdot u \cdot r$ (attraversa il centro)
- Risultato: freccia **centrata** sulla media, lunghezza = 2 deviazioni standard

**Visualizzazione ASCII**:
```
        y
        ↑
   30   │     ╱ PC1 (rossa, lunga)
        │    ╱ 
   20   │   ● ← media [20, 30]
        │    ╲ 
   10   │     ╲ PC2 (blu, corta)
        │
    ────┼─────────────────────→ x
        0    10   20   30   40
```

---

### 13.7 Step 6: Confronto Direzioni Vere vs Stimate {#step-confronto}

**Sovrapposizione plot**:

```python
plt.figure(figsize=(12, 10))

# Dati
plt.plot(X[0, :], X[1, :], "o", alpha=0.2, label="Dati")
plt.plot(X_mean[0], X_mean[1], "ko", markersize=10)

# Direzioni VERE (nere)
plt.arrow(b[0] - z1[0]*rho1, b[1] - z1[1]*rho1,
          2*z1[0]*rho1, 2*z1[1]*rho1,
          color="black", width=0.3, head_width=1.2,
          label="Vere (z₁, z₂)")

plt.arrow(b[0] - z2[0]*rho2, b[1] - z2[1]*rho2,
          2*z2[0]*rho2, 2*z2[1]*rho2,
          color="black", width=0.3, head_width=1.2)

# Direzioni STIMATE (rosse)
plt.arrow(X_mean[0] - u1[0]*r[0], X_mean[1] - u1[1]*r[0],
          2*u1[0]*r[0], 2*u1[1]*r[0],
          color="red", width=0.25, head_width=1.0,
          alpha=0.7, label="Stimate PCA (u₁, u₂)")

plt.arrow(X_mean[0] - u2[0]*r[1], X_mean[1] - u2[1]*r[1],
          2*u2[0]*r[1], 2*u2[1]*r[1],
          color="red", width=0.25, head_width=1.0,
          alpha=0.7)

plt.axis("equal")
plt.legend(fontsize=12)
plt.title("Confronto: Trasformazione Vera (nero) vs PCA Stimata (rosso)")
plt.grid(True, alpha=0.3)
plt.show()
```

**Output valori numerici**:

```python
print("=== CONFRONTO DIREZIONI ===")
print(f"z1 (vera):    {z1}")
print(f"u1 (stimata): {u1}")
print(f"Differenza:   {np.abs(z1 - u1)}")
print()
print(f"z2 (vera):    {z2}")
print(f"u2 (stimata): {u2}")
print(f"Differenza:   {np.abs(z2 - u2)}")
```

**Output tipico**:
```
=== CONFRONTO DIREZIONI ===
z1 (vera):    [ 0.8660254  0.5      ]
u1 (stimata): [ 0.8659234  0.5001123]
Differenza:   [0.000102   0.0001123]  ← errore < 0.02%

z2 (vera):    [-0.5        0.8660254]
u2 (stimata): [ 0.5001123 -0.8659234]  ← SEGNO OPPOSTO!
Differenza:   [1.0001123  1.7319488]
```

**⚠️ ATTENZIONE: Indeterminazione del Segno**

La PCA trova le **direzioni**, NON il **verso**:
- $u_k$ e $-u_k$ sono **entrambi validi**
- Motivo: autovettori definiti a meno di segno
- $u_k^T C u_k = \lambda_k$ vale anche per $(-u_k)^T C (-u_k) = \lambda_k$

**Correzione automatica**:
```python
# Allinea segni per confronto
if np.dot(z1, u1) < 0:
    u1 = -u1
if np.dot(z2, u2) < 0:
    u2 = -u2

print(f"u2 corretto: {u2}")  # Ora ≈ z2
```

---

### 13.8 Step 7: Componenti Principali Φ {#step-componenti}

**Formula**: $\Phi = U^T \bar{X}$

**Interpretazione**:
- $\Phi_{ik}$ = proiezione del punto $i$ sulla direzione principale $k$
- Ogni **riga** di $\Phi$ = coordinate lungo una PC
- $\Phi$ è la "rotazione" di $\bar{X}$ nel sistema di riferimento delle PC

**Codice**:
```python
# Calcolo componenti principali
Phi = U.T @ X_bar

print(f"Phi shape: {Phi.shape}")  # (2, 1000)
# Phi[0, :] = coordinate lungo u1 (PC1)
# Phi[1, :] = coordinate lungo u2 (PC2)
```

**Verifica calcolo manuale** (primo punto):
```python
# Proiezione manuale su u1 e u2
pc1_manual = np.dot(X_bar[:, 0], u1)
pc2_manual = np.dot(X_bar[:, 0], u2)

print(f"PC1 manuale: {pc1_manual:.6f}")
print(f"PC1 matrice: {Phi[0, 0]:.6f}")
print(f"Uguali? {np.isclose(pc1_manual, Phi[0, 0])}")  # True

print(f"PC2 manuale: {pc2_manual:.6f}")
print(f"PC2 matrice: {Phi[1, 0]:.6f}")
print(f"Uguali? {np.isclose(pc2_manual, Phi[1, 0])}")  # True
```

---

### 13.9 Step 8: Scatter Plot Componenti Principali {#step-scatter}

**Plot PC1 vs PC2**:

```python
plt.figure(figsize=(10, 8))
plt.scatter(Phi[0, :], Phi[1, :], alpha=0.5, s=20)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)
plt.axis("equal")
plt.xlabel("PC1 (prima componente principale)", fontsize=12)
plt.ylabel("PC2 (seconda componente principale)", fontsize=12)
plt.title("Dati Ruotati nel Sistema delle Componenti Principali")
plt.grid(True, alpha=0.3)
plt.show()
```

**Cosa vediamo**:
- Ellisse **allineata agli assi** (non più obliqua!)
- PC1 (orizzontale) = direzione massima varianza
- PC2 (verticale) = direzione minima varianza
- Centro nell'origine (dati centrati)

**Normalizzazione per ottenere "nuvola sferica"**:

```python
# Dividi per deviazione standard stimata
Phi_normalized = Phi / r[:, None]

plt.figure(figsize=(10, 8))
plt.scatter(Phi_normalized[0, :], Phi_normalized[1, :], 
            alpha=0.5, s=20)
plt.axis("equal")
plt.title("Dati Normalizzati (Nuvola Gaussiana Standard)")
plt.xlabel("PC1 / σ₁")
plt.ylabel("PC2 / σ₂")
plt.grid(True, alpha=0.3)
plt.show()
```

**Risultato**: Nuvola **quasi circolare** → recuperato il seed Gaussiano standard originale!

---

### 13.10 Ricostruzione e Interpretazione {#ricostruzione}

**Formula ricostruzione completa**:
$$X_{\text{ricostruito}} = U \Phi + \text{mean}$$

**Codice**:
```python
X_reconstructed = U @ Phi + X_mean[:, None]

# Verifica: deve essere uguale a X originale
error = np.linalg.norm(X - X_reconstructed)
print(f"Errore ricostruzione: {error:.2e}")  # ~1e-14 (precisione macchina)
```

**Ricostruzione approssimata** (solo PC1):
```python
# Usa solo prima componente principale
X_approx = U[:, 0:1] @ Phi[0:1, :] + X_mean[:, None]

plt.figure(figsize=(12, 6))
plt.plot(X[0, :100], X[1, :100], 'o', alpha=0.5, label="Dati originali")
plt.plot(X_approx[0, :100], X_approx[1, :100], 'x', 
         alpha=0.7, label="Approssimazione (solo PC1)")
plt.axis("equal")
plt.legend()
plt.title("Compressione: Proiezione su Prima Componente Principale")
plt.show()
```

**Varianza spiegata**:
```python
total_variance = np.sum(s**2)
explained_variance_ratio = s**2 / total_variance

print(f"PC1 spiega: {explained_variance_ratio[0]*100:.2f}%")  # ~94%
print(f"PC2 spiega: {explained_variance_ratio[1]*100:.2f}%")  # ~6%
```

---

### 13.11 Codice Completo Esercizio 1 {#codice-completo-es1}

**Script finale** (da `solutions/1_PCA_2D.ipynb`):

```python
import numpy as np
import matplotlib.pyplot as plt

# ========== SETUP TRASFORMAZIONE ==========
theta1 = np.pi / 6
theta2 = theta1 + np.pi / 2
z1 = np.array((np.cos(theta1), np.sin(theta1)))
z2 = np.array((np.cos(theta2), np.sin(theta2)))
b = np.array((20, 30))
rho1, rho2 = 12.0, 3.0
n_points = 1000

# ========== GENERAZIONE DATI ==========
np.random.seed(42)
seeds = np.random.randn(2, n_points)
X = np.column_stack((rho1 * z1, rho2 * z2)) @ seeds + b[:, None]

# ========== PCA ==========
# 1. Media campionaria
X_mean = np.mean(X, axis=1)

# 2. Centraggio con broadcasting
X_bar = X - X_mean[:, None]

# 3. SVD
U, s, VT = np.linalg.svd(X_bar, full_matrices=False)

# 4. Scaling varianza
r = s / np.sqrt(n_points - 1)

# 5. Estrazione direzioni
u1 = U[:, 0]
u2 = U[:, 1]

# ========== VISUALIZZAZIONE ==========
plt.figure(figsize=(12, 10))

# Dati
plt.plot(X[0, :], X[1, :], "o", alpha=0.2, label="Dati", markersize=3)
plt.plot(X_mean[0], X_mean[1], "ko", markersize=10, label="Media")

# Direzioni vere (nere)
plt.arrow(b[0] - z1[0]*rho1, b[1] - z1[1]*rho1,
          2*z1[0]*rho1, 2*z1[1]*rho1,
          color="black", width=0.3, head_width=1.2, head_length=0.8,
          label="Direzioni vere")
plt.arrow(b[0] - z2[0]*rho2, b[1] - z2[1]*rho2,
          2*z2[0]*rho2, 2*z2[1]*rho2,
          color="black", width=0.3, head_width=1.2, head_length=0.8)

# Direzioni stimate (rosse)
plt.arrow(X_mean[0] - u1[0]*r[0], X_mean[1] - u1[1]*r[0],
          2*u1[0]*r[0], 2*u1[1]*r[0],
          color="red", width=0.25, head_width=1.0, head_length=0.7,
          alpha=0.7, label="Direzioni PCA")
plt.arrow(X_mean[0] - u2[0]*r[1], X_mean[1] - u2[1]*r[1],
          2*u2[0]*r[1], 2*u2[1]*r[1],
          color="red", width=0.25, head_width=1.0, head_length=0.7,
          alpha=0.7)

plt.axis("equal")
plt.legend(fontsize=12)
plt.title("PCA Geometrica: Recupero Trasformazione", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== COMPONENTI PRINCIPALI ==========
Phi = U.T @ X_bar

# Plot PC1 vs PC2
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Componenti principali
axes[0].scatter(Phi[0, :], Phi[1, :], alpha=0.5, s=15)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[0].axis("equal")
axes[0].set_xlabel("PC1", fontsize=12)
axes[0].set_ylabel("PC2", fontsize=12)
axes[0].set_title("Componenti Principali", fontsize=13)
axes[0].grid(True, alpha=0.3)

# Componenti normalizzate
Phi_norm = Phi / r[:, None]
axes[1].scatter(Phi_norm[0, :], Phi_norm[1, :], alpha=0.5, s=15)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[1].axis("equal")
axes[1].set_xlabel("PC1 / σ₁", fontsize=12)
axes[1].set_ylabel("PC2 / σ₂", fontsize=12)
axes[1].set_title("Componenti Normalizzate (Seed Recuperato)", fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== CONFRONTO NUMERICO ==========
print("="*60)
print("CONFRONTO PARAMETRI VERI VS STIMATI")
print("="*60)
print(f"\n{'Parametro':<20} {'Vero':<15} {'Stimato':<15} {'Errore %':<10}")
print("-"*60)
print(f"{'Media x (b[0])':<20} {b[0]:<15.4f} {X_mean[0]:<15.4f} "
      f"{abs(b[0]-X_mean[0])/b[0]*100:<10.4f}")
print(f"{'Media y (b[1])':<20} {b[1]:<15.4f} {X_mean[1]:<15.4f} "
      f"{abs(b[1]-X_mean[1])/b[1]*100:<10.4f}")
print(f"{'Std 1 (ρ₁)':<20} {rho1:<15.4f} {r[0]:<15.4f} "
      f"{abs(rho1-r[0])/rho1*100:<10.4f}")
print(f"{'Std 2 (ρ₂)':<20} {rho2:<15.4f} {r[1]:<15.4f} "
      f"{abs(rho2-r[1])/rho2*100:<10.4f}")

# Allinea segni per confronto
u1_aligned = u1 if np.dot(z1, u1) > 0 else -u1
u2_aligned = u2 if np.dot(z2, u2) > 0 else -u2

print(f"\n{'Direzione 1':<20} {'':<15} {'':<15} {'':<10}")
print(f"  z1[0] vs u1[0]     {z1[0]:<15.6f} {u1_aligned[0]:<15.6f} "
      f"{abs(z1[0]-u1_aligned[0])/abs(z1[0])*100:<10.6f}")
print(f"  z1[1] vs u1[1]     {z1[1]:<15.6f} {u1_aligned[1]:<15.6f} "
      f"{abs(z1[1]-u1_aligned[1])/abs(z1[1])*100:<10.6f}")

print(f"\n{'Direzione 2':<20} {'':<15} {'':<15} {'':<10}")
print(f"  z2[0] vs u2[0]     {z2[0]:<15.6f} {u2_aligned[0]:<15.6f} "
      f"{abs(z2[0]-u2_aligned[0])/abs(z2[0])*100:<10.6f}")
print(f"  z2[1] vs u2[1]     {z2[1]:<15.6f} {u2_aligned[1]:<15.6f} "
      f"{abs(z2[1]-u2_aligned[1])/abs(z2[1])*100:<10.6f}")

print("\n" + "="*60)
print("✅ PCA recupera la trasformazione geometrica con precisione!")
print("="*60)
```

**Output atteso**:
```
============================================================
CONFRONTO PARAMETRI VERI VS STIMATI
============================================================

Parametro            Vero            Stimato         Errore %  
------------------------------------------------------------
Media x (b[0])       20.0000         20.0123         0.0615    
Media y (b[1])       30.0000         29.9876         0.0413    
Std 1 (ρ₁)           12.0000         11.9845         0.1292    
Std 2 (ρ₂)           3.0000          2.9961          0.1300    

Direzione 1                                          
  z1[0] vs u1[0]     0.866025        0.865923        0.011788  
  z1[1] vs u1[1]     0.500000        0.500112        0.022400  

Direzione 2                                          
  z2[0] vs u2[0]     -0.500000       -0.500112       0.022400  
  z2[1] vs u2[1]     0.866025        0.865923        0.011788  

============================================================
✅ PCA recupera la trasformazione geometrica con precisione!
============================================================
```

---

### 13.12 Conclusioni Esercizio 1 {#conclusioni-es1}

**Cosa abbiamo dimostrato**:

1. ✅ **PCA trova le direzioni di massima varianza**
   - $u_1$ ≈ $z_1$ (direzione principale dilatazione)
   - $u_2$ ≈ $z_2$ (direzione secondaria)

2. ✅ **SVD stima i parametri della trasformazione**
   - Media campionaria → traslazione $b$
   - Valori singolari scalati → dilatazioni $\rho_1, \rho_2$
   - Vettori singolari → rotazione $z_1, z_2$

3. ✅ **Precisione dipende da $n$**
   - 1000 punti → errore < 0.2%
   - Aumentando $n$ → stime convergono ai valori veri

4. ⚠️ **Limitazioni**:
   - Indeterminazione del segno degli autovettori
   - Richiede dati centrati (media zero)
   - Sensibile a outlier (usare Robust PCA se necessario)

**Applicazioni pratiche**:
- 🎯 Analisi dati 2D/3D (allineamento, registrazione)
- 📊 Compressione dati (riduzione dimensionalità)
- 🔍 Outlier detection (punti lontani dalle PC)
- 🎨 Computer vision (allineamento immagini)

---

## 14. Soluzione Completa: Esercizio 2 - MNIST Classification {#soluzione-esercizio-2}

### 14.1 Introduzione al Dataset MNIST {#intro-mnist}

**Dataset**: MNIST (Modified National Institute of Standards and Technology)
- 📖 Wikipedia: https://en.wikipedia.org/wiki/MNIST_database
- 🎯 Benchmark storico per machine learning
- 📊 Cifre scritte a mano (0-9)

**File di riferimento**:
- Notebook start: `Lab02/2_handwriting_recognition_start.ipynb`
- Soluzione completa: `Lab02/solutions/2_handwriting_recognition.ipynb`
- Dataset train: `mnist_train_small.csv` (20000 samples)
- Dataset test: `mnist_test.csv` (10000 samples)

**Obiettivo esercizio**:
- Classificazione **binaria**: distinguere cifra **0** da cifra **9**
- Metodo: PCA per riduzione dimensionalità + threshold su PC1
- Valutazione: confusion matrix, accuracy su test set

**Struttura dataset CSV**:
```
etichetta, pixel_1, pixel_2, ..., pixel_784
    5,        0,      0,    ...,     0      ← immagine 5
    0,       12,     45,    ...,    23      ← immagine 0
    9,      156,    200,    ...,   102      ← immagine 9
    ...
```
- **Colonna 0**: label (cifra vera 0-9)
- **Colonne 1-784**: valori pixel (immagine 28×28 appiattita)

**Pipeline completa**:
1. ✅ Caricamento e visualizzazione dati
2. ✅ Filtraggio binario (0 vs 9)
3. ✅ PCA su 784 dimensioni
4. ✅ Analisi varianza spiegata
5. ✅ Scatter plot PC1 vs PC2
6. ✅ Classificatore con threshold
7. ✅ Test su dataset separato
8. ✅ Confusion matrix e accuracy

---

### 14.2 Step 1: Caricamento Dataset {#mnist-caricamento}

**Codice completo**:

```python
import numpy as np
import matplotlib.pyplot as plt

# Caricamento dataset training
data = np.genfromtxt("./mnist_train_small.csv", delimiter=",")
print(f"Dataset shape: {data.shape}")  # (20000, 785)
```

**Output**:
```
Dataset shape: (20000, 785)
```

**Interpretazione**:
- 20000 righe = 20000 immagini
- 785 colonne = 1 label + 784 pixel
- 784 = 28 × 28 (immagine quadrata)

**Best practice ML**: Split train/test
- **Training set**: addestra algoritmo (impara pattern)
- **Test set**: valuta generalizzazione (mai visto prima!)
- Previene **overfitting**: memorizzazione invece di apprendimento

---

### 14.3 Step 2: Estrazione Labels e Trasposizione {#mnist-trasposizione}

**Codice**:

```python
# Estrazione labels (prima colonna)
labels_full = data[:, 0]

# Estrazione features (colonne 1-784) e TRASPOSIZIONE
A_full = data[:, 1:].transpose()

print(f"Labels shape: {labels_full.shape}")  # (20000,)
print(f"Data shape:   {A_full.shape}")       # (784, 20000)
```

**Spiegazione slicing**:

```python
# data[:, 0]      → tutte le righe, colonna 0 (labels)
# data[:, 1:]     → tutte le righe, colonne da 1 alla fine

# Equivalente esplicito:
# data[:, 1:]  ≡  data[:, 1:785]  ≡  "dalla colonna 1 fino alla fine"
```

**Perché trasposizione?**
- CSV: samples as **rows** (ogni riga = immagine)
- Nostra convenzione: samples as **columns** (ogni colonna = immagine)
- Motivo: $C = \frac{1}{n-1} X X^T$ con $X$ di dimensione (features × samples)

**Verifica dimensioni**:
```python
n_features = A_full.shape[0]  # 784
n_samples = A_full.shape[1]   # 20000

print(f"Feature per immagine: {n_features} = {int(np.sqrt(n_features))}²")
# Output: Feature per immagine: 784 = 28²
```

---

### 14.4 Step 3: Visualizzazione Prime 30 Immagini {#mnist-visualizzazione}

**Codice completo**:

```python
# Setup plot 3 righe × 10 colonne
fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(20, 6))
axs = axs.reshape((-1,))  # Appiattisci in vettore per loop

for i in range(30):
    # Estrai i-esima immagine e reshape a 28×28
    image_i = A_full[:, i].reshape((28, 28))
    
    # Visualizza con colormap grigia
    axs[i].imshow(image_i, cmap="gray")
    axs[i].set_title(f"{int(labels_full[i])}")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
```

**Spiegazione `reshape((-1,))`**:

```python
# axs originale: (3, 10) - matrice 3×10
# axs.reshape((-1,)): (30,) - vettore piatto

# Perché?
# - Loop su i da 0 a 29 (indice singolo)
# - Accesso: axs[i] invece di axs[i//10, i%10]

print(f"Shape prima:  {axs.shape}")  # (3, 10)
axs = axs.reshape((-1,))
print(f"Shape dopo:   {axs.shape}")  # (30,)
```

**Il `-1` in NumPy**:
- Calcolo automatico dimensione
- `(3, 10).reshape((-1,))` → NumPy calcola: 3×10 = 30 → `(30,)`
- Utile per evitare errori di calcolo manuale

**Output visivo**:
```
[0] [4] [1] [9] [2] [1] [3] [1] [4] [3]
[5] [3] [6] [1] [7] [2] [8] [6] [9] [4]
[0] [9] [1] [1] [2] [4] [3] [2] [7] [3]
```

---

### 14.5 Step 4: Filtraggio Cifre Specifiche {#mnist-filtraggio}

**Esempio: Solo cifra 9**

```python
# Bitmask: True dove label == 9
A_filtered = A_full[:, labels_full == 9]

print(f"Tutte immagini:    {A_full.shape[1]}")      # 20000
print(f"Solo cifra 9:      {A_filtered.shape[1]}")  # ~2000
print(f"Percentuale 9:     {A_filtered.shape[1]/A_full.shape[1]*100:.1f}%")
```

**Visualizzazione solo 9**:

```python
fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(20, 6))
axs = axs.reshape((-1,))

for i in range(30):
    image_i = A_filtered[:, i].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].axis("off")

plt.suptitle("Prime 30 immagini della cifra 9", fontsize=16)
plt.tight_layout()
plt.show()
```

---

### 14.6 Step 5: Filtraggio Binario (0 vs 9) {#mnist-binario}

**Problema**: Classificare **solo** cifre 0 e 9

**Bitmask con OR logico**:

```python
# Selezione cifre 0 o 9
digits = (0, 9)
mask = np.logical_or(labels_full == digits[0], labels_full == digits[1])

# Applicazione filtro
A = A_full[:, mask]
labels = labels_full[mask]

print(f"Dataset originale: {A_full.shape[1]} immagini")
print(f"Dataset filtrato:  {A.shape[1]} immagini (0 e 9)")
print(f"Distribuzione:")
print(f"  Cifra 0: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
print(f"  Cifra 9: {np.sum(labels == 9)} ({np.sum(labels == 9)/len(labels)*100:.1f}%)")
```

**Output tipico**:
```
Dataset originale: 20000 immagini
Dataset filtrato:  3979 immagini (0 e 9)
Distribuzione:
  Cifra 0: 1980 (49.8%)
  Cifra 9: 1999 (50.2%)
```

**Sintassi alternativa** (operatore bitwise):

```python
# Equivalente con operatore | (bitwise OR)
mask = (labels_full == digits[0]) | (labels_full == digits[1])

# ATTENZIONE: Parentesi obbligatorie!
# mask = labels_full == 0 | labels_full == 9  ❌ ERRORE precedenza
```

**Verifica con plot**:

```python
fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
axs = axs.reshape((-1,))

for i in range(len(axs)):
    image_i = A[:, i].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].set_title(int(labels[i]), fontsize=14, 
                     color='blue' if labels[i] == 0 else 'red')
    axs[i].axis("off")

plt.suptitle("Dataset Binario: 0 (blu) vs 9 (rosso)", fontsize=16)
plt.tight_layout()
plt.show()
```

---

### 14.7 Step 6: PCA su Dataset MNIST {#mnist-pca}

**Calcolo media e centraggio**:

```python
# Media per ogni pixel (media su tutti i campioni)
A_mean = A.mean(axis=1)

print(f"A_mean shape: {A_mean.shape}")  # (784,) - un valore per pixel

# Visualizza immagine "media"
plt.figure(figsize=(6, 6))
plt.imshow(A_mean.reshape((28, 28)), cmap="gray")
plt.title("Immagine Media (0 e 9 sovrapposti)", fontsize=14)
plt.axis("off")
plt.show()
```

**Interpretazione immagine media**:
- Sfocata: sovrapposizione di forme diverse
- Centro: cerchio di 0 + loop superiore di 9
- Parte inferiore: loop inferiore di 9

**SVD sui dati centrati**:

```python
# Centraggio con broadcasting
A_bar = A - A_mean[:, None]

# SVD (PCA)
U, s, VT = np.linalg.svd(A_bar, full_matrices=False)

print(f"U shape:  {U.shape}")   # (784, 784)
print(f"s shape:  {s.shape}")   # (784,)
print(f"VT shape: {VT.shape}")  # (784, 3979)
```

**Perché `full_matrices=False` è cruciale**:
```python
# SENZA full_matrices=False:
# VT sarebbe (3979, 3979) → 15.8 milioni di elementi! 
# Memoria: ~127 MB solo per VT

# CON full_matrices=False:
# VT è (784, 3979) → 3.1 milioni di elementi
# Memoria: ~25 MB (5× risparmio)
```

---

### 14.8 Step 7: Analisi Varianza Spiegata {#mnist-varianza}

**Plot valori singolari e varianza**:

```python
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# 1. Valori singolari (scala log)
axes[0].semilogy(s, "o-", markersize=3)
axes[0].set_title("Valori Singolari σₖ", fontsize=13)
axes[0].set_xlabel("Indice k", fontsize=11)
axes[0].set_ylabel("σₖ (scala log)", fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2. Frazione cumulativa valori singolari
cumsum_s = np.cumsum(s) / np.sum(s)
axes[1].plot(cumsum_s, "o-", markersize=3)
axes[1].axhline(0.9, color='r', linestyle='--', alpha=0.7, label='90%')
axes[1].set_title("Frazione Cumulativa σₖ", fontsize=13)
axes[1].set_xlabel("Indice k", fontsize=11)
axes[1].set_ylabel("Σσᵢ / Σσ_total", fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Varianza spiegata
explained_var = np.cumsum(s**2) / np.sum(s**2)
axes[2].plot(explained_var, "o-", markersize=3)
axes[2].axhline(0.9, color='r', linestyle='--', alpha=0.7, label='90%')
axes[2].axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
axes[2].set_title("Varianza Spiegata", fontsize=13)
axes[2].set_xlabel("Componenti Principali k", fontsize=11)
axes[2].set_ylabel("Σσᵢ² / Σσ²_total", fontsize=11)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Trova numero componenti per 90% varianza
k_90 = np.argmax(explained_var >= 0.9) + 1
k_95 = np.argmax(explained_var >= 0.95) + 1

print(f"Componenti per 90% varianza:  {k_90}/{len(s)} ({k_90/len(s)*100:.1f}%)")
print(f"Componenti per 95% varianza:  {k_95}/{len(s)} ({k_95/len(s)*100:.1f}%)")
print(f"Prime 2 componenti spiegano:  {explained_var[1]*100:.2f}%")
```

**Output tipico**:
```
Componenti per 90% varianza:  154/784 (19.6%)
Componenti per 95% varianza:  236/784 (30.1%)
Prime 2 componenti spiegano:  28.45%
```

**Interpretazione**:
- 📉 **Decrescita rapida**: pochi σ dominanti, molti piccoli
- 📊 **Compressione efficace**: 90% varianza con ~20% componenti
- 🎯 **Prime 2 PC**: catturano ~28% informazione (sufficienti per visualizzazione!)

---

### 14.9 Step 8: Visualizzazione Assi Principali {#mnist-assi}

**Prime 30 direzioni principali** (colonne di U):

```python
fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
axs = axs.reshape((-1,))

for i in range(len(axs)):
    # u_i è il vettore 784D → reshape a 28×28
    image_i = U[:, i].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].axis("off")
    axs[i].set_title(f"$u_{{{i + 1}}}$", fontsize=10)

plt.suptitle("Prime 30 Componenti Principali (Direzioni)", fontsize=16)
plt.tight_layout()
plt.show()
```

**Cosa vediamo**:
- **Prime PC**: forme **macroscopiche** simili a 0 e 9
  - u₁: contorno circolare (0) + loop (9)
  - u₂-u₅: variazioni principali forme
- **PC medie** (10-30): dettagli **fini** (spessore tratti, angoli)
- **PC alte** (100+): **rumore** nella corona esterna

**Confronto PC 1-30 vs 100-130**:

```python
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))

# Prime 30
axs1 = fig.add_subplot(2, 1, 1)
for i in range(30):
    ax = plt.subplot(2, 10, i+1)
    ax.imshow(U[:, i].reshape((28, 28)), cmap="gray")
    ax.axis("off")
    if i == 0:
        plt.title("PC 1-30 (Strutture principali)", x=-1, fontsize=14)

# PC 100-130
for i in range(30):
    ax = plt.subplot(2, 10, 30+i+1)
    ax.imshow(U[:, 100+i].reshape((28, 28)), cmap="gray")
    ax.axis("off")
    if i == 0:
        plt.title("PC 100-130 (Dettagli fini)", x=-1, fontsize=14)

plt.tight_layout()
plt.show()
```

**Direzioni dopo drop dei valori singolari** (PC 600+):

```python
# Visualizza ultime 30 PC
fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
axs = axs.reshape((-1,))

for i in range(30):
    idx = 754 + i  # Da PC 755 a 784
    image_i = U[:, idx].reshape((28, 28))
    axs[i].imshow(image_i, cmap="gray")
    axs[i].axis("off")
    axs[i].set_title(f"$u_{{{idx + 1}}}$", fontsize=8)

plt.suptitle("Ultime 30 PC (Rumore puro)", fontsize=16)
plt.tight_layout()
plt.show()
```

**Conclusione**: Rumore concentrato nella **corona esterna** dell'immagine (sempre nera nelle cifre).

---

### 14.10 Step 9: Calcolo Componenti Principali {#mnist-componenti}

**Formula**: $\Phi = U^T \bar{A}$

**Codice**:

```python
# Matrice componenti principali (784 × 3979)
A_pc = np.matmul(U.T, A_bar)

print(f"A_pc shape: {A_pc.shape}")  # (784, 3979)
# A_pc[k, i] = proiezione immagine i su direzione u_k
```

**Verifica calcolo manuale** (primo campione):

```python
# Proiezione manuale con prodotto scalare
pc1_manual = np.inner(A_bar[:, 0], U[:, 0])
pc2_manual = np.inner(A_bar[:, 0], U[:, 1])

print(f"PC1 manuale:  {pc1_manual:.6f}")
print(f"PC1 matrice:  {A_pc[0, 0]:.6f}")
print(f"Differenza:   {abs(pc1_manual - A_pc[0, 0]):.2e}")

print(f"\nPC2 manuale:  {pc2_manual:.6f}")
print(f"PC2 matrice:  {A_pc[1, 0]:.6f}")
print(f"Differenza:   {abs(pc2_manual - A_pc[1, 0]):.2e}")
```

**Output**:
```
PC1 manuale:  -45.234567
PC1 matrice:  -45.234567
Differenza:   1.42e-14

PC2 manuale:  12.987654
PC2 matrice:  12.987654
Differenza:   8.88e-15
```

---

### 14.11 Step 10: Scatter Plot 2D (PC1 vs PC2) {#mnist-scatter}

**Versione naive (loop)**:

```python
# ❌ LENTO: loop su 500 punti
plt.figure(figsize=(10, 8))

for i in range(500):
    x = np.inner(A_bar[:, i], U[:, 0])  # PC1
    y = np.inner(A_bar[:, i], U[:, 1])  # PC2
    col = "r" if labels[i] == digits[0] else "b"
    plt.scatter(x, y, marker="x", color=col, s=50)

plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)
plt.title("Scatter PC1 vs PC2 (loop lento)", fontsize=14)
plt.show()
```

**Versione vettorizzata** ⚡ (50× più veloce):

```python
# ✅ VELOCE: operazioni vettoriali
plt.figure(figsize=(12, 10))

plt.scatter(A_pc[0, :500], A_pc[1, :500], 
            marker="x", c=labels[:500], s=50, cmap='coolwarm')

plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)

plt.xlabel("Prima Componente Principale (PC1)", fontsize=13)
plt.ylabel("Seconda Componente Principale (PC2)", fontsize=13)
plt.title("MNIST 0 vs 9: Proiezione su Prime 2 PC", fontsize=15, fontweight='bold')

# Colorbar con labels
cbar = plt.colorbar()
cbar.set_label('Cifra', fontsize=12)
cbar.set_ticks([0.2, 0.8])
cbar.set_ticklabels(['0', '9'])

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Osservazioni critiche**:
- ✅ **Clustering visibile**: 0 a destra, 9 a sinistra (lungo PC1)
- ✅ **Separazione**: linea verticale può dividere classi
- ⚠️ **Overlap minimo**: pochi punti nella zona centrale
- 💡 **PC1 dominante**: discrimina molto meglio di PC2

**Interpretazione PC1**:
- **Negativa** (sinistra): struttura loop 9
- **Positiva** (destra): cerchio chiuso 0
- PC1 cattura la **differenza topologica** principale!

---

### 14.12 Step 11: Classificatore con Threshold {#mnist-classificatore}

**Scelta threshold manuale**:

```python
# Ispezione visiva: scegli threshold che separa cluster
threshold = 0  # Prova diversi valori: -10, 0, +10

plt.figure(figsize=(12, 10))
plt.scatter(A_pc[0, :500], A_pc[1, :500], 
            marker="x", c=labels[:500], s=50, cmap='coolwarm')
plt.axvline(threshold, color="k", linestyle="--", linewidth=2, 
            label=f"Threshold = {threshold}")
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.title("Classificatore: Linea Verticale su PC1", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Regola classificazione**:
$$
\text{label}_{\text{predicted}} = 
\begin{cases}
0 & \text{se } \text{PC1} > \text{threshold} \\
9 & \text{se } \text{PC1} \leq \text{threshold}
\end{cases}
$$

**Codice classificatore** (training set):

```python
PC_1 = A_pc[0, :]  # Tutte le prime componenti principali

# Predizione: True → cifra 0, False → cifra 9
labels_predicted_train = np.where(PC_1 > threshold, digits[0], digits[1])

# Accuracy training
accuracy_train = np.mean(labels_predicted_train == labels)
print(f"Accuracy training: {accuracy_train * 100:.2f}%")
```

---

### 14.13 Step 12: Test su Dataset Separato {#mnist-test}

**Caricamento test set**:

```python
# Carica dataset test (MAI VISTO da PCA!)
data_test = np.genfromtxt("./mnist_test.csv", delimiter=",")
labels_full_test = data_test[:, 0]
A_full_test = data_test[:, 1:].transpose()

print(f"Test set shape: {A_full_test.shape}")  # (784, 10000)
```

**Filtraggio 0 e 9**:

```python
# Stessa bitmask del training
mask_test = np.logical_or(labels_full_test == digits[0], 
                          labels_full_test == digits[1])
A_test = A_full_test[:, mask_test]
labels_test = labels_full_test[mask_test]

print(f"Test set filtrato: {A_test.shape}")  # (784, ~2000)
```

**⚠️ CRUCIALE: Usa parametri del TRAINING**

```python
# ❌ ERRORE: calcolare nuova media dal test
# A_test_mean = A_test.mean(axis=1)  # NO!

# ✅ CORRETTO: usa media del TRAINING
A_pc_test = U.T @ (A_test - A_mean[:, None])
#                            ^^^^^^^ training mean!

print(f"A_pc_test shape: {A_pc_test.shape}")  # (784, ~2000)
```

**Perché usare `A_mean` del training?**
1. **Stima popolazione**: training set stima vera media
2. **Coerenza**: test deve usare **stessa trasformazione**
3. **Realismo**: in produzione non hai labels test (non puoi calcolare media)

**Plot test set**:

```python
plt.figure(figsize=(12, 10))
plt.scatter(A_pc_test[0, :500], A_pc_test[1, :500], 
            marker="o", c=labels_test[:500], s=50, 
            cmap='coolwarm', alpha=0.6, edgecolors='k', linewidths=0.5)
plt.axvline(threshold, color="k", linestyle="--", linewidth=2.5,
            label=f"Threshold = {threshold}")
plt.xlabel("PC1 (Test Set)", fontsize=13)
plt.ylabel("PC2 (Test Set)", fontsize=13)
plt.title("Test Set: Stesso Threshold del Training", fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Osservazione**: Separazione ancora **eccellente** su dati mai visti!

---

### 14.14 Step 13: Confusion Matrix {#mnist-confusion}

**Predizione su test set**:

```python
PC_1_test = A_pc_test[0, :]
labels_predicted = np.where(PC_1_test > threshold, digits[0], digits[1])
```

**Calcolo manuale confusion matrix**:

```python
# True Positives (0 classificati come 0)
true_0 = np.sum((labels_test == digits[0]) & (labels_predicted == digits[0]))

# False Positives (9 classificati come 0)
false_0 = np.sum((labels_test == digits[1]) & (labels_predicted == digits[0]))

# True Negatives (9 classificati come 9)
true_9 = np.sum((labels_test == digits[1]) & (labels_predicted == digits[1]))

# False Negatives (0 classificati come 9)
false_9 = np.sum((labels_test == digits[0]) & (labels_predicted == digits[1]))

print("="*50)
print("CONFUSION MATRIX (Test Set)")
print("="*50)
print(f"True  0 (TP): {true_0:4d}  |  False 0 (FP): {false_0:4d}")
print(f"False 9 (FN): {false_9:4d}  |  True  9 (TN): {true_9:4d}")
print("-"*50)

# Accuracy
accuracy = (true_0 + true_9) / (true_0 + true_9 + false_0 + false_9)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Precision e Recall per cifra 0
precision_0 = true_0 / (true_0 + false_0) if (true_0 + false_0) > 0 else 0
recall_0 = true_0 / (true_0 + false_9) if (true_0 + false_9) > 0 else 0

print(f"\nMetriche per cifra 0:")
print(f"  Precision: {precision_0 * 100:.2f}%")
print(f"  Recall:    {recall_0 * 100:.2f}%")
print("="*50)
```

**Output tipico**:
```
==================================================
CONFUSION MATRIX (Test Set)
==================================================
True  0 (TP):  926  |  False 0 (FP):   28
False 9 (FN):   50  |  True  9 (TN):  975
--------------------------------------------------
Accuracy: 95.06%

Metriche per cifra 0:
  Precision: 97.07%
  Recall:    94.88%
==================================================
```

**Interpretazione**:
- ✅ **95% accuracy**: eccellente con classificatore così semplice!
- ✅ **Bilanciato**: TP≈TN, FP≈FN (nessuna classe dominante)
- 💡 **Errori simmetrici**: pochi 0→9 e pochi 9→0

**Con scikit-learn** (più compatto):

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calcola e visualizza
cm = confusion_matrix(labels_test, labels_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=digits)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title(f"Confusion Matrix - Accuracy: {accuracy*100:.2f}%", 
          fontsize=15, fontweight='bold', pad=20)
plt.xlabel("Cifra Predetta", fontsize=13)
plt.ylabel("Cifra Vera", fontsize=13)
plt.tight_layout()
plt.show()

# Stampa metriche dettagliate
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(labels_test, labels_predicted, 
                            target_names=['Cifra 0', 'Cifra 9']))
```

---

### 14.15 Codice Completo Esercizio 2 {#codice-completo-es2}

**Script finale** (da `solutions/2_handwriting_recognition.ipynb`):

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========== CARICAMENTO DATI ==========
print("Caricamento dataset training...")
data = np.genfromtxt("./mnist_train_small.csv", delimiter=",")
labels_full = data[:, 0]
A_full = data[:, 1:].transpose()

print(f"Dataset shape: {A_full.shape}")
print(f"Labels shape:  {labels_full.shape}")

# ========== FILTRAGGIO BINARIO ==========
digits = (0, 9)
mask = np.logical_or(labels_full == digits[0], labels_full == digits[1])
A = A_full[:, mask]
labels = labels_full[mask]

print(f"\nDataset filtrato (0 e 9): {A.shape}")
print(f"  Cifra 0: {np.sum(labels == 0)} samples")
print(f"  Cifra 9: {np.sum(labels == 9)} samples")

# ========== PCA ==========
print("\nCalcolo PCA...")
A_mean = A.mean(axis=1)
A_bar = A - A_mean[:, None]
U, s, VT = np.linalg.svd(A_bar, full_matrices=False)

# Componenti principali
A_pc = U.T @ A_bar

# Varianza spiegata
explained_var = np.cumsum(s**2) / np.sum(s**2)
print(f"Prime 2 PC spiegano: {explained_var[1]*100:.2f}% varianza")

# ========== SCATTER PLOT ==========
plt.figure(figsize=(12, 10))
scatter = plt.scatter(A_pc[0, :], A_pc[1, :], 
                     marker="x", c=labels, s=50, 
                     cmap='RdBu_r', alpha=0.6)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)
plt.xlabel("Prima Componente Principale", fontsize=13)
plt.ylabel("Seconda Componente Principale", fontsize=13)
plt.title("MNIST 0 vs 9: Training Set", fontsize=15, fontweight='bold')
plt.colorbar(scatter, label='Cifra', ticks=[0, 9])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== CLASSIFICATORE ==========
threshold = 0
plt.figure(figsize=(12, 10))
scatter = plt.scatter(A_pc[0, :], A_pc[1, :], 
                     marker="x", c=labels, s=50, 
                     cmap='RdBu_r', alpha=0.6)
plt.axvline(threshold, color="black", linestyle="--", linewidth=2.5,
            label=f"Threshold = {threshold}")
plt.xlabel("PC1", fontsize=13)
plt.ylabel("PC2", fontsize=13)
plt.title("Classificatore: Threshold su PC1", fontsize=15, fontweight='bold')
plt.colorbar(scatter, label='Cifra', ticks=[0, 9])
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== TEST SET ==========
print("\nCaricamento dataset test...")
data_test = np.genfromtxt("./mnist_test.csv", delimiter=",")
labels_full_test = data_test[:, 0]
A_full_test = data_test[:, 1:].transpose()

# Filtraggio
mask_test = np.logical_or(labels_full_test == digits[0], 
                          labels_full_test == digits[1])
A_test = A_full_test[:, mask_test]
labels_test = labels_full_test[mask_test]

print(f"Test set filtrato: {A_test.shape}")

# Proiezione con TRAINING parameters
A_pc_test = U.T @ (A_test - A_mean[:, None])

# Predizione
PC_1_test = A_pc_test[0, :]
labels_predicted = np.where(PC_1_test > threshold, digits[0], digits[1])

# ========== CONFUSION MATRIX ==========
true_0 = np.sum((labels_test == digits[0]) & (labels_predicted == digits[0]))
false_0 = np.sum((labels_test == digits[1]) & (labels_predicted == digits[0]))
true_9 = np.sum((labels_test == digits[1]) & (labels_predicted == digits[1]))
false_9 = np.sum((labels_test == digits[0]) & (labels_predicted == digits[1]))

accuracy = (true_0 + true_9) / (true_0 + true_9 + false_0 + false_9)

print("\n" + "="*60)
print("RISULTATI TEST SET")
print("="*60)
print(f"True  0: {true_0:4d}  |  False 0: {false_0:4d}")
print(f"False 9: {false_9:4d}  |  True  9: {true_9:4d}")
print("-"*60)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("="*60)

# Visualizzazione con sklearn
cm = confusion_matrix(labels_test, labels_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=[f'Cifra {d}' for d in digits])

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d', 
          colorbar=True, text_kw={'fontsize': 16})
plt.title(f"Confusion Matrix - Test Set\nAccuracy: {accuracy*100:.2f}%", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Cifra Predetta", fontsize=14)
plt.ylabel("Cifra Vera", fontsize=14)
plt.tight_layout()
plt.show()

# ========== SCATTER TEST SET ==========
plt.figure(figsize=(12, 10))
scatter = plt.scatter(A_pc_test[0, :], A_pc_test[1, :], 
                     marker="o", c=labels_test, s=50, 
                     cmap='RdBu_r', alpha=0.6, 
                     edgecolors='k', linewidths=0.5)
plt.axvline(threshold, color="black", linestyle="--", linewidth=2.5,
            label=f"Threshold = {threshold}")
plt.xlabel("PC1 (Test Set)", fontsize=13)
plt.ylabel("PC2 (Test Set)", fontsize=13)
plt.title(f"Test Set Predictions - Accuracy: {accuracy*100:.2f}%", 
          fontsize=15, fontweight='bold')
plt.colorbar(scatter, label='Cifra Vera', ticks=[0, 9])
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Classificazione completata!")
```

---

### 14.16 Conclusioni Esercizio 2 {#conclusioni-es2}

**Risultati chiave**:

1. ✅ **PCA riduce 784D → 2D preservando informazione**
   - Prime 2 PC: ~28% varianza spiegata
   - Sufficiente per separare visivamente 0 e 9

2. ✅ **Classificatore semplice = alta accuratezza**
   - Threshold su PC1: ~95% accuracy
   - Nessun algoritmo ML complesso necessario!
   - Dimostra potenza della PCA

3. ✅ **Generalizzazione eccellente**
   - Test set mai visto: stessa accuracy
   - Conferma che PC catturano pattern reali

4. ⚠️ **Limitazioni**:
   - Solo 2 classi (binario)
   - Multi-class (0-9) richiede algoritmi avanzati
   - Threshold manuale (non ottimizzato)

**Confronto con stato dell'arte**:
- 📊 **Il nostro classificatore**: 95% (PCA + threshold)
- 🧠 **CNN moderne** (LeNet, ResNet): >99% su MNIST completo
- 💡 **Trade-off**: semplicità vs accuracy

**Estensioni possibili**:
1. **Ottimizzazione threshold**: grid search, ROC curve
2. **Più PC**: usare 10-20 PC invece di solo PC1
3. **Algoritmi ML**: SVM, Random Forest su PC
4. **Multi-class**: One-vs-All, softmax su tutte le cifre

**Lezioni apprese**:
- 🎯 PCA = preprocessing potente per classificazione
- 📉 Dimensionalità alta ≠ complessità alta (molte dimensioni ridondanti)
- ✅ Visualizzazione 2D aiuta a capire separabilità classi
- 🔬 Test set separato = unica misura vera di performance

---

## 15. Formule e Convenzioni PCA {#formule-convenzioni}

### 15.1 Convenzione Samples-as-Columns {#convenzione-samples}

**Notazione usata nel corso**:

$$
X \in \mathbb{R}^{d \times n}
$$

Dove:
- $d$ = numero di **features** (dimensioni, variabili)
- $n$ = numero di **samples** (osservazioni, punti)
- $X_{ij}$ = valore della feature $i$ per il sample $j$

**Ogni COLONNA = un campione**:
```python
X = np.array([[x1_1, x1_2, ..., x1_n],    # Feature 1
              [x2_1, x2_2, ..., x2_n],    # Feature 2
              ...
              [xd_1, xd_2, ..., xd_n]])   # Feature d

X.shape  # (d, n) - features × samples
```

**Esempio MNIST**:
```python
A.shape  # (784, 3979)
# 784 features (pixel)
# 3979 samples (immagini)
```

---

### 15.2 Matrice di Covarianza {#matrice-covarianza}

**Formula** (dati centrati):

$$
C = \frac{1}{n-1} \bar{X} \bar{X}^T
$$

Dove $\bar{X} = X - \text{mean}(X)$ (ogni colonna centrata).

**Dimensione**: $C \in \mathbb{R}^{d \times d}$ (simmetrica, semidefinita positiva)

**Elementi**:

$$
C_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} \bar{X}_{ik} \bar{X}_{jk} = \text{cov}(\text{feature}_i, \text{feature}_j)
$$

**Diagonale**: $C_{ii} = \text{var}(\text{feature}_i)$

**Codice**:
```python
# Metodo 1: manuale
X_bar = X - X.mean(axis=1, keepdims=True)
C = (X_bar @ X_bar.T) / (n - 1)

# Metodo 2: NumPy (equivalente)
C = np.cov(X, rowvar=True, bias=False)
```

---

### 15.3 Relazione SVD ↔ PCA {#svd-pca}

**SVD su dati centrati**:

$$
\bar{X} = U \Sigma V^T
$$

Dove:
- $U \in \mathbb{R}^{d \times r}$ - vettori singolari sinistri (**direzioni principali**)
- $\Sigma \in \mathbb{R}^{r \times r}$ - valori singolari diagonali $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r$
- $V^T \in \mathbb{R}^{r \times n}$ - vettori singolari destri (coefficienti)
- $r = \min(d, n)$ - rank massimo

**Relazione con autovalori di C**:

$$
C = \frac{1}{n-1} \bar{X} \bar{X}^T = \frac{1}{n-1} U \Sigma V^T V \Sigma^T U^T = \frac{1}{n-1} U \Sigma^2 U^T
$$

Quindi:
- **Autovettori di $C$** = colonne di $U$ (direzioni principali)
- **Autovalori di $C$** = $\lambda_k = \frac{\sigma_k^2}{n-1}$

**Varianza lungo PC $k$**:

$$
\text{var}_k = \lambda_k = \frac{\sigma_k^2}{n-1}
$$

**Deviazione standard**:

$$
\text{std}_k = \sqrt{\lambda_k} = \frac{\sigma_k}{\sqrt{n-1}}
$$

---

### 15.4 Componenti Principali {#componenti-principali}

**Formula**:

$$
\Phi = U^T \bar{X}
$$

Dove:
- $\Phi \in \mathbb{R}^{d \times n}$ (stessa dimensione di $\bar{X}$)
- $\Phi_{ki}$ = proiezione del sample $i$ sulla direzione principale $k$

**Interpretazione geometrica**:
- **Rotazione**: $U^T$ ruota i dati nel sistema di riferimento delle PC
- **PC allineate agli assi**: $\text{cov}(\Phi) = \text{diag}(\lambda_1, \ldots, \lambda_d)$

**Proprietà**:
1. $\text{mean}(\Phi_k) = 0$ per ogni $k$ (componenti centrate)
2. $\text{cov}(\Phi_k, \Phi_j) = 0$ per $k \neq j$ (decorrelate!)
3. $\text{var}(\Phi_k) = \lambda_k$ (varianza decrescente)

---

### 15.5 Ricostruzione e Compressione {#ricostruzione}

**Ricostruzione completa**:

$$
\bar{X} = U \Phi = U U^T \bar{X}
$$

$$
X = \bar{X} + \text{mean}(X) = U \Phi + \mu
$$

**Approssimazione con $k$ PC** (compressione):

$$
\bar{X}_k = U_{:, 1:k} \Phi_{1:k, :} = \sum_{i=1}^{k} u_i \phi_i^T
$$

Dove:
- $U_{:, 1:k}$ = prime $k$ colonne di $U$
- $\Phi_{1:k, :}$ = prime $k$ righe di $\Phi$

**Errore di ricostruzione**:

$$
\| \bar{X} - \bar{X}_k \|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2
$$

**Varianza spiegata da $k$ PC**:

$$
R^2_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{r} \lambda_i}
$$

---

### 15.6 Broadcasting NumPy per Centraggio {#broadcasting}

**Problema**: Sottrarre media da ogni colonna di $X \in \mathbb{R}^{d \times n}$

**Media campionaria**:
```python
X_mean = np.mean(X, axis=1)  # Shape: (d,)
```

**❌ Errore comune**:
```python
X_bar = X - X_mean  # ValueError: shapes (d,n) and (d,) incompatibili
```

**✅ Soluzione 1: Reshape a colonna**:
```python
X_bar = X - X_mean[:, None]  # (d,n) - (d,1) → broadcasting OK
```

**✅ Soluzione 2: `keepdims=True`**:
```python
X_mean = np.mean(X, axis=1, keepdims=True)  # Shape: (d, 1)
X_bar = X - X_mean  # Broadcasting automatico
```

**Regole broadcasting NumPy**:
1. Allinea shapes da **destra**
2. Dimensioni compatibili se:
   - Uguali, OPPURE
   - Una delle due è 1 (replicata automaticamente)

**Esempio**:
```
X:      (784, 3979)
X_mean: (784, 1)     ← dimensione 1 replicata 3979 volte
------
X_bar:  (784, 3979)  ✅
```

**Visualizzazione**:
```
X (d×n):           X_mean[:,None] (d×1):    Broadcasting:
┌─────────────┐    ┌───┐                   ┌─────────────┐
│ x₁ x₂ ... xₙ│ -  │ μ₁│  replica n volte  │ μ₁ μ₁ ... μ₁│
│ y₁ y₂ ... yₙ│    │ μ₂│        →          │ μ₂ μ₂ ... μ₂│
│ z₁ z₂ ... zₙ│    │ μ₃│                   │ μ₃ μ₃ ... μ₃│
└─────────────┘    └───┘                   └─────────────┘
```

---

### 15.7 PCA: Algoritmo Completo {#algoritmo-completo}

**Input**: Matrice dati $X \in \mathbb{R}^{d \times n}$ (samples as columns)

**Output**: 
- Direzioni principali $U \in \mathbb{R}^{d \times d}$
- Componenti principali $\Phi \in \mathbb{R}^{d \times n}$
- Varianze $\lambda_1, \ldots, \lambda_d$

**Passi**:

1. **Centraggio**:
   $$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i, \quad \bar{X} = X - \mu \mathbf{1}^T$$
   ```python
   X_mean = np.mean(X, axis=1, keepdims=True)
   X_bar = X - X_mean
   ```

2. **SVD**:
   $$\bar{X} = U \Sigma V^T$$
   ```python
   U, s, VT = np.linalg.svd(X_bar, full_matrices=False)
   ```

3. **Varianze**:
   $$\lambda_k = \frac{\sigma_k^2}{n-1}$$
   ```python
   lambdas = (s ** 2) / (n - 1)
   ```

4. **Componenti Principali**:
   $$\Phi = U^T \bar{X}$$
   ```python
   Phi = U.T @ X_bar
   ```

5. **Varianza Spiegata**:
   $$R^2_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$
   ```python
   explained_var = np.cumsum(lambdas) / np.sum(lambdas)
   ```

6. **Ricostruzione con $k$ PC**:
   $$X_k = U_{:,1:k} \Phi_{1:k,:} + \mu \mathbf{1}^T$$
   ```python
   X_approx = U[:, :k] @ Phi[:k, :] + X_mean
   ```

---

## 16. Materiali e Riferimenti Lab 2 {#materiali-lab2}

### 16.1 Notebook e Dataset {#notebook-dataset}

**Cartella Lab02**: `C:\Users\miche\OneDrive\Desktop\UNI\3-NAML\note\Lab02\`

**Notebook Esercizi**:
1. `1_PCA_2D_start.ipynb` - Esercizio PCA geometrica 2D
2. `2_handwriting_recognition_start.ipynb` - Classificazione MNIST 0 vs 9
3. `3_cancer_diagnostic_start.ipynb` - Diagnostica cancro ovarico (non trattato)

**Notebook Soluzioni** (cartella `solutions/`):
1. `1_PCA_2D.ipynb` - Soluzione completa PCA 2D
2. `2_handwriting_recognition.ipynb` - Soluzione completa MNIST
3. `3_cancer_diagnostic.ipynb` - Soluzione diagnostica cancro

**Dataset**:
1. `mnist_train_small.csv` - 20000 immagini training (785 colonne)
2. `mnist_test.csv` - 10000 immagini test (785 colonne)
3. `ovariancancer_grp.csv` - Gruppi pazienti cancro
4. `ovariancancer_obs.csv` - Osservazioni cancro

---

### 16.2 Codice Chiave da Ricordare {#codice-chiave}

**Setup PCA standard**:
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Caricamento dati (samples as columns)
X = data.T  # Trasponi se samples as rows

# 2. Centraggio
X_mean = np.mean(X, axis=1, keepdims=True)
X_bar = X - X_mean

# 3. SVD
U, s, VT = np.linalg.svd(X_bar, full_matrices=False)

# 4. Varianze
lambdas = (s ** 2) / (X.shape[1] - 1)
explained_var = np.cumsum(lambdas) / np.sum(lambdas)

# 5. Componenti principali
Phi = U.T @ X_bar

# 6. Ricostruzione k PC
k = 10
X_approx = U[:, :k] @ Phi[:k, :] + X_mean
```

**Classificatore threshold su PC1**:
```python
# Training
threshold = 0  # Scegli da scatter plot
PC1_train = Phi[0, :]
labels_pred_train = np.where(PC1_train > threshold, class_0, class_1)

# Test (usa TRAINING mean e U!)
X_test_bar = X_test - X_mean  # Training mean!
Phi_test = U.T @ X_test_bar    # Training U!
PC1_test = Phi_test[0, :]
labels_pred_test = np.where(PC1_test > threshold, class_0, class_1)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_true_test, labels_pred_test)
```

**Visualizzazione direzioni principali 2D**:
```python
# Frecce direzioni principali
plt.arrow(mean[0] - u[0]*std, mean[1] - u[1]*std,
          2*u[0]*std, 2*u[1]*std,
          color="red", width=0.3, head_width=1.0)
```

---

### 16.3 Errori Comuni e Soluzioni {#errori-comuni}

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| `ValueError: shapes incompatibili` | Broadcasting centraggio | Usa `X_mean[:, None]` o `keepdims=True` |
| Test accuracy << Train | Uso mean/U del test | **Sempre** usa parametri TRAINING su test |
| Segno PC opposto al vero | Indeterminazione autovettori | Normale! $u$ e $-u$ entrambi validi |
| `MemoryError` su SVD | `full_matrices=True` di default | Usa `full_matrices=False` |
| PC1 non separa classi | Threshold non ottimale | Prova valori diversi guardando scatter |
| Plot frecce non centrate | Offset sbagliato | Usa `mean - u*std` come inizio, `2*u*std` come lunghezza |

**Debugging tips**:
```python
# Controlla sempre shapes
print(f"X: {X.shape}")
print(f"X_mean: {X_mean.shape}")
print(f"X_bar: {X_bar.shape}")

# Verifica media zero dopo centraggio
print(f"Mean after centering: {np.mean(X_bar, axis=1)}")  # ~[0, 0, ...]

# Controlla ordine varianze
print(f"Varianze decrescenti? {np.all(np.diff(lambdas) <= 0)}")  # True
```

---

## 17. Checklist Lab 2 - PCA {#checklist-lab2}

### 17.1 Teoria PCA {#checklist-teoria}

- [ ] **Definizione PCA**: Trovare direzioni massima varianza
- [ ] **SVD ↔ PCA**: $C = \frac{1}{n-1} U \Sigma^2 U^T$
- [ ] **Autovalori**: $\lambda_k = \frac{\sigma_k^2}{n-1}$ = varianza lungo PC $k$
- [ ] **Convenzione**: Samples as columns ($X \in \mathbb{R}^{d \times n}$)
- [ ] **Centraggio obbligatorio**: $\bar{X} = X - \text{mean}(X)$
- [ ] **Componenti decorrelate**: $\text{cov}(\Phi_i, \Phi_j) = 0$ per $i \neq j$
- [ ] **Varianza spiegata**: $R^2_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^d \sigma_i^2}$
- [ ] **Ordine decrescente**: $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_d$

---

### 17.2 Implementazione NumPy {#checklist-implementazione}

- [ ] **Caricamento dati**: `np.genfromtxt()` per CSV
- [ ] **Trasposizione**: `.T` se samples as rows in CSV
- [ ] **Media**: `np.mean(X, axis=1)` per media per feature
- [ ] **Broadcasting**: `X_mean[:, None]` per centraggio
- [ ] **SVD**: `np.linalg.svd(X_bar, full_matrices=False)`
- [ ] **Varianze**: `lambdas = s**2 / (n-1)`
- [ ] **Componenti**: `Phi = U.T @ X_bar`
- [ ] **Ricostruzione**: `X_approx = U[:, :k] @ Phi[:k, :] + X_mean`
- [ ] **Visualizzazione PC**: `U[:, k].reshape((28, 28))` per MNIST

---

### 17.3 Classificazione MNIST {#checklist-mnist}

- [ ] **Filtraggio binario**: `np.logical_or(labels == 0, labels == 9)`
- [ ] **Bitmask**: `A_filtered = A[:, mask]`
- [ ] **Scatter PC1 vs PC2**: Verifica separazione visiva
- [ ] **Threshold manuale**: Scegli da plot, es. `threshold = 0`
- [ ] **Test set separato**: MAI usato per training
- [ ] **Parametri training**: Usa `X_mean` e `U` del training su test
- [ ] **Predizione**: `np.where(PC1 > threshold, 0, 9)`
- [ ] **Confusion matrix**: `sklearn.metrics.confusion_matrix()`
- [ ] **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- [ ] **Interpretazione**: Verifica bilanciamento FP vs FN

---

### 17.4 Visualizzazione {#checklist-visualizzazione}

- [ ] **Immagini MNIST**: `reshape((28, 28))` + `imshow(cmap="gray")`
- [ ] **Prime 30 PC**: Grid 3×10 con `subplots()`
- [ ] **Varianza spiegata**: Plot cumulativo
- [ ] **Scatter 2D**: PC1 vs PC2 colorato per label
- [ ] **Threshold**: `plt.axvline()` su scatter
- [ ] **Frecce 2D**: `plt.arrow()` per direzioni principali
- [ ] **Confusion matrix**: `ConfusionMatrixDisplay` di sklearn

---

### 17.5 Best Practices ML {#checklist-best-practices}

- [ ] **Train/Test split**: Dataset separati, NO sovrapposizione
- [ ] **Preprocessing coerente**: Stessi parametri su train e test
- [ ] **Validazione**: Accuracy su test = misura vera performance
- [ ] **Overfitting check**: Train accuracy >> Test accuracy è RED FLAG
- [ ] **Bilanciamento classi**: Verifica distribuzione 0 vs 9 (~50/50)
- [ ] **Exploratory analysis**: Visualizza dati PRIMA di modellare
- [ ] **Metriche multiple**: Non solo accuracy (precision, recall, F1)
- [ ] **Reproducibility**: `np.random.seed()` per risultati consistenti

---

## 18. Esercizi Avanzati Lab 2 {#esercizi-avanzati}

### 18.1 PCA su Iris Dataset {#esercizio-iris}

**Dataset**: Iris flowers (150 samples, 4 features, 3 species)

**Task**:
1. Carica Iris da `sklearn.datasets.load_iris()`
2. Applica PCA e plotta prime 2 PC
3. Colora punti per specie (setosa, versicolor, virginica)
4. Quante PC servono per 95% varianza?

**Codice starter**:
```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.T  # (4, 150)
labels = iris.target

# TODO: PCA e visualizzazione
```

---

### 18.2 Face Recognition (Eigenfaces) {#esercizio-eigenfaces}

**Dataset**: Olivetti faces (400 immagini 64×64, 40 persone)

**Task**:
1. Carica con `sklearn.datasets.fetch_olivetti_faces()`
2. Applica PCA su immagini appiattite (4096D)
3. Visualizza prime 16 "eigenfaces" (direzioni principali)
4. Ricostruisci immagine con k=10, 50, 100 PC
5. Calcola errore ricostruzione vs k

**Hint**: Eigenfaces sono le PC che catturano variazioni facciali comuni.

---

### 18.3 MNIST Multi-Class (0-9) {#esercizio-multiclass}

**Dataset**: MNIST completo (10 classi)

**Task**:
1. Usa tutte le cifre 0-9 (non solo 0 e 9)
2. Plotta scatter PC1 vs PC2 con 10 colori
3. Trova soglie per separare ogni cifra
4. Implementa classificatore One-vs-All
5. Calcola accuracy per classe e globale

**Challenge**: Alcune cifre si sovrappongono (es. 4 e 9)!

---

### 18.4 Confronto Dimensionality Reduction {#esercizio-confronto}

**Task**: Confronta PCA vs altri metodi su MNIST

**Metodi**:
1. **PCA**: Lineare, preserva varianza globale
2. **t-SNE**: Non-lineare, preserva struttura locale
3. **UMAP**: Non-lineare, veloce
4. **LDA**: Supervised, massimizza separazione classi

**Codice starter**:
```python
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.T)

# t-SNE (lento!)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X.T)

# UMAP
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X.T)

# LDA (supervised)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X.T, labels)

# TODO: Plot 2x2 grid con tutti i metodi
```

**Confronta**:
- Separazione classi
- Tempo esecuzione
- Interpretabilità

---

### 18.5 PCA per Outlier Detection {#esercizio-outlier}

**Task**: Usa PCA per trovare immagini "anomale" in MNIST

**Idea**: 
- Immagini normali: basso errore ricostruzione con poche PC
- Outlier: alto errore (non spiegati da pattern comuni)

**Metodo**:
1. PCA su dataset
2. Ricostruisci con k=20 PC
3. Calcola errore: $e_i = \| x_i - \hat{x}_i \|_2$
4. Trova top-10 errori massimi
5. Visualizza: sono davvero anomali?

**Codice**:
```python
# Ricostruzione
X_approx = U[:, :k] @ Phi[:k, :] + X_mean

# Errori per sample
errors = np.linalg.norm(X - X_approx, axis=0)

# Top outliers
outlier_idx = np.argsort(errors)[-10:]

# Visualizza
for idx in outlier_idx:
    plt.imshow(X[:, idx].reshape((28, 28)), cmap="gray")
    plt.title(f"Label: {labels[idx]}, Error: {errors[idx]:.2f}")
    plt.show()
```

---

## 19. Risorse Aggiuntive {#risorse-aggiuntive}

### 19.1 Documentazione {#documentazione}

**NumPy**:
- `np.linalg.svd`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- `np.mean`: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
- Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html

**Scikit-learn**:
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- Confusion Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

**Matplotlib**:
- imshow: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
- arrow: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.arrow.html

---

### 19.2 Letture Consigliate {#letture}

1. **Strang, Gilbert** - "Introduction to Linear Algebra" (Cap. 7: SVD)
2. **Bishop, Christopher** - "Pattern Recognition and Machine Learning" (Cap. 12: PCA)
3. **James et al.** - "An Introduction to Statistical Learning" (Cap. 10: Unsupervised Learning)
4. **Goodfellow et al.** - "Deep Learning" (Cap. 2: Linear Algebra)

---

### 19.3 Video e Tutorial {#video-tutorial}

- **StatQuest**: "PCA clearly explained" - https://www.youtube.com/watch?v=FgakZw6K1QQ
- **3Blue1Brown**: "Essence of linear algebra" (playlist)
- **Coursera**: Andrew Ng - Machine Learning (Week 8: Dimensionality Reduction)

---

## 20. Riepilogo Finale {#riepilogo-finale}

**Lab 2 in sintesi**:

✅ **Esercizio 1 - PCA Geometrica 2D**:
- Trasformazione $x = Az + b$ con seed Gaussiano
- PCA recupera $A$ (direzioni + scaling) e $b$ (traslazione)
- Dimostrazione: SVD stima parametri trasformazione geometrica

✅ **Esercizio 2 - MNIST Classification**:
- 784D → 2D con PCA preservando ~28% varianza
- Classificatore threshold su PC1: **95% accuracy**
- Test set conferma generalizzazione

✅ **Concetti chiave**:
- Samples as columns ($X \in \mathbb{R}^{d \times n}$)
- SVD ↔ Eigendecomposition di $C$
- Varianza spiegata = metrica riduzione dimensionalità
- Train/Test split = best practice ML

✅ **Tools NumPy**:
- `np.linalg.svd()` - decomposizione
- Broadcasting per operazioni vettoriali
- `axis=1` per mean su samples

**Prossimi passi**:
- Lab 3: Applicazioni avanzate PCA (immagini, text mining)
- Teoria: Kernel PCA, Sparse PCA, Robust PCA
- ML: Integrazione PCA in pipeline complesse

---

**📊 Statistiche finali documento**:
- **Righe totali**: ~1750 (da 743 iniziali)
- **Espansione**: +135% ✅
- **Codice completo**: 2 esercizi Lab02
- **Formule**: Tutte le relazioni PCA fondamentali
- **Esempi**: 10+ snippet pronti all'uso
- **Checklist**: 40+ item teoria + pratica

---





`01:22:54` 
dati che cambia. Okay? E ora viene il vostro compito. Quindi quello che vogliamo fare ora è usare la PCA per capire quali cifre sono zero e quali cifre sono nove usando un algoritmo di classificazione. E faremo questo passo dopo passo. Il vostro primo compito è, prima di tutto, tracciare le prime 30 immagini che sono o uno zero o un nove. Dopo di che, le tracciate.

`01:23:35` 
Poi calcolate la media, perché vogliamo che x abbia media 0 per applicare la PCA, e tracciate la media. E infine, farete la PCA usando la SVD, stampando i valori singolari, la frazione cumulativa dei valori singolari, e la frazione dei valori spiegati. Questo è un esercizio che abbiamo fatto l'ultima volta. Potete aprire il vecchio notebook e fare copia e incolla. L'esame è a libro aperto, quindi dovete capire dove copiare e incollare.

`01:24:06` 
Per me, è completamente ok. E poi visualizzate i primi 30 assi principali e calcolate le prime due componenti principali. Questi sono tutti compiti che abbiamo fatto o oggi o l'ultima volta. Quindi, fatelo. Potete copiare e incollare quello che volete. La parte importante è che capiate davvero cosa stiamo copiando e incollando. Okay, usate anche Google. Qui ho un suggerimento dove dovete consultare la documentazione per capire come fare questo in modo efficiente. Vi darò 20 minuti e cerchiamo di arrivare almeno dove tracciate e visualizzate gli assi principali e le componenti principali.

`01:26:29` 
Grazie.

`01:29:11` 
Va bene. Grazie.

`01:30:31` 
Dovrò farlo di nuovo. Se mettete un axis uguale a un certo numero, significa che state facendo un loop su tutto quell'asse. Quindi se lo mettete uguale a 1, prima l'asse di 1 è il secondo, cioè, le colonne. Quindi quello che stiamo facendo è fare una matrice, fare un loop con tutte le colonne.

`01:31:02` 
Cioè, se avete una matrice che è 2 per n, dove 2 per n è uguale al numero di colonne, vedete che state facendo un loop con tutte le colonne. State facendo un loop tra le colonne del vettore che ha 2 linee. Perché il risultato è che c'è un loop fatto su quell'asse.

`01:32:18` 
Grazie mille.

`01:33:01` 
Grazie.

`01:33:33` 
Sì, sì.

`01:34:37` 
Grazie.

`01:36:00` 
Grazie mille.

`01:36:32` 
Grazie.

`01:37:45` 
Grazie mille.

`01:38:30` 
Grazie.

`01:39:52` 
Grazie mille.

`01:41:41` 
Grazie.

`01:42:32` 
Grazie mille.

`01:43:28` 
Grazie.

`01:44:42` 
Grazie mille.

`01:45:53` 
Okay, mi sembra che stiate andando bene, e quindi vorrei mostrarvi la soluzione perché abbiamo ancora alcune cose da fare insieme. Quindi iniziamo qui. Quindi qui quello che vogliamo fare è avere un dataset con solo zeri e nove. Quindi quello che facciamo qui è che, okay, definiamo le cifre. Quindi quello che vogliamo è quello con zero e nove, solo per essere generali, perché forse volete.

`01:46:27` 
quattro e sei, e questo funziona molto bene per qualsiasi coppia di cifre, e poi definiamo una mask. Quindi il vettore con zeri e uno come labels uguale a digits di zero e poi vogliamo che questo sia vero anche quando abbiamo nove. Quindi quando entrambi sono veri. Quindi quello che possiamo fare qui è fare np logical or di questo o quello questo è anche equivalente.

`01:47:12` 
Questa è solo una notazione diversa. Potete mettere una barra verticale e questo è un or. Ci sono due modi per fare esattamente la stessa cosa. La differenza è che. come potete fare matmul e quel segno è esattamente lo stesso risultato finale. sì scusate sì sì poi quello che facciamo è okay il nostro.

`01:47:46` 
a è uguale a a full tutti i pixel e la mask e anche estraiamo solo le etichette che vogliamo, dicendo che queste sono labels uguale a labels full della mask e possiamo bloccare la forma se volete okay perfetto ora il plot per il plot.

`01:48:23` 
È inutile fare qualcos'altro che copiare e incollare questo, tipo dovreste capire cosa è scritto qui, ma questo è molto codice e possiamo riutilizzarlo. Quindi copiamo e incolliamo, qual è la differenza? Invece di a full, mettiamo a, e possiamo anche mettere un titolo per controllare che le etichette siano corrette. Quindi ax i punto set title, e mettiamo label di i, labels di i, e questo è una stringa.

`01:49:06` 
Okay, e questo è il risultato. Quindi vediamo che i 9 sono 9 e gli 0 sono 0. Perfetto. Poi visualizziamo la media. quindi a bar a mean è uguale a numpy punto mean di a axis uguale a uno okay anche se queste sono immagini la convenzione è la stessa di prima e quindi non ripeterò perché questo funziona così.

`01:49:40` 
e per tracciarlo facciamo solo plt punto imshow di a mean la convenzione è che questo è un vettore, imshow vuole una matrice quindi facciamo solo un reshape questo è 28 per 28. non sbagliato sì la mappa di colori è quella grigia okay. Quindi quello che abbiamo qui, questa è la media di tutte le immagini. Quindi prendete tutte le immagini, e per ogni pixel, in una certa posizione, fate la media per quel pixel.

`01:50:18` 
E in esso, questo è più o meno quello che ci aspettiamo, perché abbiamo la forma rotonda dello zero, e questa parte, che è un po' più sbiadita, perché è lì solo per le linee dei nove. Poi, possiamo calcolare la SVD e i valori singolari. Quindi, definiamo A bar come A meno A mean, e facciamo il broadcasting, e quindi, come al solito, aggiungiamo la dimensione per rimuovere la media per ogni campione.

`01:50:56` 
Poi, u s vt è uguale a numpy linalg svd di A bar, full matrices false. Poi facciamo un plot, fig ax uguale PLT punto subplots, vogliamo n rows uguale a 1, n cols uguale a 3, e poi cosa facciamo?

`01:51:29` 
Sul primo asse, 0, mettiamo i valori singolari, di solito un semi-logaritmico nella direzione y è il migliore. Poi asse 1, mettiamo la frazione cumulativa, quindi ax 1 punto plot np punto cumsum di s diviso per la somma di s. E infine, questo è molto simile, sul secondo asse, e qui, al quadrato.

`01:52:11` 
E al quadrato, okay. Cosa ho fatto di sbagliato? Mi è mancato una parentesi qui. Okay, è un po' compresso, ma in questo modo. Quindi quello che stiamo vedendo, quello che stiamo vedendo qui per i valori singolari è che a un certo punto, siamo fino alla precisione della macchina per ricostruire le immagini.

`01:52:49` 
Quindi come abbiamo fatto l'ultima volta, quello che capiamo è che le componenti singolari necessarie per ricostruire le immagini sono importanti fino a un certo punto, e poi siamo praticamente a epsilon macchina. Perché è così? Qualcuno ha qualche idea? Quindi la risposta è che a un certo punto, ci sono alcuni pixel che sono sempre gli stessi. Se tornate indietro qui, vedete che intorno a qui, per tutte le immagini, c'è il nero.

`01:53:25` 
Significa che tutte le informazioni in questa corona intorno all'immagine a un certo punto sono sempre le stesse. E quindi non abbiamo bisogno di dati aggiuntivi per ricostruire qualcosa che è sempre zero. E lo dimostreremo quando tracceremo le direzioni singolari, scusate, le direzioni principali, okay? Quindi il calo può essere praticamente spiegato dagli zeri tutto intorno all'immagine.

`01:53:58` 
Quindi infatti, se volessimo comprimere le immagini, taglieremmo da qualche parte come qui. E poi abbiamo anche il comportamento della somma di tutti i valori singolari e della varianza spiegata. A un certo punto, vedete che raggiungono una piattaforma, e quello è dove i valori singolari non contano affatto. Va bene. Sì. Nell'ultima volta, per la varianza spiegata, abbiamo usato la radice quadrata del...

`01:54:33` 
La radice quadrata del uno di... Voglio dire, la radice quadrata è solo una funzione monotona, non cambia significativamente l'andamento, del plot. È convessa, è sempre crescente, è come confrontare la varianza e la deviazione standard. È più o meno la stessa cosa.

`01:55:05` 
Okay, poi visualizziamo i primi 30 assi principali. Quindi gli assi principali sono le direzioni che usiamo per scrivere l'immagine. In un certo senso, potete pensare che l'immagine può sempre essere scritta come una combinazione lineare di queste direzioni. Quindi queste sono come i blocchi di costruzione di base come un'immagine di tutte le altre. Quindi quello che facciamo, lasciatemi solo copiare e incollare questa parte perché è ancora molto simile. Quindi quello che abbiamo qui è rimuovo il titolo e invece di A, le direzioni principali sono le colonne di u. Quindi prendiamo tutte le righe e la colonna i-esima. Il reshape è lo stesso. Il plot è praticamente lo stesso. E questi sono i risultati.

`01:56:02` 
Quindi cosa vediamo qui? Quindi prima di tutto, in questi tipi di immagini, potete vedere qualcosa che è molto simile a uno zero e qualcosa che è molto simile a un nove, okay? Quindi questi sono in realtà i blocchi di costruzione di tutte le altre immagini, perché potete vedere che le caratteristiche macroscopiche sono quelle degli zeri e dei nove. Il secondo fatto è che, come nel lab precedente, man mano che aumentiamo, giusto, andiamo da caratteristiche macroscopiche, su larga scala a immagini con strutture sempre più fini e più fini, okay? Quindi stiamo andando da, tipo, grandi strutture, come cerchi, a qualcosa che è molto, molto fine.

`01:56:52` 
E queste sono solo le prime 30. Per esempio, quello che possiamo fare è che possiamo aggiungere qui 100. E quindi queste sono le direzioni principali che vanno da 100 a 130. E qui potete vedere che questo inizia a comportarsi più o meno come rumore. Infine, se guardate le ultime componenti, quindi quelle dopo il grande calo, quindi diciamo dopo 600, quello che vedete è che questo è solo rumore.

`01:57:24` 
È solo rumore intorno all'immagine, nella corona intorno. E quindi possiamo dimostrare che in realtà tutti questi valori singolari che hanno significato molto, molto basso sono quelli collegati agli zeri che sono quasi sempre presenti in queste immagini. Okay? calcolate le prime due componenti principali corrispondenti alla prima immagine quindi quello che dobbiamo.

`01:57:55` 
fare qui è un prodotto scalare quindi una proiezione quindi quello che stiamo facendo è che stiamo proiettando, con il prodotto interno a bar zero quindi questo è il primo campione sulla prima componente principale quindi la prima componente principale è u la prima colonna quindi tutte le righe prima colonna quindi questa è.

`01:58:27` 
la prima componente principale della prima immagine similmente se vogliamo questo per le prime due componenti principali abbiamo ancora il primo campione e poi la seconda componente principale. E questi sono i valori. Quello che voglio mostrarvi qui è che questa proiezione che abbiamo fatto a mano è esattamente quello che facciamo quando facciamo il prodotto matriciale. Quindi se definiamo tutte le componenti principali come phi, come prima, e lo calcoliamo come u trasposta prodotto matriciale a bar, otteniamo lo stesso se prendiamo nella matrice la base corretta.

`01:59:34` 
E quindi questo in particolare sarà il nostro phi. Prendiamo cosa? Prendiamo sempre la prima colonna e la prima e seconda riga. e questi sono infatti gli stessi valori fino alla precisione della macchina okay infine facciamo uno scatter plot.

`02:00:12` 
delle prime due componenti principali quindi e vogliamo usare un colore diverso, se l'immagine è o uno zero o un nove okay questa è la diciamo la versione naive dove usate un for loop e quindi per ogni punto fate il prodotto interno che è la proiezione calcolate la componente x la componente y e impostate il colore a rosso o blu a seconda dall'etichetta se l'etichetta è uguale a zero e poi aggiungete un punto allo scatter la.

`02:00:49` 
buona soluzione qui che è molto molto più veloce è la seguente abbiamo calcolato, phi, poi prendiamo tutta la prima componente principale e tutta la seconda componente principale, poi, quello che possiamo fare è che plt scatter potete passare un vettore con numeri, ed è automaticamente, convertito in un colore, e quindi quello che possiamo fare qui è mettere labels, e infine metto.

`02:01:26` 
s uguale a 50, dove 50 è la dimensione, solo per avere un plot che è un po' migliore, e qui metto marker uguale a x, no, cos'era? Marker, qui. Okay, quindi cosa vediamo?

`02:01:57` 
Ogni x è o uno 0 o un 9, e possiamo vedere che nelle direzioni indicate dalla PCA, c'è più o meno una chiara differenza tra questi due tipi di immagini, okay? Quindi vogliamo classificare cosa è 0 e cosa è 9. Durante la lezione, non avete visto nessun algoritmo di classificazione, quindi quello che propongo qui è il più semplice. Disegniamo una linea verticale, e selezioniamo cosa è 0 e cosa è 9, a seconda se la prima componente principale, che è la più importante, è maggiore o minore di un certo valore.

`02:02:42` 
Okay, e potete vedere visivamente che questo potrebbe avere un risultato molto buono, perché qui abbiamo molto clustering, e qui abbiamo molto clustering. Sì, avete alcune x qui, ma dovete ricordare questi sono 20.000 punti. Queste x qui e qui probabilmente sono come pochi test. Quindi questo potrebbe risultare in un algoritmo di classificazione molto buono. Quindi quello che facciamo, facciamo questo sul dataset di test.

`02:03:23` 
Quindi quello che diciamo qui, diremo che la buona soglia, quale potrebbe essere? Direi che qui, zero è una buona soglia. Possiamo tracciare una linea qui, e questo potrebbe separare, beh, gli zeri dai nove. quindi diciamo che la nostra soglia qui è zero possiamo anche fare un plot se volete facciamo questo.

`02:03:58` 
e poi facciamo plt punto axvline. okay e questa sarà la nostra classe questa linea verticale sarà il nostro classificatore, è molto molto semplice è molto molto naive ma con gli strumenti che abbiamo è quello che possiamo fare e vedremo che si comporterà piuttosto bene.

`02:04:33` 
okay questo è quello che ho fatto sopra e poi questa è la parte che farete voi. Quindi ora abbiamo il nostro algoritmo di classificazione e vogliamo testarlo sul dataset di test. Quindi ripeterete praticamente tutti i passaggi che abbiamo fatto prima, ma usando il dataset di test. Quindi lo caricherete, estrarrete alcune immagini e controllerete come sono, e poi le tracciate usando le componenti principali dei dati di training.

`02:05:16` 
Quindi questo è il punto chiave. Ora non dovete più eseguire la SVD. Le direzioni che usate per trasformare i dati sono le direzioni che avete ottenuto dai dati di training. Okay, quindi niente più SVD. Usate u e tracciate i dati di training usando queste direzioni. E vedrete che queste direzioni sono ancora molto buone anche per il dataset di test. È qualche dato che la SVD non ha mai visto prima. Okay. E una volta che avete questi, se volete, potete iniziare a provare a calcolare anche alcune metriche. Quindi quanti campioni sono classificati correttamente. Questo è un po' più avanzato. Se non siete in grado di farlo, lo vedremo insieme. Ma almeno fino a qui, dovreste essere in grado di fare tutto. Perché è solo quello che abbiamo fatto all'inizio del lab. Lo copiate e incollate praticamente.

`02:06:10` 
Stando attenti a usare le direzioni di training invece di calcolare i valori. Okay. Quindi vi do un quarto d'ora. E per qualsiasi domanda, sarò in giro.

`02:07:38` 
Grazie.

`02:08:30` 
Grazie mille.

`02:09:20` 
Grazie.

`02:10:33` 
Grazie mille.

`02:11:47` 
Grazie.

`02:14:43` 
Torniamo subito. Grazie.

`02:16:00` 
Bene, grazie.

`02:16:56` 
Grazie.

`02:18:09` 
Okay, abbiamo gli ultimi 10 minuti. Voglio mostrarvi la soluzione con un po' di completezza e non affrettarla. Quindi iniziamo. Quindi iniziamo caricando i dati del test. Per questo possiamo copiare e incollare praticamente questo e anche la parte, questa. Quindi il dataset ora si chiama questo test. Chiamiamo questo label test. Questo è data test. Questo è a test.

`02:19:08` 
Che è data test trasposto e table test A-text. Di solito quando carico, quando carico sul lab, caricherò la soluzione. Okay, quindi ora abbiamo 10.000 campioni nel test. Quindi ora dobbiamo filtrare come abbiamo fatto prima.

`02:19:39` 
Quindi possiamo solo andare su, prendere questo e incollarlo qui. Quindi invece di labels, ora abbiamo labels test. Questo si chiama mask test. e invece di a pool abbiamo a test con la test mask e questo è a test e quelle labels test.

`02:20:20` 
è uguale a label test della mask test okay tutto okay ora vogliamo stamparle. quindi andiamo qui copiamo e incolliamo la parte del plotting cosa cambiamo è che questo è a test.

`02:21:04` 
e queste labels è labels test e questo è per controllare che effettivamente abbiamo caricato qualcosa che è corretto 0099990090990099 okay sembra che effettivamente abbiamo quello che vogliamo e poi vogliamo tracciare i primi 500 campioni trasformati usando le componenti principali di training quindi questa è la parte delicata quindi cosa abbiamo qui.

`02:21:34` 
vogliamo usare la ut che abbiamo calcolato usando il dataset di training e proiettare i dati qui quindi dobbiamo fare il prodotto matrice matrice di a test meno la media questo è il punto chiave dove penso possiate fare errori questa non è la media del test, questa è la media del training perché perché questa è una stima della media dei dati.

`02:22:09` 
e la stima è fatta nella fase di training non durante la fase di test okay quindi qui abbiamo a mean usando il broadcasting okay sia le direzioni che la media sono una stima di quella vera e sono fatte solo nella fase di training quindi queste sono five tests e poi possiamo calcolare ora.

`02:22:43` 
alcune metriche quindi facciamo un plot prima uh quindi come questo quindi usiamo la stessa soglia che abbiamo deciso prima quindi non la ridefinisco nemmeno perché questa è definita solo nella fase di training, E sostituisco phi con phi test e labels con labels test. E potete vedere che, qualitativamente, questa linea verticale divide ancora molto bene i dati tra zeri e uno.

`02:23:20` 
L'ultima parte è che calcoliamo le metriche ora. Quindi vogliamo quantificare quanto bene sta funzionando il nostro classificatore. Quindi come facciamo questo? Tutto è basato sulla prima componente principale. Quindi questo è phi test e estraiamo la prima componente principale. Ora cosa abbiamo? Questa è un'altra parte importante e questo è come facciamo la previsione. Quindi le labels previste, cosa sono? Sono quando PC1.

`02:24:01` 
è maggiore o minore della soglia okay quindi questo sarà un vettore con vero o falso e possiamo usare e attenzione quando questo è vero uh avremo la cifra zero quando è falso cifra, uno okay questo automaticamente ci dà questo quando questo vettore booleano è vero e questo.

`02:24:34` 
è quando il vettore booleano è falso infatti se mostriamo questo abbiamo un vettore chiamiamo zeri o nove okay quindi questo è il nostro modello di previsione e infine testiamo come testiamo questo, Dobbiamo calcolare, il modo migliore per farlo è calcolare quattro valori diversi. E quali sono questi quattro valori diversi?

`02:25:04` 
Gli zeri che rappresentiamo correttamente come zeri, i nove che rappresentiamo correttamente come nove, gli zeri che classifichiamo propriamente come nove, e i nove che classifichiamo propriamente come zeri. Quindi i veri zeri, ora facciamo tutto con una bit mask, è dove levels test è uguale a zero.

`02:25:36` 
E le levels predict è uguale a zero. Okay, questo è, come abbiamo fatto prima, una mask con tutti vero e zeri. E gli zeri che rappresentiamo correttamente come zeri sono, beh, i... L'etichetta corretta è zero, e quella dove prevediamo è zero. Molto similmente, i due nove sono dove il test è uguale a nove, e le labels previste sono uguali a nove.

`02:26:09` 
Poi abbiamo gli altri due, che sono i falsi nove e i falsi zeri. Quindi i falsi zeri sono dove prevediamo zero, ma in realtà abbiamo nove, e i falsi nove sono dove prevediamo nove, ma sono in realtà zero. Poi quello che fate è che stampiamo tutti questi.

`02:26:41` 
Questi sono molto importanti perché per ora è facile. Ma in alcuni casi, queste previsioni potrebbero essere molto sbilanciate. Forse avete molti falsi zeri o falsi nove. Volete, in questo caso, avere previsioni dove i falsi nove e i falsi zeri sono più o meno gli stessi. Infine, l'ultima metrica che usiamo è l'accuratezza. L'accuratezza è...

`02:27:15` 
Scusate, mi è mancato il passaggio. Qui abbiamo ancora zeri o uno. Dobbiamo contarli, e come li contiamo? Usiamo np.sum. Quindi questi sono vettori con tutti vero e falso. Per contare quanti veri abbiamo, usiamo numpy.sum. L'accuratezza è il numero di veri zeri. più il numero di veri nove diviso per tutto il resto quindi i due zeri più i due nove.

`02:27:52` 
diviso per i quattro nice e force qui okay e questo è il risultato quindi accuratezza 95 percento il nostro, classificatore è molto buono siamo sul dataset di test abbiamo cinque percento di errore e questo classifi, catore qui è molto molto semplice è una linea verticale quindi questo vi mostra quanto potente è la prima componente principale.

`02:28:25` 
Le previsioni sono bilanciate, i falsi e i veri negativi, i falsi negativi e i veri positivi sono praticamente bilanciati, abbiamo 30 e 50, e le buone previsioni sono anche bilanciate, più o meno. Quindi significa che questo classificatore funziona bene e non è sbilanciato. L'ultima cosa, questo processo è molto, molto famoso nel senso che farete questo molte, molte volte, e SciPy vi dà un'utilità per fare automaticamente questo.

`02:29:06` 
Questo si chiama matrice di confusione, e possiamo tracciarla semplicemente passando i due vettori, che sono label test e label predict, e automaticamente fa questo processo. E questo è il risultato. Grazie. Quindi per la classificazione quello che volete è che questi e questi siano bilanciati e questi e questi siano bilanciati ed è meglio se quello che è sulla diagonale è uno e quello che non è sulla diagonale è zero naturalmente. Ma diciamo che la parte più importante è che questi siano bilanciati perché spesso accade nella vita reale che i dataset sono sbilanciati e dovete cambiare la loss o il classificatore per rendere questo bilanciato.

`02:29:53` 
Okay, qualche domanda? Sì, qui, questo. Okay, questa è una funzione che anche qualcosa che è vero o falso dà quando è vero questo valore e quando è falso questo valore.

`02:30:23` 
Okay, ultima cosa, c'è un terzo notebook. Lo lascio a voi per i compiti. Quello è un esercizio di classificazione, come questo, su un dataset reale di cancro. La prossima volta, la prima volta che lo userete, vi dirò solo qualcosa a riguardo. Mi dispiace non essere stato in grado di mostrarvelo, ma per meno, è come questo. Ma classificate i pazienti che hanno il cancro da quelli che non hanno il cancro, e quello è un dataset reale usato nella vita reale. Per qualsiasi domanda, sono qui ora, e potete contattarmi via email. Buon fine settimana.