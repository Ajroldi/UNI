## Introduzione al laboratorio e obiettivi della giornata
[00:00] È il terzo laboratorio. La giornata prevede circa quindici minuti per riprendere il terzo esercizio della sessione precedente, non mostrato in aula, passando attraverso la soluzione e riepilogando i concetti principali della PCA (Principal Component Analysis). La PCA è una tecnica di riduzione dimensionale che proietta i dati su direzioni di massima varianza, semplificando la visualizzazione e l’applicazione di modelli.
[00:20] Sono previsti due notebook sugli argomenti trattati. Il primo riguarda la regressione ai minimi quadrati, la regressione ridge e la regressione con kernel. Il secondo è dedicato all’algoritmo PageRank di Google per la valutazione delle connessioni nei grafi.
[00:40] Il primo esercizio utilizza un dataset realistico tratto da due pubblicazioni. Per alcuni pazienti sono stati misurati biomarcatori (quantità osservabili) e l’etichetta target indica se il paziente svilupperà il cancro nel corso della vita, distinguendo tra pazienti sani e con malattia. L’obiettivo è valutare la relazione tra variabili misurabili e la probabilità di sviluppare la malattia.
[01:00] Si collega l’informazione misurata sul paziente alla possibilità di sviluppare la malattia. Operativamente, si carica un dataset e si ottiene una matrice $A$ con righe pari al numero di feature (biomarcatori) e colonne pari al numero di pazienti; ogni colonna è un campione.
[01:20] La stampa di $A$ produce una matrice di coefficienti che quantificano la presenza dei biomarcatori. È disponibile anche un array GRB dove ogni elemento indica se il paziente è sano o meno, rappresentato come stringa. Per analisi numeriche è preferibile convertire le etichette in valori numerici.
[01:40] Si conta quanti pazienti hanno il cancro e quanti sono sani. Poiché il problema è di classificazione binaria, è importante il bilanciamento del dataset: un numero comparabile di campioni per ciascuna classe riduce il rischio di bias in addestramento.
[02:00] Se il dataset non è bilanciato, può essere necessario un pre-processing per riequilibrarlo, altrimenti lo squilibrio si riflette in bias nel modello. Per classificazioni binarie si desiderano numeri simili per le etichette 0 e 1.
[02:20] In questo caso ci sono 120 pazienti con cancro e circa 100 sani, un bilanciamento accettabile. Si può creare un vettore booleano “is\_same”, sostituendo le stringhe con 0 o 1. L’uso di numeri facilita la manipolazione, preservando il significato.
[02:40] Si impiega la funzione where, già vista in precedenza, per trasformare l’array booleano “group == normal” in 1 se vero e 0 altrimenti. Si procede con una esplorazione dei dati per comprendere struttura e correlazioni del dataset.
[03:00] Si selezionano due proteine a caso (ad esempio ID 0 e ID 1) e si produce uno scatter plot delle variabili. È difficile osservare una correlazione informativa tra due proteine scelte casualmente.
[03:20] Richiamando un esempio sui digit, dopo la PCA si poteva distinguere visivamente zero e nove; qui, prima della PCA, lo scatter di due proteine non permette di individuare chiaramente le categorie.
[03:40] Qualsiasi coppia di proteine può essere scelta (ad esempio la 99 e un’altra), ma il significato interpretativo resta debole: senza PCA, cercare correlazioni casuali è inefficace.
[04:00] La PCA fornisce una trasformazione adatta a visualizzare i dati e a rendere più efficace l’applicazione dei modelli, evitando tentativi manuali su tutte le coppie di variabili. Anche in 3D la distribuzione dei punti può risultare concentrata e la separazione tra categorie poco visibile.
[04:20] Per i grafici 3D con Matplotlib si usa la proiezione 3D e nello scatter si passano $x$, $y$, $z$ invece di $x$, $y$. Verrà mostrata anche una libreria più adatta in alcuni contesti.
## PCA: preparazione, SVD e interpretazione delle componenti
[04:40] Si applica la PCA. Per la PCA è utile un dataset centrato a media zero: si calcola la media di $A$ lungo le colonne (ogni colonna è un campione) e la si sottrae da $A$ ottenendo $A^{\text{bar}}$ a media zero. La centratura elimina contributi di traslazione e mette in evidenza la struttura di varianza.
[05:00] Si definisce la media come $\mu = A.\text{mean}(\text{axis}=1)$ e la matrice centrata come $A^{\text{bar}} = A - \mu$. Si applica la SVD: $A^{\text{bar}} = U \Sigma V^\top$, dove $U$ contiene le direzioni principali (autovettori nello spazio delle feature), $\Sigma$ i valori singolari (radici della varianza lungo le direzioni), e $V$ i coefficienti dei campioni nella base $U$.
[05:20] Si tracciano i valori singolari e la frazione cumulativa di varianza spiegata. I primi valori singolari sono molto rilevanti; il primo domina nettamente. La curva dei valori singolari mostra un “gomito”, cioè un punto di flesso che suggerisce una soglia utile per scegliere il numero di componenti.
[05:40] La regola del gomito: si seleziona una soglia approssimativa dove la curva piega. Qui un cut-off intorno a 25 componenti è ragionevole.
[06:00] Si considerano le direzioni principali e si proietta il dataset su di esse. Il prodotto $A^{\text{bar}}$ con le prime due colonne di $U$ produce le prime due componenti principali per ciascun campione. Queste componenti sono coordinate nei nuovi assi di massima varianza.
[06:20] Lo scatter delle prime due componenti mostra punti ancora raggruppati e una distinzione parzialmente complessa tra sani e malati, ma emerge un comportamento complessivo: anche con due componenti principali si può iniziare a visualizzare quali pazienti potrebbero essere sani o a rischio.
[06:40] Una separazione con una linea verticale è inadatta: la distribuzione nelle due componenti non è separata nettamente a sinistra e destra; il classificatore lineare semplice sarebbe insufficiente.
[07:00] Si propone una strategia più sofisticata: proiezione in 3D e separazione planare. Si introduce una libreria di visualizzazione alternativa.
## Visualizzazione 3D interattiva e separazione planare
[07:20] Si utilizza Plotly, libreria interattiva utile per esplorazione. È più pesante in memoria e leggermente più complessa, ma può semplificare la visualizzazione. Si usa “scatter3d” passando $x$, $y$, $z$ e il colore delle etichette.
[07:40] Il grafico 3D è interattivo: si può ruotare la vista e, posizionando il cursore, identificare se un punto corrisponde a un paziente con cancro o normale. La legenda è cliccabile per escludere serie. Plotly è frequentemente impiegata in applicazioni web.
[08:00] Osservando i dati in 3D, si ipotizza un piano di separazione efficace fra pazienti con cancro e normali. Si può applicare una tecnica di classificazione che traccia un piano nello spazio 3D e assegna le etichette in base al lato in cui cadono i campioni.
[08:20] L’approccio sarà approfondito successivamente. L’idea è cercare un piano che separi al meglio i punti rossi dai blu e classificare i campioni in base al lato del piano. Questa idea generalizza la separazione lineare da una soglia a una superficie planare.
[08:40] L’implementazione è disponibile in scikit-learn. Si raggiunge una QLC dell’83%: il classificatore, insieme alla PCA, stima il rischio di cancro con qualità dell’83%, un valore ragionevole potenzialmente utilizzabile con ulteriori affinamenti.
[09:00] Si esaminano le componenti principali e i loro pesi sui biomarcatori. Considerando le prime tre componenti, la prima mostra proteine con correlazioni forti positive e negative, altre trascurabili. Le componenti principali indicano combinazioni lineari di biomarcatori che spiegano la varianza.
[09:20] Serve competenza di dominio per interpretare il significato biologico delle correlazioni. L’analisi consente di identificare proteine rilevanti per il compito e altre di scarsa importanza.
[09:40] Il notebook con le soluzioni è disponibile su WeBip. Si passa al notebook sulla regressione.
## Regressione: impostazione del notebook e obiettivi
[10:00] Su WeBip, cartella “lab03”, è disponibile il notebook 1 “least square regression, kernel regression”. Si consiglia di aprire Colab e caricare il notebook per implementare il codice.
[10:20] Il primo compito è eseguire una regressione ai minimi quadrati usando la pseudo-inversa di Moore–Penrose, calcolata tramite SVD. Si implementa la pseudo-inversa basata sulla SVD.
[10:40] Si può scegliere tra SVD “full” ($\text{full\_matrices} = \text{True}$) e SVD “thin” ($\text{full\_matrices}=\text{False}$). Si confronta il risultato con la funzione standard che calcola la pseudo-inversa, verificando una differenza dell’ordine di $10^{-15}$.
[11:00] Si eseguono misure dei tempi di esecuzione, confrontando la pseudo-inversa standard con la versione via SVD, in particolare l’implementazione ottimizzata. La valutazione temporale aiuta a capire l’efficienza delle scelte implementative.
[11:20] Il codice richiesto è conciso: poche linee basate sulla formula matematica della pseudo-inversa. Si dedica tempo all’implementazione.
## Implementazione della pseudo-inversa via SVD
[11:40] Per la versione full, si esegue la SVD come: $U, s, V^\top = \text{svd}(A, \text{full\_matrices}=\text{True})$, dove $s$ contiene i valori singolari e $U$, $V^\top$ sono matrici ortogonali. La SVD decompone $A$ in rotazioni ($U$, $V$) e scalature ($\Sigma$).
[12:00] Per invertire $\Sigma$, si considerano solo i valori singolari maggiori di zero: gli zero non sono invertibili. Essendo $\Sigma$ diagonale, l’inversione consiste nel prendere i reciproci degli elementi diagonali. Si costruisce $\Sigma^\dagger$ con $1/s_i$ per $s_i>0$ e 0 altrimenti.
[12:20] La pseudo-inversa di Moore–Penrose è:
```math
A^\dagger = V \,\Sigma^\dagger\, U^\top.
```
Questa espressione deriva da $A = U \Sigma V^\top$ e dalla proprietà che $U$ e $V$ sono ortogonali, mentre $\Sigma$ è diagonale con valori singolari non negativi.
[12:40] La differenza con la versione thin riguarda $\text{full\_matrices}=\text{False}$. L’inversione della diagonale segue lo stesso principio. Con matrici non quadrate, si possono ottimizzare i prodotti evitando di costruire esplicitamente la matrice diagonale e sfruttando il broadcasting.
[13:00] La verifica mostra che la pseudo-inversa calcolata coincide con quella della funzione standard, con errore numerico dell’ordine di $10^{-16}$, compatibile con l’epsilon di macchina.
[13:20] Si misurano i tempi con “%timeit”: molte ripetizioni e report di media e deviazione standard. L’implementazione SVD thin con prodotti ottimizzati è circa due volte più veloce della versione di default, a parità di condizioni.
[13:40] Uno studio di scaling rispetto alla dimensione di $A$ darebbe conferme ulteriori, ma ci si attende differenze non drammatiche. La buona vettorializzazione e il broadcasting possono superare implementazioni standard in alcuni casi.
## Regressione ai minimi quadrati: modello lineare e stima dei parametri
[14:00] Si affronta la regressione ai minimi quadrati con dati sintetici. Si parte da un modello lineare con pendenza $m=2$ e intercetta $q=3$. La regressione ai minimi quadrati stima i parametri che minimizzano l’errore quadratico tra predizioni e osservazioni.
[14:20] Si generano $N=100$ punti per la variabile indipendente $x$ da una distribuzione gaussiana. Si aggiunge un rumore gaussiano $\varepsilon$ di intensità controllata alla variabile dipendente $y$.
[14:40] La relazione osservata è:
```math
y = m x + q + \varepsilon,
```
dove $\varepsilon$ è rumore gaussiano. Maggiore $\varepsilon$ provoca dispersione dei punti rispetto alla retta generatrice.
[15:00] Si calcola $y$ dai parametri noti e dal rumore, quindi si applica la regressione ai minimi quadrati per stimare $m$ e $q$, confrontando le stime con i valori reali.
[15:20] Si definisce la matrice di progetto $\Phi \in \mathbb{R}^{N \times 2}$:
```math
\Phi = \begin{bmatrix}
x_1 & 1 \\
x_2 & 1 \\
\vdots & \vdots \\
x_N & 1
\end{bmatrix},
```
dove la prima colonna è $x$ e la seconda è di 1. I parametri sono $w = [m, q]^\top$.
[15:40] Il problema ai minimi quadrati è:
```math
\Phi w \approx y.
```
Il sistema è sovradeterminato, quindi non ha soluzione esatta per tutti i punti. La pseudo-inversa di Moore–Penrose dà la soluzione nel senso dei minimi quadrati.
[16:00] La soluzione è:
```math
w = \Phi^\dagger y,
```
con $\Phi^\dagger$ calcolata via SVD come descritto. Il vettore $w$ contiene le stime di $m$ e $q$.
[16:20] Si verifica la vicinanza delle stime ai valori reali $m=2$ e $q=3$. Ad esempio, si può ottenere $m \approx 2{,}2$ e $q \approx 3{,}1$, coerente con il rumore.
[16:40] Per visualizzazione si traccia lo scatter $(x,y)$, la retta generatrice in rosso (con $m$ e $q$ reali) e la retta stimata in nero (con $w$). Il confronto mostra l’efficacia del fitting.
[17:00] Un rumore più basso (ampiezza 0.1) riduce la discrepanza; un rumore più alto aumenta la dispersione e rende la stima più impegnativa.
[17:20] Si prepara una valutazione su dati di test: si definisce $x_{\text{test}}$ come griglia uniforme (ad esempio in $[-3,3]$ con 1000 punti), si costruisce $\Phi_{\text{test}}$ e si calcola:
```math
y_{\text{pred}} = \Phi_{\text{test}}\, w.
```
La retta stimata è valutata su punti diversi da quelli di addestramento.
[17:40] Si verifica la coerenza delle dimensioni tra vettori e matrici, adattando le forme per consentire i prodotti matriciali corretti.
[18:00] Il risultato finale mostra la nuvola dei dati di training, la retta generatrice e la retta stimata; sui dati di test, la retta stimata viene valutata e rappresentata graficamente.
## Aumento dei campioni e impatto su accuratezza
[00:00] Se il numero di punti aumenta da 100 a 1000, i punti rimangono allineati lungo una direzione quasi rettilinea, ma sono molti di più. Le stime di $m$ e $q$ migliorano: la linea della predizione e quella del modello reale risultano più vicine. All’aumentare dei campioni cresce l’accuratezza del modello, anche con dispersione dovuta al rumore.
[00:25] Un altro fattore che incrementa l’accuratezza, solitamente non controllabile, è la riduzione del rumore: mantenendo 100 punti, se il rumore è molto piccolo è più facile ottenere una predizione corretta. In questo caso le stime di $m$ e $q$ possono risultare $2.0$ e $3.0$.
[00:45] Esiste un compromesso: più rumore rende più difficile l’estimazione, più punti la migliora. Questo spiega l’importanza di dataset estesi: con molti dati, modelli lineari e modelli complessi tendono a funzionare meglio.
[01:10] Si torna al livello di rumore precedente, fissato a $2$.
## Equazioni normali e risoluzione del sistema
[01:20] In alternativa alla pseudo-inversa, si possono risolvere le equazioni normali:
```math
(\Phi^\top \Phi)\, w = \Phi^\top y,
```
dove $\Phi$ è la matrice delle caratteristiche, $w$ è il vettore dei parametri e $y$ è il vettore dei valori osservati. Le equazioni normali derivano dalla minimizzazione dell’errore quadratico medio.
[01:40] Il sistema è quadrato e risolvibile con un risolutore lineare, passando come primo argomento $\Phi^\top \Phi$ e come secondo $\Phi^\top y$, ottenendo $w$.
[02:00] Non si calcola la pseudo-inversa completa: si utilizzano decomposizioni per matrici quadrate (ad esempio LU). Il risultato per $w$ è praticamente identico a quello con la pseudo-inversa, con differenze dell’ordine dell’epsilon di macchina.
[02:20] Vantaggi: si calcolano solo i prodotti $\Phi^\top \Phi$ e $\Phi^\top y$, rendendo l’approccio adatto a generalizzazioni verso altre regressioni.
## Regressione “ridge” su dati non lineari
[02:40] Si introduce una regressione su un fenomeno non lineare mantenendo un modello lineare. Il fenomeno ha forma simile a una sigmoide. Si definisce una funzione $f$, si generano punti $x$ gaussiani standard, si calcola il modello reale $y$, si aggiunge rumore $\varepsilon$, e si visualizzano i punti rumorosi con il modello reale.
[03:05] Applicando i minimi quadrati, la prestazione è scarsa. Si introduce la ridge regression, aggiungendo un termine di regolarizzazione $\lambda$ che penalizza pesi ampi. La regolarizzazione riduce l’overfitting e stabilizza la soluzione.
[03:25] Variando $\lambda$ cambia il comportamento del modello: con un valore adatto di $\lambda$ il modello lineare migliora leggermente, pur rimanendo non ottimale per dati fortemente non lineari. Si introduce successivamente la regressione con kernel.
[03:45] Compito operativo: generare il modello reale, aggiungere rumore, applicare i minimi quadrati e introdurre la penalizzazione ridge. Due vie:
- modificare le equazioni normali aggiungendo la penalizzazione;
- usare l’identità di Woodbury per riscrivere il problema.
[04:05] Il procedimento con Woodbury prevede di calcolare prima un vettore $\alpha$, poi ricavare i pesi $w$ e produrre il grafico risultante. Si mantiene la stessa struttura, cambiando i dati e usando l’identità per riformulare l’equazione.
## Generazione dei dati e modello reale sigmoide
[04:30] Si generano $x_i$ gaussiani standard, con 100 punti. Il valore reale $y$ è:
```math
y = \tanh(2x - 1).
```
Questa funzione ha forma a “S” appiattita (sigmoide). Lo scatter $x$ contro $y$ mostra la curva a “S”.
[04:55] Si definisce $y$ rumorosa:
```math
y_{\text{rumorosa}} = y_{\text{reale}} + \varepsilon,
```
dove $\varepsilon$ è rumore gaussiano con deviazione standard $0{,}1$. La visualizzazione mostra la forma a “S” con dispersione.
[05:15] Per la curva reale continua, si definisce $x_{\text{test}}$ uniforme in $[-3,3]$ con 1000 punti e si calcola:
```math
y_{\text{test}} = \tanh(2x_{\text{test}} - 1).
```
La linea continua rappresenta il comportamento reale su un intervallo ampio.
## Minimi quadrati con base lineare
[05:40] Si costruisce la matrice delle caratteristiche lineare:
```math
\Phi = \begin{bmatrix}
x & \mathbf{1}
\end{bmatrix},
```
dove la seconda colonna è di 1 per l’intercetta. La stima con pseudo-inversa è:
```math
w = \Phi^{+} y,
```
con $w = (m, q)^\top$.
[06:00] Si definisce anche $\Phi_{\text{test}}$ per $x_{\text{test}}$:
```math
\Phi_{\text{test}} = \begin{bmatrix}
x_{\text{test}} & \mathbf{1}
\end{bmatrix}.
```
La predizione sul test è $y_{\text{pred}} = \Phi_{\text{test}} w$.
[06:20] Il modello lineare non è adatto a una forma a “S”: il fit minimizza la distanza dai punti con maggior densità, rimanendo vicino alle aree più popolate, senza catturare la non linearità complessiva.
## Ridge regression: formulazione con equazioni normali
[06:40] Si riscrive il problema introducendo la penalizzazione sui pesi. Le equazioni normali in versione ridge sono:
```math
(\Phi^\top \Phi + \lambda I)\, w = \Phi^\top y,
```
dove $I$ è l’identità di dimensione pari al numero di parametri. Il termine $\lambda I$ penalizza pesi elevati.
[07:00] Risolvendo il sistema si ottiene $w$ e si calcola $y_{\text{pred}} = \Phi_{\text{test}} w$. L’impatto di $\lambda$: all’aumentare di $\lambda$, $m$ e $q$ vengono penalizzati di più.
[07:20] Se $\lambda$ è molto grande, il modello preferisce pendenza e intercetta piccole. Per $\lambda = 10000$, $m$ e $q$ sono quasi nulli; per $\lambda = 1000$ sono più grandi ma ancora piccoli; per $\lambda = 1$ il modello è simile a quello senza regolarizzazione.
[07:40] La penalizzazione dei pesi è diffusa anche nelle reti neurali. In questo contesto lineare, il termine regolarizzante riduce la complessità privilegiando pesi piccoli.
## Ridge regression: identità di Woodbury
[08:00] Si riscrive la soluzione usando l’identità di Woodbury, sostituendo il sistema basato su $\Phi^\top \Phi$ con uno basato su $\Phi \Phi^\top$.
[08:20] Si definisce $\alpha$ tale che:
```math
(\Phi \Phi^\top + \lambda I_n)\, \alpha = y,
```
dove $I_n$ è l’identità $n \times n$ e $n$ è il numero di campioni. Trovata $\alpha$, si ricava:
```math
w = \Phi^\top \alpha.
```
Questa soluzione è algebricamente equivalente alla ridge precedente.
[08:40] La differenza tra le due soluzioni per $w$ è dell’ordine dell’epsilon di macchina ($\approx 10^{-15}$). Se $\lambda=0$, la versione standard con equazioni normali funziona, mentre la variante con $\Phi \Phi^\top$ non è equivalente: l’identità di Woodbury richiede $\lambda I$ non nullo per mantenere l’equivalenza.
[09:00] Le due formule coincidono per ogni $\lambda > 0$ e divergono se $\lambda = 0$.
## Introduzione alla regressione con kernel
[09:20] Il termine $\Phi \Phi^\top$ può essere sostituito con una matrice kernel $K$ costruita applicando una funzione kernel ai punti. Se il kernel coincide con il prodotto scalare standard, si recupera $\Phi \Phi^\top$.
[09:40] Cambiando $K$ e introducendo non linearità nel kernel (ad esempio polinomiale o gaussiano), si mantiene il costo di una regressione lineare risolvendo un problema lineare, ma si ottiene la capacità di approssimare fenomeni non lineari.
[10:00] Operativamente, si risolve:
```math
(\Phi \Phi^\top + \lambda I_n)\, \alpha = y
```
sostituendo $\Phi \Phi^\top$ con $K$. Il flusso rimane analogo: si calcola $\alpha$ e poi la predizione.
[10:20] Procedura consigliata: inizializzare $K$ di dimensione $n \times n$ e calcolare ogni $K_{ij}$ con un doppio ciclo applicando la funzione kernel prescelta. Successivamente, sostituire il doppio ciclo con una versione vettorializzata.
## Impostazione della regolarizzazione e parametri iniziali
[00:00] Si definisce la regolarizzazione $\lambda$. Si fissano parametri $q=4$ e $\sigma=1$, da modificare in seguito per osservare l’influenza sul modello. La regolarizzazione controlla l’ampiezza dei parametri, riducendo la complessità ed evitando sovradattamento.
[00:20] L’obiettivo è impostare il modello di regressione con kernel in modo ordinato, esaminando i passaggi chiave.
## Definizione dei kernel
[00:40] Si definiscono funzioni kernel che prendono in input due scalari $x_i$ e $x_j$ e restituiscono un valore.
[00:55] Kernel prodotto (lineare):
```math
k_{\text{lin}}(x_i, x_j) = x_i x_j + 1.
```
Il termine $+1$ introduce una componente costante, equivalente a un bias.
[01:15] Kernel polinomiale di ordine $q$:
```math
k_{\text{poly}}(x_i, x_j) = (x_i x_j + 1)^q.
```
L’elevazione alla potenza consente di modellare relazioni non lineari fino al grado $q$.
[01:35] Kernel gaussiano (RBF):
```math
k_{\text{RBF}}(x_i, x_j) = \exp\!\left(-\frac{(x_i - x_j)^2}{2\sigma^2}\right).
```
La distanza $|x_i-x_j|$ è normalizzata da $\sigma$ e controlla il raggio d’influenza del contributo gaussiano.
## Regressione con kernel: struttura generale
[01:55] Si definisce una funzione di regressione con kernel che accetta la funzione kernel come argomento. Il dataset di training ha dimensione $n = X.\text{shape}[0]$. Si inizializza la matrice del kernel $K \in \mathbb{R}^{n \times n}$.
[02:30] Si riempie $K$ con un doppio ciclo su $i$ e $j$:
```math
K_{ij} = k(x_i, x_j).
```
Si costruisce così la Gram matrix del kernel.
## Calcolo dei coefficienti alfa
[02:55] Calcolata $K$, si determinano i coefficienti $\alpha$ risolvendo il sistema regolarizzato:
```math
\alpha = (K + \lambda I)^{-1} y,
```
dove $I$ è l’identità $n \times n$ e $y$ è il vettore dei valori di training.
[03:20] In questa forma non si calcola $w$ nello spazio implicito del kernel; si usa direttamente $K$ e $\alpha$ per la predizione.
## Valutazione sul test: costruzione del kernel di test
[03:45] Per valutare il modello su dati di test si costruisce $K_{\text{test}}$ tra punti di test e training:
```math
(K_{\text{test}})_{ij} = k(x^{\text{test}}_i, x^{\text{train}}_j).
```
La prima input è il punto di test, la seconda il punto di training.
[04:05] $K_{\text{test}}$ ha forma (numero di test) $\times$ (numero di training). Si riempie con un doppio ciclo su indici $i$ (test) e $j$ (training).
[04:25] La previsione sui dati di test è:
```math
\hat{y}_{\text{test}} = K_{\text{test}} \,\alpha.
```
Si usa lo stesso kernel e gli stessi parametri del training per costruire $K_{\text{test}}$.
## Visualizzazione: confronto con regressione lineare
[04:50] Si traccia lo scatter di $x$ e $y$ e la linea di riferimento $x_{\text{test}}$ contro $y_{\text{test}}$ (funzione vera). La predizione con kernel lineare ($k_{\text{lin}}$) coincide con la regressione lineare classica con bias:
```math
k_{\text{lin}}(x_i, x_j) = x_i x_j + 1 \quad \Rightarrow \quad \text{ipotesi lineare con bias}.
```
Questo conferma la coerenza tra le due formulazioni.
## Variazione del kernel: polinomiale di grado 4
[05:25] Si adotta un kernel polinomiale di grado $q=4$:
```math
k_{\text{poly}}(x_i, x_j) = (x_i x_j + 1)^4.
```
La predizione risulta più aderente ai dati quando questi presentano non linearità significative.
[05:40] Il modello assomiglia a un polinomio di grado quattro, fornendo un adattamento più flessibile rispetto alla forma lineare.
## Scelta del grado polinomiale e sovradattamento
[05:55] Variando $q$ si osservano effetti diversi: con $q=2$ si ottiene una parabola, spesso non ideale. Con $q=200$ si osserva sovradattamento: il modello si adatta quasi perfettamente ad alcuni punti, ma introduce oscillazioni marcate, tipiche dei polinomi di alto ordine.
[06:10] Il sovradattamento si manifesta nella perdita della forma complessiva della funzione pur adattando punti specifici molto bene. Una scelta moderata come $q=4$ o $q=5$ bilancia flessibilità e stabilità.
## Kernel gaussiano: interpretazione e effetto di sigma
[06:50] Il kernel gaussiano modella contributi locali attorno ai punti:
```math
k_{\text{RBF}}(x_i, x_j) = \exp\!\left(-\frac{(x_i - x_j)^2}{2\sigma^2}\right).
```
$\sigma$ controlla il raggio d’influenza: più grande è $\sigma$, più ampia è la regione di “condivisione” di informazione.
[07:05] Effetto di $\sigma$:
- $\sigma$ grande: informazione condivisa su raggio ampio; gaussiane larghe; superficie liscia.
- $\sigma$ piccolo: influenza limitata; gaussiane strette; funzione più frammentata con picchi localizzati.
[07:20] Si considera anche l’effetto di $\lambda$. Con $\lambda$ molto grande si penalizzano fortemente i parametri, tendendo ad appiattire il modello; con $\lambda$ molto piccolo si consente maggiore complessità. Un caso descritto mostra una funzione quasi orizzontale, segno di regolarizzazione molto forte nella soluzione numerica.
[07:40] Con $\sigma$ molto piccolo, si ottiene un fitting locale con picchi centrati sui punti. Ogni punto contribuisce con una gaussiana stretta.
[07:55] Aumentando $\sigma$ (ad esempio $\sigma = 0{,}3$), i raggi delle gaussiane crescono e l’informazione si estende su intervalli più ampi, producendo maggiore regolarità. Le gaussiane si sovrappongono e si ottiene una stima più liscia.
[08:15] Per un valore “buono” di $\sigma$, alcune gaussiane si fondono e il modello liscia la funzione, aderendo bene ai dati. La scelta di $\sigma$ è parte del tuning degli iperparametri.
[08:30] Un compromesso suggerito è $\sigma \approx 0{,}5$, che bilancia fit locale e influenza globale. Se $\sigma$ è eccessivamente grande (ad esempio $\sigma = 100$), l’effetto è simile a una singola gaussiana ampia: si aggregano le informazioni di tutti i punti e, se la media è prossima allo zero, il modello collassa su una stima costante.
## Scelte pratiche e tuning degli iperparametri
[08:50] La selezione di $\sigma$ è critica: un valore troppo piccolo porta a eccessiva località e possibile rumore; uno troppo grande porta a un collasso verso una stima costante. Occorre scegliere $\sigma$ con attenzione.
[09:05] Il tempo è limitato, si procede rapidamente. Sono disponibili soluzioni negli homework, utili per apprendere la tecnica.
[09:20] Un passaggio successivo è reimplementare tutto usando operazioni vettorializzate per migliorare efficienza rispetto ai doppi cicli in Python.
## Considerazioni comparative: Gaussiane vs Polinomi
[09:35] Le gaussiane sono spesso più flessibili e migliori, ma presentano un problema: senza dati fuori dal dominio osservato, la funzione può tendere a zero ai margini. Con $\sigma = 1$, si osserva tendenza a ritornare a zero ai bordi.
[09:50] I polinomi hanno comportamento più noto all’infinito: possono essere preferibili quando si desidera un comportamento asintotico specifico. In generale le gaussiane sono spesso vantaggiose, ma la scelta dipende da contesto e dati.
[10:05] Se la natura dei dati suggerisce comportamento quadratico o cubico, si preferisce un polinomio. In assenza di conoscenza a priori, si preferisce spesso il kernel gaussiano.
## Vettorializzazione della costruzione del kernel
[10:20] La costruzione del kernel con doppio ciclo è lenta. Si può scrivere il calcolo di $K$ in modo vettoriale con NumPy, evitando cicli espliciti.
[10:35] Per il kernel lineare, osservando che $K_{ij} = x_i x_j + 1$, se $X \in \mathbb{R}^{n \times 1}$ è il vettore colonna dei dati di training:
```math
K = X X^\top + \mathbf{1},
```
dove $\mathbf{1}$ è una matrice di tutti 1 di dimensione $n \times n$. Si genera $K$ con prodotti matrice-matrice evitando iterazioni.
[10:55] Precisamente, se $X$ ha forma $\mathbb{R}^{n \times 1}$, allora $X X^\top \in \mathbb{R}^{n \times n}$ e
```math
K_{ij} = X_i X_j + 1
```
riproduce il kernel lineare con bias.
[11:10] Per il kernel gaussiano si adottano strategie simili, sfruttando identità sulla distanza quadratica e broadcast in NumPy, evitando cicli espliciti.
[11:25] In generale, si preferiscono operazioni NumPy vettorializzate per effettuare implicitamente i calcoli su tutte le coppie tramite moltiplicazioni e broadcasting.
## Transizione: introduzione a PageRank
[11:40] Si passa al notebook finale: tema PageRank, algoritmo di Google per determinare i nodi più importanti in un grafo.
[11:55] È predisposto uno script Python per navigare su Wikipedia, facendo crawling su alcune pagine nella categoria “machine learning”. Si costruisce un grafo che connette le pagine in base ai link presenti.
[12:10] Esempio: dalla pagina “machine learning” partono link verso pagine correlate come “computational statistics”. Lo script crea un arco orientato nel grafo se esiste un link tra due pagine.
[12:25] È possibile ottenere statistiche di traffico web delle pagine. Si mostra che l’importanza calcolata con PageRank sul grafo è correlata alle metriche di traffico reali.
[12:40] Una pagina con alto PageRank tende a ricevere più traffico. La correlazione evidenzia l’utilità dell’approccio: PageRank quantifica l’importanza strutturale delle pagine in un grafo di link, coerente con il traffico reale.
## Introduzione e impostazione del problema
[00:00] Si utilizza NetworkX, libreria Python che offre funzioni e classi per lavorare con grafi. I grafi rappresentano nodi e archi tra di essi; sono adatti a modellare collegamenti direzionati tra pagine.
[00:15] I grafi sono utili per modellare collegamenti direzionati. NetworkX fornisce strumenti pronti per analisi efficienti su tali strutture.
[00:28] Sono forniti tre file CSV: edges, nodes e traffic. Si caricano con pandas. Il file edges contiene coppie di pagine collegate; il file nodes contiene l’elenco di pagine; il file traffic associa a ciascun nodo il traffico di un determinato giorno.
[00:48] Il dataset comprende circa 250 pagine (nodi), ~2000 collegamenti (archi) e un numero di voci di traffico pari ai nodi. Obiettivo iniziale: costruire il grafo con NetworkX a partire da questi dati.
## Costruzione del grafo diretto e integrazione dei dati
[01:05] Si costruisce un grafo diretto: gli archi hanno orientamento. Un link da A a B non implica il collegamento inverso da B ad A, riflettendo la realtà dei link tra pagine.
[01:22] Si aggiungono i nodi al grafo dalla colonna “node” del DataFrame nodes. Ogni valore rappresenta il nome di una pagina inserita come nodo.
[01:34] Si aggiungono gli archi al grafo dalle colonne “source” e “target” del DataFrame edges. Ogni coppia rappresenta un collegamento diretto dalla pagina di origine alla pagina di destinazione.
[01:45] Inserendo nodi e archi, si ottiene una struttura NetworkX pronta per analisi successive.
## Mappatura dei nodi a indici e preparazione per la matrice di transizione
[02:02] I nodi sono stringhe. Si desidera convertirli in indici numerici. Si crea un dizionario che mappa ogni pagina (stringa) a un identificatore intero. Questa mappatura consente la costruzione di matrici, traducendo i nomi delle pagine in posizioni di righe e colonne.
[02:18] La matrice di transizione richiede riferimento univoco per ogni nodo. La mappa stringa→ID permette di localizzare direttamente l’elemento di transizione nella matrice.
## Matrice stocastica M e interpretazione probabilistica
[02:35] Si definisce la matrice stocastica $M$, inizialmente una matrice di zeri $N \times N$, dove $N$ è il numero di pagine.
[02:47] Si itera su ciascun arco del grafo. Ogni arco collega due pagine. Le stringhe dei nodi di partenza e arrivo si convertono in indici $i$ e $j$ usando il dizionario.
[03:00] L’elemento $M_{j i}$ rappresenta la probabilità di saltare dalla pagina $i$ alla pagina $j$. A ciascun arco uscente dalla pagina $i$ si assegna probabilità $1/d_i$, dove $d_i$ è il grado uscente di $i$.
[03:18] Formalmente:
```math
M_{j i} = \begin{cases}
\frac{1}{d_i} & \text{se esiste un arco da } i \text{ a } j, \\
0 & \text{altrimenti.}
\end{cases}
```
$M$ è colonna-stocastica: la somma degli elementi di ogni colonna $i$ è 1, descrivendo la distribuzione di probabilità di transizione in uscita da $i$.
[03:40] Interpretazione probabilistica: un utente sulla pagina $i$ con $d_i$ link uscenti sceglie ciascun link con probabilità $1/d_i$. $M$ codifica la probabilità uniforme di transizione tra pagine connesse.
## Damping factor e matrice G (versione regolarizzata di M)
[03:57] Nella pratica si usa una versione regolarizzata della matrice di transizione che permette salti casuali verso qualunque pagina: il damping factor $d$, tipicamente 85%, modella questo comportamento.
[04:10] $d$ è la probabilità di seguire i link presenti sulla pagina; $1-d$ è la probabilità di saltare casualmente verso qualsiasi pagina.
[04:21] Si definisce la matrice $G$:
```math
G = d \, M + (1 - d) \, \frac{\mathbf{1}}{N},
```
dove $\mathbf{1}$ è la matrice di tutti 1 $N \times N$. $\mathbf{1}/N$ assegna probabilità uguali a tutte le transizioni possibili, modellando salti casuali uniformi.
[04:40] In questo modo, l’utente segue i link reali per l’85% e può saltare su qualunque pagina per il restante 15%. $G$ è ancora colonna-stocastica, garantendo che le probabilità si sommino a 1 per ogni stato di partenza.
## Calcolo del PageRank: funzione di NetworkX e confronto metodologico
[04:57] Si calcola il vettore di PageRank in due modi. Il primo usa la funzione PageRank di NetworkX, che calcola automaticamente il PageRank dei nodi, dato il grafo e il parametro $\alpha$ (equivalente a $d$).
[05:12] La funzione PageRank costruisce internamente la matrice di transizione e risolve il problema stazionario, restituendo la distribuzione di PageRank sui nodi.
[05:22] Il secondo metodo implementa l’iterazione delle potenze (power iteration) per calcolare il vettore stazionario della catena di Markov definita da $G$. Entrambi i metodi conducono a risultati coerenti.
## Implementazione dell’iterazione delle potenze
[05:39] Si definisce $G$ come:
```math
G = d \, M + (1 - d) \, \frac{\mathbf{1}}{N}.
```
Questa è la matrice di transizione regolarizzata.
[05:48] Si inizializza il vettore $p$ delle probabilità su ciascuna pagina, ponendolo uniforme:
```math
p^{(0)} = \frac{1}{N} \, \mathbf{e},
```
dove $\mathbf{e}$ è il vettore di tutti 1 di dimensione $N$.
[06:05] Si fissa una tolleranza numerica, ad esempio $10^{-8}$, e un massimo di iterazioni, ad esempio $1000$.
[06:14] Si esegue l’iterazione:
```math
p^{(k+1)} = G \, p^{(k)}.
```
Dopo ogni moltiplicazione, si normalizza il vettore per una norma (ad esempio $L^1$) per evitare problemi di scala:
```math
p^{(k+1)} \leftarrow \frac{p^{(k+1)}}{\lVert p^{(k+1)} \rVert}.
```
[06:31] Si controlla la differenza tra iterazioni successive. Se $\lVert p^{(k+1)} - p^{(k)} \rVert < \text{tolleranza}$, si considera raggiunta la convergenza e si interrompe il ciclo, altrimenti si aggiorna $p$ e si continua.
[06:43] Al termine, si normalizza $p$ affinché la somma degli elementi sia $1$:
```math
p \leftarrow \frac{p}{\sum_{i=1}^{N} p_i}.
```
Il vettore $p$ risultante è il PageRank: ogni componente $p_i$ è la probabilità stazionaria di essere sulla pagina $i$.
## Confronto tra PageRank di NetworkX e Power Iteration
[07:00] Si confrontano i vettori di PageRank ottenuti con NetworkX e con l’iterazione delle potenze. La correlazione tra i due è molto alta, prossima a uno; la differenza media è dell’ordine di $10^{-2}$.
[07:12] Le minime differenze derivano da dettagli implementativi, ma il comportamento complessivo è sovrapponibile, confermando la correttezza dell’implementazione.
[07:22] Un grafico di confronto mostra curve praticamente coincidenti, con differenze lievi.
## Visualizzazione del grafo e layout a molle
[07:39] Si visualizza il grafo usando il “Spring Layout” di NetworkX, che simula collegamenti come molle. Il sistema cerca una disposizione che minimizzi un’energia, producendo un layout leggibile. La stiffness $K$ controlla la rigidità e “random seed” fissa l’inizializzazione.
[07:48] Si disegnano i nodi, gli archi e si aggiungono etichette. Poiché il grafo è denso, si mostrano le etichette solo per il 10% dei nodi più importanti secondo il PageRank, calcolando il quantile al 90% e filtrando i nodi con importanza superiore.
[08:03] La generazione del layout richiede tempo, perché risolve una simulazione fisica. Il risultato tipico mostra le pagine più importanti raggruppate al centro, con connessioni forti; le pagine meno connesse si dispongono in periferia.
[08:40] Tra i nomi rilevanti emergono “cross validation”, “pattern recognition”, “statistical learning theory”, “generative models”. Restringendo ulteriormente al top 5% per importanza, si rendono visibili elementi prima coperti, chiarendo la struttura.
[08:56] Concetti come “confusion matrix” risultano tra i temi importanti nelle pagine di machine learning, riflettendo l’influenza strutturale dei collegamenti.
## Confronto tra PageRank e dati di traffico reale
[09:13] Si dispone del vettore di PageRank per ordinare le pagine per importanza. Si confronta con i dati di traffico nel DataFrame traffic, che riporta il traffico di alcune pagine.
[09:24] Si aggiunge al DataFrame traffic la colonna con i nomi dei nodi e la colonna con il PageRank. Ogni riga contiene: nome del nodo, traffico e PageRank.
[09:36] Si produce uno scatter con traffic.traffic sull’asse orizzontale e traffic.pageRank sull’asse verticale. A prima vista la correlazione sembra debole.
[09:44] Modificando le scale degli assi in logaritmico, emerge una tendenza: allineamento qualitativo tra importanza stimata e traffico osservato, suggerendo una relazione non casuale.
[09:54] Si calcola il coefficiente di correlazione con $\text{corrcoef}$ di NumPy, ottenendo circa il 66%, indicativo di una correlazione moderata, superiore a fenomeni casuali.
[10:08] L’evidenza supporta l’idea che l’algoritmo coglie una struttura informativa nei dati, producendo previsioni ragionevoli del traffico.
## Baseline casuale e significatività del risultato
[10:25] Si considera una baseline: importanza casuale delle pagine. Si simula permutando casualmente i valori di PageRank (o generando sequenze casuali) e si ricalcola la correlazione.
[10:36] Permutando casualmente i valori di importanza, la correlazione con il traffico reale risulta prossima a zero. Ogni esecuzione produce permutazioni diverse, ma l’esito resta vicino alla non correlazione.
[10:46] Il confronto mostra che l’algoritmo ottiene una correlazione di circa 0{,}6, significativa, indicando che il PageRank rileva informazioni utili sulla distribuzione del traffico rispetto alla randomizzazione.
## Chiusura e riferimenti operativi
[11:03] Non emergono ulteriori domande. La sessione termina. Per dubbi o richieste, è possibile contattare via email. Buon fine settimana.