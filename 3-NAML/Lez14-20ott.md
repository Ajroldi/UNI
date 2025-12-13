## Introduzione al problema del completamento di matrici e al Netflix problem
[00:00] Si considera il problema generale del completamento di matrici, detto anche *matrix completion*. Questo problema riguarda il riempimento delle voci mancanti di una matrice parzialmente osservata. Una delle applicazioni tipiche di questo quadro è quella dei sistemi di raccomandazione, in particolare il cosiddetto *Netflix problem*, in cui si vogliono predire i voti che gli utenti darebbero a film che non hanno ancora visto.
[00:10] Si immagina di avere una matrice, o più in generale un tensore, che descrive le interazioni tra utenti e prodotti. Nel caso specifico si considerano:
- $N$ utenti,
- $D$ film (o, più in generale, contenuti multimediali come serie, documentari, e così via).
Ogni utente vota solo una parte dei film disponibili. La matrice dei voti contiene quindi molti valori mancanti. Ogni riga rappresenta un utente, ogni colonna un film, e in ciascuna posizione $(i,j)$ della matrice compare il voto dell’utente $i$ per il film $j$, se tale voto è stato espresso.
[00:25] L’obiettivo è predire quale sarebbe il voto dei valori mancanti per ogni utente, in modo da poter formulare suggerimenti personalizzati. Il problema rientra nella classe dei sistemi di raccomandazione: a partire dalle preferenze parzialmente note degli utenti, si vogliono produrre raccomandazioni sui contenuti che questi potrebbero apprezzare.
[00:40] Nelle applicazioni reali il numero di utenti $N$ e il numero di film $D$ è in genere molto grande. La matrice risultante è quindi di dimensione molto elevata e al tempo stesso molto sparsa, poiché molti elementi sono mancanti. Il punto cruciale è che si vuole sfruttare il fatto che il dataset possiede una struttura a rango basso, cioè può essere approssimato bene da una matrice di rango molto minore rispetto alle sue dimensioni.
[00:55] Dire che la matrice ha rango basso significa che esistono alcuni fattori latenti, o feature derivate, di dimensione ridotta, che permettono di descrivere tutte le colonne o tutte le righe della matrice. Questi fattori latenti rappresentano caratteristiche nascoste ma rilevanti nei dati, che riassumono le principali variazioni nelle preferenze degli utenti e nelle proprietà dei film.
[01:05] Alcuni esempi di possibili feature latenti nel caso dei film sono:
- il genere del film (ad esempio drammatico, commedia, azione),
- l’epoca o l’era di produzione,
- il tipo di pubblico a cui è rivolto (ad esempio adulti o bambini).
Queste caratteristiche non sono necessariamente direttamente osservate nella matrice dei voti, ma ne determinano in modo implicito la struttura.
[01:15] Il numero di feature latenti “significative” viene indicato con $r$ ed è pari al rango della matrice. Si assume che $r$ sia molto più piccolo sia del numero di utenti $N$ sia del numero di film $D$. L’ipotesi fondamentale è quindi che il dataset possieda una struttura intrinsecamente a rango basso, e che questo possa essere sfruttato per ricostruire le voci mancanti in modo significativo.
## Gradi di libertà di una matrice a rango basso e decomposizione SVD
[01:30] Si considera ora un esempio astratto per comprendere meglio la struttura a rango basso. Sia $M$ una matrice quadrata $n \times n$ e si supponga che
```math
\mathrm{rank}(M) = r \ll n

cioè che il suo rango sia $r$ anziché $n$ (rango pieno). Si assume inoltre che $n$ sia grande. Si vuole quantificare quanti gradi di libertà sono necessari per descrivere completamente questa matrice $M$.
[01:45] L’obiettivo è determinare il numero di parametri indipendenti necessari per specificare la matrice $M$ a rango $r$. In altre parole, si vuole capire quanti gradi di libertà servono per identificare univocamente $M$.
[01:50] Si richiama quindi la decomposizione ai valori singolari (SVD) della matrice $M$:
```math
M = U \Sigma V^\top,
```
dove:
- $U$ è una matrice ortogonale $n \times n$,
- $V$ è una matrice ortogonale $n \times n$,
- $\Sigma$ è una matrice diagonale $n \times n$ contenente i valori singolari di $M$.
[02:00] Nel caso in cui $\operatorname{rank}(M) = r$, la matrice diagonale $\Sigma$ contiene solo $r$ valori singolari non nulli. Di conseguenza:
- in $\Sigma$ compaiono $r$ valori significativi,
- in $U$ e $V$ contano solo i primi $r$ vettori singolari (colonne), mentre gli altri sono associati a valori singolari nulli o trascurabili e non sono rilevanti per descrivere il contenuto informativo del dato.
[02:15] Si potrebbe pensare che questi $r$ vettori in $U$ e in $V$ richiedano $r \cdot n$ parametri ciascuno per essere specificati. Tuttavia non è così, perché i vettori singolari sono:
- normalizzati, cioè hanno norma unitaria,
- mutuamente ortogonali, cioè il prodotto scalare tra vettori diversi è nullo.
[02:25] Questi vincoli di ortogonalità e normalizzazione introducono relazioni tra le componenti dei vettori e riducono il numero effettivo di gradi di libertà. Utilizzando, ad esempio, un procedimento di ortogonalizzazione di Gram–Schmidt, si può verificare che descrivere una famiglia di $r$ vettori ortonormali in $\mathbb{R}^n$ richiede meno di $r \cdot n$ parametri liberi.
[02:40] In particolare, per ciascuna delle matrici $U$ e $V$, il numero di gradi di libertà necessari a descrivere i primi $r$ vettori singolari è pari a:
```math
\frac{(2n - r - 1)\,r}{2}.
```
Questa formula tiene conto sia della normalizzazione dei vettori sia della loro mutua ortogonalità.
[02:55] Sommando il contributo di $U$ e di $V$, si ottiene che il numero complessivo di gradi di libertà necessari per descrivere una matrice $M$ di rango $r$ è:
```math
(2n - r)\,r.
```
Questa quantità è molto più piccola di $n^2$ quando $r \ll n$.
[03:05] Supponendo, ad esempio:
- $n \approx 10^6$ (un milione di utenti e un milione di film),
- $r \approx 10^2$ (rango pari a $100$),
se la matrice fosse generica, senza struttura a rango basso, sarebbero necessari $n^2 = 10^{12}$ parametri per descriverla. Questo corrisponde al numero totale dei suoi elementi.
[03:20] Se invece si sfrutta il fatto che la matrice ha rango $r$, il numero di gradi di libertà si riduce a:
```math
(2n - r)\,r \simeq 2 \cdot 10^6 \cdot 10^2 = 2 \cdot 10^8,
```
cioè dell’ordine di $10^8$, molto più piccolo di $10^{12}$. Questo evidenzia il vantaggio essenziale della struttura a rango basso: per descrivere la matrice bastano molti meno parametri rispetto al numero totale di voci.
[03:30] La riduzione del numero di parametri mostra che, se il dataset è effettivamente a rango basso, è teoricamente possibile ricostruire la matrice completa da un numero di osservazioni molto inferiore al totale degli elementi. Questo è il principio che rende possibile il completamento di matrici su larga scala.
[03:40] È importante osservare che, se il dataset con valori mancanti non possiede una struttura a rango basso, non esiste un modo veramente significativo per riempire le voci mancanti. In questi casi si potrebbero solo adottare strategie semplicistiche, come copiare un valore vicino su riga o colonna o fare la media dei valori circostanti, ma tali procedure non hanno un significato profondo in termini di modellizzazione delle preferenze.
[03:55] Nelle applicazioni reali, quando si hanno dataset di grandi dimensioni, è ragionevole assumere che i campioni non siano completamente scorrelati tra loro. Questa correlazione tra righe o colonne induce spesso una struttura a rango basso, o approssimativamente a rango basso, che è fondamentale per progettare algoritmi efficaci di completamento dei dati mancanti.
## Formalizzazione matematica del problema di matrix completion
[04:10] Si passa alla formalizzazione matematica del problema. Si consideri una matrice $X$ parzialmente osservata. Ciò significa che:
- esiste un insieme di indici osservati $\Omega \subset \{1,\dots,m\} \times \{1,\dots,d\}$,
- per ogni coppia $(i,j) \in \Omega$ è noto il valore $X_{ij}$.
[04:20] Gli elementi $X_{ij}$ tali che $(i,j)\in\Omega$ costituiscono le osservazioni disponibili. Si indica con $|\Omega| = m$ il numero di elementi osservati.
[04:30] L’obiettivo è ricostruire la matrice completa $X$ assumendo che essa sia a rango basso. Formalmente si suppone che:
```math
\operatorname{rank}(X) = r \ll \min(m,d).
```
Questa ipotesi esprime il fatto che la matrice ha una struttura intrinseca di bassa complessità rispetto alle sue dimensioni.
[04:40] Per formulare il problema in modo compatto, si introducono due definizioni utili.
1. **Operatore di proiezione $P_\Omega$**
[04:40] L’operatore di proiezione $P_\Omega$ agisce su una matrice $A$ e produce una nuova matrice $P_\Omega(A)$ definita elemento per elemento come:
```math
\bigl(P_\Omega(A)\bigr)_{ij} =
\begin{cases}
A_{ij}, & \text{se } (i,j) \in \Omega, \\
0, & \text{altrimenti}.
\end{cases}
```
In altre parole, $P_\Omega$ mantiene nella matrice solo i valori nelle posizioni osservate, mettendo a zero tutte le altre.
[04:55] Se si considera la matrice delle osservazioni $X$, la matrice $P_\Omega(X)$ coincide con una matrice che contiene i valori noti nelle posizioni osservate e zeri altrove. Per una matrice generica $M$, $P_\Omega(M)$ seleziona i valori nelle posizioni di $\Omega$ e pone a zero tutte le altre voci.
2. **Frazione di osservazioni**
[05:10] La frazione di osservazioni è data dal rapporto tra il numero di elementi osservati e il numero totale di elementi della matrice:
```math
\text{frazione di osservazioni} = \frac{|\Omega|}{m \cdot d}.
```
Questa quantità indica quale percentuale della matrice è effettivamente nota.
[05:10] Se non si introduce alcuna ipotesi strutturale sulla matrice $X$, il problema di ricostruzione è mal posto: esistono infinite matrici diverse che coincidono con i dati osservati su $\Omega$. Qualunque modifica che interessi soltanto indici non appartenenti a $\Omega$ produce una matrice diversa ma compatibile con le osservazioni.
[05:25] La condizione di basso rango diventa quindi essenziale. Si vuole sfruttare il fatto che $X$ è a rango basso per selezionare un’unica matrice tra tutte quelle che coincidono sui dati osservati. In altre parole, si introduce un criterio che favorisca matrici con rango minimo.
## Formulazione come problema di rango minimo e difficoltà di ottimizzazione
[05:40] Si cerca, tra tutte le matrici che coincidono con le osservazioni nelle posizioni in $\Omega$, quella che ha rango minimo. Formalmente si pone il seguente problema:
```math
\min_{M \in \mathbb{R}^{m \times d}} \operatorname{rank}(M)
```
soggetto al vincolo
```math
P_\Omega(M) = P_\Omega(X),
```
cioè
```math
M_{ij} = X_{ij}, \quad \text{per tutti } (i,j) \in \Omega.
```
[05:55] Questa formulazione garantisce che la matrice incognita $M$ coincida con la matrice originale $X$ nelle posizioni osservate. Tra tutte le matrici che soddisfano questo vincolo si sceglie quella il cui rango è il più basso possibile.
[06:05] Tuttavia il problema così posto è estremamente difficile. La funzione rango è una funzione intera, non continua e non convessa, e il problema di minimizzarla sotto vincoli lineari appartiene alla classe dei problemi NP-hard. Ciò significa che non esiste, in generale, un algoritmo efficiente (in tempo polinomiale) che ne trovi la soluzione esatta per matrici di grandi dimensioni.
[06:20] È quindi necessario individuare un modo per rendere il problema trattabile, pur mantenendo l’idea fondamentale: trovare una matrice a rango basso coerente con le osservazioni disponibili.
## Convessità, non convessità e ruolo del rango
[06:30] La difficoltà del problema risiede nella non convessità della funzione $\operatorname{rank}(M)$. Per comprendere questo aspetto, è utile ricordare la differenza tra funzioni convesse e non convesse.
[06:30] Una funzione convessa ha normalmente un unico minimo globale e una forma che rende l’ottimizzazione più semplice. Metodi iterativi come la discesa del gradiente possono convergere a questo minimo globale con garanzie teoriche, sotto opportune condizioni.
[06:45] Una funzione non convessa può invece avere molti minimi locali e un paesaggio di ottimizzazione complicato. Metodi di discesa del gradiente possono fermarsi in minimi locali che non sono il minimo globale, e il risultato dipende fortemente dal punto iniziale. La funzione rango è un esempio tipico di funzione non convessa, poiché varia a salti e non in modo continuo.
[06:45] Un esempio di funzionale convesso è quello che compare nei metodi ai minimi quadrati per la risoluzione di sistemi lineari:
```math
\phi(x) = \frac{1}{2} \|Ax - b\|_2^2,
```
dove $A$ è una matrice, $b$ un vettore e $\|\cdot\|_2$ è la norma euclidea. Questa funzione è quadratica e quindi convessa. Il suo minimo coincide con la soluzione del sistema $Ax = b$ (quando la soluzione esiste ed è unica).
[06:55] In questo contesto, l’ottimizzazione è relativamente semplice, perché:
- il funzionale ha un unico minimo globale,
- i metodi iterativi come il gradiente convergono, in condizioni standard, verso tale minimo.
[07:05] Nel caso di funzioni non convesse, il paesaggio di ottimizzazione può essere molto più complesso, con numerosi minimi locali e punti di sella. Un algoritmo di discesa può fermarsi in un minimo locale e non fornire la soluzione ottimale globale. La funzione rango appartiene a questa categoria: piccoli cambiamenti nella matrice possono produrre variazioni improvvise nel rango.
[07:15] La non convessità della funzione rango rende il problema di matrix completion, formulato come minimizzazione del rango, molto difficile. D’altra parte, molte funzioni di costo che modellano problemi reali sono non convesse, come ad esempio le funzioni di loss nelle reti neurali. Questo mostra che la non convessità è spesso inevitabile, ma quando possibile è utile trovare rilassamenti convessi che approssimino il problema originale.
## Esempio di funzione rango e confronto con la norma nucleare
[07:40] Per chiarire meglio la natura non convessa della funzione rango, si consideri una matrice $2\times 2$ che dipende da due parametri reali $x$ e $y$:
```math
M(x,y) =
\begin{pmatrix}
\sqrt{4x} & x + y \\
x + y & \sqrt{4y}
\end{pmatrix}.
```
Per ogni coppia $(x,y)$ si può calcolare il rango di $M(x,y)$ e rappresentarlo come funzione di $x$ e $y$.
[07:50] Il grafico del rango in funzione di $(x,y)$ presenta una struttura a gradini: esistono regioni in cui il rango è costante e cambiamenti improvvisi quando $(x,y)$ attraversa determinate curve. Questo comportamento a salti evidenzia la non convessità della funzione rango, che non è adatta a un’ottimizzazione convessa.
[08:05] Sullo stesso dominio $(x,y)$ si può invece considerare la norma nucleare di $M(x,y)$, che risulta essere una funzione convessa dei parametri $x$ e $y$. Il confronto tra il grafico del rango e quello della norma nucleare mostra come quest’ultima vari in modo regolare e continuo, fornendo un paesaggio di ottimizzazione molto più regolare.
## Norma nucleare di una matrice e collegamento con i valori singolari
[08:20] La norma nucleare di una matrice $M$ è definita come la somma dei suoi valori singolari. Se $\sigma_1, \sigma_2, \dots, \sigma_r$ sono i valori singolari non nulli di $M$, si ha:
```math
\|M\|_* = \sum_{k=1}^r \sigma_k.
```
Questa norma misura quindi la somma delle ampiezze principali della matrice, rappresentate dai valori singolari.
[08:30] Un’espressione equivalente della norma nucleare è:
```math
\|M\|_* = \operatorname{trace}\bigl(\sqrt{M^\top M}\bigr),
```
dove:
- $M^\top M$ è una matrice simmetrica semidefinita positiva,
- $\sqrt{M^\top M}$ è la radice quadrata matriciale di $M^\top M$,
- la traccia è la somma degli autovalori di $\sqrt{M^\top M}$, che coincidono con i valori singolari di $M$.
[08:40] Se si costruisce il vettore
```math
\sigma = (\sigma_1, \sigma_2, \dots, \sigma_r),
```
contenente i valori singolari non nulli di $M$, la norma nucleare è esattamente la norma $L_1$ di questo vettore:
```math
\|M\|_* = \|\sigma\|_1 = \sum_{k=1}^r |\sigma_k|.
```
Poiché i valori singolari sono non negativi, la norma $L_1$ coincide semplicemente con la loro somma.
[08:50] La norma $L_1$ di un vettore $v = (v_1,\dots,v_n)$ è definita come:
```math
\|v\|_1 = \sum_{i=1}^n |v_i|.
```
Nel caso dei valori singolari, tale definizione coincide con la somma dei valori singolari stessi.
[09:00] La norma nucleare ha due proprietà importanti nel contesto del matrix completion:
1. è una funzione convessa della matrice $M$;
2. è strettamente legata al rango, perché dipende direttamente dai valori singolari che determinano il rango della matrice.
## Norme vettoriali $L_0$ e $L_1$ e analogia con il rango
[09:15] Per comprendere il collegamento tra il rango di una matrice e la norma nucleare, è utile richiamare l’analogo vettoriale. In particolare si considerano due funzioni sullo spazio dei vettori: la cosiddetta norma $L_0$ e la norma $L_1$.
[09:20] La “norma” $L_0$, che in realtà non è una vera norma in senso matematico, di un vettore $v$ è definita come il numero di componenti non nulle di $v$:
```math
\|v\|_0 = \#\{i : v_i \neq 0\}.
```
Questa quantità misura la cardinalità del supporto del vettore, cioè quante coordinate sono effettivamente diverse da zero.
[09:30] Minimizzare $\|v\|_0$ significa ricercare una rappresentazione sparsa, con il minor numero possibile di componenti non nulle. Tuttavia, la funzione $L_0$ è non convessa e porta a problemi di ottimizzazione difficili, spesso NP-hard, per motivi simili a quelli che si incontrano nella minimizzazione del rango.
[09:40] La norma $L_1$ di un vettore $v$ è definita come:
```math
\|v\|_1 = \sum_i |v_i|.
```
Questa funzione è convessa. Se si minimizza $\|v\|_1$ sotto opportuni vincoli, si ottengono spesso vettori sparsi, cioè con molte componenti nulle, ma attraverso un problema convesso, quindi trattabile.
[09:50] In sintesi:
- la minimizzazione di $\|v\|_0$ promuove la sparsità in modo diretto ma comporta problemi di ottimizzazione non convessi;
- la minimizzazione di $\|v\|_1$ fornisce una rilassazione convessa della minimizzazione di $\|v\|_0$, e in molti casi produce comunque soluzioni sparse.
[10:05] Nel caso delle matrici l’analogo della sparsità vettoriale è la condizione di rango basso. La funzione rango conta il numero di valori singolari non nulli, così come $\|v\|_0$ conta il numero di componenti non nulle di un vettore.
## Dalla minimizzazione del rango alla minimizzazione della norma nucleare
[10:05] Poiché la funzione rango è non convessa, si cerca un sostituto convesso che ne mantenga l’effetto qualitativo. La norma nucleare svolge questo ruolo.
[10:15] La norma nucleare è la somma dei valori singolari, quindi è la norma $L_1$ del vettore delle singolar values. Minimizzare la norma nucleare equivale a minimizzare la norma $L_1$ del vettore dei valori singolari.
[10:25] Minimizzare la norma $L_1$ del vettore dei valori singolari tende a rendere questo vettore “sparso”, cioè induce molti valori singolari a diventare nulli. Quando molti valori singolari sono nulli, la matrice ha rango basso.
[10:35] In conclusione:
- la minimizzazione della norma nucleare è una rilassazione convessa della minimizzazione della funzione rango;
- pur non coincidendovi esattamente, essa ne preserva la tendenza a selezionare matrici a rango basso, mantenendo il problema di ottimizzazione convesso.
## Formulazione convessa del problema di matrix completion
[11:00] Alla luce delle considerazioni precedenti, il problema di matrix completion può essere riformulato sostituendo la funzione rango con la norma nucleare. Si considera il problema:
```math
\min_{M \in \mathbb{R}^{m \times d}} \|M\|_*
```
soggetto al vincolo:
```math
P_\Omega(M) = P_\Omega(X).
```
[11:10] In forma equivalente:
```math
\min_M \|M\|_* \quad \text{s.t.} \quad M_{ij} = X_{ij}, \ \forall (i,j)\in\Omega.
```
[11:15] Questa nuova formulazione presenta due vantaggi principali:
1. il problema è convesso, poiché la norma nucleare è convessa e i vincoli sono lineari;
2. esistono metodi di ottimizzazione ben sviluppati per problemi convessi, con risultati teorici sulla convergenza.
[11:25] Inoltre, risultati teorici mostrano che, sotto opportune ipotesi sulla distribuzione dei dati e delle osservazioni, la minimizzazione della norma nucleare consente la ricostruzione esatta della matrice originale $X$. In altre parole, la soluzione del problema convesso coincide, con alta probabilità, con la soluzione del problema originario di rango minimo.
## Risultati teorici per il completamento di matrici a rango basso
[11:40] Si considera un modello stocastico di matrice a rango basso, detto *random orthogonal model*. In questo modello la matrice $M$ è generata come:
```math
M = \sum_{k=1}^r \sigma_k \, u_k v_k^\top,
```
dove:
- $u_k$ e $v_k$ sono vettori ortonormali,
- i vettori $u_k$ e $v_k$ sono scelti in modo casuale, uniformemente tra tutte le famiglie ortonormali possibili.
[11:50] Questo modello definisce una famiglia di matrici $M$ di dimensione $m_1 \times m_2$, con rango $r$ molto più piccolo del minimo tra $m_1$ e $m_2$:
```math
r \ll \min(m_1, m_2).
```
[12:00] Si definisce $n = \max(m_1, m_2)$. Si suppone di osservare $m$ elementi di $M$ scelti a caso, in modo uniforme, tra tutte le possibili posizioni. Si ottiene così un insieme di indici osservati $\Omega$ di cardinalità $m$.
[12:15] Un risultato fondamentale afferma che esistono costanti positive $C$ e $c$ tali che, se il numero di osservazioni $m$ soddisfa una disuguaglianza del tipo:
```math
m \ge C \, n^{1,2} r \, \log n
```
(o una stima analoga, in dipendenza della formulazione), allora il minimizzatore della norma nucleare
```math
\hat{M} = \arg\min \{ \|M\|_* : P_\Omega(M) = P_\Omega(X) \}
```
è unico ed è uguale alla matrice vera $M$ con probabilità almeno
```math
1 - C n^{-3}.
```
[12:30] Questo significa che, per $n$ grande, la probabilità di fallire la ricostruzione esatta è molto piccola, dell’ordine di $n^{-3}$. In modo sintetico si può dire che, se si hanno “abbastanza osservazioni”, allora, con alta probabilità, si può recuperare esattamente la matrice originale mediante la minimizzazione della norma nucleare.
[12:40] Quando le ipotesi del teorema (basso rango, distribuzione casuale delle osservazioni e struttura dei vettori $u_k$ e $v_k$) sono soddisfatte:
- esiste un’unica matrice coerente con le osservazioni e a norma nucleare minima,
- tale matrice coincide con la matrice vera $M$ con alta probabilità.
[12:50] In questo modo, la rilassazione convessa tramite norma nucleare risulta, in molti casi, formalmente equivalente o molto vicina alla minimizzazione diretta del rango, con il vantaggio di essere trattabile dal punto di vista computazionale.
## Problema convesso con norma nucleare e aspetti pratici
[13:05] Una volta definito il problema convesso
```math
\min_M \|M\|_* \quad \text{s.t.} \quad P_\Omega(M) = P_\Omega(X),
```
rimane da capire come risolverlo in pratica, soprattutto quando le dimensioni della matrice sono molto grandi.
[13:15] Anche se il problema è convesso, le dimensioni tipiche (ordine di milioni di utenti e milioni di oggetti) richiedono algoritmi efficienti sia in termini di tempo computazionale sia di memoria. Non è possibile usare metodi generali di programmazione convessa che trattino l’intera matrice in modo esplicito, memorizzando tutti gli elementi.
[13:25] Un algoritmo proposto per affrontare questo tipo di problema è il *Singular Value Thresholding* (SVT). Si tratta di un algoritmo iterativo che:
- sfrutta in modo esplicito la struttura a rango basso della matrice,
- richiede la computazione di decomposizioni ai valori singolari,
- può essere implementato in modo efficiente, lavorando su matrici di rango effettivo ridotto.
## Soft thresholding scalare e promozione della sparsità
[13:40] Il primo ingrediente fondamentale dell’algoritmo SVT è l’operatore di *soft thresholding* applicato a uno scalare. Dato un numero reale $x$ e una soglia $\tau > 0$, l’operatore di soft thresholding $s_\tau(x)$ è definito come:
```math
s_\tau(x) =
\begin{cases}
x - \tau, & x > \tau, \\
0, & |x| \le \tau, \\
x + \tau, & x < -\tau.
\end{cases}
```
[13:55] In altre parole:
- se $|x| \le \tau$, l’uscita è zero;
- se $x > \tau$, l’uscita è $x - \tau$;
- se $x < -\tau$, l’uscita è $x + \tau$.
[14:05] Graficamente, l’operatore di soft thresholding appiattisce verso zero i valori piccoli in valore assoluto e riduce di $\tau$ i valori grandi, mantenendo il segno. Applicato alle componenti di un vettore, questo operatore induce sparsità, perché annulla le componenti di ampiezza piccola.
## Soft thresholding sui valori singolari e operatore SVT
[14:20] L’idea successiva consiste nell’applicare il soft thresholding non agli elementi della matrice, ma ai suoi valori singolari. Si consideri una matrice $M$ con SVD
```math
M = U \Sigma V^\top,
```
dove $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots)$ è la matrice diagonale dei valori singolari.
[14:25] Si definisce l’operatore di *singular value thresholding* $D_\tau(M)$ come:
```math
D_\tau(M) = U \, S_\tau(\Sigma) \, V^\top,
```
dove $S_\tau(\Sigma)$ è la matrice diagonale ottenuta applicando l’operatore scalare $s_\tau$ a ciascun valore singolare:
```math
S_\tau(\Sigma) = \operatorname{diag}\bigl(s_\tau(\sigma_1), s_\tau(\sigma_2), \dots\bigr).
```
[14:40] L’operatore $D_\tau$:
- riduce di una quantità $\tau$ i valori singolari maggiori di $\tau$,
- annulla i valori singolari minori o uguali a $\tau$,
- lascia invariati i vettori singolari (colonne di $U$ e di $V$).
[14:50] Di conseguenza:
- i valori singolari piccoli vengono “tagliati” a zero,
- la matrice risultante $D_\tau(M)$ ha rango ridotto, perché alcuni valori singolari sono diventati nulli.
[15:00] L’operatore $D_\tau$ promuove quindi il rango basso della matrice, in modo analogo a come il soft thresholding vettoriale promuove la sparsità.
## Collegamento tra SVT e minimizzazione con norma nucleare
[15:15] Esiste un legame diretto tra l’operatore di singular value thresholding $D_\tau$ e la minimizzazione di una funzione che combina un termine ai minimi quadrati e la norma nucleare. Questo collegamento chiarisce il significato ottimizzante del soft thresholding sui valori singolari.
[15:25] Dato un dato $Y$ (matrice o tensore), l’operatore $D_\tau(Y)$ può essere scritto come:
```math
D_\tau(Y) = \arg\min_M \left\{ \frac{1}{2} \|M - Y\|_F^2 + \tau \|M\|_* \right\},
```
dove:
- $\|M - Y\|_F$ è la norma di Frobenius:
  ```math
\|M - Y\|_F^2 = \sum_{i,j} (M_{ij} - Y_{ij})^2,
```
- $\|M\|_*$ è la norma nucleare di $M$.
[15:40] Quindi, $D_\tau(Y)$ è il minimizzatore del funzionale:
```math
J(M) = \frac{1}{2} \|M - Y\|_F^2 + \tau \|M\|_*.
```
[15:50] In questo funzionale:
- il primo termine è un termine di fedeltà ai dati, perché misura lo scostamento tra $M$ e $Y$ in senso delle minime quadrate;
- il secondo termine è un termine di regolarizzazione che penalizza la norma nucleare e quindi favorisce matrici a rango basso.
[16:00] Il parametro $\tau$ regola il bilanciamento tra:
- la vicinanza ai dati (piccolo errore $\|M - Y\|_F^2$),
- la semplicità della soluzione (piccola norma nucleare, quindi rango basso).
[16:10] Il fatto che $D_\tau(Y)$ sia il minimizzatore di un funzionale che contiene la norma nucleare mostra che il soft thresholding sui valori singolari è strettamente connesso a un problema di ottimizzazione convesso regolarizzato.
## Interpretazione del funzionale con norma di Frobenius e norma nucleare
[16:25] Considerando il funzionale
```math
J(M) = \frac{1}{2} \|M - Y\|_F^2 + \tau \|M\|_*,
```
il primo termine impone che la matrice $M$ approssimi bene i dati $Y$. Più è piccolo il valore di $\|M - Y\|_F^2$, più $M$ è vicina a $Y$.
[16:35] Il secondo termine, $\tau \|M\|_*$, introduce una penalizzazione proporzionale alla norma nucleare. Minimizzare questo termine favorisce matrici a rango basso, poiché riduce la somma dei valori singolari e quindi tende a rendere molti di essi nulli.
[16:45] Insieme, i due termini definiscono un compromesso:
- una matrice $M$ troppo vicina a $Y$ potrebbe avere rango elevato se si trascurasse la regolarizzazione;
- una matrice con norma nucleare molto piccola è a rango basso, ma potrebbe essere molto distante da $Y$ se si ignorasse il termine di fedeltà ai dati.
[16:55] La soluzione del problema di minimizzazione bilancia questi due aspetti in modo ottimale (per il valore di $\tau$ scelto), producendo una matrice:
- sufficientemente vicina ai dati osservati,
- con struttura a rango basso.
## Soglia sui valori singolari e uso della SVD in pratica
[00:00] Il secondo termine della funzione obiettivo, che contiene la norma nucleare, è un termine che spinge la soluzione verso una matrice a bassa rank. In generale, quando si ha un problema ai minimi quadrati, regolarizzato o meno, la decomposizione ai valori singolari (SVD) è uno strumento fondamentale per la sua soluzione.
[00:20] In pratica, a partire da una matrice $Y$, si calcola la sua SVD:
```math
Y = U \Sigma V^\top,
```
poi si applica una sogliatura alla matrice diagonale $\Sigma$ dei valori singolari, ottenendo una versione “sogliata” di $Y$:
```math
\hat Y = U \Sigma' V^\top,
```
dove $\Sigma'$ è ottenuta applicando il soft thresholding agli elementi diagonali di $\Sigma$.
[00:50] Il soft thresholding applicato ai valori singolari consiste nel sostituire ogni valore singolare $\sigma_i$ con
```math
\sigma_i' = \max(\sigma_i - \tau, 0),
```
dove $\tau > 0$ è il parametro di soglia. I valori singolari più piccoli di $\tau$ vengono annullati, mentre quelli maggiori vengono ridotti di $\tau$. Questo procedimento realizza la minimizzazione di un funzionale con norma nucleare, in quanto la norma nucleare $\|M\|_* = \sum_i \sigma_i(M)$ è direttamente influenzata da tali operazioni.
## Formulazioni equivalenti del problema di completamento di matrici
[01:30] Dal punto di vista pratico, esistono due formulazioni equivalenti del problema di completamento di matrici a bassa rank.
La prima è una formulazione di minimizzazione non vincolata, in cui si cerca una matrice $M$ che minimizzi:
```math
\min_M \ \tau \|M\|_* + \frac{1}{2} \|\mathcal{P}_\Omega(M) - \mathcal{P}_\Omega(X)\|_F^2.
```
[01:55] In questa espressione:
- $\|M\|_*$ è la norma nucleare di $M$ e favorisce soluzioni a bassa rank;
- $\mathcal{P}_\Omega(\cdot)$ è la proiezione sull’insieme degli indici osservati $\Omega$; essa mantiene gli elementi nelle posizioni osservate e pone a zero gli altri;
- $X$ è la matrice vera, o la matrice degli osservati, e $\mathcal{P}_\Omega(X)$ rappresenta le osservazioni disponibili;
- il termine $\|\mathcal{P}_\Omega(M) - \mathcal{P}_\Omega(X)\|_F^2$ misura la fedeltà ai dati osservati, mediante la norma di Frobenius.
[02:30] L’obiettivo è che sulle posizioni osservate la matrice $M$ coincida il più possibile con la matrice dei dati $X$, mentre la norma nucleare induce la soluzione verso una matrice a bassa rank. Il parametro $\tau > 0$ regola il compromesso tra il vincolo di bassa rank (attraverso la norma nucleare) e la fedeltà ai dati.
[02:55] La stessa idea può essere formulata come problema di ottimizzazione vincolata:
```math
\min_M \ \|M\|_* \quad \text{soggetto a} \quad \mathcal{P}_\Omega(M) = \mathcal{P}_\Omega(X).
```
[03:20] Nella prima formulazione la fedeltà ai dati è espressa come un termine nel funzionale da minimizzare; nella seconda compare come un vincolo esatto sulle posizioni osservate. Le due formulazioni sono concettualmente equivalenti e, a seconda della strategia numerica scelta, può risultare più conveniente lavorare con una o con l’altra.
## Algoritmo iterativo basato su SVD e soft thresholding
### Definizione delle variabili e inizializzazione
[03:50] Per risolvere il problema di completamento di matrici si utilizza un algoritmo iterativo che alterna:
- un passo che riduce il rango della matrice candidata tramite soft thresholding sui valori singolari;
- un passo che riporta la soluzione verso il rispetto delle osservazioni disponibili.
[04:10] Come ingressi dell’algoritmo si considerano:
- la matrice dei dati parzialmente osservata $X$ e l’insieme di indici osservati $\Omega$;
- il parametro $\tau$, che è la soglia da applicare ai valori singolari (legata al termine di regolarizzazione);
- il parametro $\delta$, che rappresenta la lunghezza del passo nell’aggiornamento iterativo.
[04:35] Si introduce una costante iniziale $c_0$, che può essere calcolata o scelta in modo semplice, ad esempio $c_0 = 1$. Si definisce $Y_0 = \mathcal{P}_\Omega(X)$ (eventualmente scalata da $c_0$). Il parametro $\delta$ sarà utilizzato per aggiornare la sequenza $Y_k$ nel corso dell’algoritmo.
### Passo di sogliatura sui valori singolari
[05:05] Alla $k$-esima iterazione si definisce una matrice $M_k$ come:
```math
M_k = \mathcal{T}_\tau(Y_k),
```
dove $\mathcal{T}_\tau$ indica l’operazione di soft thresholding sui valori singolari. Più precisamente:
1. Si calcola la SVD di $Y_k$:
   ```math
Y_k = U_k \Sigma_k V_k^\top.
```
2. Si applica il soft thresholding agli elementi diagonali di $\Sigma_k$:
   ```math
(\Sigma_k')_{ii} = \max\bigl((\Sigma_k)_{ii} - \tau, 0\bigr).
```
3. Si ricostruisce la matrice sogliata:
   ```math
M_k = U_k \Sigma_k' V_k^\top.
```
[05:40] In questo modo si decomprime $Y_k$ nei suoi valori singolari, si riducono i valori singolari tramite sogliatura e si ricompone una matrice $M_k$ che ha rango ridotto rispetto a $Y_k$, o comunque tende ad avere rango più basso.
### Passo di aggiornamento verso le osservazioni
[06:05] Una volta ottenuto $M_k$, si aggiorna $Y$ secondo la regola:
```math
Y_{k+1} = Y_k + \delta \, \mathcal{P}_\Omega(X - M_k).
```
[06:25] In questa formula:
- $\mathcal{P}_\Omega(X - M_k)$ è la proiezione del residuo $X - M_k$ sulle sole posizioni osservate;
- tale residuo proiettato rappresenta la direzione lungo la quale ci si muove per avvicinare $M_k$ alle osservazioni;
- $\delta$ è la lunghezza del passo: se $\delta$ è grande, il passo è più lungo; se è piccolo, il passo è più corto.
[06:55] Questa regola di aggiornamento corrisponde, formalmente, a un passo di discesa del gradiente su un problema convesso che combina un termine di fedeltà ai dati e un termine di regolarizzazione con norma nucleare. Il residuo proiettato rappresenta il gradiente rispetto al termine di fedeltà ai dati, e il parametro $\delta$ regola l’ampiezza dello spostamento lungo questa direzione.
[07:25] In sintesi:
- il calcolo di $M_k$ tramite soft thresholding implementa il passo di riduzione del rango mediante minimizzazione approssimata della norma nucleare;
- il termine $\mathcal{P}_\Omega(X - M_k)$ implementa il passo di ri-allineamento della soluzione alle osservazioni;
- il parametro $\delta$ regola la combinazione di queste due componenti.
[07:50] L’iterazione viene ripetuta per un certo numero di passi o fino al soddisfacimento di un criterio di arresto basato sulla convergenza.
## Alternanza tra riduzione del rango e fedeltà ai dati
[08:10] Ogni iterazione dell’algoritmo si compone di due fasi principali:
1. sogliatura dei valori singolari di $Y_k$ per ottenere $M_k$, che tende a ridurre la rank;
2. aggiornamento di $Y_k$ verso una matrice che rispetta le osservazioni, aggiungendo $\delta \, \mathcal{P}_\Omega(X - M_k)$.
[08:35] L’effetto complessivo è un’alternanza tra:
- una tendenza a far diminuire il rango della soluzione, tramite soft thresholding;
- una tendenza a soddisfare i dati osservati, tramite il passo orientato dal residuo proiettato.
[08:55] Questa alternanza implica che l’andamento del rango non sia monotono. Durante il passo di sogliatura i valori singolari vengono ridotti e il rango tende a diminuire. Durante il passo di aggiornamento con il residuo $X - M_k$, gli elementi della matrice vengono modificati e il rango può aumentare, restare invariato o comunque non seguire una decrescita costante.
[09:25] Non si ha quindi la garanzia che il rango decresca ad ogni iterazione. Possono verificarsi oscillazioni del rango, ma ciò non impedisce la convergenza alla soluzione ottimale della minimizzazione della norma nucleare, purché i parametri siano scelti in modo appropriato e si eseguano un numero sufficiente di iterazioni.
## Convergenza dell’algoritmo e scelta del passo $\delta$
[10:15] L’algoritmo descritto viene applicato a un problema convesso, e ciò consente di stabilire risultati di convergenza. In particolare, è dimostrato che, scegliendo il parametro $\delta$ nell’intervallo:
```math
0 < \delta < 2,
```
l’algoritmo converge alla soluzione ottimale del problema di minimizzazione della norma nucleare, soggetto ai vincoli sulle osservazioni.
[10:45] La convergenza è di tipo lineare e la velocità di convergenza dipende dal valore di $\delta$. In pratica valori di $\delta$ compresi tra $1.2$ e $1.5$ forniscono un buon compromesso tra stabilità e rapidità, anche se questa indicazione è di tipo empirico.
[11:10] Dal punto di vista interpretativo:
- $\delta$ è la lunghezza del passo lungo la direzione del residuo proiettato;
- se $\delta$ è grande, si compiono passi lunghi, il che può velocizzare la convergenza ma anche introdurre instabilità o oscillazioni;
- se $\delta$ è piccolo, i passi sono più corti, il numero di iterazioni cresce, ma il comportamento tende a essere più regolare.
[11:40] L’unico vincolo teorico essenziale per la convergenza è che $\delta$ appartenga all’intervallo $(0,2)$. All’interno di questo intervallo la scelta più opportuna è guidata da considerazioni pratiche di efficienza e stabilità.
## Criterio di arresto e complessità computazionale
[12:10] Essendo una procedura iterativa, è necessario stabilire un criterio di arresto. Sono possibili due approcci:
1. fissare a priori un numero di iterazioni $K$;
2. utilizzare un criterio basato sull’errore o sul residuo.
[12:30] Un criterio di terminazione basato sul residuo proiettato consiste nel verificare la condizione:
```math
\frac{\|\mathcal{P}_\Omega(M_k - X)\|_F}{\|\mathcal{P}_\Omega(X)\|_F} \leq \varepsilon,
```
dove:
- $\|\cdot\|_F$ è la norma di Frobenius,
- $\varepsilon > 0$ è una tolleranza fissata,
- $M_k$ è la matrice alla $k$-esima iterazione.
[12:55] Questo criterio confronta l’errore relativo sulle posizioni osservate tra la soluzione corrente e i dati, e arresta l’algoritmo quando tale errore scende al di sotto di una soglia prefissata.
[13:15] Per quanto riguarda la complessità computazionale, il costo complessivo dell’algoritmo è dell’ordine di:
```math
O(k \, r \, n \, d),
```
dove:
- $k$ è il numero di iterazioni eseguite,
- $r$ è la rank effettiva o approssimata della soluzione,
- $n$ e $d$ sono le dimensioni della matrice.
[13:40] Anche se $n$ e $d$ sono grandi, se la rank $r$ è piccola il prodotto $r \, n \, d$ rimane gestibile. L’algoritmo è quindi adatto a problemi in cui la matrice incognita è a bassa rank, come nel matrix completion in molte applicazioni reali.
## Scelta del parametro di soglia $\tau$
[14:05] Un aspetto importante per le prestazioni del metodo è la scelta del parametro di soglia $\tau$, che controlla l’intensità del soft thresholding sui valori singolari e quindi il grado di riduzione del rango.
Si possono distinguere tre strategie principali di scelta.
### 1. Scelta teorica basata su una formula
[14:25] Una prima possibilità è utilizzare una scelta teorica, ponendo ad esempio:
```math
\tau = \gamma \sqrt{n d},
```
dove:
- $n$ e $d$ sono le dimensioni della matrice,
- $\gamma$ è un parametro tipicamente scelto in un intervallo, ad esempio tra $5$ e $10$.
[14:55] Questa scelta lega la soglia alle dimensioni del problema, con un fattore di scala $\gamma$ che incorpora considerazioni empiriche o teoriche sulla struttura del rumore o delle osservazioni mancanti.
### 2. Scelta tramite cross-validation
[15:15] Una seconda strategia consiste nell’impiegare la cross-validation. In questo caso:
- si fissa un intervallo di ricerca per $\tau$, ad esempio:
  ```math
\tau \in [1, 10 \sqrt{n d}];
```
- si definisce una griglia di possibili valori di $\tau$ in questo intervallo;
- per ciascun valore di $\tau$ si esegue l’algoritmo e si valuta l’errore su un sottoinsieme dei dati usato per la validazione;
- si seleziona il valore di $\tau$ che minimizza l’errore di validazione.
[15:45] Il parametro di soglia viene così scelto in modo guidato dai dati, sulla base delle prestazioni effettive nella ricostruzione.
### 3. Strategia adattiva
[16:05] Una terza possibilità è utilizzare una strategia adattiva, in cui il valore di $\tau$ viene modificato durante le iterazioni. Si può ad esempio:
- iniziare con un valore relativamente grande,
- ridurre gradualmente $\tau$ man mano che l’algoritmo procede.
[16:20] Nelle prime fasi, una soglia più alta forza una riduzione significativa del rango; nelle fasi successive, una soglia più piccola consente di affinare la soluzione con maggior precisione.
[16:35] Queste tre strategie (scelta teorica, cross-validation, scelta adattiva) sono concettualmente analoghe alle strategie usate per la scelta del passo nei metodi di discesa del gradiente in altri contesti.
## Scelta del parametro $\delta$ e stabilità dell’algoritmo
[18:10] Per il parametro $\delta$ (step size) non esistono formule altrettanto strutturate come per $\tau$, al di là del vincolo teorico $0 < \delta < 2$. In pratica:
- si sceglie $\delta$ in un intervallo ristretto, ad esempio tra $1$ e $1.5$;
- tale scelta è giustificata empiricamente da un compromesso tra velocità di convergenza e stabilità.
[18:35] Interpretando $\delta$ come lunghezza del passo:
- se $\delta$ è grande, i passi sono lunghi; ciò può velocizzare la convergenza ma anche introdurre possibili oscillazioni;
- se $\delta$ è piccolo, i passi sono corti; il numero di iterazioni necessario aumenta, ma il comportamento risulta in genere più stabile.
[19:00] Valori moderati di $\delta$, all’interno dell’intervallo garantito di convergenza $(0,2)$, consentono di evitare instabilità mantenendo una buona efficienza.
## Applicazioni del completamento di matrici a bassa rank
[19:20] Il problema del completamento di matrici, nato originariamente nel contesto dei sistemi di raccomandazione (come nel problema Netflix), è in realtà molto generale. Lo stesso schema teorico può essere applicato in diversi ambiti.
[19:40] In ambito di *computer vision*:
- per la rimozione dello sfondo in un video, dove il video viene modellato come una matrice (o un tensore) e lo sfondo è visto come componente a bassa rank;
- per il *video completion* o l’*inpainting* di immagini, in cui si vogliono ricostruire parti mancanti di un’immagine o di una sequenza video.
[20:05] In problemi di identificazione di sistemi, quando le misure provenienti da sensori presentano dati mancanti, è possibile utilizzare il completamento di matrici per ricostruire le grandezze non osservate.
[20:20] In bioinformatica e genomica, i dati sperimentali possono essere parziali o rumorosi, e il completamento di matrici a bassa rank permette di ricostruire informazioni mancanti o ridurre il rumore.
[20:35] In fisica quantistica, la ricostruzione di stati quantistici può essere ricondotta a un problema di completamento di matrici con struttura a bassa rank.
[20:50] In ambito finanziario, per problemi di ottimizzazione di portafoglio con rendimenti mancanti: se non si dispone di tutte le serie storiche di rendimento per i vari titoli, si può tentare di completare la matrice dei rendimenti mediante strategie analoghe.
[21:10] In tutte queste situazioni, il punto comune è:
- la presenza di una matrice con dati mancanti,
- la ricerca di una soluzione a bassa rank coerente con le osservazioni,
- l’uso della norma nucleare come surrogato convesso del rango.
[21:30] Il quadro teorico basato su:
- rilassamento convesso tramite norma nucleare,
- algoritmi iterativi basati su SVD e soft thresholding,
risulta quindi applicabile a un’ampia gamma di problemi oltre al caso dei sistemi di raccomandazione.
## Riepilogo del quadro sul completamento di matrici
[21:50] Gli elementi essenziali del metodo di completamento di matrici a bassa rank possono essere riassunti come segue:
- la minimizzazione diretta della rank è un problema non convesso e difficile da trattare;
- la norma nucleare $\|M\|_*$ viene introdotta come surrogato convesso della rank, sfruttando il fatto che è la somma dei valori singolari e promuove soluzioni a bassa rank;
- il problema è riformulato come:
  ```math
\min_M \ \|M\|_* \quad \text{soggetto a} \quad \mathcal{P}_\Omega(M) = \mathcal{P}_\Omega(X),
```
  oppure, in forma regolarizzata:
  ```math
\min_M \ \tau \|M\|_* + \frac{1}{2} \|\mathcal{P}_\Omega(M) - \mathcal{P}_\Omega(X)\|_F^2;
```
- si sviluppa una procedura iterativa basata sulla SVD e sul soft thresholding dei valori singolari, alternando:
  - passi di riduzione del rango (sogliatura dei valori singolari),
  - passi di avvicinamento ai dati osservati (aggiornamento con il residuo proiettato).
## Introduzione alle Support Vector Machines (SVM)
[23:05] Si introduce ora un secondo argomento: le *Support Vector Machines* (SVM). Queste derivano da idee simili a quelle dei metodi ai minimi quadrati e dei metodi kernel, ma utilizzano funzioni di perdita diverse.
[23:25] Le SVM si suddividono in due grandi famiglie:
- *Support Vector Classification* (SVC), per problemi di classificazione;
- *Support Vector Regression* (SVR), per problemi di regressione.
La differenza principale riguarda l’obiettivo:
- nella classificazione, si vuole separare punti appartenenti a classi diverse tramite un iperpiano (o una frontiera indotta da un kernel);
- nella regressione, si vuole approssimare una funzione, spesso un iperpiano in uno spazio di feature, tollerando un certo errore entro un tubo di ampiezza $\varepsilon$.
[23:55] Nel caso della classificazione, l’obiettivo è costruire un iperpiano che separi i dati di due (o più) classi massimizzando il margine, cioè la distanza tra l’iperpiano e i punti più vicini di ciascuna classe. Nel caso della regressione, si vuole trovare una funzione che approssimi i dati il meglio possibile, ignorando gli errori che rientrano all’interno di una fascia (tubo) di ampiezza $\varepsilon$ attorno alla funzione stessa.
## Support Vector Classification: margine massimo
[24:20] Per la classificazione, “ottimale” significa che l’iperpiano di separazione è quello che massimizza il margine, cioè la distanza minima tra l’iperpiano e i punti più vicini di ciascuna classe.
[24:35] In uno scenario con due insiemi di punti in un piano, si vuole trovare una retta di separazione (o un iperpiano in dimensioni superiori) tale che:
- i punti di una classe stiano da una parte,
- i punti dell’altra classe stiano dall’altra,
- la distanza tra la retta e i punti più vicini (di entrambe le classi) sia massima.
[24:55] Questo concetto di margine si traduce matematicamente nella ricerca di un vettore dei pesi $w$ e di un bias $b$ che definiscono l’iperpiano, minimizzando la norma di $w$ sotto vincoli lineari che esprimono la corretta classificazione dei punti, con eventuali variabili di slack per ammettere errori o violazioni del margine.
## Support Vector Regression: tubo di ampiezza $\varepsilon$
[25:20] Nel caso della regressione con SVM (SVR), l’idea di ottimalità è diversa. Si vuole trovare una funzione di regressione $f(x)$ tale che:
- il maggior numero possibile di punti dati cada all’interno di un tubo di ampiezza $\varepsilon$ attorno a $f(x)$;
- gli errori all’interno del tubo non vengano penalizzati;
- gli errori al di fuori del tubo vengano penalizzati in modo appropriato.
[25:45] Il tubo di ampiezza $\varepsilon$ (epsilon-tube) è la regione compresa tra $f(x) - \varepsilon$ e $f(x) + \varepsilon$. I punti che si trovano in questa regione sono considerati accettabili e non contribuiscono alla funzione di costo; quelli al di fuori determinano invece la perdita.
[26:05] I punti che si trovano sulla frontiera del tubo o al di fuori di esso sono i *support vectors*, cioè quelli che determinano effettivamente la soluzione di regressione. I punti all’interno del tubo non influenzano direttamente la soluzione finale e sono “tollerati” dall’errore ammesso di ampiezza $\varepsilon$.
[26:25] La scelta di $\varepsilon$ influisce sul numero di support vectors:
- se $\varepsilon$ è grande, il tubo è largo, molti punti cadono al suo interno e non vengono penalizzati; il numero di support vectors diminuisce;
- se $\varepsilon$ è piccolo, il tubo è stretto, più punti si trovano al di fuori del tubo e diventano support vectors; la soluzione può diventare più complessa.
## Funzione di perdita $\varepsilon$-insensitive
[26:50] Nella SVR si utilizza la funzione di perdita *$\varepsilon$-insensitive*. Indicato con $y_i$ il valore osservato e con $f(x_i)$ la predizione, la perdita è definita come:
```math
L_\varepsilon(y_i, f(x_i)) =
\begin{cases}
0, & \text{se } |y_i - f(x_i)| \le \varepsilon, \\
|y_i - f(x_i)| - \varepsilon, & \text{se } |y_i - f(x_i)| > \varepsilon.
\end{cases}
```
[27:20] In altre parole:
- se l’errore assoluto $|y_i - f(x_i)|$ è minore o uguale a $\varepsilon$, la perdita è zero;
- se l’errore supera $\varepsilon$, si penalizza solo l’eccedenza rispetto a $\varepsilon$.
[27:35] Geometricamente, questa funzione corrisponde a ignorare gli errori all’interno del tubo di ampiezza $\varepsilon$ e a penalizzare quelli esterni, proporzionalmente alla distanza oltre il tubo.
## Formulazione del problema di Support Vector Regression
[27:55] La formulazione del problema di SVR ricalca concettualmente quella di altri problemi di ottimizzazione regolarizzata. Si introduce una funzione obiettivo da minimizzare, insieme a vincoli che definiscono le condizioni sul tubo di ampiezza $\varepsilon$.
[28:15] Una forma classica del problema di SVR è:
```math
\min_{w,b,\xi_i,\xi_i^*} \ \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
```
soggetto ai vincoli:
```math
\begin{cases}
y_i - \langle w, x_i \rangle - b \le \varepsilon + \xi_i, \\
\langle w, x_i \rangle + b - y_i \le \varepsilon + \xi_i^*, \\
\xi_i \ge 0, \ \xi_i^* \ge 0,
\end{cases}
```
per $i = 1, \dots, n$.
[28:50] Qui:
- $w$ è il vettore dei pesi che definisce la funzione di regressione,
- $b$ è il termine di bias,
- $\xi_i$ e $\xi_i^*$ sono variabili di slack, che misurano quanto il punto $i$ si trova al di sopra o al di sotto del tubo,
- $C > 0$ è un parametro di regolarizzazione che bilancia:
  - la “piattezza” del modello (minimizzazione di $\|w\|^2$),
  - la tolleranza per le violazioni del tubo (attraverso la somma delle slack variables).
[29:20] Il termine $\frac{1}{2}\|w\|^2$ spinge verso soluzioni con pesi di piccola norma, favorendo funzioni di regressione non eccessivamente variabili. Il termine $C \sum_{i} (\xi_i + \xi_i^*)$ penalizza i punti che escono dal tubo di ampiezza $\varepsilon$.
[29:40] Le variabili di slack $\xi_i$ e $\xi_i^*$ sono vincolate a essere non negative e rappresentano le eccedenze oltre il tubo. Il parametro $C$ controlla quanto sia costoso penalizzare queste eccedenze: valori grandi di $C$ penalizzano fortemente le violazioni, portando a modelli che si adattano maggiormente ai dati; valori piccoli consentono invece maggiori violazioni.
## Lagrangiana e condizioni di ottimalità nella SVR
[30:05] Per risolvere il problema di SVR si utilizza la formulazione duale del problema, introducendo la Lagrangiana. Si introducono moltiplicatori di Lagrange, ad esempio $\alpha_i$ e $\alpha_i^*$, per i vincoli sulle disuguaglianze.
[30:25] La Lagrangiana assume la forma:
```math
\mathcal{L}(w,b,\xi_i,\xi_i^*;\alpha_i,\alpha_i^*) = \frac{1}{2}\|w\|^2 + C \sum_i (\xi_i + \xi_i^*) + \sum_i \alpha_i (y_i - \langle w, x_i \rangle - b - \varepsilon - \xi_i) + \sum_i \alpha_i^* (\langle w, x_i \rangle + b - y_i - \varepsilon - \xi_i^*) + \dots
```
dove i puntini rappresentano eventuali termini per i vincoli $\xi_i \ge 0$, $\xi_i^* \ge 0$.
[30:55] Si derivano le condizioni di stazionarietà prendendo le derivate parziali della Lagrangiana rispetto alle variabili primali $w$, $b$, $\xi_i$, $\xi_i^*$ e ponendole uguali a zero. Da queste condizioni si ricava la forma di $w$ in termini delle variabili duali e dei dati.
[31:15] Una condizione fondamentale è:
```math
w = \sum_{i=1}^n (\alpha_i - \alpha_i^*) x_i.
```
[31:30] Questa formula mostra che il vettore dei pesi $w$ è sempre una combinazione lineare dei dati $x_i$, con coefficienti dati dalla differenza tra le variabili duali $\alpha_i$ e $\alpha_i^*$. Questo risultato è analogo a quanto avviene nei metodi kernel per i minimi quadrati, e sarà fondamentale per estendere la SVR al caso non lineare.
## Estensione kernel e rappresentazione non lineare
[32:05] Poiché $w$ è espresso come combinazione lineare dei dati $x_i$, è possibile introdurre una mappa di feature $\phi(x)$ e un kernel $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j)\rangle$. In questo modo si possono affrontare problemi non lineari nello spazio originale dei dati, trattandoli come problemi lineari nello spazio delle feature.
[32:25] In pratica:
- si sostituisce il prodotto scalare $\langle w, x\rangle$ con una somma di kernel:
  ```math
f(x) = \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(x_i, x) + b;
```
- i dati $x_i$ vengono implicitamente mappati nello spazio di feature tramite $\phi$, ma non è necessario calcolare esplicitamente $\phi$ grazie alla funzione kernel;
- si mantiene la struttura matematica della SVR, ma la frontiera nel dominio originale può assumere forme non lineari.
[32:55] Il concetto di iperpiano viene quindi trasferito dallo spazio originale allo spazio delle feature: un iperpiano in questo spazio corrisponde a una frontiera generalmente non lineare nello spazio dei dati originali. Ciò consente di modellare relazioni complesse mantenendo l’impianto teorico delle SVM.
## Espressione finale del modello di regressione SVM
[33:20] Una volta determinati i valori delle variabili duali $\alpha_i$, $\alpha_i^*$ e del bias $b$, la funzione di regressione SVM assume la forma:
```math
f(x) = \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(x_i, x) + b.
```
[33:40] Gli indici $i$ per cui $(\alpha_i - \alpha_i^*) \neq 0$ corrispondono ai support vectors. Solo questi punti contribuiscono alla predizione $f(x)$:
- i punti che cadono all’interno del tubo di ampiezza $\varepsilon$ hanno in genere coefficienti nulli e non influenzano la soluzione;
- i punti che si trovano sulla frontiera del tubo o al di fuori di esso determinano il modello.
[34:05] Per effettuare una previsione su un nuovo punto $x$, si sommano i contributi dei support vectors, pesati dai coefficienti $(\alpha_i - \alpha_i^*)$, tramite i valori del kernel $K(x_i,x)$ e si aggiunge il bias $b$.
## Scelta del kernel e forma della funzione di regressione
[34:30] In implementazioni pratiche è possibile scegliere tra diversi kernel, ad esempio:
- kernel lineare,
- kernel polinomiale,
- kernel RBF (radial basis function), che corrisponde al kernel gaussiano,
- altri kernel specifici.
[34:50] La scelta del kernel determina la forma della funzione di regressione:
- con un kernel RBF la regressione può adattarsi a strutture molto flessibili e non lineari dei dati;
- con un kernel polinomiale di grado elevato si possono modellare relazioni polinomiali complesse;
- con un kernel lineare (o un kernel polinomiale di grado 1) si ottiene una regressione lineare nello spazio originale.
[35:15] Quando i dati non seguono una tendenza lineare, un kernel lineare richiede in genere un numero elevato di support vectors per rappresentare adeguatamente la struttura dei dati. Il modello risultante, pur essendo linearmente parametrizzato, può dover utilizzare molti punti di supporto.
[35:35] Con un kernel RBF, spesso si ottiene una regressione più ben adattata ai dati, con un numero inferiore di support vectors, perché la flessibilità del kernel consente di modellare meglio l’andamento non lineare.
## Obiettivo delle SVM: pochi support vectors e buona rappresentazione
[35:55] In generale, l’idea di fondo delle SVM, sia nella classificazione sia nella regressione, è ottenere una rappresentazione dei dati che:
- utilizzi un numero di support vectors il più piccolo possibile;
- mantenga una buona capacità di approssimare o separare i dati.
[36:15] Ridurre il numero di support vectors significa avere un modello più compatto che:
- è più semplice da valutare, poiché richiede meno operazioni per ogni predizione;
- tende a essere meno soggetto a overfitting;
- conserva la capacità di rappresentare correttamente la struttura dei dati.
[36:30] La scelta del kernel e dei parametri (come $\varepsilon$, $C$ e gli iperparametri specifici del kernel) influisce sia sulla complessità del modello sia sul numero di support vectors.
## Funzione di perdita hinge e Support Vector Classification
[36:50] Per la classificazione, la funzione di perdita utilizzata nelle SVM è la *hinge loss*. Se $y_i \in \{-1, +1\}$ è l’etichetta di classe e $f(x_i)$ è la funzione di decisione, la hinge loss è definita come:
```math
L_{\text{hinge}}(y_i, f(x_i)) = \max(0, 1 - y_i f(x_i)).
```
[37:15] Questa funzione di perdita:
- è nulla se $y_i f(x_i) \ge 1$, cioè se il punto è correttamente classificato e si trova al di là del margine;
- aumenta linearmente quando $y_i f(x_i) < 1$, cioè quando il punto è mal classificato o troppo vicino all’iperpiano di separazione.
[37:30] La formulazione della SVM di classificazione minimizza una funzione composta da:
- un termine di regolarizzazione, tipicamente $\frac{1}{2}\|w\|^2$, che promuove un margine ampio;
- un termine legato alla hinge loss, che penalizza i punti mal classificati o troppo vicini all’iperpiano.
## Collegamento con i metodi ai minimi quadrati
[38:05] Dal punto di vista concettuale, le SVM sono vicine ai metodi ai minimi quadrati già discussi, con alcune differenze cruciali:
- si cambia la funzione di perdita: invece della perdita quadratica si impiega la hinge loss (per la classificazione) o la $\varepsilon$-insensitive loss (per la regressione);
- si introducono parametri come $\varepsilon$ e $C$ per controllare la tolleranza agli errori e la complessità del modello;
- si usano i kernel per passare da modelli lineari a modelli non lineari, mantenendo una struttura di ottimizzazione simile.
[38:35] In pratica, gli schemi già sviluppati per i metodi ai minimi quadrati possono essere adattati:
- sostituendo la funzione di perdita quadratica con la hinge loss o con la $\varepsilon$-insensitive loss;
- mantenendo la possibilità di utilizzare kernel per ottenere modelli non lineari;
- introducendo i parametri $C$ ed $\varepsilon$ e studiando il loro effetto sul numero di support vectors e sulla qualità della regressione o della classificazione.
## Ruolo di $\varepsilon$ e selezione dei punti significativi
[39:00] Nella SVR il parametro $\varepsilon$ ha un ruolo fondamentale nella selezione dei punti significativi:
- i punti che cadono all’interno del tubo di ampiezza $\varepsilon$ non contribuiscono alla funzione di costo e non diventano support vectors;
- i punti che si trovano al di fuori del tubo (o sui suoi bordi) generano variabili di slack non nulle e corrispondono a coefficienti $(\alpha_i - \alpha_i^*)$ diversi da zero.
[39:25] Il valore di $\varepsilon$ controlla il compromesso tra:
- complessità del modello (misurata dal numero di support vectors),
- accuratezza della previsione (quanto strettamente la funzione si avvicina ai dati).
[39:40] In analogia con il completamento di matrici, si tratta di scegliere opportunamente un parametro che regola la tolleranza all’errore e la complessità della soluzione, mantenendo un equilibrio tra capacità di generalizzazione e adattamento ai dati.
[39:55] Con questa panoramica si conclude il quadro introduttivo su:
- completamento di matrici tramite norma nucleare e SVD,
- support vector machines per classificazione e regressione, con attenzione al ruolo dei kernel, dei parametri di regolarizzazione e delle funzioni di perdita.