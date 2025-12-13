## Capitolo 1: Analisi della Convergenza del Metodo del Gradiente
### Introduzione alla Stima dell'Errore e Ipotesi Aggiuntive
[00:00] L'analisi riprende dalla definizione di un limite superiore, o *bound*, per l'errore tra i valori della funzione obiettivo. L'obiettivo è stimare un limite per la differenza media tra il valore della funzione calcolato a ogni iterazione e il suo valore minimo. Tale quantità è espressa come:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*))
$$
dove:
- $T$ è il numero totale di iterazioni dell'algoritmo.
- $x_t$ è il vettore dei parametri (o iterata) alla $t$-esima iterazione.
- $x^*$ è la soluzione ottima, ovvero il punto di minimo della funzione.
- $f(\cdot)$ è la funzione obiettivo che si intende minimizzare.
[00:11] Partendo dalla sola ipotesi di convessità della funzione $f$, l'analisi di base ha condotto alla seguente disuguaglianza fondamentale:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \frac{\gamma}{2T} \sum_{t=1}^{T} ||\nabla f(x_t)||^2 + \frac{1}{2\gamma T} ||x_1 - x^*||^2
$$
Questa relazione costituisce il punto di partenza per derivare risultati più specifici sulla convergenza, i quali si ottengono introducendo ipotesi aggiuntive sulla funzione obiettivo $f$.
### Caso 1: Funzione Convessa con Gradiente Limitatato
#### Ipotesi e Tesi del Teorema
[00:30] Il primo risultato si ottiene introducendo l'ipotesi che il gradiente della funzione sia limitato. Questa proprietà è anche nota come Lipschitzianità della funzione rispetto al gradiente.
- **Definizione di Gradiente Limitatato**: Una funzione $f$ si dice a gradiente limitato se esiste una costante positiva $B$ tale che la norma del suo gradiente è sempre inferiore o uguale a tale costante per ogni punto del dominio. Matematicamente:
  $$
  ||\nabla f(x)|| \le B \quad \forall x
  $$
[00:40] Si introduce un'ulteriore ipotesi riguardante il punto di partenza dell'algoritmo, $x_1$. Si assume che la distanza tra $x_1$ e la soluzione ottima $x^*$ sia limitata da una costante $R$:
$$
||x_1 - x^*|| \le R
$$
[00:51] Sotto queste due ipotesi, e scegliendo un passo di apprendimento (o *learning rate*) $\gamma$ specifico, definito come:
$$
\gamma = \frac{R}{B\sqrt{T}}
$$
dove $T$ è il numero totale di iterazioni, si ottiene la seguente stima per la differenza media dei valori funzionali:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \frac{RB}{\sqrt{T}}
$$
[01:06] Questa relazione implica che la differenza tra il valore della funzione calcolato nell'iterata "migliore" ($x_{best}$) e il valore ottimo $f(x^*)$ decresce con un tasso dell'ordine di $1/\sqrt{T}$.
#### Numero di Iterazioni e Tasso di Convergenza
[01:15] Se si desidera che la soluzione trovata, $x_{best}$, soddisfi una certa tolleranza $\epsilon$, ovvero che l'errore sia inferiore a tale soglia:
$$
f(x_{best}) - f(x^*) \le \epsilon
$$
[01:23] Sfruttando la stima precedente, è possibile determinare il numero di iterazioni $T$ necessarie per garantire tale accuratezza. Il numero di iterazioni risulta essere dell'ordine di:
$$
T \approx \frac{R^2 B^2}{\epsilon^2}
$$
[01:34] Questo risultato indica che il tasso di convergenza, in relazione alla tolleranza $\epsilon$ desiderata, è dell'ordine di $1/\epsilon^2$. Di conseguenza, per raggiungere una tolleranza molto piccola (ad esempio, $\epsilon = 10^{-2}$), il numero di iterazioni teoricamente richiesto può diventare estremamente elevato.
#### Dimostrazione del Risultato
[01:52] La dimostrazione del teorema è relativamente diretta e si basa sull'applicazione delle ipotesi aggiuntive alla disuguaglianza fondamentale.
[02:04] Si parte dalla disuguaglianza ottenuta nell'analisi di base:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \frac{\gamma}{2T} \sum_{t=1}^{T} ||\nabla f(x_t)||^2 + \frac{1}{2\gamma T} ||x_1 - x^*||^2
$$
Il primo termine al secondo membro viene maggiorato utilizzando l'ipotesi di gradiente limitato, per cui $||\nabla f(x_t)||^2 \le B^2$. Il secondo termine viene maggiorato utilizzando l'ipotesi sulla distanza iniziale, per cui $||x_1 - x^*||^2 \le R^2$.
[02:13] Applicando queste maggiorazioni, si ottiene:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \frac{\gamma}{2T} \sum_{t=1}^{T} B^2 + \frac{R^2}{2\gamma T}
$$
La sommatoria $\sum_{t=1}^{T} B^2$ è pari a $T \cdot B^2$, quindi la disuguaglianza si semplifica in:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \frac{\gamma B^2}{2} + \frac{R^2}{2\gamma T}
$$
[02:24] Il teorema postula l'esistenza di un valore ottimale per il passo di apprendimento $\gamma$. Per trovarlo, si considera il lato destro della disuguaglianza come una funzione di $\gamma$, che si può denotare con $Q(\gamma)$:
$$
Q(\gamma) = \frac{\gamma B^2}{2} + \frac{R^2}{2\gamma T}
$$
[02:32] Per trovare il valore di $\gamma$ che minimizza $Q(\gamma)$, si calcola la sua derivata prima rispetto a $\gamma$ e la si pone uguale a zero:
$$
\frac{dQ}{d\gamma} = \frac{B^2}{2} - \frac{R^2}{2\gamma^2 T} = 0
$$
[02:42] Risolvendo l'equazione per $\gamma$, si ottiene il valore ottimale:
$$
\gamma_{opt} = \frac{R}{B\sqrt{T}}
$$
[02:49] Sostituendo questo valore ottimale di $\gamma$ nell'espressione di $Q(\gamma)$, si ottiene il risultato finale enunciato dal teorema:
$$
\frac{1}{T} \sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \frac{RB}{\sqrt{T}}
$$
[02:58] Questo primo risultato stabilisce che, sotto le ipotesi di funzione convessa e a gradiente limitato, l'errore medio decresce con un fattore proporzionale a $1/\sqrt{T}$, portando a un tasso di convergenza dell'ordine di $1/\epsilon^2$.
### Caso 2: Funzione L-smooth (a gradiente Lipschitziano)
#### La Condizione di Decrescita Sufficiente
[03:10] Si analizza ora il caso in cui si aggiunge l'ipotesi di L-smoothness della funzione.
- **Definizione di Funzione L-smooth**: Una funzione $f$ è detta L-smooth, o a gradiente L-Lipschitziano, se esiste una costante $L > 0$ tale che il suo gradiente soddisfa la seguente condizione per ogni coppia di punti $x, y$:
  $$
  ||\nabla f(x) - \nabla f(y)|| \le L ||x - y|| \quad \forall x, y
  $$
  Una conseguenza diretta di questa proprietà è la seguente disuguaglianza, che fornisce un limite superiore quadratico alla funzione:
  $$
  f(y) \le f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}||y-x||^2
  $$
[03:15] In questo scenario, si sceglie un passo di apprendimento costante pari a $\gamma = 1/L$. Con questa scelta, è possibile dimostrare una proprietà fondamentale nota come "condizione di decrescita sufficiente" (*sufficient decrease*).
[03:23] Questa condizione garantisce che, se la funzione è L-smooth e si utilizza $\gamma = 1/L$, i valori della funzione obiettivo calcolati a ogni iterazione del metodo del gradiente formano una sequenza strettamente decrescente. Matematicamente, si ha:
$$
f(x_{t+1}) \le f(x_t) - \frac{1}{2L} ||\nabla f(x_t)||^2
$$
Poiché il termine $||\nabla f(x_t)||^2$ è sempre non negativo, si ha $f(x_{t+1}) < f(x_t)$ (a meno che non si sia già raggiunto un punto stazionario dove il gradiente è nullo), garantendo una sequenza decrescente di valori funzionali.
#### Dimostrazione della Decrescita Sufficiente
[03:40] La dimostrazione parte dalla disuguaglianza che definisce la proprietà di L-smoothness.
[03:46] In tale disuguaglianza, si sostituisce la relazione ricorsiva del metodo del gradiente con $\gamma = 1/L$:
$$
x_{t+1} = x_t - \frac{1}{L} \nabla f(x_t)
$$
Ponendo $y = x_{t+1}$ e $x = x_t$, la disuguaglianza di L-smoothness diventa:
$$
f(x_{t+1}) \le f(x_t) + \nabla f(x_t)^T(x_{t+1}-x_t) + \frac{L}{2}||x_{t+1}-x_t||^2
$$
[03:55] Si sostituisce ora l'espressione per la differenza $(x_{t+1}-x_t) = -\frac{1}{L}\nabla f(x_t)$:
$$
f(x_{t+1}) \le f(x_t) + \nabla f(x_t)^T \left(-\frac{1}{L}\nabla f(x_t)\right) + \frac{L}{2}\left\|-\frac{1}{L}\nabla f(x_t)\right\|^2
$$
[04:01] Semplificando i termini si ottiene:
- Il prodotto scalare diventa: $-\frac{1}{L} \nabla f(x_t)^T \nabla f(x_t) = -\frac{1}{L} ||\nabla f(x_t)||^2$.
- Il termine quadratico diventa: $\frac{L}{2} \frac{1}{L^2} ||\nabla f(x_t)||^2 = \frac{1}{2L} ||\nabla f(x_t)||^2$.
[04:08] Combinando i termini, la disuguaglianza si trasforma in:
$$
f(x_{t+1}) \le f(x_t) - \frac{1}{L} ||\nabla f(x_t)||^2 + \frac{1}{2L} ||\nabla f(x_t)||^2
$$
[04:18] Sommando i termini simili contenenti la norma del gradiente, si ottiene la relazione finale, che dimostra la condizione di decrescita sufficiente:
$$
f(x_{t+1}) \le f(x_t) - \frac{1}{2L} ||\nabla f(x_t)||^2
$$
### Caso 3: Funzione Convessa e L-smooth
#### Ipotesi e Tasso di Convergenza
[04:25] Si considera ora il caso di una funzione che è contemporaneamente convessa e L-smooth.
[04:30] Mentre il risultato precedente mostrava una decrescita tra iterazioni consecutive, l'obiettivo qui è dimostrare che, dopo $T$ iterazioni, l'errore all'ultima iterata è limitato da una costante che dipende dal numero di iterazioni e dalla distanza iniziale dalla soluzione. La disuguaglianza da dimostrare è:
$$
f(x_T) - f(x^*) \le \frac{L ||x_1 - x^*||^2}{2T}
$$
[04:46] Assumendo, come nel primo caso, che la distanza iniziale dalla soluzione sia limitata da una costante $R$ (cioè $||x_1 - x^*|| \le R$), il risultato può essere riscritto come:
$$
f(x_T) - f(x^*) \le \frac{L R^2}{2T}
$$
[04:56] Se si desidera raggiungere una tolleranza $\epsilon$, il numero di iterazioni $T$ necessario per garantire $f(x_T) - f(x^*) \le \epsilon$ è dell'ordine di:
$$
T \approx \frac{L R^2}{2\epsilon}
$$
[05:04] La differenza fondamentale rispetto al primo caso (gradiente limitato) è che il numero di iterazioni è ora proporzionale a $1/\epsilon$, anziché a $1/\epsilon^2$. L'aggiunta dell'ipotesi di L-smoothness ha quindi permesso di ottenere un tasso di convergenza significativamente migliore.
#### Dimostrazione del Risultato
[05:15] La dimostrazione parte nuovamente dall'analisi di base, utilizzando un passo di apprendimento $\gamma = 1/L$.
[05:20] Si sfrutta la condizione di decrescita sufficiente, che può essere riscritta per fornire un limite superiore alla norma del gradiente in funzione della decrescita della funzione obiettivo:
$$
||\nabla f(x_t)||^2 \le 2L (f(x_t) - f(x_{t+1}))
$$
[05:29] Partendo dalla disuguaglianza fondamentale derivata dalla convessità, $f(x_t) - f(x^*) \le \nabla f(x_t)^T(x_t - x^*)$, e sommando su tutte le iterazioni da $t=1$ a $T$, si ottiene:
$$
\sum_{t=1}^{T} (f(x_t) - f(x^*)) \le \sum_{t=1}^{T} \nabla f(x_t)^T(x_t - x^*)
$$
[05:35] L'idea chiave della dimostrazione consiste nel sommare la condizione di decrescita sufficiente su tutte le iterazioni.
[05:40] Questa operazione porta a una somma telescopica. Sommando i termini $f(x_t) - f(x_{t+1})$ per $t$ da 1 a $T$, i termini intermedi si cancellano a vicenda.
[05:50] Il risultato della somma telescopica è la differenza tra il valore della funzione alla prima iterazione e quello all'iterazione $T+1$:
$$
\sum_{t=1}^{T} (f(x_t) - f(x_{t+1})) = f(x_1) - f(x_{T+1})
$$
Poiché $x^*$ è il punto di minimo, si ha $f(x^*) \le f(x_{T+1})$. Di conseguenza, la somma può essere maggiorata come segue: $f(x_1) - f(x_{T+1}) \le f(x_1) - f(x^*)$.
[06:00] Combinando i vari passaggi, si ottiene un limite per la media dei gradienti.
[06:05] Questo limite viene inserito nell'equazione originale dell'analisi di base.
[06:15] Dopo aver riorganizzato i termini, si ottiene una disuguaglianza che permette di raggruppare i valori funzionali sul lato sinistro.
[06:20] Anche in questo passaggio si presenta una somma telescopica, il cui risultato finale è limitato superiormente da $\frac{L}{2} ||x_1 - x^*||^2$.
[06:28] Grazie alla condizione di decrescita sufficiente, sappiamo che la sequenza dei valori funzionali $f(x_t)$ è decrescente, ovvero $f(x_{t+1}) \le f(x_t)$. Questo implica che $f(x_T)$ è il valore più piccolo tra tutti quelli calcolati durante le $T$ iterazioni.
[06:39] Di conseguenza, la somma degli errori funzionali può essere minorata come segue:
$$
\sum_{t=1}^{T} (f(x_t) - f(x^*)) \ge T (f(x_T) - f(x^*))
$$
[06:48] Combinando tutti questi risultati, si giunge alla disuguaglianza finale:
$$
f(x_T) - f(x^*) \le \frac{L ||x_1 - x^*||^2}{2T}
$$
Questo limite mostra come l'errore all'ultima iterazione dipenda dalla distanza iniziale dalla soluzione e decresca linearmente con il numero di iterazioni $T$.
### Caso 4: Funzione $\mu$-fortemente Convessa e L-smooth
#### Ipotesi e Risultati Principali
[07:00] Si introduce un'ulteriore e più stringente condizione sulla funzione: la $\mu$-forte convessità.
- **Definizione di Funzione $\mu$-fortemente Convessa**: Una funzione $f$ si dice $\mu$-fortemente convessa se esiste una costante $\mu > 0$ tale che la seguente disuguaglianza è valida per ogni coppia di punti $x, y$:
  $$
  f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}||y-x||^2
  $$
  Questa condizione implica che la funzione è limitata inferiormente da una parabola quadratica.
[07:05] In questo caso, si ottengono due risultati fondamentali: un limite sulla distanza tra le iterate e la soluzione ottima, e un limite sulla differenza dei valori funzionali.
[07:13] Si dimostra che la distanza al quadrato tra l'iterata $x_{t+1}$ e la soluzione $x^*$ si riduce a ogni passo di un fattore costante:
$$
||x_{t+1} - x^*||^2 \le \left(1 - \frac{\mu}{L}\right) ||x_t - x^*||^2
$$
Il rapporto $\kappa = L/\mu$ è noto come **numero di condizionamento** del problema di ottimizzazione.
[07:25] Sfruttando questa relazione di contrazione, si può dimostrare che il numero di passi $T$ richiesto per raggiungere una data tolleranza $\epsilon$ è dell'ordine di:
$$
T \approx \kappa \log\left(\frac{1}{\epsilon}\right)
$$
Questo tipo di convergenza, detta lineare (o geometrica), è molto più rapida di quelle viste in precedenza.
#### Il Concetto di Numero di Condizionamento
[07:35] Il numero di condizionamento è un concetto centrale in analisi numerica.
[07:41] Per un problema numerico generico, che può essere rappresentato in forma astratta come $F(x, d) = 0$ (dove $x$ è l'incognita e $d$ sono i dati), il numero di condizionamento quantifica la sensibilità della soluzione $x$ a piccole variazioni (perturbazioni) nei dati $d$.
[08:00] Un esempio classico è quello dei sistemi di equazioni lineari $Ax=b$. Per una matrice quadrata e invertibile $A$, il numero di condizionamento è definito come:
$$
\kappa(A) = ||A|| \cdot ||A^{-1}||
$$
dove la norma può essere una qualsiasi norma matriciale indotta. Questo numero è sempre maggiore o uguale a 1.
[08:10] Un numero di condizionamento elevato indica che il problema è "mal condizionato": piccole perturbazioni nei dati di input (ad esempio, nel vettore $b$ o nella matrice $A$) possono provocare grandi variazioni nella soluzione $x$.
[08:18] La velocità di convergenza di molti metodi iterativi, come il metodo del gradiente o il metodo del gradiente coniugato, dipende fortemente dal numero di condizionamento della matrice del sistema. Per il metodo del gradiente, la dipendenza è da $\kappa(A)$, mentre per il gradiente coniugato, un metodo più avanzato, è da $\sqrt{\kappa(A)}$.
[08:33] Questo concetto si estende al contesto dell'ottimizzazione non lineare, dove il numero di condizionamento $\kappa = L/\mu$ governa la velocità di convergenza dell'algoritmo del gradiente.
[08:50] Un esempio famoso di matrice mal condizionata è la matrice di Hilbert, per la quale una minima perturbazione nel termine noto $b$ può produrre una soluzione completamente diversa da quella attesa.
#### Dimostrazione dei Risultati
[09:08] La dimostrazione si basa su una delle relazioni ottenute nell'analisi di base, che lega il prodotto scalare del gradiente con la differenza tra l'iterata corrente e la soluzione ottima. Questa relazione, che indichiamo come BA1 (*Basic Analysis 1*), è:
$$
\nabla f(x_t)^T(x_t - x^*) = \frac{1}{2\gamma} (||x_t - x^*||^2 - ||x_{t+1} - x^*||^2) + \frac{\gamma}{2} ||\nabla f(x_t)||^2
$$
[09:25] Si utilizza inoltre la definizione di $\mu$-forte convessità. Scegliendo $x = x_t$ e $y = x^*$ nella disuguaglianza di forte convessità e riarrangiando i termini, si ottiene un limite inferiore per il prodotto scalare $\nabla f(x_t)^T(x_t - x^*)$. Questa relazione, che indichiamo come SC2 (*Strong Convexity 2*), è:
$$
\nabla f(x_t)^T(x_t - x^*) \ge f(x_t) - f(x^*) + \frac{\mu}{2} ||x_t - x^*||^2
$$
[09:50] Combinando le due condizioni BA1 e SC2, e dopo alcuni passaggi algebrici, si ottiene una relazione che limita l'errore all'iterazione $t+1$:
$$
||x_{t+1} - x^*||^2 \le (1 - \mu\gamma) ||x_t - x^*||^2 - 2\gamma(f(x_t) - f(x^*)) + \gamma^2 ||\nabla f(x_t)||^2
$$
[10:05] In questa disuguaglianza, il termine $(1 - \mu\gamma) ||x_t - x^*||^2$ rappresenta il fattore di contrazione desiderato. Gli ultimi due termini, $- 2\gamma(f(x_t) - f(x^*)) + \gamma^2 ||\nabla f(x_t)||^2$, possono essere visti come un "rumore" che perturba la convergenza.
[10:12] Il fattore $(1 - \mu\gamma)$, se minore di 1, garantisce una riduzione dell'errore a ogni passo. L'obiettivo è dimostrare che la somma dei termini di "rumore" è negativa o nulla, in modo da non ostacolare la convergenza.
[10:25] Scegliendo il passo di apprendimento $\gamma = 1/L$, si possono sfruttare le proprietà della funzione.
[10:30] Dalle proprietà di L-smoothness e convessità, si può derivare la seguente disuguaglianza, che lega l'errore funzionale alla norma del gradiente:
$$
f(x_t) - f(x^*) \ge \frac{1}{2L} ||\nabla f(x_t)||^2
$$
[10:48] Sostituendo questa relazione nel termine di "rumore" e utilizzando $\gamma = 1/L$, si ha:
$$
\text{Rumore} = -\frac{2}{L}(f(x_t) - f(x^*)) + \frac{1}{L^2} ||\nabla f(x_t)||^2
$$
Utilizzando la disuguaglianza $f(x_t) - f(x^*) \ge \frac{1}{2L} ||\nabla f(x_t)||^2$, si ottiene:
$$
\text{Rumore} \le -\frac{2}{L}\left(\frac{1}{2L} ||\nabla f(x_t)||^2\right) + \frac{1}{L^2} ||\nabla f(x_t)||^2 = -\frac{1}{L^2} ||\nabla f(x_t)||^2 + \frac{1}{L^2} ||\nabla f(x_t)||^2 = 0
$$
Una derivazione più attenta mostra che il termine di rumore è strettamente non positivo.
[11:10] Poiché il termine di rumore è non positivo, la disuguaglianza per l'errore si semplifica notevolmente:
$$
||x_{t+1} - x^*||^2 \le (1 - \mu\gamma) ||x_t - x^*||^2
$$
[11:18] Sostituendo il valore scelto per il passo di apprendimento, $\gamma = 1/L$, si ottiene la relazione di contrazione lineare dell'errore:
$$
||x_{t+1} - x^*||^2 \le \left(1 - \frac{\mu}{L}\right) ||x_t - x^*||^2
$$
[11:24] Applicando questa relazione ricorsivamente per $T$ iterazioni, si ottiene la convergenza lineare della distanza delle iterate dalla soluzione:
$$
||x_T - x^*||^2 \le \left(1 - \frac{\mu}{L}\right)^T ||x_1 - x^*||^2
$$
[11:33] Per dimostrare il secondo risultato, relativo alla convergenza dei valori funzionali, si utilizza una caratterizzazione della L-smoothness.
[11:38] Si osserva che il gradiente calcolato nel punto di minimo è nullo, ovvero $\nabla f(x^*) = 0$.
[11:42] Sfruttando la proprietà di L-smoothness, si può dimostrare la seguente relazione:
$$
f(x_t) - f(x^*) \le \frac{L}{2} ||x_t - x^*||^2
$$
Sostituendo in questa disuguaglianza il risultato ottenuto per la convergenza della distanza delle iterate, si dimostra che anche i valori funzionali convergono linearmente al valore ottimo.
### Riepilogo dei Tassi di Convergenza
[12:00] La tabella seguente riassume i tassi di convergenza ottenuti per il metodo del gradiente in base alle diverse ipotesi sulla funzione obiettivo $f$. Il tasso è espresso come il numero di iterazioni $T$ necessarie per raggiungere una tolleranza $\epsilon$.
| Ipotesi sulla Funzione $f$ | Numero di Iterazioni $T$ per Tolleranza $\epsilon$ |
| :--- | :--- |
| Convessa, Gradiente Limitatato | $O(1/\epsilon^2)$ |
| Convessa, L-smooth | $O(1/\epsilon)$ |
| $\mu$-fortemente Convessa, L-smooth | $O(\kappa \log(1/\epsilon))$ |
[12:10] Come si evince dalla tabella, l'aggiunta di ipotesi che rendono la funzione "meglio comportata" (più regolare e strutturata) si traduce in un miglioramento significativo del tasso di convergenza dell'algoritmo.
## Capitolo 2: Scelta del Passo di Apprendimento e Ottimizzazione Vincolata
### La Scelta Pratica del Passo di Apprendimento (Learning Rate)
[12:20] Tutte le dimostrazioni di convergenza analizzate si basano sull'assunzione che il passo di apprendimento $\gamma$ abbia un valore specifico, calcolato in funzione di costanti come $L$ (costante di Lipschitz), $B$ (limite del gradiente) o $R$ (limite sulla distanza iniziale).
[12:35] Nella pratica, queste costanti sono quasi sempre sconosciute. Di conseguenza, sorge il problema di come scegliere un valore ragionevole per $\gamma$.
[12:44] Il passo di apprendimento è un iperparametro cruciale del modello, che viene tipicamente ottimizzato attraverso un processo di "tentativi ed errori" (*trial and error*), guidato dall'osservazione dell'andamento della funzione di costo su un insieme di dati di validazione.
[12:58] Esistono, tuttavia, metodi più strutturati per la scelta di $\gamma$, come il metodo della "ricerca lineare" (*line search*).
### Il Metodo della Line Search
[13:02] L'idea fondamentale della ricerca lineare è scegliere il passo di apprendimento $\gamma$ in modo "intelligente" a ogni iterazione dell'algoritmo.
[13:07] A differenza dell'analisi teorica, dove $\gamma$ è una costante, in questo approccio $\gamma$ diventa adattivo: a ogni iterazione $k$, si sceglie un passo specifico $\gamma_k$.
[13:16] Partendo dall'iterata corrente $x_k$ e dalla direzione di discesa $d_k = -\nabla f(x_k)$, si cerca il valore di $\gamma$ che minimizza la funzione monodimensionale $\phi(\gamma)$, definita come il valore della funzione obiettivo lungo la direzione di discesa:
$$
\phi(\gamma) = f(x_k + \gamma d_k)
$$
[13:25] Esistono due strategie principali per eseguire questa minimizzazione:
1.  **Ricerca Lineare Esatta (*Exact Line Search*)**: Trova il valore di $\gamma$ che minimizza esattamente la funzione $\phi(\gamma)$.
2.  **Ricerca Lineare Inesatta (*Inexact Line Search*) o Backtracking**: Cerca un valore approssimato di $\gamma$ che garantisca una diminuzione sufficiente del valore della funzione, senza richiedere la minimizzazione esatta.
### Exact Line Search
[13:40] L'obiettivo della ricerca lineare esatta è trovare il valore $\gamma > 0$ che minimizza la funzione $\phi(\gamma) = f(x_k + \gamma d_k)$.
[13:48] Poiché $x_k$ e $d_k$ sono noti all'iterazione $k$, $\phi(\gamma)$ è una funzione della sola variabile $\gamma$.
[14:05] Per trovare il minimo, si calcola la derivata prima di $\phi(\gamma)$ rispetto a $\gamma$ e la si pone uguale a zero. Applicando la regola della catena, si ottiene:
$$
\phi'(\gamma) = \nabla f(x_k + \gamma d_k)^T d_k = 0
$$
[14:15] Poiché la nuova iterata è $x_{k+1} = x_k + \gamma d_k$, la condizione $\phi'(\gamma)=0$ ha un'interpretazione geometrica chiara: il gradiente calcolato nella nuova iterata, $\nabla f(x_{k+1})$, deve essere ortogonale alla direzione di ricerca precedente, $d_k$.
[14:26] **Vantaggi e Svantaggi**:
- **Vantaggi**: Se la funzione $f$ ha una forma semplice (ad esempio, quadratica), è possibile trovare una formula analitica chiusa per il $\gamma$ ottimale. Teoricamente, questo metodo garantisce la massima riduzione del valore funzionale a ogni passo e, di conseguenza, converge nel minor numero di iterazioni.
- **Svantaggi**: Se $f$ è una funzione complessa, come accade nella maggior parte delle applicazioni reali, trovare il minimo esatto di $\phi(\gamma)$ può essere computazionalmente molto oneroso o addirittura impossibile. Per questo motivo, la ricerca lineare esatta è quasi mai utilizzata in pratica.
### Inexact Line Search (Backtracking)
[15:10] Il metodo del backtracking è un approccio approssimato che persegue lo stesso obiettivo della ricerca esatta, ma senza la necessità di trovare il minimo preciso.
[15:18] Si basa su una procedura iterativa per trovare un valore di $\gamma$ che sia "sufficientemente buono".
[15:29] **Algoritmo di Backtracking**:
1.  Si scelgono tre parametri: una stima iniziale per il passo, $\bar{\gamma}$, e due costanti $c \in (0, 1)$ e $\tau \in (0, 1)$.
    - $c$ è una costante che controlla la condizione di decrescita sufficiente (nota come condizione di Armijo).
    - $\tau$ è un fattore di riduzione (*shrink factor*) utilizzato per diminuire il passo.
2.  Si inizializza il passo di apprendimento con la stima iniziale: $\gamma = \bar{\gamma}$.
3.  Si verifica la seguente condizione, nota come **condizione di Armijo**:
    $$
    f(x_k + \gamma d_k) > f(x_k) + c \gamma \nabla f(x_k)^T d_k
    $$
    Poiché la direzione di discesa è $d_k = -\nabla f(x_k)$, la condizione può essere riscritta come:
    $$
    f(x_k - \gamma \nabla f(x_k)) > f(x_k) - c \gamma ||\nabla f(x_k)||^2
    $$
    Questa condizione verifica se il passo $\gamma$ produce una diminuzione "sufficiente" del valore della funzione. Il lato destro rappresenta una retta con pendenza inferiore a quella della funzione nel punto $x_k$.
4.  **Ciclo**: Finché la condizione di Armijo è vera, significa che il passo $\gamma$ è troppo grande e non garantisce una decrescita adeguata. Si riduce quindi $\gamma$ moltiplicandolo per il fattore di riduzione $\tau$:
    $$
    \gamma \leftarrow \tau \gamma
    $$
5.  **Uscita**: Quando la condizione di Armijo diventa falsa, si è trovato un valore di $\gamma$ adeguato. Si imposta il passo per l'iterazione corrente del gradiente, $\gamma_k$, a questo valore di $\gamma$ e si procede con l'aggiornamento di $x_k$ a $x_{k+1}$.
[16:25] In sintesi, all'interno di ogni iterazione del metodo del gradiente, si esegue un ciclo di sotto-iterazioni (il backtracking) per calcolare un passo di apprendimento $\gamma_k$ appropriato per l'aggiornamento successivo.
### Introduzione alla Scelta del Passo di Apprendimento (Gamma)
[00:00] L'algoritmo di *backtracking line search* è una delle tecniche disponibili nelle librerie di apprendimento automatico per la scelta del parametro $\gamma$. Un'altra possibilità consiste nell'utilizzare una cosiddetta **schedulazione** per $\gamma$.
[00:06] L'idea di base della schedulazione è quella di ridurre progressivamente il valore del passo di apprendimento man mano che l'algoritmo si avvicina alla soluzione, in modo da poterla approssimare con maggiore precisione. Un esempio pratico di questa strategia consiste nel definire il passo $\gamma$ all'iterazione $k$ come inversamente proporzionale a $k$:
$$
\gamma_k = \frac{1}{k}
$$
Questa è solo una delle molteplici scelte possibili per la schedulazione del passo di apprendimento.
### Analisi del Backtracking Line Search
#### Vantaggi del Backtracking Line Search
[00:20] Il metodo del *backtracking line search* presenta diversi vantaggi significativi:
- **Praticità ed Efficienza**: È una procedura iterativa che si dimostra pratica ed efficiente nella maggior parte dei casi.
- **Robustezza**: Garantisce quasi sempre di ottenere un valore ragionevole per $\gamma$, adattandosi alle caratteristiche locali della funzione.
- **Diffusione**: È una tecnica ampiamente utilizzata e implementata in molti pacchetti software e applicazioni pratiche.
#### Svantaggi del Backtracking Line Search
[00:30] D'altra parte, questo metodo introduce anche alcuni svantaggi. Il principale è la necessità di definire ulteriori iperparametri, ovvero le costanti $c$, $\tau$ e la stima iniziale $\bar{\gamma}$. Sebbene il metodo permetta di determinare automaticamente un buon valore per $\gamma$, introduce nuove costanti che devono essere scelte a priori.
[00:42] È importante notare che, sebbene in linea di principio i valori di $c$ e $\tau$ possano essere adattati a diverse funzioni, in pratica si utilizzano quasi sempre valori di default. Anche nelle implementazioni di librerie note come TensorFlow o PyTorch, i valori predefiniti per questi parametri si dimostrano efficaci per la stragrande maggioranza delle applicazioni.
[00:59] Un altro svantaggio è legato al numero di iterazioni. Poiché il valore di $\gamma$ calcolato con il backtracking non è il valore esatto che minimizza la funzione lungo la direzione di discesa (come avverrebbe con l'*exact line search*), il numero totale di iterazioni necessarie per raggiungere la stessa accuratezza sarà tendenzialmente superiore.
### Riepilogo dei Metodi per la Scelta di Gamma
[01:12] La tabella seguente riassume le caratteristiche dei diversi approcci al metodo del gradiente discendente analizzati, in base alla strategia di scelta del passo di apprendimento $\gamma$:
1.  **Gradiente Discendente con $\gamma$ fisso**: Semplice da implementare, ma la scelta di $\gamma$ è critica e non adattiva.
2.  **Gradiente Discendente con *Exact Line Search***: Garantisce la massima discesa a ogni passo, ma è computazionalmente troppo oneroso per la maggior parte delle applicazioni reali.
3.  **Gradiente Discendente con *Backtracking Line Search***: Un compromesso pratico che adatta $\gamma$ a ogni iterazione, garantendo una convergenza robusta a costo di introdurre nuovi iperparametri.
### Ottimizzazione Vincolata: Introduzione
[01:20] Finora, l'analisi del gradiente discendente ha riguardato esclusivamente problemi di **ottimizzazione non vincolata**, ovvero la minimizzazione di una funzione $f(x)$ senza alcuna restrizione sulla variabile $x$.
[01:27] In molte applicazioni pratiche, può essere necessario minimizzare una funzione soggetta a determinati vincoli su $x$.
[01:33] Un problema di ottimizzazione vincolata può essere formulato come la minimizzazione di una funzione $f(x)$ soggetta al vincolo che la soluzione $x$ appartenga a un insieme chiuso e convesso, denotato con $C$.
#### Il Legame con la Regolarizzazione
[01:44] Il concetto di ottimizzazione vincolata è strettamente legato alla **regolarizzazione**. Quando si applica la regolarizzazione, si cerca di minimizzare una funzione di costo con il vincolo aggiuntivo che il vettore dei pesi abbia norma minima o sia sparso. Questi vincoli qualitativi sono stati tradotti matematicamente nell'appartenenza del vettore dei pesi a una palla unitaria misurata in norma L2 o in norma L1. Entrambi questi insiemi (le palle L2 e L1) sono esempi di insiemi chiusi e convessi.
[02:06] Altri esempi comuni di vincoli includono:
- **Vincoli di non negatività**: i componenti del vettore soluzione devono essere non negativi ($x_i \ge 0$).
- **Vincoli a scatola (*box constraints*)**: ogni componente del vettore dei pesi deve avere un valore compreso tra un limite inferiore e uno superiore ($l_i \le x_i \le u_i$).
- **Appartenenza a una palla unitaria**: come già menzionato, il vettore deve appartenere a una palla di raggio fissato, definita rispetto a una certa norma (es. L1, L2).
### Il Problema del Gradiente Discendente con Vincoli
[02:17] L'applicazione del metodo del gradiente discendente standard a un problema vincolato presenta una criticità fondamentale. Anche se l'iterata corrente, $x_k$, appartiene all'insieme ammissibile $C$, non vi è alcuna garanzia che l'iterata successiva, $x_{k+1} = x_k - \gamma \nabla f(x_k)$, rimanga all'interno di $C$.
[02:28] L'idea per risolvere questo problema è modificare l'algoritmo del gradiente discendente per garantire che tutte le iterate generate rimangano sempre all'interno dell'insieme convesso $C$ che definisce i vincoli.
### L'Operatore di Proiezione
[02:38] La soluzione formale a questo problema consiste nell'introdurre l'**operatore di proiezione**.
[02:42] Dato un punto $y$ e un insieme chiuso e convesso $C$, l'operatore di proiezione su $C$, denotato con $\text{proj}_C(y)$, restituisce il punto appartenente a $C$ che ha la minima distanza da $y$. Questo concetto è stato già incontrato nel contesto dell'approssimazione ai minimi quadrati, dove il vettore delle etichette reali veniva proiettato sullo spazio generato dalle colonne della matrice dei dati.
[03:00] Se il punto $y$ si trova già all'interno dell'insieme $C$, la sua proiezione è il punto stesso. Se, invece, $y$ è esterno a $C$, la sua proiezione sarà un punto che si trova sulla frontiera di $C$.
[03:07] Ad esempio, se $C$ è un disco nel piano e $y$ è un punto esterno al disco, la sua proiezione sarà il punto sulla circonferenza del disco che si trova sulla retta congiungente il centro del disco e $y$.
### Il Metodo del Gradiente Proiettato
[03:14] Il **metodo del gradiente proiettato** (*projected gradient method*) è una tecnica pratica, implementata in diverse librerie software, che estende il gradiente discendente ai problemi vincolati.
[03:27] L'idea fondamentale di questo metodo può essere riassunta nel motto: "Prima discendi, poi proietta" (*Descend, then project*).
[03:32] L'algoritmo funziona nel seguente modo:
1.  Si parte da un punto $x_k$ che appartiene all'insieme ammissibile $C$.
2.  Si esegue un passo standard di discesa del gradiente, ottenendo un punto intermedio $y_k = x_k - \gamma \nabla f(x_k)$, che potrebbe trovarsi al di fuori di $C$.
3.  Si proietta questo punto intermedio sull'insieme ammissibile $C$ per ottenere la nuova iterata $x_{k+1}$.
[03:41] L'intera operazione di aggiornamento può essere scritta in una singola riga:
$$
x_{k+1} = \text{proj}_C (x_k - \gamma \nabla f(x_k))
$$
dove $\text{proj}_C$ indica l'operatore di proiezione sull'insieme $C$.
[03:53] Un punto cruciale per l'efficienza di questo metodo è che il calcolo della proiezione non deve essere computazionalmente troppo oneroso.
### Esempi di Proiezione
#### Proiezione L2
[04:03] Un esempio importante è la proiezione sulla palla unitaria (o di raggio $r$) definita dalla norma L2. In questo caso, l'insieme $C$ è $C = \{x : ||x||_2 \le r\}$.
[04:10] L'algoritmo prevede di calcolare il passo del gradiente e poi di proiettare il risultato su questa palla.
[04:18] Se il punto ottenuto dopo il passo di discesa, $y$, si trova all'interno o sulla frontiera della palla (ovvero $||y||_2 \le r$), allora il punto è già ammissibile e la sua proiezione coincide con il punto stesso.
[04:31] Se, invece, il punto $y$ si trova all'esterno della palla ($||y||_2 > r$), la proiezione consiste nel "restringere" la lunghezza del vettore $y$ per riportarlo sulla frontiera, mantenendone la direzione. Questo si ottiene riscalando il vettore $y$ per il fattore $r/||y||_2$.
[04:46] La formula compatta per la proiezione sulla palla L2 di raggio $r$ è:
$$
\text{proj}_C(y) = y \cdot \min\left(1, \frac{r}{||y||_2}\right)
$$
Questa operazione è computazionalmente molto semplice, poiché richiede solo il calcolo della norma del vettore e una moltiplicazione scalare.
[05:00] Questo risultato è coerente con l'interpretazione della regolarizzazione L2, che mira a trovare soluzioni con norma piccola o limitata da un valore $r$.
[05:13] Come già osservato, questo tipo di vincolo non promuove la sparsità della soluzione, ma favorisce pesi di piccola entità.
#### Proiezione L1
[05:21] Se si considera la proiezione sulla palla unitaria definita dalla norma L1, non è possibile trovare un'espressione analitica esplicita e semplice per l'operatore di proiezione, specialmente in spazi multidimensionali.
[05:28] Tuttavia, esistono algoritmi numerici efficienti che permettono di calcolare la proiezione L1 in un tempo ragionevole, tipicamente con una complessità computazionale di $O(n \log n)$, dove $n$ è la dimensione dello spazio.
[05:38] La proprietà fondamentale della palla L1, già osservata in precedenza, è la sua forma geometrica (un rombo in 2D, un iper-ottaedro in dimensioni superiori), che promuove la **sparsità**. Pertanto, proiettare una soluzione su tale insieme favorisce l'azzeramento di alcune delle sue componenti.
### Collegamento tra Ottimizzazione Vincolata e Regolarizzazione
[05:50] Il problema di ottimizzazione vincolata, che consiste nel minimizzare $f(x)$ con il vincolo $x \in C$, può essere risolto con il metodo del gradiente proiettato.
[06:07] La controparte di questo problema è il problema di minimizzazione regolarizzata, formulato come:
$$
\min_x f(x) + \lambda \Omega(x)
$$
dove:
- $f(x)$ è la funzione di costo.
- $\Omega(x)$ è un termine di regolarizzazione che penalizza soluzioni indesiderate (es. $\Omega(x) = ||x||_p$).
- $\lambda$ è un iperparametro (spesso interpretato come un moltiplicatore di Lagrange) che bilancia il peso tra la minimizzazione di $f(x)$ e la penalità di regolarizzazione.
Questo tipo di problema può essere risolto con una variante del metodo del gradiente nota come **gradiente discendente prossimale**.
[06:20] L'operatore prossimale è una generalizzazione dell'operatore di proiezione. La sua applicazione garantisce una discesa lungo la funzione regolarizzata.
[06:34] Esiste una relazione profonda tra i due approcci: per un dato valore del raggio $r$ nel problema vincolato, esiste sempre un valore del moltiplicatore $\lambda$ tale che il problema vincolato e il problema regolarizzato ammettono la stessa soluzione. In altre parole, l'ottimizzazione vincolata e la regolarizzazione sono due formulazioni equivalenti dello stesso problema di base.
#### Caso della Norma L2 (Ridge Regression)
[06:51] Nel caso della norma L2, si ha la seguente corrispondenza:
- **Ottimizzazione Vincolata**: $\min_x f(x)$ con il vincolo $||x||_2 \le r$.
- **Regolarizzazione (Ridge Regression)**: $\min_x f(x) + \lambda ||x||_2^2$.
[07:00] In questo caso, l'operatore prossimale associato alla regolarizzazione L2 corrisponde all'operatore di *weight decay* (decadimento dei pesi).
[07:08] L'aggiornamento tramite *weight decay* consiste nel ridurre il valore dei pesi a ogni iterazione tramite un fattore di riscalamento. Questo approccio è anche noto come *soft penalty*, poiché tutti i pesi vengono contratti verso lo zero, senza essere forzatamente azzerati.
#### Caso della Norma L1 (LASSO)
[07:22] Per la norma L1, l'analogia è simile. La differenza principale risiede nella complessità dell'operatore.
[07:34] Nel problema regolarizzato con norma L1 (LASSO), l'operatore prossimale corrispondente è noto come **operatore di soft thresholding**.
[07:43] L'espressione analitica di questo operatore è esattamente quella derivata in precedenza analizzando le proprietà della regolarizzazione L1. Esso agisce "tagliando" le componenti del vettore vicine a zero e contraendo le altre, promuovendo così la sparsità.
### Riepilogo del Gradiente Proiettato e Prossimale
[08:03] In sintesi:
- Il **gradiente proiettato** si applica a problemi di ottimizzazione vincolata. L'algoritmo consiste in un passo di discesa del gradiente seguito da una proiezione sull'insieme delle soluzioni ammissibili.
- Il **gradiente prossimale** si applica a problemi di ottimizzazione regolarizzata e utilizza un operatore (l'operatore prossimale) che generalizza l'idea della proiezione.
- I casi con vincoli (o regolarizzazione) L1 e L2 sono di fondamentale importanza pratica.
[08:18] Sebbene il gradiente discendente (e le sue varianti) sia un metodo fondamentale e didatticamente molto istruttivo, non è l'algoritmo più utilizzato nelle applicazioni moderne di machine learning. Il metodo di elezione è una sua variante stocastica.
[08:31] La discussione proseguirà con l'analisi del gradiente discendente stocastico.
## Capitolo 3: Il Gradiente Discendente Stocastico (SGD)
### Introduzione al Gradiente Discendente Stocastico (SGD)
[08:42] Si introduce ora il metodo di ottimizzazione più importante nel contesto dell'apprendimento automatico: il **gradiente discendente stocastico** (SGD, *Stochastic Gradient Descent*).
[08:50] L'idea fondamentale dell'SGD nasce dalla struttura tipica delle funzioni di costo utilizzate nell'apprendimento automatico.
[09:00] Come visto in precedenza, la funzione di costo globale $F(x)$ è tipicamente definita come una media (o una somma) di funzioni di costo elementari $f_i(x)$, dove ciascuna $f_i(x)$ è calcolata su un singolo campione del dataset. La forma tipica è:
$$
F(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(x)
$$
dove $N$ è il numero totale di campioni nel dataset e $f_i(x)$ è la funzione di costo (ad esempio, l'errore quadratico) valutata sull'$i$-esimo campione.
### Differenza tra Gradiente Discendente e Gradiente Discendente Stocastico
[09:20] La regola di aggiornamento del gradiente discendente standard (spesso chiamato *batch gradient descent*) è:
$$
x_{t+1} = x_t - \gamma \nabla F(x_t) = x_t - \gamma \frac{1}{N} \sum_{i=1}^{N} \nabla f_i(x_t)
$$
Nel gradiente discendente stocastico, invece di calcolare il gradiente dell'intera funzione di costo $F$, che richiede di processare tutti gli $N$ campioni, si approssima il gradiente globale utilizzando una sola delle sue componenti.
[09:33] Ad ogni iterazione $t$, si seleziona casualmente un indice $i$ dall'insieme $\{1, 2, \dots, N\}$, e si approssima il gradiente vero $\nabla F(x_t)$ con il gradiente della singola componente $\nabla f_i(x_t)$.
[09:44] Ad esempio, se la funzione di costo è una somma di 100 termini, si sceglie casualmente un indice (es. 55), si calcola il gradiente solo di quel singolo termine e si utilizza questa approssimazione per aggiornare i parametri.
[09:56] La "stocasticità" del metodo deriva proprio da questa selezione casuale dell'indice a ogni iterazione. La regola di aggiornamento dell'SGD è quindi:
$$
x_{t+1} = x_t - \gamma \nabla f_i(x_t)
$$
dove $i$ è un indice scelto casualmente a ogni passo $t$.
### Vantaggi Computazionali dell'SGD
[10:08] Il motivo principale per cui si adotta questo approccio è di natura computazionale. Nelle applicazioni reali, il numero di campioni $N$ può essere enorme (milioni o miliardi). Calcolare il gradiente completo $\nabla F$ a ogni iterazione, che richiede un passaggio su tutto il dataset, diventa proibitivamente costoso.
[10:18] Utilizzando l'SGD, invece di calcolare una somma di $N$ gradienti, se ne calcola solo uno. Questo rende ogni passo di aggiornamento circa $N$ volte più economico e, di conseguenza, $N$ volte più veloce rispetto a un passo del gradiente discendente standard.
### Analisi Teorica dell'SGD
[10:29] La domanda fondamentale è se questa approssimazione sia legittima, ovvero se l'algoritmo SGD sia comunque in grado di ottimizzare la funzione di costo. La risposta è affermativa, e la sua giustificazione teorica si basa su alcune ipotesi chiave.
#### Ipotesi Fondamentale: Stimatore Imparziale del Gradiente
[10:44] Si denota con $g_t = \nabla f_i(x_t)$ il gradiente stocastico calcolato all'iterazione $t$. L'ipotesi di fondamentale importanza per l'analisi della convergenza dell'SGD è che questo gradiente stocastico sia uno **stimatore imparziale** (*unbiased estimator*) del vero gradiente.
[10:55] Matematicamente, ciò significa che il valore atteso del gradiente stocastico $g_t$, condizionato al valore dell'iterata $x_t$, è uguale al gradiente completo. Se la selezione dell'indice $i$ è uniforme, si ha:
$$
\mathbb{E}_{i}[g_t | x_t = x] = \mathbb{E}_{i}[\nabla f_i(x)] = \frac{1}{N} \sum_{i=1}^{N} \nabla f_i(x) = \nabla F(x)
$$
Questa proprietà garantisce che, in media, la direzione di aggiornamento dell'SGD coincida con la direzione del gradiente vero.
#### Convergenza in Valore Atteso
[11:10] Un'altra conseguenza della natura stocastica dell'algoritmo è che i risultati di convergenza non sono più validi puntualmente (cioè per una singola traiettoria), ma lo diventano **in valore atteso** (*in expectation*).
[11:21] In pratica, ciò significa che le dimostrazioni di convergenza viste per il gradiente deterministico possono essere adattate al caso stocastico applicando l'operatore di valore atteso e lavorando con le quantità medie anziché con i valori esatti.
[11:30] Utilizzando la proprietà di convessità e l'ipotesi di stimatore imparziale, è possibile derivare relazioni fondamentali analoghe a quelle del caso deterministico, ma valide in valore atteso.
### Risultati di Convergenza per l'SGD
[11:47] Si presentano ora i principali risultati di convergenza per l'SGD, senza entrare nei dettagli delle dimostrazioni.
#### Caso di Funzioni Convesse e Lipschitziane
[11:53] Per funzioni convesse e Lipschitziane, si mantengono ipotesi simili a quelle del gradiente discendente deterministico:
- La distanza iniziale dalla soluzione, $||x_0 - x^*||$, è limitata.
- Esiste un limite superiore per la varianza del gradiente stocastico, ovvero per il valore atteso del quadrato della sua norma.
[12:07] Il risultato di convergenza ottenuto è molto simile a quello del caso deterministico, con la differenza che il limite superiore si applica all'errore in valore atteso.
#### Caso di Funzioni Fortemente Convesse
[12:17] Per le funzioni fortemente convesse, sebbene la dimostrazione sia più complessa, si ottiene un risultato significativo. Se la funzione è differenziabile, fortemente convessa e il gradiente stocastico ha varianza limitata, si ottiene un limite superiore per la differenza (in valore atteso) tra il valore della funzione e il valore ottimo.
[12:30] Questo limite dipende dalla varianza del gradiente, dalla costante di forte convessità $\mu$ e dal numero di iterazioni $T$.
[12:40] Analogamente al caso deterministico, anche nel contesto stocastico l'ipotesi di forte convessità migliora la velocità di convergenza. La complessità, in termini di numero di iterazioni necessarie per raggiungere un'accuratezza $\epsilon$, passa da $O(1/\epsilon^2)$ (per funzioni convesse) a $O(1/\epsilon)$ (per funzioni fortemente convesse).
### Strategie di Campionamento nell'SGD
[13:14] Un aspetto cruciale nell'implementazione pratica dell'SGD è la strategia con cui si seleziona l'indice $i$ dall'insieme $\{1, \dots, N\}$ a ogni iterazione. Esistono due approcci principali.
#### 1. Campionamento con Reinserimento (With Replacement)
[13:25] In questa strategia, a ogni iterazione dell'SGD, si seleziona casualmente un indice dall'intero insieme di indici disponibili.
[13:32] L'indice scelto viene immediatamente "reinserito" nell'insieme, rendendolo disponibile per essere selezionato anche nelle iterazioni successive. Ciò implica che lo stesso campione potrebbe essere scelto più volte consecutivamente.
[13:42] **Vantaggio**: Il vantaggio principale di questo approccio è di natura teorica. Assumere che a ogni iterazione tutti gli indici siano disponibili garantisce che i gradienti stocastici calcolati in iterazioni diverse siano variabili casuali indipendenti e identicamente distribuite (i.i.d.). Questa ipotesi è fondamentale per dimostrare rigorosamente i risultati di convergenza dell'SGD.
[14:02] **Svantaggio**: Lo svantaggio è pratico: alcuni campioni potrebbero essere selezionati molte volte all'interno di un ciclo sul dataset, mentre altri potrebbero non essere mai scelti.
#### 2. Campionamento senza Reinserimento (Without Replacement)
[14:12] In questa strategia, una volta che un indice $i$ viene selezionato, non è più disponibile per le selezioni successive fino a quando tutti gli altri indici non sono stati scelti almeno una volta.
[14:21] **Implementazione pratica**: Questo approccio è molto più comune e semplice da implementare. All'inizio di ogni **epoca** (un passaggio completo sul dataset), si crea una lista degli indici da $1$ a $N$, la si mescola casualmente (*shuffle*), e poi si scorrono sequenzialmente gli indici di questa lista mescolata. Poiché l'ordine iniziale è casuale, la selezione degli indici risulta casuale ma senza ripetizioni all'interno di un'epoca.
[14:44] **Vantaggi**:
- Ogni punto del dataset viene utilizzato esattamente una volta per ogni epoca.
- Sperimentalmente, si osserva che questa strategia converge più velocemente rispetto al campionamento con reinserimento.
[14:58] **Svantaggio**: Dal punto di vista teorico, il fatto che gli indici non vengano reinseriti rende l'analisi della convergenza molto più complessa. I gradienti stocastici calcolati all'interno di un'epoca non sono più indipendenti, e per questo motivo la maggior parte dei risultati teorici si basa sulla strategia con reinserimento.
### Ottimizzazione in Machine Learning vs. Ottimizzazione Generale
[15:13] È fondamentale comprendere la differenza tra l'obiettivo dell'ottimizzazione in un contesto generale e quello dell'ottimizzazione nell'apprendimento automatico.
[15:18] **Ottimizzazione Generale (es. Ingegneria)**: Si consideri un problema di ottimizzazione della forma di un profilo alare per minimizzare la resistenza aerodinamica. L'obiettivo è trovare il valore esatto dei parametri di forma che minimizza la funzione "resistenza". In questo caso, si desidera trovare la soluzione migliore possibile, ovvero il minimo globale della funzione obiettivo.
[16:00] **Ottimizzazione in Machine Learning**: L'obiettivo non è semplicemente minimizzare la funzione di costo $F(w)$ sull'insieme di addestramento. L'obiettivo finale è trovare un vettore di pesi $w$ che non solo dia un valore basso alla funzione di costo, ma che permetta al modello di **generalizzare** bene a dati nuovi e mai visti.
[16:10] In altre parole, nel machine learning, raggiungere il minimo perfetto della funzione di costo sul training set spesso corrisponde a una situazione di **overfitting**, in cui il modello impara a memoria i dati di addestramento ma perde la sua capacità di fare previsioni accurate su nuovi dati.
[16:18] Questa differenza di obiettivi spiega perché il gradiente discendente stocastico è così efficace nell'apprendimento automatico. Non è necessario trovare il minimo esatto della funzione di costo, ma è sufficiente avvicinarsi a una regione di minimo che garantisca una buona generalizzazione. Al contrario, l'SGD non sarebbe la scelta ideale in un contesto ingegneristico che richiede di trovare il minimo esatto di una funzione obiettivo.
## Capitolo 4: Comportamento e Varianti del Gradiente Stocastico
### Introduzione all'Early Stopping e al Gradiente Stocastico
[00:00] L'argomento dell'arresto anticipato (*early stopping*) è stato già trattato, ma ora viene contestualizzato con un esempio significativo legato al metodo della discesa del gradiente stocastico.
[00:13] Si consideri un semplice problema di regressione lineare. La funzione di costo globale, $F(x)$, è la somma di $n$ funzioni di costo elementari, $f_i(x)$.
[00:24] Ciascuna funzione di costo elementare $f_i(x)$, associata all'$i$-esimo campione, ha la forma di una parabola:
$$
f_i(x) = \frac{1}{2} (a_i x - b_i)^2
$$
dove $x$ è il parametro scalare da ottimizzare, mentre $a_i$ e $b_i$ sono costanti specifiche del campione.
[00:36] Il punto di minimo di una singola parabola $f_i(x)$ si trova annullando il termine quadratico, ovvero per $x_i = \frac{b_i}{a_i}$.
[00:43] Il minimo globale della funzione di costo complessiva $F(x) = \sum_{i=1}^n f_i(x)$ si trova nel punto $x^*$ che corrisponde a una media pesata dei minimi individuali:
$$
x^* = \frac{\sum_{i=1}^n a_i b_i}{\sum_{i=1}^n a_i^2}
$$
Questo valore si ottiene calcolando il gradiente della funzione $F(x)$ e ponendolo uguale a zero.
### Interpretazione Grafica del Gradiente Stocastico
[00:54] La situazione può essere visualizzata graficamente. Le parabole tratteggiate rappresentano le singole funzioni di costo elementari $f_i(x)$, mentre la parabola nera rappresenta la funzione di costo globale $F(x)$, che è la loro somma.
[01:09] Ogni parabola elementare ha il proprio punto di minimo, e anche la funzione di costo globale ha un suo minimo globale, $x^*$.
[01:16] Analizziamo il comportamento del gradiente stocastico. Supponiamo di trovarci in un punto $x$ e di selezionare casualmente un campione (ad esempio, quello associato alla parabola blu) per calcolare il gradiente.
[01:28] Se il punto $x$ si trova in una zona lontana dal minimo globale (la "regione lontana" o *far-out region*), si osserva che la pendenza (il gradiente) di quasi tutte le parabole elementari ha lo stesso segno.
[01:40] In questa regione, il segno del gradiente di una singola parabola scelta a caso coincide con il segno del gradiente della funzione di costo globale (la parabola nera).
[01:48] In pratica, ciò significa che lontano dal minimo, il gradiente calcolato su un singolo campione casuale è un'indicazione affidabile della direzione di discesa verso il minimo globale. Sebbene l'esempio sia monodimensionale per semplicità, questo principio si estende a casi multidimensionali.
[02:03] In questa zona, il gradiente stocastico fornisce una direzione di discesa ragionevolmente buona.
### La Regione di Confusione
[02:08] La situazione cambia radicalmente quando l'algoritmo si avvicina al minimo ed entra nella cosiddetta "regione di confusione" (*region of confusion*).
[02:11] In questa zona, le pendenze delle singole parabole elementari possono avere segni discordanti. A seconda del campione scelto casualmente (ad esempio, quello associato alla parabola blu o a quella verde), il segno del gradiente può essere opposto.
[02:17] Ciò implica che, all'interno di questa regione, la direzione del passo di aggiornamento dipende in modo critico dal campione specifico selezionato, e il comportamento dell'algoritmo diventa molto più rumoroso e imprevedibile.
[02:28] Una simulazione del percorso seguito dal gradiente stocastico mostra questo comportamento. Partendo da un punto lontano, si osserva una fase iniziale in cui, sebbene la traiettoria non sia liscia, il metodo si muove progressivamente verso la regione del minimo.
[02:51] Successivamente, l'algoritmo entra in una fase in cui il suo comportamento diventa apparentemente caotico, con oscillazioni attorno al minimo. Questo effetto è amplificato se si utilizza un passo di apprendimento (*learning rate*) più grande.
[03:01] Questa è la manifestazione pratica della regione di confusione: l'uso di un singolo campione per calcolare il gradiente non garantisce più che la direzione scelta sia corretta, nemmeno per quanto riguarda il segno.
[03:14] Questo comportamento è tipico dell'SGD: una fase iniziale di rapida discesa, seguita da un movimento rumoroso e oscillatorio attorno al punto di minimo.
### Implicazioni della Regione di Confusione e Overfitting
[03:25] Dal punto di vista dell'apprendimento automatico, questo comportamento rumoroso non è necessariamente negativo.
[03:29] La presenza della regione di confusione impedisce all'algoritmo di convergere esattamente al minimo teorico della funzione di costo sul training set. Il punto finale raggiunto è solo in prossimità del minimo, ma non coincide con esso.
[03:36] Questo può essere un vantaggio, poiché aiuta a prevenire l'**overfitting**. Evitando di adattarsi perfettamente ai dati di addestramento, il modello può mantenere una migliore capacità di generalizzazione.
[03:41] Aumentare il numero di iterazioni non garantisce che l'algoritmo si avvicini progressivamente e stabilmente al minimo; continuerà a oscillare nella sua vicinanza.
[03:53] Questa immagine di una discesa rapida seguita da oscillazioni è fondamentale per comprendere il comportamento tipico del gradiente stocastico.
### Analisi Teorica della Convergenza del Gradiente Stocastico
[04:00] Se utilizzato con un tasso di apprendimento $\gamma$ costante, il gradiente stocastico non può convergere esattamente al minimo $x^*$, ma continuerà a "rimbalzare" (*bounce around*) nelle sue vicinanze.
[04:08] Esiste un risultato teorico che formalizza questo comportamento, analizzando la distanza al quadrato (in valore atteso) tra l'iterata $x_t$ e la soluzione ottima $x^*$.
[04:15] Per una funzione fortemente convessa, la disuguaglianza è la seguente:
$$
\mathbb{E}[\|x_t - x^*\|^2] \leq (1 - \gamma \mu)^t \|x_0 - x^*\|^2 + \frac{\gamma B^2}{\mu}
$$
dove:
- $\mathbb{E}[\cdot]$ indica il valore atteso.
- $x_t$ è la soluzione all'iterazione $t$.
- $x^*$ è la soluzione ottimale.
- $x_0$ è il punto di partenza.
- $\gamma$ è il tasso di apprendimento (costante).
- $\mu$ è la costante di forte convessità della funzione.
- $B^2$ è una costante che limita la varianza del gradiente stocastico.
[04:20] Il primo termine, $(1 - \gamma \mu)^t \|x_0 - x^*\|^2$, è analogo a quello visto per il gradiente discendente deterministico. Poiché $(1 - \gamma \mu)$ è una costante minore di 1, questo termine decresce esponenzialmente con il numero di iterazioni $t$ e tende a zero.
[04:28] Se questo fosse l'unico termine, l'algoritmo convergerebbe linearmente alla soluzione esatta.
[04:42] Tuttavia, nel caso stocastico è presente un secondo termine additivo, $\frac{\gamma B^2}{\mu}$.
[04:49] Questo termine è legato al tasso di apprendimento $\gamma$, alla varianza del gradiente $B^2$ e alla costante di forte convessità $\mu$. Se $\gamma$ è mantenuto costante, questo termine non andrà mai a zero.
[04:56] Dal punto di vista teorico, è proprio questo termine di errore residuo la causa della regione di confusione e delle oscillazioni attorno al minimo.
[05:08] Poiché $B$ e $\mu$ sono proprietà intrinseche della funzione, l'unico modo per ridurre questo errore residuo è agire sul tasso di apprendimento $\gamma$.
[05:16] Questo risultato teorico suggerisce la necessità di adottare strategie in cui $\gamma$ non sia costante, ma venga ridotto progressivamente (*scheduled*) durante l'addestramento, in modo che anche il secondo termine tenda a zero.
### Il Metodo Mini-Batch Gradient Descent
[05:26] In pratica, né il gradiente discendente classico (batch) né quello puramente stocastico (con un solo campione) sono le scelte più comuni. Si utilizza invece un approccio intermedio: la **discesa del gradiente mini-batch** (*mini-batch gradient descent*).
[05:34] Riepilogando i tre approcci:
- **Gradient Descent (Batch)**: Utilizza tutti gli $N$ campioni per calcolare il gradiente a ogni passo.
- **Stochastic Gradient Descent (SGD)**: Utilizza un solo campione casuale per calcolare il gradiente.
- **Mini-Batch Gradient Descent**: Utilizza un sottoinsieme di $m$ campioni (il *mini-batch*), con $1 < m < N$, per calcolare il gradiente.
[05:46] Ad esempio, se si dispone di un milione di campioni, invece di usarne uno solo o tutti, si possono selezionare casualmente 10, 32 o 128 campioni e calcolare il gradiente medio su questo sottoinsieme.
[05:58] La dimensione del mini-batch, $m$, è un iperparametro del modello che deve essere scelto.
[06:10] Una volta fissata la dimensione $m$, il gradiente approssimato $\tilde{g}_t$ viene calcolato come la media dei gradienti relativi ai campioni nel mini-batch selezionato all'iterazione $t$:
$$
\tilde{g}_t = \frac{1}{m} \sum_{i \in \mathcal{I}_t} \nabla f_i(x_t)
$$
dove $\mathcal{I}_t$ è l'insieme degli indici degli $m$ campioni scelti casualmente.
[06:19] La regola di aggiornamento utilizza questo gradiente mediato:
$$
x_{t+1} = x_t - \gamma \tilde{g}_t
$$
### Vantaggi del Metodo Mini-Batch
[06:26] L'approccio mini-batch offre due vantaggi fondamentali.
[06:28] **1. Riduzione della Varianza**: Utilizzando una media su $m$ campioni invece di un singolo campione, si riduce la varianza dell'approssimazione del gradiente. La stima $\tilde{g}_t$ del gradiente vero è quindi più accurata rispetto a quella ottenuta con l'SGD puro, portando a una convergenza generalmente più rapida e stabile.
[06:44] **2. Calcolo Parallelo**: Un altro beneficio cruciale è la possibilità di parallelizzare i calcoli. Tutti i gradienti $\nabla f_i(x_t)$ per i campioni nel mini-batch devono essere valutati nello stesso punto $x_t$.
[06:53] Se si dispone di un processore con più core (o, più comunemente, di una GPU), è possibile distribuire il calcolo dei singoli gradienti sui diversi core, eseguendoli in parallelo e accelerando notevolmente la computazione.
[07:05] Questo è uno dei motivi principali per cui l'approccio mini-batch è universalmente adottato nel deep learning.
[07:10] La struttura di calcolo del mini-batch (eseguire la stessa operazione su dati diversi) è ideale per le architetture delle **GPU (Graphics Processing Units)**.
[07:20] Le GPU sono ottimizzate per eseguire la stessa operazione su grandi quantità di dati in parallelo (architettura SIMD/SIMT). Questo spiega le differenze drammatiche di prestazioni osservate quando si esegue un addestramento su GPU rispetto a una CPU.
### Svantaggi e Scelta della Dimensione del Mini-Batch
[07:33] L'approccio mini-batch presenta anche delle considerazioni da tenere a mente.
[07:36] Il "rumore" introdotto dal gradiente stocastico, che causa le oscillazioni attorno al minimo, può essere benefico in scenari con funzioni di costo non convesse e con molti minimi locali. I salti casuali possono aiutare l'algoritmo a "sfuggire" da un minimo locale sub-ottimale.
[07:51] Un salto sufficientemente grande potrebbe spostare il punto al di fuori della regione di attrazione di un minimo locale, permettendogli di esplorare altre aree dello spazio dei parametri e potenzialmente trovare un minimo migliore.
[08:04] In generale, si ritiene che questo rumore tenda a guidare l'algoritmo verso minimi che si trovano in regioni "piatte" (*flat minima*), i quali spesso garantiscono una migliore generalizzazione del modello.
[08:10] Se si utilizza una dimensione del mini-batch $m$ troppo grande, l'algoritmo si avvicina al comportamento deterministico del gradient descent. Questo riduce l'entità del rumore benefico.
[08:17] Sebbene il rumore sia presente anche nel metodo mini-batch, la sua ampiezza è ridotta a causa della media su più campioni.
[08:25] Scegliere un valore di $m$ troppo elevato non è sempre una buona strategia, poiché può portare a una convergenza verso minimi "acuti" (*sharp minima*) e a un maggiore rischio di overfitting.
[08:31] Trovare il valore ottimale per la dimensione del mini-batch $m$ è quindi un compito importante nell'ottimizzazione degli iperparametri.
### Strategie per la Gestione del Tasso di Apprendimento (Learning Rate Scheduling)
[08:37] Per migliorare ulteriormente la convergenza e gestire il comportamento dell'algoritmo, si utilizzano strategie per variare dinamicamente il tasso di apprendimento $\gamma$ durante l'addestramento.
[08:42] Una strategia comune è quella di far decrescere $\gamma$ in modo inversamente proporzionale al numero di iterazioni o di epoche.
[08:45] Un'altra possibilità è creare uno **schedulatore** (*scheduler*), ovvero una funzione che regola il valore di $\gamma$ secondo una logica predefinita. Le principali librerie di deep learning, come TensorFlow e PyTorch, offrono strumenti avanzati per definire questi schedulatori.
[08:55] Ad esempio, si può impostare un valore di $\gamma$ elevato nella fase iniziale dell'addestramento (la "far-out region") per accelerare la discesa, per poi ridurlo quando la funzione di costo smette di migliorare (raggiunge un plateau).
[09:08] Questa strategia, nota come *step decay* o *reduce on plateau*, è una delle più comuni e efficaci nelle implementazioni pratiche.
### Oltre il Gradiente Stocastico: Metodi di Ottimizzazione Avanzati
[09:17] La discesa del gradiente stocastico è stata per anni l'algoritmo fondamentale (*workhorse*) per l'ottimizzazione nel deep learning.
[09:26] Tuttavia, il suo utilizzo in scenari sempre più complessi ha evidenziato alcuni limiti, come la sensibilità alla scelta del learning rate e la difficoltà a navigare paesaggi di costo complessi. Di conseguenza, sono stati sviluppati numerosi metodi alternativi, che rimangono comunque nell'ambito dei metodi del primo ordine (cioè, basati solo sul calcolo del gradiente).
[09:42] Tra questi metodi avanzati, i più noti sono:
- **Momentum**: Introduce un termine di "inerzia" per smorzare le oscillazioni e accelerare la discesa.
- **Nesterov Accelerated Gradient (NAG)**: Una variante migliorata del Momentum.
- **AdaGrad**: Adatta il learning rate per ogni singolo parametro.
- **Adam**: Combina le idee di Momentum e di adattamento del learning rate, ed è oggi uno degli ottimizzatori più utilizzati.
[09:50] Queste varianti dell'SGD sono state progettate per accelerare la convergenza e migliorare la stabilità dell'addestramento, e saranno oggetto di analisi successive.
[10:01] La lezione odierna si conclude qui.