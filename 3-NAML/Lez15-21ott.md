# Lez15-21ott - Differenziazione Automatica

## 🎯 Obiettivi della Lezione

### Concetti Teorici
- Comprendere i **4 metodi** per calcolare derivate (manuale, numerico, simbolico, automatico)
- Distinguere **AD** da metodi numerici e simbolici
- Studiare i **Wengert lists** e decomposizione in operazioni elementari
- Analizzare **grafi computazionali**: nodi, archi, dipendenze
- Comprendere **forward mode** vs **reverse mode** (backpropagation)
- Studiare il costo computazionale: n×cost vs m×cost

### Algoritmi e Metodi
- **Forward mode (tangent)**: variabili v̇ᵢ = ∂vᵢ/∂xⱼ, buono per n≪m
- **Reverse mode (adjoint)**: variabili v̄ᵢ = ∂y/∂vᵢ, buono per n≫m (neural networks!)
- **Chain rule**: applicazione sistematica in AD
- **Matrix-free methods**: calcolare J·v e Jᵀ·w senza Jacobiano esplicito
- **Numeri duali**: ε² = 0, f(x+ε) = f(x) + f'(x)·ε

### Applicazioni Pratiche
- Gradient descent per neural network training
- Calcolo efficiente di gradienti con molti parametri
- Backpropagation come caso speciale di reverse mode AD

## 📚 Prerequisiti

### Matematica
- **Calcolo**: derivate, regola della catena, derivate parziali
- **Algebra Lineare**: matrici Jacobiane, prodotti matrice-vettore
- **Analisi Numerica**: floating-point errors, condizionamento

### Concetti
- **Gradient descent**: θ_(k+1) = θ_k - η∇L(θ_k)
- **Chain rule**: (f∘g)'(x) = f'(g(x))·g'(x)
- **Jacobian matrix**: matrice delle derivate parziali

## 📑 Indice Completo

### [Parte 1: Motivazione e Metodi](#parte-1-motivazione-e-metodi) (`00:00` - `28:28`)
1. [Introduzione alla Differenziazione](#introduzione-alla-differenziazione) - `00:00:00`
2. [Motivazione: Gradient Descent per Neural Networks](#motivazione-gradient-descent) - `00:02:15`
3. [Metodi per Calcolare Derivate](#metodi-per-calcolare-le-derivate) - `00:09:10`
4. [Manuale: Elegante ma Error-Prone](#manuale-elegante-ma-error-prone) - `00:11:24`
5. [Numerico (Finite Differences): Facile ma Instabile](#numerico-finite-differences) - `00:14:38`
6. [Simbolico (Maple/Mathematica): Expression Bloat](#simbolico-expression-bloat) - `00:19:52`
7. [Automatico: Best Choice per ML](#automatico-best-choice) - `00:24:06`

### [Parte 2: AD Principles](#parte-2-ad-principles) (`28:28` - `51:27`)
8. [Differenziazione Automatica (AD)](#differenziazione-automatica-ad) - `00:28:28`
9. [AD non è: Numerico, Simbolico](#ad-non-è) - `00:30:41`
10. [Wengert List: Decomposizione Elementare](#wengert-list) - `00:35:04`
11. [Operazioni Elementari: log, exp, +, -, ×, ÷](#operazioni-elementari) - `00:37:29`
12. [Grafo Computazionale: Nodi e Archi](#grafo-computazionale) - `00:42:13`
13. [Edge Derivatives](#edge-derivatives) - `00:45:38`

### [Parte 3: Forward Mode](#parte-3-forward-mode) (`51:27` - `01:04:15`)
14. [Forward Mode (Tangent)](#forward-mode) - `00:51:27`
15. [Variabili Tangenti: v̇ᵢ = ∂vᵢ/∂xⱼ](#variabili-tangenti) - `00:54:12`
16. [Inizializzazione: v̇=[1,0] per ∂/∂x₁](#inizializzazione-forward) - `00:57:36`
17. [Propagazione Forward](#propagazione-forward) - `01:00:49`
18. [Costo: n Passi per n Input](#costo-n-passi) - `01:03:22`

### [Parte 4: Reverse Mode (Backpropagation)](#parte-4-reverse-mode) (`01:04:15` - `01:16:36`)
19. [Reverse Mode (Adjoint)](#reverse-mode) - `01:04:15`
20. [Variabili Adjoints: v̄ᵢ = ∂y/∂vᵢ](#variabili-adjoints) - `01:06:28`
21. [Inizializzazione: ȳ=1](#inizializzazione-reverse) - `01:08:54`
22. [Backward Propagation: v̄ᵢ = Σⱼ v̄ⱼ·∂vⱼ/∂vᵢ](#backward-propagation) - `01:11:37`
23. [Costo: m Passi per m Output](#costo-m-passi) - `01:14:23`
24. [Backpropagation = Reverse Mode AD](#backpropagation--reverse-mode) - `01:16:36`

### [Parte 5: Matrix-Free Methods e Numeri Duali](#parte-5-matrix-free-methods) (`01:16:36` - `01:44:59`)
25. [Matrix-Free Methods](#matrix-free-methods) - `01:16:36`
26. [Calcolare J·v senza J](#calcolare-jv-senza-j) - `01:19:18`
27. [Calcolare Jᵀ·w con Reverse Mode](#calcolare-jᵀw) - `01:22:45`
28. [Numeri Duali: ε² = 0](#numeri-duali) - `01:23:40`
29. [f(x+ε) = f(x) + f'(x)·ε](#fxε--fx--fxε) - `01:29:36`
30. [Esempio: x²/cos(x) in π](#esempio-numeri-duali) - `01:35:11`
31. [Prodotto e Composizione](#prodotto-e-composizione) - `01:40:27`
32. [Derivate di Ordine Superiore](#derivate-ordine-superiore) - `01:44:59`

---

## Parte 1: Motivazione e Metodi

## Introduzione alla Differenziazione

## Introduzione alla Differenziazione

`00:00:00` 
Okay, oggi inizieremo con, in realtà copriremo questo argomento, che è in qualche modo fuori dal mainstream che abbiamo considerato finora, ma è molto importante. E tra un momento, vi dirò perché questo è di particolare importanza in quello che vedremo.

**Context: Perché la Differenziazione è Cruciale per il Machine Learning**

Nel machine learning e deep learning, il **calcolo efficiente delle derivate** è fondamentale per:

1. **Training dei modelli**: Gradient descent e sue varianti (SGD, Adam, RMSprop)
2. **Backpropagation**: Algoritmo chiave per neural networks
3. **Ottimizzazione**: Trovare parametri ottimali θ* che minimizzano loss L(θ)
4. **Sensitivity analysis**: Capire l'impatto di ogni parametro sull'output

**Sfida**: Reti neurali moderne hanno **milioni/miliardi di parametri** → servono metodi efficienti!

**Riferimento Slide**: AutoDiff.pdf, Slide 1 - Titolo "Automatic Differentiation" con schema neural network

`00:00:39` 
Se ricordate, quando abbiamo considerato, per esempio, i minimi quadrati. problema, quello che abbiamo fatto è scrivere una funzione di costo, poi minimizzare la funzione di costo, al fine di trovare il vettore dei pesi che minimizza la funzione di costo, e quindi possiamo usare quel vettore per creare un modello lineare, lineare in generale se usate la metrica kernel. In quel caso particolare.

### Caso Fortunato: Least Squares (Soluzione Chiusa)

**Problema di Least Squares**:
$$
\min_{w} L(w) = \|Xw - y\|^2 = \sum_{i=1}^{N} (w^T x_i - y_i)^2
$$

**Soluzione in forma chiusa** (caso sovradeterminato $N > d$):

$$
\boxed{w^* = (X^T X)^{-1} X^T y = X^{\dagger} y}
$$

Dove $X^{\dagger}$ è la **pseudo-inversa** di Moore-Penrose.

**Via SVD** (più stabile numericamente):

Se $X = U \Sigma V^T$ (SVD), allora:
$$
w^* = V \Sigma^{-1} U^T y = \sum_{i=1}^{r} \frac{u_i^T y}{\sigma_i} v_i
$$

**Gradiente** (per verificare):
$$
\nabla_w L(w) = 2X^T(Xw - y) = 0 \quad \Rightarrow \quad X^T X w = X^T y
$$

Queste sono le **normal equations**.

`00:01:13` 
eravamo molto fortunati nel senso che la soluzione del problema di minimizzazione può essere scritta usando o non usando la decomposizione ai valori singolari in forma chiusa, quindi abbiamo una formula esplicita per W, e quindi tutto è abbastanza semplice. Nelle applicazioni pratiche, in particolare quando introdurremo le reti neurali,

**Fortuna**: Per LS, non serve algoritmo iterativo → formula diretta!

**Ma per Neural Networks**: **NESSUNA soluzione chiusa!**

`00:01:44` 
funziona. la possibilità di venire con una soluzione analitica o un'espressione in forma chiusa per i, vettori d'onda, peso e varianze in quel caso, che minimizzano la funzione di costo è impossibile. Quindi dovremo ricorrere a un algoritmo di minimizzazione numerica e la maggior parte di essi, potrei dire tutti, coinvolgeranno in un modo o nell'altro.

## Motivazione: Gradient Descent per Neural Networks

**Problema Generale**:

Data una **loss function** $L(\theta)$ dove $\theta = [\theta_1, \ldots, \theta_p]^T \in \mathbb{R}^p$ sono i **parametri del modello**:

$$
\min_{\theta \in \mathbb{R}^p} L(\theta)
$$

**Esempi**:
- **Regression**: $L(\theta) = \frac{1}{N}\sum_{i=1}^{N} (f_{\theta}(x_i) - y_i)^2$ (MSE)
- **Binary Classification**: $L(\theta) = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\sigma(f_{\theta}(x_i))) + (1-y_i)\log(1-\sigma(f_{\theta}(x_i)))]$ (binary cross-entropy)
- **Multi-class**: $L(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(p_k(x_i; \theta))$ (categorical cross-entropy)

**Nessuna formula chiusa** → Serve algoritmo iterativo!

`00:02:19` 
qualche variazione del metodo della discesa del gradiente. Quindi in altri modi, la formula più generale che considereremo è questa. Quindi useremo una procedura iterativa dove il vettore sconosciuto, qui è chiamato theta, che è comune nel contesto delle reti neurali o W, qualunque cosa, è calcolato come risultato di un processo iterativo.

### Gradient Descent Algorithm

**Formula Base**:

$$
\boxed{\theta^{(k+1)} = \theta^{(k)} - \eta \nabla L(\theta^{(k)})}
$$

Dove:
- $\theta^{(k)}$: parametri all'iterazione $k$
- $\eta > 0$: **learning rate** (step size)
- $\nabla L(\theta)$: **gradiente** della loss rispetto a $\theta$

$$
\nabla L(\theta) = \begin{bmatrix}
\frac{\partial L}{\partial \theta_1} \\
\frac{\partial L}{\partial \theta_2} \\
\vdots \\
\frac{\partial L}{\partial \theta_p}
\end{bmatrix} \in \mathbb{R}^p
$$

`00:03:23` 
Lo stiamo muovendo verso il minimo con una lunghezza di passo eta, che in quel contesto è chiamato tasso di apprendimento. In pratica, dal punto di vista numerico, è la lunghezza del passo che stiamo facendo lungo la direzione del gradiente. OK, quindi tornando alla parabola che abbiamo visto prima, se siamo qui, questa è la direzione del gradiente e ci stiamo muovendo qui, per esempio.

**Interpretazione Geometrica**:

```
      L(θ)
        |
    ____|____
   /    |    \
  /     |     \
 /      ↓      \    ← Direzione: -∇L (opposta al gradiente!)
/    θ^(k+1)    \
────────────────── θ
     θ^(k)
```

Il **gradiente** $\nabla L(\theta^{(k)})$ punta nella direzione di **massima crescita** di $L$.

Ci muoviamo nella **direzione opposta** $-\nabla L$ per **minimizzare** $L$.

**Step size** $\eta$: Quanto lontano ci muoviamo in quella direzione.

**Riferimento Slide**: AutoDiff.pdf, Slide 3 - Diagramma gradient descent su superficie loss 3D con traiettoria iterativa

`00:03:56` 
OK, questo è tutto. Poi una volta che abbiamo questa valutazione, possiamo iterare. Il, qual è il punto? Il punto è che questa è l'espressione più semplice. Poi ci sono molte variazioni partendo da quella. Vedremo alcune di queste variazioni che coinvolgono termini aggiuntivi al fine di rendere la convergenza più robusta o di accelerare la convergenza.

**Varianti di Gradient Descent**:

**1. Gradient Descent with Momentum**:
$$
\begin{align}
v^{(k+1)} &= \beta v^{(k)} + \eta \nabla L(\theta^{(k)}) \\
\theta^{(k+1)} &= \theta^{(k)} - v^{(k+1)}
\end{align}
$$

Parametro: $\beta \in [0,1)$ (tipicamente 0.9)

**2. Nesterov Accelerated Gradient (NAG)**:
$$
\begin{align}
v^{(k+1)} &= \beta v^{(k)} + \eta \nabla L(\theta^{(k)} - \beta v^{(k)}) \\
\theta^{(k+1)} &= \theta^{(k)} - v^{(k+1)}
\end{align}
$$

**3. AdaGrad** (adaptive learning rate):
$$
\theta^{(k+1)} = \theta^{(k)} - \frac{\eta}{\sqrt{G^{(k)} + \epsilon}} \odot \nabla L(\theta^{(k)})
$$

Dove $G^{(k)} = \sum_{i=0}^{k} [\nabla L(\theta^{(i)})]^2$ (element-wise).

**4. RMSprop**:
$$
\begin{align}
G^{(k+1)} &= \gamma G^{(k)} + (1-\gamma) [\nabla L(\theta^{(k)})]^2 \\
\theta^{(k+1)} &= \theta^{(k)} - \frac{\eta}{\sqrt{G^{(k+1)} + \epsilon}} \odot \nabla L(\theta^{(k)})
\end{align}
$$

**5. Adam** (Adaptive Moment Estimation) ← **most popular!**:
$$
\begin{align}
m^{(k+1)} &= \beta_1 m^{(k)} + (1-\beta_1) \nabla L(\theta^{(k)}) \\
v^{(k+1)} &= \beta_2 v^{(k)} + (1-\beta_2) [\nabla L(\theta^{(k)})]^2 \\
\hat{m} &= \frac{m^{(k+1)}}{1 - \beta_1^{k+1}}, \quad \hat{v} = \frac{v^{(k+1)}}{1 - \beta_2^{k+1}} \\
\theta^{(k+1)} &= \theta^{(k)} - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \odot \hat{m}
\end{align}
$$

Parametri tipici: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

`00:04:32` 
Quindi, ci sono molte ricette. Ma da questa formula, è evidente che dobbiamo calcolare il gradiente. Okay, quindi data la funzione di perdita L, qualunque formula abbiate per la funzione di costo, dovete calcolare il gradiente o le derivate di questa funzione di costo. In realtà, vedremo che.

**Punto Chiave**: Tutte queste varianti richiedono $\nabla L(\theta)$!

**Challenge**: Come calcolare $\nabla L(\theta)$ efficientemente per $p \sim 10^6 \text{-} 10^9$ parametri?

`00:05:03` 
Immagino che sappiate tutti che il gradiente è un metodo del primo ordine, quindi la convergenza del metodo del gradiente, discesa del gradiente, è uno. Se volete avere una convergenza più veloce, potete fare affidamento sul metodo di Newton. Immagino che abbiate visto il metodo di Newton per trovare la radice di una funzione. In quel caso, per la funzione, dovete calcolare la derivata seconda della funzione.

`00:05:40` 
Se siete nel caso multidimensionale, dovrete calcolare l'azione della funzione. Qui dobbiamo calcolare il gradiente. Se siamo in più dimensioni, dovremo calcolare lo Jacobiano della funzione incrociata. Ma in ogni caso, il messaggio è che non importa quale metodo state usando per minimizzare, per trovare il vettore dei parametri che minimizza la funzione incrociata in tutti i casi che state considerando,

`00:06:15` 
vi sarà richiesto di calcolare alcune derivate, o derivate prime o secondarie. Quindi il punto e questo è il primo punto. Il punto è, come possiamo calcolare queste derivate? È chiaro che se avete in mente, è anche una semplice architettura di una rete neurale sapete che l'idea è che alimentate la rete neurale per esempio se avete se state considerando il caso di.

`00:06:51` 
classificazione di immagini quindi avete un gruppo di immagini che sono o gatti o cani, diciamo che avete migliaia di immagini userete 800 di esse per allenare la rete neurale quindi le state alimentando supponiamo per semplicità che queste immagini siano in scala di grigi quindi state alimentando l'immagine alla rete e come output avrete o uno.

`00:07:24` 
o zero uno può essere gatto zero può essere cane e quello che volete fare in quel caso è minimizzare la, una funzione di costo che calcola la differenza tra l'output della rete neurale e l'etichetta vera dell'immagine di input. OK, poi testerete il vostro modello addestrato usando le 200 immagini che avete scartato all'inizio.

`00:07:59` 
Quindi in quel caso, è chiaro che la funzione di costo è qualcosa che coinvolge la differenza tra l'etichetta vera e le etichette previste. E dovete calcolare la derivata rispetto a cosa? Rispetto a tutti i parametri che sono presenti nella vostra rete neurale. In realtà, in linea di principio, dovete calcolare la derivata dell'output rispetto all'input.

`00:08:35` 
Ma dato che, alla fine della giornata, quello che volete fare è venire con i valori di tutti i parametri della rete, dovete essere in grado di capire qual è l'effetto della variazione di ogni parametro nella rete sull'output della rete stessa. Una sorta di sensibilità. Quindi questo è il punto chiave. Ed è chiaro che, in pratica...

`00:09:10` 
Quindi, il calcolo delle derivate in quella situazione è molto importante, e ora dobbiamo capire come possiamo farlo.

## Metodi per Calcolare le Derivate

`00:09:10` 
Quali sono le possibilità per calcolare le derivate? In realtà, abbiamo quattro modi possibili per farlo. Il primo è calcolare manualmente. Avete la vostra funzione di costo, calcolate manualmente tutte le derivate.

`00:09:47` 
È chiaro che nelle applicazioni pratiche questo è irrealizzabile a causa del numero di derivate che dovete calcolare, e inoltre, in generale, è, beh, discuteremo più tardi i vantaggi e gli svantaggi. Il secondo è usando la differenziazione numerica. Avete mai sentito parlare di differenze finite per calcolare derivate? Chi non ha mai sentito parlare di questo? Okay, quindi le differenze finite sono un altro modo a cui potete pensare per calcolare derivate.

`00:10:28` 
Poi potete usare Maple, MATLAB, o Mathematica, o il toolbox simbolico, scusate, il pacchetto simbolico di Python per calcolare in modo simbolico l'espressione delle derivate. Okay. E infine, possiamo adottare la numerica, la differenziazione automatica. Posso anticipare che queste quattro possibilità è il modo che useremo e il modo che è il metodo che è usato in pratica.

`00:11:15` 
Okay. Quindi vediamo per ciascuno di questi metodi. Quali sono i vantaggi e gli svantaggi. Quindi per il manuale. Può essere, se siete in grado di semplificare la formula, potete venire con un'espressione molto elegante, ed è di particolare importanza, per esempio, se volete dimostrare qualcosa, perché calcolando manualmente le derivate, potete avere un'espressione esplicita per il risultato,

`00:11:55` 
e quindi potete dimostrare qualcosa sulle sensibilità, la convergenza, e così via. D'altra parte, specialmente se avete espressioni complesse, complicate, questo approccio è soggetto a errori, e inoltre, richiede molto tempo. Poi abbiamo il metodo numerico.

`00:12:26` 
principalmente trovando differenze. È facile da implementare, molto, molto facile. Funziona essenzialmente su qualsiasi funzione. Può essere usato come una scatola nera. Avete la vostra formula delle differenze finite. Diciamo che voglio usare, per esempio, questa formula per calcolare la derivata.

`00:12:57` 
La derivata prima di una funzione, di una funzione generica, xf in un punto, diciamo, chiamato x0. Quello che dobbiamo decidere è cos'è h, e poi è solo una questione di due valutazioni della funzione e una divisione. Quali sono i problemi con questo? Questo approccio apparentemente è molto promettente, è semplice, ma ovviamente è un'approssimazione e inoltre è rischioso perché in questa formula quello che stiamo facendo è che stiamo commettendo quello che è di solito chiamato un crash in virgola mobile.

`00:13:56` 
Nel senso che qui ovviamente se vogliamo calcolare il valore della derivata di f vicino a x0, h dovrebbe essere abbastanza piccolo. Il punto è che qui, di solito x0 è una quantità finita, diciamo 1, e h può essere una quantità dell'ordine di, per esempio, 10 alla meno 4.

`00:14:30` 
Quindi il primo crimine qui è che dal punto di vista della virgola mobile, state sommando due quantità che sono di ordine di grandezza abbastanza diverso, 1 e 10 alla meno 4. E questo può essere complicato in alcune situazioni, e l'altro problema è questa operazione, perché qui state sottraendo due quantità che sono molto vicine una all'altra, e dal punto di vista della virgola mobile, questo potrebbe causare cancellazioni.

`00:15:07` 
Okay, quindi, e in... In pratica, quando considerate qualsiasi formula numerica, quello che potete vedere è che, essenzialmente, quando state usando quel tipo di formula, quindi quella formula, state commettendo due tipi di errori.

`00:16:03` 
Uno è il cosiddetto errore di troncamento, e l'altro è il cosiddetto errore di arrotondamento. Okay, qui, per esempio, ho tracciato in marrone, in arancione, l'errore di arrotondamento.

`00:16:36` 
L'errore di arrotondamento è qualcosa che si comporta come 1 su h, come potete vedere. E non dipende, essenzialmente, dal metodo che state usando. Mentre l'errore di troncamento dipende dal metodo che state usando. Potete usare, per esempio, questo è il primo metodo di Jordan, quindi l'errore che state commettendo approssimando f primo in x0 con questa quantità,

`00:17:14` 
è qualcosa che va a zero come h, okay? Mentre se usate questo tipo di formula di x zero, più h meno f di x zero meno h diviso per due h, quindi questa è una cosiddetta formula d-centrata, e questa è una formula centrata. Questa è di secondo ordine.

`00:17:51` 
Okay, quindi questo è quello che viene chiamato l'errore di troncamento, e l'errore di troncamento dipende dal metodo che state usando. Quindi è di solito la potenza del parametro h, okay? Qui nel grafico che potete vedere, è dato per n uguale a 2.

`00:18:21` 
Okay, se volete tracciare la stessa immagine per n uguale a 1, avete un comportamento diverso dell'errore. Quello che è importante è che questi due errori si stanno sommando, e come potete vedere, c'è effettivamente un valore ottimale di h. Che minimizza l'errore, okay?

`00:18:54` 
Quindi, in altri termini, non è, in generale, non è vero che se andate a valori sempre più piccoli di h, state migliorando l'approssimazione delle derivate che state calcolando, e questo è dovuto al fatto che avete questi due errori e il comportamento di questi due errori è opposto, sono opposti, quindi uno sta diminuendo, l'altro sta aumentando.

`00:19:28` 
con l'h, okay? In pratica, per esempio, qui ho solo riportato, questo dovrebbe essere... il grande O, quindi qui avete le due formule, quindi formula D-centrata e centrata. E in questo esempio, ho solo considerato le due formule applicate alla funzione seno.

`00:20:04` 
Quindi come abbiamo detto prima nelle slide, una buona proprietà delle differenze finite è che possono essere usate come una scatola nera. Quindi una volta che ho definito questa formula, è tutto. Devo solo passare il punto dove voglio calcolare la derivata, l'incremento che voglio usare, e la funzione che sto considerando.

`00:20:37` 
E poi posso usare questo. Questa è una di quelle due formule, o posso usare una qualsiasi di esse. E qui quello che ho fatto è tracciare l'errore, qui avete scala logaritmica su entrambi gli assi per la funzione seno.

`00:21:13` 
Quello che potete vedere è che chiaramente in questo grafico, la pendenza in queste due parti vi dice l'ordine del metodo, l'ordine dell'errore di troncamento. Qui la pendenza è più alta, quindi significa che è di ordine 2, questo è di ordine 1. Cosa succede qui, in questa porzione? Qui, l'approssimazione in virgola mobile, frecce di cancellazione, e così via, sono predominanti.

`00:21:54` 
E quindi, come potete vedere, una volta che siete in questa porzione, o in questa porzione dei valori di h, l'approssimazione che potete ottenere usando la distanza finita per una derivata è molto, molto scarsa. Potete vedere che avete un comportamento oscillante, e non potete fare affidamento su quel tipo di calcolo.

`00:22:38` 
Okay, poi abbiamo il terzo metodo, che è il simbolico, Maple, Mathcad, Mathematica, e così via. Essenzialmente, ha gli stessi vantaggi del calcolo manuale. È anche meno soggetto a errori perché è automatico. Ma ha due svantaggi.

`00:23:13` 
È computazionalmente abbastanza intensivo nel senso che se avete un complesso... espressione da differenziare il calcolo può essere abbastanza lungo e poi potete creare quello che viene chiamato il gonfiamento dell'espressione quindi potete finire con un'espressione per la derivata che è molto molto lunga e poi dovete passare attraverso la semplificazione raccogliere termini e per casi semplici la maggior parte.

`00:23:52` 
degli strumenti sono in grado di eseguire qualche tipo di semplificazione automaticamente ma se avete un'espressione molto complessa dovete entrare nella semplificazione dell'espressione complessa quindi l'ultimo punto è la differenziazione automatica, ho, voglio solo enfatizzare una cosa. Nel manuale e nel simbolico, quello che avete come risultato è un'espressione, okay?

`00:24:36` 
Nel caso numerico, avete un valore, okay? Qui, una volta che abbiamo scelto hx0 e h e f, ovviamente, questo è un numero, okay? Quindi stiamo valutando la derivata in un certo punto. E se tornate all'espressione che avevamo qui,

`00:25:07` 
Qui abbiamo il gradiente della funzione L rispetto ai parametri theta, e dobbiamo valutare questo gradiente in theta, e theta la maggior parte delle volte è o il vecchio valore o qualche tipo di estrapolazione. Ma in ogni caso, è qualcosa che vi darà un numero o un vettore nel caso di una funzione multidimensionale.

`00:25:43` 
Quindi in questa espressione, quello a cui siamo principalmente interessati, non è avere un'espressione esplicita per il gradiente o la derivata. Vogliamo avere un valore. Questo è quello che conta. Okay? Avere un valore. Quindi, in questo senso, la differenziazione automatica è il cavallo da lavoro di qualsiasi framework di machine learning e deep learning.

`00:26:19` 
E, in realtà, in TensorFlow, PyTorch, qualsiasi she-kit-learn, qualsiasi libreria che implementa algoritmi di deep learning o machine learning, avete, all'interno, qualche tipo di strumento per la differenziazione automatica. Durante il...

## Differenziazione Automatica (AD)

`00:26:51` 
Nel laboratorio, vedrete uno degli... strumenti, che si chiama JAXA. È uno strumento per la differenziazione automatica che è usato all'interno di, TensorFlow. Quali sono gli svantaggi di questo tipo di approccio? Consuma molta memoria, quindi ha un.

`00:27:23` 
impronta di memoria abbastanza alta, e l'implementazione non è per niente semplice. Tuttavia, durante il laboratorio, vedrete come implementare una libreria AD molto semplice, al fine di capire quali sono i principali problemi che potete incontrare anche in.

`00:27:56` 
questo argomento. Okay, quindi qui è solo di nuovo sulla differenziazione numerica. Nei commenti precedenti, abbiamo enfatizzato il fatto che potete venire con valori che non sono significativi a causa di errori di cancellazione, quindi problemi di virgola mobile.

`00:28:28` 
Qui, quello che è enfatizzato è il costo computazionale. Quindi, diciamo che siamo nella buona regione delle differenze finite, quindi non abbiamo problemi relativi alla virgola mobile. E se avete una funzione da Rn a R, quindi una funzione con n input e solo un output, allora per calcolare la derivata parziale,

`00:29:01` 
per calcolare il gradiente, dobbiamo eseguire cosa? Questo tipo di operazione per ognuna delle n, variabili che definiscono l'input, okay? Ed è chiaro che se... Torniamo al caso precedente dell'immagine. Supponiamo che la vostra immagine sia mille per mille.

`00:29:33` 
Quindi significa che, in quel caso, n sarebbe un milione. Quindi in pratica, significa che dovete calcolare un milione di derivate dell'output rispetto a ciascuno dei parametri di input, giusto? E se avete una funzione che è da rm a rm, Quindi, avete 1 milione di input e 100 output, diciamo che è un problema di classificazione,

`00:30:11` 
e avete 100 possibili famiglie di oggetti che volete classificare. Bene, in quel caso, è chiaro che dovete calcolare lo Jacobiano, il che significa che dovreste calcolare n per m valutazioni di funzioni, che è molto, molto costoso dal punto di vista computazionale.

`00:30:48` 
Okay, questo è quello che abbiamo visto prima. Quindi... Quello che stiamo sostenendo è l'uso della differenziazione automatica. Prima di definire cos'è ADE, diciamo cosa ADE non è. Perché a volte il termine differenziazione automatica è usato in modo non molto appropriato.

`00:31:25` 
Quindi prima di tutto, non è una differenziazione numerica. Quindi in altri termini, non state usando alcun tipo di formula approssimata, ma state calcolando il valore. Fate attenzione. Il valore del valore o i valori, a seconda di quale tipo di funzione state considerando.

`00:31:56` 
Esattamente, okay. Quindi non c'è H coinvolto nel calcolo. Non è un calcolo simbolico, differenziazione simbolica. In che senso? Nel senso che ADE non fornisce un'espressione esplicita per la derivata. Fornisce un valore, okay? Quindi questo è totalmente diverso da cos'è la differenziazione simbolica. Quando usate qualsiasi strumento simbolico, inserite una funzione. Per esempio, in MATLAB, se definite la vostra funzione f, diciamo che avete f, quindi definite la vostra x come una variabile simbolica, forse reale.

`00:32:53` 
Poi usate, definite f come il seno di x, e poi se chiedete per gif di f rispetto a x, quello che otterrete è il coseno, okay? Quindi questo è il risultato dell'uso di uno strumento simbolico. AD non fa questo. Sta operando in un modo totalmente diverso.

`00:33:28` 
E l'idea di, l'idea principale di AD è dividere qualsiasi funzione, per quanto complessa vogliate, in blocchi elementari. E i blocchi elementari di solito sono alcune, sottrazioni, moltiplicazione, divisione, funzione trigonometrica ed esponenziali.

`00:33:58` 
e logaritmi essenzialmente. Quindi una volta che avete queste funzioni semplici siete quasi finiti nel senso che la differenziazione automatica si basa fortemente sul fatto che siete in grado di dividere, di rappresentare qualsiasi funzione come la, composizione di funzioni elementari. Okay? Quindi quello che vedremo ora è.

`00:34:32` 
come fare. è come, dal punto di vista operativo, creare questa divisione in semplici.

## Wengert List

`00:35:04` 
funzione. Il primo oggetto, quell'oggetto o concetto che dobbiamo introdurre è la cosiddetta lista di Wengert. Cos'è la lista di Wengert? È essenzialmente un modo di descrivere. in passi qual è la sequenza di operazioni che dovete eseguire per calcolare.

`00:35:36` 
il valore di qualsiasi funzione in termini di funzioni elementari. Quindi, quando... Vedremo tra un momento un esempio. L'idea è che, data una funzione, supponiamo che abbiate una funzione di due variabili, create questa lista di Wengert, prima di tutto, definendo le prime variabili aggiuntive che sono usate, che sono chiamate v.

`00:36:16` 
Le prime v sono essenzialmente le variabili di input, quindi se avete, per esempio, due input, avrete v di meno 1, che è uguale a x1, e v0, che è uguale a x2, dove x1 e x2 sono i due valori. che definiscono il punto che definisce il punto dove volete valutare la vostra funzione siate.

`00:36:50` 
attenti per il momento stiamo vogliamo valutare la funzione non siamo al punto di calcolare derivate ora vogliamo solo capire come usare la lista banger per calcolare il valore della funzione come composizione di funzioni elementari poi avete una porzione della lista banger che è composta dalle cosiddette variabili intermedie che sono essenzialmente passi intermedi.

`00:37:26` 
che sono richiesti per calcolare gli output che sono le ultime variabili che sono. calcolate nella lista di Wengert.

`00:38:33` 
Okay, quindi questo è un modo molto formale di definire la lista di Wengert. Con un esempio, tutto sarà molto più chiaro. Quindi, consideriamo questa funzione f di x1, x2, che è uguale al logaritmo naturale di x1 più x1, x2. Quindi, la lista di Wengert è data dalle prime due variabili aggiuntive, b meno 1 e b0,

`00:39:08` 
che sono uguali a x1 e x2, gli input. Quindi, supponiamo che più tardi vedremo forse che x1 è uguale a 1, x2 è uguale a 2. Okay, due valori. Poi, abbiamo la porzione della lista di Wengert chiamata variabili intermedie, che sono b1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x33, x34, x35, x36, x37, x38, x39, x40, x40, x41, x42, x43, x44, x44, x45, x46, x47, x48, x49, x49, x50, x51, x52, x52, x53, x54, x55, x55, x56, x57, x58, x58, x59, x60, La prima variabile intermedia calcola il logaritmo di x1 che, in termini di variabili aggiuntive, è il logaritmo di v meno 1, okay?

`00:39:46` 
Poi abbiamo la seconda parte, che è v2, è il prodotto di x1, x2, quindi il prodotto di v meno 1 e v0. E infine, dobbiamo sommare questi due contributi. Come potete vedere, in questa porzione, abbiamo diviso il calcolo di quella funzione, che è complessa, in molti passi, ogni passo coinvolgendo solo una funzione elementare.

`00:40:22` 
Logaritmo. moltiplicazione addizione okay e infine l'output y è uguale a v3 quindi questa è uh il valore intermedio questa è la porzione della variabile intermedia questo è l'output in questo caso è solo uno questi sono gli input okay perché uh per la funzione elementare sapete.

`00:40:55` 
facilmente come calcolare le derivate questo è il punto chiave perché per una funzione semplice funzione elementare sapete come calcolare le derivate quindi uh per esempio la derivata del logaritmo è uno su okay quindi una volta che avete una sorta di dizionario di funzione elementare e corrispondenti, derivate insieme alla regola di differenziazione per la somma il prodotto e così via.

`00:41:33` 
siete finiti okay in pratica è esattamente quello che state facendo quando state differenziando qualsiasi funzione okay è un modo formale di rappresentare quello che tutti stanno facendo quando state calcolando una derivata okay un altro modo di rappresentare la la funzione è attraverso un calcolo.

## Grafo Computazionale

`00:42:13` 
quello che viene chiamato il grafo computazionale quindi il grafo computazionale è in qualche in qualche senso una rappresentazione grafica della lista banger, Okay, nel grafo computazionale, di nuovo, abbiamo essenzialmente tre parti. La prima parte è data dagli input. C'è una parte intermedia dove avete il calcolo dei valori intermedi e poi l'output. Qui, per esempio, è chiaro che x1 entra nel logaritmo ed entra in questo prodotto con x2. Questi output, il prodotto e il logaritmo, sono sommati per dare l'output.

`00:43:00` 
Okay, quindi questo è il grafo computazionale della funzione che abbiamo. Abbiamo definito prima. Essenzialmente, nei nodi del grafo, Abbiamo le variabili, quindi qui avremo come variabili il prodotto di questi due, in quel nodo avremo il logaritmo di x1, qui la somma di queste due variabili intermedie, e i bordi del grafo rappresentano le dipendenze di ogni variabile dalle altre variabili del grafo.

`00:43:47` 
Okay, una volta che avete questo grafo, il prossimo passo è introdurre le cosiddette derivate sui bordi. Quali sono le derivate sui bordi? Essenzialmente, qui state solo sostituiti nei nodi b1, b2, e b3 secondo la definizione che abbiamo dato nella lista di Wengert, okay?

`00:44:24` 
b1 è il logaritmo, b2 è il prodotto, b3 è la somma del precedente. Quali sono le derivate sui bordi? Essenzialmente, dovete fornire l'espressione della derivata. Per esempio, su questo bordo, quello che dobbiamo fornire è la derivata di b2 rispetto a b-1, okay?

`00:45:03` 
V2 era il prodotto di V meno 1 e V0. Quindi la derivata parziale di V2 rispetto a V meno 1 è V0. Qui è la derivata parziale di V2 rispetto a V0, quindi è V meno 1. Qui, la derivata parziale di V3 rispetto a V2. Ricordate che V3 è la somma di V1 e V2, quindi la derivata parziale è solo 1 e lo stesso su questo bordo.

`00:45:49` 
Perché queste derivate sul bordo sono importanti? Perché ora... Se volete calcolare le derivate dell'output rispetto all'input, diciamo che voglio calcolare dy su dv0. Quindi voglio calcolare la derivata di questa funzione, che è in realtà v3,

`00:46:25` 
rispetto a v0. Quello che devo fare è solo il prodotto di questi due termini, di queste due derivate sui bordi. È dv3 su dv2, dv2 su dv0. È la derivata, è la regola della catena. Nient'altro che la regola della catena. Sfruttando la riduzione della funzione in funzioni elementari semplici,

`00:46:55` 
e poi calcolando... Queste derivate sui bordi, che non sono altro che quello che vi stavo dicendo prima. Avete un dizionario di derivate di funzioni semplici che vi permette di scrivere queste derivate sui bordi. Una volta che avete quelle derivate sui bordi, potete calcolare quello che volete.

`00:47:25` 
È ovvio che se ora voglio calcolare la derivata di v3 rispetto a v meno 1, allora in quel caso dovete stare attenti, perché per andare da qui a qui, avete due percorsi, questo e questo. Quindi dovete sommarli, come nella regola della catena.

`00:47:56` 
È esattamente quello che avreste fatto usando la regola della catena, nient'altro. Quindi, ora se torniamo alla lista di Wengert che abbiamo creato, e ora vogliamo usarla. In pratica, la lista di Wengert non è mai creata in questo modo.

`00:48:31` 
È sempre creata partendo dai valori qui. Quindi, la vera lista di Wengert è questa. Quindi, diciamo che abbiamo la divisione in funzioni semplici, e assegniamo a x1 il valore 2, e a x2 il valore 5. Poi possiamo calcolare le variabili intermedie e anche a y, che è uguale a b3.

`00:49:02` 
Quindi è logaritmo di 2, questo è 10, quindi l'output è il logaritmo di 2 più 10. Ora, come posso usare questa idea, questa funzione, per calcolare la derivata? Quindi supponiamo che voglio calcolare la derivata di f rispetto a x1. Quindi, in altri termini, se torniamo a questa immagine, calcoleremo la derivata di questo nodo rispetto a questo, per un dato insieme di valori 2 e 5.

`00:49:45` 
Qual è l'idea? L'idea è che... Introducete un nuovo insieme di variabili che sono di solito denotate con i punti e sono chiamate variabili tangenti. Ogni variabile non è altro che la derivata di quella variabile.

`00:50:17` 
In questo caso, rispetto a x1. Quindi la variabile rispetto a cui voglio considerare nella derivata. È chiaro che per la prima, la derivata di b-1 rispetto a x1 è 1, perché b-1 è 1 stesso. Quindi qui 1 e qui 0.

`00:50:48` 
E questo è un punto importante. Essenzialmente, se avete... una funzione che dipende da, quindi avete una funzione f che va da rn a r, okay? Quindi avete n, variabili di input. Quindi qui potete calcolare df su dx1 fino a df su dxn, okay? Nel,

`00:51:27` 
quello che viene chiamato la modalità tangente o della differenziazione automatica, l'idea è che, se inizializzate la procedura qui con un vettore v meno fino a v zero punto, uguale a, supponiamo che vogliamo derivare rispetto a.

`00:51:57` 
una variabile qui, una variabile intermedia, dovrò mettere tutti gli zeri qui a parte l'uno. Che voglio considerare come differenziazione. Perché? Perché ricordate che questi vi punti sono la derivata di vi rispetto a xj, dove j è la variabile che volete considerare nella differenziazione.

`00:52:30` 
Quindi in quel caso, supponiamo che j sia una delle m variabili di input. In questo caso, dato che voglio calcolare la derivata rispetto a x1, l'inizializzazione sarà v punto meno 1 uguale a 1, v punto 0 uguale a 0. È chiaro che se volete calcolare la derivata rispetto a x2, avreste n.

`00:53:01` 
Quello è 0, 1. Poi, il prossimo passo coinvolge queste variabili intermedie, okay? In particolare, come vi ho detto prima, quello che devo fare è applicare le derivate di queste funzioni elementari. Quindi, la derivata del logaritmo è 1 su, per, è una funzione composta, per la derivata di v1 rispetto a x1, okay?

`00:53:38` 
Quindi, 1 su v meno 1 per v punto meno 1, okay? È solo la regola della catena applicata a questa funzione. Che dire di qui? Qui abbiamo, è un prodotto. Quindi, abbiamo la derivata del primo per il secondo più il primo per la derivata del secondo. Questo è 0, perché v punto 0 è 0, quindi questo termine non dà alcun contributo.

`00:54:11` 
Rimaniamo solo con questa porzione, che è v meno 1 punto, che è 1, per v0. v0 dal valore della funzione è 5. Quindi qui ho 5. Infine, v3 punto è questo. Quindi in questo caso, applichiamo la differenziazione della somma. Quindi abbiamo solo v1 punto più v2 punto.

`00:54:44` 
v1 punto è 1 mezzo. v2 punto è 5, quindi v3 punto è 5.5, che è cosa? Il valore di df su dx1 nel punto 2, 5, okay? Quindi l'idea è molto semplice. Dividete la funzione in funzioni elementari,

`00:55:14` 
usate la regola della catena e le derivate di quelle funzioni semplici per venire con il valore della derivata, okay? Questa è l'idea. È chiaro che, per esempio, in questo caso, se volete calcolare anche df su dx2, dovete fare un altro passaggio dell'algoritmo nel senso che dovete partire da 0, 1, e dovete calcolare questi valori di nuovo, okay?

`00:55:53` 
Quindi, il messaggio è che se avete un output, ma avete un milione di input, allora per calcolare il gradiente, dovete ripetere questa procedura un milione di volte, okay?

`00:56:29` 
E, ovviamente, questo potrebbe essere molto impegnativo dal punto di vista computazionale, e quindi, qui entra in gioco quella che viene chiamata la modalità inversa di AE, che è, e questa è la ragione per cui voglio passare un po' di tempo su questo argomento, è alla base di quello che viene chiamato algoritmo di backpropagation, di cui forse avete già sentito parlare, nel contesto delle reti neurali.

`00:57:08` 
Backpropagation, anche se nel contesto della rete neurale è stato scoperto negli anni 60 o 80, proposto, in realtà non è altro che la modalità inversa di AE, Applicata a un problema specifico. Okay, quindi è una sorta di riscoperta di qualcosa che era ben noto, ma in altri contesti. Quindi qual è l'idea della modalità inversa? Se guardate attentamente questa immagine, questa tabella, potete vedere che qual è il problema qui?

`00:57:54` 
Il problema è che se ho le variabili finali e voglio completare la derivata rispetto a una qualsiasi di queste variabili di input, devo cambiare l'inizializzazione della procedura ogni volta perché ho variabili finali. Quindi l'idea è invertire. Invece di partire dall'input, voglio partire dall'output.

`00:58:24` 
Quindi, l'idea è che sto impostando queste nuove variabili che sono denotate con la barra, dy barra, che è dy su dvi, ma partendo dall'output. Quindi, sto impostando y barra uguale a 1, che è ovviamente sempre vero perché è dy su dy, okay?

`00:58:58` 
E la cosa buona è che con solo un passaggio dall'output all'input, potete calcolare tutte le derivate dell'output rispetto all'input, okay? Quindi, con solo un passaggio attraverso il grafo computazionale o attraverso la lista di Wengert... le derivate corrispondenti, potete calcolare tutte le derivate. È.

`00:59:29` 
importante notare che ovviamente in generale potreste avere questo tipo di situazione dove avete la funzione con n input e m output. Nelle reti neurali di solito avete n molto maggiore di m, quindi il numero di caratteristiche è molto maggiore del numero di etichette o output che volete considerare. Quindi questo.

`01:00:05` 
è la ragione per cui nelle reti neurali la propagazione all'indietro è di importanza fondamentale, perché avete molti input e volete calcolare la sensibilità degli output rispetto a tutti gli input. Quindi, se siete in grado di venire con un algoritmo che è meno impegnativo dal punto di vista computazionale per calcolare tutte queste derivate, siete un vincitore, okay?

`01:00:39` 
In altre situazioni dove M è molto maggiore di N, ovviamente, in questi casi, la procedura in avanti è migliore, okay? Quindi, non sto affermando che l'inverso è sempre una buona scelta. È una buona scelta quando siamo in questa situazione, okay? Okay, quindi...

`01:01:11` 
Qui è in qualche modo riassunto quello che vi ho appena detto. La modalità formale è buona quando aggiungete n molto più piccolo di m. Ma quello che ci interessa la maggior parte delle volte è questo caso, che è il caso che appare più spesso nel contesto del machine learning.

`01:01:49` 
E se dovete calcolare lo Jacobiano, data una funzione che è caratterizzata da n input e m output, i costi per calcolare lo Jacobiano sono dati da, o n volte il costo del calcolo della valutazione della funzione f e supponiamo che per valutare la funzione f dovete eseguire cinque operazioni in virgola mobile allora significa che.

`01:02:24` 
per calcolare con il motore formale avrete bisogno di n un milione di volte cinque okay nella modalità inversa supponiamo che l'output sia 100 avete solo 100 per cinque che è molto più piccolo okay. questo è un metodo una regola empirica la maggior parte delle volte le funzioni di perdita che considereremo.

`01:03:05` 
Direi, molte volte, sono funzioni scalari, quindi con solo un output, per esempio, se avete i minimi quadrati, o generalizzati in qualche senso, è la freccia, quindi è solo un numero, okay? Quindi, M è uguale a uno, e l'input può essere caratterizzato da molte caratteristiche. Quindi, nel machine learning, in generale, la regola empirica è che la modalità inversa è più adatta.

`01:03:47` 
Okay, quali sono gli svantaggi della modalità inversa? Gli svantaggi della modalità inversa... La modalità inversa è correlata a... il fatto che dovete tenere a mente molti valori vedremo tra un momento e, quindi se la vostra funzione è complicata è complessa allora la quantità di memoria di cui avete bisogno per.

`01:04:27` 
memorizzare tutte le variabili intermedie che sono richieste per il calcolo può essere impegnativa. okay qui c'è un esempio quindi supponiamo che abbiamo la funzione che abbiamo considerato prima.

`01:04:57` 
per applicare la modalità inversa dobbiamo prima di tutto fare un passo in avanti per calcolare tutte le variabili b meno uno fino a v3 in questo caso okay poi dobbiamo calcolare per istanziare il passo inverso impostiamo v3 barra uguale a y barra uguale a uno sappiamo che v3 è uguale a v1 più.

`01:05:34` 
v2 okay quindi v1 barra ricordate che vi barra è dy su dvi okay quindi v v1 barra è, Questa è la notazione che stiamo usando.

`01:06:07` 
E come posso esprimere questo termine? Beh, sappiamo che y è v3, e qui posso usare essenzialmente la regola della catena. Quindi, posso dire dv3 su, sì, dv3 e dv3 su dv1.

`01:06:40` 
Questo è v3 barra, e questo è dv3 su dv1, che è 1 anche, okay? E potete vedere da qui. Okay, dv3 su dv1. Poi lo stesso per v2 barra. Poi avete v meno 1 e v0.

`01:07:11` 
Qui è uguale a v meno 1 più qualcosa. Quindi è v2 barra, che è dv2, dy su dv2, per dv2 su dv1. E qui avete 1. E poi avete dv2 su dv meno 1, che è dato da.

`01:07:44` 
questa espressione. Quindi v0. E v0 è qualcosa che sappiamo. È 5. Quindi qui è 5. Nello stesso modo, v0 barra è uguale a 2. E infine, v meno 1 barra è v1 barra dv1 su dv meno 1, che è esattamente come prima. Qui avete il logaritmo.

`01:08:15` 
Quindi questa derivata è 1 su v meno 1, e avete 0.5. Quindi alla fine, quello che avete è che per la f su dx1 è, 5 più 0.5, e per v0 è solo 2, okay? Quindi con solo un passo nella modalità all'indietro, nella modalità inversa, avete le due variabili, le due, scusate, le due derivate.

`01:09:00` 
Potete anche tornare a questa immagine, e potete essenzialmente impostare, come abbiamo visto, dv3 su dv3 uguale a 1, e poi tornare indietro attraverso il grafo computazionale. Quindi, da qui a qui, da qui a qui, e da qui a qui, e esattamente, otterrete esattamente lo stesso risultato.

`01:09:40` 
Quindi, quello che abbiamo descritto qui è esattamente quello che potete anche ottenere muovendovi sul grafo dall'output all'input. Se ricordate, lasciatemi tornare alla prima immagine, alla prima formula qui.

## Numeri Duali

`01:10:27` 
In quel caso, quello di cui abbiamo bisogno è il gradiente di f, o potrebbe essere lo Jacobiano, se avete la funzione con più input e più output. In alcune situazioni, siete interessati a calcolare il prodotto scalare del gradiente di f con un vettore dato, o il prodotto della matrice J con un vettore dato, quando questa operazione può sorgere.

`01:11:16` 
Per esempio, state usando... Uh... metodo iterativo che coinvolge la soluzione di un sistema lineare nel durante il processo potete avere la necessità di calcolare questa quantità quindi in quel caso in quel caso per calcolare o questo termine o questo possiamo sfruttare.

`01:11:56` 
il cosiddetto approccio matrix-free l'approccio matrix-free è un approccio che, è usato molte volte nell'analisi numerica e essenzialmente l'idea è che, concentriamoci su questo caso avete una matrice e siete interessati, a calcolare l'effetto di questa matrice su un vettore dato.

`01:12:27` 
Okay? In realtà, non siete interessati realmente alla matrice stessa, ma a questo prodotto, okay? La ragione per cui sono chiamati matrix-free è perché, in pratica, questi metodi non richiedono la costruzione esplicita della matrice, in questo caso la matrice J.

`01:13:02` 
Qual è l'idea? Se, quando usate la modalità in avanti, se... Impostate x punto, ricordate che x punto in quello che abbiamo visto qui era questo vettore essenzialmente, 1, 0, okay?

`01:13:33` 
E 1, 0 era la ricetta per calcolare la derivata della funzione f rispetto a x1. 0, 1 è la ricetta per calcolare la derivata di f rispetto a x2, okay? Supponiamo che siate interessati a calcolare il, in questo caso, è solo il gradiente, il prodotto scalare tra il gradiente di f e un vettore r dato da 1, 2, okay?

`01:14:15` 
Allora quello che potete fare è... inizializzare questo vettore x punto con uno due quindi se voi potete controllare se inizializzate qui, la procedura non con uno zero ma con uno due l'output che otterrete qui alla fine della procedura è esattamente questo il gradiente di f punto r dove r è il vettore che avete.

`01:14:55` 
usato per inizializzare la procedura quindi significa che uh uh senza la necessità del calcolo esplicito del gradiente potete venire con questo valore lo stesso è vero se avete. una situazione dove avete più di un output in quel caso se impostate il vettore iniziale x punto.

`01:15:30` 
uguale a b allora l'output y sarà lo jacobiano del il prodotto dello jacobiano di f per b se fate lo stesso nella modalità inversa invece che nella modalità in avanti quello che potete ottenere è che se impostate ricordate che nella modalità inversa dovete impostare l'output.

`01:16:05` 
okay quindi se l'output è caratterizzato da m variabili allora, Supponiamo che M sia 5, e voglio considerare il vettore 1, 2, 3, 4, 5. Se inizializzo l'output con questo vettore e eseguo esattamente questi passi,

`01:16:36` 
quello con cui verrò è un vettore che è il prodotto dello Jacobiano per il vettore B, con solo un passaggio in avanti. Perché in questo caso ho il vettore? Perché ricordate che con il modello inverso, state calcolando tutte le derivate. Quindi se avete una situazione dove qui avete M maggiore di 1, venite con il vettore finale.

`01:17:11` 
E quindi in ogni caso, siete in grado di calcolare queste quantità senza la necessità di creare esplicitamente la matrice J o il vettore gradiente di S. Stavo menzionando prima che in alcune situazioni, siete interessati a usare metodi che sono di ordine superiore a uno.

`01:17:49` 
E quindi potete usare, per esempio, il metodo di Newton. Il metodo di Newton in 1D coinvolge la presenza della derivata seconda della funzione. Nel caso dimensionale superiore. Avete bisogno di avere l'addizione. Ma anche in questo caso, vedremo più tardi, che quando considerate il metodo di Newton, quello che è di solito richiesto è essere in grado di calcolare il prodotto dell'azione per un fattore di incremento dato, delta x.

`01:18:45` 
Quindi di nuovo, quello che dovete fare è trovare un buon modo per calcolare questa quantità, in qualche modo in modo simile a prima. Quindi se avete, per esempio, una funzione con f input e un output, Potete applicare la modalità inversa al gradiente e quindi il gradiente è stato calcolato con solo un passaggio della modalità inversa e potete venire con le n componenti del gradiente.

`01:19:38` 
Okay, questo è esattamente quello che abbiamo visto. Poi dovete, per ogni componente del gradiente, dovete calcolare una colonna dell'azione. Okay, quindi dovete ricalcolare n passi di n passi inversi. Okay, quindi alla fine, quello di cui avete bisogno è n al quadrato. Okay, e di nuovo, se avete.

`01:20:10` 
Okay, quindi dovete calcolare n passi di n passi inversi. Okay, quindi dovete calcolare n passi di n passi inversi, una situazione dove n è più grande questo approccio non è fattibile quindi qual è l'uh l'idea, l'idea è quello che viene chiamato l'approccio inverso e in avanti e l'idea principale è introdurre una funzione aggiuntiva che è g di x che è il gradiente per il vettore b.

`01:20:48` 
che è cos'è la derivata direzionale di b di f lungo b okay. e ora se prendete il gradiente di questa funzione allora avete esattamente, l'azione per il vettore v, okay? Questa è l'idea.

`01:21:18` 
Quindi, ricordate che nella modalità inversa, il primo passo coinvolge un passaggio in avanti per calcolare tutti i valori delle variabili di cui avete bisogno. Okay? In questo caso, dato che stiamo applicando, stiamo considerando questa nuova funzione g, eseguiamo il passaggio in avanti su g per calcolare tutti i valori,

`01:21:53` 
tutti i valori intermedi. Poi calcolate il gradiente, sulla funzione che avete calcolato qui. solo usando un singolo passaggio inverso, e quello con cui potete venire è l'azione per b. Quindi l'idea è, calcolare una g con un passo in avanti qui, poi usando un passaggio inverso per calcolare il gradiente di g,

`01:22:36` 
che è in particolare l'azione di f per b, e y per b perché qui avete inizializzato la procedura con x punto uguale a b. Okay, in, come avete visto, anche se la modalità inversa è quella che di solito adottiamo perché è importante per gestire situazioni dove avete più input, molti più input che output,

`01:23:40` 
abbiamo, se guardate, per esempio, a questa tabella, dobbiamo essere in grado di calcolare derivate semplici, come, per esempio, log di v1, qui devo calcolare le derivate di questa funzione. Okay? E in questo senso, uno strumento importante che è un altro blocco di costruzione di grandi librerie per la differenziazione automatica è uno strumento che si chiama numeri duali.

`01:24:20` 
Cosa sono i numeri duali? I numeri duali sono nuovi numeri, molto simili ai numeri complessi. Sapete che i numeri complessi sono caratterizzati da una parte reale e una parte immaginaria, dove avete l'unità immaginaria.

`01:24:51` 
Quindi un numero complesso generico è una parte. A più IB. Okay, un numero duale è qualcosa della forma a più epsilon d. Sappiamo che in questo contesto, la caratterizzazione di i è che la radice quadrata di i è meno 1. Okay, qual è la caratterizzazione di epsilon?

`01:25:24` 
Epsilon è un cosiddetto numero nilpotente. È un numero che è diverso da zero, ma il quadrato di epsilon è zero. A prima vista, si potrebbe dire che è qualcosa che suona molto strano, perché non è qualcosa che è molto comune alla nostra intuizione.

`01:25:55` 
Ma forse se... Stop. Consideriamo una situazione che può essere una rappresentazione assertiva del numero duale è questa. Quindi supponiamo che il numero duale a più b epsilon sia scritto come questa matrice.

`01:26:25` 
Quindi a più epsilon b è a a 0 b, che è essenzialmente come scrivere a per la matrice identità più b per, lasciatemi chiamarlo, e. o epsilon se vogliamo, ma ora è una matrice dove epsilon è questa matrice.

`01:27:07` 
Questa matrice è diversa da zero, ma se calcolate epsilon al quadrato, è uguale a zero. È la matrice nulla. OK, quindi in questo contesto, può suonare abbastanza strano se reinterpretate in termini di rappresentazione matriciale. Forse è più facile afferrare il significato di questo epsilon.

`01:27:40` 
Quindi formalmente, Questa è la definizione, okay, quindi abbiamo il numero a più epsilon b, e questo è il numero duale, dove è simile a quello a cui siamo abituati per il numero complesso, a è chiamata la parte reale, e b è chiamata la parte duale, okay.

`01:28:12` 
Perché sono interessanti? Perché se considerate una funzione f, qualsiasi funzione, e valutate questa funzione in un numero duale caratterizzato da una parte duale uguale a uno, quindi x più epsilon, Allora potete usare l'espansione di Taylor per esprimere questa quantità, quindi è f di x più f primo di x per epsilon, che è l'incremento, più la derivata seconda su due fattoriale per epsilon al quadrato, e poi avete tutti gli altri termini.

`01:29:02` 
Ma da questo termine in poi, sono tutti zero, perché epsilon al quadrato è zero, e tutte le altre potenze di epsilon. Quindi, alla fine, quello che abbiamo è che f di x più epsilon è uguale a f di x più f primo di x per epsilon. Quindi, in altre parole, se avete una funzione f, e valutate la funzione f.

`01:29:36` 
In un numero duale di epsilon. Quello con cui venite è un numero duale, questo è un numero duale, dove la parte reale è il valore della funzione in x, scusate, la parte reale è il valore della funzione in x, e la parte duale è il valore della derivata prima in x. Quindi con una valutazione, avete il valore della funzione e il valore della derivata, okay? Questo è il potere della teoria dei numeri duali, almeno in questo contesto.

`01:30:19` 
Cerchiamo di elaborare un po' su questo con un esempio. Supponiamo che abbiamo la funzione x al quadrato. su coseno di x e impostiamo x uguale a pi più epsilon okay quindi secondo quello che noi.

`01:30:50` 
abbiamo visto nella slide precedente dobbiamo valutare la funzione f in x più epsilon, dobbiamo calcolare il numeratore che è pi più epsilon al quadrato che non è altro che pi al quadrato più due pi epsilon e al denominatore, abbiamo coseni di pi più epsilon potete usare la formula per l'addizione del.

`01:31:24` 
coseni e potete venire con meno uno okay, Quindi, ora dovete eseguire la divisione, e alla fine avete questo numero duale, che è meno pi al quadrato meno 2 pi epsilon. Quello che affermiamo è che meno pi al quadrato è il valore della funzione f in pi, e meno 2 pi è il valore della derivata di f in pi,

`01:32:07` 
che è esattamente quello che otterreste con il calcolo esplicito. Alcune proprietà dei numeri duali che sono utili. Per favore praticate. Se avete la somma di due numeri duali, allora è come il numero complesso, avete la somma delle parti reali e la somma delle parti duali.

`01:32:46` 
Per il prodotto, esattamente come nel caso del numero complesso, in quel caso qui avreste avuto i al quadrato, che è meno uno. In questo caso, avete epsilon al quadrato, che è zero. Quindi alla fine, quello che avete è solo questo risultato, okay? Quindi avete un nuovo numero duale con la parte reale uguale a hc e la parte duale uguale a ad più bc.

`01:33:20` 
Cosa significa, quel prodotto? Essenzialmente, è un altro modo di scrivere, se volete, di derivare la regola per derivare il prodotto di due funzioni, okay? Perché ricordate che il prodotto di due funzioni è f per g primo è f primo g più f g primo, che è esattamente quello che avete qui.

`01:34:00` 
Okay? Essenzialmente. A, d, quindi la parte reale del primo per la parte duale del secondo, e b, c, okay? Che dire della funzione composta? Quindi supponiamo che abbiate h di x uguale a g di f di f di x. Ricordate, l'idea è sempre, devo valutare la funzione nel numero duale dato da x più epsilon.

`01:34:38` 
Quindi, h in x più epsilon è g di f valutato in x più epsilon. Ora, diamo un'occhiata a questo. Questo è f valutato in questo numero duale, che è f in x più primo in x per epsilon. Questo è il risultato che avevamo prima con l'espansione di Taylor. E ora, qui, ho g valutato su questo numero duale.

`01:35:13` 
Okay? La differenza principale è che ora la parte reale, scusate, la parte duale del numero dove sto valutando la funzione non è più uguale a uno, ma è uguale a f primo di x. Quindi abbiamo g, il valore di g nella parte reale, più il valore, la derivata di g valutata nella parte reale per questo, che è esattamente la formula per la derivazione di una funzione composta.

`01:35:59` 
g di f di x più, o se volete... Questa è la formula, g primo di f di x per f primo di x, okay? Quindi questo è... la formula per la derivazione della funzione composta se volete questo è un altro modo di, derivare la regola della catena di nuovo in alcune situazioni siamo interessati a.

`01:36:41` 
calcolare derivate di ordine superiore sono i numeri duali utili anche per questo tipo di situazione se modificate un po' la definizione sì in che senso nel senso che se invece di chiedere che epsilon, sia diverso da zero e il quadrato sia uguale a zero possiamo chiedere che epsilon sia, diverso da zero epsilon al quadrato sia diverso da zero ma epsilon al cubo sia uguale a zero.

`01:37:21` 
quindi di nuovo sfruttando l'espansione di taylor ora se valutiamo la funzione f in questo numero duale, abbiamo questi tre termini quindi con una singola valutazione stiamo ottenendo, il valore della funzione il valore della derivata prima, e il valore della derivata seconda fino a un fattore di uno okay quindi in in altre parole.

`01:38:03` 
voi, con solo una valutazione, potete venire con questo valore. Se avete una funzione di più variabili, allora dovete eseguire questo trucco per ogni variabile. Quindi essenzialmente, potete impostare, per esempio, se volete calcolare la derivata rispetto a x k,

`01:38:36` 
allora dovete impostare x k più epsilon e tutti gli altri solo x j per le altre componenti del vettore di input. E il risultato sarà la derivata di quello rispetto a x k. Qui c'è un esempio del calcolo della derivata seconda, quindi se avete, per esempio, la funzione x al cubo, impostiamo l'input x uguale a 2, 1, 0, e possiamo calcolare x al cubo.

`01:39:35` 
calcolando x al secondo e poi moltiplicando per x di nuovo, quello che ottenete è questa funzione, e abbiamo il valore della funzione in 8, il valore della derivata, e derivata prima e derivata seconda, okay? Voglio solo essere chiaro, i numeri duali, esattamente come la differenziazione automatica in generale, non sono intesi, per esempio, se avete f di x uguale a 1 su x al quadrato, okay?

`01:40:19` 
In linea di principio, si potrebbe dire, usando il formalismo dei numeri duali, posso dire che, okay, qui, posso valutare questa funzione in x più epsilon al quadrato, che è 1 su x più 2x epsilon più epsilon al quadrato. Questo è 0, quindi posso cancellare.

`01:40:50` 
E poi, qui, quello che posso fare è, posso moltiplicare. per x meno 2x epsilon, e quindi qui avete x al quadrato, poi avreste avuto 4x epsilon al quadrato, che è zero,

`01:41:23` 
e qui avete x meno 2x epsilon, quindi avete 1 su, vediamo qui,

`01:42:00` 
Quindi siamo qui al quadrato, quindi abbiamo 1 su x al quadrato, che è il valore della funzione, meno 2 per 1 su x al cubo epsilon.

`01:42:35` 
E questa è la derivata. Ma in generale, non siamo interessati all'uso del numero duale per ottenere... Formula teorica, ma per dipingere numeri, il potente di tutto questo metodo si basa sul fatto che non vogliamo calcolare formule, ma numeri. Okay, qui c'è la stessa idea per calcolare il gradiente della funzione x1 per coseno di x2.

`01:43:30` 
Quindi l'idea è, come abbiamo detto prima, impostare la variabile. Qui siamo interessati alla derivata prima, quindi stiamo impostando la prima variabile uguale a 2 più epsilon, perché vogliamo considerare 2, e vogliamo considerare la derivata rispetto a x1, e la seconda è pi più 0, e poi nella seconda invertiamo l'idea, 2 più 0 e pi più epsilon.

`01:44:11` 
E avete le due derivate, meno 1 e 0, quindi il gradiente nel punto 2 pi è meno 1 e 0. Quindi questa è l'idea del metodo.

`01:44:59` 
Okay, forse possiamo fermarci qui. C'è qualche domanda su questo argomento?
---

## Appendice: Formule Chiave di Automatic Differentiation

### 4 Metodi per Calcolare Derivate

| Metodo | Pro | Contro | Complessit� |
|--------|-----|--------|-------------|
| **Manuale** | Elegante, formula esplicita | Error-prone, tedioso | O() tempo umano |
| **Numerico** (finite diff.) | Facile, black-box | Instabile (cancellation), O(h) error | O(n) evaluations |
| **Simbolico** (Maple) | Esatto, formula chiusa | Expression bloat, lento | O(exp(n)) crescita |
| **Automatic (AD)** | Esatto + efficiente, scalabile | Richiede implementazione | O(n) o O(m) |

### Finite Differences (Metodo Numerico)

**Forward Difference** (1st order: O(h)):
`
f'(x)  [f(x+h) - f(x)] / h
`

**Central Difference** (2nd order: O(h)):
`
f'(x)  [f(x+h) - f(x-h)] / (2h)
`

**Errore Totale**:
- **Truncation error**: O(h^n)  decresce con h
- **Roundoff error**: O(e/h)  cresce con h piccolo
- **Optimal h**: h_opt ~ e per central diff (e = machine precision  10)

**Problemi**:
- Cancellation: f(x+h)  f(x) quando h  0
- Floating-point errors dominano per h < 10
- Serve h "giusto": n� troppo grande n� troppo piccolo

**Riferimento Slide**: AutoDiff.pdf, Slide 5 - Grafico log-log: errore vs h (V-shape) con optimum

### Symbolic Differentiation (Maple/Mathematica)

**Pro**: Formula esatta chiusa

**Problema: Expression Bloat**

Esempio: f(x) = xsin(x)cos(x)exp(x)

**Derivata 1**:
`
f'(x) = sin(x)cos(x)exp(x) + xcos(x)exp(x) - xsin(x)exp(x) + xsin(x)cos(x)exp(x)
`
(~40 operations)

**Derivata 2**: ~200 operations
**Derivata 3**: ~1000+ operations

**Expression bloat**: Size cresce esponenzialmente con ordine derivata!

**Memoria**: Formule enormi  impraticabile per DL

### Automatic Differentiation (AD)

**Principio**: Decomporre f in operazioni elementari, applicare chain rule sistematicamente

**Operazioni Elementari**:
`
+, -, , , exp, log, sin, cos, tan, sqrt, ^, ...
`

Tutte hanno derivate **note**!

**Wengert List** (Evaluation Trace):

Esempio: f(x, x) = ln(x) + xx - sin(x)

`
v = x                    (input)
v  = x                    (input)
v  = ln(v)              v = ln(x)
v  = v  v             v = xx
v  = sin(v)              v = sin(x)
v  = v + v              v = ln(x) + xx
v  = v - v              v = f(x,x)  [output]
`

**Computational Graph**:
`
    x          x
               
   ln()        sin()
              
    v      v
       v      
              
        + 
              
        -  
              
        y
`

**Edge Derivatives**: v?/v? per ogni arco (i  j)

### Forward Mode AD

**Idea**: Calcolare v?/x? per ogni variabile intermedia v?

**Tangent variables** (derivatives w.r.t. input x?):
`
v? = v?/x?
`

**Inizializzazione** (seed):
`
Per calcolare f/x:
  v = 1  (x/x = 1)
  v  = 0  (x/x = 0)
`

**Propagazione Forward** (insieme a valori primal):

Per ogni operazione v? = f(v?, v):
`
Primal:   v? = f(v?, v)
Tangent:  v? = (f/v?)v? + (f/v)v
`

**Esempio**:
- v? = v?  v    v? = vv? + v?v  (product rule!)
- v? = sin(v?)    v? = cos(v?)v?     (chain rule!)

**Costo**: 
- **n input**  serve **n passi forward** (uno per ogni f/x?)
- **Buono per**: n  m (pochi input, molti output)
- **Complessit�**: O(n  cost(f))

### Reverse Mode AD (Backpropagation)

**Idea**: Calcolare y/v? per ogni variabile intermedia v?

**Adjoint variables** (derivatives of output w.r.t. intermediate):
`
v? = y/v?
`

**Inizializzazione**:
`
? = y/y = 1  (seed)
`

**Backward Propagation** (dopo forward pass):

Per v? che appare in v? = f(..., v?, ...):
`
v? += v?  (v?/v?)   [accumula da tutti j che usano v?]
`

**Ordine**: Dal nodo output verso input (reverse topological order)

**Esempio** (ln(x) + xx - sin(x)):

Forward pass (calcola valori):
`
v = ln(x),  v = xx,  v = sin(x)
v = v + v,  v = v - v
`

Backward pass (calcola adjoints):
`
v = 1
v = v  1 = 1         [v/v = 1]
v = v  (-1) = -1     [v/v = -1]
v = v  1 = 1         [v/v = 1]
v = v  1 = 1         [v/v = 1]
x = v  x + v  cos(x) = x - cos(x)  [f/x]
x = v  (1/x) + v  x = 1/x + x      [f/x]
`

**Costo**:
- **m output**  serve **m passi backward** (uno per ogni output y?)
- **Buono per**: n  m (molti input, pochi output)  **Neural Networks!**
- **Complessit�**: O(m  cost(f))
- Per NN: m = 1 (scalar loss)  **1 backward pass** calcola TUTTI i gradienti!

**Backpropagation = Reverse Mode AD applicato a NN loss**

### Forward vs Reverse Mode

| Aspetto | Forward Mode | Reverse Mode |
|---------|--------------|--------------|
| Calcola | v?/x? (column of Jacobian) | y/v? (row of Jacobian) |
| Variabili | Tangents v? | Adjoints v? |
| Direzione | Input  Output | Output  Input |
| Passes | n forward (for n inputs) | m backward (for m outputs) |
| Costo | O(n  cost(f)) | O(m  cost(f)) |
| Meglio per | n  m (few inputs) | n  m (few outputs) |
| NN Training |  (milioni di parametri) |  (1 scalar loss) |

**Per NN**: n ~ 10-10 parametri, m = 1 loss  **Reverse mode vince!**

### Matrix-Free Methods

**Problema**: Jacobian J  R^(mn) � troppo grande da memorizzare (mn elementi)

**Soluzione**: Calcolare **prodotti Jacobiano-vettore** senza costruire J!

**1. Jacobian-Vector Product (JVP)**: Jv

Usando **Forward Mode**:
`
(Jv)? = S? (f?/x?)v? = f/xv
`

Seed: v = v  Forward mode  ottieni Jv

**Costo**: O(cost(f)) (1 forward pass!)

**2. Vector-Jacobian Product (VJP)**: w^TJ

Usando **Reverse Mode**:
`
(w^TJ)? = S? w?(f?/x?)
`

Seed: ? = w  Reverse mode  ottieni w^TJ

**Costo**: O(cost(f)) (1 backward pass!)

**Applicazioni**:
- **Hessian-vector products**: fv (for Newton's method)
- **Gradient computation**: L = J^T?Loss (reverse mode!)

### Numeri Duali

**Definizione**: Estensione di R con elemento e tale che **e = 0**

Numero duale: x = x + x'e

**Aritmetica**:
`
(x + x'e) + (y + y'e) = (x+y) + (x'+y')e
(x + x'e)  (y + y'e) = xy + (x'y + xy')e  [e = 0!]
`

**Funzioni Elementari** (Taylor expansion + e = 0):

`
exp(x + x'e) = exp(x) + x'exp(x)e
ln(x + x'e)  = ln(x) + (x'/x)e
sin(x + x'e) = sin(x) + x'cos(x)e
cos(x + x'e) = cos(x) - x'sin(x)e
`

**Forward Mode con Numeri Duali**:

Per calcolare f'(a):
`
f(a + e) = f(a) + f'(a)e
`

Valuta f con aritmetica duale  parte e d� f'(a)!

**Esempio**: f(x) = x/cos(x) in x = p

`
x = p + e
x = p + 2pe
cos(x) = cos(p) + (-sin(p))e = -1
x/cos(x) = p/(-1) + 2pe/(-1) = -p - 2pe

 f(p) = -p, f'(p) = -2p  
`

**Derivate di Ordine Superiore**: Usare numeri "tri-duali" etc.

### Complessit� Computazionale

**Notazione**:
- n: # input (parametri)
- m: # output  
- ops(f): # operazioni elementari in f

**Metodi**:

| Metodo | Complessit� | Note |
|--------|-------------|------|
| Manual | O() | Tempo umano! |
| Finite Diff (forward) | O(n  ops(f)) | n evaluations di f |
| Finite Diff (central) | O(2n  ops(f)) | 2n evaluations |
| Symbolic | O(exp(ops(f))) | Expression bloat! |
| **AD Forward** | O(n  ops(f)) | n forward passes |
| **AD Reverse** | O(m  ops(f)) | m backward passes |

**Per Neural Networks** (n ~ 10, m = 1):
- Forward mode: O(10  ops) 
- Reverse mode: O(1  ops) = O(ops) 

**Memoria**:
- Forward: O(ops) (store intermediates durante forward)
- Reverse: O(ops) (store tape per backward)

### Chain Rule in AD

**Univariate**: (f  g)'(x) = f'(g(x))  g'(x)

**Multivariate** (vector chain rule):

Se y = f(u, ..., u) e u? = g?(x, ..., x):

`
y/x? = S? (y/u?)  (u?/x?)
`

**Forward Mode**: Propaga u?/x? (bottom-up)
**Reverse Mode**: Propaga y/u? (top-down)

**Esempio Composizione**: h(x) = f(g(g(x)))

`
Forward:  g = g'(x), g = g'(g)g, ? = f'(g)g
Reverse:  h = 1, ? = hf'(g), ? = ?g'(g), x = ?g'(x)
`

### Implementazioni Pratiche

**Librerie AD**:

**Python**:
- **PyTorch**: torch.autograd (reverse mode, dynamic graphs)
- **TensorFlow**: tf.GradientTape (reverse mode)
- **JAX**: jax.grad, jax.jacfwd, jax.jacrev (both modes!)
- **Autograd**: autograd.grad (numpy-based)

**Julia**:
- **ForwardDiff.jl**: Forward mode (numeri duali)
- **ReverseDiff.jl**: Reverse mode
- **Zygote.jl**: Source-to-source AD

**C++**:
- **CppAD**: Operator overloading
- **ADOL-C**: Tape-based

**Esempio PyTorch**:
`python
import torch

# Define function with requires_grad=True
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x[0]**2 + x[0]*x[1] - torch.sin(x[1])

# Backward pass (reverse mode AD)
y.backward()

# Gradient
print(x.grad)  # [y/x, y/x]
`

**Esempio JAX** (both modes):
`python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.dot(x, x)

x = jnp.array([1.0, 2.0, 3.0])

# Reverse mode (grad)
grad_f = jax.grad(f)
print(grad_f(x))  # [2.0, 4.0, 6.0]

# Forward mode (JVP)
v = jnp.array([1.0, 0.0, 0.0])
_, jvp = jax.jvp(f, (x,), (v,))
print(jvp)  # f/x = 2.0
`

---

## Riferimenti Bibliografici

1. **Griewank, A., & Walther, A. (2008)**. "Evaluating derivatives: principles and techniques of algorithmic differentiation." *SIAM*.

2. **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018)**. "Automatic differentiation in machine learning: a survey." *Journal of Machine Learning Research*, 18, 1-43.

3. **Nocedal, J., & Wright, S. J. (2006)**. "Numerical optimization." *Springer*.

4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. "Deep Learning." *MIT Press*. Chapter 6: "Backpropagation and Other Differentiation Algorithms".

5. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986)**. "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.

---

**Fine Lezione 15 - Automatic Differentiation**

*Prossima lezione: Optimization Methods (Gradient Descent Variants, Newton's Method, Conjugate Gradient)*
