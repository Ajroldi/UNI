## [00:00] Introduzione e obiettivi del laboratorio
Contesto: questo è il quinto laboratorio, dedicato alla discesa del gradiente (gradient descent). Lavoreremo in due parti:
- Implementazione di gradient descent e varianti:
  - passo fisso (learning rate costante),
  - backtracking (passo adattivo per garantire diminuzione sufficiente),
  - exact line search per funzioni quadratiche (passo ottimo analitico).
- Applicazioni e test su funzioni di benchmark non convesse, quindi su due problemi di apprendimento:
  - regressione lineare (loss quadratica),
  - Support Vector Machine (SVM) per regressione e classificazione.
Nota operativa: useremo JAX per il calcolo automatico del gradiente (autodiff).
## [00:00] Controllo della soluzione e impostazione del problema
- Tutto a posto; procediamo con il controllo della soluzione.
- Punto di partenza: copiare/incollare il codice base e verificare le differenze. Nel caso del backtracking, non usiamo un learning rate fisso ma lo calcoliamo dinamicamente.
## [02:00] Perché iniziare dalla regressione lineare
- La regressione lineare è un caso di test ideale: semplice ma completo, contiene gli elementi base del training (loss, gradiente, learning rate, criteri di arresto, iterazioni).
- Nella prossima lezione useremo gli stessi “ingredienti” per addestrare reti neurali.
- Aggiungeremo poi una SVM per regressione o classificazione: gli ingredienti non cambiano, aumenta solo la complessità del modello.
## [04:00] Piano del notebook
Implementeremo:
- Gradient descent “vanilla” (passo fisso).
- Gradient descent con backtracking (passo adattivo).
- Exact line search per funzioni quadratiche (passo analitico ottimale).
Testeremo gli algoritmi su tre funzioni di benchmark non convesse con molti minimi locali, ottime per valutare stabilità e convergenza. Il gradiente sarà calcolato con JAX.
## [06:00] Visualizzare funzioni 2D: meshgrid e contour
Obiettivo: visualizzare il comportamento del gradient descent su funzioni 2D, mostrando livelli (contour) e traiettorie iterative.
Passi:
1) Discretizzazioni 1D
- Scegli una griglia per x (es. da −5 a 5 con 50 punti).
- Scegli una griglia per y (es. da −5 a 5 con 30 punti).
2) Griglia 2D
- Usa meshgrid (np.meshgrid o jnp.meshgrid) con i vettori x e y.
- Ottieni matrici X e Y: X ripete i valori di x per riga; Y ripete i valori di y per colonna.
3) Valutazione della funzione
- Per z = sin(x) · sin(y), calcola Z applicando sin a X e Y.
4) Plot dei livelli
- Usa plt.contourf(X, Y, Z) per le linee di livello riempite.
- Sovrapponi le traiettorie del metodo con plt.plot.
- Aggiungi colorbar per mappare valori e colori (per sinusoidi: max 1, min −1).
Idea chiave: meshgrid trasforma griglie 1D in una griglia 2D per valutare e visualizzare funzioni f(x, y) su tutta la superficie.
## [11:00] Funzioni di benchmark per ottimizzazione
- Useremo funzioni 2D note con più minimi locali (cfr. liste su Wikipedia).
- Obiettivo: valutare come gli algoritmi si comportano in scenari non convessi, dove è facile “bloccarsi” in minimi locali.
- Le formule esatte non sono centrali: le funzioni servono come test.
## [13:00] Gradient Descent (passo fisso) – Algoritmo e parametri
Parametri:
- grad\_f: funzione che calcola il gradiente della loss.
- x0: punto iniziale.
- lr: learning rate (ampiezza del passo).
- tol: soglia per arresto su ||grad||.
- max\_iter: massimo numero di iterazioni.
Procedura:
1) Inizializza x = copia di x0 (per evitare aliasing) e crea “path” con x.
2) Loop per iterazione in [1, max\_iter]:
- Calcola g = grad\_f(x).
- Aggiorna x = x − lr · g.
- Salva x nel path.
- Se ||g|| < tol, arresta (punto stazionario, idealmente un minimo).
Buona pratica: usare “\_” per variabili non usate nel loop, per chiarezza.
## [15:30] Conclusione e note operative (prima parte)
- Verifica della soluzione: tutto ok.
- Indicazioni operative: proseguire con implementazioni e test.
## [18:00] Gradient Descent con backtracking (learning rate dinamico)
Idea: adattare il passo ad ogni iterazione per ottenere una diminuzione “sufficiente” (condizione tipo Armijo).
Procedura:
1) Copia la struttura di gradient descent standard.
2) A ogni iterazione:
- Imposta t = 1 (passo iniziale grande).
- Finché la condizione di backtracking non è soddisfatta, riduci t = β · t (β < 1, tipicamente 0.8).
- Aggiorna x = x − t · g.
- Salva nel path.
Vantaggi: evita overshooting (passi troppo grandi) e stagnazione (passi troppo piccoli), migliorando stabilità e velocità di convergenza. È più costoso per iterazione, ma riduce il numero di iterazioni necessarie.
## [21:00] Exact line search per funzioni quadratiche
Scenario: funzione quadratica f(x) = 1/2 xᵀ A x − bᵀ x + c, con A simmetrica definita positiva.
Concetti:
- Gradiente: ∇f(x) = A x − b (o Ax + b, in base alla convenzione dei segni).
- Passo ottimo lungo −g: t* = (gᵀ g) / (gᵀ A g).
Procedura:
1) Calcola g = grad\_f(x).
2) Calcola t* con la formula analitica.
3) Aggiorna x = x − t* · g.
4) Salva nel path e verifica arresto su ||g||.
Vantaggi: passo ottimale lungo la direzione corrente, convergenza molto rapida su quadratiche (validità limitata a questo caso).
## [24:00] Suggerimenti pratici e controllo dei plot
- Farsi un’idea qualitativa della forma delle funzioni aiuta a validare i grafici.
- La colorbar permette di leggere i valori associati ai colori.
- Sovrapporre le traiettorie ai contour mostra il percorso verso i minimi (il blu indica valori bassi).
## [27:00] Criteri di arresto e implementazione
- Arresto tipico: ||grad\_f(x)|| < tol.
- Impostare sempre max\_iter per evitare loop infiniti.
- Salvare copie di x nel path per non sovrascrivere punti precedenti.
## [30:00] Conclusione del modulo su Gradient Descent
- Abbiamo implementato:
  - gradient descent “vanilla”,
  - backtracking,
  - exact line search su quadratiche.
- Prossimi passi:
  - completare varianti e test su benchmark,
  - visualizzare traiettorie e interpretare i risultati in scenari non connessi.
- Osservare come il learning rate influenza stabilità, velocità e capacità di evitare minimi locali.
## [03:15] Test automatici e visualizzazione
- Setup: si parte da x0 = (4, 4), si compila JIT il gradiente e si eseguono:
  - gradient descent standard,
  - gradient descent con backtracking.
- Salviamo ultimo punto e traiettoria, poi visualizziamo:
  - A sinistra: contorno della funzione (molti minimi locali, unico minimo globale in (0,0)).
  - A destra: numero di iterazioni vs errore dal minimo.
- Risultati:
  - Senza backtracking: passi piccoli, stagnazione dopo molte iterazioni.
  - Con backtracking: passo iniziale grande, poi adattivo, errore ridotto più rapidamente.
## [05:00] Parametri e stabilità del learning rate
- Esperimenti con lr = 1 nel metodo standard:
  - Possibile rallentamento o instabilità, dipende dalla funzione.
- Problemi di connessione e ripresa: si riparte dopo interruzioni tecniche, la registrazione continua.
## [06:30] Overshooting e minimi locali
- Con lr = 1: overshooting evidente, la linea rossa salta tra valli (minimi locali). Passo troppo grande, instabilità su funzioni non convesse.
- Con lr troppo piccolo (es. 0.001): progressi minimi, servono molte iterazioni.
- Anche su quadratiche, gradient descent standard può essere inefficiente: backtracking dà passi ragionevoli e convergenza in decine di iterazioni fino a ~1e−6, coerente con i limiti della precisione single (epsilon macchina ~1e−8).
## [09:00] Exact line search su quadratiche: effetti
- Passo ottimo lungo la direzione del gradiente a ogni iterazione.
- Richiede ancora meno iterazioni rispetto al backtracking, ma vale solo per quadratiche.
## [10:00] Collegamento al training di reti neurali
- Gli ingredienti usati (loss, gradiente, learning rate, criteri di arresto, iterazioni) sono gli stessi impiegati per addestrare reti neurali, anche modelli molto grandi.
- Comprendere bene questi concetti è essenziale.
## [10:30] Problema di regressione lineare: setup e obiettivi
- Dati sintetici: retta “vera” y = 1.5x + 3 con rumore gaussiano.
- Obiettivo: apprendere θ0 (intercetta) e θ1 (pendenza) minimizzando la MSE.
- Dataset: 200 punti, x uniformi in [0, 10]; seed = 0 per riproducibilità.
## [11:30] Train/Test split e validazione
- Suddividere i dati in training e test (in ML spesso anche validation).
- Test set: dati mai visti in addestramento, serve per valutare generalizzazione.
- Comodo usare train\_test\_split di scikit‑learn con percentuale e seed.
## [12:30] Definizione del modello, loss e gradiente (con JAX)
Ingredienti:
- Modello lineare: ŷ = θ0 + θ1 · x.
- Loss MSE: media dei quadrati degli errori ŷ − y.
- Gradiente della loss: si ottiene con jax.grad sulla funzione MSE.
Dettaglio sugli argomenti:
- jax.grad differenzia rispetto al primo argomento della funzione.
- Assicurarsi che la firma sia MSE(θ, x, y) per ottenere ∂/∂θ.
## [13:30] Aggiornamento dei parametri (Gradient Descent)
- A ogni iterazione:
  - Calcolare MSE(θ, x\_batch, y\_batch).
  - Calcolare ∇θ MSE.
  - Aggiornare θ ← θ − lr · ∇θ MSE.
- Questo corrisponde a muoversi nello spazio dei parametri verso il minimo della loss.
## [14:15] Stochastic Gradient Descent (SGD): principi
- Invece di usare tutto il dataset, si usa un mini-batch:
  - Vantaggio: costo per iterazione più basso.
  - Svantaggio: gradiente più rumoroso, ma spesso utile per generalizzare.
- Procedura:
  - Per ogni epoca, permutare il dataset.
  - Estrarre un batch.
  - Calcolare gradiente e aggiornare θ.
- Esercizio: implementare le funzioni fondamentali (brevi ma cruciali).
## [03:10] Ciclo di addestramento: epoche, permutazioni, mini-batch
Terminologia:
- Epoca: un passaggio completo sul training set.
- max\_epochs: quante volte ripetiamo il ciclo.
Procedura:
- A ogni epoca, aggiornare il random key di JAX e permutare gli indici.
- Iterare per i in 0..len(X\_train) a passo batch\_size:
  - Selezionare indici del batch.
  - Creare X\_batch e Y\_batch.
  - Aggiornare θ con lo step SGD.
## [04:10] Gestione della casualità in JAX
- JAX richiede gestione esplicita del random key.
- Aggiornare il key a ogni epoca e generare nuove permutazioni.
- Beneficio: batch casuali aumentano la stabilità media dei gradienti.
## [05:00] Parametri di training: inizializzazione e tuning
- Inizializzazione: θ = [0, 0].
- Parametri: θ iniziale, X\_train/Y\_train, X\_test/Y\_test, learning\_rate, max\_epochs, batch\_size.
- Effetto del batch\_size:
  - Grande: gradiente più accurato, LR potenzialmente maggiore, costo per aggiornamento più alto.
  - Piccolo: gradiente più rumoroso, LR più piccolo e più epoche.
- Esempio: 100 epoche di SGD portano θ vicino ai parametri reali (≈ [3, 1.5]).
## [06:10] Visualizzazione: dati e retta appresa
- Tracciare punti di train/test con scatter.
- Costruire x\_plot = linspace(0, 10, 1000).
- Valutare y\_plot = θ0 + θ1 · x\_plot e tracciare la retta.
- Obiettivo: verificare visivamente l’adattamento della retta ai dati.
## [06:40] Chiusura: importanza dei concetti
- Concetti fondamentali:
  - Modello (previsione),
  - Loss (MSE come esempio),
  - Gradiente (autodiff JAX),
  - Aggiornamento (SGD, mini-batch),
  - Casualità (random key, permutazioni),
  - Tuning (learning\_rate, batch\_size, epoche).
- Questi elementi saranno riutilizzati in modelli più complessi.
## [00:00] Modello – Idea chiave
Contesto: dato un input x e parametri θ, il modello produce una previsione ŷ.
Definizione:
- θ0: intercetta,
- θ1: pendenza,
- x: input,
- ŷ = θ0 + θ1 · x.
Estensione a modelli complessi (reti neurali):
- x diventa un vettore,
- θ diventa una collezione di matrici (pesi) per strati,
- si combinano trasformazioni lineari e attivazioni,
- il principio resta: dato x e θ, il modello produce ŷ.
## [00:45] Funzione di perdita (loss)
- Misura la differenza tra previsioni e dati reali.
- MSE: media dei quadrati degli errori e = ŷ − y.
- Se i dati sono più grandi (matrici ampie), aumentano i calcoli e il costo computazionale.
## [01:30] Gradiente della loss con JAX
- Definire MSE(θ, x, y).
- Usare jax.grad(MSE) per ottenere ∂MSE/∂θ.
- Importante: jax.grad differenzia rispetto al primo argomento, mantenere l’ordine corretto (θ come primo argomento).
## [02:05] Aggiornamento del gradiente
- Su un mini-batch: gradients = grad\_MSE(θ, X\_batch, Y\_batch).
- Aggiornare: θ ← θ − learning\_rate · gradients.
- È la discesa lungo la direzione opposta al gradiente.
## [02:30] SGD – differenze e motivazioni
- Mini-batch al posto del full-batch:
  - Meno costoso per iterazione,
  - Introduce rumore utile per generalizzazione.
- x e y sono necessari per calcolare loss e gradiente, anche se si differenzia rispetto a θ.
## [03:10] Ciclo di addestramento: epoche e mini-batch
- Permutare il dataset a ogni epoca (random key di JAX).
- Estrarre mini-batch con gli indici permutati.
- Aggiornare θ per ciascun batch.
## [04:10] Random key in JAX
- Gestire manualmente lo stato del generatore.
- Nuove permutazioni a ogni epoca migliorano la qualità media dei gradienti.
## [05:00] Parametri di training e effetti
- θ iniziale = [0, 0].
- Parametri: learning\_rate, max\_epochs, batch\_size.
- Trade-off del batch\_size:
  - Grande: gradiente più accurato, aggiornamenti più costosi.
  - Piccolo: gradiente più rumoroso, serve LR minore e più epoche.
## [06:10] Visualizzazione del modello
- Scatter dei dati di train/test.
- Tracciare la retta ŷ = θ0 + θ1 · x su un asse x continuo.
- Verifica visiva della bontà dell’adattamento.
## [00:00] Risultati e metrica sul test set
- Il modello è addestrato sul training (punti blu).
- Valutazione: calcolare la metrica sul test (punti arancioni), ad es. MSE(θ\_opt, X\_test, Y\_test) = 0.75.
- Tracciare la linea di regressione e interpretare il risultato.
- Regola fondamentale: le metriche si calcolano sempre sul test set, non sul training.
## [01:00] Verso modelli più complessi: SVR e SVM
- Dalla regressione lineare passiamo a:
  1) Support Vector Regression (SVR) con loss “tubolare” (epsilon-insensitive),
  2) SVM di classificazione lineare in 2D con un iperpiano separatore.
- L’idea generale resta: dati X e Y, definire loss, gradiente e aggiornamento.
## [02:00] Implementazione SVR: strumenti e organizzazione
- Librerie: JAX, NumPy, Matplotlib, train\_test\_split.
- Struttura in una classe “SupportVectorRegression” per gestire parametri, loss, training e predizione.
## [03:00] Loss epsilon-insensitive e regolarizzazione
Concetti:
- Loss tubolare: errori entro ±epsilon non sono penalizzati; oltre epsilon, la loss cresce.
- Regolarizzazione L2: λ · ||W||² per controllare la grandezza dei parametri ed evitare overfitting.
Formula operativa:
- ŷ = X · w\_slope + b.
- epsilon-loss per campione: max(0, |ŷ − y| − epsilon).
- Loss totale: mean(epsilon-loss) + λ · sum(params²).
## [04:30] Training SVR: inizializzazione, gradiente e update
- X come matrice (righe = campioni, colonne = feature).
- Inizializzazione: zeri, dimensione = n\_feature + 1 (bias).
- grad = jax.grad(loss).
- Step SGD con JIT: w ← w − lr · grad(w, X, Y).
- Loop su maxiter epoche, aggiornando i parametri ad ogni step.
## [06:00] Predizione SVR
- Per dati X e parametri w:
  - ŷ = X · w\_slope + b.
- Struttura identica al caso lineare, con loss diversa (epsilon-insensitive).
## [07:00] Dati, split e visualizzazione SVR
- Generare dati sintetici lineari con rumore e suddividere 80/20 train/test.
- Nota JAX: linspace produce un vettore 1D; fare reshape a una colonna (N × 1).
- Visualizzazione:
  - Linea rossa = modello,
  - Banda rosa = “tubo” ±epsilon (loss zero),
  - Punti blu = training,
  - Croci verdi = test.
- Valutazione: loss tubolare minimizzata sui blu; MSE calcolata sui verdi.
## [09:00] SVM di classificazione (hinge loss)
Obiettivo: classificazione binaria con separatore lineare.
Differenze:
- Y categoriale (tipicamente −1/+1).
- X 2D: due feature per campione.
- Loss hinge: max(0, 1 − y · decision), con decision = X · w + b.
- Regolarizzazione L2 come nella SVR.
## [10:30] Visualizzazione dati di classificazione
- Scatter:
  - x[:, 0] sull’asse X,
  - x[:, 1] sull’asse Y,
  - colore c = y per classi 0/1.
- In 2D è intuitivo esaminare la separazione lineare.
## [11:00] Esercizio SVM: implementazione
- Copiare la struttura SVR e adattare:
  - la loss (hinge),
  - le dimensioni (due feature).
- Accuratezza attesa > 90% con parametri adeguati.
## [12:00] jnp.maximum elemento per elemento
- jnp.maximum applica l’operazione per elemento (con broadcasting).
- Esempio concettuale: per ogni componente, prende max(0, valore).
## [13:00] Soluzione SVM: decision, hinge loss, training
- Decision: decision = X · w\_slope + b (eventuale reshape di w\_slope per coerenza dimensionale).
- Hinge loss: mean(max(0, 1 − y · decision)) + λ · sum(params²).
- Training:
  - n\_feature = 2,
  - parametri iniziali a zero (w1, w2, b),
  - grad = jax.grad(loss),
  - step JIT: w ← w − lr · grad(w, X, Y),
  - loop su maxiter.
Purezza e JIT:
- Con jax.jit, le funzioni devono essere pure rispetto ai parametri aggiornati.
- Passare esplicitamente w alla funzione step; evitare di catturare self.w per non confondere il caching di JAX.
## [15:00] Predizione SVM
- Predizione binaria: y\_pred = sign(decision).
- Se serve mappare a 0/1: −1 → 0, +1 → 1.
## [16:00] Valutazione, risultati e debugging
Pipeline:
- Split train/test, training, parametri appresi, valutazione sul test.
- Consigli:
  - Verificare la loss (hinge),
  - Controllare forme di X e dei parametri (reshape),
  - Assicurare purezza delle funzioni JIT,
  - Tuning di learning rate ed epoche.
- Accuratezze tipiche ~95% se tutto è coerente.
## [17:00] Frontiera di decisione: meshgrid e contour
- Tecnica: meshgrid per coprire lo spazio, contour/contourf per linee di livello e regioni.
- Visualizzazione:
  - punti cerchi = training,
  - croci = test.
- Interpretazione: la frontiera separa le classi; qualche errore è atteso ma la maggior parte dei punti risulta corretta.
## [18:00] Conclusione
- Abbiamo visto:
  - SVR con loss epsilon-insensitive e regolarizzazione L2,
  - SVM di classificazione con hinge loss e regolarizzazione,
  - Importanza delle forme (matrici vs vettori) e della purezza con JIT in JAX,
  - Visualizzazione della frontiera di separazione e valutazione su test.
- I concetti presentati sono fondamentali e riutilizzabili in modelli più complessi.