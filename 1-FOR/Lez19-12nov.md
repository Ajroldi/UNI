Data e ora: 2025-12-11 12:09:08
Luogo: [Inserisci luogo]: [Inserisci luogo]
Corso: [Inserisci nome corso]: [Inserisci nome corso]
## Panoramica
La lezione ha rivisto le condizioni per trovare una direzione ammissibile crescente in un problema di programmazione lineare e ha mostrato come certificare l’ottimalità. Concetti chiave: identificazione dei vincoli attivi, rappresentazione grafica delle condizioni di miglioramento e uso del problema duale come certificato di ottimalità. La lezione è poi passata al passo algoritmico per determinare la massima ampiezza ammissibile (lambda) lungo una direzione di miglioramento, controllando i vincoli non attivi, gettando le basi per l’algoritmo del Simplesso.
## Contenuti rimanenti
1. Come determinare una direzione efficiente tra tutte le direzioni ammissibili.
2. Come determinare una soluzione ammissibile di partenza per un sistema di disequazioni.
3. Implementazione dettagliata dell’algoritmo risolutivo.
4. Uso dell’algoritmo per determinare una soluzione ammissibile di partenza.
5. Altre proprietà importanti per l’analisi post-ottimalità.
6. Revisione degli esercizi per la prossima settimana.
## Contenuti trattati
### 1. Condizioni per una direzione ammissibile crescente
- Una direzione ammissibile crescente (ψ) da un punto ammissibile (x-bar) deve soddisfare due condizioni:
    - **Condizione di crescita:** Il prodotto scalare tra il gradiente della funzione obiettivo (c) e la direzione (ψ) deve essere maggiore di zero (cψ > 0). Questa condizione è indipendente dal punto x-bar.
    - **Condizione di ammissibilità:** Per tutti i vincoli attivi (i ∈ I(x-bar)), il prodotto scalare tra il gradiente del vincolo (ai) e la direzione (ψ) deve essere minore o uguale a zero (aiψ ≤ 0).
- Geometricamente, la condizione di crescita definisce un semispazio, quella di ammissibilità un cono. L’intersezione è l’insieme delle direzioni ammissibili crescenti.
### 2. Applicazione a punti di esempio
- Il problema "porte e finestre" è stato usato per illustrare i concetti.
- **Punti non ottimali (4, 0) e (4, 3):** Per entrambi sono stati identificati i vincoli attivi e stabilite le condizioni per una direzione ammissibile crescente. In entrambi i casi, la rappresentazione geometrica mostra intersezione non vuota tra le regioni di crescita e ammissibilità, confermando che esistono direzioni di miglioramento e i punti non sono ottimali.
- **Punto ottimo (2, 6):** Qui i vincoli attivi sono 2 e 3. La rappresentazione geometrica mostra intersezione vuota tra le regioni di crescita e ammissibilità. L’assenza di soluzione indica che non esistono direzioni ammissibili crescenti, quindi (2, 6) è soluzione ottima.
### 3. Dualità e certificato di ottimalità
- Il problema della ricerca di una direzione ammissibile crescente si può formulare come un PL "ristretto". Se esiste soluzione, questo PL è illimitato.
- Per il Lemma di Farkas, il PL ristretto ha soluzione se e solo se il suo duale ("duale ristretto") è irrealizzabile. Viceversa, se il PL ristretto non ha soluzione (cioè non esistono direzioni ammissibili crescenti), il duale ristretto ammette soluzione.
- Il duale ristretto cerca moltiplicatori non negativi (η) tali che il gradiente della funzione obiettivo (c) sia combinazione lineare non negativa dei gradienti dei vincoli attivi (Σ ηi * ai = c, ηi ≥ 0).
- Una soluzione (eta-bar) del duale ristretto si può usare per costruire una soluzione ammissibile (y-bar) per il duale completo ponendo `y_i = eta_i_bar` per i vincoli attivi e `y_i = 0` per i non attivi.
- Si è dimostrato che il valore obiettivo di questa soluzione duale (`y_bar * b`) coincide con quello primale (`c * x_bar`).
- Per il Teorema della Dualità Forte, quando una soluzione primale e una duale ammissibili hanno lo stesso valore obiettivo, entrambe sono ottime. Questo processo fornisce un certificato formale di ottimalità.
### 4. Logica del Simplesso e ampiezza del passo (lambda)
- La logica dell’algoritmo risolutivo è un processo iterativo:
    1. Si parte da un punto ammissibile (es. un vertice della regione ammissibile).
    2. Si cerca una direzione ammissibile crescente.
    3. Se non esiste, il punto corrente è ottimo.
    4. Se si trova una direzione (ψ), ci si sposta lungo di essa fino a un nuovo punto `x' = x_bar + lambda * psi`.
- La massima ampiezza `lambda > 0` si determina trovando il primo vincolo non attivo che viene raggiunto.
- Per ogni vincolo non attivo `i`, si analizza la disequazione `lambda * (a_i * psi) <= b_i - a_i * x_bar`.
    - Se `a_i * psi <= 0`, la direzione si allontana o è parallela al vincolo, quindi non è limitante.
    - Se `a_i * psi > 0`, la direzione punta verso il vincolo, imponendo un limite superiore a lambda: `lambda <= (b_i - a_i * x_bar) / (a_i * psi)`.
- Il valore finale di `lambda` è il minimo di questi limiti superiori. Così il nuovo punto `x'` resta ammissibile e si ferma sul nuovo vincolo.
- Se per tutti i vincoli non attivi `a_i * psi <= 0`, il problema è illimitato.
## Domande degli studenti
Nessuna domanda posta dagli studenti.