Data e ora: 2025-12-11 12:59:48
Luogo: [Inserisci luogo]: Aula 312
Corso: [Inserisci nome corso]: [Inserisci nome corso]
## Panoramica
Le lezioni hanno trattato il processo meccanico di formulazione dei problemi duali a partire dai problemi primali utilizzando una tabella di corrispondenza, includendo un esempio dettagliato passo-passo. Il corso è poi passato ad applicazioni pratiche ed esercizi relativi ai problemi di flusso massimo/taglio minimo, come la ricostruzione di un flusso dato un taglio e la modellazione di scenari reali come pattugliamenti di polizia e pianificazione di agenti. È stata mostrata una tecnica chiave di modellazione, il path covering tramite split dei nodi, come problema di flusso a costo minimo.
## Contenuti rimanenti
1. Completare l'esercizio sulla scrittura del duale per il problema del cammino minimo.
2. Completare l'esercizio sulla scrittura del duale per il problema del flusso massimo.
3. Completare l'esercizio di formulazione del problema duale (per le variabili x2 e oltre).
## Contenuti trattati
### 1. Formulazione dei problemi duali
- Rivisti i casi simmetrici e asimmetrici di problemi duali e introdotti due metodi per formulare un duale: trasformare il primale in forma standard o usare una tabella di corrispondenza.
- Spiegata in dettaglio la tabella di corrispondenza, mostrando come gli elementi di un problema primale (min/max, variabili, vincoli, coefficienti) si mappano sugli elementi del duale.
- Fornita una guida passo-passo all'uso della tabella per costruire il duale di un problema di minimizzazione con vincoli misti. Questo includeva:
    - Associare una variabile duale a ciascun vincolo primale.
    - Costruire la funzione obiettivo duale a partire dai termini noti del primale.
    - Derivare i vincoli duali dalle colonne (variabili primali).
    - Determinare il tipo di vincolo duale (≤, =, ≥) in base alle restrizioni di segno delle variabili primali.
    - Determinare le restrizioni di segno delle variabili duali in base al tipo di vincolo primale.
- Proposto un esercizio agli studenti: scrivere i duali dei problemi del cammino minimo e del flusso massimo.
> **Suggerimenti AI**
> La tua spiegazione della tabella di corrispondenza è stata molto metodica e chiara. Il processo passo-passo per costruire il duale è stato efficace. Nel determinare il segno delle variabili duali, ti sei momentaneamente confuso ma ti sei ripreso rapidamente; potrebbe essere utile fermarsi e indicare direttamente la riga corretta sulla tabella per assicurarsi che gli studenti seguano. Quando prosegui l'argomento, considera di presentare il problema primale in formato matrice o tabella per facilitare la lettura delle colonne e la formazione dei vincoli duali.
### 2. Modellazione con flusso: pianificazione agenti/giocatori e path covering
- Descritto un problema di pianificazione: minimizzare il numero di agenti necessari per visitare un insieme di giocatori, dove ciascun giocatore è disponibile in un giorno specifico e lo spostamento tra giocatori richiede tempo. Questo è stato inquadrato come un problema generale di path covering.
- Il problema non può essere risolto con il flusso massimo standard poiché non garantisce che tutti i nodi (lavori) siano coperti.
- La tecnica chiave è trasformarlo in un problema di flusso a costo minimo risolvibile, splittando ogni nodo lavoro 'i' in due nodi, 'i' (inizio) e 'i-primo' (fine).
- Si crea un "grafo di compatibilità" dove esiste un arco da un nodo primo (fine lavoro i) a un nodo non primo (inizio lavoro j) se un singolo agente può gestire entrambi in sequenza: `giorno_visita(i) + tempo_viaggio(i, j) <= giorno_visita(j)`.
- La rete di flusso è strutturata per forzare un flusso esattamente pari a 1 su ciascun arco interno del lavoro (da i a i-primo), garantendo che ogni lavoro sia coperto.
- Per minimizzare il numero di agenti (percorsi), si aggiunge un arco artificiale dal pozzo T alla sorgente S. L'obiettivo è minimizzare il flusso su questo arco T-S, che corrisponde al numero totale di percorsi utilizzati. Tutti gli altri costi sono zero.
- È stato usato un esempio reale di Italgas, evidenziando come ridurre un problema a un problema di flusso sia più efficiente che modellarlo erroneamente come un problema più complesso come il TSP.
> **Suggerimenti AI**
> La spiegazione della tecnica di split dei nodi è stata molto chiara e metodica. L'esempio reale di Italgas è stato un ottimo modo per contestualizzare l'importanza della scelta del modello giusto. Per la prossima lezione, sarebbe utile iniziare ricapitolando rapidamente la definizione di grafo di compatibilità e la condizione `d_i + c_ij <= d_j` prima di spiegare come si usa il flusso su questa struttura.
### 3. Ricostruzione del flusso da un taglio minimo
- Presentato un problema di "reverse engineering": dato un taglio di capacità minima, ricostruire un flusso ammissibile che lo realizzi.
- Stabilita la proprietà chiave: gli archi forward (da lato S a lato T) devono essere saturi (flusso = capacità), e gli archi backward devono avere flusso nullo.
- Mostrato come calcolare il flusso richiesto in ingresso o uscita dai nodi a cavallo del taglio in base ai flussi saturi sugli archi.
- Spiegati due metodi per determinare il flusso nei sottografi lato S e lato T: per ispezione nei grafi semplici, oppure modellando ciascun lato come un problema ausiliario di flusso massimo per verificare se il flusso richiesto può essere instradato.
- Chiarito che se non si trova un flusso ammissibile per uno dei due sottografi, il taglio dato non era quello di capacità minima.
> **Suggerimenti AI**
> Questa è stata un'ottima applicazione pratica del teorema max-flow min-cut. Il concetto di "reverse engineering" è coinvolgente. L'uso di un problema ausiliario di flusso massimo è una tecnica potente, ma la spiegazione è stata un po' rapida. Quando lo introduci, potresti esplicitare l'obiettivo: "Dobbiamo verificare se la rete lato S ha abbastanza capacità per fornire X unità al taglio. Possiamo testarlo creando un nuovo problema temporaneo di flusso massimo."
### 4. Modellazione con max-flow/min-cut: il problema delle pattuglie di polizia
- Introdotto un problema reale: minimizzare il numero di pattuglie di polizia necessarie per intercettare tutto il traffico da una sorgente (S) a una destinazione (T).
- Modellato il problema come la ricerca di un taglio minimo nel grafo della rete stradale, dove ogni arco ha capacità 1 (rappresenta una pattuglia).
- Aggiunta una complicazione: le pattuglie devono essere ad almeno una distanza 'k' dalla sorgente 'S'.
- Proposte due soluzioni per gestire il vincolo di distanza:
    1.  **Modifica delle capacità:** Prima si risolve un problema di cammino minimo per trovare la distanza da S a tutti i nodi. Per ogni arco che porta a un nodo "troppo vicino" (distanza < k), si imposta la capacità a infinito, rendendo proibitivo includerlo in un taglio minimo.
    2.  **Modifica del grafo:** Si uniscono tutti i nodi entro distanza 'k' dalla sorgente in un unico "super-sorgente" e si cerca il taglio minimo nel grafo modificato.
> **Suggerimenti AI**
> Questo è stato un ottimo esempio di come combinare algoritmi diversi (cammino minimo e flusso massimo). La spiegazione delle due soluzioni alternative è stata chiara. Per renderlo ancora più concreto, potresti accennare brevemente ai pro e contro, ad esempio: "Modificare le capacità mantiene intatta la struttura del grafo, mentre unire i nodi semplifica il grafo su cui si esegue l'algoritmo finale."
## Domande degli studenti
1. **Cosa succede se non si riesce a trovare un flusso ammissibile in una delle due parti (NS o NT) ricostruendo da un taglio?**
   - Se non si trova un flusso ammissibile, significa che in una delle due parti (NS o NT) non c'è abbastanza capacità. Questo implica che il taglio fornito non è quello di capacità minima e il vero taglio minimo si trova altrove nel grafo.
2. **[Riguardo l'arco backward in un taglio minimo]**
   - È irrilevante perché va dal lato T (NT) al lato S (NS). Sappiamo che affinché il flusso che attraversa il taglio sia uguale alla capacità del taglio, il flusso sugli archi backward deve essere nullo. Questa proprietà vale solo per un taglio di capacità minima.
3. **[Riguardo il problema delle pattuglie di polizia] Quindi si usa lo stesso grafo per due problemi diversi?**
   - Esattamente. Si usa la stessa struttura di grafo ma per due problemi diversi. Prima si considerano le lunghezze/ distanze degli archi per risolvere un problema di cammino minimo. Si prendono le etichette (distanze) da quella soluzione. Poi, usando quelle etichette, si modificano le capacità degli archi per risolvere il secondo problema, cioè il max-flow/min-cut per trovare le posizioni delle pattuglie.
4. **[Riguardo il path covering] Perché non minimizzare direttamente il flusso sugli archi uscenti da S?**
   - Il numero di percorsi è una variabile, quindi non si saprebbe che valore assegnare all'offerta in S e alla domanda in T. Per gestire questa incognita, si introduce l'arco artificiale da T a S. Il flusso su questo arco rappresenta il numero totale di percorsi, e minimizzarlo diventa l'obiettivo. Questo trucco mantiene il bilancio di flusso in tutti i nodi originali pari a zero, mentre il flusso obbligatorio 1 sugli archi interni forza i percorsi a visitare ogni nodo.