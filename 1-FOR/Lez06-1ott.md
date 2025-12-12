## Formulazione Machine Learning: Gestione Sistemi Non Fattibili
### Contesto Problema: Sistemi Disuguaglianze Non Fattibili
Nel machine learning, conoscenza e dati possono essere modellati come sistema di disuguaglianze. Tuttavia, a causa di incertezza o imprecisione dei dati, questi sistemi sono spesso non fattibili, significando che nessuna soluzione `x` può soddisfare simultaneamente tutti i vincoli. L'obiettivo è modificare le disuguaglianze per rendere il sistema fattibile.

### Metodo 1: Minimizzare la Somma delle Perturbazioni
Questo approccio comporta perturbare i lati destri delle disuguaglianze per renderle meno stringenti.

#### Dettagli Formulazione
*   **Insiemi:**
    *   `I`: Insieme di `m` disuguaglianze.
    *   `V`: Insieme di `n` variabili.
*   **Parametri:**
    *   `aij`: Matrice coefficienti.
    *   `bi`: Lato destro per il primo insieme di disuguaglianze.
    *   `lj`: Limite inferiore per le variabili.
*   **Variabili:**
    *   `xj`: Variabili decisionali originali.
    *   `si`: Perturbazione non negativa (slack) per disuguaglianza `i`.
    *   `tj`: Perturbazione non negativa (slack) per variabile `j`.
*   **Vincoli Perturbati:**
    *   `Σ(aij * xj) ≤ bi + si` per ogni `i` in `I`. Aggiungere `si` (dove `si ≥ 0`) innalza il lato destro, indebolendo il vincolo.
    *   `xj ≥ lj - tj` per ogni `j` in `V`. Sottrarre `tj` (dove `tj ≥ 0`) abbassa il lato destro, indebolendo il vincolo.
*   **Funzione Obiettivo:**
    *   Minimizzare l'impatto totale delle perturbazioni: `minimizza Σ(si) + Σ(tj)`.

#### Svantaggio
Questo metodo può risultare nel perturbare un gran numero di disuguaglianze di piccole quantità, che può alterare significativamente l'integrità del modello originale.

### Metodo 2 (Variante): Minimizzare il Numero di Disuguaglianze Trascurate
Questa variante si concentra sul prendere una decisione binaria per ogni disuguaglianza: mantenerla com'è o scartarla interamente.

#### Dettagli Formulazione
*   **Nuove Variabili:**
    *   `s'i`: Variabile binaria (1 se disuguaglianza `i` è trascurata, 0 altrimenti).
    *   `t'j`: Variabile binaria (1 se limite `j` è trascurato, 0 altrimenti).
*   **Vincoli Modificati (Metodo "Big M"):**
    *   `Σ(aij * xj) ≤ bi + M * s'i`. Quando `s'i = 1`, la costante grande `M` rende il lato destro effettivamente infinito, rendendo il vincolo non vincolante.
    *   `xj ≥ lj - M * t'j`. Quando `t'j = 1`, il limite inferiore diventa effettivamente infinito negativo.
*   **Funzione Obiettivo:**
    *   Minimizzare il conteggio dei vincoli trascurati: `minimizza Σ(s'i) + Σ(t'j)`.

### Migliori Pratiche Formulazione
*   **Mantenere Linearità:** Il metodo "Big M" è preferito rispetto al moltiplicare il lato sinistro di un vincolo per una variabile binaria, poiché quest'ultimo introdurrebbe non-linearità, rendendo il problema molto più difficile per i risolutori. La regola d'oro è mantenere le formulazioni lineari quando possibile.
*   **Applicazione:** Questa tecnica di trascurare selettivamente vincoli è altamente utile in altri domini, come problemi di pianificazione.

## Introduzione alla Teoria dei Grafi e Flussi di Rete
### Concetti Centrali e Terminologia
*   **Modellazione con Grafi:** Problemi del mondo reale possono essere modellati rappresentando locazioni chiave o entità come **nodi** (o vertici) e le connessioni tra essi come **archi** (o spigoli). Attributi come costo o tempo possono essere assegnati agli archi. Una volta modellato, il problema è risolto sul grafo astratto.
*   **Esempio:** Per trovare un itinerario in una città, stazioni ferroviarie e incroci sono nodi, strade sono archi, e tempi di percorrenza sono attributi degli archi (costi).

### L'Origine della Teoria dei Grafi: I Sette Ponti di Königsberg
*   **Il Problema:** Nella città di Königsberg del XVIII secolo, fu posta la domanda: È possibile fare una passeggiata che attraversi ognuno dei suoi sette ponti esattamente una volta e ritorni al punto di partenza?
*   **Modello Astratto e Soluzione di Eulero:**
    *   Il matematico Leonhard Euler modellò il problema rappresentando le quattro masse terrestri come nodi e i sette ponti come archi.
    *   Dimostrò che era impossibile. Il suo ragionamento si basava sul **grado** di un nodo (il numero di archi connessi ad esso).
    *   Per entrare e poi uscire da un nodo, richiede una coppia di archi. Per un tour completo che inizia e finisce nello stesso punto, ogni nodo deve avere un grado pari.
    *   Poiché i nodi nel grafo di Königsberg avevano gradi dispari, nessun tale percorso (un **ciclo Euleriano**) poteva esistere.

### Definizioni Chiave dei Grafi
#### Percorsi e Cicli
*   **Percorso:** Una sequenza di archi consecutivi.
*   **Percorso Semplice:** Un percorso senza archi ripetuti.
*   **Percorso Elementare:** Un percorso senza nodi ripetuti (e quindi senza archi ripetuti). Questa è la definizione predefinita di "percorso" per questo corso.
*   **Ciclo:** Un percorso che inizia e finisce nello stesso nodo.
*   **Ciclo Hamiltoniano:** Un ciclo elementare che visita ogni nodo nel grafo esattamente una volta.
*   **Ciclo Euleriano:** Un ciclo che attraversa ogni arco nel grafo esattamente una volta.

#### Tipi di Grafi
*   **Grafo Non Diretto:** Gli archi non hanno direzione; la connessione tra nodi `i` e `j` è simmetrica.
*   **Grafo Diretto:** Gli archi hanno una direzione specifica; un percorso da `i` a `j` non implica un percorso da `j` a `i`.

## Problema Design di Rete: Le 13 Isole (Albero di Copertura Minimo)
Questo problema comporta trovare un Albero di Copertura Minimo (MST).

### Dichiarazione Problema
*   **Contesto:** Una città costruita su 13 isole (nodi) ha tutti i suoi ponti di collegamento (archi) distrutti da uno tsunami.
*   **Obiettivo:** Ricostruire un sottoinsieme di ponti per assicurare che sia possibile viaggiare tra due isole qualsiasi.
*   **Scopo:** Raggiungere piena connettività con il minimo costo totale di ricostruzione possibile. Il costo di ricostruzione di ogni ponte potenziale è dato.

### Caratteristiche Soluzione
*   **Numero di Archi:** Perché un grafo con `n` nodi sia connesso senza ridondanza, deve contenere esattamente `n-1` archi. Per 13 isole, questo significa che devono essere costruiti 12 ponti.
    *   Meno di 12 archi risulta in un grafo disconnesso.
    *   Più di 12 archi introduce ridondanza (cicli) e costo non necessario.
*   **Costo Ottimale:** Il costo totale minimo per connettere tutte le 13 isole è **53**.
*   **Esempio Selezione Archi:** Durante il processo di soluzione, archi con peso 6 furono considerati. Tre tali archi (H-J, J-K, e L-K) furono aggiunti, contribuendo 18 al costo totale. Un altro arco di peso 6 fu scartato perché aggiungerlo avrebbe creato un ciclo.

### Strategie Soluzione (Algoritmi)
Due strategie primarie furono identificate per trovare l'insieme ottimale di ponti.

#### Strategia 1: Crescita Basata su Nodi (Algoritmo di Prim)
1.  Iniziare con un nodo arbitrario singolo (o un piccolo cluster di nodi connessi).
2.  Identificare ripetutamente l'arco più economico che connette un nodo dentro il cluster corrente a un nodo fuori dal cluster.
3.  Aggiungere questo arco e il nuovo nodo al cluster.
4.  Continuare finché tutti i 12 archi sono selezionati e tutti i 13 nodi sono connessi.

#### Strategia 2: Selezione Basata su Costi (Algoritmo di Kruskal)
1.  Ordinare tutti gli archi possibili in ordine crescente del loro costo.
2.  Iterare attraverso la lista ordinata di archi.
3.  Per ogni arco, aggiungerlo alla soluzione **se e solo se** connette due componenti precedentemente non connessi (cioè, non forma un ciclo con gli archi già selezionati).
4.  Fermarsi una volta che `n-1` (12) archi sono stati aggiunti.

### Intuizione Chiave: Evitare Cicli
Il principio centrale in entrambe le strategie di successo è l'evitamento dei cicli. Aggiungere un arco che crea un ciclo è ridondante perché i nodi che connette sono già raggiungibili l'uno dall'altro. Nella strategia basata su costi (Kruskal), un arco è scartato se forma un ciclo perché gli archi che già formano un percorso tra i suoi estremi sono garantiti essere più economici (o di costo uguale).

## Proprietà Centrali degli Alberi di Copertura Minimi
### Aggiungere un Arco a un MST
*   **Risultato:** Aggiungere un arco esterno (es. C-B) a un MST completato crea un **ciclo unico**.
*   **Ragionamento:** Un MST, per definizione, contiene già esattamente un percorso tra due nodi qualsiasi. Aggiungere un nuovo arco tra due nodi crea un secondo percorso alternativo, formando così un singolo ciclo.

### La Proprietà del Ciclo
*   **Regola:** Il costo di un arco esterno aggiunto a un MST è **maggiore o uguale** al costo di qualsiasi altro arco dentro il ciclo che crea.
*   **Ragionamento:** Durante la costruzione MST (usando un algoritmo come Kruskal), l'arco esterno fu precedentemente scartato perché un percorso che connetteva i suoi estremi esisteva già. Poiché gli archi sono processati in ordine crescente di costo, tutti gli archi in quel percorso esistente devono avere un costo minore o uguale all'arco scartato.

### Rimuovere un Arco da un MST
*   **Risultato:** Rimuovere qualsiasi arco singolo (es. F-L) da un MST completato divide il grafo in **due componenti disconnessi** (due sottoinsiemi di nodi irraggiungibili).
*   **Ragionamento:** L'MST è una soluzione minimale per la connettività. Rimuovere qualsiasi sua parte rompe la connessione, risultando in due gruppi separati di nodi senza percorso tra loro. Il confine tra questi due componenti è chiamato "taglio".

### La Proprietà del Taglio
*   **Regola:** Il costo di un arco dentro un MST (es. F-L) è **minore o uguale** al costo di qualsiasi altro arco che attraversa lo stesso taglio.
*   **Ragionamento:** Questa proprietà è fondamentale per algoritmi come Prim. L'arco fu scelto per l'MST precisamente perché era il collegamento a costo minimo che connetteva i suoi due estremi (e i componenti a cui appartengono) su tutti gli altri archi di collegamento possibili.

## Formulazione Matematica del Problema MST
### Definizione Problema
*   Il problema è formalmente conosciuto come problema **Albero di Copertura Minimo (MST)**.
    *   **Copertura:** L'albero deve toccare tutti i nodi nel grafo.
    *   **Minimo:** La somma dei costi degli archi selezionati è minimizzata.
    *   **Albero:** C'è un percorso unico tra due nodi qualsiasi.

### Componenti Formulazione
*   **Insiemi:**
    *   `N`: L'insieme di tutti i nodi.
    *   `A`: L'insieme di tutti gli archi disponibili.
*   **Parametri:**
    *   `wij`: Il peso o costo associato con ogni arco `(i, j)`.
*   **Variabili:**
    *   `xij`: Una variabile binaria che è `1` se arco `(i, j)` è selezionato per l'albero, e `0` altrimenti.

### Funzione Obiettivo
*   L'obiettivo è minimizzare il costo totale degli archi selezionati.
*   **Formula:** `Minimizza Σ (wij * xij)` per tutti gli archi `(i, j)` nell'insieme `A`.

### Vincoli e Connettività
*   **Idea Iniziale Difettosa:** Richiedere almeno un arco per uscire da ogni nodo (`Σ xij ≥ 1`) è insufficiente, poiché permette la creazione di componenti disconnessi multipli.
*   **Formulazione Corretta (Formulazione Cut-set):** Per assicurare piena connettività, per qualsiasi sottoinsieme proprio e non vuoto di nodi `S`, almeno un arco selezionato deve connettere un nodo in `S` a un nodo fuori da `S`.
*   **Formula:** Per ogni sottoinsieme `S` (dove `S` è un sottoinsieme proprio di `N` e non vuoto), deve valere: `Σ xij ≥ 1` per tutti gli archi dove `i ∈ S` e `j ∈ (N-S)`.

### Analisi Critica della Formulazione
*   **Svantaggio:** Il numero di vincoli richiesti da questa formulazione è esponenziale (approssimativamente `2^n`, dove `n` è il numero di nodi).
*   **Paradosso:** Mentre la formulazione matematica è computazionalmente massiva (esponenziale), il problema MST stesso è considerato "facile" e può essere risolto efficientemente da algoritmi greedy (es. Kruskal, Prim) in tempo polinomiale.

## 📅 Prossimi Accordi
*   [ ] Leggere Sezione 2 del Capitolo 2 nelle note di lezione per notazione completa grafi e metodi di rappresentazione.
*   [ ] Prepararsi per l'attività di valutazione formativa (Kahoot) sulla rappresentazione grafi, programmata per la prossima settimana.
*   [ ] La sessione successiva continuerà dalla formulazione matematica dell'MST, ricapitolerà le sue proprietà, ed estenderà il problema al Problema del Commesso Viaggiatore (TSP).
*   [ ] Un breve video con la dimostrazione per il secondo algoritmo (Kruskal/Basato su Costi) sarà pubblicato online.
*   [ ] (Opzionale) Guardare il video raccomandato sull'ottimizzazione grafi.
*   [ ] Rivedere il concetto di trascurare selettivamente vincoli per applicazioni future, particolarmente in problemi di pianificazione.
*   [ ] Gli studenti possono scrivere le loro osservazioni nel loro "libro di viaggio".
*   [ ] (Per studenti con preoccupazioni privacy) Inviare email all'istruttore per far rimuovere il proprio nome dalla tabella punti bonus pubblicata.