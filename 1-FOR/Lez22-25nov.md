Data e ora: 2025-12-11 12:21:55
Luogo: [Luogo]
Corso: [Nome corso]
## Panoramica
La lezione ha introdotto i vincoli di integrità nell’ottimizzazione, evidenziandone il potere modellistico e le implicazioni teoriche; ha mostrato la modellazione disgiuntiva (big-M) per unioni di regioni; ha formulato un problema di schedulazione tipo flow-shop con variabili binarie di precedenza per imporre la capacità delle macchine; ha linearizzato un obiettivo min–max (makespan); e ha illustrato come modellare funzioni di costo spezzate lineari usando variabili di segmento e attivazione binaria per imporre l’uso sequenziale.
## Contenuti rimanenti
1. Esplorare cosa succede ignorando i vincoli di integrità (risultati del rilassamento continuo) — previsto per domani
2. Questioni di efficienza e scelta/taratura corretta delle costanti big-M — previsto per domani
3. Altre applicazioni dei vincoli disgiuntivi — menzionate ma rimandate
4. Strumenti/dimostrazioni di ottimalità per problemi interi (oltre la teoria LP) — riconosciuti come non direttamente applicabili e non trattati
5. Arricchimento della cassetta degli attrezzi modellistica oltre gli esempi visti — previsto per lezioni future
## Contenuti trattati
### 1. Motivazione e implicazioni dei vincoli di integrità nei problemi lineari
- I vincoli di integrità restringono l’insieme ammissibile, spesso da infinito a finito.
- Vantaggio modellistico: permettono di rappresentare fenomeni non esprimibili con soli vincoli lineari (continui), es. oggetti indivisibili come porte/finestre.
- Svantaggio teorico: la teoria LP basata su convessità (dualità, slackness complementare, ottimalità dei vertici) non vale più con l’integrità.
- L’insieme ammissibile intero corrisponde ai punti reticolari interi nel poliedro LP; a volte un vertice LP è già intero (caso fortunato), ma piccole variazioni nei dati possono portare a soluzioni LP frazionarie.
- Bisogna sfruttare la conoscenza LP dove possibile ma inventare nuovi metodi per i problemi interi; la enumerazione esaustiva è un punto di partenza ma inefficiente.
### 2. Vincoli disgiuntivi e modellazione big-M per unioni di regioni convesse
- Obiettivo: modellare una regione ammissibile non convessa come unione di insiemi convessi (due triangoli Q e R).
- L’intersezione dei vincoli dà una regione più piccola e sbagliata; l’unione richiede di attivare un insieme di vincoli e disattivare l’altro.
- Identificare i vincoli condivisi tra i due triangoli (es. x2 ≥ 0, x1 ≥ 0 o x2 ≤ 5).
- Introdurre uno switch binario y per selezionare quale vincolo “diagonale” è attivo:
  - x1 − x2 ≤ 0 attivo se y = 1; x1 + x2 ≤ 5 rilassato tramite + M·y.
  - Se y = 0, l’altro diagonale è attivo e il primo rilassato tramite + M·(1 − y) o formulazione equivalente.
- Il big-M sposta i vincoli per “spegnerli”; M deve essere sufficientemente grande da rendere il vincolo rilassato non vincolante per tutti i punti rilevanti.
- La tecnica mostra la modellazione di vincoli “o–o”; attenzione che M troppo grande peggiora le prestazioni del risolutore.
### 3. Schedulazione su pipeline di M macchine: formulazione time-index con switch di precedenza
- Problema: N lavori processati in sequenza su M macchine in ordine; ogni lavoro j ha tempi di lavorazione p_{j,i} sulla macchina i.
- Vincoli:
  - Flusso/pipeline: l’inizio su macchina i non può essere prima del completamento su macchina i−1 per lo stesso lavoro: T_{i,j} ≥ T_{i−1,j} + p_{j,i−1}, per i = 2,…,M.
  - Capacità macchina: al più un lavoro per volta; per ogni coppia di lavori (j,k) su macchina i, o j precede k o viceversa:
    - Se j precede k: T_{i,j} + p_{j,i} ≤ T_{i,k}.
    - Se k precede j: T_{i,k} + p_{k,i} ≤ T_{i,j}.
- Vincoli disgiuntivi tramite variabile binaria Y_{i,j,k}:
  - Si usa big-M per rilassare la disuguaglianza di precedenza non selezionata, attivando solo una per coppia.
  - Interpretazione: Y_{i,j,k} = 1 se j precede k su macchina i; l’altra disuguaglianza viene rilassata con +M di conseguenza.
- Le variabili T sono continue (tempi di inizio), Y binarie (decisioni di ordinamento).
### 4. Linearizzazione dell’obiettivo makespan (min–max)
- Obiettivo: minimizzare il tempo di completamento dell’ultimo lavoro sull’ultima macchina (makespan).
- Tempo di completamento per lavoro j su macchina M: T_{M,j} + p_{j,M}.
- Linearizzazione min–max: introdurre variabile ausiliaria Z tale che Z ≥ T_{M,j} + p_{j,M} per tutti j; obiettivo è minimizzare Z.
- Così il problema min–max diventa un obiettivo lineare con vincoli lineari.
### 5. Modellazione di funzioni di costo spezzate lineari con variabili di segmento e attivazione binaria
- Esempio di funzione spezzata lineare (costi telefonici): segmenti con costi unitari diversi su intervalli.
- Rappresentare x (uso totale) come somma degli usi di segmento: x = Z1 + Z2 + Z3, dove ogni Zi è la lunghezza del segmento.
- La funzione di costo diventa lineare nelle variabili di segmento: f(x) = b0 + c1·Z1 + c2·Z2 + c3·Z3.
- Senza vincoli, l’ottimizzazione alloccherebbe sui segmenti più economici fuori ordine; serve imporre l’uso sequenziale:
  - Introdurre variabili binarie Y1, Y2, Y3 che indicano se il segmento i è usato (Zi > 0).
  - Vincoli di collegamento: 0 ≤ Zi ≤ (upper bound del segmento i)·(Y appropriata o combinazione).
  - Sequenzialità: es. Z1 ≤ 140·Y2 assicura che Z1 sia pieno prima che inizi Z2; analogamente Z2 ≤ 80·Y3 impone l’ordine su Z3.
- Idea generale: usare binarie per forzare che i segmenti successivi si attivino solo dopo che i precedenti sono pieni; riferimento a casi più generali con gap nelle note.
## Domande degli studenti
1. Il problema di schedulazione è chiaro? Puoi rispiegare il problema?
- Il docente ha ripetuto il setup: i lavori entrano nella macchina 1, proseguono fino alla M; l’obiettivo è minimizzare il tempo di uscita dell’ultimo lavoro; le decisioni riguardano ordine e tempi rispettando la sequenza e la capacità delle macchine.
2. Che variabili bisogna introdurre per il problema di schedulazione?
- Il docente ha guidato gli studenti a proporre variabili di inizio T_{i,j} (continue, ≥ 0) per ogni coppia lavoro–macchina, e poi variabili binarie di precedenza Y_{i,j,k} per imporre la capacità tramite vincoli disgiuntivi.
3. Come si impone “o j precede k o k precede j” su una macchina?
- Il docente ha introdotto vincoli disgiuntivi con una binaria Y_{i,j,k} e termini big-M per attivare una disuguaglianza e rilassare l’altra, garantendo esclusività.
4. Come si linearizza l’obiettivo min–max makespan?
- Il docente ha suggerito di introdurre una variabile ausiliaria Z con vincoli Z ≥ T_{M,j} + p_{j,M} per tutti i lavori j, poi minimizzare Z.