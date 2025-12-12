Data e ora: 2025-12-11 12:04:36
Luogo: [Inserisci luogo]: [Inserisci luogo]
Corso: [Inserisci nome corso]: [Inserisci nome corso]
## Panoramica
Questa lezione si è concentrata sul consolidamento della conoscenza della dualità nella programmazione lineare (PL) attraverso diversi esempi dettagliati e l'esplorazione della sua interpretazione geometrica. Sono stati trattati: la costruzione del duale per un problema PL generale, per il problema del cammino minimo, per il problema del flusso massimo e per un problema astratto con più insiemi di variabili. È stata spiegata anche la connessione tra il duale del flusso massimo e il problema del taglio minimo. Successivamente, è stata introdotta l'interpretazione geometrica del duale come cono finitamente generato e sono state derivate le condizioni per trovare una direzione ammissibile di miglioramento da un punto dato, ovvero la condizione di miglioramento (`c*xi > 0`) e la condizione di ammissibilità per i vincoli attivi (`ai*xi <= 0`).
## Contenuti rimanenti
1.  Una rappresentazione visiva (figura) del cono definito dalle condizioni per una direzione ammissibile crescente.
2.  Altri esempi di ricerca di direzioni ammissibili crescenti.
3.  Analisi di cosa succede quando il sistema di condizioni per una direzione ammissibile crescente non ha soluzione, e il collegamento con l'ottimalità.
4.  Come risolvere problemi di programmazione lineare usando la dualità per certificare l'ottimalità.
## Contenuti trattati
### 1. Costruzione del duale di un problema PL generale
- Il numero di variabili duali è uguale al numero di vincoli primali, e il numero di vincoli duali è uguale al numero di variabili primali.
- La funzione obiettivo del duale si ottiene dai termini noti (secondo membro) del problema primale.
- Ogni vincolo duale si costruisce raccogliendo i coefficienti di una singola variabile primale su tutti i vincoli primali.
- Il segno dei vincoli e delle variabili duali si determina tramite una tabella di conversione in base alle caratteristiche del problema primale.
- È stata accettata una correzione di uno studente riguardo una variabile mancante in un vincolo.
> **Suggerimenti AI**
> La spiegazione passo-passo della costruzione del duale è stata molto metodica e chiara. Usare la tabella come riferimento costante è un ottimo metodo didattico. Quando hai chiarito la confusione dello studente sui nomi delle variabili (X vs. Y) nella tabella, la spiegazione "guarda la colonna, non i nomi" è stata efficace. Per migliorare ancora, potresti pre-etichettare le colonne della tabella come "Problema Primale" e "Problema Duale" prima di iniziare l'esempio, così da prevenire confusione.
### 2. Dualità nel problema del cammino minimo
- Il problema primale è stato formulato come minimizzazione dei costi degli archi soggetta ai vincoli di conservazione del flusso. La matrice dei coefficienti è stata identificata come matrice di incidenza nodo-arco.
- Il duale è stato costruito associando variabili duali (potenziali, `pi`) a ciascun nodo.
- L'obiettivo duale diventa massimizzare la differenza di potenziale tra nodo di arrivo e nodo di partenza (`pi_t - pi_s`).
- I vincoli duali assumono la forma `pi_j - pi_i <= c_ij`, collegata all'analogia del "problema della corda".
- Il procedimento è stato mostrato prima su un esempio concreto e poi generalizzato.
> **Suggerimenti AI**
> Partire da un esempio concreto prima di generalizzare ha reso il tema più digeribile. Il collegamento al "problema della corda" è stato ottimo per ancorare il nuovo materiale a concetti già noti. Dopo aver derivato la forma generale del duale (`pi_j - pi_i <= c_ij`), potresti chiedere esplicitamente "A cosa vi fa pensare questo vincolo?" per stimolare il collegamento autonomo degli studenti.
### 3. Dualità nel problema del flusso massimo
- Il primale del flusso massimo è stato formulato come massimizzazione del flusso su un arco fittizio da t a s in un modello di circolazione, con vincoli di conservazione e capacità.
- Il duale è stato costruito con variabili `pi` per i nodi e `mu` per gli archi.
- L'obiettivo duale è minimizzare la somma di `u_ij * mu_ij` (capacità per la variabile duale).
- L'interpretazione: fissando `pi` a 0 o 1, il problema diventa una partizione dei nodi in due insiemi (S e T) e la minimizzazione della capacità del taglio tra essi.
- Si dimostra così che il duale del flusso massimo è il problema del taglio minimo, spiegando il teorema max-flow min-cut.
> **Suggerimenti AI**
> L'esposizione di come il duale del flusso massimo si riduca al taglio minimo è stata il punto forte della lezione. I passaggi logici con `pi_s=1` e `pi_t=0` e l'analisi dei valori di `mu_ij` sono stati molto chiari. Per renderlo ancora più visivo, potresti disegnare un grafo e colorare i nodi secondo il valore di `pi` (es. rosso per `pi=1`, blu per `pi=0`).
### 4. Costruzione del duale per un problema astratto
- Il problema coinvolgeva due insiemi di variabili (`x_ij` e `y_j`) e più gruppi di vincoli.
- Passaggio cruciale: portare tutte le variabili al primo membro dei vincoli.
- Variabili duali (`alpha_i`, `beta_j`, `gamma_ij`) introdotte per ciascun gruppo di vincoli.
- I vincoli duali si costruiscono considerando un gruppo di variabili primali alla volta (`x_ij` e `y_j`) e trovando i loro coefficienti su tutti i gruppi di vincoli.
- L'ultimo passo è determinare i segni delle variabili duali in base al tipo di vincolo primale.
> **Suggerimenti AI**
> Ottimo esercizio di sintesi per testare le abilità meccaniche degli studenti senza un contesto applicativo. L'enfasi sul "riportare tutto a sinistra" è un consiglio pratico fondamentale. La spiegazione su come gestire variabili con più indici (come `x_ij`) e la loro presenza in vincoli con indici singoli (come `alpha_i`) è stata chiara. Per fissare il metodo, potresti proporre subito un esercizio simile da svolgere in autonomia.
### 5. Interpretazione geometrica del problema duale
- L'espressione `y*A` nel duale è stata spiegata come combinazione lineare non negativa dei gradienti dei vincoli, che geometricamente genera un cono finito.
- La condizione `y*A = c` è stata interpretata come verifica che il vettore `c` (gradiente della funzione obiettivo) appartenga al cono generato dai gradienti dei vincoli (`a_i`).
- Sono stati usati esempi visivi in R2 per illustrare il concetto.
> **Suggerimenti AI**
> La spiegazione del concetto di cono è stata molto chiara, soprattutto con gli esempi in R2. Scomporre il vincolo `y*A = c` nella domanda "c appartiene al cono?" aiuta a costruire intuizione. Per rafforzare, potresti accennare al perché sia un cono *finitamente generato* (perché i vincoli sono in numero finito).
### 6. Ricerca di una direzione ammissibile di miglioramento
- L'obiettivo è verificare se una soluzione ammissibile `x_bar` può essere migliorata spostandosi verso un nuovo punto `x_prime = x_bar + lambda * xi` (con `lambda > 0`).
- Due condizioni: il nuovo punto deve migliorare la funzione obiettivo (`c*x_prime > c*x_bar`) e restare ammissibile (`A*x_prime <= b`).
- Sostituendo e semplificando, si ottiene la **condizione di miglioramento**: `c*xi > 0`, cioè la direzione `xi` deve formare un angolo < 90° con il gradiente di `c`.
- La **condizione di ammissibilità** si analizza concentrandosi sull'insieme attivo `I(x_bar)`—i vincoli in uguaglianza in `x_bar`. Per questi, la condizione si riduce a `ai*xi <= 0`.
> **Suggerimenti AI**
> Il paragone tra essere "in mezzo alla stanza" e "contro un muro" è stato molto efficace per introdurre il concetto di vincoli attivi. La derivazione è stata logica e chiara. Quando hai introdotto l'insieme attivo `I(x_bar)`, potresti esplicitare che per i vincoli non attivi si può sempre trovare un `lambda` piccolo a piacere, perciò ci si concentra solo su quelli attivi per la direzione.
### 7. Esempio: ricerca di una direzione ammissibile crescente
- I concetti sono stati applicati al problema "porte e finestre" nel punto (4, 0).
- I vincoli attivi (1 e 5) sono stati identificati sia graficamente che algebricamente.
- Le condizioni per una direzione ammissibile crescente in (4,0) sono:
    - Miglioramento: `30*xi1 + 50*xi2 > 0`
    - Ammissibilità (dai vincoli attivi): `xi1 <= 0` e `xi2 >= 0`.
- È stato assegnato come compito visualizzare il cono risultante per la prossima lezione.
> **Suggerimenti AI**
> Questo esempio pratico è stato fondamentale per fissare la teoria astratta. Mostrare sia il metodo grafico che quello algebrico per trovare i vincoli attivi è stato ottimo. Dopo aver scritto `-xi2 <= 0`, potresti subito riscriverlo come `xi2 >= 0` per rendere più intuitivo il cono finale (`xi1 <= 0`, `xi2 >= 0` e la disequazione sull'obiettivo).
## Domande degli studenti
1.  **Puoi ripetere? (Errore nel primo esempio)**
    - Sì, hai ragione. Non c'è y4. È stato un mio errore. Nella quarta equazione non c'è x3, quindi qui non c'è nulla. Grazie.
2.  **[Domanda implicita sulla regola di corrispondenza nella tabella della dualità]**
    - Il docente ha chiarito che non bisogna fissarsi sui nomi delle variabili (X vs Y) nella tabella, ma sulle colonne che rappresentano il tipo di problema (es. massimo o minimo). Il problema di partenza determina la "colonna di partenza" e il duale si trova nella "colonna di arrivo".
3.  **[Domanda implicita sull'origine della regola della variabile non negativa per un problema `max <=`]**
    - Il docente ha spiegato che è una convenzione naturale, citando il "problema del venditore di pillole" dove il `max` con vincoli `<=` porta a variabili duali (prezzi) non negative. Ha anche detto che questa tabella sarà fornita all'esame.
4.  **Domande? Siete convinti? Se avete dubbi, fermatemi pure.**
    - [Nessuna risposta udibile dagli studenti]
5.  **Domande sulla scrittura del duale?**
    - [Nessuna risposta udibile dagli studenti]