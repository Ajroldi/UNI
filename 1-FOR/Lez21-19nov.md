Data e ora: 2025-12-11 12:18:57
Luogo: [Inserisci luogo]: [Inserisci luogo]
Corso: [Inserisci nome corso]: [Inserisci nome corso]
## Panoramica
Queste lezioni si sono concentrate sulla risoluzione di problemi di programmazione lineare analizzando soluzioni potenziali e vertici (basi). La prima lezione ha introdotto il metodo della slackness complementare per verificare l’ottimalità di soluzioni date e per svolgere analisi post-ottimalità (column generation). La seconda lezione ha collegato questi concetti al quadro classico dei manuali sulle basi, analizzandoli sia analiticamente (ammissibilità primale/duale) sia geometricamente (intersezioni, coni e gradienti). L’obiettivo generale era mostrare che una base è ottima se e solo se è ammissibile sia per il problema primale che per quello duale.
## Contenuti trattati
### 1. Verifica dell’ottimalità delle soluzioni tramite slackness complementare
La procedura base per verificare se una soluzione è ottima consiste nel: scrivere il problema duale, scrivere le equazioni di slackness complementare, controllare l’ammissibilità primale della soluzione data, usare le variabili primali non nulle per costruire una soluzione duale complementare e infine verificare l’ammissibilità di questa soluzione duale.
- **Esempio 1 (Soluzione ottima x’):** Una soluzione x’ data è stata confermata ottima. La soluzione duale complementare y’ è stata ricavata e trovata ammissibile per il problema duale.
- **Esempio 2 (Soluzione ammissibile ma non ottima x’’):** Una seconda soluzione x’’ è risultata ammissibile per il problema primale. Tuttavia, la sua soluzione duale complementare y’’ violava un vincolo duale, dimostrando che x’’ non era ottima.
### 2. Analisi post-ottimalità e column generation
Questa sezione ha esplorato come valutare la convenienza di aggiungere nuovi prodotti (variabili) a un piano di produzione ottimale esistente.
- **Concetto:** Aggiungere una nuova variabile (colonna) al problema primale corrisponde ad aggiungere un nuovo vincolo al problema duale.
- **Metodo:** Per determinare se un nuovo prodotto è conveniente, si verifica se la soluzione duale ottima originale soddisfa il nuovo vincolo duale corrispondente a quel prodotto. Se il vincolo è violato, il nuovo prodotto è potenzialmente conveniente e va approfondito.
- **Esempio:** Per un problema di produzione con tre nuovi prodotti potenziali, la soluzione duale ottima originale è stata verificata rispetto ai tre nuovi vincoli duali. Si è visto che solo il prodotto 5 violava il suo vincolo, indicando che era l’unico da aggiungere al piano di produzione. Questa tecnica è stata introdotta come **Column Generation**.
### 3. Collegamento ai concetti classici di PL (basi e soluzioni di base)
I metodi del corso sono stati formalmente collegati alla terminologia classica della programmazione lineare, dove i vertici sono definiti come soluzioni di base.
- **Definizione:** Una base (B) è un sottoinsieme di ‘n’ indici di vincolo corrispondenti a una sottomatrice invertibile (AB) della matrice dei vincoli A.
- **Schema di analisi:**
    - **Ammissibilità primale:** Verificare se il vertice definito dalla base soddisfa tutti gli altri vincoli (non di base).
    - **Ammissibilità duale:** Verificare se la soluzione duale complementare (trovata usando la base) soddisfa tutti i vincoli duali (es. non negatività).
- **Condizione di ottimalità:** Una base è ottima se e solo se è ammissibile sia primalmente che dualmente.
### 4. Analisi geometrica e analitica delle basi
È stato svolto un esercizio per verificare diverse basi potenziali (B1, B2, B3) per un dato problema di PL, combinando visualizzazione geometrica e calcoli analitici.
- **Base B1 (Singolare):** Questa base corrispondeva a due rette di vincolo parallele. Geometricamente non formano un vertice unico. Analiticamente, la sottomatrice corrispondente era singolare (determinante nullo), confermando che non è una base valida.
- **Base B2 (Primalmente inammissibile, dualmente ammissibile):**
    - **Geometricamente:** Identificata come un vertice fuori dalla regione ammissibile.
    - **Analiticamente:** Il controllo di ammissibilità primale falliva, in accordo con l’osservazione geometrica. Il controllo di ammissibilità duale passava, cioè il vettore dei costi stava nel cono generato dai gradienti dei vincoli attivi.
- **Base B3 (Ottima):**
    - **Geometricamente:** Identificata come un vertice sul bordo della regione ammissibile.
    - **Analiticamente:** Questa base risultava ammissibile sia per il problema primale che per quello duale, quindi era la base ottima.
## Domande degli studenti
1.  **Perché non ci sono vincoli di segno sulle variabili y?**
    - È una conseguenza diretta delle regole di costruzione del duale. I vincoli di uguaglianza nel primale corrispondono a variabili duali libere (senza segno).
2.  **Le due soluzioni (x’ e x’’) potrebbero essere equivalenti?**
    - Potrebbero, ma in questo caso specifico non lo erano perché davano valori obiettivo diversi.
3.  **Con la column generation si possono aggiungere entrambe le variabili utili?**
    - Sì, si possono aggiungere tutte le variabili potenzialmente utili. Tuttavia, serve un’analisi ulteriore (come il simplesso) per determinare quale sia davvero la più conveniente da inserire in base.
4.  **L’inammissibilità primale influenza l’ammissibilità duale?**
    - No, le due condizioni si verificano indipendentemente. Una base può essere ammissibile per una, per l’altra, per entrambe o per nessuna. Una base è ottima solo se soddisfa entrambe le condizioni contemporaneamente.