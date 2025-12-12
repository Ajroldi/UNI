Data e ora: 2025-12-11 21:16:18
Luogo: [Inserisci luogo]
Corso: [Inserisci nome corso]
## Panoramica
Il laboratorio si è aperto con una spiegazione dettagliata dei progetti piccolo e grande, coprendo enunciato, vincoli, obiettivi e modalità di consegna. Si è poi passati a un laboratorio pratico sull’analisi di sensitività per un problema di product mix, mostrando come identificare i vincoli restrittivi e valutare variazioni di prezzo sia con metodi brute-force che con strumenti teorici come variabili duali e costi ridotti.
## Contenuti rimanenti
Nessun contenuto rimanente pianificato o identificato
## Contenuti trattati
### 1. Spiegazione dei progetti: piccolo e grande
- **Progetto piccolo:** Formulare un problema di programmazione lineare mista per calcolare i percorsi ottimali di quattro droni che analizzano un edificio.
- **Obiettivo:** Minimizzare il tempo di ritorno allabase dell’ultimo drone.
- **Vincoli e dettagli:**
    - Tutti i droni partono dallo stesso punto iniziale e devono tornare.
    - Tutti i punti specificati devono essere visitati.
    - Velocità di movimento: salita 1 m/s, discesa 2 m/s, orizzontale 1.5 m/s.
    - È richiesto un pre-processing dei punti in base a condizioni di connettività specifiche.
    - Due istanze (edifici) fornite con diverse coordinate di base e condizioni di ingresso.
- **Progetto grande:** Estensione del piccolo con vincoli di batteria, aumentando la complessità.
- **Modalità di consegna:**
    - Scadenza progetto piccolo: 10 dicembre.
    - Scadenza progetto grande da definire con il Prof. Maluchelli.
    - Consegna tramite cartella su Webe/Omweave.
    - Il codice deve essere eseguibile con un comando specifico.
    - Output: percorso di ogni drone.
    - Dettagli amministrativi per la consegna di gruppo saranno chiariti.
> **Suggerimenti AI**
> La spiegazione dei progetti è stata chiara e ben strutturata; seguire il PDF ha aiutato gli studenti. Per chiarire regole complesse come le velocità (“La combinazione di movimenti è il massimo tra orizzontale e salita”), fermarsi su un esempio numerico aiuta: “Se un drone sale di 3 metri e si sposta di 4 metri orizzontalmente, il tempo è dato dal massimo dei due, non dalla somma.” Così la regola astratta diventa concreta.

### 2. Analisi di sensitività: problema di product mix (parte 1)
- **Problema:** Massimizzare il ricavo dalla produzione di quattro profumi usando cinque ingredienti con disponibilità limitata.
- **Formulazione:** Programmazione lineare standard: Massimizza C*X soggetto a A*X <= B.
- **Task 1: Implementazione modello:** Gli studenti hanno completato una funzione Python usando il pacchetto MIP per modellare e risolvere il PL.
- **Task 2: Leve di miglioramento:** Si è visto che aumentare la disponibilità degli ingredienti (vettore B) è la leva per aumentare il profitto senza cambiare prezzi o composizione; dimostrato aumentando tutte le disponibilità del 10% e risolvendo di nuovo.
- **Task 3: Vincoli restrittivi:**
    - **Metodo brute-force:** Si è aumentata la disponibilità di ogni ingrediente uno alla volta osservando la variazione dell’obiettivo; il vincolo più restrittivo dava il maggior aumento di profitto.
    - **Metodo teorico:** Introdotte le variabili duali (prezzi ombra) come misura corretta di sensitività; estratti i valori duali con `.pi` dal modello risolto, confermando i risultati brute-force.
> **Suggerimenti AI**
> Il passaggio da brute-force a variabili duali è stato ottimo per mostrare il “perché” della teoria. Quando si introducono i prezzi ombra, aiuta una breve intuizione economica: “Il prezzo ombra è il massimo che pagheresti per un’unità in più di quell’ingrediente. Se è 150€ e comprarne di più costa meno, conviene.” Questo ancoraggio reale aiuta a fissare il concetto.

### 3. Analisi di sensitività: costo ridotto
- **Problema:** Un profumo (il numero 3) non veniva prodotto nella soluzione ottima; si è chiesto di quanto deve aumentare il suo prezzo per renderne conveniente la produzione.
- **Metodo brute-force:** Usato un ciclo `while` per aumentare il prezzo di 5€ e risolvere finché la produzione diventava positiva.
- **Metodo teorico:** Introdotto il costo ridotto per variabili al vincolo inferiore (zero); spiegato come il miglioramento richiesto nel coefficiente obiettivo affinché la variabile entri in base.
- **Implementazione:** Estratto il costo ridotto con `.rc` in MIP per calcolare l’aumento esatto di prezzo necessario.
> **Suggerimenti AI**
> Qui il contrasto tra approssimazione iterativa e soluzione analitica esatta via costo ridotto è stato chiaro, rafforzato dal live coding. Per collegare meglio la soglia brute-force, calcolarla esplicitamente: “Se il prezzo originale è 216€ e il costo ridotto è 144€, la soglia è 216 + 144 = 360€,” che coincide con il range 355–360€ trovato col ciclo.

### 4. Intuizione sulla column generation (product mix parte 2)
- **Estensione problema:** Da 4 a 40 tipi di profumo.
- **Task 1: Selezione del miglior nuovo profumo:**
    - **Brute-force:** Iterato sui nuovi profumi, aggiungendone uno alla volta al set originale, risolvendo ogni volta il PL a 5 variabili per trovare il maggior incremento di ricavo.
    - **Metodo teorico:** Mostrata l’equivalenza con il calcolo dei costi ridotti per tutti i profumi non di base usando i duali del PL a 4 profumi; profumi con costo ridotto positivo sono candidati a migliorare la soluzione.
- **Task 2 & 3: Aggiunta sequenziale e soluzione finale:**
    - Accennato a ripetere dopo aver aggiunto il miglior nuovo profumo.
    - Risolto il problema completo con tutti i 40 profumi.
- **Key Takeaway:** Il processo manuale e iterativo di trovare e aggiungere variabili (colonne) con costo ridotto favorevole è l’intuizione base dietro l’algoritmo di column generation.
> **Suggerimenti AI**
> Ottimo modo di costruire l’intuizione sulla column generation. La domanda “Possiamo automatizzare?” è stata la transizione perfetta. Esplicitare il collegamento: “Risolvi un problema piccolo, usa i duali per prezzare nuove opzioni, aggiungi la migliore, ripeti—questo è esattamente ciò che la column generation automatizza nei problemi su larga scala.” 💡

## Domande degli studenti
1. **Sulle velocità dei droni: se scende e va orizzontale, il tempo è il massimo tra le due?**
- Sì. È il massimo tra componente di discesa e orizzontale.
2. **La consegna del progetto è personale o di gruppo?**
- Mi risulta personale, ma verificherò con il Prof. Manatelli. Si manterrà una consegna per gruppo e probabilmente ci sarà un modulo per indicare i membri. La convenzione di naming ufficiale sarà aggiornata.
3. **La scadenza del progetto resta la stessa?**
- Sì, la scadenza resta invariata.
4. **Perché aumentiamo la disponibilità (i `b_i`) e non la diminuiamo per i vincoli restrittivi?**
- Perché stiamo massimizzando: rilassiamo i vincoli aumentando le risorse per vedere se il profitto cresce. In un problema di minimizzazione, si ridurrebbero per vedere se il costo scende.
5. **Usando `.rc` per il costo ridotto, fa iterazione brute-force?**
- No. Usa una formula analitica diretta dalla teoria LP per calcolare il costo ridotto—è un calcolo esatto, non iterativo.