> Data e Ora: 2025-09-25 22:16:51
> Luogo: [Inserire Luogo]
> [Inserire Titolo]
`Casi d'uso` `Ingegneria dei requisiti` `UML`
## Tema
Questa lezione spiega come gli scenari evolvono in casi d'uso strutturati per derivare requisiti funzionali e non funzionali. Copre i ruoli degli attori, i flussi di eventi, le condizioni di ingresso/uscita, le eccezioni e i requisiti speciali. Si enfatizza il raffinamento iterativo tra obiettivi, assunzioni, fenomeni e requisiti. Vengono introdotti artefatti UML—diagrammi dei casi d'uso, di sequenza, a stati, di attività e di classe—incluse relazioni come generalizzazione, estensione e inclusione, oltre all'analisi nome-verbo per la modellazione del dominio e la specifica della cardinalità.
## Conclusioni
1. Gli scenari e i casi d'uso sono utilizzati per la pre-analisi per comprendere il dominio applicativo, le aspettative degli stakeholder e i confini tra il mondo e la macchina.
2. Gli stakeholder includono futuri utenti, clienti, utenti indiretti (es. cittadini che chiamano un pronto soccorso), operatori e interazioni con sistemi legacy e regolamenti/standard (es. software aeronautico).
3. Gli scenari sono descrizioni informali di situazioni specifiche che coinvolgono il software da realizzare.
4. I casi d'uso generalizzano le descrizioni degli scenari in modelli preliminari di requisiti strutturati.
5. La struttura dei casi d'uso include: nome (verbo/azione), attori (ruoli come ufficiale sul campo, dispatcher; il sistema è implicito), condizioni di ingresso, flusso di eventi, condizioni di uscita, eccezioni e requisiti speciali.
6. Gli attori sono ruoli, non individui specifici; gli scenari possono fare riferimento a persone specifiche, ma i casi d'uso astraggono ai ruoli.
7. I casi d'uso possono fare riferimento ad altri casi d'uso (es. allocare risorse riferito all'invio di emergenze).
8. Le eccezioni descrivono la gestione dei flussi non normali (es. perdita di connessione tra ufficiale sul campo e sala di controllo attiva notifiche immediate).
9. I requisiti speciali catturano aspetti non funzionali, spesso vincoli temporali (es. conferme e tempi di risposta entro 30 secondi).
10. I casi d'uso aiutano a derivare sia requisiti funzionali che non funzionali; l'identificazione dei requisiti è iterativa e circolare con la definizione dei casi d'uso.
## Punti Salienti
- ` "I casi d'uso sono utili per creare requisiti, e viceversa—la definizione dei requisiti può portare all'identificazione di nuovi casi d'uso." ` 
- ` "Dobbiamo creare requisiti in una forma non ambigua, concentrandoci sul sistema." ` 
- ` "Più esploriamo e analizziamo il nostro problema, e più definiamo il diagramma delle classi, più la nostra comprensione del problema migliora." ` 
## Capitoli e Argomenti
### Scenari e Casi d'Uso nell'Ingegneria dei Requisiti
> Gli scenari sono descrizioni informali di situazioni specifiche che coinvolgono il software da realizzare, utilizzati durante la pre-analisi per ottenere comprensione dagli stakeholder, documenti, regolamenti e sistemi legacy. I casi d'uso generalizzano e strutturano le informazioni degli scenari in un modello preliminare di requisiti, comprendente attori (ruoli), condizioni di ingresso, flusso di eventi, condizioni di uscita, eccezioni e requisiti speciali (spesso vincoli non funzionali). Il software da realizzare è implicito nei casi d'uso e non modellato come attore esplicito.
* **Punti Chiave**
  * Scenari: descrizioni informali e specifiche di situazioni per ottenere comprensione del dominio e delle aspettative.
  * Casi d'uso: generalizzazioni strutturate degli scenari che formano modelli preliminari di requisiti.
  * Gli attori nei casi d'uso sono ruoli (es. ufficiale sul campo, dispatcher), non individui; il sistema è implicito.
  * Le condizioni di ingresso definiscono quando un caso d'uso può iniziare (es. vero per segnalazione di emergenza, o utente loggato).
  * Il flusso di eventi dettaglia i passaggi che coinvolgono le interazioni tra attori e sistema.
  * Le condizioni di uscita definiscono lo stato dopo il completamento (es. conferma e risposta selezionata ricevute).
  * Le eccezioni catturano i flussi non normali (es. perdita di connessione attiva notifiche immediate).
  * I requisiti speciali catturano vincoli non funzionali (es. tempi entro 30 secondi).
  * I casi d'uso possono fare riferimento ad altri casi d'uso (es. allocare risorse).
  * La maggior parte del software interagisce con sistemi legacy; regolamenti/standard possono vincolare il design (es. aeronautico).
* **Spiegazione**
  Gli scenari sono raccolti da interazioni con gli stakeholder e revisioni di documenti/regolamenti, inclusa l'analisi di sistemi legacy, per formare una comprensione iniziale quando i confini tra il mondo e la macchina non sono ancora completamente noti. Da questi scenari, gli analisti derivano casi d'uso con una struttura organizzata per supportare un ragionamento sistematico. Dare ai casi d'uso nomi con azioni verbali aiuta a chiarire lo scopo. Gli attori sono definiti da ruoli per astrarre da individui specifici utilizzati negli scenari. Il flusso di eventi si concentra sulle interazioni tra attori e il software da realizzare, con condizioni di ingresso/uscita che forniscono stati pre/post. Le eccezioni delineano come il sistema dovrebbe comportarsi in condizioni anormali, e i requisiti speciali articolano vincoli non funzionali come i tempi. I casi d'uso possono fare riferimento ad altri casi d'uso per modularità. Questi artefatti strutturati alimentano l'identificazione iterativa dei requisiti, sia funzionali che non funzionali, e le lacune nei requisiti possono portare a casi d'uso nuovi o rivisti.
* **Esempi** 
  > Attori: ufficiale sul campo, dispatcher. Condizione di ingresso: vero (un'emergenza può essere segnalata in qualsiasi momento). Flusso di eventi: l'ufficiale sul campo attiva la segnalazione di emergenza sul terminale; il sistema presenta un modulo; l'ufficiale sul campo seleziona il livello di emergenza e compila i dettagli; invia il modulo; il dispatcher viene notificato; il dispatcher rivede le informazioni, assegna le risorse (riferimento al caso d'uso allocare risorse). Condizione di uscita: l'ufficiale sul campo ha ricevuto la conferma e la risposta selezionata. Eccezioni: perdita di connessione tra ufficiale sul campo e sala di controllo attiva notifiche immediate sia per l'ufficiale sul campo che per il dispatcher. Requisiti speciali: il rapporto dell'ufficiale sul campo viene confermato entro 30 secondi; la risposta selezionata arriva non oltre 30 secondi dopo essere stata inviata dal dispatcher. 
  * Identificare gli attori per ruolo: ufficiale sul campo, dispatcher.
  * Impostare la condizione di ingresso: vero per segnalazioni sempre disponibili.
  * Dettagliare il flusso passo-passo collegando azioni degli attori e risposte del sistema.
  * Fare riferimento a casi d'uso correlati per l'allocazione delle risorse.
  * Definire la condizione di uscita garantendo la conferma e la consegna della risposta selezionata.
  * Elencare le eccezioni per la gestione della perdita di connessione con notifiche immediate.
  * Specificare vincoli temporali non funzionali di 30 secondi per conferma e consegna della risposta.
  > Contesto: un progetto su uno scheduler di appuntamenti intelligente che tiene conto del tempo di viaggio, supporta la pianificazione dei percorsi e compiti flessibili. Scenario: Luigi riceve una telefonata dal suo capo che lo informa che è stata pianificata una riunione per il giorno successivo dalle 13 alle 15. Programma attuale: dalle 8 alle 12:30 lavoro sul progetto (evento a tempo fisso, tipo lavoro, possibilmente etichettato privato) tenuto a casa; dalle 13:30 alle 14:30 ora di pranzo (evento a tempo flessibile, tipo personale), durata minima 30 minuti, può essere tenuto ovunque. Obiettivo: inserire la riunione dalle 13 alle 15 nel programma. Passaggi: Luigi apre l'applicazione mobile, accede, inserisce un nuovo evento dalla home page; seleziona tipo evento lavoro, poi riunione; aggiunge descrizione "riunione con il capo"; imposta inizio alle 13, fine alle 15; imposta luogo di partenza casa e luogo della riunione Harrison Street; il sistema calcola i percorsi possibili da casa a Harrison Street, li mostra, conferma il tempo sufficiente; il sistema avvisa: "l'ora di pranzo si sovrappone alla riunione con il capo." Comportamento: in questo scenario il sistema avvisa solo; in alternativa il sistema potrebbe proporre un programma diverso. Caso d'uso: Inserire evento a tempo fisso. Attori: utente; servizio GPS (es. Google Maps) per calcolare percorsi e tempi di viaggio. 
  * Catturare i dettagli dello scenario inclusi orari specifici, tipi, luoghi e vincoli (durata minima pranzo 30 minuti, luogo flessibile).
  * Derivare il nome del caso d'uso con una frase verbale: Inserire evento a tempo fisso.
  * Identificare gli attori: utente e servizio GPS come attore non umano per il calcolo dei percorsi.
  * Dettagliare i passaggi del flusso: accesso, inserimento evento, selezione tipo, descrizione, orari di inizio/fine, luoghi, calcolo percorsi, conferma, avviso sovrapposizione.
  * Notare le alternative di comportamento (solo avviso vs proposta di modifiche al programma).
  * Utilizzare il servizio GPS per misurare il tempo di viaggio e convalidare la fattibilità entro i vincoli di programma.
* **Considerazioni** 
* Nominare i casi d'uso con verbi di azione per riflettere le attività (es. segnalare emergenza).
* Definire gli attori per ruolo, non per identità; trattare il software da realizzare come implicito.
* Includere condizioni di ingresso e uscita per delimitare il flusso di eventi.
* Documentare eccezioni per situazioni anormali (es. perdita di connessione).
* Registrare requisiti speciali per vincoli non funzionali, soprattutto temporali (es. entro 30 secondi).
* Fare riferimento a casi d'uso correlati per modularizzare attività complesse (es. allocare risorse).
* Studiare sistemi legacy, regolamenti e standard durante la pre-analisi.
* Iterare tra casi d'uso e requisiti; aspettarsi un raffinamento circolare.
* Tenere conto di utenti indiretti influenzati dal comportamento del sistema (es. cittadini nel dispatching delle ambulanze).
* Utilizzare servizi esterni (es. GPS) come attori non umani quando forniscono funzionalità essenziali.
### Derivazione Iterativa dei Requisiti dai Casi d'Uso
> I casi d'uso forniscono una base strutturata per derivare requisiti funzionali e non funzionali. Il processo è iterativo: la redazione dei requisiti può rivelare casi d'uso mancanti; nuovi o raffinati casi d'uso portano a requisiti aggiuntivi o rivisti.
* **Punti Chiave**
  * Casi d'uso ai requisiti: derivare dichiarazioni esplicite funzionali e non funzionali.
  * Requisiti ai casi d'uso: le lacune spingono all'identificazione di nuovi casi d'uso.
  * Raffinamento circolare e iterativo tra artefatti.
  * I requisiti non funzionali derivano spesso dalle sezioni sui requisiti speciali (es. tempistiche entro 30 secondi).
  * I requisiti funzionali sorgono dal flusso di eventi e dalle interazioni degli attori.
* **Spiegazione**
  Partendo da descrizioni strutturate dei casi d'uso, gli analisti estraggono requisiti funzionali traducendo il flusso degli eventi in capacità che il sistema deve supportare (es. supportare gli ufficiali sul campo nella segnalazione delle emergenze; supportare i dispatcher nell'allocazione delle risorse). I requisiti non funzionali sono derivati dai requisiti speciali e dalle eccezioni, inclusi vincoli temporali come conferme entro 30 secondi e risposte non oltre 30 secondi dopo l'invio. Durante questa estrazione, spesso emergono bisogni insoddisfatti o ambiguità, che portano alla creazione o modifica dei casi d'uso. Questo ciclo iterativo continua fino a quando requisiti e casi d'uso catturano sufficientemente le aspettative degli stakeholder e il comportamento del sistema.
* **Esempi** 
  > Dai casi d'uso segnalare emergenza e allocare risorse: il sistema deve supportare gli ufficiali sul campo nella segnalazione di un'emergenza; deve avere un tempo di risposta inferiore a 30 secondi quando reagisce a richieste relative all'emergenza; deve supportare i dispatcher nell'allocazione delle risorse all'incidente. 
  * Mappare ogni passaggio del caso d'uso a capacità funzionali (segnalazione, notifica di emergenza, allocazione delle risorse).
  * Tradurre i requisiti speciali in requisiti non funzionali (tempo di risposta inferiore a 30 secondi).
  * Assicurare allineamento tra eccezioni e requisiti di affidabilità (notifiche immediate in caso di perdita di connessione).
### Attori dei Casi d'Uso e Flussi di Eventi
> Gli attori nei casi d'uso non sono limitati agli esseri umani; possono essere servizi, software preesistenti, dispositivi, sensori o organizzazioni. I flussi di eventi forniscono descrizioni strutturate e generalizzano scenari precedenti. I casi d'uso racchiudono interpretazioni specifiche di procedure, e le scelte di design dell'interazione tra sistema e utente impattano sull'implementazione del software. Le definizioni dei casi d'uso dovrebbero includere condizioni di ingresso/uscita, eccezioni e requisiti speciali.
* **Punti Chiave**
  * Gli attori possono essere umani o non umani (servizi, software, dispositivi, sensori, organizzazioni).
  * I flussi di eventi strutturano lo scenario e generalizzano descrizioni precedenti.
  * La selezione della modalità di interazione influisce sull'implementazione del sistema.
  * I componenti del caso d'uso includono condizioni di ingresso, condizioni di uscita, eccezioni e requisiti speciali.
* **Spiegazione**
  La lezione sottolinea che un attore rappresenta qualsiasi entità esterna che interagisce con il sistema. Quando si modella un caso d'uso per uno scheduler, l'attore potrebbe essere un utente umano, un servizio di calendario o un dispositivo GPS. Il flusso degli eventi organizza i passaggi e le condizioni del caso d'uso. Scegliere, ad esempio, un'interfaccia utente conversazionale rispetto a un'interfaccia basata su moduli cambia il flusso dell'interazione e la successiva implementazione. Le condizioni di uscita definiscono il completamento, le eccezioni gestiscono deviazioni come problemi di connettività, e i requisiti speciali coprono vincoli come prestazioni o accuratezza.
* **Esempi** 
  > In uno scenario di pianificazione, il software interagisce con dispositivi (GPS), servizi (API di calendario) e utenti. Il flusso degli eventi include la creazione di eventi, impostazione di tempo/luogo/tipo, conferma degli eventi, avvisi del sistema per sovrapposizioni e calcolo dei percorsi. Le eccezioni come la perdita di connessione sono gestite da un caso d'uso separato (gestione problema di connessione). 
  * Identificare attori: utente, servizio calendario, dispositivo GPS.
  * Definire flusso di eventi: creare evento; utente imposta tempo/luogo/tipo; utente conferma; sistema avvisa su sovrapposizioni; sistema calcola percorso; sistema memorizza dati evento.
  * Specificare eccezioni: la perdita di connessione estende il caso principale d'uso tramite gestione problema di connessione.
* **Considerazioni** 
* Modellare esplicitamente attori non umani con l'icona dell'attore; possono rappresentare dispositivi, servizi o organizzazioni.
* Documentare condizioni di ingresso/uscita, eccezioni e requisiti speciali in ogni caso d'uso.
* Allineare le decisioni sulla modalità di interazione con le esigenze degli stakeholder, riconoscendo l'impatto sull'implementazione.
* Mantenere il software da realizzare implicito nei diagrammi dei casi d'uso; non aggiungere un nodo software separato.
* Utilizzare stereotipi per chiarire i ruoli degli attori (inizia, partecipa).
* **Circostanze Speciali** 
* Se si incontrano problemi di connettività durante un caso d'uso (es. segnalare emergenza), come dovrebbe essere affrontato? Modellare un caso d'uso esteso (gestione problema di connessione) che specifica le procedure di gestione e viene attivato sotto condizioni definite.
### Identificazione e Classificazione dei Fenomeni
> I fenomeni sono categorizzati per chiarire i confini e le responsabilità del sistema: solo mondo (si verificano interamente al di fuori del sistema), condivisi/controlati dal mondo (input controllati esternamente ma condivisi con il sistema), controllati dalla macchina (il sistema avvia o controlla interazioni che coinvolgono fenomeni condivisi), e solo macchina (attività interne al sistema).
* **Punti Chiave**
  * Fenomeni solo mondo accadono al di fuori del sistema (es. l'utente partecipa a riunioni).
  * Fenomeni condivisi controllati dal mondo: input controllati esternamente condivisi con il sistema (es. l'utente imposta tempo/luogo/tipo, conferma).
  * Fenomeni controllati dalla macchina: azioni del sistema che influenzano lo stato condiviso (es. avviso sovrapposizioni).
  * Fenomeni solo macchina: computazioni interne e gestione dei dati (es. calcolo percorsi, memorizzazione informazioni evento).
* **Spiegazione**
  Separando i fenomeni, gli ingegneri identificano ciò che il sistema può influenzare e ciò che deve osservare. Gli eventi solo mondo informano le assunzioni ma non sono controllabili. I dati condivisi controllati dal mondo devono essere convalidati e assimilati. Le uscite controllate dalla macchina riflettono le responsabilità del sistema di notificare o prevenire problemi. I processi solo macchina garantiscono coerenza interna e prestazioni (es. algoritmi di routing, persistenza).
* **Esempi** 
  > Solo mondo: l'utente imposta e partecipa a riunioni. Condiviso: l'utente imposta tempo/luogo/tipo, conferma. Controllato dalla macchina: il sistema avvisa di eventi sovrapposti. Solo macchina: il sistema calcola percorsi, memorizza informazioni sugli eventi. 
  * Elencare tutti gli eventi osservati e classificarli in base al controllo.
  * Decidere la validazione per gli input condivisi.
  * Definire condizioni e messaggi per avvisi controllati dalla macchina.
  * Progettare algoritmi interni per computazioni solo macchina.
* **Considerazioni** 
* Assicurarsi che i confini siano chiari per evitare requisiti che dettino comportamenti umani.
* Utilizzare la mappatura dei fenomeni per guidare la formulazione e la testabilità dei requisiti.
* Validare rigorosamente gli input condivisi per mantenere le assunzioni di dominio.
* **Circostanze Speciali** 
* Se gli input condivisi (es. dettagli evento) sono inconsistenti, come dovrebbe essere affrontato? Definire i comportamenti del sistema per il rilevamento dei conflitti e i prompt per l'utente, mantenendo le assunzioni di dominio.
### Obiettivi, Assunzioni di Dominio e Requisiti
> Da più casi d'uso, gli ingegneri derivano obiettivi precisi, assunzioni di dominio e requisiti di sistema. Gli obiettivi dichiarano i risultati desiderati; le assunzioni vincolano le interpretazioni; i requisiti specificano il comportamento verificabile del sistema in una forma non ambigua concentrandosi sul sistema. I requisiti condizionali (es. stile R5) catturano comportamenti dipendenti dal contesto.
* **Punti Chiave**
  * Gli obiettivi guidano la direzione della soluzione e possono essere raffinati iterativamente.
  * Esempi di obiettivi: gli utenti tengono traccia degli eventi a cui partecipano; gli utenti evitano di pianificare eventi sovrapposti.
  * Assunzioni di dominio: le informazioni sugli eventi inserite dall'utente sono corrette; i dati GPS sono accurati.
  * I requisiti devono concentrarsi sul sistema e utilizzare una formulazione coerente 'il sistema deve...'; con condizioni quando applicabile.
* **Spiegazione**
  Il processo è iterativo: gli obiettivi derivati dagli input degli stakeholder e dai casi d'uso possono cambiare quando assunzioni o requisiti si rivelano non fattibili. I requisiti devono essere precisi (es. 'Quando un utente conferma un evento, il sistema deve aggiornare...'). Assunzioni come l'accuratezza del GPS giustificano la stima dei percorsi; se violate, le uscite del sistema potrebbero essere errate, il che deve essere riconosciuto.
* **Esempi** 
  > Obiettivi: tenere traccia degli eventi, evitare sovrapposizioni. Assunzioni: informazioni sugli eventi corrette, GPS accurato. Requisiti: il sistema deve consentire la creazione di nuovi eventi; il sistema deve consentire l'impostazione di tempo/luogo/tipo; il sistema deve avvisare delle sovrapposizioni; quando un utente conferma un evento, il sistema deve aggiornare i registri pertinenti. 
  * Mappare ogni obiettivo a un insieme di requisiti.
  * Elencare le assunzioni e notare i rischi se non valide (es. inaccuratezze del GPS).
  * Definire requisiti condizionali per i cambiamenti di stato (la conferma attiva aggiornamenti).
* **Considerazioni** 
* Mantenere coerenza e non ambiguità nei requisiti.
* Concentrarsi sui requisiti sul comportamento del sistema, non sulle azioni umane.
* Catturare condizioni e attivatori all'interno delle dichiarazioni di requisito.
* **Circostanze Speciali** 
* Se un obiettivo non può essere realizzato sotto le attuali assunzioni, come dovrebbe essere affrontato? Rivedere gli obiettivi, regolare le assunzioni o ridefinire i requisiti per allinearsi con un'analisi fattibile.
### Modelli Interrelati e Raffinamento Iterativo
> Scenari, fenomeni, obiettivi, casi d'uso, assunzioni di dominio e requisiti sono strettamente connessi. Creare o modificare uno di essi può rendere necessarie aggiornamenti agli altri. L'ingegneria dei requisiti inizia con l'elicitation e la modellazione dei casi d'uso, per poi procedere a modelli più completi che comprendono il dominio applicativo e i requisiti e assunzioni formalizzati.
* **Punti Chiave**
  * I casi d'uso e i fenomeni trasferiscono informazioni osservate in modelli.
  * Gli obiettivi potrebbero aver bisogno di aggiornamenti quando si creano requisiti o assunzioni.
  * L'analisi degli obiettivi può rivelare casi d'uso mancanti.
  * Tutti gli elementi devono essere analizzati insieme per coerenza.
* **Spiegazione**
  La lezione sottolinea un ciclo di feedback: interviste con gli stakeholder e analisi dei documenti informano scenari e fenomeni. Questi vengono mappati in obiettivi e casi d'uso. I requisiti e le assunzioni preliminari vengono controllati per fattibilità, costringendo potenzialmente a modifiche degli obiettivi e a nuovi casi d'uso. I cicli iterativi stabilizzano il set di modelli.
* **Esempi** 
  > I casi d'uso iniziali (segnalare emergenza, aprire incidente, allocare risorse) portano all'identificazione dei partecipanti (ufficiale sul campo, dispatcher, allocatore risorse, risorsa) e casi d'uso ausiliari (visualizza mappa, gestisci problema di connessione). La generalizzazione (segnala problema) e le relazioni extend/include emergono durante il raffinamento. 
  * Partire dalle procedure core.
  * Aggiungere ruoli attore e attori non umani.
  * Modularizzare tramite include ed estensione.
  * Astrarre tramite generalizzazione per unificare casi specializzati.
* **Considerazioni** 
* Pianificare aggiornamenti iterativi tra gli elementi del modello.
* Documentare la motivazione per le modifiche per mantenere la tracciabilità.
* Assicurare allineamento tra obiettivi, assunzioni e requisiti.
### UML nell'Ingegneria dei Requisiti
> UML è un linguaggio di specifica con più tipi di diagrammi, utilizzabile sia a livello di requisiti che di design. Per i requisiti, ci si concentra su diagrammi dei casi d'uso (specifici per i requisiti), diagrammi di sequenza, diagrammi a stati, diagrammi di attività e diagrammi di classe. I diagrammi dinamici (blu) modellano il comportamento nel tempo; i diagrammi statici (rosa) modellano la struttura.
* **Punti Chiave**
  * I diagrammi dei casi d'uso forniscono panoramiche delle interazioni attore-caso d'uso e sono l'unico diagramma UML specifico per i requisiti.
  * I diagrammi di sequenza, a stati e di attività catturano le dinamiche; i diagrammi di classe catturano la struttura statica del dominio.
  * Nel design, si utilizzano diagrammi di componenti e distribuzione; i diagrammi di collaborazione sono equivalenti ai diagrammi di sequenza; i diagrammi di oggetti possono essere menzionati ma non utilizzati.
* **Spiegazione**
  UML offre una palette di diagrammi. I diagrammi dei casi d'uso evidenziano chi interagisce con cosa. I diagrammi di sequenza modellano i flussi di messaggi; le macchine a stati modellano stati e transizioni del ciclo di vita; i diagrammi di attività modellano i flussi di lavoro; i diagrammi di classe modellano entità e relazioni. Il software da realizzare è implicito nei diagrammi dei casi d'uso, implementando tutti i casi d'uso inclusi.
* **Esempi** 
  > Attori: ufficiale sul campo, dispatcher, allocatore risorse, risorsa. Casi d'uso: segnalare emergenza, aprire incidente, allocare risorse, segnalare problema, gestire problema di connessione, visualizza mappa. Ruoli: l'ufficiale sul campo inizia a segnalare un'emergenza; il dispatcher partecipa alla segnalazione dell'emergenza e inizia ad aprire incidenti e allocare risorse. Relazioni: segnalare un'emergenza generalizza segnalare un problema; gestire problema di connessione estende segnalare un'emergenza; visualizza mappa è inclusa da aprire incidente e allocare risorse. 
  * Disegnare associazioni attore-caso d'uso.
  * Aggiungere stereotipi (inizia, partecipa).
  * Modellare relazioni di generalizzazione, estensione e inclusione.
  * Mantenere il software da realizzare implicito; evitare di aggiungere un'icona di sistema.
* **Considerazioni** 
* Scegliere diagrammi di sequenza rispetto ai diagrammi di collaborazione se la leggibilità è una priorità.
* Utilizzare stereotipi per chiarire i ruoli degli attori nei diagrammi dei casi d'uso.
* Separare preoccupazioni dinamiche da statiche quando si selezionano i tipi di diagrammi.
* **Circostanze Speciali** 
* Se gli stakeholder non hanno mai visto UML prima, come dovrebbe essere affrontato? Fornire materiale aggiuntivo e un capitolo UML dedicato da studiare prima di procedere.
### Relazioni tra Casi d'Uso: Generalizzazione, Estensione, Inclusione
> La generalizzazione modella relazioni astratte-a-specializzate dei casi d'uso; l'estensione modella comportamenti opzionali attivati sotto condizioni specifiche; l'inclusione modella comportamenti modulari obbligatori di cui il caso d'uso base dipende.
* **Punti Chiave**
  * Generalizzazione: il caso d'uso base può essere astratto; i casi d'uso specializzati ereditano e affinano il comportamento.
  * Estensione: il caso d'uso estensore è opzionale; può specificare condizioni di estensione (es. aiuto quando l'utente preme un pulsante).
  * Inclusione: il caso d'uso incluso è obbligatorio; il caso d'uso base è incompleto senza di esso.
* **Spiegazione**
  La generalizzazione è parallela all'ereditarietà delle classi; i casi d'uso astratti mancano di flussi concreti e devono essere specializzati. L'estensione disaccoppia funzionalità opzionali, spesso eccezionali o ausiliarie (es. aiuto, gestione connessione). L'inclusione centralizza interazioni condivise obbligatorie (es. autenticazione, visualizzazione mappa) per evitare duplicazioni e garantire coerenza.
* **Esempi** 
  > Estensione: l'aiuto alla transazione del bancomat si estende alla transazione del bancomat (opzionale; attivata quando l'utente preme aiuto). Inclusione: l'autenticazione del cliente è inclusa nella transazione del bancomat (obbligatoria). 
  * Definire condizioni di estensione per l'aiuto.
  * Assicurarsi che l'autenticazione venga invocata durante il flusso della transazione.
  > Segnalare un'emergenza è una specializzazione di segnalare un problema (generalizzazione). La gestione del problema di connessione estende la segnalazione di emergenza (opzionale, gestisce eccezioni). Visualizza mappa è inclusa da aprire incidente e allocare risorse (funzionalità condivisa obbligatoria). 
  * Astrarre segnalare un problema per unificare segnalazione di emergenza/rapina.
  * Attivare la gestione della connessione sui guasti.
  * Riutilizzare l'interazione di visualizzazione mappa ovunque sia necessaria la visualizzazione della posizione.
* **Considerazioni** 
* Utilizzare l'inclusione per evitare la duplicazione di interazioni comuni tra più casi d'uso.
* Utilizzare l'estensione per flussi opzionali o eccezionali per mantenere i casi d'uso principali focalizzati.
* Utilizzare la generalizzazione per strutturare famiglie di casi d'uso correlati.
* **Circostanze Speciali** 
* Se un caso d'uso estensore deve attivarsi solo sotto trigger specifici, come dovrebbe essere affrontato? Specificare esplicite condizioni di estensione nella descrizione e nel diagramma (es. 'quando l'utente preme Aiuto').
### Diagrammi delle Classi per la Modellazione del Dominio
> I diagrammi delle classi forniscono una vista statica del dominio applicativo definendo entità (classi), attributi, tipi e relazioni. Supportano la modellazione dei requisiti chiarendo i concetti e i vincoli del dominio prima del design.
* **Punti Chiave**
  * Identificare le entità di dominio e i loro attributi con tipi precisi.
  * Utilizzare i diagrammi delle classi per catturare la struttura statica; complementare con diagrammi dinamici per il comportamento.
  * I diagrammi delle classi fanno parte della modellazione dei requisiti quando descrivono il dominio applicativo.
* **Spiegazione**
  Nell'esempio dell'asta, la classe Asta include attributi come aperta: Booleano e data: Data, che indicano il suo stato e tempistiche. Tale modellazione aiuta ad allineare i requisiti riguardo l'apertura/chiusura delle aste, la pianificazione degli eventi o l'associazione di incidenti con tempi e stati.
* **Esempi** 
  > Entità: asta. Attributi: aperta: Booleano; data: Data. Il diagramma può ulteriormente includere metodi o relazioni con offerte, articoli o utenti, sebbene la lezione si sia concentrata sugli attributi. 
  * Definire attributi con tipi espliciti.
  * Relazionare classi per catturare la semantica del dominio.
* **Considerazioni** 
* Assicurarsi che i tipi degli attributi siano espliciti e coerenti.
* Utilizzare i diagrammi delle classi precocemente per far emergere ambiguità di dominio.
### Diagrammi delle Classi UML per l'Analisi dei Requisiti
> I diagrammi delle classi a livello di requisito catturano il vocabolario del dominio: entità (classi), attributi rilevanti e associazioni con specifiche complete (nomi, cardinalità). Le operazioni sono tipicamente omesse a meno che non aiutino nella comprensione; a livello di design, le operazioni diventano centrali.
* **Punti Chiave**
  * Concentrarsi sull'identificazione delle entità di dominio rilevanti (classi) e solo sugli attributi necessari.
  * Definire le associazioni con precisione, comprese le cardinalità (es. uno, stella, 0..1).
  * Esempi di cardinalità: molti-a-molti (Partecipante–Asta), uno-a-molti (Asta–Articolo), uno opzionale (Articolo–Offerta offerta più alta relazione 0..1).
  * Chiarire che le associazioni possono esistere anche quando un lato ha zero istanze (es. aste senza partecipanti).
  * Utilizzare la terminologia UML appropriata: 'associazioni' piuttosto che 'relazioni'.
  * Includere tipi di base specifici per il dominio per gli attributi: Booleano, float, Data, Ora, Timestamp.
* **Spiegazione**
  La lezione sottolinea che i diagrammi delle classi a livello di requisito sono uno strumento per costruire un vocabolario condiviso del dominio problema. Il processo inizia identificando le entità che contano per comprendere l'applicazione, seguito dalla selezione solo di quegli attributi che sono rilevanti. Le associazioni sono la parte più critica; portano semantica attraverso nomi e cardinalità (es. 1, *, 0..1). La lezione illustra come mappare requisiti narrativi in diagrammi strutturati e come una precisa cardinalità eviti ambiguità. Nota anche che le operazioni di solito non sono necessarie in questa fase a meno che non migliorino la chiarezza, mentre nel design guadagnano importanza.
* **Esempi** 
  > Entità: Asta, Articolo, Offerta, Organizzatore, Direttore, Assistente, Partecipante (due tipi: remoto e di persona), app di terze parti. Associazioni: Il partecipante partecipa all'asta ("*" a "*"). L'asta coinvolge articoli: un'asta include uno o più articoli (almeno uno), e ogni articolo è coinvolto in non più di un'asta (cardinalità vicino all'articolo indica vincolo). L'offerta riguarda un singolo articolo; possono esistere più offerte per un singolo articolo. Associazione aggiuntiva: offerta più alta tra articolo e offerta con cardinalità 0..1 su entrambi i lati per rappresentare l'offerta vincente per articolo. Ruoli organizzativi: l'organizzatore organizza l'asta (uno o più organizzatori per asta; un organizzatore può organizzare più aste). Il direttore dirige l'asta (unico direttore per asta). L'assistente assiste l'asta (più assistenti per asta; gli assistenti possono assistere più aste). Il partecipante fa offerte. Le app di terze parti monitorano le aste e accedono alle informazioni su aste future e passate per costruire statistiche e pubblicizzare le aste. Gli attributi possono utilizzare tipi Booleano, float e Data, Ora, Timestamp. 
  * Partire dalla descrizione testuale: 'sviluppare un sistema—un gestore di aste—per la gestione delle aste di articoli.'
  * Applicare analisi nome-verbo: i sostantivi (Asta, Articolo, Lotto, Offerta, Direttore, Partecipante, Organizzatore, Assistente, app di terze parti) suggeriscono classi o attributi; i verbi (coinvolge, offre, registra, organizza, dirige, assiste, monitora) suggeriscono associazioni o operazioni.
  * Derivare associazioni: L'asta coinvolge un lotto di articoli -> la cardinalità implica una relazione uno-a-molti Asta-Articolo; Articolo in non più di un'asta -> il lato Articolo è 'uno'.
  * Identificare Offerta come entità (non solo come verbo) perché le offerte devono essere registrate; definire associazione Offerta-Articolo (riguarda), e derivare cardinalità (una Articolo, molte Offerte).
  * Aggiungere associazione offerta più alta Articolo-Offerta (0..1 ciascun lato) per catturare la semantica dell'offerta vincente.
  * Modellare ruoli e le loro associazioni all'asta: L'organizzatore organizza (1..* organizzatori per asta), Il direttore dirige (esattamente un direttore per asta), L'assistente assiste (0..* assistenti per asta).
  * Includere tipi di Partecipante (remoto/di persona) e Partecipante fa offerte.
  * Aggiungere app di terze parti che monitorano le aste e accedono ai dati per statistiche e pubblicità.
  * Notare scenari di partecipazione opzionale: le aste possono avere zero partecipanti; gli individui possono voler partecipare quando non esiste alcuna asta.
* **Considerazioni** 
* Utilizzare la terminologia e la sintassi UML 'associazione' con precisione per evitare interpretazioni errate.
* Nominare esplicitamente le associazioni e definire le cardinalità per rendere le specifiche precise.
* Includere solo attributi rilevanti nella fase dei requisiti per mantenere i diagrammi focalizzati.
* Considerare se le operazioni aiutano nella comprensione; includerle selettivamente a livello di requisiti.
* Utilizzare tipi specifici per il dominio (Data, Ora, Timestamp) in modo coerente per gli attributi temporali.
* **Circostanze Speciali** 
* Se si incontra un'asta senza partecipanti, come dovrebbe essere affrontato? Modellare l'associazione Partecipante-Asta per consentire zero partecipanti e gestire di conseguenza notifiche/UI.
* Se un articolo non deve essere in più di un'asta, come dovrebbe essere affrontato? Forzare una cardinalità di 1 sul lato Articolo dell'associazione Asta-Articolo e convalidare durante la configurazione.
* Se le offerte devono essere registrate e auditabili, come dovrebbe essere affrontato? Trattare Offerta come un'entità con associazioni a Articolo e Partecipante, non solo come un'operazione.
* Se si devono distinguere le offerte vincenti, come dovrebbe essere affrontato? Aggiungere una specifica associazione 'offerta più alta' con cardinalità 0..1 a Articolo-Offerta per catturare il vincolo di un'unica offerta vincente.
### Analisi Nome-Verbo per Derivare Diagrammi delle Classi
> Un approccio sistematico per mappare requisiti testuali in diagrammi delle classi identificando i sostantivi come classi candidate o attributi e i verbi come associazioni o operazioni candidate. Aiuta a derivare entità, relazioni e cardinalità da descrizioni narrative.
* **Punti Chiave**
  * I sostantivi possono rappresentare classi (es. Asta, Articolo, Offerta) o attributi (es. lotto).
  * I verbi possono rappresentare associazioni (es. coinvolge, riguarda) o operazioni (es. offre).
  * Il contesto determina se un termine diventa una classe o un attributo; esigenze di registrazione o persistenza suggeriscono entità.
  * Le cardinalità spesso emergono da qualificatori come 'lotto', 'uno per uno', 'multiplo'.
  * L'iterazione dell'analisi affina il diagramma e approfondisce la comprensione.
* **Spiegazione**
  La lezione dimostra l'analisi nome-verbo nello scenario dell'asta: 'ogni asta coinvolge un lotto di articoli' produce un'associazione Asta-Articolo con cardinalità uno-a-molti; 'gli articoli vengono offerti uno per uno' suggerisce che Offerta è sia un'operazione che un'entità. Riferimenti alla 'registrazione delle offerte' spingono Offerta nel set delle classi. Frasi successive producono ruoli (Organizzatore, Direttore, Assistente) e sistemi esterni (app di terze parti). Le decisioni sulla cardinalità sono supportate da frasi come 'uno per uno' e 'più partecipanti'.
* **Esempi** 
  > Testo: 'ogni asta coinvolge un lotto di articoli che vengono offerti uno per uno ... l'assistente registra l'offerta in un'applicazione ... i partecipanti possono offrire ... se non arriva una nuova offerta entro quattro minuti dall'ultima, l'offerta si chiude e l'offerta più alta vince ... le app di terze parti accedono alle informazioni su aste future e passate per costruire statistiche e pubblicizzare le aste.' Sostantivi: asta, articoli, lotto, offerta, assistente, applicazione, partecipante, direttore, organizzatore, pannello, stanza, sistema, organizzatori, applicazioni di terze parti, statistiche. Verbi: coinvolge, offre, registra, può offrire, arriva, chiude, vince, gestisce, accede, costruisce, disegna, pubblicizza. Decisioni: Lotto informa la cardinalità Asta-Articolo (uno-a-molti). Offerta diventa una classe (necessità di registrazione). 'Offerta più alta vince' motiva una specifica associazione (offerta più alta) tra Articolo e Offerta con cardinalità 0..1. I ruoli diventano classi con associazioni all'Asta. Le app di terze parti diventano una classe esterna che monitora l'Asta. 
  * Estrarre sostantivi dalla narrativa e classificarli come classi, attributi o entità esterne.
  * Mappare i verbi ad associazioni o operazioni; determinare quali verbi implicano entità persistenti.
  * Assegnare cardinalità utilizzando quantificatori ('lotto', 'uno per uno', 'multiplo').
  * Affinare iterativamente le associazioni e aggiungere nomi (riguarda, offerta più alta).
* **Considerazioni** 
* Non tutti i sostantivi devono diventare classi; alcuni sono attributi o possono essere omessi se irrilevanti.
* I verbi possono indicare associazioni o operazioni; decidere in base alla persistenza e alla semantica del dominio.
* Utilizzare indizi narrativi per impostare esplicitamente le cardinalità (evitare assunzioni).
### Associazioni, Cardinalità e Classi di Associazione
> Le associazioni in UML definiscono relazioni tra classi. La cardinalità esprime quante istanze possono partecipare. Le classi di associazione aggiungono attributi a un'associazione specifica.
* **Punti Chiave**
  * Cardinalità comuni: uno (1), molti (*), intervalli come 0..1, 1..*, 2..*.
  * Associazioni opzionali consentono zero istanze (es. asta con zero partecipanti).
  * Più associazioni tra le stesse classi possono esprimere semantiche diverse (es. Articolo–Offerta: riguarda vs. offerta più alta).
  * Le classi di associazione (es. Frequenza tra PercorsoUrbano e Fermata) catturano informazioni aggiuntive sul collegamento stesso.
  * Nominare le associazioni chiarisce le semantiche (riguarda, offerta più alta).
* **Spiegazione**
  L'esempio dell'asta mostra partecipazione molti-a-molti e relazioni asta-articolo uno-a-molti con vincoli sul lato articolo. Le associazioni delle offerte dimostrano sia relazioni generali che speciali. Nel Trip Tracker, Frequenza come classe di associazione arricchisce PercorsoUrbano–Fermata con semantiche temporali (ogni x secondi o x minuti), mentre PercorsoADistanza–Fermata utilizza voci di Orario con orari esatti.
* **Esempi** 
  > Percorso–Fermata: Il percorso si riferisce ad almeno due fermate (2..*). PercorsoUrbano–Fermata utilizza una classe di associazione Frequenza per rappresentare fermate periodiche (ogni x secondi o x minuti). PercorsoADistanza–Fermata utilizza Orario e VoceOrario (inizio, intermedio, fine) per rappresentare orari programmati esatti. 
  * Definire associazione base Percorso–Fermata con cardinalità minima 2.
  * Aggiungere specializzazione: PercorsoUrbano (metropolitana, autobus, tram) e PercorsoADistanza.
  * Allegare Frequenza come classe di associazione a PercorsoUrbano–Fermata.
  * Modellare Orario con categorie VoceOrario per PercorsoADistanza–Fermata.
* **Considerazioni** 
* Assicurarsi che le cardinalità siano esplicite e giustificate dal testo (es. almeno un articolo per asta).
* Utilizzare classi di associazione quando la relazione stessa ha proprietà (es. frequenza).
### Diagrammi di Sequenza nei Requisiti
> I diagrammi di sequenza rappresentano il flusso di eventi in un caso d'uso, concentrandosi sulle interazioni tra oggetti tramite scambi di messaggi, con il tempo che scorre verticalmente.
* **Punti Chiave**
  * Le linee di vita rappresentano i partecipanti (es. Utente, Sistema).
  * Le frecce denotano i messaggi: le richieste sincrone si aspettano risposte e bloccano il mittente; i messaggi asincroni sono unidirezionali, il mittente continua senza attendere.
  * Le frecce tratteggiate rappresentano le risposte.
  * Le attivazioni mostrano il periodo di computazione attivato da un messaggio.
  * Le frecce di auto-chiamata rappresentano un'elaborazione interna all'interno di una linea di vita.
  * Le caselle di titolo possono incorniciare il diagramma ma sono opzionali.
* **Spiegazione**
  La lezione mostra un sequenza ad alto livello in cui l'Utente si iscrive, il Sistema risponde (può essere OK o non OK), il Sistema calcola periodicamente problemi internamente e notifica asincronicamente l'Utente. L'ordinamento indica che la notifica avviene solo dopo l'iscrizione. La sintassi trasmette semantica: le diverse punte delle frecce indicano messaggi sincroni rispetto ad asincroni; le risposte devono corrispondere alle richieste.
* **Esempi** 
  > Esempio errato da 'TravLander': Una richiesta da Utente a Sistema è seguita dall'attivazione del Sistema, poi il Sistema invia un'altra richiesta a Utente ma non viene mostrata alcuna risposta; la apparente risposta è erroneamente rappresentata come un'altra freccia di richiesta. Questo uso improprio rompe la prevista corrispondenza richiesta-risposta e confonde l'interpretazione. 
  * Identificare la richiesta sincrona iniziale.
  * Aspettarsi e rappresentare una corrispondente risposta con una freccia tratteggiata.
  * Evitare di rappresentare risposte utilizzando punte di freccia di richiesta.
  * Assicurarsi che le attivazioni siano allineate con il tempismo dei messaggi e le responsabilità della linea di vita.
* **Considerazioni** 
* Mantenere la correttezza sintattica per preservare la comprensione condivisa.
* Gli strumenti potrebbero non imporre la sintassi; gli autori devono autovalidare i diagrammi.
* Modellare esplicitamente risposte e tipi di messaggi per evitare ambiguità.
### Operazioni di Sistema e Interazione Utente nel Trip Tracker
> Il sistema Trip Tracker offre operazioni allineate con le esigenze degli utenti e il monitoraggio dei percorsi: iscriversi, disiscriversi, calcolareProblemi, notificare. L'utente è modellato per ricevere notifiche.
* **Punti Chiave**
  * Iscriversi e disiscriversi gestiscono gli interessi degli utenti in percorsi specifici.
  * CalcolareProblemi determina periodicamente problemi sui percorsi.
  * Notificare invia informazioni sui problemi agli utenti (asincrono).
  * Il sistema interagisce con entità core di dominio: Percorso, Fermata, PercorsoUrbano, PercorsoADistanza, Orario, Frequenza.
  * La sequenza impone che la notifica segua l'iscrizione.
* **Spiegazione**
  Il diagramma del Trip Tracker evidenzia le operazioni a livello di sistema direttamente nel modello dei requisiti per chiarire i comportamenti esterni. Gli utenti possono iscriversi/disiscriversi ai percorsi; il sistema calcola problemi e notifica gli utenti. La distinzione tra percorsi urbani e a lunga distanza si collega a diverse semantiche temporali (frequenza rispetto a orario).
* **Esempi** 
  > La linea di vita dell'utente invia iscrizione al Sistema; il Sistema memorizza l'iscrizione e risponde (OK o non OK). La linea di vita del Sistema esegue successivamente calcolaProblemi internamente (auto-chiamata), poi invia notifiche asincrone all'Utente. PercorsoUrbano ha fermate basate su frequenza (ogni x secondi o x minuti). PercorsoADistanza utilizza voci di orario esatte (inizio, intermedio, fine). 
  * Modellare lo stato di iscrizione dell'utente.
  * Implementare il calcolo periodico dei problemi.
  * Utilizzare notifiche asincrone per disaccoppiare il tempismo della consegna.
  * Rappresentare semantiche temporali tramite classe di associazione e strutture di orario.
* **Considerazioni** 
* Utilizzare notifiche asincrone per evitare di bloccare le operazioni del sistema.
* Assicurarsi che lo stato di iscrizione controlli le notifiche.
