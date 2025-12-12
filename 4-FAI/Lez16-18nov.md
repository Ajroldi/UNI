> Data Ora: 2025-12-11 13:00:14
> Luogo: [Luogo]
> Corso: [Nome Corso]
## Panoramica
Le lezioni hanno introdotto la pianificazione classica come sintesi tra ricerca e rappresentazioni di stato basate sulla logica, messo a confronto i linguaggi STRIPS e PDDL, e applicato questi concetti al mondo dell'aspirapolvere e al Block World. I temi centrali hanno incluso la rappresentazione di stati, obiettivi e azioni (precondizioni, liste add/delete, applicabilità, effetti), assunzioni fondative (nome unico, chiusura del dominio, mondo chiuso), schemi d'azione e binding dei parametri, e un trattamento operativo del frame problem tramite liste add/delete. La formalizzazione del Block World ha coperto costanti, predicati, domini vs. fluenti, e azioni canoniche (unstack/stack/pick-up/put-down), con attenzione all'applicabilità e alle transizioni di stato. Un esempio di pianificazione PDDL con robot–chiave–porta–scatola in due stanze ha evidenziato vincoli negli schemi (uguaglianza/disuguaglianza), ridondanza di locked/unlocked sotto CWA, e trucchi di modellazione STRIPS (posizione implicita della chiave quando è in mano). Le lezioni hanno anche valutato un modello di pianificazione generato da ChatGPT, diagnosticando problemi di binding dei parametri e di liste degli effetti.
## Contenuti Rimanenti
1. Gestione delle precondizioni negative in STRIPS tramite formulazioni alternative (predicati compilati o strategie di modellazione) — promesso ma non ancora trattato.
2. Il frame problem: introdotto con analogia e soluzione operativa, ma assiomi logici formali e discussione più profonda rimandati.
3. Procedure/algoritmi di inferenza dettagliati per la logica del primo ordine — riconosciuto come non trattato.
4. Sottoinsieme stabile delle funzionalità PDDL: da studiare; elenco specifico delle feature non ancora trattato.
5. Comportamento dell'implementazione del planner quando si cancellano letterali non esistenti: variabilità notata; nessuna esplorazione o esempio più approfondito.
6. Assiomi logici formali per l'evoluzione temporale dei fluenti (es., regole F(t+1)) da scrivere nella prossima lezione.
7. Funzionalità PDDL dettagliate oltre STRIPS di base (trattamento completo di letterali negativi, vincoli di uguaglianza/disuguaglianza, espressioni algebriche).
8. Definizione completa delle azioni omessa nell'esempio a due stanze (es., put-key-in-box, varianti open/close door).
9. Come trovare un piano (algoritmi di pianificazione e metodi di ricerca), esplicitamente programmato per la prossima volta.
## Contenuti Trattati
### 1. Pianificazione Classica: Motivazione e Posizione all'interno della Ricerca
- La pianificazione combina ricerca e rappresentazioni di stato basate sulla logica per migliorare l'efficienza rispetto agli agenti puramente logici.
- La pianificazione classica specializza la ricerca generale con rappresentazioni strutturate per stati, azioni e obiettivi.
- Obiettivo: trovare una sequenza di azioni dallo stato iniziale a uno stato che soddisfi l'obiettivo.
- Suggerimenti: ricapitolare brevemente un'inefficienza concreta degli agenti puramente logici e includere un semplice diagramma di flusso che collochi la pianificazione rispetto a ricerca e logica.
### 2. STRIPS e PDDL: Linguaggi e Compromessi
- STRIPS: storica, logica del primo ordine semplificata (predicati, variabili, costanti; niente funzioni).
- PDDL: famiglia di linguaggi, un superset di STRIPS; le versioni variano e non sono supersets rigorosi l'una dell'altra.
- Compromesso espressività vs. efficienza computazionale: linguaggi più ricchi aumentano il costo di elaborazione.
- Storia: STRIPS per Shakey; PDDL per l'International Planning Competition.
- Piano di concentrarsi su un nucleo stabile di funzionalità PDDL.
- Suggerimenti: aneddoti storici concisi; elenco comparativo delle feature STRIPS vs. PDDL; un esempio che illustri il “no functions” di STRIPS.
### 3. Analogia Sfondo–Primo Piano e Frame Problem
- Analogia con l'animazione: sfondi fissi (non cambiamento) vs. primi piani che cambiano (effetti).
- Frame problem: specificare cosa rimane uguale vs. cosa cambia tra passi temporali.
- Fluenti indicizzati nel tempo: F(t+1) vale se creato al tempo t o se persiste da t senza eliminazione.
- Suggerimenti: un template pseudo-logico per l'evoluzione temporale; invitare analogie alternative (es., “diff” del controllo di versione).
### 4. Soluzione Operativa STRIPS/PDDL al Frame Problem
- Scelta operativa: le azioni elencano add/delete; tutto il resto persiste allo stato successivo.
- Questo evita assiomi esaustivi di non-cambiamento e aiuta l'efficienza del planner.
- Suggerimenti: percorrere un micro esempio S, A e S’ risultante.
### 5. Rappresentare un Problema di Pianificazione: Setup del Mondo dell'Aspirapolvere
- Ambiente: due stanze (R1, R2); un robot si muove e aspira sporco; obiettivo: entrambe le stanze pulite.
- Costanti: robot, R1, R2; Predicati: in(x, y), clean(x).
- Formulazioni valide multiple; le scelte rappresentazionali influenzano chiarezza ed efficienza (es., sporco esplicito).
- Suggerimenti: codifiche alternative affiancate (con/senza “sporco”); criteri per rappresentazioni migliori (meno predicati, precondizioni più semplici).
### 6. Formalizzazione del Block World: Costanti, Predicati e Scelte di Astrazione
- Scenario: pinza robotica che muove blocchi; tavolo arbitrariamente grande.
- Stato iniziale di esempio: on(A, table), on(C, A); clear(B), clear(C); handempty; costanti: A, B, C, table.
- Predicati: block(x), on(x,y), clear(x), handempty (0-ario).
- Astrazione: omettere colori, materiali, ecc.
- Suggerimenti: confronto in due colonne tra costanti e predicati; piccolo diagramma di blocchi impilati; chiarire i predicati 0-ari.
### 7. Predicati Derivati vs. Primitivi e Assunzione di Mondo Chiuso (CWA)
- Sotto CWA, tutto ciò che non è elencato nello stato è falso.
- Implicazione: occorre elencare esplicitamente i fatti clear; non si può derivare clear(A) dall'assenza di on(qualcosa, A).
- Distinguere predicati di dominio (immutabili, es., block(A)) da fluenti (variabili, es., on/clear/handempty).
- Suggerimenti: piccola tavola di verità per CWA; esercizio che marca dominio vs. fluenti.
### 8. Rappresentazione dello Stato e Assunzioni di Base
- Stati: congiunzioni di letterali positivi e ground; niente negazione, variabili o disgiunzione.
- Assunzione del Nome Unico: costanti distinte denotano oggetti distinti.
- Assunzione di Chiusura del Dominio: esistono solo gli oggetti denotati dalle costanti.
- Assunzione di Mondo Chiuso: i fatti non nello stato sono falsi; spiega l'assenza di letterali negativi negli stati.
- Suggerimenti: verifica interattiva in cui gli studenti inferiscono fatti falsi via CWA; piccolo esercizio che rinforza i vincoli.
### 9. Rappresentazione dell'Obiettivo e Soddisfacimento
- Obiettivi: congiunzioni di letterali; STRIPS non consente letterali negativi negli obiettivi; PDDL li consente.
- Esempio: clean(R1) ∧ clean(R2); la disgiunzione (es., clean(R1) ∨ clean(R2)) non è consentita.
- Soddisfacimento: i letterali positivi dell'obiettivo devono essere nello stato; i letterali negativi devono essere assenti.
- Gli obiettivi possono contenere variabili; clean(x) è soddisfatto da qualunque letterale ground corrispondente; la negazione in PDDL significa assenza di fatti corrispondenti (es., tutte le stanze sporche se non c'è clean(x)).
- Suggerimenti: esempio svolto con variabili e negazione; chiarire i limiti delle variabili rispetto alla disgiunzione.
### 10. Rappresentazione delle Azioni: Precondizioni, Add/Delete, Effetti
- Le azioni specificano nome, precondizioni, lista delete, lista add; talvolta combinate in un'unica lista di effetti con letterali positivi/negativi.
- Esempio “right”: precondizione in(robot, R1); delete in(robot, R1); add in(robot, R2).
- Precondizioni: congiuntive; STRIPS consente solo positive; PDDL consente positive e negative; si possono usare variabili.
- Applicabilità: le precondizioni positive devono essere nello stato; le precondizioni negative devono essere assenti.
- Transizione di stato: copia lo stato, poi delete, poi add; i fatti non coinvolti persistono (intuizione del frame).
- Suggerimenti: pseudocodice conciso per apply(action, state); collegare “i fatti non coinvolti persistono” al frame problem.
### 11. Schemi d'Azione e Binding dei Parametri
- Lo schema d'azione compatta famiglie di azioni con parametri (es., suck(x): precondizione in(robot, x); add clean(x)).
- L'istanza vincola i parametri alle costanti; tutti i parametri devono apparire nelle precondizioni per essere vincolati.
- Gli stati sono insiemi: aggiungere un letterale già presente non ha effetto; verifiche di applicabilità e di obiettivo si riducono a appartenenza all'insieme.
- Suggerimenti: checklist degli errori (tutti i parametri nelle precondizioni; gli effetti fanno riferimento solo a parametri vincolati); esercizio per correggere binding mancanti.
### 12. Insieme di Azioni per l'Aspirapolvere e Scelte di Progetto
- Azioni definite: left, right, suck R1, suck R2; notate ridondanze e problemi di scalabilità.
- Le azioni suck consentono la pulizia a prescindere dallo stato di sporco; aggiungere NOT clean come precondizione è solo PDDL (non STRIPS).
- Formulazioni valide multiple producono diverse caratteristiche di ricerca.
- Suggerimenti: proporre una modellazione compatibile con STRIPS (es., aggiungere il predicato dirty(x)); confrontare lunghezze dei piani/fattori di diramazione tra codifiche.
### 13. Schemi d'Azione nel Block World: Unstack/Stack/Pick-Up/Put-Down
- Unstack(x,y): precondizioni—handempty, block(x), block(y), clear(x), on(x,y); effetti—delete handempty, clear(x), on(x,y); add holding(x), clear(y).
- Esempio di applicabilità: unstack(C, A) quando tutte le precondizioni sono presenti.
- Transizione di stato: prima delete poi add; gli altri fatti persistono.
- Azioni aggiuntive: stack(x,y), pick-up(x) dal tavolo, put-down(x) sul tavolo.
- Azioni specifiche per il tavolo: usare stack/unstack con table può creare conflitti add/delete su clear(table); l'ordine add/delete dipende dall'implementazione, quindi azioni separate sono più sicure.
- Suggerimenti: mostrare lo schema problematico e gli esiti divergenti sotto diversi ordini add/delete; checklist per applicabilità e transizione.
### 14. Funzionalità PDDL e Rappresentazione in Python (Breve)
- La PDDL di base consente letterali negativi in precondizioni/obiettivi; alcune versioni supportano uguaglianze/disuguaglianze ed espressioni algebriche.
- Rappresentazione in Python: codificare stato iniziale, obiettivi, azioni con precondizioni/effetti e predicati di dominio.
- Suggerimenti: collegare un frammento di codice minimo la prossima volta; evidenziare un esempio concreto di precondizione negativa.
### 15. Esempio di Pianificazione in Due Stanze: Robot–Chiave–Porta–Scatola
- Costanti: robot, R1, R2, key, door, box.
- Predicati: in(entity, room), locked(door), unlocked(door), room(r), holding(robot, key).
- Stato iniziale: in(robot, R2), in(key, R2), unlocked(door), room(R1), room(R2).
- Obiettivo: locked(door) ∧ in(key, box).
- Azioni:
  - grasp-key: se robot e chiave sono co-localizzati, rimuovi in(key, room) e aggiungi holding(robot, key).
  - lock-door: richiede di tenere la chiave; commuta unlocked in locked.
  - move(x,y): richiede robot in x, door unlocked, room(x), room(y); necessita disuguaglianza x ≠ y per evitare move(x,x) degenere.
  - put-key-in-box: indicata come necessaria ma non completamente definita.
- Commenti di modellazione:
  - I vincoli di disuguaglianza impongono movimenti significativi.
  - La CWA consente di modellare con solo locked o solo unlocked; le azioni vanno riviste di conseguenza.
  - Trucco STRIPS: quando si tiene la chiave, omettere la posizione esplicita della chiave; inferire la posizione da quella del robot.
- Suggerimenti: mostrare l'inattesa move(R2, R2); assegnare un compito per riscrivere le azioni usando solo locked sotto CWA; chiarire l'effetto di grasp-key e mostrare la propagazione implicita della posizione durante move.
### 16. Pianificare come Costruire un Piano
- Piano: sequenza di azioni che porta dallo stato iniziale a uno stato che soddisfi l'obiettivo.
- Distinzione: un obiettivo non è uno stato; più stati possono soddisfare un obiettivo.
- Piano di esempio: da in(robot, R1) e clean(R1), eseguire right; poi suck(R2) per ottenere clean(R1) ∧ clean(R2).
- Suggerimenti: elencare piani alternativi validi per discutere ottimalità e strategie di ricerca; accennare brevemente alla ricerca forward vs. backward.
### 17. Valutare un Modello di Pianificazione Generato da ChatGPT
- Costanti: Left, Right; predicati: dirty, clean, robot_location.
- Azione move: problemi includono variabile R non usata, variabile T non vincolata negli effetti, potenziale istanza move-to-same-location.
- Azione suck: problemi simili con variabili non usate/non vincolate; usa sia dirty sia clean, potenzialmente ridondante sotto CWA.
- Cancellare letterali inesistenti: dipende dall'implementazione; in generale evitare.
- Suggerimenti: presentare azioni corrette; attività di gruppo per identificare e correggere errori; collegare gli errori a regole viste (binding dei parametri, stati-come-insiemi, CWA).
## Domande degli Studenti
1. L'azione move funziona?
- Risposta: Non necessariamente. Ha una variabile R non usata e una variabile T non vincolata nella lista add; l'applicabilità vincola F ma lascia T non collegata, rendendo l'azione mal definita. Anche se F = D, l'azione può essere inutile. Tutti i parametri devono apparire nelle precondizioni per essere vincolati.
2. Vedi qualcosa di strano nell'azione suck?
- Risposta: Stesso problema di move — una variabile R non usata. Concettualmente per il resto va bene. Si è anche notata la ridondanza di usare sia dirty sia clean e il rischio di cancellare letterali non esistenti.
3. “Non c'è nessuna costante implicita dato che abbiamo handempty? handempty non implica una costante ‘hand’?”
- Risposta: Una costante denota un oggetto; handempty è un predicato 0-ario (a valore di verità). L'oggetto “mano” non è rappresentato esplicitamente; si potrebbe modellarlo, ma in questa astrazione handempty rimane un predicato.
4. “Possiamo usare unstack/stack con il tavolo invece di azioni speciali pick-up/put-down?”
- Risposta: Allentare block(y) per consentire y = table introduce effetti in conflitto (es., cancellare e aggiungere clear(table)). Poiché l'ordine di applicazione add/delete dipende dall'implementazione, il comportamento diventa inaffidabile; usare azioni specifiche per il tavolo.
5. “Perché non potevamo prendere il tavolo (o fare unstack dal tavolo)?”
- Risposta: In unstack, x deve essere un blocco e y è il supporto. Si prendono i blocchi, non il tavolo. Consentire y = table porta al conflitto su clear(table) descritto sopra.
6. “Possiamo usare solo uno tra locked/unlocked?”
- Risposta: Sì, sotto CWA. Se usi solo locked, l'assenza implica non locked. Le azioni devono essere riviste di conseguenza per mantenere coerenza; prova la versione a predicato singolo come esercizio.
7. “L'azione move(x,y) è applicabile senza imporre x ≠ y?”
- Risposta: Senza x ≠ y, sia move(R2, R1) sia move(R2, R2) diventano applicabili, creando azioni degeneri. Aggiungi vincoli di disuguaglianza per evitare istanze move-to-same-room.