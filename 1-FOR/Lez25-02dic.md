Data e ora: 2025-12-11 12:32:34
Luogo: [Inserisci luogo]
Corso: [Inserisci nome corso]
## Panoramica
Le lezioni hanno trattato la logistica dei progetti e la struttura dell’esame, l’intuizione geometrica per disuguaglianze valide e di faccia, e la derivazione dei tagli frazionari di Gomory dal rilassamento LP, culminando in un esempio svolto a due vincoli con variabili di slack, identificazione della base e calcoli matriciali. Il processo è stato esteso per generare e interpretare tagli di Gomory (incluso un caso in cui due tagli coincidevano), discutere l’aggiornamento della base dopo l’aggiunta di tagli e introdurre il Branch and Bound tramite un esempio di knapsack. Sono stati anche illustrati aspetti logistici del corso e i piani futuri.
## Contenuti rimanenti
1. Discussione della soluzione all’ultima domanda d’esame (rimandata alla prossima settimana).
2. Sondaggio per decidere se il progetto grande sarà nel primo o secondo appello.
3. Completamento esplicito dei due tagli di Gomory per il primo esempio (setup fatto; tagli non scritti completamente).
4. Calcolo di AB^(-1) per la base 3×3 e derivazione del prossimo taglio (tempo finito per invertire la matrice).
5. Applicazione iterativa dei tagli fino alla soluzione intera ottima (rimandata; focus rimasto sulla generazione dei tagli).
6. Upper bound più sofisticati per esercizi di Branch and Bound (previsti per domani).
7. Esempio applicato a un problema intero generale con interpretazione geometrica (previsto per domani).
8. Approfondimento sui problemi di vehicle routing (ipotizzato per l’ultima settimana).
9. Presentazione generale di euristiche per problemi difficili (alternativa per l’ultima settimana).
## Contenuti trattati
### 1. Logistica dei progetti e struttura dell’esame
- Cartella per l’ultima domanda aperta in ritardo; sarà dato tempo extra e le soluzioni saranno discusse la prossima settimana.
- Progetto piccolo vs. progetto grande:
  - Il progetto grande estende il piccolo; il codice del piccolo può essere riutilizzato.
  - Fare il progetto grande include i punti del test di laboratorio e annulla il voto del piccolo.
  - Se non si è sicuri di fare il grande, procedere col piccolo.
- Sondaggio questa settimana per decidere se il progetto grande sarà nel primo o secondo appello.
- Regole d’esame: l’esame va sostenuto tutto insieme; prima e seconda parte non sono separate.
- Valutazione progetto grande: basata sul codice caricato e sulle risposte date in sede d’esame; non serve relazione.
- Scadenze:
  - Progetto grande: qualche giorno/una settimana prima dell’appello (es. febbraio se l’appello è a febbraio).
  - Progetto piccolo: 10 dicembre (da confermare).
- I progetti sono di gruppo; in sede d’esame ogni studente scrive le proprie risposte.
- Suggerimenti AI:
  - Organizzazione: confermare per iscritto le date esatte (scadenza piccolo e appello grande) e aggiornare subito la piattaforma.
  - Metodi: mostrare le opzioni del sondaggio live o fornire motivazioni per ogni opzione per un voto informato.
  - Chiarezza: riassumere la regola di sovrapposizione/annullamento con un diagramma di flusso.
  - Migliorie: fornire una timeline di esempio e una checklist (formazione gruppi, setup repository, formato consegna).
### 2. Inviluppo convesso e disuguaglianze valide vs. di faccia (intuizione geometrica)
- Task: dati punti interi ammissibili in R², scrivere i vincoli per il loro inviluppo convesso.
- Disuguaglianza valida: mantiene tutti i punti interi ammissibili dal lato ammissibile; ne esistono molte.
- Disuguaglianza di faccia: tocca l’inviluppo convesso lungo una faccia (in R², un lato), cioè almeno due punti.
- Costruite disuguaglianze con gradienti:
  - Gradiente (−1, 1): −x1 + x2 ≤ 3.
  - Gradiente (1, −1): x1 − x2 ≤ 3.
  - Gradiente (1, 1): x1 + x2 ≤ 13.
  - Gradiente (−1, −1): −x1 − x2 ≤ −7.
- Suggerimenti AI:
  - Organizzazione: pre-etichettare punti e gradienti sulla figura per ridurre avanti/indietro.
  - Metodi: sovrapporre assi e segnare intercette per velocizzare la traduzione in disuguaglianze.
  - Chiarezza: giustificare brevemente perché “toccare almeno due punti” indica una faccia in R².
  - Migliorie: aggiungere una routine “metodo delle intercette” come elenco puntato per standardizzare le derivazioni.
### 3. Definizioni e processo per i tagli (tagli frazionari di Gomory)
- Si parte dal rilassamento LP AX = B con non negatività; si considera la partizione di base:
  - Variabili di base XB, non di base XN = 0 nella soluzione LP; si partiziona A in AB e AN.
  - Si moltiplica per AB⁻¹ per ottenere XB + ĀXN = B̄, dove Ā = AB⁻¹AN e B̄ = AB⁻¹B.
- Se una variabile di base è frazionaria, si sceglie la sua riga t e si scrive:
  - XH + Σ Ā_{t j} Xj = B̄_t.
  - Si sostituiscono i coefficienti a sinistra con i pavimenti e si applica ≤; si applica il pavimento anche al termine noto:
    - XH + Σ ⌊Ā_{t j}⌋ Xj ≤ ⌊B̄_t⌋.
- Questa disuguaglianza taglia la soluzione di base frazionaria perché X̄H > ⌊B̄_t⌋ se X̄H è frazionaria.
- Suggerimenti AI:
  - Organizzazione: numerare le regole con formule compatte.
  - Metodi: ripetere gli indici esatti quando si corregge a metà spiegazione.
  - Chiarezza: illustrare perché l’uguaglianza sarebbe inammissibile dopo il pavimento.
  - Migliorie: fornire template e checklist: scegli variabile di base frazionaria → prendi la sua riga → pavimento LHS/RHS → verifica che il taglio violi la soluzione LP.
### 4. Esempio svolto: ILP a due vincoli, slack, identificazione base e calcoli matriciali
- Rappresentazione geometrica:
  - Prima vincolo: intercette x1 = 6 (x2 = 0), x2 = 12/5 (x1 = 0); gradiente circa (2, 5).
  - Secondo vincolo: intercette x1 ≈ 3⅓ (x2 = 0), x2 = 5 (x1 = 0); gradiente (3, 2).
  - Direzione obiettivo (2, 1); soluzione LP ottima x̄ = (10/3, 0).
- Aggiunte slack x3 e x4 per convertire le disuguaglianze in uguaglianze:
  - x3 = 0 sul bordo del primo vincolo; x4 = 0 sul bordo del secondo.
- Base/non base nel punto ottimo:
  - Non base: x2 = 0 e x4 = 0 → indici (2, 4).
  - Base: x1 = 10/3 e x3 da slack → indici (1, 3).
- Costruiti AB, AN e B; calcolato AB⁻¹ (determinante −3) e B̄ = AB⁻¹B:
  - x1 = 10/3; x3 = 16/3 → entrambi frazionari, quindi due tagli di Gomory.
- Suggerimenti AI:
  - Organizzazione: dichiarare AB e AN finali con etichette e ordine variabili coerente (x1, x2, x3, x4).
  - Metodi: correggendo le slack, aggiornare i diagrammi per rafforzare l’accuratezza.
  - Chiarezza: confermare mapping righe/colonne prima di scrivere le matrici.
  - Migliorie: scrivere esplicitamente i due tagli di Gomory per t = 1 (h = 1) e t = 2 (h = 3), mostrare come un taglio elimina x̄; fornire valori numerici di Ā per illustrare i pavimenti.
### 5. Generazione dei tagli di Gomory dal rilassamento LP (esempio a due vincoli)
- Identificati Ā = AB^(-1)AN: 2/3, 1/3, 11/3, −2/3 per le non base x2 e x4.
- Chiarito l’indicizzazione: righe di Ā corrispondono a indici di base; colonne a indici non base.
- Costruito Taglio 1 (riga di x1):
  - x1 + ⌊2/3⌋ x2 + ⌊1/3⌋ x4 ≤ ⌊10/3⌋ ⇒ x1 ≤ 3.
  - Interpretazione geometrica: il bordo x1 = 3 rende il punto frazionario precedente inammissibile.
- Costruito Taglio 2 (riga di x3):
  - x3 + ⌊11/3⌋ x2 + ⌊−2/3⌋ x4 ≤ ⌊16/3⌋ ⇒ x3 + 3x2 − x4 ≤ 5.
  - Eliminati gli slack via x3 = 12 − 2x1 − 5x2 e x4 = 10 − 3x1 − 2x2; si semplifica a x1 ≤ 3 (coincidenza).
- Implicazioni:
  - Aggiungere il taglio introduce una nuova slack (es. x5) e può espandere la base per le iterazioni successive.
  - La generazione ripetuta di tagli porta verso la soluzione intera ma può essere lenta e mal condizionata.
- Suggerimenti AI:
  - Organizzazione: ricapitolare la formula generale di Gomory prima di presentare i valori di Ā.
  - Metodi: usare la sostituzione 2D per visualizzare i tagli quando compaiono slack.
  - Chiarezza: standardizzare la terminologia di indici (B per base, N per non base).
  - Migliorie: presentare un esempio numerico dove i pavimenti danno coefficienti non banali; checklist breve per costruire un taglio.
### 6. Aggiornamento base e setup iterazione dopo aggiunta taglio
- Nuova soluzione LP dopo x1 ≤ 3: (x1, x2) = (3, 1/2), ancora frazionaria.
- Aggiunta disuguaglianza e slack: x1 + x5 = 3, con x5 = 0 nella nuova soluzione.
- Base/non base: B = {1, 2, 3}, N = {4, 5}; notato x3 ≠ 0 dalla geometria.
Prossimi passi: formare AB dalle colonne scelte; calcolare AB^(-1); usare b = (12, 10, 3)^T; derivare Ā per il prossimo taglio.
Focus esame: enfasi sulla generazione dei tagli; i solver LP gestiscono rilassamento ed estrazione della base.
Suggerimenti AI:
  - Organizzazione: mostrare AB e AN con righe/colonne etichettate e dimensioni.
  - Metodi: pre-calcolare AB^(-1) o mostrare uno strumento numerico per completare un ciclo in classe.
  - Chiarezza: dichiarare esplicitamente il vettore b per evitare confusione.
  - Migliorie: fornire un handout con il workflow “post-taglio”.
### 7. Limiti e caratteristiche di convergenza dei metodi dei piani di taglio
- Iterare i tagli porta verso la soluzione intera ma può soffrire di malcondizionamento delle matrici e convergenza lenta.
- Suggerimenti AI:
  - Organizzazione: menzionare brevemente metodi ibridi (tagli + branching) per contestualizzare il cambio di strategia.
  - Metodi: mostrare un indicatore di malcondizionamento o un esempio dove i tagli si bloccano.
  - Chiarezza: aggiungere un esempio concreto di stallo per rafforzare la comprensione.
### 8. Introduzione al Branch and Bound tramite esempio knapsack
- Branch and Bound presentato come alternativa più intelligente all’enumerazione esaustiva.
- Istanza knapsack: capacità 8; valori [4, 1, 3, 1]; pesi [5, 4, 3, 1].
- Costruito albero binario delle decisioni ramificando su x1, x2, x3, x4.
- Potatura per inammissibilità:
  - x1 = 1 e x2 = 1 supera la capacità (5 + 4 > 8) → pota.
  - x1 = 1 e x3 = 1 forza x4 = 0 (5 + 3 = 8) → pota.
- Soluzione ammissibile iniziale tramite euristica corretta a valore 5; upper bound ai nodi calcolati (es. ignorando la capacità per UB grezzo).
- Potatura per dominanza: pota i sottoalberi con UB ≤ incumbent; esplora quelli con UB > incumbent.
- Soluzione finale trovata valore 7 con esplorazione limitata.
- Suggerimenti AI:
  - Organizzazione: esplicitare le correzioni e riassumere gli aggiornamenti dell’incumbent.
  - Metodi: tracciare i nodi con una tabella semplice (variabili fissate, capacità residua, UB, valore corrente, decisione).
  - Chiarezza: dichiarare sempre come si calcolano gli UB rispetto alle decisioni fissate.
  - Migliorie: introdurre UB knapsack frazionario per limiti più stretti; ricapitolare potatura per inammissibilità vs. dominanza.
### 9. Procedura generica Branch and Bound
- Componenti:
  - P: problema originale/sottoproblema; V: valore incumbent; UB dal rilassamento dà soluzione X e flag di ammissibilità.
- Casi:
  - Se X è ammissibile intero per il sottoproblema, restituisci il suo valore (stop).
  - Se il sottoproblema è inammissibile, restituisci −∞.
  - Se UB > incumbent (massimizzazione), ramifica in sottoproblemi disgiunti che coprono l’insieme ammissibile.
- Ricorsione e visita:
  - Applica B&B a ogni sottoproblema; aggiorna l’incumbent se migliorato.
  - L’ordine di visita è indotto dalla ricorsione; ordini alternativi richiedono gestione esplicita (es. code, priorità).
- Collegamento ai tagli:
  - Se il rilassamento LP dà soluzione intera, si può terminare subito “gratis”.
- Suggerimenti AI:
  - Organizzazione: presentare come passi numerati con flowchart.
  - Metodi: mappare i casi generici ai nodi knapsack esplicitamente.
  - Chiarezza: chiarire “incumbent” con esempi in corso d’opera.
  - Migliorie: fornire pseudocodice e note sulla scelta della variabile di branching.
### 10. Logistica del corso e richieste
- La lezione singola della prossima settimana sarà martedì online per problemi di trasporto; streaming disponibile; docente remoto.
- Ultima settimana: opzione approfondimento vehicle routing o presentazione generale di euristiche, a seconda dell’interesse.
- Richiesta: inviare richieste di esercizi su tutto il programma, non solo programmazione lineare intera.
- Suggerimenti AI:
  - Organizzazione: pubblicare logistica e richieste esercizi sulla piattaforma.
  - Metodi: invitare brevi motivazioni per i temi richiesti per personalizzare la pratica.
  - Chiarezza: fornire data/ora e link piattaforma per iscritto.
## Domande degli studenti
1. Scadenza progetto grande:
   - Risposta: la scadenza sarà fissata qualche giorno/una settimana prima dell’appello; se l’appello è a febbraio, la scadenza sarà a febbraio.
2. Scadenza progetto piccolo:
   - Risposta: la scadenza è il 10 dicembre (in attesa di conferma).
3. Interpretazione slack:
   Domanda: “Quindi quando x4 è zero?”
   Risposta: Sì, x4 = 0 indica di essere sul bordo del secondo vincolo; mapping corretto A1 → x3 e A2 → x4, confermando x4 = 0 sul bordo del secondo vincolo.