## Docenti e Assistenti del Corso
*   **Docente (Relatore):** Peter Lucaranzi (non Francesco Amigoni).
*   **Altro Docente:** Francesco Amigoni.
*   **Lezione Ospite:** La Professoressa Viola Chiafonati terrà una lezione sugli aspetti filosofici e analitici dell'IA.
*   **Assistenti (TAs):** Alberto e Paolo condurranno le esercitazioni per approfondire i temi del corso.
    *   Le esercitazioni servono per l'apprendimento e non sono copie dirette dei problemi d'esame.
## Logistica e Struttura del Corso
*   **Materiale del Corso:** Tutti i materiali (slide, codice, esami di esempio) sono disponibili sulle pagine dei docenti (WeBip). Il contenuto è identico sia sulla pagina di Peter che su quella di Francesco.
*   **Lezioni:**
    *   Non è previsto lo streaming live.
    *   Le lezioni sono registrate e saranno caricate, di solito entro la settimana successiva.
    *   Il corso è diviso in due sezioni con docenti condivisi per argomenti specifici. Questo permette agli studenti di seguire la lezione di un'altra sezione se perdono la propria.
*   **Orario:**
    *   **Sezione 1:** Martedì pomeriggio e venerdì mattina.
    *   **Sezione 2:** Lunedì mattina e venerdì pomeriggio.
## Programma e Argomenti del Corso
*   **IA Classica:** Il corso introduce i concetti di IA classica.
    *   **Argomenti di Peter Lucaranzi:** Spazio degli stati, ricerca non informata e informata, ricerca avversaria, problemi di soddisfacimento vincoli.
    *   **Argomenti di Francesco Amigoni:** Logica, fondamenti ed etica dell'IA.
*   **Conoscenza dei Giochi Richiesta:** Gli studenti devono conoscere le regole di giochi semplici come "Breakthrough" (versione semplificata della dama), poiché potrebbero essere usati negli esami.
## Esame e Valutazione
*   **Formato:** Un unico esame scritto, da sostenere in presenza.
    *   **Durata:** 1,5 ore.
    *   **Struttura:** Tipicamente 4-5 domande, per un totale di 30 punti.
    *   **Voto:** Un punteggio di 32 punti corrisponde a "cum laude".
*   **Iscrizione all'Esame:**
    *   Gli studenti **devono** iscriversi all'esame entro la scadenza ufficiale.
    *   Iscriversi anche solo un minuto dopo la scadenza comporta l'impossibilità di sostenere l'esame.
*   **Dopo l'Esame:**
    *   La discussione degli elaborati avviene solo in presenza.
    *   I docenti possono convocare gli studenti che superano la prova scritta per un eventuale esame orale.
    *   Non esistono altri metodi (es. progetti, domande extra) per superare il corso o aumentare il voto.
*   **Lezioni Speciali:**
    *   Una lezione finale sarà dedicata alla simulazione di un esame.
    *   La lezione della Professoressa Viola Chiafonati su etica/filosofia dell'IA fa parte del materiale d'esame. Gli argomenti specifici cambiano ogni anno.
## Materiali di Studio e Risorse
*   **Libro di Testo:** La fonte principale del materiale del corso. È considerato la "Bibbia".
*   **Guida allo Studio:** È disponibile una versione aggiornata (v3). Dettaglia gli argomenti di ogni lezione e li collega ai capitoli e sezioni del libro di testo. Alcune parti sono designate per lo studio autonomo e sono materiale d'esame obbligatorio.
*   **Slide:** Servono da supporto al libro di testo. Possono essere aggiornate, quindi è importante scaricare sempre la versione più recente.
*   **Codice Sorgente:** Disponibile per gli studenti per sperimentare.
*   **Esami di Esempio:** Esami passati con soluzioni sono disponibili online.
## Comunicazione e Processo di Studio
*   **Annunci:** Tutti gli annunci del corso sono pubblicati su WeBeep. Le email non vengono usate per comunicazioni ufficiali.
*   **Domande:** Tutte le domande devono essere postate sul forum WeBeep, non inviate via email. Questo garantisce chiarezza e permette a tutti gli studenti di beneficiare della discussione.
*   **Workflow di Studio Consigliato:**
    1.  Seguire la lezione o guardare la registrazione.
    2.  Studiare le sezioni corrispondenti sul libro di testo.
    3.  Sperimentare con il codice sorgente e i notebook forniti.
    4.  Rivedere i problemi d'esame correlati.
    5.  Inviare eventuali domande sul forum.
## Introduzione all'Intelligenza Artificiale
*   **Definizione di IA:** La definizione di IA non è fissa; cambia nel tempo e varia tra le diverse comunità scientifiche.
    *   **Esempio:** Negli anni '90, l'IA era associata ai programmi di scacchi; oggi è spesso associata a modelli come ChatGPT.
*   **Doppia Natura dell'IA:** L'IA è un crocevia tra:
    *   **Scienza:** Indagare la natura dell'intelligenza.
    *   **Ingegneria:** Costruire sistemi che mostrano intelligenza.
*   **Contesto Storico e Cambio di Paradigma:** Il termine "intelligenza artificiale" è stato coniato in una proposta del 1955 per un workshop a Dartmouth nel 1956. Prima della fine degli anni '90, l'IA era vista come un insieme di algoritmi disparati. Intorno al 2000, la comunità IA si è spostata verso un paradigma unificante: l'**agente razionale**. Questo concetto offre un quadro universale per vedere qualsiasi sistema IA, sia esso un robot fisico o un programma software. Il campo si è poi ampliato includendo rappresentazione della conoscenza, machine learning, reti neurali, deep learning e IA generativa.
*   **Quattro Approcci all'IA:**
    1.  **Pensare come un umano (Scienze Cognitive):** Creare sistemi che modellano il processo di pensiero umano.
    2.  **Pensare razionalmente (Logica):** Costruire sistemi basati su ragionamenti matematicamente solidi e dimostrabili.
    3.  **Agire come un umano (Test di Turing):** Sviluppare sistemi che imitano il comportamento umano così bene da essere indistinguibili da una persona.
    4.  **Agire razionalmente (Agenti Razionali):** **Questo è il focus del corso.** L'obiettivo è creare agenti che agiscono per ottenere il miglior risultato possibile in base a una misura di performance, senza dover imitare il pensiero o il comportamento umano.
## Paradigma dell'Agente Razionale
*   **Definizione Base:** Un agente razionale è un sistema che interagisce con un ambiente. Usa sensori per percepire l'ambiente e attuatori per compiere azioni.
*   **Obiettivo:** Per ogni sequenza di percezioni, un agente razionale seleziona un'azione che si prevede **massimizzi la sua misura di performance**. Questa decisione si basa sulle evidenze fornite dalla sequenza di percezioni e su eventuali conoscenze incorporate.
*   **Principio Chiave:** Il focus è sull'ottenere **comportamento razionale** e il risultato che ne deriva. L'implementazione specifica (es. semplici regole `if-then` o una rete neurale complessa) è secondaria rispetto all'efficacia del risultato. L'obiettivo è il comportamento intelligente, non necessariamente imitare i processi cognitivi umani.
    *   **Esempio:** Un semplice programma di regole `if-then` per un aspirapolvere è considerato agente intelligente se pulisce la stanza in modo razionale ed efficace. L'intelligenza è nel comportamento, non nella complessità del codice.
## Caratterizzazione dell'Ambiente di Compito (PEAS)
*   **Framework:** Un ambiente di compito è definito tramite il framework PEAS:
    *   **P**erformance Measure: I criteri di successo (es. apprendimento dello studente, velocità, sicurezza).
    *   **E**nvironment: Il contesto in cui opera l'agente (es. insieme di studenti, scacchiera, autostrada).
    *   **A**ctuators: I meccanismi che l'agente usa per agire (es. mostrare esercizi, muovere un pezzo, sterzare un'auto).
    *   **S**ensors: I meccanismi che l'agente usa per percepire (es. input da tastiera, una telecamera sulla scacchiera, lidar).
*   **Esempio: Tutor di Inglese Automatico**
    *   **Performance:** Massimizzare i risultati di apprendimento degli studenti.
    *   **Environment:** L'insieme degli studenti che usano il sistema.
    *   **Actuators:** Mostrare esercizi, fornire feedback, sintesi vocale.
    *   **Sensors:** Inserimenti da tastiera, input vocale dello studente.
*   **Tipi di Ambienti:** La natura dell'ambiente determina la complessità dell'agente richiesto.
    *   **Osservabilità:** Completamente osservabile (es. scacchi) vs. Parzialmente osservabile (es. auto autonoma con punti ciechi).
    *   **Agenti:** Agente singolo vs. Multi-agente.
    *   **Determinismo:** Deterministico (es. mossa negli scacchi) vs. Stocastico (es. lancio di dadi, probabilità note) vs. Non deterministico (probabilità degli esiti sconosciute).
    *   **Temporalità:** Episodico (azioni indipendenti) vs. Sequenziale (l'azione corrente influenza le successive).
    *   **Cambiamento:** Statico (l'ambiente non cambia mentre l'agente pensa) vs. Dinamico (l'ambiente cambia indipendentemente).
    *   **Tipo di Dati:** Discreto (es. scacchi) vs. Continuo (es. guida).
    *   **Conoscenza:** Nota (le regole dell'ambiente sono note) vs. Sconosciuta.
## Gerarchia delle Architetture degli Agenti
L'obiettivo ingegneristico è scegliere l'architettura più semplice che possa risolvere il problema. La complessità aumenta nel seguente ordine, e un'architettura più semplice non può implementare una più complessa.
### 1. Agente Riflesso Semplice
*   **Funzione:** Agisce solo sulla percezione corrente, ignorando la storia delle percezioni.
*   **Meccanismo:** Usa semplici regole condizione-azione (es. `if [condizione], then [azione]`).
*   **Esempi:** Primi modelli Roomba, automazione domestica smart tramite IFTTT ("If This, Then That").
*   **Limite:** Altamente inefficiente in ambienti non completamente osservabili. Può bloccarsi in loop e non apprende dalle azioni passate.
### 2. Agente Riflesso Basato su Modello
*   **Funzione:** Mantiene uno stato interno per tracciare aspetti del mondo che non può vedere. Ha un modello di come funziona il mondo.
*   **Meccanismo:** Usa lo stato interno, un "modello di transizione" (come evolve il mondo) e un "modello sensore" per capire la situazione.
*   **Esempio:** Un Roomba moderno che costruisce una mappa della stanza per tracciare le aree pulite e non pulite.
*   **Limite:** Pur comprendendo meglio l'ambiente, non ha un obiettivo specifico di lungo termine o "goal".
### 3. Agente Basato su Obiettivi
*   **Funzione:** Oltre al modello, questo agente ha informazioni esplicite di "goal". Sceglie le azioni per raggiungere quell'obiettivo.
*   **Meccanismo:** Può usare ricerca e pianificazione per trovare un percorso verso lo stato obiettivo.
*   **Esempio:** Un navigatore GPS che trova la rotta verso una destinazione o un'IA che pianifica mosse per vincere.
*   **Limite:** Gli obiettivi sono binari (raggiunti o no). Questa architettura fatica con situazioni che richiedono compromessi o preferenze (es. percorso più veloce vs più sicuro).
### 4. Agente Basato su Utilità
*   **Funzione:** Generalizza l'approccio basato su obiettivi usando una funzione di "utilità" che assegna un punteggio numerico agli stati, rappresentando il grado di desiderabilità.
*   **Meccanismo:** Sceglie l'azione che porta allo stato con la massima utilità attesa, permettendo di gestire obiettivi in conflitto e fare compromessi razionali.
*   **Esempio:** Un sistema di guida autonoma che deve bilanciare obiettivi multipli come velocità, sicurezza ed efficienza.
*   **Superiorità:** Un agente basato su utilità può implementare un agente basato su obiettivi assegnando alta utilità agli stati obiettivo e bassa agli altri. Il contrario non è possibile.
## 📅 Prossimi Appuntamenti
- [ ] Attivare le notifiche su WeBip per annunci e aggiornamenti materiali del corso.
- [ ] Usare il forum WeBeep per tutte le domande sul corso invece di inviare email.
- [ ] Se necessario, prendere appuntamento per ricevimento invece di usare le ore di ricevimento formali.
- [ ] **Iscriversi all'esame prima della scadenza.** È possibile annullare l'iscrizione se i piani cambiano.
- [ ] Scaricare l'ultima versione delle slide e della Guida allo Studio (attualmente v3).
- [ ] Studiare i capitoli richiesti del libro di testo per il materiale introduttivo (Capitoli 1 e 2) e sulla caratterizzazione dei compiti (PEAS).
- [ ] Imparare le regole dei giochi specificati (es. Breakthrough) poiché potrebbero comparire nelle domande d'esame.
- [ ] Ricordare che la lezione della Professoressa Viola Chiafonati sugli aspetti filosofici è materiale d'esame obbligatorio.
- [ ] Notare che una lezione finale sarà dedicata alla simulazione d'esame.
- [ ] Aspettarsi che le registrazioni delle lezioni siano pubblicate la settimana dopo la lezione live.
- [ ] Prepararsi a domande d'esame che richiedono di classificare ambienti di compito in base alle loro proprietà (es. osservabile, deterministico, statico).