## Sistemi e Metodi per Big Data e Dati Non Strutturati (SMBUD) — Riassunto Completo
### Panoramica e filosofia del corso
- Nome ufficiale: Sistemi e Metodi per Big Data e Dati Non Strutturati (SMBUD)
- Ambito: Fondamenti di big data, database non relazionali (NoSQL) e calcolo scalabile su grandi dataset
- Approccio: Equilibrio tra teoria e pratica; ogni tecnologia coperta da una lezione teorica e una esercitazione
- Enfasi:
  - Big data: caratteristiche, vantaggi, vincoli, rischi, applicazioni e le Quattro V (Volume, Varietà, Velocità, Veridicità)
  - Database non relazionali: passaggio da relazionale a NoSQL, modelli dati, linguaggi di interrogazione, architetture e considerazioni di design
  - Calcolo scalabile: framework distribuiti (Hadoop e Spark), data warehousing, analytics in streaming e dashboard in tempo reale
- Prerequisiti attesi:
  - Database relazionali: modellazione dati e SQL di base
  - Programmazione di base: controllo di flusso; Python preferito (C/C++ accettati)
- Materiali e piattaforme didattiche:
  - WeBeep ospita slide (teoria completa già caricata), calendario, riferimenti, registrazioni e trascrizioni esercitazioni
  - Materiali esercitativi rilasciati progressivamente; consigliato un libro introduttivo; libro specifico in preparazione
### Calendario, streaming e logistica
- Formato: Lezioni e esercitazioni guidate da docente e assistenti; lezioni registrate e disponibili per revisione
- Streaming: Disponibile la prima settimana (forse anche la seconda) per motivi logistici; poi interrotto; registrazioni sempre disponibili
- Logistica aula: Segnalato inizialmente un posto extra; streaming utile per visibilità/logistica
- Calendario:
  - Nessuna lezione: 23 ottobre (Graduation Day)
  - Tre quiz/test in corso programmati (date indicative sul calendario)
### Team didattico e collegamento con l'industria
- Docente: Prof. Marco Brambilla (Informatica, Edificio 20)
  - Ruoli: Docente del corso; Direttore dei corsi di laurea triennale e magistrale in Informatica; Responsabile del Data Science Lab
- Assistenti:
  - Andrea Tocchetti (Postdoc, Data Science Lab)
  - Riccardo Campi (Dottorando, Data Science Lab)
- Data Science Lab: ~15 membri tra dottorandi/postdoc; forte collegamento con l'industria tramite startup in tecnologie web, AI/ML, edge e embedded AI
### Protocollo di comunicazione e contatti
- Contatto principale: Email a docente e assistenti
- Tag richiesto: Includere “SMBUD” nell'oggetto o nel corpo per garantire filtraggio e gestione tempestiva
- Volume email: Il docente riceve 150–300 email/giorno; il tag SMBUD è fondamentale
- Ricevimento: Edificio 20
- Nota sul ruolo di direttore: Contattare il docente come Direttore indica escalation; usare solo se necessario
### Logistica didattica e struttura delle lezioni
- Sequenza:
  1) Ripasso database relazionali (con almeno una esercitazione)
  2) Modelli dati e architetture NoSQL
  3) Calcolo su larga scala (Hadoop: HDFS e componenti selezionati; Spark con Python)
  4) Data warehousing e analytics in tempo reale
- Esercitazioni:
  - Due sottogruppi paralleli (divisione alfabetica), stesso giorno/orario e contenuti identici
  - Docenti: Tocchetti e Campi
  - Vincolo attuale: Trovata solo una aula; seconda aula in attesa
- Risorse piattaforma:
  - WeBeep: materiali, calendario (argomenti e docenti), registrazioni/trascrizioni, appunti informali, riferimenti
  - Visibilità: Se i materiali non sono visibili, iscrizione incompleta o problema tecnico—contattare per risoluzione
### Tecnologie e sistemi Big Data trattati
- Categorie NoSQL:
  - Database a grafo (copertura approfondita)
  - Database a documenti (copertura approfondita)
  - Database colonnari
  - Key-value store
  - Database per ricerca/informazione
  - Database vettoriali
  - Feature store
  - Semantica: breve menzione
- Considerazioni architetturali:
  - Gestione dati distribuita su larga scala (on-premises o cloud)
  - Trade-off di design tra modelli dati, linguaggi di interrogazione e pattern di integrazione
- Calcolo scalabile:
  - Hadoop: Focus su HDFS e componenti pertinenti
  - Spark: programmazione pratica in Python e processing distribuito
### Valutazione, esami e progetti
- Componenti:
  - Quiz in corso: svolti in ambiente sicuro sui laptop degli studenti; contribuiscono al punteggio
  - Progetto opzionale: lavoro pratico; dettagli dopo la copertura iniziale dei contenuti
  - Esame finale:
    - Teoria: mix di vero/falso, scelta multipla e domande aperte brevi; possibile aggiunta orale in base alla valutazione
    - Pratica: scrittura di query per sistemi/linguaggi trattati; programmazione Spark o simili
- Logistica esame:
  - Esempi di esami passati forniti come riferimento (non vincolanti; tipologia esercizi può variare)
  - Valutazione su tecnologie e linguaggi insegnati; creatività oltre il corso non richiesta
  - Esame finale su laptop in valutazione; decisione attesa entro 1–2 settimane
- Tempistiche:
  - Progetto e dettagli esame comunicati entro 1–2 settimane dai primi argomenti
### Opportunità di ricerca e tesi
- Focus Data Science Lab:
  - Data science
  - Machine learning spiegabile e trasparente
- Opportunità:
  - Sessione a fine corso presenterà proposte di tesi e temi di ricerca
  - Studenti interessati a tesi/progetti dovranno contattare più avanti nel semestre o nel prossimo
### Scenario di mercato Big Data e considerazioni strategiche
- Tre pilastri delle soluzioni big data:
  1) Strumenti infrastrutturali: storage, raccolta, gestione dati locale/distribuita/cloud
  2) Strumenti di data analytics: ML, AI, statistica, esplorazione, visualizzazione
  3) Livello applicativo: soluzioni rivolte al cliente
- Provider full-stack:
  - Principali attori: Amazon, Google, IBM, Microsoft, Oracle, SAP, ecc.
  - Strategia: stack integrati tra infrastruttura, analytics e applicazioni
  - Vantaggi per il cliente: gestione unica, sistemi integrati, delega privacy/rischi, sviluppo più rapido
  - Rischi: lock-in del fornitore, qualità disomogenea tra componenti, amplificazione dei punti di fallimento
- Open source:
  - Centrale nel big data; molte piattaforme commerciali basate su OSS
  - Pro: flessibilità, risparmio, profondità ecosistema
  - Contro: deployment/supporto/implementazione autogestiti
- Fonti dati:
  - Dati aziendali interni e provider esterni (salute, sport, IoT, social, finanza, telecom, ecc.)
  - Ampia disponibilità di dataset acquistabili
### Modello di creazione del valore: ciclo virtuoso guidato dai dati
- Fasi:
  - Raccolta/archiviazione dati (infrastruttura)
  - Analisi (data science, statistica, ML/AI)
  - Generazione valore (efficienza, risparmio, aumento vendite/clienti, ROI misurabile)
- Principio:
  - Il big data deve portare a risultati concreti—raccogliere/analizzare senza applicazione spreca risorse
  - Ciclo continuo: l'impatto genera nuovi dati → ulteriore analisi → valore raffinato
### Caso studio: Netflix e Big Data nella pratica
- Evoluzione:
  - Era DVD: ottimizzazione logistica per consegna in ~20 ore a livello nazionale; previsione stock/domanda con dati limitati (eventi di visione, rating, metadati di base); competizione con videonoleggi
  - Era streaming: log ricchi di interazioni (click, scroll, play/pausa, rewind) permettono profilazione e personalizzazione granulare
- Meccaniche di personalizzazione:
  - ~70.000 generi alternativi creati tramite clustering su feature e comportamento
  - 270M+ utenti clusterizzati per gusti e comportamento; matching cluster utenti-cluster contenuti per personalizzare homepage e liste → maggiore soddisfazione e retention
- Strategia contenuti guidata dai dati:
  - Esempio: investimento diritti House of Cards (~$20–$30M) guidato da analisi preferenze, affinità regista/attori, performance comparabili e posizionamento di mercato
- Ottimizzazione interfaccia:
  - Test A/B/multivariati; artwork/preview adattivi per massimizzare engagement e ridurre churn
- Rischi:
  - Privacy; filter bubble; possibile manipolazione attenzione
  - Trade-off aziendali: over-ottimizzazione riduce diversità catalogo; mitigazione tramite algoritmi di diversificazione
### Rischi chiave e mitigazione: bias da personalizzazione
- Personalizzazione a lungo termine restringe l'esposizione, favorendo "bolle di realtà" e polarizzazione sociale
- Influenza i gusti utente su 5–10 anni; riduce la diversità di utilizzo del catalogo
- Mitigazione: strategie di diversificazione intenzionale nei sistemi di raccomandazione per introdurre varietà controllata e contrastare l'overfitting
### Fattori economici abilitanti ed evoluzione infrastrutturale
- Costi storage:
  - Anni '90: ~$500–$600/GB
  - ~10 anni fa: ~$0,02/GB
  - Oggi (consumer): verso ~$0,001/GB
- Connettività:
  - Circa 2000: ~$1.200/mese per banda larga base, disponibilità limitata
  - Oggi: ~€16–€18/mese banda larga consumer, accessibile globalmente
- Risultato: storage/connettività più economici → più dispositivi/contenuti/utenti → più investimenti → tecnologia più veloce → più uso dati; mercato scala a decine/centinaia di zettabyte annui
- Contesto storico:
  - Vincoli passati rendevano lo streaming HD impossibile; servivano DVD
  - Attuale ubiquità streaming HD resa possibile dai miglioramenti infrastrutturali
### Gap operativi e nuovo pensiero sui dati
- Gap: la raccolta dati cresce esponenzialmente; capacità di analisi/decisione in ritardo → valore nascosto e inefficienze
- Requisiti: nuovi strumenti e approcci per estrarre valore su larga scala; piattaforme flessibili per volume, varietà, velocità, veridicità
- Persone prima della tecnologia:
  - Riqualificazione organizzativa e cambiamenti culturali necessari oltre l'adozione di strumenti
- Cambiamenti di paradigma:
  - Da campionamento ad analisi su tutta la popolazione (es. Netflix logga tutti gli utenti; droni coprono tutto)
  - Da ipotesi a esplorazione data-first (es. genomica bottom-up)
  - Da report mensili statici a dashboard in tempo reale con freschezza al minuto e drill-down interattivo
### Conclusioni
- Il corso inizia con supporto strutturato (registrazioni, streaming iniziale limitato) e passa a focus in presenza
- Forte collegamento con ricerca attiva e industria tramite Data Science Lab
- Protocollo chiaro di tagging email SMBUD per comunicazione efficiente
- Percorsi tesi e temi di ricerca chiariti verso fine corso
### Prossimi passi e attività da svolgere
- Logistica corso:
  - Usare “SMBUD” in tutte le email relative al corso per filtraggio
  - Accedere alle lezioni registrate per ripasso o recupero
  - Usare lo streaming nella prima settimana (forse seconda), poi pianificare la disattivazione
  - Nota sull'ufficio docente: Edificio 20
  - Confermare assegnazione aula esercitazioni; trovare seconda aula per sessioni parallele
  - Pubblicare divisione finale gruppi esercitazione e assegnazione aule su WeBeep
  - Assicurarsi che il calendario WeBeep mostri tutte le sessioni con argomenti e docenti assegnati
  - Caricare progressivamente materiali esercitazione; fornire registrazioni e trascrizioni
  - Svolgere tre quiz in corso come da calendario; rilasciare esempi esami passati con disclaimer non vincolante
  - Annunciare decisione su formato esame finale (carta vs. laptop) entro 1–2 settimane
  - Comunicare dettagli precisi su progetto ed esame dopo i primi argomenti (tempistiche 1–2 settimane)
  - Giorno senza lezione: 23 ottobre (Graduation Day)
- Sequenza di apprendimento:
  - Prepararsi a un rapido ripasso database relazionali, poi NoSQL, Hadoop/Spark e warehousing
- Ricerca e tesi:
  - Partecipare alla sessione finale di corso su proposte tesi e temi laboratorio
  - Contattare più avanti nel semestre o nel prossimo per opportunità tesi/progetto
- Pratica strategica:
  - Implementare diversificazione nelle raccomandazioni per evitare bolle di realtà
  - Dare priorità alla pulizia dati (pianificare ~70% sforzo in progetti AI/ML)
  - Progettare piattaforme flessibili per volume, varietà, velocità, veridicità
  - Avanzare la riqualificazione per passare da paradigmi relazionali/statici a big data, real-time e data-driven
  - Passare da campionamento ad analisi su tutta la popolazione dove possibile
  - Stabilire analytics in streaming real-time e dashboard interattive
  - Pianificare logistica dati su scala estrema, inclusi trasferimenti fisici quando le reti sono impraticabili