# SOFTWARE ENGINEERING 2 - LEZIONE 1: INTRODUZIONE AL CORSO
*Corso della Prof.ssa Elisabetta Di Nitto - Politecnico di Milano*

## 📋 INFORMAZIONI TECNICHE
- **Video:** SE2 20250917
- **Data elaborazione:** 25 settembre 2025, ore 13:32:45
- **Sistema:** Ultra GPU-Optimized
- **Modello Whisper:** large-v3 su CUDA
- **Segmenti audio:** 2196
- **Word timestamps:** 19776
- **Frame estratti:** 300
- **Contenuto matematico:** 272 frame

---

## 🗂️ INDICE DELLA LEZIONE

### PARTE I: INFORMAZIONI LOGISTICHE DEL CORSO
- **[00:33 - 03:00]** Introduzione e organizzazione streaming
- **[02:25 - 03:30]** Divisione in tre classi parallele
- **[03:33 - 04:00]** Docenti e assistenti per esercitazioni
- **[08:53 - 15:27]** Materiali didattici e libro di testo

### PARTE II: OBIETTIVI E PROGRAMMA DEL CORSO
- **[04:00 - 08:53]** Obiettivi: focus sui sistemi complessi su larga scala
- **[05:01 - 07:41]** Argomenti: cicli di vita, gestione progetti, linguaggi di specifica
- **[07:41 - 08:43]** Analisi dei requisiti, progettazione, validazione e verifica

### PARTE III: MODALITÀ D'ESAME E VALUTAZIONE
- **[16:03 - 34:07]** Sistema di valutazione flessibile (2 parti + homework opzionale)
- **[17:15 - 27:25]** Opzioni: esame scritto, progetto R&DD, progetto ricerca
- **[34:11 - 50:07]** Dettagli progetti R&DD e Implementation & Testing
- **[50:07 - 58:28]** Homework di valutazione documenti e modalità esame scritto

### PARTE IV: CONTENUTI INTRODUTTIVI
- **[68:28 - 74:52]** Importanza dell'ingegneria del software
- **[69:25 - 73:06]** Esempio: outage mondiale CrowdStrike del 19 luglio 2024
- **[73:58 - 76:05]** Definizione di Software Engineering
- **[76:22 - 82:25]** Competenze richieste per un software engineer
- **[82:25 - 90:29]** Caratteristiche del prodotto software e qualità

---

## 📚 CONTENUTO DELLA LEZIONE

### 🎯 PARTE I: INFORMAZIONI LOGISTICHE {#informazioni-logistiche}
*Timing: 00:33 - 15:27*

#### Introduzione al Corso
**[00:33 - 01:05]**
> "Buongiorno. Questa è la prima lezione di Software Engineering 2. In questa prima lezione ci concentreremo sulle regole d'esame, gli obiettivi del corso, ecc."

**Organizzazione streaming:**
- Oggi streaming aperto per tutti (prima lezione)
- Domani streaming aperto (lezione eccezionale con Prof. Rossi)
- Dalle prossime lezioni: streaming chiuso per incoraggiare la partecipazione in aula
- Registrazioni sempre disponibili per tutti

#### Divisione in Classi
**[02:25 - 03:30]**
> "Come sapete, abbiamo tre classi diverse per Software Engineering 2. Quindi siamo una delle classi, probabilmente [...] il vostro cognome inizia con una lettera tra A e forse E o G."

**Struttura organizzativa:**
- **3 classi parallele** con stesso programma
- **Docenti:** Elisabetta Di Nitto (A-D), Matteo Rossi (E-O), Matteo Camilli (P-Z)
- **Website comune** e materiali condivisi
- **Esercitazioni:** Simone Reale (dottorando in quantum software engineering)

#### Eccezione per Domani
**[10:00 - 10:51]**
> "Domani devo partire per una conferenza, quindi il Professor Rossi mi sostituirà e insegnerà a entrambe le classi. La lezione sarà nella nostra aula, che è la 5.03, e lo streaming sarà aperto per tutti."

**Dettagli:**
- **Aula:** 5.03 (230 posti)
- **Streaming:** Aperto per evitare sovraffollamento
- **Link:** Webex room del Prof. Rossi (sarà inserito nel programma)

### 📖 PARTE II: MATERIALI E OBIETTIVI {#materiali-obiettivi}
*Timing: 04:00 - 15:27*

#### Obiettivi del Corso
**[04:00 - 04:44]**
> "Gli obiettivi del corso sono riassunti qui. L'idea è darvi una panoramica di principi e tecniche dell'ingegneria del software. In particolare, il nostro focus sarà sui sistemi su larga scala."

**Focus principale:**
- **Sistemi complessi** e progetti su larga scala
- **Differenza** dai corsi base di SE (sistemi piccoli)
- **Soluzioni** per supportare lo sviluppo di sistemi più grandi

#### Argomenti Trattati
**[05:01 - 08:43]**

**Panoramica generale:**
- Cicli di vita del software
- **Gestione progetti** (2 lezioni) e stima complessità
- **Standard** come baseline per attività professionali
- **Linguaggi di specifica:** UML (semi-formale) e Alloy (formale con tool di analisi)

**Analisi e progettazione:**
- **Analisi dei requisiti:** identificazione, analisi, documentazione
- **Progettazione software:** approcci multipli e stili di riferimento
- **Validazione e verifica:** attività per garantire qualità

#### Materiali Didattici
**[13:40 - 15:27]**

**Nuovo libro di testo:**
- **Titolo:** Specificamente progettato per questo corso
- **Pubblicazione:** Inizio ottobre 2025 (3 ottobre)
- **Contenuto:** Include capitoli non insegnati per completezza
- **Versione online:** In fase di negoziazione con biblioteca e editore

**Altri materiali:**
- **Libri suggeriti** disponibili in formato elettronico
- **Slides e materiali** su WeBeep
- **Revisioni continue** del materiale per ogni lezione

### ⚖️ PARTE III: MODALITÀ D'ESAME {#modalita-esame}
*Timing: 16:03 - 58:28*

#### Filosofia di Valutazione
**[16:41 - 17:08]**
> "La nostra filosofia è che vogliamo cercare di accomodare i gusti e le esigenze di tutti, e quindi abbiamo un gran numero di opzioni che sono riassunte da questa tabella."

#### Schema di Valutazione
**[17:15 - 20:32]**

**Struttura base:**
- **2 parti obbligatorie** che devono essere combinate
- **Homework opzionale** (2 punti aggiuntivi)
- **Parti indipendenti:** possibili in sessioni diverse

**Parte 1 (16 punti max, min 8):**
- Written Exam 1
- Research Project (alternativo)

**Parte 2 (14 punti max, min 7):**
- R&DD Project (Requirement Analysis & Design Document)
- Written Exam 2 (alternativo)

#### Raccomandazioni del Docente
**[21:25 - 23:56]**
> "La mia forte raccomandazione è di fare almeno questo progetto qui, perché imparare come fare l'analisi dei requisiti [...] è qualcosa che possiamo fare solo per tentativi ed errori e solo attraverso un'esperienza concreta."

**Motivazioni:**
- **R&DD Project fortemente consigliato** per esperienza pratica
- **Written Exam 2** adatto solo a professionisti con esperienza
- **Apprendimento pratico** essenziale per competenze di progettazione

#### Progetto R&DD (Requirement Analysis & Design Document)
**[34:11 - 41:15]**

**Obiettivi:**
- Mettere in pratica gli approcci appresi
- **Attività di gruppo** (fino a 3 persone)
- Singoli consentiti ma sconsigliati

**Supporto didattico:**
- **4 sessioni di esercitazione** dedicate
- Incontri individuali disponibili
- **Homework collegato** per preparazione

**Processo di apprendimento:**
**[35:16 - 35:57]**
> "Come sperimenterete lavorando in gruppo, un aspetto importante che imparate sul campo è che dovete coordinarvi con gli altri e assicurarvi che il gruppo nel suo complesso abbia successo."

#### Implementation & Testing (Progetto Completo)
**[36:51 - 41:15]**

**Caratteristiche:**
- **Solo per gruppi** di 2-3 studenti (stesso gruppo R&DD)
- **Prototipo funzionante** con alcune funzionalità
- **Testing di accettazione** tra gruppi diversi
- **Documenti viventi** che evolvono nel processo

**Vantaggi:**
**[40:59 - 41:14]**
> "Fare questo progetto [...] ha il vantaggio di permettervi di attraversare veramente tutto il processo e ottenere un'esperienza di cosa significa partire dalla concezione dell'idea fino all'implementazione effettiva."

#### Dettagli Punteggi e Vincoli
**[29:18 - 33:26]**

**Composizione punteggio:**
- **Parte 1:** 16 punti (min 8)
- **Parte 2:** 14 punti (min 7) 
- **Homework:** 2 punti
- **Totale:** 32 punti (30 e lode se ≥31)

**Vincoli importanti:**
- **Punteggio minimo** per ogni parte
- **Totale ≥18** per superare l'esame
- **Parti indipendenti:** mantenimento punteggi separato

### 💻 PARTE IV: CONTENUTI INTRODUTTIVI {#contenuti-introduttivi}
*Timing: 68:28 - 90:29*

#### Importanza dell'Ingegneria del Software
**[68:28 - 69:51]**

**Pervasività del software:**
- Software ovunque nella società moderna
- Dipendenza crescente per servizi essenziali
- **Impatto critico** su sicurezza e economia

#### Caso Studio: Outage CrowdStrike
**[69:25 - 73:06]**

**Evento:** 19 luglio 2024
> "CrowdStrike ha causato il crash di sistemi Windows aziendali in tutto il mondo"

**Impatti documentati:**
- **Trasporti:** Cancellazione voli, problemi ferroviari
- **Sanità:** Sistemi ospedalieri offline
- **Servizi finanziari:** Interruzioni bancarie
- **Emergenze:** Sistemi 911 non funzionanti

**Lezione appresa:**
Dimostrazione concreta dell'importanza della qualità del software e delle metodologie di ingegneria del software.

#### Definizione di Software Engineering
**[73:58 - 76:05]**

**Definizione formale:**
> "Campo dell'informatica dedicato alla progettazione, sviluppo, testing e manutenzione di applicazioni software"

**Componenti chiave:**
- **Progettazione sistematica**
- **Sviluppo controllato**
- **Testing rigoroso**  
- **Manutenzione continua**

#### Competenze del Software Engineer
**[76:22 - 82:25]**

**[76:22 - 76:47]**
> "Le competenze di programmazione non sono sufficienti"

**Categorie di competenze:**

**1. Tecniche:**
- Programmazione (base necessaria ma non sufficiente)
- Architetture software
- Database e sistemi distribuiti
- Testing e debugging

**2. Gestione progetti:**
- Pianificazione e scheduling
- Stima costi e tempi
- Gestione risorse
- Controllo qualità

**3. Cognitive:**
- Problem solving
- Pensiero sistemico
- Astrazione e modellazione

**4. Aziendali:**
- Comunicazione con stakeholder
- Comprensione del business
- Gestione requisiti cliente

#### Il Prodotto Software
**[80:30 - 90:29]**

**Caratteristiche distintive:**
**[80:30 - 85:55]**
- **Intangibile:** Non fisico, difficile da visualizzare completamente
- **Malleabile:** Facilmente modificabile (illusione di semplicità)
- **Non si usura:** Non degrado fisico ma evoluzione funzionale
- **Sviluppo custom:** Ogni sistema è unico

**Qualità del prodotto software (ISO/IEC 25010:2023):**

**Qualità funzionali:**
- **Functional Suitability:** Adeguatezza funzionale
- **Performance Efficiency:** Efficienza prestazioni
- **Compatibility:** Compatibilità con altri sistemi
- **Usability:** Facilità d'uso
- **Reliability:** Affidabilità
- **Security:** Sicurezza
- **Maintainability:** Manutenibilità
- **Portability:** Portabilità

#### Qualità di Processo
**[88:04 - 90:29]**

**Produttività:**
**[88:04 - 88:44]**
> "Capacità di produrre una 'buona' quantità di output (prodotti software) data una certa quantità di input (risorse)"

**Puntualità:**
**[89:03 - 89:49]**
> "Capacità di rispondere alle richieste di cambiamento in modo tempestivo"

**Equilibrio critico:**
- Bilanciamento tra produttività e qualità
- Gestione tempi vs. completezza
- Adattabilità ai cambiamenti del mercato

---

## 🎯 DATE IMPORTANTI E SCADENZE

### Progetto R&DD
- **Assegnazione progetto:** 10 ottobre 2025
- **Registrazione gruppi:** Da definire
- **Consegna documenti:** Da definire
- **Discussione:** Sessione invernale (gennaio-febbraio)

### Homework
- **Pubblicazione:** Collegato alle esercitazioni
- **Valutazione:** Su documenti studenti anno precedente
- **Punteggio:** 0-2 punti (1 punto per RASD + 1 punto per DD)

### Esami Scritti
- **5 appelli** durante l'anno accademico
- **Durata:** 1.5 ore per parte (3 ore se entrambe le parti)
- **Modalità:** Open book (no dispositivi elettronici tranne e-reader basic)

---

## 📋 INFORMAZIONI PRATICHE

### Registrazioni e Streaming
- **Policy:** Best effort per le registrazioni
- **Backup:** Materiale anno precedente disponibile
- **Streaming:** Chiuso tranne casi eccezionali
- **Accesso:** Registrazioni sempre disponibili su WeBeep

### Materiali di Studio
- **Primario:** Nuovo libro di testo (ottobre 2025)
- **Supporto:** Slides riviste per ogni lezione
- **Approfondimenti:** Libri suggeriti in biblioteca digitale
- **Website:** WeBeep con programma dettagliato e link registrazioni

### Supporto Studenti
- **Esercitazioni:** 4 sessioni dedicate per progetti
- **Ricevimento:** Disponibile su appuntamento
- **Gruppi:** Supporto per coordinamento e risoluzione problemi
- **Comunicazioni:** Announcements via website comune delle tre classi