> Data Ora: 2025-12-11 13:04:21
> Luogo: [Luogo]
> Corso: [Nome Corso]
## Panoramica
La lezione ha passato in rassegna i metodi di pianificazione classica e gli approcci moderni basati su SAT. È iniziata formalizzando i problemi di pianificazione (stati, obiettivi, schemi d'azione con precondizioni/liste add/delete) ed ha esplorato la pianificazione forward come ricerca sugli stati, includendo questioni di efficienza ed euristiche derivate da problemi rilassati. Ha poi introdotto la pianificazione backward come ricerca sugli obiettivi tramite regressione, trattando azioni rilevanti, controlli di inconsistenza e vincoli sugli obiettivi per potare. Motivata dai limiti dei metodi forward/backward, la lezione ha presentato la pianificazione SAT (SATPlan) come alternativa pratica: codificare problemi STRIPS/PDDL in logica proposizionale, eseguire una ricerca iterativa sulla lunghezza del piano con un risolutore SAT (DPLL) ed estrarre piani da modelli soddisfacenti. Un esempio con scatola e anello ha illustrato la pipeline SATPlan, mostrando insoddisfacibilità a L=0 e L=1 e un piano soddisfacente a L=2.
## Contenuti Rimanenti
1. Walkthrough dettagliato di implementazioni Python per pianificazione forward e backward (funzioni, strutture dati, operazioni su insiemi).
2. Derivazioni formali e prove delle proprietà della regressione (es., natura meno vincolante).
3. Catalogo sistematico di vincoli di dominio/stato/obiettivo per potare rami inconsistenti (oltre i casi esemplificati).
4. Confronto empirico delle prestazioni di pianificazione forward vs. backward su diversi domini.
5. Alternative di progettazione di euristiche oltre i rilassamenti di delete/precondizioni (planning graph, landmarks).
6. Panoramica di altri algoritmi di pianificazione efficienti (oltre SAT planning).
7. Dimostrazione di strumenti di conversione automatica STRIPS/PDDL→proposizionale.
8. Passi dettagliati di conversione in CNF per tutti gli assiomi a L=2 (assegnato come homework).
## Contenuti Trattati
### 1. Formalizzazione dei Problemi di Pianificazione e Definizione di Soluzione
- I problemi di pianificazione sono definiti con predicati logici per lo stato iniziale e l'obiettivo, e schemi d'azione specificati da precondizioni, liste add e liste delete; gli schemi sono istanziati in azioni concrete.
- Una soluzione è una sequenza di azioni concrete che trasforma lo stato iniziale in uno stato che soddisfa l'obiettivo.
Suggerimenti AI
- Organizzazione: Aggiungere una slide con l'elenco dei componenti formali (Initial, Goal, Actions: Pre, Add, Del) per un rapido scaffold visivo.
- Metodi: Includere un micro-esempio (2 azioni, 3 predicati) per distinguere schemi vs. sequenze di azioni.
- Chiarezza: Mostrare uno schema con due istanziazioni concrete affiancate.
- Miglioramento: Usare un prompt interattivo: “Dato questo obiettivo, elenca due possibili istanziazioni di schema.”
### 2. Pianificazione Forward come Ricerca sugli Stati
- Parte dallo stato iniziale, applica tutte le istanze d'azione applicabili, genera stati successori e continua finché l'obiettivo è soddisfatto.
- Si può usare qualunque strategia di ricerca (BFS, DFS, UCS, A*); la pianificazione mappa alla ricerca definendo stato iniziale, azioni, funzione di transizione (applica add/delete), test dell'obiettivo (controllo di sottoinsieme) e costi dei passi.
- Considerazioni pratiche: Rappresentare gli stati come insiemi di predicati; gestire i loop con closed list; vincolare i domini per evitare azioni inutili (es., move R2→R2).
- Spunti implementativi: Applicabilità delle azioni e test dell'obiettivo via inclusione di insiemi; risultato via manipolazione di insiemi.
Suggerimenti AI
- Organizzazione: Elencare esplicitamente i componenti della ricerca in una slide per fissare la mappatura.
- Metodi: Generalizzare le insidie di applicabilità con “vincoli di dominio” come best practice.
- Comprensione Studenti: Convertire dal vivo uno stato di blocchi visuale in insiemi di predicati.
- Miglioramento: Coprire brevemente la rilevazione di duplicati nelle closed list e aggiungere un think-pair-share su vincoli per ridurre azioni inutili.
### 3. Euristiche per la Pianificazione Forward tramite Problemi Rilassati
- La ricerca informata (es., A*) usa euristiche derivate da problemi rilassati (ignorare alcune precondizioni o le liste delete).
- Risolvere ottimamente il problema rilassato; la lunghezza della soluzione fornisce un limite inferiore ammissibile per il problema originale.
- Esempio: Nel blocks world, ignorare la precondizione “clear” fornisce la lunghezza del piano rilassato come limite inferiore.
Suggerimenti AI
- Organizzazione: Contrapporre “ignora delete” vs. “ignora precondizioni” e annotare le condizioni di ammissibilità.
- Metodi: Fornire un esempio numerico che calcoli h(s) sotto delete relaxation.
- Chiarezza: Enfatizzare la risoluzione efficiente del rilassato; menzionare approssimazioni (es., conteggio dei letterali obiettivo non soddisfatti).
- Miglioramento: Notare le insidie di euristiche troppo ottimistiche e suggerire aggiustamenti specifici di dominio.
### 4. Limiti della Pianificazione Forward e Introduzione alla Pianificazione Backward
- La pianificazione forward soffre spesso di alti fattori di diramazione e inefficienza.
- La pianificazione backward ricerca sugli obiettivi, regredendo tramite azioni rilevanti; ci si ferma quando l'obiettivo corrente è soddisfatto dallo stato iniziale.
- Anche la pianificazione backward è una famiglia di algoritmi (applicabili diverse strategie di ricerca).
Suggerimenti AI
- Organizzazione: Usare un diagramma affiancato che contrasti esplorazione di stati vs. obiettivi.
- Metodi: Collegare al backward chaining con un'analogia esplicita.
- Chiarezza: Ribadire la soddisfazione dell'obiettivo come inclusione di predicati nello stato iniziale.
- Miglioramento: Fornire una checklist per scegliere la backward (obiettivi decomponibili, dominio che supporta regressione).
### 5. Azioni Rilevanti e Significato Intuitivo
- Un'azione è rilevante per l'obiettivo G se aggiunge almeno un sotto-obiettivo (letterale) di G; la rilevanza indica che potrebbe essere l'ultima azione di qualche soluzione.
- La rilevanza non implica necessità; alcune azioni rilevanti possono essere inutilizzabili a causa di interferenze.
Suggerimenti AI
- Organizzazione: Presentare la definizione formale prima dell'intuizione.
- Metodi: Aggiungere un controesempio dove la rilevanza fallisce per interferenza.
- Chiarezza: Rinforzare “rilevante ≠ obbligatorio”.
- Miglioramento: Esercitarsi nell'identificare azioni rilevanti per un obiettivo multi-letterale.
### 6. Regressione: Definizione, Intuizione e Algoritmo
- La regressione di G attraverso A produce R(G, A) tale che qualsiasi stato che soddisfa R(G, A) rende A applicabile e l'esecuzione di A porta a uno stato che soddisfa G.
- Intuizione: Trattare A come potenziale ultimo passo; includere le precondizioni di A e mantenere le parti di G non soddisfatte che non sono aggiunte da A.
- Inconsistenza: Se A cancella qualche sotto-obiettivo di G, è inconsistente e non può essere usata per la regressione.
- Algoritmo:
  - Se Delete(A) ∩ G ≠ ∅ → inconsistente.
  - Altrimenti, R(G, A) = Pre(A) ∪ (G \ Add(A)).
Suggerimenti AI
- Organizzazione: Numerare i passi dell'algoritmo per prendere appunti.
- Metodi: Fornire un secondo esempio di fallimento per inconsistenza.
- Chiarezza: Enfatizzare “meno vincolante” escludendo i letterali già aggiunti da A.
- Miglioramento: Offrire un template di regressione per esercizi.
### 7. Flusso Backward, Inconsistenza e Vincoli sugli Obiettivi
- Regressare iterativamente l'obiettivo corrente tramite azioni rilevanti; fermarsi quando l'obiettivo regredito vale nello stato iniziale; leggere il piano dal basso verso l'alto.
- Insidie: Obiettivi intrinsecamente inconsistenti (letterali mutex), cattivo ordinamento dei sotto-obiettivi che porta a vicoli ciechi.
- Tecniche: Usare vincoli su obiettivi/stato per potare rami insoddisfacibili; possono servire vincoli multipli.
Suggerimenti AI
- Organizzazione: Introdurre un'euristica per scegliere i sotto-obiettivi da regredire (effetti che corrispondono a sotto-obiettivi non soddisfatti senza conflitti).
- Metodi: Brainstorming di vincoli di dominio con gli studenti.
- Chiarezza: Rinforzare la terminologia “obiettivo ≠ stato”.
- Miglioramento: Fornire una checklist di rilevazione inconsistenze (Delete(A) ∩ goal, vincoli di dominio, plausibilità dell'ultimo passo).
### 8. Limiti di Forward e Backward (Motivazione Pratica)
- La backward può essere più efficiente della forward ma spesso incontra vicoli ciechi difficili da rilevare; le euristiche per A* backward sono difficili da progettare.
- In pratica, entrambi i metodi sono spesso superati da algoritmi specializzati come SAT planning.
Suggerimenti AI
- Motivazione: Richiamare un esempio di vicolo cieco in forward per fissare la memoria.
- Quantificazione: Confrontare prestazioni empiriche (dimensioni dei problemi) per mostrare l'inefficienza pratica.
### 9. SAT Planning: Idea di Alto Livello e Flusso di Lavoro
- Ridurre un problema di pianificazione a SAT: rappresentare in STRIPS/PDDL → tradurre in formule proposizionali → eseguire un SAT solver (DPLL) → estrarre un piano da un modello soddisfacente.
- Un'assegnazione soddisfacente corrisponde a un piano valido sotto una codifica corretta; l'efficienza sfrutta solver SAT altamente ottimizzati.
Suggerimenti AI
- Ritmo: Presentare la pipeline come diagramma numerato e riprenderla dopo l'esempio.
- Anteprima: Spiegare il ruolo degli assiomi (precondizioni, fluenti, esclusioni) per demistificare la “magia”.
### 10. Ricerca Iterativa della Lunghezza del Piano (L=0,1,2,…)
- Cercare aumentando la lunghezza L del piano: codificare e testare SAT a ogni L.
- Se insoddisfacibile, incrementare L; fermarsi a una soglia di risorse o al trovare un piano.
Suggerimenti AI
- Visualizzazione: Includere un flowchart (costruisci codifica → controllo SAT → se unsat, incrementa L).
- Completezza: Notare condizioni ed euristiche per limitare L.
### 11. Problema di Esempio in STRIPS (Scatola, Anello, Mano)
- Stato iniziale: closed(b), in(r,b), free.
- Obiettivo: hold(r).
- Schemi d'azione: open(x) e pick_out(x,y).
- Costanti: scatola b, anello r. Predicati: closed, open, in, free, hold.
- Nota: Sono possibili codifiche alternative (es., solo closed o solo open).
Suggerimenti AI
- Chiarezza: Indicare esplicitamente l'obiettivo come hold(r) quando introdotto.
- Supporto: Fornire una piccola tabella che mappi gli elementi STRIPS ai ruoli.
### 12. Simboli Indicizzati nel Tempo e Assunzione di Mondo Chiuso
- Definire simboli proposizionali per ciascun fluente e azione a ogni passo temporale t.
- Nelle codifiche SAT, la falsità deve essere asserita esplicitamente; l'assunzione di mondo chiuso non si applica automaticamente.
Suggerimenti AI
- Distinzione: Separare visivamente simboli di fluenti vs. azioni con indici temporali.
- Promemoria: “In STRIPS, i fatti non elencati sono falsi; nelle codifiche SAT, asserire le falsità.”
### 13. Reificazione di Azioni e Fluenti
- Le azioni sono reificate in simboli proposizionali (vero/falso al tempo t).
- I fluenti rappresentano predicati variabili nel tempo; simboli di azione e di fluente coesistono.
Suggerimenti AI
- Esempio: Mostrare “pick_out(r,b) vero a t” e i suoi effetti sui fluenti a t+1.
- Checklist: “Per ogni schema d'azione, crea simboli azione; per ogni istanza di predicato, crea simboli fluente.”
### 14. Codifica a L=0 e L=1; CNF e Insoddisfacibilità
- L=0: Codificare iniziale e obiettivo al tempo 0; tipicamente insoddisfacibile per problemi non banali.
- L=1: Aggiungere azioni e famiglie di assiomi; tradurre in CNF; l'esempio è insoddisfacibile (nessun piano di lunghezza 1).
- DPLL: La propagazione unitaria porta a contraddizione a L=1.
Suggerimenti AI
- Concreto: Includere un frammento CNF minimo e 2–3 passi di propagazione unitaria.
- Dispensa: Fornire le clausole CNF esatte per L=1 da annotare.
### 15. Assiomi di Precondizione (Applicabilità)
- Forma: a_t → (pre1_t ∧ pre2_t ∧ …); codificare per ogni istanza d'azione e passo temporale, poi convertire in CNF.
- La direzione logica è di necessità (l'azione implica le precondizioni), non sufficienza.
Suggerimenti AI
- Pattern: Mostrare l'implicazione e la sua forma CNF.
- Esercizio: Scrivere gli assiomi di precondizione per un'azione aggiuntiva (es., close(b)).
### 16. Assiomi sui Fluenti (Transizioni di Stato)
- Per ogni fluente F: F_{t+1} è vero sse (un'azione a t rende F vero) OPPURE (F_t è vero e nessuna azione a t rende F falso).
- Istanziazioni di esempio:
  - OpenB_{t+1} sse open(b)_t O OpenB_t.
  - InRB_{t+1} sse InRB_t E NON pick_out(r,b)_t.
  - HoldR_{t+1} sse pick_out(r,b)_t O HoldR_t.
Suggerimenti AI
- Template: Presentare un assioma fluente generico e istanziare per fluente.
- Terminologia: Introdurre i frame axioms e “nessun cambiamento se non azionato”.
### 17. Assiomi di Esclusione delle Azioni (No Azioni in Parallelo)
- Codificare l'esclusione reciproca: per ogni coppia (i≠j), ¬a_i_t ∨ ¬a_j_t a ogni tempo t.
- Anche se le precondizioni impediscono la concorrenza, includere clausole di esclusione come pratica generale.
Suggerimenti AI
- Scalabilità: Menzionare mutex basati su risorse/interferenze (stile Graphplan).
- Mnemonico: “Clausole not-both a coppie a ogni t.”
### 18. Codifica per L=2; Modello DPLL ed Estrazione del Piano
- L'obiettivo deve valere a t=2; assiomi duplicati per t=0 e t=1.
- DPLL trova un'assegnazione soddisfacente:
  - azioni a t=0: open(b) vero; pick_out(r,b) falso.
  - azioni a t=1: open(b) falso; pick_out(r,b) vero.
  - stato a t=2: open(b), not in(r,b), hold(r) (obiettivo soddisfatto).
- Piano estratto: [open(b) a t=0, pick_out(r,b) a t=1].
Suggerimenti AI
- Presentazione: Mostrare una timeline compatta con stati e azioni per t; evidenziare il piano.
- Nota: L'ordine delle assegnazioni è guidato dal solver (es., euristiche come VSIDS) ed è semanticamente agnostico.
### 19. Comportamento DPLL e Semantica della Pianificazione
- Il SAT solver opera come una black box cercando assegnazioni soddisfacenti; la codifica garantisce che qualunque modello soddisfacente corrisponda a un piano valido nonostante l'ordine opportunistico delle assegnazioni del solver.
Suggerimenti AI
- Rinforzo: Aggiungere una slide “solver black box vs. codifica strutturata”.
- Estensione: Considerare di mostrare una breve traccia di propagazione unitaria in una lezione futura.
### 20. Riepilogo del Processo SATPlan
- Partire da STRIPS/PDDL; tradurre automaticamente in logica proposizionale con assiomi di precondizione, fluente ed esclusione; eseguire SAT; estrarre il piano dal modello.
- Nonostante la complessità, SATPlan è efficiente e ampiamente usato in pratica.
Suggerimenti AI
- Chiusura: Assegnare un foglio di lavoro che etichetti assiomi di applicabilità, effetti ed esclusività in un micro-dominio (es., interruttore della luce) per rinforzare il trasferimento.
## Domande degli Studenti
1. Perché la pianificazione forward è considerata una famiglia di algoritmi?
- Perché si può applicare qualsiasi strategia di ricerca (ampiezza, profondità, A*, costo uniforme); la pianificazione è una specializzazione della ricerca generale.
2. L'azione “mettere la chiave nella scatola” è rilevante per l'obiettivo che richiede la chiave nella scatola?
- Sì, aggiunge il sotto-obiettivo in-key-box; tuttavia, se cancella altri sotto-obiettivi necessari, diventa inconsistente e non può essere usata per la regressione.
3. Cosa c'è di problematico nel considerare applicabili azioni come move R2→R2?
- Creano loop (mitigati da closed list) e gonfiano il fattore di diramazione con azioni inutili, sprecando calcolo senza aiutare la ricerca.
4. Perché free può essere rappresentato come not hold(r)?
- Perché c'è un solo anello; la mano o tiene quell'unico anello o è libera. Con più anelli, questa equivalenza non vale.