> Data Ora: 2025-12-11 12:51:12
> Luogo: [Luogo]
> Corso: [Nome Corso]
## Panoramica
La serie di lezioni ha esplorato le procedure di inferenza nella logica proposizionale con enfasi sul model checking e sull'algoritmo DPLL (Davis–Putnam–Logemann–Loveland) per la soddisfacibilità. È iniziata con il ragionamento tramite tavole di verità, le loro proprietà e limitazioni, quindi è passata a SAT, conversione in CNF e flusso di lavoro di DPLL, includendo terminazione anticipata ed euristiche (letterali puri, clausole unitarie). La conseguenza logica è stata collegata alla soddisfacibilità tramite confutazione (α ⊨ β se e solo se α ∧ ¬β è insoddisfacibile). Esempi svolti hanno coperto trasformazioni in CNF, splitting/backtracking e un compito applicato di ragionamento che traduce linguaggio naturale (ad es., uno scenario con un unicorno) in logica proposizionale per testare conseguenze. La discussione ha collegato questi metodi alla soddisfazione di vincoli e all'NP-completezza, e ha anticipato approcci di dimostrazione teorematica per la prossima lezione.
## Contenuti Rimanenti
1. Famiglia delle procedure di inferenza basate su dimostrazione teorematica (trattazione completa la prossima volta), incluse prove di correttezza/completazza.
2. Esecuzioni dettagliate di DPLL oltre le brevi illustrazioni: esecuzione passo-passo del pseudocodice su istanze CNF concrete.
3. Discussione più ampia e prove formali di correttezza e completezza per DPLL; analisi della complessità e benchmark prestazionali pratici.
4. Miglioramenti avanzati di DPLL: euristiche di selezione delle variabili, apprendimento di clausole, backjumping.
5. Strutture dati e indicizzazione intelligenti (ad es., watched literals); spiegazione della rappresentazione CNF basata su insiemi che abilita efficienza.
6. Varianti come 2SAT e 3SAT e le loro proprietà.
7. Applicazioni di DPLL alla pianificazione e ad altri domini; traduzioni CNF estese per esempi di homework.
## Contenuti Trattati
### 1. Riepilogo: Agenti Logici e Procedure di Inferenza
- Gli agenti logici codificano conoscenza con formule ben formate in logica proposizionale o del primo ordine.
- L'inferenza chiede se α implica β (α ⊨ β).
- Due famiglie:
  - Model checking (focus di queste lezioni).
  - Theorem proving (prossima lezione).
- Proprietà chiave:
  - Correttezza (soundness): le frasi derivate sono realmente conseguite.
  - Completezza: se β è conseguenza di α, la procedura può derivare β.
> Suggerimenti AI
> - Usa un diagramma che contrasti model checking vs. theorem proving e riprendilo alla fine.
> - Fornisci una slide di definizioni concise per correttezza/completezza con esempi in una riga.
### 2. Ragionamento con Tavole di Verità (Basi di Model Checking)
- Conseguenza: α ⊨ β sse ogni modello che soddisfa α soddisfa anche β.
- I modelli assegnano valori di verità a n simboli proposizionali → 2^n modelli.
- Procedura: enumerare tutti i modelli; verificare α ⇒ β; estendibile a KB ⊨ β congiungendo le frasi della KB.
- Esempio:
  - A: (R ∧ W) → RCA
  - B: R ∧ ¬RCA
  - C: ¬W
  - Con simboli R, W, RCA, solo il modello R=true, W=false, RCA=false rende vere A e B; rende vera anche C → A ∧ B ⊨ C.
- Proprietà:
  - Corretta, completa e decidibile per vocabolari proposizionali finiti.
- Limitazioni:
  - Esplosione esponenziale (2^n righe) e limitata interpretabilità umana.
> Suggerimenti AI
> - Mostra il sottoinsieme di righe della tavola dove A e B sono vere, evidenziandole.
> - Contrasta con uno schema di deduzione naturale o risoluzione per l'interpretabilità.
> - Rinforza la crescita: chiedi “Quanti modelli per 5 simboli?”
### 3. Dalla Conseguenza a SAT e Complessità del Problema
- SAT: determinare se una formula ha un'assegnazione soddisfacente.
- Varianti includono 2SAT e 3SAT; SAT è NP-completo, sebbene molte istanze pratiche siano risolvibili efficientemente.
- Riduzione della conseguenza: α ⊨ β sse α ∧ ¬β è insoddisfacibile.
> Suggerimenti AI
> - Nota brevemente NP vs. co-NP e la riduzione α ∧ ¬β.
> - Men ziona applicazioni (verifica, pianificazione) per motivare.
### 4. CNF (Forma Normale Coniuntiva) e Rappresentazione
- CNF: congiunzione di clausole; ogni clausola è una disgiunzione di letterali (simbolo o sua negazione).
- La CNF preserva l'equivalenza logica.
- Rappresentare come insiemi di insiemi per comodità computazionale (indicizzazione, aggiornamenti rapidi).
- Definizioni:
  - Clausola unitaria: letterale singolo.
  - Letterali complementari: A e ¬A.
  - Letterale puro: simbolo compare con una sola polarità tra le clausole.
> Suggerimenti AI
> - Mostra una mappatura concreta a insiemi (es., {(A, B), (¬C, D)}) e operazioni di base (rimuovere clausole soddisfatte).
> - Confronta rappresentazioni ad albero vs. a insiemi per illustrarne i vantaggi.
### 5. Conversione di una Formula in CNF (Passo per Passo)
- Esempio: A ↔ (B ∨ C).
  1. Elimina bicondizionale: (A → B ∨ C) ∧ (B ∨ C → A).
  2. Elimina implicazioni: (¬A ∨ B ∨ C) ∧ (¬(B ∨ C) ∨ A).
  3. Spingi negazioni (De Morgan): ¬(B ∨ C) ≡ (¬B ∧ ¬C).
  4. Distribuisci: (¬A ∨ B ∨ C) ∧ (¬B ∨ A) ∧ (¬C ∨ A).
> Suggerimenti AI
> - Etichetta ogni trasformazione (es., “Eliminazione dell'implicazione”) e annota gli intermedi.
> - Fornisci un secondo esempio rapido (es., ¬(A → B) ∨ C) per esercizio.
### 6. Algoritmo DPLL: Idee di Base e Procedura
- Obiettivo: verificare SAT per una formula CNF tramite ricerca in profondità su assegnazioni parziali.
- Terminazione anticipata:
  - Se tutte le clausole sono vere sotto l'assegnazione parziale → SAT.
  - Se una clausola diventa falsa → UNSAT per quel ramo.
- Euristiche:
  - Letterale puro: se X compare solo come X (o solo come ¬X), assegna per soddisfare tutte le sue occorrenze (es., solo ¬X → imposta X=false).
  - Clausola unitaria: assegnazione forzata per soddisfare la clausola a singolo letterale.
- Flusso del pseudocodice:
  - Verifica condizioni di terminazione.
  - Applica propagazione unitaria ed eliminazione dei letterali puri.
  - Se non applicabili, scegli una variabile e fai split (true/false), con backtracking quando necessario.
- Collegamento ai CSP: analogo alla ricerca con backtracking con euristiche.
> Suggerimenti AI
> - Confronta le euristiche dei CSP (es., MRV) con le scelte di DPLL.
> - Esegui dal vivo una CNF mostrando propagazione unitaria ed eliminazione dei letterali puri.
### 7. Esempi di Flusso DPLL
- Caso soddisfacibile:
  - Parti con clausola unitaria ¬C → imposta C=false; semplifica.
  - Identifica letterali puri (es., ¬A, D) e scegli per frequenza.
  - Assegna ¬A; le clausole rimanenti sono vere; il modello {C=false, A=false} soddisfa tutte le clausole; nessuno split necessario.
- Splitting e backtracking:
  - Clausole esempio: a∨b, a∨¬b, ¬a∨b; inizialmente nessun puro/unitario.
  - Split su A: A=true → clausola unitaria B → B=true fornisce un modello.
  - Se un ramo fallisce, torna indietro e prova l'assegnazione alternativa (A=false).
- Prova di insoddisfacibilità:
  - Esempio: a∨b, a∨¬b, ¬a∨b, ¬a∨¬b; split su A.
  - Ramo A=true produce B e ¬B → contraddizione.
  - Ramo A=false produce B e ¬B → contraddizione.
  - Tutti i rami si chiudono → UNSAT.
> Suggerimenti AI
> - Usa un piccolo albero decisionale per split/backtracking.
> - Giustifica brevemente la scelta della variabile di split (euristiche di occorrenza/attività).
> - Per l'unsat, mostra una tavola di verità compatta per A, B per confermare l'assenza di assegnazioni soddisfacenti.
### 8. Conseguenza via Confutazione e DPLL
- α ⊨ β sse nessun modello soddisfa α ∧ ¬β.
- Teorema di confutazione: α ⊨ β ⇔ unsat(α ∧ ¬β).
- DPLL verifica la soddisfacibilità di α ∧ ¬β per decidere la conseguenza.
> Suggerimenti AI
> - Presenta un visual “Conseguenza ⇄ Unsat”: α ⊨ β ⇄ unsat(α ∧ ¬β).
> - Includi micro-esempi in cui la conseguenza vale e fallisce.
### 9. Esempio Svolto di Conseguenza (Alice e l'Impermeabile)
- Obiettivo: provare A, B ⊨ C mostrando unsat(A ∧ B ∧ ¬C).
- Traduci in CNF; nega C appropriatamente.
- Usa clausole unitarie e letterali puri; non si sono verificati split.
- Una clausola diventa falsa sotto assegnazioni forzate → contraddizione → UNSAT.
- Concludi A, B ⊨ C.
> Suggerimenti AI
> - Dichiara esplicitamente: “Non si sono verificati split; le assegnazioni forzate determinano univocamente il percorso.”
> - Tieni visibile la CNF iniziale mentre applichi ogni assegnazione forzata.
### 10. Efficienza e Applicazioni Pratiche di DPLL
- Il DPLL di base scala moderatamente; i solver SAT moderni gestiscono milioni di variabili con ottimizzazioni.
- Le euristiche contano: scelta della variabile e ordine dei valori influenzano le prestazioni.
- Tecniche:
  - Watched literals: monitoraggio efficiente delle clausole.
  - Conflict-driven clause learning (CDCL): imparare dai conflitti; backjumping.
- Applicazioni: verifica hardware/software, sintesi, verifica di protocolli, pianificazione.
> Suggerimenti AI
> - Nomina brevemente e dai un'intuizione per watched literals e CDCL.
> - Condividi un caso reale (ad es., model checking basato su SAT che trova un bug).
### 11. Esempio Applicato Finale: Ragionamento sull'Unicorno con DPLL
- Simboli: MY (mythical), MR (mortal), MM (mammal), HR (horned), MG (magical).
- Affermazioni:
  - MY → ¬MR.
  - ¬MY → MR ∧ MM.
  - (¬MR ∨ MM) → HR.
  - HR → MG.
- Domande: L'unicorno è mitico? magico? cornuto?
- Metodo: traduci in CNF; testa α ∧ ¬β per ciascun β.
  - Mitico? α ∧ ¬MY è soddisfacibile (es., ¬MY, MR, MG) → non conseguente.
  - Magico? α ∧ ¬MG è insoddisfacibile → MG è conseguente.
  - Cornuto? α ∧ ¬HR è insoddisfacibile → HR è conseguente.
> Suggerimenti AI
> - Fornisci le derivazioni CNF complete in una dispensa.
> - Mantieni i simboli coerenti e mostra un grafo di dipendenze.
> - Invita a fare previsioni prima di eseguire DPLL.
## Domande degli Studenti
1. “Se vuoi puoi vedere che sulla tavola, ok, c'è ehm che A e B sono vere, ok, e C è vera ma c'è un altro caso quindi e B è vera ma C è vera, oh questo è sbagliato qui, questo stavo solo controllando le ultime tre colonne perché oh no devi controllare.”
- Risposta: Controlla le colonne delle formule complete A (verde), B (verde) e C (blu), non le sottoformule. L'unico modello in cui A e B sono entrambe vere (R=true, W=false, RCA=false) rende vera anche C.
2. “Voglio dire, è un numero finito di combinazioni.”
- Risposta: Con un numero finito di simboli proposizionali, ci sono finitamente molti modelli, quindi la procedura con tavole di verità termina sempre; la logica proposizionale è decidibile.
3. Alfa è una frase? Sì o no.
- Risposta: Sì. Alfa è la congiunzione delle frasi date.
4. Perché possiamo concludere l'insoddisfacibilità nell'esempio di Alice e l'impermeabile senza controllare altri rami?
- Risposta: Le clausole unitarie e i letterali puri forzano assegnazioni; non ci sono stati split, quindi non esistevano rami alternativi da controllare.