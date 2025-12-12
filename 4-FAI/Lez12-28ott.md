> Data Ora: 2025-12-11 13:47:22
> Luogo: [Inserisci Luogo]
> Corso: [Inserisci Nome Corso]
## Panoramica
La serie di lezioni ha introdotto agenti logici e logica formale, collegandosi ai contenuti precedenti del corso sulla risoluzione di problemi (CSP, pianificazione di percorsi, ricerca avversaria). Ha presentato la logica come linguaggio formale con sintassi e semantica e ha confrontato la logica proposizionale (PL) con la logica del primo ordine (FOL). I concetti chiave includevano basi di conoscenza (KB), operazioni tell/ask, motori di inferenza, conseguenza logica vs. implicazione, modelli e interpretazioni, quantificatori (∀, ∃), uguaglianza, validità/soddisfacibilità/insoddisfacibilità e procedure di inferenza (model checking e dimostrazione teorematica). Le illustrazioni pratiche hanno riguardato una mini-KB Prolog (Yazin, Ruba, Yusuf), un esempio proposizionale di diagnosi auto e esempi FOL (parentela; King John e Richard; predicati loves/likes), evidenziando unità granulari di conoscenza e tipici errori nell'uso dei quantificatori. Il corso ha enfatizzato l'approccio dichiarativo, la distinzione di ruolo tra KB e motore di inferenza, e la praticità di sistemi ibridi che combinano ricerca e ragionamento logico.
## Contenuti Rimanenti
1. Sintassi e semantica dettagliate della logica proposizionale (connettivi, tavole di verità, conseguenza logica; parentesi e regole di precedenza).
2. Sintassi completa e semantica formale della logica del primo ordine (quantificatori, predicati, funzioni, domini, interpretazioni; rappresentazione ad albero delle frasi FOL).
3. Procedure di inferenza formale e sistemi di dimostrazione (modus ponens, risoluzione, derivazioni; algoritmi; correttezza e completezza).
4. Esempio pratico di agente puramente logico in azione (da trattare nelle prossime lezioni).
5. Integrazione di tecniche di ricerca con ragionamento logico negli agenti (pianificazione ibrida, es. STRIPS più ricerca euristica).
6. Confronto tra ragionamento LLM e ragionamento logico simbolico (discussione rimandata).
7. Logiche aggiuntive oltre la breve menzione (di ordine superiore, modale).
8. Esercizio di knowledge engineering (traduzione da linguaggio naturale a logica).
## Contenuti Trattati
### 1. Riepilogo dei Tipi di Problemi Precedenti e Rappresentazioni di Stato
- Classi di problemi:
  - Ricerca di stato (es. CSP) dove i costi sono associati agli stati.
  - Ricerca di percorso (es. pianificazione di percorsi, puzzle) dove i costi sono associati alle azioni.
  - Problemi avversari riconosciuti ma fuori dallo schema generale attuale.
- Rappresentazioni di stato:
  - Stati atomici (etichette/ID; confronto di uguaglianza).
  - Stati fattorizzati (coppie variabile–valore, tipiche nei CSP).
- Motivazione a passare a stati strutturali (oggetti + relazioni) per abilitare ragionamenti più ricchi.
### 2. Agenti Logici: Concetto, Motivazione e Architettura
- Gli agenti logici usano frasi logiche per rappresentare la conoscenza di stati e azioni e per prendere decisioni.
- KB: collezione di frasi logiche; due operazioni:
  - tell: asserisce conoscenza.
  - ask: interroga la conoscenza.
- Approccio dichiarativo: caratterizzare gli agenti da ciò che sanno.
- Vantaggi:
  - Rappresentazione esplicita della conoscenza.
  - Economia tramite inferenza (derivare fatti impliciti quando necessario).
  - Interrogazione flessibile oltre problemi fissi da iniziale a obiettivo.
- Ciclo dell'agente puramente logico:
  - Percepisce → traduce in frasi → tell KB → ask KB per azione → tell azione al tempo t → incrementa tempo → agisce → ripete.
- Nota pratica: Gli agenti puramente logici sono spesso poco pratici; i sistemi reali combinano ricerca e ragionamento logico.
### 3. KB vs. Motore di Inferenza: Ruoli e Distinzione
- La KB contiene frasi statiche, rappresentate esplicitamente (fatti e regole).
- Il motore di inferenza ragiona sulla KB per dedurre nuovi fatti e rispondere a interrogazioni.
- Esempio di collegamento: Dato male(Yazin) e regole person(X) → mortal(X); male(X) → person(X), il motore di inferenza deriva mortal(Yazin).
### 4. Esempio Prolog: Tell/Ask e Implicazione
- Fatti: male(Yazin), female(Ruba), male(Yusuf).
- Regole: person(X) ⇒ mortal(X); female(X) ⇒ person(X); male(X) ⇒ person(X).
- Query:
  - mortal(Ziggy): falso/sconosciuto (ipotesi del mondo chiuso; assenza di conoscenza).
  - mortal(Yazin): vero tramite catena male(Yazin) ⇒ person(Yazin) ⇒ mortal(Yazin).
- Nota: Direzionalità delle clausole Prolog (RHS implica LHS nella forma di clausola) e mappatura alla notazione naturale di implicazione.
### 5. La Logica come Linguaggio Formale: Sintassi vs. Semantica
- Sintassi: regole per formare frasi ben formate.
- Semantica: significato tramite verità nei modelli/mondi.
- Modelli:
  - In PL: assegnazione di valori di verità ai simboli proposizionali.
  - In FOL: dominio di oggetti più interpretazioni per costanti, predicati e funzioni.
### 6. Logica Proposizionale: Sintassi, Semantica e Modellazione
- Sintassi:
  - Frasi atomiche: simboli proposizionali (p, q, …).
  - Connettivi: ¬, ∧, ∨, →, ↔; formazione ricorsiva di frasi complesse.
  - Parentesi raggruppano sottoformule; regole di precedenza complete da trattare più avanti.
- Semantica e tavole di verità:
  - Un modello assegna vero/falso a ogni simbolo; con n simboli, 2^n modelli.
  - Esempio: p → q è vero quando p è falso (verità vacua).
- Rappresentazioni ad albero (prospettiva di programmazione):
  - Operatore radice; sottoalberi sinistro/destro come sottoformule.
  - L'ordine conta per connettivi non commutativi (es. →).
- Rilevanza e limiti:
  - PL adatta a informazioni parziali, disgiuntive e negative; valutazione composizionale e indipendente dal contesto.
  - Limiti espressivi: non può riferirsi a oggetti specifici o identità; domini strutturati richiedono molti fatti separati.
- Esempio: Diagnosi auto che mappa regole naturali a simboli PL e implicazioni; la più piccola unità di conoscenza sono proposizioni atomiche (fatti vero/falso come A, B, C).
### 7. Differenze tra Logica Proposizionale e Logica del Primo Ordine
- PL: intere frasi come unità atomiche; nessun oggetto/relazione esplicito o identità tra occorrenze.
- FOL: oggetti (specifici/generici) e relazioni; può asserire identità e struttura tra occorrenze di variabili; supporta costanti, funzioni, predicati, uguaglianza e quantificatori.
- Esempio di contrasto: “mother(Lulu, Fifi)” esprimibile in FOL; PL non può catturare identità di oggetti o struttura relazionale con soli simboli A, B.
- Guida alla scelta dello strumento: PL basta per stati semplici e non strutturati; FOL preferibile per domini strutturati con oggetti e relazioni.
### 8. Introduzione alla Logica del Primo Ordine: Termini, Predicati, Relazioni e Uguaglianza
- Oggetti denotati direttamente (costanti come KingJohn, 2) o indirettamente tramite funzioni (es. father(Paolo)).
- Variabili (x, y) si riferiscono a oggetti indeterminati.
- Simboli di predicato esprimono relazioni (es. Brother(x, y)); frasi atomiche sono predicati applicati a termini.
- Predicato di uguaglianza: term1 = term2 è vero se entrambi denotano lo stesso oggetto; assiomi standard noti (riflessività, simmetria, transitività, sostituzione).
- Esempi: Brother(KingJohn, Richard); Longer(LegLeft(Richard), LegLeft(KingJohn)); Parent(father(Paolo), x).
### 9. Esempi di Logica del Primo Ordine e Quantificatori
- Esempio di parentela:
  - Predicati: P(x,y): parent(x,y); O(x,y): older(x,y); M(x,y): mother(x,y).
  - Conoscenza:
    - ∀x∀y (P(x,y) ⇒ O(x,y)).
    - ∀x∀y (M(x,y) ⇒ P(x,y)).
    - M(Lulu, 50).
  - Catena di inferenza: M(Lulu,50) ⇒ P(Lulu,50) ⇒ O(Lulu,50).
- Quantificatori:
  - Universale: ∀x α è vero se α vale per tutti x nel dominio; uso tipico con implicazione (es. ∀x (at(x, Polyme) → smart(x))).
  - Esistenziale: ∃x α è vero se α vale per qualche x; uso tipico con congiunzione (es. ∃x (at(x, Polyme) ∧ smart(x))).
  - Errori comuni:
    - ∀ con ∧ cambia il significato in “tutti soddisfano entrambi”, non appartenenza condizionale.
    - ∃ con → può essere vero vacuamente per i non membri.
  - Proprietà dei quantificatori:
    - Quantificatori dello stesso tipo commutano (∀x∀y ≡ ∀y∀x; ∃x∃y ≡ ∃y∃x).
    - Tipi diversi non commutano: ∃x ∀y loves(x,y) è diverso da ∀y ∃x loves(x,y).
  - Equivalenze di negazione:
    - ∀x φ ≡ ¬∃x ¬φ; ∃x φ ≡ ¬∀x ¬φ.
### 10. Semantica, Modelli e Interpretazioni in FOL
- Un modello comprende:
  - Dominio di oggetti.
  - Interpretazioni che mappano i simboli del linguaggio (costanti, predicati, funzioni) a elementi e relazioni del dominio.
- Verità atomica: un predicato vale quando la relazione interpretata vale tra i termini interpretati.
- Le frasi complesse seguono le tavole di verità dei connettivi.
- Esempio: brother(KingJohn, Richard) → brother(Richard, KingJohn) può essere vera o falsa a seconda dell'interpretazione; i nomi differiscono dagli oggetti denotati.
### 11. Conseguenza, Procedure di Inferenza e Meta-proprietà
- Conseguenza:
  - α ⊨ β significa che ogni modello di α è un modello di β (inclusione di insiemi di modelli).
  - La KB è spesso trattata come congiunzione di frasi: KB ≡ α1 ∧ α2 ∧ …; KB ⊨ β si allinea con α ⊨ β.
  - Esempio: α1 = (¬Q ∧ R ∧ S ∧ W) implica α2 = (¬Q); α2 ha più modelli di α1.
- Procedure di inferenza:
  - Model checking: verifica la conseguenza valutando sui modelli (con metodi pratici per gestire la complessità).
  - Theorem proving: deriva conclusioni tramite regole (es., da P e P → Q si inferisce Q).
  - Correttezza: le conclusioni derivate sono realmente conseguite; completezza: tutte le conclusioni conseguite possono essere derivate (da trattare prossimamente).
- Meta-proprietà:
  - Validità: vera in tutti i modelli (es., A ∨ ¬A).
  - Soddisfacibile: vera in almeno un modello; Insoddisfacibile: vera in nessun modello (es., A ∧ ¬A).
  - Teorema di Deduzione: KB ⊨ α sse KB → α è valida.
  - Teorema di Confutazione: KB ⊨ α sse (KB ∧ ¬α) è insoddisfacibile (base della prova per contraddizione).
### 12. Contesto Storico e Motivazionale del Ragionamento Simbolico
- La logica precede l'IA; ha radici nel pensiero greco antico.
- Motivazione: il ragionamento simbolico completa i metodi statistici; sistemi moderni (es., knowledge graph, regole di compliance) sfruttano la logica insieme al deep learning.
- Collocazione nel corso: le prossime tecniche svilupperanno metodi di IA simbolica classica e la loro integrazione con ricerca e apprendimento.
### 13. Riflessione Guidata: LLM vs. Ragionamento Logico
- Gli studenti hanno riflettuto sulle differenze tra ragionamento logico simbolico e ragionamento basato su LLM (probabilistico vs. deduttivo, data-driven vs. rule-based, trasparenza dell'inferenza).
- Discussione rinviata a una sessione successiva.
## Domande degli Studenti
1. Quanti di voi conoscono Prolog?
  - Risposta: Prolog è un linguaggio di programmazione basato sulla logica; non è richiesto per il corso, usato a scopo illustrativo.
2. Potete concludere che Ziggy è mortale?
  - Risposta: No; mancando conoscenza su Ziggy, la query è falsa o sconosciuta (ipotesi del mondo chiuso).
3. È vero che Yazin è mortale?
  - Risposta: Sì; tramite le implicazioni male(Yazin) ⇒ person(Yazin) ⇒ mortal(Yazin).
4. “La frase brother(King John, Richard) implica brother(Richard, King John) è vera in questo modello?”
  - Risposta: Dipende dall'interpretazione; senza mappature specificate dai simboli agli oggetti/relazioni, non si può concludere la verità.