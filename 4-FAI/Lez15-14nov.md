Data Ora: 2025-12-11 12:55:33
Luogo: [Inserisci Luogo]: [Inserisci Luogo]
Corso: [Inserisci Nome Corso]: [Inserisci Nome Corso]
## Panoramica
Questo sommario copre tre lezioni sull'inferenza logica. La prima lezione ha introdotto gli algoritmi di dimostrazione teorematica, mettendoli a confronto con il model checking e definendo correttezza e completezza. Si è concentrata sull'algoritmo di risoluzione, spiegandone il meccanismo di prova per confutazione con un esempio svolto. La seconda lezione ha esplorato strategie per ottimizzare la risoluzione (Unit, Input, Linear, Set of Support) e ha definito le clausole di Horn e le clausole definite. Questo ha portato a walkthrough dettagliati degli algoritmi di Forward e Backward Chaining. L'ultima lezione ha applicato questi concetti modellando il ragionamento di un agente nel Wumpus World usando la logica proposizionale. Ha mostrato come costruire una base di conoscenza e usare la conseguenza logica (tramite DPLL) per effettuare mosse sicure, discutendo anche i limiti della logica proposizionale, come la monotonicità e la rappresentazione del tempo.
## Contenuti Trattati
### 1. Introduzione alla Dimostrazione Teorematica e Proprietà Chiave
- Le procedure di inferenza si classificano in due famiglie: model checking e dimostrazione teorematica. Gli algoritmi di dimostrazione operano manipolando sintatticamente le formule per derivarne di nuove.
- **Correttezza (Soundness):** Un algoritmo è corretto se ogni formula che deriva è anche conseguenza della formula iniziale. È la proprietà più importante, perché garantisce risultati affidabili.
- **Completezza (Completeness):** Un algoritmo è completo se può derivare ogni formula che è conseguenza della formula iniziale.
- Sono stati discussi tre algoritmi di dimostrazione: Risoluzione, Forward Chaining e Backward Chaining.
### 2. L'Algoritmo di Risoluzione
- La risoluzione è un algoritmo di dimostrazione corretto e completo che lavora su formule in Forma Normale Coniuntiva (CNF).
- Opera per confutazione: per provare `KB entails alpha`, dimostra che `KB AND (NOT alpha)` è insoddisfacibile derivando la "clausola vuota".
- **Regola di Risoluzione:** Date due clausole che contengono letterali complementari (es., `L` e `NOT L`), si genera una nuova clausola (il risolvente) combinando tutti gli altri letterali delle clausole originali.
- **Esempio:** Risolvendo `(A or B)` e `(not B)` si ottiene `A`. Risolvendo `A` e `not A` si ottiene la clausola vuota, che segnala una contraddizione.
- **Processo:** L'algoritmo genera iterativamente nuovi risolventi e li aggiunge all'insieme di clausole. Termina quando si trova la clausola vuota (provando la conseguenza) o quando non si possono più generare nuove clausole (confutando la conseguenza). L'arresto è garantito perché il numero di clausole possibili è finito.
### 3. Strategie di Risoluzione
- Per gestire il costo computazionale di verificare tutte le possibili coppie di clausole, si usano strategie come euristiche per decidere quali coppie risolvere per prime.
- **Risoluzione Unitaria (Unit Resolution):** Una delle due clausole deve essere unitaria (un solo letterale). È incompleta tranne che per le clausole di Horn.
- **Risoluzione Input (Input Resolution):** Almeno una clausola deve provenire dall'insieme iniziale. Anch'essa è incompleta tranne che per le clausole di Horn.
- **Risoluzione Lineare (Linear Resolution):** Generalizzazione della risoluzione input in cui una clausola proviene dall'insieme iniziale o è un'antenata dell'altra. Questa strategia è completa.
- **Risoluzione con Insieme di Supporto (Set of Support):** Richiede un sottoinsieme delle clausole iniziali (l'"insieme di supporto", S) ritenuto portare a contraddizione. A ogni passo, una clausola deve provenire da S. Questa strategia è completa.
### 4. Clausole di Horn e Algoritmi di Chaining
- Una **clausola di Horn** ha al più un letterale positivo quando scritta in CNF. Una **clausola definita** ha esattamente un letterale positivo. Sono rilevanti per modellare regole (implicazioni) e fatti.
- **Forward Chaining:** Algoritmo guidato dai dati che usa clausole definite e il *modus ponens*. Parte dai fatti noti e ne deriva di nuovi fino a raggiungere l'obiettivo o a esaurire le derive. È corretto e completo per basi di conoscenza con clausole definite.
- **Backward Chaining:** Algoritmo guidato dall'obiettivo che usa anch'esso clausole definite. Parte da una query (l'obiettivo) e lavora all'indietro, creando un grafo AND-OR di sotto-obiettivi da provare. Spesso è più efficiente in pratica perché si focalizza solo sulle regole rilevanti. Evitare i cicli e memorizzare i risultati (caching) sono dettagli implementativi chiave.
### 5. Applicazione: L'Agente del Wumpus World
- La logica proposizionale può modellare la conoscenza e il ragionamento di un agente in un ambiente come il Wumpus World.
- **Rappresentazione:** Si definiscono simboli proposizionali per gli stati in ciascuna cella (es., `Pij` per un pozzo, `Bij` per una brezza). Le regole dell'ambiente (es., si percepisce una brezza se una cella adiacente ha un pozzo) sono codificate come formule logiche in una base di conoscenza (KB).
- **Ragionamento:** L'agente usa la conseguenza logica per prendere decisioni. Per verificare se una mossa è sicura (es., verso la cella 2,1), interroga se la sua KB implica l'assenza di un pericolo (`KB entails not P21`). Questo si fa tipicamente usando un algoritmo come DPLL tramite prova per confutazione.
- **Aggiornamenti di Conoscenza:** Man mano che l'agente si muove e ottiene nuove percezioni (es., sente una brezza), aggiunge questi fatti alla sua KB. Poiché la logica è monotona, la KB solo cresce e l'agente usa questa conoscenza aggiornata per i ragionamenti successivi.
### 6. Limiti della Logica Proposizionale
- **Inefficienza:** Un agente puramente logico è costoso computazionalmente. Serve una regola distinta per ogni proprietà in ogni cella.
- **Rappresentazione del Tempo:** Per gestire ambienti dinamici servirebbero simboli proposizionali per ogni stato a ogni istante (es., `Pij_t0`, `Pij_t1`). Questo porta al **problema del frame**: la difficoltà di definire regole che colleghino gli stati tra tempi diversi. Questi limiti motivano logiche e sistemi di pianificazione più avanzati.
## Compiti Assegnati
1. Formalizza risoluzione, forward chaining e backward chaining come problemi di ricerca (definendo stato iniziale, azioni, test di obiettivo, ecc.).
2. Formalizza il problema di soddisfacibilità (SAT) come problema di soddisfazione di vincoli (CSP) (definendo variabili, domini, vincoli).
3. Riesci a trovare un esempio di clausola di Horn che non sia una clausola definita?
## Domande degli Studenti
1. **Se potessi mantenere solo una di queste due proprietà, correttezza o completezza, quale terresti?**
   - La correttezza è la più importante. È meglio che tutto ciò che l'algoritmo produce sia corretto e affidabile, anche se non trova tutte le formule conseguite possibili.
2. **Puoi dirmi un algoritmo di inferenza completo ma molto stupido?**
   - Un algoritmo che genera tutte le possibili formule. Sarebbe completo perché alla fine genererebbe tutte le formule conseguite, ma sarebbe inutile e scorretto perché genererebbe anche un numero infinito di formule non conseguite.
3. **Che cosa stai includendo in S [l'insieme di supporto]?**
   - In S stai mettendo le frasi che potrebbero portarti all'insoddisfacibilità.
4. **Come posso ottenere L? In quanti modi posso ottenere L? (Nel contesto di un esempio di backward chaining).**
   - Sono state identificate due possibili vie: da 'A e P' oppure da 'A e B'. L'insegnante ha poi spiegato che uno di questi percorsi porta a un ciclo e va evitato.
5. **Perché qui è scritto resolvents e non resolvents?**
   - La trascrizione si interrompe prima che il docente fornisca una risposta a questa domanda.