## Algoritmi di Ricerca: Test Obiettivo, Complessità e Gestione Frontiera
- Tempistica test obiettivo:
  - Eseguire il test obiettivo "in uscita" (quando si estrae un nodo dalla frontiera) può fermare la ricerca prima alla prima scoperta di uno stato obiettivo.
  - Impatto sull'algoritmo:
    - Versione iterativa: Potenziali risparmi di tempo fermandosi quando un obiettivo è estratto; non cambia l'ottimalità o comportamento fondamentale; ancora non garantisce ottimalità se esiste un obiettivo più economico altrove.
    - Versione ricorsiva: Il test obiettivo su figli ipotetici non può essere eseguito poiché i figli non sono esplicitamente generati; la valutazione è limitata al percorso corrente a causa della frontiera basata su call stack.
- Due implementazioni per ricerca:
  - Versione ricorsiva:
    - La frontiera è implicita tramite il call stack (nessuna coda esplicita).
    - Complessità spaziale: Lineare nella profondità (memorizza solo il percorso corrente tramite record chiamate).
    - Complessità temporale: Esponenziale nel branching e profondità (esplora lo stesso spazio di ricerca).
    - Vantaggi: "Più economica" in spazio; evita di memorizzare la frontiera completa e nodi figli.
    - Vincoli: Non può valutare o testare nodi non attualmente visitati; non può eseguire test precoce su nodi ipotetici non visti.
  - Versione iterativa:
    - Frontiera esplicita (coda/coda priorità) contenente nodi figli generati.
    - Caratterizzazione complessità: O(B^M) tempo e spazio nel caso peggiore, dove B è fattore branching e M è profondità soluzione.
    - Comportamento: Genera tutti i figli ad ogni espansione e li accoda; permette test obiettivo all'estrazione (uscita).
- Conclusione su arresto precoce:
  - Il test obiettivo precoce nella ricerca iterativa può risparmiare alcune espansioni quando un obiettivo è raggiunto, ma non garantisce ottimalità se esistono obiettivi migliori più in alto o altrove.
  - Il comportamento ricorsivo depth-first-like non può beneficiare del test obiettivo su nodi non generati a causa della mancanza di frontiera esplicita.

## Ricerca Euristica (Informata): Principi e Requisiti
- Definizione e ruolo:
  - La ricerca informata usa informazioni esterne (euristiche) oltre la definizione del problema per prioritizzare nodi nella frontiera.
  - La selezione nodi è fatta tramite una funzione di valutazione F(n), tipicamente alla testa/cima di una coda priorità.
- Caratteristiche euristiche:
  - Fornisce una stima (non costo esatto) di quanto promettente sia un nodo relativamente al raggiungimento dell'obiettivo.
  - Deve essere computabile efficientemente per qualsiasi stato incontrato.
  - Proprietà ideale: Valori più bassi per nodi più promettenti; uguale a zero all'obiettivo.
- Processo generazione euristica:
  - Derivate da versioni semplificate del problema originale (es. ignorare ostacoli o vincoli).
  - Analogie:
    - Distanza euclidea per approssimazione linea retta senza ostacoli.
    - Distanza Manhattan per movimento griglia senza diagonali.
- Ordinamento frontiera:
  - Ricerca greedy best-first: F(n) = H(n); frontiera prioritizzata solo dal valore euristico.
  - Tabelle euristiche possono essere usate quando lo spazio stati è piccolo; altrimenti calcolare su richiesta.
- Accuratezza e dominanza:
  - Un'euristica H2 domina H1 se H2(n) ≥ H1(n) per tutti i nodi n (mentre entrambe sono ammissibili).
  - Euristiche dominanti sono più informate/accurate e tipicamente guidano la ricerca più efficientemente.
  - Esempio di euristica povera: H(n) = 0 per tutti gli n non offre guida.

## Esempio: Euristiche nell'8-Puzzle (H1 vs H2)
- Stato obiettivo: Tessere disposte nell'ordine corretto (spazio vuoto non contato in H1).
- Definizioni euristiche:
  - H1 (tessere mal posizionate): Conteggio tessere non nella loro posizione obiettivo; zero all'obiettivo; ammissibile ma grossolana.
  - H2 (distanza Manhattan): Somma su tessere delle differenze assolute in riga e colonna tra posizioni correnti e obiettivo; zero all'obiettivo; ammissibile e più raffinata.
- Dominanza:
  - H2 domina H1 nell'8-puzzle: H2(n) ≥ H1(n) per tutti gli n; fornisce guida più accurata.
- Comportamento frontiera sotto greedy best-first:
  - Con H1: Più pareggi; può esplorare rami subottimali ed espandere più nodi.
  - Con H2: Migliore discriminazione; meno espansioni; evita alcuni vicoli ciechi.
- Risultato:
  - H2 riduce espansioni non necessarie e può raggiungere l'obiettivo più direttamente rispetto a H1, benché greedy best-first non sia ancora garantito ottimale o completo.

## Esempio Navigazione: Pathfinding Guidato da Euristica
- Setup: Movimento griglia in quattro direzioni verso un obiettivo con muri/ostacoli.
- Euristiche:
  - Distanza Manhattan: Somma delle differenze orizzontali e verticali; tipica per griglie 4-direzioni senza diagonali; ammissibile e consistente in quel setting.
  - Distanza euclidea: Stima linea retta; ammissibile e consistente quando mosse diagonali sono permesse con costi appropriati.
- Comportamento:
  - L'euristica tira le espansioni verso la direzione obiettivo.
  - Quando bloccata, alternative simmetriche possono avere valori euristici simili; il tie-breaking può causare deviazioni.
- Limitazione:
  - Greedy best-first usando solo H può rimanere intrappolato o ciclare a causa di miopia euristica (nessuna considerazione del costo percorso G).

## Ricerca Greedy Best-First: Proprietà e Complessità
- Funzione valutazione: F(n) = H(n).
- Completezza: Non completa; può entrare in loop infiniti o cicli dipendendo dal comportamento euristico e tie-breaking.
- Ottimalità: Non ottimale in generale.
- Complessità (caso peggiore):
  - Tempo: Esponenziale nella profondità, tipicamente O(B^M).
  - Spazio: Esponenziale a causa della memorizzazione frontiera.

## Ricerca A*: Combinare Costo Percorso ed Euristica
- Modifica centrale:
  - F(n) = G(n) + H(n)
    - G(n): Costo percorso da radice a nodo n (certo).
    - H(n): Costo stimato da n a obiettivo (ottimistico).
- Intuizione:
  - Buon progresso mostra G aumenta compensato da calo in H; scarso progresso mostra G aumenta mentre H resta uguale o aumenta.
- Relazione ad altri algoritmi:
  - Uniform Cost Search (UCS) è A* con H(n) = 0, quindi F(n) = G(n).
  - Greedy Best-First Search corrisponde a usare solo H(n).
- Benefici:
  - Incorpora sia progresso finora (G) che sforzo rimanente stimato (H).
  - Con euristiche ammissibili e consistenti, A* è completo, ottimale e ottimalmente efficiente al f-bound della soluzione.

## Requisiti Qualità Euristica per A*
- Ammissibilità:
  - L'euristica non sovrastima mai il costo vero per raggiungere l'obiettivo: 0 ≤ H(n) ≤ H*(n); H(obiettivo) = 0.
  - Costruita tramite semplificazioni ottimistiche (es. ignorare ostacoli).
- Consistenza (Monotonicità):
  - Per tutti gli archi (n → n′) con costo passo c(n, n′): H(n) ≤ c(n, n′) + H(n′).
  - Assicura G non decrescente e F non crescente lungo percorsi nella ricerca grafo.
  - Consistenza implica ammissibilità; non viceversa.
- Impatto:
  - Con H consistente, la prima volta che un nodo è estratto dalla frontiera, il suo G(n) è ottimale.
  - Euristiche inconsistenti possono causare blocco subottimale nella ricerca grafo; ricerca albero può recuperare tramite ri-espansione.

## Note Implementazione e Variazioni
- Ordinamento coda priorità (frontiera):
  - Greedy best-first: prioritizzato da H(n).
  - A*: prioritizzato da G(n) + H(n).
- UCS come caso speciale A*:
  - Impostare H(n) = 0 produce UCS; baseline quando nessuna euristica informativa è disponibile.
- Tie-breaking:
  - Definire politiche deterministiche; non influenza ottimalità sotto consistenza ma influenza ordine attraversamento.
- Gestire euristiche inconsistenti:
  - Mantenere puntatori genitore/figlio e permettere aggiornamenti per migliorare costi per stati già visti.
  - Reinserimento/aggiornamento nodi può mitigare blocco subottimale.

## Ottimalità, Completezza ed Efficienza di A*
- Completezza:
  - A* è completo quando i costi passo sono positivi (≥ ε > 0) e esiste una soluzione di costo ottimale finito c*.
- Ottimalità:
  - Ricerca albero: A* è ottimale con euristiche ammissibili.
  - Ricerca grafo: A* è ottimale con euristiche consistenti.
- Efficienza:
  - A* espande tutti i nodi con F(n) < c* e può espandere alcuni con F(n) = c*, ma non espande mai nodi con F(n) > c*.
  - Ottimalmente efficiente alla frontiera c* tra algoritmi usando euristiche ammissibili.

## A* Pesato e Satisficing
- A* Pesato:
  - F(n) = G(n) + w · H(n), w > 1 per enfatizzare guida euristica.
  - Esplorazione più veloce; tipicamente perde garanzie ammissibilità e ottimalità quando w > 1.
- Satisficing:
  - Accettare soluzioni quasi-ottimali (es. ≤ 1.1 × c*), scambiando ottimalità per meno espansioni.

## Iterative Deepening A* (IDA*): Concetti Chiave, Meccaniche e Proprietà
- Idea centrale:
  - IDA* applica iterative deepening su f-cost (f = G + H), implementando A* come DFS con cutoff.
- Processo iterativo limitato da costo:
  - Cutoff iniziale impostato a f(radice) = G(inizio) + H(inizio).
  - DFS espande solo nodi con f ≤ cutoff corrente; nodi con f > cutoff sono potati.
  - Se nessun obiettivo è trovato, alzare cutoff al più piccolo f osservato che supera il cutoff corrente e ripetere.
- Perché DFS:
  - Risparmi memoria: lineare nella profondità; evita la grande lista aperta di A* standard.
- Proprietà:
  - Completezza e ottimalità quando H è ammissibile e incrementi cutoff seguono valori f superanti minimi.
  - Complessità:
    - Memoria: Lineare nella profondità.
    - Tempo: Spesso superiore ad A* a causa di esplorazione ripetuta; nessuna frontiera memorizzata.
- Impatto pratico:
  - Efficace per puzzle grandi sotto memoria ristretta (es. 15- e 24-puzzle).

## Conclusioni Pratiche
- La ricerca ricorsiva è efficiente in memoria ma non può valutare rami non visti; ricerca iterativa gestisce frontiera esplicita abilitando valutazione più ampia e arresto precoce.
- Le euristiche sono semplificazioni problema: euristiche più informate (dominanti) portano a meno espansioni e migliore guida.
- Greedy best-first è veloce ma non sicuro in completezza/ottimalità; A* bilancia certezza (G) con guida informata (H), abilitando ricerca robusta con euristiche ammissibili e consistenti.
- Manhattan vs Euclidea:
  - Usare Manhattan in mondi griglia allineati ad assi; Euclidea per spazi continui o capaci di diagonali.
- UCS come A* con H=0: baseline utile quando nessuna euristica significativa è disponibile.
- IDA*: scambia tempo per memoria, mantenendo qualità soluzione A* sotto euristiche ammissibili.

## 📅 Prossimi Accordi ed Elementi d'Azione
- [ ] Preparare prossima lezione su backtracking per problemi soddisfacimento vincoli (focus su frontiera implicita ricerca ricorsiva e tracciamento percorso).
- [ ] Introdurre e formalizzare proprietà euristiche: ammissibilità e consistenza, con dimostrazioni completezza e ottimalità A* sotto queste condizioni.
- [ ] Fornire esempi codice: implementazioni greedy best-first e A* con e senza tabelle raggiunto/visitato; includere strategie tie-breaking e aggiornamenti per euristiche inconsistenti.
- [ ] Progettare esperimenti dimostrativi contrastando H1 (tessere mal posizionate) vs H2 (distanza Manhattan) su 8-puzzle e varianti tessere colorate; misurare espansioni nodi.
- [ ] Creare esempi griglia navigazione comparando euristiche Manhattan ed Euclidea; illustrare gestione ostacoli e limitazioni euristiche.
- [ ] Chiarire dichiarazioni complessità: distinguere tempo vs spazio, ricorsivo vs iterativo, e formalizzare analisi O(B^M).
- [ ] Configurare modalità UCS (H=0) per comparazioni prestazioni baseline; prototipare A* Pesato con pesi variabili e misurare qualità soluzione vs espansioni.
- [ ] Validare ammissibilità e consistenza euristica per dominio specifico (regole movimento, modello costo); implementare politiche tie-breaking.
- [ ] Implementare IDA*: impostare cutoff iniziale a f(radice), eseguire DFS con f-thresholding, e benchmarkare A* vs IDA* su memoria e runtime.