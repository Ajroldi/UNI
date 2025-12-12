> Data Ora: 2025-12-11 12:12:52
> Luogo: [Luogo]
> Corso: [Nome Corso]
## Panoramica
In queste lezioni il corso è passato dalla ricerca classica agli agenti che apprendono, presentando il reinforcement learning (RL) come naturale estensione che arricchisce ricerca e pianificazione. Il materiale ha introdotto la definizione di machine learning di Tom Mitchell, confrontato apprendimento supervisionato, non supervisionato e per rinforzo, e collegato il RL alla ricerca tramite concetti come funzioni di valore, politiche e ricompense ritardate. I fondamenti del RL sono stati sviluppati tramite esempi di navigazione su griglia, tris e problemi tipo mountain-car/pendolo, trattando la proprietà di Markov, ritorni scontati, design delle ricompense, cicli SARSA, idee di Bellman e aggiornamenti pratici Q-learning. Le lezioni hanno anche confrontato approcci tabellari con approssimazione tramite funzioni profonde (es. DQN/CNN), evidenziando esplorazione, convergenza, efficienza campionaria e risultati empirici contro vari avversari. L'apprendimento non supervisionato è stato presentato come pre-processing per ricerca/pianificazione, mentre il RL è stato sottolineato per i suoi legami stretti con ricerca, pianificazione e gioco. Sono stati citati demo multimediali; dove non disponibili, sono stati forniti confronti e risultati verbali.
## Contenuti Rimanenti
1. Rappresentazione di concetti complessi tramite logica e applicazione di questi metodi alla pianificazione (prossime lezioni con il Prof. Amigoni).
2. Derivazione formale della formula di massimizzazione per la ricompensa cumulativa e teoria completa dell'equazione di Bellman (rimandata; prevista per un futuro corso ML).
3. Differenze dettagliate tra A* e RL nei compiti di navigazione ed esempi estesi di pianificazione (rimandati al Prof. Amigoni).
4. Dettagli di implementazione del deep reinforcement learning oltre il Q-learning tabellare (architetture, stabilizzatori) e specifiche CNN.
5. Parametrizzazione di UCB o alternative tipo bandit con funzioni apprese nelle politiche RL (solo citate).
6. Strategie di inizializzazione della Q-table e effetti precisi della regolazione degli iperparametri (learning rate, gamma, epsilon).
7. Garanzie formali di convergenza con approssimazione tramite funzioni e condizioni; tabelle di trasposizione e riduzioni di simmetria nei giochi.
8. Video dimostrativi completi per BFS/A*, esecuzioni RL ed esperimenti di self-play (citati ma non mostrati).
## Contenuti Trattati
### 1. Inquadramento del Corso: dalla Ricerca agli Agenti che Apprendono
- Conclusa la parte sulla ricerca (es. BFS/A*) e introdotto l'apprendimento come potenziamento quando l'ambiente è parzialmente ignoto.
- Prossimo focus: rappresentazione logica e applicazioni di pianificazione.
- L'apprendimento non supervisionato visto come pre-processing per la ricerca e il RL come strettamente legato a ricerca e gioco.
### 2. Architettura dell'Agente che Apprende e Ruolo di Performance/Critico
- Presentata l'architettura dell'agente che apprende: elemento di performance (agente), critico (misura/feedback di performance), elemento di apprendimento (adatta il comportamento), generatore di problemi (dati di pratica).
- Sottolineata la necessità di misure di performance (es. voti/test) per guidare l'apprendimento.
- Tipi di agenti (goal-based, utility-based, rule-based, semplice riflesso, model-based) in questo schema.
### 3. Definizione di Tom Mitchell e Tipi di Machine Learning
- Definizione: un programma apprende dall'esperienza E su compiti T con misura di performance P se la performance migliora con l'esperienza.
- Supervisionato: dati con target; previsioni (es. classificazione azioni o clienti).
- Non supervisionato: scoperta di struttura da dati senza target.
- Rinforzo: prova ed errore con ricompense; massimizzare la ricompensa cumulativa (spesso scontata).
### 4. Apprendimento Non Supervisionato come Preprocessing per Ricerca/Pianificazione
- Esempio di mappatura robotica: dati sensoriali continui e rumorosi trasformati in classi discrete tramite clustering per formare stati grafo per A*/pianificazione.
- Evidenziata la fattibilità: evitare stati per ogni centimetro; concentrarsi su regioni informative.
### 5. Perché Sottolineare il Reinforcement Learning in Questa Lezione
- Il non supervisionato spesso prepara i dati per la ricerca; il RL si integra con ricerca/gioco (es. AlphaZero usa MCTS più RL).
- Setup RL: agente con sensori agisce in un ambiente e apprende a migliorare la performance.
### 6. Concetti di Reinforcement Learning: Ricompensa Ritardata, SARSA, Funzioni di Valore
- Affrontate ricompense ritardate e conseguenze a lungo termine (studiare per esami, mosse iniziali a scacchi).
- Introdotto ciclo SARSA: Stato–Azione–Ricompensa–Stato–Azione.
- Definite funzioni di valore:
  - V(s): ritorno atteso di essere nello stato s sotto comportamento ottimale.
  - Q(s, a): ritorno atteso da azione a in stato s e poi comportamento ottimale.
- Insegnati concetti tabellari; nota la preferenza pratica per approssimatori profondi (es. DQN).
### 7. Esempio Pendolo/Mountain-Car e Definizione Ricompensa
- Ambiente con azioni: accelera a sinistra, accelera a destra o nessuna accelerazione, usando la gravità per raggiungere l'obiettivo.
- Ricompensa: 0 all'obiettivo; −1 altrimenti (formattazione chiarita).
- Stato include posizione e velocità; le funzioni di valore formano superfici continue sulle variabili di stato.
- Discretizzazione usata come semplificazione nonostante la continuità.
### 8. Politiche e Trade-off Esplorazione–Sfruttamento
- Politiche deterministiche (greedy) vs. epsilon-greedy: con probabilità ε azione casuale; altrimenti la migliore attuale.
- Collegate ai multi-armed bandit ma con esiti RL ritardati vs. ricompense bandit immediate.
- UCB citato ma non usato in questi esempi RL.
### 9. Proprietà di Markov e Assunzione di Dinamica a Un Passo
- Assunta proprietà di Markov: prossimo stato e ricompensa dipendono solo da stato e azione attuali (non dalla storia completa).
- Evidenziate difficoltà pratiche nel verificare la Markovità; esistono estensioni per osservabilità parziale.
### 10. Ritorni Scontati e Fattore di Sconto (Gamma)
- Obiettivo: massimizzare la somma delle ricompense future; sconto tramite γ in [0,1] limita i ritorni e codifica la preferenza temporale.
- γ=0 privilegia ricompense immediate; γ→1 valorizza esiti a lungo termine.
- Citata intuizione del bound geometrico: G_t ≤ R_max / (1−γ) se le ricompense sono limitate.
### 11. Design della Funzione Ricompensa: Ricompensa Obiettivo vs. Penalità Azione e Rischi
- Ricompensa obiettivo (solo terminale) vs. penalità azione (feedback denso, es. −1 su azioni non sicure).
- Evidenziati rischi: incentivi errati, sicurezza, comportamenti indesiderati (esempio penalità pick-up tardivo).
- Sottolineata la difficoltà e l'importanza di un design attento delle ricompense.
### 12. Funzioni di Valore e Concetti di Bellman Expectation
- Chiariti V e Q e la loro relazione con le idee di Bellman (menzionate forme di expectation e optimality ma non derivate).
- Focus su algoritmi pratici più che sulla teoria completa.
### 13. Q-Learning: Storia, Inizializzazione, Regola di Aggiornamento e Parametri
- Nota storica: Q-learning (circa 1989) ha spostato il focus da V a Q.
- Strategie di inizializzazione: valori casuali, pessimisti o ottimisti (influenzano l'esplorazione).
- Aggiornamento canonico:
  - Q(s_t, a_t) ← Q(s_t, a_t) + β [ r_{t+1} + γ max_{a’} Q(s_{t+1}, a’) − Q(s_t, a_t) ].
- Parametri: fattore di sconto γ, learning rate β; epsilon-greedy per la selezione delle azioni; stati terminali gestiti senza bootstrapping.
### 14. Esplorazione e Convergenza nel Reinforcement Learning
- Esplorazione pratica spesso epsilon-greedy; convergenza teorica richiede di provare ogni coppia stato-azione infinite volte (tabellare).
- I metodi tabellari scalano male; approssimatori (es. DQN) migliorano la generalizzazione ma senza garanzie.
- Agire su valori approssimati avvia il bootstrapping dell'apprendimento, con potenziale instabilità sotto approssimazione.
### 15. Modellazione Navigazione su Griglia: BFS/A* vs. Q-Learning
- BFS: esplora la frontiera (verde), restituisce un singolo percorso che soddisfa il goal test.
- RL: progetta ricompense (es. +1 all'obiettivo, −1 altrove), parte con Q casuali e comportamento casuale; in ~200–400+ episodi apprende azioni ottimali negli stati.
- RL produce una politica completa da tutti gli stati vs. la ricerca che restituisce un percorso; A* può essere più veloce con euristiche informate, mentre RL si assume il costo dell'esplorazione.
### 16. Tris come Problema RL Avversario
- Setup: stato della scacchiera, fino a nove azioni; ricompense +1 vittoria, −1 sconfitta, 0 pareggio.
- Avversari trattati come parte dell'ambiente; semplice Q-learning tabellare senza tabelle di trasposizione; learning rate, gamma, epsilon usati.
### 17. Q-Learning vs. Minimax Deterministico
- Minimax gioca in modo ottimale e deterministico (es. apertura al centro).
- All'inizio RL perde; con l'apprendimento, gli esiti tendono al pareggio (RL impara le risposte al gioco fisso), come Minimax vs. Minimax.
### 18. Q-Learning vs. Minimax Casuale
- Minimax casuale randomizza tra azioni di pari valore, ampliando copertura e difficoltà.
- L'apprendimento tende comunque al pareggio; occasionali vittorie Minimax riflettono esplorazione/apprendimento incompleti.
### 19. Self-Play Q-Learning Tabellare
- Entrambi gli agenti apprendono, incontrando più coppie stato-azione che contro avversari deterministici.
- La convergenza al pareggio richiede molte più partite (>6.000), riflettendo lo spazio stati effettivo più grande (3^9 scacchiere) e l'esplorazione reciproca.
### 20. Q-Learning vs. Giocatore Casuale
- Avversari casuali forniscono ampia esplorazione, avvicinando la copertura teorica.
- Q-learning vince generalmente; pareggi e occasionali vittorie random rivelano regioni non apprese.
- Anche giochi piccoli possono richiedere decine di migliaia di episodi (~50.000) per un apprendimento tabellare completo.
### 21. Approssimazione Funzionale con Reti Neurali Profonde (Idee AlphaZero)
- Sostituito Q-table con reti profonde (es. CNN) che codificano le scacchiere come piani di input multipli (X, O, vuoti).
- Contro avversari random o Minimax casuale, la convergenza è molto più rapida (~200–300 esperimenti) rispetto al tabellare (migliaia), grazie a interpolazione e generalizzazione.
- Notata instabilità e mancanza di garanzie; tipici stabilizzatori (target network, experience replay) citati nello spirito.
## Domande degli Studenti
1. Come potremmo sapere se un'azione è buona o cattiva?
- Risposta: Non lo sappiamo immediatamente; questo è il nucleo della sfida nel RL. Le ricompense basate sull'obiettivo segnalano gli esiti solo alla fine, mentre le ricompense basate su penalità d'azione forniscono feedback immediato (ad es., −1 quando si urta nell'evitare ostacoli). Scegliere una funzione di ricompensa appropriata è difficile; penalità disallineate possono peggiorare il comportamento (esempio della tariffa per ritiro tardivo).
2. Il Q-learning impara ad affrontare tutte le versioni di Minimax, o solo un comportamento specifico?
- Risposta: Il Minimax deterministico tende a ripetere le stesse azioni (ad es., apertura al centro), quindi il Q-learning principalmente impara a contrastare quel comportamento deterministico specifico, portando a pareggi simili a Minimax vs. Minimax. Il Minimax randomizzato amplia il comportamento effettivo dell'avversario, rendendo l'apprendimento più difficile.