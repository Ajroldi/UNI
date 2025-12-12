Data Ora: 2025-12-11 10:37:27
Luogo: [Inserisci Luogo]: [Inserisci Luogo]
Corso: [Inserisci Nome Corso]: [Inserisci Nome Corso]
## Panoramica
Questa serie di lezioni è passata dai problemi di ricerca standard alla ricerca avversaria per giochi multi-giocatore. La prima lezione ha introdotto i concetti base della ricerca avversaria, le sue caratteristiche uniche come l'alternanza dei turni e l'algoritmo Minimax. La seconda lezione ha trattato l'ottimizzazione avanzata con il taglio Alpha-Beta, i benefici in termini di complessità, l'importanza dell'ordine delle mosse e considerazioni pratiche come le funzioni di valutazione per giochi con profondità di ricerca limitata. Le ultime lezioni hanno illustrato l'algoritmo Expectiminimax per giochi stocastici, incluso un esempio di potatura, ed esteso i concetti ai giochi con più avversari. La lezione si è conclusa con annunci riguardanti esercitazioni future e materiali di approfondimento.
## Contenuti Trattati
### 1. Introduzione alla Ricerca Avversaria
- Gli algoritmi di ricerca standard (come A*) sono insufficienti per giochi con avversari (es. scacchi) perché non considerano le contromosse.
- La ricerca avversaria è progettata per situazioni con un avversario che cerca di vincere, focalizzandosi su giochi a due giocatori, a turni, a somma zero, con informazione perfetta e deterministici o stocastici.
- A differenza della ricerca standard dove una soluzione è una sequenza di azioni, nella ricerca avversaria la "soluzione" è la singola mossa migliore per il turno corrente.
- L'albero di ricerca deve essere ricostruito a ogni mossa perché l'azione dell'avversario cambia lo stato, e memorizzare l'intero albero è proibitivo in termini di memoria.
### 2. Formalizzazione dei Giochi Avversari
- I giocatori sono designati come **Max** (cerca di massimizzare l'utilità) e **Min** (cerca di minimizzare l'utilità).
- La formulazione del gioco include uno stato iniziale, azioni legali, una funzione risultato, un **Terminal-Test(s)** (sostituisce il goal test) e una funzione **Utility(s, p)** che assegna un valore numerico a uno stato terminale per un giocatore.
- I valori di utilità sono tipicamente +1 per una vittoria di Max, -1 per una vittoria di Min e 0 per un pareggio.
- Il ciclo di gioco prevede il controllo dello stato terminale, la determinazione del giocatore corrente, la scelta di una mossa e l'aggiornamento dello stato.
### 3. L'Algoritmo Minimax
- Minimax trova la mossa ottimale assumendo che anche l'avversario (Min) giochi in modo ottimale.
- Il processo prevede una ricerca ricorsiva in profondità (DFS) per costruire l'albero del gioco fino ai nodi terminali, applicando la funzione di utilità alle foglie e "risalendo" questi valori.
- Nei nodi **Max**, il valore è il **massimo** tra i valori dei figli.
- Nei nodi **Min**, il valore è il **minimo** tra i valori dei figli.
- Il nodo radice (Max) sceglie l'azione che porta al figlio con il valore massimo risalito.
- L'algoritmo è ottimale ma ha complessità temporale O(b^d), rendendolo impraticabile per giochi complessi come gli scacchi.
### 4. Taglio Alpha-Beta
- Il taglio Alpha-Beta è un'ottimizzazione di Minimax che restituisce esattamente lo stesso risultato potando i rami che non influenzano la decisione finale.
- Usa due parametri:
    - **Alpha (α):** Il miglior valore (punteggio più alto) trovato finora per il giocatore MAX.
    - **Beta (β):** Il miglior valore (punteggio più basso) trovato finora per il giocatore MIN.
- La potatura avviene quando `alpha >= beta`, cioè il percorso attuale non può essere migliore di uno già trovato.
- **Complessità:** L'efficienza dipende dall'ordine delle mosse.
    - Caso migliore: O(b^(h/2)), permettendo di cercare circa il doppio in profondità.
    - Caso medio: O(b^(3h/4)).
- Per una potatura ottimale, le mosse che si prevede abbiano la massima utilità per MAX (o minima per MIN) dovrebbero essere esplorate per prime.
### 5. Giochi con Profondità Limitata e Funzioni di Valutazione
- Per giochi complessi, cercare fino ai nodi terminali è impraticabile. Si impone un limite di profondità alla ricerca.
- A questo limite, si usa una **funzione di valutazione** per stimare l'utilità attesa di uno stato non terminale.
- Usare una funzione di valutazione significa che l'ottimalità non è più garantita. La qualità di questa funzione è critica per le prestazioni dell'agente.
- La funzione di valutazione deve essere compatibile con l'intervallo di valori della funzione di utilità.
- Si può applicare la **ricerca ad approfondimento iterativo** per ottenere un algoritmo "anytime" che cerca progressivamente più a fondo finché non scade il tempo.
### 6. Expectiminimax per Giochi Stocastici
- Per giochi con elementi di caso (es. lancio di dadi nel Backgammon), si usa **Expectiminimax**.
- L'albero del gioco include **nodi di caso** oltre ai nodi MAX e MIN.
- In un nodo di caso, il valore è la media pesata degli esiti, in base alle loro probabilità.
- La logica dell'algoritmo è restituire l'utilità per i nodi terminali, massimizzare per i nodi max, minimizzare per i nodi min e calcolare il valore atteso per i nodi di caso.
- Il taglio Alpha-Beta può essere applicato, ma richiede di conoscere i limiti dei valori possibili per essere efficace.
### 7. Minimax per Giochi Multi-Avversario
- Il concetto di minimax può essere esteso a giochi con più di due giocatori (es. Monopoli).
- **Approccio 1:** Trattare ogni giocatore come un massimizzatore indipendente, con ogni turno rappresentato da un nodo in cui quel giocatore massimizza la propria utilità.
- **Approccio 2:** Raggruppare tutti gli avversari in un unico "super-avversario", semplificando l'albero ad alternare tra la tua mossa (max) e le mosse combinate di tutti gli altri.
### 8. Conclusione della Lezione e Annunci
- È stato consigliato agli studenti di esercitarsi con un gioco interattivo di Tris in un notebook Python contro un agente minimax.
- È stato ricordato l'imminente esercitazione sulla ricerca non informata.
- Gli studenti sono stati invitati a ripassare il materiale sulla ricerca non informata in preparazione alla lezione.
## Domande degli Studenti
1. **Ipoteticamente, potrei ottenere più [del valore atteso di 3]? Potrei ottenere di più, ad esempio... potrei arrivare fino a 14. Dipende tutto da come gioca questo giocatore, giusto?**
   - Sì, potresti ottenere un punteggio più alto, ma Minimax opera sotto l'assunzione che anche il tuo avversario giochi in modo ottimale per minimizzare il tuo punteggio. Quindi l'algoritmo calcola la mossa migliore contro un avversario perfetto, portando a un risultato atteso di 3 da quel ramo, non 14.
2. **Posso fermarmi la prima volta che troviamo un'azione che porta a un nodo MIN strettamente positivo?**
   - No, la decisione di potare è sempre relativa ai valori attuali di alpha e beta, non al fatto che un valore sia positivo o negativo. Ti fermi solo quando sai che l'avversario (MIN) ha una mossa che per lui è migliore di quella che tu (MAX) hai già assicurato.
3. **Vogliamo solo vincere, non essere ottimali, giusto?**
   - Minimax è progettato per trovare la mossa *ottimale*, cioè massimizzare l'utilità, non solo raggiungere uno stato di "vittoria" (es. punteggio > 0). In molti giochi o scenari, l'obiettivo è massimizzare il proprio punteggio o rendimento, non solo vincere di misura.