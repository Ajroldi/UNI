> Data e Ora: 2025-12-11 18:49:11
> Luogo: Luogo
> Corso: Intelligenza Artificiale / Reti Bayesiane e Inferenza Probabilistica
## Panoramica
La sessione ha trattato la transizione da ambienti deterministici alla gestione dell'incertezza negli agenti, ponendo le basi teoriche e meccaniche dei Modelli Grafici Probabilistici. Il corso ha esplorato le fonti di incertezza (osservabilità parziale, non-determinismo), i limiti della Logica e l'introduzione della Teoria delle Decisioni (Utilità Attesa Massima). Il focus principale è stato sulle **Reti Bayesiane**: la loro struttura (DAG), la semantica e la distinzione tra variabili casuali e valori. Argomenti tecnici chiave hanno incluso la Regola della Catena, la Marginalizzazione, la Normalizzazione e tre tipi di ragionamento (Causale, Evidenziale, Inter-causale). La sessione si è conclusa con un'introduzione ai metodi di inferenza approssimata, in particolare il Campionamento in Avanti e il Campionamento per Rifiuto.
## Contenuti Trattati
### 1. Introduzione all'Incertezza e agli Stati di Credenza
- **Passaggio all'Incertezza:** Dalla conoscenza perfetta ad ambienti incerti causati da percezione imperfetta (sensori), osservabilità parziale, non-determinismo e agenti avversari.
- **Stato di Credenza:** Definito come l'insieme di tutti i possibili stati fisici in cui l'agente potrebbe trovarsi.
- **Esempi:**
    - *Percezione Robotica:* Uso di pixel a griglia e scansioni laser per illustrare l'incertezza dei sensori.
    - *Aspirapolvere:* Il non vedere una stanza crea uno stato di credenza con più possibilità.
- **Pianificazione vs. Esecuzione:** Distinzione tra "pianificare di muoversi" (simulazione mentale nello spazio delle credenze) ed "eseguire un movimento" (azione fisica).
### 2. Logica vs. Teoria delle Decisioni
- **Limiti della Logica:** Discusso il "Problema della Qualificazione" (impossibilità di elencare tutte le condizioni) e l'incapacità dell'operatore `OR` di esprimere la probabilità (ad es., `Mal di denti -> Carie` non è sempre vero).
- **Teoria delle Decisioni:** Definita come Teoria della Probabilità + Teoria dell'Utilità.
- **Utilità Attesa Massima (MEU):** La razionalità è definita come la scelta dell'azione che massimizza la MEU.
- **Problema dell'Aeroporto:** Mostra il compromesso tra probabilità di successo (prendere il volo) e costo/utilità del tempo nel decidere quando partire da casa.
### 3. Fondamenti di Reti Bayesiane
- **Definizione:** Una Rete Bayesiana è un Grafo Aciclico Diretto (DAG) che rappresenta le dipendenze tra variabili casuali per rappresentare in modo compatto la Distribuzione di Probabilità Congiunta.
- **Variabili vs. Valori:** Distinzione tra Variabile Casuale (lettere maiuscole, es. Appuntamento) e suoi valori osservati (minuscole, es. presente/assente).
- **Struttura:** I nodi rappresentano variabili casuali; gli archi rappresentano relazioni genitore-figlio che implicano influenza o causalità.
- **Tabelle di Probabilità Condizionata (CPT):**
    - La dimensione dipende dal numero di genitori e dagli stati di valore.
    - **Normalizzazione:** Per una combinazione fissata di valori dei genitori, la somma delle probabilità dei valori del figlio deve essere 1.
- **Esempi Usati:**
    - *Pioggia/Treno:* Pioggia -> Manutenzione -> Treno -> Appuntamento.
    - *Valutazione Studente:* Difficoltà (D), Conoscenza (K), Voto (G), SAT (S) e Lettera (L).
### 4. Meccanica della Probabilità e Inferenza
- **Probabilità Congiunta:** Tutte le interrogazioni si riducono al calcolo della probabilità congiunta delle variabili coinvolte.
- **Regola della Catena:** La probabilità congiunta è il prodotto della probabilità di ciascuna variabile condizionata sui propri genitori.
    - *Esempio di Fattorizzazione:* $P(R, M, T, A) = P(R)P(M\mid R)P(T\mid M,R)P(A\mid T)$.
- **Marginalizzazione:** Sommare sui possibili valori delle variabili nascoste/mancanti (ad es., sommare 'Treno' per trovare la probabilità di 'Appuntamento').
- **Normalizzazione ($\alpha$):** Usata per garantire che i risultati sommino a 1, di fatto l'inverso della probabilità dell'evidenza.
### 5. Tipi di Ragionamento (Inferenza)
- **Ragionamento Causale:** Probabilità di un effetto dato un causa (es., probabilità di Lettera favorevole data bassa Conoscenza).
- **Ragionamento Evidenziale:** Probabilità di una causa dato un effetto (es., probabilità di esame difficile dato un voto 'C').
- **Ragionamento Inter-causale:** Effetto di "spiegazione a vicenda"—conoscere una causa (esame difficile) influisce sulla probabilità di un'altra causa (Conoscenza dello studente) quando l'effetto (Voto) è noto.
### 6. Indipendenza e D-Separation
- **Indipendenza Condizionale:** Conoscere una variabile può cambiare lo stato di indipendenza di variabili connesse.
- **Strutture a V (Collider):** Scenario in cui due genitori indipendenti diventano dipendenti quando il figlio comune è osservato.
- **Percorsi Attivi:** Un percorso è attivo se l'influenza reciproca scorre tra variabili.
### 7. Inferenza Approssimata (Campionamento)
- **Motivazione:** Il calcolo esatto è spesso troppo complesso per reti reali.
- **Campionamento in Avanti:** Generare campioni attraversando la rete dall'alto verso il basso in base alle probabilità delle CPT.
- **Campionamento per Rifiuto:** Gestire l'evidenza (ad es., "Il treno è in orario") scartando i campioni che non corrispondono all'evidenza.
    - *Limitazione:* Inefficiente per eventi rari.
## Contenuti Rimanenti / Omissi
1.  **Calcoli:** Specifiche inferenze (Probabilità di A) e derivazioni matematiche dettagliate delle proprietà di indipendenza condizionale sono state troncate o saltate.
2.  **Apprendimento:** Algoritmi per apprendere la struttura delle Reti Bayesiane dai dati.
3.  **Variabili Complesse:** Gestione di variabili casuali continue nelle Reti Bayesiane.
4.  **D-Separation:** Discussione approfondita su "percorsi attivi" e regole di D-separation solo accennata; dettagli omessi.
## Domande degli Studenti
1.  **Vacuum World:** Perché il robot non può distinguere gli stati se si muove?
    - *Risposta:* Durante la fase di *pianificazione*, l'agente sta solo simulando mentalmente il movimento e non ha ancora percepito la nuova stanza.
2.  **Logica:** È vero che avere una carie implica dolore?
    - *Risposta:* No. Qui la logica fallisce perché puoi avere una carie senza dolore; è necessaria la probabilità.
3.  **Problema dell'Aeroporto:** Come si affronta l'incertezza nella vita reale?
    - *Risposta:* Stimiamo intuitivamente i tempi e li bilanciamo contro il "costo" di perdere il volo (teoria dell'utilità intuitiva).
4.  **Struttura della Rete:** Perché 'Treno' è genitore di 'Appuntamento'?
    - *Risposta:* I genitori tipicamente rappresentano cause; il treno in ritardo influenza lo stato dell'appuntamento.
5.  **CPT:** Dove la somma della probabilità condizionata è 1?
    - *Risposta:* Per uno specifico valore dei genitori (riga), la somma delle probabilità del figlio (colonne) deve essere 1.
6.  **Variabili:** "Puoi dirmi quali sono le variabili casuali in questo caso?"
    - *Risposta:* Il docente ha chiarito i nomi delle variabili (Difficoltà, Conoscenza) in base all'intento dello studente.
7.  **Campionamento:** "Cosa faresti [per il Campionamento per Rifiuto se il treno è in orario]?"
    - *Risposta:* Correttamente identificato che consideriamo solo i casi/campioni in cui il treno è in orario (scartando gli altri).
## Suggerimenti AI
- **Visualizzare gli Stati di Credenza:** Disegnare un albero decisionale per mostrare come uno stato di credenza si ramifica in altri durante la pianificazione. Rafforza la differenza tra simulazione mentale ed esecuzione.
- **Formula MEU:** Scrivere $P(\text{Risultato}\mid\text{Azione}) \times \text{Utilità}(\text{Risultato})$ esplicitamente alla lavagna durante l'esempio dell'aeroporto per ancorare la discussione intuitiva alla matematica.
- **Marginalizzazione:** Scrivere esplicitamente la formula di sommatoria per la marginalizzazione accanto al diagramma della rete per aiutare a visualizzare il "sommare via".
- **Comprensione delle CPT:** Chiedere agli studenti di interpretare una cella specifica in una CPT (ad es., "Cosa rappresenta questo 0,8?") per assicurarsi che leggano la semantica della tabella, non solo le dimensioni.
- **Calcolo di Inferenza:** Eseguire *un* calcolo causale completo alla lavagna con numeri reali. Poiché la derivazione è stata saltata/assegnata, serve un modello da seguire.
- **Percorsi Attivi:** Usare un ausilio visivo per evidenziare i percorsi "bloccati" o "aperti" per rendere più concreti il ragionamento inter-causale e le strutture a V.
- **Eventi Rari:** Prima di spiegare i limiti del Campionamento per Rifiuto, chiedere alla classe di proporre soluzioni per eventi rari come introduzione a temi futuri quali la Pesatura di Verosimiglianza.