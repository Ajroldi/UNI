## Strutture Dati Fondamentali di Python
### Tuple
Una tupla è una raccolta di elementi racchiusi tra parentesi tonde, come `(18.849..., 28.274...)`. La funzione `type()` può essere utilizzata per verificare che una variabile sia una tupla.
*   **Indicizzazione**: Gli elementi sono accessibili utilizzando un'indicizzazione basata su zero con parentesi quadre (es. `R[0]`, `R[1]`), che differisce dall'indicizzazione basata su uno di MATLAB.
*   **Immutabilità**: Le tuple sono **immutabili**, il che significa che i loro elementi non possono essere modificati dopo la creazione. Tentare di riassegnare un elemento (es. `R[1] = 6`) genererà un `TypeError`.
*   **Eterogeneità**: Le tuple possono contenere elementi di tipi di dati diversi, ad esempio: `c = (1, 'a', "string", mt.sqrt(2))`.
### Liste (`[]`) vs. Tuple (`()`)
*   **Liste**: Definite con parentesi quadre `[]`, le liste sono **mutabili**, il che significa che i loro elementi possono essere modificati dopo la creazione.
*   **Tuple**: Definite con parentesi tonde `()`, le tuple sono **immutabili**.
*   L'operatore `+` sulle liste esegue la concatenazione, non l'addizione matematica.
### Dizionari (`{}`)
*   Definiti con parentesi graffe `{}`, i dizionari memorizzano dati in coppie `chiave: valore`.
*   Sono utili per creare tabelle di ricerca, come la mappatura dei numeri romani ai valori interi (es. `'M': 1000`).
*   I valori sono accessibili tramite la loro chiave: `symbols['M']` restituisce `1000`.
### Stringhe
Le stringhe si comportano come tuple e liste per quanto riguarda l'indicizzazione.
*   **Indicizzazione Positiva**: Inizia da `0` per il primo carattere.
*   **Indicizzazione Negativa**: Inizia da `-1` per l'ultimo carattere. `s[-1]` è un modo comune per accedere all'ultimo elemento.
## Manipolazione delle Sequenze: Indicizzazione e Slicing
Lo slicing viene utilizzato per estrarre porzioni di sequenze come stringhe, liste o tuple.
### Sintassi dello Slicing
*   La sintassi di base è `[start:stop]`. Il carattere all'indice `stop` è **escluso** (non incluso). `s[0:4]` estrae i caratteri agli indici 0, 1, 2 e 3.
*   **Omissione degli Indici**:
    *   `s[:4]` parte dall'inizio (indice `0`).
    *   `s[1:]` arriva fino alla fine, includendo l'ultimo carattere.
*   La sintassi completa è `[start:stop:step]`, che differisce dal `start:step:stop` di MATLAB. Il valore `step` determina l'incremento. Ad esempio, `s[::2]` estrae ogni secondo carattere.
### Inversione con Slicing
*   Utilizzando uno step negativo di `-1` si inverte la direzione di attraversamento.
*   Il comando `s[::-1]` è un'idioma comune per invertire un'intera sequenza.
## Funzioni e Moduli
### Definizione e Output delle Funzioni
Una funzione può essere creata per eseguire un compito specifico, come `circle_one` che calcola l'area e la lunghezza di un cerchio.
*   Le funzioni possono includere una `docstring` che fornisce documentazione, accessibile tramite `help(function_name)`.
*   Una funzione può restituire più valori, che sono confezionati in una **tupla**. Ad esempio, `circle_one(3)` restituisce due valori numerici come una tupla.
### Gestione degli Output delle Funzioni
In Python, una funzione restituisce un insieme fisso di output, a differenza di MATLAB dove il numero di output può variare. Ci sono diversi modi per gestire questi output:
1.  **Decomposizione degli Output**: La tupla restituita può essere "decomposta" assegnando i suoi elementi a più variabili direttamente: `L, A = circle_one(3)`.
2.  **Scarto degli Output Non Desiderati**: Utilizzare un trattino basso (`_`) come segnaposto per ignorare un valore restituito e risparmiare memoria: `_, A = circle_one(3)`.
3.  **Selezione Inline**: Chiamare la funzione e selezionare immediatamente un elemento specifico dalla tupla restituita in una singola riga: `A = circle_one(3)[1]`.
### Importazione e Ispezione dei Moduli
Ci sono diversi modi per importare i moduli:
1.  `import math`: Richiede il prefisso con il nome del modulo (es. `math.pi`).
2.  `import math as mt`: Utilizza un alias, richiedendo l'alias come prefisso (es. `mt.pi`).
3.  `from math import pi`: Importa direttamente un oggetto specifico, senza bisogno di prefisso.
4.  `from math import cos, sin, pi`: Importa più oggetti specifici.
La funzione `dir()` (es. `dir(mt)`) può essere utilizzata per elencare tutte le funzioni, costanti e altri oggetti contenuti in un modulo importato.
## Controllo del Flusso e Logica
### Logica Condizionale e Indentazione
*   Una dichiarazione `if-elif-else` viene utilizzata per eseguire codice basato su condizioni.
*   L'operatore `==` viene utilizzato per il confronto logico (es. `if unit == 'K'`), mentre `=` è per l'assegnazione di variabili.
*   **Indentazione** è una parte obbligatoria della sintassi di Python utilizzata per definire i blocchi di codice per `if`, `else`, cicli e funzioni.
### Cicli
*   Un ciclo `while` può essere utilizzato per ripetere un blocco di codice finché una condizione è vera. Ad esempio, `while 1.0 + epsilon != 1.0`.
### Operatori Chiave
*   `//` (Divisione Intera): Esegue la divisione e restituisce il risultato intero (es. `13 // 2` è `6`).
*   `%` (Modulo): Restituisce il resto di una divisione (es. `13 % 2` è `1`).
*   `!=`: Operatore di confronto "Diverso da".
*   `+=`, `-=`: Operatori di incremento e decremento per un codice conciso (es. `decimal += value`).
### Il Valore `None`
Il valore `None` può essere assegnato a una variabile per rappresentare l'assenza di un valore, spesso utilizzato per gestire una condizione di errore.
## Problemi di Programmazione e Algoritmi
### Conversione di Temperature
Il compito è scrivere una funzione che converte la temperatura da un singolo input di stringa come `"75K"` o `"25F"`.
*   **Processo**:
    1.  **Estrazione Dati**: Il valore numerico e l'unità vengono estratti dalla stringa di input utilizzando lo slicing (es. `temper[:-1]` per il numero e `temper[-1]` per l'unità).
    2.  **Casting di Tipo**: La stringa numerica estratta viene convertita in un numero utilizzando `float()`.
    3.  **Logica Condizionale**: Un blocco `if-elif-else` determina quale formula di conversione applicare in base all'unità (`'K'` o `'F'`).
### Calcolo dell'Epsilon della Macchina
L'epsilon della macchina (`ε`) è il numero più piccolo tale che `1.0 + ε` è diverso da `1.0`.
*   **Algoritmo**:
    1.  Inizializza `epsilon = 1.0`.
    2.  Utilizza un ciclo `while` con la condizione `1.0 + epsilon != 1.0`.
    3.  All'interno del ciclo, dividi `epsilon` per 2 (`epsilon /= 2`).
    4.  Il ciclo si ferma quando `epsilon` è troppo piccolo per essere registrato. L'epsilon della macchina finale è `2 *` l'ultimo valore di `epsilon`.
*   **Verifica**: Il risultato corrisponde alla costante `eps` nella libreria NumPy.
### Conversione da Numero Romano a Numero Decimale
Questo algoritmo utilizza un dizionario per mappare i simboli romani ai valori interi.
*   **Algoritmo**:
    1.  Inizializza un risultato `decimale` a `0`.
    2.  Scorri la stringa del numero romano utilizzando `range()`, confrontando ogni carattere (`sinistra`) con quello successivo (`destra`).
    3.  **Regola di Sottrazione**: Se `valore(sinistra) < valore(destra)`, sottrai `valore(sinistra)` dal totale (`decimale -= valore`).
    4.  **Regola di Addizione**: Altrimenti, aggiungi `valore(sinistra)` al totale (`decimale += valore`).
    5.  Dopo il ciclo, aggiungi il valore dell'ultimo carattere al totale.
### Conversione da Intero a Binario
Questo algoritmo utilizza la divisione ripetuta per convertire un intero nella sua rappresentazione binaria.
*   **Algoritmo**:
    1.  Inizializza una lista vuota per memorizzare i resti.
    2.  Utilizza un ciclo `while` che continua finché il numero non è `0`.
    3.  All'interno del ciclo, aggiungi il resto (`numero % 2`) alla lista e aggiorna il numero utilizzando la divisione intera (`numero //= 2`).
    4.  Dopo il ciclo, inverti la lista dei resti (`[::-1]`) per ottenere la corretta sequenza binaria.
    5.  La lista finale può essere formattata in una stringa pulita utilizzando `str()` e il metodo `.replace()`.
## 📅 Prossimi Appuntamenti
*   [ ] Scrivere una piccola funzione che calcola le lunghezze dei lati B e C di un triangolo rettangolo, data la lunghezza del lato A e dell'angolo gamma. Ricorda che `sin` e `cos` del modulo `math` si aspettano angoli in radianti.
*   [ ] Continuare a lavorare sulla funzione di conversione della temperatura che accetta un input di stringa come "75K".
*   [ ] Completare l'esercizio per scrivere una funzione per convertire un intero nella sua rappresentazione esadecimale. L'algoritmo è simile alla conversione binaria ma utilizza una base di 16.
*   [ ] La prossima sessione introdurrà il modulo NumPy.