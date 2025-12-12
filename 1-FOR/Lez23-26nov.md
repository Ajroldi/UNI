Data e ora: 2025-12-11 12:27:29
Luogo: [Luogo tradotto]
Corso: [Nome corso tradotto]
## Panoramica
Questa lezione ha coperto i fondamenti della Programmazione Lineare Intera (ILP), passando dall’intuizione geometrica a tecniche algoritmiche avanzate. Si è iniziato esplorando la relazione tra rilassamento LP e insieme ammissibile intero, usando attività interattive per mostrare il concetto di inviluppo convesso e i limiti dell’arrotondamento euristico. Sono state introdotte le basi teoriche come unimodularità e totale unimodularità. Si è poi passati all’algoritmo dei piani di taglio, dettagliando il metodo di Chvátal-Gomory per generare disuguaglianze valide. Si è concluso con un approfondimento sui tagli di Gomory, inclusa la derivazione teorica tramite base ottima e un’applicazione numerica passo-passo su un problema di minimizzazione.
## Contenuti rimanenti
1.  Rappresentazione geometrica dei tagli generati.
## Contenuti trattati
### 1. Introduzione alla Programmazione Lineare Intera (ILP) e attività geometrica
-   Presentate due strategie principali per risolvere ILP: sfruttare l’eredità LP e l’enumerazione esaustiva.
-   Sottolineata l’importanza critica della formulazione in ILP rispetto a LP; formulazioni diverse definiscono lo stesso insieme intero ma poliedri diversi.
-   Svolta un’attività in classe con scheda e “Kahoot” per risolvere geometricamente un problema 2D.
-   Dimostrato che la soluzione ottima del rilassamento LP (frazionaria) non si arrotonda necessariamente alla soluzione intera ottima.
-   Mostrato che l’euristica del “punto intero più vicino” spesso fallisce (es. punto (2,5) era il più vicino ma (4,3) era ottimo).
### 2. Il concetto di inviluppo convesso
-   Definito l’inviluppo convesso come il più piccolo poliedro convesso che contiene tutti i punti interi ammissibili.
-   Se l’inviluppo convesso è noto, risolvere il rilassamento LP dà automaticamente la soluzione intera ottima.
-   Svantaggio: descrivere l’inviluppo convesso richiede in genere un numero esponenziale di disuguaglianze rispetto alla dimensione.
-   Citati software come “Porta” che convertono tra vertici e disuguaglianze.
-   Concluso che non serve l’inviluppo completo, ma solo la descrizione locale vicino all’ottimo.
### 3. Unimodularità e totale unimodularità
-   Definita matrice unimodulare: ogni sottomatrice a rango pieno ha determinante 1 o -1.
-   Definita matrice totalmente unimodulare (TU): *ogni* sottomatrice ha determinante 0, 1 o -1.
-   Collegato il concetto ai flussi di rete: la matrice incidenza nodo-arco è TU.
-   Conseguenza: se la matrice dei vincoli $A$ è TU e il termine noto $b$ è intero, la soluzione LP è garantita intera (proprietà di integrità).
-   Dimostrazione matematica tramite regola di Cramer/inversa di matrice (determinante 1/-1 al denominatore implica nessuna frazione).
### 4. Problema di localizzazione di impianti ed esercizio di formulazione
-   Ripreso il problema di localizzazione: decidere dove installare centri di servizio (antenne) e assegnare clienti (cellulari) per minimizzare il costo.
-   Confrontate due formulazioni per il vincolo “centro attivo”:
    1.  $x_{ij} \le y_j$ (formulazione forte)
    2.  $\sum x_{ij} \le U_j y_j$ (formulazione debole / stile big-M)
-   Spiegato che la seconda è più compatta da scrivere, ma la prima è “più stretta” e vicina all’inviluppo convesso, migliorando le prestazioni del risolutore.
-   **Esercizio sfida:** Introdotto un nuovo vincolo: i clienti *devono* essere assegnati al *centro attivo più vicino* (non a uno qualsiasi). Assegnato come esercizio a casa.
### 5. Disuguaglianze valide, tagli e algoritmo dei piani di taglio
-   Definito **Rilassamento ($P'$):** Stesso problema senza vincoli di integrità.
-   Definita **Disuguaglianza valida:** Disuguaglianza soddisfatta da tutti i punti interi ammissibili ($F$).
-   Definito **Taglio:** Disuguaglianza valida che esclude la soluzione ottima LP corrente ($x'$).
-   Schema dell’**algoritmo dei piani di taglio:**
    1.  Risolvi il rilassamento LP.
    2.  Se la soluzione è intera, Stop.
    3.  Altrimenti, trova un taglio che separa la soluzione frazionaria dall’insieme intero.
    4.  Aggiungi il taglio e ripeti.
### 6. Tagli di Chvátal-Gomory (metodo di generazione)
-   Introdotto un metodo universale per generare disuguaglianze valide (metodo di Chvátal).
-   **Step 1:** Combinazione lineare non negativa delle disuguaglianze esistenti (con moltiplicatori $u_i$).
-   **Step 2:** Si “indeboliscono” i coefficienti a sinistra arrotondandoli per difetto ($\lfloor g_j \rfloor$).
-   **Step 3:** Si arrotonda per difetto il termine noto ($\lfloor \gamma \rfloor$), giustificato perché variabili intere per coefficienti interi danno somma intera.
### 7. Applicazione al problema di matching (grafo triangolo)
-   Formulazione del problema di matching massimo con variabili binarie ($x_{ij}$) e vincoli (somma archi incidenti $\le 1$).
-   Confronto tra soluzione intera (selezione di un arco) e soluzione LP continua (tutte variabili a $1/2$ per il triangolo).
-   Dimostrazione della procedura di Chvátal-Gomory su questo esempio:
    -   scelta dei moltiplicatori ($1/2$ per ogni vincolo);
    -   combinazione lineare;
    -   arrotondamento dei coefficienti e del termine noto.
-   Ottenuta la disuguaglianza valida $x_{12} + x_{13} + x_{23} \le 1$ che esclude la soluzione frazionaria.
### 8. Teoria dei tagli di Gomory (base ottima)
-   Formulazione del problema in forma duale standard (uguaglianze, non negatività, integrità).
-   Definizione del rilassamento LP ($\bar{P}$) e partizione della base ottima ($x_B$ e $x_N$).
-   Riscrittura algebrica dei vincoli tramite inversa della base: $x_B + A_B^{-1}A_N x_N = A_B^{-1}b$.
-   Scelta di una riga sorgente dove la variabile di base ($x_{B_h}$) è frazionaria.
-   Derivazione della formula del taglio di Gomory applicando il pavimento ai coefficienti e al termine noto della riga sorgente.
-   Dimostrazione teorica che il taglio è valido per soluzioni intere ma esclude la soluzione LP frazionaria corrente.
### 9. Esempio numerico di tagli di Gomory
-   Risoluzione passo-passo di un problema di minimizzazione con due vincoli e variabili intere.
-   Identificazione della base ottima $B=\{1, 3\}$ e calcolo di $A_B^{-1}$ e $A_N$.
-   Calcolo della soluzione ottima frazionaria ($x_1 = 13/3, x_3 = 35/3$).
-   Calcolo dettagliato dei coefficienti per le variabili non di base tramite prodotto vettore-matrice.
-   Generazione di due tagli distinti (Taglio 1 e Taglio 2) applicando il pavimento ai coefficienti calcolati.
-   Enfasi su come arrotondare i numeri negativi (es. pavimento di $-1/3$ è $-1$).
-   Verifica numerica che i tagli generati sono violati dalla soluzione frazionaria corrente.
## Domande degli studenti
1.  **[Dalla discussione: "È una mia scelta... come trovo i coefficienti?"]**
    -   Il docente ha spiegato che nell’esempio iniziale di matching la scelta dei moltiplicatori (1/2) era arbitraria per mostrare una disuguaglianza valida. Il punto chiave è trovare coefficienti che garantiscano un taglio, cosa che il metodo di Gomory (spiegato dopo) fa in modo sistematico.