# RICERCA OPERATIVA - LEZIONE: VINCOLI E TRASFORMAZIONI
*Corso del Prof. Federico Malucelli - Politecnico di Milano*

## 📋 INFORMAZIONI TECNICHE
- **Video:** Federico Malucelli's Personal Room-20250923 1010-1
- **Data elaborazione:** 24 Settembre 2025, ore 13:45
- **Sistema:** Analisi ULTRA con qualità massima
- **AI:** Whisper base su CPU | **Timestamp:** 6.414 parole precise
- **Frame:** 200 estratti ultra-densi | **Formule:** 155 frame matematici

---

## 🗂️ INDICE DELLA LEZIONE

### PARTE I: RIPASSO PROBLEMA DELLO ZAINO
- **[10:49 - 13:00]** Soluzione ottimale: 75 punti con 3 medium + 1 large
- **[11:22 - 11:27]** Vincolo capacità: 25 chilogrammi
- **[11:43 - 12:10]** Metodologie valide: forza bruta, euristiche, regole empiriche

### PARTE II: TIPOLOGIE DI VINCOLI
- **[13:26 - 13:52]** Vincoli di disponibilità (≤)
- **[14:08 - 14:15]** Vincoli di fabbisogno (≥) 
- **[14:15 - 14:18]** Trasformazioni tra vincoli (parte più importante)
- **[14:25 - 14:29]** Vincoli di miscelazione (blending)
- **[14:29 - 14:34]** Vincoli logici e conservazione flusso

### PARTE III: PROBLEMA DELLA DIETA
- **[15:05 - 15:45]** Scenario poke bowl: almeno 700 calorie
- **[17:00 - 18:00]** Definizione variabili decisionali continue
- **[18:00 - 20:00]** Formulazione matematica vincolo calorico

### PARTE IV: TRASFORMAZIONI BLACK BOX
- **[20:14 - 22:46]** Concetto Black Box per standardizzazione
- **[22:46 - 23:18]** Trasformazione ≥ → ≤: moltiplicare per -1
- **[23:18 - 27:02]** Trasformazione = → ≤: variabili slack

### PARTE V: VINCOLI DI MISCELAZIONE
- **[30:15 - 36:06]** Industria siderurgica e petrolifera
- **[31:19 - 35:09]** Esempio: 30% calorie da cibi non trasformati
- **[35:09 - 36:06]** Linearizzazione del vincolo non lineare

### PARTE VI: VINCOLI LOGICI E GIOCO KAHOOT
- **[36:23 - 37:55]** Variabili binarie come variabili logiche
- **[40:30 - 49:25]** Quiz interattivo su traduzioni logiche
- **[50:27 - 53:41]** Problema di soddisfacibilità (SAT)

### PARTE VII: VINCOLI DI ESCLUSIONE
- **[54:02 - 62:18]** "O salmone O pollo": formulazione con variabili binarie
- **[57:19 - 62:36]** Variabili di collegamento (linking constraints)
- **[71:11 - 72:12]** Esercizio per la settimana: vincolo "tutto o niente"

### PARTE VIII: CONSERVAZIONE DEL FLUSSO
- **[72:59 - 90:00]** Esempio campo petrolifero
- **[75:35 - 82:24]** Formulazione matematica per ogni nodo
- **[82:51 - 90:00]** Forma standardizzata con variabili slack

---

## 📚 CONTENUTO DELLA LEZIONE

### 🎒 PARTE I: RIPASSO PROBLEMA DELLO ZAINO {#problema-zaino}
*Timing: 10:49 - 13:00*

#### Soluzione del Problema Settimanale
**[10:49 - 11:01]**
> "Bene, iniziamo. La domanda della settimana scorsa era questa. Dovete decidere cosa potete mettere nel vostro zaino."

**Parametri del problema:**
- **Obiettivo:** Massimizzare il punteggio di sopravvivenza (valore verde nella tabella)
- **Vincolo:** Capacità zaino = 25 chilogrammi
- **Tipo:** Selezionare un oggetto per categoria senza superare la capacità

#### Soluzione Ottimale Rivelata
**[11:27 - 11:41]**
> "La soluzione ottimale è questa: 75 punti. E si ottiene selezionando medium, medium, medium, e large nella categoria frigo."

**Note del professore:**
**[11:43 - 12:10]**
> "Non era importante ottenere la soluzione ottimale. L'intento di questa attività era attivare il vostro ragionamento di ottimizzazione."

**Metodologie accettate:**
- Enumerazione per forza bruta
- Euristiche
- Regole empiriche
- Qualsiasi approccio ragionevole

### 🔧 PARTE II: CLASSIFICAZIONE DEI VINCOLI {#tipologie-vincoli}
*Timing: 13:26 - 14:34*

#### Vincoli di Disponibilità (≤)
**[13:45 - 14:06]**
> "Abbiamo già visto il primo tipo di vincolo, che è il più comune: il vincolo di disponibilità. Avete un gruppo di risorse, risorse limitate. Dovete usarle e il vostro uso non può superare la disponibilità."

**Caratteristiche:**
- **Forma matematica:** `Uso ≤ Disponibilità`
- **Esempio:** Problema assemblaggio cellulari
- **Significato:** Risorse limitate che non possono essere superate

#### Vincoli di Fabbisogno (≥)
**[14:08 - 14:15]**
> "Ora vedremo i vincoli di fabbisogno. Vedremo come trasformare i vincoli da un tipo all'altro. E questa sarà la parte più importante di oggi."

**Caratteristiche:**
- **Forma matematica:** `Quantità ≥ Fabbisogno_minimo`
- **Significato:** Requisiti minimi da soddisfare
- **Opposto:** Dei vincoli di disponibilità

### 🍱 PARTE III: ESEMPIO PRATICO - PROBLEMA DELLA DIETA {#problema-dieta}
*Timing: 15:05 - 20:00*

#### Contesto del Problema
**[15:05 - 15:30]**
> "Vi ho detto che una tipica applicazione della Ricerca Operativa è la dieta. Non c'è continuità al Politecnico, quindi dovete andare in giro e cercare del cibo da qualche parte."

**Scenario:** Poke bowl con selezione di ingredienti
**Vincolo nutrizionale:** Almeno 700 calorie per le lezioni pomeridiane

#### Tabella Ingredienti e Valori Nutrizionali
| Ingrediente | Calorie per unità |
|-------------|------------------|
| Riso nero | 300 |
| Cereali | 250 |
| Lattuga | 35 |
| Carote | 40 |
| Ananas | 180 |
| Fagioli | 100 |
| Salmone | 200 |
| Pollo | 550 |
| Salsa | 400 |

#### Definizione delle Variabili
**[16:41 - 17:39]**
> "Quali sono le variabili? [...] Non è binario. È la quantità. Supponiamo che siate in grado di chiedere esattamente una certa quantità di grammi di riso o di lattuga o carote. Non è zero-uno. È una variabile continua."

**Variabili decisionali:**
```
x_riso, x_cereali, x_lattuga, x_carote, x_ananas, 
x_fagioli, x_salmone, x_pollo, x_salsa ≥ 0
```

#### Formulazione del Vincolo Calorico
**[18:35 - 20:00]**
```
300·x_riso + 250·x_cereali + 35·x_lattuga + 40·x_carote + 
180·x_ananas + 100·x_fagioli + 200·x_salmone + 
550·x_pollo + 400·x_salsa ≥ 700
```

### ⚡ PARTE IV: TRASFORMAZIONI BLACK BOX {#trasformazioni-black-box}
*Timing: 20:14 - 27:02*

#### Concetto di Scatola Nera
**[20:14 - 20:46]**
> "Assumiamo che, come abbiamo fatto la scorsa volta, abbiamo una scatola nera, una macchina che risolve problemi in questa forma. Massimizza una certa funzione obiettivo. E tutti i vincoli devono essere minore o uguale."

**Specifiche del Black Box:**
- **Input:** Parametri C, A, B
- **Output:** Soluzione ottimale X* e valore
- **Forma standard:** Tutti i vincoli devono essere ≤

<img width="335" height="127" alt="image" src="https://github.com/user-attachments/assets/bd29567f-bb79-4e83-8785-adf51b17e2a9" />

#### Trasformazione ≥ → ≤
**[22:38 - 22:57]**
> "È sufficiente moltiplicare per meno uno qui e meno uno qui, entrambi i lati per meno uno. E essendo una disuguaglianza, devo invertire la disuguaglianza."

**Procedimento:**
```
A·x ≥ b  →  -A·x ≤ -b
```

#### Trasformazione = → ≤ (Variabili Slack)
**[23:18 - 25:27]**

**Per vincoli ≤:**
> "Aggiungiamo qui la parte mancante per raggiungere B. E questa è non negativa. Si chiama variabile slack."

```
A·x ≤ b  →  A·x + s = b  (con s ≥ 0)
```

**Per vincoli ≥:**
```
A·x ≥ b  →  A·x - s = b  (con s ≥ 0)
```

**Significato delle variabili slack:**
**[25:56 - 26:01]**
> "Il significato è cosa manca per raggiungere il confine del vincolo."

#### Trasformazione = → {≤, ≥}
**[27:12 - 29:46]**
**Interpretazione geometrica:**
> "L'uguaglianza è una linea. Questa è una linea. La linea è l'intersezione di due semipiani."

```
A·x = b  →  {A·x ≤ b
           {A·x ≥ b
```

### 🏭 PARTE V: VINCOLI DI MISCELAZIONE {#vincoli-miscelazione}
*Timing: 30:15 - 36:06*

#### Origine Storica
**[30:19 - 30:43]**
> "I vincoli di miscelazione sono almeno dal punto di vista storico molto importanti. Provengono dall'industria siderurgica, che è stata una delle prime applicazioni, o dall'industria petrolifera, che sono state tra le prime applicazioni della ricerca operativa."

**Settori di applicazione:**
- Industria siderurgica (miscele di minerali per produrre acciaio)
- Industria petrolifera
- Qualsiasi processo che richieda percentuali specifiche di componenti

#### Esempio: Vincolo di Percentuale nella Dieta
**[31:19 - 31:44]**
> "Nel nostro problema della dieta, il dietologo dice: per essere sani, dovete mangiare 700 calorie, ma almeno il 30% di quelle calorie deve provenire da cibo non trasformato, ad esempio frutta e verdura."

**Ingredienti "non trasformati":** Lattuga, carote, ananas, fagioli

#### Formulazione Non Lineare (Problema!)
**[33:42 - 34:57]**
```
(35·x_lattuga + 40·x_carote + 180·x_ananas + 100·x_fagioli) / 
(300·x_riso + ... + 400·x_salsa) ≥ 0.30
```

**Problema:** Il vincolo è **non lineare**!

#### Linearizzazione della Formulazione
**[35:26 - 36:06]**
> "È non lineare. Tuttavia, possiamo linearizzarlo molto facilmente considerando il fatto che il denominatore è diverso da zero, maggiore di zero. E possiamo moltiplicare tutto a destra e a sinistra per questa quantità."

**Risultato linearizzato:**
```
35·x_lattuga + 40·x_carote + 180·x_ananas + 100·x_fagioli ≥ 
0.30 · (300·x_riso + ... + 400·x_salsa)
```

### 🎯 PARTE VI: VINCOLI LOGICI E QUIZ INTERATTIVO {#vincoli-logici}
*Timing: 36:23 - 53:41*

#### Introduzione ai Vincoli Logici
**[36:23 - 36:59]**
> "Se abbiamo variabili logiche, allora possiamo formulare qualsiasi formula di logica proposizionale come vincolo lineare. E la cosa è molto facile. È abbastanza intuitivo."

**Corrispondenze base:**
- **OR:** È più o meno una somma
- **AND:** È più o meno l'intersezione

#### Quiz Kahoot: Traduzioni Logiche
**[40:30 - 49:25]** Sessione interattiva con 4 domande principali

##### Domanda 1: OR Logico (x ∨ y)
**[40:53 - 42:48]**

**Tavola di verità:** Falso solo quando entrambi sono falsi
**Risposta corretta:** `x + y ≥ 1`

**Spiegazione:**
> "Se guardate i vincoli che vi ho proposto, l'unico che taglia fuori questo caso è x + y maggiore uguale di uno: almeno uno dei due deve essere uno."

##### Domanda 2: AND Logico (x ∧ y)
**[43:05 - 44:54]**

**Tavola di verità:** Vero solo quando entrambi sono veri
**Risposta corretta:** `x ≥ 1 AND y ≥ 1`

**Spiegazione:**
> "L'unica condizione per modellare l'AND logico è l'intersezione: qualcosa deve valere contemporaneamente. Quindi dobbiamo avere che questo e questo devono essere uguali a uno simultaneamente."

##### Domanda 3: Implicazione (x → y)
**[45:19 - 47:43]**

**Tavola di verità:** Falso solo quando x=vero e y=falso
**Risposta corretta:** `x ≤ y`

**Spiegazione:**
> "Non possiamo avere che una conseguenza falsa segua da un'ipotesi vera. In questo caso, vedete x uguale a uno non può essere minore o uguale a y uguale a zero."

##### Domanda 4: XOR (x ⊕ y)
**[48:09 - 49:18]**

**Tavola di verità:** Vero quando le variabili hanno valori diversi
**Risposta corretta:** `x + y = 1`

**Spiegazione:**
> "L'idea è partire dai casi vietati. I casi vietati sono entrambi uno o entrambi zero. È come avere x + y uguale a uno."

#### Problema di Soddisfacibilità (SAT)
**[50:20 - 53:23]**
**[50:27 - 50:35]**
> "Per un informatico c'è un problema, un problema. La madre di tutti i problemi: la soddisfacibilità."

**Definizione:** Data una formula logica proposizionale, trovare un'assegnazione di verità che la renda vera.

**Importanza:**
**[51:03 - 51:25]**
> "È la madre di tutti i problemi perché nella complessità computazionale ogni problema può essere ridotto al problema di soddisfacibilità. Quindi se siete in grado di risolvere in tempo polinomiale il problema di soddisfacibilità, potete risolvere qualsiasi problema anche quelli che non sono ancora stati inventati."

### 🔗 PARTE VII: VINCOLI DI ESCLUSIONE {#vincoli-esclusione}
*Timing: 54:02 - 72:12*

#### Problema Pratico: "O Salmone O Pollo"
**[54:02 - 54:18]**
> "Se andate al negozio di poke, sapete che c'è una regola. O pagate di più o potete prendere solo un tipo di proteina. Quindi nel nostro caso specifico, era o salmone o pollo."

#### Approccio Non Lineare (Da evitare!)
**[56:20 - 57:15]**
**Suggerimento di uno studente:** "Il prodotto deve essere zero"
```
x_salmone · x_pollo = 0
```

**Problema:** Vincolo **non lineare**!

#### Soluzione con Variabili Binarie
**[57:19 - 59:55]**

**Step 1: Introdurre variabili "flag"**
```
y_salmone = {1 se x_salmone > 0
           {0 altrimenti

y_pollo = {1 se x_pollo > 0
         {0 altrimenti
```

**Step 2: Vincolo logico**
```
y_salmone + y_pollo ≤ 1
```

#### Vincoli di Collegamento (Linking Constraints)
**[62:19 - 62:36]**
> "Ricordate che questo è un errore tipico: dimenticate questo vincolo di collegamento, ma i vincoli di collegamento sono della massima importanza."

**Formulazione:**
```
x_salmone ≤ M · y_salmone
x_pollo ≤ M · y_pollo
```

dove M è una costante sufficientemente grande

**Spiegazione del funzionamento:**
**[63:19 - 63:45]**
> "In questo modo vedete che se x_salmone è zero, possiamo impostare y a zero e siamo a posto. Se x_salmone è maggiore di zero, l'unico modo per averlo maggiore di zero senza violare questo vincolo è impostare y_salmone a uno."

#### Esercizio per la Settimana Successiva
**[71:11 - 72:12]**
> "Dovete scrivere questo vincolo qui, che è scritto a parole. O la variabile continua come salmone è uguale a zero o è all'interno di un dato intervallo dove il lato sinistro è diverso da zero."

**Problema da risolvere:**
```
x_salmone = 0  OPPURE  a ≤ x_salmone ≤ b
```
dove a > 0 e b > a

**Significato:** "Tutto o niente" - o non prendi salmone, o ne prendi almeno una quantità minima a.

### 🌊 PARTE VIII: VINCOLI DI CONSERVAZIONE DEL FLUSSO {#conservazione-flusso}
*Timing: 72:59 - 90:00*

#### Esempio: Campo Petrolifero
**[73:12 - 74:29]**

**Componenti del sistema:**
- 2 pozzi (disponibilità: 10.000 e 15.000 unità)
- 1 stazione di pompaggio 
- 2 raffinerie (fabbisogno: 13.000 e 9.000 unità)
- Rete di tubi colleganti

**Obiettivo:** Descrivere come distribuire il petrolio lungo i tubi rispettando i vincoli di capacità

#### Definizione delle Variabili
**[75:49 - 76:15]**
```
x_ij = flusso di petrolio nel tubo da i a j
```

#### Vincoli per Tipo di Nodo

##### Pozzi (Vincoli di Disponibilità)
**[76:34 - 78:03]**

**Pozzo 1:**
```
x_1,2 + x_1,3 ≤ 10.000
```

**Pozzo 2:**
**[78:28 - 79:29]**
> "Come misuro la quantità di petrolio che estraggo dal pozzo 2? Il petrolio che esce dal pozzo 2 ma devo sottrarre la quantità che arriva dal pozzo 1 perché quella quantità non è estratta dal pozzo 2."

```
x_2,3 + x_2,4 - x_1,2 ≤ 15.000
```

##### Stazione di Pompaggio (Conservazione del Flusso)
**[79:45 - 80:19]**
> "Questo è facile perché, come ho detto, tutto quello che entra deve uscire."

**Stazione 3:**
```
x_1,3 + x_2,3 - x_3,4 - x_3,5 = 0
```

##### Raffinerie (Vincoli di Fabbisogno)

**Raffineria 5:**
```
x_3,5 + x_4,5 ≥ 9.000
```

**Raffineria 4:**
**[81:21 - 81:53]**
> "Per la raffineria 4, dobbiamo usare lo stesso argomento che usiamo per il pozzo 2. Qual è la quantità di petrolio che si ferma nel 4? È la quantità totale che arriva nel 4, meno la quantità che va al 5."

```
x_2,4 + x_3,4 - x_4,5 ≥ 13.000
```

#### Forma Standardizzata con Variabili Slack
**[82:58 - 89:55]**

**Trasformazione in uguaglianze:**
```
Pozzo 1:     -x_1,2 - x_1,3 + s_1 = -10.000
Pozzo 2:     +x_1,2 - x_2,3 - x_2,4 + s_2 = -15.000
Stazione 3:  +x_1,3 + x_2,3 - x_3,4 - x_3,5 = 0
Raffineria 4: +x_2,4 + x_3,4 - x_4,5 - s_4 = 13.000
Raffineria 5: +x_3,5 + x_4,5 - s_5 = 9.000
```

#### Convenzioni di Segno
**[85:13 - 87:17]**
> "Di solito la convenzione è: qualcosa in uscita, negativo; qualcosa in entrata, positivo."

**Proprietà importante:**
**[84:44 - 85:01]**
> "Le variabili appaiono nel nostro sistema di vincoli esattamente due volte. Una volta quando consideriamo il flusso in uscita, e una volta quando consideriamo il flusso in entrata."

#### Interpretazione delle Variabili Slack
**[88:45 - 89:49]**
> "È come avere qualcosa che esce dai nodi e non va da nessuna parte. È come avere un arco senza fine nel nostro grafo."

**Significato pratico:**
- Nei pozzi: petrolio estratto ma "sprecato"
- Nelle raffinerie: capacità di raffinazione non utilizzata

---

## 📝 CONCETTI CHIAVE DA RICORDARE

### Trasformazioni Fondamentali
1. **≥ → ≤:** Moltiplicare per -1 e invertire il segno
2. **= → ≤:** Aggiungere variabili slack non negative
3. **= → {≤,≥}:** Un'uguaglianza è l'intersezione di due disuguaglianze

### Vincoli Logici - Traduzioni Base
- **OR (∨):** `x + y ≥ 1`
- **AND (∧):** `x ≥ 1 AND y ≥ 1`
- **Implicazione (→):** `x ≤ y`
- **XOR (⊕):** `x + y = 1`

### Linking Constraints per Variabili Binarie
**Schema generale:**
```
variabile_continua ≤ M · variabile_binaria
```

### Proprietà dei Vincoli di Flusso
- Ogni variabile di flusso appare esattamente 2 volte
- Una volta positiva (nodo di arrivo), una volta negativa (nodo di partenza)
- Le variabili slack rappresentano "sprechi" o capacità inutilizzata
