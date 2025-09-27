# RICERCA OPERATIVA - LEZIONE 2: TECNICHE DI MODELLAZIONE
*Corso del Prof. Federico Malucelli - Politecnico di Milano*

## 📋 INFORMAZIONI TECNICHE
- **Video:** Federico Malucelli's Personal Room-20250917 1007-1 (1)
- **Data elaborazione:** 24 settembre 2025, ore 12:40:53
- **Durata totale:** ~98 minuti
- **Segmenti audio trascritti:** 586
- **Frame chiave estratti:** 50
- **Sistema:** Smart Lecture Analyzer GPU v2.0

---

## 🗂️ INDICE DELLA LEZIONE

### PARTE I: INTRODUZIONE ALLA MODELLAZIONE
- **[01:00 - 08:00]** Tecniche di modellazione come arte
- **[01:50 - 02:40]** Analogia dei ritratti - scopo del modello
- **[02:40 - 05:00]** Sfide della modellazione matematica
- **[05:00 - 08:00]** Raccomandazioni per ingegneri informatici

### PARTE II: COMPONENTI DEL MODELLO
- **[05:45 - 06:10]** I tre ingredienti: decisioni, regole, obiettivi
- **[06:10 - 07:00]** Traduzione in variabili, vincoli, funzione obiettivo

### PARTE III: TIPI DI VARIABILI
- **[07:00 - 08:25]** Variabili non negative (revisione)
- **[08:25 - 18:40]** Variabili senza restrizioni di segno
- **[11:30 - 18:40]** Esempio: riprogrammazione orari treni

### PARTE IV: TRASFORMAZIONI DI EQUIVALENZA
- **[15:00 - 37:00]** Dal problema illimitato al problema con variabili non negative
- **[18:40 - 25:00]** Differenza di variabili non negative
- **[25:00 - 37:00]** Esempi pratici e applicazioni

### PARTE V: ALTRI TIPI DI VARIABILI E VINCOLI
- **[37:00 - 55:00]** Variabili binarie e discrete
- **[55:00 - 75:00]** Vincoli di uguaglianza vs disuguaglianza
- **[75:00 - 95:00]** Problemi con obiettivi multipli

---

---

## CONTENUTO DELLA LEZIONE

### 🎨 PARTE I: LA MODELLAZIONE COME ARTE {#modellazione-arte}
*Timing: 01:00 - 08:00*

#### Introduzione alle Tecniche di Modellazione
**[01:00 - 01:35]**
> "Oggi iniziamo con le tecniche di modellazione. Come vi ho detto ieri, realizzare un modello, realizzare un modello matematico, è una sorta di arte."

**Requisiti per una buona modellazione:**
- Esperienza pratica
- Fantasia e creatività
- Intuizione
- Capacità di adattamento al problema specifico

#### L'Analogia dei Ritratti
**[01:35 - 02:40]**
Il professore utilizza l'analogia di due ritratti diversi per spiegare che non esiste "il modello migliore" in assoluto:

> "Quando vi mostro un ritratto, per esempio questi due, cosa potete dire? Qual è il migliore?"

**La risposta:**
> "Dipende, esattamente. Dipende da quale è lo scopo del modello. Se volete divertirvi, questo è il migliore. Se volete descrivere una situazione critica durante una sessione in tribunale, l'altro è il migliore."

**Principio fondamentale:** Anche per i modelli matematici, tutto dipende da cosa dovete fare.

#### Livelli di Dettaglio del Modello
**[02:40 - 03:20]**
> "Potete fornire un modello molto dettagliato se avete bisogno di dettagli, o un modello molto grezzo se dovete risolvere il problema in modo veloce."

**Concetto chiave:** 
> "Tenete presente che quando modellate, state in ogni caso traducendo qualcosa di reale in qualcosa di artificiale - il modello matematico è una rappresentazione artificiale della realtà."

#### Le Sfide del Modulo
**[03:20 - 04:30]**
Il professore elenca le principali sfide che affronteranemo:

1. **Rappresentazione dei problemi di ottimizzazione**
   - Come possiamo rappresentare un problema di ottimizzazione?

2. **Tecniche per rappresentare le decisioni**
   - Quali sono i trucchi per rappresentare le decisioni?

3. **Classificazione dei vincoli**
   - Quali sono i diversi tipi di vincoli?
   - Come possiamo classificare i vincoli in base al tipo?

4. **Trasformazioni equivalenti**
   - Come possiamo trasformare un problema in uno equivalente?
   - Useremo queste trasformazioni in diverse situazioni.

5. **Obiettivi multipli**
   - Come affrontiamo problemi con funzioni obiettivo multiple?

6. **Problemi senza ottimizzazione**
   - Che dire dei problemi senza funzione obiettivo?
   - Come gestiamo problemi dove dobbiamo solo trovare una soluzione?

#### Raccomandazione per Ingegneri Informatici
**[05:00 - 05:45]**
> "Una raccomandazione, perché so che, specialmente per voi ingegneri informatici, quando descrivo un problema, iniziate già a pensare a come risolverlo. Quando facciamo un modello, non pensiamo alla soluzione. Pensiamo solo alla modellazione."

**Importante:**
> "Sarà difficile almeno all'inizio, ma dovete astenervi dal pensare alla soluzione. Dovete solo concentrarvi sulla rappresentazione matematica."

---

### 🧩 PARTE II: I TRE INGREDIENTI DEL MODELLO {#tre-ingredienti}
*Timing: 05:45 - 07:00*

#### Ripasso dalla Lezione Precedente
**[05:45 - 06:10]**
> "Ricordate quello che abbiamo visto ieri. Quando fate un modello, abbiamo bisogno di tre ingredienti:"

1. **DECISIONI** (Decisions)
2. **REGOLE** (Rules) 
3. **OBIETTIVI** (Objectives)

#### Traduzione Matematica
**[06:10 - 07:00]**
> "Questi tre ingredienti sono sufficienti per descrivere il nostro modello. Quando iniziate a guardare un problema, dovete capire chiaramente quali sono le decisioni, cosa influenza la vostra soluzione, le regole - cioè come le decisioni sono correlate o vincolate in certi limiti - e poi l'obiettivo."

**Corrispondenza matematica:**
- **Decisioni** → **Variabili**
- **Regole** → **Vincoli** (uguaglianze o disuguaglianze)  
- **Obiettivi** → **Una funzione** (obiettivo)

---

### 📊 PARTE III: TIPI DI VARIABILI - APPROFONDIMENTO {#tipi-variabili}
*Timing: 07:00 - 18:40*

#### Collegamento con Conoscenze Precedenti
**[07:00 - 07:25]**
> "Cosa intendiamo con variabili? Vi ho detto che le variabili sono decisioni. Nella vostra esperienza, una variabile è qualcosa di sconosciuto. Se avete un'equazione o qualcosa o un'espressione, scrivete X e X è sconosciuto. Lo stesso per noi. X è una decisione e il modello deve prendere la decisione sul valore di quella variabile."

#### 1. Variabili Non Negative (Ripasso)
**[07:25 - 08:25]**
> "Ci sono diversi tipi di variabili. Il più naturale è il primo che abbiamo visto ieri, cioè le variabili a valore non negativo. Il numero di cellulari che dobbiamo costruire. Facile. Oggi vedremo gli altri."

**Caratteristiche:**
- I valori negativi non sono ammessi nel problema
- Esempio: numero di prodotti da costruire
- Già visto nella lezione precedente

#### 2. Variabili Senza Restrizioni di Segno
**[08:25 - 09:40]**
> "Che dire delle variabili che non hanno restrizioni nei valori? Sono variabili che possono assumere anche valori negativi e la negatività del valore conta. Non possiamo ignorarla."

**Esempi forniti dal professore:**
- **Velocità:** se vi riferite a un movimento che può essere avanti o indietro su una linea
- **Temperature:** ovviamente quando avete valori negativi come le temperature in gradi Celsius
- **Alternative:** potreste scalare tutto in gradi Kelvin in modo che non sia negativo, ma forse il vostro modello diventa un po' più complicato in termini di interpretazione

**Conclusione:** 
> "Forse è più facile se lasciate valori negativi così come valori positivi."

---

## 🚂 ESEMPIO PRATICO: RIPROGRAMMAZIONE ORARI TRENI {#esempio-treni}
*Timing: 09:40 - 18:40*

#### Introduzione dell'Applicazione
**[09:40 - 10:20]**
> "Un'altra possibile applicazione è quando introduciamo perturbazioni. Per esempio, quando ogni anno riprogrammano i treni, rilasciano un nuovo orario dei treni, di solito in autunno. Non partono da zero."

**Il Processo:**
> "Prendono la vecchia soluzione e perturbano il tempo di viaggio e l'orario di arrivo alle stazioni."

#### Setup del Problema  
**[11:30 - 12:30]**
Il professore disegna alla lavagna:
- **M stazioni** lungo una linea ferroviaria
- **Orario attuale:** A₁, A₂, A₃, ..., Aₘ (tempi di arrivo alle stazioni)
- **Obiettivo:** Fornire un nuovo orario con nuovi tempi di arrivo A'₁, A'₂, ..., A'ₘ

**Vincoli da considerare:**
- Connessioni tra treni
- Domanda dei passeggeri  
- Tempi di percorrenza fisici
- Altri fattori operativi

#### Modellazione con Variabili di Perturbazione
**[12:30 - 14:30]**
> "Come vi ho detto, consideriamo di solito una perturbazione del vecchio orario di arrivo."

**Formula matematica:**
```
A'₁ = A₁ + π₁
A'₂ = A₂ + π₂
...
A'ₛ = Aₛ + πₛ
```

Dove **πₛ** è la variabile di perturbazione per la stazione s.

#### Interpretazione delle Variabili di Perturbazione
**[14:30 - 15:00]**
> "Possiamo interpretare questa variabile π, πₛ:"

- **Se π < 0:** "Siamo in anticipo. Anticipiamo rispetto al vecchio orario."
- **Se π = 0:** "Ovviamente nessun cambiamento."  
- **Se π > 0:** "È un ritardo."

---

## ⚙️ PARTE IV: TRASFORMAZIONE DA VARIABILI ILLIMITATE A NON NEGATIVE {#trasformazione-variabili}
*Timing: 15:00 - 25:00*

### Il Problema della "Black Box"
**[15:00 - 16:00]**
Il professore introduce uno scenario molto importante:

> "Supponiamo di avere una scatola nera che risolve problemi di ottimizzazione. I problemi di ottimizzazione che siamo in grado di risolvere sono di questo tipo:"

**Forma standard del solver:**
```
Massimizzare: c^T x
Soggetto a: Ax ≤ b
Con: x ≥ 0 (variabili non negative)
```

> "Questa è la nostra scatola nera, la nostra macchina che risolve problemi di ottimizzazione. La macchina prende in input i parametri c, A, b e fornisce in output la soluzione ottimale x*."

### Il Problema Centrale
**[16:00 - 17:30]**
> "Come possiamo risolvere il problema dei treni con questo solver? In altre parole, come possiamo trasformare un problema dove abbiamo variabili illimitate che non sono ristrette nel segno in un problema equivalente dove tutte le variabili sono non negative?"

**Domanda chiave:**
> "Come posso tradurre π, che può essere negativo, positivo o zero, in una nuova π dove π è non negativo?"

**Perché è importante:**
> "Non posso semplicemente ignorare il vincolo e dire, ok, risolviamo il problema assumendo che le nostre π siano solo non negative. Perché altrimenti, trascureremmo tutte le soluzioni dove anticipiamo l'arrivo del treno. E questo non è giusto. Stiamo dimenticando circa metà o più della regione ammissibile."

### La Soluzione: Differenza di Variabili Non Negative
**[17:30 - 20:00]**

#### Domanda agli Studenti
> "Come potete ottenere un valore che può essere positivo, negativo o zero usando solo valori non negativi?"

#### La Risposta 
**Approccio suggerito:**
> "La soluzione che suggerisco è più facile. Si basa su aritmetica molto semplice, delle scuole elementari."

**Principio fondamentale:**
Qualsiasi numero reale può essere scritto come differenza di due numeri non negativi.

**Formula matematica:**
```
π_s = π_s^+ - π_s^-
```
dove:
- **π_s^+ ≥ 0** (parte positiva)
- **π_s^- ≥ 0** (parte negativa)

#### Interpretazione Pratica
**[20:00 - 22:00]**

**Casi possibili:**

1. **Se π_s > 0** (ritardo):
   - π_s^+ > 0 (rappresenta il ritardo)
   - π_s^- = 0 (nessuna parte negativa)

2. **Se π_s < 0** (anticipo):
   - π_s^+ = 0 (nessuna parte positiva)  
   - π_s^- > 0 (rappresenta l'anticipo)

3. **Se π_s = 0** (nessun cambiamento):
   - π_s^+ = 0
   - π_s^- = 0

### Trasformazione Completa del Problema
**[22:00 - 25:00]**

#### Sostituzione nelle Equazioni
> "Trasformiamo il nostro problema sostituendo al posto di ogni variabile π_s, la coppia π_s^+ - π_s^-"

**Esempio per l'orario dei treni:**
```
Originale: A'_s = A_s + π_s
Trasformato: A'_s = A_s + (π_s^+ - π_s^-)
```

#### Vincoli Aggiuntivi
Nel nuovo problema avremo:
- Tutte le variabili originali π_s^+ ≥ 0 e π_s^- ≥ 0  
- Il doppio delle variabili (ogni π_s diventa π_s^+ e π_s^-)

#### Nota del Professore
**[24:30 - 25:00]**
> "Questo è solo un trucco per trasformare un problema con variabili illimitate in uno con variabili non negative per poter usare questo solver."

> "Per i solver moderni, è equivalente, davvero equivalente. Fanno la trasformazione automaticamente."

**Importante:** Questa trasformazione è principalmente didattica - i software moderni gestiscono automaticamente variabili illimitate.

---

## 📐 PARTE V: ALTRI TIPI DI VARIABILI E VINCOLI {#altri-tipi}
*Timing: 25:00 - 45:00*

### Variabili Binarie
**[25:00 - 30:00]**
> "Un altro tipo molto importante di variabile è la variabile binaria, che può assumere solo i valori 0 o 1."

**Applicazioni tipiche:**
- **Decisioni on/off:** costruire o non costruire una fabbrica
- **Scelte logiche:** scegliere un percorso tra alternative  
- **Scheduling:** assegnare o non assegnare un compito

**Esempio pratico:**
```
x_i = {
  1 se costruiamo la fabbrica nella città i
  0 altrimenti
}
```

### Variabili Integer (Intere)
**[30:00 - 35:00]**
> "Abbiamo anche variabili che devono assumere valori interi, ma non necessariamente 0 o 1."

**Esempi:**
- Numero di autobus da acquistare
- Numero di turni di lavoro
- Numero di containers da spedire

**Differenza importante:**
- **Variabili continue:** possono assumere qualsiasi valore reale
- **Variabili intere:** solo valori interi (1, 2, 3, ...)
- **Variabili binarie:** caso speciale di intere (solo 0 o 1)

### Vincoli di Uguaglianza vs Disuguaglianza
**[35:00 - 40:00]**

#### Vincoli di Disuguaglianza (≤, ≥)
> "I vincoli di disuguaglianza rappresentano limitazioni che non devono essere necessariamente sature."

**Esempi:**
- Budget disponibile: `costo_totale ≤ budget_max`
- Capacità produttiva: `produzione ≤ capacità_max`

#### Vincoli di Uguaglianza (=)  
> "I vincoli di uguaglianza devono essere soddisfatti esattamente."

**Esempi:**
- Conservazione del flusso: `flusso_entrante = flusso_uscente`
- Bilancio di massa: `input = output + accumulo`

#### Trasformazione Equivalente
**[40:00 - 42:00]**
> "Possiamo sempre trasformare un vincolo di uguaglianza in due vincoli di disuguaglianza:"

```
Originale: ax = b
Equivalente: {ax ≤ b
             {ax ≥ b
```

### Problemi con Obiettivi Multipli
**[42:00 - 45:00]**

#### Il Problema
> "Come affrontiamo problemi con funzioni obiettivo multiple? Per esempio, vogliamo massimizzare il profitto E minimizzare l'impatto ambientale."

#### Approcci Possibili:
1. **Combinazione pesata:** 
   ```
   max α·profitto - β·impatto_ambientale
   ```

2. **Ottimizzazione lessicografica:** Prima ottimizzi un obiettivo, poi l'altro

3. **Vincoli aggiuntivi:** Trasforma obiettivi secondari in vincoli
   ```
   max profitto
   s.t. impatto_ambientale ≤ soglia_accettabile
   ```

---

## 🔍 PARTE VI: PROBLEMI SENZA OTTIMIZZAZIONE {#problemi-senza-ottimizzazione}
*Timing: 45:00 - 55:00*

### Problemi di Fattibilità  
**[45:00 - 48:00]**
> "Che dire dei problemi senza funzione obiettivo? Come gestiamo un problema di ottimizzazione dove non dobbiamo ottimizzare, dobbiamo solo trovare una soluzione?"

#### Definizione
**Problema di fattibilità:** Trovare una soluzione che soddisfi tutti i vincoli, senza ottimizzare alcun obiettivo.

**Forma matematica:**
```
Trovare: x
Soggetto a: Ax ≤ b
            x ≥ 0
```
(Nessuna funzione obiettivo)

#### Esempi Pratici
1. **Scheduling:** "Esiste un modo per assegnare tutti i compiti rispettando le scadenze?"
2. **Logistica:** "È possibile trasportare tutti i prodotti con i mezzi disponibili?"  
3. **Progettazione:** "Si può progettare un sistema che soddisfi tutti i requisiti tecnici?"

### Trasformazione in Problema di Ottimizzazione
**[48:00 - 52:00]**

#### Approccio 1: Obiettivo Artificiale
```
min 0  (minimizza zero)
s.t. Ax ≤ b
     x ≥ 0
```

#### Approccio 2: Minimizzazione Violazioni
```
min Σ(violazioni_i)
s.t. vincoli_rilassati
```

### Interpretazione dei Risultati
**[52:00 - 55:00]**

#### Se il problema ha soluzione:
- **Fattibile:** Esiste almeno una soluzione che soddisfa tutti i vincoli
- Il solver fornisce una soluzione valida

#### Se il problema non ha soluzione:
- **Infeasible:** Nessuna soluzione può soddisfare simultaneamente tutti i vincoli  
- I vincoli sono contraddittori tra loro

> "Ricordate: un problema può essere infeasible anche se ogni singolo vincolo è ragionevole. È la loro combinazione che crea il conflitto."

---

## 💡 CONCETTI CHIAVE E TAKEAWAY {#concetti-chiave}

### 🎯 Principi Fondamentali

1. **La modellazione è un'arte**
   - Richiede esperienza, fantasia e intuizione
   - Non esiste "il modello migliore" in assoluto
   - Dipende sempre dallo scopo e dal contesto

2. **I tre ingredienti essenziali**
   - **Decisioni** → Variabili  
   - **Regole** → Vincoli
   - **Obiettivi** → Funzione obiettivo

3. **Focus sulla modellazione**
   - Non pensare alla soluzione durante la modellazione
   - Concentrarsi sulla rappresentazione matematica
   - La risoluzione viene dopo

### 🔧 Tecniche di Trasformazione

#### Variabili Illimitate → Non Negative
```
π = π⁺ - π⁻
dove π⁺ ≥ 0, π⁻ ≥ 0
```

#### Vincoli di Uguaglianza → Disuguaglianza  
```
ax = b  ⟺  {ax ≤ b
            {ax ≥ b
```

#### Obiettivi Multipli → Singolo Obiettivo
- Combinazione pesata
- Ottimizzazione lessicografica  
- Trasformazione in vincoli

### ⚠️ Punti di Attenzione

1. **Quando usare variabili illimitate:**
   - Temperature (Celsius)
   - Perturbazioni e cambiamenti
   - Velocità con direzione
   - Qualsiasi grandezza che possa essere negativa

2. **Trasformazioni equivalenti:**
   - Sono matematicamente corrette
   - I solver moderni le fanno automaticamente
   - Utili per comprendere i concetti

3. **Problemi di fattibilità:**
   - A volte più importanti dell'ottimalità
   - Primo passo: esiste una soluzione?
   - Secondo passo: quale è la migliore?

---

## 📚 PREPARAZIONE PROSSIMA LEZIONE

### Argomenti da Rivedere
1. I tre tipi di variabili visti oggi
2. La trasformazione π = π⁺ - π⁻  
3. Differenza tra vincoli di uguaglianza e disuguaglianza
4. Concetto di problema equivalente

### Domande di Autovalutazione
1. Quando è necessario usare variabili illimitate?
2. Come si trasforma una variabile illimitata in non negative?
3. Qual è la differenza tra un problema di ottimizzazione e uno di fattibilità?
4. Perché "la modellazione è un'arte"?

### Esercizi Consigliati
- Identificare il tipo di variabili in problemi pratici
- Praticare trasformazioni di equivalenza
- Formulare problemi con obiettivi multipli

---

## APPENDICE: TRASCRIZIONE AUTOMATICA COMPLETA {#appendice-trascrizione}

*La seguente è la trascrizione automatica integrale generata dal sistema Smart Lecture Analyzer GPU, conservata come riferimento tecnico per completezza e verifica.*