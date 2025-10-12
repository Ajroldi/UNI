# 📘 Lezione 7 - Alloy: Operazioni e Asserzioni (Parte 1)

**Corso:** Ingegneria del Software  
**Data:** 8 Ottobre  
**Argomento:** Alloy - Operazioni Add/Delete, Asserzioni e Verifica  
**Durata Parte 1:** 00:00:21 - 00:40:00 (~40 minuti)

---

## 📑 Indice dei Contenuti - Parte 1

### [1. Review: Address Book e Join Operator](#1-review-address-book-e-join-operator) `00:00:21 - 00:19:30`
   - [1.1 Recap delle Signature](#11-recap-delle-signature) `00:00:21 - 00:02:28`
   - [1.2 Vincoli di Cardinalità](#12-vincoli-di-cardinalità) `00:02:28 - 00:04:00`
   - [1.3 Predicati con Constraint](#13-predicati-con-constraint) `00:04:00 - 00:06:35`
   - [1.4 Join Operator: Esempi Dettagliati](#14-join-operator-esempi-dettagliati) `00:06:35 - 00:11:09`
   - [1.5 Quantificazione con "some"](#15-quantificazione-con-some) `00:11:09 - 00:16:27`
   - [1.6 Rilevamento di Inconsistenze](#16-rilevamento-di-inconsistenze) `00:16:27 - 00:19:30`

### [2. Visualizzazione con Show Predicate](#2-visualizzazione-con-show-predicate) `00:19:30 - 00:29:15`
   - [2.1 Esecuzione del Comando Run](#21-esecuzione-del-comando-run) `00:19:30 - 00:22:00`
   - [2.2 Mondi Generati dall'Analyzer](#22-mondi-generati-dallanalyzer) `00:22:00 - 00:29:15`

### [3. Operazione Add: Aggiungere Elementi](#3-operazione-add-aggiungere-elementi) `00:29:15 - 00:37:12`
   - [3.1 Corner Case: Stesso Libro Prima e Dopo](#31-corner-case-stesso-libro-prima-e-dopo) `00:29:15 - 00:30:23`
   - [3.2 Perché B = B' Soddisfa il Predicato](#32-perché-b--b-soddisfa-il-predicato) `00:30:23 - 00:32:24`
   - [3.3 Decisioni di Design](#33-decisioni-di-design) `00:32:24 - 00:34:04`
   - [3.4 Altri Mondi Validi](#34-altri-mondi-validi) `00:34:04 - 00:36:10`
   - [3.5 Operatore Prodotto Cartesiano (->)](#35-operatore-prodotto-cartesiano---) `00:36:10 - 00:37:12`

### [4. Operazione Delete: Rimuovere Elementi](#4-operazione-delete-rimuovere-elementi) `00:37:12 - 00:40:34`
   - [4.1 Specifica dell'Operazione Delete](#41-specifica-delloperazione-delete) `00:37:12 - 00:38:52`
   - [4.2 Esecuzione e Corner Cases](#42-esecuzione-e-corner-cases) `00:38:52 - 00:40:34`

---

## 1. Review: Address Book e Join Operator

### 1.1 Recap delle Signature

`⏱️ 00:00:21 - 00:02:28`

Iniziamo rivedendo l'esempio dell'**Address Book** introdotto nella lezione precedente. Ricordiamo le signature fondamentali:

```alloy
sig Name {}
sig Addr {}
sig Book {
  addr: Name -> lone Addr
}
```

💡 **Insight - Struttura dell'Address Book**

La specifica definisce tre tipi fondamentali:
- **`Name`**: Rappresenta i nomi (atomi senza struttura interna)
- **`Addr`**: Rappresenta gli indirizzi (atomi senza struttura interna)
- **`Book`**: Rappresenta un libro contenente la relazione `addr`

📝 **Nota sulla Relazione Ternaria**

La dichiarazione `addr: Name -> lone Addr` all'interno di `Book` crea una **relazione ternaria**:
- Si legge: "in un Book, ogni Name è associato a **al più un** Addr"
- Il vincolo `lone` (lone = 0 o 1) garantisce che ogni nome abbia **0 oppure 1 indirizzo**, mai più di uno

🔍 **Osservazione - Atomi e Istanze**

Quando eseguiamo un predicato, l'Alloy Analyzer genera **mondi** (worlds) contenenti:
- Atomi concreti come `N0, N1, N2` (istanze di Name)
- Atomi concreti come `A0, A1, A2` (istanze di Addr)
- Istanze di Book con coppie concrete nella relazione `addr`

**Esempio di mondo concreto:**
```
Book0:
  addr = { (N0, A1), (N1, A0) }
```

Questo significa:
- Nel `Book0`, il nome `N0` è associato all'indirizzo `A1`
- Il nome `N1` è associato all'indirizzo `A0`
- Altri nomi (se esistono) non hanno indirizzi associati

---

### 1.2 Vincoli di Cardinalità

`⏱️ 00:02:28 - 00:04:00`

Il vincolo `lone` nella specifica ha un effetto importante sui mondi possibili generati dall'Analyzer.

📊 **Tabella: Molteplicità in Alloy**

| Keyword | Cardinalità | Significato | Esempio |
|---------|-------------|-------------|---------|
| `lone` | 0..1 | Al più uno | Un nome ha 0 o 1 indirizzo |
| `one` | 1..1 | Esattamente uno | Ogni persona ha esattamente 1 età |
| `some` | 1..* | Almeno uno | Un corso ha almeno 1 studente |
| `set` | 0..* | Qualsiasi numero | Un autore può scrivere 0+ libri |

⚠️ **Attenzione - Predicati Sempre Veri**

Con il vincolo `lone` nella specifica, alcuni predicati risultano **sempre veri** (tautologie):
- Non possiamo mai avere un nome associato a 2+ indirizzi
- Predicati che richiedono questa situazione sono **insoddisfacibili**

**Esempio di predicato sempre vero:**
```alloy
pred nessunNomeConDueIndirizzi[b: Book] {
  all n: Name | #(n.(b.addr)) <= 1
}
```

Questo predicato è **ridondante** perché il vincolo `lone` lo garantisce già!

---

### 1.3 Predicati con Constraint

`⏱️ 00:04:00 - 00:06:35`

Possiamo creare predicati che **aggiungono ulteriori vincoli** oltre a quelli della specifica base.

📝 **Esempio 1: Libro con Almeno 2 Elementi**

```alloy
pred mostraDueElementi[b: Book] {
  #b.addr > 1
}
```

🔍 **Analisi del Predicato**

- **`b.addr`**: Operatore **join** (.) che estrae l'insieme di coppie `(Name, Addr)` dal libro `b`
- **`#`**: Operatore di **cardinalità** che conta gli elementi dell'insieme
- **`> 1`**: Richiede almeno 2 coppie nell'insieme

**Cosa otteniamo eseguendo `run mostraDueElementi`?**

L'Analyzer genera mondi dove il libro contiene **almeno 2 associazioni nome→indirizzo**.

**Esempio di mondo soddisfacente:**
```
Book0:
  addr = { (N0, A0), (N1, A1), (N2, A2) }
```

Qui `#Book0.addr = 3 > 1` ✅

---

### 1.4 Join Operator: Esempi Dettagliati

`⏱️ 00:06:35 - 00:11:09`

L'operatore **join** (`.`) è fondamentale in Alloy. Vediamo esempi passo-passo per comprenderne il funzionamento.

💡 **Insight - Operatore Join (.)**

Il join è un'operazione che **compone relazioni** eliminando l'elemento intermedio:
- Data una relazione `R: A -> B` e una relazione `S: B -> C`
- Il join `R.S` produce una relazione `A -> C`
- Per ogni coppia `(a, b)` in R e `(b, c)` in S, otteniamo `(a, c)` in R.S

📊 **Esempio Passo-Passo: Join Semplice**

Supponiamo di avere:

```alloy
sig Name {}
sig Addr {}
sig Book {
  addr: Name -> lone Addr
}

pred esempio[b: Book] {
  #(name.(b.addr)) > 1
}
```

**Analisi dettagliata:**

1️⃣ **`b.addr`** produce un insieme di coppie:
```
b.addr = { (N0, A1), (N1, A0), (N2, A1) }
```

2️⃣ Supponiamo che `name` sia un atomo specifico, ad esempio `N0`

3️⃣ **`N0.(b.addr)`** = join tra il singleton `{N0}` e l'insieme di coppie `b.addr`

**Passo intermedio:**
- Cerchiamo tutte le coppie in `b.addr` che iniziano con `N0`
- Troviamo: `(N0, A1)`
- Il risultato del join è l'insieme dei secondi elementi: `{A1}`

4️⃣ **`#(N0.(b.addr))`** = cardinalità dell'insieme `{A1}` = **1**

5️⃣ La condizione `> 1` richiede che il risultato contenga **più di 1 elemento**

⚠️ **Attenzione - Vincolo Lone**

Con il vincolo `lone` nella specifica, **ogni nome può avere al massimo 1 indirizzo**, quindi:
- `#(n.(b.addr))` può essere solo **0 o 1**
- La condizione `> 1` non può mai essere soddisfatta con il vincolo `lone`!

📝 **Esempio più complesso: Join Multiplo**

```alloy
pred esempioComplesso[b: Book] {
  some n: Name | #(n.(b.addr)) > 1
}
```

Questo predicato dice:
> "Esiste **almeno un** nome `n` tale che `n` è associato a **più di 1** indirizzo nel libro `b`"

**Sequenza di valutazione:**

1. L'Analyzer prova tutti i possibili atomi `n` di tipo `Name`
2. Per ciascun `n`, calcola `n.(b.addr)` (gli indirizzi associati a `n`)
3. Conta la cardinalità del risultato
4. Verifica se esiste almeno un `n` per cui `#(n.(b.addr)) > 1`

**Risultato con vincolo `lone`:**
- Il predicato è **insoddisfacibile** (inconsistente con la specifica)
- L'Analyzer restituisce "No instance found" ❌

---

### 1.5 Quantificazione con "some"

`⏱️ 00:11:09 - 00:16:27`

La quantificazione **esistenziale** con `some` permette di esprimere proprietà come "esiste almeno un elemento che..."

📝 **Sintassi della Quantificazione**

```alloy
some variabile: Tipo | formula
```

Significa:
> "Esiste **almeno un** elemento del tipo `Tipo` tale che `formula` è vera"

🔍 **Esempio: Nome con Più Indirizzi**

```alloy
pred unNomeConPiuIndirizzi[b: Book] {
  some n: Name | #(n.(b.addr)) > 1
}
```

**Domanda chiave:** Questo predicato è **soddisfacibile** data la nostra specifica con vincolo `lone`?

**Analisi passo-passo:**

1️⃣ Il predicato richiede l'esistenza di un nome `n` con più di 1 indirizzo

2️⃣ La specifica dice `addr: Name -> lone Addr`
   - `lone` = al più 1 indirizzo per nome

3️⃣ **Contraddizione logica!**
   - Il predicato richiede `#(n.(b.addr)) > 1`
   - La specifica garantisce `#(n.(b.addr)) <= 1`
   - Non esiste un mondo che soddisfi entrambi i vincoli

⚠️ **Attenzione - Inconsistenza**

Quando un predicato è **inconsistente** con la specifica:
- L'Alloy Analyzer non trova alcuna istanza
- Messaggio: "No instance found"
- Questo è un segnale che:
  - Il predicato è troppo restrittivo, OPPURE
  - La specifica impedisce il comportamento desiderato

💡 **Insight - Ragionamento sulla Soddisfacibilità**

Prima di eseguire un predicato, possiamo **ragionare logicamente**:

**Domanda:** È possibile avere un nome con 2+ indirizzi?

**Risposta:**
- Il vincolo `lone` nella dichiarazione `addr: Name -> lone Addr` dice esplicitamente "al più uno"
- Quindi **NO**, è impossibile
- Il predicato `some n: Name | #(n.(b.addr)) > 1` è **insoddisfacibile**

✅ **Regola Pratica - Verifica Mentale**

Prima di scrivere un predicato, chiediti:
1. Quali vincoli impone la specifica base?
2. Il mio predicato è compatibile con questi vincoli?
3. Sto richiedendo qualcosa di impossibile?

---

### 1.6 Rilevamento di Inconsistenze

`⏱️ 00:16:27 - 00:19:30`

Uno degli usi più potenti dell'Alloy Analyzer è **rilevare inconsistenze** nelle specifiche.

🎯 **Obiettivo - Rilevamento Automatico**

L'Analyzer può:
1. Rilevare quando un predicato è **incompatibile** con la specifica
2. Mostrare "No instance found" quando non esistono mondi validi
3. Aiutarci a capire se la specifica è troppo restrittiva o incorretta

📊 **Esempio: Predicato Inconsistente**

```alloy
sig Name {}
sig Addr {}
sig Book {
  addr: Name -> lone Addr  // Vincolo: al più 1 indirizzo per nome
}

pred trovaNomeConDueIndirizzi[b: Book] {
  some n: Name | #(n.(b.addr)) > 1  // Richiede: almeno 2 indirizzi
}
```

**Esecuzione:**
```alloy
run trovaNomeConDueIndirizzi
```

**Risultato:**
```
No instance found.
Predicate may be inconsistent.
```

🔍 **Analisi dell'Inconsistenza**

**Perché non trova istanze?**

1. La specifica impone: `addr: Name -> lone Addr`
   - Interpretazione: per ogni Book, la relazione `addr` associa ogni Name a **al più 1** Addr
   
2. Il predicato richiede: `#(n.(b.addr)) > 1`
   - Interpretazione: esiste un Name associato a **più di 1** Addr

3. **Conflitto logico:**
   - Vincolo specifica: `#(n.(b.addr)) ≤ 1` per ogni `n`
   - Vincolo predicato: `∃n: #(n.(b.addr)) > 1`
   - Questi due vincoli sono **mutuamente esclusivi**

💡 **Insight - Utilità Pratica**

Il rilevamento di inconsistenze è utile per:
- **Validare specifiche**: Scoprire vincoli in conflitto
- **Debugging**: Capire perché certi comportamenti non sono possibili
- **Esplorazione**: Verificare se le nostre assunzioni sono corrette

📝 **Nota - Show Predicate**

Per esplorare la specifica, possiamo usare un predicato `show` generico:

```alloy
pred show[b: Book] {}
```

Questo predicato **non aggiunge vincoli**, quindi l'Analyzer genera mondi che soddisfano **solo la specifica base**.

**Esecuzione:**
```alloy
run show
```

**Risultato:** Mondi generati rispettando solo il vincolo `lone` della specifica.

---

## 2. Visualizzazione con Show Predicate

### 2.1 Esecuzione del Comando Run

`⏱️ 00:19:30 - 00:22:00`

Il comando `run` è il meccanismo principale per **esplorare** le specifiche Alloy.

📝 **Sintassi Base del Comando Run**

```alloy
run nomePredicato
```

Opzionalmente, possiamo specificare **limiti** (scope):
```alloy
run nomePredicato for 3
run nomePredicato for 5
run nomePredicato for 3 but 4 Book
```

💡 **Insight - Scope e Bounded Analysis**

- **Scope**: Il numero massimo di atomi per ogni tipo
- `for 3`: Massimo 3 atomi per ogni signature
- `for 3 but 4 Book`: Massimo 3 atomi per ogni tipo, ma 4 per Book

⚠️ **Attenzione - Performance**

Scope più grandi = tempo di esecuzione maggiore:
- `for 3`: Molto veloce (millisecondi)
- `for 5`: Veloce (secondi)
- `for 10`: Lento (minuti per specifiche complesse)
- `for 100`: Molto lento o impraticabile

✅ **Regola Pratica - Scegliere lo Scope**

1. **Esplorazione iniziale:** `for 3` (veloce, mostra casi piccoli)
2. **Testing:** `for 5` (buon compromesso)
3. **Verifica approfondita:** `for 8-10` (solo se necessario)

📊 **Esempio: Show Predicate**

```alloy
pred show[b: Book] {
  // Nessun vincolo aggiuntivo
}

run show
```

**Cosa succede:**
1. L'Analyzer genera mondi che soddisfano la specifica base
2. Mostra il primo mondo trovato
3. Possiamo cliccare "Next" per vedere altri mondi

**Mondi possibili:**
- Libro vuoto: `addr = {}`
- Libro con 1 elemento: `addr = {(N0, A0)}`
- Libro con 2 elementi: `addr = {(N0, A0), (N1, A1)}`
- ...

---

### 2.2 Mondi Generati dall'Analyzer

`⏱️ 00:22:00 - 00:29:15`

L'Alloy Analyzer genera diversi **mondi** (istanze) che soddisfano i vincoli specificati.

🔍 **Osservazione - Ordine di Presentazione**

L'Analyzer mostra tipicamente:
1. **Mondi più semplici** per primi (meno atomi)
2. **Corner cases** (casi limite)
3. **Mondi più complessi** successivamente

💡 **Insight - Corner Cases**

I **corner cases** sono situazioni al limite che spesso rivelano problemi nella specifica:
- Insiemi vuoti
- Elementi isolati
- Relazioni cicliche
- Situazioni "degeneri"

📝 **Esempio: Sequenza di Mondi**

Eseguendo `run show for 3`, potremmo vedere:

**Mondo 1 - Vuoto:**
```
Book0:
  addr = {}
```
Nessuna associazione nome→indirizzo.

**Mondo 2 - Un Elemento:**
```
Book0:
  addr = {(N0, A0)}
```
Un solo nome con un indirizzo.

**Mondo 3 - Due Elementi:**
```
Book0:
  addr = {(N0, A0), (N1, A1)}
```
Due nomi, ciascuno con il proprio indirizzo.

**Mondo 4 - Corner Case:**
```
Book0:
  addr = {(N0, A0), (N1, A0)}
```
Due nomi che condividono **lo stesso indirizzo** (permesso dalla specifica!).

⚠️ **Attenzione - Interpretazione dei Mondi**

Ogni mondo mostrato è **valido** secondo la specifica. Se vediamo mondi "strani":
- Non è un errore dell'Analyzer
- La specifica **permette** quella situazione
- Potremmo dover **aggiungere vincoli** se vogliamo escluderla

✅ **Regola Pratica - Analisi dei Mondi**

Quando esploriamo con `show`:
1. Osserva i primi mondi (spesso corner cases)
2. Chiediti: "Questo mondo ha senso nel mio dominio?"
3. Se un mondo è indesiderato, aggiungi vincoli per escluderlo
4. Continua a cliccare "Next" per vedere varianti

---

## 3. Operazione Add: Aggiungere Elementi

### 3.1 Corner Case: Stesso Libro Prima e Dopo

`⏱️ 00:29:15 - 00:30:23`

Quando definiamo operazioni che modificano lo stato (come `add`), possono emergere **corner cases** interessanti.

📝 **Definizione dell'Operazione Add**

```alloy
pred add[b, b': Book, n: Name, a: Addr] {
  b'.addr = b.addr + (n -> a)
}
```

**Interpretazione:**
- `b`: Libro **prima** dell'operazione
- `b'`: Libro **dopo** l'operazione
- `n -> a`: Coppia (nome, indirizzo) da aggiungere
- `b.addr + (n -> a)`: Unione insiemistica

💡 **Insight - Unione Insiemistica**

L'operatore `+` in Alloy rappresenta l'**unione di insiemi**:
```
A + B = {x | x ∈ A ∨ x ∈ B}
```

Se `(n, a)` è già in `b.addr`, l'unione non cambia l'insieme!

🔍 **Corner Case Scoperto dall'Analyzer**

Eseguendo:
```alloy
pred showAdd[b, b': Book, n: Name, a: Addr] {
  add[b, b', n, a]
}

run showAdd
```

**Risultato inaspettato:**
```
Book0:
  addr = {(N0, A0)}

b = Book0
b' = Book0  // Stesso libro!
n = N0
a = A0
```

⚠️ **Attenzione - B e B' Identici**

L'Analyzer mostra un mondo dove:
- `b` e `b'` sono **lo stesso atomo** `Book0`
- Non c'è stato nessun cambiamento!
- Ma il predicato `add` è comunque **soddisfatto**

**Perché questo accade?** Vediamo nella prossima sezione.

---

### 3.2 Perché B = B' Soddisfa il Predicato

`⏱️ 00:30:23 - 00:32:24`

Analizziamo matematicamente perché `b = b'` può soddisfare il predicato `add`.

🔍 **Analisi Matematica**

Il predicato dice:
```alloy
b'.addr = b.addr + (n -> a)
```

Sostituendo `b' = b`:
```
b.addr = b.addr + (n -> a)
```

**Domanda:** Quando questa equazione è vera?

**Risposta:** Quando `(n -> a)` è **già presente** in `b.addr`!

💡 **Insight - Proprietà dell'Unione Insiemistica**

Per l'unione di insiemi vale:
```
S = S ∪ {x}  ⟺  x ∈ S
```

Se l'elemento è già nell'insieme, l'unione non lo modifica.

📊 **Esempio Concreto**

Supponiamo:
```
b.addr = {(N0, A0), (N1, A1)}
n = N0
a = A0
```

Calcoliamo `b'.addr`:
```
b'.addr = b.addr + (n -> a)
        = {(N0, A0), (N1, A1)} ∪ {(N0, A0)}
        = {(N0, A0), (N1, A1)}  // L'elemento era già presente!
```

Quindi `b'.addr = b.addr`, e se l'Analyzer sceglie `b' = b`, il predicato è soddisfatto! ✅

📝 **Nota - Non è un Errore**

Questo comportamento è **corretto** dal punto di vista logico:
- Il predicato non dice che `b ≠ b'`
- Sta solo specificando la relazione tra `b'.addr` e `b.addr`
- Se aggiungiamo un elemento già presente, nessun cambiamento è matematicamente corretto

---

### 3.3 Decisioni di Design

`⏱️ 00:32:24 - 00:34:04`

Dobbiamo decidere se questo comportamento (B = B') è **desiderabile** o meno nel nostro modello.

🎯 **Opzioni di Design**

**Opzione 1 - Tollerare il Caso:**
- Accettare che `add` possa non avere effetto se l'elemento esiste già
- Comportamento simile a `set.add()` in molti linguaggi (operazione idempotente)

**Opzione 2 - Forzare il Cambiamento:**
- Richiedere esplicitamente che `b ≠ b'`
- Aggiungere vincolo: `n -> a not in b.addr`

📝 **Modifica della Specifica - Opzione 2**

Se vogliamo forzare un cambiamento effettivo:

```alloy
pred addStrict[b, b': Book, n: Name, a: Addr] {
  b'.addr = b.addr + (n -> a)
  b != b'  // Forza libri diversi
}
```

Oppure, equivalentemente:

```alloy
pred addOnlyNew[b, b': Book, n: Name, a: Addr] {
  b'.addr = b.addr + (n -> a)
  n -> a not in b.addr  // L'elemento NON deve esistere già
}
```

💡 **Insight - Semantica delle Operazioni**

La scelta dipende dalla **semantica** che vogliamo:

| Semantica | Vincolo | Comportamento |
|-----------|---------|---------------|
| **Idempotente** | Nessuno | Add ripetuto = nessun effetto |
| **Precondizione** | `n -> a not in b.addr` | Add fallisce se elemento esiste |
| **Cambiamento forzato** | `b != b'` | Deve sempre modificare lo stato |

✅ **Regola Pratica - Documentare la Scelta**

Qualunque scelta facciamo, dobbiamo:
1. **Documentarla** nei commenti
2. **Testarla** con asserzioni
3. **Verificare** che corrisponda al comportamento desiderato nel dominio

---

### 3.4 Altri Mondi Validi

`⏱️ 00:34:04 - 00:36:10`

L'Analyzer mostra tipicamente prima i **corner cases**, poi casi più "normali".

🔍 **Osservazione - Strategia dell'Analyzer**

L'Alloy Analyzer:
1. Cerca mondi con **meno atomi** prima (più semplici)
2. Mostra **corner cases** che spesso rivelano problemi
3. Permette di esplorare casi più complessi cliccando "Next"

📊 **Esempio: Mondo "Normale" per Add**

Dopo il corner case `b = b'`, cliccando "Next" potremmo vedere:

```
Book0:
  addr = {(N0, A0)}

Book1:
  addr = {(N0, A0), (N1, A1)}

b = Book0
b' = Book1
n = N1
a = A1
```

**Interpretazione:**
- Libro iniziale `Book0` contiene `{(N0, A0)}`
- Aggiungiamo `(N1, A1)`
- Libro risultante `Book1` contiene `{(N0, A0), (N1, A1)}`
- Qui `b ≠ b'` ✅

💡 **Insight - Importanza dei Corner Cases**

I corner cases sono **fondamentali** per:
- **Scoprire ambiguità** nella specifica
- **Validare assunzioni** implicite
- **Migliorare** la precisione del modello

Se un corner case è **indesiderato**, dobbiamo aggiungere vincoli espliciti per escluderlo.

---

### 3.5 Operatore Prodotto Cartesiano (->)

`⏱️ 00:36:10 - 00:37:12`

L'operatore `->` (freccia) rappresenta il **prodotto cartesiano** tra insiemi.

📝 **Definizione - Prodotto Cartesiano**

Per due insiemi `A` e `B`, il prodotto cartesiano `A -> B` è:
```
A -> B = {(a, b) | a ∈ A, b ∈ B}
```

💡 **Insight - Casi Speciali**

**Caso 1: Singleton -> Singleton**
```alloy
n -> a  // dove n è un singolo Name, a è un singolo Addr
```
Risultato: Un insieme contenente **una sola coppia** `{(n, a)}`

**Caso 2: Insieme -> Singleton**
```alloy
Name -> a  // dove a è un singolo Addr
```
Risultato: Tutte le coppie `{(n0, a), (n1, a), (n2, a), ...}` per ogni Name

**Caso 3: Singleton -> Insieme**
```alloy
n -> Addr  // dove n è un singolo Name
```
Risultato: Tutte le coppie `{(n, a0), (n, a1), (n, a2), ...}` per ogni Addr

**Caso 4: Insieme -> Insieme**
```alloy
Name -> Addr
```
Risultato: **Tutte le possibili coppie** tra Name e Addr

📊 **Tabella: Esempi di Prodotto Cartesiano**

| Espressione | Input | Output |
|-------------|-------|--------|
| `N0 -> A1` | Singleton × Singleton | `{(N0, A1)}` |
| `{N0, N1} -> A0` | Set × Singleton | `{(N0, A0), (N1, A0)}` |
| `N0 -> {A0, A1}` | Singleton × Set | `{(N0, A0), (N0, A1)}` |
| `{N0, N1} -> {A0, A1}` | Set × Set | `{(N0, A0), (N0, A1), (N1, A0), (N1, A1)}` |

🔍 **Applicazione nell'Operazione Add**

Nel predicato `add`:
```alloy
b'.addr = b.addr + (n -> a)
```

- `n -> a` crea la coppia singola da aggiungere
- `+` fa l'unione con le coppie esistenti in `b.addr`

---

## 4. Operazione Delete: Rimuovere Elementi

### 4.1 Specifica dell'Operazione Delete

`⏱️ 00:37:12 - 00:38:52`

Analogamente all'operazione `add`, possiamo definire un'operazione `delete` per rimuovere elementi.

📝 **Definizione dell'Operazione Delete**

```alloy
pred delete[b, b': Book, n: Name] {
  b'.addr = b.addr - (n -> Addr)
}
```

**Interpretazione:**
- `b`: Libro **prima** della rimozione
- `b'`: Libro **dopo** la rimozione
- `n`: Nome da rimuovere
- `n -> Addr`: **Tutte le coppie** che iniziano con `n`
- `-`: Operatore di **differenza insiemistica**

💡 **Insight - Differenza Insiemistica**

L'operatore `-` in Alloy rappresenta la **differenza di insiemi**:
```
A - B = {x | x ∈ A ∧ x ∉ B}
```

Rimuove da A tutti gli elementi che sono anche in B.

🔍 **Analisi Dettagliata**

**Cosa significa `n -> Addr`?**

- `Addr` rappresenta **tutti gli indirizzi** possibili
- `n -> Addr` = tutte le coppie `{(n, a0), (n, a1), (n, a2), ...}`
- Quindi `b.addr - (n -> Addr)` rimuove **qualsiasi coppia** che inizia con `n`

📊 **Esempio Concreto**

Supponiamo:
```
b.addr = {(N0, A0), (N1, A1), (N2, A0)}
n = N1
```

Calcoliamo `n -> Addr`:
```
N1 -> Addr = {(N1, A0), (N1, A1), (N1, A2), ...}
```

Calcoliamo `b'.addr`:
```
b'.addr = b.addr - (N1 -> Addr)
        = {(N0, A0), (N1, A1), (N2, A0)} - {(N1, A0), (N1, A1), (N1, A2), ...}
        = {(N0, A0), (N2, A0)}  // Rimossa (N1, A1)
```

✅ **Risultato:** Il nome `N1` non è più associato a nessun indirizzo!

📝 **Nota - Vincolo Lone**

Dato che abbiamo `addr: Name -> lone Addr`, ogni nome ha **al più 1 indirizzo**.

Quindi:
- `n -> Addr` contiene al massimo 1 coppia effettivamente presente in `b.addr`
- L'operazione rimuove **quella singola coppia** (se esiste)

⚠️ **Attenzione - Elemento Non Presente**

Se `n` non ha indirizzi in `b`:
```
b.addr = {(N0, A0), (N2, A1)}
n = N1  // N1 non è nel libro
```

Allora:
```
b'.addr = {(N0, A0), (N2, A1)} - {} = {(N0, A0), (N2, A1)}
```

Il libro **non cambia**! (Corner case simile ad `add`)

---

### 4.2 Esecuzione e Corner Cases

`⏱️ 00:38:52 - 00:40:34`

Eseguiamo l'operazione `delete` e analizziamo i mondi generati.

📝 **Predicato per Visualizzazione**

```alloy
pred showDelete[b, b': Book, n: Name] {
  delete[b, b', n]
}

run showDelete
```

Oppure possiamo eseguire direttamente:
```alloy
run delete
```

💡 **Insight - Multiple Run Commands**

In una specifica Alloy, possiamo avere **molteplici comandi** `run` e `check`:
- Possiamo scegliere quale eseguire
- Utile per testare diverse operazioni separatamente
- Ogni comando può avere scope diversi

🔍 **Primo Mondo - Corner Case**

L'Analyzer mostra tipicamente prima il corner case:

```
Book0:
  addr = {}

b = Book0
b' = Book0  // Stesso libro!
n = N0
```

**Interpretazione:**
- Libro vuoto iniziale
- Proviamo a rimuovere `N0`
- Il libro rimane vuoto (ovviamente, `N0` non c'era)
- `b = b'` perché nessun cambiamento è avvenuto

⚠️ **Attenzione - Delete Senza Effetto**

Come per `add`, anche `delete` può non avere effetto se:
- L'elemento da rimuovere **non esiste** nel libro
- In questo caso `b'.addr = b.addr`

📊 **Secondo Mondo - Operazione Riuscita**

Cliccando "Next", vediamo un caso "normale":

```
Book2:
  addr = {(N0, A1), (N1, A0)}

Book1:
  addr = {(N0, A1)}

b = Book2
b' = Book1
n = N1
```

**Interpretazione:**
- Libro iniziale `Book2` contiene `{(N0, A1), (N1, A0)}`
- Rimuoviamo il nome `N1`
- Libro risultante `Book1` contiene solo `{(N0, A1)}`
- La coppia `(N1, A0)` è stata rimossa con successo ✅

💡 **Insight - Simmetria con Add**

Entrambe le operazioni (`add` e `delete`) hanno corner cases simili:
- `add` può non avere effetto se l'elemento esiste già
- `delete` può non avere effetto se l'elemento non esiste

Se vogliamo evitare questi corner cases, dobbiamo aggiungere **precondizioni**:

```alloy
pred deleteStrict[b, b': Book, n: Name] {
  some a: Addr | n -> a in b.addr  // Precondizione: n deve esistere
  b'.addr = b.addr - (n -> Addr)
}
```

---

## 🎯 Riepilogo Parte 1

In questa prima parte della lezione abbiamo:

✅ **Riveduto i Concetti Base di Alloy:**
- Signature, relazioni ternarie, vincoli di cardinalità
- Join operator e composizione di relazioni
- Quantificazione esistenziale e universale

✅ **Esplorato Predicati Avanzati:**
- Predicati con vincoli di cardinalità (`#b.addr > 1`)
- Rilevamento di inconsistenze (predicati insoddisfacibili)
- Uso del comando `run` con scope

✅ **Definito Operazioni Mutabili:**
- **Add**: Aggiunge una coppia nome→indirizzo
- **Delete**: Rimuove tutte le coppie per un dato nome
- Operatori insiemistici: unione (`+`), differenza (`-`), prodotto cartesiano (`->`)

✅ **Analizzato Corner Cases:**
- Operazioni senza effetto (elemento già presente / non presente)
- Situazioni dove `b = b'` (nessun cambiamento di stato)
- Importanza di corner cases per validare specifiche

✅ **Discusso Decisioni di Design:**
- Semantica idempotente vs precondizioni
- Quando forzare cambiamenti di stato
- Come documentare e testare le scelte

---

### 📚 Concetti Chiave da Ricordare

1. **Join Operator (.)**: Compone relazioni eliminando l'elemento intermedio
2. **Vincoli Cardinalità**: `lone`, `one`, `some`, `set` limitano le molteplicità
3. **Inconsistenze**: Predicati che contraddicono la specifica → "No instance found"
4. **Corner Cases**: Situazioni al limite spesso rivelano ambiguità
5. **Operatori Insiemistici**: `+` (unione), `-` (differenza), `->` (prodotto cartesiano)

---

### 🔜 Prossimi Argomenti (Parte 2)

Nella prossima parte vedremo:
- **Asserzioni (assert)**: Verificare proprietà della specifica
- **Counter-examples**: Quando le asserzioni falliscono
- **Correzione di Asserzioni**: Come modificare specifiche basandosi sui counter-examples
- **Funzioni**: Definire operazioni riusabili
- **Bounded Verification**: Limiti e garanzie dell'analisi Alloy

---

*Fine Parte 1 - Continua nella Parte 2...*

---
---

# 📘 Lezione 7 - Alloy: Operazioni e Asserzioni (Parte 2)

**Corso:** Ingegneria del Software  
**Data:** 8 Ottobre  
**Argomento:** Asserzioni, Counter-examples, Funzioni, Bounded Verification  
**Durata Parte 2:** 00:40:34 - 01:04:48 (~24 minuti)

---

## 📑 Indice dei Contenuti - Parte 2

### [5. Asserzioni: Verificare Proprietà](#5-asserzioni-verificare-proprietà) `00:40:34 - 00:48:12`
   - [5.1 Introduzione alle Asserzioni](#51-introduzione-alle-asserzioni) `00:40:34 - 00:41:42`
   - [5.2 Esempio: delUndoesAdd](#52-esempio-delundoesadd) `00:41:42 - 00:44:48`
   - [5.3 Asserzioni vs Predicati](#53-asserzioni-vs-predicati) `00:44:48 - 00:45:53`
   - [5.4 Esecuzione del Check Command](#54-esecuzione-del-check-command) `00:45:53 - 00:47:39`
   - [5.5 Interpretazione del Counter-example](#55-interpretazione-del-counter-example) `00:47:39 - 00:48:12`

### [6. Correzione delle Asserzioni](#6-correzione-delle-asserzioni) `00:48:12 - 00:54:29`
   - [6.1 Analisi del Problema](#61-analisi-del-problema) `00:48:12 - 00:49:19`
   - [6.2 Modifica dell'Asserzione](#62-modifica-dellasserzione) `00:49:19 - 00:50:24`
   - [6.3 Quantificazione Universale](#63-quantificazione-universale) `00:50:24 - 00:52:47`
   - [6.4 Implicazione vs AND](#64-implicazione-vs-and) `00:52:47 - 00:54:29`

### [7. Bounded Verification](#7-bounded-verification) `00:54:29 - 01:02:22`
   - [7.1 Concetto di Analisi Bounded](#71-concetto-di-analisi-bounded) `00:54:29 - 00:57:08`
   - [7.2 Scope e Performance](#72-scope-e-performance) `00:57:08 - 00:58:18`
   - [7.3 Bounded Verification vs Testing](#73-bounded-verification-vs-testing) `00:58:18 - 00:59:31`
   - [7.4 Vantaggi e Svantaggi](#74-vantaggi-e-svantaggi) `00:59:31 - 01:02:22`

### [8. Funzioni in Alloy](#8-funzioni-in-alloy) `01:02:22 - 01:04:48`
   - [8.1 Definizione di Funzioni](#81-definizione-di-funzioni) `01:02:22 - 01:03:34`
   - [8.2 Esempio: Funzione Lookup](#82-esempio-funzione-lookup) `01:03:34 - 01:04:11`
   - [8.3 Uso delle Funzioni nelle Asserzioni](#83-uso-delle-funzioni-nelle-asserzioni) `01:04:11 - 01:04:48`

---

## 5. Asserzioni: Verificare Proprietà

### 5.1 Introduzione alle Asserzioni

`⏱️ 00:40:34 - 00:41:42`

Le **asserzioni** (assertions) sono uno strumento potente in Alloy per **verificare proprietà** della nostra specifica.

💡 **Insight - Cosa sono le Asserzioni**

Un'asserzione è una **proprietà** che dovrebbe essere **sempre vera** nella nostra specifica:
- Non genera mondi come i predicati
- Viene **verificata** dall'Analyzer
- Se è falsa, l'Analyzer genera un **counter-example**

📝 **Sintassi delle Asserzioni**

```alloy
assert NomeAsserzione {
  // Formula logica che dovrebbe essere sempre vera
}
```

Per verificare un'asserzione:
```alloy
check NomeAsserzione
```

🎯 **Obiettivo - Verifica di Proprietà**

Le asserzioni servono per:
- **Validare** che operazioni si comportino come previsto
- **Scoprire bug** nella specifica
- **Documentare** proprietà attese del sistema
- **Testare** invarianti e postcondizioni

📊 **Tabella: Predicati vs Asserzioni**

| Aspetto | Predicato | Asserzione |
|---------|-----------|------------|
| **Sintassi** | `pred nome { ... }` | `assert nome { ... }` |
| **Comando** | `run nome` | `check nome` |
| **Scopo** | Generare mondi | Verificare proprietà |
| **Output (vero)** | Mondi che soddisfano | "No counterexample" |
| **Output (falso)** | "No instance found" | Counter-example |
| **Uso** | Esplorare specifiche | Validare correttezza |

---

### 5.2 Esempio: delUndoesAdd

`⏱️ 00:41:42 - 00:44:48`

Vediamo un esempio concreto: verificare che `delete` sia l'operazione **inversa** di `add`.

📝 **Definizione dell'Asserzione**

```alloy
assert delUndoesAdd {
  all b, b', b'': Book, n: Name, a: Addr |
    (add[b, b', n, a] and delete[b', b'', n]) 
    implies 
    b.addr = b''.addr
}
```

🔍 **Analisi Dettagliata**

**Interpretazione in italiano:**
> "Per ogni tripla di libri `b`, `b'`, `b''`, ogni nome `n` e ogni indirizzo `a`:  
> Se aggiungiamo `(n, a)` a `b` ottenendo `b'`,  
> e poi rimuoviamo `n` da `b'` ottenendo `b''`,  
> allora il libro finale `b''` deve essere uguale al libro iniziale `b`"

**In formula logica:**
```
∀b, b', b'': Book, ∀n: Name, ∀a: Addr:
  (add(b, b', n, a) ∧ delete(b', b'', n)) ⟹ (b.addr = b''.addr)
```

💡 **Insight - Operazioni Inverse**

L'asserzione esprime il concetto di **operazioni inverse**:
- **Add** aggiunge un elemento
- **Delete** rimuove un elemento
- Applicarle in sequenza dovrebbe **annullare l'effetto**

📊 **Visualizzazione Grafica**

```
Stato Iniziale (b)     Add (n,a)      Stato Intermedio (b')    Delete n      Stato Finale (b'')
┌────────────────┐     ───────>       ┌────────────────────┐    ───────>     ┌────────────────┐
│ addr = {       │                    │ addr = {           │                 │ addr = {       │
│   (N1, A0)     │                    │   (N1, A0),        │                 │   (N1, A0)     │
│ }              │                    │   (n, a)  ← nuovo  │                 │ }              │
└────────────────┘                    └────────────────────┘                 └────────────────┘
                                                                               
                       Asserzione: b.addr = b''.addr dovrebbe essere VERA
```

⚠️ **Attenzione - Precondizione Implicita**

L'asserzione **assume implicitamente** che:
- `add` e `delete` siano eseguibili
- Non ci siano vincoli che impediscano le operazioni

Vedremo che questa assunzione può causare problemi!

---

### 5.3 Asserzioni vs Predicati

`⏱️ 00:44:48 - 00:45:53`

È fondamentale comprendere la **differenza** tra predicati e asserzioni.

💡 **Insight - Ruoli Diversi**

**Predicati:**
- Sono **vincoli** che devono essere veri in un mondo specifico
- Vengono usati con `run` per **generare** mondi
- Se soddisfacibili, l'Analyzer mostra un mondo
- Se insoddisfacibili, mostra "No instance found"

**Asserzioni:**
- Sono **proprietà globali** che dovrebbero valere sempre
- Vengono usate con `check` per **verificare** correttezza
- Se vere, l'Analyzer dice "No counterexample found"
- Se false, l'Analyzer genera un **counter-example**

📊 **Tabella Comparativa Dettagliata**

| Aspetto | Predicato | Asserzione |
|---------|-----------|------------|
| **Natura** | Vincolo locale | Proprietà globale |
| **Verifica** | Soddisfacibilità | Validità |
| **Comando** | `run predName` | `check assertName` |
| **Successo** | Mostra mondo | "No counterexample" |
| **Fallimento** | "No instance" | Mostra counter-example |
| **Interpretazione Successo** | "Esiste un mondo così" | "La proprietà è sempre vera" |
| **Interpretazione Fallimento** | "Nessun mondo possibile" | "Ecco un caso dove è falsa" |

🔍 **Esempio Comparativo**

**Come Predicato:**
```alloy
pred delUndoesAddPred[b, b', b'': Book, n: Name, a: Addr] {
  add[b, b', n, a]
  delete[b', b'', n]
  b.addr = b''.addr
}

run delUndoesAddPred
```
Questo **genera un mondo** dove la sequenza add-delete riporta allo stato iniziale.

**Come Asserzione:**
```alloy
assert delUndoesAdd {
  all b, b', b'': Book, n: Name, a: Addr |
    (add[b, b', n, a] and delete[b', b'', n]) implies b.addr = b''.addr
}

check delUndoesAdd
```
Questo **verifica** se la proprietà vale **sempre**, per ogni possibile combinazione di libri, nomi e indirizzi.

✅ **Regola Pratica - Quando Usare Cosa**

**Usa Predicati quando:**
- Vuoi **esplorare** la specifica
- Vuoi **vedere esempi** di mondi validi
- Vuoi **testare** se una situazione è possibile

**Usa Asserzioni quando:**
- Vuoi **verificare** una proprietà
- Vuoi **validare** che un invariante sia rispettato
- Vuoi **documentare** assunzioni sul sistema

---

### 5.4 Esecuzione del Check Command

`⏱️ 00:45:53 - 00:47:39`

Eseguiamo ora l'asserzione per verificare se `delete` annulla veramente `add`.

📝 **Comando di Verifica**

```alloy
check delUndoesAdd
```

Opzionalmente con scope:
```alloy
check delUndoesAdd for 3
check delUndoesAdd for 5
```

🎯 **Domanda Cruciale**

Prima di eseguire, possiamo chiederci:
> "L'asserzione genererà un counter-example o sarà valida?"

💡 **Insight - Pensiero Critico**

Ricordiamo i **corner cases** visti per `add`:
- Se `(n, a)` è **già presente** in `b`, allora `b' = b`
- Quindi `add` potrebbe non avere effetto!

Se `add` non ha effetto ma `delete` rimuove l'elemento, allora:
- Partiamo da `b` con `(n, a)`
- `add` non cambia nulla: `b' = b` (ancora con `(n, a)`)
- `delete` rimuove `n`: `b''` senza `(n, a)`
- Risultato: `b.addr ≠ b''.addr` ❌

⚠️ **Attenzione - Counter-example Atteso**

Ci aspettiamo un **counter-example**!

🔍 **Esecuzione e Risultato**

```alloy
check delUndoesAdd
```

**Output dell'Analyzer:**
```
Counterexample found.
```

L'Analyzer mostra un mondo specifico dove l'asserzione **non vale**!

📊 **Visualizzazione del Counter-example**

```
Book1:
  addr = {}  (libro vuoto)

Book2:
  addr = {(N0, A1)}  (contiene N0→A1)

b = Book2
b' = Book2  // Stesso libro! Add non ha avuto effetto
b'' = Book1  // Libro vuoto dopo delete
n = N0
a = A1
```

**Verifica:**
1. `add[Book2, Book2, N0, A1]`: ✅ Vero (N0→A1 già presente)
2. `delete[Book2, Book1, N0]`: ✅ Vero (rimuove N0)
3. `Book2.addr = Book1.addr`: ❌ **FALSO!**
   - `Book2.addr = {(N0, A1)}`
   - `Book1.addr = {}`
   - Non sono uguali!

💡 **Insight - Counter-example Rivelatore**

Il counter-example ci mostra il **corner case** problematico:
- Se l'elemento esiste già, `add` è idempotente (nessun effetto)
- Ma `delete` lo rimuove comunque
- Quindi la sequenza add-delete **non** riporta allo stato iniziale

---

### 5.5 Interpretazione del Counter-example

`⏱️ 00:47:39 - 00:48:12`

Come interpretare e usare il counter-example per migliorare la specifica?

🔍 **Analisi del Counter-example**

Il counter-example ci dice:
> "Esiste una situazione dove l'asserzione è falsa"

**Domanda:** È un **bug nella specifica** o nella **asserzione**?

In questo caso:
- La **specifica** (operazioni `add` e `delete`) è corretta
- L'**asserzione** è troppo forte (assume una precondizione non specificata)

💡 **Insight - Precondizione Mancante**

L'asserzione assume implicitamente che:
> "L'elemento `(n, a)` **non è presente** in `b` prima dell'add"

Ma non abbiamo specificato questa **precondizione**!

✅ **Soluzione - Rafforzare l'Asserzione**

Dobbiamo aggiungere la precondizione esplicita:
```alloy
assert delUndoesAddFixed {
  all b, b', b'': Book, n: Name, a: Addr |
    (n -> a not in b.addr and  // ← PRECONDIZIONE AGGIUNTA
     add[b, b', n, a] and 
     delete[b', b'', n]) 
    implies 
    b.addr = b''.addr
}
```

Ora diciamo esplicitamente:
> "Se `(n, a)` **non** è già in `b`, allora add seguito da delete riporta allo stato iniziale"

📝 **Nota - Esplorare Altri Counter-examples**

L'Analyzer permette di vedere **altri counter-examples** cliccando "Next":
- Può mostrare diverse configurazioni problematiche
- Utile per capire tutti i casi limite
- Aiuta a raffinare la specifica

---

## 6. Correzione delle Asserzioni

### 6.1 Analisi del Problema

`⏱️ 00:48:12 - 00:49:19`

Analizziamo in dettaglio **perché** l'asserzione originale fallisce e **come** correggerla.

🔍 **Riepilogo del Problema**

**Asserzione originale:**
```alloy
assert delUndoesAdd {
  all b, b', b'': Book, n: Name, a: Addr |
    (add[b, b', n, a] and delete[b', b'', n]) implies b.addr = b''.addr
}
```

**Problema identificato:**
- L'asserzione **non specifica** quando la proprietà dovrebbe valere
- Assume implicitamente che l'elemento non sia già presente
- Ma questa assunzione **non è espressa** nella formula

📊 **Scenario Problematico**

```
Situazione iniziale:
b.addr = {(N0, A1)}  // N0 è già nel libro

Operazione add[b, b', N0, A1]:
b'.addr = b.addr + {(N0, A1)} = {(N0, A1)}  // Nessun cambiamento
b' = b  // Stesso libro

Operazione delete[b', b'', N0]:
b''.addr = b'.addr - (N0 -> Addr) = {}  // N0 rimosso

Risultato:
b.addr = {(N0, A1)}
b''.addr = {}
b.addr ≠ b''.addr  ❌ L'asserzione fallisce!
```

💡 **Insight - Causa Radice**

La causa del problema è la **semantica idempotente** dell'operazione `add`:
- `add` definita come: `b'.addr = b.addr + (n -> a)`
- Se `(n, a) ∈ b.addr`, allora `b.addr ∪ {(n, a)} = b.addr`
- Quindi `add` può **non modificare** lo stato

Ma `delete` **rimuove sempre** l'elemento (se presente):
- `delete` definita come: `b'.addr = b.addr - (n -> Addr)`
- Rimuove `n` anche se era già presente prima dell'`add`

⚠️ **Attenzione - Asimmetria**

C'è un'**asimmetria** tra `add` e `delete`:
- `add` è **condizionale**: aggiunge solo se non presente
- `delete` è **incondizionale**: rimuove sempre

Questa asimmetria causa il fallimento dell'asserzione!

---

### 6.2 Modifica dell'Asserzione

`⏱️ 00:49:19 - 00:50:24`

Correggiamo l'asserzione aggiungendo la **precondizione** necessaria.

📝 **Asserzione Corretta**

```alloy
assert delUndoesAddFixed {
  all b, b', b'': Book, n: Name, a: Addr |
    (n -> a not in b.addr and        // ← PRECONDIZIONE
     add[b, b', n, a] and 
     delete[b', b'', n]) 
    implies 
    b.addr = b''.addr
}
```

🔍 **Analisi della Correzione**

**Parte aggiunta:** `n -> a not in b.addr`

**Significato:**
- "La coppia `(n, a)` **non è presente** in `b.addr`"
- Questo garantisce che `add` **effettivamente aggiunga** qualcosa di nuovo
- Quindi `b' ≠ b` (il libro cambia davvero)

💡 **Insight - Precondizione Esplicita**

Ora l'asserzione dice esplicitamente:
> "Se partiamo da un libro che **non contiene** `(n, a)`,  
> e aggiungiamo `(n, a)` ottenendo `b'`,  
> e poi rimuoviamo `n` ottenendo `b''`,  
> allora `b''` è uguale a `b`"

📊 **Scenario con Precondizione**

```
Precondizione: n -> a not in b.addr
b.addr = {(N1, A0)}  // N0 NON è nel libro ✅

Operazione add[b, b', N0, A1]:
b'.addr = {(N1, A0)} ∪ {(N0, A1)} = {(N1, A0), (N0, A1)}  // Elemento aggiunto
b' ≠ b  // Libro modificato

Operazione delete[b', b'', N0]:
b''.addr = {(N1, A0), (N0, A1)} - {(N0, Addr)} = {(N1, A0)}  // N0 rimosso

Risultato:
b.addr = {(N1, A0)}
b''.addr = {(N1, A0)}
b.addr = b''.addr  ✅ L'asserzione vale!
```

✅ **Verifica della Correzione**

```alloy
check delUndoesAddFixed
```

**Output atteso:**
```
No counterexample found.
Assertion may be valid.
```

🎯 **Obiettivo Raggiunto**

Con la precondizione, l'asserzione è **valida** entro lo scope di verifica!

📝 **Nota - Aggiunta al Codice**

Nella specifica Alloy, aggiungiamo:

```alloy
// Operazione add
pred add[b, b': Book, n: Name, a: Addr] {
  b'.addr = b.addr + (n -> a)
}

// Operazione delete
pred delete[b, b': Book, n: Name] {
  b'.addr = b.addr - (n -> Addr)
}

// Asserzione CORRETTA con precondizione
assert delUndoesAddFixed {
  all b, b', b'': Book, n: Name, a: Addr |
    (n -> a not in b.addr and        // Precondizione essenziale
     add[b, b', n, a] and 
     delete[b', b'', n]) 
    implies 
    b.addr = b''.addr
}

check delUndoesAddFixed
```

---

### 6.3 Quantificazione Universale

`⏱️ 00:50:24 - 00:52:47`

Approfondiamo il significato della **quantificazione universale** nelle asserzioni.

📝 **Sintassi della Quantificazione**

```alloy
all variabile: Tipo | formula
```

**Significato:**
> "Per **ogni** elemento del tipo `Tipo`, la `formula` deve essere vera"

💡 **Insight - Quantificazione nell'Asserzione**

Nell'asserzione:
```alloy
all b, b', b'': Book, n: Name, a: Addr | ...
```

Stiamo dicendo:
> "Per **ogni possibile combinazione** di:  
> - tre libri `b`, `b'`, `b''`  
> - un nome `n`  
> - un indirizzo `a`  
> la proprietà deve valere"

🔍 **Esempio Concreto di Enumerazione**

Supponiamo scope: `for 3`
- Book: `{Book0, Book1, Book2}`
- Name: `{N0, N1, N2}`
- Addr: `{A0, A1, A2}`

L'Analyzer verifica la formula per **tutte le combinazioni**:

```
Combinazione 1:
  b = Book0, b' = Book0, b'' = Book0, n = N0, a = A0

Combinazione 2:
  b = Book0, b' = Book0, b'' = Book0, n = N0, a = A1

Combinazione 3:
  b = Book0, b' = Book0, b'' = Book0, n = N0, a = A2

...

Combinazione N:
  b = Book2, b' = Book2, b'' = Book2, n = N2, a = A2
```

**Totale combinazioni:** 3³ × 3 × 3 = **243 combinazioni**

⚠️ **Attenzione - Esplosione Combinatoria**

Con scope più grandi, il numero di combinazioni **cresce esponenzialmente**:
- `for 3`: 243 combinazioni
- `for 5`: 5³ × 5 × 5 = **3125 combinazioni**
- `for 10`: 10³ × 10 × 10 = **100,000 combinazioni**

Ecco perché l'analisi diventa più lenta con scope maggiori!

📊 **Tabella: Valutazione delle Combinazioni**

| b | b' | b'' | n | a | Precondizione | add | delete | Conclusione | Risultato |
|---|----|----|---|---|--------------|-----|--------|-------------|-----------|
| B0 | B0 | B1 | N0 | A0 | `n->a not in b` | ✅ | ✅ | `b = b''`? | Verifica |
| B1 | B1 | B1 | N0 | A0 | `n->a in b` | - | - | - | Precondizione falsa, implicazione vera |
| B0 | B1 | B2 | N1 | A1 | `n->a not in b` | ✅ | ✅ | `b = b''`? | Verifica |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

💡 **Insight - Implicazione e Precondizioni**

Quando la **precondizione** è falsa, l'**implicazione** è automaticamente vera:
```
Falso ⟹ Qualunque cosa = Vero
```

Quindi l'Analyzer **salta** le combinazioni dove la precondizione non vale!

✅ **Regola Pratica - Interpretazione**

Quando scriviamo:
```alloy
all x: T | P(x) implies Q(x)
```

Stiamo dicendo:
> "Per ogni `x` dove `P(x)` è vero, allora `Q(x)` deve essere vero"

Se `P(x)` è falso, l'implicazione è banalmente vera (non ci interessa quel caso).

---

### 6.4 Implicazione vs AND

`⏱️ 00:52:47 - 00:54:29`

Un errore comune è confondere **implicazione** (`implies`) con **congiunzione** (`and`).

⚠️ **Errore Comune - Usare AND invece di IMPLIES**

**SBAGLIATO:**
```alloy
assert delUndoesAddWrong {
  all b, b', b'': Book, n: Name, a: Addr |
    (n -> a not in b.addr and 
     add[b, b', n, a] and 
     delete[b', b'', n] and     // ← AND invece di IMPLIES
     b.addr = b''.addr)
}
```

**CORRETTO:**
```alloy
assert delUndoesAddCorrect {
  all b, b', b'': Book, n: Name, a: Addr |
    (n -> a not in b.addr and 
     add[b, b', n, a] and 
     delete[b', b'', n]) 
    implies                      // ← IMPLIES corretto
    b.addr = b''.addr
}
```

🔍 **Differenza Semantica**

**Con AND (sbagliato):**
```
∀b, b', b'', n, a: 
  (precondizione ∧ add ∧ delete ∧ conclusione)
```
Significa: "Per ogni combinazione, **tutte e quattro** le condizioni devono essere vere"

**Con IMPLIES (corretto):**
```
∀b, b', b'', n, a: 
  (precondizione ∧ add ∧ delete) ⟹ conclusione
```
Significa: "Per ogni combinazione **dove le prime tre** sono vere, **allora** la conclusione deve essere vera"

📊 **Tabella Comparativa**

| Scenario | Versione AND | Versione IMPLIES |
|----------|-------------|------------------|
| Precondizione vera, conclusione vera | ✅ Vero | ✅ Vero |
| Precondizione vera, conclusione falsa | ❌ Falso | ❌ Falso |
| Precondizione falsa, conclusione vera | ❌ Falso | ✅ Vero |
| Precondizione falsa, conclusione falsa | ❌ Falso | ✅ Vero |

💡 **Insight - AND è Troppo Restrittivo**

Con AND, richiediamo che:
- La precondizione sia vera, **E**
- Le operazioni siano eseguite, **E**
- La conclusione sia vera

Ma questo è **troppo forte**! Non vogliamo richiedere che la conclusione sia **sempre** vera, ma solo **quando le premesse sono vere**.

🔍 **Esempio di Fallimento con AND**

```alloy
check delUndoesAddWrong
```

**Counter-example trovato:**
```
b = Book0 con addr = {(N0, A0)}
b' = Book1 con addr = {}
b'' = Book2 con addr = {(N1, A1)}
n = N1
a = A2
```

**Verifica:**
- `n -> a not in b.addr`: ✅ `(N1, A2)` non è in `{(N0, A0)}`
- `add[b, b', N1, A2]`: ❌ **FALSO** (b' non contiene N1→A2)
- Formula con AND: ❌ **FALSO** (una delle condizioni è falsa)

Con AND, l'asserzione richiede che **tutte le combinazioni** soddisfino tutte le condizioni, il che è impossibile!

✅ **Regola Pratica - Quando Usare Cosa**

**Usa IMPLIES quando:**
- Hai una **premessa** (precondizione) e una **conclusione** (postcondizione)
- Vuoi dire: "SE succede X, ALLORA deve valere Y"
- Pattern: `(precondizioni) implies (conclusione)`

**Usa AND quando:**
- Tutte le condizioni devono valere **simultaneamente**
- Non c'è relazione di causa-effetto
- Pattern: Vincoli su un singolo mondo

📝 **Nota - Errore Frequente negli Esami**

Secondo il professore, usare AND invece di IMPLIES nelle asserzioni è un **errore comune** negli esami!

**Consiglio:** Chiediti sempre:
> "Sto specificando condizioni simultanee, o sto dicendo 'se X allora Y'?"

---

## 7. Bounded Verification

### 7.1 Concetto di Analisi Bounded

`⏱️ 00:54:29 - 00:57:08`

L'analisi che l'Alloy Analyzer esegue è sempre **bounded** (limitata).

💡 **Insight - Cosa significa "Bounded"**

**Bounded verification** significa:
- L'analisi considera solo **insiemi finiti** di atomi
- Il numero di atomi è limitato dallo **scope**
- Non possiamo verificare su insiemi **infiniti**

📝 **Definizione Formale**

**Verifica bounded:**
> Verifica di proprietà entro un **limite fissato** (bound) sul numero di elementi considerati.

**Esempio:**
```alloy
check delUndoesAddFixed for 3
```
Verifica la proprietà considerando:
- Al massimo 3 istanze di `Book`
- Al massimo 3 istanze di `Name`
- Al massimo 3 istanze di `Addr`

🔍 **Confronto: Verifica Bounded vs Unbounded**

| Aspetto | Verifica Bounded | Verifica Unbounded |
|---------|-----------------|-------------------|
| **Insiemi** | Finiti | Potenzialmente infiniti |
| **Scope** | Fissato (es. for 3) | Illimitato |
| **Completezza** | Solo entro i limiti | Su tutti i casi possibili |
| **Decidibilità** | Sempre terminante | Può non terminare |
| **Garanzie** | "Corretto fino a N elementi" | "Corretto per ogni N" |
| **Tool** | Alloy, SAT solvers | Theorem provers (Coq, Isabelle) |

⚠️ **Attenzione - Limiti della Bounded Verification**

**Cosa possiamo dire:**
> "La proprietà vale per tutti i mondi con **al più N elementi**"

**Cosa NON possiamo dire:**
> "La proprietà vale per **ogni possibile mondo**"

**Esempio:**
```alloy
check myAssertion for 5
// Output: No counterexample found
```

**Interpretazione corretta:**
> "Non ci sono counter-examples con ≤5 elementi per ogni tipo"

**Interpretazione SBAGLIATA:**
> "La proprietà è sempre vera" (potrebbe essere falsa con 6+ elementi!)

📊 **Esempio: Limiti dello Scope**

Supponiamo un'asserzione che **fallisce** solo con 6+ elementi:

```alloy
assert propertyX {
  // Proprietà che richiede almeno 6 elementi per fallire
  all s: SomeSet | #s < 6 implies someCondition[s]
}

check propertyX for 5
// Output: No counterexample found ✅

check propertyX for 6
// Output: Counterexample found! ❌
```

Con scope 5, non troviamo il counter-example (che richiede 6 elementi)!

💡 **Insight - Bounded ≠ Incompleto**

Anche se bounded, la verifica è **completa** entro i limiti:
- **Esplora completamente** lo spazio limitato
- **Trova tutti** i counter-examples entro lo scope
- Fornisce **certezza** nei limiti specificati

---

### 7.2 Scope e Performance

`⏱️ 00:57:08 - 00:58:18`

Lo **scope** influenza sia la **copertura** che la **performance** dell'analisi.

📝 **Sintassi dello Scope**

```alloy
check assertionName for N
```

Dove `N` è il numero massimo di istanze per ogni signature.

**Scope personalizzato:**
```alloy
check assertionName for 3 but 5 Book
```
- Al massimo 3 istanze per ogni tipo
- Al massimo 5 istanze per `Book` (override)

💡 **Insight - Trade-off Scope vs Performance**

| Scope | Pro | Contro |
|-------|-----|--------|
| **Piccolo** (3-5) | ⚡ Veloce (millisecondi-secondi) | 🔍 Copertura limitata |
| **Medio** (5-8) | ⚖️ Buon compromesso | ⏱️ Moderatamente lento |
| **Grande** (10+) | 🎯 Alta copertura | 🐌 Molto lento (minuti-ore) |

🔍 **Test Pratico di Performance**

```alloy
// Veloce
check myAssertion for 3
// Tipicamente: < 1 secondo

// Medio
check myAssertion for 5
// Tipicamente: 1-10 secondi

// Lento
check myAssertion for 10
// Tipicamente: minuti (dipende dalla complessità)

// Molto lento
check myAssertion for 100
// Probabilmente impraticabile!
```

⚠️ **Attenzione - Complessità Esponenziale**

Il tempo di analisi **cresce esponenzialmente** con lo scope:
- Raddoppiare lo scope può **decuplicare** il tempo
- Con specifiche complesse, anche scope 10 può essere proibitivo

📊 **Esempio Pratico**

**Specifica semplice (Address Book):**
- `for 3`: ~0.1 secondi
- `for 5`: ~2 secondi
- `for 10`: ~60 secondi

**Specifica complessa (10+ signatures, molte relazioni):**
- `for 3`: ~10 secondi
- `for 5`: ~5 minuti
- `for 10`: potrebbe non terminare in tempi ragionevoli

✅ **Regola Pratica - Scegliere lo Scope**

1. **Sviluppo iniziale:** `for 3` (feedback veloce)
2. **Testing regolare:** `for 4-5` (buon compromesso)
3. **Verifica finale:** `for 8-10` (se il tempo lo permette)
4. **CI/CD:** `for 5-6` (bilanciamento tempo/copertura)

💡 **Insight - Scope Incrementale**

Strategia consigliata:
```alloy
// Prima iterazione - veloce
check myAssertion for 3

// Se passa, aumenta
check myAssertion for 5

// Se passa ancora, aumenta (se fattibile)
check myAssertion for 8
```

Se trovi un counter-example con scope piccolo, **non serve aumentare** (hai già trovato il problema)!

---

### 7.3 Bounded Verification vs Testing

`⏱️ 00:58:18 - 00:59:31`

Confrontiamo la **bounded verification** con il **testing tradizionale**.

📊 **Tabella Comparativa Dettagliata**

| Aspetto | Testing Tradizionale | Bounded Verification |
|---------|---------------------|---------------------|
| **Approccio** | Campionamento input | Esplorazione esaustiva |
| **Copertura** | Casi selezionati | Tutti i casi entro scope |
| **Spazio esplorato** | Sparse (pochi punti) | Denso (tutto lo spazio) |
| **Garanzie** | "Funziona su questi input" | "Funziona per tutti gli input ≤N" |
| **Completezza** | Parziale | Completa (entro limiti) |
| **Bug nascosti** | Possono sfuggire | Trovati se entro scope |
| **Scalabilità** | Buona (parallelizzabile) | Limitata (esponenziale) |
| **Tool** | JUnit, pytest, etc. | Alloy, model checkers |

💡 **Insight - Due Paradigmi Diversi**

**Testing:**
```
Spazio degli input: ████████████████████████
Input testati:      ●     ●  ●       ●    ●
                    ↑ Campionamento sparse
```

**Bounded Verification:**
```
Spazio limitato:    ████████
Esplorazione:       ████████  ← Tutto coperto!
                    
Spazio oltre limit: ░░░░░░░░  ← Non coperto
```

🔍 **Esempio Concreto**

**Scenario:** Verificare la proprietà "delete annulla add"

**Con Testing:**
```java
@Test
public void testDeleteUndoesAdd() {
    Book b = new Book();
    b.add("Alice", "123 Main St");
    Book b2 = b.copy();
    b2.delete("Alice");
    assertEquals(b, b2);
}
```
- Testa **un caso specifico** (Alice, 123 Main St)
- Se passa, sappiamo che funziona per questo input
- Bug con altri nomi/indirizzi potrebbero sfuggire

**Con Bounded Verification:**
```alloy
assert delUndoesAdd {
  all b, b', b'': Book, n: Name, a: Addr |
    (n -> a not in b.addr and 
     add[b, b', n, a] and delete[b', b'', n]) 
    implies b.addr = b''.addr
}
check delUndoesAdd for 5
```
- Verifica **tutti i possibili** libri, nomi, indirizzi (entro scope 5)
- Se passa, sappiamo che funziona per **tutte le combinazioni** ≤5 elementi
- Counter-example **sicuramente trovato** se esiste entro lo scope

✅ **Regola Pratica - Quando Usare Cosa**

**Usa Testing quando:**
- Hai **implementazione concreta** da testare
- Vuoi verificare **comportamento runtime**
- Necessiti di **performance** (testing scala bene)
- Casi di test sono **facilmente identificabili**

**Usa Bounded Verification quando:**
- Hai **specifica formale** da validare
- Vuoi **garanzie matematiche** (entro limiti)
- Necessiti di **copertura esaustiva** (piccolo spazio)
- Vuoi **trovare corner cases** nascosti

**Usa ENTRAMBI quando:**
- Progetto critico che richiede alta affidabilità
- Bounded verification sulla specifica + Testing sull'implementazione

---

### 7.4 Vantaggi e Svantaggi

`⏱️ 00:59:31 - 01:02:22`

Riassumiamo i **vantaggi** e gli **svantaggi** della bounded verification.

✅ **Vantaggi della Bounded Verification**

1. **Esplorazione Esaustiva Entro Limiti**
   - Copre **tutti** i casi possibili entro lo scope
   - Non devi pensare a quali casi testare
   - Trova corner cases che potresti non immaginare

2. **Garanzie Matematiche**
   - Se passa, la proprietà **è vera** (entro scope)
   - Non ci sono "forse" o "probabilmente"
   - Certezza matematica nel dominio limitato

3. **Trovamento Garantito di Bug**
   - Se un bug esiste entro lo scope, **lo trova**
   - Non dipende dalla "fortuna" di scegliere il test giusto
   - Counter-examples sono automaticamente minimali

4. **Indipendente dall'Implementazione**
   - Verifica la **specifica**, non il codice
   - Trova problemi di design prima dell'implementazione
   - Validazione early nel ciclo di sviluppo

5. **Documentazione Vivente**
   - Asserzioni documentano proprietà attese
   - Specifiche eseguibili e verificabili
   - Contratto formale del sistema

❌ **Svantaggi della Bounded Verification**

1. **Limitatezza dello Scope**
   - Verifica solo insiemi **finiti** e **piccoli**
   - Bug che emergono con molti elementi sfuggono
   - Non fornisce garanzie "per ogni N"

2. **Complessità Esponenziale**
   - Tempo cresce esponenzialmente con scope
   - Scope grandi diventano impraticabili
   - Limiti pratici su dimensione verificabile

3. **Curva di Apprendimento**
   - Sintassi e semantica di Alloy richiedono studio
   - Pensare in termini di logica relazionale
   - Non intuitivo per chi è abituato al codice imperativo

4. **Astrattezza**
   - Specifica ≠ Implementazione
   - Gap tra modello e codice reale
   - Serve comunque testing dell'implementazione

5. **Performance**
   - Analisi può essere lenta (minuti per scope medio)
   - Non adatto per feedback immediato in TDD
   - Richiede pazienza per verifiche complesse

📊 **Quando la Bounded Verification è Più Efficace**

| Scenario | Efficacia | Motivazione |
|----------|----------|-------------|
| **Protocolli** | ⭐⭐⭐⭐⭐ | Pochi stati, proprietà critiche |
| **Data structures** | ⭐⭐⭐⭐ | Invarianti complessi, corner cases |
| **Business logic** | ⭐⭐⭐ | Regole intricate, precondizioni |
| **UI logic** | ⭐⭐ | Troppi stati, dipendente da input utente |
| **Performance** | ⭐ | Non valuta tempi di esecuzione |

💡 **Insight - Complementarità**

Bounded verification e testing **non si escludono**, ma si **complementano**:

**Workflow ideale:**
1. **Specifica** in Alloy
2. **Verifica bounded** con `check`
3. **Implementazione** in linguaggio di programmazione
4. **Unit testing** dell'implementazione
5. **Integration testing** del sistema completo

Ogni fase cattura diversi tipi di errori!

✅ **Regola Pratica - Best Practices**

**Per massimizzare l'efficacia:**
1. Inizia con **scope piccolo** (3) per feedback rapido
2. Aumenta **gradualmente** se passa (5, poi 8)
3. Usa bounded verification in **fase di design**
4. Combina con **testing** in fase di implementazione
5. Documenta **assunzioni** sullo scope nei commenti

**Esempio di commento:**
```alloy
// Verificato per scope 8
// Proprietà critica: testare con scope ≥10 in CI/CD
check delUndoesAddFixed for 8
```

---

## 8. Funzioni in Alloy

### 8.1 Definizione di Funzioni

`⏱️ 01:02:22 - 01:03:34`

Alloy permette di definire **funzioni** per rendere le specifiche più modulari e leggibili.

📝 **Sintassi delle Funzioni**

```alloy
fun nomeFunzione[parametri]: TipoRitorno {
  // Corpo della funzione
  espressione
}
```

**Componenti:**
- `fun`: Keyword per definire una funzione
- `nomeFunzione`: Nome identificativo
- `[parametri]`: Lista di parametri con tipi
- `: TipoRitorno`: Tipo del valore restituito
- `{ espressione }`: Corpo che calcola il risultato

💡 **Insight - Funzioni vs Predicati**

| Aspetto | Funzione | Predicato |
|---------|----------|-----------|
| **Keyword** | `fun` | `pred` |
| **Ritorna** | Un **valore** (set, atom, etc.) | **Booleano** (vero/falso) |
| **Uso** | Calcolare espressioni | Definire vincoli |
| **Esempio** | `fun sum[a, b: Int]: Int` | `pred isValid[b: Book]` |

🔍 **Differenze Chiave**

**Funzione:**
```alloy
fun double[x: Int]: Int {
  add[x, x]
}
```
Restituisce un **valore intero** (il doppio di x).

**Predicato:**
```alloy
pred isEven[x: Int] {
  rem[x, 2] = 0
}
```
Restituisce un **booleano** (vero se x è pari).

✅ **Regola Pratica - Quando Usare Funzioni**

**Usa funzioni quando:**
- Devi **calcolare** un valore da usare in espressioni
- Vuoi **riutilizzare** logica di calcolo
- Necessiti di **astrazione** per espressioni complesse

**Esempio pratico:**
```alloy
// Invece di ripetere questa espressione ovunque:
b.addr & (Name -> a)

// Definisci una funzione:
fun namesWithAddr[b: Book, a: Addr]: set Name {
  b.addr.a
}

// E usala nelle asserzioni/predicati
```

---

### 8.2 Esempio: Funzione Lookup

`⏱️ 01:03:34 - 01:04:11`

Vediamo un esempio concreto di funzione utile nell'Address Book.

📝 **Definizione della Funzione Lookup**

```alloy
fun lookup[b: Book, n: Name]: set Addr {
  n.(b.addr)
}
```

**Parametri:**
- `b: Book` - Il libro in cui cercare
- `n: Name` - Il nome da cercare

**Tipo di ritorno:**
- `set Addr` - L'insieme di indirizzi associati a `n` in `b`

**Corpo:**
- `n.(b.addr)` - Join tra il nome e la relazione addr del libro

🔍 **Analisi del Comportamento**

**Cosa fa `lookup`:**
1. Prende un libro `b` e un nome `n`
2. Estrae la relazione `b.addr` (tutte le coppie nome→indirizzo del libro)
3. Fa il join con `n` (filtra solo le coppie che iniziano con `n`)
4. Restituisce gli indirizzi (secondi elementi delle coppie)

📊 **Esempio Concreto**

```alloy
// Libro con contenuto:
Book0.addr = {(N0, A1), (N1, A0), (N2, A1)}

// Chiamate alla funzione:
lookup[Book0, N0] = {A1}       // N0 → A1
lookup[Book0, N1] = {A0}       // N1 → A0
lookup[Book0, N2] = {A1}       // N2 → A1
lookup[Book0, N3] = {}         // N3 non nel libro
```

💡 **Insight - Vincolo Lone**

Con il vincolo `lone` (`addr: Name -> lone Addr`):
- `lookup` restituisce sempre un insieme con **0 o 1 elementi**
- Mai più di un indirizzo per nome
- Quindi `set Addr` ha cardinalità ≤ 1

**Possibili risultati:**
- `{}` - Nome non presente
- `{A0}` - Nome presente con un indirizzo

⚠️ **Attenzione - Set vs Singleton**

Anche se c'è al massimo 1 elemento, il tipo è `set Addr`, non `Addr`:
```alloy
// CORRETTO
fun lookup[b: Book, n: Name]: set Addr { ... }

// SBAGLIATO (se il nome non c'è, non possiamo restituire un singolo Addr)
fun lookup[b: Book, n: Name]: Addr { ... }
```

📝 **Nota - Alternativa con "lone"**

Potremmo anche dichiarare:
```alloy
fun lookup[b: Book, n: Name]: lone Addr {
  n.(b.addr)
}
```

Questo dice esplicitamente "0 o 1 indirizzo", ma `set Addr` è più generale.

---

### 8.3 Uso delle Funzioni nelle Asserzioni

`⏱️ 01:04:11 - 01:04:48`

Le funzioni rendono le asserzioni più **leggibili** e **manutenibili**.

📝 **Asserzione con Funzione Lookup**

```alloy
assert addLocal {
  all b, b': Book, n1, n2: Name, a: Addr |
    (add[b, b', n1, a] and n1 != n2) 
    implies 
    lookup[b, n2] = lookup[b', n2]
}
```

**Interpretazione:**
> "L'operazione `add` è **locale**: aggiungere `(n1, a)` non modifica gli indirizzi di altri nomi."

🔍 **Analisi Dettagliata**

**Cosa dice l'asserzione:**
1. Partiamo da libro `b`
2. Aggiungiamo `(n1, a)` ottenendo `b'`
3. Consideriamo un **altro nome** `n2` (diverso da `n1`)
4. Gli indirizzi di `n2` devono essere **uguali** in `b` e `b'`

**In altre parole:**
- Modificare `n1` non deve influenzare `n2`
- Proprietà di **località** (local operation)

💡 **Insight - Vantaggi dell'Uso di Funzioni**

**Senza funzione (meno leggibile):**
```alloy
assert addLocal {
  all b, b': Book, n1, n2: Name, a: Addr |
    (add[b, b', n1, a] and n1 != n2) 
    implies 
    n2.(b.addr) = n2.(b'.addr)  // ← Espressione ripetuta e oscura
}
```

**Con funzione (più leggibile):**
```alloy
assert addLocal {
  all b, b': Book, n1, n2: Name, a: Addr |
    (add[b, b', n1, a] and n1 != n2) 
    implies 
    lookup[b, n2] = lookup[b', n2]  // ← Intento chiaro!
}
```

✅ **Vantaggi:**
1. **Leggibilità**: `lookup[b, n2]` è più chiaro di `n2.(b.addr)`
2. **Riuso**: Stessa funzione usabile in più asserzioni/predicati
3. **Manutenibilità**: Cambiare l'implementazione in un solo posto
4. **Astrazione**: Nasconde dettagli implementativi (join operation)

📊 **Esempio di Verifica**

```alloy
check addLocal for 5
// Verifica che add non modifichi altri nomi
// entro scope di 5 elementi per tipo
```

**Risultato atteso:**
```
No counterexample found.
```

L'asserzione dovrebbe passare: aggiungere un nome non influenza altri nomi! ✅

🔍 **Scenario di Test**

```
Libro iniziale (b):
  addr = {(N0, A0), (N1, A1)}

Operazione: add[b, b', N2, A2]  // Aggiungiamo N2

Libro risultante (b'):
  addr = {(N0, A0), (N1, A1), (N2, A2)}

Verifica per N0 (n2 = N0):
  lookup[b, N0] = {A0}
  lookup[b', N0] = {A0}
  Uguali! ✅

Verifica per N1 (n2 = N1):
  lookup[b, N1] = {A1}
  lookup[b', N1] = {A1}
  Uguali! ✅

Conclusione: add è locale ✅
```

✅ **Regola Pratica - Best Practices con Funzioni**

1. **Nomina chiaramente**: `lookup`, `getAddresses`, `findNames`
2. **Documenta**: Aggiungi commenti su cosa restituisce
3. **Riusa**: Usa la stessa funzione in predicati e asserzioni
4. **Testa**: Verifica che le funzioni calcolino correttamente

**Esempio di funzione ben documentata:**
```alloy
// Restituisce l'insieme di indirizzi associati al nome n nel libro b.
// Se n non è presente nel libro, restituisce l'insieme vuoto.
// Con il vincolo 'lone', l'insieme ha cardinalità 0 o 1.
fun lookup[b: Book, n: Name]: set Addr {
  n.(b.addr)
}
```

---

## 🎯 Riepilogo Parte 2

In questa seconda parte della lezione abbiamo approfondito:

✅ **Asserzioni (assert):**
- Verificano **proprietà globali** della specifica
- Usano il comando `check` invece di `run`
- Generano **counter-examples** se la proprietà è falsa
- Esempio: `delUndoesAdd` per verificare che delete annulla add

✅ **Counter-examples:**
- Mostrano **situazioni specifiche** dove l'asserzione fallisce
- Rivelano **corner cases** e **bug** nella specifica
- Permettono di **iterare** e migliorare il modello
- Esempio: scoperto che add idempotente causa fallimento

✅ **Correzione di Asserzioni:**
- Aggiungere **precondizioni** esplicite
- Usare **implicazione** (`implies`) invece di `and`
- Esempio: `n -> a not in b.addr` prima di add

✅ **Quantificazione Universale:**
- `all x: T | formula` verifica per **ogni** elemento
- L'Analyzer esplora **tutte le combinazioni**
- Attenzione all'**esplosione combinatoria**

✅ **Bounded Verification:**
- Analisi **limitata** a insiemi finiti (scope)
- **Esplorazione esaustiva** entro i limiti
- Garanzie **matematiche** nel dominio bounded
- Complementare al **testing tradizionale**

✅ **Funzioni:**
- Calcolano **valori** (non booleani)
- Sintassi: `fun nome[params]: TipoRitorno { expr }`
- Esempio: `lookup[b, n]` per ottenere indirizzi
- Migliorano **leggibilità** e **riuso**

---

### 📚 Concetti Chiave da Ricordare

1. **Asserzioni ≠ Predicati**: Asserzioni verificano proprietà, predicati generano mondi
2. **Counter-examples**: Strumento fondamentale per debug e raffinamento
3. **Implicazione**: Usare `implies`, non `and`, per precondizione→conclusione
4. **Bounded**: Verifica completa entro scope, ma limitata a insiemi finiti
5. **Funzioni**: Astraggono calcoli, rendono specifiche più leggibili

---

### 🔜 Prossimi Argomenti (Parte 3)

Nella prossima parte finale vedremo:
- **Facts**: Vincoli che devono sempre valere
- **Family Tree**: Esempio complesso con gerarchie
- **Chiusura Transitiva**: Operatore `^` per relazioni transitive
- **Specializzazione**: Signature astratte ed estensioni
- **Riepilogo**: Struttura completa di un documento Alloy

---

*Fine Parte 2 - Continua nella Parte 3...*

---
---

# 📘 Lezione 7 - Alloy: Operazioni e Asserzioni (Parte 3)

**Corso:** Ingegneria del Software  
**Data:** 8 Ottobre  
**Argomento:** Riepilogo Alloy, Family Tree, Facts, Chiusura Transitiva  
**Durata Parte 3:** 01:02:57 - 01:30:00 (~27 minuti)

---

## 📑 Indice dei Contenuti - Parte 3

### [9. Riepilogo: Struttura di un Documento Alloy](#9-riepilogo-struttura-di-un-documento-alloy) `01:02:57 - 01:04:48`
   - [9.1 Componenti di una Specifica Alloy](#91-componenti-di-una-specifica-alloy) `01:02:57 - 01:03:34`
   - [9.2 Comandi Run e Check](#92-comandi-run-e-check) `01:03:34 - 01:04:11`
   - [9.3 Bounded Analysis Recap](#93-bounded-analysis-recap) `01:04:11 - 01:04:48`

### [10. Esempio: Family Tree](#10-esempio-family-tree) `01:04:48 - 01:15:44`
   - [10.1 Definizione delle Signature](#101-definizione-delle-signature) `01:04:48 - 01:08:19`
   - [10.2 Signature Astratte ed Estensioni](#102-signature-astratte-ed-estensioni) `01:08:19 - 01:10:00`
   - [10.3 Esecuzione Iniziale e Mondi Caotici](#103-esecuzione-iniziale-e-mondi-caotici) `01:10:00 - 01:11:40`
   - [10.4 Predicato "Essere il Proprio Nonno"](#104-predicato-essere-il-proprio-nonno) `01:11:40 - 01:15:44`

### [11. Facts: Vincoli Globali](#11-facts-vincoli-globali) `01:15:44 - 01:19:08`
   - [11.1 Introduzione ai Facts](#111-introduzione-ai-facts) `01:15:44 - 01:17:31`
   - [11.2 Facts vs Predicati](#112-facts-vs-predicati) `01:17:31 - 01:18:07`
   - [11.3 Esempio: Nessun Antenato di Se Stesso](#113-esempio-nessun-antenato-di-se-stesso) `01:18:07 - 01:19:08`

### [12. Chiusura Transitiva](#12-chiusura-transitiva) `01:19:08 - 01:27:31`
   - [12.1 Operatore di Chiusura Transitiva (^)](#121-operatore-di-chiusura-transitiva-) `01:19:08 - 01:20:14`
   - [12.2 Definizione Formale](#122-definizione-formale) `01:20:14 - 01:20:44`
   - [12.3 Esempio: Lista Concatenata](#123-esempio-lista-concatenata) `01:20:44 - 01:25:27`
   - [12.4 Chiusura Riflessiva Transitiva (*)](#124-chiusura-riflessiva-transitiva-) `01:25:27 - 01:26:00`
   - [12.5 Applicazione al Family Tree](#125-applicazione-al-family-tree) `01:26:00 - 01:27:31`

### [13. Facts Avanzati](#13-facts-avanzati) `01:27:31 - 01:29:59`
   - [13.1 Vincolo Simmetrico: Wife e Husband](#131-vincolo-simmetrico-wife-e-husband) `01:27:31 - 01:28:18`
   - [13.2 Stile Logico vs Stile Set Theory](#132-stile-logico-vs-stile-set-theory) `01:28:18 - 01:29:59`

### [14. Conclusioni e Prospettive](#14-conclusioni-e-prospettive) `01:29:59 - 01:30:00`

---

## 9. Riepilogo: Struttura di un Documento Alloy

### 9.1 Componenti di una Specifica Alloy

`⏱️ 01:02:57 - 01:03:34`

Prima di procedere con esempi più complessi, ricapitoliamo la **struttura generale** di un documento Alloy.

📝 **Componenti Principali**

Un documento Alloy può contenere:

1. **Signatures** (`sig`)
2. **Facts** (`fact`)
3. **Predicates** (`pred`)
4. **Functions** (`fun`)
5. **Assertions** (`assert`)
6. **Comments** (commenti)

💡 **Insight - Organizzazione del Codice**

```alloy
// 1. SIGNATURES - Definizione dei tipi
sig Type1 { ... }
sig Type2 { ... }

// 2. FACTS - Vincoli che devono sempre valere
fact FactName { ... }

// 3. PREDICATES - Vincoli parametrizzati
pred predName[params] { ... }

// 4. FUNCTIONS - Calcoli riutilizzabili
fun funName[params]: ReturnType { ... }

// 5. ASSERTIONS - Proprietà da verificare
assert assertName { ... }

// 6. COMMANDS - Run e Check
run predName for N
check assertName for N
```

📊 **Tabella: Componenti di Alloy**

| Componente | Keyword | Scopo | Eseguibile |
|-----------|---------|-------|-----------|
| **Signature** | `sig` | Definire tipi e relazioni | No |
| **Fact** | `fact` | Vincoli sempre veri | No (sempre applicati) |
| **Predicate** | `pred` | Vincoli parametrizzati | Sì (con `run`) |
| **Function** | `fun` | Calcolare valori | No (usata in altri costrutti) |
| **Assertion** | `assert` | Proprietà da verificare | Sì (con `check`) |
| **Comment** | `//` o `/* */` | Documentazione | No |

✅ **Regola Pratica - Ordine Consigliato**

**Organizzazione raccomandata:**
1. Commenti introduttivi (descrizione della specifica)
2. Signatures (tipi base del dominio)
3. Facts (invarianti globali)
4. Functions (utilità riutilizzabili)
5. Predicates (scenari e operazioni)
6. Assertions (proprietà da verificare)
7. Commands (run/check per testing)

---

### 9.2 Comandi Run e Check

`⏱️ 01:03:34 - 01:04:11`

I comandi `run` e `check` sono il modo per **interagire** con l'Alloy Analyzer.

📝 **Sintassi dei Comandi**

**Run - Esegue un predicato:**
```alloy
run predicateName
run predicateName for N
run predicateName for N but M Signature
run { formula } for N
```

**Check - Verifica un'asserzione:**
```alloy
check assertionName
check assertionName for N
check assertionName for N but M Signature
```

💡 **Insight - Differenze Run vs Check**

| Aspetto | `run` | `check` |
|---------|-------|---------|
| **Input** | Predicato | Asserzione |
| **Obiettivo** | Trovare mondi | Trovare counter-examples |
| **Output (successo)** | Mostra mondo(i) | "No counterexample found" |
| **Output (fallimento)** | "No instance found" | Mostra counter-example |
| **Interpretazione** | "Esiste un mondo così?" | "La proprietà vale sempre?" |

🔍 **Esempi Pratici**

```alloy
// Genera mondi dove il libro ha almeno 2 elementi
run { some b: Book | #b.addr > 1 } for 3

// Verifica che delete annulla add (con precondizione)
check delUndoesAddFixed for 5

// Esplora la specifica senza vincoli aggiuntivi
run {} for 3

// Verifica con scope personalizzato
check addLocal for 3 but 5 Book
```

✅ **Regola Pratica - Workflow di Sviluppo**

**Ciclo tipico:**
1. **Definisci** signatures e facts
2. **Esplora** con `run {}` per vedere mondi base
3. **Definisci** predicati per scenari specifici
4. **Testa** predicati con `run`
5. **Scrivi** asserzioni per proprietà attese
6. **Verifica** asserzioni con `check`
7. **Itera** basandosi sui counter-examples

---

### 9.3 Bounded Analysis Recap

`⏱️ 01:04:11 - 01:04:48`

Ricordiamo brevemente i concetti chiave dell'analisi bounded.

💡 **Insight - Bounded Analysis**

L'Alloy Analyzer esegue una **bounded analysis** (analisi limitata):
- Considera solo **insiemi finiti** di atomi
- Il numero è limitato dallo **scope** (`for N`)
- Non può verificare proprietà su insiemi **infiniti**

📊 **Limitazioni e Garanzie**

| Aspetto | Bounded Analysis |
|---------|-----------------|
| **Scope** | Fissato (es. `for 5`) |
| **Completezza** | Solo entro i limiti |
| **Garanzia** | "Vero per ≤N elementi" |
| **Decidibilità** | Sempre termina |
| **Performance** | Esponenziale con N |

⚠️ **Attenzione - Interpretazione dei Risultati**

**Quando `check` non trova counter-examples:**
- ✅ La proprietà vale per tutti i mondi con ≤N elementi
- ❌ NON garantisce che valga per N+1, N+2, ... elementi

**Best practice:** Testare con scope crescenti (3, 5, 8, 10) per aumentare la confidenza.

---

## 10. Esempio: Family Tree

### 10.1 Definizione delle Signature

`⏱️ 01:04:48 - 01:08:19`

Passiamo a un esempio più complesso: modellare un **albero genealogico** (family tree).

📝 **Specifica Iniziale**

```alloy
abstract sig Person {
  father: lone Man,
  mother: lone Woman
}

sig Man extends Person {
  wife: lone Woman
}

sig Woman extends Person {
  husband: lone Man
}
```

🔍 **Analisi Dettagliata delle Signature**

**1. Signature Astratta `Person`:**
```alloy
abstract sig Person { ... }
```
- `abstract`: Non possono esistere istanze dirette di `Person`
- Ogni persona deve essere o `Man` o `Woman`
- Come le classi astratte in OOP

**2. Relazioni Familiari:**
```alloy
father: lone Man
mother: lone Woman
```
- Ogni persona ha **al più un** padre (tipo `Man`)
- Ogni persona ha **al più una** madre (tipo `Woman`)
- `lone` = 0 o 1 (opzionale)

**3. Estensioni:**
```alloy
sig Man extends Person { wife: lone Woman }
sig Woman extends Person { husband: lone Man }
```
- `Man` **estende** `Person` (eredita `father`, `mother`)
- Aggiunge relazione `wife` (opzionale)
- `Woman` similmente estende e aggiunge `husband`

💡 **Insight - Gerarchia di Tipi**

```
         Person (abstract)
         /           \
       Man          Woman
     (+ wife)     (+ husband)
```

Ogni istanza sarà **o** `Man` **o** `Woman`, mai `Person` diretto.

📊 **Tabella: Molteplicità nelle Relazioni**

| Relazione | Tipo | Molteplicità | Significato |
|-----------|------|-------------|-------------|
| `father` | `lone Man` | 0..1 | Padre opzionale |
| `mother` | `lone Woman` | 0..1 | Madre opzionale |
| `wife` | `lone Woman` | 0..1 | Moglie opzionale |
| `husband` | `lone Man` | 0..1 | Marito opzionale |

⚠️ **Attenzione - Assenza di Vincoli**

Questa specifica **non impone** alcun vincolo di sensatezza:
- Una persona può essere padre/madre di se stessa
- Relazioni cicliche sono permesse
- Moglie e marito possono non corrispondere

Vedremo come aggiungere vincoli con i **facts**!

---

### 10.2 Signature Astratte ed Estensioni

`⏱️ 01:08:19 - 01:10:00`

Approfondiamo il concetto di **signature astratte** e **estensioni**.

💡 **Insight - Abstract Signatures**

Una signature **astratta** (`abstract sig`) è come una classe astratta in OOP:
- **Non può avere istanze dirette**
- Deve avere **sottotipi concreti**
- Serve per **condividere** attributi comuni

📝 **Sintassi delle Estensioni**

```alloy
abstract sig SuperType { ... }

sig SubType1 extends SuperType { ... }
sig SubType2 extends SuperType { ... }
```

**Semantica:**
- `SubType1` **eredita** tutti i campi di `SuperType`
- `SubType1` può **aggiungere** nuovi campi propri
- Ogni istanza di `SubType1` è anche di tipo `SuperType`

🔍 **Esempio: Person, Man, Woman**

```alloy
abstract sig Person {
  father: lone Man,    // Ereditato da Man e Woman
  mother: lone Woman   // Ereditato da Man e Woman
}

sig Man extends Person {
  wife: lone Woman     // Solo per Man
}

sig Woman extends Person {
  husband: lone Man    // Solo per Woman
}
```

**Risultato:**
- Ogni `Man` ha: `father`, `mother`, `wife`
- Ogni `Woman` ha: `father`, `mother`, `husband`
- Non possono esistere `Person` che non siano né `Man` né `Woman`

📊 **Tabella: Confronto con OOP**

| Alloy | Java/C++ | Semantica |
|-------|----------|-----------|
| `abstract sig` | `abstract class` | Non istanziabile |
| `sig X extends Y` | `class X extends Y` | Ereditarietà |
| Campi in sig | Attributi in classe | Proprietà |
| `lone`, `one`, etc. | Molteplicità | Vincoli di cardinalità |

✅ **Regola Pratica - Quando Usare Abstract**

**Usa `abstract sig` quando:**
- Vuoi condividere attributi tra tipi
- Nessuna istanza dovrebbe essere "solo" del supertipo
- Esempio: `Person` (deve essere `Man` o `Woman`)

**Usa `sig` normale quando:**
- Il tipo può avere istanze dirette
- Non è necessaria specializzazione
- Esempio: `Book`, `Name`, `Addr`

---

### 10.3 Esecuzione Iniziale e Mondi Caotici

`⏱️ 01:10:00 - 01:11:40`

Eseguiamo la specifica del Family Tree senza vincoli aggiuntivi.

📝 **Predicato Show di Default**

```alloy
pred show {}

run show
```

Oppure semplicemente:
```alloy
run {}
```

**Cosa succede:** L'Analyzer genera mondi che soddisfano **solo** le signature, senza vincoli logici.

🔍 **Mondi Generati**

**Mondo 1 - Vuoto:**
```
// Nessuna persona
```

**Mondo 2 - Una Donna:**
```
Woman0
  father = none
  mother = none
  husband = none
```

**Mondo 3 - Relazioni Caotiche:**
```
Woman0
  father = none
  mother = Woman0  // ← Madre di se stessa! ⚠️
  husband = none
```

**Mondo 4 - Ancora Più Caotico:**
```
Man0
  father = Man0     // ← Padre di se stesso! ⚠️
  mother = Woman0
  wife = Woman0

Woman0
  father = Man0
  mother = Woman0   // ← Madre di se stessa! ⚠️
  husband = Man0
```

⚠️ **Attenzione - Mondi Insensati**

Questi mondi sono **validi** secondo la specifica attuale, ma **insensati** dal punto di vista del dominio:
- Persone genitori di se stesse
- Relazioni cicliche
- Wife/husband non simmetrici

💡 **Insight - La Specifica è Troppo Permissiva**

Ciò che è **ovvio per noi** (una persona non può essere suo padre) **non è ovvio per Alloy**!

Dobbiamo **esplicitare** tutti i vincoli di sensatezza.

✅ **Soluzione - Aggiungere Facts**

Useremo i **facts** per escludere mondi insensati:
- Nessuno è antenato di se stesso
- Wife e husband devono corrispondere
- ... altri vincoli di coerenza

---

### 10.4 Predicato "Essere il Proprio Nonno"

`⏱️ 01:11:40 - 01:15:44`

Prima di aggiungere vincoli, esploriamo una situazione curiosa: **può qualcuno essere il proprio nonno?**

📝 **Definizione della Funzione Grandpa**

```alloy
fun grandpa[p: Person]: set Person {
  p.(mother + father).(mother + father)
}
```

🔍 **Analisi Passo-Passo**

**Cosa calcola `grandpa[p]`?**

1. **`mother + father`**: Unione delle relazioni madre e padre
   - Risultato: Relazione che contiene **tutti i genitori**

2. **`p.(mother + father)`**: Join di `p` con i genitori
   - Risultato: **I genitori di `p`** (padre e/o madre)

3. **`.(mother + father)` di nuovo**: Join dei genitori con i loro genitori
   - Risultato: **I nonni di `p`** (nonni paterni e materni)

💡 **Insight - Composizione di Relazioni**

```
p → (mother + father) → (mother + father)
│         │                    │
│         └──> Genitori di p   └──> Genitori dei genitori di p
└──> Persona p                         (= Nonni di p)
```

📊 **Esempio Concreto**

```
Person0
  father = Man1
  mother = Woman1

Man1 (padre di Person0)
  father = Man2    // Nonno paterno
  mother = Woman2

Woman1 (madre di Person0)
  father = Man3    // Nonno materno
  mother = Woman3

Risultato: grandpa[Person0] = {Man2, Man3}
```

📝 **Predicato: Essere il Proprio Nonno**

```alloy
pred isOwnGrandpa[p: Person] {
  p in grandpa[p]
}
```

Oppure sintassi alternativa:
```alloy
pred isOwnGrandpa[p: Person] {
  p in p.(mother + father).(mother + father)
}
```

**Interpretazione:**
> "La persona `p` è nell'insieme dei propri nonni"

🔍 **Esecuzione del Predicato**

```alloy
run isOwnGrandpa
```

**Domanda:** Troverà un'istanza?

**Risposta:** **Sì!** Senza vincoli, è possibile costruire cicli dove qualcuno è il proprio nonno.

**Esempio di mondo trovato:**

```
Man0
  father = Man1
  mother = Woman0
  wife = Woman0

Man1
  father = Man0  // ← Man0 è nonno di se stesso!
  mother = Woman0
  wife = none

Woman0
  father = Man1
  mother = Woman0  // ← Ciclo!
  husband = Man0
```

**Verifica:**
- `Man0.father = Man1`
- `Man1.father = Man0`
- Quindi `Man0` → `Man1` → `Man0` (ciclo di 2 generazioni)

💡 **Insight - Perché è Possibile?**

Senza vincoli sui cicli, Alloy può creare **strutture circolari**:
- A è padre di B
- B è padre di A
- Quindi A è nonno di A

Questo è **matematicamente valido** ma **semanticamente insensato**!

⚠️ **Attenzione - Joke del Professore**

Il professore menziona che ci sono situazioni reali (combinazioni familiari complesse con matrimoni) dove qualcuno potrebbe tecnicamente diventare il proprio nonno, ma sono casi estremi e inusuali.

Nel nostro modello, vogliamo **impedire** questi cicli.

---

## 11. Facts: Vincoli Globali

### 11.1 Introduzione ai Facts

`⏱️ 01:15:44 - 01:17:31`

I **facts** sono vincoli che devono **sempre** essere veri nella specifica.

📝 **Sintassi dei Facts**

```alloy
fact FactName {
  // Formula logica che deve sempre valere
}
```

Oppure fact anonimo:
```alloy
fact {
  // Formula logica
}
```

💡 **Insight - Facts vs Predicati**

| Aspetto | Fact | Predicato |
|---------|------|-----------|
| **Sempre applicato** | ✅ Sì | ❌ No (solo se chiamato) |
| **Parametri** | ❌ No | ✅ Sì |
| **Comando** | - (automatico) | `run` |
| **Scopo** | Vincoli globali | Scenari specifici |
| **Esempio** | Invarianti | Operazioni |

🔍 **Differenza Fondamentale**

**Predicato:**
```alloy
pred noCycles[p: Person] {
  p not in p.^(mother + father)
}

run noCycles  // Applicato solo quando eseguito
```

**Fact:**
```alloy
fact noCycles {
  all p: Person | p not in p.^(mother + father)
}

// Sempre applicato, in ogni run/check!
```

✅ **Regola Pratica - Quando Usare Facts**

**Usa Facts per:**
- **Invarianti** che devono sempre valere
- **Vincoli strutturali** del dominio
- **Regole del mondo** che non cambiano mai

**Esempi:**
- Nessuno è antenato di se stesso
- Una moglie deve avere quel marito
- Un libro non può avere duplicati (se desiderato)

**NON usare Facts per:**
- Vincoli che valgono solo in alcuni scenari
- Operazioni parametrizzate
- Condizioni che dipendono dallo stato

---

### 11.2 Facts vs Predicati

`⏱️ 01:17:31 - 01:18:07`

Approfondiamo la distinzione tra facts e predicati.

📊 **Tabella Comparativa Dettagliata**

| Caratteristica | Facts | Predicati |
|----------------|-------|-----------|
| **Keyword** | `fact` | `pred` |
| **Parametri** | No | Sì |
| **Applicazione** | Automatica (sempre) | Esplicita (con `run`) |
| **Scope** | Globale | Locale (al run) |
| **Verifica** | In ogni `run` e `check` | Solo quando chiamato |
| **Uso tipico** | Invarianti di dominio | Operazioni e scenari |
| **Modificabile** | No (parte della spec) | Sì (chiama o non chiama) |

💡 **Insight - I Facts Ristringono lo Spazio dei Mondi**

**Senza facts:**
```
Spazio dei mondi possibili: ██████████████████
                           (tutto permesso)
```

**Con facts:**
```
Spazio dei mondi possibili: ████░░░░░░░░░░░░░░
                           ↑        ↑
                        Validi   Esclusi dai facts
```

I facts **riducono** i mondi che l'Analyzer può generare.

🔍 **Esempio Pratico**

**Specifica senza fact:**
```alloy
sig Person {
  age: Int
}

// Nessun vincolo sull'età
run {} // Può generare età negative, > 150, etc.
```

**Specifica con fact:**
```alloy
sig Person {
  age: Int
}

fact validAge {
  all p: Person | p.age >= 0 and p.age <= 150
}

run {} // Solo età 0-150
```

✅ **Regola Pratica - Progettazione**

**Workflow consigliato:**
1. Inizia con **solo signature** (esplora mondi liberi)
2. Identifica mondi **insensati** o **indesiderati**
3. Aggiungi **facts** per escluderli
4. Verifica con `run` che i mondi siano ora sensati
5. Scrivi **predicati** per scenari specifici
6. Verifica **asserzioni** sulle proprietà

---

### 11.3 Esempio: Nessun Antenato di Se Stesso

`⏱️ 01:18:07 - 01:19:08`

Aggiungiamo un fact per impedire che qualcuno sia antenato di se stesso.

📝 **Primo Tentativo (Ingenuo)**

```alloy
fact noSelfAncestor {
  all p: Person | 
    p not in p.mother and
    p not in p.father and
    p not in p.mother.mother and
    p not in p.mother.father and
    p not in p.father.mother and
    p not in p.father.father and
    // ... dove ci fermiamo?
}
```

⚠️ **Problema - Incompletezza**

Questo approccio è:
- **Verboso**: Dobbiamo elencare tutti i livelli
- **Incompleto**: Dove ci fermiamo? Bisnonni? Trisavoli?
- **Fragile**: Difficile da mantenere

💡 **Soluzione - Chiusura Transitiva**

Alloy fornisce un operatore speciale: **chiusura transitiva** (`^`)

📝 **Versione Corretta con Chiusura Transitiva**

```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}
```

**Interpretazione:**
> "Per ogni persona `p`, `p` non è nella chiusura transitiva della relazione genitore"

**In altre parole:**
> "Nessuno è antenato di se stesso (a qualsiasi livello)"

✅ **Vantaggi:**
- **Conciso**: Una sola riga
- **Completo**: Copre **tutti** i livelli (genitori, nonni, bisnonni, ...)
- **Elegante**: Esprime l'intento chiaramente

Vedremo nel dettaglio la chiusura transitiva nella prossima sezione!

---

## 12. Chiusura Transitiva

### 12.1 Operatore di Chiusura Transitiva (^)

`⏱️ 01:19:08 - 01:20:14`

La **chiusura transitiva** è un operatore potente per lavorare con relazioni.

📝 **Sintassi**

```alloy
R^  // Chiusura transitiva di R
R*  // Chiusura riflessiva transitiva di R
```

Dove `R` è una relazione (es. `father`, `mother + father`, `next`, etc.)

💡 **Insight - Cosa Significa Chiusura Transitiva**

**Chiusura transitiva di una relazione R:**
> L'insieme di tutte le coppie `(a, c)` tali che esiste un **cammino** da `a` a `c` seguendo R.

**Formalmente:**
```
R^ = R ∪ (R.R) ∪ (R.R.R) ∪ (R.R.R.R) ∪ ...
```

**In parole:**
- R: Coppie raggiungibili in **1 passo**
- R.R: Coppie raggiungibili in **2 passi**
- R.R.R: Coppie raggiungibili in **3 passi**
- ...

📊 **Tabella: Operatori di Chiusura**

| Operatore | Nome | Definizione | Include Riflessività |
|-----------|------|-------------|---------------------|
| `R^` | Chiusura transitiva | R ∪ R.R ∪ R.R.R ∪ ... | ❌ No |
| `R*` | Chiusura riflessiva transitiva | iden ∪ R^ | ✅ Sì |

**Dove `iden`** è la relazione identità: `{(x, x) | x ∈ Dominio}`

🔍 **Differenza Chiave**

**`R^` (transitiva):**
- Cammini di lunghezza **≥ 1**
- **Non** include `(x, x)` a meno che non ci sia un ciclo

**`R*` (riflessiva transitiva):**
- Cammini di lunghezza **≥ 0**
- **Sempre** include `(x, x)` per ogni `x`

---

### 12.2 Definizione Formale

`⏱️ 01:20:14 - 01:20:44`

Vediamo la definizione matematica formale della chiusura transitiva.

📝 **Definizione Matematica**

**Chiusura Transitiva:**
```
R^ = R ∪ (R ∘ R) ∪ (R ∘ R ∘ R) ∪ ...
```

Dove `∘` è l'operatore di **composizione di relazioni**.

**Chiusura Riflessiva Transitiva:**
```
R* = iden ∪ R^
```

Dove `iden` è la **relazione identità**: `{(x, x) | ∀x}`

💡 **Insight - Composizione di Relazioni**

**Composizione `R ∘ S`:**
```
(a, c) ∈ R ∘ S  ⟺  ∃b: (a, b) ∈ R ∧ (b, c) ∈ S
```

**In Alloy:**
```alloy
R.S  // Operatore . è la composizione
```

📊 **Esempio di Espansione**

Consideriamo relazione `parent` (genitore):

```
parent^ = parent ∪ 
          (parent.parent) ∪ 
          (parent.parent.parent) ∪ 
          ...

       = genitori ∪ 
         nonni ∪ 
         bisnonni ∪ 
         ...
```

🔍 **Visualizzazione Grafica**

```
       A
      / \
     B   C
    / \   \
   D   E   F
  /
 G

parent:   {(B,A), (C,A), (D,B), (E,B), (F,C), (G,D)}

parent^:  {(B,A), (C,A), (D,B), (E,B), (F,C), (G,D),  ← 1 passo
           (D,A), (E,A), (F,A),                        ← 2 passi (nonni)
           (G,B),                                       ← 2 passi
           (G,A)}                                       ← 3 passi (bisnonno)
```

✅ **Regola Pratica - Quando Usare ^**

**Usa `^` quando:**
- Vuoi considerare **tutti i passi** di una relazione
- Esempio: Tutti gli antenati (non solo genitori)
- Esempio: Tutti i nodi raggiungibili in un grafo

**Esempio tipico:**
```alloy
// Tutti gli antenati di p
p.^(mother + father)

// Tutti i discendenti di p
^(mother + father).p
```

---

### 12.3 Esempio: Lista Concatenata

`⏱️ 01:20:44 - 01:25:27`

Vediamo un esempio concreto di chiusura transitiva: una **lista concatenata**.

📝 **Modello della Lista**

```alloy
sig Node {
  next: lone Node
}
```

**Significato:**
- Ogni nodo ha **al più un** successore
- Rappresenta una lista (potenzialmente con cicli, senza vincoli aggiuntivi)

🔍 **Esempio Concreto**

```
Lista: N0 → N1 → N2 → N3

Rappresentazione in Alloy:
Node0: next = Node1
Node1: next = Node2
Node2: next = Node3
Node3: next = none
```

**Relazione `next`:**
```
next = {(N0, N1), (N1, N2), (N2, N3)}
```

💡 **Insight - Calcolo della Chiusura Transitiva**

**Passo 1: `next` (1 passo)**
```
{(N0, N1), (N1, N2), (N2, N3)}
```

**Passo 2: `next.next` (2 passi)**

Calcoliamo la composizione:
- `(N0, N1)` e `(N1, N2)` → `(N0, N2)` ✓
- `(N1, N2)` e `(N2, N3)` → `(N1, N3)` ✓

Risultato:
```
{(N0, N2), (N1, N3)}
```

**Passo 3: `next.next.next` (3 passi)**

- `(N0, N2)` e `(N2, N3)` → `(N0, N3)` ✓

Risultato:
```
{(N0, N3)}
```

**Passo 4: `next^` (chiusura completa)**

Unione di tutti i passi:
```
next^ = {(N0, N1), (N1, N2), (N2, N3),  ← 1 passo
         (N0, N2), (N1, N3),            ← 2 passi
         (N0, N3)}                      ← 3 passi
```

📊 **Visualizzazione della Chiusura**

```
Lista:  N0 → N1 → N2 → N3

next^:  N0 ────────────────┐
        │   ─────────┐     │
        │   ↓        ↓     ↓
        N0 → N1 → N2 → N3
             │    ↓    
             └────┘    
```

Tutte le frecce rappresentano coppie in `next^`.

🔍 **Interpretazione Semantica**

**`N0.^next`** = Tutti i nodi **raggiungibili** da N0:
```
N0.^next = {N1, N2, N3}
```

**`^next.N3`** = Tutti i nodi da cui **si può raggiungere** N3:
```
^next.N3 = {N0, N1, N2}
```

💡 **Insight - Utilizzo Pratico**

**Verificare che la lista sia aciclica:**
```alloy
fact acyclic {
  all n: Node | n not in n.^next
}
```

**Verificare che esista un inizio (nodo senza predecessore):**
```alloy
fact hasHead {
  some n: Node | no ^next.n
}
```

**Verificare che esista una fine (nodo senza successore):**
```alloy
fact hasTail {
  some n: Node | no n.next
}
```

---

### 12.4 Chiusura Riflessiva Transitiva (*)

`⏱️ 01:25:27 - 01:26:00`

La **chiusura riflessiva transitiva** include anche la **relazione identità**.

📝 **Definizione**

```alloy
R* = iden + R^
```

**Dove `iden`** è la relazione identità:
```
iden = {(x, x) | ∀x ∈ Dominio}
```

💡 **Insight - Differenza tra ^ e ***

| Operatore | Include (x, x) | Lunghezza Cammino |
|-----------|---------------|-------------------|
| `R^` | No (solo se ciclo) | ≥ 1 |
| `R*` | Sempre | ≥ 0 |

🔍 **Esempio Concreto**

Riprendiamo la lista: `N0 → N1 → N2 → N3`

**`next^` (chiusura transitiva):**
```
{(N0, N1), (N1, N2), (N2, N3),
 (N0, N2), (N1, N3),
 (N0, N3)}
```

**`next*` (chiusura riflessiva transitiva):**
```
{(N0, N0), (N1, N1), (N2, N2), (N3, N3),  ← Identità aggiunta
 (N0, N1), (N1, N2), (N2, N3),
 (N0, N2), (N1, N3),
 (N0, N3)}
```

📊 **Quando Usare * vs ^**

| Situazione | Usa | Motivo |
|------------|-----|--------|
| Antenati **stretti** | `^` | Escludi se stesso |
| Antenati **o se stesso** | `*` | Includi se stesso |
| Raggiungibilità **diretta** | `^` | Solo se c'è un cammino |
| Raggiungibilità **o stesso nodo** | `*` | Ogni nodo raggiunge se stesso |

✅ **Esempio Pratico**

**Verificare che p sia antenato (o uguale) a se stesso:**
```alloy
// Sempre vero con *
all p: Person | p in p.*(mother + father)

// Falso (normalmente) con ^
all p: Person | p in p.^(mother + father)  // Solo se c'è ciclo
```

---

### 12.5 Applicazione al Family Tree

`⏱️ 01:26:00 - 01:27:31`

Applichiamo la chiusura transitiva al nostro esempio del Family Tree.

📝 **Fact: Nessun Antenato di Se Stesso**

```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}
```

🔍 **Analisi Dettagliata**

**Cosa significa:**
> "Per ogni persona `p`, `p` non può essere raggiunto da `p` seguendo la relazione genitore (madre o padre)"

**In altre parole:**
- Nessuna persona è genitore di se stessa
- Nessuna persona è nonno/a di se stessa
- Nessuna persona è bisnonno/a di se stessa
- ... (a qualsiasi livello)

💡 **Insight - Relazione Genitore**

**`(mother + father)`**: Unione delle relazioni
```
mother + father = {(p, m) | m è madre di p} ∪ 
                  {(p, f) | f è padre di p}
```

**Risultato:** Tutte le coppie `(figlio, genitore)`

**`^(mother + father)`**: Chiusura transitiva
```
^(mother + father) = tutti gli antenati
```

**`p.^(mother + father)`**: Tutti gli antenati di `p`

**`p not in p.^(mother + father)`**: `p` non è tra i suoi antenati ✓

📊 **Esempio di Applicazione**

**Senza il fact:**
```
Man0
  father = Man1
  mother = Woman0

Man1
  father = Man0  // ← Ciclo! Man0 è nonno di se stesso

// Questo è VALIDO senza il fact
```

**Con il fact:**
```
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

// Ora il mondo sopra è ESCLUSO
// L'Analyzer non lo genererà mai
```

✅ **Verifica dell'Effetto**

```alloy
run {} for 5

// Prima del fact: possono apparire cicli
// Dopo il fact: nessun ciclo negli antenati
```

🔍 **Esecuzione del Predicato "Proprio Nonno"**

```alloy
pred isOwnGrandpa[p: Person] {
  p in grandpa[p]
}

run isOwnGrandpa

// Senza fact: trova istanze
// Con fact: "No instance found" ✅
```

Il fact ha **eliminato** la possibilità di essere il proprio nonno!

---

## 13. Facts Avanzati

### 13.1 Vincolo Simmetrico: Wife e Husband

`⏱️ 01:27:31 - 01:28:18`

Aggiungiamo un vincolo di coerenza tra le relazioni `wife` e `husband`.

📝 **Problema da Risolvere**

Nella specifica attuale:
```alloy
sig Man extends Person {
  wife: lone Woman
}

sig Woman extends Person {
  husband: lone Man
}
```

**Situazioni incoerenti permesse:**
```
Man0
  wife = Woman0

Woman0
  husband = Man1  // ← Man1, non Man0!
```

Vogliamo garantire: **Se M ha W come moglie, allora W deve avere M come marito.**

💡 **Insight - Simmetria della Relazione Matrimoniale**

**Proprietà desiderata:**
```
M.wife = W  ⟺  W.husband = M
```

Per **ogni** coppia (M, W).

📝 **Fact: Simmetria Wife-Husband (Stile Logico)**

```alloy
fact wifeHusbandSymmetry {
  all m: Man, w: Woman |
    (m.wife = w) iff (w.husband = m)
}
```

**Interpretazione:**
> "Per ogni uomo `m` e donna `w`: `m` ha `w` come moglie **se e solo se** `w` ha `m` come marito"

🔍 **Analisi dell'Operatore IFF**

**`iff` (if and only if)** = equivalenza logica `⟺`

```
A iff B  ≡  (A implies B) and (B implies A)
```

**Nel nostro caso:**
```
(m.wife = w) iff (w.husband = m)

Significa:
1. Se m.wife = w, allora w.husband = m
2. Se w.husband = m, allora m.wife = w
```

✅ **Effetto del Fact**

**Senza il fact:**
```
Man0: wife = Woman0
Woman0: husband = Man1  // ← Incoerente! ❌
```

**Con il fact:**
```
// Questo mondo è ESCLUSO
// Solo mondi coerenti sono permessi:

Man0: wife = Woman0
Woman0: husband = Man0  // ← Coerente! ✅
```

---

### 13.2 Stile Logico vs Stile Set Theory

`⏱️ 01:28:18 - 01:29:59`

Il fact precedente può essere espresso in **due stili** equivalenti.

📝 **Stile 1: Logico (già visto)**

```alloy
fact wifeHusbandSymmetry {
  all m: Man, w: Woman |
    (m.wife = w) iff (w.husband = m)
}
```

**Caratteristiche:**
- Usa quantificatori (`all`)
- Usa operatori logici (`iff`)
- Stile "imperativo"

📝 **Stile 2: Set Theory**

```alloy
fact wifeHusbandSymmetry {
  wife = ~husband
}
```

**Caratteristiche:**
- Usa operatori di insieme
- Usa trasposizione (`~`)
- Stile "dichiarativo"

💡 **Insight - Operatore di Trasposizione (~)**

**Trasposizione `~R`**: Inverte le coppie nella relazione

```
R = {(a, b), (c, d), (e, f)}
~R = {(b, a), (d, c), (f, e)}
```

**Nel nostro caso:**

```
wife = {(Man0, Woman0), (Man1, Woman1)}
~husband = {(m, w) | (w, m) ∈ husband}
```

Se `husband = {(Woman0, Man0), (Woman1, Man1)}`, allora:
```
~husband = {(Man0, Woman0), (Man1, Woman1)}
```

Quindi `wife = ~husband` ✅

🔍 **Confronto dei Due Stili**

| Aspetto | Stile Logico | Stile Set Theory |
|---------|-------------|------------------|
| **Leggibilità** | Più esplicito | Più conciso |
| **Espressività** | Caso per caso | Relazione globale |
| **Lunghezza** | Verboso | Compatto |
| **Preferenza** | Principianti | Esperti |

📊 **Esempio Comparativo**

**Proprietà:** Ogni persona ha al più un coniuge

**Stile logico:**
```alloy
fact oneSpouse {
  all m: Man | lone m.wife
  all w: Woman | lone w.husband
}
```

**Stile set theory:**
```alloy
fact oneSpouse {
  wife in Man lone -> Woman
  husband in Woman lone -> Man
}
```

💡 **Insight - Equivalenza**

I due stili sono **semanticamente equivalenti**:
- Esprimono la stessa proprietà
- Producono gli stessi vincoli
- L'Analyzer li tratta identicamente

✅ **Regola Pratica - Quale Usare?**

**Usa stile logico quando:**
- Sei alle prime armi con Alloy
- La proprietà è più chiara espressa "caso per caso"
- Vuoi massimizzare la leggibilità

**Usa stile set theory quando:**
- Hai esperienza con Alloy
- La proprietà è naturalmente una relazione tra insiemi
- Vuoi codice più conciso

**Esempio ideale per set theory:**
```alloy
fact symmetricRelation {
  R = ~R  // R è simmetrica
}
```

**Esempio ideale per stile logico:**
```alloy
fact complexConstraint {
  all x: X, y: Y, z: Z |
    (p[x, y] and q[y, z]) implies r[x, z]
}
```

---

## 14. Conclusioni e Prospettive

`⏱️ 01:29:59 - 01:30:00`

Abbiamo completato un'introduzione completa ad Alloy!

🎯 **Riepilogo Generale**

In questa lezione (Parti 1, 2, 3) abbiamo coperto:

✅ **Fondamenti:**
- Signature, relazioni, molteplicità
- Operatori: join (.), unione (+), differenza (-), prodotto cartesiano (->)
- Predicati, funzioni, asserzioni, facts

✅ **Operazioni:**
- Add e Delete nell'Address Book
- Corner cases e semantica idempotente
- Precondizioni e postcondizioni

✅ **Verifica:**
- Asserzioni per validare proprietà
- Counter-examples per debug
- Bounded verification e i suoi limiti

✅ **Concetti Avanzati:**
- Chiusura transitiva (`^`) e riflessiva transitiva (`*`)
- Facts per vincoli globali
- Signature astratte e gerarchie
- Stili di specifica (logico vs set theory)

📊 **Struttura Completa di un Documento Alloy**

```alloy
// ============================================
// MODULO E IMPORTS
// ============================================
module FamilyTree

// ============================================
// SIGNATURES - Tipi e Relazioni
// ============================================
abstract sig Person {
  father: lone Man,
  mother: lone Woman
}

sig Man extends Person {
  wife: lone Woman
}

sig Woman extends Person {
  husband: lone Man
}

// ============================================
// FACTS - Vincoli Globali
// ============================================
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

fact wifeHusbandSymmetry {
  wife = ~husband
}

// ============================================
// FUNCTIONS - Calcoli Riutilizzabili
// ============================================
fun ancestors[p: Person]: set Person {
  p.^(mother + father)
}

fun grandparents[p: Person]: set Person {
  p.(mother + father).(mother + father)
}

// ============================================
// PREDICATES - Scenari e Operazioni
// ============================================
pred show {}

pred marriedCouple[m: Man, w: Woman] {
  m.wife = w
  w.husband = m
}

// ============================================
// ASSERTIONS - Proprietà da Verificare
// ============================================
assert noOneIsTheirOwnGrandparent {
  all p: Person | p not in grandparents[p]
}

assert marriageIsSymmetric {
  all m: Man, w: Woman |
    (m.wife = w) implies (w.husband = m)
}

// ============================================
// COMMANDS - Run e Check
// ============================================
run show for 5

check noOneIsTheirOwnGrandparent for 5
check marriageIsSymmetric for 5
```

💡 **Insight - Best Practices Finali**

1. **Inizia Semplice**: Definisci signature base, poi aggiungi vincoli
2. **Esplora Liberamente**: Usa `run {}` per vedere cosa permette la specifica
3. **Identifica Problemi**: Cerca mondi insensati
4. **Aggiungi Vincoli**: Usa facts per escludere mondi indesiderati
5. **Verifica Proprietà**: Scrivi asserzioni per proprietà attese
6. **Itera**: Usa counter-examples per raffinare la specifica
7. **Documenta**: Commenta l'intento di facts e asserzioni
8. **Testa con Scope Crescenti**: 3 → 5 → 8 per aumentare confidenza

✅ **Quando Usare Alloy**

**Alloy è eccellente per:**
- 🎯 Validare **modelli concettuali** prima dell'implementazione
- 🔍 Trovare **corner cases** e bug di design
- 📋 **Documentare** specifiche formali
- 🔬 Esplorare **proprietà** di sistemi complessi
- 🏗️ Progettare **data structures** e **protocolli**

**Alloy NON sostituisce:**
- ❌ Testing dell'implementazione
- ❌ Verifica di performance
- ❌ Analisi di sicurezza completa
- ❌ Proof formali (usa theorem provers per quello)

📚 **Concetti Chiave da Ricordare**

1. **Signatures**: Definiscono tipi e strutture
2. **Facts**: Vincoli che valgono sempre
3. **Predicates**: Vincoli parametrizzati per scenari
4. **Functions**: Calcoli che restituiscono valori
5. **Assertions**: Proprietà da verificare
6. **Chiusura Transitiva**: `^` per relazioni multi-livello
7. **Bounded Analysis**: Verifica completa entro limiti finiti
8. **Counter-examples**: Guida iterativa al miglioramento

🔜 **Prossimi Passi**

Per approfondire Alloy:
1. **Pratica**: Modella domini familiari (es. sistema universitario, biblioteca)
2. **Esplora**: Leggi specifiche di esempio nella documentazione Alloy
3. **Sperimenta**: Prova varianti di vincoli e osserva l'effetto
4. **Progetta**: Usa Alloy in progetti reali per validare design
5. **Community**: Partecipa a forum e discuti specifiche con altri

📖 **Risorse Aggiuntive**

- **Alloy Documentation**: http://alloytools.org
- **Software Abstractions (libro)**: Daniel Jackson
- **Tutorial**: http://alloytools.org/tutorials.html
- **Examples**: Repository di specifiche di esempio

---

## 🎯 Riepilogo Completo delle 3 Parti

### Parte 1: Fondamenti
- ✅ Address Book: signature, relazioni, join operator
- ✅ Predicati con vincoli di cardinalità
- ✅ Operazioni Add e Delete
- ✅ Corner cases e semantica idempotente

### Parte 2: Verifica
- ✅ Asserzioni e counter-examples
- ✅ Correzione con precondizioni
- ✅ Implicazione vs AND
- ✅ Bounded verification
- ✅ Funzioni per riutilizzo

### Parte 3: Concetti Avanzati
- ✅ Family Tree con gerarchie
- ✅ Facts per invarianti globali
- ✅ Chiusura transitiva (^ e *)
- ✅ Signature astratte
- ✅ Stili di specifica

---

## 📚 Cheat Sheet Alloy

### Molteplicità
- `lone`: 0 o 1
- `one`: esattamente 1
- `some`: ≥ 1
- `set`: qualsiasi numero (default)

### Operatori Relazionali
- `.`: Join (composizione)
- `+`: Unione
- `-`: Differenza
- `&`: Intersezione
- `->`: Prodotto cartesiano
- `~`: Trasposizione
- `^`: Chiusura transitiva
- `*`: Chiusura riflessiva transitiva

### Quantificatori
- `all x: T | formula`: Per ogni x
- `some x: T | formula`: Esiste almeno un x
- `no x: T | formula`: Non esiste x
- `lone x: T | formula`: Esiste al più un x
- `one x: T | formula`: Esiste esattamente un x

### Operatori Logici
- `and`: Congiunzione (∧)
- `or`: Disgiunzione (∨)
- `not`: Negazione (¬)
- `implies`: Implicazione (⟹)
- `iff`: Equivalenza (⟺)

### Cardinalità
- `#expr`: Numero di elementi in expr

### Costrutti Principali
- `sig Name { ... }`: Definire signature
- `abstract sig`: Signature astratta
- `sig A extends B`: Estensione
- `fact Name { ... }`: Vincolo globale
- `pred name[...] { ... }`: Predicato
- `fun name[...]: Type { ... }`: Funzione
- `assert name { ... }`: Asserzione
- `run pred for N`: Eseguire predicato
- `check assert for N`: Verificare asserzione

---

*Fine della Lezione 7 - Alloy: Operazioni e Asserzioni*

---

**Nota:** Questa lezione ha coperto i concetti fondamentali e avanzati di Alloy per la specifica e verifica di sistemi software. La pratica costante con esempi concreti è essenziale per padroneggiare il linguaggio e il suo Analyzer.
