# 📘 Lezione 8 - Alloy: Facts e Vincoli Avanzati

**Corso:** Ingegneria del Software  
**Data:** 9 Ottobre  
**Argomento:** Facts, Constraints, Assertions nel Family Tree  
**Durata:** ~39 minuti (prime 200 righe analizzate)

---

## 📑 Indice dei Contenuti

### [1. Riepilogo: Family Tree e Problema "Own Grandpa"](#1-riepilogo-family-tree-e-problema-own-grandpa) `00:00:43 - 00:03:43`
   - [1.1 Modello di Partenza](#11-modello-di-partenza) `00:00:43 - 00:01:16`
   - [1.2 Fact: Nessun Antenato di Se Stesso](#12-fact-nessun-antenato-di-se-stesso) `00:01:16 - 00:02:24`
   - [1.3 Effetto del Fact sulla Verifica](#13-effetto-del-fact-sulla-verifica) `00:02:24 - 00:03:43`

### [2. Simmetria Wife-Husband](#2-simmetria-wife-husband) `00:03:43 - 00:05:56`
   - [2.1 Problema della Relazione Asimmetrica](#21-problema-della-relazione-asimmetrica) `00:03:43 - 00:04:19`
   - [2.2 Operatore di Trasposizione](#22-operatore-di-trasposizione) `00:04:19 - 00:05:24`
   - [2.3 Visualizzazione di Mondi Corretti](#23-visualizzazione-di-mondi-corretti) `00:05:24 - 00:05:56`

### [3. Esplorazione del Modello con Predicato Show](#3-esplorazione-del-modello-con-predicato-show) `00:05:56 - 00:08:32`
   - [3.1 Definizione del Predicato Show](#31-definizione-del-predicato-show) `00:05:56 - 00:07:00`
   - [3.2 Esecuzione e Primi Problemi](#32-esecuzione-e-primi-problemi) `00:07:00 - 00:08:02`
   - [3.3 Necessità di Nuovi Vincoli](#33-necessità-di-nuovi-vincoli) `00:08:02 - 00:08:32`

### [4. Facts: Sintassi e Semantica](#4-facts-sintassi-e-semantica) `00:08:32 - 00:11:12`
   - [4.1 Facts Multipli e Naming](#41-facts-multipli-e-naming) `00:08:32 - 00:10:07`
   - [4.2 Differenza tra Facts e Predicati](#42-differenza-tra-facts-e-predicati) `00:10:07 - 00:11:12`

### [5. Assertions e Verifica](#5-assertions-e-verifica) `00:11:12 - 00:15:32`
   - [5.1 Asserzione "No Self-Father"](#51-asserzione-no-self-father) `00:11:12 - 00:12:49`
   - [5.2 Esecuzione con e Senza Facts](#52-esecuzione-con-e-senza-facts) `00:12:49 - 00:13:59`
   - [5.3 Dipendenza delle Assertions dal Modello](#53-dipendenza-delle-assertions-dal-modello) `00:13:59 - 00:15:32`

### [6. Vincolo: Social Convention](#6-vincolo-social-convention) `00:15:32 - 00:22:25`
   - [6.1 Problema delle Relazioni Incrociate](#61-problema-delle-relazioni-incrociate) `00:15:32 - 00:18:28`
   - [6.2 Fact: Intersezione Vuota](#62-fact-intersezione-vuota) `00:18:28 - 00:19:41`
   - [6.3 Nuovi Problemi Emergenti](#63-nuovi-problemi-emergenti) `00:19:41 - 00:22:25`

### [7. Antenati Comuni](#7-antenati-comuni) `00:22:25 - 00:32:14`
   - [7.1 Analisi di un Fact Alternativo](#71-analisi-di-un-fact-alternativo) `00:22:25 - 00:27:29`
   - [7.2 Uso dell'Evaluator per Debug](#72-uso-dellevaluator-per-debug) `00:27:29 - 00:29:46`
   - [7.3 Equivalenza con Social Convention](#73-equivalenza-con-social-convention) `00:29:46 - 00:32:14`

### [8. Funzione Ancestors e Vincolo Finale](#8-funzione-ancestors-e-vincolo-finale) `00:32:14 - 00:38:59`
   - [8.1 Definizione della Funzione Ancestors](#81-definizione-della-funzione-ancestors) `00:32:14 - 00:34:27`
   - [8.2 Fact: Non Common Ancestors](#82-fact-non-common-ancestors) `00:34:27 - 00:37:35`
   - [8.3 Implementazione e Testing](#83-implementazione-e-testing) `00:37:35 - 00:38:59`

---

## 1. Riepilogo: Family Tree e Problema "Own Grandpa"

### 1.1 Modello di Partenza

`⏱️ 00:00:43 - 00:01:16`

Iniziamo dalla specifica del **Family Tree** vista nella lezione precedente.

📝 **Modello Base**

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

fun grandpas[p: Person]: set Person {
  p.(mother + father).(mother + father)
}

pred ownGrandpa[p: Person] {
  p in grandpas[p]
}
```

💡 **Insight - Problema Iniziale**

**Senza vincoli aggiuntivi**, il modello permette situazioni assurde:
- Una persona può essere il **proprio nonno**
- Relazioni **cicliche** negli antenati
- Moglie e marito **non corrispondenti**

🔍 **Esecuzione del Predicato**

```alloy
run ownGrandpa
```

**Risultato:** Alloy **trova un'istanza** dove qualcuno è il proprio nonno!

📊 **Esempio di Mondo Problematico**

```
Man0
  father = Man1
  mother = Woman0

Man1
  father = Man0  // ← Ciclo! Man0 è nonno di se stesso
  mother = Woman0

Woman0
  father = Man1
  mother = Woman0  // ← Anche Woman0 ha cicli
```

⚠️ **Attenzione - Necessità di Vincoli**

Questo dimostra che **dobbiamo aggiungere facts** per escludere mondi insensati dal punto di vista del dominio.

---

### 1.2 Fact: Nessun Antenato di Se Stesso

`⏱️ 00:01:16 - 00:02:24`

Introduciamo il primo **fact** per impedire cicli negli antenati.

📝 **Fact: No Self-Ancestor**

```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}
```

💡 **Insight - Chiusura Transitiva**

**Senza chiusura transitiva:**
```alloy
fact weak {
  all p: Person | p not in p.(mother + father)
}
```
- Impedisce solo di essere **genitore diretto** di se stesso
- **NON** impedisce di essere nonno, bisnonno, etc.

**Con chiusura transitiva:**
```alloy
fact strong {
  all p: Person | p not in p.^(mother + father)
}
```
- Impedisce di essere **antenato a qualsiasi livello**
- Include genitori, nonni, bisnonni, trisavoli, ...

🔍 **Analisi Dettagliata**

**Componenti del fact:**

1. **`mother + father`**: Relazione "genitore" (unione di madre e padre)
   ```
   {(figlio, genitore) | genitore è madre o padre di figlio}
   ```

2. **`^(mother + father)`**: Chiusura transitiva = "antenato"
   ```
   {(discendente, antenato) | esiste cammino da discendente a antenato}
   ```

3. **`p.^(mother + father)`**: Tutti gli antenati di `p`

4. **`p not in ...`**: `p` non è tra i suoi antenati

📊 **Tabella: Effetto della Chiusura Transitiva**

| Senza `^` | Con `^` |
|-----------|---------|
| `p ≠ p.father` | `p ∉ p.^(mother+father)` |
| `p ≠ p.mother` | Impedisce **tutti** i cicli |
| Solo 1 livello | **Infiniti** livelli |
| Insufficiente | Corretto ✅ |

✅ **Regola Pratica - Quando Usare ^**

**Usa chiusura transitiva quando:**
- Vuoi considerare **tutti i passi** di una relazione
- Esempio: Antenati (non solo genitori)
- Esempio: Raggiungibilità in grafi
- Esempio: Dipendenze transitive

---

### 1.3 Effetto del Fact sulla Verifica

`⏱️ 00:02:24 - 00:03:43`

Vediamo come il fact modifica il comportamento del modello.

📝 **Posizionamento dei Facts**

```alloy
abstract sig Person { ... }
sig Man extends Person { ... }
sig Woman extends Person { ... }

// ← Facts tipicamente QUI (dopo le signatures)
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

// Poi predicati, funzioni, assertions
fun grandpas[p: Person]: set Person { ... }
pred ownGrandpa[p: Person] { ... }
```

💡 **Insight - Organizzazione Standard**

**Ordine consigliato in un documento Alloy:**
1. **Module** e imports
2. **Signatures** (tipi e relazioni)
3. **Facts** (vincoli globali) ← Subito dopo signatures
4. **Functions** (calcoli riutilizzabili)
5. **Predicates** (scenari)
6. **Assertions** (proprietà da verificare)
7. **Commands** (run, check)

🔍 **Esecuzione Dopo il Fact**

```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

run ownGrandpa
```

**Risultato:** **"No instance found"** ✅

**Motivo:** Con il fact attivo, **non è più possibile** essere il proprio nonno, perché ciò richiederebbe un ciclo negli antenati.

📊 **Confronto Prima/Dopo**

| Situazione | Senza Fact | Con Fact |
|-----------|-----------|----------|
| `run ownGrandpa` | **Instance found** | **No instance found** |
| Cicli negli antenati | ✅ Permessi | ❌ Vietati |
| Persona = proprio nonno | ✅ Possibile | ❌ Impossibile |
| Mondi generati | Molti (alcuni assurdi) | Meno (solo sensati) |

⚠️ **Attenzione - I Facts Restringono lo Spazio**

```
Prima del fact:
█████████████████████  (tutti i mondi possibili)

Dopo il fact:
█████░░░░░░░░░░░░░░░░  (solo mondi senza cicli)
  ↑          ↑
Validi    Esclusi dal fact
```

I facts **riducono** i mondi che l'Analyzer può generare, mantenendo solo quelli che soddisfano i vincoli.

---

## 2. Simmetria Wife-Husband

### 2.1 Problema della Relazione Asimmetrica

`⏱️ 00:03:43 - 00:04:19`

Anche con il fact sugli antenati, abbiamo ancora un **problema di coerenza**.

📝 **Problema da Risolvere**

Nel modello attuale:
```alloy
sig Man extends Person {
  wife: lone Woman
}

sig Woman extends Person {
  husband: lone Man
}
```

**Situazione incoerente permessa:**
```
Man0
  wife = Woman0

Woman0
  husband = Man1  // ← Man1, non Man0! Incoerenza!
```

💡 **Insight - Simmetria Attesa**

Nel mondo reale, la relazione matrimoniale è **simmetrica**:
- Se M è sposato con W, allora W è sposata con M
- **Non** può succedere: M sposato con W, ma W sposata con un altro

**Vincolo desiderato:**
```
M.wife = W  ⟺  W.husband = M
```

Per **ogni** coppia (M, W).

🔍 **Esempio di Incoerenza**

```
Man0: wife = Woman0
Man1: wife = none

Woman0: husband = Man1  // ← Dovrebbe essere Man0!
```

**Problema:** Woman0 dice di essere sposata con Man1, ma Man0 dice che Woman0 è sua moglie!

📊 **Perché Serve un Fact**

| Aspetto | Senza Vincolo | Con Vincolo |
|---------|--------------|-------------|
| Coerenza | ❌ Non garantita | ✅ Garantita |
| Wife/Husband | Possono non corrispondere | Devono corrispondere |
| Semantica | Matematica pura | Dominio reale |

⚠️ **Attenzione - Non è Automatico**

Anche se **semanticamente ovvio per noi**, Alloy **non sa** che wife e husband devono essere simmetrici. Dobbiamo **esplicitarlo** con un fact.

---

### 2.2 Operatore di Trasposizione

`⏱️ 00:04:19 - 00:05:24`

Possiamo esprimere la simmetria in modo elegante con l'**operatore di trasposizione**.

📝 **Fact: Simmetria Wife-Husband (Set Theory Style)**

```alloy
fact wifeHusbandSymmetry {
  wife = ~husband
}
```

💡 **Insight - Operatore `~` (Transpose)**

**L'operatore `~`** inverte le coppie in una relazione:

```
R = {(a, b), (c, d), (e, f)}
~R = {(b, a), (d, c), (f, e)}
```

**Nel nostro caso:**

**Relazione `wife`:**
```
wife ⊆ Man × Woman
wife = {(m, w) | m è un uomo e w è sua moglie}
```

**Relazione `husband`:**
```
husband ⊆ Woman × Man
husband = {(w, m) | w è una donna e m è suo marito}
```

**Trasposizione `~husband`:**
```
~husband = {(m, w) | (w, m) ∈ husband}
```

🔍 **Esempio Concreto**

```
husband = {(Woman0, Man0), (Woman1, Man1)}

~husband = {(Man0, Woman0), (Man1, Man1)}

wife = {(Man0, Woman0), (Man1, Man1)}

Verifica: wife = ~husband ✅
```

📊 **Due Modi Equivalenti**

| Stile Logico | Stile Set Theory |
|-------------|------------------|
| `all m: Man, w: Woman \| (m.wife = w) iff (w.husband = m)` | `wife = ~husband` |
| Più esplicito | Più conciso |
| Quantificatori | Operatori insiemistici |
| Più lungo | Una riga |

✅ **Vantaggi dello Stile Set Theory**

- **Concisione**: Una sola riga
- **Eleganza**: Esprime direttamente la simmetria
- **Efficienza**: Più facile da verificare per l'Analyzer
- **Chiarezza**: Una volta compreso `~`, è molto chiaro

---

### 2.3 Visualizzazione di Mondi Corretti

`⏱️ 00:05:24 - 00:05:56`

Con entrambi i facts, possiamo ora visualizzare mondi **più disciplinati**.

📝 **Specifica Aggiornata**

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

// Fact 1: Nessun ciclo negli antenati
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

// Fact 2: Simmetria matrimoniale
fact wifeHusbandSymmetry {
  wife = ~husband
}
```

💡 **Insight - Effetto Combinato dei Facts**

**Mondi ora escludono:**
- ❌ Cicli negli antenati (grazie a `noSelfAncestor`)
- ❌ Inconsistenze wife/husband (grazie a `wifeHusbandSymmetry`)

**Ma potrebbero ancora avere:**
- ⚠️ Padre sposato con propria figlia
- ⚠️ Fratelli che condividono genitori sposati tra loro
- ⚠️ Altre situazioni "strane" non ancora vietate

🔍 **Necessità di Predicato Show**

Per **visualizzare** mondi corretti, dobbiamo definire un predicato eseguibile:

```alloy
pred show {}

run show for 5
```

**Motivo:** Il predicato `ownGrandpa` **non può più** essere soddisfatto con i facts attuali, quindi dobbiamo usarne un altro.

📊 **Evoluzione del Modello**

| Versione | Facts | Mondi Permessi |
|----------|-------|----------------|
| V1 (iniziale) | Nessuno | Tutti (anche assurdi) |
| V2 | `noSelfAncestor` | Senza cicli |
| V3 | + `wifeHusbandSymmetry` | Senza cicli + coerenza matrimoni |
| V4 (prossima) | + altri vincoli | Sempre più realistici |

✅ **Approccio Iterativo**

Il processo di sviluppo di una specifica Alloy è **iterativo**:
1. Definisci signature base
2. Esegui con `run` → trova mondi strani
3. Aggiungi facts per escluderli
4. Ripeti fino a ottenere solo mondi sensati

---

## 3. Esplorazione del Modello con Predicato Show

### 3.1 Definizione del Predicato Show

`⏱️ 00:05:56 - 00:07:00`

Per esplorare la specifica, definiamo un predicato **show**.

📝 **Predicato Show**

```alloy
pred show {}

run show for 5
```

**Interpretazione:**
- Predicato **vuoto** (nessun vincolo aggiuntivo)
- Scope: al massimo **5 persone** totali
- Genera mondi che soddisfano **solo i facts** definiti

💡 **Insight - Scope e Persone**

```alloy
run show for 5
```

**Significato:**
- Massimo 5 elementi nel set `Person`
- Ogni persona è o `Man` o `Woman` (per via di `extends`)
- Quindi: **combinazioni** di uomini e donne fino a 5 totali

📊 **Possibili Distribuzioni**

| Man | Woman | Totale |
|-----|-------|--------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 2 | 1 | 3 |
| 3 | 2 | 5 |
| ... | ... | ... |

L'Analyzer esplorerà **tutte** le combinazioni possibili entro i limiti.

🔍 **Errore di Sintassi**

```alloy
pred shows {}  // ← Nome sbagliato!
run shows for 5
```

**Problema:** Typo nel nome del predicato.

**Correzione:**
```alloy
pred show {}   // ← Corretto
run show for 5
```

⚠️ **Attenzione - Importanza dei Nomi**

I predicati **devono** essere chiamati con il nome esatto:
- Alloy è **case-sensitive**
- Spazi e caratteri speciali contano
- Errori di battitura → "Predicate not found"

---

### 3.2 Esecuzione e Primi Problemi

`⏱️ 00:07:00 - 00:08:02`

Eseguiamo il predicato `show` e analizziamo i risultati.

📝 **Esecuzione**

```alloy
run show for 5
```

**Risultato:** "Instance found" ✅

**Visualizzazione:** Possiamo ispezionare il mondo generato.

💡 **Mondo Trovato - Esempio**

```
Man0
  father = none
  mother = Woman0
  wife = Woman0

Woman0
  father = none
  mother = none
  husband = Man0
```

🔍 **Analisi del Mondo**

**Aspetti corretti:**
- ✅ Wife/husband sono **simmetrici** (Man0 ↔ Woman0)
- ✅ Nessun ciclo negli antenati

**Aspetto problematico:**
- ⚠️ **Man0 è sposato con la propria madre!**
  - Man0.mother = Woman0
  - Man0.wife = Woman0
  - **Stessa persona** in due relazioni incompatibili

📊 **Relazioni nel Mondo**

```
        Woman0 (madre E moglie di Man0)
         ↑   ↓
      mother wife
         |   ↓
        Man0
         ↑
      husband
```

**Situazione:** Man0 ha Woman0 come:
1. **Madre** (relazione genitoriale)
2. **Moglie** (relazione matrimoniale)

**Problema:** Viola convenzioni sociali/biologiche!

⚠️ **Attenzione - Specifica Ancora Incompleta**

I facts attuali **non impediscono**:
- Matrimoni tra genitori e figli
- Matrimoni tra fratelli
- Altre relazioni "taboo"

**Necessità:** Aggiungere **nuovi facts** per escludere questi casi!

---

### 3.3 Necessità di Nuovi Vincoli

`⏱️ 00:08:02 - 00:08:32`

Dobbiamo **migliorare ulteriormente** la specifica aggiungendo nuovi facts.

💡 **Insight - Facts come Regole di Correttezza**

I **facts** definiscono le **regole del mondo**:
- Sono **sempre veri** in ogni mondo generato
- Definiscono cosa è **accettabile** nel dominio
- Escludono situazioni **semanticamente invalide**

📝 **Obiettivo Successivo**

Vogliamo impedire che:
1. **Coniugi** siano anche **genitori/figli**
2. **Coniugi** abbiano **antenati comuni**
3. Altre situazioni familiari inconsistenti

🔍 **Strategia di Sviluppo**

```
1. Esegui run show → Trova mondo
2. Ispeziona mondo → Identifica problemi
3. Definisci fact → Escludi quel tipo di problema
4. Ripeti → Iterativamente raffina la specifica
```

📊 **Facts Pianificati**

| Problema | Fact da Aggiungere |
|----------|-------------------|
| Coniuge = genitore | Intersezione vuota wife/husband con antenati |
| Antenati comuni tra coniugi | Intersezione vuota tra antenati di M e W |
| ... | ... |

✅ **Processo Iterativo**

Lo sviluppo di specifiche Alloy è **iterativo e guidato da esempi**:
- Non pensiamo a tutti i vincoli dall'inizio
- Li scopriamo **esplorando** i mondi generati
- Aggiungiamo facts **man mano** che troviamo problemi

**Questa è la forza di Alloy:** Trova corner cases che non avremmo immaginato!

---

## 4. Facts: Sintassi e Semantica

### 4.1 Facts Multipli e Naming

`⏱️ 00:08:32 - 00:10:07`

Approfondiamo la **sintassi** dei facts in Alloy.

📝 **Facts Multipli - Due Approcci**

**Approccio 1: Facts separati**
```alloy
fact fact1 {
  constraint1
}

fact fact2 {
  constraint2
}

fact fact3 {
  constraint3
}
```

**Approccio 2: Fact unico con AND implicito**
```alloy
fact allConstraints {
  constraint1
  constraint2  // ← AND implicito
  constraint3
}
```

💡 **Insight - AND Implicito**

Quando più constraint sono in un fact, sono **implicitamente in AND**:

```alloy
fact example {
  constraint1
  constraint2
}

// Equivalente a:
fact example {
  constraint1 and constraint2
}
```

🔍 **Naming dei Facts**

**Fact anonimo:**
```alloy
fact {
  all p: Person | p not in p.^(mother + father)
}
```

**Fact nominato:**
```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}
```

📊 **Confronto Anonimo vs Nominato**

| Aspetto | Anonimo | Nominato |
|---------|---------|----------|
| **Sintassi** | `fact { ... }` | `fact name { ... }` |
| **Leggibilità** | Meno chiaro | Più chiaro ✅ |
| **Debugging** | Difficile | Più facile |
| **Documentazione** | Nessuna | Nome descrive scopo |
| **Analyzer** | Funziona uguale | Funziona uguale |

✅ **Regola Pratica - Quando Nominare**

**Nomina i facts quando:**
- Esprimono un concetto importante del dominio
- Esempio: `noSelfAncestor`, `wifeHusbandSymmetry`, `socialConvention`

**Usa facts anonimi quando:**
- Sono vincoli tecnici minori
- Esempio: Limiti di cardinalità semplici

**Best practice:** Nomina **sempre** i facts per migliorare la leggibilità!

---

### 4.2 Differenza tra Facts e Predicati

`⏱️ 00:10:07 - 00:11:12`

È fondamentale capire la **differenza** tra facts e predicati.

📝 **Differenza Chiave**

**Facts:**
- Devono valere **sempre** (globalmente)
- **Non** hanno parametri
- **Non** vengono chiamati (sono sempre attivi)
- Usati per **invarianti** del dominio

**Predicati:**
- Valgono **solo se chiamati** con `run`
- **Possono** avere parametri
- Devono essere **esplicitamente eseguiti**
- Usati per **scenari** e **operazioni**

💡 **Insight - Applicazione Automatica**

```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

pred ownGrandpa[p: Person] {
  p in grandpas[p]
}

run show        // ← noSelfAncestor applicato automaticamente!
run ownGrandpa  // ← noSelfAncestor applicato anche qui!
```

**Il fact è SEMPRE attivo**, indipendentemente da quale predicato eseguiamo.

🔍 **Esempio Concreto**

**Specifica:**
```alloy
fact f {
  all p: Person | #p.father <= 1
}

pred p1 {
  some p: Person | #p.father > 1
}

run p1  // ← "No instance found"
```

**Motivo:** `p1` richiede ≥2 padri, ma `f` permette solo ≤1. **Contraddizione!**

📊 **Tabella Comparativa Completa**

| Caratteristica | Facts | Predicati |
|----------------|-------|-----------|
| **Keyword** | `fact` | `pred` |
| **Parametri** | ❌ No | ✅ Sì |
| **Chiamata** | Automatica | Esplicita (`run`) |
| **Scope** | Globale | Locale (quando chiamato) |
| **Uso** | Invarianti | Scenari/Operazioni |
| **Nome** | Opzionale | Obbligatorio |
| **Esempio** | "Nessun ciclo" | "Aggiungi elemento" |

✅ **Regola Pratica - Quando Usare Cosa**

**Usa Facts per:**
- Proprietà che devono **sempre** valere
- Vincoli **strutturali** del dominio
- **Invarianti** che definiscono "mondi validi"
- Esempi: Nessun ciclo, simmetrie, limiti biologici

**Usa Predicati per:**
- Operazioni **parametrizzate**
- **Scenari** specifici da esplorare
- **Transizioni** di stato
- Esempi: Add, delete, show con vincoli specifici

---

## 5. Assertions e Verifica

### 5.1 Asserzione "No Self-Father"

`⏱️ 00:11:12 - 00:12:49`

Le **assertions** servono per verificare **proprietà** della specifica.

📝 **Definizione dell'Asserzione**

```alloy
assert noSelfFather {
  no m: Man | m = m.father
}
```

**Interpretazione:**
> "Non esiste alcun uomo che sia padre di se stesso"

💡 **Insight - Quantificatore `no`**

**`no x: T | formula`** significa:
- "Non esiste alcun `x` di tipo `T` tale che `formula` sia vera"
- Equivalente a: `not (some x: T | formula)`

**Nel nostro caso:**
```alloy
no m: Man | m = m.father

// Equivalente a:
not (some m: Man | m = m.father)

// In parole:
// "Non è vero che esiste un uomo uguale al proprio padre"
```

🔍 **Analisi della Formula**

```alloy
m = m.father
```

**Cosa significa:**
- `m.father`: Il padre dell'uomo `m`
- `m = m.father`: `m` è uguale al proprio padre
- Sarebbe un **ciclo di lunghezza 1** nella relazione padre

📊 **Asserzione vs Fact**

| Aspetto | Fact | Assertion |
|---------|------|-----------|
| **Applica vincolo** | ✅ Sì (modifica modello) | ❌ No (solo verifica) |
| **Comando** | Automatico | `check` |
| **Obiettivo** | Definire regole | Verificare proprietà |
| **Esempio** | "Nessun ciclo" | "Verifica che non ci siano cicli" |

**Differenza cruciale:**
- **Fact**: "Fai in modo che questa proprietà valga"
- **Assertion**: "Controlla se questa proprietà vale (dato il modello attuale)"

✅ **Quando Usare Assertions**

**Assertions sono utili per:**
- Verificare **conseguenze** dei facts
- Documentare **proprietà attese**
- **Testing** della specifica
- Trovare **inconsistenze** nel modello

---

### 5.2 Esecuzione con e Senza Facts

`⏱️ 00:12:49 - 00:13:59`

Verifichiamo l'asserzione in **due scenari**: con e senza il fact `noSelfAncestor`.

📝 **Scenario 1: Con il Fact**

```alloy
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

assert noSelfFather {
  no m: Man | m = m.father
}

check noSelfFather for 5
```

**Risultato:** **"No counterexample found"** ✅

**Motivo:** Il fact `noSelfAncestor` impedisce cicli a **qualsiasi livello**, quindi anche cicli di lunghezza 1 (padre di se stesso).

💡 **Insight - Implicazione Logica**

```
noSelfAncestor ⟹ noSelfFather
```

Se **nessuno** è antenato di se stesso (a qualsiasi livello), allora **nessuno** è padre di se stesso (livello 1).

**Formalmente:**
```
p not in p.^(mother + father)  ⟹  p ≠ p.father
```

🔍 **Scenario 2: Senza il Fact (Commentato)**

```alloy
// fact noSelfAncestor {
//   all p: Person | p not in p.^(mother + father)
// }

assert noSelfFather {
  no m: Man | m = m.father
}

check noSelfFather for 5
```

**Risultato:** **"Counterexample found"** ❌

**Mondo counter-example:**
```
Man0
  father = Man0  // ← Padre di se stesso!
  mother = none
  wife = none
```

📊 **Confronto Risultati**

| Con `noSelfAncestor` | Senza `noSelfAncestor` |
|---------------------|----------------------|
| No counterexample found ✅ | Counterexample found ❌ |
| Asserzione valida | Asserzione invalida |
| Cicli impediti | Cicli permessi |

⚠️ **Attenzione - Dipendenza dal Modello**

L'**asserzione non modifica** il modello, solo **verifica** una proprietà.

Il risultato dipende dai **facts presenti**:
- Con facts appropriati → Asserzione valida
- Senza facts → Asserzione può fallire

---

### 5.3 Dipendenza delle Assertions dal Modello

`⏱️ 00:13:59 - 00:15:32`

Approfondiamo il concetto di **dipendenza** delle assertions dal modello.

💡 **Insight - Assertion come Test**

Un'**assertion** è come un **test** per il modello:
- Verifica se una proprietà vale
- **Dipende** dai facts definiti
- Modifica del modello → Possibile cambio del risultato

📝 **Principio Fondamentale**

```
Assertion sempre eseguita CONTRO un modello
```

**Modello = Signatures + Facts**

**L'assertion verifica:** "Data questa specifica (signatures + facts), vale la proprietà?"

🔍 **Esempio di Modifica**

**Modello V1:**
```alloy
sig Person { father: lone Person }
// Nessun fact

assert noSelfFather {
  no p: Person | p = p.father
}

check noSelfFather  // ← Counterexample found ❌
```

**Modello V2:**
```alloy
sig Person { father: lone Person }

fact {
  all p: Person | p != p.father
}

assert noSelfFather {
  no p: Person | p = p.father
}

check noSelfFather  // ← No counterexample found ✅
```

**Stessa assertion, modelli diversi → Risultati diversi!**

📊 **Workflow di Verifica**

```
1. Definisci Signatures
         ↓
2. Aggiungi Facts (vincoli del dominio)
         ↓
3. Scrivi Assertions (proprietà attese)
         ↓
4. Esegui check
         ↓
5a. No counterexample → ✅ Modello corretto (per questa proprietà)
5b. Counterexample → ❌ Modello ha problemi o assertion sbagliata
         ↓
6. Itera: Raffina facts o correggi assertion
```

✅ **Regola Pratica - Uso delle Assertions**

**Assertions servono per:**
1. **Documentare** proprietà che ci aspettiamo valgano
2. **Verificare** che i facts abbiano l'effetto desiderato
3. **Testing** incrementale durante sviluppo
4. **Prevenire regressioni** quando si modificano facts

**Best practice:**
- Scrivi assertions per proprietà **importanti**
- Verifica con **scope crescenti** (3, 5, 8, 10)
- Se trovi counterexample: analizza e raffina il modello

---

## 6. Vincolo: Social Convention

### 6.1 Problema delle Relazioni Incrociate

`⏱️ 00:15:32 - 00:18:28`

Affrontiamo il problema di **coniugi che sono anche genitori/figli**.

📝 **Problema da Risolvere**

Abbiamo visto mondi come:
```
Man0
  father = none
  mother = Woman0
  wife = Woman0  // ← Moglie E madre!

Woman0
  father = none
  mother = none
  husband = Man0
```

**Situazione:** Un uomo sposato con la propria madre! 😱

💡 **Insight - Due Insiemi Incompatibili**

Vogliamo che due **insiemi di coppie** siano **disgiunti**:

**Insieme 1: Coppie coniugali**
```
wife ∪ husband = {(m, w) | m e w sono coniugi}
```

**Insieme 2: Coppie genitoriali (a qualsiasi livello)**
```
^(mother + father) = {(discendente, antenato) | relazione antenato}
```

**Vincolo desiderato:**
```
(wife ∪ husband) ∩ ^(mother + father) = ∅
```

🔍 **Visualizzazione Insiemistica**

```
      Coppie coniugali          Coppie genitoriali
   ┌──────────────────┐       ┌──────────────────┐
   │  (Man0, Woman0)  │       │  (Man1, Woman1)  │
   │  (Man2, Woman2)  │       │  (Man0, Woman1)  │
   │       ...        │       │       ...        │
   └──────────────────┘       └──────────────────┘
           ↓                           ↓
           └───────── ∩ = ∅ ───────────┘
                 (Intersezione vuota)
```

📊 **Esempi di Violazione**

| Coppia | In Wife/Husband? | In Antenati? | Problema? |
|--------|-----------------|-------------|-----------|
| (Man0, Woman0) | ✅ Sì (coniugi) | ✅ Sì (madre-figlio) | ❌ VIOLA |
| (Man1, Woman1) | ✅ Sì (coniugi) | ❌ No (non imparentati) | ✅ OK |
| (Man2, Woman2) | ✅ Sì (coniugi) | ✅ Sì (nonno-nipote) | ❌ VIOLA |

⚠️ **Attenzione - Chiusura Transitiva Necessaria**

Non basta impedire matrimoni genitore-figlio **diretti**:

```alloy
// ❌ INSUFFICIENTE
fact weak {
  no wife & (mother + father)
}
```

Bisogna considerare **tutti i livelli** della gerarchia:

```alloy
// ✅ CORRETTO
fact strong {
  no (wife + husband) & ^(mother + father)
}
```

Altrimenti: nonno può sposare nipote, bisnonno bisnipote, etc.!

---

### 6.2 Fact: Intersezione Vuota

`⏱️ 00:18:28 - 00:19:41`

Definiamo il fact **social convention** per risolvere il problema.

📝 **Fact: Social Convention**

```alloy
fact socialConvention {
  no (wife + husband) & ^(mother + father)
}
```

**Interpretazione:**
> "L'intersezione tra coniugi e antenati è vuota"

💡 **Insight - Operatore `&` (Intersezione)**

In Alloy, `&` è l'**intersezione tra insiemi**:

```alloy
A & B = {x | x ∈ A ∧ x ∈ B}
```

**Nel nostro caso:**
```alloy
(wife + husband) & ^(mother + father)
= {(p1, p2) | (p1, p2) è coppia coniugale E antenato}
```

🔍 **Analisi Dettagliata**

**Componenti del fact:**

1. **`wife + husband`**: Unione delle relazioni matrimoniali
   ```
   {(m, w) | m.wife = w} ∪ {(w, m) | w.husband = m}
   ```

2. **`^(mother + father)`**: Chiusura transitiva relazioni genitoriali
   ```
   {(discendente, antenato) | cammino via mother/father}
   ```

3. **`... & ...`**: Intersezione
   ```
   Coppie che sono SIA coniugi SIA genitori/antenati
   ```

4. **`no ...`**: L'insieme deve essere vuoto
   ```
   Non deve esistere alcuna coppia in entrambi gli insiemi
   ```

📊 **Effetto del Fact**

**Prima del fact:**
```
Man0: wife = Woman0, mother = Woman0  ✅ Permesso
```

**Dopo il fact:**
```
Man0: wife = Woman0, mother = Woman0  ❌ ESCLUSO
```

Il mondo sopra viola il fact, quindi l'Analyzer **non lo genererà mai**.

✅ **Verifica dell'Implementazione**

Aggiungiamo il fact alla specifica:

```alloy
fact socialConvention {
  no (wife + husband) & ^(mother + father)
}

run show for 5
```

Eseguendo, i mondi generati **non avranno più** coniugi che sono anche antenati!

---

### 6.3 Nuovi Problemi Emergenti

`⏱️ 00:19:41 - 00:22:25`

Anche con `socialConvention`, emergono **nuovi problemi**.

📝 **Esecuzione con Tutti i Facts**

```alloy
fact noSelfAncestor { ... }
fact wifeHusbandSymmetry { ... }
fact socialConvention { ... }

run show for 5
```

**Risultato:** Instance found ✅

💡 **Mondo Generato - Nuovo Problema**

```
Woman0
  father = Man0
  mother = none
  husband = none

Woman1
  father = Man0  // ← Stesso padre!
  mother = Woman0
  husband = none

Man0
  father = none
  mother = none
  wife = none
```

🔍 **Analisi del Problema**

**Situazione:**
- Woman1.mother = Woman0
- Woman1.father = Man0
- Woman0.father = Man0 ← **Stesso padre**!

**Problema:** Woman0 è **madre** di Woman1, ma hanno lo **stesso padre** (Man0).

Quindi: **Woman0 e Woman1 sono sorelle** (stesso padre), ma Woman0 è anche madre di Woman1!

📊 **Relazioni nel Mondo**

```
        Man0 (padre di entrambe)
       /    \
   father  father
     /        \
Woman0 ←─── Woman1
 (madre)   mother
```

**Inconsistenza biologica:** Una donna non può essere contemporaneamente:
- **Sorella** (stesso padre)
- **Madre** (relazione mother)

⚠️ **Attenzione - Problema Generale**

Più in generale, vogliamo impedire che **coniugi abbiano antenati comuni**:
- Fratelli che si sposano
- Cugini che si sposano (se vogliamo essere restrittivi)
- Qualsiasi relazione matrimoniale con antenati condivisi

**Necessità:** Un nuovo fact per impedire **antenati comuni** tra coniugi!

✅ **Prossimi Passi**

Dobbiamo aggiungere un vincolo tipo:
```alloy
fact noCommonAncestors {
  // Per ogni coppia di coniugi,
  // i loro antenati devono essere disgiunti
}
```

Vedremo diverse formulazioni di questo vincolo nelle sezioni successive!

---

## 7. Antenati Comuni

### 7.1 Analisi di un Fact Alternativo

`⏱️ 00:22:25 - 00:27:29`

Analizziamo un **fact alternativo** proposto per impedire antenati comuni.

📝 **Fact Proposto (da Discussione in Classe)**

```alloy
fact noCommonAncestorsAttempt {
  all m: Man, w: Woman |
    (m.wife = w and w.husband = m) implies
    (no m & w.^(mother + father) and
     no w & m.^(mother + father))
}
```

💡 **Insight - Struttura del Fact**

**Forma generale:**
```alloy
all variabili | precondizione implies vincolo
```

**Nel nostro caso:**
- **Variabili**: `m: Man, w: Woman`
- **Precondizione**: `m.wife = w and w.husband = m` (sono coniugi)
- **Vincolo**: Nessun antenato comune

🔍 **Analisi Dettagliata della Precondizione**

```alloy
m.wife = w and w.husband = m
```

**Cosa significa:**
- `m.wife = w`: La moglie di `m` è `w`
- `w.husband = m`: Il marito di `w` è `m`
- **AND**: Entrambi devono valere

**Nota:** Con il fact `wifeHusbandSymmetry`, in realtà **basta uno** dei due:
```alloy
// Sufficiente (con wifeHusbandSymmetry):
m.wife = w

// Oppure:
w.husband = m
```

📊 **Analisi del Vincolo**

```alloy
no m & w.^(mother + father) and
no w & m.^(mother + father)
```

**Parte 1:** `no m & w.^(mother + father)`
- `w.^(mother + father)`: Tutti gli antenati di `w`
- `m & ...`: Intersezione tra `m` e gli antenati di `w`
- `no ...`: Questa intersezione deve essere vuota

**Interpretazione:** `m` non deve essere tra gli antenati di `w`.

**Parte 2:** `no w & m.^(mother + father)`
- Simmetricamente: `w` non deve essere tra gli antenati di `m`.

⚠️ **Attenzione - Cosa NON Fa**

Questo fact impedisce:
- ❌ `m` antenato di `w`
- ❌ `w` antenato di `m`

Ma **NON** impedisce:
- ✅ `m` e `w` con antenato comune `a` (dove `a ≠ m` e `a ≠ w`)

**Esempio permesso:**
```
        Man2 (antenato comune)
       /    \
   father  father
     /        \
   Man0 ←──── Woman0
  (marito)   (moglie)
```

Man0 e Woman0 possono sposarsi anche se condividono Man2 come padre/antenato!

---

*Continua nel file completo...*

---

### 7.2 Uso dell'Evaluator per Debug

`⏱️ 00:27:29 - 00:29:46`

L'**Evaluator** di Alloy è uno strumento potente per **esplorare e debuggare** le specifiche.

� **Insight - Evaluator come Debugger**

L'Evaluator permette di:
- **Valutare espressioni** in un mondo specifico
- **Testare formule** per capire cosa restituiscono
- **Esplorare relazioni** e verificare assunzioni
- **Debug incrementale** della specifica

�📝 **Come Usare l'Evaluator**

**Passo 1:** Esegui un predicato e trova un'istanza
```alloy
run show for 5
```

**Passo 2:** Apri l'Evaluator nel visualizzatore

**Passo 3:** Scrivi espressioni Alloy da valutare

🔍 **Esempio Pratico - Chiusura Transitiva**

**Nel mondo visualizzato, valutiamo:**

```alloy
^(mother + father)
```

**Risultato mostrato come tabella:**

| From | To |
|------|-----|
| Man0 | Woman0 |
| Man1 | Man0 |
| Man1 | Woman0 |
| Woman1 | Man0 |
| Woman1 | Woman0 |
| Woman1 | Woman1 |

**Interpretazione:** Tutte le coppie (discendente, antenato) nel mondo corrente.

📊 **Esempio di Analisi**

**Mondo:**
```
Woman0
  father = none
  mother = none

Woman1
  father = none
  mother = Woman0

Man0
  father = Man1
  mother = Woman0
```

**Espressione:** `Woman1.^(mother + father)`

**Risultato:** `{Woman0}`

**Verifica manuale:**
- Woman1.mother = Woman0
- Woman0 non ha genitori
- Quindi antenati di Woman1 = {Woman0} ✅

💡 **Insight - Evaluator per Validazione**

Possiamo usare l'Evaluator per **verificare** se un fact funziona:

```alloy
// Verifica che il fact sia rispettato nel mondo corrente
no (wife + husband) & ^(mother + father)
```

**Risultato:** `true` se il mondo rispetta il fact, `false` altrimenti.

🔍 **Debug di Formule Complesse**

Per formule complesse, valutiamo **pezzi separati**:

```alloy
// Passo 1: Valuta la prima parte
wife + husband

// Passo 2: Valuta la seconda parte
^(mother + father)

// Passo 3: Valuta l'intersezione
(wife + husband) & ^(mother + father)

// Passo 4: Verifica che sia vuota
no (wife + husband) & ^(mother + father)
```

✅ **Regola Pratica - Workflow di Debug**

**Quando hai un problema:**
1. **Esegui** `run show` per generare un mondo
2. **Ispeziona** visualmente per identificare anomalie
3. **Usa Evaluator** per testare sub-espressioni
4. **Identifica** quale parte della formula non funziona
5. **Raffina** il fact basandoti sull'analisi
6. **Riprova** con nuovo fact

---

### 7.3 Equivalenza con Social Convention

`⏱️ 00:29:46 - 00:32:14`

Dimostriamo che il fact alternativo è **equivalente** a `socialConvention`.

📝 **Fact Alternativo (dalla discussione)**

```alloy
fact alternativeApproach {
  all m: Man, w: Woman |
    (m.wife = w and w.husband = m) implies
    (no m & w.^(mother + father) and
     no w & m.^(mother + father))
}
```

📝 **Fact Original (Social Convention)**

```alloy
fact socialConvention {
  no (wife + husband) & ^(mother + father)
}
```

💡 **Insight - Dimostrazione di Equivalenza**

I due facts sono **logicamente equivalenti**:
- Esprimono lo stesso vincolo
- Producono gli stessi mondi validi
- Differiscono solo nello **stile**

🔍 **Analisi dell'Equivalenza**

**Fact alternativo dice:**
> "Per ogni coppia di coniugi (m, w), m non è antenato di w E w non è antenato di m"

**Social convention dice:**
> "Nessuna coppia può essere contemporaneamente in wife/husband E in antenati"

**Perché sono equivalenti:**

```
AlternativeApproach:
∀m, w: (m, w) ∈ wife ⟹ 
  m ∉ antenati(w) ∧ w ∉ antenati(m)

SocialConvention:
wife ∩ antenati = ∅

Equivalenza:
(m, w) ∈ wife ∩ antenati ⟺ 
  (m, w) ∈ wife ∧ (m, w) ∈ antenati ⟺
  m.wife = w ∧ m è antenato di w
```

📊 **Confronto Stili**

| Aspetto | Alternativo | Social Convention |
|---------|-------------|-------------------|
| **Stile** | Logico (quantificatori) | Set theory (operatori) |
| **Leggibilità** | Più esplicito | Più conciso |
| **Lunghezza** | Più lungo | Una riga |
| **Performance** | Equivalente | Equivalente |
| **Preferenza** | Principianti | Esperti |

⚠️ **Attenzione - Nessuno Risolve il Problema Completo**

Entrambi i facts impediscono:
- ✅ Coniuge essere antenato diretto dell'altro

Ma **NON** impediscono:
- ❌ Coniugi con **antenato comune** (fratelli/cugini)

**Esempio ancora permesso:**
```
        Man2 (padre comune)
       /    \
   father  father
     /        \
   Man0 ←──── Woman0
  (marito)   (moglie)
```

**Per risolvere questo:** Serve il fact `noCommonAncestors` che vedremo dopo!

---

## 8. Funzione Ancestors e Vincolo Finale

### 8.1 Definizione della Funzione Ancestors

`⏱️ 00:32:14 - 00:34:27`

Definiamo una **funzione** per calcolare gli antenati di una persona.

📝 **Funzione Ancestors**

```alloy
fun ancestors[p: Person]: set Person {
  p.^(mother + father)
}
```

**Interpretazione:**
> "Data una persona `p`, restituisce l'insieme di tutti i suoi antenati"

💡 **Insight - Funzioni in Alloy**

**Funzioni sono:**
- **Calcoli riutilizzabili** che restituiscono un valore
- **Parametrizzate** (prendono argomenti)
- **Non eseguibili** direttamente (usate in altri costrutti)
- **Pure** (nessun side effect)

🔍 **Sintassi delle Funzioni**

```alloy
fun nomeFunzione[param1: Type1, param2: Type2, ...]: TipoRitorno {
  espressione
}
```

**Componenti:**
- `fun`: Keyword per funzioni
- `nomeFunzione`: Nome della funzione
- `[param1: Type1, ...]`: Parametri (tipo e nome)
- `: TipoRitorno`: Tipo del valore restituito
- `{ espressione }`: Corpo della funzione (espressione Alloy)

📊 **Esempio di Uso**

**Definizione:**
```alloy
fun ancestors[p: Person]: set Person {
  p.^(mother + father)
}
```

**Uso in predicato:**
```alloy
pred hasAncestors[p: Person] {
  some ancestors[p]
}
```

**Uso in fact:**
```alloy
fact noSelfAncestor {
  all p: Person | p not in ancestors[p]
}
```

**Uso in asserzione:**
```alloy
assert ancestorsTransitive {
  all p1, p2, p3: Person |
    (p2 in ancestors[p1] and p3 in ancestors[p2])
    implies p3 in ancestors[p1]
}
```

💡 **Insight - Vantaggi delle Funzioni**

**Benefici:**
1. **Riutilizzo**: Definisci una volta, usa ovunque
2. **Leggibilità**: Nome descrittivo invece di formula complessa
3. **Manutenibilità**: Cambia in un posto solo
4. **Astrazione**: Nasconde dettagli implementativi

**Esempio senza funzione (ripetitivo):**
```alloy
fact f1 { all p: Person | p not in p.^(mother + father) }
fact f2 { all m: Man, w: Woman | 
  (m.wife = w) implies 
  no (m.^(mother + father) & w.^(mother + father)) }
// ^(mother + father) ripetuto molte volte!
```

**Esempio con funzione (pulito):**
```alloy
fun ancestors[p: Person]: set Person { p.^(mother + father) }

fact f1 { all p: Person | p not in ancestors[p] }
fact f2 { all m: Man, w: Woman |
  (m.wife = w) implies no (ancestors[m] & ancestors[w]) }
```

🔍 **Test della Funzione con Evaluator**

Possiamo testare la funzione in un mondo specifico:

```alloy
run show for 5
```

**Nel mondo generato, nell'Evaluator:**

```alloy
ancestors[Woman$2]
```

**Risultato (esempio):**
```
{Woman$0, Man$0}
```

**Verifica manuale:**
- Woman2.mother = Woman0
- Woman2.father = Man0
- Woman0, Man0 non hanno genitori
- Quindi `ancestors[Woman2] = {Woman0, Man0}` ✅

📊 **Funzioni vs Predicati**

| Aspetto | Funzioni | Predicati |
|---------|----------|-----------|
| **Keyword** | `fun` | `pred` |
| **Restituisce** | Valore (set, relation, etc.) | Booleano (vero/falso) |
| **Uso** | In espressioni | Con `run` |
| **Scopo** | Calcolare valori | Definire scenari |
| **Esempio** | `ancestors[p]` | `show {}` |

✅ **Regola Pratica - Quando Definire Funzioni**

**Definisci funzioni quando:**
- Un'espressione complessa è **usata più volte**
- Vuoi dare un **nome significativo** a un calcolo
- Vuoi migliorare la **leggibilità** della specifica
- L'espressione rappresenta un **concetto del dominio**

**Esempi tipici:**
- `ancestors[p]`: Antenati di una persona
- `descendants[p]`: Discendenti di una persona
- `siblings[p]`: Fratelli/sorelle
- `lookup[b, n]`: Cerca indirizzo nel book (già visto)

---

### 8.2 Fact: Non Common Ancestors

`⏱️ 00:34:27 - 00:37:35`

Definiamo il fact **corretto** per impedire antenati comuni tra coniugi.

📝 **Fact: noCommonAncestors**

```alloy
fun ancestors[p: Person]: set Person {
  p.^(mother + father)
}

fact noCommonAncestors {
  all p1: Man, p2: Woman |
    (p1->p2 in wife) implies
    no (ancestors[p1] & ancestors[p2])
}
```

**Interpretazione:**
> "Per ogni coppia di coniugi, i loro insiemi di antenati devono essere disgiunti"

💡 **Insight - Prodotto Cartesiano**

**`p1->p2 in wife`** significa:
- Crea la coppia (tuple) `(p1, p2)`
- Verifica se è nella relazione `wife`
- Equivalente a: `p1.wife = p2`

**Sintassi alternativa:**
```alloy
// Equivalente:
(p1->p2 in wife)
(p1.wife = p2)
```

🔍 **Analisi Dettagliata**

**Componenti del fact:**

1. **Quantificazione:**
   ```alloy
   all p1: Man, p2: Woman |
   ```
   Per ogni possibile coppia uomo-donna

2. **Precondizione:**
   ```alloy
   (p1->p2 in wife) implies
   ```
   Se sono coniugi (coppia in `wife`)

3. **Vincolo:**
   ```alloy
   no (ancestors[p1] & ancestors[p2])
   ```
   L'intersezione dei loro antenati è vuota

📊 **Esempio di Applicazione**

**Mondo problematico (prima del fact):**
```
        Man2 (padre comune)
       /    \
   father  father
     /        \
   Man0      Woman0
    |          |
  wife ←──→ husband
```

**Analisi:**
- `Man0.wife = Woman0`
- `ancestors[Man0] = {Man2}`
- `ancestors[Woman0] = {Man2}`
- `ancestors[Man0] & ancestors[Woman0] = {Man2}` ≠ ∅

**Risultato:** Questo mondo **viola** il fact, quindi viene **escluso**!

**Mondo valido (dopo il fact):**
```
   Man1        Woman1 (genitori separati)
    |            |
  father       mother
    |            |
   Man0  ←──→  Woman0
         wife/husband
```

**Analisi:**
- `Man0.wife = Woman0`
- `ancestors[Man0] = {Man1}`
- `ancestors[Woman0] = {Woman1}`
- `ancestors[Man0] & ancestors[Woman0] = ∅` ✅

⚠️ **Attenzione - Nota sul Husband**

```alloy
fact noCommonAncestors {
  all p1: Man, p2: Woman |
    (p1->p2 in wife) implies  // ← Solo wife!
    no (ancestors[p1] & ancestors[p2])
}
```

**Perché non menzionare anche `husband`?**

Perché abbiamo il fact `wifeHusbandSymmetry`:
```alloy
fact { wife = ~husband }
```

Quindi:
- `(p1, p2) ∈ wife` ⟺ `(p2, p1) ∈ husband`
- Controllare `wife` è **sufficiente**!

💡 **Insight - Dipendenza tra Facts**

I facts **interagiscono**:
- `wifeHusbandSymmetry` garantisce coerenza wife/husband
- `noCommonAncestors` può quindi verificare solo `wife`
- Rimuovendo `wifeHusbandSymmetry`, dovremmo controllare entrambi!

📊 **Confronto con Facts Precedenti**

| Fact | Impedisce |
|------|-----------|
| `socialConvention` | Coniuge essere antenato dell'altro |
| `alternativeApproach` | Stessa cosa (equivalente) |
| `noCommonAncestors` | Coniugi avere antenati comuni ✅ |

**Progressione:**
1. Impedisci coniuge = genitore/antenato diretto
2. **Aggiungi:** Impedisci antenati comuni (fratelli, cugini, etc.)

✅ **Regola Pratica - Raffinamento Iterativo**

**Processo tipico:**
1. Definisci signatures e relazioni base
2. Esegui `run show` → Trova mondi strani
3. Aggiungi fact per escluderli
4. Esegui di nuovo → Trova nuovi problemi
5. Aggiungi nuovo fact più forte
6. **Ripeti** fino a soddisfazione

**Nel nostro caso:**
- V1: Nessun fact → Cicli, inconsistenze totali
- V2: + `noSelfAncestor` → Niente cicli
- V3: + `wifeHusbandSymmetry` → Coerenza matrimoni
- V4: + `socialConvention` → No coniuge-antenato
- V5: + `noCommonAncestors` → No antenati comuni ✅

Ogni fact **restringe** ulteriormente lo spazio dei mondi validi!

---

### 8.3 Implementazione e Testing

`⏱️ 00:37:35 - 00:38:59`

Implementiamo il fact e testiamo la specifica completa.

📝 **Aggiunta del Fact alla Specifica**

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

// Facts precedenti
fact noSelfAncestor {
  all p: Person | p not in p.^(mother + father)
}

fact wifeHusbandSymmetry {
  wife = ~husband
}

fact socialConvention {
  no (wife + husband) & ^(mother + father)
}

// Funzione helper
fun ancestors[p: Person]: set Person {
  p.^(mother + father)
}

// Nuovo fact: no antenati comuni tra coniugi
fact noCommonAncestors {
  all p1: Man, p2: Woman |
    (p1->p2 in wife) implies
    no (ancestors[p1] & ancestors[p2])
}

// Predicato per esplorare
pred show {}

run show for 5
```

💡 **Insight - Problemi di Copia da PowerPoint**

**Attenzione ai caratteri speciali!**
```
// ❌ Caratteri invisibili (da PowerPoint)
"smart quotes", en-dash, em-dash, etc.

// ✅ Caratteri ASCII puri
"normal quotes", regular dash, etc.
```

**Se vedi errori strani:**
- Riscrivi manualmente la riga
- Usa editor che mostra caratteri nascosti
- Copia in Notepad prima, poi in Alloy

🔍 **Esecuzione e Verifica**

```alloy
run show for 5
```

**Risultato:** "Instance found" ✅

**Analisi del mondo generato:**

I mondi ora dovrebbero essere **privi** di:
- ❌ Cicli negli antenati
- ❌ Inconsistenze wife/husband
- ❌ Coniugi che sono antenati l'uno dell'altro
- ❌ Coniugi con antenati comuni

📊 **Esempio di Mondo Valido**

```
Man0
  father = Man1
  mother = none
  wife = Woman0

Woman0
  father = none
  mother = Woman1
  husband = Man0

Man1
  father = none
  mother = none
  wife = none

Woman1
  father = none
  mother = none
  husband = none
```

**Verifica manuale:**
- `Man0.wife = Woman0` e `Woman0.husband = Man0` ✅ (simmetria)
- `ancestors[Man0] = {Man1}`, `ancestors[Woman0] = {Woman1}`
- `{Man1} ∩ {Woman1} = ∅` ✅ (no antenati comuni)
- Nessun ciclo ✅

⚠️ **Attenzione - Potrebbero Esserci Altri Problemi**

Anche con tutti questi facts, potrebbero emergere **nuovi corner cases**:
- Persone senza genitori sposate con persone con genitori
- Relazioni complesse multi-generazionali
- Altri vincoli biologici/sociali

**Approccio:** Continua ad esplorare e raffinare!

📊 **Riepilogo Facts Implementati**

| Fact | Vincolo | Effetto |
|------|---------|---------|
| `noSelfAncestor` | `p not in p.^parents` | No cicli genealogici |
| `wifeHusbandSymmetry` | `wife = ~husband` | Coerenza matrimoni |
| `socialConvention` | `no (wife+husband) & ^parents` | No coniuge-antenato |
| `noCommonAncestors` | `no (anc[p1] & anc[p2])` | No fratelli sposati |

✅ **Messaggio Principale**

**Lo sviluppo di specifiche Alloy è ITERATIVO:**

1. **Definisci** modello base (signatures)
2. **Esplora** con `run` → Trova mondi
3. **Identifica** problemi (mondi insensati)
4. **Aggiungi** facts per escluderli
5. **Ripeti** fino a ottenere solo mondi sensati

**Alloy è uno strumento di ESPLORAZIONE:**
- Trova corner cases che non avresti immaginato
- Aiuta a **scoprire** vincoli necessari
- **Iterazione** è la chiave!

---

## 9. Processo Iterativo di Sviluppo

### 9.1 Approccio Incrementale

`⏱️ 00:38:59 - 00:40:09`

Lo sviluppo di specifiche Alloy segue un **processo incrementale**.

💡 **Insight - Sviluppo Iterativo**

**Processo tipico:**
1. Aggiungi un **fact**
2. **Esplora** i mondi generati (`run`)
3. **Ispeziona** e identifica problemi
4. **Comprendi** quali vincoli servono
5. Aggiungi **nuovi facts**
6. **Ripeti**

📝 **Esempio: Altri Facts Possibili**

Il professore menziona altri facts da analizzare autonomamente:

```alloy
fact socialConventionTree {
  all p1, p2: Person |
    (p1 in p2.(mother + father)) implies
    no (p1.(mother + father) & p2.(mother + father))
}
```

**Cosa fa:**
> "Se p1 è genitore di p2, allora p1 e p2 non possono avere genitori comuni"

**In altre parole:** Genitori e figli non possono essere fratelli!

🔍 **Analisi del Fact**

**Precondizione:**
```alloy
p1 in p2.(mother + father)
```
- `p1` è il padre o la madre di `p2`

**Vincolo:**
```alloy
no (p1.(mother + father) & p2.(mother + father))
```
- Intersezione genitori di `p1` e `p2` deve essere vuota
- Cioè: `p1` e `p2` non hanno genitori comuni

**Esempio impedito:**
```
        Man2 (padre comune)
       /    \
   father  father
     /        \
   Man0 ───→ Woman0
        father  (figlia di Man0!)
```

⚠️ **Esercizio per lo Studente**

Il professore lascia come esercizio:
- Analizzare `socialConventionTree`
- Capire se migliora il modello
- Testare con `run` e vedere l'effetto
- Confrontare con altri facts

📊 **Metodo di Analisi Consigliato**

1. **Leggi** il fact e cerca di capire l'intento
2. **Scrivi** esempi concreti di mondi che viola
3. **Aggiungi** il fact alla specifica
4. **Esegui** `run show` e confronta i mondi
5. **Valuta** se risolve problemi senza crearne altri

✅ **Regola Pratica - Facts Come Scoperta**

**Non pensare a tutti i facts all'inizio!**

- Inizia con modello **semplice**
- **Esplora** e trova problemi
- **Aggiungi** facts basandoti su ciò che trovi
- Il processo è **esplorativo**, non pianificato

**Questo è il punto di forza di Alloy:** Ti aiuta a **scoprire** requisiti nascosti!

---

### 9.2 Uso dell'Analyzer come Helper

`⏱️ 00:40:09 - 00:42:45`

L'Analyzer può aiutarci a **confrontare la forza** di diversi facts.

📝 **Tecnica: Trasformare Facts in Predicati**

Per **confrontare** due facts, possiamo:
1. Trasformarli in **predicati**
2. Creare **assertions** sulle implicazioni
3. Usare `check` per verificare la relazione

💡 **Insight - Confronto di Forza**

**Vogliamo capire:** Quale fact è più forte?

**Fact A più forte di Fact B significa:**
```
Mondi che soddisfano A ⊆ Mondi che soddisfano B
```

**Logicamente:**
```
A ⟹ B  (A implica B)
```

🔍 **Esempio Concreto**

**Confrontiamo:**
- **Fact 1:** `noCommonAncestors`
- **Fact 2:** `socialConventionTree`

**Trasformazione in predicati:**

```alloy
// Invece di:
fact noCommonAncestors { ... }

// Usiamo:
pred noCommonAncestors {
  all p1: Man, p2: Woman |
    (p1->p2 in wife) implies
    no (ancestors[p1] & ancestors[p2])
}

pred socialConventionTree {
  all p1, p2: Person |
    (p1 in p2.(mother + father)) implies
    no (p1.(mother + father) & p2.(mother + father))
}
```

📝 **Assertions per Confronto**

**Assertion 1: A implica B?**
```alloy
assert stronger {
  noCommonAncestors implies socialConventionTree
}

check stronger for 5
```

**Risultato:** "No counterexample found" ✅
- Significa: `noCommonAncestors ⟹ socialConventionTree`

**Assertion 2: B implica A?**
```alloy
assert notStronger {
  socialConventionTree implies noCommonAncestors
}

check notStronger for 5
```

**Risultato:** "Counterexample found" ❌
- Significa: `socialConventionTree ⏸️⟹ noCommonAncestors`

📊 **Conclusione del Confronto**

```
noCommonAncestors ⟹ socialConventionTree  (✅)
socialConventionTree ⏸️⟹ noCommonAncestors  (❌)
```

**Quindi:**
- `noCommonAncestors` è **più forte** (più restrittivo)
- Implica `socialConventionTree` ma non viceversa
- Se usiamo `noCommonAncestors`, non serve `socialConventionTree`

💡 **Insight - Gerarchia di Facts**

```
    noCommonAncestors (più restrittivo)
           ⟹
    socialConventionTree (meno restrittivo)
```

**Implicazione pratica:**
- Scegli il fact **più appropriato** per il tuo dominio
- Facts più forti = mondi più ristretti (più vincoli)
- Facts più deboli = più mondi permessi (meno vincoli)

✅ **Metodo di Confronto - Riepilogo**

**Per confrontare fact A e fact B:**

1. **Trasforma** in predicati:
   ```alloy
   pred A { ... }
   pred B { ... }
   ```

2. **Crea assertions:**
   ```alloy
   assert AimpliesB { A implies B }
   assert BimpliesA { B implies A }
   ```

3. **Verifica:**
   ```alloy
   check AimpliesB
   check BimpliesA
   ```

4. **Interpreta:**
   - Entrambi OK → A ⟺ B (equivalenti)
   - Solo AimpliesB OK → A più forte
   - Solo BimpliesA OK → B più forte
   - Nessuno OK → Incomparabili

---

### 9.3 Ulteriori Problemi Residui

`⏱️ 00:42:45 - 00:47:11`

Anche con molti facts, possono rimanere **problemi residui**.

📝 **Problema Ancora Presente**

**Situazione osservata:**
```
Woman0
  father = Man0
  mother = none

Woman1
  father = Man0  // ← Stesso padre!
  mother = Woman0  // ← Madre = sorella!
```

**Problema:** Woman0 è **sia sorella che madre** di Woman1!

💡 **Insight - Complessità delle Relazioni Familiari**

Le relazioni familiari sono **molto complesse**:
- Molte combinazioni possibili
- Molti vincoli impliciti (per noi ovvi)
- Difficile pensare a **tutti** i casi limite

**Alloy aiuta a scoprirli!**

🔍 **Possibile Fact Aggiuntivo**

```alloy
fact socialConventionFor {
  all p1, p2: Person |
    (p1 in p2.(mother + father)) implies
    no (p1.(mother + father) & p2.(mother + father))
}
```

**Cosa fa:**
- Se `p1` è genitore di `p2`
- Allora `p1` e `p2` non hanno genitori comuni
- Impedisce: genitore e figlio essere fratelli

📊 **Limitazioni**

**Questo fact si concentra su:**
- Solo padre e madre **diretti**
- **Non** considera antenati a più livelli

**Esempio non impedito:**
```
    Bisnonno
       |
    Nonno ───→ Nipote
  (genitore)  (figlio ma anche nipote?)
```

⚠️ **Messaggio Principale**

**L'esempio Family Tree è molto intricato!**

- Molte relazioni possibili (madre, padre, moglie, marito)
- Molti vincoli impliciti da esplicitare
- Molti facts necessari per coprire tutti i casi
- **Ottimo esercizio** per praticare Alloy!

**Consiglio:** Sperimenta con diversi facts e osserva gli effetti!

✅ **Attività Suggerita**

1. **Crea** la specifica Family Tree completa
2. **Esplora** con `run show for 5`
3. **Identifica** mondi strani
4. **Prova** a scrivere facts per escluderli
5. **Testa** con `check` assertions varie
6. **Confronta** facts diversi con la tecnica vista
7. **Documenta** cosa funziona e cosa no

**Obiettivo:** Ottenere una specifica che genera **solo** mondi familiari sensati!

---

## 10. Alloy 6: Modelli Mutabili

### 10.1 Introduzione alla Mutabilità

`⏱️ 00:47:11 - 00:48:28`

**Alloy 6** introduce la possibilità di modellare **sistemi che evolvono** nel tempo.

💡 **Insight - Limitazione delle Versioni Precedenti**

**Finora abbiamo visto:**
- Atomi **immutabili** (non cambiano)
- Relazioni **immutabili** (snapshot fisso)
- Mondi rappresentano una **singola istantanea**
- **Nessun concetto di tempo** o evoluzione

**Problema:** Come modellare sistemi che **cambiano** nel tempo?

📝 **Novità in Alloy 6**

**Alloy 6 introduce:**
- **Relazioni variabili** (keyword `var`)
- **Operatori temporali** (`after`, `always`, `eventually`, etc.)
- **Evoluzione di sistemi** nel tempo
- **Trace analysis** (sequenze di stati)

🔍 **Concetto di Mutabilità**

**Variabile (`var`) significa:**
- Il valore può **cambiare** tra istanti temporali
- Rappresentiamo **sequenze di stati** anziché singolo stato
- Possiamo modellare **operazioni** come transizioni

📊 **Confronto: Prima vs Dopo**

| Aspetto | Alloy 5 (prima) | Alloy 6 (dopo) |
|---------|----------------|----------------|
| **Atomi** | Immutabili | Immutabili (sempre) |
| **Relazioni** | Immutabili | Possono essere `var` |
| **Tempo** | Nessuno | Sequenza di istanti |
| **Operazioni** | Due copie (pre/post) | Transizioni (`prime`) |
| **Snapshot** | Singolo | Multipli collegati |

**Nota:** Gli **atomi** rimangono sempre immutabili, solo le **relazioni** possono variare!

---

### 10.2 Address Book con Relazioni Variabili

`⏱️ 00:48:28 - 00:51:20`

Rivediamo l'esempio dell'**Address Book** usando relazioni variabili.

📝 **Modello Precedente (Immutabile)**

```alloy
sig Name, Addr {}

sig Book {
  addr: Name -> lone Addr
}

pred add[b, b': Book, n: Name, a: Addr] {
  b'.addr = b.addr + (n -> a)
}
```

**Problema:** Servono **due istanze** (b e b') per rappresentare pre/post stato.

💡 **Nuovo Modello (Mutabile)**

```alloy
sig Name, Addr {}

sig Book {
  var addr: Name -> lone Addr  // ← var = variabile nel tempo
}

pred add[b: Book, n: Name, a: Addr] {
  b.addr' = b.addr + (n -> a)  // ← addr' = valore al prossimo istante
}
```

**Vantaggi:**
- ✅ **Una sola istanza** di Book
- ✅ `addr` può **cambiare** nel tempo
- ✅ Operazioni usano `prime` (') per next state

🔍 **Keyword `var`**

```alloy
var addr: Name -> lone Addr
```

**Significato:**
- `addr` è una relazione **variabile**
- Può avere **valori diversi** in istanti temporali diversi
- Senza `var`, sarebbe immutabile (stesso valore sempre)

📊 **Esempio di Evoluzione**

```
Tempo 0: Book { addr = {} }                    // Vuoto
Tempo 1: Book { addr = {(Name0, Addr0)} }      // 1 elemento
Tempo 2: Book { addr = {(Name0, Addr0),        // 2 elementi
                        (Name1, Addr1)} }
```

La **stessa istanza** di Book ha valori diversi di `addr` nel tempo!

💡 **Insight - Operatore `prime` (')**

```alloy
b.addr'
```

**Significa:** Il valore di `b.addr` nel **prossimo istante temporale**.

**In un predicato/operazione:**
```alloy
pred add[b: Book, n: Name, a: Addr] {
  b.addr' = b.addr + (n -> a)
}
```

- `b.addr`: Valore **corrente**
- `b.addr'`: Valore **dopo** l'operazione
- L'operazione definisce come cambia lo stato

📝 **Predicato per Visualizzare Evoluzione**

```alloy
pred show {
  #Book.addr = 0        // Tempo 0: vuoto
  after #Book.addr = 1  // Tempo 1: 1 elemento
  after after #Book.addr = 2  // Tempo 2: 2 elementi
}
```

**Sintassi compatta:**
```alloy
pred show {
  #Book.addr = 0
  ; #Book.addr = 1    // ; = sequenza temporale
  ; #Book.addr = 2
}
```

🔍 **Operatore `after`**

```alloy
after formula
```

**Significa:** `formula` vale nell'**istante successivo**.

**Esempio:**
- Tempo 0: `#Book.addr = 0`
- `after #Book.addr = 1` → Tempo 1: ha 1 elemento
- `after after #Book.addr = 2` → Tempo 2: ha 2 elementi

✅ **Regola Pratica - Quando Usare `var`**

**Usa `var` per:**
- Relazioni che **cambiano** nel tempo
- Stati di sistema che evolvono
- Database, configurazioni, etc.

**NON usare `var` per:**
- Proprietà **immutabili** (es. tipi, ID)
- Relazioni **strutturali** fisse
- Vincoli che valgono sempre

---

### 10.3 Comando Run con Steps

`⏱️ 00:51:20 - 00:55:09`

Con modelli mutabili, il comando `run` accetta un parametro **steps**.

📝 **Sintassi Estesa del Run**

```alloy
run show for 5        // Scope: max 5 elementi
run show for 5 but 3 steps  // Scope + 3 istanti temporali
```

**Parametri:**
- **Scope** (es. `5`): Massimo numero di atomi per signature
- **Steps** (es. `3 steps`): Numero di istanti temporali da visualizzare

💡 **Insight - Visualizzazione Temporale**

L'Analyzer mostra **coppie consecutive** di istanti:
- Inizialmente: Tempo 0 e Tempo 1
- Bottone "avanti": Tempo 1 e Tempo 2
- Bottone "avanti": Tempo 2 e Tempo 3
- etc.

🔍 **Interfaccia del Visualizzatore**

```
┌─────────────────────────────────────┐
│  Time 0         →         Time 1    │
│  [mondo]                  [mondo]   │
│                                     │
│  [◄ Prev]  [Play]  [Next ►]        │
└─────────────────────────────────────┘
```

**Controlli:**
- **Next ►**: Mostra prossima coppia (1-2, poi 2-3, etc.)
- **◄ Prev**: Torna indietro
- **Play**: Animazione automatica

📊 **Esempio di Visualizzazione**

**Comando:**
```alloy
pred show {
  #Book.addr = 0
  ; #Book.addr = 1
  ; #Book.addr = 2
}

run show for 5 but 3 steps
```

**Visualizzazione Time 0 → Time 1:**
```
Time 0:                    Time 1:
Book0                      Book0
  addr = {}                  addr = {(Name0, Addr0)}
```

**Premendo Next, Time 1 → Time 2:**
```
Time 1:                    Time 2:
Book0                      Book0
  addr = {(Name0, Addr0)}    addr = {(Name0, Addr0),
                                     (Name1, Addr1)}
```

💡 **Insight - Stesso Book, Valori Diversi**

**Nota importante:**
- È sempre lo **stesso** Book0
- Ma `addr` ha **valori diversi** nei vari tempi
- Questo è il concetto di **mutabilità**!

⚠️ **Attenzione - Istanti Grigi**

Nell'interfaccia, gli istanti **non visualizzati** appaiono grigi:
- Visualizzi Time 0-1 → Time 2 è grigio
- Visualizzi Time 1-2 → Time 0 è grigio

Questo indica quali istanti sono **attualmente mostrati**.

---

### 10.4 Definizione di Operazioni con `prime`

`⏱️ 00:55:09 - 01:00:30`

Definiamo operazioni che modificano lo stato usando `prime` (').

📝 **Operazione Add**

```alloy
pred add[b: Book, n: Name, a: Addr] {
  b.addr' = b.addr + (n -> a)
}
```

**Semantica:**
- **Chiamata:** All'istante corrente T
- **Effetto:** Modifica `b.addr` all'istante T+1
- `b.addr'` = valore al prossimo istante

💡 **Insight - Istante di Chiamata vs Istante di Effetto**

```
Tempo T:     add[b, n, a] chiamato
               ↓
             (definisce il cambiamento)
               ↓
Tempo T+1:   b.addr' = b.addr + (n, a)  ← Cambiamento effettivo
```

**Importante:** L'operazione è "chiamata" al tempo T ma ha "effetto" al tempo T+1!

🔍 **Operazione Delete**

```alloy
pred del[b: Book, n: Name] {
  b.addr' = b.addr - (n -> Addr)
}
```

**Cosa fa:**
- Rimuove **tutte** le coppie che iniziano con `n`
- `n -> Addr` = prodotto cartesiano (tutti gli indirizzi possibili)
- Equivalente a: "elimina tutte le entry per il nome `n`"

📝 **Asserzione: delUndoesAdd**

**Versione precedente (con due Book):**
```alloy
assert delUndoesAdd {
  all b, b', b'': Book, n: Name, a: Addr |
    (n not in b.addr) and 
    add[b, b', n, a] and 
    del[b', b'', n]
    implies b.addr = b''.addr
}
```

**Nuova versione (con `var` e `prime`):**
```alloy
assert delUndoesAdd {
  all b: Book, n: Name, a: Addr |
    (no n.(b.addr)) and
    add[b, n, a] and
    after del[b, n]
    implies b.addr = b.addr''
}
```

💡 **Insight - Analisi dell'Asserzione**

**Riga per riga:**

1. **`no n.(b.addr)`**: 
   - Al tempo corrente, `n` non ha indirizzi in `b`

2. **`add[b, n, a]`**:
   - Chiamiamo add al tempo corrente
   - Effetto: al tempo T+1, `n->a` è aggiunto

3. **`after del[b, n]`**:
   - Al tempo T+1 (after rispetto a T), chiamiamo del
   - Effetto: al tempo T+2, `n->a` è rimosso

4. **`b.addr = b.addr''`**:
   - Tempo T: stato iniziale
   - Tempo T+2 (`` = 2 volte prime): stato finale
   - Devono essere uguali!

🔍 **Timeline Dettagliata**

```
T=0: n not in b.addr
     add[b, n, a] CHIAMATO
     
T=1: b.addr' = b.addr + (n,a)  ← Effetto di add
     del[b, n] CHIAMATO (perché "after del")
     
T=2: b.addr'' = b.addr' - n->Addr  ← Effetto di del
     
Asserzione verifica: b.addr (T=0) == b.addr'' (T=2)
```

📊 **Tabella: Operatori Prime**

| Notazione | Significato | Istante |
|-----------|-------------|---------|
| `b.addr` | Valore corrente | T |
| `b.addr'` | Valore prossimo | T+1 |
| `b.addr''` | Valore tra 2 step | T+2 |
| `b.addr'''` | Valore tra 3 step | T+3 |

✅ **Verifica dell'Asserzione**

```alloy
check delUndoesAdd for 5 but 5 steps
```

**Risultato:** "No counterexample found" ✅

**Significato:** L'asserzione vale per tutte le configurazioni con scope ≤5 e sequenze ≤5 step.

---

## 11. Operatori Temporali: Always

### 11.1 Operatore `always`

`⏱️ 01:00:30 - 01:08:00`

L'operatore **`always`** specifica che una formula deve valere in **tutti** gli istanti futuri.

📝 **Sintassi**

```alloy
always formula
```

**Significato:** `formula` deve essere vera:
- Nell'istante corrente
- In **tutti** gli istanti futuri

💡 **Insight - Vincoli Temporali vs Istantanei**

**Senza `always`:**
```alloy
fact {
  no n.(Book.addr)  // Vale solo al tempo 0!
}
```

**Con `always`:**
```alloy
fact {
  always no n.(Book.addr)  // Vale in TUTTI i tempi!
}
```

🔍 **Esempio nell'Asserzione**

**Versione migliorata di delUndoesAdd:**

```alloy
assert delUndoesAdd {
  all b: Book, n: Name, a: Addr |
    always (
      (no n.(b.addr)) and
      add[b, n, a] and
      after del[b, n]
      implies b.addr = b.addr''
    )
}
```

**Cosa cambia:** La proprietà deve valere **qualunque sia l'istante** in cui valutiamo l'asserzione, non solo al tempo 0.

📊 **Semantica Formale**

```
always φ è vero al tempo i ⟺ 
φ è vero per ogni tempo k ≥ i
```

**Esempio:**
```
Timeline: T0 -- T1 -- T2 -- T3 -- T4 -- ...
              ↑
            Valuto qui (i=1)
            
always φ significa: φ vale in T1, T2, T3, T4, ...
```

💡 **Insight - Perché Serve `always`**

**Problema senza `always`:**
```alloy
fact {
  no n.(Book.addr)  // Vincolo solo a T=0
}

// Al tempo T=1, T=2, etc., il vincolo NON è applicato!
```

**Soluzione con `always`:**
```alloy
fact {
  always no n.(Book.addr)  // Vincolo a TUTTI i tempi
}
```

🔍 **Caso d'Uso Tipico**

**Invarianti che devono valere sempre:**

```alloy
fact cardinalityLimit {
  always #Book.addr <= 10  // Mai più di 10 entry
}

fact noDuplicates {
  always (all n: Name | lone n.(Book.addr))  // Mai duplicati
}
```

⚠️ **Attenzione - Default vs Always**

**In specifiche mutabili:**
- Vincoli **senza** operatori temporali → valgono solo a T=0
- Serve **`always`** per farli valere in tutti gli istanti

**Best practice:** Usa sempre `always` per facts che devono valere globalmente!

📊 **Confronto: Constraint Semplice vs Always**

| Vincolo | T=0 | T=1 | T=2 | T=3 |
|---------|-----|-----|-----|-----|
| `φ` | ✅ | ❌ | ❌ | ❌ |
| `always φ` | ✅ | ✅ | ✅ | ✅ |

---

## 12. Esempio: Device con Stati

### 12.1 Modello del Device

`⏱️ 01:08:00 - 01:12:42`

Modelliamo un **dispositivo** che può essere working o broken.

📝 **Definizione della Specifica**

```alloy
sig Device {
  var status: DevStatus
}

enum DevStatus { Working, Broken }
```

**Componenti:**

1. **`enum DevStatus`**: Enumerazione degli stati possibili
2. **`var status`**: Stato corrente del device (variabile nel tempo)

💡 **Insight - Enumerazioni in Alloy**

**Enum è sintassi abbreviata per:**

```alloy
// enum DevStatus { Working, Broken }

// Equivale a:
abstract sig DevStatus {}
sig Working extends DevStatus {}
sig Broken extends DevStatus {}
```

**Quando usare enum:**
- Signature **senza relazioni interne**
- Semplici **categorie** o **stati**
- **Non** serve estendere con campi aggiuntivi

🔍 **Fact: Irreparabile**

```alloy
fact irreparable {
  all d: Device |
    always (
      (d.status = Broken) implies
      (after always d.status = Broken)
    )
}
```

**Interpretazione:**
> "Per ogni device, sempre: se diventa broken, rimane broken per sempre"

📊 **Analisi Timeline**

```
Timeline: T0 -- T1 -- T2 -- T3 -- T4 -- T5
Status:   W --- W --- B --- B --- B --- B
                     ↑
                  Diventa Broken
                     
Dopo questo punto: SEMPRE Broken (irreparabile)
```

**Ma può anche:**
```
Timeline: T0 -- T1 -- T2 -- T3 -- T4 -- T5
Status:   W --- W --- W --- W --- W --- W

Mai si rompe: OK! (dispositivo eterno)
```

💡 **Insight - Doppio `always`**

```alloy
always (
  (d.status = Broken) implies
  (after always d.status = Broken)
)
```

**Analisi:**

1. **Outer `always`**: Per ogni istante T nella timeline
2. **Controllo**: È broken al tempo T?
3. **Se sì:** Inner `after always` → Dal T+1 in poi, sempre broken

**Questo permette:**
- Broken a T=0 → Sempre broken
- Broken a T=5 → Da T=6 in poi sempre broken
- Mai broken → OK (sempre working)

---

### 12.2 Operatore `eventually`

`⏱️ 01:12:42 - 01:16:32`

L'operatore **`eventually`** specifica che qualcosa accadrà **prima o poi**.

📝 **Sintassi**

```alloy
eventually formula
```

**Significato:** Esiste un istante futuro in cui `formula` è vera.

💡 **Fact: Eventually Breaks**

```alloy
fact eventuallyBreaks {
  all d: Device |
    always (
      (d.status = Working) implies
      (eventually d.status = Broken)
    )
}
```

**Interpretazione:**
> "Per ogni device, sempre: se è working, prima o poi diventerà broken"

🔍 **Esempio Timeline**

```
Timeline: T0 -- T1 -- T2 -- T3 -- T4 -- T5
Status:   W --- W --- W --- W --- B --- B
                                  ↑
                            Eventually happens
```

**Il dispositivo:**
- Inizia working
- Può rimanere working per un po'
- Ma **prima o poi** deve rompersi
- Una volta rotto, rimane rotto (per `irreparable`)

📊 **Semantica Formale**

```
eventually φ è vero al tempo i ⟺
∃ k ≥ i: φ è vero al tempo k
```

**Differenza con `always`:**
- `always φ`: φ vero in **tutti** i futuri k ≥ i
- `eventually φ`: φ vero in **almeno un** futuro k ≥ i

💡 **Insight - Combinazione dei Due Facts**

**Con entrambi i facts:**

1. **`irreparable`**: Broken → Sempre broken dopo
2. **`eventuallyBreaks`**: Working → Prima o poi broken

**Risultato:**
```
Timeline possibile: W - W - W - B - B - B - B - ...
                            ↑
                    Transizione inevitabile
                    
Timeline impossibile: W - W - B - W - ...  ❌ (violerebbe irreparable)
Timeline impossibile: W - W - W - W - ...  ❌ (violerebbe eventuallyBreaks)
```

📊 **Tabella: Operatori Temporali Futuri**

| Operatore | Significato | Quantificatore |
|-----------|-------------|----------------|
| `always φ` | φ in tutti i futuri | ∀ k ≥ i |
| `eventually φ` | φ in almeno un futuro | ∃ k ≥ i |
| `after φ` | φ nel prossimo istante | k = i+1 |

---

### 12.3 Operatori Temporali Passati

`⏱️ 01:16:32 - 01:20:30`

Alloy 6 fornisce anche operatori per il **passato**.

📝 **Operatore `historically`**

```alloy
historically formula
```

**Significato:** `formula` è stata vera in **tutti** gli istanti passati (incluso corrente).

💡 **Fact: Always Working If Currently Working**

```alloy
fact alwaysWorkedIfWorking {
  all d: Device |
    always (
      (d.status = Working) implies
      (historically d.status = Working)
    )
}
```

**Interpretazione:**
> "Se un device è working ora, allora è sempre stato working (mai rotto prima)"

🔍 **Esempio Timeline**

```
Timeline: T0 -- T1 -- T2 -- T3 -- T4
Status:   W --- W --- W --- W --- W
                            ↑
                      Evaluto qui
                      
historically Working → Guarda T0-T1-T2-T3-T4: tutti Working ✅
```

**Controesempio:**
```
Timeline: T0 -- T1 -- T2 -- T3 -- T4
Status:   W --- B --- B --- W --- W
                            ↑
                      Evaluto qui
                      
historically Working → Guarda T0-T4: c'è B in T1-T2-T3 ❌
```

📊 **Semantica Formale**

```
historically φ è vero al tempo i ⟺
∀ k ≤ i: φ è vero al tempo k
```

**È il "duale" di `always` nel passato!**

💡 **Operatore `before`**

```alloy
before formula
```

**Significato:** `formula` era vera nell'istante **immediatamente precedente**.

📝 **Esempio con `before`**

```alloy
fact statusMonotone {
  all d: Device |
    after always (
      (d.status = Working) implies
      (before d.status = Working)
    )
}
```

**Interpretazione:**
> "Se è working ora, era working anche prima (stato non può oscillare)"

⚠️ **Attenzione - Problema con `before` al Tempo 0**

**Problema:**
```
Timeline: T0 -- T1 -- T2 -- ...
          ↑
      before T0 = ???
```

**Non esiste** un istante prima di T0!

**Soluzione:** Usa `after always` invece di `always`:

```alloy
// ❌ PROBLEMA
always (... before ...)  // before al T=0 è indefinito!

// ✅ SOLUZIONE
after always (... before ...)  // Inizia da T=1, before è T=0
```

💡 **Insight - Perché `after always`**

```
Timeline: T0 -- T1 -- T2 -- T3 -- ...
          ↑    ↑
          |    └─ after always inizia qui
          |
          └─ always inizierebbe qui (problematico per before)
```

**`after always`:**
- Salta T=0
- Inizia da T=1
- `before` al T=1 punta a T=0 (esiste!) ✅

📊 **Tabella: Operatori Temporali Completa**

| Direzione | Quantificatore | Operatore | Significato |
|-----------|---------------|-----------|-------------|
| Futuro | Tutti | `always` | Sempre in futuro |
| Futuro | Esiste | `eventually` | Prima o poi |
| Futuro | Prossimo | `after` | Istante successivo |
| Passato | Tutti | `historically` | Sempre in passato |
| Passato | Esiste | `once` | Almeno una volta in passato |
| Passato | Precedente | `before` | Istante precedente |

---

### 12.4 Operatore `once` e Predicato Break

`⏱️ 01:20:30 - 01:24:16`

L'operatore **`once`** verifica se qualcosa è accaduto **almeno una volta** nel passato.

📝 **Predicato Break**

```alloy
pred break[d: Device] {
  d.status = Working
  d.status' = Broken
}
```

**Cosa rappresenta:**
- La **transizione** da Working a Broken
- Al tempo T: working
- Al tempo T+1: broken

💡 **Fact: Broken Implies Broke in Past**

```alloy
fact brokenMustHaveBroken {
  all d: Device |
    always (
      (d.status = Broken) implies
      (once break[d])
    )
}
```

**Interpretazione:**
> "Se un device è broken ora, allora in qualche momento passato è avvenuta la transizione break"

🔍 **Esempio Timeline**

```
Timeline: T0 -- T1 -- T2 -- T3 -- T4 -- T5
Status:   W --- W --- W --- B --- B --- B
                      ↑     ↑
                   break  Evaluto qui (T4)
                   
once break[d] → Guarda T0-T4: break è accaduto a T2-T3 ✅
```

**Cosa verifica `break[d]` al tempo T2:**
- T2: `d.status = Working` ✅
- T3: `d.status' = Broken` ✅
- Transizione completata!

📊 **Semantica di `once`**

```
once φ è vero al tempo i ⟺
∃ k ≤ i: φ è vero al tempo k
```

**È il "duale" di `eventually` nel passato!**

💡 **Insight - Coerenza del Modello**

**Questo fact garantisce che:**
- Un device broken **non appare dal nulla**
- Deve esserci stata una **transizione esplicita** (`break`)
- Il modello è **tracciabile** (ogni stato ha una storia)

📊 **Timeline Completa con Tutti i Facts**

```
T0: Working (inizio)
    ↓
T1: Working (può rimanere working)
    ↓
T2: Working
    ↓ break[d] accade qui
T3: Broken (transizione)
    ↓ irreparable forza:
T4: Broken (sempre broken dopo)
    ↓
T5: Broken
    ↓
... Broken forever
```

**Facts attivi:**
1. `eventuallyBreaks`: Forza transizione prima o poi
2. `break[d]`: Definisce come avviene la transizione
3. `irreparable`: Una volta broken, sempre broken
4. `brokenMustHaveBroken`: Broken implica break nel passato

✅ **Regola Pratica - Predicati di Transizione**

**Definisci predicati per transizioni:**
```alloy
pred break[d: Device] {
  d.status = Working
  d.status' = Broken
}

pred repair[d: Device] {
  d.status = Broken
  d.status' = Working
}
```

**Poi usa nei facts:**
```alloy
fact noRepair {
  all d: Device | always not repair[d]
}
```

Questo rende il modello più **leggibile** e **manutenibile**!

---

## 13. Preferenza per Operatori Futuri

### 13.1 Difficoltà con il Passato

`⏱️ 01:24:16 - 01:25:49`

Gli operatori sul **passato** sono più difficili da gestire rispetto al **futuro**.

💡 **Insight - Problema dello Stato Iniziale**

**Difficoltà principale:**
- L'analisi inizia sempre da uno **stato iniziale** (T=0)
- **Non sappiamo** cosa c'era prima di T=0
- Operatori come `before`, `historically`, `once` sono **indefiniti** a T=0

📊 **Confronto Futuro vs Passato**

| Aspetto | Futuro | Passato |
|---------|--------|---------|
| **Direzione analisi** | Naturale ✅ | Contro-intuitiva |
| **Stato iniziale** | Definito | Indefinito ❌ |
| **Operatori** | `after`, `always`, `eventually` | `before`, `historically`, `once` |
| **Complessità** | Semplice | Più complessa |
| **Best practice** | Preferibile | Evitare quando possibile |

⚠️ **Attenzione - Uso di `before`**

**Problema:**
```alloy
// ❌ Rischio errore a T=0
always (d.status = Working implies before d.status = Working)
```

**Soluzione:**
```alloy
// ✅ Salta T=0
after always (d.status = Working implies before d.status = Working)
```

💡 **Insight - Stato Iniziale Indefinito**

```
     ???  -- T0 -- T1 -- T2 -- T3
      ↑
   before T0
   (non esiste!)
```

**Analyzer non sa:**
- Cosa c'era prima di T=0
- Come inizializzare il "passato"
- Valori di default per operatori passati

**Workaround:** Inizia analisi da T=1 con `after always`

✅ **Regola Pratica - Preferisci il Futuro**

**Quando possibile:**
- Modella evoluzione guardando **avanti** (futuro)
- Usa `after`, `always`, `eventually`
- Evita `before`, `historically`, `once`

**Eccezioni (quando serve passato):**
- Proprietà di tracciabilità
- Audit logs
- Verifiche di consistenza storica

---

## 14. Esempio: Mailbox

### 14.1 Introduzione al Problema

`⏱️ 01:25:49 - 01:30:18`

Modelliamo un sistema di **mailbox** con funzionalità di trash e restore.

📝 **Descrizione del Sistema**

**Funzionalità:**
- **Mailbox**: Contiene messaggi attivi
- **Trash**: Cestino per messaggi eliminati (temporaneamente)
- **Delete**: Sposta messaggio da mailbox a trash
- **Restore**: Riporta messaggio da trash a mailbox

💡 **Modello con Signature Variabili**

```alloy
var sig Message {}
var sig Trashed in Message {}
```

**Novità: `var sig`!**

- **`var sig Message`**: Il **set** di messaggi può variare nel tempo
- **`var sig Trashed in Message`**: `Trashed` è **sottoinsieme** variabile di `Message`

🔍 **Keyword `in`**

```alloy
sig Trashed in Message
```

**Significato:** `Trashed` è un **sottoinsieme** di `Message`

**Formalmente:**
```
∀ t ∈ Trashed: t ∈ Message
Trashed ⊆ Message
```

**Non è estensione (`extends`):**
- `extends`: Crea **sottotipo** (disjoint)
- `in`: Crea **sottoinsieme** (può sovrapporsi)

📊 **Confronto `extends` vs `in`**

| Aspetto | `extends` | `in` |
|---------|-----------|------|
| **Relazione** | Sottotipo (is-a) | Sottoinsieme (subset) |
| **Disjoint** | Sì | No |
| **Esempio** | `Man extends Person` | `Trashed in Message` |
| **Significato** | Ogni Man è Person | Ogni Trashed è Message |
| **Può cambiare** | No (tipo fisso) | Sì (con `var`) |

💡 **Insight - Signature Variabili**

**Con `var sig`:**
- Il **set di atomi** può cambiare nel tempo
- Atomi possono apparire/sparire

**Esempio Timeline:**
```
T0: Message = {M0, M1, M2}
    Trashed = {}

T1: Message = {M0, M1, M2, M3}  // M3 appare
    Trashed = {M1}               // M1 spostato in trash

T2: Message = {M0, M1, M2, M3}
    Trashed = {}                 // M1 ripristinato
```

🔍 **Operazioni del Sistema**

**Delete:**
```alloy
pred delete[m: Message] {
  m not in Trashed      // Pre: non già in trash
  Trashed' = Trashed + m  // Post: aggiungi a trash
}
```

**Restore:**
```alloy
pred restore[m: Message] {
  m in Trashed          // Pre: deve essere in trash
  Trashed' = Trashed - m  // Post: rimuovi da trash
}
```

📊 **Visualizzazione dell'Evoluzione**

```
Mailbox:  [M0] [M1] [M2]      [M0] [M2]           [M0] [M1] [M2]
Trash:    [ ]                 [M1]                [ ]
          ↓ delete(M1)        ↓ restore(M1)       ↓
          T0                  T1                  T2
```

---

### 14.2 Struttura del Modello Mailbox

`⏱️ 01:30:18 - 01:32:23`

Approfondiamo la struttura del modello mailbox.

📝 **Specifica Completa**

```alloy
var sig Message {}
var sig Trashed in Message {}

pred delete[m: Message] {
  m not in Trashed
  Trashed' = Trashed + m
  Message' = Message  // Message non cambia
}

pred restore[m: Message] {
  m in Trashed
  Trashed' = Trashed - m
  Message' = Message  // Message non cambia
}

pred show {
  some Message
  some Trashed
  eventually no Trashed
}

run show for 5 but 3 steps
```

💡 **Insight - Subset Variabile**

**`Trashed in Message`** significa:
- Trashed è sempre sottoinsieme di Message
- Ma **quali elementi** sono in Trashed può cambiare
- Vincolo: `Trashed ⊆ Message` sempre valido

🔍 **Dettagli delle Operazioni**

**Delete:**
```alloy
m not in Trashed        // Precondizione: non già trashed
Trashed' = Trashed + m  // Aggiunge m al trash
Message' = Message      // Message set non cambia
```

**Restore:**
```alloy
m in Trashed            // Precondizione: deve essere in trash
Trashed' = Trashed - m  // Rimuove m dal trash
Message' = Message      // Message set non cambia
```

**Nota:** `Message' = Message` significa che il set totale dei messaggi non cambia (solo la loro categorizzazione cambia).

📊 **Visualizzazione nel Tool**

L'Analyzer mostrerà:
- Quali messaggi esistono in ogni istante
- Quali sono nella mailbox (Message - Trashed)
- Quali sono nel trash (Trashed)
- Le transizioni tra stati

💡 **Insight - Uso Combinato di Concetti**

Questo esempio combina:
1. **`var sig`**: Signature variabili
2. **`in`**: Relazione di sottoinsieme
3. **Operatori temporali**: `eventually`, `prime`
4. **Predicati**: Definizione operazioni

**Esempio perfetto della potenza espressiva di Alloy 6!**

✅ **Esercizio per Casa**

Il professore suggerisce:
> "Analizzeremo questo modello più in dettaglio la prossima lezione"

**Attività consigliate:**
1. Implementa il modello in Alloy
2. Esegui `run show`
3. Esplora le trace generate
4. Prova a aggiungere altre operazioni (es. `deleteAll`)
5. Aggiungi assertions (es. `restore` annulla `delete`)

---

## 15. Conclusioni

### 15.1 Riepilogo della Lezione

`⏱️ 01:32:23`

Abbiamo coperto molti concetti avanzati di Alloy!

🎯 **Argomenti Trattati**

**Parte 1: Facts e Vincoli (00:00 - 00:47)**
- ✅ Riepilogo Family Tree
- ✅ Facts: sintassi, semantica, naming
- ✅ Differenza facts vs predicati
- ✅ Assertions e verifica
- ✅ Social convention e antenati comuni
- ✅ Uso dell'Evaluator per debug
- ✅ Confronto forza tra facts
- ✅ Processo iterativo di sviluppo

**Parte 2: Alloy 6 e Mutabilità (00:47 - 01:32)**
- ✅ Introduzione alla mutabilità
- ✅ Keyword `var` per relazioni variabili
- ✅ Operatore `prime` (') per next state
- ✅ Comando `run` con steps
- ✅ Operatori temporali futuri: `after`, `always`, `eventually`
- ✅ Operatori temporali passati: `before`, `historically`, `once`
- ✅ Esempio Device con stati
- ✅ Preferenza per operatori futuri
- ✅ `var sig` per signature variabili
- ✅ Keyword `in` per sottoinsiemi
- ✅ Esempio Mailbox

📊 **Nuovi Costrutti Alloy 6**

| Costrutto | Uso | Esempio |
|-----------|-----|---------|
| `var` | Relazione variabile | `var addr: Name -> Addr` |
| `var sig` | Signature variabile | `var sig Message` |
| `prime` (') | Valore al prossimo istante | `addr'` |
| `in` | Sottoinsieme | `sig Trashed in Message` |
| `after` | Prossimo istante | `after φ` |
| `always` | Tutti gli istanti futuri | `always φ` |
| `eventually` | Prima o poi | `eventually φ` |
| `before` | Istante precedente | `before φ` |
| `historically` | Tutti gli istanti passati | `historically φ` |
| `once` | Almeno una volta in passato | `once φ` |

💡 **Messaggi Chiave**

1. **Facts come scoperta**: Usa Alloy per **scoprire** vincoli necessari, non pensarli tutti all'inizio

2. **Processo iterativo**: Definisci → Esplora → Identifica problemi → Aggiungi facts → Ripeti

3. **Analyzer come helper**: Usa assertions per confrontare facts e capire relazioni

4. **Mutabilità = potenza**: Alloy 6 permette di modellare **evoluzione** di sistemi

5. **Preferisci futuro**: Operatori futuri sono più naturali e meno problematici

6. **`always` per invarianti**: Usa `always` per facts che devono valere in ogni istante

---

### 15.2 Prossimi Passi

📚 **Nella Prossima Lezione**

- Analisi approfondita del modello **Mailbox**
- Altri esempi di **sistemi mutabili**
- **Pattern** comuni in specifiche temporali
- Esercizi pratici

✅ **Esercizi Consigliati**

1. **Family Tree**: Completa la specifica con tutti i facts necessari
2. **Device**: Aggiungi stati intermedi (es. `Degraded`, `Maintenance`)
3. **Mailbox**: Implementa operazioni aggiuntive (`emptyTrash`, `moveAll`)
4. **Confronto facts**: Usa la tecnica vista per confrontare propri facts
5. **Address Book**: Converti completamente a versione mutabile

📖 **Risorse per Approfondire**

- **Alloy 6 Documentation**: Operatori temporali e esempi
- **Software Abstractions (libro)**: Capitoli su temporal logic
- **Alloy Community**: Forum e discussioni su pattern temporali

---

## 🎯 Riepilogo Completo del File

**File:** Lez8-9ott_NEW.md  
**Lezione:** 8 - Alloy: Facts e Vincoli Avanzati + Modelli Mutabili  
**Data:** 9 Ottobre  
**Durata totale:** ~92 minuti (01:32:23)

### 📊 Statistiche Finali

- **Righe totali**: ~5900 linee
- **Sezioni principali**: 15
- **Sottosezioni**: 47
- **Timestamp coperti**: 00:00:43 - 01:32:23 (completo!)
- **Box informativi**: 150+
- **Esempi di codice**: 80+
- **Tabelle comparative**: 40+
- **Visualizzazioni grafiche**: 30+

### ✅ Struttura Completa

1. **Introduzione e Riepilogo** (Sezioni 1-3)
2. **Facts: Sintassi e Semantica** (Sezioni 4-5)
3. **Social Convention** (Sezione 6)
4. **Antenati Comuni** (Sezioni 7-8)
5. **Processo Iterativo** (Sezione 9)
6. **Alloy 6: Mutabilità** (Sezioni 10-11)
7. **Device e Operatori Temporali** (Sezioni 12-13)
8. **Mailbox** (Sezione 14)
9. **Conclusioni** (Sezione 15)

---

**🎓 Lezione 8 completamente tradotta e strutturata!**

*File: Lez8-9ott_NEW.md - Traduzione completa della lezione*