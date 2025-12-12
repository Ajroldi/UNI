# Ricerca Operativa - Lezione 3

**Data:** 23 Settembre | **Prof. Federico Malucelli** | *Politecnico di Milano*

## Obiettivo della Lezione

Imparare a **modellare i vincoli** in Programmazione Lineare:
- Riconoscere i principali tipi di vincoli
- Trasformarli in forma standard senza cambiare il problema
- Collegare vincoli logici e di flusso con esempi concreti

## 1. Ripasso: Problema dello Zaino (Knapsack)

**Setup:**
- Oggetti con valore `vᵢ` e peso `aᵢ`
- Capacità massima `b`
- Variabili: `xᵢ ∈ {0,1}` (prendo o non prendo l'oggetto)

**Modello matematico:**
- Vincolo di capacità: `Σᵢ aᵢxᵢ ≤ b`
- Obiettivo: `max Σᵢ vᵢxᵢ`

**Messaggio chiave:** Dal testo narrativo → variabili + vincoli + funzione obiettivo

## 2. Tipologie di Vincoli

### Vincoli di Disponibilità (≤)
Limitano l'uso di **risorse scarse**:
- Forma: `Σⱼ aⱼxⱼ ≤ b`
- Esempi: ore macchina, capacità magazzino, budget massimo

### Vincoli di Fabbisogno (≥)
Impongono un **minimo da raggiungere**:
- Forma: `Σⱼ aⱼxⱼ ≥ b`
- Esempi: apporto minimo calorie, domanda minima, livello di servizio

### Perché la distinzione è importante
Algebricamente posso trasformare un tipo nell'altro, ma **l'interpretazione cambia**. Capire se esprimiamo "non superare" o "garantire almeno" è essenziale.

## 3. Diet Problem (Problema della Dieta)

**Applicazione classica:** Scelta di alimenti per minimizzare il costo rispettando vincoli nutrizionali.

**Esempio: Poke Bowl**
- Variabili: `xⱼ ≥ 0` = quantità del cibo j
- Dati: costo `cⱼ`, contenuti nutrizionali `aᵢⱼ`, requisiti minimi `bᵢ`

**Modello:**
```
min Σⱼ cⱼxⱼ
s.t. Σⱼ aᵢⱼxⱼ ≥ bᵢ  ∀i (vincoli nutrizionali)
     xⱼ ≥ 0
```

**Vincolo esempio:** Almeno 700 calorie
```
300·x_riso + 250·x_cereali + 35·x_lattuga + ... ≥ 700
```

## 4. Trasformazioni di Vincoli ("Black Box")

**Scopo:** Portare qualsiasi modello nella forma richiesta dal solver senza cambiare le soluzioni possibili.

### Da ≥ a ≤
```
Σⱼ aᵢⱼxⱼ ≥ bᵢ  ⟺  -Σⱼ aᵢⱼxⱼ ≤ -bᵢ
```

### Uguaglianza come due disuguaglianze
```
Σⱼ aᵢⱼxⱼ = bᵢ  ⟺  {Σⱼ aᵢⱼxⱼ ≤ bᵢ
                   {Σⱼ aᵢⱼxⱼ ≥ bᵢ
```

### Variabili Slack/Surplus
**Per vincoli ≤:**
```
Σⱼ aᵢⱼxⱼ ≤ bᵢ  ⟺  Σⱼ aᵢⱼxⱼ + sᵢ = bᵢ, sᵢ ≥ 0
```

**Per vincoli ≥:**
```
Σⱼ aᵢⱼxⱼ ≥ bᵢ  ⟺  Σⱼ aᵢⱼxⱼ - tᵢ = bᵢ, tᵢ ≥ 0
```

**Interpretazione:** Lo slack misura "quanto margine rimane" prima di saturare il vincolo.

## 5. Vincoli di Miscelazione (Blending)

Usati quando interessa la **composizione percentuale** di una miscela.

**Esempio:** Almeno 30% delle calorie da cibi "non trasformati"

**Vincolo non lineare (problematico):**
```
(Σⱼ∈G qⱼxⱼ) / (Σⱼ qⱼxⱼ) ≥ 0.30
```

**Linearizzazione** (moltiplicando per il denominatore):
```
Σⱼ∈G qⱼxⱼ ≥ 0.30 · Σⱼ qⱼxⱼ
```

**Risultato:** Un vincolo lineare adatto alla PL.

## 6. Vincoli Logici con Variabili Binarie

Le decisioni logiche si modellano con variabili 0/1.

### Traduzioni Base
- **OR (x ∨ y):** `x + y ≥ 1`
- **AND (x ∧ y):** `x ≥ 1 AND y ≥ 1`
- **Implicazione (x → y):** `x ≤ y`
- **XOR (x ⊕ y):** `x + y = 1`

### Collegamento con SAT
Ogni formula logica booleana può essere riscritta come sistema di vincoli lineari su variabili binarie.

## 7. Vincoli di Esclusione e "Tutto o Niente"

### Esclusione Semplice
Progetti incompatibili i e j: `yᵢ + yⱼ ≤ 1`

### Vincoli di Collegamento (Linking Constraints)
**Problema:** "O salmone O pollo" nella poke bowl

**Soluzione:**
1. Variabili binarie flag:
```
y_salmone = 1 se x_salmone > 0, 0 altrimenti
y_pollo = 1 se x_pollo > 0, 0 altrimenti
```

2. Vincolo di esclusione:
```
y_salmone + y_pollo ≤ 1
```

3. Linking constraints:
```
x_salmone ≤ M · y_salmone
x_pollo ≤ M · y_pollo
```

dove M è una costante sufficientemente grande.

### Vincolo "Tutto o Niente"
```
x = 0  OPPURE  a ≤ x ≤ b  (con a > 0)
```

Schema: Introdurre binaria y e usare linking constraints per forzare x nell'intervallo [a,b] quando y=1.

## 8. Conservazione del Flusso su Grafi

**Esempio:** Rete di tubi in campo petrolifero

### Componenti
- **Pozzi (sorgenti):** vincoli di disponibilità
- **Raffinerie (pozzi):** vincoli di fabbisogno
- **Nodi intermedi:** conservazione del flusso

### Variabili
`xᵢⱼ` = flusso da nodo i a nodo j

### Vincoli per Tipo di Nodo

**Pozzi (disponibilità):**
```
Σⱼ xᵢⱼ - Σⱼ xⱼᵢ ≤ capacità_estrattiva
```

**Raffinerie (fabbisogno):**
```
Σⱼ xⱼᵢ - Σⱼ xᵢⱼ ≥ domanda
```

**Nodi intermedi (conservazione):**
```
Σⱼ xⱼᵢ = Σⱼ xᵢⱼ
```
(tutto ciò che entra deve uscire)

### Proprietà Importanti
- Ogni variabile di flusso appare esattamente 2 volte
- Una volta positiva (nodo arrivo), una volta negativa (nodo partenza)
- Convenzione: flusso uscente negativo, flusso entrante positivo

### Variabili Slack nei Problemi di Flusso
Rappresentano:
- Nei pozzi: petrolio estratto ma "sprecato"
- Nelle raffinerie: capacità di raffinazione non utilizzata

## Concetti Chiave

### Trasformazioni Fondamentali
1. **≥ → ≤:** Moltiplicare per -1 e invertire
2. **= → ≤:** Aggiungere slack non negative
3. **= → {≤,≥}:** Uguaglianza = intersezione di due disuguaglianze

### Principi Generali
- I vincoli traducono la **realtà** nel modello matematico
- Riconoscere il tipo di vincolo aiuta a modellare più velocemente
- Le trasformazioni servono solo a standardizzare, non cambiano le soluzioni
- Molte applicazioni diverse si riducono a programmi lineari/misti con gli stessi mattoni

### Focus sulla Modellazione
- Non pensare alla soluzione durante la modellazione
- Concentrarsi sulla rappresentazione matematica corretta
- L'interpretazione semantica è fondamentale quanto la forma algebrica