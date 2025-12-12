# Ricerca Operativa - Lezione 2

**Prof. Federico Malucelli** | *Politecnico di Milano*

## La Modellazione come Arte

La modellazione matematica è un'arte che richiede esperienza, fantasia e intuizione. Non esiste "il modello migliore" in assoluto - tutto dipende dallo scopo e dal contesto.

**Principio fondamentale:** Il modello matematico è una rappresentazione artificiale della realtà. Può essere molto dettagliato (se servono dettagli) o grezzo (per risolvere velocemente).

## I Tre Ingredienti Essenziali

Ogni modello richiede tre componenti fondamentali:

1. **DECISIONI** → tradotte in **Variabili**
2. **REGOLE** → tradotte in **Vincoli** (uguaglianze o disuguaglianze)
3. **OBIETTIVI** → tradotti in **Funzione obiettivo**

## Tipi di Variabili

### 1. Variabili Non Negative (x ≥ 0)
- Esempio: numero di prodotti da costruire
- I valori negativi non hanno senso nel contesto del problema

### 2. Variabili Illimitate (x ∈ ℝ)
Possono assumere valori positivi, negativi o zero quando la negatività ha significato:
- **Temperature** (°C)
- **Velocità** con direzione
- **Perturbazioni** rispetto a uno stato iniziale

### 3. Variabili Binarie (x ∈ {0,1})
Per decisioni on/off:
- Costruire o non costruire una fabbrica
- Scegliere un percorso tra alternative
- Assegnare o non assegnare un compito

### 4. Variabili Intere (x ∈ ℤ)
Quando servono valori discreti:
- Numero di autobus da acquistare
- Numero di turni di lavoro
- Numero di container da spedire

## Esempio Pratico: Riprogrammazione Orari Treni

**Problema:** Riprogrammare gli orari dei treni partendo da un orario esistente.

**Setup:**
- M stazioni con orari attuali: A₁, A₂, ..., Aₘ
- Nuovi orari: A'ₛ = Aₛ + πₛ
- πₛ = variabile di perturbazione (può essere positiva, negativa o zero)

**Interpretazione di πₛ:**
- π < 0 → anticipo rispetto all'orario precedente
- π = 0 → nessun cambiamento
- π > 0 → ritardo

## Trasformazione: Variabili Illimitate → Non Negative

**Problema:** I solver standard accettano solo variabili non negative (x ≥ 0).

**Soluzione:** Rappresentare ogni variabile illimitata come differenza di due variabili non negative:

```
πₛ = πₛ⁺ - πₛ⁻
```

dove πₛ⁺ ≥ 0 e πₛ⁻ ≥ 0

**Casi:**
- Se πₛ > 0 (ritardo): πₛ⁺ > 0, πₛ⁻ = 0
- Se πₛ < 0 (anticipo): πₛ⁺ = 0, πₛ⁻ > 0
- Se πₛ = 0: πₛ⁺ = 0, πₛ⁻ = 0

**Nota:** I solver moderni gestiscono automaticamente questa trasformazione.

## Vincoli: Uguaglianza vs Disuguaglianza

### Vincoli di Disuguaglianza (≤, ≥)
Limitazioni che non devono essere necessariamente sature:
- Budget: costo_totale ≤ budget_max
- Capacità: produzione ≤ capacità_max

### Vincoli di Uguaglianza (=)
Devono essere soddisfatti esattamente:
- Conservazione del flusso: flusso_entrante = flusso_uscente
- Bilancio di massa: input = output + accumulo

### Trasformazione Equivalente
Un vincolo di uguaglianza può essere scritto come due disuguaglianze:
```
ax = b  ⟺  {ax ≤ b
            {ax ≥ b
```

## Problemi con Obiettivi Multipli

Quando ci sono più obiettivi (es. massimizzare profitto E minimizzare impatto ambientale):

**Approcci:**
1. **Combinazione pesata:** max α·profitto - β·impatto_ambientale
2. **Ottimizzazione lessicografica:** ottimizza prima un obiettivo, poi l'altro
3. **Vincoli aggiuntivi:** trasforma obiettivi secondari in vincoli

## Problemi di Fattibilità

**Definizione:** Problemi senza funzione obiettivo - l'obiettivo è solo trovare una soluzione ammissibile.

**Forma matematica:**
```
Trovare: x
Soggetto a: Ax ≤ b
            x ≥ 0
```

**Esempi:**
- Scheduling: "Esiste un modo per assegnare tutti i compiti?"
- Logistica: "È possibile trasportare tutti i prodotti?"
- Progettazione: "Si può progettare un sistema che soddisfi i requisiti?"

**Trasformazione in problema di ottimizzazione:**
- Minimizzare zero: min 0
- Minimizzare violazioni: min Σ(violazioni)

**Risultati:**
- **Fattibile:** esiste almeno una soluzione
- **Infeasible:** nessuna soluzione soddisfa tutti i vincoli

## Concetti Chiave

### Principi Fondamentali
- La modellazione richiede esperienza e pratica
- Non pensare alla soluzione durante la modellazione
- Concentrarsi sulla rappresentazione matematica
- La risoluzione viene dopo

### Trasformazioni Equivalenti
- Sono matematicamente corrette
- I solver moderni le gestiscono automaticamente
- Utili per comprendere i concetti teorici

### Focus sulla Fattibilità
- Primo passo: esiste una soluzione?
- Secondo passo: quale è la migliore?
- Un problema può essere infeasible anche se ogni singolo vincolo è ragionevole

## Preparazione Prossima Lezione

**Da rivedere:**
1. I quattro tipi di variabili
2. La trasformazione π = π⁺ - π⁻
3. Differenza tra vincoli di uguaglianza e disuguaglianza
4. Concetto di problema equivalente

**Domande autovalutazione:**
- Quando usare variabili illimitate?
- Come trasformare variabili illimitate in non negative?
- Differenza tra problema di ottimizzazione e fattibilità?
- Perché "la modellazione è un'arte"?