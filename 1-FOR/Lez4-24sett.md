## Riassunto: Puzzle di Modellazione, Tecniche di Ottimizzazione e Formulazioni Applicate
### Riepilogo Puzzle: Variabile in Intervallo o Zero (Modellazione con Variabile 0–1)
- Obiettivo: Modellare x_sal ∈ [A, B] (A, B ≥ 0) o x_sal = 0 con una variabile binaria.
- Variabili:
  - y_sal ∈ {0,1}: y_sal = 1 se x_sal ∈ [A, B]; y_sal = 0 se x_sal = 0.
- Vincoli:
  - x_sal ≥ A · y_sal
  - x_sal ≤ B · y_sal
- Correttezza:
  - Se y_sal = 0 ⇒ x_sal = 0 (data la non negatività).
  - Se y_sal = 1 ⇒ A ≤ x_sal ≤ B.
- Guida per la consegna:
  - Se il tuo modello corrisponde: commenta "corretto."
  - Se chiaramente sbagliato: "sbagliato."
  - Usa "per favore controlla" solo quando genuinamente incerto dopo aver valutato l'equivalenza.
### Uso di un Risolutore di Ottimizzazione per Testare la Fattibilità (Nessuna Funzione Obiettivo)
- Problema: Determinare se esiste x che soddisfa A x ≤ b usando un risolutore che richiede min c^T x s.t. A x ≤ b.
- Trucco:
  - Scegliere un vincolo a1^T x ≤ b1 come obiettivo: minimizzare a1^T x.
  - Mantenere altri vincoli a2^T x ≤ b2, …, am^T x ≤ bm.
- Risultati:
  - Se i vincoli 2..m sono infattibili ⇒ originale infattibile.
  - Se viene trovata una soluzione ottimale x*:
    - Se a1^T x* ≤ b1 ⇒ x* è fattibile per il sistema originale.
    - Se a1^T x* > b1 ⇒ originale infattibile (contraddizione con ottimalità).
- Nota: Ricerca binaria e idee correlate funzionano anch'esse; il messaggio centrale è sfruttare il risolutore per inferire la fattibilità.
<img width="552" height="725" alt="image" src="https://github.com/user-attachments/assets/08a642e0-4b78-4927-be9e-331876dfc83e" />
  
# Facility Location Problem

## Insiemi
- **S = {1, ..., m}**: siti candidati per le antenne
- **C = {1, ..., n}**: clienti da servire

## Parametri
- **M**: capacità massima (n° clienti per antenna)
- **Pᵢⱼ**: potenza emessa da antenna i ∈ S per cliente j ∈ C
- **cⱼ**: costo installazione antenna j ∈ S

## Variabili
- **yⱼ** ∈ {0,1}: antenna j attiva (1) o no (0), ∀j ∈ S
- **xᵢⱼ** ∈ {0,1}: cliente i assegnato ad antenna j (1) o no (0), ∀i ∈ C, ∀j ∈ S

## Vincoli

### Copertura completa
Ogni cliente deve essere servito da esattamente un'antenna:

$$\sum_{j \in S} x_{ij} = 1 \qquad \forall i \in C$$

### Vincolo di capacità
Il numero di clienti assegnati a un'antenna non può superare la sua capacità:

$$\sum_{i \in C} x_{ij} \leq M \cdot y_j \qquad \forall j \in S$$

**Implicazione**: Se $y_j = 0$ allora $x_{ij} = 0$ (antenna spenta → nessun cliente servito)

**Vincolo equivalente**: $x_{ij} \leq y_j \quad \forall i \in C, \forall j \in S$

---

## Funzioni Obiettivo

### Obiettivo 1: Minimizzare costo installazione

$$\min \sum_{j \in S} c_j \cdot y_j$$

Minimizza il costo totale di installazione delle antenne attive.

---

### Obiettivo 2: Minimizzare potenza massima emessa

**Formulazione non lineare**:

$$\min \max_{i \in C, j \in S} \{P_{ij} \cdot x_{ij}\}$$

**Linearizzazione** con variabile ausiliaria **d** (potenza massima):

$$\min \quad d$$

$$\text{s.t.} \quad d \geq P_{ij} \cdot x_{ij} \qquad \forall i \in C, \forall j \in S$$

La variabile $d$ rappresenta la massima potenza emessa nel sistema.

---

### Obiettivo 3: Massimizzare copertura con vincolo di budget

**Opzione A**: Massimizzare copertura pesata con costo

$$\max \quad d \cdot \sum_{i \in C} \sum_{j \in S} x_{ij} - \sum_{j \in S} c_j \cdot y_j$$

**Opzione B**: Vincolo di budget fisso **B**

$$\max \quad \sum_{i \in C} \sum_{j \in S} x_{ij}$$

$$\text{s.t.} \quad \sum_{j \in S} c_j \cdot y_j \leq B$$

Massimizza il numero totale di assegnamenti cliente-antenna rispettando il budget disponibile.
- [ ] Iniziare con Esercizio 6 e portare formulazioni preliminari alla prossima sessione.
- [ ] Esplorare la raccolta esercizi e consultare schemi di soluzione secondo necessità.
- [ ] Segnalare qualsiasi link HTML non funzionante/inaccessibile all'istruttore.
