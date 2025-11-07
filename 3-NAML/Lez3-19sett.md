---

# Lezione 3 - Venerdì 19 Settembre 2025

**Argomenti principali:**
- Progetto 7: Sequenza di Fibonacci (ricorsione e iterazione)
- Esercizi: Rapporto aureo e sequenza di Recamán
- Progetto 8: Stima di π con metodo Monte Carlo
- Introduzione a NumPy (array, operazioni vettoriali)
- Introduzione a Matplotlib (plotting 2D e 3D)
- Progetto 9: Visualizzazione 3D della funzione di Himmelblau

---

## 📌 Progetto 7: Numeri di Fibonacci

### Competenze Sviluppate
- **Ricorsione** (chiamate di funzioni a se stesse)
- **Funzione `range()`** per generare sequenze
- **List comprehension** (costruzione concisa di liste)
- **Slicing** con indici negativi (`list[-1]`, `list[-2]`)
- **Analisi di complessità** (ricorsivo O(2^n) vs iterativo O(n))

### Definizione Matematica

La sequenza di Fibonacci è definita dalla seguente relazione ricorsiva:

$$
F(0) = 0
$$
$$
F(1) = 1
$$
$$
F(n) = F(n-1) + F(n-2) \quad \text{per } n \geq 2
$$

**Sequenza:** 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...

**Interpretazione:** Ogni numero è la somma dei due precedenti.

---

### 🔴 Implementazione 1: Fibonacci Ricorsivo

**Idea:** Tradurre direttamente la definizione matematica in codice Python.

```python
def recfib(n):
    """
    Calcola l'n-esimo numero di Fibonacci usando ricorsione.
    
    Args:
        n (int): Indice nella sequenza (n >= 0)
    
    Returns:
        int: n-esimo numero di Fibonacci
    
    Nota: Non include validazione dell'input!
    """
    # Caso base: F(0) = 0, F(1) = 1
    if n <= 1:
        return n
    else:
        # Caso ricorsivo: F(n) = F(n-1) + F(n-2)
        return recfib(n-1) + recfib(n-2)
```

**Esempio di utilizzo:**
```python
>>> recfib(10)
55
```

**Caratteristiche di Python:**
- Python supporta la ricorsione **di default** (a differenza di Fortran che richiede dichiarazione esplicita)
- Limite di ricorsione: ~1000 livelli (configurabile con `sys.setrecursionlimit()`)

---

### ⚠️ Problema della Ricorsione: Complessità Esponenziale

**Esempio:** Calcolo di `recfib(5)`

```
recfib(5)
├── recfib(4)
│   ├── recfib(3)
│   │   ├── recfib(2)
│   │   │   ├── recfib(1) → 1
│   │   │   └── recfib(0) → 0
│   │   └── recfib(1) → 1
│   └── recfib(2)
│       ├── recfib(1) → 1
│       └── recfib(0) → 0
└── recfib(3)
    ├── recfib(2)
    │   ├── recfib(1) → 1
    │   └── recfib(0) → 0
    └── recfib(1) → 1
```

**Osservazioni:**
- `recfib(3)` viene calcolato **2 volte**
- `recfib(2)` viene calcolato **3 volte**
- `recfib(1)` viene calcolato **5 volte**
- `recfib(0)` viene calcolato **3 volte**

**Complessità temporale:** O(2^n) - crescita esponenziale!

| n | Chiamate ricorsive | Tempo stimato |
|---|---|---|
| 10 | 177 | < 1ms |
| 20 | 21,891 | ~10ms |
| 30 | 2,692,537 | ~1s |
| 40 | 331,160,281 | ~2 minuti |

**Conclusione:** La versione ricorsiva è **inefficiente** per n grandi (> 30).

---

### 📚 Interludio: Funzione `range()`

Prima di continuare con Fibonacci, vediamo la funzione `range()` che genera sequenze di numeri.

**Sintassi:** `range(start, stop, step)`
- `start` (opzionale): valore iniziale (default: 0)
- `stop` (obbligatorio): valore finale **escluso**
- `step` (opzionale): incremento (default: 1)

**Esempi:**

```python
# Solo stop (da 0 a 5, escluso 6)
>>> R1 = range(6)
>>> print(R1)
range(0, 6)
>>> list(R1)
[0, 1, 2, 3, 4, 5]

# Start, stop, step
>>> R2 = range(10, 20, 2)
>>> list(R2)
[10, 12, 14, 16, 18]

# Step negativo (conteggio all'indietro)
>>> R3 = range(20, 10, -1)
>>> list(R3)
[20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
```

**Note:**
- `range()` restituisce un oggetto di tipo `range`, non una lista
- Usa `list()` per convertirlo in lista
- È **lazy** (genera valori al volo, non li memorizza tutti)

---

### 🎯 List Comprehension

**Definizione:** Modo conciso di creare liste applicando un'espressione a ogni elemento di una sequenza.

**Sintassi base:** `[espressione for variabile in sequenza]`

**Esempio 1: Quadrati dei primi 10 numeri**
```python
>>> quadrati = [n**2 for n in range(10)]
>>> quadrati
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

Equivalente a:
```python
quadrati = []
for n in range(10):
    quadrati.append(n**2)
```

**Esempio 2: Primi 15 numeri di Fibonacci con ricorsione**
```python
>>> F = [recfib(n) for n in range(15)]
>>> F
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
```

**Verifica tipo:**
```python
>>> type(F)
<class 'list'>
```

**⚠️ Attenzione:** Questo approccio è **lento** perché ogni `recfib(n)` ricalcola tutto da zero!

---

### 🔍 Slicing: Estrarre Elementi da Liste

**Sintassi generale:** `L[start:stop:step]` (tutti i parametri sono opzionali)

**Esempi con la lista Fibonacci:**
```python
F = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
#    0  1  2  3  4  5  6   7   8   9  10  11   12   13   14  (indici positivi)
#  -15-14-13-12-11-10 -9  -8  -7  -6  -5  -4   -3   -2   -1  (indici negativi)

# Elementi dall'indice 1 al 3 (4 escluso)
>>> F[1:4]
[1, 1, 2]

# Dal 5° elemento alla fine
>>> F[5:]
[5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Elementi ogni 3 posizioni (da 3 a 10)
>>> F[3:11:3]
[2, 13, 55]

# Step negativo (dal 10 al 5, indietro)
>>> F[10:5:-1]
[55, 34, 21, 13, 8]

# Inversione completa della lista
>>> F[::-1]
[377, 233, 144, 89, 55, 34, 21, 13, 8, 5, 3, 2, 1, 1, 0]
```

**Regola importante:** `stop` è **sempre escluso** (come in `range()`)

**Indici negativi:**
- `F[-1]` → ultimo elemento (377)
- `F[-2]` → penultimo elemento (233)
- Utili per accedere alla fine senza conoscere la lunghezza
---

### 🟢 Implementazione 2: Fibonacci Iterativo (Efficiente)

**Idea:** Costruire la lista completa una volta sola, senza ridondanza.

```python
def fib(n):
    """
    Calcola i primi n+1 numeri di Fibonacci (da F(0) a F(n)).
    
    Args:
        n (int): Indice massimo nella sequenza
    
    Returns:
        list: Lista [F(0), F(1), ..., F(n)]
    
    Complessità: O(n) tempo, O(n) spazio
    """
    # Inizializza con i primi due valori
    fib_seq = [0, 1]
    
    # Calcola i valori successivi
    for i in range(2, n+1):
        # Somma ultimi due elementi: fib_seq[-1] + fib_seq[-2]
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    
    return fib_seq
```

**Spiegazione:**
- `fib_seq[-1]` → ultimo elemento (F(i-1))
- `fib_seq[-2]` → penultimo elemento (F(i-2))
- `.append()` → aggiunge nuovo elemento alla fine

**Esempio di utilizzo:**
```python
>>> F_list = fib(15)
>>> F_list
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

# Per ottenere solo l'n-esimo numero (ultimo della lista)
>>> F_15 = fib(15)[-1]
>>> F_15
610
```

**Promemoria sull'indicizzazione:**
- Le liste Python sono **indicizzate da zero**: `F_list[0]` è il primo elemento
- Indici negativi partono dalla fine: `F_list[-1]` è l'ultimo, `F_list[-2]` è il penultimo

---

### 📊 Confronto: Ricorsivo vs Iterativo

| Caratteristica | Ricorsivo (`recfib`) | Iterativo (`fib`) |
|---|---|---|
| **Complessità temporale** | O(2^n) - esponenziale | O(n) - lineare |
| **Complessità spaziale** | O(n) stack | O(n) lista |
| **Tempo per n=30** | ~1 secondo | < 1 millisecondo |
| **Tempo per n=40** | ~2 minuti | < 1 millisecondo |
| **Vantaggio** | Codice semplice/elegante | Velocità, efficienza |
| **Svantaggio** | Lentissimo per n>30 | Più codice |

**Conclusione:** Usa sempre la versione **iterativa** per applicazioni reali!

---

## 🔬 Esercizio 1: Convergenza al Rapporto Aureo

### Obiettivo
Calcolare i seguenti rapporti lungo la sequenza di Fibonacci:

$$
r_1(n) = \frac{F(n)}{F(n+1)}, \quad r_2(n) = \frac{F(n)}{F(n+2)}, \quad r_3(n) = \frac{F(n)}{F(n+3)}
$$

Cosa noti per $n$ grande (es. $n \approx 30$)?

---

### Il Rapporto Aureo φ (Phi)

**Definizione matematica:**

$$
\varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749...
$$

**Proprietà:**
- Anche chiamato "Sezione Aurea" o "Golden Ratio"
- Soluzione di $\varphi^2 = \varphi + 1$
- Il reciproco è $\frac{1}{\varphi} = \varphi - 1 \approx 0.618...$

**Legame con Fibonacci:**

$$
\lim_{n \to \infty} \frac{F(n+1)}{F(n)} = \varphi
$$

Quindi:

$$
\lim_{n \to \infty} \frac{F(n)}{F(n+1)} = \frac{1}{\varphi} = \frac{\sqrt{5} - 1}{2} \approx 0.618...
$$

---

### Implementazione: Metodo 1 (Ciclo For)

```python
# Calcola i primi 40 numeri di Fibonacci
F1 = fib(40)

# Calcola il rapporto r1 = F(n) / F(n+1)
R1 = []
for n in range(40):  # Da n=0 a n=39
    rapporto = F1[n] / F1[n+1]
    R1.append(rapporto)

print(R1)
```

**Output (primi e ultimi valori):**
```python
[0.0, 1.0, 0.5, 0.666..., 0.6, 0.625, 0.615..., 0.619..., ..., 0.618033988...]
```

**Calcolo del valore teorico:**
```python
import math as mt
phi_reciproco = (mt.sqrt(5) - 1) / 2
print(phi_reciproco)  # 0.6180339887498949
```

**Osservazioni:**
- I primi valori sono lontani dal limite (0.0, 1.0, 0.5, ...)
- Già da n=10 siamo vicini a 0.618...
- Da n=30 in poi la precisione è **eccellente** (15 cifre decimali)

---

### Implementazione: Metodo 2 (List Comprehension)

```python
# Versione più compatta con list comprehension
F1 = fib(40)
R1 = [F1[n] / F1[n+1] for n in range(40)]
```

**⚠️ Attenzione:** Non usare `recfib()` qui! Sarebbe:
```python
# ❌ LENTISSIMO - NON FARE!
R1_slow = [recfib(n) / recfib(n+1) for n in range(40)]
```
Questo ricalcola `recfib(n)` e `recfib(n+1)` separatamente, con complessità totale O(2^n * n).

---

### Visualizzazione della Convergenza

```python
import matplotlib.pyplot as plt

F1 = fib(40)
R1 = [F1[n] / F1[n+1] for n in range(40)]

# Valore teorico
phi_inv = (mt.sqrt(5) - 1) / 2

# Plot del rapporto
plt.plot(R1, 'b-', label=r'$F(n) / F(n+1)$')
plt.axhline(y=phi_inv, color='r', linestyle='--', label=r'$1/\varphi \approx 0.618$')
plt.xlabel('n')
plt.ylabel('Rapporto')
plt.title('Convergenza al Reciproco del Rapporto Aureo')
plt.legend()
plt.grid(True)
plt.show()
```

**Risultato:** La curva blu converge rapidamente alla linea rossa tratteggiata.

---

### Esercizio Esteso: Altri Rapporti

**Calcola r2 e r3:**
```python
F1 = fib(42)  # Serve F(42) per r3 con n=39

R1 = [F1[n] / F1[n+1] for n in range(40)]
R2 = [F1[n] / F1[n+2] for n in range(40)]
R3 = [F1[n] / F1[n+3] for n in range(40)]
```

**Valori limite teorici:**

$$
\lim_{n \to \infty} r_1(n) = \frac{1}{\varphi} \approx 0.618
$$

$$
\lim_{n \to \infty} r_2(n) = \frac{1}{\varphi^2} \approx 0.382
$$

$$
\lim_{n \to \infty} r_3(n) = \frac{1}{\varphi^3} \approx 0.236
$$

**Verifica numerica:**
```python
phi = (1 + mt.sqrt(5)) / 2
print(f"1/φ   = {1/phi:.10f}")       # 0.6180339887
print(f"1/φ²  = {1/phi**2:.10f}")    # 0.3819660113
print(f"1/φ³  = {1/phi**3:.10f}")    # 0.2360679775

print(f"\nR1[39] = {R1[39]:.10f}")
print(f"R2[39] = {R2[39]:.10f}")
print(f"R3[39] = {R3[39]:.10f}")
```

**Output:**
```
1/φ   = 0.6180339887
1/φ²  = 0.3819660113
1/φ³  = 0.2360679775

R1[39] = 0.6180339887
R2[39] = 0.3819660113
R3[39] = 0.2360679775
```

**Conclusione:** Perfetta convergenza a potenze negative di φ!
---

## 🔢 Esercizio 2: Sequenza di Recamán

### Definizione

La sequenza di Recamán è una sequenza matematica interessante definita come:

$$
R(0) = 0
$$

Per $n \geq 1$:

$$
R(n) = \begin{cases}
R(n-1) - n & \text{se } R(n-1) - n \geq 0 \text{ e } R(n-1) - n \notin \{R(0), ..., R(n-1)\} \\
R(n-1) + n & \text{altrimenti}
\end{cases}
$$

**In parole:**
- Parti da 0
- Ad ogni passo, **prova a sottrarre** n
- Se il risultato è **negativo** O **già presente** nella sequenza → **aggiungi** n invece
- Altrimenti sottrai n

**Sequenza:** 0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9, 24, 8, 25, 43, 62, ...

---

### Analisi Manuale: Primi 10 Termini

| n | R(n-1) | Candidato (R(n-1) - n) | ≥ 0? | Già presente? | Azione | R(n) |
|---|---|---|---|---|---|---|
| 0 | - | - | - | - | Inizializzazione | **0** |
| 1 | 0 | 0 - 1 = -1 | ❌ No | - | Aggiungi: 0+1 | **1** |
| 2 | 1 | 1 - 2 = -1 | ❌ No | - | Aggiungi: 1+2 | **3** |
| 3 | 3 | 3 - 3 = 0 | ✅ Sì | ✅ Sì (R(0)=0) | Aggiungi: 3+3 | **6** |
| 4 | 6 | 6 - 4 = 2 | ✅ Sì | ❌ No | Sottrai: 6-4 | **2** |
| 5 | 2 | 2 - 5 = -3 | ❌ No | - | Aggiungi: 2+5 | **7** |
| 6 | 7 | 7 - 6 = 1 | ✅ Sì | ✅ Sì (R(1)=1) | Aggiungi: 7+6 | **13** |
| 7 | 13 | 13 - 7 = 6 | ✅ Sì | ✅ Sì (R(3)=6) | Aggiungi: 13+7 | **20** |
| 8 | 20 | 20 - 8 = 12 | ✅ Sì | ❌ No | Sottrai: 20-8 | **12** |
| 9 | 12 | 12 - 9 = 3 | ✅ Sì | ✅ Sì (R(2)=3) | Aggiungi: 12+9 | **21** |

**Risultato:** `[0, 1, 3, 6, 2, 7, 13, 20, 12, 21]`

---

### Competenze Python Utilizzate

1. **Operatore `in`**: Verifica appartenenza a una lista
   ```python
   >>> 5 in [1, 2, 3, 4, 5]
   True
   >>> 10 in [1, 2, 3, 4, 5]
   False
   ```

2. **Operatore logico `and`**: Entrambe le condizioni devono essere vere
   ```python
   >>> (5 >= 0) and (5 not in [1, 2, 3])
   True
   >>> (5 >= 0) and (5 not in [1, 5, 10])
   False
   ```

3. **Comparazione `>=`**: Maggiore o uguale
   ```python
   >>> 0 >= 0
   True
   >>> -1 >= 0
   False
   ```

---

### Implementazione in Python

```python
def recaman(n):
    """
    Genera i primi n termini della sequenza di Recamán.
    
    Args:
        n (int): Numero di termini da generare
    
    Returns:
        list: Lista [R(0), R(1), ..., R(n-1)]
    
    Esempio:
        >>> recaman(10)
        [0, 1, 3, 6, 2, 7, 13, 20, 12, 21]
    """
    # Inizializza con R(0) = 0
    R = [0]
    
    # Genera termini da R(1) a R(n-1)
    for i in range(1, n):
        # Candidato: sottrai i dall'ultimo valore
        candidate_minus = R[i-1] - i  # oppure R[-1] - i
        
        # Controlla entrambe le condizioni
        if (candidate_minus >= 0) and (candidate_minus not in R):
            # Condizioni soddisfatte: sottrai
            R.append(candidate_minus)
        else:
            # Altrimenti: aggiungi
            R.append(R[i-1] + i)  # oppure R[-1] + i
    
    return R
```

**Dettagli di implementazione:**
- `R[i-1]` o `R[-1]` → ultimo elemento della lista
- `candidate_minus not in R` → verifica che il valore non sia già presente
- `and` richiede **entrambe** le condizioni vere

---

### Test della Funzione

```python
>>> recaman(10)
[0, 1, 3, 6, 2, 7, 13, 20, 12, 21]

>>> recaman(15)
[0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9]

>>> recaman(20)
[0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9, 24, 8, 25, 43, 62]
```

**Verifica online:** [OEIS A005132](https://oeis.org/A005132)

---

### Variazione con `and` e `&`

**Esempio dal notebook:**
```python
F2 = fib(10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
x = 22

# Operatore logico 'and'
>>> (x in F2) and (x >= 0)
False

# Operatore bitwise '&' (per array NumPy)
>>> (x in F2) & (x >= 0)
False
```

**Differenza:**
- `and` → operatore logico Python (per singoli valori booleani)
- `&` → operatore bitwise (usato per array NumPy elemento per elemento)

Per la sequenza di Recamán, **usa `and`**.

---

### Proprietà della Sequenza di Recamán

1. **Non tutti i numeri positivi appaiono**
   - Alcuni numeri (es. 4, 5, 14, 17, ...) non appaiono mai nella sequenza!
   - Non si sa se la sequenza contenga tutti i numeri positivi (problema aperto)

2. **Crescita irregolare**
   - Alterna sottrazioni e addizioni in modo imprevedibile
   - Dopo migliaia di termini, ci sono ancora "salti" inaspettati

3. **Visualizzazione**
   - Viene spesso visualizzata come una serie di archi semicircolari
   - Video famoso su YouTube mostra la sequenza musicalmente

---

### Esercizio Esteso: Numeri Mancanti

**Domanda:** Quali numeri positivi **non** appaiono nei primi 100 termini?

```python
R = recaman(100)

# Trova tutti i numeri mancanti fino a max(R)
max_val = max(R)
numeri_mancanti = [n for n in range(max_val) if n not in R]

print(f"Primi 10 numeri mancanti: {numeri_mancanti[:10]}")
print(f"Totale numeri mancanti (0-{max_val}): {len(numeri_mancanti)}")
```

**Output tipico:**
```
Primi 10 numeri mancanti: [4, 5, 14, 17, 18, 26, 27, 28, 29, 35]
Totale numeri mancanti (0-...): 42
```

---

### Confronto con Fibonacci

| Caratteristica | Fibonacci | Recamán |
|---|---|---|
| **Regola** | Somma ultimi 2 | Sottrai o aggiungi |
| **Crescita** | Esponenziale | Irregolare |
| **Prevedibilità** | Alta (formula chiusa) | Bassa (caotica) |
| **Complessità calcolo** | O(n) | O(n²) (per `not in`) |
| **Tutti i valori unici?** | Sì (crescente) | Sì (per definizione) |
| **Ottimizzazione** | Iterazione | Usa `set` per velocità |

**Versione ottimizzata con `set`:**
```python
def recaman_fast(n):
    """Versione ottimizzata con set per O(1) lookup."""
    R = [0]
    R_set = {0}  # Per controllo rapido di appartenenza
    
    for i in range(1, n):
        candidate = R[-1] - i
        if (candidate >= 0) and (candidate not in R_set):
            R.append(candidate)
            R_set.add(candidate)
        else:
            new_val = R[-1] + i
            R.append(new_val)
            R_set.add(new_val)
    
    return R
```

**Complessità:** O(n) invece di O(n²)!
---

## 🎲 Progetto 8: Stima di π con il Metodo Monte Carlo

### Competenze Sviluppate
- Operatore `+=` (e varianti `-=`, `*=`, etc.)
- Introduzione agli **array NumPy** (n-dimensionali)
- Metodi `.append()` e concatenazione liste (`+`)
- Introduzione a **Matplotlib** (visualizzazione dati)
- Funzioni casuali con `random.uniform()`
- Concetti di simulazione statistica

---

### 🎯 Problema: Stimare π Geometricamente

**Setup geometrico:**
1. Quadrato di lato 2 centrato nell'origine: $[-1, 1] \times [-1, 1]$
   - Area quadrato = $2 \times 2 = 4$

2. Cerchio inscritto di raggio 1 (centrato nell'origine)
   - Area cerchio = $\pi r^2 = \pi \cdot 1^2 = \pi$

```
    y
    1 ┌─────────────┐
      │    .-""-.   │
      │  .'       '. │
      │ /     •     \│
    0 ├──────────────┤  x
      │ \           /│
      │  '.       .' │
   -1 │    '-...-'   │
      └─────────────┘
     -1      0      1
```

---

### 📐 Teoria del Metodo Monte Carlo

**Idea:** Genera N punti casuali $(x, y)$ nel quadrato. La probabilità che un punto cada dentro il cerchio è:

$$
P(\text{punto dentro}) = \frac{\text{Area cerchio}}{\text{Area quadrato}} = \frac{\pi}{4}
$$

**Stimatore:**

$$
\frac{N_{\text{dentro}}}{N_{\text{totale}}} \approx \frac{\pi}{4}
$$

Quindi:

$$
\pi \approx 4 \cdot \frac{N_{\text{dentro}}}{N}
$$

**Criteri di appartenenza:** Un punto $(x, y)$ è dentro il cerchio se:

$$
d = \sqrt{x^2 + y^2} \leq 1
$$

---

### 🔧 Implementazione Base

```python
import math as mt
import random

def compute_pi_mc(N):
    """
    Stima π usando il metodo Monte Carlo.
    
    Args:
        N (int): Numero di punti casuali da generare
    
    Returns:
        float: Stima di π
    """
    circle_points = 0  # Contatore punti dentro il cerchio
    square_points = 0  # Contatore punti totali
    
    for i in range(N):
        # Genera coordinate casuali in [-1, 1] × [-1, 1]
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)
        
        # Calcola distanza dall'origine
        dist = mt.sqrt(rand_x**2 + rand_y**2)
        
        # Verifica se il punto è dentro il cerchio
        if dist <= 1:
            circle_points += 1
        
        square_points += 1
    
    # Stima π
    pi_est = 4.0 * circle_points / square_points
    
    return pi_est
```

**Uso della funzione:**
```python
>>> compute_pi_mc(1000)
3.144  # Varia ad ogni esecuzione!

>>> compute_pi_mc(10000)
3.1384

>>> compute_pi_mc(100000)
3.14256
```

---

### 📊 Convergenza e Precisione

**Test con diverse dimensioni:**

| N | Stima π | Errore % | Tempo |
|---|---|---|---|
| 100 | ~3.12 | ~0.7% | < 1ms |
| 1,000 | ~3.14 | ~0.05% | < 1ms |
| 10,000 | ~3.142 | ~0.02% | ~5ms |
| 100,000 | ~3.1416 | ~0.001% | ~50ms |
| 1,000,000 | ~3.14159 | ~0.0001% | ~500ms |

**Osservazioni:**
- La convergenza è **stocastica** (non monotona)
- Errore decresce come $O(1/\sqrt{N})$ (legge radice quadrata)
- Per dimezzare l'errore, serve **4 volte** più punti

**⚠️ Nota sulla casualità:**
```python
# Risultati diversi ad ogni esecuzione (senza seed)
>>> compute_pi_mc(1000)
3.144
>>> compute_pi_mc(1000)
3.128
>>> compute_pi_mc(1000)
3.152
```

---

### 🔢 Operatore `+=` e Varianti

**Prima di continuare:** Gli operatori di incremento/decremento compatti.

```python
# Forma lunga vs forma compatta
x = 5
x = x + 1   # x diventa 6
x += 1      # Equivalente, ma più conciso

# Altri operatori simili
circle_points = 0
circle_points += 1    # circle_points = circle_points + 1
circle_points -= 1    # circle_points = circle_points - 1
circle_points *= 2    # circle_points = circle_points * 2
circle_points /= 2    # circle_points = circle_points / 2
circle_points //= 2   # circle_points = circle_points // 2 (divisione intera)
circle_points %= 3    # circle_points = circle_points % 3 (modulo)
```

**Nel contesto Monte Carlo:**
```python
if dist <= 1:
    circle_points += 1  # Incrementa contatore

square_points += 1      # Incrementa sempre
```

---

### 📦 Concatenazione Liste

**Due metodi per aggiungere elementi:**

```python
# Metodo 1: Concatenazione con +
a = [1, 2, 3]
b = [4, 5, 6]
c = a + b
print(c)  # [1, 2, 3, 4, 5, 6]

# Aggiungere un singolo elemento
lista = [1, 2, 3]
lista = lista + [5]  # lista diventa [1, 2, 3, 5]

# Metodo 2: Append (modifica in-place)
lista = [1, 2, 3]
lista.append(5)  # lista diventa [1, 2, 3, 5]
```

**Differenze:**
- `+` crea una **nuova lista** (più lento, ma sicuro)
- `.append()` modifica la lista **esistente** (più veloce)

**Quale usare?**
- Per singoli elementi: `.append()` è più veloce
- Per concatenare liste: `+` o `.extend()`

---

## 🎨 Versione con Visualizzazione

### Obiettivo
Modificare la funzione per restituire:
1. Stima finale di π
2. Coordinate $(x, y)$ dei punti **dentro** il cerchio
3. Coordinate $(x, y)$ dei punti **fuori** dal cerchio
4. Vettore con tutte le stime progressive di π (storia)

---

### 🔵 Introduzione a NumPy

**Cos'è NumPy?**
- Libreria fondamentale per calcolo scientifico in Python
- Fornisce **array n-dimensionali** (più efficienti delle liste)
- Operazioni vettoriali (senza cicli espliciti)
- Funzioni matematiche ottimizzate

**Import standard:**
```python
import numpy as np  # Alias 'np' è convenzione universale
```

---

### Creare Array NumPy

**1. Array di zeri:**
```python
>>> vv = np.zeros(5)
>>> vv
array([0., 0., 0., 0., 0.])

>>> type(vv)
<class 'numpy.ndarray'>

# Matrice 3×2 di zeri
>>> vv = np.zeros((3, 2))
>>> vv
array([[0., 0.],
       [0., 0.],
       [0., 0.]])
```

**2. Array di uni:**
```python
>>> vvv = np.ones((3, 2))
>>> vvv
array([[1., 1.],
       [1., 1.],
       [1., 1.]])
```

**3. Array vuoto (non inizializzato):**
```python
>>> vec_pi_est = np.empty(10)
>>> vec_pi_est  # Valori casuali (garbage memory)
array([6.23e-307, 1.33e-306, ...])
```

**⚠️ Nota:** `np.empty()` è più veloce ma contiene **valori casuali**. Va bene se sovrascriverai tutto.

---

**4. Da lista Python:**
```python
>>> A = np.array([[1, 2], [3, 4]])
>>> A
array([[1, 2],
       [3, 4]])

>>> type(A)
<class 'numpy.ndarray'>
```

**⚠️ Errore comune:** Liste non omogenee
```python
>>> l = [[1, 2], [3, 4, 5]]  # Lunghezze diverse!
>>> A = np.array(l)
>>> A
array([list([1, 2]), list([3, 4, 5])], dtype=object)  # Non una matrice!
```

---

### Operazioni con Array NumPy

**Somma elemento per elemento:**
```python
>>> v1 = np.array([1, 2, 3])
>>> v2 = np.array([4, 5, 6])
>>> v1 + v2
array([5, 7, 9])
```

**Prodotto vettoriale:**
```python
>>> np.cross(v1, v2)
array([-3,  6, -3])
```

**Moltiplicazione matriciale:**
```python
>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([[5, 6], [7, 8], [9, 10]])
>>> v = np.array([1, 2])

# Prodotto matrice-vettore
>>> A @ v          # Oppure np.dot(A, v)
array([5, 11])

# Prodotto matrice-matrice
>>> B @ A          # B è 3×2, A è 2×2 → risultato 3×2
array([[23, 34],
       [31, 46],
       [39, 58]])
```

**⚠️ Differenza importante:**
```python
>>> A * B   # ❌ ERRORE se dimensioni incompatibili
>>> A @ B   # ✅ Prodotto matriciale (algebra lineare)
```

---

### Implementazione Avanzata Monte Carlo

```python
import math as mt
import random
import numpy as np

def compute_pi_mc_g(N):
    """
    Stima π con Monte Carlo e memorizza dati per visualizzazione.
    
    Args:
        N (int): Numero di punti casuali
    
    Returns:
        tuple: (pi_est, x_in, y_in, x_out, y_out, vec_pi_est)
            - pi_est: stima finale di π
            - x_in, y_in: liste coordinate punti dentro
            - x_out, y_out: liste coordinate punti fuori
            - vec_pi_est: array NumPy con storia stime π
    """
    circle_points = 0
    square_points = 0
    
    # Liste per coordinate (usiamo liste Python)
    x_in = []
    y_in = []
    x_out = []
    y_out = []
    
    # Array NumPy per storia stime π
    vec_pi_est = np.empty(N)
    
    for i in range(N):
        # Genera punto casuale
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)
        
        # Calcola distanza
        dist = mt.sqrt(rand_x**2 + rand_y**2)
        
        # Classifica punto
        if dist <= 1:
            circle_points += 1
            x_in = x_in + [rand_x]      # Concatenazione
            y_in = y_in + [rand_y]
            # Oppure: x_in.append(rand_x)
        else:
            x_out.append(rand_x)        # Append
            y_out.append(rand_y)
        
        square_points += 1
        
        # Calcola e memorizza stima corrente
        pi_est = 4.0 * circle_points / square_points
        vec_pi_est[i] = pi_est
    
    return pi_est, x_in, y_in, x_out, y_out, vec_pi_est
```

**Dettagli implementativi:**
- **Liste per coordinate:** Dimostrano versatilità (possono crescere dinamicamente)
- **Array NumPy per stime:** Preallocato (più efficiente, dimensione nota)
- **Concatenazione vs append:** Entrambi dimostrati (equivalenti per singoli elementi)

---

### Output e Gestione Tuple

```python
>>> N = 10
>>> output = compute_pi_mc_g(N)
>>> type(output)
<class 'tuple'>

>>> output
(3.2, [0.234, ...], [0.567, ...], [-0.123, ...], [0.890, ...], array([0., 4., ...]))
```

**Unpacking della tupla:**
```python
# Assegna tutti i valori
>>> pi_est, x_in, y_in, x_out, y_out, vec_pi_est = compute_pi_mc_g(1000)

# Scarta valori non necessari con underscore
>>> pi_est, _, _, _, _, _ = compute_pi_mc_g(1000)
>>> print(f"π ≈ {pi_est}")
π ≈ 3.144
```

**⚠️ Nota:** Con N=10 la stima non è significativa! Usa almeno N=1000.
---

## 📊 Visualizzazione con Matplotlib

### Introduzione a Matplotlib

**Matplotlib** è la libreria Python standard per creare grafici 2D/3D.

**Import convenzione:**
```python
import matplotlib.pyplot as plt
```

**Stile:** Sintassi simile a MATLAB per familiarità.

---

### Fondamenti NumPy per Plotting

**1. `np.linspace()` vs `np.arange()` vs `range()`**

```python
# range(): genera interi, stop escluso
>>> list(range(0, 10, 2))
[0, 2, 4, 6, 8]

# np.arange(): come range, ma restituisce array NumPy
>>> np.arange(0, 10, 2)
array([0, 2, 4, 6, 8])

# np.linspace(): punti equidistanti, INCLUDE stop per default
>>> np.linspace(0, 10, 5)
array([ 0. ,  2.5,  5. ,  7.5, 10. ])

# Per escludere stop
>>> np.linspace(0, 10, 5, endpoint=False)
array([0., 2., 4., 6., 8.])
```

**Differenze chiave:**

| Funzione | Stop incluso? | Tipo output | Uso tipico |
|---|---|---|---|
| `range()` | ❌ No | range object | Cicli for |
| `np.arange()` | ❌ No | ndarray | Array numerici |
| `np.linspace()` | ✅ Sì (default) | ndarray | Grafici (punti uniformi) |

---

**2. Funzioni trigonometriche vettoriali**

```python
import math as mt

# math.cos: solo per SCALARI
>>> mt.cos(0)
1.0
>>> mt.cos([0, 1, 2])  # ❌ ERRORE!
TypeError: must be real number, not list

# np.cos/np.sin: funzionano su ARRAY (vettoriali)
>>> x = np.array([0, np.pi/2, np.pi])
>>> np.cos(x)
array([ 1.0000e+00,  6.1232e-17, -1.0000e+00])

>>> np.sin(x)
array([0.0000e+00, 1.0000e+00, 1.2246e-16])
```

**Workflow tipico per plot:**
```python
# Genera 100 punti da 0 a 10
x = np.linspace(0, 10, 100)

# Calcola y = cos(x) e y = sin(x) vettorialmente
y_cos = np.cos(x)
y_sin = np.sin(x)
```

---

### Plot Base con `plt.plot()`

**Sintassi:**
```python
plt.plot(x, y, format_string)
```

**Esempio 1: Singolo grafico**
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y_cos = np.cos(x)

plt.plot(x, y_cos, 'r--')  # Linea rossa tratteggiata
plt.show()
```

**Esempio 2: Serie multiple**
```python
x = np.linspace(0, 10, 100)
y_cos = np.cos(x)
y_sin = np.sin(x)

# Due serie nello stesso grafico
plt.plot(x, y_cos, 'r--', x, y_sin, 'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Funzioni Trigonometriche')
plt.legend(['cos(x)', 'sin(x)'])
plt.grid(True)
plt.show()
```

---

### Format String (Stili)

**Sintassi:** `'[colore][marcatore][linea]'`

**Esempi comuni:**

| String | Significato |
|---|---|
| `'r--'` | Rosso, linea tratteggiata |
| `'b'` | Blu, linea continua |
| `'go'` | Verde, cerchi (no linea) |
| `'k-.'` | Nero, linea punto-tratteggio |

**Colori:**
- `'r'` (red), `'g'` (green), `'b'` (blue), `'c'` (cyan)
- `'m'` (magenta), `'y'` (yellow), `'k'` (black), `'w'` (white)

**Stili linea:**
- `'-'` (continua), `'--'` (tratteggiata), `'-.'` (punto-tratteggio), `':'` (punteggiata)

**Marcatori:**
- `'o'` (cerchio), `'s'` (quadrato), `'^'` (triangolo), `'*'` (stella)

**Oppure usa keyword:**
```python
plt.plot(x, y, color='blue', linewidth=2, linestyle='--')
# Abbreviazioni: c='b', lw=2, ls='--'
```

---

### Subplot: Grafici Affiancati

**Creare layout 1×2 (una riga, due colonne):**

```python
fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1 e ax2 sono "assi" (subplot)
# Puoi plottare su ciascuno separatamente
```

**Equivalente esplicito:**
```python
fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax2 = axes[1]
```

**Altri layout:**
```python
# 2 righe, 1 colonna
fig, (ax1, ax2) = plt.subplots(2, 1)

# 2×2 griglia
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
```

---

## 🎨 Visualizzazione Monte Carlo Completa

### Subplot 1: Punti nel Quadrato e Cerchio

```python
import matplotlib.pyplot as plt
import numpy as np

# Esegui simulazione
N = 1000
pi_est, x_in, y_in, x_out, y_out, vec_pi_est = compute_pi_mc_g(N)

# Crea figura con 2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2)

# ===== SUBPLOT 1: Scatter plot dei punti =====

# 1. Disegna cerchio unitario
angles = np.linspace(0, 2*np.pi, 100)
xs = np.cos(angles)
ys = np.sin(angles)
ax1.plot(xs, ys, color='blue')

# 2. Imposta rapporto aspetto uguale (cerchio non distorto)
ax1.set_aspect(1)

# 3. Converti liste in array NumPy (per Matplotlib)
x_in = np.array(x_in)
y_in = np.array(y_in)
x_out = np.array(x_out)
y_out = np.array(y_out)

# 4. Scatter plot: punti dentro (rosso) e fuori (verde)
ax1.scatter(x_in, y_in, c='r', s=5, alpha=0.6)    # c=colore, s=size, alpha=trasparenza
ax1.scatter(x_out, y_out, c='g', s=5, alpha=0.6)

# 5. Aggiungi griglia
ax1.grid(True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'Monte Carlo: N={N}')
```

**Dettagli:**
- `np.linspace(0, 2*np.pi, 100)` → 100 angoli per disegnare cerchio
- `xs = np.cos(angles)`, `ys = np.sin(angles)` → coordinate cerchio
- `.set_aspect(1)` → rapporto 1:1 (evita cerchio schiacciato)
- `.scatter()` → plot a nuvola di punti (non linee)
- `c='r'` → colore rosso
- `s=5` → dimensione punti
- `alpha=0.6` → trasparenza 60%

---

### Subplot 2: Convergenza di π

```python
# ===== SUBPLOT 2: Storia delle stime di π =====

# Plot vettore stime (indice implicito come asse x)
ax2.plot(vec_pi_est, 'b-', linewidth=0.8)

# Linea orizzontale per π vero
ax2.axhline(y=np.pi, color='r', linestyle='--', label=r'$\pi$ vero')

# Imposta rapporto aspetto per visualizzazione migliore
ax2.set_aspect(250)  # Espande asse y rispetto a x

# Etichette e griglia
ax2.set_xlabel('Iterazione')
ax2.set_ylabel(r'Stima $\pi$')
ax2.set_title('Convergenza')
ax2.grid(True)
ax2.legend()

# Mostra entrambi i subplot
plt.tight_layout()  # Evita sovrapposizione etichette
plt.show()
```

**Dettagli:**
- `ax2.plot(vec_pi_est)` → se non specifichi x, usa indici [0, 1, 2, ...]
- `.axhline(y=np.pi)` → linea orizzontale a y=π
- `.set_aspect(250)` → rende visibile la convergenza (scala y)
- `r'$\pi$'` → LaTeX per simbolo matematico π
- `.tight_layout()` → aggiusta spaziatura automaticamente

**Comportamento visivo:**
- All'inizio: stime molto variabili (pochi punti)
- Dopo ~100 iterazioni: oscillazioni più piccole
- Con N grande: asintoticamente vicino a π

---

### Risultato Completo

```python
import math as mt
import random
import numpy as np
import matplotlib.pyplot as plt

# Funzione completa (già definita sopra)
def compute_pi_mc_g(N):
    # ... (implementazione precedente)
    pass

# Esegui simulazione
N = 1000
pi_est, x_in, y_in, x_out, y_out, vec_pi_est = compute_pi_mc_g(N)

# Crea visualizzazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Punti
angles = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(angles), np.sin(angles), 'b-')
ax1.scatter(np.array(x_in), np.array(y_in), c='r', s=5, alpha=0.6, label='Dentro')
ax1.scatter(np.array(x_out), np.array(y_out), c='g', s=5, alpha=0.6, label='Fuori')
ax1.set_aspect(1)
ax1.grid(True)
ax1.legend()
ax1.set_title(f'N={N}, π≈{pi_est:.4f}')

# Subplot 2: Convergenza
ax2.plot(vec_pi_est, 'b-', linewidth=0.8)
ax2.axhline(y=np.pi, color='r', linestyle='--', label=r'$\pi=3.14159...$')
ax2.set_aspect(250)
ax2.grid(True)
ax2.legend()
ax2.set_xlabel('Iterazione')
ax2.set_ylabel(r'Stima $\pi$')
ax2.set_title('Convergenza')

plt.tight_layout()
plt.show()
```

**Verifica risultato:**
```python
>>> print(f"Stima: {pi_est:.6f}")
>>> print(f"Vero:  {np.pi:.6f}")
>>> print(f"Errore: {abs(pi_est - np.pi):.6f}")
Stima: 3.144000
Vero:  3.141593
Errore: 0.002407
```

---

### Operazioni Array NumPy Avanzate

**Dal notebook - Altri esempi utili:**

```python
# Conversione lista → array
>>> l = [1, 2, 3, 4, 5]
>>> la = np.array(l)
>>> type(la)
<class 'numpy.ndarray'>

# Diagonale di una matrice
>>> la = np.array([[1, 2], [3, 4]])
>>> np.diag(la)
array([1, 4])

# Matrice diagonale con offset
>>> ww = np.array([1, 2, 3, 4, 5])
>>> AA = np.diag(ww, -1)  # Diagonale sotto quella principale
>>> AA
array([[0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0],
       [0, 2, 0, 0, 0, 0],
       [0, 0, 3, 0, 0, 0],
       [0, 0, 0, 4, 0, 0],
       [0, 0, 0, 0, 5, 0]])

# Dimensione e forma
>>> np.size(AA)
36
>>> AA.shape
(6, 6)
>>> np.shape(AA)
(6, 6)

# Estrazione righe/colonne
>>> gg = AA[:, 2]  # Colonna 2
>>> np.shape(gg)
(6,)
>>> hh = AA[4, :]  # Riga 4
>>> np.shape(hh)
(6,)

# Numeri casuali
>>> BB = np.random.rand(6, 4)  # Matrice 6×4 di valori uniformi [0, 1)
>>> BB
array([[0.234, 0.567, ...],
       ...])

# Moltiplicazione matriciale
>>> AA @ BB   # Equivalente a np.dot(AA, BB)
>>> AA * BB   # ❌ ERRORE se dimensioni diverse (operazione elemento per elemento)
```

**⚠️ Reminder:** `@` è prodotto matriciale, `*` è elemento per elemento.
---

## 🏔️ Progetto 9: Visualizzazione 3D di Funzioni

### Competenze Sviluppate
- Plot di superfici 3D (`plot_surface`)
- Uso di `meshgrid` per creare griglie 2D
- Modulo `mpl_toolkits.mplot3d`
- Grafici a contorno (`contour`)
- Ottimizzazione e analisi visuale di funzioni

---

### 🎯 Problema: Funzione di Himmelblau

**Definizione matematica:**

$$
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
$$

**Proprietà interessanti:**
- **4 minimi globali identici** (tutti con $f = 0$):
  1. $(3.0, 2.0)$
  2. $(-2.805118, 3.131312)$
  3. $(-3.779310, -3.283186)$
  4. $(3.584428, -1.848126)$

- **1 massimo locale**:
  - $(-0.270845, -0.923039)$ con $f \approx 181.617$

**Uso:** Test comune per algoritmi di ottimizzazione (multi-modal).

**Obiettivo:** Visualizzare $f(x, y)$ per $x, y \in [-5, 5]$.

---

### 🔲 Creazione della Griglia con `meshgrid`

**Problema:** Per valutare $f(x, y)$ su una griglia 2D, servono **matrici** di coordinate.

**Soluzione:** `np.meshgrid()` crea griglie da vettori 1D.

**Esempio concettuale:**
```python
# Vettori 1D per x e y
x_vals = np.arange(-5, 5, 0.2)  # Da -5 a 5, step 0.2
y_vals = x_vals                 # Stessa griglia per y

# Crea meshgrid
X, Y = np.meshgrid(x_vals, y_vals)
```

**Cosa fa `meshgrid`?**
- `X` → matrice dove ogni **riga** ripete `x_vals`
- `Y` → matrice dove ogni **colonna** ripete `y_vals`

**Esempio 3×3:**
```python
>>> x = np.array([1, 2, 3])
>>> y = np.array([10, 20, 30])
>>> X, Y = np.meshgrid(x, y)

>>> X
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])

>>> Y
array([[10, 10, 10],
       [20, 20, 20],
       [30, 30, 30]])
```

**Risultato:** Ogni coppia `(X[i,j], Y[i,j])` rappresenta un punto nella griglia.

**Equivalente a MATLAB:** Identica sintassi!

---

### 📐 Implementazione della Funzione

```python
def Himmelblau(x):
    """
    Calcola la funzione di Himmelblau.
    
    Args:
        x: array-like con [x_coord, y_coord]
           Può essere vettori per operazioni vettoriali
    
    Returns:
        float o ndarray: valore della funzione
    
    Formula: f(x,y) = (x² + y - 11)² + (x + y² - 7)²
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
```

**⚠️ Notazione:** `x` è un array con `x[0]` = coordinata x, `x[1]` = coordinata y.

---

### 🌐 Grafico 3D con `plot_surface`

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Per plotting 3D

# 1. Crea vettori per assi
xaxis = np.arange(-5, 5, 0.1)  # 100 punti da -5 a 5
yaxis = np.arange(-5, 5, 0.1)

# 2. Crea meshgrid
X, Y = np.meshgrid(xaxis, yaxis)

# 3. Calcola Z su tutta la griglia (vettorializzato)
Z = Himmelblau([X, Y])

# 4. Crea figura con proiezione 3D
fig = plt.figure(figsize=(10, 7))
axis = fig.add_subplot(111, projection='3d')
# Oppure: axis = plt.subplot(projection='3d')

# 5. Plot superficie
axis.plot_surface(X, Y, Z, cmap='jet')

# 6. Etichette e titolo
axis.set_xlabel('x')
axis.set_ylabel('y')
axis.set_zlabel('f(x, y)')
axis.set_title('Funzione di Himmelblau')

# 7. Mostra grafico
plt.show()

# 8. (Opzionale) Salva figura
fig.savefig('himmelblau_3d.png', dpi=300)
```

**Dettagli:**
- `projection='3d'` → abilita assi 3D
- `cmap='jet'` → color map (blu→rosso per valori bassi→alti)
- Altre colormap: `'viridis'`, `'plasma'`, `'coolwarm'`, `'terrain'`
- `.savefig()` → salva immagine (formati: png, pdf, svg, ...)

---

### 🎨 Colormap e Parametri

**Colormap comuni:**
```python
# Colori freddi → caldi
axis.plot_surface(X, Y, Z, cmap='jet')       # Blu → Verde → Giallo → Rosso
axis.plot_surface(X, Y, Z, cmap='viridis')   # Blu/Viola → Verde → Giallo
axis.plot_surface(X, Y, Z, cmap='plasma')    # Viola → Rosa → Giallo

# Divergenti (per enfatizzare zero)
axis.plot_surface(X, Y, Z, cmap='coolwarm')  # Blu ← Bianco → Rosso
axis.plot_surface(X, Y, Z, cmap='seismic')   # Blu ← Bianco → Rosso

# Terreno
axis.plot_surface(X, Y, Z, cmap='terrain')   # Simula altitudine geografica
```

**Altri parametri:**
```python
axis.plot_surface(X, Y, Z, 
                  cmap='viridis',
                  alpha=0.8,           # Trasparenza (0-1)
                  linewidth=0.2,       # Spessore linee griglia
                  antialiased=True,    # Smoothing
                  edgecolor='gray')    # Colore linee griglia
```

---

### 📉 Grafici a Contorno 2D

**Motivazione:** Per **ottimizzazione**, i contour plot sono spesso migliori delle superfici 3D:
- Più facile vedere percorsi di discesa
- Mostra isovalori (curve di livello)
- Non oscura parti della funzione

**Implementazione:**
```python
fig = plt.figure(figsize=(8, 6))
axis = plt.subplot()

# Contour plot con 100 livelli
contour = axis.contour(X, Y, Z, levels=100, cmap='jet')

# (Opzionale) Aggiungi barra colori
plt.colorbar(contour, label='f(x, y)')

# Etichette
axis.set_xlabel('x')
axis.set_ylabel('y')
axis.set_title('Contour Plot - Funzione di Himmelblau')

# Rapporto aspetto uguale
axis.set_aspect('equal')

plt.show()
```

**Varianti:**
```python
# Contour riempito (filled)
axis.contourf(X, Y, Z, levels=100, cmap='jet')

# Contour con linee + riempimento
axis.contourf(X, Y, Z, levels=100, cmap='jet', alpha=0.7)
axis.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5)
```

---

### 🔍 Visualizzazione dei Minimi

**Aggiungi marker per i 4 minimi:**
```python
# Coordinate dei minimi
minima = np.array([
    [3.0, 2.0],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126]
])

# Su grafico 3D
axis.scatter(minima[:, 0], minima[:, 1], 
             [0, 0, 0, 0],  # z=0 per tutti i minimi
             c='red', s=100, marker='*', 
             label='Minimi globali')
axis.legend()

# Su contour plot
axis.scatter(minima[:, 0], minima[:, 1], 
             c='red', s=100, marker='*', 
             edgecolors='black', linewidths=2,
             label='Minimi globali', zorder=10)
axis.legend()
```

**Dettagli:**
- `zorder=10` → disegna sopra i contour
- `edgecolors='black'` → bordo nero per visibilità

---

### 📊 Confronto 3D vs Contour

**Codice completo per confronto:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definizione funzione
def Himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Griglia
xaxis = np.arange(-5, 5, 0.1)
yaxis = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(xaxis, yaxis)
Z = Himmelblau([X, Y])

# Minimi
minima = np.array([[3.0, 2.0], [-2.805118, 3.131312],
                   [-3.779310, -3.283186], [3.584428, -1.848126]])

# Crea figura con 2 subplot
fig = plt.figure(figsize=(16, 6))

# Subplot 1: 3D
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
ax1.scatter(minima[:, 0], minima[:, 1], [0]*4, 
            c='red', s=100, marker='*', label='Minimi')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.set_title('Superficie 3D')
ax1.legend()

# Subplot 2: Contour
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
ax2.contour(X, Y, Z, levels=20, colors='black', linewidths=0.3, alpha=0.4)
ax2.scatter(minima[:, 0], minima[:, 1], 
            c='red', s=150, marker='*', edgecolors='white', 
            linewidths=2, zorder=10, label='Minimi')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Plot')
ax2.set_aspect('equal')
ax2.legend()
plt.colorbar(contour, ax=ax2, label='f(x, y)')

plt.tight_layout()
plt.show()
```

---

### 🧮 Analisi Numerica dei Minimi

```python
# Verifica che i minimi siano effettivamente zeri
for i, (x, y) in enumerate(minima, 1):
    f_val = Himmelblau([x, y])
    print(f"Minimo {i}: f({x:.6f}, {y:.6f}) = {f_val:.10e}")
```

**Output:**
```
Minimo 1: f(3.000000, 2.000000) = 0.0000000000e+00
Minimo 2: f(-2.805118, 3.131312) = 2.8477098736e-12
Minimo 3: f(-3.779310, -3.283186) = 1.4210854715e-14
Minimo 4: f(3.584428, -1.848126) = 1.4210854715e-14
```

**Conclusione:** Tutti i minimi sono effettivamente $\approx 0$ (errori numerici trascurabili).

---

### 🎓 Concetti Avanzati

**1. Slicing di array 2D:**
```python
# Estrai colonna 3 dalla matrice AA
>>> gg = AA[:, 2]
>>> np.shape(gg)
(6,)

# Estrai riga 4
>>> hh = AA[4, :]
>>> np.shape(hh)
(6,)
```

**2. Prodotto scalare:**
```python
>>> v = np.array([1, 2])
>>> AA = np.array([[1, 2], [3, 4]])
>>> np.dot(v, AA)  # Vettore-matrice
array([7, 10])
>>> AA @ v         # Matrice-vettore
array([5, 11])
```

**3. Numeri casuali:**
```python
>>> BB = np.random.rand(6, 4)  # Uniforme [0, 1)
>>> BB.shape
(6, 4)
```

**4. Append per array NumPy:**
```python
>>> v = np.array([1, 2, 3])
>>> v_new = np.append(v, 4)  # Restituisce NUOVO array
>>> v_new
array([1, 2, 3, 4])
>>> v  # Originale NON modificato
array([1, 2, 3])
```

**⚠️ Differenza con liste:**
- Liste: `.append()` modifica in-place
- NumPy: `np.append()` crea nuovo array (immutabilità parziale)

---

### 📚 Librerie di Plotting - Panoramica

**Matplotlib:**
- Libreria **generica** e **fondamentale**
- Sufficiente per la maggior parte delle esigenze
- Sintassi tipo MATLAB
- **Usa questa per iniziare**

**Seaborn:**
- Basata su Matplotlib
- Specializzata in **visualizzazioni statistiche**
- Grafici: distribuzioni, box plot, heatmap, pair plot
- Estetica moderna di default

**Bokeh:**
- Grafici **interattivi** per web
- Slider, zoom, pan, tooltip
- Migliore per dashboard e applicazioni web
- Più semplice di Plotly per interattività avanzata

**Altri:**
- **Plotly:** Interattività avanzata, 3D, animazioni
- **Altair:** Dichiarativo, basato su grammatica grafica
- **ggplot:** Port di R ggplot2 (meno popolare in Python)

**Raccomandazione:** Inizia con **Matplotlib**, poi esplora Seaborn e Bokeh.
  - plt.contour(X, Y, Z) per disegnare le curve di livello di Z sul dominio XY.
  - Tracciare i percorsi dai punti iniziali ai minimi su una mappa 2D.
## Logistica del corso e contenuti futuri
- Materiali a breve termine:
  - Il notebook fino alla parte sui grafici sarà pubblicato entro questa sera.
- Prossime lezioni:
  - Lunedì: introduzione a SciPy (minimizzazione, integrazione, ODE, calcolo scientifico) e Pandas (gestione e analisi dei dati).
  - Martedì: inizio della parte teorica del libro.
- Laboratori:
  - Approfondimento su NumPy e Matplotlib oltre le basi.
## Esempi pratici e concetti chiave
- Monte Carlo π:
  - Usare array preallocati quando N è noto per prestazioni migliori.
  - Memorizzare le stime di π per ogni iterazione per analisi di convergenza.
- Grafici:
  - Matplotlib è la principale; usare sottoplot per visualizzazioni affiancate.
  - Scatter per i punti; plot per le linee continue.
  - Regolare i rapporti d'aspetto per migliorare la leggibilità.
- NumPy:
  - Preferire np.cos/np.sin per gli array; math.cos è per scalari.
  - Scegliere linspace per includere l'endpoint; modificarlo secondo necessità.
- Scelte di visualizzazione:
  - Matplotlib per grafici generali; Seaborn per statistiche; Bokeh per esigenze interattive.
  - Grafici a contorno per traiettorie di ottimizzazione; grafici di superficie per la forma della funzione.
- Contenitori Python:
  - Le liste sono eterogenee e flessibili (concatenazione con “+”, append per efficienza).
  - Gli array NumPy sono contenitori numerici omogenei con operazioni vettoriali efficienti.

---

## 📚 Riepilogo Concetti Chiave della Lezione 3

### Ricorsione e Iterazione
- **Ricorsione:** Elegante ma O(2^n) - solo per problemi piccoli o con memoization
- **Iterazione:** O(n) - preferibile per performance
- **Fibonacci:** Caso studio perfetto per confronto

### Strutture Dati Python
- **List comprehension:** `[expr for item in seq]` - conciso ed espressivo
- **Slicing:** `[start:stop:step]` con indici negativi `[-1]`, `[-2]`
- **Range:** `range(start, stop, step)` - stop escluso

### NumPy Fondamenti
- **Array n-dimensionali:** `np.array()`, `np.zeros()`, `np.empty()`
- **Operazioni vettoriali:** Niente cicli! `np.cos(array)`, `array1 + array2`
- **Meshgrid:** `np.meshgrid(x, y)` per griglie 2D
- **Prodotto matriciale:** `@` oppure `np.dot()`
- **Differenza con liste:** Array più veloci per calcolo numerico

### Matplotlib Essenziale
- **Import:** `import matplotlib.pyplot as plt`
- **Plot base:** `plt.plot(x, y, 'r--')` con format string
- **Subplot:** `fig, (ax1, ax2) = plt.subplots(1, 2)`
- **Scatter:** `ax.scatter(x, y, c='r', s=10)` per punti
- **3D:** `from mpl_toolkits.mplot3d import Axes3D` + `projection='3d'`

### Metodo Monte Carlo
- **Principio:** Simulazione statistica per approssimazioni
- **Convergenza:** O(1/√N) - errore decresce lentamente
- **Applicazioni:** Integrazione, probabilità, ottimizzazione

### Algoritmi Implementati
1. **Fibonacci ricorsivo/iterativo** - Complessità, golden ratio
2. **Recamán** - Condizioni logiche, operatore `in`
3. **Monte Carlo π** - Random sampling, visualizzazione
4. **Himmelblau 3D** - Meshgrid, superficie, contour

---

## 📋 Materiali di Riferimento Utilizzati

Questa lezione è stata ricostruita utilizzando:

1. **Python_Lecture3.ipynb** (574 righe) - Notebook principale contenente:
   - Progetto 7: Fibonacci (ricorsivo/iterativo) con esercizi golden ratio
   - Esercizio: Sequenza di Recamán
   - Progetto 8: Stima di π con Monte Carlo (base + visualizzazione)
   - Introduzione NumPy: array, operazioni, meshgrid
   - Introduzione Matplotlib: plot, subplot, scatter, 3D
   - Progetto 9: Funzione Himmelblau (superficie 3D e contour)

2. **Lez3-19sett.md** (file base - 244 righe) - Conteneva già ottima sintesi

**Note:** Nessuna cartella PDF separata per il 19 settembre - tutti i contenuti sono nel notebook.

---

## 📅 Prossimi Appuntamenti

### Per la Prossima Lezione (Lunedì 22 Settembre)
- [ ] **Completare gli esercizi proposti:**
  - ✅ Fibonacci: Calcola rapporti $r_2 = F(n)/F(n+2)$ e $r_3 = F(n)/F(n+3)$
  - ✅ Recamán: Trova i primi 10 numeri mancanti nella sequenza
  - ✅ Monte Carlo: Esegui con N=100,000 e confronta precisione
  - ✅ Himmelblau: Identifica visualmente i 4 minimi sul contour plot

- [ ] **Rivedere i concetti chiave:**
  - Differenza tra ricorsione e iterazione
  - Quando usare liste vs array NumPy
  - Come creare meshgrid per funzioni 2D
  - Subplot e scatter plot in Matplotlib

- [ ] **Esercizi aggiuntivi suggeriti:**
  - **Fibonacci esteso:** Implementa con memoization (caching) per velocizzare ricorsione
  - **Monte Carlo 3D:** Estendi a sfera in cubo 3D per stimare volume
  - **Altre funzioni test:** Prova Rosenbrock, Rastrigin, Ackley
  - **Animazione:** Usa `matplotlib.animation` per mostrare convergenza Monte Carlo

### Setup e Pratica
- [ ] Installare/verificare NumPy e Matplotlib: `pip install numpy matplotlib`
- [ ] Sperimentare con diverse colormap per 3D plot
- [ ] Provare a modificare `set_aspect()` per vedere effetti
- [ ] Confrontare velocità liste vs NumPy con `%timeit` (Jupyter)

### Cosa Aspettarsi Prossimamente
- **Lunedì 22/09:** Continuazione NumPy (broadcasting, indexing avanzato)
- **Questa settimana:** Più operazioni con array, algebra lineare
- **Prossima settimana:** Pandas per data analysis, lettura CSV/Excel

---

## 💡 Suggerimenti per la Pratica

### 1. Memoization per Fibonacci Ricorsivo
```python
# Usa un dizionario per cachare risultati
cache = {}

def recfib_memo(n):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    result = recfib_memo(n-1) + recfib_memo(n-2)
    cache[n] = result
    return result

# Ora anche recfib_memo(100) è veloce!
```

### 2. Timing delle Funzioni
```python
import time

# Metodo 1: manuale
start = time.time()
result = fib(1000)
end = time.time()
print(f"Tempo: {end - start:.4f} secondi")

# Metodo 2: timeit (più accurato)
import timeit
tempo = timeit.timeit('fib(100)', globals=globals(), number=1000)
print(f"Tempo medio: {tempo/1000:.6f} secondi")
```

### 3. Generatore per Fibonacci (Memory Efficient)
```python
def fib_generator(n):
    """Genera Fibonacci senza memorizzare tutta la lista."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Uso
for i, fib_num in enumerate(fib_generator(20)):
    print(f"F({i}) = {fib_num}")
```

### 4. Vettorializzazione NumPy Avanzata
```python
# Invece di ciclo for
risultato = []
for x in X:
    risultato.append(np.sin(x) ** 2 + np.cos(x) ** 2)

# Usa operazioni vettoriali (molto più veloce!)
risultato = np.sin(X)**2 + np.cos(X)**2  # Tutti ≈ 1
```

### 5. Debugging Grafici
```python
# Problemi comuni e soluzioni

# 1. Cerchio distorto
ax.set_aspect('equal')  # Fix!

# 2. Non vedo i punti
ax.scatter(x, y, s=50, zorder=10)  # Aumenta size e z-order

# 3. Colori non si vedono
ax.scatter(x, y, c='red', edgecolors='black', linewidths=1)

# 4. Grafico troppo piccolo
fig, ax = plt.subplots(figsize=(10, 8))  # Dimensioni in pollici

# 5. Testo tagliato
plt.tight_layout()  # Prima di show()
```

---

## 🎯 Checklist di Padronanza

Verifica di aver capito questi concetti:

**Fibonacci e Ricorsione:**
- [ ] Capisco perché la ricorsione è O(2^n)
- [ ] So quando usare ricorsione vs iterazione
- [ ] Capisco il concetto di golden ratio
- [ ] So implementare memoization

**Recamán:**
- [ ] Capisco la logica con `and` e `in`
- [ ] So perché alcuni numeri non appaiono mai
- [ ] Capisco la differenza con Fibonacci (crescita)

**NumPy:**
- [ ] So creare array con `zeros`, `ones`, `empty`, `array`
- [ ] Capisco differenza array vs liste Python
- [ ] So usare operazioni vettoriali (no cicli)
- [ ] Capisco `meshgrid` per funzioni 2D
- [ ] So fare prodotto matriciale con `@`
- [ ] Capisco slicing 2D: `A[:, 2]` vs `A[2, :]`

**Matplotlib:**
- [ ] So creare plot base con `plt.plot()`
- [ ] Capisco format string (`'r--'`, `'bo'`, etc.)
- [ ] So creare subplot con `subplots()`
- [ ] Capisco differenza `plot` vs `scatter`
- [ ] So creare grafico 3D con `projection='3d'`
- [ ] So fare contour plot con `contour()`/`contourf()`

**Monte Carlo:**
- [ ] Capisco il principio geometrico (area cerchio/quadrato)
- [ ] So perché convergenza è O(1/√N)
- [ ] Capisco la casualità (seed per riproducibilità)
- [ ] So visualizzare risultati con scatter

---

## 🚀 Sfide Avanzate (Opzionali)

### Sfida 1: Monte Carlo Multi-Dimensionale
Estendi a 3D: stima il volume di una sfera unitaria in un cubo $[-1, 1]^3$.
**Hint:** Volume sfera = $\frac{4}{3}\pi r^3$, cubo = $8$.

### Sfida 2: Animazione Convergenza
Usa `matplotlib.animation.FuncAnimation` per animare la convergenza di π.

### Sfida 3: Ottimizzazione Himmelblau
Implementa gradient descent per trovare un minimo partendo da punto casuale.

### Sfida 4: Fibonacci Matrice
Formula chiusa con matrici:
$$
\begin{bmatrix} F(n+1) \\ F(n) \end{bmatrix} = 
\begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}^n
\begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$
Implementa e confronta velocità con iterazione.

### Sfida 5: Fractal con Matplotlib
Disegna il set di Mandelbrot usando `meshgrid` e `contourf`.

---

## 📖 Risorse Extra

### Documentazione Ufficiale
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Python Official Docs - Recursion](https://docs.python.org/3/faq/programming.html#how-do-i-create-a-multidimensional-list)

### Tutorial Interattivi
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) - Copia/modifica esempi!
- [Real Python - NumPy](https://realpython.com/numpy-tutorial/)

### Video Consigliati
- Corey Schafer: "NumPy Tutorial" (YouTube)
- Keith Galli: "Complete Python NumPy Tutorial"
- sentdex: "Matplotlib Tutorial Series"

### Libri (Opzionali)
- **"Python Data Science Handbook"** - Jake VanderPlas (gratis online)
- **"NumPy Beginner's Guide"** - Ivan Idris

---

## 🔍 Debugging e Errori Comuni

### TypeError: 'float' object is not subscriptable
```python
# ❌ Errore
def func(x):
    return x[0]**2
func(5)  # ERRORE!

# ✅ Fix: accetta sia scalari che array
def func(x):
    x = np.atleast_1d(x)  # Converte scalare in array
    return x[0]**2
```

### IndexError: list index out of range
```python
# ❌ Errore con range(40) ma lista ha 41 elementi
for n in range(40):
    print(F[n+1])  # F[40] non esiste!

# ✅ Fix
for n in range(len(F)-1):  # Oppure range(40) con F[n] invece di F[n+1]
```

### ValueError: shapes mismatch
```python
# ❌ Errore
A = np.array([[1, 2]])    # Shape (1, 2)
B = np.array([[3], [4]])  # Shape (2, 1)
A + B  # ERRORE dimensioni incompatibili senza broadcasting

# ✅ Fix: usa transpose o reshape
A.T + B  # Ora (2, 1) + (2, 1) = OK
```

---

**Fine Lezione 3 - 19 Settembre 2025**

**Prossima lezione:** Lunedì 22 Settembre - NumPy avanzato e Pandas intro