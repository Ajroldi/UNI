## Lezione 2: Python Avanzato - Strutture Dati e Algoritmi
### 16 Settembre 2025

Questa lezione approfondisce concetti Python più avanzati attraverso progetti pratici. Continueremo l'approccio "learning by doing" iniziato nella prima lezione, introducendo strutture dati fondamentali e algoritmi classici.

---

## Strutture Dati Fondamentali di Python

### Tuple: Sequenze Immutabili

Le **tuple** sono sequenze ordinate di elementi racchiusi tra **parentesi tonde** `()`.

```python
# Esempio di tupla creata dalla lezione precedente
R = (18.84955592153876, 28.274333882308138)
```

**Caratteristiche principali:**

#### 1. Verificare il Tipo
```python
print(type(R))  # <class 'tuple'>
```

#### 2. Indicizzazione (Zero-Based)
A differenza di MATLAB che usa indicizzazione da 1, Python inizia da **indice 0**:

```python
print(R[0])  # 18.849... (primo elemento)
print(R[1])  # 28.274... (secondo elemento)
```

**Indicizzazione negativa** - accede agli elementi dalla fine:
```python
print(R[-1])  # 28.274... (ultimo elemento)
print(R[-2])  # 18.849... (penultimo elemento)
```

#### 3. Immutabilità ⚠️
Le tuple **NON possono essere modificate** dopo la creazione:

```python
R[1] = 6  # ❌ TypeError: 'tuple' object does not support item assignment
```

Questo garantisce l'integrità dei dati ma richiede di creare una nuova tupla per modifiche.

#### 4. Eterogeneità
Le tuple possono contenere elementi di **tipi diversi**:

```python
import math as mt
c = (1, 'a', "stringa", mt.sqrt(2))
print(c)  # (1, 'a', 'stringa', 1.4142135623730951)
print(type(c[0]))  # <class 'int'>
print(type(c[1]))  # <class 'str'>
print(type(c[3]))  # <class 'float'>
```

### Liste: Sequenze Mutabili

Le **liste** sono sequenze ordinate racchiuse tra **parentesi quadre** `[]`.

```python
# Creazione di liste
numeri = [1, 2, 3, 4, 5]
mista = [1, 'ciao', 3.14, True]
vuota = []
```

**Differenze chiave con le tuple:**

| Caratteristica | Tuple `()` | Liste `[]` |
|----------------|-----------|-----------|
| Mutabilità | ❌ Immutabili | ✅ Mutabili |
| Sintassi | `(1, 2, 3)` | `[1, 2, 3]` |
| Velocità | Più veloci | Più lente |
| Uso | Dati fissi | Dati dinamici |

```python
# Liste SONO mutabili
lista = [1, 2, 3]
lista[1] = 99
print(lista)  # [1, 99, 3] ✅ Funziona!

# Operazioni su liste
lista.append(4)     # Aggiunge alla fine
print(lista)        # [1, 99, 3, 4]
```

**⚠️ Attenzione:** L'operatore `+` con le liste fa **concatenazione**, non somma matematica:

```python
l1 = [1, 2, 3]
l2 = [4, 5, 6]
print(l1 + l2)  # [1, 2, 3, 4, 5, 6] (concatenazione)
# NON [5, 7, 9] (somma elemento per elemento - quello richiede NumPy)
```

### Dizionari: Coppie Chiave-Valore

I **dizionari** sono collezioni **non ordinate** di coppie chiave-valore, racchiuse tra **parentesi graffe** `{}`.

```python
# Creazione di un dizionario
studente = {
    'nome': 'Mario',
    'cognome': 'Rossi',
    'età': 22,
    'voti': [28, 30, 27]
}

# Accesso ai valori tramite chiave
print(studente['nome'])  # 'Mario'
print(studente['voti'])  # [28, 30, 27]
```

**Esempio pratico: Numeri Romani**
```python
# Mappatura simboli romani → valori decimali
symbols = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000
}

print(symbols['M'])  # 1000
print(symbols['V'])  # 5
```

**Operazioni utili:**
```python
# Verificare se una chiave esiste
print('M' in symbols)        # True
print('pippo' in symbols)    # False

# Ottenere tutte le chiavi
print(symbols.keys())        # dict_keys(['I', 'V', 'X', ...])

# Ottenere tutti i valori
print(symbols.values())      # dict_values([1, 5, 10, ...])
```

**Dizionario con tipi misti:**
```python
g = {
    'pippo': 3,
    'pluto': 7,
    'elio': 5,
    4: 7,                    # Chiave intera
    'paperino': [2, 3]       # Valore lista
}

print(g['elio'])    # 5
print(g[4])         # 7
print(g['paperino'][0])  # 2
```

### Stringhe: Comportamento da Sequenza

Le **stringhe** in Python si comportano come tuple per quanto riguarda l'indicizzazione e lo slicing.

```python
s = "Python"

# Indicizzazione positiva (da sinistra)
print(s[0])   # 'P' (primo carattere)
print(s[1])   # 'y'
print(s[5])   # 'n' (ultimo carattere)

# Indicizzazione negativa (da destra)
print(s[-1])  # 'n' (ultimo)
print(s[-2])  # 'o' (penultimo)
print(s[-6])  # 'P' (primo)
```

Le stringhe sono definite con virgolette singole `'...'` o doppie `"..."`:

```python
stringa1 = 'Ciao'
stringa2 = "Mondo"
stringa3 = 'Python è "fantastico"!'  # Virgolette miste
```

---

## Manipolazione delle Sequenze: Indicizzazione e Slicing

Lo **slicing** permette di estrarre porzioni (sottostringhe, sottoliste) da sequenze.

### Sintassi Base dello Slicing

**Formato:** `sequenza[start:stop]`

- `start`: indice iniziale (incluso)
- `stop`: indice finale (**escluso**)

```python
s = "numerical"

print(s[0:4])   # 'nume' (caratteri 0,1,2,3 - stop escluso!)
print(s[1:5])   # 'umer' (caratteri 1,2,3,4)
print(s[2:7])   # 'meric' (caratteri 2,3,4,5,6)
```

**Visualizzazione degli indici:**
```
 n   u   m   e   r   i   c   a   l
 0   1   2   3   4   5   6   7   8   (indici positivi)
-9  -8  -7  -6  -5  -4  -3  -2  -1   (indici negativi)
```

### Omissione degli Indici

```python
s = "numerical"

# Ometti start → parte dall'inizio
print(s[:4])    # 'nume' (equivale a s[0:4])

# Ometti stop → arriva fino alla fine
print(s[1:])    # 'umerical' (dal carattere 1 alla fine)

# Ometti entrambi → copia dell'intera sequenza
print(s[:])     # 'numerical'
```

### Slicing con Step (Passo)

**Formato completo:** `sequenza[start:stop:step]`

⚠️ **Differenza da MATLAB:** In Python è `start:stop:step`, MATLAB usa `start:step:stop`

```python
s = "numerical"

# Step positivo
print(s[::2])    # 'nmrcl' (ogni 2° carattere: 0,2,4,6,8)
print(s[1::2])   # 'ueia' (ogni 2° carattere partendo da 1)
print(s[0:8:3])  # 'nec' (caratteri 0,3,6)

# Esempi con liste
numeri = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numeri[2:8:2])   # [2, 4, 6] (da indice 2 a 8, step 2)
print(numeri[::3])     # [0, 3, 6, 9] (ogni 3° elemento)
```

### Inversione con Slicing Negativo

Uno **step negativo** inverte la direzione di attraversamento:

```python
s = "numerical"

# Inversione completa (idioma comune!)
print(s[::-1])   # 'laciremun' (sequenza invertita)

# Slicing con step negativo
print(s[8:2:-1])  # 'lacire' (da 8 a 2, all'indietro)
print(s[::-2])    # 'lcrmn' (ogni 2° carattere, invertito)
```

**Inversione di liste:**
```python
lista = [1, 2, 3, 4, 5]
print(lista[::-1])  # [5, 4, 3, 2, 1]

# Inversione di numeri binari
binario = [1, 0, 1, 1]
binario_inv = binario[::-1]
print(binario_inv)  # [1, 1, 0, 1]
```

---

## Funzioni e Moduli: Revisione Approfondita

### Gestione degli Output Multipli

Come visto nella Lezione 1, le funzioni Python possono restituire più valori (tupla). Ci sono **3 modi** per gestirli:

```python
def circle_one(r):
    """Calcola circonferenza e area di un cerchio."""
    import math
    L = 2 * math.pi * r
    A = math.pi * r**2
    return L, A  # Restituisce una tupla
```

#### Metodo 1: Assegnazione Singola (tupla intera)
```python
risultato = circle_one(3)
print(risultato)       # (18.849..., 28.274...)
print(risultato[0])    # 18.849... (circonferenza)
print(risultato[1])    # 28.274... (area)
```

#### Metodo 2: Unpacking (decomposizione)
```python
L, A = circle_one(3)   # Decomposizione in variabili separate
print(f"Circonferenza: {L}")
print(f"Area: {A}")
```

#### Metodo 3: Scarto Selettivo con `_`
Usa il trattino basso `_` per **ignorare** valori non necessari:

```python
# Voglio solo l'area, non la circonferenza
_, area_solo = circle_one(3)
print(area_solo)  # 28.274...

# Voglio solo la circonferenza
circ_solo, _ = circle_one(3)
print(circ_solo)  # 18.849...
```

**Vantaggio:** Risparmio di memoria e codice più leggibile.

#### Metodo 4: Accesso Inline
Seleziona direttamente un elemento dalla tupla restituita:

```python
# Prendi solo il secondo valore (indice 1)
area = circle_one(3)[1]
print(area)  # 28.274...

# Prendi solo il primo valore (indice 0)
circonferenza = circle_one(3)[0]
print(circonferenza)  # 18.849...
```

### Importazione Moduli: Riepilogo Completo

```python
import math

# 1. Import standard
import math
print(math.pi)  # Richiede prefisso 'math.'

# 2. Import con alias (convenzione standard)
import math as mt
print(mt.pi)    # Richiede prefisso alias 'mt.'

# 3. Import selettivo di elementi specifici
from math import pi
print(pi)       # Nessun prefisso necessario

# 4. Import multiplo selettivo
from math import cos, sin, pi
print(sin(pi/2))  # 1.0
print(cos(0))     # 1.0

# 5. Import globale (❌ sconsigliato!)
from math import *  # Importa TUTTO - inquina il namespace
```

### Ispezione dei Moduli con `dir()`

La funzione `dir()` mostra **tutti gli oggetti** disponibili in un modulo:

```python
import math as mt

# Elenca tutto il contenuto del modulo
print(dir(mt))

# Output include:
# ['acos', 'asin', 'atan', 'ceil', 'cos', 'e', 'exp', 'factorial',
#  'floor', 'log', 'log10', 'pi', 'pow', 'sin', 'sqrt', 'tan', ...]

# Documentazione di funzioni specifiche
help(mt.sqrt)    # Mostra come usare sqrt()
help(mt.log)     # Mostra come usare log()
```

---
## Controllo del Flusso e Logica

### Logica Condizionale: `if-elif-else`

Le **dichiarazioni condizionali** permettono di eseguire codice diverso in base a condizioni logiche.

**Sintassi:**
```python
if condizione1:
    # Blocco eseguito se condizione1 è True
    istruzioni
elif condizione2:
    # Blocco eseguito se condizione1 è False e condizione2 è True
    istruzioni
else:
    # Blocco eseguito se tutte le condizioni precedenti sono False
    istruzioni
```

**⚠️ Attenzione all'Indentazione:**
L'indentazione (4 spazi o 1 tab) è **obbligatoria** in Python per definire i blocchi di codice. Non è solo una questione di stile - errori di indentazione causano `IndentationError`.

**Esempio pratico:**
```python
temperatura = 25

if temperatura > 30:
    print("Fa molto caldo!")
elif temperatura > 20:
    print("Temperatura piacevole")
elif temperatura > 10:
    print("Fa fresco")
else:
    print("Fa freddo!")
```

### Operatori di Confronto

| Operatore | Significato | Esempio |
|-----------|-------------|---------|
| `==` | Uguale a | `a == b` |
| `!=` | Diverso da | `a != b` |
| `<` | Minore di | `a < b` |
| `>` | Maggiore di | `a > b` |
| `<=` | Minore o uguale | `a <= b` |
| `>=` | Maggiore o uguale | `a >= b` |

**⚠️ Errore Comune:**
```python
# ❌ SBAGLIATO: = è per assegnazione
if unit = 'K':
    print("Kelvin")

# ✅ CORRETTO: == è per confronto
if unit == 'K':
    print("Kelvin")
```

### Cicli: `while`

Il ciclo **while** ripete un blocco di codice finché una condizione rimane `True`.

**Sintassi:**
```python
while condizione:
    # Blocco ripetuto finché condizione è True
    istruzioni
```

**Esempio: Countdown**
```python
count = 5
while count > 0:
    print(count)
    count -= 1  # Equivale a: count = count - 1
print("Lancio!")

# Output:
# 5
# 4
# 3
# 2
# 1
# Lancio!
```

**Ciclo infinito (usa con cautela!):**
```python
while True:
    risposta = input("Digita 'esci' per uscire: ")
    if risposta == 'esci':
        break  # Esce dal ciclo
    print(f"Hai scritto: {risposta}")
```

### Cicli: `for`

Il ciclo **for** itera su sequenze (liste, stringhe, range, ecc.).

**Sintassi:**
```python
for elemento in sequenza:
    # Blocco eseguito per ogni elemento
    istruzioni
```

**Esempi con `range()`:**
```python
# range(stop) - numeri da 0 a stop-1
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(start, stop) - numeri da start a stop-1
for i in range(2, 7):
    print(i)  # 2, 3, 4, 5, 6

# range(start, stop, step) - con passo personalizzato
for i in range(10, 20, 2):
    print(i)  # 10, 12, 14, 16, 18

# range con step negativo (countdown)
for i in range(5, 0, -1):
    print(i)  # 5, 4, 3, 2, 1
```

**Iterazione su stringhe:**
```python
parola = "PYTHON"
for lettera in parola:
    print(lettera)
# Output: P, Y, T, H, O, N (uno per riga)
```

**Iterazione su liste:**
```python
colori = ['rosso', 'verde', 'blu']
for colore in colori:
    print(f"Mi piace il {colore}")
```

**Funzione `len()` - lunghezza di sequenze:**
```python
stringa = "MCLXXI"
print(len(stringa))  # 6

lista = [1, 2, 3, 4, 5]
print(len(lista))    # 5
```

**Combinare `range()` e `len()` per accedere agli indici:**
```python
Roman = 'VII'
for i in range(len(Roman) - 1):  # len('VII') = 3, range(2) = [0, 1]
    c1 = Roman[i]      # Carattere corrente
    c2 = Roman[i + 1]  # Carattere successivo
    print(i, c1, c2)

# Output:
# 0 V I
# 1 I I
```

### Operatori Aritmetici Speciali

#### Divisione Intera: `//`
Restituisce solo la parte intera del quoziente:
```python
print(13 / 2)   # 6.5 (divisione normale)
print(13 // 2)  # 6 (solo parte intera)
print(7 // 2)   # 3
print(20 // 3)  # 6
```

#### Modulo: `%`
Restituisce il **resto** della divisione:
```python
print(13 % 2)   # 1 (13 = 6*2 + 1)
print(7 % 2)    # 1 (7 = 3*2 + 1)
print(20 % 3)   # 2 (20 = 6*3 + 2)
print(10 % 5)   # 0 (divisione esatta)
```

**Uso pratico:** Verificare se un numero è pari o dispari:
```python
numero = 17
if numero % 2 == 0:
    print(f"{numero} è pari")
else:
    print(f"{numero} è dispari")
```

#### Operatori di Incremento/Decremento

```python
x = 10

# Incremento
x += 5   # Equivale a: x = x + 5
print(x)  # 15

# Decremento
x -= 3   # Equivale a: x = x - 3
print(x)  # 12

# Moltiplicazione
x *= 2   # Equivale a: x = x * 2
print(x)  # 24

# Divisione
x /= 4   # Equivale a: x = x / 4
print(x)  # 6.0

# Divisione intera
x //= 2  # Equivale a: x = x // 2
print(x)  # 3.0

# Modulo
x %= 2   # Equivale a: x = x % 2
print(x)  # 1.0
```

### Il Valore Speciale: `None`

`None` è un valore speciale che rappresenta l'**assenza di un valore**. È utile per:

1. **Inizializzare variabili**
2. **Gestire errori**
3. **Indicare che una funzione non ha risultato valido**

```python
# Esempio: gestione errori
def dividi(a, b):
    if b == 0:
        print("Errore: divisione per zero!")
        return None  # Nessun valore valido
    else:
        return a / b

risultato = dividi(10, 0)
if risultato is None:
    print("Operazione non riuscita")
else:
    print(f"Risultato: {risultato}")
```

**Verificare se una variabile è `None`:**
```python
valore = None

# ✅ Modo corretto
if valore is None:
    print("Valore è None")

# ⚠️ Funziona ma meno idiomatico
if valore == None:
    print("Valore è None")
```

---

## Progetto 4: Conversione Temperature Avanzata

### Obiettivi di Apprendimento
- Manipolazione di stringhe con slicing
- Estrazione di sottostringhe
- Casting di tipi (string → float)
- Logica condizionale con `if-elif-else`
- Operatore di confronto `==`
- Uso di `None` per gestire errori

### Il Problema

Nella Lezione 1 abbiamo creato funzioni per convertire temperature, ma richiedevano input separati per valore e unità. Ora miglioriamo: **la funzione deve accettare un'unica stringa** come `'75K'` o `'32F'` ed estrarre automaticamente valore e unità.

**Requisiti:**
- Input: stringa nel formato `'valoreUNITA'` (es: `'273K'`, `'-40F'`)
- Conversioni supportate:
  - Kelvin → Fahrenheit: $°F = 1.8(K - 273.15) + 32$
  - Fahrenheit → Kelvin: $K = \frac{°F - 32}{1.8} + 273.15$
- Gestione errori: unità non riconosciute

### Implementazione Completa

```python
def convtemp(temp):
    """
    Converte temperatura da Kelvin a Fahrenheit e viceversa.
    
    INPUT:
        temp - stringa nel formato 'valoreK' o 'valoreF'
               Esempio: '273K', '32F', '-40F'
    
    OUTPUT:
        temperatura convertita (float)
        oppure None se l'unità non è riconosciuta
    """
    # Estrazione del valore numerico (tutto tranne l'ultimo carattere)
    temp_value = float(temp[0:-1])  # oppure: float(temp[:-1])
    
    # Estrazione dell'unità di misura (ultimo carattere)
    temp_type = temp[-1]
    
    # Debug (opzionale - commenta queste righe in produzione)
    # print(temp_value, type(temp_value))
    # print(temp_type, type(temp_type))
    
    # Logica condizionale per determinare la conversione
    if temp_type == 'K':
        # Kelvin → Fahrenheit
        out_temp = 1.8 * (temp_value - 273.15) + 32
    elif temp_type == 'F':
        # Fahrenheit → Kelvin
        out_temp = (temp_value - 32) / 1.8 + 273.15
    else:
        # Unità non riconosciuta
        print('Errore: unità non valida! Usa K o F.')
        out_temp = None
    
    return out_temp
```

### Come Funziona: Analisi Dettagliata

**Passo 1: Estrazione del Valore Numerico**
```python
temp = '273K'
temp_value = float(temp[0:-1])  # temp[0:-1] = '273'
# float('273') → 273.0
```

**Slicing breakdown:**
- `temp[0:-1]` significa "dal primo carattere (indice 0) all'ultimo escluso (indice -1)"
- Equivale a `temp[:-1]` (start omesso = 0)
- Per `'273K'`: estrae `'273'`
- Per `'-40F'`: estrae `'-40'` (il segno negativo è incluso!)

**Passo 2: Estrazione dell'Unità**
```python
temp_type = temp[-1]  # Ultimo carattere
# Per '273K' → 'K'
# Per '32F' → 'F'
```

**Passo 3: Decisione e Conversione**
```python
if temp_type == 'K':
    # Formula Kelvin → Fahrenheit
    out_temp = 1.8 * (temp_value - 273.15) + 32
elif temp_type == 'F':
    # Formula Fahrenheit → Kelvin
    out_temp = (temp_value - 32) / 1.8 + 273.15
else:
    # Gestione errore
    print('Errore!')
    out_temp = None
```

### Test della Funzione

```python
# Test 1: Kelvin → Fahrenheit
risultato1 = convtemp('273K')
print(risultato1)  # ~32.0 (punto di congelamento dell'acqua)

# Test 2: Fahrenheit → Kelvin
risultato2 = convtemp('32F')
print(risultato2)  # ~273.15

# Test 3: Temperatura negativa
risultato3 = convtemp('-40F')
print(risultato3)  # ~233.15 K

# Test 4: Verifica punto speciale (-40°F = -40°C = 233.15K)
risultato4 = convtemp('-40K')
print(risultato4)  # ~-459.67°F

# Test 5: Errore - unità non valida
risultato5 = convtemp('25C')  # 'C' non supportato
# Output: "Errore: unità non valida! Usa K o F."
print(risultato5)  # None

# Test 6: Uso della funzione help
help(convtemp)  # Mostra la docstring
```

### Definizione delle Stringhe: Virgolette Singole vs. Doppie

Python accetta sia virgolette **singole** `'...'` che **doppie** `"..."` per le stringhe:

```python
stringa1 = 'Ciao mondo'
stringa2 = "Ciao mondo"
# Sono equivalenti

# Vantaggio delle virgolette miste: include l'altro tipo
messaggio1 = "Python è 'fantastico'!"
messaggio2 = 'Il libro dice: "Python è potente"'

# Virgolette triple per stringhe multi-riga
testo_lungo = """
Questa è una stringa
che si estende
su più righe.
"""
```

### Esercizio 4: Estensione Multi-Formato

**Compito:** Estendi la funzione `convtemp()` per supportare anche i **gradi Celsius** (`'C'`). Aggiungi un secondo parametro che specifica il formato di output desiderato.

**Nuova firma della funzione:**
```python
def convtemp_extended(temp, output_unit):
    """
    Converte tra Kelvin, Fahrenheit e Celsius.
    
    INPUT:
        temp - stringa nel formato 'valoreUNITA' (K, F, o C)
        output_unit - stringa che specifica l'unità di output ('K', 'F', o 'C')
    
    OUTPUT:
        temperatura convertita nell'unità desiderata
    """
    # TODO: Implementa la funzione
    pass
```

**Formule di conversione Celsius:**
- Celsius → Kelvin: $K = °C + 273.15$
- Kelvin → Celsius: $°C = K - 273.15$
- Celsius → Fahrenheit: $°F = \frac{9}{5}°C + 32$
- Fahrenheit → Celsius: $°C = \frac{5}{9}(°F - 32)$

---
## Progetto 5: Calcolo dell'Epsilon della Macchina

### Obiettivi di Apprendimento
- Ciclo `while` con condizioni complesse
- Operatore logico `!=` (diverso da)
- Concetto di **precisione in virgola mobile**
- Importanza della rappresentazione numerica nei computer

### Il Contesto: Cosa è l'Epsilon della Macchina?

I computer rappresentano i numeri in virgola mobile con **precisione finita**. L'**epsilon della macchina** ($\epsilon$) è il più piccolo numero positivo tale che:

$$
1.0 + \epsilon \neq 1.0
$$

In altre parole, è il numero più piccolo che, sommato a 1, produce un risultato distinguibile da 1 stesso nella rappresentazione in virgola mobile del computer.

**Perché è importante?**
- Determina la precisione dei calcoli numerici
- Essenziale per algoritmi di ottimizzazione e analisi numerica
- Aiuta a capire gli errori di arrotondamento

### Algoritmo per il Calcolo

**Idea:**
1. Parti da `epsilon = 1.0`
2. Continua a dimezzare `epsilon` finché `1.0 + epsilon` è ancora diverso da `1.0`
3. Quando il ciclo termina, `epsilon` è troppo piccolo - quindi raddoppialo

**Implementazione:**

```python
def machine_epsilon():
    """
    Calcola l'epsilon della macchina.
    
    OUTPUT:
        epsilon - il più piccolo numero tale che 1.0 + epsilon != 1.0
    """
    e = 1.0
    
    # Continua a dimezzare finché 1+e è distinguibile da 1
    while (1 + e != 1):
        e = e / 2
    
    # Quando il ciclo termina, e è troppo piccolo, quindi raddoppialo
    return 2 * e

# Calcolo
eps = machine_epsilon()
print(f"Epsilon della macchina: {eps}")
print(f"Formato scientifico: {eps:e}")
```

**Output tipico (su sistemi a 64-bit):**
```
Epsilon della macchina: 2.220446049250313e-16
Formato scientifico: 2.220446e-16
```

### Analisi Dettagliata del Ciclo

```python
e = 1.0

# Iterazione 1: e = 1.0
# 1 + 1.0 = 2.0 ≠ 1.0 ✅ continua
# e = 1.0 / 2 = 0.5

# Iterazione 2: e = 0.5
# 1 + 0.5 = 1.5 ≠ 1.0 ✅ continua
# e = 0.5 / 2 = 0.25

# ... molte iterazioni ...

# Iterazione N: e ≈ 2.22e-16
# 1 + 2.22e-16 ≠ 1.0 ✅ continua
# e = 2.22e-16 / 2 ≈ 1.11e-16

# Iterazione N+1: e ≈ 1.11e-16
# 1 + 1.11e-16 == 1.0 ❌ STOP!
# (troppo piccolo per essere rappresentato)
```

**Perché raddoppiare alla fine?**
Quando il ciclo si ferma, `e` è già stato dimezzato una volta di troppo. L'ultimo valore valido era `2 * e`.

### Verifica con NumPy

NumPy fornisce l'epsilon della macchina come costante:

```python
import numpy as np

# Epsilon della macchina da NumPy
eps_numpy = np.finfo(float).eps
print(f"NumPy epsilon: {eps_numpy}")

# Confronto con il nostro calcolo
eps_nostro = machine_epsilon()
print(f"Nostro epsilon: {eps_nostro}")
print(f"Sono uguali? {eps_numpy == eps_nostro}")
```

**Altre informazioni utili con NumPy:**
```python
import numpy as np

info = np.finfo(float)
print(f"Epsilon: {info.eps}")
print(f"Numero massimo: {info.max}")
print(f"Numero minimo positivo: {info.tiny}")
print(f"Precisione (cifre decimali): {info.precision}")
```

### Implicazioni Pratiche

```python
# Attenzione ai confronti con numeri float!
a = 0.1 + 0.1 + 0.1
b = 0.3

print(a == b)  # False! ⚠️ (per problemi di arrotondamento)
print(a)       # 0.30000000000000004
print(b)       # 0.3

# Soluzione: confronto con tolleranza
tolerance = 1e-10
print(abs(a - b) < tolerance)  # True ✅

# Oppure usa math.isclose() (Python 3.5+)
import math
print(math.isclose(a, b))  # True ✅
```

---

## Progetto 6: Conversione Numero Romano → Decimale

### Obiettivi di Apprendimento
- Uso di **dizionari** per mappature chiave-valore
- Ciclo `for` con `range()`
- Funzione `len()` per ottenere la lunghezza
- Operatori `+=` e `-=`
- Operatore logico `<` (minore di)
- Algoritmo con logica condizionale complessa

### Il Sistema dei Numeri Romani

I numeri romani usano 7 simboli base:

| Simbolo | Valore |
|---------|--------|
| I | 1 |
| V | 5 |
| X | 10 |
| L | 50 |
| C | 100 |
| D | 500 |
| M | 1000 |

**Regole di composizione:**
1. **Addizione:** Simboli con valore decrescente o uguale si sommano
   - `VI` = 5 + 1 = 6
   - `XXX` = 10 + 10 + 10 = 30
   
2. **Sottrazione:** Un simbolo più piccolo prima di uno più grande si sottrae
   - `IV` = 5 - 1 = 4 (non IIII)
   - `IX` = 10 - 1 = 9
   - `XL` = 50 - 10 = 40
   - `CM` = 1000 - 100 = 900

**Esempi:**
- `VII` = 5 + 1 + 1 = 7
- `MCMXC` = 1000 + (1000-100) + (100-10) = 1990
- `MMXXV` = 1000 + 1000 + 10 + 10 + 5 = 2025

### Preparazione: Concetti Chiave

#### Dizionari per la Mappatura
```python
# Crea un dizionario che mappa simboli romani a valori decimali
symbols = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000
}

# Accesso ai valori
print(symbols['M'])  # 1000
print(symbols['V'])  # 5
```

#### Lunghezza di Stringhe
```python
Roman = 'MCLXXI'
print(len(Roman))  # 6 (numero di caratteri)
```

#### Funzione `range()` per Iterare su Indici
```python
# Converti range in lista per vedere i valori
print(list(range(10)))      # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(list(range(2, 10, 2))) # [2, 4, 6, 8] (start, stop, step)
```

#### Iterare su Coppie di Caratteri Consecutivi
```python
Roman = 'VII'
for i in range(len(Roman) - 1):  # range(2) = [0, 1]
    c1 = Roman[i]      # Carattere corrente
    c2 = Roman[i + 1]  # Carattere successivo
    print(i, c1, c2)

# Output:
# 0 V I
# 1 I I
```

### L'Algoritmo: Logica Dettagliata

**Strategia:**
1. Scorri tutti i caratteri **tranne l'ultimo**
2. Per ogni coppia di caratteri consecutivi (sinistra, destra):
   - Se `valore(sinistra) < valore(destra)` → **sottrai** (es: IV)
   - Altrimenti → **somma** (es: VI)
3. Alla fine, aggiungi il valore dell'ultimo carattere

**Perché funziona?**
- In `IV`, quando incontriamo `I` (valore 1) e vediamo che segue `V` (valore 5), sappiamo che `I` deve essere sottratto
- In `VI`, quando incontriamo `V` (valore 5) e vediamo che segue `I` (valore 1), sappiamo che `V` deve essere sommato

### Implementazione Completa

```python
def Roman2Decimal(Roman):
    """
    Converte un numero romano in formato decimale.
    
    INPUT:
        Roman - stringa contenente un numero romano (es: 'VII', 'MCMXC')
    
    OUTPUT:
        Decimal - valore decimale corrispondente (int)
    """
    # Dizionario di mappatura simboli → valori
    symbols = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    
    Decimal = 0
    
    # Itera su tutti i caratteri tranne l'ultimo
    for i in range(len(Roman) - 1):
        left = Roman[i]          # Simbolo corrente
        right = Roman[i + 1]     # Simbolo successivo
        
        # Confronta i valori dei due simboli
        if symbols[left] < symbols[right]:
            # Sottrazione (es: IV, IX, XL)
            Decimal -= symbols[left]
        else:
            # Addizione (es: VI, XX)
            Decimal += symbols[left]
    
    # Aggiungi sempre l'ultimo carattere (mai sottratto)
    Decimal += symbols[Roman[-1]]
    
    return Decimal
```

### Esempi di Esecuzione Passo-Passo

#### Esempio 1: `'VII'` (7)
```python
Roman = 'VII'
symbols = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
Decimal = 0

# Iterazione 1: i=0
left = 'V' (5), right = 'I' (1)
5 < 1? No → Somma
Decimal = 0 + 5 = 5

# Iterazione 2: i=1
left = 'I' (1), right = 'I' (1)
1 < 1? No → Somma
Decimal = 5 + 1 = 6

# Aggiungi ultimo carattere
Decimal = 6 + symbols['I'] = 6 + 1 = 7 ✅
```

#### Esempio 2: `'IX'` (9)
```python
Roman = 'IX'
Decimal = 0

# Iterazione 1: i=0
left = 'I' (1), right = 'X' (10)
1 < 10? Sì → Sottrai
Decimal = 0 - 1 = -1

# Aggiungi ultimo carattere
Decimal = -1 + symbols['X'] = -1 + 10 = 9 ✅
```

#### Esempio 3: `'MCMXC'` (1990)
```python
Roman = 'MCMXC'
Decimal = 0

# i=0: M,C → 1000 > 100 → +1000 = 1000
# i=1: C,M → 100 < 1000 → -100 = 900
# i=2: M,X → 1000 > 10 → +1000 = 1900
# i=3: X,C → 10 < 100 → -10 = 1890
# Ultimo: +C (+100) = 1990 ✅
```

### Test della Funzione

```python
# Test vari
print(Roman2Decimal('VII'))     # 7
print(Roman2Decimal('IV'))      # 4
print(Roman2Decimal('IX'))      # 9
print(Roman2Decimal('XLII'))    # 42
print(Roman2Decimal('MCMXC'))   # 1990
print(Roman2Decimal('MMXXV'))   # 2025
print(Roman2Decimal('MMMCMXCIX'))  # 3999 (massimo numero romano standard)
```

### Concetti Aggiuntivi: Operazioni su Liste

```python
# Concatenazione di liste con +
l1 = [1, 2, 3]
l2 = [4, 5, 6]
l3 = l1 + l2
print(l3)  # [1, 2, 3, 4, 5, 6]

# Verificare appartenenza con 'in'
print(4 in l1)  # False
print(2 in l1)  # True

# Estrarre chiavi da dizionario
g = {'pippo': 3, 'pluto': 7}
print(g.keys())    # dict_keys(['pippo', 'pluto'])
print(list(g.keys()))  # ['pippo', 'pluto']
```

---

## Progetto 7: Conversione Intero → Binario

### Obiettivi di Apprendimento
- Algoritmo di **divisione ripetuta**
- Ciclo `while` con condizione `!= 0`
- Operatori `%` (modulo) e `//` (divisione intera)
- Metodo `.append()` per aggiungere elementi a liste
- Inversione di liste con slicing `[::-1]`
- Manipolazione di stringhe con `.replace()`

### Rappresentazione Binaria (Base 2)

I computer memorizzano i numeri in **base 2** (binario) usando solo 0 e 1.

**Conversione da decimale a binario:**
- Dividi ripetutamente per 2
- Raccogli i resti
- Inverti l'ordine dei resti

**Esempio manuale: 13 → binario**
```
13 ÷ 2 = 6 resto 1  ↓
 6 ÷ 2 = 3 resto 0  ↓
 3 ÷ 2 = 1 resto 1  ↓
 1 ÷ 2 = 0 resto 1  ↓

Resti: [1, 0, 1, 1]
Invertiti: [1, 1, 0, 1] → 1101₂
```

**Verifica:** $1 \times 2^3 + 1 \times 2^2 + 0 \times 2^1 + 1 \times 2^0 = 8 + 4 + 0 + 1 = 13$ ✅

### Operatori Necessari

#### Operatore Modulo `%`
Restituisce il **resto** della divisione:
```python
print(7 % 2)   # 1
print(13 % 2)  # 1
print(8 % 2)   # 0 (divisione esatta)
```

#### Operatore Divisione Intera `//`
Restituisce il **quoziente** senza decimali:
```python
print(7 // 2)   # 3
print(13 // 2)  # 6
print(8 // 2)   # 4
```

**Confronto con `int()` e divisione normale:**
```python
print(int(7 / 2))  # 3 (divisione normale poi troncamento)
print(7 // 2)      # 3 (divisione intera diretta)
# Risultato uguale, ma // è più efficiente
```

### Implementazione dell'Algoritmo

```python
def int2bin(N):
    """
    Converte un intero in rappresentazione binaria.
    
    INPUT:
        N - intero positivo da convertire
    
    OUTPUT:
        b - lista di cifre binarie (es: [1, 1, 0, 1] per 13)
    """
    b = []  # Lista vuota per i resti
    
    # Continua finché N non diventa 0
    while N != 0:
        rem = N % 2      # Resto della divisione per 2 (0 o 1)
        N = N // 2       # Quoziente intero (per prossima iterazione)
        b.append(rem)    # Aggiungi resto alla lista
    
    # Inverti la lista per ottenere l'ordine corretto
    b = b[::-1]
    
    return b
```

### Analisi Passo-Passo: Conversione di 25

```python
N = 25
b = []

# Iterazione 1:
rem = 25 % 2 = 1
N = 25 // 2 = 12
b = [1]

# Iterazione 2:
rem = 12 % 2 = 0
N = 12 // 2 = 6
b = [1, 0]

# Iterazione 3:
rem = 6 % 2 = 0
N = 6 // 2 = 3
b = [1, 0, 0]

# Iterazione 4:
rem = 3 % 2 = 1
N = 3 // 2 = 1
b = [1, 0, 0, 1]

# Iterazione 5:
rem = 1 % 2 = 1
N = 1 // 2 = 0
b = [1, 0, 0, 1, 1]

# Ciclo termina (N = 0)
# Inverti: b = [1, 1, 0, 0, 1]
# Risultato: 25 = 11001₂ ✅
```

### Formattazione dell'Output

La lista `[1, 1, 0, 0, 1]` è corretta ma non molto leggibile. Possiamo formattarla meglio:

```python
# Applica la funzione
Nb = int2bin(25)
print(Nb)  # [1, 1, 0, 0, 1]

# Converti a stringa
Nb_str = str(Nb)
print(Nb_str)  # '[1, 1, 0, 0, 1]'

# Rimuovi parentesi quadre
Nb_str = Nb_str.replace('[', '').replace(']', '')
print(Nb_str)  # '1, 1, 0, 0, 1'

# Rimuovi virgole
Nb_str = Nb_str.replace(',', '')
print(Nb_str)  # '1 1 0 0 1'

# Rimuovi spazi
Nb_str = Nb_str.replace(' ', '')
print(Nb_str)  # '11001' ✅ Perfetto!
```

**Versione compatta (chain di replace):**
```python
Nb = int2bin(25)
Nb_str = str(Nb).replace('[', '').replace(']', '').replace(',', '').replace(' ', '')
print(Nb_str)  # '11001'
```

**Alternativa più elegante (join):**
```python
Nb = int2bin(25)
Nb_str = ''.join(str(digit) for digit in Nb)
print(Nb_str)  # '11001'
```

### Test Completo

```python
# Test vari numeri
print(int2bin(13))   # [1, 1, 0, 1] → 1101₂
print(int2bin(7))    # [1, 1, 1] → 111₂
print(int2bin(16))   # [1, 0, 0, 0, 0] → 10000₂
print(int2bin(255))  # [1, 1, 1, 1, 1, 1, 1, 1] → 11111111₂
print(int2bin(1))    # [1] → 1₂
print(int2bin(0))    # [] → caso speciale! (Vedere nota sotto)
```

**⚠️ Caso speciale: N = 0**
Il nostro algoritmo restituisce lista vuota `[]` per N=0. Possiamo gestirlo:

```python
def int2bin_fixed(N):
    """Versione migliorata che gestisce N=0."""
    if N == 0:
        return [0]
    
    b = []
    while N != 0:
        b.append(N % 2)
        N = N // 2
    return b[::-1]

print(int2bin_fixed(0))  # [0] ✅
```

### Operazioni su Liste: Append

```python
# Aggiungere elementi a una lista
resti = []
resti.append(1)
print(resti)  # [1]

resti.append(0)
print(resti)  # [1, 0]

resti.append(1)
print(resti)  # [1, 0, 1]
```

### Inversione di Sequenze

```python
# Inversione di stringa
s = 'numerical'
print(s[::-1])  # 'laciremun'

# Inversione di lista
a = [1, 0, 0, 1, 1, 0, 1]
b = a[::-1]
print(b)  # [1, 0, 1, 1, 0, 0, 1]

# L'originale rimane immutato
print(a)  # [1, 0, 0, 1, 1, 0, 1]
```

---

## Esercizio: Conversione Intero → Esadecimale

### Il Compito

Scrivi una funzione `int2hex(N)` che converte un intero in rappresentazione **esadecimale** (base 16).

**Sistema esadecimale:**
- Usa 16 simboli: 0-9 e A-F
- Valore simboli: 0=0, 1=1, ..., 9=9, A=10, B=11, C=12, D=13, E=14, F=15

**Algoritmo:** Simile al binario, ma dividi per 16 invece di 2.

**Esempio: 255 → esadecimale**
```
255 ÷ 16 = 15 resto 15 (F)  ↓
 15 ÷ 16 =  0 resto 15 (F)  ↓

Resti: [F, F]
Invertiti: [F, F] → FF₁₆
```

**Verifica:** $15 \times 16^1 + 15 \times 16^0 = 240 + 15 = 255$ ✅

### Soluzione Completa

```python
def int2hex(N):
    """
    Converte un intero in rappresentazione esadecimale.
    
    INPUT:
        N - intero positivo da convertire
    
    OUTPUT:
        h - stringa esadecimale (es: 'FF' per 255)
    """
    # Dizionario per mappare resti a simboli esadecimali
    symbs = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F'
    }
    
    h = ''  # Stringa vuota per risultato
    
    # Gestione caso speciale N=0
    if N == 0:
        return '0'
    
    # Divisione ripetuta per 16
    while N > 0:
        rem = N % 16           # Resto (0-15)
        h = symbs[rem] + h     # Prepend (aggiungi all'inizio)
        N = N // 16            # Quoziente intero
    
    return h
```

**Perché `symbs[rem] + h` invece di `h + symbs[rem]`?**
- Vogliamo costruire la stringa **dall'ultimo al primo** carattere
- `symbs[rem] + h` inserisce il nuovo carattere **all'inizio**
- Così non serve invertire alla fine

### Test della Funzione

```python
print(int2hex(255))   # 'FF'
print(int2hex(16))    # '10'
print(int2hex(255))   # 'FF'
print(int2hex(4095))  # 'FFF'
print(int2hex(10))    # 'A'
print(int2hex(15))    # 'F'
print(int2hex(0))     # '0'
```

### Confronto con Funzione Built-in Python

Python ha una funzione integrata `hex()`:

```python
print(hex(255))  # '0xff' (nota il prefisso '0x')
print(hex(16))   # '0x10'

# Per rimuovere il prefisso:
print(hex(255)[2:])  # 'ff'
print(hex(255)[2:].upper())  # 'FF'
```

---
---

## Riepilogo Concetti Chiave della Lezione 2

### Strutture Dati
- **Tuple `()`**: Immutabili, veloci, per dati fissi
- **Liste `[]`**: Mutabili, dinamiche, metodo `.append()`
- **Dizionari `{}`**: Coppie chiave-valore, accesso rapido
- **Stringhe**: Comportamento da sequenza, indicizzazione e slicing

### Controllo del Flusso
- **`if-elif-else`**: Decisioni con operatori `==`, `!=`, `<`, `>`
- **`while`**: Ripetizione basata su condizione
- **`for`**: Iterazione su sequenze con `range()`, `len()`

### Operatori Speciali
- **`//`**: Divisione intera (quoziente)
- **`%`**: Modulo (resto)
- **`+=`, `-=`**: Incremento/decremento compatto
- **`[::-1]`**: Inversione di sequenze

### Algoritmi Implementati
1. **Conversione temperature** (slicing, if-elif, None)
2. **Epsilon macchina** (while, precisione float)
3. **Romano → Decimale** (dizionari, for, logica condizionale)
4. **Intero → Binario** (divisione ripetuta, append, inversione)
5. **Intero → Esadecimale** (base 16, dizionario, prepend)

---

## 📚 Materiali di Riferimento Utilizzati

Questa lezione è stata ricostruita utilizzando i seguenti materiali:

1. **Python_Lecture2_v1_1.ipynb** (267 righe) - Notebook principale contenente:
   - Progetto 4: Conversione temperature avanzata (string input)
   - Progetto 5: Calcolo epsilon della macchina
   - Progetto 6: Conversione Romano → Decimale
   - Progetto 7: Conversione Intero → Binario
   - Esercizio: Conversione Intero → Esadecimale
   - Tutti gli esempi di codice e concetti su strutture dati

2. **Lez2-16sett.md** (file base - 150 righe) - Conteneva già un'ottima sintesi dei concetti

**Note:** Le slide PDF specifiche per il 16 settembre non esistono come file separato - i contenuti Python sono tutti nei notebook.

---

## 📅 Prossimi Appuntamenti

### Per la Prossima Lezione (Venerdì 19 Settembre - Laboratorio)
- [ ] **Completare gli esercizi proposti:**
  - ✅ Esercizio 4: Estendi `convtemp()` per supportare Celsius con output personalizzabile
  - ✅ Esercizio 5: Funzione per triangolo rettangolo (dalla Lezione 1)
  - ✅ Esercizio 6: Conversione Intero → Esadecimale (soluzione fornita)

- [ ] **Rivedere i concetti chiave:**
  - Differenze tra tuple, liste e dizionari
  - Slicing con start:stop:step
  - Logica con if-elif-else
  - Cicli while vs. for

- [ ] **Esercizi aggiuntivi suggeriti:**
  - Scrivi `Decimal2Roman(N)` (inverso del Progetto 6)
  - Implementa `bin2int(binary_str)` che converte stringa binaria in intero
  - Crea una calcolatrice con dizionario: `operations = {'+': add, '-': sub, ...}`

### Setup e Pratica
- [ ] Portare laptop per il laboratorio pratico di venerdì
- [ ] Sperimentare con i progetti su Google Colab o Anaconda
- [ ] Provare a "rompere" il codice per capire i messaggi di errore
- [ ] Leggere la documentazione Python su strutture dati built-in

### Cosa Aspettarsi Prossimamente
- **Venerdì 19/09:** Laboratorio pratico con esercizi guidati su tutto il Python base
- **Settimana prossima:** Introduzione a **NumPy** per calcolo scientifico
- **Poi:** Matplotlib per visualizzazione, Pandas per data science

---

## 💡 Suggerimenti per la Pratica

### 1. Debugging e Comprensione Errori
Quando incontri un errore, leggi il messaggio attentamente:
```python
# TypeError: 'tuple' object does not support item assignment
# Significa: stai cercando di modificare una tupla (immutabile)

# IndexError: string index out of range
# Significa: stai accedendo a un indice che non esiste

# KeyError: 'pippo'
# Significa: la chiave non esiste nel dizionario
```

### 2. Print Debugging
Usa `print()` liberamente per capire cosa succede:
```python
def debug_example(x):
    print(f"Ingresso: x = {x}")
    result = x * 2
    print(f"Dopo moltiplicazione: result = {result}")
    return result
```

### 3. Usa `help()` e `dir()`
```python
help(str.replace)  # Come usare replace()
help(list.append)  # Come usare append()
dir([])            # Tutti i metodi delle liste
```

### 4. Sperimenta nella REPL
Python ha una console interattiva perfetta per testare:
```python
>>> [1,2,3] + [4,5,6]  # Test rapido
[1, 2, 3, 4, 5, 6]
>>> 'ciao'[::-1]
'oaic'
```

### 5. Commenta il Codice
```python
# ❌ Commento inutile
x = x + 1  # incrementa x

# ✅ Commento utile
x += 1  # Passa al prossimo elemento della sequenza di Fibonacci
```

---

## 🎯 Checklist di Padronanza

Verifica di aver capito questi concetti chiave:

**Strutture Dati:**
- [ ] So quando usare tuple vs. liste
- [ ] Capisco come accedere a elementi con indici positivi/negativi
- [ ] So creare e usare dizionari per mappature
- [ ] Capisco perché le tuple sono immutabili

**Slicing:**
- [ ] So usare `[start:stop]` per estrarre sottostringhe/sottoliste
- [ ] Capisco che stop è escluso
- [ ] So usare `[::-1]` per invertire
- [ ] Capisco la sintassi completa `[start:stop:step]`

**Controllo Flusso:**
- [ ] So scrivere blocchi if-elif-else con corretta indentazione
- [ ] Capisco la differenza tra `=` (assegnazione) e `==` (confronto)
- [ ] So quando usare `while` vs. `for`
- [ ] Capisco come `range()` genera sequenze

**Algoritmi:**
- [ ] Capisco l'algoritmo di divisione ripetuta (binario/esadecimale)
- [ ] Capisco la logica romano→decimale (sottrazione vs. addizione)
- [ ] So cosa è l'epsilon della macchina e perché è importante
- [ ] So manipolare stringhe con slicing e replace()

---

## 🚀 Sfide Avanzate (Opzionali)

Per chi vuole approfondire:

### Sfida 1: Calcolatrice Romana
Scrivi funzioni per sommare e sottrarre numeri romani direttamente (senza convertirli in decimale).

### Sfida 2: Validatore Romano
Crea una funzione `is_valid_roman(s)` che verifica se una stringa è un numero romano valido secondo le regole storiche.

### Sfida 3: Conversione Intelligente
Scrivi `convert_any_base(N, from_base, to_base)` che converte numeri tra basi arbitrarie (2-36).

### Sfida 4: Performance
Misura il tempo di esecuzione di `int2bin()` per numeri grandi usando il modulo `time`. Puoi renderlo più veloce?

---

**Fine Lezione 2 - 16 Settembre 2025**

**Prossima lezione:** Venerdì 19 Settembre - Laboratorio pratico con NumPy