# Lezione 16 - 4 Novembre: Dualità nella Programmazione Lineare

---

## 📑 Indice

1. **[Introduzione alla Dualità](#1-introduzione-alla-dualità)** `00:02:09 - 00:07:38`
   - 1.1 [Setup: Il problema dei contadini (primale)](#11-setup-il-problema-dei-contadini-primale)
   - 1.2 [Formulazione matematica del problema primale](#12-formulazione-matematica-del-problema-primale)

2. **[Il Problema Duale: I Venditori di Pillole](#2-il-problema-duale-i-venditori-di-pillole)** `00:07:38 - 00:13:30`
   - 2.1 [Descrizione del problema duale](#21-descrizione-del-problema-duale)
   - 2.2 [Formulazione matematica del problema duale](#22-formulazione-matematica-del-problema-duale)
   - 2.3 [Relazione tra i due problemi](#23-relazione-tra-i-due-problemi)

3. **[Confronto e Proprietà Primale-Duale](#3-confronto-e-proprietà-primale-duale)** `00:13:30 - 00:23:12`
   - 3.1 [Intuizione sul confronto dei valori](#31-intuizione-sul-confronto-dei-valori)
   - 3.2 [Attività pratica di risoluzione](#32-attività-pratica-di-risoluzione)
   - 3.3 [Analisi dei risultati](#33-analisi-dei-risultati)

4. **[Risultati del Kahoot e Analisi](#4-risultati-del-kahoot-e-analisi)** `00:40:05 - 00:53:30`
   - 4.1 [Soluzione del problema primale](#41-soluzione-del-problema-primale)
   - 4.2 [Soluzione del problema duale](#42-soluzione-del-problema-duale)
   - 4.3 [Vincoli attivi e interpretazione](#43-vincoli-attivi-e-interpretazione)

5. **[Teorema Fondamentale della Dualità Debole](#5-teorema-fondamentale-della-dualità-debole)** `00:53:30 - 01:03:49`
   - 5.1 [Notazione matriciale](#51-notazione-matriciale)
   - 5.2 [Dimostrazione della dualità debole](#52-dimostrazione-della-dualità-debole)
   - 5.3 [Conseguenze: illimitatezza e vuoto](#53-conseguenze-illimitatezza-e-vuoto)

6. **[Secondo Esempio: Problema di Trasporto](#6-secondo-esempio-problema-di-trasporto)** `01:03:49 - 01:14:37`
   - 6.1 [Il problema primale: proprietari di fabbriche](#61-il-problema-primale-proprietari-di-fabbriche)
   - 6.2 [Il problema duale: azienda logistica](#62-il-problema-duale-azienda-logistica)
   - 6.3 [Confronto tra le formulazioni](#63-confronto-tra-le-formulazioni)

7. **[Notazione Matriciale del Problema di Trasporto](#7-notazione-matriciale-del-problema-di-trasporto)** `01:14:37 - 01:26:21`
   - 7.1 [Definizione delle matrici e vettori](#71-definizione-delle-matrici-e-vettori)
   - 7.2 [Forma compatta primale e duale](#72-forma-compatta-primale-e-duale)
   - 7.3 [Verifica della proprietà di dualità](#73-verifica-della-proprietà-di-dualità)

8. **[Coppie Standard Primale-Duale](#8-coppie-standard-primale-duale)** `01:26:21 - 01:33:14`
   - 8.1 [Prima coppia standard](#81-prima-coppia-standard)
   - 8.2 [Seconda coppia standard](#82-seconda-coppia-standard)
   - 8.3 [Trasformazione di problemi non standard](#83-trasformazione-di-problemi-non-standard)

---

## 1. Introduzione alla Dualità

### 1.1 Setup: Il problema dei contadini (primale)

`00:02:09`

🌾 **Attività di role-playing:**

La classe viene divisa in due gruppi per comprendere il concetto di dualità:
- **Metà 1:** Contadini (problema primale)
- **Metà 2:** Venditori di pillole (problema duale)

`00:02:41`

**Scenario per i contadini:**

Siete agricoltori e dovete nutrire i vostri polli.

Per nutrirli, avete due tipi di mangime disponibili:
- **Mangime A**
- **Mangime B**

`00:03:13`

**Costi:**
- Mangime A: **12 centesimi** per unità
- Mangime B: **16 centesimi** per unità

**Requisiti nutrizionali (dal dietologo):**

Ogni pollo deve consumare almeno:
- **14 unità** di carboidrati
- **20 unità** di proteine
- **9 unità** di vitamine

`00:03:48`

**Composizione nutrizionale dei mangimi:**

| Componente | Mangime A (per unità) | Mangime B (per unità) |
|-----------|----------------------|----------------------|
| **Carboidrati** | 2 | 2 |
| **Proteine** | 4 | 2 |
| **Vitamine** | 1 | 3 |

🎯 **Obiettivo:** Decidere come nutrire i polli utilizzando i mangimi A e B, **minimizzando la spesa**.

### 1.2 Formulazione matematica del problema primale

`00:04:23`

**Variabili decisionali:**

$$
\begin{aligned}
x_A &= \text{unità di mangime A per pollo} \geq 0 \\
x_B &= \text{unità di mangime B per pollo} \geq 0
\end{aligned}
$$

`00:05:18`

**Funzione obiettivo:**

`00:05:55`

$$
\min \; Z = 12 x_A + 16 x_B
$$

Minimizziamo il costo totale per pollo.

`00:06:30`

**Vincoli nutrizionali:**

Dobbiamo raggiungere **almeno** i requisiti minimi:

**Vincolo 1 - Carboidrati:**
$$
2x_A + 2x_B \geq 14
$$

**Vincolo 2 - Proteine:**
$$
4x_A + 2x_B \geq 20
$$

`00:07:07`

**Vincolo 3 - Vitamine:**
$$
1x_A + 3x_B \geq 9
$$

**Vincoli di segno:**
$$
x_A, x_B \geq 0
$$

`00:07:38`

✅ **Modello completo (PRIMALE):**

$$
\begin{aligned}
\min \quad & 12 x_A + 16 x_B \\
\text{s.t.} \quad & 2x_A + 2x_B \geq 14 \\
& 4x_A + 2x_B \geq 20 \\
& x_A + 3x_B \geq 9 \\
& x_A, x_B \geq 0
\end{aligned}
$$

---

## 2. Il Problema Duale: I Venditori di Pillole

### 2.1 Descrizione del problema duale

`00:07:38`

💊 **Scenario per i venditori:**

Non siete agricoltori. Siete **venditori di prodotti**, e i vostri prodotti sono **pillole**.

`00:08:09`

**Tipi di pillole:**

Avete un magazzino pieno di tre tipi di pillole:
- **Pillola C** (carboidrati): fornisce 1 unità di carboidrati
- **Pillola P** (proteine): fornisce 1 unità di proteine
- **Pillola V** (vitamine): fornisce 1 unità di vitamine

`00:08:46`

**Strategia di vendita:**

Andate dai vostri amici contadini e dite:

> "Perché state ancora mescolando i mangimi nel modo tradizionale? Comprate le nostre pillole! Risparmierete denaro e non sovralimenterete i polli. I polli saranno più sani!"

`00:09:19`

**Obiettivo del venditore:**

Volete vendere le pillole. L'obiettivo di un venditore è **massimizzare il profitto** o **massimizzare il ricavo**.

**Decisioni da prendere:**

`00:09:54`

Avete le pillole. Sapete quante pillole vendere per pollo (i numeri blu: 14, 20, 9).

❓ **Cosa manca?**

Il **prezzo**! Il prezzo non è deciso, è la vostra **variabile decisionale**.

Dovete fissare il prezzo per ogni tipo di pillola.

### 2.2 Formulazione matematica del problema duale

`00:09:54`

**Variabili decisionali:**

$$
\begin{aligned}
y_C &= \text{prezzo della pillola C (carboidrati)} \geq 0 \\
y_P &= \text{prezzo della pillola P (proteine)} \geq 0 \\
y_V &= \text{prezzo della pillola V (vitamine)} \geq 0
\end{aligned}
$$

`00:10:34`

**Funzione obiettivo:**

Vogliamo massimizzare il ricavo:

$$
\max \; W = 14 y_C + 20 y_P + 9 y_V
$$

**Ricavo = prezzo × quantità**

Le quantità sono i requisiti nutrizionali (numeri blu).

`00:11:17`

### 2.3 Relazione tra i due problemi

`00:11:50`

❓ **Potete fissare i prezzi come volete?**

Potete dire: "Le proteine costano 1 milione di euro, le vitamine 2 milioni"?

**Risposta:** No! Dovete essere **competitivi** rispetto ai mangimi.

`00:12:28`

**Vincoli di competitività:**

Per ogni mangime, il costo delle pillole equivalenti deve essere ≤ al costo del mangime.

**Vincolo 1 - Simulare mangime A:**

Con una unità di mangime A otteniamo: 2C + 4P + 1V

Il costo equivalente in pillole deve essere competitivo:

$$
2y_C + 4y_P + 1y_V \leq 12
$$

`00:12:59`

**Vincolo 2 - Simulare mangime B:**

Con una unità di mangime B otteniamo: 2C + 2P + 3V

$$
2y_C + 2y_P + 3y_V \leq 16
$$

`00:13:30`

✅ **Modello completo (DUALE):**

$$
\begin{aligned}
\max \quad & 14 y_C + 20 y_P + 9 y_V \\
\text{s.t.} \quad & 2y_C + 4y_P + 1y_V \leq 12 \\
& 2y_C + 2y_P + 3y_V \leq 16 \\
& y_C, y_P, y_V \geq 0
\end{aligned}
$$

---

## 3. Confronto e Proprietà Primale-Duale

### 3.1 Intuizione sul confronto dei valori

`00:13:30`

🤔 **I due problemi sono comparabili?**

È difficile dirlo direttamente. Facciamo questo ragionamento:

`00:14:01`

**Scenario 1 - Soluzione feasible primale:**

Supponiamo che **non abbiate seguito alcun corso di RO**. Trovate una soluzione ammissibile $\bar{x}$ per il problema dei contadini.

- $\bar{x}$ è ammissibile (valori non negativi)
- Nutrite i polli correttamente (≥ 14, 20, 9)
- Ottenete un certo costo $\bar{V}$

`00:15:03`

**Scenario 2 - Soluzione feasible duale:**

Anche voi (venditori) non avete seguito alcun corso. Trovate una soluzione ammissibile $\bar{y}$.

- I prezzi sono competitivi rispetto ai mangimi
- Ma forse troppo economici, non guadagnate molto
- Ottenete un ricavo $\bar{V}'$

`00:15:35`

**Confronto dei valori:**

❓ Se voglio confrontare $\bar{V}$ (costo) con $\bar{V}'$ (ricavo), cosa posso dire **con certezza**?

`00:16:18`

📊 **Scenari estremi:**

**Per i contadini:**
- Scenario peggiore: sovralimentate i polli fino all'obesità
- Polli molto grassi
- Spendete **moltissimo** denaro

`00:16:53`

**Per i venditori:**
- Scenario peggiore: offerta super competitiva
- Caso limite: regalate tutto (prezzi = 0)
- Ricavo = **0**

`00:17:31`

💡 **Intuizione:**

$$
\bar{V}' \leq \bar{V}
$$

Il ricavo del problema di massimizzazione è sempre ≤ al costo del problema di minimizzazione.

### 3.2 Attività pratica di risoluzione

`00:17:31`

📝 **Attività in classe:**

Nel WeBeep, trovate un notebook nella cartella **"Material for activities in class"**:
- File: `two_prob_incomplete.ipynb`

`00:19:23`

Il notebook contiene:
- I vettori dei requisiti (numeri blu: 14, 20, 9)
- Il vettore dei costi (rosso: 12, 16)
- La matrice dei contributi nutrizionali
- Struttura di base per entrambi i problemi

`00:20:00`

⚠️ **Nota tecnica:**

Il notebook è per la versione scaricata di Python, non per Colab. Dovete:
1. Copiare il preambolo (sezione import) dai vostri notebook di laboratorio
2. Incollarlo nel file scaricato
3. Altrimenti non funziona (nessun browser)

`00:20:37`

**Compito:**
1. Scrivete il modello per i contadini
2. Eseguitelo
3. Fate lo stesso per i venditori di pillole
4. Potete lavorare insieme, condividendo il codice
5. Poi faremo un recap con il codice e vedremo i risultati

### 3.3 Analisi dei risultati

`00:25:20`

**Notazione matriciale:**

Usiamo gli stessi dati per entrambi i problemi:

**Vettore costi c (riga):**
$$
\mathbf{c} = \begin{pmatrix} 12 & 16 \end{pmatrix}
$$

**Matrice contributi A:**
$$
\mathbf{A} = \begin{pmatrix}
2 & 2 \\
4 & 2 \\
1 & 3
\end{pmatrix}
$$

`00:25:54`

**Vettore requisiti b (colonna):**
$$
\mathbf{b} = \begin{pmatrix} 14 \\ 20 \\ 9 \end{pmatrix}
$$

**Problema primale (forma matriciale):**
$$
\begin{aligned}
\min \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A}\mathbf{x} \geq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{aligned}
$$

**Problema duale (forma matriciale):**
$$
\begin{aligned}
\max \quad & \mathbf{y}^T \mathbf{b} \\
\text{s.t.} \quad & \mathbf{y}^T \mathbf{A} \leq \mathbf{c}^T \\
& \mathbf{y} \geq 0
\end{aligned}
$$

---

## 4. Risultati del Kahoot e Analisi

### 4.1 Soluzione del problema primale

`00:40:05`

🎮 **Kahoot interattivo:**

Il Kahoot è diviso in due parti:
- Quando vedete l'immagine del **pollo**: rispondono i contadini
- Quando vedete l'immagine delle **pillole**: rispondono i venditori

`00:41:24`

**Domanda 1 - Valore ottimo del problema di minimizzazione:**

Gli studenti usano uno slider per indicare il valore intero.

✅ **Risposta:**

$$
Z^* = 88 \text{ centesimi per pollo}
$$

`00:41:56`

**Domanda 2 - Valori delle variabili:**

`00:46:08`

**Valore di $x_A$** (quantità mangime A):

$$
x_A^* \approx 6 \text{ (arrotondato)}
$$

`00:47:12`

**Valore di $x_B$** (quantità mangime B):

$$
x_B^* \approx 1 \text{ (arrotondato)}
$$

### 4.2 Soluzione del problema duale

`00:49:35`

**Domanda 3 - Valore ottimo del problema di massimizzazione:**

✅ **Risposta:**

$$
W^* = 88 \text{ centesimi}
$$

`00:42:33`

🔔 **Osservazione importante:**

I due valori ottimi sono **uguali**!
- Primale (minimizzazione): 88
- Duale (massimizzazione): 88

Non è magia, c'è una ragione matematica!

`00:50:18`

**Prezzi delle pillole:**

**Prezzo pillola C** (carboidrati):
$$
y_C^* \approx 5 \text{ centesimi}
$$

`00:51:08`

**Prezzo pillola P** (proteine):
$$
y_P^* = 0 \text{ centesimi}
$$

`00:51:41`

**Prezzo pillola V** (vitamine):
$$
y_V^* \approx 2 \text{ centesimi}
$$

**Verifica ricavo:**
$$
14(5) + 20(0) + 9(2) = 70 + 0 + 18 = 88 \; ✓
$$

### 4.3 Vincoli attivi e interpretazione

`00:47:42`

**Domanda 4 - Quali vincoli sono attivi (tight)?**

Un vincolo è **attivo** quando vale come **uguaglianza stretta**.

Sostituiamo $x_A^* = 6, x_B^* = 1$:

`00:48:21`

**Vincolo 1 (carboidrati):**
$$
2(6) + 2(1) = 12 + 2 = 14 \; ✓ \text{ ATTIVO}
$$

**Vincolo 2 (proteine):**
$$
4(6) + 2(1) = 24 + 2 = 26 > 20 \; ✗ \text{ NON ATTIVO}
$$

`00:48:59`

**Vincolo 3 (vitamine):**
$$
1(6) + 3(1) = 6 + 3 = 9 \; ✓ \text{ ATTIVO}
$$

✅ **Risposta:** Vincoli **1 e 3** sono attivi.

`00:52:19`

💡 **Interpretazione importante:**

**Perché $y_P^* = 0$?**

Guardate il vincolo delle proteine nel problema primale:

$$
4x_A + 2x_B \geq 20
$$

Con la soluzione ottima: $26 > 20$

Stiamo **sovralimentando** i polli di proteine!

`00:52:57`

Per i contadini, comprare pillole di proteine **non è un problema** perché già ne hanno troppe con i mangimi.

`00:53:30`

Se volete essere competitivi, dovete dare le pillole di proteine **gratis**.

Invece, carboidrati e vitamine sono vincoli **tight** (attivi), quindi hanno un prezzo positivo.

---

## 5. Teorema Fondamentale della Dualità Debole

### 5.1 Notazione matriciale

`00:54:27`

Scriviamo i due problemi in **forma matriciale** generale, in modo che le osservazioni valgano per qualsiasi tipo di problema.

**Problema PRIMALE:**

$$
\begin{aligned}
\min \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A}\mathbf{x} \geq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{aligned}
$$

`00:55:01`

**Problema DUALE:**

$$
\begin{aligned}
\max \quad & \mathbf{y}^T \mathbf{b} \\
\text{s.t.} \quad & \mathbf{y}^T \mathbf{A} \leq \mathbf{c}^T \\
& \mathbf{y} \geq 0
\end{aligned}
$$

### 5.2 Dimostrazione della dualità debole

`00:55:34`

📚 **Teorema della Dualità Debole:**

Sia $\bar{x}$ ammissibile per il primale e $\bar{y}$ ammissibile per il duale. Allora:

$$
\mathbf{c}^T \bar{x} \geq \mathbf{y}^T \mathbf{b}
$$

`00:56:05`

**Dimostrazione:**

Partiamo dalla ammissibilità di $\bar{x}$:

$$
\mathbf{A}\bar{x} \geq \mathbf{b}
$$

`00:56:42`

Moltiplichiamo a sinistra per $\bar{y}^T$ (che è non negativo):

$$
\bar{y}^T \mathbf{A} \bar{x} \geq \bar{y}^T \mathbf{b}
$$

Poiché $\bar{y} \geq 0$, il segno della disuguaglianza **non cambia**.

`00:57:17`

Ora ricordiamo che $\bar{y}$ è ammissibile per il duale:

$$
\bar{y}^T \mathbf{A} \leq \mathbf{c}^T
$$

Quindi:

$$
\mathbf{c}^T \bar{x} \geq \bar{y}^T \mathbf{A} \bar{x} \geq \bar{y}^T \mathbf{b}
$$

$$
\boxed{\mathbf{c}^T \bar{x} \geq \bar{y}^T \mathbf{b}}
$$

**Conclusione:**

Il valore di una soluzione ammissibile del problema di **minimizzazione** è sempre ≥ al valore di una soluzione ammissibile del problema di **massimizzazione**.

`00:57:54`

📊 **Rappresentazione grafica:**

```
            Primale (MIN)
            ↓
─────────── V̄ ───────────→ (vogliamo diminuire)
            88

─────────── V̄' ──────────→ (vogliamo aumentare)
            88
            ↑
            Duale (MAX)
```

`00:58:32`

✅ **Corollario (certificazione di ottimalità):**

Se $\mathbf{c}^T \bar{x} = \bar{y}^T \mathbf{b}$, allora $\bar{x}$ e $\bar{y}$ sono **ottimi** per i rispettivi problemi.

**Ragione:**

- $\bar{x}$ non può fare meglio di $\mathbf{c}^T \bar{x}$ perché qualsiasi soluzione ammissibile del primale ha valore ≥ $\bar{y}^T \mathbf{b}$
- $\bar{y}$ non può fare meglio di $\bar{y}^T \mathbf{b}$ perché qualsiasi soluzione ammissibile del duale ha valore ≤ $\mathbf{c}^T \bar{x}$

### 5.3 Conseguenze: illimitatezza e vuoto

`00:59:36`

📌 **Conseguenza 1:**

Se il problema **duale** (massimizzazione) è **illimitato**:

$$
\max \mathbf{y}^T \mathbf{b} = +\infty
$$

Cosa possiamo dire del primale?

`01:00:14`

**Esempio:** Con i numeri che vi ho dato, il ricavo diventa +∞. Diventate più ricchi di Elon Musk!

`01:00:44`

**Risposta:**

Il problema **primale** (minimizzazione) **non ha soluzione ammissibile** (è vuoto).

**Dimostrazione:**

`01:01:22`

Se il primale avesse una soluzione ammissibile $\bar{x}$, avrebbe un valore finito $\mathbf{c}^T \bar{x}$.

Ma per la dualità debole: $\mathbf{y}^T \mathbf{b} \leq \mathbf{c}^T \bar{x}$ per ogni $\bar{y}$ ammissibile.

Questo fermerebbe la crescita del duale, impedendo che vada a +∞. **Contraddizione!**

`01:02:45`

📌 **Conseguenza 2:**

Simmetricamente, se il problema **primale** (minimizzazione) è **illimitato**:

$$
\min \mathbf{c}^T \mathbf{x} = -\infty
$$

Allora il problema **duale** **non ha soluzione ammissibile** (è vuoto).

`01:03:17`

⚠️ **Attenzione:**

**Non possiamo dire il contrario!**

Non possiamo dire: "Se il mio problema è vuoto, allora l'altro è illimitato."

Potrebbero esserci casi in cui **entrambi** i problemi (primale e duale) sono **vuoti**.

Quindi l'implicazione vale solo in una direzione:
- ✅ Illimitato → Altro è vuoto
- ❌ Vuoto → Altro è illimitato (NON NECESSARIAMENTE)

---

## 6. Secondo Esempio: Problema di Trasporto

### 6.1 Il problema primale: proprietari di fabbriche

`01:03:49`

🏭 **Nuovo scenario di role-playing:**

Siete proprietari di **fabbriche** e **negozi**.

**Setup:**
- **n fabbriche**
- **m negozi**
- Producete **un tipo di bene**

`01:04:20`

**Capacità di produzione:**

Ogni fabbrica ha una produzione specifica:
$$
A_1, A_2, \ldots, A_n
$$

**Domanda dei negozi:**

Ogni negozio ha una richiesta specifica:
$$
D_1, D_2, \ldots, D_m
$$

`01:04:57`

**Assunzione di bilanciamento:**

Per semplicità, assumiamo che la produzione soddisfi esattamente la domanda:

$$
\sum_{i=1}^n A_i = \sum_{j=1}^m D_j
$$

**Costi di trasporto:**

Avete i vostri camion e un costo per trasportare una unità:

$$
b_{ij} = \text{costo unitario di trasporto da fabbrica } i \text{ a negozio } j
$$

`01:05:51`

**Formulazione del problema:**

È un tipo di problema di flusso. Dovete decidere la quantità di bene da trasportare da ogni fabbrica a ogni negozio.

**Variabili decisionali:**

$$
y_{ij} = \text{quantità di bene trasportato da fabbrica } i \text{ a negozio } j
$$

`01:06:45`

**Funzione obiettivo:**

Vogliamo **minimizzare** il costo totale di trasporto:

$$
\min \sum_{i=1}^n \sum_{j=1}^m b_{ij} y_{ij}
$$

`01:07:16`

**Vincoli - Svuotamento delle fabbriche:**

Per ogni fabbrica $i = 1, \ldots, n$:

$$
\sum_{j=1}^m y_{ij} = A_i
$$

Il flusso in uscita dalla fabbrica $i$ deve essere uguale alla sua produzione.

`01:07:47`

**Vincoli - Rifornimento dei negozi:**

Per ogni negozio $j = 1, \ldots, m$:

$$
\sum_{i=1}^n y_{ij} = D_j
$$

Il flusso in entrata al negozio $j$ deve essere uguale alla sua domanda.

`01:09:00`

**Vincoli di segno:**

$$
y_{ij} \geq 0 \quad \forall i, j
$$

✅ **Problema completo (PRIMALE di trasporto):**

$$
\begin{aligned}
\min \quad & \sum_{i=1}^n \sum_{j=1}^m b_{ij} y_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^m y_{ij} = A_i \quad \forall i = 1, \ldots, n \\
& \sum_{i=1}^n y_{ij} = D_j \quad \forall j = 1, \ldots, m \\
& y_{ij} \geq 0 \quad \forall i, j
\end{aligned}
$$

### 6.2 Il problema duale: azienda logistica

`01:09:33`

🚚 **Scenario duale:**

Avete un'**azienda logistica**. Avete già sistemi di trasporto tra qualsiasi coppia di siti.

Dite ai proprietari delle fabbriche:

> "Perché vi preoccupate ancora dei vostri camion che si rompono? Affidateci il trasporto!"

`01:10:03`

**Strategia:**

Comprate tutto dalle fonti (fabbriche) e vendete tutto alle destinazioni (negozi).

**Decisioni:**

Sapete che dovete trasportare quelle quantità esatte. Non potete lasciare nulla indietro.

Avete già il sistema, quindi **non avete costi** di trasporto.

`01:10:35`

**Obiettivo:**

Dovete avere una funzione obiettivo **opposta**:
- Loro: minimizzazione
- Voi: **massimizzazione**

**Variabili decisionali:**

Dovete decidere i **prezzi** per ogni unità:

$$
\begin{aligned}
\lambda_i &= \text{prezzo di acquisto per unità dalla fabbrica } i \\
\mu_j &= \text{prezzo di vendita per unità al negozio } j
\end{aligned}
$$

`01:11:57`

**Funzione obiettivo:**

Vogliamo massimizzare il **profitto**:

$$
\text{Profitto} = \text{Ricavo dalla vendita} - \text{Spesa per l'acquisto}
$$

$$
\max \sum_{j=1}^m \mu_j D_j - \sum_{i=1}^n \lambda_i A_i
$$

Le quantità $A_i$ e $D_j$ sono **fissate**.

`01:12:46`

**Vincoli di competitività:**

Per ogni coppia $(i, j)$, dobbiamo simulare l'effetto del trasporto di una unità da $i$ a $j$.

Compriamo una unità in $i$ (costo $\lambda_i$) e vendiamo in $j$ (ricavo $\mu_j$).

Il **costo netto** deve essere competitivo rispetto al costo diretto $b_{ij}$:

$$
\mu_j - \lambda_i \leq b_{ij} \quad \forall i, j
$$

`01:13:58`

**Vincoli di segno:**

`01:14:37`

⚠️ **Attenzione:** Non abbiamo specificato vincoli di segno espliciti per $\lambda$ e $\mu$.

Le variabili sono **illimitate nel segno** (unrestricted in sign):

$$
\lambda_i, \mu_j \in \mathbb{R} \quad \text{(possono essere negative)}
$$

**Ragione:**

Potremmo dover andare molto lontano per comprare qualcosa. Invece di pagare, vogliamo essere pagati perché siamo obbligati ad andare lì.

Lo stesso vale per i prezzi di vendita: possiamo permettere valori negativi.

### 6.3 Confronto tra le formulazioni

`01:14:37`

📊 **Osservazione delle regolarità:**

Guardiamo le figure (i coefficienti).

`01:15:11`

**Coefficiente $b_{ij}$:**

- **Primale:** Appare nella **funzione obiettivo**
- **Duale:** Appare nel **lato destro** (right-hand side)

Proprio come nel problema mangimi-pillole!

**Coefficienti $A_i$ e $D_j$:**

- **Primale:** Appaiono nel **lato destro** (right-hand side)
- **Duale:** Appaiono nella **funzione obiettivo**

`01:15:47`

**Elemento che crea confusione:**

C'è un **segno meno** nel duale:

$$
\max \sum \mu_j D_j - \sum \lambda_i A_i
$$

Questo segno meno non era presente nel problema mangimi-pillole.

`01:16:23`

**Spiegazione:**

Abbiamo un sistema di **uguaglianze** nei vincoli del primale.

Non cambia nulla se mettiamo un meno davanti:

$$
\sum_{j=1}^m y_{ij} = A_i \iff -\sum_{j=1}^m y_{ij} = -A_i
$$

`01:16:53`

**Matrice dei coefficienti:**

È più difficile vedere che stiamo usando la **stessa matrice** nei due problemi.

`01:17:29`

Nel duale, consideriamo una coppia $(i,j)$ che rappresenta la connessione $i \to j$, che corrisponde alla variabile $y_{ij}$.

Dove appare questa variabile nei vincoli del primale?

- Appare una volta nel vincolo con indice $i$ (fabbrica)
- Appare una volta nel vincolo con indice $j$ (negozio)

`01:18:01`

Nel primo insieme di vincoli (fabbriche), la variabile è moltiplicata per **+1**.

Nel secondo insieme (negozi), è moltiplicata per **+1**.

Ma se aggiungiamo il segno meno davanti al primo insieme, otteniamo:

$$
\mu_j - \lambda_i \leq b_{ij}
$$

Il coefficiente $-1$ moltiplica la variabile associata alla fabbrica ($\lambda_i$).

Il coefficiente $+1$ moltiplica la variabile associata al negozio ($\mu_j$).

`01:18:38`

Quindi anche questi due problemi sono **strettamente connessi**.

---

## 7. Notazione Matriciale del Problema di Trasporto

### 7.1 Definizione delle matrici e vettori

`01:18:38`

Rappresentiamo i due problemi in **forma matriciale**.

`01:19:12`

**Vettore c (riga, 1 × (n+m)):**

$$
\mathbf{c} = \begin{pmatrix} -A_1 & -A_2 & \cdots & -A_n & D_1 & D_2 & \cdots & D_m \end{pmatrix}
$$

Includiamo le produzioni (con segno meno) e le domande.

`01:19:49`

**Vettore b (colonna, nm × 1):**

$$
\mathbf{b} = \begin{pmatrix}
b_{11} \\
b_{12} \\
\vdots \\
b_{1m} \\
\vdots \\
b_{nm}
\end{pmatrix}
$$

Tutti i costi di trasporto in una lunga colonna.

`01:20:20`

**Matrice A (nm × (n+m)):**

È la matrice di **incidenza nodo-arco** (trasposta).

Ogni riga corrisponde a un arco $(i,j)$.

`01:21:19`

Per l'arco $(1,1)$:
- Colonna 1 (fabbrica 1): **-1**
- Colonna $n+1$ (negozio 1): **+1**
- Tutto il resto: 0

```
        Fabbriche (1...n) | Negozi (1...m)
Arco 11:   -1  0 ... 0   |   +1  0 ... 0
Arco 12:   -1  0 ... 0   |    0 +1 ... 0
  ⋮         ⋮             |     ⋮
Arco ij:    0 ... -1 ... |   ... +1 ...
  ⋮         ⋮             |     ⋮
Arco nm:    0 ...  0  -1 |    0 ...  0 +1
```

In generale, per l'arco $(i,j)$:
- Posizione $i$: **-1**
- Posizione $n+j$: **+1**

`01:21:57`

La matrice ha:
- **nm righe** (una per ogni arco)
- **n+m colonne** (una per ogni nodo)

`01:22:39`

**Vettore variabili x (colonna, (n+m) × 1):**

$$
\mathbf{x} = \begin{pmatrix}
\lambda_1 \\
\lambda_2 \\
\vdots \\
\lambda_n \\
\mu_1 \\
\mu_2 \\
\vdots \\
\mu_m
\end{pmatrix}
$$

Tutti i prezzi (acquisto e vendita) in una colonna.

**Vettore variabili y (riga, 1 × nm):**

$$
\mathbf{y} = \begin{pmatrix}
y_{11} & y_{12} & \cdots & y_{1m} & \cdots & y_{nm}
\end{pmatrix}
$$

Tutte le quantità trasportate in una riga.

### 7.2 Forma compatta primale e duale

`01:23:41`

**Problema di MINIMIZZAZIONE (primale):**

$$
\begin{aligned}
\min \quad & \mathbf{y} \mathbf{b} \\
\text{s.t.} \quad & \mathbf{y} \mathbf{A} = \mathbf{c} \\
& \mathbf{y} \geq 0
\end{aligned}
$$

Nota: $\mathbf{y}\mathbf{A}$ produce un vettore riga, $\mathbf{c}$ è un vettore riga.

`01:24:12`

**Problema di MASSIMIZZAZIONE (duale):**

$$
\begin{aligned}
\max \quad & \mathbf{c} \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A}^T \mathbf{x} \leq \mathbf{b} \\
& \text{(nessun vincolo di segno su } \mathbf{x}\text{)}
\end{aligned}
$$

Possiamo chiamare questo **primale** e l'altro **duale**, o viceversa.

I ruoli possono essere scambiati senza problemi.

### 7.3 Verifica della proprietà di dualità

`01:24:46`

📚 **Proprietà di dualità debole:**

Se abbiamo:
- $\bar{y}$ soluzione ammissibile per il primale (minimizzazione)
- $\bar{x}$ soluzione ammissibile per il duale (massimizzazione)

Allora:

$$
\mathbf{c} \bar{x} \leq \bar{y} \mathbf{b}
$$

`01:25:19`

**Dimostrazione:**

Partiamo dall'ammissibilità di $\bar{x}$:

$$
\mathbf{A}^T \bar{x} \leq \mathbf{b}
$$

Moltiplichiamo a sinistra per $\bar{y}$ (non negativo):

$$
\bar{y} \mathbf{A}^T \bar{x} \leq \bar{y} \mathbf{b}
$$

`01:25:51`

Ora ricordiamo che $\bar{y}$ è ammissibile:

$$
\bar{y} \mathbf{A} = \mathbf{c}
$$

Quindi: $\bar{y} \mathbf{A}^T$ è il trasposto di $\mathbf{c}$, ma operativamente abbiamo:

$$
\mathbf{c} \bar{x} \leq \bar{y} \mathbf{b}
$$

È ancora più semplice rispetto alla dimostrazione precedente!

`01:26:21`

**Conseguenze:**

- Se uno dei due problemi è illimitato, l'altro deve essere vuoto
- Se l'uguaglianza vale, possiamo certificare l'ottimalità

---

## 8. Coppie Standard Primale-Duale

### 8.1 Prima coppia standard

`01:26:21`

Dall'esempio del trasporto, abbiamo la coppia:

**PRIMALE (min con uguaglianze):**

$$
\begin{aligned}
\min \quad & \mathbf{y} \mathbf{b} \\
\text{s.t.} \quad & \mathbf{y} \mathbf{A} = \mathbf{c} \\
& \mathbf{y} \geq 0
\end{aligned}
$$

`01:27:23`

**DUALE (max con disuguaglianze ≤):**

$$
\begin{aligned}
\max \quad & \mathbf{c} \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A}^T \mathbf{x} \leq \mathbf{b} \\
& \mathbf{x} \text{ illimitato in segno}
\end{aligned}
$$

### 8.2 Seconda coppia standard

`01:27:54`

Dall'esempio mangimi-pillole, abbiamo la coppia:

**PRIMALE (min con ≥):**

$$
\begin{aligned}
\min \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A}\mathbf{x} \geq \mathbf{b} \\
& \mathbf{x} \geq 0
\end{aligned}
$$

**DUALE (max con ≤):**

$$
\begin{aligned}
\max \quad & \mathbf{y}^T \mathbf{b} \\
\text{s.t.} \quad & \mathbf{y}^T \mathbf{A} \leq \mathbf{c}^T \\
& \mathbf{y} \geq 0
\end{aligned}
$$

`01:28:25`

📌 **Memorizzazione:**

Se vi do un problema in uno di questi **quattro formati**, potete scrivere immediatamente il duale.

Non importa il significato delle variabili, dei vincoli, della competitività, ecc. Non ci interessa.

Scriviamo il duale usando queste forme standard.

### 8.3 Trasformazione di problemi non standard

`01:28:55`

❓ **Problema:** Cosa fare quando vi do un problema che **non rientra** in nessuno di questi quattro formati?

**Esempio di problema misto:**

$$
\begin{aligned}
\min \quad & 4x_1 + 2x_2 \\
\text{s.t.} \quad & \text{alcuni vincoli } \leq \\
& \text{alcuni vincoli } \geq \\
& \text{alcune variabili } x_i \geq 0 \\
& \text{alcune variabili illimitate in segno} \\
& \text{alcune variabili } x_i \leq 0
\end{aligned}
$$

`01:29:27`

Come si scrive il duale di questo?

`01:30:01`

**Opzione 1: Trasformazione in forma standard**

Trasformiamo il problema in un problema equivalente che rientra in una delle coppie standard.

`01:30:34`

**Passo 1 - Uniformare i vincoli:**

Vogliamo che tutti i vincoli siano del tipo ≥ (per usare la seconda coppia standard).

Se abbiamo un vincolo:

$$
a_i^T x \leq b_i
$$

Moltiplichiamo per $-1$:

$$
-a_i^T x \geq -b_i
$$

`01:31:37`

**Passo 2 - Uniformare le variabili:**

Tutte le variabili devono essere non negative.

**Se $x_j$ è illimitata in segno:**

Introduciamo due variabili non negative $x_j^+, x_j^- \geq 0$ e scriviamo:

$$
x_j = x_j^+ - x_j^-
$$

Sostituiamo ovunque $x_j$ con $x_j^+ - x_j^-$.

`01:32:08`

**Se $x_j \leq 0$:**

Introduciamo una variabile equivalente:

$$
x_j' = -x_j \geq 0
$$

Sostituiamo $x_j$ con $-x_j'$ ovunque.

`01:32:41`

**Dopo la trasformazione:**

Abbiamo un problema che rientra in una delle forme standard. A quel punto possiamo scrivere il duale immediatamente.

`01:33:14`

**Opzione 2: Usare una tabella**

Domani vedremo come usare una **tabella** per scrivere direttamente il duale senza fare trasformazioni.

---

## 📊 Tabelle Riassuntive

### Confronto Primale-Duale (Problema Mangimi-Pillole)

| Elemento | Primale (Contadini) | Duale (Venditori) |
|----------|---------------------|-------------------|
| **Obiettivo** | MIN costo | MAX ricavo |
| **Variabili** | $x_A, x_B$ (quantità mangimi) | $y_C, y_P, y_V$ (prezzi pillole) |
| **Numero variabili** | 2 | 3 |
| **Vincoli** | 3 (requisiti nutrizionali ≥) | 2 (competitività ≤) |
| **Coefficienti obiettivo** | 12, 16 (costi mangimi) | 14, 20, 9 (quantità richieste) |
| **Termini noti** | 14, 20, 9 (requisiti) | 12, 16 (costi mangimi) |
| **Matrice coefficienti** | $A$ (3×2) | $A^T$ (2×3) |

**Osservazione chiave:** Ciò che è nell'obiettivo nel primale è nel lato destro nel duale, e viceversa.

### Soluzione Ottima (Problema Mangimi-Pillole)

| Problema | Valore Ottimo | Variabili |
|----------|---------------|-----------|
| **Primale** | $Z^* = 88$ cent | $x_A^* \approx 6$, $x_B^* \approx 1$ |
| **Duale** | $W^* = 88$ cent | $y_C^* \approx 5$, $y_P^* = 0$, $y_V^* \approx 2$ |

**Vincoli attivi primale:** 1 (carboidrati) e 3 (vitamine)

**Vincolo non attivo:** 2 (proteine) → sovralimentazione → $y_P^* = 0$

### Teorema della Dualità Debole

**Enunciato:**

Sia $\bar{x}$ ammissibile per il primale e $\bar{y}$ ammissibile per il duale. Allora:

$$
\mathbf{c}^T \bar{x} \geq \bar{y}^T \mathbf{b}
$$

**Corollari:**

| Situazione | Conseguenza |
|-----------|-------------|
| $\mathbf{c}^T \bar{x} = \bar{y}^T \mathbf{b}$ | $\bar{x}$ e $\bar{y}$ sono **ottimi** |
| Duale illimitato ($+\infty$) | Primale **vuoto** (no soluz. ammissibili) |
| Primale illimitato ($-\infty$) | Duale **vuoto** (no soluz. ammissibili) |

⚠️ **Attenzione:** Non vale il contrario! Se un problema è vuoto, l'altro **non è necessariamente** illimitato (potrebbero essere entrambi vuoti).

### Coppie Standard Primale-Duale

**Coppia 1 (da problema trasporto):**

| Primale | Duale |
|---------|-------|
| $\min \mathbf{y}\mathbf{b}$ | $\max \mathbf{c}\mathbf{x}$ |
| $\mathbf{y}\mathbf{A} = \mathbf{c}$ | $\mathbf{A}^T\mathbf{x} \leq \mathbf{b}$ |
| $\mathbf{y} \geq 0$ | $\mathbf{x}$ illimitato in segno |

**Coppia 2 (da problema mangimi-pillole):**

| Primale | Duale |
|---------|-------|
| $\min \mathbf{c}^T\mathbf{x}$ | $\max \mathbf{y}^T\mathbf{b}$ |
| $\mathbf{A}\mathbf{x} \geq \mathbf{b}$ | $\mathbf{y}^T\mathbf{A} \leq \mathbf{c}^T$ |
| $\mathbf{x} \geq 0$ | $\mathbf{y} \geq 0$ |

### Trasformazioni per Forma Standard

Per trasformare un problema non standard in forma standard:

| Elemento non standard | Trasformazione | Forma standard |
|----------------------|----------------|----------------|
| Vincolo $a^Tx \leq b$ (quando vogliamo ≥) | Moltiplica per -1 | $-a^Tx \geq -b$ |
| Variabile $x_j$ illimitata | $x_j = x_j^+ - x_j^-$ | $x_j^+, x_j^- \geq 0$ |
| Variabile $x_j \leq 0$ | $x_j' = -x_j$ | $x_j' \geq 0$ |
| Obiettivo MAX (quando vogliamo MIN) | $\max f = -\min(-f)$ | $\min (-f)$ |

### Problema di Trasporto

**Dati:**
- $n$ fabbriche con produzione $A_1, \ldots, A_n$
- $m$ negozi con domanda $D_1, \ldots, D_m$
- Costo unitario trasporto: $b_{ij}$
- Bilanciamento: $\sum A_i = \sum D_j$

**Primale (proprietari):**

$$
\begin{aligned}
\min \quad & \sum_{i,j} b_{ij} y_{ij} \\
\text{s.t.} \quad & \sum_j y_{ij} = A_i \quad \forall i \\
& \sum_i y_{ij} = D_j \quad \forall j \\
& y_{ij} \geq 0
\end{aligned}
$$

**Duale (azienda logistica):**

$$
\begin{aligned}
\max \quad & \sum_j \mu_j D_j - \sum_i \lambda_i A_i \\
\text{s.t.} \quad & \mu_j - \lambda_i \leq b_{ij} \quad \forall i,j \\
& \lambda_i, \mu_j \text{ illimitati}
\end{aligned}
$$

**Interpretazione:**
- $\lambda_i$: prezzo acquisto unità da fabbrica $i$
- $\mu_j$: prezzo vendita unità a negozio $j$
- Vincolo: profitto per unità $(i \to j)$ ≤ costo diretto trasporto

