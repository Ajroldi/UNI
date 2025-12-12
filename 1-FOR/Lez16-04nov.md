# Lezione 16 - Dualità nella Programmazione Lineare

## Introduzione: Problema dei Contadini (Primale)

**Scenario**: Agricoltori devono nutrire polli con due mangimi (A e B).

**Dati**:
- Costi: Mangime A = 12¢, Mangime B = 16¢
- Requisiti minimi per pollo: 14 carboidrati, 20 proteine, 9 vitamine
- Composizione mangimi (per unità):

| Nutriente | Mangime A | Mangime B |
|-----------|-----------|-----------|
| Carboidrati | 2 | 2 |
| Proteine | 4 | 2 |
| Vitamine | 1 | 3 |

**Formulazione**:
```
min  12x_A + 16x_B
s.t. 2x_A + 2x_B ≥ 14  (carboidrati)
     4x_A + 2x_B ≥ 20  (proteine)
     x_A + 3x_B ≥ 9    (vitamine)
     x_A, x_B ≥ 0
```

## Problema Duale: Venditori di Pillole

**Scenario**: Venditori offrono pillole nutrizionali (C, P, V) che forniscono 1 unità di carboidrati/proteine/vitamine.

**Strategia**: Convincere i contadini a comprare pillole invece di mangimi.

**Variabili**: $y_C, y_P, y_V$ = prezzi delle pillole (da decidere)

**Obiettivo**: Massimizzare ricavo vendendo le quantità necessarie (14, 20, 9)

**Vincoli di competitività**: Il costo delle pillole equivalenti a 1 unità di mangime deve essere ≤ al costo del mangime.

**Formulazione**:
```
max  14y_C + 20y_P + 9y_V
s.t. 2y_C + 4y_P + y_V ≤ 12   (competitività vs A)
     2y_C + 2y_P + 3y_V ≤ 16  (competitività vs B)
     y_C, y_P, y_V ≥ 0
```

## Relazione Primale-Duale

**Osservazione chiave**: Stessa matrice A in entrambi i problemi
- Coefficienti obiettivo primale → termini noti duale
- Termini noti primale → coefficienti obiettivo duale
- Matrice A nel primale → A^T nel duale

**Forma matriciale**:
- Primale: $\min \mathbf{c}^T\mathbf{x}$ con $\mathbf{Ax} \geq \mathbf{b}$, $\mathbf{x} \geq 0$
- Duale: $\max \mathbf{y}^T\mathbf{b}$ con $\mathbf{y}^T\mathbf{A} \leq \mathbf{c}^T$, $\mathbf{y} \geq 0$

## Soluzione Ottima

**Primale**: $Z^* = 88¢$, con $x_A^* \approx 6$, $x_B^* \approx 1$

**Duale**: $W^* = 88¢$, con $y_C^* \approx 5$, $y_P^* = 0$, $y_V^* \approx 2$

**Vincoli attivi primale**: 
- Carboidrati: $2(6) + 2(1) = 14$ ✓ ATTIVO
- Proteine: $4(6) + 2(1) = 26 > 20$ ✗ NON ATTIVO
- Vitamine: $6 + 3(1) = 9$ ✓ ATTIVO

**Interpretazione**: Prezzo proteine = 0 perché c'è sovralimentazione (vincolo non attivo)

## Teorema della Dualità Debole

**Enunciato**: Per ogni $\bar{x}$ ammissibile (primale) e $\bar{y}$ ammissibile (duale):
$$\mathbf{c}^T\bar{x} \geq \bar{y}^T\mathbf{b}$$

**Dimostrazione**:
1. Da $\mathbf{A}\bar{x} \geq \mathbf{b}$, moltiplicando per $\bar{y}^T \geq 0$: $\bar{y}^T\mathbf{A}\bar{x} \geq \bar{y}^T\mathbf{b}$
2. Da $\bar{y}^T\mathbf{A} \leq \mathbf{c}^T$: $\mathbf{c}^T\bar{x} \geq \bar{y}^T\mathbf{A}\bar{x}$
3. Combinando: $\mathbf{c}^T\bar{x} \geq \bar{y}^T\mathbf{b}$ ✓

**Corollario (certificato ottimalità)**: Se $\mathbf{c}^T\bar{x} = \bar{y}^T\mathbf{b}$, allora $\bar{x}$ e $\bar{y}$ sono ottimi.

**Conseguenze**:
- Duale illimitato (+∞) → Primale vuoto (nessuna soluzione ammissibile)
- Primale illimitato (-∞) → Duale vuoto
- ⚠️ NON vale il contrario: vuoto ≠ implica illimitato

## Problema di Trasporto

**Primale (proprietari fabbriche/negozi)**:
- n fabbriche con produzione $A_1, \ldots, A_n$
- m negozi con domanda $D_1, \ldots, D_m$
- Costo trasporto: $b_{ij}$ per unità da i a j
- Variabili: $y_{ij}$ = quantità trasportata

```
min  Σᵢⱼ bᵢⱼ yᵢⱼ
s.t. Σⱼ yᵢⱼ = Aᵢ  ∀i  (svuotamento fabbriche)
     Σᵢ yᵢⱼ = Dⱼ  ∀j  (rifornimento negozi)
     yᵢⱼ ≥ 0
```

**Duale (azienda logistica)**:
- Compra dalle fabbriche a prezzo $\lambda_i$
- Vende ai negozi a prezzo $\mu_j$
- Variabili: $\lambda_i, \mu_j$ (illimitate in segno)

```
max  Σⱼ μⱼDⱼ - Σᵢ λᵢAᵢ
s.t. μⱼ - λᵢ ≤ bᵢⱼ  ∀i,j  (competitività)
     λᵢ, μⱼ ∈ ℝ (illimitate)
```

**Interpretazione**: Il profitto per trasportare da i a j deve essere ≤ al costo diretto.

## Coppie Standard Primale-Duale

**Coppia 1 (da trasporto)**:
```
Primale:                    Duale:
min  y·b                   max  c·x
s.t. y·A = c               s.t. Aᵀx ≤ b
     y ≥ 0                      x illimitato
```

**Coppia 2 (da mangimi-pillole)**:
```
Primale:                    Duale:
min  cᵀx                   max  yᵀb
s.t. Ax ≥ b                s.t. yᵀA ≤ cᵀ
     x ≥ 0                      y ≥ 0
```

## Trasformazione in Forma Standard

Per problemi non standard, trasformare così:

**Uniformare vincoli**:
- Se $a^Tx \leq b$ (e vogliamo ≥): moltiplica per -1 → $-a^Tx \geq -b$

**Uniformare variabili**:
- Se $x_j$ illimitata: sostituisci con $x_j = x_j^+ - x_j^-$ dove $x_j^+, x_j^- \geq 0$
- Se $x_j \leq 0$: sostituisci con $x_j' = -x_j \geq 0$

**Procedura**:
1. Trasforma in forma standard
2. Scrivi il duale usando le coppie standard
3. (Alternativa: usa tabella, vista nella lezione successiva)

## Notazione Matriciale Trasporto

**Vettore c** (1×(n+m)): $[-A_1, -A_2, \ldots, -A_n, D_1, D_2, \ldots, D_m]$

**Vettore b** (nm×1): Tutti i $b_{ij}$ in colonna

**Matrice A** (nm×(n+m)): Matrice incidenza nodo-arco
- Riga per arco (i,j): -1 in posizione i, +1 in posizione n+j, 0 altrove

**Forma compatta**:
- Primale: $\min \mathbf{y}\mathbf{b}$ con $\mathbf{y}\mathbf{A} = \mathbf{c}$, $\mathbf{y} \geq 0$
- Duale: $\max \mathbf{c}\mathbf{x}$ con $\mathbf{A}^T\mathbf{x} \leq \mathbf{b}$, x illimitato