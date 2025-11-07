# Lezione 9 - 3 Ottobre: Lab 1 NAML - SVD e Applicazioni

**Corso**: Numerical Analysis for Machine Learning  
**Docenti**: Matteo Caldano (TA), Elena Basile (Tutor)  
**Data**: 3 Ottobre 2024  
**Durata**: 13:30 - 16:00  
**Materiale**: Google Colab + Notebook su WeBeep

---

## 📑 Indice Completo

### **Parte 1: Fondamenti**
1. **[Introduzione e Organizzazione](#1-introduzione-e-organizzazione)** `00:00:02 - 00:02:31`
   - 1.1 [Contatti e struttura del corso](#11-contatti-e-struttura)
   - 1.2 [Ambiente di lavoro: Google Colab](#12-ambiente-google-colab)
   
2. **[Recap Teorico: SVD e Compressione](#2-recap-teorico-svd)** `00:02:31 - 00:09:29`
   - 2.1 [Decomposizione SVD: A = UΣV^T](#21-decomposizione-svd)
   - 2.2 [Approssimazione rango-k e Teorema Eckart-Young](#22-approssimazione-rango-k)
   - 2.3 [Complessità spaziale: perché è compressione](#23-complessità-spaziale)
   - 2.4 [Randomized SVD: algoritmo efficiente](#24-randomized-svd-algoritmo)

### **Parte 2: Setup e Fondamentali NumPy**
3. **[Google Colab: Setup e Workflow](#3-google-colab-setup)** `00:09:29 - 00:11:14`
   - 3.1 [Upload notebook da WeBeep](#31-upload-notebook)
   - 3.2 [Connessione a runtime](#32-connessione-runtime)
   
4. **[NumPy Basics: Random e Riproducibilità](#4-numpy-basics-random)** `00:11:14 - 00:13:54`
   - 4.1 [Random seed per risultati deterministici](#41-random-seed)
   - 4.2 [Generazione matrici random](#42-generazione-matrici)
   
5. **[SVD in Python: NumPy vs SciPy](#5-svd-python)** `00:13:54 - 00:19:26`
   - 5.1 [Full vs Thin SVD](#51-full-vs-thin-svd)
   - 5.2 [Uso della documentazione](#52-documentazione)
   - 5.3 [Ricostruzione matriciale](#53-ricostruzione-matriciale)
   - 5.4 [Verifica con norma del residuo](#54-verifica-residuo)

### **Parte 3: Ottimizzazione Performance**
6. **[Confronto Performance: Loop vs Vettorizzazione](#6-performance-comparison)** `00:23:24 - 00:34:46`
   - 6.1 [Metodo naive: loop Python (lento)](#61-metodo-naive-loop)
   - 6.2 [Operatore @ ottimizzato (30x più veloce)](#62-operatore-ottimizzato)
   - 6.3 [Broadcasting NumPy (2x più veloce di @)](#63-broadcasting-numpy)
   - 6.4 [Analisi scalabilità](#64-analisi-scalabilità)

### **Parte 4: Lab 1 - Compressione Immagini**
7. **[Image Compression con SVD](#7-image-compression)** `00:34:46 - 01:08:42`
   - 7.1 [Caricamento e preprocessing immagini](#71-caricamento-preprocessing)
   - 7.2 [Conversione grayscale e colormap](#72-conversione-grayscale)
   - 7.3 [Esercizio: SVD su immagini](#73-esercizio-svd-immagini)
   - 7.4 [Visualizzazione valori singolari](#74-visualizzazione-valori-singolari)
   - 7.5 [Scelta del cutoff per compressione](#75-scelta-cutoff)
   - 7.6 [Ricostruzione con diversi k](#76-ricostruzione-diversi-k)
   - 7.7 [Matrici rango-1 e eigenimages](#77-matrici-rango-1)
   - 7.8 [Confronto Tarantula Nebula vs Mondrian](#78-confronto-immagini)
   - 7.9 [Limitazione: dipendenza da orientamento](#79-limitazione-orientamento)

### **Parte 5: Lab 2 - Randomized SVD**
8. **[Implementazione Randomized SVD](#8-randomized-svd-implementazione)** `01:08:42 - 01:39:43`
   - 8.1 [Esercizio: codificare algoritmo rSVD](#81-esercizio-rsvd)
   - 8.2 [Soluzione commentata](#82-soluzione-commentata)
   - 8.3 [Confronto valori singolari: full vs randomized](#83-confronto-valori-singolari)
   - 8.4 [Effetto di k sulla accuratezza](#84-effetto-k)
   - 8.5 [Ricostruzione immagini: originale vs SVD vs rSVD](#85-ricostruzione-immagini)
   - 8.6 [Test su Mondrian: performance superiori](#86-test-mondrian)

### **Parte 6: Lab 3 - Image Denoising**
9. **[Denoising con Hard Thresholding](#9-denoising-hard-thresholding)** `01:41:02 - 02:26:04`
   - 9.1 [Aggiunta rumore artificiale gaussiano](#91-aggiunta-rumore)
   - 9.2 [Tre metodi di thresholding](#92-tre-metodi-thresholding)
     - Formula matrici quadrate
     - Rumore sconosciuto (formula slide 7)
     - 90% energia (baseline)
   - 9.3 [Esercizio: calcolo soglie e ricostruzione](#93-esercizio-soglie)
   - 9.4 [Soluzione: implementazione threshold](#94-soluzione-threshold)
   - 9.5 [Visualizzazione: clean vs error](#95-visualizzazione-clean-error)
   - 9.6 [Confronto metodi: trade-off rumore/dettagli](#96-confronto-metodi)

### **Parte 7: Homework e Conclusioni**
10. **[Homework: Background Removal da Video](#10-homework-background-removal)** `02:26:04 - 02:32:06`
    - 10.1 [Concetto: separazione sfondo/foreground](#101-concetto-separazione)
    - 10.2 [Preprocessing video: grayscale + subsampling](#102-preprocessing-video)
    - 10.3 [Trasformazione video → matrice](#103-trasformazione-matrice)
    - 10.4 [SVD per estrazione sfondo (k=1)](#104-svd-estrazione-sfondo)
    - 10.5 [Compito: sostituire con rSVD](#105-compito-rsvd)
    
11. **[Conclusioni e Feedback](#11-conclusioni)** `02:30:59 - 02:32:06`

---

## 🎯 Obiettivi del Laboratorio

**Competenze Teoriche:**
- ✅ Comprendere SVD come strumento di compressione
- ✅ Conoscere Randomized SVD per grandi matrici
- ✅ Applicare hard thresholding per denoising

**Competenze Pratiche:**
- ✅ Usare NumPy/SciPy per algebra lineare
- ✅ Ottimizzare codice: evitare loop Python
- ✅ Sfruttare broadcasting NumPy
- ✅ Visualizzare dati con Matplotlib
- ✅ Lavorare con Google Colab

**Applicazioni:**
1. **Image Compression**: comprimere immagini con SVD
2. **Image Denoising**: rimuovere rumore con thresholding
3. **Background Removal**: separare sfondo/foreground in video

---

## 📚 Prerequisiti

**Matematica:**
- Algebra lineare: matrici, prodotti, norme
- SVD: U, Σ, V^T
- Teorema Eckart-Young

**Python:**
- Sintassi base (variabili, loop, funzioni)
- Liste e array
- Slicing

**Librerie:**
```python
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
```

---

---

## <a name="introduzione"></a>1. Introduzione al Laboratorio

`00:00:02`

Okay, quindi buon pomeriggio a tutti. Questo è il primo laboratorio di analisi numerica per il machine learning. Io sono Matteo Caldano, e lei è Elena Basile. Io sono il teaching assistant, e lei è la tutor. Queste sono le nostre email, quindi se durante il lab avete domande, sentitevi liberi di contattarci via email. Vi mostrerò del codice, ed Elena sarà in giro per assicurarsi che tutti stiano seguendo quello che sto mostrando.

#### Contatti

- **Matteo Caldano** (TA): matteo.caldano@polimi.it
- **Elena Basile** (Tutor): elena.basile@polimi.it

`00:00:43`

Qual è la struttura di questa sessione di laboratorio? Quindi nei primi tipo 10 minuti, quarto d'ora, di solito mi piace fare un breve recap degli algoritmi e dei topic che useremo durante il lab per il coding e questo è solo alla lavagna, tutto ciò di cui abbiamo bisogno. Riguardo al programma, vorrei iniziare all'una e mezza e finire alle quattro per due motivi: prima di tutto così potete andare a casa prima alle quattro e secondo perché...

`00:01:22`

Questa è una sessione interattiva, quindi anche se non faccio pause, quello che di solito vi do è del tempo per fare il codice voi stessi e durante queste finestre temporali potete anche uscire, prendere un caffè, andare in bagno, fare quello che volete voi. Siete al master, quindi siete liberi di fare ciò che pensate sia meglio. Okay? Detto questo, vi mostrerò tutto in Google Colab.

`00:01:57`

E Google Colab è l'unico ambiente che supporto, nel senso che non vi aiuterò a risolvere problemi se state usando VS Code e qualche ambiente Conda o Mamba. Siete liberi di farlo. Li uso io stesso, ma Google Colab è molto, molto facile da usare e anche è quasi uno standard dell'industria. Quindi è un ambiente molto usato e inoltre non vale la pena per me risolvere tutti i vostri problemi di installazione di qualche ambiente locale.

`00:02:31`

Okay, inoltre durante il lab, mentre sto parlando, sentitevi liberi di farmi qualsiasi domanda, alzate la mano, commentate qualcosa, tipo state andando troppo veloci, state andando troppo lenti, per favore spiegate questo di nuovo, okay? Manteniamo questo interattivo. Avete domande? Okay, quindi iniziamo con un ripasso, e oggi parliamo di SVD, e in particolare SVD per compressione. Vedremo tre applicazioni: una per la compressione di immagini, una per il denoising di immagini (avete visto l'algoritmo di soglia forte, l'algoritmo di hard threshold durante la lezione), e infine vi mostrerò, se abbiamo tempo, un esempio per...

### Recap SVD e Compressione

`00:03:22`

Rimozione dello sfondo, e questo sarà anche parzialmente alla lavagna. Quindi, SVD: abbiamo una matrice A, e la scriviamo come U sigma V^T, questa è pseudo-diagonale, e questa e questa sono ortogonali, nel senso che U per U^T è l'identità. Okay, poi, come sfruttiamo SVD oggi è che, okay, la comprimiamo, una matrice, nella matrice A_k, che è uguale a sigma_1 · U_1 · V_1^T, più... sigma_k · U_k · V_k^T.

**Decomposizione SVD**:
```
A = U Σ V^T
```

Dove:
- **U** (m×m): matrice ortogonale sinistra (U·U^T = I)
- **Σ** (m×n): matrice pseudo-diagonale con valori singolari σᵢ
- **V^T** (n×n): matrice ortogonale destra trasposta

**Approssimazione rango-k**:
```
A_k = Σ(i=1 to k) σᵢ · uᵢ ⊗ vᵢ^T
```

`00:04:19`

Pensiamo ad A_k come una versione compressa di A. E sapete dal teorema di Eckart-Young che questa è la migliore approssimazione di A sia in norma due che nella norma di Frobenius. Tuttavia, A_k ha la stessa dimensione di A. Quindi perché questa è una compressione di A se hanno la stessa dimensione? A_k per noi è una versione compressa di A...

`00:04:50`

Perché salviamo solo i vettori u, v e i valori sigma. Okay? Quindi quando memorizziamo A_k non la memorizziamo come una matrice piena, ma invece il modo in cui effettivamente risparmiamo spazio è salvando solo i vettori singolari e i valori singolari. E quindi in questo modo invece di avere una complessità quadratica abbiamo una complessità lineare e in particolare uh useremo uh m...

`00:05:24`

k per ogni termine, u ha dimensione m, v ha dimensione n, e sigma ha dimensione 1. Quindi questo va linearmente rispetto a m e n, fissato k. Quindi in realtà stiamo buttando via molti dati se facciamo questo.

**Complessità Spaziale**:
- **Matrice completa A**: O(m×n)
- **Decomposizione rango-k**: O(k(m+n+1)) ≈ O(k(m+n))
- **Risparmio**: significativo quando k << min(m,n)

`00:05:58`

Quindi pensate che m e n sono molto molto grandi, tipo migliaia, centinaia di migliaia, e invece k è piccolo, magari 100. Quindi stiamo buttando via molti calcoli per calcolare tutti i valori singolari e tutti i vettori u e v con SVD. E la randomized SVD, rSVD... È una sorta di soluzione a questo perché siamo interessati solo ai casi più importanti, e la randomized SVD sfrutta proprietà di algebra lineare randomizzata per trovare un'approssimazione della SVD solo per alcuni dei valori singolari più importanti.

### Randomized SVD

`00:06:40`

Quindi durante la lezione, abbiamo visto questo abbastanza in dettaglio. Scrivo qui solo i passi. Quindi quando effettivamente codiamo l'algoritmo, abbiamo qui alla lavagna tutto. Quindi iniziamo con A·P, che è la matrice di sketch random, che appartiene a R^(n×l). Definiamo Z uguale ad A·P.

`00:07:19`

Grazie. In questo modo stiamo campionando lo spazio delle colonne di A. Poi calcoliamo la decomposizione QR di Z e quindi stiamo trovando una base ortonormale per A. Poi calcoliamo Y uguale a Q^T per A, quindi stiamo proiettando A sulla base Q che abbiamo calcolato nella QR. Calcoliamo U_Y applicando la decomposizione ai valori singolari a Y e infine Y deve essere sollevato nello spazio di A e usiamo Q per questo, quindi la U della SVD sarà uguale a Q per U_Y.

#### Algoritmo Randomized SVD

**Input**: A (m×n), k (rango target)

**Step**:
1. **P = randn(n, k)** - Sketch Gaussiano
2. **Z = A @ P** - Campionamento spazio colonne
3. **Q, R = qr(Z)** - Base ortonormale
4. **Y = Q^T @ A** - Proiezione
5. **U_Y, σ, V^T = svd(Y)** - SVD ridotta
6. **U = Q @ U_Y** - Ricostruzione

`00:08:23`

Ok? Qualche domanda? Questo dovrebbe essere materiale che abbiamo visto durante la lezione. Ok, quindi passiamo a Google Colab. Quindi, condividerò il mio schermo.

`00:08:54`

Ok. Puoi spegnere la luce?

`00:09:29`

Preferite così, o vedete abbastanza così, o preferite con la luce spenta?

---

## <a name="setup-colab"></a>2. Setup Google Colab

`00:10:00`

Spenta. Grazie. Okay, quindi ho caricato su WeBeep i notebook che useremo oggi.

`00:10:37`

E quello che fate è andare su Google Colab e poi caricarli. Quindi, per esempio, qui sto navigando dal mio computer, il primo lab, e iniziamo con il numero uno basic SVD. Dove vi mostrerò come usare SVD dalle librerie Python.

#### Accesso a Google Colab

1. Browser → `colab.research.google.com`
2. Login con account Google (gratuito)
3. CPU di base fornita da Google

#### Upload Notebook

```
File → Upload Notebook → Browse → Downloads/01_linear_algebra_basics.ipynb
```

**Workflow**:
1. Scarica notebook da WeBeep
2. Upload su Colab
3. Connect runtime
4. Esegui celle con `Shift+Enter` o pulsante Play

`00:11:14`

Okay, il notebook è caricato. Tutti sono qui? Okay. Quindi, SVD può essere fatto sia con SciPy che con NumPy. Qui stiamo importando le nostre librerie che useremo. Ci vuole un secondo per Colab per connettersi a una macchina virtuale che Google ci dà.

---

## <a name="linear-algebra-basics"></a>3. Basi di Algebra Lineare con NumPy

### Import e Random Seed

#### Import Essenziali

```python
import numpy as np
from scipy.linalg import svd
```

**Convenzioni**:
- `numpy as np`: convenzione standard, tutti riconoscono `np`
- `from scipy.linalg import svd`: importiamo solo SVD da SciPy

`00:11:45`

Okay. Sì. Okay. Va bene così. Okay. Poi generiamo una matrice random. Quindi, prima di tutto, vedete questo comando. Questo comando serve per impostare il seed. Questo è molto usato nel machine learning perché volete che il vostro risultato sia riproducibile.

#### Random Seed per Riproducibilità

```python
# Senza seed: risultati diversi ogni volta
A = np.random.random((5, 6))  # Matrice 5×6 uniforme [0,1]

# Con seed: risultati deterministici
np.random.seed(42)
A = np.random.random((5, 6))  # Sempre la stessa matrice A
```

`00:12:18`

Okay. Quindi, non volete che il vostro codice cambi ogni volta che lo eseguite, volete che se addestrate un modello lo stesso modello possa essere riaddestrato con lo stesso codice e quindi anche se i numeri che stiamo generando sono casuali, impostare il seed fa sì che i numeri generati casualmente siano sempre gli stessi. Ecco perché questi sono anche chiamati numeri pseudo-casuali. Cambiando il numero dentro il seed che fornite create numeri casuali diversi quindi per...

`00:12:51`

Esempio se invece di zero metto un numero diverso, la matrice che genererò sarà diversa, quindi con zero abbiamo questa matrice e se eseguo di nuovo e di nuovo lo stesso, il risultato resta lo stesso, ma se metto uno ho una matrice molto diversa che resta la stessa ogni volta che eseguo questo. Il modo in cui la genero è, okay, uso la libreria numpy, ha un sotto-modulo chiamato random...

### Generazione Matrici Random

`00:13:23`

E chiamo rand, che ci dà un numero casuale generato uniformemente in [0, 1]. E la matrice ha dimensione 5 per 4. Infine, se nell'ultima riga metto una variabile, questa variabile verrà visualizzata. E quindi questa è la matrice casuale che ho generato. Parliamo ora di SVD. Quindi abbiamo due...

**Quando usare il seed**:
- ✅ Consegna homework (risultati riproducibili)
- ✅ Debug codice (comportamento prevedibile)
- ✅ Testing statistico (risultati confrontabili)
- ❌ Produzione (serve vera randomness)

### SVD con NumPy e SciPy

`00:13:54`

Implementazioni diverse che sono praticamente le stesse, una fornita da numpy e una fornita da scipy. Sono dentro il sotto-modulo di algebra lineare e la chiamate con la funzione SVD. Ricevono in input due parametri: la matrice che vogliamo fattorizzare e un secondo parametro che è full_matrices. Dovreste aver visto durante la lezione che abbiamo due diversi tipi di decomposizione SVD, quella completa e quella sottile.

`00:14:25`

In quella sottile, quello che stiamo facendo è che per quanto riguarda sigma, la matrice pseudo-diagonale, stiamo buttando via tutti gli zeri. Invece, per quanto riguarda u o v, a seconda se la matrice è molto larga e molto corta, o molto sottile e molto alta, stiamo scartando i vettori singolari, sia di u che di v.

`00:14:56`

Di solito, preferiremo usare quella senza full_matrices, perché non abbiamo bisogno dei vettori singolari e degli zeri nella matrice pseudo-diagonale, okay? E questa è la nostra prima decomposizione SVD. Quindi chiamiamo da scipy, il modulo di algebra lineare, la funzione SVD con parametro a, e questo ci dà u, s...

`00:15:27`

E v già trasposta. Possiamo accedere alla shape di a, quindi il numero di righe e il numero di colonne, con l'attributo shape. Vedete che s non è una matrice, ma è solo un vettore. Poiché sigma è una matrice pseudo-diagonale, numpy è molto pragmatico e ci dà solo una diagonale, perché non è necessario memorizzare tutti gli altri zeri.

#### Shape delle Matrici

```python
A = np.random.random((5, 6))
print(A.shape)  # Output: (5, 6)
```

**Attributo `shape`**: restituisce tupla `(n_rows, n_cols)`

`00:16:02`

Possiamo anche vederli nell'ultima riga, ad esempio, possiamo mettere U e questo ci dà U e questa sarà la nostra matrice U. Quindi il messaggio da portare a casa qui è: ricordate che V è già trasposta e che S è solo una diagonale. Inoltre, un altro punto molto importante è imparare a usare la documentazione.

### Documentazione

`00:16:32`

Forse non ricordate esattamente come funzionano questi comandi. Per un programmatore, sapere come usare la documentazione è un'abilità molto importante. Quindi quello che potete fare è semplicemente andare su Google, cercare numpy SVD. Andate al primo risultato e potete leggere la documentazione e tutto ciò di cui avete bisogno è scritto qui, quindi ad esempio cosa restituisce: abbiamo s, vettori con i valori singolari dentro ogni vettore ordinati in ordine decrescente...

`00:17:04`

Ordine. Abbiamo vh, array unitari, e u che sono gli altri array unitari. Avete anche documentazione su full_matrices: se true, u e vh hanno la shape m per n e n per n, altrimenti le shape sono m per k e k per n dove k è il minimo tra m e n. Naturalmente questo è molto piccolo, probabilmente...

`00:17:36`

Quindi dato che anche l'esame è a libro aperto, imparate a usare la documentazione perché potete usarla e sfruttarla è un'abilità molto importante. Possiamo fare lo stesso anche con il sotto-modulo di algebra lineare di scipy e ad esempio se facciamo questo otteniamo più o meno lo stesso. Okay, diciamo che ora vogliamo ricostruire la nostra originale...

#### Accesso alla Documentazione

**Metodo 1: Question Mark (Colab)**
```python
?np.random.random  # Apre pannello documentazione laterale
```

**Metodo 2: help() (VS Code/locale)**
```python
help(np.random.random)  # Stampa testo completo
```

**Contenuto Documentazione**:
- Descrizione funzione
- Parametri e tipi
- Valori di ritorno
- **Esempi pratici** ⭐ (molto utili!)

> **💡 Best Practice**: Leggere documentazione prima di usare nuove funzioni

### Ricostruzione Matrice

`00:18:17`

Matrice A. Quindi prima di tutto dobbiamo ricostruire la matrice S. Un modo molto carino di fare questo è il seguente. Quindi prima di tutto inizializziamo S come una matrice di tutti zeri e la stessa shape di A e poi iteriamo per ogni i nella lunghezza, uguale da zero al numero di valori singolari e mettiamo sulla diagonale di s il valore s_i e...

`00:18:49`

In questo modo abbiamo creato la nostra matrice con tutti zeri a parte quelli sulla diagonale. Abbiamo un comando simile che automaticamente vi fa questo dal modulo di algebra lineare L.A. di SciPy chiamato diagsvd. Passate s, la shape di a e questo è questo. Quindi abbiamo una matrice piena di zeri a parte la pseudo-diagonale dove abbiamo i valori singolari. E ora siamo pronti...

`00:19:26`

A ricostruire a e verificare che effettivamente questa è una decomposizione di a. Quindi calcolerò a... Prima facciamo la moltiplicazione matriciale di s per vt, e poi facciamo di nuovo la moltiplicazione matriciale di questi con u. Questo è esattamente equivalente a fare questo. In NumPy, avete due modi diversi di fare la moltiplicazione matriciale. Avete la funzione np.matmul, a cui date due parametri, la matrice sinistra e la matrice destra che volete moltiplicare.

`00:20:03`

Oppure potete usare l'operatore @, che è la moltiplicazione matrice-matrice. Se venite da MATLAB, fate attenzione, non confondete questo con solo l'asterisco, che è il prodotto elemento per elemento. Quindi, se non avete mai usato MATLAB, tappate le orecchie per un secondo, ma se avete usato MATLAB, fate attenzione, perché potreste... Questo potrebbe essere un po' confuso, perché in Python, l'asterisco è quello che in MATLAB è punto-asterisco, ed è il prodotto elemento per elemento, mentre il simbolo @ in Python è in MATLAB equivalente solo all'asterisco, e questo è MATLAB.

`00:20:55`

Okay, per verificare che questa è effettivamente una decomposizione, controlliamo la norma del residuo. Quindi calcoliamo A_svd moltiplicando le varie matrici, la differenza tra queste e la nostra matrice originale, e la dividiamo per la norma di A. Quello che abbiamo non è esattamente zero, ma è vicino al machine epsilon, e quindi ogni volta che usate numeri floating point, dovreste sapere che...

`00:21:28`

Non vedrete quasi mai zeri. Dovreste solo verificare se l'errore relativo è più del machine epsilon. E questo è molto vicino al machine epsilon. Quindi possiamo considerare questi due oggetti come uguali. OK, sto andando troppo lento o va bene? Va bene. OK, perfetto. Cosa succede se mettiamo full_matrices uguale a false?

`00:22:07`

Beh, quello che cambia è ora che la shape di u e la shape di v sono leggermente diverse e in particolare, avranno la shape uguale a... vorrete usare il minimo tra m e n. Quindi ora, invece di costruire la matrice pseudo-diagonale S, possiamo costruire la matrice diagonale vera S con il comando da NumPy, diag.

`00:22:41`

Quindi diag è un comando che prende in input un vettore e restituisce una matrice quadrata, dove sulla diagonale mettete tutti gli elementi di s minuscola. Quindi ora, per ricostruire la matrice A, come prima, usiamo lo stesso processo. E quello che facciamo è, okay, moltiplichiamo U per il prodotto matrice-matrice di S per VT. E il risultato è lo stesso di prima ed è molto vicino alla precisione macchina.

---

## <a name="compressione-matriciale"></a>4. Compressione Matriciale con SVD

`00:23:24`

Parliamo ora di come effettivamente comprimere i dati. Quindi fino ad ora quello che abbiamo fatto è che abbiamo preso la matrice completa e l'abbiamo decomposta e poi abbiamo verificato che la decomposizione era effettivamente una decomposizione ricostruendo la matrice completa. Se volete comprimere A, quello che facciamo è che tronchiamo questa ricostruzione al k-esimo valore singolare. Quindi calcoliamo A_k come la somma, come vedete, come vi ho mostrato alla lavagna, del prodotto esterno dei primi...

`00:24:02`

k u_i, v_i pesati per il loro valore singolare. Tuttavia è importante fare questo in modo efficiente. Quindi, se fate un loop e iniziate effettivamente sommando a una matrice di zeri tutti questi componenti, quello che succederà è che questo sarà molto lento. E ora vi mostrerò questo. Quindi, importiamo la libreria time per cronometrare quanto tempo impiega una certa operazione.

`00:24:37`

E ora stiamo usando una matrice che è molto, molto più grande. 1.000 per 1.500. E quindi, prima di tutto, decomponiamo questa e calcoliamo la nostra matrice S maiuscola, che è la matrice diagonale. Questo ci metterà un po', un secondo.

### Confronto Performance: Naive vs Ottimizzato

`00:25:09`

E quindi, il primo approccio che vi mostro è quello naive. Quindi, quello che stiamo facendo qui. Okay, iniziamo il nostro timer. Inizializziamo la nostra matrice come piena di zeri, quindi una matrice di zeri con la stessa dimensione di a, e poi iteriamo su tutti i valori singolari, e aggiungiamo cosa? Il valore singolare i per il prodotto esterno dei due vettori. Quindi il prodotto esterno è solo u_i per v trasposta, e possiamo eseguirlo facilmente.

`00:25:57`

Questo ci metterà alcuni secondi, quindi potete vedere qui a sinistra che c'è un numero, che è il numero di secondi che la cella ha impiegato per eseguire, quindi questa è un'approssimazione del tempo. Tuttavia, abbiamo salvato il tempo effettivo dentro questa variabile loop_time. Ora confronteremo questo con la moltiplicazione matriciale fornita da NumPy.

`00:26:30`

Quindi questo è esattamente lo stesso, ma invece di farlo manualmente, stiamo sfruttando l'operatore @. Ha impiegato meno di un secondo, meno di un secondo in particolare è una frazione di secondo. Infine, l'ultimo approccio che voglio mostrarvi è forse il più importante, perché vi mostra un assaggio di ciò di cui NumPy è veramente capace...

### Broadcasting NumPy

`00:27:03`

E un po' anche della parte più sofisticata di NumPy. Quindi abbiamo che S, la matrice S maiuscola, che è k per k, è diagonale. E in particolare, la diagonale è uguale a s minuscola. Quindi quello che abbiamo, quindi se vogliamo calcolare il prodotto di u·s, ogni elemento dalla matrice, dalla regola prodotto riga-colonna...

`00:27:34`

È uguale alla somma di u_ik per s_kj. Tuttavia, s_kj è diverso da 0 solo se k e j sono uguali, perché dobbiamo essere sulla diagonale. E quindi questa somma è in realtà solo un valore. Che è u_ij per s_j. Okay, se fate questo, se pensate al prodotto riga per colonna...

`00:28:07`

Quello che stiamo facendo qui, sarebbe... una colonna dove solo un valore sarà diverso da zero e avete una riga di u che è tutta diversa da zero ma se fate il prodotto di questa riga per questa colonna qui abbiamo tutti zeri e quindi la somma sarà sempre zero a parte quando questo collide con questo e quindi in realtà non avete una sommatoria ma solo un prodotto di due numeri. NumPy non sa che la matrice...

`00:28:42`

È piena di zeri ma potete sfruttare questa parte voi stessi facendo questo: quindi avete u e la moltiplicate elemento per elemento con s minuscola. Quindi quello che state effettivamente facendo è che avete questa matrice... prendiamo la prima riga e poi avete s minuscola che è questa e state facendo l'elemento per...

`00:29:17`

Elemento di questa riga per questo vettore e ripetendo questo per ogni riga di u e questo è fatto automaticamente da questa sintassi. Quindi quello che sta succedendo qui è che NumPy capisce che u è una matrice e s è un vettore anche se non hanno la stessa shape e quindi non potete fare il prodotto elemento per elemento. Quello che fa è qualcosa che si chiama broadcasting, quindi sta cambiando la shape di...

`00:29:53`

s per far corrispondere la shape di u e in particolare sta facendo questo riga per riga, quindi sta prendendo s e la sta moltiplicando elemento per elemento con la prima riga della matrice a e sta ripetendo questo processo per ogni riga della matrice u. Qual è il vantaggio di questo processo è che state evitando la sommatoria e state evitando di memorizzare la matrice S che potrebbe essere molto costosa perché ha bisogno in spazio...

`00:30:29`

Memoria che è il quadrato di n e n potrebbe essere molto grande, quindi se avete la matrice A che è 1000 per 1000 dovete salvare un milione di numeri e man mano che n aumenta questo è molto molto costoso. E salvare solo s minuscola potrebbe essere un grande risparmio di tempo e anche un grande risparmio di memoria. Quindi eseguiamo questa cella. Anche questa gira in meno di un secondo, ma confrontiamo i risultati. Quindi qui c'è solo molta stampa dei vari tempi.

`00:31:06`

E questi sono i risultati. Quindi, tempo per ricostruire la vecchia matrice usando il naive for loop, 7 secondi. Tempo per ricostruire la matrice con l'operatore @ ottimizzato, 0.1 secondi. E quindi, vedete che questo è più di un miglioramento di 30 volte. Questo è dato dal fatto che l'operatore @ chiama codice C o Fortran che è stato compilato da qualcun altro e messo sulla vostra macchina da NumPy.

`00:31:43`

Invece, quando fate questo nel modo naive, nel for loop, tutto gira in Python. E quindi, è molto, molto costoso. Python è un linguaggio molto costoso. Tuttavia, sfrutta codice C o Fortran scritto da persone molto brave e lo nasconde dietro un'interfaccia Python. Quindi, dovete essere in grado di usare questa interfaccia efficacemente. E questa è la parte più importante riguardo NumPy. Infine, il modo vettorizzato, quello che usa broadcasting...

`00:32:14`

È quasi due volte più veloce dell'operatore @. La differenza tra le varie ricostruzioni è molto piccola. È 5 per 10 alla meno 15, che è ancora nel vicinato del machine epsilon, e quindi possiamo considerare queste ricostruzioni uguali. Sì, questo dovrebbe scalare. Voglio dire, man mano che la matrice diventa più grande, queste differenze dovrebbero diventare sempre più grandi.

`00:32:49`

Uno, perché il for loop scala davvero, davvero male in Python. E secondo, tutto è quadratico. Quindi, per esempio, il broadcasting rimuove il costo di memoria quadratico e il for loop extra. Quindi, man mano che la matrice diventa sempre più grande, penso che i guadagni in performance diventino sempre più grandi.

`00:33:22`

Scusa, non sto sentendo molto bene. In Python, in NumPy. Quindi è molto difficile perché dipende dall'implementazione dal vostro PC e così via. Ma potrei vedervi essere tipo meno di un secondo in Python è un'ora. Sì, sì. Perché i loop Python sono davvero, davvero lenti e scalano davvero male. Invece il codice NumPy è ottimizzato ed è in C. Tipo non abbiamo range checking perché il range checking è fatto solo una volta. Invece in Python ogni iterazione dovete essere sicuri di essere nel range. Dovete sollevare l'eccezione corretta e così via.

`00:34:09`

Quindi tipo è chiave. I loop Python sono tipo molto, molto cattivi. Non dovreste mai usarli a meno che. Ne avete bisogno se state parlando di roba di algebra lineare grande, OK, e molti, molti lab, vedremo come non usare i for loop, specialmente più avanti. Vi mostrerò JAX, che è una libreria di deep learning, in realtà, più tecnicamente, è una libreria di differenziazione automatica. E poi ci sono funzioni molto speciali per evitare di fare for loop del tutto.

---

## <a name="compressione-immagini"></a>5. Lab 2: Compressione Immagini

`00:34:46`

OK, altre domande? OK, quindi questa era solo l'introduzione. La prossima sarà effettivamente un esercizio. Quindi passiamo al prossimo notebook. Quindi lo carichiamo. OK, apriamo file, upload notebook, browse. Il secondo è image compression. Quindi ora effettivamente quello che stiamo facendo è applicare SVD per effettivamente comprimere un'immagine.

### Caricamento Immagini

`00:35:21`

Prima di tutto, prima di iniziare questo, quindi connettiamoci, quindi clicco connect, ci stiamo connettendo a una macchina virtuale che Google ci sta dando, avremo bisogno di alcuni dati, abbiamo bisogno di un'immagine da effettivamente comprimere, e ho caricato su WeBeep due immagini per mostrarvi due risultati diversi, che si chiamano Tarantula Nebula e Mondrian, e come le carichiamo? Beh, apriamo i file cliccando su questa icona cartella a sinistra, questa arancione, e poi una volta che siamo in questo menu, possiamo cliccare questo file con una freccia verso l'alto e caricare su questo storage di sessione alcuni file, e in particolare, caricheremo queste due immagini che comprimeremo.

`00:36:19`

Okay, quindi ora vedete che nella nostra macchina virtuale, abbiamo i nostri due file che possiamo usare per questo esercizio. Chiuderò il pannello per avere più spazio perché altrimenti non vedrete niente. Okay, tutti bene? Okay, quindi prima di tutto, leggiamo l'immagine. Quindi useremo matplotlib per leggere l'immagine, matplotlib.pyplot per fare del plotting...

`00:36:52`

NumPy per fare la nostra roba di algebra lineare. Definiamo dov'è l'immagine. L'immagine è nella cartella corrente. Quindi questo punto significa cartella corrente e uno slash e definiamo il nome, punto jpeg con l'estensione. Poi usiamo questa funzione imread per leggere l'immagine puntata da questo image_path.

`00:37:26`

Okay, abbiamo letto A. Se mettiamo una A qui, possiamo vedere cosa? Beh, automaticamente, il notebook è molto bravo e capisce che questa è un'immagine, ma in realtà, la parte importante è questa. Questo è un NDArray, quindi un array di numpy, con dimensione 500 per 600 per 3. Quindi il 3 qui è i 3 canali, rosso, blu e verde, RGB...

`00:37:57`

E poi questi invece sono le dimensioni, il numero di pixel nell'altezza e il numero di pixel nella larghezza. Infatti, se fate shape, ottenete la shape, che è questa tupla. Come visualizziamo l'immagine? Matplotlib.pyplot.plt ha una funzione, imshow, che anche una matrice la tratta come un'immagine.

`00:38:32`

E quindi possiamo visualizzarla. E quindi questa è la nostra Tarantula Nebula. Il prossimo passo è convertire questa immagine in scala di grigi. Quindi perché dobbiamo fare questo? Solo per semplicità, dato che lavora con matrici, quello che potremmo fare è che trattiamo ogni canale indipendentemente dall'altro, e applichiamo questo, o quello che stiamo per vedere, tre volte, una per ogni canale.

### Conversione Grayscale

`00:39:09`

Per semplicità, trasformiamo solo questo in scala di grigi e applichiamo SVD solo al canale grigio, che è la media tra rosso, verde e blu. Questo è solo per semplicità. Potete ottenere molto facilmente un'immagine a colori applicando questa procedura a ogni canale, ma per evitare confusioni, prima di tutto passiamo a scala di grigi. Come facciamo questo? Con questo comando. Okay, questo è un altro comando NumPy, e proveremo a capirlo.

`00:39:39`

Quindi, facciamo la media, applichiamo la mean, il nostro argomento è A, e poi abbiamo l'ultimo argomento con una keyword, che è axis uguale a 2. Cosa significa questo? Questo significa che la media, la mean, è applicata solo su questo asse. Cos'è l'asse? Questi sono gli assi. Quindi, questo è l'asse 0, questo è l'asse 1, questo è l'asse 2, alterato.

`00:40:11`

E quindi, se applichiamo la mean sull'asse 2, significa che stiamo facendo una media per ogni elemento del tensore concernente solo il terzo asse. Quindi, quello che stiamo effettivamente facendo è che stiamo andando pixel per pixel e prendendo la media attraverso il canale RGB. Questo è il modo NumPy. Questo è il modo ottimizzato perché non stiamo facendo un for loop. NumPy vi dà esattamente quello di cui avete bisogno solo leggendo la documentazione.

`00:40:43`

Quindi, questo è il modo ottimizzato perché non stiamo facendo un for loop. NumPy vi dà esattamente quello di cui avete bisogno solo leggendo la documentazione. Quindi, se non ricordate come funziona la mean, andate su Google, cercate NumPy mean. E leggete la documentazione quindi vedete che il primo parametro è un array like array contenente numeri la cui media è desiderata poi avete axis asse o assi quindi qui vedete che potete fare la mean attraverso più di un asse sui quali le medie sono calcolate il default è calcolare la media dell'array appiattito questo significa che l'array dà fuori solo un numero di default che è la...

`00:41:18`

Media del vecchio tensore. Quindi questa è la nostra immagine in scala di grigi e ora la plottiamo. Oh non è un gran ché, perché... Il motivo è che matplotlib non sa effettivamente che questa è un'immagine grigia. Per matplotlib, questo è qualche tipo di dato, che è in forma matriciale.

`00:41:52`

Questo potrebbe anche essere tipo un surface plot di qualche funzione bidimensionale. E di default, usa quella che si chiama una color map che ha senso per la maggior parte dei dati. Quindi se questo fosse, diciamo, un surface plot, e stiamo plottando le curve di livello di qualche funzione, questa è la palette di default, che si chiama Viridis, che è una palette molto buona per la visualizzazione dati. Tuttavia, questa è un'immagine in scala di grigi per noi, quindi quello che possiamo dire è...

`00:42:27`

Okay, non usare la color map di default, CMAP, ma usa quella grigia, perché per noi questa è un'immagine grigia. E quindi per noi, ha senso visualizzare questo in grigio. E quindi questa è la nostra Tarantula Nebula in scala di grigi, okay? Non spaventatevi per il fatto che ci siano colori diversi, è solo una rappresentazione diversa dello stesso oggetto, perché per noi, questi dati matriciali sono un'immagine. Per un'altra persona, questi dati matriciali potrebbero significare qualcos'altro. Questo punto è chiaro? Okay.

### SVD su Immagini

`00:43:08`

Qual è la dimensione dell'immagine? Come prima, x è un array intero. Abbiamo l'attributo shape, e possiamo calcolarlo con shape, e vediamo che abbiamo mantenuto la shape di prima senza il terzo canale. Quindi 500 per 600 pixel. Okay. Ho caricato il notebook con i problemi, quello con le soluzioni, ma voi non dovreste avere la soluzione. Qui dovreste avere uno spazio vuoto. È vero? Okay.

`00:43:43`

Quindi ora... Dovete fare questo. Quindi eseguiamo la SVD su questa immagine. Quindi ora vi darò la mia nuova risposta. E quello che potete provare a fare è applicare la SVD usando il file SVD e provare a capire quali sono le shape dei risultati. E se avete tempo, potete anche iniziare a visualizzare i valori singolari e quale è il loro comportamento nella cella.

`00:44:16`

Quindi prendiamoci tipo 10 minuti. Quindi se volete, potete andare e prendere un caffè o fare una pausa. E io ed Elena saremo in giro. E se avete domande, alzate solo la mano e proveremo ad aiutarvi. OK. È sull'altro lato dello schermo, perché questo è il backslash, che funziona solo su Windows, e l'altro su Linux, che è quello sull'altro lato.

_[Pausa per esercizio - 00:46:01 a 00:52:40]_

`00:52:40`

Quindi, quello che possiamo vedere qui è esattamente la versione di come vogliamo che appaia. E possiamo anche fare questo con l'X-Ray, quindi possiamo dire che disegniamo questa matrice su una scala di grigi. E in realtà, a volte non fate nemmeno il meglio. Per fare una matrice immagine grigia, quello che fate è mettere alcuni pesi a seconda del colore, perché alcuni colori sono più importanti di altri. Ma diciamo che il modo più semplice è semplicemente mediare e trattare la matrice che si ottiene come se fosse una matrice grigia.

`00:53:10`

E poi diciamo di metterla come un mapping di colori grigio, ma potrebbe essere qualche altro mapping.

### Visualizzazione Valori Singolari

`00:54:23`

Okay, andiamo avanti. Quindi per questa decomposizione SVD, usiamo la SVD di numpy, e usiamo la versione thin. E quindi questo è il risultato. Abbiamo una matrice che è 500 per 500. La nostra diagonale con 500 elementi, e poi infine, la trasposta di V, che è 500 per 600.

`00:54:53`

Il prossimo passo è plottare i trend di queste quantità. I valori singolari, sigma, la frazione cumulativa dei valori singolari. Quindi sommiamo uno per uno tutti i valori singolari fino a raggiungere la somma completa, e vogliamo vedere quanto la somma fino al k-esimo valore è vicina alla somma totale, e la frazione di varianza spiegata, che è la stessa di prima, ma con un quadrato.

`00:55:32`

Quindi questa è la sintassi subplot per inizializzare una figura con una riga e tre colonne, e poi questa è la dimensione della figura. Poi cosa facciamo? Okay, sul primo asse, cioè sul primo box plot, plottiamo in scala semilogaritmica y tutti i valori singolari. Sul secondo blocco, plottiamo la somma cumulativa, np dot cumsum function per somma cumulativa, divisa per la somma di s.

`00:56:08`

Quindi questo andrà da 0 a 1. E infine, la radice quadrata della somma cumulativa del quadrato, chiamata anche varianza spiegata. Quindi questi sono i plot. Quindi il primo commento riguarda l'uso degli assi corretti. Nel senso che questi sono alcuni dati che hanno alcuni trend e a volte usare il plot normale nella scala lineare non vi dà informazioni molto buone.

`00:56:48`

Per esempio, qui potete vedere che il valore singolare più grande è solo uno ed è molto, molto più grande di tutti gli altri. E quindi sembra che questa scala di plotting non sia molto buona perché non potete veramente vedere cosa sta succedendo qui sotto. E quindi un modo per risolvere questo è usare una scala semi-log y. Potreste anche provare a usare una scala log log, che è logaritmica in entrambi gli assi.

`00:57:21`

E quindi quello che vedete qui è in realtà che se vedete questo in scala logaritmica, in realtà i valori singolari importanti sono ancora qui. E poi c'è un... come un comportamento a gomito, e poi tutto questo diventa molto piccolo, molto rapidamente. Sì? Quando plottate, cambiate il nome della funzione. Potete fare qualcosa di simile anche per le altre scale.

`00:57:54`

Per esempio, potete mettere semi-log Y anche per gli altri. La cosa importante è che sappiate che a volte è necessario giocare un po' con le scale, per provare a capire quale è il vero comportamento dei dati, nel senso che a volte sarebbe più utile usare una scala invece dell'altra. Ci sono alcuni tipi di plot che di solito hanno solo una scala. Per quanto riguarda i valori singolari, a volte è utile usare una scala logaritmica, a volte è utile usarne una normale.

### Scelta del Taglio per Compressione

`00:58:31`

Quindi, questi tipi di grafici sono molto utili anche per capire dove fare il nostro taglio, se vogliamo comprimere la matrice. Quindi, quello che vogliamo fare è rimuovere tutti i valori che sono molto, molto piccoli, perché se un valore singolare è piccolo, significa che quella parte della matrice non è molto utile. Quindi, se, per esempio, usiamo questo plot, potremmo dire che un buon posto per tagliare sarebbe tipo qui intorno, intorno a 100, perché poi tutti gli altri molto, molto rapidamente scendono a zero.

`00:59:10`

Un altro modo per fare questo è, okay, questa è la frazione cumulativa dei valori singolari. Dove raggiungiamo, tipo, per esempio, il 90%? Quindi a quale punto di questa curva, i valori singolari spiegano il 90% della somma cumulativa di tutti gli altri, e questo succederebbe per meno qui intorno. Quindi, vedete, il 90% della somma cumulativa spiegata è qui, interseca qui, e quindi tagliereste a circa 300.

`00:59:43`

Questo potrebbe essere un approccio, potreste anche usare un approccio dove dite, okay, vogliamo spiegare il 95% o il 99% della varianza, okay, il 99% della varianza è qui, e quindi il posto dove tagliare questo è circa qui, che è circa 100. Non c'è una soluzione universale su dove tagliare per comprimere la matrice.

### Ricostruzione con Diversi k

`01:00:13`

Siate consapevoli di questa regola empirica per avere un'idea di dove potrebbe essere un buon posto per fare la compressione. E poi di solito quello che fate è che ispezionate manualmente i dati e provate a capire se la vostra scelta era buona o no. Quindi è anche molta esperienza come in molti topic di machine learning. Ora visualizziamo le migliori matrici.

`01:00:44`

Quindi la matrice che otteniamo mantenendo k valori singolari. Quindi questi sono sei casi diversi e quindi inizializziamo il plot con due righe e tre colonne per un totale di sei. E poi iteriamo su tutti i casi che vogliamo usare. OK, per esempio, qui stiamo usando K uguale a uno. Stiamo tagliando una tabella a due uguale a cinque, 10, 15 e poi iteriamo su tutti questi punti di taglio.

`01:01:17`

OK, il nostro K sarà l'indice. E poi ricostruiamo la chiave. Come facciamo questo? Mantenendo le prime K colonne di u. Quindi questa è una sintassi di slicing. Avete visto una sintassi di slicing in Python prima? Sì. OK, quindi con questo, stiamo mantenendo tutte le righe e le prime K colonne di u.

`01:01:48`

E poi per quanto riguarda questa parte, stiamo mantenendo i primi K valori singolari di s minuscola e di v trasposta. Stiamo mantenendo le prime K righe e tutte le colonne. Questa è la nostra A_k ricostruita. La plottiamo in imshow, e poi impostiamo il titolo, uguale a K. Quindi questo è il risultato.

`01:02:21`

Faccio un po' di zoom indietro. Quindi questo è quello che succede se usiamo solo un valore singolare, e il primo vettore u, v. E vedete che man mano che usiamo sempre più vettori per ricostruire la matrice...

`01:02:51`

Stiamo approssimando sempre di più quella finale. Quindi questo è con 15. Possiamo ripetere questi con valori diversi di k, quindi per esempio invece di usare 1, 2, 5, 10, 15 e 50, possiamo usare 1, 5, 50, 100, 200, 300 e 500, questi non sono più 6, okay, quindi qui vedete che vediamo quasi niente.

`01:03:56`

Dovreste vedere anche dal proiettore che questo e questo sono un po' sfocati, ma invece quando raggiungete qualcosa come 300 o 500, penso che dal proiettore possiate vedere quasi nessuna differenza da questo a questo. E in questo caso, stiamo già risparmiando tipo il 50% dello spazio perché stiamo mantenendo solo metà dei valori singolari perché la dimensione è più o meno 600.

### Matrici Rango-1 e Componenti

`01:04:36`

Tuttavia, questo non è ancora un risultato molto buono perché non stiamo comprimendo troppo, e questo è anche dovuto al fatto che questo tipo di immagine ha molte caratteristiche casuali. Quindi questa è una galassia. E questo assomiglia molto a rumore. Invece, quando dopo cambierete l'immagine, vedrete che la compressione sarà molto, molto migliore perché nel dipinto di Mondrian, ci sono caratteristiche verticali e orizzontali molto chiare, tipo molto geometriche.

`01:05:11`

E in questo caso, SVD funziona molto molto meglio perché siamo più lontani dal rumore nero. Il prossimo passo è visualizzare la k-esima matrice rango 1. Cosa è la k-esima matrice rango 1? Sono le matrici che compongono la matrice originale quando fattorizziamo con SVD.

`01:05:48`

Questo significa che vogliamo visualizzare tutti i componenti perché stiamo creando qualcosa che è come una funzione di base, una base funzionale per la nostra immagine originale e vogliamo vedere quali sono le direzioni rispetto alle quali lo spazio è diviso. E la k-esima matrice rango uno vi dà queste. E in particolare, queste sono ordinate. Quindi la prima sarà la più importante. La seconda sarà la seconda più importante perché queste sono pesate dai valori singolari, che nella SVD sono ordinati dal più grande al più piccolo.

`01:06:28`

In particolare, la k-esima matrice rango uno è solo questa, che è il prodotto esterno del vettore U per il vettore v. Voglio dire, è solo questo componente. E la k-esima matrice rango uno vi dà queste. Vedete che a è una combinazione lineare di tutti questi componenti e questo componente è come la direzione di una certa funzione di business e questi sono i modi e dato che questi sono i...

`01:06:59`

Valori singolari sono ordinati dal più grande al più piccolo, stiamo vedendo quali sono i componenti più importanti fino a quello meno importante. Quindi questo è il risultato. Una parte importante qui è che...

`01:07:33`

Man mano che k diventa sempre più grande, quello che stiamo vedendo sono caratteristiche sempre più fini, in particolare, come vi stavo dicendo prima, SVD è molto bravo a trovare caratteristiche molto grandi e molto geometriche. Vedete qui molto chiaramente che ci sono linee verticali e orizzontali. Abbiamo caratteristiche molto, diciamo, a grana grossa. E invece, man mano che K diventa sempre più grande, le caratteristiche meno importanti che la SVD trova sono quelle dove ci sono oggetti sempre più fini.

`01:08:03`

Quindi potete vedere, in un certo senso, che queste linee verticali e orizzontali costruiscono oggetti che sono sempre più fini, e questi sono quelli che sono più simili al rumore. Ed è per questo che SVD trova difficile comprimere questi tipi di immagini, perché sono composte, per la maggior parte, da rumore. E la parte di rumore è quella che la SVD pensa sia la meno importante.

---

## <a name="randomized-svd-esercizio"></a>6. Randomized SVD - Esercizio

`01:08:42`

Okay, è tutto chiaro? Okay, per il prossimo passo quello che farete è la randomized SVD. Quindi, è un po' più difficile del passo di prima ma quello che faremo è implementare questo algoritmo. Okay? Quindi ora vi abituerete a trasformare algoritmi matematici in algoritmi python. Uh quindi qui dovete implementare una funzione randomized_svd, passate A, la matrice...

`01:09:15`

Che volete calcolare la randomized svd e k che è la dimensione k della matrice P da cui proiettate. I due comandi che dovete conoscere ora sono i seguenti: il primo è np.random.randn. E la differenza da prima è che quella piccola n significa che il rumore invece di essere uniforme è normale e poi invece di passare, e poi passate ancora le dimensioni quindi lo sketch random è n per K. E quindi qui passate n e k per costruire lo sketch random.

`01:09:58`

Poi l'altra parte è che avete la decomposizione QR e per quanto riguarda la decomposizione QR. Dovete scoprire che l'algebra lineare, la decomposizione QR a cui passate la vostra matrice Z, e vi restituirà Q e R così. Okay, quindi l'algebra lineare QR restituisce esattamente quello che pensate, le matrici Q e R e per costruire questa...

`01:10:30`

Usate la funzione randn. Che è normale. Numero random con dimensione n per k, okay? Vi do, diciamo, 10 minuti, anche un quarto d'ora se ne avete bisogno, e provate a implementare la randomized SVD e applicarla ad A, e provate a fare tutto quello che abbiamo fatto prima. Ora vi do 20 minuti, e provate a ripetere tutto quello che abbiamo fatto usando la randomized SVD.

`01:11:03`

E tutto è esattamente lo stesso a parte implementare questo algoritmo che restituisce U, sigma e V con quello randomizzato. Quindi restituite questa U, questa sigma e questa VT. Queste sono date dalla SVD applicata a Y, e la U moltiplicando UI per la matrice Q che ottenete dalla QR. E poi il codice è esattamente lo stesso di prima. E vi do 20 minuti per questo.

_[Pausa per esercizio - 01:11:57 a 01:27:45]_

### Soluzione Randomized SVD

`01:27:45`

Okay, come state andando? L'avete fatto? Mi sembra che più o meno tutti voi abbiate provato, e quindi per rimanere nei tempi, vediamo la soluzione. Quindi prima di tutto, otteniamo la shape di A. Perché abbiamo bisogno di M e N per calcolare la matrice sketch. Quindi potete ottenere N in modi diversi. Per esempio, qui ottengo sia M, che è questo underscore, e N. E in realtà non ho bisogno di M. E quindi la cosa che di solito fate in Python è che se non avete bisogno di una variabile, mettete solo un underscore.

`01:28:31`

Un altro modo per ottenere N sarebbe fare A dot shape. Questa shape è una tupla con due elementi, e ottenete il secondo, e ottenete il secondo passando uno. E questo è esattamente lo stesso. Poi calcolate la matrice sketch P. Quindi np dot random, random normal di dimensione N per K. Z è A moltiplicazione matriciale P. Poi calcoliamo la decomposizione QR.

`01:29:01`

np dot linear algebra dot QR. Passiamo Z e abbiamo in output Q e R. Non usiamo R e quindi come convenzione metto un underscore. Potreste mettere una R qui, non è un errore in alcun modo, ma le best practice in Python sono se non avete bisogno di una variabile mettete un underscore e quindi anche chi sta leggendo il codice sa che non useremo questa variabile. Y è Q trasposta per A?

`01:29:31`

Questo è un modo per fare la trasposta dot capital T. Un altro modo che è esattamente uguale è numpy dot transpose di Q. Questi sono due modi per fare la stessa cosa, tipo potete fare la moltiplicazione matrice matrice con il simbolo @ o numpy dot matmul. Calcolate la SVD con le matrici thin. Le SY e VTY sono quelle di cui abbiamo bisogno, ma U è nello spazio proiettato e dobbiamo...

`01:30:06`

Tornare nello spazio completo, e quindi usiamo la matrice Q per tornare nello spazio completo, e quindi restituiamo U, S e VT. Poi usiamo la nostra funzione per calcolare la SVD randomizzata, e otteniamo la U random, S random e VT random, e poi questo è esattamente lo stesso plot di prima, quindi non sto commentando il codice, sto commentando il risultato, l'unica cosa che aggiungiamo è questo plot logaritmico, e qui uso questo stile, quindi questa stringa qui...

`01:30:44`

Definisce lo stile del plot, la O significa che usate marker circolari, il più significa che usate marker più, e questo dash significa che abbiamo una linea continua, e questo è il plot, quindi quello che stiamo vedendo è che... Stiamo matchando i valori singolari della SVD completa abbastanza bene fino a, diciamo, i primi 10, molto bene, poi fino a tipo 30, abbastanza bene, e poi inizia a non essere molto buono intorno a 30 o 40.

`01:31:24`

Qui vedete che inizia a esserci una differenza piuttosto grande. Questo è perché con K, calcolate approssimativamente i valori singolari, e a un certo punto, gli algoritmi random si rompono. Se volete valori singolari più corretti, dovete aumentare K. Questo significa che i calcoli sono più costosi, perché stiamo mantenendo di più dello spazio originale quando proiettiamo e troviamo la base.

`01:31:57`

Tuttavia, questo significa anche che avremo più valori singolari che sono molto buoni. Invece, qui, prima era fino a 30, ora abbiamo valori singolari molto buoni, almeno fino a 100. Quindi, man mano che aumentate k, vi avvicinate sempre di più all'applicare la SVD completa. Applichiamo la randomized SVD esattamente come prima, quindi questo è praticamente lo stesso codice, e confrontiamo l'immagine originale, la prima, con quella ricostruita, con i primi k valori singolari dalla SVD originale, e quella con quella randomizzata.

`01:32:48`

Dal proiettore, penso che possiate vedere differenze molto piccole tra le tre. Dal vostro laptop, dovreste vedere la differenza. Questa è un po' più rumorosa. Tuttavia, penso che i risultati siano abbastanza buoni, e con randomized SVD, stiamo risparmiando molti costi computazionali.

### Confronto con Mondrian

`01:33:22`

Avete domande? Le soluzioni saranno caricate su WeBeep dopo un paio di giorni dal lab. Quindi potete aspettarvi che la cartella delle soluzioni sia sotto lab01 in pochi giorni con tutte le soluzioni che vi sto mostrando ora. Altre domande? E se no, voglio mostrare cosa succede con l'altra immagine. Quindi eseguiremo di nuovo tutto il codice, ma cambiando l'immagine. Quindi invece della Tarantula Nebula, vi mostro ora il Mondrian. Possiamo fare un run all qui e eseguiremo tutto di nuovo. Quindi vedete, questa è l'immagine. È un dipinto molto famoso. E vedete molto chiaramente un'immagine geometrica, orizzontale e verticale, allineate con l'asse. E questo è molto importante.

`01:34:22`

In scala di grigi, come prima, e poi i valori singolari. Vedete in particolare qui che a un certo punto raggiungiamo qualcosa che è molto vicino all'epsilon macchina, e questo significa che in realtà siamo stati in grado di ricostruire la matrice quasi esattamente con solo un punto qui. Tutti gli altri valori singolari qui sono praticamente solo rumore, e quindi in realtà con caratteristiche molto geometriche, SVD funziona molto bene, e infatti siamo quasi senza perdite se tagliamo qui.

`01:35:05`

Queste sono le ricostruzioni, quindi questa è solo la prima, diciamo, eigenimage, e poi abbiamo ricostruzioni con sempre più piani. Ma già con k uguale a 50 dal proiettore, penso che possiate vedere quasi nessuna differenza. Queste sono le eigenimages, e come abbiamo notato prima, iniziamo con caratteristiche molto grandi, e qui potete vedere quadrati con un diametro piuttosto grande, e man mano che k aumenta, le eigenimages che sono la direzione su cui proiettiamo l'immagine e poi la ricostruiamo diventano sempre più fini.

`01:35:47`

Se applicate il randomized a queste, funziona ancora molto bene, e abbiamo ora k uguale a 200, che forse per un'immagine di dimensione 300 per 700 è un po' più grande, quindi possiamo anche provare con qualcosa come 50, e k uguale a 50, abbiamo ancora un accordo molto buono tra il valore singolare reale e quello randomizzato.

`01:36:23`

La differenza è molto, molto piccola. E queste sono le immagini ricostruite. Abbiamo 50 con k uguale a 5. E penso che possiate vedere quasi la differenza tra quella originale, quella con i valori singolari reali, e quella economica con i valori singolari randomizzati. Un ultimo commento prima di chiudere questo notebook è...

`01:36:55`

Voglio davvero, davvero mostrarvi quanto sia importante che le caratteristiche geometriche siano allineate con gli assi. Perché se le caratteristiche geometriche non sono allineate con gli assi, molte cose si rompono. In particolare, quello che sto aggiungendo con questa riga è una funzione da SciPy e in particolare il modulo image che ruota l'immagine. Quindi prendiamo A, che è la nostra immagine, e la ruotiamo di 20 gradi.

`01:37:28`

Eseguiamo ancora tutto. Okay, per riferimento, questa è l'immagine. Come prima, un po' di nero qua e là. Rotazione di 20 gradi. Valori singolari, molto, molto diversi. Prima, avevamo tipo un plateau e un grande drop. Ora, tutto si rompe. Non siamo più in grado di avere una decomposizione perfetta dell'immagine con solo una rotazione di 20 gradi. Abbiamo bisogno di tutti i valori singolari e ancora siamo ancora al decimo alla seconda.

`01:38:00`

Prima, arrivavamo a 10 alla meno 11. E queste sono le ricostruzioni. Potete vedere che le caratteristiche a grana grossa non sono più riconoscibili. Abbiamo qualcosa che è molto brutto. E poi, naturalmente, man mano che k aumenta, la ricostruzione migliora sempre di più, ma qui potete vedere molto rumore ancora con k uguale a p.

`01:38:32`

Ci sono immagini, a parte la prima, le altre sono un po' simili, con lo stesso comportamento per cui man mano che k diventa sempre più grande, abbiamo strutture sempre più fini. Ma quello che voglio mostrarvi è che tutto si rompe un po' anche per la SVD randomizzata. Okay, i risultati sono peggiori. Per la SVD randomizzata, abbiamo a 50, abbiamo un'immagine molto, molto rumorosa che anche dal proiettore potete vedere è molto, molto diversa da questa, che è molto, molto diversa da questa.

`01:39:03`

Quindi il messaggio da portare a casa è: SVD funziona per questo tipo di approcci solo quando le caratteristiche sono molto geometriche e molto allineate con l'asse, solo una piccola rotazione rompe tutto. Il codice era esattamente lo stesso a parte questa rotazione di 20 gradi e la SVD randomizzata ora funziona piuttosto male, direi. Qualche domanda prima di chiudere questo argomento, naturalmente potete contattarmi via email su questo argomento in qualsiasi momento volete. Okay, ultimo notebook di oggi, oh.

`01:39:43`

Scusatemi, scusa, potrebbe essere un approccio quindi se volete comprimere un'immagine, nella vita reale, SVD non è l'algoritmo che volete usare. Okay, questa è solo un'applicazione, che è, penso, molto significativa perché vi fa capire un po' di più SVD.

`01:40:21`

Non rimanete concentrati sull'applicazione perché se volete comprimere immagini, ci sono algoritmi come JPEG che funzionano a meraviglia. Quindi, non rimanete bloccati nell'applicare SVD necessariamente per questo tipo di applicazioni. Quella che vedremo dal prossimo lab, la prossima applicazione SVD, invece, sarà molto più realistica e molto più potente. Questa è più per mostrarvi le limitazioni di SVD e abituarvi un po' a usare SVD. L'approccio che avete suggerito potrebbe funzionare, ma ancora per l'elaborazione di immagini non è qualcosa che suggerirei.

---

## <a name="denoising"></a>7. Lab 3: Denoising con SVD

`01:41:02`

Okay, ultimo notebook di oggi. Eh, è rimozione del rumore. Quindi per questo, dato che siamo abituati a usare queste immagini, le useremo ancora. Se aprite un nuovo notebook, il runtime cancellerà tutti i vostri dati. Quindi dobbiamo caricare le immagini di nuovo. Quindi fate attenzione se usate questo per un progetto. Google non salva i vostri dati a meno che non montiate il vostro Google Drive. Okay. E questa è una cosa che per ora vi sto mostrando.

`01:41:53`

Non vi sto mostrando. Quindi tutti i dati che abbiamo nella vostra cartella e che avete caricato, hanno un arco temporale molto limitato. Quindi fate attenzione. Quindi ci stiamo connettendo. E ora carico le immagini, e questo è l'avviso per esattamente quello che abbiamo appena detto. Assicuratevi che i vostri file siano salvati altrove. Questo file runtime sarà cancellato quando il runtime viene terminato.

`01:42:34`

Okay, quindi quello che stiamo facendo ora è che stiamo prendendo queste immagini, stiamo aggiungendo artificialmente del rumore, e stiamo usando SVD per pulire le immagini dal rumore. Perché funziona? Perché il rumore ha caratteristiche molto, molto fini che sono i valori singolari meno importanti nella SVD. Quindi se sogliamo correttamente i valori singolari, stiamo rimuovendo le caratteristiche fini che sono le caratteristiche di rumore.

### Aggiunta Rumore Artificiale

`01:43:15`

Quindi ora partiamo dal dipinto, che sappiamo funziona molto bene. Lo carichiamo con la funzione imread, e poi trasformiamo l'immagine in scala di grigi. Quindi usiamo la mean di numpy, come abbiamo fatto prima. Ora solo per scopi di normalizzazione, normalizziamo tutto per il massimo, quindi sappiamo che tutti i valori nella matrice sono tra 0 e 1. Questo è solo per semplicità per ora. Ora, questo non è davvero rilevante per l'algoritmo.

`01:43:48`

Poi definiamo gamma, il rumore, come 0.1, e creiamo l'immagine rumorosa. Quindi prendiamo x, e aggiungiamo numeri normali casuali per gamma. Infine, clippiamo il rumore. Quindi dato che abbiamo normalizzato l'immagine tra zero e uno non ha senso avere valori che sono più piccoli di zero o più grandi di uno quindi usiamo questa funzione chiamata numpy clip, che si assicura che tutti i valori che sono fuori un certo range siano messi esattamente al...

`01:44:25`

Confine del range quindi se abbiamo qui per esempio abbiamo 1.1 sarà clippato a 1. Se abbiamo qualcosa tipo meno due sarà clippato a zero e ora possiamo plottare le due immagini quindi alcuni plot, una riga due colonne dimensione figura questi sono gli assi sul primo blocco mettiamo la nostra immagine originale mappa colori grigio sul secondo blocco asse uno stiamo mostrando l'immagine rumorosa e questo è il risultato.

`01:45:02`

Quindi questa è la nostra immagine originale e qui potete vedere che abbiamo aggiunto il nostro rumore. È sempre un buon controllo plottare gli oggetti con cui stiamo lavorando per assicurarsi che siano effettivamente quello che crediamo siano. Poi vogliamo applicare SVD per rimuovere il rumore. E avete visto nelle slide, probabilmente dell'ultima lezione, che ci sono diversi modi per trovare questa soglia.

### Tre Metodi di Thresholding

`01:45:38`

Qui vi propongo di usare tre metodi diversi. Il primo è per matrici quadrate. Anche se questa matrice non è quadrata, in realtà funziona abbastanza bene finché la matrice non è molto deformata. Quindi partiamo con questo. Poi vi chiedo anche di implementare la soglia con rumore sconosciuto. Questa è una formula dalle slide. Eh, tipo la sette in particolare da queste, eh, eh, yet. E infine, come benchmark, di solito mettete anche qualcosa che è molto semplice e in particolare 90% di energia. E con questo intendo, eh, 0.9, eh, la frazione di 0.9 della somma cumulativa dei, eh, valori singolari.

`01:46:30`

Quindi prima vi ho mostrato il grafico con la somma cumulativa e sogliamo dove raggiungiamo il 90% della somma cumulativa totale. E qui vi darò 15 minuti. Eh, provate a calcolare queste soglie e ricostruire le immagini, quindi è esattamente quello che abbiamo fatto prima. Significa che dovete calcolare la SVD trovare il k di decadimento e poi ricostruire l'immagine. La differenza con prima è che K non è dato come indice, ma come valore dei valori singolari che calcolerete. E poi trovate i valori singolari dove questo k è. Quindi, per esempio, applicate questa formula.

`01:47:16`

Trovate che tau è 27. Trovate dove 27 è tra tutti i valori singolari. Trovate che i valori singolari con valore 27 è il 51. E quindi mettete K uguale a 50 e ricostruite AK con K uguale a 50. È chiaro questo? Okay, sarò in giro. Usiamo 15 minuti.

_[Pausa per esercizio - 01:47:50 a 02:05:27]_

### Soluzione Denoising

`02:05:27`

Okay ragazzi, vediamo la soluzione, perché altrimenti non possiamo mantenere la scaletta. Quindi, questa è la nostra SVD. Una volta che abbiamo la nostra SVD, calcoliamo le soglie nei tre modi. Quindi, la prima è la formula per matrici quadrate. Questa è solo per matrici quadrate, ma funziona ancora piuttosto bene per matrici che non sono quadrate.

`02:05:58`

Quindi, proviamo a usarla. Quindi, m e n è uguale a x dot shape, e otteniamo la shape, e poi il nostro cutoff è uguale a cosa? È 4 diviso per la radice quadrata di j, tutto per la radice quadrata di n per gamma. Questo è il nostro cutoff. Poi la somma cumulativa, per ora la chiamo solo cs, è uguale a numpy.cumulativeSum di s diviso per numpy.cumulativeSum di s.

`02:06:47`

E vogliamo calcolare a quale indice questo vettore cs, che tutti i valori stanno tra 0 e 1, raggiunge 0.9. Quindi possiamo trovarlo facendo numpy.min di np.where di cs maggiore di 0.9. Quindi lasciatemi spiegare un po' questo. Questo è l'indice e quindi la soglia.

`02:07:29`

È uguale a S di r nove. Quindi spieghiamo un po' cos'è tutta questa roba. Quindi plottiamo. Stampiamo per ora CS. Quindi CS è questo vettore con la somma cumulativa dei valori singolari. E partite dal primo. Quindi il primo valore singolare spiega il 20% di tutti gli altri valori singolari fino alla fine dove raggiungete il 100%. Okay, quindi la somma cumulativa sta tra perché è normalizzata per la somma totale sta tra.

`02:08:03`

E vogliamo trovare a quale indice abbiamo almeno il 90%. Quindi qui, perché qui significa che stiamo mantenendo tutti i valori singolari che fanno almeno il 90% di tutti gli altri. Come facciamo? Quindi prima di tutto, usiamo questo numpy where. Quindi se facciamo CS maggiore di 0.9 abbiamo un vettore con tutti true o false a seconda di questa uguaglianza.

`02:08:38`

Quindi vogliamo trovare o l'ultimo che è false o il primo che è true. In particolare vogliamo tagliarlo qui, il primo che è true. Quindi come facciamo? Prendiamo il minimo perché se facciamo np.where abbiamo tutti gli indici dove questa uguaglianza è true.

`02:09:16`

Quindi questi sono tutti gli indici da 244 alla fine e poi prendiamo il primo. Quindi in realtà, se facciamo o zero o il mean è lo stesso. Quindi questo è il nostro indice. Quindi il nostro 90 è solo l'indice. E ora mettiamo questo indice in tutti i valori singolari. E quindi abbiamo il valore che in realtà è la soglia.

`02:09:54`

No, questo è il where. Oh, scusa. Tutto è, vedete che tutto è dentro queste parentesi. Significa che questo è un array di un array. Quindi dovete fare zero, zero.

`02:10:25`

E ora, cutoff 90 dovrebbe essere un numero. Okay. È chiaro questo, o dovrei spiegarlo di nuovo? Dalle vostre facce, penso che sia meglio se lo spiego di nuovo. Quindi, CS è la somma cumulativa. Okay? È un vettore che dice, okay, questo è il primo valore singolare.

`02:10:56`

Questo spiega il 20% di tutti i valori singolari. Il secondo valore singolare, insieme al primo, spiega il 26% di tutti gli altri valori singolari. Perché questa è la somma cumulativa normalizzata per la somma totale. Volete trovare quale è il valore singolare che spiega il 90%, a quale punto di tutti i valori singolari, quel valore singolare con tutti quelli che vengono prima spiegano il 90 di tutti quelli quindi dobbiamo...

`02:11:29`

Trovare quale è l'indice proprio qui questo è il primo valore singolare che spiega più del 90 quindi se mettiamo un maggiore di 90 qui avete un vettore con tutti false prima e tutti true dopo e in particolare vogliamo trovare questo indice come troviamo questo indice abbiamo una funzione.

`02:11:59`

np dot where che vi dà tutti gli indici che sono true quindi vi dà tutti gli indici di questa parte infatti se eseguiamo questo avete tutti questi indici. Ora, questo non è un array, se fate davvero attenzione, avete queste parentesi qui, e queste parentesi qui, quindi questo è una tupla di array, e vogliamo questo, quindi con il primo, otteniamo l'array, con il secondo, otteniamo il primo elemento, questo è l'indice dove vogliamo tagliare i valori singolari, e quindi troviamo il valore singolare che è a questo punto, 0, 0.

`02:13:07`

Questo è il posto dove vogliamo tagliare la soglia, la nostra SVD. Infine, usiamo la formula con rumore sconosciuto, dobbiamo solo implementare questo, quindi prima di tutto definiamo beta è come m diviso per n, omega è uguale a 0.55 per beta alla terza meno 0.95 per beta al quadrato, più 1.82 per beta, più 1.43.

`02:13:54`

E poi il cutoff con rumore sconosciuto è uguale a omega per np dot median di s. Questa è la formula. Omega di beta per il valore singolare mediano. Questi sono i tre cutoff.

### Visualizzazione Soglie e Ricostruzione

`02:14:25`

Ora li plottiamo. Quindi, PLT dot semi log y di s. Lo plottiamo in nero. E possiamo dire label uguale valori singolari. PLT dot... E poi vogliamo plottare alcune linee orizzontali per le soglie.

`02:14:56`

Vogliamo plottare dove vogliamo tagliare. Quindi, alcune linee orizzontali. E c'è una funzione in PLT chiamata axeHorizontalLine. E possiamo mettere solo il valore. E metterà una linea orizzontale a quel valore. Quindi, vogliamo cutoff, color rosso, label cutoff square, e poi plt.axh line, cutoff 90, color verde, label, cutoff 90 energia, e infine, color blu, questo è cutoff rumore sconosciuto, per visualizzare la legenda, plt.legend.

`02:16:51`

Oh, okay, color, non potete solo metterlo così, dovete mettere, okay, la lettera, questo color, okay, e questo è il plot. Quindi questo è come tutti i valori singolari stanno andando, ci sono all'inizio alcuni molto importanti, e poi c'è un gomito, quello rosso è se usiamo la formula quadrata. Quello blu è quello per matrici non quadrate con rumore sconosciuto.

`02:17:24`

Potete vedere che sono molto simili e seguono quasi la regola del gomito. Una regola del pollice molto buona che potete usare in statistica è tagliare le cose quando c'è un gomito e infatti stiamo tagliando intorno a questa caratteristica e poi l'ultimo è 90% di energia che è molto molto più conservativo. Probabilmente stiamo mantenendo roba di cui non abbiamo bisogno che comprende un po' di rumore.

`02:18:03`

Ora dobbiamo effettivamente trovare l'indice dove ricostruire la matrice dalla soglia. Quindi, per questo, definiamo una funzione che chiamiamo denoise, alla quale passiamo u, s, e vt, e la soglia. Restituirà la funzione senza il rumore.

`02:18:37`

In particolare, come facciamo? Come prima, manteniamo le prime k colonne di u. Moltiplichiamo questo per nt.diag di s, i primi k valori, per vt, le prime k righe.

`02:19:14`

Ora, quello che resta da fare è trasformare la soglia in k. Come facciamo? Questo è molto simile a quello che abbiamo fatto prima con il where. Infatti, il k sarà uguale a cosa? Quindi, abbiamo nnp.where, s è maggiore della soglia. Quindi, questi sono tutti i valori singolari che vogliamo mantenere.

`02:19:49`

Scusa. Questi sono tutti i valori singolari che vogliamo mantenere. Quindi, questo sarà un array con true e false. E quelli con true sono quelli che vogliamo mantenere. Con il where, estraiamo gli indici, e vogliamo mantenere tutti fino al più grande di questi. Quindi, nnp.max. Più uno. Perché il più uno? Perché lo slicing in Python è chiuso aperto. Quindi se fate lo slice di tutto da zero a dieci, ottenete i primi nove elementi. Scusa, ottenete l'elemento da zero a nove compresi. Se fate lo slice da zero a tre, ottenete zero, uno e due. E vogliamo mantenere questo perché questo sono quelli che vogliamo mantenere. E quindi aggiungiamo uno allo slice.

`02:20:50`

Quindi ora le nostre matrici saranno, chiamiamole XClean, che è denoise USVT cutoff. E chiamiamo X90 denoise con us bt uh cutoff 90. è tutto chiaro okay finale ultimo passo plottiamo tutto quindi il...

`02:21:41`

Passo più importante è capire se abbiamo quello che abbiamo fatto è effettivamente quello che pensavamo e quindi vogliamo plottare l'immagine originale quella denoised e confrontare i due tipi di rumore quindi figure axis bst dot subplots due righe due colonne grande dimensione chiamiamola 10 per 10.

`02:22:14`

AXE D00 imshow qui plottiamo quella pulita, XClean, CHMAP grigio, e possiamo impostare il titolo di 00, set title clean. Similmente, sulla stessa riga, questo lo chiamiamo clean 90, per quello con soglia 90%, e nella riga sotto, quello che facciamo è che plottiamo la differenza rispetto a quello originale.

`02:23:03`

Quindi, riga sotto, plottiamo l'errore per controllare qual è la differenza rispetto a quello senza il rumore, ed è la differenza tra np.apps di x che è quello originale quello a cui abbiamo il rumore e quello pulito e quindi vediamo quanto la nostra procedura di pulizia è vicina all'immagine originale, quindi immagine originale aggiungiamo rumore artificiale denoisiamo l'immagine e vogliamo controllare.

`02:23:35`

Qual è la differenza tra quello originale e il risultato che abbiamo ottenuto con il nostro algoritmo okay tutto sembra buono okay questi sono i risultati eh è un po' troppo largo, okay tutto funziona quindi.

`02:24:09`

Questo è quello con la soglia che ho calcolato con la formula, e quello con il 90% di energia. Quello che dovreste vedere meglio dal vostro laptop è che qui, avete ancora parecchio rumore. Infatti, dall'errore, quello che vedete è che, in generale, questo colore è un po' più verso il bianco. Infatti, qui, l'errore, in media, è un po' più grande. Infatti, abbiamo più rumore. Invece, qui, c'è un taglio molto più conservativo, e quello che avete è che, in generale, il rumore è molto più basso.

`02:24:47`

L'errore è molto piccolo. Tuttavia, ci sono alcune caratteristiche, come questa, questa, dove, dato che siamo stati molto conservativi con il numero di valori singolari che abbiamo mantenuto, in realtà, non siamo in grado di ricostruire completamente l'immagine. Questi sono i due lati. Penserei che se allargate solo un po' questa soglia, e andate qui, questo sarebbe in realtà quello ottimale. Tuttavia, abbiamo un'approssimazione molto buona a questa formula. In realtà, quello che vi ho mostrato è quello con la linea rossa. Se applicate quello con la linea blu, probabilmente otterreste risultati migliori.

`02:25:32`

Facciamo questo cambiando questo. Quindi, sì, direi che è un po' meglio. C'erano alcune differenze più grandi qui prima. Vediamo qui. Okay.

---

## <a name="background-removal"></a>8. Homework: Background Removal

`02:26:04`

Avete qualche domanda? Se no vi mostrerò brevemente l'homework quello che vi ho lasciato voglio solo fare alcuni brevi commenti quindi su questo notebook penso che l'applicazione sia molto bella perché quello che vogliamo fare è che abbiamo un video e che è da una diciamo...

`02:26:34`

Telecamera di sicurezza e vogliamo capire qual è lo sfondo del video e quali sono effettivamente e invece quali sono le parti del video con persone che si muovono e.

`02:27:10`

E la parte divertente qui è che possiamo usare SVD per ottenere risultati piuttosto belli. In particolare, questo è il video. E vedete che ci sono persone che camminano in giro. Cosa dite delle persone che camminano in giro rispetto allo sfondo? Quindi quello che facciamo è, prima di tutto, c'è del pre-processing. Questa è roba Python più avanzata. Di solito, durante l'esame, non vi è richiesto di fare questo, ma penso che sia molto bello se volete provare a capire cosa sta succedendo.

`02:27:40`

Quindi qui, stiamo facendo del pre-processing. In particolare, quello che stiamo facendo è che stiamo trasformando il video in scala di grigi. Stiamo estraendo solo alcuni frame perché tutti i frame sono troppi, e stiamo ridimensionando il video per essere un po' più piccolo. E in particolare, questo è un esempio del risultato. La SVD è computazionalmente molto pesante, quindi abbiamo... Sottocampionato molto il numero di pixel in ogni immagine, abbiamo messo tutto in scala di grigi quindi non dobbiamo trattare ogni canale indipendentemente, e abbiamo estratto solo alcuni frame.

`02:28:16`

Poi, questa è la parte chiave, abbiamo trasformato il video in una matrice. Qui, ogni colonna è un frame del video, e poi impiliamo insieme tutte le colonne una con le altre. E questo è molto simile a quello che faremo nel prossimo laboratorio con la PCA. E questa è un'idea molto bella perché in questo modo, abbiamo una matrice dove lo sfondo sono le linee costanti nel video.

`02:28:50`

E abbiamo visto che SVD è molto bravo a trovare quali sono le caratteristiche geometriche dell'immagine e in particolare può capire molto facilmente con questo qual è lo sfondo perché sono le linee orizzontali costanti questo sarà la prima, eigenimage che abbiamo estratto prima okay perché questa è una costante possiamo ricostruire la prima componente di ricostruzione di questa immagine è la prima linea una immagine che è la linea costante dietro invece le persone che si muovono sono queste ombre che si muovono qua e là e questo.

`02:29:25`

Saranno le caratteristiche finali che vengono estratte dopo dalla SVD quindi quello che facciamo è applichiamo questo e una volta che abbiamo applicato la SVD ricostruiamo solo con il primo, valore singolare qui e valori simili uguale a uno questo è lo sfondo, E dopo, se fate un po' di ginnastica con le shape e tutto, ottenete questo.

`02:29:58`

Questi sono alcuni frame del video. Avete il video originale, lo sfondo estratto, e le persone. Quindi questo è il concetto versus l'immagine. E questa è la differenza rispetto a quello originale e lo sfondo estratto. E qui avete le persone estratte. Ora, l'esercizio per voi a casa è cambiare questo e usare, invece della SVD, quella randomizzata.

`02:30:29`

Perché qui stiamo usando solo un valore singolare. E abbiamo visto che calcolare la SVD... Non è utile, perché stiamo buttando via tutto quello che non è il valore singolare, e quella randomizzata è molto, molto buona in questo caso, perché possiamo mettere k uguale a, tipo, 20, 10, il primo valore singolare sarà davvero, davvero buono, ed è tutto quello di cui abbiamo bisogno, e vedrete che facendo questo, il vostro algoritmo sarà ordini di grandezza più veloce, e questo è il vostro homework.

### Conclusione

`02:30:59`

Quindi, questo è tutto per oggi, e prima che andiate, lasciatemi solo farvi una domanda. Il lab di oggi, è stato troppo veloce, o è stato okay, riguardo alla velocità, più o meno? È stato okay? Okay, perfetto. Quindi, per qualsiasi cosa, scrivetemi un'email, e buon weekend.

_[Fine del laboratorio - 02:32:06 a 02:40:18: Conversazioni informali con studenti]_

---

## 12. Codice Completo: Notebook Soluzioni {#codice-completo}

### 12.1 Image Compression - Soluzione Completa

**Setup e Caricamento Immagine:**

```python
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

# Carica immagine (TarantulaNebula.jpg o mondrian.jpg)
image_path = "./TarantulaNebula.jpg"
A = imread(image_path)

# Opzionale: ruota immagine
# A = rotate(A, 20, reshape=False)

# Visualizza originale (RGB)
plt.imshow(A)
plt.axis("off")
plt.show()
```

**Conversione Grayscale e SVD:**

```python
# Conversione grayscale (media dei canali RGB)
X = np.mean(A, axis=2)

# Visualizza grayscale
img = plt.imshow(X, cmap="gray")
plt.axis("off")
plt.show()

print(f"Dimensioni immagine: {X.shape}")

# SVD completa (thin SVD)
U, s, VT = np.linalg.svd(X, full_matrices=False)
print(f"U: {U.shape}, s: {s.shape}, VT: {VT.shape}")
```

**Analisi Valori Singolari (3 Plot):**

```python
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# 1. Valori singolari (scala log)
axes[0].semilogy(s, "o-")
axes[0].set_title("Singular values")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Singular value (log scale)")

# 2. Frazione cumulativa: ∑σᵢ / ∑σⱼ
axes[1].plot(np.cumsum(s) / np.sum(s), "o-")
axes[1].set_title("Cumulative fraction of singular values")
axes[1].set_xlabel("Index")
axes[1].set_ylabel("Cumulative fraction")

# 3. Varianza spiegata: √(∑σᵢ² / ∑σⱼ²)
axes[2].plot(np.sqrt(np.cumsum(s**2) / np.sum(s**2)), "o-")
axes[2].set_title("Explained variance")
axes[2].set_xlabel("Index")
axes[2].set_ylabel("Explained variance")

plt.tight_layout()
plt.show()
```

**Ricostruzione Rank-k (k = 1, 2, 5, 10, 15, 50):**

```python
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axs = axs.reshape((-1,))
idxs = [1, 2, 5, 10, 15, 50]

for i in range(len(idxs)):
    k = idxs[i]
    # Ricostruzione: A_k = U[:,:k] @ Σ[:k,:k] @ VT[:k,:]
    A_k = np.matmul(U[:, :k], np.matmul(np.diag(s[:k]), VT[:k, :]))
    
    axs[i].imshow(A_k, cmap="gray")
    axs[i].set_title(f"k = {k}")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
```

**Eigenimages (Matrici Rank-1):**

```python
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axs = axs.reshape((-1,))
idxs = [1, 2, 3, 4, 5, 6]

for i, k in enumerate(idxs):
    # k-esima matrice rank-1: σₖ·uₖ·vₖᵀ
    # Senza σₖ: uₖ·vₖᵀ (eigenimage pura)
    ukvk = np.outer(U[:, k - 1], VT[k - 1, :])
    
    axs[i].imshow(ukvk, cmap="gray")
    axs[i].set_title(f"{k}-th rank-1 matrix")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
```

**Interpretazione:** 
- Le prime eigenimages catturano pattern **globali** (bordi, contrasti principali)
- Le successive catturano **dettagli fini** (texture, piccoli oggetti)

---

### 12.2 Randomized SVD - Implementazione Completa

**Algoritmo rSVD (Versione Tutor):**

```python
def randomized_SVD(A, k):
    """
    Randomized SVD per approssimazione rank-k.
    
    Parametri:
    - A: matrice m×n
    - k: rank desiderato
    
    Output:
    - U: m×k (colonne ortonormali)
    - sy: k valori singolari approssimati
    - VTy: k×n (righe ortonormali)
    
    Complessità: O(mnk) invece di O(mn²)
    """
    m, n = A.shape
    
    # 1. Proiezione random: cattura range di A
    P = np.random.randn(n, k)
    Z = A @ P
    
    # 2. Ortonormalizza con QR
    Q, _ = np.linalg.qr(Z)
    
    # 3. Proietta A su sottospazio Q
    Y = Q.T @ A
    
    # 4. SVD ridotta su Y (k×n, molto più piccola!)
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    
    # 5. Ricostruisci U nello spazio originale
    U = Q @ Uy
    
    return U, sy, VTy
```

**Uso e Confronto con SVD Full:**

```python
k = 100

# Randomized SVD
U_rand, s_rand, VT_rand = randomized_SVD(X, k)

# Confronto valori singolari (3 plot)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 1. Valori singolari (log-log)
axs[0].loglog(s, "o-", label="Full SVD")
axs[0].loglog(s_rand, "+-", label="Randomized SVD")
axs[0].set_title("Singular values")
axs[0].legend()

# 2. Somma cumulativa
axs[1].semilogx(np.cumsum(s), "o-", label="Full SVD")
axs[1].semilogx(np.cumsum(s_rand), "+-", label="Randomized SVD")
axs[1].set_title("Cumulative fraction")
axs[1].legend()

# 3. Varianza spiegata
axs[2].semilogx(np.cumsum(s**2), "o-", label="Full SVD")
axs[2].semilogx(np.cumsum(s_rand**2), "+-", label="Randomized SVD")
axs[2].set_title("Explained variance")
axs[2].legend()

plt.tight_layout()
plt.show()
```

**Confronto Visuale (Originale vs Full SVD vs rSVD):**

```python
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Originale
axs[0].imshow(X, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

# Full SVD (rank-k)
A_svd = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]
axs[1].imshow(A_svd, cmap="gray")
axs[1].set_title(f"Full SVD (k={k})")
axs[1].axis("off")

# Randomized SVD (rank-k)
A_rsvd = U_rand @ np.diag(s_rand) @ VT_rand
axs[2].imshow(A_rsvd, cmap="gray")
axs[2].set_title(f"Randomized SVD (k={k})")
axs[2].axis("off")

plt.tight_layout()
plt.show()
```

**Quando Usare rSVD:**
- ✅ Matrici **grandi** (m, n > 1000)
- ✅ Rank **piccolo** (k << min(m,n))
- ✅ Precisione moderata sufficiente (~1-3% errore)
- ❌ NO per piccole matrici (overhead superiore a beneficio)

---

### 12.3 Denoising - Tre Metodi a Confronto

**Setup con Rumore Artificiale:**

```python
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Carica immagine (mondrian.jpg o TarantulaNebula.jpg)
image_path = "./mondrian.jpg"
A = imread(image_path)

# Conversione grayscale NORMALIZZATA [0,1]
X = np.mean(A, axis=2) / A.max()

# Aggiungi rumore gaussiano
gamma = 0.1  # Deviazione standard rumore
np.random.seed(42)  # Riproducibilità
X_noisy = X + gamma * np.random.randn(*X.shape)
X_noisy = np.clip(X_noisy, 0, 1)  # Clip a [0,1]

# Visualizza originale vs noisy
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(X, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(X_noisy, cmap="gray")
axs[1].set_title("Noisy Image")
axs[1].axis("off")

plt.show()
```

**SVD su Immagine Noisy:**

```python
U, s, VT = np.linalg.svd(X_noisy, full_matrices=False)
m, n = X.shape
```

**Calcolo Tre Soglie:**

```python
# METODO 1: Hard Threshold (matrici quadrate, rumore noto)
cutoff = (4 / np.sqrt(3)) * np.sqrt(n) * gamma
print(f"Cutoff (square formula): {cutoff:.4f}")

# METODO 2: 90% Energia Cumulativa (baseline conservativa)
cumsum_threshold = 0.90
cdS = np.cumsum(s) / np.sum(s)
r90 = np.min(np.where(cdS > cumsum_threshold)[0])
cutoff90 = s[r90]
print(f"Cutoff (90% energy): {cutoff90:.4f} (rank {r90})")

# METODO 3: Rumore Sconosciuto (formula β-dipendente)
sigma_med = np.median(s)
beta = m / n
omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
cutoff_unknown = omega * sigma_med
print(f"Cutoff (unknown noise): {cutoff_unknown:.4f}")
```

**Visualizzazione Soglie su Spettro:**

```python
plt.figure(figsize=(10, 6))
plt.semilogy(s, "k.-", label="Singular values")
plt.axhline(cutoff, color="b", label="Hard threshold (known noise)")
plt.axhline(cutoff90, color="g", label="90% cumulative energy")
plt.axhline(cutoff_unknown, color="r", linestyle="--", 
            label="Hard threshold (unknown noise)")
plt.xlabel("Index")
plt.ylabel("Singular value (log scale)")
plt.title("Threshold Comparison on Noisy Image Spectrum")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Funzione Denoising (Generica):**

```python
def denoise(U, s, VT, threshold):
    """
    Denoising via hard thresholding SVD.
    
    Parametri:
    - U, s, VT: fattori SVD
    - threshold: soglia τ
    
    Output:
    - Immagine denoised (ricostruzione rank-r con σᵢ > τ)
    """
    # Trova ultimo indice con σ > τ
    r = np.max(np.where(s > threshold)[0])
    
    # Ricostruisci con primi r+1 componenti
    return U[:, :(r+1)] @ np.diag(s[:(r+1)]) @ VT[:(r+1), :]

# Applica tre metodi
Xclean_hard = denoise(U, s, VT, cutoff)
Xclean_90 = denoise(U, s, VT, cutoff90)
Xclean_unknown = denoise(U, s, VT, cutoff_unknown)
```

**Confronto Visuale (Denoised + Errori):**

```python
# Calcola errori assoluti rispetto originale
err_hard = np.abs(X - Xclean_hard)
err_90 = np.abs(X - Xclean_90)

# Norme Frobenius
err_norm_hard = np.linalg.norm(err_hard, ord="fro")
err_norm_90 = np.linalg.norm(err_90, ord="fro")

# Plot 2×2: denoised + errori
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Riga 1: immagini denoised
axs[0, 0].imshow(Xclean_hard, cmap="gray")
axs[0, 0].set_title("Denoised: Hard Threshold")
axs[0, 0].axis("off")

axs[0, 1].imshow(Xclean_90, cmap="gray")
axs[0, 1].set_title("Denoised: 90% Energy")
axs[0, 1].axis("off")

# Riga 2: mappe errori (stessa scala colore)
max_err = max(err_hard.max(), err_90.max())

axs[1, 0].imshow(err_hard, cmap="hot", vmin=0, vmax=max_err)
axs[1, 0].set_title(f"Error Hard (‖·‖_F = {err_norm_hard:.2e})")
axs[1, 0].axis("off")

axs[1, 1].imshow(err_90, cmap="hot", vmin=0, vmax=max_err)
axs[1, 1].set_title(f"Error 90% (‖·‖_F = {err_norm_90:.2e})")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()
```

**Interpretazione Risultati:**

| Metodo | Rank | Rumore Residuo | Perdita Dettagli |
|--------|------|----------------|------------------|
| **Hard Threshold** | ~5-10 | Medio | Basso |
| **90% Energy** | ~400 | Alto (troppo conservativo) | Minimo |
| **Unknown Noise** | ~8-15 | Basso | Basso-Medio |

**Trade-off:**
- **Hard threshold** (noto): Equilibrio ottimale rumore/dettagli
- **90% energy**: Mantiene troppi componenti → include rumore
- **Unknown noise**: Simile a hard (buona approssimazione!)

---

### 12.4 Background Removal - Homework

**Concetto:** Video = matrice (pixel × frame). SVD separa:
- **Sfondo** (costante) → 1° valore singolare
- **Foreground** (movimento) → valori singolari successivi

**Preprocessing Video:**

```python
import cv2
import numpy as np

def preprocess_video(video_path, subsample=10, resize_factor=0.3):
    """
    Preprocessing:
    1. Grayscale
    2. Subsampling (ogni N frame)
    3. Resize (riduzione risoluzione)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % subsample == 0:
            # Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize
            h, w = gray.shape
            gray_small = cv2.resize(gray, 
                                    (int(w*resize_factor), int(h*resize_factor)))
            frames.append(gray_small)
        
        idx += 1
    
    cap.release()
    return np.array(frames)

# Carica e preprocessa
frames = preprocess_video("security_camera.mp4", subsample=5, resize_factor=0.25)
print(f"Frames: {frames.shape}")  # (n_frames, height, width)
```

**Trasformazione Video → Matrice:**

```python
# Flatten ogni frame in colonna
n_frames, h, w = frames.shape
X = frames.reshape(n_frames, h*w).T  # (pixels × frames)
print(f"Matrix X: {X.shape}")  # (height*width, n_frames)
```

**Estrazione Sfondo con SVD (k=1):**

```python
# SVD
U, s, VT = np.linalg.svd(X, full_matrices=False)

# Ricostruisci solo con 1° valore singolare (sfondo)
background_flat = U[:, :1] @ np.diag(s[:1]) @ VT[:1, :]

# Reshape a video
background_frames = background_flat.T.reshape(n_frames, h, w)

# Foreground = Originale - Sfondo
foreground_frames = frames - background_frames
```

**Visualizzazione (Frame Esempio):**

```python
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

idx = 50  # Frame esempio

axs[0].imshow(frames[idx], cmap="gray")
axs[0].set_title("Original Frame")
axs[0].axis("off")

axs[1].imshow(background_frames[idx], cmap="gray")
axs[1].set_title("Background (k=1)")
axs[1].axis("off")

axs[2].imshow(foreground_frames[idx], cmap="gray", vmin=0)
axs[2].set_title("Foreground (People)")
axs[2].axis("off")

plt.tight_layout()
plt.show()
```

**HOMEWORK:** Sostituire SVD con **Randomized SVD**:

```python
# Usa randomized_SVD(X, k=20) invece di np.linalg.svd(X)
# Primo valore singolare sarà comunque accuratissimo
# Speedup: 10-50× per video grandi!

U_rand, s_rand, VT_rand = randomized_SVD(X, k=20)
background_rsvd = U_rand[:, :1] @ np.diag(s_rand[:1]) @ VT_rand[:1, :]
# ... resto identico
```

---

## 13. Materiali e Riferimenti {#materiali-lab}

### Notebook Lab01

1. **`1_basics_svd_start.ipynb`**: 
   - SVD con NumPy/SciPy
   - Full vs Thin SVD
   - Confronto performance (loop vs vettorizzazione)

2. **`2_image_compression_start.ipynb`**:
   - Compressione immagini
   - Eigenimages (matrici rank-1)
   - Randomized SVD

3. **`3_noise_removal_start.ipynb`**:
   - Hard thresholding
   - Confronto metodi (known/unknown noise, 90% energy)

4. **`4_background_removal_start.ipynb`**:
   - Separazione sfondo/foreground
   - SVD su video (homework)

**Immagini Test:**
- `TarantulaNebula.jpg` (alta entropia, molti dettagli)
- `mondrian.jpg` (bassa entropia, geometria semplice)

### Codice Chiave da Ricordare

**Thin SVD (raccomandato):**
```python
U, s, VT = np.linalg.svd(A, full_matrices=False)
# U: m×q, s: q, VT: q×n (q = min(m,n))
```

**Ricostruzione rank-k (vettorizzata):**
```python
# VELOCE: broadcasting NumPy
A_k = (U[:, :k] * s[:k]) @ VT[:k, :]

# MEDIO: operatore @
A_k = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]

# LENTO: loop Python (evitare!)
for i in range(k):
    A_k += s[i] * np.outer(U[:, i], VT[i, :])
```

**Randomized SVD (implementazione completa):**
```python
def randomized_SVD(A, k):
    _, n = A.shape
    P = np.random.randn(n, k)
    Z = A @ P
    Q, _ = np.linalg.qr(Z)
    Y = Q.T @ A
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    return U, sy, VTy
```

**Denoising (hard threshold):**
```python
def denoise(U, s, VT, tau):
    r = np.max(np.where(s > tau)[0])
    return U[:, :(r+1)] @ np.diag(s[:(r+1)]) @ VT[:(r+1), :]
```

---

## 14. Checklist Completa Lab 1 {#checklist-lab}

### Setup e Fondamentali

- [ ] **Colab setup**: Upload notebook, connetti runtime GPU (se disponibile)
- [ ] **NumPy random**: `np.random.seed(X)` per riproducibilità
- [ ] **SVD syntax**: `U, s, VT = np.linalg.svd(A, full_matrices=False)`
- [ ] **Full vs Thin**: Thin (economico) preferito per rank-k

### Performance Python

- [ ] **Evitare loop**: Usare operazioni matriciali NumPy
- [ ] **Broadcasting**: `(U * s) @ VT` 2× più veloce di `U @ diag(s) @ VT`
- [ ] **Profiling**: `time.time()` o `%timeit` (Jupyter) per benchmark

### Image Compression

- [ ] **Conversione grayscale**: `X = np.mean(A, axis=2)`
- [ ] **Analisi spettro**: Plot σ (log), cumsum, varianza
- [ ] **Cutoff scelta**: Regola del gomito o % varianza
- [ ] **Ricostruzione rank-k**: Visualizzare k = [1, 5, 10, 50, 100]
- [ ] **Eigenimages**: `np.outer(U[:, k], VT[k, :])` per k-esima

### Randomized SVD

- [ ] **Implementazione**: 5 step (Omega → QR → project → SVD → U)
- [ ] **Parametri**: k (rank), seed per riproducibilità
- [ ] **Confronto spettro**: Plot σ full vs σ_rand (primi k)
- [ ] **Accuratezza**: Verificare differenza trascurabile per k << n

### Denoising

- [ ] **Aggiunta rumore**: `X_noisy = X + gamma * randn(*shape)`
- [ ] **Tre soglie**: Hard (known), unknown noise (β-formula), 90% energy
- [ ] **Funzione denoise**: Threshold, trova rank r, ricostruisci
- [ ] **Confronto visuale**: Plot denoised + error map
- [ ] **Trade-off**: Rumore vs perdita dettagli

### Background Removal (Homework)

- [ ] **Video → matrice**: Flatten frame in colonne
- [ ] **SVD k=1**: Estrai sfondo (componente costante)
- [ ] **Foreground**: Originale - sfondo
- [ ] **Randomized SVD**: Sostituire con rSVD(k=20) per speedup

---

## 15. Esercizi Avanzati {#esercizi-avanzati}

### Esercizio 1: Compressione Ottimale

Trova **k minimo** per errore relativo < 5%:

```python
def find_optimal_k(U, s, VT, X_orig, target_error=0.05):
    """
    Ricerca binaria per k ottimale.
    """
    for k in range(1, len(s)):
        X_k = (U[:, :k] * s[:k]) @ VT[:k, :]
        rel_error = np.linalg.norm(X_orig - X_k, 'fro') / np.linalg.norm(X_orig, 'fro')
        if rel_error < target_error:
            return k, rel_error
    return len(s), 0.0

k_opt, err_opt = find_optimal_k(U, s, VT, X)
print(f"Optimal k: {k_opt} (error: {err_opt:.2%})")
```

### Esercizio 2: Denoising Adattivo

Usa **cross-validation** per trovare γ ottimale:

```python
gammas = [0.05, 0.1, 0.15, 0.2]
errors = []

for gamma in gammas:
    X_noisy = X + gamma * np.random.randn(*X.shape)
    U, s, VT = np.linalg.svd(X_noisy, full_matrices=False)
    
    cutoff = (4/np.sqrt(3)) * np.sqrt(n) * gamma
    X_clean = denoise(U, s, VT, cutoff)
    
    err = np.linalg.norm(X - X_clean, 'fro')
    errors.append(err)

best_gamma = gammas[np.argmin(errors)]
print(f"Best gamma: {best_gamma}")
```

### Esercizio 3: rSVD con Power Iterations

Migliora accuratezza rSVD con **power iterations**:

```python
def randomized_SVD_power(A, k, q=2):
    """
    rSVD con power iterations.
    
    q iterazioni: (AA^T)^q Ω per amplificare gap spettrale.
    """
    _, n = A.shape
    P = np.random.randn(n, k)
    
    # Power iterations
    Z = A @ P
    for _ in range(q):
        Z = A @ (A.T @ Z)
    
    Q, _ = np.linalg.qr(Z)
    Y = Q.T @ A
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    
    return U, sy, VTy

# Test: confronta q=0 vs q=3
U1, s1, VT1 = randomized_SVD(X, 50)
U2, s2, VT2 = randomized_SVD_power(X, 50, q=3)

plt.semilogy(s[:50], 'k-', label='Full SVD')
plt.semilogy(s1, 'b--', label='rSVD (q=0)')
plt.semilogy(s2, 'r:', label='rSVD (q=3)')
plt.legend()
plt.title("Effect of Power Iterations on rSVD Accuracy")
plt.show()
```

### Esercizio 4: Video Denoising

Applica denoising **frame-by-frame** a video:

```python
def denoise_video(frames, gamma):
    """
    Denoise ogni frame indipendentemente.
    """
    denoised_frames = []
    
    for frame in frames:
        U, s, VT = np.linalg.svd(frame, full_matrices=False)
        m, n = frame.shape
        cutoff = (4/np.sqrt(3)) * np.sqrt(n) * gamma
        
        frame_clean = denoise(U, s, VT, cutoff)
        denoised_frames.append(frame_clean)
    
    return np.array(denoised_frames)

# Applica
frames_clean = denoise_video(frames, gamma=0.05)
```

**Confronto:** Frame-by-frame vs SVD globale (come homework)?

---

## 🎯 Punti Chiave Finali Lab 1

1. **SVD = Swiss Army Knife**:
   - Compressione: mantenere primi k
   - Denoising: threshold valori singolari
   - Separazione: sfondo vs foreground

2. **Vettorizzazione è CRITICA**:
   - Broadcasting NumPy: 60× più veloce di loop
   - Mai usare `for` su array grandi

3. **Randomized SVD**:
   - 10-100× speedup per matrici grandi
   - Accuratissimo per k piccolo
   - **Homework essenziale**: sostituire in background removal

4. **Denoising ≠ Compressione**:
   - Threshold intelligente (teoria informazione)
   - 90% energia TROPPO conservativo
   - Formula hard: $(4/\sqrt{3})\sqrt{n}\gamma$

5. **Trade-off Universale**:
   - ↑ k: più dettagli, più rumore
   - ↓ k: meno rumore, meno dettagli
   - Ottimale: dipende da applicazione!

---

**Fine Lezione 9 - Lab 1: SVD e Applicazioni Pratiche**

