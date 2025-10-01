## Decomposizione ai Valori Singolari (SVD) — Forme, Proprietà, Geometria, Interpretazioni Data/ML, Compressione, Calcolo e Teorema Chiave
### Definizioni Centrali e Notazione
- Qualsiasi matrice A ∈ R^(m×n) ammette A = U Σ V^T.
  - U ∈ R^(m×m) e V ∈ R^(n×n) sono ortogonali.
  - Σ ∈ R^(m×n) è (semi-)diagonale con valori singolari non negativi σ1 ≥ σ2 ≥ … ≥ σk ≥ 0.
- Rango(A) = r uguale al numero di valori singolari non nulli.
- Al massimo min(m, n) valori singolari possono essere non nulli; se A è rango-deficiente, solo i primi r sono non nulli.
### Varianti SVD e Dimensioni
- SVD Completa:
  - U: m×m, Σ: m×n, V: n×n (mantiene tutte le basi ortogonali, incluse quelle moltiplicate per zeri in Σ).
- SVD Economica (Sottile/Compatta):
  - U_econ: m×n (prime n colonne di U), Σ_econ: n×n, V: n×n.
  - Consapevole della forma (specialmente quando m > n); omette componenti base che moltiplicano zeri strutturali in Σ.
- SVD Ridotta (Consapevole del Rango):
  - Assume rango(A) = r < min(m, n).
  - U_r: m×r, Σ_r: r×r, V_r^T: r×n.
  - Mantiene solo direzioni corrispondenti a valori singolari non nulli; U_r span Col(A), V_r span Row(A).
- SVD Troncata (Approssimazione):
  - Scegliere p < n (tipicamente p ≤ r se rango-deficiente).
  - U_p: m×p, Σ_p: p×p, V_p^T: p×n.
  - Approssima A usando le prime p componenti; comunemente calcolata tramite SVD completa/economica e poi troncata.
### Intuizioni su Rango e Struttura
- Per m > n, le ultime (m−n) righe di Σ sono zeri; le colonne finali di U non influenzano A.
- Spazi colonna/riga:
  - Le colonne di U span Col(A).
  - Le colonne di V span Col(A^T) = Row(A).
  - SVD ridotta mantiene basi allineate con valori singolari non nulli.
### Relazioni con Matrici tipo Covarianza
- A^T A = V (Σ^T Σ) V^T:
  - Σ^T Σ è diagonale con elementi σ_i^2; questi sono autovalori di A^T A.
  - I vettori singolari destri (colonne di V) sono autovettori; σ_i^2 quantificano la varianza lungo queste direzioni.
- A A^T = U (Σ Σ^T) U^T:
  - Σ Σ^T è diagonale con elementi σ_i^2; i vettori singolari sinistri (colonne di U) sono autovettori.
- Implicazione pratica:
  - σ_i^2 servono come autovalori sia di A^T A che di A A^T, informando analisi di varianza, minimi quadrati e conditioning.
### Interpretazione Geometrica di A = U Σ V^T
- Trasforma x tramite A via:
  1) V^T: rotazione/riflessione (ortogonale; preserva lunghezze/angoli).
  2) Σ: scaling anisotropico lungo direzioni singolari (cerchio → ellisse scalata per σ_i).
  3) U: rotazione/riflessione all'orientamento finale.
- Effetto netto: trasformazione ortogonale → scaling lungo assi principali → trasformazione ortogonale.
### Interpretazione Data Science / Machine Learning
- SVD fornisce coordinate caratteristiche ortogonali:
  - Massimizza varianza (σ_i^2 grandi indicano direzioni dominanti).
  - Decorrelazione caratteristiche (minimizza covarianza).
- I vettori singolari destri (V) definiscono direzioni principali nello spazio caratteristiche; i valori singolari indicano importanza.
- SVD troncata esegue riduzione dimensionalità/estrazione caratteristiche per approssimazione efficiente e compressione.
### Parametrizzazione: 2D vs 3D (e superiori)
- 2D (2×2):
  - A ha 4 parametri; SVD usa θ (rotazione U), σ1, σ2 (scale), φ (rotazione V^T) → totale 4.
- 3D (3×3):
  - A ha 9 parametri; U e V^T sono rotazioni (parametrizzate da asse + angolo o angoli di Eulero), Σ ha tre scale (σ1, σ2, σ3). Il conteggio si allinea con gradi di libertà; generalizza a dimensioni superiori.
### SVD Troncata per Compressione Immagini
- Setup:
  - Immagine a colori (~600×400) memorizzata come tre matrici 600×400 (RGB).
  - Rango max ≈ min(600, 400) = 400 (concettualmente).
- Approssimazione rango-k:
  - Usare prime k colonne di U, primi k valori singolari, prime k righe di V^T.
- Impatto visivo e compressione:
  - k ≈ 300: differenza visibile minima per molte scene.
  - k ≈ 200: qualità molto buona.
  - k ≈ 100: piccole differenze; coda omessa spesso bassa energia.
  - k ≈ 10: sfocatura evidente; compressione forte.
- Rapporti di compressione:
  - k moderato può produrre ~4× compressione; immagini geometriche allineate agli assi possono raggiungere ~7.5× con alta qualità.
### Comportamento Spettro Valori Singolari e Sensibilità Orientamento
- Distribuzione energia:
  - I valori singolari principali portano la maggior parte dell'energia; il decadimento della coda dipende dal contenuto.
- Dipendenza dall'immagine:
  - Pattern geometrici allineati agli assi → decadimento rapido → alta comprimibilità.
  - Texture casuali → decadimento lento, quasi lineare → comprimibilità peggiore.
- Effetti orientamento:
  - Quadrato allineato agli assi: un σ dominante; estremamente comprimibile.
  - Quadrato ruotato: spettro si diffonde; più σ necessari per dettagli → comprimibilità minore.
- Intuizione ML:
  - Le rotazioni alterano distribuzioni valori singolari; l'augmentation dati basata su rotazione introduce efficacemente nuove varianti per training.
### Costruzione Numerica via A^T A e A A^T
- Esempio 1: 2×2 rango pieno
  - Calcolare X = A^T A e Y = A A^T (entrambi SPD).
  - Autovalori di X e Y combaciano; valori singolari σ_i = √λ_i.
  - U da autovettori di Y; V da autovettori di X o via V = A^T U Σ^−1.
- Esempio 2: Rango-1
  - A^T A ha autovalori {λ1, 0}; σ1 = √λ1, σ2 = 0.
  - v1 da autovettore di λ1; u1 = (1/σ1) A v1.
  - A = u1 σ1 v1^T esattamente (SVD ridotta sufficiente).
### Proprietà e Decomposizioni Correlate
- A ortogonale (quadrata):
  - Se A^T A = I, allora Σ = I e U = A (a meno di convenzione), V = I; A è una trasformazione ortogonale pura.
- A simmetrica e Decomposizione Polare:
  - Usando SVD, A può essere scritta A = Q S dove Q è ortogonale e S è simmetrica PSD.
  - Interpreta deformazione come rotazione + stretch; ampiamente usata in meccanica.
- Limite norma spettrale:
  - ||A b|| ≤ σ1 ||b|| per ogni b.
  - Per qualsiasi autoppia (λ, x): |λ| ≤ σ1.
  - Il raggio spettrale di A è limitato superiormente dal valore singolare più grande.
### Strategia Computazionale e Metodi Randomizzati
- Scelta basata su dimensionalità:
  - Per A ∈ R^{n×d}: se d ≫ n, calcolare A A^T (n×n); se n ≫ d, calcolare A^T A (d×d).
- Recupero componenti:
  - Da A A^T = U Σ^2 U^T, ottenere V via V = A^T U Σ^−1.
  - Da A^T A = V Σ^2 V^T, ottenere U via U = A V Σ^−1.
- Fattibilità pratica:
  - SVD esatta può essere costosa per dati su larga scala.
  - SVD randomizzata fornisce approssimazioni a basso rango efficienti e accurate ed è ampiamente usata nell'industria.
### Teorema di Eckart–Young: Migliore Approssimazione Basso Rango
- Espansione rango-1:
  - A = Σ_i σ_i u_i v_i^T.
- Troncamento:
  - A_k = Σ_{i=1}^k σ_i u_i v_i^T è rango-k.
- Ottimalità:
  - A_k è la migliore approssimazione rango-k di A sia in norma spettrale che di Frobenius.
- Implicazione compressione:
  - Mantenere le prime k componenti singolari produce la ricostruzione rango-k dimostrabilmente ottimale.
### Prossimi Accordi ed Elementi d'Azione
- [ ] Usare SVD economica in MATLAB/Python quando m > n per evitare componenti moltiplicate per zero.
- [ ] Per dati rango-deficienti, calcolare SVD ridotta (m×r, r×r, r×n) per mantenere direzioni significative.
- [ ] Per approssimazione/compressione, eseguire SVD troncata con i primi p valori/vettori singolari (p ≤ r).
- [ ] Sfruttare A^T A = V Σ^2 V^T e A A^T = U Σ^2 U^T per analisi varianza e minimi quadrati.
- [ ] Dimostrare SVD randomizzata su immagini per quantificare risparmi computazionali e qualità ricostruzione.
- [ ] Presentare dimostrazione usando |λ| ≤ σ1 per risultati correlati.
- [ ] Esplorare parametrizzazione rotazione 3D in SVD per abbinare gradi di libertà 3×3.
- [ ] Confrontare effetti orientamento immagini e collegare a pratiche data augmentation.