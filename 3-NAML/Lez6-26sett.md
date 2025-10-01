## Strumenti di Algebra Lineare per il Corso: Panoramica e Riassunto Integrato
- Focus su matrici quadrate e rettangolari; decomposizioni chiave e proprietà necessarie per argomenti successivi (minimi quadrati, ottimizzazione, machine learning).
- Sequenza di concetti:
  - Decomposizione agli autovalori (EVD) per matrici quadrate; funzioni di matrici tramite autoppie; trasformazioni di similarità.
  - Fattorizzazione QR (completa ed economica), matrici ortogonali, procedura di Gram–Schmidt.
  - Teorema spettrale per matrici simmetriche; ortogonalità e realtà di autovettori/autovalori.
  - Matrici Simmetriche Definite Positive (SPD): caratterizzazioni, forma energetica e fattorizzazione di Cholesky.
  - Decomposizione ai Valori Singolari (SVD): completa, ridotta/economica, basata sul rango, troncata; interpretazione per dati, benefici di memoria e rilevanza ML.
---
## Decomposizione agli Autovalori (EVD) di Matrici Quadrate
- Definizione:
  - Per A ∈ R^{n×n}, le autoppie (λi, xi) soddisfano A xi = λi xi.
  - Forma matriciale: A X = X Λ con X = [x1 … xn], Λ = diag(λ1,…,λn). Se X è invertibile: A = X Λ X^{-1}.
- Interpretazione geometrica:
  - A agisce come trasformazione lineare: gli autovettori mantengono la direzione; scalano secondo gli autovalori.
- Potenze e funzioni di matrici:
  - A^k xi = λi^k xi per k ∈ N.
  - Funzioni analitiche f(A): f(A) xi = f(λi) xi (es. e^A xi = e^{λi} xi). Si collega alle soluzioni di ODE e al metodo delle potenze.
---
## Matrici Simili: Invarianza e Trasformazione degli Autovettori
- Similarità:
  - B = M^{-1} A M con M invertibile ⇒ A e B sono simili.
- Relazione autovalori:
  - Se B y = λ y, allora A (M y) = λ (M y). Gli autovalori sono identici; gli autovettori si mappano tramite w = M y.
- Interpretazione:
  - Le matrici simili rappresentano la stessa mappa lineare sotto un cambio di base M.
---
## SVD e Algebra Lineare: Fattorizzazione QR, Autovalori, Teorema Spettrale
### Contesto di Fattorizzazione e Sviluppo Storico
- Per una matrice A di rango pieno, la fattorizzazione LU potrebbe non esistere a causa di pivot nulli e potenziali fallimenti nell'eliminazione di Gauss; scambi di righe (pivoting/permutazioni) possono risolvere questo problema.
- Al contrario, la fattorizzazione QR esiste sempre: A = QR, dove Q è ortogonale (o ortonormale) e R è triangolare superiore. Questo evita le limitazioni dell'eliminazione di Gauss.
- Lo sviluppo storico abbraccia le basi teoriche (Householder, Givens, Gram-Schmidt) e gli algoritmi computazionali (innovazioni guidate dalla stabilità).
### Definizioni e Framework Matematico
- Matrice ortogonale: Q^T Q = I (la trasposta è uguale all'inversa), preserva lunghezze e angoli; ||Qx|| = ||Qx|| per ogni x.
- Fattorizzazione QR: A (m×n) = Q (m×m) R (m×n), dove Q è ortogonale e R è triangolare superiore.
- QR economica: Per m > n, considerare A (m×n) = Q₁ (m×n) R₁ (n×n), dove Q₁ contiene solo le prime n colonne di Q e R₁ è n×n triangolare superiore; più efficiente in memoria per matrici "alte".
### Algoritmi Computazionali per QR
- Riflessioni di Householder: Approccio sistematico usando matrici di riflessione per azzerare le voci colonna per colonna; numericamente stabile.
- Rotazioni di Givens: Approccio mirato usando matrici di rotazione 2×2 per eliminare voci specifiche; utile per matrici sparse o quando è necessaria un'eliminazione selettiva di voci.
- Processo di Gram-Schmidt: Ortonormalizzazione sequenziale; concettualmente semplice ma meno numericamente stabile senza modifiche (es. Gram-Schmidt modificato).
### Applicazioni e Vantaggi di QR
- Risoluzione di sistemi lineari: Ax = b diventa QRx = b, poi Rx = Q^T b. Poiché R è triangolare superiore, la back-substitution risolve efficientemente.
- Stabilità numerica: Generalmente più stabile della fattorizzazione LU, specialmente per matrici mal condizionate.
- Minimi quadrati: Per sistemi sovradeterminati (m > n), QR fornisce le basi per risolvere min ||Ax - b||₂.
### Introduzione alla Teoria degli Autovalori
- Problema degli autovalori: Ax = λx, dove x (autovettore) è un vettore non nullo e λ (autovalore) è uno scalare.
- Polinomio caratteristico: det(A - λI) = 0; le radici sono gli autovalori.
- Significato geometrico: gli autovettori rappresentano direzioni dove A agisce come un puro scaling; gli autovalori rappresentano i fattori di scaling.
---
## Teorema Spettrale per Matrici Simmetriche
- Per S simmetrica ∈ R^{n×n} (S = S^T), esiste Q ortogonale tale che S = Q Λ Q^T.
- Differenze da EVD generale:
  - Le colonne di Q sono una base ortonormale; Λ ha autovalori reali.
- Ortogonalità degli autovettori:
  - Per S simmetrica, gli autovettori associati ad autovalori distinti sono ortogonali; spazi nulli e di colonna sono ortogonali negli shift rilevanti (S − αI).
- Realtà degli autovalori:
  - Il quoziente di Rayleigh λ = (x^T S x)/(x^T x) è reale per S simmetrica reale.
---
## Matrici Simmetriche Definite Positive (SPD): Caratterizzazioni e Conseguenze
- Caratterizzazioni equivalenti per S simmetrica:
  1) Tutti gli autovalori λi > 0.
  2) Forma quadratica: v^T S v > 0 per ogni v ≠ 0.
  3) Fattorizzazione: S = B^T B con colonne indipendenti; per SPD, S = L L^T (Cholesky).
- Implicazioni:
  - 1 ⇒ 2 tramite espansione su autobasi e ortogonalità.
  - 3 ⇒ 2 poiché v^T S v = ||B v||^2 ≥ 0, zero solo in v = 0 con rango di colonna pieno.
- Note e applicazioni:
  - Interpretazione energetica: v^T S v come energia di sistema.
  - Ottimizzazione: quadratiche convesse (es. 1/2 v^T S v − v^T b); norma energetica ||v||_S = √(v^T S v).
---
## Decomposizione ai Valori Singolari (SVD): Definizioni, Strutture, Proprietà e Usi Pratici
### Concetti Centrali e Strutture
- SVD generalizza idee spettrali a qualsiasi A ∈ R^{m×n}.
- Forme e copertura:
  - SVD completa: A = U Σ V^T con U ∈ R^{m×m}, V ∈ R^{n×n} ortogonali, Σ ∈ R^{m×n} diagonale (σi non negativi).
  - SVD economica (ridotta): A = Û Σ̂ V^T con Û ∈ R^{m×r}, Σ̂ ∈ R^{r×r}, V ∈ R^{n×n}, r = rango(A). Spesso presentata con V̂ ∈ R^{n×r}.
  - SVD ridotta: mantenere solo R componenti singolari quando rango R < min(m, n).
  - SVD troncata: mantenere le prime k < R componenti per approssimazione/compressione.
- Rango e dimensionalità:
  - Le matrici del mondo reale sono spesso rango-deficienti (R < min(m, n)); rappresentazioni efficaci usano le prime R o meno componenti.
### Obiettivi nella Trasformazione delle Caratteristiche
- Ridurre covarianza (decorrelazione) tra caratteristiche.
- Massimizzare varianza nelle nuove caratteristiche per produrre direzioni indipendenti e informative.
- Esempio (valutazioni film):
  - Gli utenti possono essere ridondanti; SVD trova "spettatori rappresentativi" producendo una base compatta e indipendente.
### Relazione SVD Fondamentale e Forma Vettoriale
- Relazione base: A = U Σ V^T con V ortogonale (V^T V = I).
- Mappatura vettoriale: A vᵢ = σᵢ uᵢ. Se rango(A) = R, σ₁…σ_R > 0 e σ_{R+1}… = 0; v_{i>R} giacciono in null(A).
### Decomposizione come Somma di Matrici di Rango 1
- A = Σ_{i=1}^k σᵢ uᵢ vᵢ^T, dove k = min(m, n) o R se rango-deficiente.
- Valori singolari ordinati in discesa (σ₁ ≥ σ₂ ≥ …); i termini principali catturano la struttura dominante.
### Applicazione Compressione Immagini
- A come matrice immagine (es. 1000×1000).
- Ricostruzione tramite termini di rango 1 pesati per σᵢ.
- Spesso solo i primi 20–30 σᵢ sono grandi; scartare termini successivi comprime preservando qualità.
- SVD troncata è alla base di algoritmi di compressione pratici.
### Esistenza e Proprietà SVD tramite A^T A
- A^T A ∈ R^{n×n} è simmetrica e semidefinita positiva: x^T A^T A x = ||A x||² ≥ 0.
- Decomposizione spettrale: A^T A = B Λ B^T con B ortogonale e Λ non negativa.
- Equivalenza spazio nullo: null(A^T A) = null(A) ⇒ rango(A^T A) = rango(A).
- Costruire SVD:
  - Da A^T A vᵢ = λᵢ vᵢ con λᵢ ≥ 0, porre σᵢ = √λᵢ e uᵢ = (A vᵢ)/σᵢ.
  - Proprietà: uᵢ sono ortonormali; AA^T uᵢ = σᵢ² uᵢ; relazione per-vettore A vᵢ = σᵢ uᵢ.
  - Per i > R: σᵢ = 0 e A vᵢ = 0; questi giacciono in null(A).
- Note pratiche:
  - In Python: svd(A, full_matrices=True/False) seleziona completa vs economica; economica risparmia memoria per m, n grandi.
---
## Connessioni ad Algoritmi e Uso Futuro
- Minimi quadrati:
  - La fattorizzazione QR è centrale per risolutori min ||Ax − b||₂; SVD fornisce pseudo-inversa e robustezza rivelatrice del rango.
- Calcoli autovalori:
  - Il metodo delle potenze converge all'autovettore dominante; il quoziente di Rayleigh stima gli autovalori.
- Ottimizzazione:
  - La struttura SPD assicura forme quadratiche convesse; abilita metodi gradient e gradient coniugato.
---
## Prossimi Accordi ed Elementi d'Azione
- [ ] Applicare proprietà autovalori a funzioni di matrici in esempi legati a ODE (e^A, metodo delle potenze).
- [ ] Usare fattorizzazione QR per derivare e implementare risolutori minimi quadrati; confrontare QR completa vs ridotta.
- [ ] Implementare Gram–Schmidt classico; notare problemi numerici; confrontare con Gram–Schmidt modificato/Householder.
- [ ] Praticare trasformazioni di similarità: verificare invarianza autovalori e mappatura autovettori tramite M.
- [ ] Dimostrare e usare il teorema spettrale in esercizi: ortogonalità e realtà di autoppie per matrici simmetriche.
- [ ] Lavorare attraverso caratterizzazioni SPD; verificare positività energetica; implementare Cholesky per sistemi SPD.
- [ ] Calcolare SVD (completa, economica, ridotta, troncata) su dati campione; interpretare valori/vettori singolari per varianza e covarianza.
- [ ] Esplorare implementazioni Python: qr(A, mode=...) e svd(A, full_matrices=...) per osservare forme ed effetti memoria.
- [ ] Mostrare esempio online di compressione immagini usando SVD troncata nella prossima sessione.