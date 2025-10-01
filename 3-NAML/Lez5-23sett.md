## Focus e ambito del corso
- Approfondimento di algebra lineare e algebra lineare numerica con enfasi su applicazioni in machine learning e data science.
- Famiglie di problemi trattate:
  - Sistemi lineari AX = B: condizioni di risolvibilità e relazione con gli spazi fondamentali.
  - Decomposizione spettrale: AX = λX; interpretazione geometrica degli autovettori (direzione preservata, lunghezza scalata).
  - Decomposizione ai valori singolari (SVD): AV = σU; vettori singolari (V, U) e valori singolari (σ); applicabile a qualsiasi matrice, anche singolare/non quadrata; considerata una generalizzazione della decomposizione spettrale ed essenziale per data science/ML.
  - Minimizzazione dei minimi quadrati: regressione collegata agli strumenti di algebra lineare.
  - Fattorizzazioni di matrici: fattorizzazioni didattiche e pratiche; ordine di trattazione flessibile (sistemi → fattorizzazioni → autovalori/SVD → minimizzazione).
## Rappresentazione dei dati con matrici
- Qualsiasi dataset può essere rappresentato come una matrice A ∈ R^{m×n}:
  - m = numero di campioni (righe).
  - n = numero di feature (colonne).
- Esempio:
  - Dataset di migliaia di immagini in scala di grigi, ciascuna 1000×1000 pixel.
  - Ogni immagine → vettore di lunghezza 1.000.000 (intensità pixel 0–255).
  - Dataset completo → matrice con migliaia di righe e 1.000.000 colonne.
- Le feature non numeriche possono essere codificate numericamente (ad esempio, one-hot encoding) così le matrici possono rappresentare qualsiasi dataset.
## Moltiplicazione matrice-vettore: prospettiva dello spazio delle colonne
- L'operazione AX produce un vettore che è una combinazione lineare delle colonne di A:
  - AX = x1·a1 + x2·a2 + …, dove ai sono le colonne di A.
- Spazio delle colonne C(A):
  - Insieme di tutte le combinazioni lineari delle colonne di A.
  - Il risultato di AX giace sempre in C(A).
### Esempi e interpretazione geometrica
- A1 (3×2) con colonne a1 = [1,2,1]^T, a2 = [3,1,-1]^T:
  - AX = x1·a1 + x2·a2 → C(A1) è un piano passante per l'origine (rango 2).
- A2 (3×3) = [[1,2,3],[4,5,6],[7,8,9]]:
  - a3 = 2·a2 − a1 → non indipendente; C(A2) coincide con C(A2′) dove A2′ = [[1,2],[4,5],[7,8]].
  - C(A2) è un piano (rango 2), sottospazio di R^3; a3 giace nel piano.
- A3 (3×3) = [[1,2,3],[4,5,6],[7,8,10]]:
  - Tre colonne indipendenti → C(A3) = R^3 (rango 3).
- A4 (3×3) = [[1,2,3],[2,4,6],[3,6,9]]:
  - a2 = 2·a1, a3 = 3·a1 → solo una colonna indipendente.
  - C(A4) è una retta passante per l'origine con direzione [1,2,3]^T (rango 1).
## Rango: definizione ed esempi
- Rango r(A):
  - Numero di colonne linearmente indipendenti.
  - Coincide con la dimensione dello spazio delle colonne C(A).
- Esempi di rango:
  - r(A1) = 2, r(A2) = 2, r(A2′) = 2, r(A3) = 3, r(A4) = 1.
## Risolvibilità dei sistemi lineari AX = B
- AX = B ha soluzione se e solo se B ∈ C(A).
  - Poiché AX ∈ C(A) per ogni X; se B ∉ C(A), l'uguaglianza è impossibile.
- Implicazione pratica: verificare se B giace nello spazio delle colonne determina la risolvibilità.
## Costruzione di una base per lo spazio delle colonne e fattorizzazione CR
- Procedura per trovare una base di C(A):
  - Inizia con c1 = a1 (prima colonna).
  - Per ogni colonna successiva ak:
    - Se ak è proporzionale a un vettore di base esistente (o combinazione lineare della base attuale), scarta.
    - Altrimenti, aggiungi ak alla base.
- Esempio con A2 = [[1,2,3],[4,5,6],[7,8,9]]:
  - Vettori di base: a1, a2 (a3 = 2·a2 − a1).
  - r(A2) = 2.
### Fattorizzazione CR (prospettiva forma ridotta per righe)
- Costruisci C selezionando colonne indipendenti: C = [a1 a2] = [[1,2],[4,5],[7,8]] (3×2).
- Trova R (2×3) tale che C·R = A:
  - Le colonne di R contengono i coefficienti per ricostruire ogni colonna originale come combinazione lineare delle colonne di C:
    - r1 = [1,0]^T → C·r1 = a1.
    - r2 = [0,1]^T → C·r2 = a2.
    - r3 = [−1,2]^T → C·r3 = a3 = 2·a2 − a1.
- A2 = C·R fornisce una fattorizzazione didattica; si collega alle forme ridotte per righe (rref).
- Note:
  - Se A ha rango massimo per colonne (ad esempio, A3), allora C = A e R = I.
  - Le colonne di C non sono ortogonali o normalizzate; utilità pratica limitata ma chiarisce la struttura.
## Invarianza del rango rispetto alla trasposizione
- Per A2′ = [[1,2,3],[4,5,6]] (3×2), considera A2′^T = [[1,4],[2,5],[3,6]]:
  - Terza colonna uguale a 2·seconda − prima; rango = 2.
- Risultato generale: rango(A^T) = rango(A); dim C(A^T) = dim C(A).
## Moltiplicazione matrice-matrice: decomposizione colonna-riga (prodotto esterno)
- Moltiplicazione standard: Se A ∈ R^{m×n}, B ∈ R^{n×p}, allora AB ∈ R^{m×p}, calcolata tramite prodotti riga per colonna.
- Vista colonna-riga:
  - Dividi A nelle sue colonne {cA1, cA2, …, cAn}.
  - Dividi B nelle sue righe {rB1, rB2, …, rBn}.
  - AB = Σ_{k=1}^n (cAk · rBk), dove ogni termine è un prodotto esterno (m×1 per 1×p → m×p).
- Ogni termine prodotto esterno è di rango 1 per costruzione; il prodotto completo è una somma di contributi di rango 1.
- Importanza concettuale:
  - Forma la base per rappresentare matrici come somme di componenti di rango 1.
  - Si collega direttamente a SVD e PCA.
  - PCA si basa su approssimazioni a basso rango derivate da SVD, rendendo centrale la visione della somma di rango 1.
### Esempio e contributi di rango 1
- Esempio: A = [[1,2],[3,4]], B = [[2,1],[2,3]].
  - cA1·rB1 = [1,3]^T · [2,1] = [[2,1],[6,3]], rango 1.
  - cA2·rB2 = [2,4]^T · [2,3] = [[4,6],[8,12]], rango 1.
  - Somma = [[6,7],[14,15]], identica al risultato della moltiplicazione standard.
- Intuizione:
  - AB è una somma di matrici di rango 1; fondamentale per SVD e approssimazioni a basso rango.
## Sottospazi fondamentali, ortogonalità e dimensioni
- Dato A ∈ R^{m×n} con rango(A) = r.
- Spazi delle colonne:
  - Col(A) ⊂ R^m, dim(Col(A)) = r.
  - Col(A^T) ⊂ R^n, dim(Col(A^T)) = r.
- Nuclei (= null spaces):
  - Null(A) ⊂ R^n (vettori x con A x = 0).
  - Null(A^T) ⊂ R^m (vettori y con A^T y = 0).
- Relazioni di ortogonalità:
  - Col(A^T) ⟂ Null(A): ogni x ∈ Null(A) è ortogonale a ogni riga di A.
  - Col(A) ⟂ Null(A^T): ogni y ∈ Null(A^T) è ortogonale a ogni colonna di A.
- Complementi ortogonali negli spazi ambienti:
  - In R^m: Col(A) e Null(A^T) sono complementi ortogonali; dim(Col(A)) = r, dim(Null(A^T)) = m − r.
  - In R^n: Col(A^T) e Null(A) sono complementi ortogonali; dim(Col(A^T)) = r, dim(Null(A)) = n − r.
- Teorema rango-nullo:
  - dim(Null(A)) = n − r e dim(Null(A^T)) = m − r.
### Interpretazione di A x = 0 tramite prodotti scalari
- Matrice di esempio: A = [[1,2,3],[4,5,6],[7,8,9]], x = [x1; x2; x3], condizione A x = 0.
- Vista moltiplicazione riga–colonna:
  - Siano r1, r2, r3 le righe di A.
  - A x = 0 implica r_i · x = 0 per i = 1,2,3 (prodotti scalari).
- Implicazioni:
  - x è ortogonale a tutte le righe di A; quindi x ∈ Null(A) ⇒ x ⟂ Col(A^T).
### Proprietà di sottospazio dei nuclei
- Chiusura e scalabilità del nucleo:
  - Il vettore 0 è in Null(A).
  - Se x, y ∈ Null(A), allora x + y ∈ Null(A).
  - Se x ∈ Null(A) e α ∈ R, allora αx ∈ Null(A).
### Base costruttiva per Null(A) tramite fattorizzazione a blocchi
- Setup: A ∈ R^{m×n}, rango(A) = r.
- Partiziona A in A1 (m×r) con colonne linearmente indipendenti e A2 (m×(n−r)) le restanti colonne dipendenti.
- Esprimi A2 come combinazione lineare di A1: A2 = A1 · B, con B ∈ R^{r×(n−r)}.
- Quindi A = [A1, A1 B].
- Costruisci K ∈ R^{n×(n−r)}: K = [−B; I_{n−r}].
-- Calcola A K: A K = [A1, A1 B] [−B; I] = A1(−B) + A1 B = 0 ⇒ le colonne di K giacciono in Null(A) e sono linearmente indipendenti.
-- Qualsiasi U ∈ Null(A) può essere scritto come U = K U2:
  - Partiziona U = [U1; U2]; AU = 0 ⇒ A1(U1 + B U2) = 0 ⇒ U1 = −B U2.
  - Quindi U = [−B U2; U2] = K U2.
-- Conclusione:
  - Le colonne di K formano una base per Null(A); dim Null(A) = n − r.
  - Ragionamento simmetrico porta a dim Null(A^T) = m − r.
## Matrici ortogonali, proiezioni e geometria
- Matrice ortogonale Q: Q^T Q = I; det(Q) = ±1.
- Conservazione della norma:
  - Per Y = QX: ||Y||^2 = X^T Q^T Q X = ||X||^2; le trasformazioni ortogonali sono rigide (preservano la lunghezza).
- Rotazione 2D:
  - R(θ) = [[cos θ, −sin θ],[sin θ, cos θ]]; ortogonale, det = +1; ruota di θ.
- Riflesso rispetto a un piano Π con normale unitaria n:
  - v_⊥ = (v · n) n; riflesso w = v − 2(v · n) n.
  - Matrice: R_ref = I − 2 n n^T; ortogonale, det = −1; R_ref^{-1} = R_ref.
- Proiezione ortogonale su Π:
  - P = I − n n^T; singolare (det = 0); non invertibile per perdita di informazione.
- Chiarimenti sulle proiezioni:
  - Proiezione di a sulla direzione b (non unitaria): vettore proiettato = (a · b) (b / ||b||^2).
## Collegamenti con machine learning e data science
- SVD:
  - Decompone A come Σ_i σ_i u_i v_i^T (somma di matrici di rango 1); applicabilità universale; alla base di PCA, riduzione dimensionale, filtraggio del rumore e modellazione a basso rango.
- PCA:
  - Sfrutta i valori/vettori singolari dominanti per approssimazioni a basso rango che catturano la varianza e riducono la dimensionalità.
- Minimi quadrati:
  - Regressione vista come minimizzazione dei residui; la risolvibilità è legata allo spazio delle colonne e al rango (equazioni normali, pseudoinversa).
- Rango e spazio delle colonne:
  - Determinano l'identificabilità del modello, la ridondanza delle feature e le condizioni di risolvibilità per AX = B.
- Prospettiva colonna-riga:
  - Fornisce comprensione strutturale dei prodotti e motiva le approssimazioni a rango vincolato usate in ML.
- Collegamento con metodi numerici:
  - Riflessi di Householder (da matrici di riflessione) per fattorizzazione QR.
  - Rotazioni di Givens (da matrici di rotazione) per calcoli di autovalori/QR.
## Fatti chiave e conclusioni
- I prodotti tra matrici si decompongono in somme di matrici di rango 1; fondamentale per SVD/PCA.
- Quattro sottospazi fondamentali e dimensioni:
  - Col(A) ⊂ R^m, dim r; Col(A^T) ⊂ R^n, dim r.
  - Null(A^T) ⊂ R^m, dim m − r; Null(A) ⊂ R^n, dim n − r.
- Relazioni ortogonali:
  - Col(A) ⟂ Null(A^T) in R^m; Col(A^T) ⟂ Null(A) in R^n.
  - Ogni coppia forma complementi ortogonali negli spazi ambienti.
- Metodo costruttivo con K = [−B; I_{n−r}] fornisce una base per Null(A).
- Le matrici di proiezione sono singolari; riflessioni e rotazioni sono ortogonali con det ±1.
## Esempi e contabilità dimensionale
- Esempio numerico: A = [[1,2,3],[4,5,6],[7,8,9]] mostra che i vettori di Null(A) sono ortogonali alle righe.
- Partizionamento delle dimensioni:
  - A1: m×r; A2: m×(n−r); B: r×(n−r); K: n×(n−r); A K: m×(n−r) zero.
## 📅 Prossimi passi e attività da svolgere
- [ ] Ripassare i sistemi lineari AX = B con attenzione al criterio di risolvibilità B ∈ C(A).
- [ ] Costruire basi per gli spazi delle colonne usando controlli di indipendenza su matrici di esempio (A1, A2, A3, A4).
- [ ] Esercitarsi con la fattorizzazione CR: formare C dalle colonne indipendenti e calcolare R per ricostruire A.
- [ ] Verificare rango(A^T) = rango(A) su altri esempi di matrici.
- [ ] Rielaborare esempi di moltiplicazione matrice-matrice usando la vista colonna-riga (prodotto esterno) e identificare componenti di rango 1.
- [ ] Completare e interiorizzare la dimostrazione che le colonne di K generano Null(A); formalizzare dim Null(A) = n − r e dim Null(A^T) = m − r.
- [ ] Prepararsi ai prossimi moduli: decomposizione spettrale, fondamenti e applicazioni SVD, minimizzazione dei minimi quadrati e fattorizzazione QR tramite riflessioni di Householder e rotazioni di Givens.
- [ ] Convertire feature non numeriche in rappresentazioni numeriche (ad esempio, one-hot) per dataset basati su matrici.
- [ ] Esplorare le implicazioni pratiche di rango e spazio delle colonne nel preprocessing dei dati (ridondanza delle feature, dimensionalità).
- [ ] Partecipare al prossimo incontro di venerdì; riprendere dopo una breve pausa alle 15:00.