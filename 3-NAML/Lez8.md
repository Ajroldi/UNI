## Denoising di Immagini tramite SVD: Concetti, Sogliatura, Algoritmi e Fondamenti
### Idea Centrale: SVD e Separazione Segnale-Rumore
- Qualsiasi matrice A può essere fattorizzata come A = U Σ V^T (SVD), esprimendo A come somma di componenti di rango 1 ordinate per valori singolari decrescenti. Questo ordinamento fornisce una gerarchia di contributi al contenuto della matrice.
- Nel denoising, l'obiettivo è ricostruire una stima pulita X̃ da un'osservazione rumorosa Y = X + rumore (es. gaussiano additivo), filtrando il rumore piuttosto che comprimere o preservare la varianza complessiva.
- La troncatura basata sulla varianza (es. mantenere 90% energia) è inadatta per il denoising perché:
  - Mantiene il rumore insieme al segnale.
  - Il rumore bianco gonfia lo spettro dei valori singolari, spesso producendo uno spettro quasi-completo senza chiara struttura a basso rango.
- La chiave è separare i valori singolari associati al vero contenuto dell'immagine da quelli dominati dal rumore.

### Strategia di Sogliatura Rigida per SVD
- Definire una soglia τ e mantenere solo i valori singolari σ_i di Y che superano τ; scartare quelli sotto τ (impostare a zero).
- Razionale: Nelle matrici rettangolari rumorose (immagini), lo spettro dei valori singolari spesso mostra una transizione netta ("ginocchio") che separa i valori singolari relativi al segnale da quelli relativi al rumore. Questo permette di definire un τ ottimale.
- Caso rettangolare generale (Y ∈ ℝ^{m×n}, assumere m ≥ n):
  - Rapporto d'aspetto: β = m/n ≥ 1.
  - Valore singolare mediano: σ_med, usato come stimatore data-driven della magnitudine del rumore.
  - Soglia: τ = ω(β) · σ_med, dove ω(β) è un polinomio in β (approssimazione closed-form dalla letteratura recente circa 2014). Questo τ non richiede livello di rumore a priori; solo β e σ_med.
- Matrici quadrate con livello di rumore noto:
  - Se Y ∈ ℝ^{n×n} con rumore gaussiano additivo di magnitudine γ (o c), usare τ = (4/√3) · √n · γ. Questo è semplice ed esplicito quando il livello di rumore è noto.

### Ottimalità dell'Errore della Ricostruzione
- Lo stimatore con sogliatura rigida X̃ = U Σ̂ V^T minimizza l'errore di ricostruzione al quadrato ||X̃ − X||_F² tra gli stimatori di sogliatura SVD sotto il modello di rumore additivo Y = X + rumore.
- Questo si allinea con il denoising ai minimi quadrati e sfrutta le identità della norma di Frobenius che collegano i valori singolari all'energia.

### Workflow: Denoising Basato su SVD
- Input: Matrice immagine rumorosa Y.
- Passi:
  1. Calcolare SVD: Y = U Σ V^T.
  2. Calcolare σ_med (valore singolare mediano di Y).
  3. Stimare il rumore implicitamente tramite σ_med se γ è sconosciuto.
  4. Calcolare τ:
     - Caso generale: τ = ω(β) · σ_med, β = m/n (o scambiare per garantire β ≥ 1).
     - Quadrata, rumore noto: τ = (4/√3) · √n · γ.
  5. Formare Σ̂ mantenendo σ_i ≥ τ; impostare σ_i < τ a 0 (soglia rigida). Sia R il numero mantenuto.
  6. Ricostruire: X̃ = U Σ̂ V^T usando le prime R colonne di U, i primi R valori singolari in Σ, e le prime R righe di V^T.
- Interpretazione: Produce una rappresentazione a basso rango focalizzata su componenti informativi, distinta dalla compressione-per-varianza.

### Esempio di Setup, Parametri e Risultati
- Immagine originale: Costruita da u e v come sovrapposizioni di componenti seno e coseno (simili a onde), con valori singolari puliti 2 e 1.
- Rumore: Aggiunto con parametro di intensità (σ/γ).
- Caso quadrato, rumore noto:
  - Calcolare τ = (4/√3) × √n × γ.
  - Mantenere valori singolari sᵢ > τ e ricostruire con R componenti mantenuti.
- Risultato: Filtraggio del rumore efficace con algoritmo semplice; non recupero perfetto ma forte miglioramento.

### Mantenimento Energia vs Soglia Ottimale: Trade-off
- Approccio mantieni 90% varianza:
  - Può richiedere R ≈ centinaia di componenti (es. R ≈ 401 in un caso dimostrato).
  - L'immagine risultante rimane rumorosa a causa dei componenti mantenuti dominati dal rumore.
- Approccio basato su soglia:
  - Sfrutta il "gap" spettrale e spesso mantiene pochissimi componenti (es. R = 1 nel caso illustrato).
  - Produce ricostruzioni significativamente più pulite.

### Osservazioni Spettrali e Selezione Pratica
- Gli spettri dei valori singolari di immagini rumorose tipicamente mostrano:
  - Un piccolo numero di valori singolari grandi (segnale).
  - Molti valori singolari più piccoli (rumore).
- Implicazione pratica: La sogliatura grafica è viabile—localizzare il salto spettrale e impostare τ appena sotto il gap.

### Confronto Concettuale: Fourier vs SVD
- Serie di Fourier:
  - Base fissa (sinusoidi), indipendente dai dati; generale ma non personalizzata.
- SVD:
  - Base adattiva ai dati; tipicamente più efficiente per rappresentazione e denoising nelle immagini.

### Sfide Computazionali e SVD Randomizzata (rSVD)
- La SVD classica è computazionalmente pesante per matrici molto grandi (milioni di righe/colonne), con alte richieste di tempo e memoria.
- Lemma di Johnson–Lindenstrauss:
  - Fornisce riduzione dimensionale preservando distanze a coppie entro (1 ± ε), con dimensione target k ≈ O(log(n)/ε²).
  - Abilita algoritmi veloci e memory-efficient per grandi dataset.
- rSVD: Approssimazione scalabile di SVD sfruttando proiezioni casuali.
  - Precondizioni: I dati hanno struttura a bassa dimensione effettiva (basso rango intrinseco).
  - Passi:
    1. Generare Ω casuale ∈ ℝ^{n×k}.
    2. Formare Y = AΩ (cattura il range di A).
    3. QR economica: Y = Q R (Q ortonormale).
    4. Proiettare: B = Q^T A (k × n).
    5. SVD di B: B = Ũ Σ V^T.
    6. Sollevare: U = Q Ũ; A ≈ U Σ V^T.
  - Benefici: Più veloce, parallelizzabile, con limiti d'errore teorici.

### Miglioramenti Pratici per rSVD
- Oversampling:
  - Se il rango desiderato è r, usare r + p (p ≈ 5–10) colonne in Ω; troncare a r successivamente.
- Iterazioni di potenza:
  - Applicare a (A A^T)^q A per amplificare la separazione dei valori singolari superiori.
  - Migliora la cattura delle componenti dominanti; implementare tramite moltiplicazioni ripetute con A e A^T prima di QR.

### Esempio Giocattolo: Accuratezza rSVD
- Matrice esempio: [[1,3,2], [5,3,1], [3,4,5]] con valori singolari veri ≈ 9.34, 3.24, 1.6.
- rSVD con k=2 recupera ≈ 9.34 e ≈ 3.0; le iterazioni di potenza migliorano la stima del secondo valore singolare.

### Intuizioni Visive e Spettrali
- Matrici casuali: Spettri dei valori singolari distribuiti, indicativi di rumore.
- Iterazione di potenza: Concentra lo spettro, accentuando i valori singolari principali e facilitando la selezione delle componenti dominanti.

### Geometria dei Dati: Perché SVD Funziona per le Immagini
- Lo spazio delle immagini è vasto:
  - 20×20 bianco/nero: 2^400 possibilità; scala di grigi 256^400; RGB ancora di più.
- Le immagini naturali occupano un sottosinsieme minuscolo (varietà) di questo spazio:
  - Implica basso rango effettivo; pochi valori singolari catturano contenuto significativo.

### Norme Matriciali e Migliore Approssimazione a Basso Rango
- Norme:
  - Norma di Frobenius: ||A||_F = sqrt(Σ a_{ij}²); identità: ||A||_F² = tr(A^T A) = Σ σ_i²; invariante sotto trasformazioni ortogonali/unitarie.
  - p-norme indotte: ||A||_p = sup_{||x||_p=1} ||Ax||_p; p=1 (max somma colonna), p=∞ (max somma riga), p=2 (norma spettrale = σ_max(A)).
  - La submoltiplicatività vale per norme indotte; Frobenius è submoltiplicativa ma non indotta.
- Norma spettrale:
  - ||A||_2 = σ_1 = sqrt(λ_max(A^T A)); significato geometrico: fattore di stretching massimo su vettori unitari.
- Teorema di Eckart–Young–Mirsky:
  - SVD troncata A_k = Σ_{i=1}^k σ_i u_i v_i^T è la migliore approssimazione di rango k sotto ||·||_2 e ||·||_F.
  - Errori: ||A − A_k||_2 = σ_{k+1}; ||A − A_k||_F = sqrt(Σ_{i=k+1} σ_i²).

### Connessioni al Completamento Matrici
- Sistemi di raccomandazione (es. film × utenti): Matrici sparse di valutazioni con voci mancanti.
- La sogliatura dei valori singolari aiuta a recuperare struttura a basso rango e inferire voci mancanti separando segnale da rumore/inconsistenza.

### Punti Chiave
- SVD fornisce una rappresentazione strutturata potente ma necessita sogliatura per distinguere rumore da segnale per denoising.
- La sogliatura rigida dei valori singolari offre denoising efficace e teoricamente supportato sfruttando transizioni spettrali.
- τ ottimale può essere calcolato usando solo rapporto d'aspetto e σ_med, o tramite forma closed quando la magnitudine del rumore è nota (caso quadrato).
- La X̃ ricostruita tramite sogliatura rigida minimizza l'errore norma-Frobenius sotto il modello di rumore additivo.
- Per problemi su larga scala, rSVD con oversampling e iterazioni di potenza offre approssimazioni scalabili e accurate.

### Prossimi Accordi ed Elementi d'Azione
- [ ] Implementare l'algoritmo di sogliatura rigida SVD: calcolare U, Σ, V^T; stimare σ_med; calcolare τ; formare Σ̂; ricostruire X̃ = U Σ̂ V^T.
- [ ] Validare l'approssimazione ω(β) e garantire selezione β corretta (β ≥ 1; aggiustare per ordinamento m,n).
- [ ] Eseguire esempi numerici per entrambi i casi: rumore sconosciuto (τ = ω(β)·σ_med) e rumore noto in matrici quadrate (τ = (4/√3)·√n·γ).
- [ ] Confrontare prestazioni denoising vs troncatura basata su varianza (90% energia) per evidenziare inadeguatezza di quest'ultima per dati rumorosi.
- [ ] Plottare spettro valori singolari e marcare τ per validare visivamente il gap e i ranghi mantenuti.
- [ ] Implementare rSVD con oversampling (r + p, p ≈ 5–10) e iterazioni di potenza (q ≈ 3); valutare qualità ricostruzione e runtime/memoria vs SVD classica.
- [ ] Documentare limiti d'errore e linee guida selezione parametri pratici (r, p, q, ε) per dataset futuri.
- [ ] Preparare note e schema per applicare sogliatura valori singolari a task completamento matrici; fornire riferimenti su lemma Johnson–Lindenstrauss e algebra lineare randomizzata.
- [ ] Presentare dimostrazione del caso norma-Frobenius di Eckart–Young–Mirsky e rivedere definizioni p-norme indotte con esempi.