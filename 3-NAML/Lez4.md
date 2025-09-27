## Gestione e visualizzazione dei dati con Pandas e Matplotlib; breve introduzione a SciPy; modelli compartimentali (SIR) e ODE Predatore–Preda
### Obiettivi principali
- Fornire una panoramica concisa e pratica di:
  - Pandas per caricamento, pulizia, trasformazione, raggruppamento e selezione dei dati
  - Matplotlib per la visualizzazione dei dati dei DataFrame (grafici a linee, grafici a torta)
  - Panoramica dei moduli SciPy e dove trovare soluzioni per problemi numerici, inclusi i solver ODE per modelli SIR e Predatore–Preda
---
## Pandas: Fondamenti dei DataFrame e workflow pratico
### Concetto di DataFrame
- Un DataFrame assomiglia a una porzione di un foglio Excel:
  - Strutturato per righe e colonne
  - Righe e colonne possono avere nomi (etichette), facilitando la manipolazione significativa dei dati
- Due tipi di dati evidenziati:
  - Serie temporali
  - DataFrame (focus principale)
### Esempio tipico di dati CSV (dataset COVID-19)
- Colonne:
  - data, paese, confermati, guariti, decessi
- Il dataset copre più paesi e date dall'inizio della pandemia, per circa due anni
### Lettura dei dati
- Funzione principale: pandas.read_csv(...)
  - Configurabile tramite parametri opzionali:
    - Delimitatore
    - Gestione dell'intestazione (salta righe iniziali)
    - Valori mancanti (gestione NaN, specifica valori NA)
    - Terminatore di riga
    - Parsing di tipi di colonna specifici (ad esempio, date)
- Raccomandazione: Impostare i tipi di dati corretti durante la lettura iniziale per accuratezza e prestazioni successive
### Impostazione dei tipi corretti (parsing delle date)
- Esempio: analizzare la colonna “data” come datetime
  - Parametro: parse_dates=['data']
  - Risultato:
    - La colonna “data” diventa datetime64
    - Permette operazioni su intervalli di date e filtraggio per data
### Ispezione dei DataFrame
- df.info(): panoramica dettagliata della struttura (numero di elementi, intervallo degli indici, colonne e tipi, uso memoria)
- df.shape: tupla (righe, colonne)
- df.dtypes: tipo per colonna
- df.head(), df.tail(): anteprima delle prime/ultime righe
### Accesso e creazione di colonne
- Seleziona una colonna: df['paese']
- Crea nuove colonne derivate:
  - Somma tra colonne:
    - df['totale_confermati'] = df[['confermati','guariti','decessi']].sum(axis=1)
  - Rapporti:
    - df['rapporto'] = df['confermati'] / df['decessi'] (gestire divisione per zero → NaN)
### Raggruppamento e aggregazione
- Raggruppa per data per ottenere i totali giornalieri mondiali:
  - mondiali = df.groupby('data')[['confermati','guariti','decessi']].sum()
- Risultato:
  - DataFrame indicizzato per data con totali aggregati (ad esempio, confermati, guariti, decessi e totali derivati)
### Filtraggio per paese e data
- Filtra per paese:
  - italia_df = df[df['paese'] == 'Italia']
  - italia_giornaliero = italia_df.groupby('data')[['confermati','guariti','decessi']].sum()
- Filtra per giorno specifico:
  - giorno = '2022-04-16'
  - ultimo_giorno_df = df[df['data'] == giorno]
  - Ordina e seleziona i primi 10:
    - top10 = ultimo_giorno_df.sort_values('confermati', ascending=False).iloc[:10]
  - Esempio top-10: USA, India, Brasile, Francia, Germania, Regno Unito, Russia, Corea del Sud, Italia, Turchia
---
## Visualizzazione con Matplotlib dai DataFrame
### Grafico a linee da dati raggruppati (mondiali nel tempo)
- Usa il metodo plot del DataFrame:
  - ax = mondiali.plot(figsize=(larghezza, altezza))
  - Personalizza gli assi:
    - ax.set_xlabel('Data'); ax.set_ylabel('Numero di casi')
    - ax.set_title('Andamento COVID-19 mondiale')
    - ax.grid(...); ax.legend(loc=..., bbox_to_anchor=...)
- Punti chiave:
  - Pandas integra la gestione delle date sull'asse x se il tipo datetime è impostato
  - Più colonne (confermati, guariti, decessi, totale_confermati) vengono visualizzate come linee separate
### Grafico a torta per i primi 10 paesi (per casi confermati in un giorno specifico)
- Crea figura e assi:
  - fig = plt.figure(figsize=(...))
  - ax = fig.add_subplot(1,1,1)
- Grafico:
  - ax.pie(top10['confermati'], labels=top10['paese'], autopct='%.1f%%')
- Utilizzi:
  - Confronto istantaneo tra categorie (paesi) per una singola data
---
## SciPy: panoramica di un ecosistema di calcolo numerico
### Posizionamento
- SciPy è un pacchetto completo per il calcolo numerico costruito su NumPy
- Progettato per il calcolo numerico con array NumPy come struttura dati principale
### Panoramica dei moduli
- Sottomoduli che coprono:
  - Trasformate (FFT)
  - Integrazione
  - Interpolazione (1D, n-dimensionale, dati sparsi)
  - Algebra lineare
  - Elaborazione immagini
  - Ottimizzazione
  - Elaborazione segnali
  - Statistica
- Punto di forza: documentazione estesa con esempi
### Messaggio pratico
- Se un problema coinvolge un metodo numerico noto o un algoritmo, SciPy probabilmente offre un'implementazione o una variante vicina
- Incoraggiamento a consultare la documentazione SciPy per soluzioni e codice di esempio
---
## Panoramica dei modelli compartimentali (SIR) e soluzione numerica con SciPy
### Concetto
- I conteggi di confermati, guariti e decessi sono modellati come compartimenti nei modelli compartimentali
- Modello più semplice: SIR (Suscettibili S, Infetti I, Guariti R) senza compartimento decessi
- Obiettivo: descrivere l'evoluzione temporale di S, I, R tramite un sistema di ODE
### Equazioni SIR e parametri
- Equazioni:
  - dS/dt = −β S I / N
  - dI/dt = β S I / N − γ I
  - dR/dt = γ I
- Parametri:
  - N: popolazione totale a t=0
  - β (beta): tasso di contatto efficace
  - γ (gamma): tasso medio di guarigione (1/γ = periodo medio di infezione)
- Intuizione dinamica:
  - S diminuisce con l'aumentare delle infezioni
  - I aumenta per i contatti e diminuisce per le guarigioni
  - R aumenta al tasso γ I
### Soluzione numerica con SciPy (solve_ivp)
-- Librerie: NumPy, Matplotlib, SciPy integrate.solve_ivp
-- Metodi del solver: RK45 (default), RK23, BDF, ecc.
-- Firma della funzione per solve_ivp:
  - fun(t, y, ...params), dove y = [S, I, R]
-- Schema di implementazione:
  - Scompatta y in S, I, R; restituisci [dSdt, dIdt, dRdt]
-- Condizioni iniziali e controllo del tempo:
  - y0: [S0, I0, R0] (ad esempio, S0=N−1, I0=1, R0=0)
  - t_span: [0, 160] (giorni)
  - t_eval: np.linspace(0, 160, 161) per campionamento uniforme
-- Chiamata a solve_ivp:
  - fun, t_span, y0, t_eval, args=(N, β, γ)
-- Interpretazione dell'output:
  - sol.success, sol.message, sol.t, sol.y con S(t)=sol.y[0], I(t)=sol.y[1], R(t)=sol.y[2]
### Grafico dei risultati SIR
-- Usa sottoplot di matplotlib
-- Traccia sol.t vs S/N (blu), I/N (rosso), R/N (verde)
-- Regola alpha, larghezza linea, etichette, limiti assi, griglia, legenda
-- Comportamento:
  - Iniziale: S/N ≈ 1, I/N ≈ 0
  - Centrale: I/N sale a un picco mentre S/N diminuisce
  - Finale: I/N diminuisce; R/N aumenta verso lo stato stazionario
---
## Modello Predatore–Preda (Lotka–Volterra)
### Struttura del modello
- Variabili: preda (x) e predatore (y)
- Parametri: α, β, γ, δ
- Equazioni:
  - dx/dt = αx − βxy
  - dy/dt = −δy + γxy
### Impostazione numerica
- Usa solve_ivp con fun(t, X, α, β, γ, δ); scompatta X → x, y
- Esempio di condizioni iniziali: x0 = 10 (preda), y0 = 5 (predatore)
- Intervallo temporale e t_eval simili a SIR
### Risultati
- I grafici temporali mostrano dinamiche oscillanti di preda e predatore
- Il grafico nello spazio delle fasi (x vs y) rivela traiettorie simili a orbite chiuse (approssimate numericamente)
### Equilibri
- Imposta le derivate a zero:
  - x(α − βy) = 0
  - y(−δ + γx) = 0
- Equilibrio non banale (x* > 0, y* > 0):
  - x* = δ/γ
  - y* = α/β
- Forma delle orbite e stabilità dipendono da condizioni iniziali e parametri
---
## Guida pratica e inquadramento Data Science
### Buone pratiche
- Leggere CSV con parsing intenzionale dei tipi (soprattutto date)
- Ispezionare i DataFrame subito (info, dtypes, head/tail)
- Creare feature derivate tramite operazioni vettoriali sulle colonne
- Raggruppare/aggregare su dimensioni significative (data, paese)
- Visualizzare usando funzioni di plotting integrate nei DataFrame per insight rapidi
### Prospettiva Data Science
- L'indice di riga rappresenta tipicamente i campioni
- Le colonne rappresentano le feature
- Questa mappatura è usata in tutte le analisi e i laboratori
### Competenze acquisite nel progetto
- Capacità di:
  - Leggere dataset con pandas.read_csv
  - Ispezionare e comprendere la struttura e i tipi dei DataFrame
  - Estrarre colonne e filtrare righe per condizioni
  - Creare nuove colonne derivate e gestire scenari NaN
  - Raggruppare/aggregare dati per dimensioni chiave e interpretare i risultati
  - Visualizzare DataFrame con Matplotlib (grafici a linee e a torta), personalizzando assi, etichette, griglia e legenda
  - Implementare e visualizzare modelli ODE (SIR, Predatore–Preda) usando solve_ivp di SciPy
---
## Esercizi aggiuntivi, risorse e logistica del corso
### Documentazione SciPy ed ecosistema Python
- Studiare la documentazione delle funzioni e gli esempi per capire parametri e modalità d'uso
- Arricchire Python base con:
  - NumPy (array)
  - Matplotlib (grafici)
  - Pandas (dataframe; valutare alternative per dataset molto grandi)
  - SciPy (metodi numerici)
- Argomenti di algebra lineare (in arrivo):
  - Matrici, risoluzione di sistemi lineari
  - Notebook di riferimento: differenze finite per −u'' = f (costruire matrici, gestire array, risolvere sistemi con NumPy/SciPy)
### Logistica del corso
- Prossima lezione prevista per domani, in presenza (salvo cambiamenti)
---
## 📅 Prossimi passi e attività da svolgere
- [ ] Importare un dataset CSV con pandas.read_csv e impostare parse_dates per le colonne data dove necessario
- [ ] Usare DataFrame.info, shape, dtypes, head, tail per ispezionare struttura e contenuto
- [ ] Creare almeno due colonne derivate (ad esempio, totale_confermati = confermati + guariti + decessi; rapporto = confermati / decessi) e gestire i casi NaN
- [ ] Raggruppare i dati per data per produrre aggregati giornalieri mondiali (confermati, guariti, decessi, totale_confermati)
- [ ] Filtrare i dati per un paese specifico (ad esempio, Italia) e generare totali giornalieri raggruppati, poi visualizzare come grafico a linee
- [ ] Selezionare i dati di un giorno specifico, ordinare per confermati decrescente, estrarre i primi 10 paesi e produrre un grafico a torta con etichette e percentuali
- [ ] Personalizzare i grafici (titoli, etichette assi, stile griglia, posizione legenda) usando gli handle di Matplotlib
- [ ] Consultare la documentazione SciPy e identificare i sottomoduli rilevanti per i prossimi compiti (ad esempio, interpolazione, ottimizzazione)
- [ ] Implementare il modello SIR in Python usando solve_ivp con N, β=0.2, γ=0.1 (1/γ=10), e condizioni iniziali S0=N−1, I0=1, R0=0; produrre grafici normalizzati di S/N, I/N, R/N su t ∈ [0, 160]
- [ ] Implementare il sistema Predatore–Preda (Lotka–Volterra) con parametri (α, β, γ, δ) e condizioni iniziali (x0=10, y0=5); generare grafici temporali e nello spazio delle fasi
- [ ] Verificare e annotare il punto di equilibrio per il modello Predatore–Preda: x* = δ/γ, y* = α/β; illustrare la dipendenza da parametri e condizioni iniziali
- [ ] Esaminare la documentazione SciPy solve_ivp (metodi, t_eval, args, opzioni solver) e sperimentare metodi alternativi (RK23, BDF) per casi stiff/non-stiff
- [ ] Esplorare il progetto differenze finite per −u'' = f: costruire la matrice, risolvere il sistema lineare con NumPy/SciPy; usare come riferimento per i prossimi contenuti di algebra lineare
- [ ] Indagare strumenti oltre Pandas per dataset molto grandi; raccogliere riferimenti per possibili alternative
- [ ] Partecipare alla lezione in presenza di domani su argomenti di algebra e collegamenti Python alle funzioni di algebra lineare