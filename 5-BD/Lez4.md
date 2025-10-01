## Riassunto: Concetti Modellazione ER, Ambito Esame e Esempi Lavorati

### Concetti Chiave: Modelli ER e Componenti
- Scopo: Rappresentare concetti dati ad alto livello e come le entità si collegano; usato per comprendere la struttura prima dell'implementazione relazionale o NoSQL.
- Elementi centrali:
  - Entità: Oggetti dati (es. Persona, Auto).
  - Relazioni: Collegamenti tra entità, con cardinalità che indicano vincoli di quantità.
  - Cardinalità: 0-a-1, 0-a-n, 1-a-1, 1-a-n; n-a-n tipicamente divise in due relazioni 1-a-n.
  - Attributi: Proprietà delle entità (es. nome, altezza, colore occhi).
  - Chiavi primarie: Identificano univocamente le istanze; evitare nomi come chiavi primarie a meno che specificato, preferire ID ufficiali (es. ID personale/codice fiscale).
  - Chiavi primarie composite: Combinano attributi multipli per identificare univocamente un'entità.
  - Attributi compositi: Attributi composti da sotto-parti (es. indirizzo = via + numero + città + provincia).
- Gerarchie ISA (is-a):
  - Struttura: Genitore (es. Animale) con figli (es. Gatto, Cane).
  - Totale vs Parziale:
    - Totale: Il genitore è astratto; solo i figli hanno istanze.
    - Parziale: Le istanze possono esistere sia nel genitore che nei figli.
  - Esclusivo vs Sovrapposto:
    - Esclusivo: L'istanza appartiene esattamente a un tipo figlio.
    - Sovrapposto: L'istanza può appartenere a tipi figli multipli simultaneamente.

### Ambito Esame, Direzione di Lettura ed Euristiche di Design
- Orientamento:
  - Focus principalmente sui concetti relazionali; comprensione ER richiesta anche per argomenti NoSQL.
  - Trasformazione ER-a-logico-a-fisico insegnata tramite euristiche; il designer applica giudizio.
  - Entità deboli/relazioni deboli sono fuori ambito.
- Lettura relazioni:
  - Seguire direzione di lettura e notazione fornite dall'esame, anche se diverse da convenzioni precedenti.
- Assunzioni:
  - Permissibili quando il testo manca di dettagli, ma non devono contraddire testo dato o conoscenza della lezione.
  - Esempio: Usare email come identificatore unico per Cliente quando gli attributi non sono specificati.
- Euristiche e scelte di design:
  - Possono esistere design multipli validi; valutati per correttezza e consistenza.
  - Leggere tutto l'esame prima di modellare; query successive possono guidare struttura, specialmente in NoSQL (es. MongoDB).

---

## Esempio 1: Modello ER Sistema di Noleggio Auto

- Obiettivo: Gestire clienti, auto, ispezioni, operazioni e fatture noleggio.
- Elementi richiesti forniti dal cliente: Carta d'identità (ID nove cifre, nome, cognome, data nascita, ecc.), patente di guida (attributi, numero patente), carta di credito.
- Elenco auto e controllo preliminare:
  - Auto descritte da marca, prezzo al giorno, velocità max, consumo carburante per chilometro.
  - Mostrare solo auto ispezionate negli ultimi tre mesi.
  - Ispezioni includono data, nome azienda, ed elenco operazioni eseguite.
- Flusso noleggio e fatturazione:
  - Cliente seleziona auto, riceve chiavi.
  - Sistema memorizza fattura noleggio e attributi.
  - Al ritorno, cliente viene addebitato; fattura marcata chiusa dopo pagamento e fornita al cliente.

### Entità
- Cliente
- Auto
- Carta d'Identità
- Patente di Guida
- Carta di Credito
- Ispezione Auto
- Operazione
- Fattura Noleggio

### Attributi e Chiavi (Rappresentativi)
- Carta d'Identità:
  - Attributi: ID (nove cifre), nome, cognome, data nascita, ecc.
  - Chiave primaria: ID
- Patente di Guida:
  - Attributi: Numero patente (chiave primaria), tipi veicoli autorizzati (composito/gruppo di booleani: camion, auto, autobus, motocicletta)
- Auto:
  - Attributi: marca, prezzo al giorno, velocità massima, consumo carburante per chilometro
- Ispezione Auto:
  - Attributi: data, nome azienda; include elenco operazioni
- Fattura Noleggio:
  - Attributi: campi relativi alla fattura; stato fattura (es. chiusa dopo pagamento)

### Relazioni e Cardinalità
- Cliente–Patente di Guida: 1-a-1 (ognuno appartiene univocamente all'altro)
- Cliente–Carta d'Identità: 1-a-1
- Cliente–Carta di Credito: 1-a-n (un cliente ha una o più carte; ogni carta appartiene a un cliente)
- Auto–Ispezione Auto:
  - Auto: 0-a-n ispezioni
  - Ispezione: n-a-1 all'Auto (ogni ispezione per esattamente un'auto)
- Ispezione Auto–Operazione:
  - Ispezione: 1-a-n operazioni
  - Operazione: 0-a-n ispezioni (operazioni riutilizzabili tra ispezioni)
- Cliente, Auto, Fattura Noleggio (analisi ternaria):
  - Stesso cliente può noleggiare stessa auto più volte → fatture multiple.
  - Fisso due entità a una; derivo la terza:
    - Un'Auto + Un Cliente → Fatture Noleggio Multiple possibili
    - Un'Auto + Una Fattura Noleggio → Un Cliente
    - Un Cliente + Una Fattura Noleggio → Un'Auto
  - Guida al design:
    - Relazioni ternarie spesso evitabili; preferire binarie assicurando identificazione unica tramite percorsi.
    - Percorso Auto → Fattura Noleggio → Cliente è unico; collegamento diretto Auto–Cliente può essere non necessario.

### Flusso Processo e Vincoli
- Pre-noleggio: Auto deve avere ispezione negli ultimi 3 mesi per essere mostrata.
- Selezione: Cliente sceglie auto; chiavi fornite; fattura creata e memorizzata.
- Ritorno e pagamento: Addebito al ritorno; fattura marcata chiusa; fattura fornita.

---

## Esempio 2: Modello ER Associazione Musei Olandesi

- Ambito: Tracciare musei, sale (stanze), opere d'arte, artisti, visitatori, biglietti, negozi, gadget, visitatori giornalieri e vendite.

### Entità
- Museo
- Sala (Stanza)
- Opera d'Arte
- Tipo
- Artista
- Visitatore
- Biglietto
- Negozio
- Gadget

### Attributi e Chiavi
- Museo:
  - Attributi: nome unico (chiave primaria), posizione, indirizzo, orario apertura, orario chiusura (stesso ogni giorno), durata media visita.
- Sala:
  - Attributi: identificatore unico 3 caratteri, titolo, uno o più tipi.
  - Ogni sala appartiene esattamente a un museo; un museo ha una o più sale.
- Opera d'Arte:
  - Attributi: titolo unico, descrizione, tipo.
  - Prodotta da un solo artista; un artista produce una o più opere d'arte.
- Tipo:
  - Descrive sale e opere d'arte.
  - Sala: uno o più tipi; tipo può essere assegnato a sale multiple.
  - Opera d'Arte: esattamente un tipo; tipo può essere assegnato a opere d'arte multiple.
- Artista:
  - Attributi: nome, cognome, data nascita, città nascita, descrizione vita.
  - Chiave primaria (per testo): composita di nome + cognome + data nascita (nota: generalmente evitare nei sistemi reali; preferire ID istituzionali).
- Biglietto:
  - Appartiene a un singolo museo; può essere comprato da esattamente un visitatore.
  - Orario visita può essere modellato sulla relazione Visitatore–Biglietto per permettere orari variabili per acquisto; alternativamente su Biglietto se un compratore e un orario.
- Gadget:
  - Attributi: codice a barre unico, nome prodotto, prezzo.

### Relazioni e Cardinalità
- Museo–Sala: 1-a-n; Sala–Museo: n-a-1 (esclusiva a un museo)
- Sala–Tipo: n-a-n (Sala ha uno o più tipi; Tipo assegnato a sale multiple)
- Opera d'Arte–Tipo: Opera d'Arte 1-a-1 a Tipo; Tipo 1-a-n a Opera d'Arte
- Artista–Opera d'Arte: 1-a-n (un artista, molte opere); Opera d'Arte–Artista: n-a-1
- Museo–Biglietto: 1-a-n; Biglietto–Museo: n-a-1
- Visitatore–Biglietto: 1-a-n; Biglietto–Visitatore: n-a-1
  - Orario visita sulla relazione quando valori variano per collegamento Visitatore–Biglietto; nella mappatura relazionale, con 1-a-n, orario visita spesso finisce su Biglietto a meno che la relazione sia n-a-n.
- Negozio–Gadget:
  - Negozi possono offrire 0-a-n gadget.
  - Gadget possono essere venduti in 1-a-n negozi, o modellati n-a-n se gadget appaiono in negozi multipli.
- Visitatore–Acquisto Gadget: 0-a-n (visitatore può comprare nessuno o molti gadget)

### Interpretazione Vincoli e Implicazioni Modellazione
- Ogni opera d'arte ha esattamente un tipo.
- Una sala contiene opere d'arte multiple; nessun vincolo impone tipo opera d'arte uniforme per sala.
- Non rimuovere la relazione opera d'arte–tipo basata su "tipo stanza" assunto; il testo non garantisce tipi uniformi per stanza.
- Mantenere tipo a livello opera d'arte per classificare propriamente ogni opera d'arte.

### Acquisto Biglietto: Posizionamento Orario Visita
- Visitatore compra biglietti multipli; ogni biglietto ha esattamente un compratore.
- Orario visita può essere:
  - Sulla relazione Visitatore–Biglietto se visitatori multipli per biglietto o orari per-visitatore sono necessari.
  - Su Biglietto se ogni biglietto ha un singolo orario visita e compratore.
- Nella mappatura relazionale, attributi relazione si spostano a tabelle entità a meno che la relazione sia molti-a-molti; dato 1-a-n o 1-a-1, orario visita tipicamente finisce su Biglietto.
- Nota correzione diagramma: Errori cardinalità (es. Biglietto inteso 1-a-N in alcuni contesti) saranno corretti; attributi non necessari rimossi prima di condividere materiali.

### Prestito e Prestazione Opere d'Arte tra Musei
- Requisito: Tracciare il periodo che ogni opera d'arte trascorre in ogni museo; periodo prestito (inizio/fine) concordato prima del trasferimento.
- Modifica proposta:
  - Collegare Opera d'Arte a Sala con attributi:
    - Data inizio esposizione
    - Data fine esposizione
  - Permettere all'Opera d'Arte di associarsi con sale multiple (e quindi musei) nel tempo; imporre periodi non sovrapposte a livello implementazione.
- Razionale:
  - Ogni sala appartiene esattamente a un museo; Opera d'Arte–Sala con periodo implica presenza museo senza aggiungere relazione Opera d'Arte–Museo.
- Alternativa:
  - Opera d'Arte–Museo diretta con attributi periodo è fattibile ma ridondante rispetto a sfruttare Opera d'Arte–Sala.

### Specializzazione Negozi e Gerarchie Prodotti (ISA)
- Nuovi tipi negozi:
  - Ristorante (vende cibo)
  - Negozio dolci (vende dolci; tipo specializzato di cibo)
  - Negozio gadget (vende gadget)
- Categorie prodotti e attributi:
  - Attributi generali Prodotto si applicano a tutti i prodotti.
  - Cibo: stesso di Gadget più data scadenza.
  - Dolci: specializzazione di Cibo.
- Gerarchie ISA e ereditarietà:
  - ISA Negozio: Negozio → {Ristorante, Negozio dolci, Negozio gadget}
    - Totale ed Esclusivo: Negozio astratto; ogni istanza è esattamente un figlio.
    - Figli ereditano attributi e relazioni di Negozio.
  - ISA Prodotto: Prodotto → {Cibo, Gadget}
    - Totale ed Esclusivo: Prodotto astratto; ogni prodotto è Cibo o Gadget.
    - Cibo include data scadenza; Gadget no.
  - ISA Cibo: Cibo → {Dolci}
    - Scelta Sovrapposto vs Esclusivo non critica con un figlio; scegliere basato su bisogni dominio.
- Implicazioni relazioni:
  - Relazione Visitatore–Prodotto Compra ereditata da Cibo, Gadget, e Dolci.
  - Relazione Ristorante–Cibo Vende; Dolci eredita, quindi ristoranti possono vendere dolci.
  - Relazione Negozio dolci–Dolci Vende specifica per dolci.
  - Dolci eredita attributi Prodotto, attributi Cibo (inclusa scadenza), Compra, e relazioni Vende applicabili.

### Amicizia tra Artisti (Solo Pittori e Scultori)
- Requisito: Modellare amicizia tra artisti le cui opere sono esibite; pittori fanno amicizia solo con pittori; scultori solo con scultori.
- ISA e auto-relazioni:
  - ISA Artista: Artista → {Pittore, Scultore}
  - Definire due auto-relazioni:
    - Amicizia(Pittore ↔ Pittore)
    - Amicizia(Scultore ↔ Scultore)
  - Non definire Amicizia a livello Artista per prevenire amicizie cross-tipo non permesse.
- Scelte tipo gerarchia:
  - Totale vs Parziale: Aperto; dipende se solo pittori/scultori sono tracciati o tipi artista più ampi. Risposte multiple accettabili.
  - Esclusivo vs Sovrapposto: Entrambi difendibili (artisti che scolpiscono e dipingono); scegliere basato su assunzioni dominio senza contraddire il testo.
- Attenzione modellazione: Assicurare vincoli riflessi nel posizionamento relazioni per imporre amicizie permesse a livello schema.

---

## Differenze NoSQL vs SQL e Guida Trasformazione
- Database documenti (es. MongoDB) supportano array e documenti incorporati; vincoli relazionali potrebbero non applicarsi.
- Trasformazioni da ER a logico/fisico variano con tecnologia target; euristiche guidano variazioni accettabili.
- Leggere tutti requisiti e query prima di modellare; task successivi possono influenzare strategie embedding o scelte relazioni.

---

## Interazione Lezione e Note Meta
- Approccio interattivo: Check-in audience; esempi scenario-driven (es. visita Museo Van Gogh).
- Strumenti: Usare strumento diagramma ER per esercizi.
- Timing: Notificare istruttore se sessioni superano cinque minuti.
- Materiali: Alcuni esercizi creati da professori esterni; discussione incoraggiata se soluzioni appaiono disallineate.

---

## Elementi d'Azione
- [ ] Usare email come identificatore Cliente unico quando attributi mancano, assicurando nessun conflitto con vincoli.
- [ ] Evitare usare nomi come chiavi primarie; preferire attributi identificazione ufficiali (es. ID personale/codice fiscale).
- [ ] Quando si incontrano cicli relazioni ternarie, tentare rimuovere una relazione verificando identificazione unica tramite percorsi binari.
- [ ] Posizionare attributi specifici relazione (es. orario visita) sulla relazione quando valori variano per collegamento; validare se spostamento a entità è accettabile.
- [ ] Leggere intero esame e tutte query prima di modellare; allineare struttura con requisiti query, specialmente per task NoSQL (es. MongoDB).
- [ ] Seguire direzione lettura e notazione fornite dall'esame per relazioni.
- [ ] Applicare scelte gerarchia ISA (totale/parziale, esclusivo/sovrapposto) consistenti con regole dominio durante trasformazione.
- [ ] Focus su comprensione ER alto livello; saltare entità/relazioni deboli per questo ambito esame.
- [ ] Usare euristiche insegnate per trasformare modelli ER in design logici e fisici; documentare scelte design quando esistono opzioni multiple.
- [ ] Correggere errori diagramma ER e attributi non necessari prima di condividere materiale; aggiornare cardinalità (es. relazioni Biglietto intese 1-a-N).
- [ ] Implementare vincoli integrità per prevenire periodi esposizione opere d'arte sovrapposte.
- [ ] Decidere tipi gerarchia per Artista e Cibo→Dolci basati su assunzioni dominio raffinate.
- [ ] Preparare materiali riferimento su implementazione ISA e semantiche sovrapposto/esclusivo.
- [ ] Programmare discussione follow-up per tipi gerarchia e mappatura relazionale attributi relazione.
- [ ] Incontrarsi in due settimane per sessione Neo4j.