## Contesto Sessione e Obiettivi
- Sessione di classe mantenuta a causa di slot temporali limitati rimanenti; cancellazione evitata per mantenere programma intatto.
- Obiettivo: introdurre approcci big data basati su database non relazionali, iniziando con database a grafi.
- Ambito serie: passare da assunzioni relazionali legacy a modelli flessibili adatti per requisiti big data (varietà, scala, schemi evolutivi, nuovi pattern di query).

## Terminologia e Chiarezza Concettuale: "NoSQL"
- "NoSQL" è ambiguo; interpretazioni multiple:
  - No-SQL (non-SQL): sistemi che non usano SQL come linguaggio di query (es. database relazionale italiano degli anni '90 senza SQL).
  - Non relazionale (uso industriale primario): database non basati sul modello relazionale; nessuna tabella; strutture dati alternative.
  - Not-only-SQL: sistemi che supportano SQL tra altri paradigmi.
  - "NewSQL" e varianti correlate: uso più recente di SQL su architetture non relazionali o ibride.
- Conclusione: "NoSQL" manca di significato tecnico preciso; preferire famiglie concrete di database non relazionali.

## Prospettiva Storica
- Database non relazionali precedono sistemi relazionali (modelli gerarchici anni '60 e altri, es. NASA).
- Metà anni '70 a ~2000: database relazionali dominano (~99.99% quota di mercato).
- Circa 2000-2010: rivoluzione big data con nuove famiglie—grafi (~2000), Bigtable/famiglie colonne (~2004), documenti (~2005-2006), chiave-valore, colonnari, ecc. (hashtag "NoSQL" emerge alla fine del decennio).
- Dinamiche di mercato:
  - Driver: problemi dati su scala massiva in aziende web-scale senza soluzioni off-the-shelf adatte.
  - Originatori: Google, LinkedIn, Facebook, Twitter, Netflix, Amazon.
  - Collaborazione: co-sviluppo inter-aziendale, spesso open source.
  - Fornitori DB aziendali tradizionali (Microsoft, Oracle, IBM, SAS, SAP) largamente assenti dalla prima ondata innovativa.

## Inquadramento Tecnico Mirato: Famiglie di Database Non Relazionali
- Quattro famiglie primarie da coprire in profondità (teoria, modellazione, implementazione, codifica, strumenti):
  1. Database chiave-valore
  2. Database colonne/famiglie-colonne/stile-Bigtable
  3. Database documenti
  4. Database grafi
- Due famiglie aggiuntive (copertura livello superiore):
  - Database basati su ricerca/Information Retrieval (IR)
  - Database vettoriali
- Nota: L'ordine è arbitrario; database relazionali rimangono rilevanti ma insufficienti per nuovi requisiti.

## Concetto Centrale: Grafi come Tipi di Dati Astratti
- I grafi sono tipi di dati astratti con nodi (vertici) e archi, accanto ad altre strutture CS (tabelle, liste).
- Distinguere modello astratto da rappresentazioni pratiche in-memoria/data-store.

## Fondamenti Teoria dei Grafi per Pensiero Database
- Definizione grafo: vertici e archi, con mappature da archi a vertici incidenti; rappresentato visualmente o testualmente.
- Tipi e proprietà:
  - Diretto vs non diretto
  - Connettività (connesso, fortemente connesso, disconnesso)
  - Densità (sparso vs denso)
  - Cicli, sottografi irraggiungibili
- Significato e applicazioni:
  - Radici più antiche della CS (es. Sette Ponti di Königsberg di Eulero).
  - Problemi canonici: Commesso Viaggiatore, colorazione grafi.
  - Usi industriali: EDA, navigazione GPS, pianificazione, ottimizzazione, allocazione risorse, robotica (grafi visibilità), networking/Internet.
- Arricchire grafi con dati:
  - Pesi archi e proprietà su nodi/archi (numerici, testo, etichette) abilitano modellazione ricca e ottimizzazione (es. costo diretto 5 vs tramite nodo costo 3).
  - Analisi strutturale: grado nodo come proxy centralità (influencer vs utenti ordinari).
  - Sottografi: estrarre sottoinsiemi è centrale per query e analisi.

## Implementazioni Grafi: Opzioni e Compromessi
- Approcci implementazione:
  - Matrici adiacenza o incidenza
  - Liste adiacenza
  - Set/collezioni statiche di nodi/archi
- Sfida: scalabilità
  - Grafi piccoli sono triviali; grafi massivi (es. miliardi di nodi, centinaia miliardi di archi) richiedono strutture altamente ottimizzate.
  - Implementazioni efficienti e scalabili sono poche; soluzioni custom raramente soddisfano prestazioni web-scale.

## Database Grafi: Scopo e Vantaggi
- Costruiti per fornire implementazioni modello grafi scalabili e affidabili per dataset enormi (reti sociali, grafi web).
- Bisogno chiave:
  - Database relazionali inadatti per memorizzare e attraversare relazioni semantiche nonostante il nome "relazionale".
  - Modellare relazioni in DB relazionali richiede chiavi, tabelle ponte, e join—innaturale e potenzialmente inefficiente.
  - Esempio: noleggi—tre tabelle e due join vs connessione diretta (Marinaio)-[:RISERVA]->(Barca) in un grafo.
- Cambio concettuale:
  - Adiacenza diretta index-free (riferimenti pointer-like) sostituisce percorsi join.
  - Benefici: modellazione intuitiva relazioni-first, attraversamento efficiente, e query più semplici.

## Query Grafi: Perché SQL Non Funziona e Pattern Matching
- SQL è inadatto:
  - Nessuna tabella o schema fisso; grafi evolvono dinamicamente con forme arbitrarie.
- Alternativa: pattern matching (matching grafi)
  - Esprimere query come pattern forma-grafo più condizioni.
  - Esecuzione trova corrispondenze sottografi nel grafo più grande.
  - Risultati sono sottografi; robusti a evoluzione schema/topologia.

## Focus Tecnologico: Panoramica Neo4j
- Database grafi nativo rappresentativo, originario circa 2000 (Neo Technologies); implementato in Java; open source.
- Proprietà:
  - Storage grafi nativo (non basato su relazionale)
  - Garanzie transazionali ACID; risultati query consistenti
  - Deployment e uso facili
  - Schema-free (senza schema):
    - Nodi/archi creati in qualsiasi momento con attributi e connessioni arbitrarie
    - Nessun vincolo attributi/relazioni predefinito
    - Alta flessibilità; richiede disciplina modellazione per evitare caos
- Ambito:
  - Progettato per gestione dati e query su nodi o piccoli sottografi
  - Non per analisi statistiche intero-grafo; usare strumenti analisi specializzati per computazioni globali

## Modellazione Schema-Free: Etichette, Attributi e Flessibilità
- Etichette (non tipi):
  - Nodi possono avere zero, una, o multiple etichette (es. Attore, Regista)
  - Etichette aiutano categorizzazione; non impongono forma o vincoli
  - Stessa etichetta può estendersi su nodi con attributi diversi
- Attributi:
  - Proprietà chiave-valore arbitrarie; tipi dati dedotti dai valori
  - Nessun set attributi obbligatorio; forme possono variare liberamente
- Guida:
  - Stabilire governance e convenzioni per prevenire modelli inconsistenti

## Cypher: Linguaggio Query Grafi
- Linguaggio dichiarativo per Neo4j, analogo a manipolazione/query dati SQL; evolvendo verso ISO GQL.
- Notazione grafi testuale:
  - Nodi: (alias:Etichetta {attributi})
  - Relazioni: (a)-[relEtichetta]->(b)
- Esempi CREATE:
  - Nodo: CREATE (neo:Crew {name: 'Neo'})
  - Relazione: CREATE (neo)-[:KNOWS]->(morph) (richiede morph definito o trovato)
- Alias:
  - Identificatori temporanei dentro una query; non persistiti
  - Necessari per riferire elementi attraverso clausole nella stessa istruzione
- Etichette multiple:
  - CREATE (n:Attore:Regista {name: 'Clint Eastwood'})
- Nessun DDL:
  - Creazione schema assente; manipolazione dati diretta e matching
- Import bulk:
  - LOAD CSV FROM 'file.csv' AS row
    CREATE (:Customer {companyName: row.companyName, customerID: row.customerID, phone: row.phone})

## Modello Prestazioni: Adiacenza Index-Free
- Archi codificano adiacenza diretta pointer-like tra nodi.
- Attraversamenti saltano nodo-a-nodo senza lookup indici o computazione join.
- Abilita prestazioni, scalabilità, ed esplorazione percorsi intuitiva.

## Guida Modellazione Pratica
- Usare etichette per categorizzazione; non assumere vincoli strutturali dalle etichette.
- Definire convenzioni minimali:
  - Attributi comuni (es. ID), tipi relazioni chiari, direzionalità
- Governare proliferazione attributi per mantenere consistenza.
- Obiettivi query:
  - Favorire pattern matching piccoli sottografi
  - Integrare strumenti specializzati per analisi intero-grafo

## Caso d'Uso Esempio Riformulato: Noleggi
- Relazionale:
  - Tabelle Marinai, Barche, Prenotazioni; join per risolvere chi prenota quale barca
- Grafo:
  - Nodi: Marinaio, Barca
  - Relazione: (Marinaio)-[:PRENOTA {tempo: ...}]->(Barca)
  - Attraversamento diretto evita join multi-tabella; naturale ed efficiente per query relazioni

## Note Linguaggio e Standardizzazione
- Cypher è dichiarativo e focalizzato su pattern.
- Standardizzazione ISO in corso per GQL (Graph Query Language) derivato da concetti Cypher.

## Dati Chiave, Fatti, Esempi e Conclusioni
- Dati/fatti:
  - DB non relazionali esistevano negli anni '60; monopolio relazionale da metà anni '70 a ~2000.
  - 2000-2010: emergenza database grafi, Bigtable/colonne, documenti, chiave-valore; esempi notevoli includono Bigtable, Dynamo, MongoDB.
  - Open source divenne norma per nuovi sistemi.
  - Fornitori DB aziendali tradizionali largamente non furono originatori.
- Esempi:
  - DB relazionale italiano anni '90 senza SQL (letterale "no SQL")
  - Esempio costo percorso peso-arco (5 diretto vs 3 tramite nodo intermedio)
  - Influencer social media vs utente ordinario tramite grado nodo
- Conclusioni:
  - "NoSQL" è ampio e ambiguo; usare nomi famiglie precise.
  - Modelli non relazionali affrontano bisogni flessibilità/scala non soddisfatti da sistemi relazionali circa 2000.
  - Teoria grafi sottende database grafi; proprietà su nodi/archi trasformano grafi matematici in modelli dati.
  - Database grafi abilitano modellazione relazioni-first e query attraversamento-centriche, partenza pulita dal pensiero relazionale.

## Piano Corso per Oggi: Database Grafi
- Cambio cardine da concetti relazionali: accantonare tabelle, join, SQL; adottare gestione dati basata su grafi.
- Piano copertura:
  - Fondamenti teoria grafi minimi (completati a livello introduttivo)
  - Implementazioni database grafi (modellazione, storage, query)

## Note Operative
- Neo4j mantiene garanzie ACID; risultati query consistenti.
- Deployment è user-friendly; disponibilità open-source facilita prova immediata.

## 📅 Prossimi Accordi ed Elementi d'Azione
- [ ] Procedere con modulo database grafi: passare da teoria grafi fondamentale a dettagli implementazione database grafi.
- [ ] Fornire sessioni dedicate per le quattro famiglie non relazionali principali (chiave-valore, colonne/Bigtable, documenti, grafi): teoria, modellazione, implementazione, codifica e strumenti.
- [ ] Includere sessioni panoramica per database basati su ricerca/IR e database vettoriali più avanti nel corso.
- [ ] Mantenere calendario classe senza cancellazioni a causa di slot disponibili limitati.
- [ ] Enfatizzare terminologia precisa attraverso materiali: usare nomi famiglie invece del ambiguo "NoSQL."
- [ ] Nessuna lezione giovedì 2 ottobre; riconvocarsi settimana prossima.
- [ ] Continuare copertura Neo4j settimana prossima (salvare dati, uso linguaggio query, esempi pratici).
- [ ] Compito opzionale: guardare The Matrix (contesto denominazione per Neo, Cypher).
- [ ] Preparare sorgenti CSV o dati legacy per demo LOAD CSV.
- [ ] Stabilire linee guida modellazione per gestire flessibilità schema-free (etichette, attributi, tipi relazioni).
- [ ] Identificare requisiti analisi intero-grafo e selezionare strumenti appropriati separati da Neo4j.