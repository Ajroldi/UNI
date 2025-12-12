# Lezione 5 - 1 Ottobre: UML Use Case e Class Diagram

## 📑 Indice

1. [Introduzione](#introduzione) `00:00:13`
2. [UML Use Case Diagram - Admission Manager](#use-case-admission) `00:01:27 - 00:14:20`
   - [Identificazione degli Attori](#identificazione-attori)
   - [Casi d'Uso del Genitore](#casi-uso-genitore)
   - [Casi d'Uso dell'Amministratore Scolastico](#casi-uso-admin)
   - [Relazioni tra Attori e Use Case](#relazioni-attori)
3. [Esercizio Fast Pop Coin](#esercizio-fast-pop-coin) `00:14:20 - 00:43:05`
   - [Descrizione del Sistema](#descrizione-sistema-fpc)
   - [Scenari Principali](#scenari-principali)
   - [Condizioni Speciali e Classi Utenti](#condizioni-speciali)
4. [Distinzione Mondo-Macchina di Jackson](#jackson-mondo-macchina) `00:17:41 - 00:22:48`
   - [Fenomeni Solo del Mondo](#fenomeni-mondo)
   - [Fenomeni Condivisi](#fenomeni-condivisi)
   - [Controllo: Mondo vs Macchina](#controllo-mondo-macchina)
5. [UML Class Diagram - Fast Pop Coin](#class-diagram-fpc) `00:22:48 - 00:32:29`
   - [Entità Principali](#entita-principali)
   - [Gerarchia degli Utenti](#gerarchia-utenti)
   - [Relazioni tra Entità](#relazioni-entita)
6. [UML Use Case Diagram - Fast Pop Coin](#use-case-fpc) `00:32:29 - 00:40:41`
   - [Attori del Sistema](#attori-sistema-fpc)
   - [Casi d'Uso Principali](#casi-uso-principali-fpc)
   - [Partecipazione e Iniziazione](#partecipazione-iniziazione)
7. [Esercizio Milk Vending Machine](#milk-vending-machine) `00:40:41 - fine`
   - [Specifiche del Sistema](#specifiche-vending)
   - [Regole di Funzionamento](#regole-funzionamento)
   - [Quiz Interattivo](#quiz-interattivo)

---

## <a name="introduzione"></a>1. Introduzione

`00:00:13 - 00:01:27`

> **📝 Nota Iniziale**
> 
> Questa lezione si concentra sulla pratica di modellazione UML, in particolare:
> - **Use Case Diagram**: Per catturare i requisiti funzionali
> - **Class Diagram**: Per modellare il dominio del sistema
> - **Distinzione Mondo-Macchina**: Framework di Jackson per analizzare fenomeni
> 
> Ricordarsi sempre di avviare la registrazione all'inizio delle lezioni!

Oggi affronteremo tre esercizi pratici:
1. **Admission Manager** - Sistema di gestione domande scolastiche
2. **Fast Pop Coin** - Sistema di monete virtuali per festival musicali
3. **Milk Vending Machine** - Distributore automatico di latte con regole speciali

---

## <a name="use-case-admission"></a>2. UML Use Case Diagram - Admission Manager

### <a name="identificazione-attori"></a>2.1 Identificazione degli Attori

`00:01:27 - 00:03:29`

Il primo passo fondamentale per disegnare un **UML Use Case Diagram** è identificare gli **attori** coinvolti nel sistema.

> **💡 Regola Pratica**
> 
> Per identificare gli attori:
> 1. Leggi attentamente il testo dei requisiti
> 2. Cerca i **sostantivi** che rappresentano **agenti** delle azioni
> 3. Identifica chi **interagisce direttamente** con il sistema

#### Attori Identificati nel Testo

**Dal testo dei requisiti**:
1. **Genitore** (Parent) - Attore più ovvio, utente principale del sistema
2. **Amministratore Scolastico** (School Administrator) - Gestisce le domande

**Dalla soluzione proposta**:
3. **Amministratore di Sistema** (System Administrator) - Non presente esplicitamente nel testo, ma necessario per gestire gli amministratori scolastici

> **⚠️ Inconsistenza Rilevata**
> 
> Ci sono alcune **inconsistenze** tra i requisiti forniti e il diagramma use case nella soluzione. Le analizzeremo insieme durante la lezione per capire come gestirle.

### <a name="casi-uso-genitore"></a>2.2 Casi d'Uso del Genitore

`00:03:29 - 00:09:24`

> **📐 Regola Generale**
> 
> Una volta completati i requisiti, **ogni requisito** mappa approssimativamente a un **singolo use case**.
> 
> Per ora creiamo un diagramma semplice, senza ereditarietà o relazioni complesse (approfondiremo nelle prossime lezioni).

#### Use Case 1: Register (Registrazione)

`00:04:00 - 00:04:39`

**Requisito 1.2**: "I genitori devono registrarsi nel sistema"

```
Use Case: Register
Attore: Parent
Label: initiate
```

**Significato della label "initiate"**:
- Il genitore è il **primo attore** coinvolto nel flusso di eventi
- È lui che **avvia** l'azione di registrazione

#### Use Case 2: Login (Accesso)

`00:04:39 - 00:05:17`

**Requisito**: "Admission Manager permette ai genitori di effettuare il login nel sistema"

```
Use Case: Login
Attore: Parent
Label: initiate
```

Il genitore **inizia** anche questo use case, essendo l'unico attore coinvolto.

#### Use Case 3: Send Application (Invio Domanda)

`00:05:17 - 00:06:36`

**Requisito 1.5**: "Admission Manager permette ai genitori di inviare una domanda per la scuola"

```
Use Case: Send Application
Attori: 
  - Parent (initiate)
  - School Administrator (participates)
```

> **🔍 Differenza Importante**
> 
> A differenza di Register e Login che hanno un **solo partecipante**, qui abbiamo:
> - **Parent**: initiate (avvia l'azione)
> - **School Administrator**: participates (è presente nel flusso di eventi)
> 
> **Perché?** Quando la domanda viene inviata, lo School Administrator riceve una **notifica**. Quindi è coinvolto nel flusso di eventi anche se non è lui ad avviare l'azione.

#### Use Case 4: Check Application Status (Controlla Stato Domanda)

`00:07:44 - 00:08:15`

**Requisito**: Controllare lo stato della propria domanda

```
Use Case: Check Application Status
Attore: Parent (initiate)
```

#### Use Case 5: Withdraw Application (Ritira Domanda)

`00:08:15 - 00:08:52`

**Requisito**: "Admission Manager permette ai genitori di ritirare una domanda"

```
Use Case: Withdraw Application
Attore: Parent (initiate)
```

#### Use Case 6: Modify Profile (Modifica Profilo)

`00:08:52 - 00:09:24`

**Requisito originale**: "Indicare nel profilo che vogliono essere notificati"

**Generalizzazione proposta**: Invece di un use case troppo specifico per le notifiche, creiamo:

```
Use Case: Modify Profile
Attore: Parent (initiate)
```

> **💡 Best Practice di Modellazione**
> 
> Quando un requisito è **troppo specifico** (es. "modificare solo impostazione notifiche"), è meglio **generalizzare** a un use case più ampio (es. "modificare profilo") che include quella funzionalità e altre simili.

### <a name="casi-uso-admin"></a>2.3 Casi d'Uso dell'Amministratore Scolastico

`00:09:24 - 00:13:46`

#### Gestione degli Amministratori

`00:09:24 - 00:11:06`

**Requisito 1.1 (originale)**: "Amministratore di sistema apre finestre di applicazione"

**Interpretazione realistica**:

```
System Administrator → Add School Administrator
School Administrator → Set Application Window
```

> **🎯 Decisione di Design**
> 
> Invece di far usare allo School Administrator lo stesso use case "Register" dei genitori, è più realistico che:
> 1. Il **System Administrator** aggiunga lo School Administrator al sistema
> 2. Lo **School Administrator** poi configuri le finestre di applicazione
> 
> Questo riflette meglio la realtà operativa.

**⚠️ IMPORTANTE per l'esame**:

Nel contesto dell'esame, dovreste **attenervi strettamente alla descrizione del testo**. Evitate di disegnare elementi che non esistono nei requisiti. Questa interpretazione è una discussione didattica.

#### Use Case 7: Set Application Window

`00:11:06 - 00:11:37`

```
Use Case: Set Application Window
Attore: School Administrator (initiate)
```

#### Use Case 8: Retrieve Applications (Recupera Domande)

`00:11:37 - 00:12:50`

**Requisito**: Lo School Administrator deve poter recuperare le domande

```
Use Case: Retrieve Applications
Attore: School Administrator (initiate)
```

Questo è uno dei **use case principali** per l'amministratore scolastico.

#### Use Case 9: Leave Comments (Lascia Commenti)

`00:12:50`

**Requisito**: Possibilità di lasciare commenti sulle domande

```
Use Case: Leave Comments
Attore: School Administrator (initiate)
```

#### Use Case 10: Accept/Reject Application (Accetta/Rifiuta Domanda)

`00:12:50 - 00:13:46`

Probabilmente il **use case più importante** per lo School Administrator:

```
Use Case: Accept or Reject Application
Attori:
  - School Administrator (initiate)
  - Parent (participates)
```

> **📬 Partecipazione del Genitore**
> 
> Il Parent **partecipa** a questo use case perché nel flusso di eventi:
> - Lo School Administrator prende la decisione
> - Il Parent viene **notificato** dell'esito
> 
> Quindi entrambi sono coinvolti, anche se solo l'amministratore inizia l'azione.

### <a name="relazioni-attori"></a>2.4 Relazioni tra Attori e Use Case

`00:13:46 - 00:14:20`

#### Riepilogo delle Label

Nella lezione di oggi utilizziamo principalmente due label:

1. **initiate**: L'attore **avvia** il use case, è il primo a interagire
2. **participates**: L'attore è **coinvolto** nel flusso di eventi ma non lo avvia

#### Struttura Completa

```
System Administrator
  └─ initiate → Add School Administrator

Parent
  ├─ initiate → Register
  ├─ initiate → Login
  ├─ initiate → Send Application
  │   └─ School Administrator participates
  ├─ initiate → Check Application Status
  ├─ initiate → Withdraw Application
  ├─ initiate → Modify Profile
  └─ participates ← Accept/Reject Application

School Administrator
  ├─ initiate → Set Application Window
  ├─ initiate → Retrieve Applications
  ├─ initiate → Leave Comments
  ├─ initiate → Accept/Reject Application
  │   └─ Parent participates
  └─ participates ← Send Application
```

> **📚 Materiale Aggiuntivo**
> 
> Un diagramma use case completo e più leggibile sarà disponibile nelle slide del corso.

---

## <a name="esercizio-fast-pop-coin"></a>3. Esercizio Fast Pop Coin

### <a name="descrizione-sistema-fpc"></a>3.1 Descrizione del Sistema

`00:14:20 - 00:15:29`

> **🎵 Contesto: Sistema di Monete per Festival Musicale**
> 
> Un'azienda privata di sicurezza organizza eventi e vuole un'applicazione che gestisca l'emissione e le transazioni di monete nel contesto di un **festival musicale di medie dimensioni**.
> 
> **Obiettivo**: Permettere ai partecipanti del festival e agli operatori di spendere una quantità prestabilita di denaro in relativa sicurezza, **senza portare portafogli** e altri beni in giro per l'evento.

#### Vantaggi del Sistema

- **Sicurezza**: Nessun contante fisico da portare in giro
- **Controllo**: Quantità di denaro pre-allocata
- **Tracciabilità**: Tutte le transazioni registrate
- **Analisi**: Monitoraggio in tempo reale dei ricavi

### <a name="scenari-principali"></a>3.2 Scenari Principali

`00:14:52 - 00:15:29`

Il software deve gestire **almeno tre scenari**:

#### Scenario 1: Emissione Monete

**Modalità di acquisto**:
- Attraverso **sportelli cassa** (cashier desks) con cassieri
- Attraverso **ATM** (distributori automatici)

**Processo**:
1. L'utente dà denaro contante
2. Riceve monete virtuali in cambio
3. Con eventuali sconti in base alla categoria

#### Scenario 2: Cashback

**Operazione duale** all'emissione di monete:
- Restituzione del denaro non speso
- Conversione monete → contanti

#### Scenario 3: Tracciamento Transazioni

**Nei vari negozi del festival**:
- Registrazione di ogni acquisto
- Tracciamento spese con monete
- Analisi dei dati di vendita

### <a name="condizioni-speciali"></a>3.3 Condizioni Speciali e Classi Utenti

`00:15:29 - 00:17:08`

#### Gerarchia delle Classi di Utenti

Nell'ambito dell'emissione di monete esistono **quattro classi** di acquirenti:

| Classe Utente | Sconto | Descrizione |
|---------------|--------|-------------|
| **VIP** | 30% | Ospiti VIP dell'evento |
| **Event Organization People** | 50% | Personale dell'organizzazione |
| **Ticket Holder Class A** | 20% | Biglietto di classe A |
| **Regular Ticket Holder** | 0% | Biglietto standard |

> **💳 Processo di Autenticazione**
> 
> Quando acquista monete, l'utente deve:
> 1. **Autenticarsi** inserendo la propria carta d'identità nell'ATM o dandola al cassiere
> 2. Il sistema determina la **classe** di appartenenza
> 3. Applica lo **sconto appropriato**
> 4. L'utente ottiene le monete dopo aver inserito/dato la corrispondente quantità di denaro

#### Regole del Cashback

Nel contesto del cashback:
- L'utente deve autenticarsi con la **carta d'identità**
- Il sistema verifica la quantità di denaro appropriata da restituire
- **Considera ruolo e privilegi** per calcolare l'importo corretto

#### Tracciamento delle Vendite

`00:17:08 - 00:17:41`

Durante l'evento:
- Ogni **commesso del negozio** (shop clerk) tiene traccia attraverso Fast Pop Coin di:
  - Vendite di prodotti
  - Monete ricevute
  
Fast Pop Coin si affida a un **servizio analitico di terze parti** per:
- Controllare periodicamente se il festival sta guadagnando
- Eseguire **analisi costi-benefici**
- Confrontare costi dei prodotti venduti vs spese organizzative

---

## <a name="jackson-mondo-macchina"></a>4. Distinzione Mondo-Macchina di Jackson

`00:17:41 - 00:22:48`

### <a name="fenomeni-mondo"></a>4.1 Fenomeni Solo del Mondo

`00:17:41 - 00:19:17`

> **📐 Framework di Jackson-Zave**
> 
> La distinzione tra **mondo** (world) e **macchina** (machine) è fondamentale nell'ingegneria del software:
> - **Mondo**: Ambiente esterno, realtà fisica
> - **Macchina**: Sistema software che stiamo progettando
> - **Fenomeni condivisi**: Interfaccia tra mondo e macchina

#### Definizione di Fenomeno World-Only

> **💡 Regola Pratica**
> 
> Un fenomeno è **world-only** (solo del mondo) quando:
> 
> **NON può essere percepito dalla macchina**
> 
> Il sistema non può sapere se questo fenomeno è avvenuto o meno.

#### Esempi di Fenomeni World-Only in Fast Pop Coin

`00:18:12 - 00:19:48`

**1. Perché un utente compra un biglietto di Classe A**

```
Fenomeno: Un utente acquista un biglietto di Classe A
Tipo: World-only
Motivo: Il sistema non è connesso al processo decisionale dell'utente
```

Il sistema **non sa**:
- Le motivazioni dell'acquisto
- Il processo mentale dell'utente
- Perché ha scelto Classe A invece di Regular

**2. Contrattazione di un VIP per l'evento**

```
Fenomeno: Un VIP viene contrattato per l'evento
Tipo: World-only
Motivo: Accordo contrattuale esterno al sistema
```

Il sistema **non sa**:
- Se un VIP specifico è stato contrattato
- I termini del contratto
- Le negoziazioni avvenute

**3. Inizio dell'evento**

```
Fenomeno: L'evento inizia fisicamente
Tipo: World-only
Motivo: Decisione organizzativa non tracciata dal sistema
```

**4. Azioni fisiche nel mondo reale**

```
Fenomeno: Un utente dà fisicamente denaro al cassiere
Tipo: World-only
Motivo: Azione fisica non direttamente osservabile dal sistema
```

L'atto **fisico** di dare denaro avviene nel mondo reale. Il sistema può solo registrare l'**effetto** (denaro ricevuto), non l'azione stessa.

### <a name="fenomeni-condivisi"></a>4.2 Fenomeni Condivisi

`00:19:48 - 00:22:14`

I **fenomeni condivisi** (shared phenomena) sono osservabili **sia dal mondo che dalla macchina**.

Ora dobbiamo determinare se sono **controllati dal mondo** o **controllati dalla macchina**.

#### Esempio 1: Utente Inserisce Monete in ATM

`00:20:21 - 00:21:37`

```
Fenomeno: Un utente inserisce monete in un ATM
Tipo: Shared (condiviso)
Controllato da: World (mondo)
```

**Domanda**: Perché è condiviso?

**Risposta**: 
- Il sistema ha **visibilità** su questa azione
- L'ATM **riconosce** l'inserimento delle monete
- Il sistema può **registrare** l'evento

**Domanda**: Perché è controllato dal mondo?

**Risposta**:
- L'azione è **iniziata dall'utente** (attore del mondo)
- L'utente decide **quando** inserire le monete
- L'utente decide **quante** monete inserire
- Il sistema **reagisce** all'azione, non la controlla

> **🎯 Principio Chiave**
> 
> Un fenomeno condiviso è **world-controlled** quando:
> - È **iniziato** da un attore del mondo (persona, evento fisico)
> - Il sistema può **osservarlo** ma non può **impedirlo o iniziarlo**
> - Il sistema **risponde** al fenomeno

#### Esempio 2: Sistema Abilita Cashback

`00:21:37 - 00:22:14`

```
Fenomeno: Il sistema abilita il cashback dopo aver controllato 
          la carta d'identità e il numero di monete inserite
Tipo: Shared (condiviso)
Controllato da: Machine (macchina)
```

**Domanda**: Perché è controllato dalla macchina?

**Risposta**:
- È la **macchina** che ha il controllo sul flusso di eventi
- Il sistema **decide** se abilitare il cashback
- Il sistema esegue le **verifiche** (ID card, monete)
- Il sistema **autorizza** l'operazione

> **🎯 Principio Chiave**
> 
> Un fenomeno condiviso è **machine-controlled** quando:
> - È **iniziato o deciso** dal sistema
> - Il sistema ha **autorità decisionale**
> - Il mondo può **osservarlo** ma non controllarlo direttamente

#### Esempio 3: Carta d'Identità Inserita in ATM

`00:22:14 - 00:22:48`

```
Fenomeno: Una carta d'identità viene inserita in un ATM
Tipo: Shared (condiviso)
Controllato da: World (mondo)
```

**Motivazione**:
- Azione **fisica** dell'utente
- Sistema può **leggere** la carta
- Sistema può **riconoscere** l'inserimento
- Ma è l'**utente** che controlla quando e quale carta inserire

### <a name="controllo-mondo-macchina"></a>4.3 Controllo: Mondo vs Macchina

`00:22:48`

#### Schema di Classificazione

```
┌─────────────────────────────────────────────────────────────┐
│                  CLASSIFICAZIONE FENOMENI                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  WORLD-ONLY                                                   │
│  ├─ Non osservabile dalla macchina                          │
│  ├─ Decisioni umane, motivazioni                            │
│  └─ Eventi fisici non strumentati                           │
│                                                               │
│  SHARED - WORLD CONTROLLED                                   │
│  ├─ Osservabile da entrambi                                 │
│  ├─ Iniziato da attori del mondo                           │
│  ├─ Sistema reagisce/registra                               │
│  └─ Esempio: utente inserisce carta, preme bottone         │
│                                                               │
│  SHARED - MACHINE CONTROLLED                                 │
│  ├─ Osservabile da entrambi                                 │
│  ├─ Iniziato/deciso dal sistema                            │
│  ├─ Mondo riceve output                                     │
│  └─ Esempio: sistema autorizza, sistema dispensa          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

> **📚 Importanza dell'Analisi**
> 
> Questa distinzione è fondamentale per:
> - Definire i **confini del sistema**
> - Identificare le **responsabilità** del software
> - Progettare le **interfacce** correttamente
> - Scrivere **requisiti chiari**

---

## <a name="class-diagram-fpc"></a>5. UML Class Diagram - Fast Pop Coin

`00:22:48 - 00:32:29`

### <a name="entita-principali"></a>5.1 Entità Principali

`00:22:48 - 00:25:30`

> **🎯 Obiettivo del Class Diagram**
> 
> In questa fase iniziale, il class diagram è usato in modo **preliminare**:
> - **NON** stiamo modellando unità di implementazione dettagliate
> - **NON** stiamo fornendo il diagramma finale per gli sviluppatori
> - **STIAMO** descrivendo il **dominio** in cui il sistema opera
> 
> È un uso diverso da quello standard dell'UML Class Diagram!

#### Approccio: Identificare i Sostantivi

`00:24:30 - 00:25:30`

> **💡 Strategia di Modellazione**
> 
> Il modo migliore per procedere con un class diagram di dominio è:
> 
> 1. Ragionare sulle **entità più importanti** (sostantivi nel testo)
> 2. Identificare cosa viene **manipolato** dagli attori
> 3. Creare classi per ogni **concetto significativo**
> 
> Per ora **ignoriamo**:
> - Attributi dettagliati
> - Metodi
> - Molteplicità precise
> 
> Li vedremo nelle prossime lezioni!

#### Entità Centrale: User

`00:25:30`

**L'entità più importante**: **User** (Utente)

```
┌──────────┐
│   User   │
└──────────┘
```

L'utente è al **centro** del nostro sistema di monete.

### <a name="gerarchia-utenti"></a>5.2 Gerarchia degli Utenti

`00:25:30 - 00:27:16`

L'utente è caratterizzato da una **gerarchia** - ci sono diversi tipi di utenti con privilegi diversi.

#### Specializzazioni dell'Utente

```
                  ┌──────────┐
                  │   User   │
                  └────┬─────┘
                       │
           ┌───────────┼───────────┐
           │           │           │
      ┌────▼────┐ ┌───▼────┐ ┌───▼──────────┐
      │   VIP   │ │ Event  │ │Ticket Holder │
      │         │ │Organiz.│ └──────┬───────┘
      └─────────┘ └────────┘        │
                             ┌───────┴────────┐
                             │                │
                      ┌──────▼──────┐  ┌─────▼────┐
                      │Class A      │  │ Regular  │
                      │Ticket Holder│  │  Ticket  │
                      └─────────────┘  │  Holder  │
                                       └──────────┘
```

**Specializzazioni principali**:
1. **VIP**: Sconto 30%
2. **Event Organizer**: Sconto 50%
3. **Ticket Holder**: Categoria generale che si specializza ulteriormente in:
   - **Class A Ticket Holder**: Sconto 20%
   - **Regular Ticket Holder**: Nessuno sconto

> **📝 Nota sulla Modellazione**
> 
> Questa gerarchia riflette le **classi di sconto** descritte nei requisiti. Usiamo l'**ereditarietà** per modellare queste specializzazioni.

### <a name="relazioni-entita"></a>5.3 Relazioni tra Entità

`00:27:16 - 00:32:29`

#### Oggetti Manipolati dall'Utente

`00:27:16 - 00:28:24`

Leggendo il testo, identifichiamo gli oggetti con cui l'utente interagisce:

**1. Money (Denaro)**

```
┌──────────┐         ┌───────┐
│   User   │────────>│ Money │
└──────────┘   dà    └───────┘
```

L'utente **dà denaro** agli ATM o ai cassieri.

**2. Coin (Moneta)**

```
┌──────────┐         ┌──────┐
│   User   │<────────│ Coin │
└──────────┘ ottiene └──────┘
```

L'utente **ottiene monete** in cambio del denaro.

> **🔍 Decisione di Modellazione**
> 
> Modelliamo **Money** e **Coin** come **due entità separate** perché:
> - Rappresentano concetti diversi (contanti fisici vs monete virtuali)
> - Hanno regole di conversione (con sconti)
> - Sono tracciati separatamente nel sistema

**3. Identity Card (Carta d'Identità)**

`00:28:24`

```
┌──────────┐         ┌──────────────┐
│   User   │────────>│Identity Card │
└──────────┘  mostra └──────────────┘
```

L'utente deve **mostrare la carta d'identità** per:
- Ottenere monete
- Ricevere cashback

> **💭 Alternativa di Modellazione**
> 
> In altri contesti potremmo modellare la Identity Card come un **attributo** dello User.
> 
> Ma qui, poiché stiamo modellando il dominio a **livello molto alto**, usiamo un'entità separata per ogni sostantivo significativo.

**4. Product (Prodotto)**

`00:28:58`

```
┌──────────┐         ┌─────────┐
│   User   │────────>│ Product │
└──────────┘  compra └─────────┘
```

L'utente **acquista prodotti** usando le monete.

**Possibile attributo** (non lo modelliamo ora):
- `cost`: costo del prodotto

#### Entità di Interazione

`00:29:01 - 00:30:41`

L'utente interagisce con altre entità per convertire denaro in monete:

**5. Cashier (Cassiere)**

```
┌──────────┐         ┌─────────┐
│   User   │<───────>│ Cashier │
└──────────┘         └─────────┘
```

Il cassiere:
- Riceve denaro dall'utente
- Dà monete all'utente
- Converte monete in denaro (cashback)

**6. ATM (Distributore Automatico)**

```
┌──────────┐         ┌─────┐
│   User   │<───────>│ ATM │
└──────────┘         └─────┘
```

L'ATM:
- Accetta denaro
- Dispensa monete
- Effettua cashback

> **📝 Nota nelle Slide**
> 
> Nelle slide, Cashier e ATM sono caratterizzati anche da **metodi** che rappresentano le operazioni di:
> - Conversione denaro → monete
> - Conversione monete → denaro
> 
> Per ora non li modelliamo esplicitamente.

#### Attori Aggiuntivi e Entità di Business

`00:30:41 - 00:32:29`

**7. Shop Clerk (Commesso del Negozio)**

```
┌────────────┐
│ Shop Clerk │
└──────┬─────┘
       │
       │ registra
       │
   ┌───▼────┐
   │  Sale  │
   └───┬────┘
       │
       │ monitora
       │
┌──────▼───────────┐
│Analytics Service │
└──────────────────┘
```

**Perché modelliamo esplicitamente Sale?**

`00:31:15 - 00:31:52`

Il testo dice:

> "Fast Pop Coin si affida a un servizio analitico di terze parti per controllare periodicamente se il festival sta guadagnando"

Questo significa che:
- Ogni **vendita** è **monitorata**
- Le vendite sono **entità** con una loro esistenza
- Non sono solo transazioni effimere

**8. Analytics Service (Servizio Analitico)**

Il servizio analitico:
- È **direttamente connesso** alle vendite (Sale)
- Analizza i dati di vendita
- Genera report di costi-benefici

#### Diagramma Completo delle Entità

`00:31:52 - 00:32:29`

```
                    ┌──────────────────┐
                    │ System           │
                    │ Administrator    │
                    └──────────────────┘

                    ┌──────────┐
                    │   User   │
                    └────┬─────┘
                         │
         ┌───────────────┼────────────────┐
         │               │                │
    ┌────▼────┐   ┌──────▼──────┐  ┌────▼──────────┐
    │   VIP   │   │   Event     │  │Ticket Holder  │
    └─────────┘   │ Organizer   │  └───────┬───────┘
                  └─────────────┘          │
                                    ┌──────┴──────┐
                                    │             │
                             ┌──────▼────┐ ┌─────▼────────┐
                             │  Class A  │ │   Regular    │
                             └───────────┘ └──────────────┘

┌──────────┐         ┌───────┐         ┌──────┐
│   User   │────────>│ Money │         │ Coin │<───────┐
└────┬─────┘         └───────┘         └──────┘        │
     │                                                  │
     │               ┌──────────────┐                   │
     └──────────────>│Identity Card │                   │
     │               └──────────────┘                   │
     │                                                  │
     │               ┌─────────┐                        │
     └──────────────>│ Product │                        │
     │               └─────────┘                        │
     │                                                  │
     ├──────────────>┌─────────┐──────────────────────┘
     │               │ Cashier │
     │               └─────────┘
     │
     └──────────────>┌─────┐──────────────────────────┐
                     │ ATM │                          │
                     └─────┘                          │
                                                      │
                                                      │
┌────────────┐       ┌──────┐       ┌─────────────────▼─┐
│ Shop Clerk │──────>│ Sale │<──────│ Analytics Service │
└────────────┘       └──────┘       └───────────────────┘
```

> **✅ Completamento**
> 
> Abbiamo modellato l'**intera immagine** in termini di entità del dominio!
> 
> Nelle prossime lezioni aggiungeremo:
> - Molteplicità sulle relazioni
> - Etichette sugli archi
> - Attributi e metodi
> - Vincoli e note

---

## <a name="use-case-fpc"></a>6. UML Use Case Diagram - Fast Pop Coin

`00:32:29 - 00:40:41`

### <a name="attori-sistema-fpc"></a>6.1 Attori del Sistema

`00:33:40 - 00:35:33`

> **💡 Strategia**
> 
> Il primo passo è sempre **leggere il testo** e identificare gli attori.
> 
> In questo caso, abbiamo già fatto metà del lavoro con il **class diagram**! Molti attori corrispondono alle entità identificate.

#### Attori Identificati

Dal class diagram e dai requisiti:

1. **User** (Utente)
   - Attore principale del sistema
   - Include tutte le specializzazioni (VIP, Event Organizer, Ticket Holders)

2. **Cashier** (Cassiere)
   - Opera agli sportelli fisici
   - Gestisce conversioni manuali

3. **Shop Clerk** (Commesso del Negozio)
   - Registra le vendite
   - Interagisce durante gli acquisti

4. **Analytics Service** (Servizio Analitico)
   - Entità **non connessa** direttamente agli altri attori
   - Sistema di terze parti
   - Riceve dati dal sistema

> **🔍 Osservazione**
> 
> L'Analytics Service è interessante perché:
> - È un attore **esterno**
> - Non è una persona fisica
> - È un **sistema** che interagisce con il nostro sistema

### <a name="casi-uso-principali-fpc"></a>6.2 Casi d'Uso Principali

`00:35:33 - 00:40:41`

Ora identifichiamo le **azioni più importanti** che devono essere eseguite.

> **📐 Domanda Chiave**
> 
> Dal punto di vista del nostro sistema, quali sono i **requisiti più importanti**?
> Quali **azioni** devono essere supportate?

#### Use Case 1 e 2: Change Coins/Money through ATM

`00:35:33 - 00:37:00`

**Cambio Monete tramite ATM**:

```
Use Case: Change Coins through ATM
Attore: User (initiate)
Descrizione: L'utente usa l'ATM per convertire denaro in monete
```

**Cambio Denaro tramite ATM** (Cashback):

```
Use Case: Change Money through ATM
Attore: User (initiate)
Descrizione: L'utente usa l'ATM per convertire monete in denaro
```

> **📝 Nota**
> 
> In entrambi i casi:
> - Solo **User** coinvolto
> - Label: **initiate**
> - Nessun altro attore **partecipa** (l'ATM è lo strumento, non un attore separato nel flusso)

#### Use Case 3 e 4: Change Coins/Money through Cashier

`00:37:00 - 00:38:57`

**Cambio Monete tramite Cassiere**:

```
Use Case: Change Coins through Cashier
Attori:
  - Cashier (initiate)
  - User (participates)
```

**Cambio Denaro tramite Cassiere** (Cashback):

```
Use Case: Change Money through Cashier
Attori:
  - Cashier (initiate)
  - User (participates)
```

> **🔄 Differenza Importante**
> 
> **Perché il Cashier inizia e lo User partecipa?**
> 
> Se ragioniamo sul **flusso di eventi**:
> 1. Il **Cashier** ha il **controllo** della transazione
> 2. Il Cashier **inizia** l'operazione nel sistema
> 3. Lo User è **coinvolto** (dà denaro, riceve monete) ma non controlla il sistema
> 
> È il **Cashier** che opera l'interfaccia del sistema!

Questo è l'**opposto** del caso ATM dove lo User ha il controllo diretto.

#### Use Case 5: Register Purchase

`00:38:57 - 00:39:34`

Una volta che l'utente va in un negozio e compra qualcosa:

```
Use Case: Register Purchase
Attori:
  - Shop Clerk (initiate)
  - User (participates)
```

**Perché questo use case?**

Dal class diagram e dalla descrizione:
- Dobbiamo **registrare ogni vendita** (Sale)
- Le vendite vengono analizzate dal servizio analitico
- Non sono transazioni "usa e getta"

**Flusso**:
1. Shop Clerk **inizia** la registrazione
2. User **partecipa** all'acquisto
3. Il sistema registra la Sale

#### Use Case 6: Send Data About Purchases

`00:39:34 - 00:40:41`

Il servizio analitico riceve dati sugli acquisti:

```
Use Case: Send Data About Purchases
Attore: Analytics Service (participates)
Iniziato da: Machine (sistema stesso)
```

> **🤔 Situazione Particolare**
> 
> Questo è un caso **abbastanza strano** perché:
> - **NON** è iniziato da un attore esterno
> - È iniziato dalla **macchina stessa** (sistema)
> - Il sistema invia periodicamente dati
> - L'Analytics Service solo **partecipa** (riceve i dati)

**Caratteristiche**:
- **Trigger automatico**: Il sistema decide quando inviare
- **Periodicità**: Controlli periodici menzionati nei requisiti
- **Flusso machine-initiated**: Raro ma possibile nei use case

### <a name="partecipazione-iniziazione"></a>6.3 Partecipazione e Iniziazione

#### Riepilogo Use Case Diagram

```
User
  ├─ initiate → Change Coins through ATM
  ├─ initiate → Change Money through ATM
  ├─ participates ← Change Coins through Cashier
  ├─ participates ← Change Money through Cashier
  └─ participates ← Register Purchase

Cashier
  ├─ initiate → Change Coins through Cashier
  │   └─ User participates
  └─ initiate → Change Money through Cashier
      └─ User participates

Shop Clerk
  └─ initiate → Register Purchase
      └─ User participates

Analytics Service
  └─ participates ← Send Data About Purchases
      (initiated by Machine)
```

> **✅ Completato**
> 
> Questo è il diagramma use case di **alto livello** per Fast Pop Coin!
> 
> Non abbiamo usato strumenti avanzati di UML (che vedremo nelle prossime lezioni), ma abbiamo catturato tutte le funzionalità principali del sistema.

---

## <a name="milk-vending-machine"></a>7. Esercizio Milk Vending Machine

`00:40:41 - fine`

### <a name="specifiche-vending"></a>7.1 Specifiche del Sistema

`00:40:41 - 00:42:32`

> **🥛 Esercizio: Distributore di Latte con Regole Speciali**
> 
> Un distributore automatico di latte con regole particolari di funzionamento.

#### Tipi di Monete Accettate

Il distributore accetta **tre tipi** di monete:

| Moneta | Valore |
|--------|--------|
| Quarter | $0.25 |
| Half Dollar | $0.50 |
| Dollar | $1.00 |

### <a name="regole-funzionamento"></a>7.2 Regole di Funzionamento

`00:40:41 - 00:43:05`

#### Regola 1: Ordine Crescente Obbligatorio

> **⚠️ Vincolo Importante**
> 
> Il distributore accetta monete **SOLO in ordine crescente**.

**Esempio**:
- ✅ Posso inserire: $0.25 → $0.50 → $1.00
- ✅ Posso inserire: $0.25 → $0.25 → $0.50
- ❌ NON posso inserire: $0.50 → $0.25 (ordine decrescente!)

**Eccezione**:
- Posso tornare a inserire monete più piccole **dopo aver richiesto il resto**
- Richiedere il resto **riavvia il processo**

#### Regola 2: Erogazione Automatica del Latte

```
SE denaro_inserito >= $1.00:
    THEN:
        - Eroga una bottiglia di latte
        - Sottrai $1.00 dal denaro nella macchina
        - Lascia il resto nella macchina
```

**Esempio**:
- Inserisco: $0.50 + $0.50 + $0.25 = $1.25
- Sistema:
  - ✅ Eroga bottiglia ($1.00)
  - 💰 Resto in macchina: $0.25
  - L'utente può continuare ad aggiungere denaro per altra bottiglia

#### Regola 3: Richiesta Resto

`00:42:00 - 00:42:32`

**In qualsiasi momento**, l'utente può richiedere il resto:

```
Azione: Request Change (Richiedi Resto)
Effetti:
  - Restituisce tutto il denaro non utilizzato
  - Può avvenire anche se denaro = $0
  - RIAVVIA il processo
  - Permette di inserire di nuovo monete piccole
```

**Esempio**:
1. Inserisco $0.50
2. Inserisco $1.00 (totale $1.50)
3. Ricevo bottiglia (resto $0.50 in macchina)
4. **NON** posso inserire $0.25 (ordine decrescente!)
5. **Richiedo resto** → ricevo $0.50
6. Ora **posso** inserire $0.25 (processo riavviato)

#### Regola 4: Fidelity Card

`00:42:32 - 00:43:05`

Il distributore accetta **carte fedeltà**:

```
CON carta fedeltà:
    Prezzo bottiglia = $0.75 invece di $1.00

SE carta_inserita E denaro >= $0.75:
    THEN:
        - Eroga bottiglia
        - Sottrai $0.75 dal denaro
        - Lascia resto in macchina
```

**Vantaggi**:
- **Sconto 25%** ($1.00 → $0.75)
- Stesso comportamento per il resto
- Stesse regole di ordine crescente

### <a name="quiz-interattivo"></a>7.3 Quiz Interattivo

`00:43:05 - fine`

> **🎮 Attività Pratica: Quiz Online**
> 
> Invece di lavorare alla lavagna, facciamo un quiz online interattivo!

#### Compito

`00:43:05 - 00:43:48`

Con riferimento alla **distinzione di Jackson tra mondo e macchina**:

**Domanda 1**: Identificare i fenomeni rilevanti per il Milk Vending Machine:
- Fenomeni **world-only**
- Fenomeni **shared**

**Domanda 2**: Per i fenomeni shared, specificare:
- Controllati dal **mondo**?
- Controllati dalla **macchina**?

Fornire una breve descrizione se necessario.

#### Modalità Quiz

`00:43:48 - 00:47:38`

**Setup**:
- Piattaforma online con sistema di quiz
- 3 minuti per registrarsi
- Nomi divertenti incoraggiati (per ridere insieme!)
- Riferimenti a Dante apparsi... 😄

**Incentivi**:
```
"Niente premi questa volta, ma ci sarà sicuramente una leaderboard!"
"Forse per la prossima volta ci saranno premi..."
```

**Atmosfera**:
- Competizione amichevole
- Focus sulla **top 5** della leaderboard
- Momento per svegliarsi (lezione alle 8:30!)

#### Esempi di Fenomeni da Analizzare

Possibili fenomeni da classificare:

**Fenomeni Potenzialmente World-Only**:
- Utente decide di comprare latte
- Utente ha una carta fedeltà fisica
- Motivazione per richiedere il resto

**Fenomeni Potenzialmente Shared-World**:
- Utente inserisce moneta
- Utente preme bottone "richiedi resto"
- Utente inserisce carta fedeltà

**Fenomeni Potenzialmente Shared-Machine**:
- Macchina eroga bottiglia
- Macchina restituisce resto
- Macchina rifiuta moneta (ordine sbagliato)
- Sistema calcola totale

---

## 📝 Note Finali e Concetti Chiave

### Riepilogo della Lezione

Abbiamo affrontato tre esercizi completi:

1. **Admission Manager**
   - Use case diagram con attori multipli
   - Relazioni initiate/participates
   - Gestione di inconsistenze nei requisiti

2. **Fast Pop Coin**
   - Distinzione mondo-macchina di Jackson
   - Class diagram di dominio (alto livello)
   - Use case diagram con attori umani e sistemi
   - Gerarchia di specializzazione (utenti con sconti)

3. **Milk Vending Machine**
   - Regole complesse di funzionamento
   - Quiz interattivo per apprendimento attivo

### 🎯 Concetti Chiave da Ricordare

#### UML Use Case Diagram

- **Identificare attori**: Leggere il testo, trovare agenti delle azioni
- **Mappare requisiti**: Un requisito ≈ un use case (regola generale)
- **Label initiate**: Attore che avvia il flusso di eventi
- **Label participates**: Attore coinvolto nel flusso ma che non inizia
- **Attenersi al testo**: Nell'esame, seguire strettamente i requisiti

#### Distinzione Mondo-Macchina

- **World-only**: NON percepibile dalla macchina
- **Shared-World**: Osservabile da entrambi, iniziato dal mondo
- **Shared-Machine**: Osservabile da entrambi, iniziato/deciso dalla macchina
- **Importanza**: Definisce confini e responsabilità del sistema

#### UML Class Diagram (Dominio)

- **Uso preliminare**: Descrivere il dominio, non implementazione dettagliata
- **Identificare entità**: Sostantivi significativi nel testo
- **Modellare relazioni**: Collegamenti tra entità
- **Gerarchie**: Specializzazioni quando ci sono categorie diverse
- **Semplificazione**: Per ora senza attributi, metodi, molteplicità

### 📚 Prossime Lezioni

Approfondiremo:
- **Strumenti avanzati** di UML class diagram
- **Molteplicità** e **vincoli** sulle relazioni
- **Attributi e metodi** delle classi
- **Pattern** di modellazione comuni
- **Relazioni** include/extend nei use case

### ⚠️ Consigli per l'Esame

1. **Leggere attentamente** la descrizione del problema
2. **Identificare** tutti gli attori prima di disegnare
3. **Mappare** requisiti a use case sistematicamente
4. **Non inventare** funzionalità non menzionate
5. **Distinguere** tra mondo e macchina per requisiti chiari
6. **Giustificare** le scelte di modellazione

---

**Fine Lezione 5**

*Prossima lezione: Approfondimento UML Class Diagram - Attributi, Metodi e Molteplicità*
