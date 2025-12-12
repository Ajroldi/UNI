# Lezione 6 - 2 Ottobre: State Machine, Activity Diagram e Introduzione ad Alloy

## 📑 Indice

1. [Introduzione e Riepilogo UML](#introduzione) `00:00:13 - 00:01:33`
2. [State Machine Diagram](#state-machine) `00:01:33 - 00:14:36`
   - [Concetti Base e Stati](#concetti-base-stati)
   - [Esempio ATM Bancario](#esempio-atm)
   - [Differenza tra State e Sequence Diagram](#differenza-state-sequence)
3. [Activity Diagram](#activity-diagram) `00:14:36 - 00:25:37`
   - [Processi e Flussi di Attività](#processi-flussi)
   - [Swim Lanes](#swim-lanes)
   - [Fork, Join e Decisioni](#fork-join-decisioni)
   - [Differenza tra Sequence e Activity Diagram](#differenza-sequence-activity)
4. [UML per Requirement Engineering - Riepilogo](#uml-requirement) `00:25:37 - 00:32:30`
   - [Quando Usare Ogni Diagramma](#quando-usare)
   - [Limitazioni di UML](#limitazioni-uml)
5. [Caso di Studio: Sistema Frenante Airbus](#caso-airbus) `00:32:30 - 00:45:29`
   - [Class Diagram del Sistema](#class-diagram-airbus)
   - [State Machine del Velivolo](#state-machine-airbus)
   - [Use Case e Sequence Diagram](#use-case-sequence-airbus)
   - [Limiti di UML per Requisiti Formali](#limiti-uml-requisiti)
6. [Introduzione ad Alloy](#introduzione-alloy) `00:45:29 - 00:59:00`
   - [Cos'è Alloy](#cosa-alloy)
   - [Risorse e Strumenti](#risorse-strumenti)
   - [Caratteristiche Principali](#caratteristiche-alloy)
   - [Alloy 6: Novità e Differenze](#alloy-6)
7. [Primo Esempio: Address Book](#esempio-address-book) `00:59:00 - fine`
   - [Definizione delle Signature](#signature-address-book)
   - [Relazioni e Cardinalità](#relazioni-cardinalita)
   - [Operazione di Join](#operazione-join)
   - [Esecuzione con Alloy Analyzer](#alloy-analyzer)

---

## <a name="introduzione"></a>1. Introduzione e Riepilogo UML

`00:00:13 - 00:01:33`

> **📝 Nota Iniziale**
> 
> Spero che oggi ci sia meno rumore rispetto all'ultima volta! La scorsa settimana ci siamo concentrati su come utilizzare UML nell'ambito della requirement engineering, in particolare:
> - **Use Case Diagram**
> - **Class Diagram** 
> - **Sequence Diagram**

Oggi completiamo questa parte focalizzandoci su:

1. **State Machine Diagram** - Per modellare gli stati delle entità
2. **Activity Diagram** - Per descrivere processi e flussi
3. **Esempio specifico** - Applicazione pratica
4. **Alloy** - Introduzione al linguaggio di specifica formale

---

## <a name="state-machine"></a>2. State Machine Diagram

### <a name="concetti-base-stati"></a>2.1 Concetti Base e Stati

`00:01:33 - 00:03:48`

> **🎯 Quando Usare State Machine Diagram**
> 
> I diagrammi di stato sono utili quando identifichiamo **una o più entità** nel dominio del problema per le quali è importante:
> - Analizzare tutti i **possibili stati**
> - Modellare le **transizioni tra stati**
> - Seguire il **ciclo di vita** dell'entità

#### Esempio Introduttivo: Entità "Problem"

Riprendiamo il class diagram di uno dei due esempi analizzati la scorsa settimana. In questo diagramma abbiamo un'entità chiamata **Problem** che è particolarmente importante per noi.

**Domanda chiave**: Quali sono gli stati possibili di questa entità Problem durante il suo ciclo di vita?

#### Diagramma di Stato Semplice

```
[●] → [Open] → [Taken Charge] → [Under Fixing] → [Solved] → [■]
```

**Elementi grafici**:
- **● (cerchio pieno)**: Stato iniziale
- **Rettangoli con angoli arrotondati**: Stati del sistema
- **Frecce**: Transizioni tra stati
- **■ (cerchio con punto)**: Stato finale/terminale

**Flusso dell'entità Problem**:
1. **Open** - Il problema è aperto (stato iniziale)
2. **Taken Charge** - Il problema è stato preso in carico
3. **Under Fixing** - Il problema è in fase di risoluzione
4. **Solved** - Il problema è risolto (stato finale)

> **💡 Significato degli Stati**
> 
> - Lo stato **Open** è evidenziato come iniziale dall'icona ●
> - Dopo **Solved**, lo state machine termina (icona ■)
> - Questo è un diagramma molto semplice, possiamo costruirne di più complessi!

### <a name="esempio-atm"></a>2.2 Esempio ATM Bancario

`00:03:48 - 00:13:28`

Vediamo ora un esempio più complesso che mostra fino a dove possiamo spingerci con gli state machine diagram.

> **🏧 Caso di Studio: Bank ATM**
> 
> Modelliamo il comportamento di un **ATM bancario** (Bancomat) attraverso i suoi possibili stati e transizioni.

#### Struttura Gerarchica degli Stati

`00:03:48 - 00:05:30`

**Stati con Sub-Stati**:

Uno degli aspetti più interessanti è lo stato **Serving Customer** (Servizio Cliente) che può essere **esploso** in sotto-stati:

```
Serving Customer
├─ [●] → Customer Authentication
├─ Selecting Transaction
├─ Transaction
└─ [■]
```

> **📐 Stati Composti**
> 
> Quando entriamo nello stato "Serving Customer" (grazie alla transizione appropriata):
> 1. Lo **state machine interno** si attiva
> 2. Entriamo immediatamente in **Customer Authentication** (stato iniziale interno)
> 3. Poi ci spostiamo in **Selecting Transaction**
> 4. Quindi in **Transaction**
> 5. Infine terminiamo

**Notazione speciale**:
- Gli stati **Customer Authentication** e **Transaction** hanno due piccole icone che indicano che sono **ulteriormente esplosi**
- La descrizione dettagliata dei loro sotto-stati è in **diagrammi separati**
- Questo evita che il diagramma diventi illeggibile

> **⚠️ Principio di Modularità**
> 
> A un certo punto non possiamo descrivere tutti i sotto-diagrammi nello stesso diagramma, altrimenti diventerebbe **incomprensibile**. Possiamo quindi differire la descrizione dello state diagram di un certo stato in un diagramma separato.

#### Annotazioni Testuali

`00:05:30 - 00:07:28`

Oltre a stati e frecce, abbiamo **elementi testuali** associati a:
- Le **frecce** (transizioni)
- Gli **stati** stessi

**Esempio: Stato "Serving Customer"**

```
Serving Customer
│
├─ entry / read card
├─ exit / eject card
│
└─ [sub-states: Customer Authentication → Selecting Transaction → Transaction]
```

> **📝 Entry e Exit Actions**
> 
> - **entry / read card**: Quando **entriamo** in questo stato, leggiamo la carta
> - **exit / eject card**: Quando **usciamo** da questo stato, espelliamo la carta
> 
> Queste azioni avvengono **prima/dopo** l'interazione con i sotto-stati!

**Flusso completo**:
1. Entriamo in "Serving Customer" → **leggiamo la carta**
2. Attraversiamo i sotto-stati (autenticazione, selezione, transazione)
3. Usciamo da "Serving Customer" → **espelliamo la carta**

#### Stati Principali e Transizioni

`00:07:28 - 00:13:28`

**Stato Iniziale: Off**

```
[●] → [Off] --turn on / startup--> [Self Test]
```

- **Stato iniziale**: ATM spento (**Off**)
- **Evento trigger**: **turn on** (qualcuno accende il sistema)
- **Azione durante transizione**: **startup** (avvio del sistema)
- **Stato successivo**: **Self Test** (auto-test del sistema)

> **🔍 Anatomia di una Transizione**
> 
> ```
> [Stato A] --evento / azione--> [Stato B]
> ```
> 
> - **evento**: Ciò che **scatena** la transizione
> - **azione**: Cosa viene **eseguito durante** la transizione
> - Se manca l'evento, la transizione può avvenire **in qualsiasi momento**

**Transizioni da Self Test**:

```
Self Test
├─ --failure--> Out of Service
└─ ---------> Idle
```

- **Con evento "failure"**: Vai a "Out of Service"
- **Senza evento**: Vai a "Idle" (può avvenire in qualsiasi momento)

> **💡 Regola delle Transizioni**
> 
> - **Con evento specifico**: La transizione avviene SOLO quando l'evento si verifica
> - **Senza evento**: La transizione può avvenire in qualsiasi momento
> - Se c'è un **evento specifico** (es. failure), quella transizione ha **priorità**

**Stato Idle - Hub Centrale**:

Da "Idle" possono partire **tre transizioni diverse**:

```
Idle
├─ --turn off / shutdown--> Off
├─ --service--> Maintenance
└─ --card inserted--> Serving Customer
```

1. **turn off / shutdown** → **Off**
   - Evento: qualcuno spegne l'ATM
   - Azione: esegue shutdown prima di spegnersi

2. **service** → **Maintenance**
   - Evento: necessità di manutenzione
   - L'ATM entra in modalità manutenzione

3. **card inserted** → **Serving Customer**
   - Evento: qualcuno inserisce una carta
   - Inizia il servizio al cliente

**Uscite da Serving Customer**:

```
Serving Customer
├─ ---------> Idle (completamento normale)
├─ --cancel--> Idle
└─ --failure--> Out of Service
```

> **⚠️ Eventi che Possono Avvenire in Qualsiasi Momento**
> 
> Gli eventi **cancel** e **failure** possono verificarsi in **qualsiasi momento** durante "Serving Customer":
> - Anche mentre si selezionano le transazioni
> - Anche durante l'autenticazione
> - Interrompono immediatamente il flusso corrente

**Stato Maintenance**:

```
Maintenance
├─ ---------> Self Test (completamento)
├─ --failure--> Out of Service
└─ <--service-- Out of Service
```

#### Riepilogo Completo del Ciclo

```
    ┌────────────────────────────────────────────────┐
    │                                                │
[●]─→[Off]──turn on/startup──→[Self Test]          │
              ▲                    │ │              │
              │                    │ └failure       │
        turn off/                  │    │           │
        shutdown                   │    ▼           │
              │                    │ [Out of        │
              │                    │  Service]      │
           [Idle]◄─────────────────┘    ▲ │         │
              │ │                        │ │         │
              │ └──service──→[Maintenance]┘         │
              │                    │                 │
         card │                    └─────────────────┘
         inserted                 (completamento)
              │
              ▼
    [Serving Customer]
    │ entry / read card
    │ exit / eject card
    │
    ├─ [●]→[Customer Authentication]ⓘ
    ├─ [Selecting Transaction]
    ├─ [Transaction]ⓘ
    └─ [■]
    
    ⓘ = ulteriormente esploso in altri diagrammi
```

> **📚 Capacità Espressive**
> 
> Come puoi vedere, con gli state machine diagram possiamo esprimere:
> 1. **Stati** e loro ciclo di vita
> 2. **Stati gerarchici** con sotto-stati
> 3. **Transizioni triggerate** da eventi specifici
> 4. **Azioni** eseguite durante le transizioni
> 5. **Entry/Exit actions** per gli stati
> 6. **Modularizzazione** con diagrammi separati

### <a name="differenza-state-sequence"></a>2.3 Differenza tra State e Sequence Diagram

`00:13:28 - 00:14:36`

Sia State Diagram che Sequence Diagram permettono di **esprimere comportamenti**, ma con focus diversi:

| Aspetto | State Diagram | Sequence Diagram |
|---------|---------------|------------------|
| **Focus** | Cambiamenti in un **singolo oggetto** | **Interazione tra oggetti** |
| **Livello** | **Class-level** documentation | **Instance-level** documentation |
| **Descrive** | Come un'entità evolve nel tempo | Uno **scenario specifico** |
| **Validità** | Ogni istanza segue quello state machine | Rappresenta **un caso particolare** |

> **💡 Documentazione Class-Level vs Instance-Level**
> 
> **State Diagram (Class-level)**:
> - Definiamo il diagramma per l'entità "Problem" o "ATM"
> - **Ogni istanza** di quella entità seguirà quello state machine
> - Ogni particolare problema, ogni particolare ATM evolverà attraverso quegli stati
> 
> **Sequence Diagram (Instance-level)**:
> - Descrive **una possibile evoluzione** specifica
> - Rappresenta un **caso particolare** o scenario
> - Altri scenari richiederebbero sequence diagram diversi

#### Esempio: Sequence Diagram con Assunzioni

Se consideriamo un sequence diagram visto in precedenza:

```
Subscriber → System: subscribe
System → Subscriber: OK
System → User: notify problem
```

Questo sequence diagram **assume** che:
- La subscription riceve feedback **OK** (caso di successo)
- Se ci fosse un **errore** alla subscribe, la notifica non avverrebbe
- Questo non è descritto qui, ma richiederebbe un **altro sequence diagram**

> **📊 Un Sequence Diagram = Un Caso Specifico**
> 
> Nell'esempio sopra descriviamo:
> - Il **caso positivo** (good case)
> - La subscription è **successful**
> - Il sistema riesce a tenere traccia dell'interesse dell'utente
> 
> Altri casi (errori, eccezioni) richiedono sequence diagram separati.

**Contesto dell'esempio**: Sistema di trasporto pubblico che supporta gli utenti nel tracciare lo stato delle varie route dell'infrastruttura.

---

## <a name="activity-diagram"></a>3. Activity Diagram

### <a name="processi-flussi"></a>3.1 Processi e Flussi di Attività

`00:16:40 - 00:18:23`

> **🎯 Focus degli Activity Diagram**
> 
> Il focus è sui **processi**. Quando vogliamo descrivere i dettagli di un certo processo, possiamo usare gli activity diagram.

> **📝 Nota per chi ha seguito Information Systems**
> 
> Per chi ha già studiato **BPMN diagrams** (Business Process Model and Notation):
> - Gli activity diagram UML sono **essenzialmente simili o equivalenti**
> - I BPMN sono più **ricchi** perché specializzati nei processi
> - Gli activity diagram UML possono comunque descrivere processi efficacemente

#### Focus di un Activity Diagram

Gli activity diagram si concentrano su:
1. **Attività** all'interno dei processi
2. **Flusso del processo** attraverso le diverse attività

#### Esempio: Selezione Esami

`00:17:42 - 00:19:31`

Vediamo un esempio di activity diagram relativo alla **selezione di esami**.

**Scenario**: Descrive cosa fa l'utente quando deve selezionare esami e registrarsi per sostenerli.

```
[●]
 ↓
[Login]◄─────┐
 │           │
 ├─user valid
 │           │
 ↓        user not
[Choose    valid
 Exam]      
 │
 ├──────────┬──────────┐
 │          │          │
 ▼          ▼          ▼
[Register  [Send Email]
 on Web]   (external exam)
 │          │
 └──────────┴──────────→[■]
```

**Flusso del processo**:

1. **Login** - L'utente effettua il login
   - Se **user not valid** (credenziali errate) → torna a Login
   - Se **user valid** → procede

2. **Choose Exam** - Scelta dell'esame
   - Due opzioni possibili

3. **Due percorsi alternativi**:
   - **Register on Web**: Registrazione tramite web (esame interno)
   - **Send Email to Instructor**: Email al professore (esame esterno di un'altra università)

4. **Terminazione** - Entrambi i percorsi portano al punto finale

> **💡 Esami Esterni**
> 
> L'opzione "Send Email" si verifica quando:
> - Scegliamo un **esame esterno** (di un'altra università)
> - Il sistema informativo **non supporta** la registrazione web
> - È necessario contattare **direttamente il professore**

### <a name="swim-lanes"></a>3.2 Swim Lanes

`00:21:19 - 00:22:21`

#### Elementi di Decisione

`00:19:31 - 00:20:46`

**Icone a forma di rombo (◊)**:

Questi simboli sono usati per rappresentare:
1. **Decision points** (punti di decisione)
2. **Merge points** (punti di fusione del flusso)

```
    ┌─────────┐
    │ Login   │
    └────┬────┘
         │
         ◊ ──user not valid──┐
         │                   │
    user valid               │
         │                   │
    ┌────▼────┐              │
    │ Choose  │              │
    │ Exam    │              │
    └─────────┘              │
         ▲                   │
         └───────────────────┘
```

> **🔄 Due Usi del Rombo**
> 
> 1. **Decision** (decisione): Il flusso si divide in percorsi alternativi
>    - Basato su una **condizione** (es. "user valid")
>    - Un solo percorso viene seguito
> 
> 2. **Merge** (fusione): Flussi da direzioni diverse si uniscono
>    - Convergenza dopo percorsi alternativi
>    - Continuazione unificata

**Loop con decisioni**:

La decisione può anche rappresentare **cicli**:
- Azione ripetuta **più volte**
- Simile ai control flow nei programmi

#### Concetto di Swim Lane

`00:21:19 - 00:22:21`

> **🏊 Swim Lanes - Corsie di Nuoto**
> 
> Le **swim lanes** (corsie) sono un modo per specificare **chi o cosa** sta eseguendo un determinato insieme di azioni.

**Esempio con due swim lanes**:

```
┌─────────────────┬────────────────────┐
│    STUDENT      │      SYSTEM        │
├─────────────────┼────────────────────┤
│                 │                    │
│ [Choose Exam]   │                    │
│       │         │                    │
│       ├─────────┼──────────┐         │
│       │         │          │         │
│       │         │ [Check Dependencies]
│       │         │          │         │
│       │         │    ┌─────┴────┐    │
│       │         │    │          │    │
│       │         │ [Store]  [Tell     │
│       │         │           Instructor]
│       │         │    │          │    │
│       │         │    └─────┬────┘    │
│       │         │          │         │
│       └─────────┼──────────┘         │
│                 │                    │
└─────────────────┴────────────────────┘
```

**Responsabilità**:
- **Student** (Studente): Esegue "Choose Exam"
- **System** (Sistema): Esegue "Check Dependencies", "Store", "Tell Instructor"

> **💼 Beneficio delle Swim Lanes**
> 
> In qualsiasi activity diagram, le swim lanes permettono di:
> - Evidenziare **chi è responsabile** di cosa
> - Separare visivamente **ruoli diversi**
> - Chiarire l'**interazione tra attori**

### <a name="fork-join-decisioni"></a>3.3 Fork, Join e Decisioni

`00:22:21 - 00:23:44`

#### Elementi già Visti

**Merge e Decision**:
- Già discussi in precedenza
- Usano lo stesso simbolo ◊
- **Merge**: Due percorsi convergono
- **Decision**: Un percorso si divide

#### Nuovi Elementi: Fork e Join

> **🔀 Fork e Join - Processi Paralleli**
> 
> Permettono di rappresentare **sotto-processi paralleli** che vengono eseguiti contemporaneamente.

**Fork (Biforcazione)**:

```
       │
       │
   ════▼════  ← Fork (barra orizzontale spessa)
       ║
    ┌──┴──┐
    │     │
    ▼     ▼
 [Store] [Tell
         Instructor]
```

- **Simbolo**: Barra orizzontale spessa
- **Significato**: I sotto-processi partono **in parallelo**
- Entrambe le attività vengono eseguite **simultaneamente**

**Join (Ricongiunzione)**:

```
 [Store] [Tell
         Instructor]
    │     │
    └──┬──┘
       ║
   ════▼════  ← Join (barra orizzontale spessa)
       │
       │
```

- **Simbolo**: Barra orizzontale spessa (come fork)
- **Significato**: Attende che **tutti** i sotto-processi paralleli terminino
- L'esecuzione **parallela si ferma**
- Si continua in modo **sequenziale**

#### Esempio Completo con Fork e Join

`00:22:59 - 00:23:44`

```
System Swim Lane:

    [Check Dependencies]
           │
        OK │
           │
       ════▼════ Fork
           ║
        ┌──┴──┐
        │     │
        ▼     ▼
    [Store] [Tell Instructor]
        │     │
        └──┬──┘
           ║
       ════▼════ Join
           │
          [■] End
```

**Descrizione del flusso**:

1. Il sistema **controlla le dipendenze** (Check Dependencies)
2. Se tutto è a posto (**OK**):
   - **Fork**: Iniziano due processi in parallelo:
     - **Store**: Memorizza i dati
     - **Tell Instructor**: Notifica il docente (es. via email)
3. **Join**: Attende che entrambi terminino
4. **Fine**: Processo completato

> **📝 Nota sui Sotto-Processi**
> 
> In questo esempio i sotto-processi paralleli sono composti da una **singola attività** ciascuno.
> 
> In generale, ogni sotto-processo parallelo può contenere **multiple attività** in sequenza o anche altri fork/join annidati!

### <a name="differenza-sequence-activity"></a>3.4 Differenza tra Sequence e Activity Diagram

`00:23:44 - 00:25:37`

| Aspetto | Sequence Diagram | Activity Diagram |
|---------|------------------|------------------|
| **Focus Principale** | **Interazione tra oggetti** | **Attività e flusso** |
| **Enfasi** | **Scambio di messaggi** | **Processo e sequenza azioni** |
| **Elementi Centrali** | Oggetti, attori, messaggi | Attività, decisioni, flussi |
| **Adatto per** | **Protocolli di interazione** | **Descrizione di processi** |
| **Swim Lanes** | Non previste nativamente | Identificano attori/oggetti esecutori |

#### Intersezione tra i Due

> **🔄 Sovrapposizione**
> 
> Qualcosa descritto con un Sequence Diagram **potrebbe in principio** essere descritto anche con un Activity Diagram, ma:
> 
> - **Activity Diagram**: Enfatizza molto di più le **attività**
> - **Sequence Diagram**: Enfatizza molto di più lo **scambio di messaggi**

#### Quando Usare Quale

**Sequence Diagram** - Ideale per:
- Descrivere **protocolli di interazione**
- Mostrare **comunicazione** tra componenti
- Evidenziare **ordine temporale** dei messaggi
- Analizzare **sincronizzazione**

**Activity Diagram** - Ideale per:
- Descrivere **processi di business**
- Modellare **workflow**
- Rappresentare **algoritmi ad alto livello**
- Mostrare **decisioni e flussi alternativi**

**Activity Diagram con Swim Lanes**:
- Le swim lanes permettono di identificare attori/oggetti
- Si avvicina ai Sequence Diagram per descrivere interazioni
- Ma rimane focalizzato sul **processo** più che sui **messaggi**

---

## <a name="uml-requirement"></a>4. UML per Requirement Engineering - Riepilogo

`00:25:37 - 00:32:30`

### <a name="quando-usare"></a>4.1 Quando Usare Ogni Diagramma

#### Domanda 1: Cosa Dovrebbe Fare il Sistema?

`00:25:37 - 00:26:16`

> **🎯 Use Case Diagram**
> 
> **Domanda**: Cosa dovrebbe fare il nostro sistema in termini di trasformazione di input in output?
> 
> **Risposta**: Creiamo **Use Case Diagram**

**Processo di creazione**:
1. Abbiamo in mente alcuni **scenari**
2. I **use case** emergono dagli scenari
3. I **use case diagram** forniscono una **panoramica** di tutti i possibili use case

> **💡 Use Case come Panoramica**
> 
> I use case diagram sono un modo per fornire una **visione d'insieme** di tutti i possibili use case per un certo software da realizzare.

#### Domanda 2: Qual è la Struttura del Mondo?

`00:26:16 - 00:27:31`

> **🏗️ Class Diagram**
> 
> **Domanda**: Qual è la struttura del mondo su cui vogliamo concentrarci e in cui vive il nostro software?
> 
> **Risposta**: Creiamo **Class Diagram**

**Utilità del Class Diagram**:

I class diagram sono uno strumento utile per aiutarci a **ragionare**, focalizzandoci su:

1. **Identificazione delle entità del dominio**
2. **Identificazione delle relazioni** tra entità
3. **Associazioni e molteplicità**
4. **Attributi** delle entità specifiche del dominio
5. **Operazioni ad alto livello** offerte dalle entità

> **🔍 Class Diagram come Strumento di Analisi**
> 
> Non è solo documentazione finale, ma uno **strumento di ragionamento** che ci aiuta a:
> - Comprendere il dominio
> - Identificare elementi importanti
> - Strutturare il problema

#### Domanda 3: Come Analizzare gli Stati Interni?

`00:27:31 - 00:28:30`

> **🔄 State Machine Diagram**
> 
> **Condizione**: Se nel class diagram ci sono entità per cui è importante analizzare gli **stati interni**
> 
> **Azione**: Utilizzare **State Machine Diagram**

**Quando è rilevante**:
- Entità con **ciclo di vita complesso**
- Comportamenti **dipendenti dallo stato**
- Necessità di modellare **transizioni critiche**

#### Domanda 4: Come Modellare le Interazioni?

`00:28:30 - 00:30:26`

> **💬 Sequence Diagram**
> 
> **Focus**: Interazioni tra **sistema e ambiente esterno**
> 
> **Livello**: Requirement analysis (analisi dei requisiti)

**Caratteristiche al livello dei requisiti**:

```
┌────────┐      ┌────────┐      ┌────────┐
│ Actor1 │      │ System │      │ Actor2 │
└───┬────┘      └───┬────┘      └───┬────┘
    │               │               │
    │──message1────►│               │
    │               │──message2────►│
    │◄──reply2──────┤               │
    │               │◄──reply1──────┤
```

**Tipicamente**:
- Diagrammi **molto semplici**
- Uno o più **utenti** che interagiscono
- Il **sistema come black box** (senza dettagli interni)
- Focus su **attori e sistema**

> **🎯 Sistema come Black Box**
> 
> A livello di requirement analysis:
> - **NON** entriamo nei dettagli di come il sistema è fatto
> - Questo sarà il focus della **fase di design**
> - Descriviamo solo **interazioni** tra attori e sistema

**Possibili scenari**:

1. **Interazione Attore-Sistema**:
   - Attore → Sistema
   - Sistema → Attore

2. **Interazione tra Attori** (mediata):
   - Attore1 → Sistema → Attore2

3. **Interazione Diretta tra Attori**:
   - Attore1 → Attore2 (senza sistema)
   - **Importante**: Definire **domain assumption**
   - Chiarire che avviene **fuori dal sistema**

> **📝 Domain Assumption per Interazioni Esterne**
> 
> Se due attori interagiscono **direttamente** senza il sistema:
> - È rilevante descriverlo nei requisiti
> - Diventa una **domain assumption**
> - Esempio: "Assumiamo che questa interazione specifica avvenga fuori dal sistema, tra questi due attori"

#### Domanda 5: Come Modellare i Processi?

`00:30:26 - 00:31:28`

> **📊 Activity Diagram**
> 
> **Alternative**: Nei casi in cui vogliamo focalizzarci sui **processi** piuttosto che sulle interazioni
> 
> **Possibilità**: Creare activity diagram al posto (o insieme) ai sequence diagram

**Activity Diagram con Swim Lanes**:

```
┌────────┬─────────┬────────┐
│ Actor1 │ System  │ Actor2 │
├────────┼─────────┼────────┤
│[Action]│         │        │
│   │    │         │        │
│   └────┼─[Action]│        │
│        │    │    │        │
│        │    └────┼─[Action]
└────────┴─────────┴────────┘
```

**Differenza con Sequence Diagram**:
- **Tre swim lanes**: Actor1, System, Actor2
- Specifica le **azioni** che i tre attori eseguono
- Focus su **come le azioni sono correlate** (flusso del processo)

### <a name="limitazioni-uml"></a>4.2 Limitazioni di UML

`00:31:28 - 00:32:30`

#### Posizionamento di UML nel Requirement Engineering

```
┌─────────────────────────────────────┐
│      EXTERNAL WORLD                 │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Domain                      │  │
│  │  - Phenomena                 │  │
│  │  - Scenarios                 │  │
│  │                              │  │
│  │  ┌────────────────────────┐  │  │
│  │  │  Requirements Models   │  │  │
│  │  │  - Goals               │  │  │
│  │  │  - Domain Assumptions  │  │  │
│  │  │  - Requirements        │  │  │
│  │  └────────────────────────┘  │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Utilizzo di UML**:

- **Use Case**: Aiutano a identificare requirements, domain assumptions, goals
- **Class Diagram e Activity Diagram**: Livello del dominio/fenomeni
- **Sequence Diagram e Activity Diagram**: Livello requirements models

> **✅ Siamo Pronti?**
> 
> A questo punto, a meno che non ci siano domande, abbiamo un esempio su cui concentrarci...

---

## <a name="caso-airbus"></a>5. Caso di Studio: Sistema Frenante Airbus

`00:32:30 - 00:45:29`

### <a name="class-diagram-airbus"></a>5.1 Class Diagram del Sistema

`00:33:00 - 00:36:17`

> **✈️ Contesto: Incidente Airbus**
> 
> Abbiamo già discusso dell'incidente che ha coinvolto un Airbus. Ora ci concentriamo su come potremmo usare UML per descrivere la logica del **sistema frenante**.
> 
> Il software che doveva attivare il sistema frenante **non ha funzionato** in quell'incidente particolare.

#### Identificazione delle Entità Principali

`00:33:39 - 00:35:38`

> **🔍 Semplificazione**
> 
> Stiamo **semplificando eccessivamente** il problema, ma in questa semplificazione, gli elementi emersi dall'analisi dell'incidente sono quelli che vediamo qui.

**Entità del Class Diagram**:

```
┌──────────┐
│ Aircraft │
└────┬─────┘
     │
     │ has
     ▼
┌──────────┐        senses      ┌─────────────────┐
│  Wheels  │◄──────────────────┤ Rotation Sensor │
└──────────┘                    └─────────────────┘

┌──────────┐
│ Aircraft │
└────┬─────┘
     │
     │ includes
     ▼
┌──────────────────────────┐
│ Aircraft Braking         │
│ Controller               │
└────────┬─────────────────┘
         │
         │ receives signals from
         ▼
    ┌─────────────────┐
    │ Rotation Sensor │
    └─────────────────┘
         │
         │ enables
         ▼
    ┌─────────────────┐
    │ Reverse Thrust  │
    │ System          │
    └─────────────────┘
```

**Relazioni identificate**:

1. **Aircraft - Wheels**
   - Relazione: "has" (ha)
   - Cardinalità: Semplificata (3-4 ruote considerate come entità singola)

2. **Wheels - Rotation Sensor**
   - Relazione: "senses" (rileva)
   - Il sensore rileva lo **stato delle ruote** (rotanti o ferme)

3. **Aircraft - Aircraft Braking Controller**
   - Relazione: "includes" (include)
   - Il controller è **parte dell'aircraft**

4. **Rotation Sensor → Aircraft Braking Controller**
   - Il controller **riceve segnali** dal sensore

5. **Aircraft Braking Controller → Reverse Thrust System**
   - Il controller **abilita** il sistema di inversione spinta

> **📝 Nota sulla Cardinalità**
> 
> Per semplicità, **non specifichiamo** la cardinalità specifica per la relazione Aircraft-Wheels.
> 
> Sappiamo che un aereo ha 3 ruote (o 4 a seconda del modello), ma a questo **livello di astrazione** possiamo considerare le ruote come un'**entità singola**.

### <a name="state-machine-airbus"></a>5.2 State Machine del Velivolo

`00:36:17 - 00:38:08`

> **❓ Domanda**
> 
> Date queste entità, possiamo esprimere gli **elementi importanti** del nostro problema?

Possiamo usare uno **state machine diagram** per rappresentare gli stati di un aereo.

#### Stati del Velivolo

`00:36:17 - 00:37:30`

**Focus**: Situazione specifica del volo e atterraggio

```
[●]
 ↓
[Parked]
 ↓
[Moving on Runway for Taking Off]
 ↓
[Taking Off]
 ↓
[Flying]
 ↓
[Landing] ◄───────────── FOCUS PRINCIPALE
 ↓
[Moving on Runway for Landing] ◄── FOCUS SPECIFICO
 ↓
[Moving on to Parking Position]
 ↓
[Parked]
 ↓
[■]
```

**Ciclo completo del volo**:

1. **Parked** (Parcheggiato) - Stato iniziale
2. **Moving on Runway for Taking Off** - Movimento sulla pista per decollo
3. **Taking Off** - Decollo
4. **Flying** - Volo
5. **Landing** - Atterraggio ⚠️
6. **Moving on Runway for Landing** - Movimento sulla pista dopo atterraggio ⚠️⚠️
7. **Moving on to Parking Position** - Movimento verso posizione parcheggio
8. **Parked** - Parcheggiato (stato finale)

> **🎯 Focus dell'Analisi**
> 
> Vogliamo concentrarci in particolare sulla **transizione** tra:
> - **Landing** → **Moving on Runway for Landing**
> 
> Questa è la fase critica per il **sistema frenante**!

#### Utilità dello State Machine

`00:37:00 - 00:38:08`

> **💡 Perché Questo Diagramma?**
> 
> Potremmo trovare utile evidenziare questo state machine diagram in particolare per **comprendere** che ci concentreremo essenzialmente su:
> - La **porzione inferiore** del diagramma
> - Più specificatamente la **transizione Landing → Moving on Runway**

**Alternative di modellazione**:

> **🤔 Esercizio di Riflessione**
> 
> Probabilmente state già pensando che potreste aver descritto questo in modo **diverso**, e va benissimo!
> 
> In generale, qualsiasi sistema può essere modellato in **molteplici modi**. Come esercizio, potete pensare ad altri possibili state machine diagram per questo esempio.

### <a name="use-case-sequence-airbus"></a>5.3 Use Case e Sequence Diagram

#### Use Case Diagram

`00:38:08 - 00:39:40`

Un diagramma use case molto semplice per evidenziare:
- Gli attori importanti
- Il caso d'uso principale

```
┌──────────────────┐
│ Rotation Sensor  │
└────────┬─────────┘
         │
         │ initiate
         ▼
    ┌────────────────────────┐
    │  Enable Reverse Thrust │ (Use Case)
    └────────────────────────┘
         │
         │ participates
         ▼
┌────────────────────┐
│ Reverse Thrust     │
│ System             │
└────────────────────┘
```

**Attori e Ruoli**:

1. **Rotation Sensor** (Sensore di Rotazione)
   - Ruolo: **initiator** (iniziatore)
   - Produce **segnali** che attivano il use case

2. **Reverse Thrust System** (Sistema di Inversione Spinta)
   - Ruolo: **participant** (partecipante)
   - **Risponde** all'attivazione

**Use Case Centrale**:
- **Enable Reverse Thrust** (Abilita Inversione Spinta)
- Attivato dal Rotation Sensor
- Il Reverse Thrust System partecipa

> **📝 Possibile Estensione**
> 
> Potremmo anche specificare esplicitamente il ruolo dei due attori:
> - **Rotation Sensor**: initiator
> - **Reverse Thrust System**: participant

#### Sequence Diagram

`00:39:40 - 00:41:43`

Sequence diagram associato a questo use case:

```
┌──────────────┐  ┌───────────────────┐  ┌──────────────┐
│ Rotation     │  │ Aircraft Braking  │  │ Reverse      │
│ Sensor       │  │ System            │  │ Thrust System│
└──────┬───────┘  └─────────┬─────────┘  └──────┬───────┘
       │                    │                    │
       │ wheels rotating    │                    │
       │───────────────────►│                    │
       │  (async)           │                    │
       │                    │                    │
       │                   ╔╪═══════════════════╗│
       │                   ║│   Activity Box    ║│
       │                   ║│                   ║│
       │                   ║│ enable RT         ║│
       │                   ║│──────────────────►║│
       │                   ║│   (sync)          ║│
       │                   ║│                   ║│
       │                   ║│◄─────────────────║│
       │                   ║│   reply           ║│
       │                   ╚╪═══════════════════╝│
       │                    │                    │
```

**Descrizione del flusso**:

1. **Messaggio Asincrono**: "wheels rotating"
   - **Da**: Rotation Sensor
   - **A**: Aircraft Braking System
   - **Tipo**: Asynchronous message (freccia con punta aperta)
   - **Significato**: Non aspetta risposta, la controparte può continuare

2. **Activity Box**: Attività attivata nel sistema
   - Rappresentata dal **rettangolo** sul lifeline del sistema
   - Indica che il sistema sta **elaborando**

3. **Messaggio Sincrono**: "enable reverse thrust"
   - **Da**: Aircraft Braking System
   - **A**: Reverse Thrust System
   - **Tipo**: Synchronous message (freccia con punta piena)
   - **Significato**: Aspetta una risposta

4. **Reply**: Risposta al messaggio sincrono
   - **Da**: Reverse Thrust System
   - **A**: Aircraft Braking System
   - Completamento dell'operazione

> **🔄 Messaggi Asincroni vs Sincroni**
> 
> **Asincrono** (Asynchronous):
> - **Non aspetta** una risposta
> - La controparte può **continuare il suo lavoro** senza essere bloccata
> - Utile per **notifiche**, **eventi**
> 
> **Sincrono** (Synchronous):
> - **Aspetta** una risposta (blocking)
> - Almeno una parte del sistema sarà **in attesa** del reply
> - Nei sistemi con **esecuzione parallela**, solo la parte coinvolta viene bloccata

> **⚠️ Importante**
> 
> È fondamentale per noi **distinguere** tra messaggi asincroni e sincroni quando caratterizziamo l'interazione tra le parti del sistema.

### <a name="limiti-uml-requisiti"></a>5.4 Limiti di UML per Requisiti Formali

`00:41:43 - 00:45:29`

#### Cosa Abbiamo Descritto Finora

`00:41:43 - 00:42:45`

Fino a questo punto abbiamo descritto i **fenomeni**:

✅ **Elementi identificati**:
- Ruote che **ruotano**
- Sensori che **identificano** la rotazione
- Sistema di inversione spinta che viene **attivato** dai segnali
- Use case "**Enable Reverse Thrust**"

#### La Domanda Critica

`00:42:15 - 00:42:45`

> **❓ Domanda Chiave**
> 
> Tornando alla nostra analisi del caso... cosa ci **manca** a questo punto?
> 
> Se ci riferiamo all'analisi precedente, stiamo descrivendo:
> - Goals (obiettivi)?
> - Domain Properties (proprietà del dominio)?
> - Requirements (requisiti)?
> 
> **Risposta**: **NO**

**Perché no?**

> **⚠️ Limitazione Principale di UML**
> 
> UML **non** ha la possibilità di definire:
> - **Constraints** (vincoli)
> - **Rules** (regole)
> - Condizioni formali

#### Esempio di Requisiti Mancanti

`00:42:45 - 00:44:00`

Dalla nostra analisi precedente del caso Airbus:

**Requisiti formali**:

```
GOAL:
  Reverse is enabled IF AND ONLY IF wheels push on

DOMAIN PROPERTY:
  Wheels push on IF AND ONLY IF wheels turning
```

> **🚫 Inesprimibile in UML**
> 
> Questi tipi di **condizioni** o **vincoli** **NON possono essere espressi** in UML puro!

**Cosa può fare UML**:
- ✅ Aiutarci a **chiarire** il dominio
- ✅ Identificare **attori**
- ✅ Identificare **elementi importanti**
- ✅ Descrivere **interazioni**
- ✅ Modellare **stati e transizioni**

**Cosa NON può fare UML**:
- ❌ Esprimere **goals formali**
- ❌ Definire **domain properties** con logica
- ❌ Specificare **requirements** con vincoli logici

#### Approccio Ibrido: UML + Linguaggio Naturale

`00:44:00 - 00:45:29`

> **💡 Soluzione Parziale**
> 
> Possiamo esprimere goals, properties e requirements **informalmente** in linguaggio naturale, ma con disciplina:

**Best Practice**:

1. **Usare solo termini definiti in UML**
   - Ogni termine deve apparire nel **class diagram**
   - Ogni entità deve essere **modellata**

2. **Mantenere consistenza**
   - Non usare entità non descritte
   - Riferirsi sempre agli elementi UML

3. **Essere disciplinati**
   - Chiarire il **ruolo** di ogni entità prima di usarla
   - Collegare linguaggio naturale ai diagrammi

**Limitazioni dell'approccio**:
- ❌ UML **di per sé** non aiuta in questo
- ❌ Rimane **informale** (linguaggio naturale)
- ❌ Non c'è **verifica automatica**
- ❌ Possibili **ambiguità** e **imprecisioni**

#### Introduzione ad Alloy - La Soluzione

`00:44:37 - 00:45:29`

> **🎯 Alloy Entra in Gioco**
> 
> **Alloy** entra nel gioco esattamente per questa ragione!

**Cosa offre Alloy**:

✅ **Espressività completa**:
- Possiamo esprimere **goals**
- Possiamo esprimere **domain properties**
- Possiamo esprimere **requirements**
- Possiamo definire **constraints formali**

✅ **Verifica formale**:
- Verificare che dati requirements e domain properties raggiungiamo il goal
- Verificare se la definizione è **completa** o **incompleta**
- **Formal verification** automatica

✅ **Descrizione completa**:
- In un linguaggio come Alloy possiamo descrivere il sistema in modo **più completo**
- Possiamo anche **verificare** proprietà formalmente

**Esempio con Alloy**:

```alloy
// Possiamo scrivere formalmente:
assert ReverseThrustGoal {
  all t: Time | 
    reverse_enabled[t] <=> wheels_push_on[t]
}

fact WheelsProperty {
  all t: Time |
    wheels_push_on[t] <=> wheels_turning[t]
}
```

E poi **verificare automaticamente** se queste proprietà sono soddisfatte!

> **📚 Materiale di Studio**
> 
> **Per UML e Requirement Engineering**:
> - Capitoli **3-8** del libro presentano questa parte
> 
> **Per Alloy**:
> - **NON** presente nella edizione corrente del libro
> - Stiamo pianificando di preparare un nuovo capitolo
> - Per ora, materiale online e slide del corso

> **📖 Nota sul Libro**
> 
> Abbiamo fatto questo sforzo di preparare il libro proprio per:
> - **Cristallizzare** il contenuto del corso
> - Rendere **chiaro** il materiale
> 
> Il libro è alla **prima edizione**, quindi probabilmente ci sono:
> - Errori
> - Problemi
> - Imprecisioni
> 
> **🙏 Richiesta**: Se usate il libro e trovate qualsiasi problema o errore di qualsiasi tipo, saremo **molto grati** se ce lo fate sapere!

---

## <a name="introduzione-alloy"></a>6. Introduzione ad Alloy

### <a name="cosa-alloy"></a>6.1 Cos'è Alloy

`00:45:29 - 00:47:44`

> **🔧 Alloy - Linguaggio di Specifica Formale**
> 
> Alloy è un **linguaggio di specifica** (specification language) che permette di modellare sistemi e software in modo formale e verificabile.

### <a name="risorse-strumenti"></a>6.2 Risorse e Strumenti

`00:45:59 - 00:47:44`

#### Documentazione Disponibile

**Libro Online** (Principale):
- Risorsa **apertamente disponibile**
- Primo libro su Alloy
- Il **practical book** online dovrebbe essere **più che sufficiente** per il nostro uso di Alloy

**Sito Ufficiale**: Informazioni su:
- Lo **strumento** (Alloy Analyzer)
- Download dell'**Alloy Analyzer**
- Documentazione aggiuntiva
- **Esempi e tutorial**

#### Alloy Analyzer

> **🔬 Tool di Verifica**
> 
> **Alloy Analyzer** è lo strumento che ci aiuta a:
> - **Simulare** le specifiche
> - **Verificare** che certe proprietà valgano
> - Esplorare istanze del modello

**Download**: Disponibile sul sito ufficiale di Alloy

### <a name="caratteristiche-alloy"></a>6.3 Caratteristiche Principali

`00:47:44 - 00:53:12`

#### Notazione Formale per la Specifica

> **📐 Alloy è...**
> 
> Una **notazione formale** utilizzata per specificare modelli di sistemi e software.

**Utilizzo**:
- ✅ A livello di **requirements** (requisiti)
- ✅ A livello di **design** (progettazione)

**Differenza nell'utilizzo**:
- Nei due contesti ci focalizzeremo su **elementi diversi**
- Gli strumenti Alloy rimangono gli stessi

#### Alloy Analyzer - Capacità

`00:47:44 - 00:49:24`

Come già menzionato, Alloy viene fornito con uno **strumento** che ci aiuta a:

1. **Simulare** le specifiche
2. **Verificare** che certe proprietà valgano

**Model Checking**:

> **🔍 Model Checking**
> 
> Quando usiamo Alloy per verificare proprietà, stiamo essenzialmente facendo un'attività che tecnicamente si chiama **model checking**.

**Per chi seguirà il corso**:

> **📚 Corso: Formal Methods for Software Engineering**
> 
> Chi seguirà quel corso si focalizzerà sul **model checking** come disciplina a sé stante:
> - Comprendere come verificare proprietà
> - Tecniche avanzate di verifica formale
> - Fondamenti teorici

**Il nostro approccio**:
- Otteniamo un **assaggio** (taste) di cosa si può fare con model checking
- Non approfondiamo la teoria completa
- Focus sull'**uso pratico** di Alloy

#### Aspetto Object-Oriented

`00:49:24 - 00:50:25`

> **🎨 Somiglianza con OOP**
> 
> Alloy **assomiglia** a un linguaggio object-oriented.

**Differenza principale con linguaggi come Java/C++/C#**:

| Linguaggi OOP | Alloy |
|---------------|-------|
| **Imperativo/Procedurale** | **Dichiarativo** |
| Focus su "come implementare" | Focus su "proprietà" |
| Descrizione del flusso | Descrizione delle caratteristiche |

> **⚠️ Alloy NON è per l'Aritmetica**
> 
> Alloy ha una **forte fondazione matematica**, ma questo NON significa che sia capace di:
> - ❌ Eseguire operazioni matematiche complesse
> - ❌ Calcoli aritmetici sofisticati
> 
> È **abbastanza limitato** per le operazioni matematiche usuali (somma, moltiplicazione, ecc.)

**Fondazione matematica = Logica**:
- Alloy è basato sulla **logica**
- La logica è il suo punto di forza
- Non l'aritmetica tradizionale

### <a name="alloy-6"></a>6.4 Alloy 6: Novità e Differenze

`00:50:25 - 00:59:00`

#### Versione Attuale: Alloy 6

`00:50:25 - 00:51:32`

> **🆕 Versione Corrente**
> 
> Ci riferiamo all'**ultima versione di Alloy**, che è la **versione 6**.

**Cambiamenti significativi**:
- Ha introdotto alcuni **cambiamenti importanti** rispetto alla versione precedente (Alloy 5)

**Storia del corso**:
- In passato usavamo **Alloy 5**
- Dal 2022-2023 circa usiamo **Alloy 6**

> **⚠️ Attenzione agli Esami Vecchi**
> 
> Dovete fare **attenzione** quando studiate dagli esami disponibili:
> - Esami da **academic year prima del 2022-2023**: usano **Alloy 5**
> - Le soluzioni sono **leggermente diverse**
> - Potete usarli, ma fate attenzione alle differenze di sintassi
> 
> In **Alloy 6** le cose vengono fatte in modo **leggermente diverso**.

#### Uso: Modellazione Concettuale

`00:51:32 - 00:53:12`

> **🎯 Alloy per Conceptual Modeling**
> 
> Alloy è utilizzato per la **modellazione concettuale** (conceptual modeling).

**Caratteristica Dichiarativa**:

> **💡 Cosa Significa "Dichiarativo"?**
> 
> L'approccio dichiarativo **non è solo** per la specifica, ma anche per la **programmazione**.

**Dichiarativo vs Procedurale**:

Esistono linguaggi di programmazione che seguono:
- **Stile Procedurale**: Più comune (Java, C++, Python, ecc.)
- **Stile Dichiarativo**: Meno comune ma potente (SQL, Prolog, Alloy, ecc.)

#### Definizione di Programmazione Dichiarativa

`00:52:02 - 00:53:12`

> **📐 Paradigma Dichiarativo**
> 
> Un paradigma di programmazione che esprime le **caratteristiche** di un calcolo (computation) ma **NON descrive come** eseguire quel calcolo.

**In altre parole**:

```
Dichiarativo:
- Descrivi COSA vuoi ottenere
- NON descrivi COME ottenerlo

Procedurale:
- Descrivi COSA vuoi ottenere
- Descrivi esplicitamente COME ottenerlo (passo per passo)
```

**Non descrive esplicitamente il control flow**:
- Non ci sono `if-then-else` espliciti nel senso tradizionale
- Non ci sono loop `for`/`while` nel senso tradizionale
- Si descrivono **proprietà** e **vincoli**

> **💭 In Termini Semplici**
> 
> Diciamo **cosa** un programma dovrebbe realizzare, piuttosto che **come** dovrebbe realizzarlo.

**Vedremo meglio con gli esempi**: La differenza diventerà più chiara quando vedremo esempi pratici!

#### Applicazioni di Alloy

`00:53:12 - 00:56:30`

**In Software Design**:
- Modellare **componenti** del software
- Descrivere **architetture**
- Definire **interfacce**

**In Requirement Analysis**:
- Descrivere formalmente **elementi del dominio**
- Definire **proprietà** di questi elementi
- Specificare **operazioni** che il software offre all'esterno

**Verifica Formale Automatica**:

> **✅ Capacità di Verifica**
> 
> Alloy può aiutare a verificare:
> - Se certe **proprietà** saranno soddisfatte dal sistema
> - Se certi **vincoli** non saranno mai violati

#### Limitazioni della Verifica a Livello di Modello

`00:54:20 - 00:55:28`

> **🤔 Obiezione Ragionevole**
> 
> Potreste argomentare: "Se ragioniamo a livello del **modello**, questo non fornisce la garanzia che la nostra **implementazione** offrirà le stesse proprietà o soddisferà gli stessi vincoli."

**Risposta**:

✅ **È vero!** Ma...

> **💡 Valore della Verifica Formale**
> 
> Se **NON** definite queste proprietà e vincoli esplicitamente, e **NON** provate che sono verificati a livello di modellazione:
> 
> - Forse non otterrete la **giusta comprensione** del problema
> - Non avrete una guida per creare un programma che fa ciò che ci si aspetta
> - Manterrete i vincoli che devono essere definiti

**Alloy come strumento di comprensione**:
- Aiuta a **capire meglio** il problema
- Fornisce una **specifica formale** verificabile
- Guida l'**implementazione corretta**

> **📝 Filosofia**
> 
> La verifica formale in generale è uno **strumento** che ci aiuta a comprendere un certo problema **meglio**.
> 
> Dobbiamo usarla in questo modo!

#### Livello di Astrazione Appropriato

`00:55:28 - 00:56:30`

> **⚖️ Bilanciamento Importante**
> 
> Una delle questioni che affrontiamo: Se modelliamo **troppi dettagli**...

**Problema del dettaglio eccessivo**:

```
Troppi Dettagli →
  ↓
Verifica Molto Complessa →
  ↓
Tempo di Verifica Troppo Alto →
  ↓
Impossibilità di Definire un Modello Chiaro →
  ↓
Non Riusciamo a Provare le Proprietà ❌
```

**La sfida**:

> **🎯 Trovare il Giusto Livello**
> 
> Dobbiamo trovare il **giusto livello di astrazione**:
> 
> - Da un lato: Non rendere il problema **troppo complesso**
> - Dall'altro: Aiutare ad affrontare i problemi che incontreremo durante design e implementazione

**Difficoltà**:
- Trovare il **giusto bilanciamento** tra questi due aspetti
- È un'**arte** più che una scienza esatta
- Verrà con l'**esperienza**

> **📚 Impareremo con la Pratica**
> 
> Vedremo attraverso gli esempi come trovare questo equilibrio!

#### Utilizzo nell'Industria

`00:56:30 - 00:57:39`

> **🏭 Alloy nell'Industria**
> 
> Anche se potreste pensare che la verifica formale sia "solo roba da università", in realtà viene **usata in aziende**.

**Esempi concreti**:

**Intel**:
- Paper che presentano l'uso di formal methods
- Verifica di proprietà hardware

**NASA**:
- Paper sull'applicazione di metodi formali
- Verifica di software critico

**ST Microelectronics** (Italia):
- Conferenza sui formal methods (circa un anno fa, in quest'aula!)
- Spiegazione di come usano formal methods per provare proprietà di:
  - **Hardware** (microprocessori)
  - **Software** che interagisce con hardware

> **📈 Trend in Crescita**
> 
> I metodi formali **non** sono qualcosa di strano o puramente accademico:
> - Stanno prendendo **momentum** (slancio)
> - Le tecniche stanno diventando più **sofisticate**
> - Possono realmente essere **usate nell'industria**

#### Cos'è Alloy - Dettagli Tecnici

`00:57:39 - 00:59:00`

**Composizione di Alloy**:

> **🧩 Ingredienti di Alloy**
> 
> Alloy è essenzialmente una **miscela** (mixture) di:
> - **Logica del primo ordine** (First-Order Logic)
> - **Calcolo relazionale** (Relational Calculus)

**Aritmetica**:
- **NON** ha aritmetica completa
- Ha solo una **piccola quantità** di aritmetica
- Operazioni base: somma, moltiplicazione, ecc.
- Ma è abbastanza **limitata**

**Gerarchie di Entità**:
- ✅ Permette di definire **gerarchie** di entità
- Simile all'ereditarietà in OOP
- Utile per noi nella modellazione

**Modularizzazione**:
- ✅ Permette di **modularizzare** le specifiche
- Possiamo definire **moduli** con porzioni di specifica
- Non lo vedremo in dettaglio nel corso (useremo modelli semplici)
- Ma possiamo usare **moduli esterni** già disponibili in Alloy

**Novità in Alloy 6**:

> **🆕 Cambiamento Principale: Mutabilità**
> 
> Il cambiamento principale rispetto alle versioni precedenti di Alloy:

**Alloy 5 e precedenti**:
- Tutti gli elementi erano **immutabili**
- Non potevano cambiare nel tempo

**Alloy 6**:
- Possiamo ora modellare l'**evoluzione del tempo**
- Possiamo descrivere entità che **cambiano stato** mentre il tempo evolve
- Introduciamo **elementi mutabili** (mutable elements)

> **⏰ Modellare il Tempo**
> 
> In modo abbastanza naturale, ora possiamo:
> - Descrivere come i nostri modelli **evolvono**
> - Usare elementi mutabili per rappresentare **cambiamenti di stato**

**Utilità**:
- ✅ Buono per **esplorare** il problema
- ✅ Focus su **piccole specifiche** (small specification)
- ✅ Lo strumento (Alloy Analyzer) aiuta nell'**analisi**

**Domande che possiamo porre**:
- La specifica è **soddisfacibile** (satisfiable)?
- Questo predicato è **vero**?
- Questa proprietà è **sempre vera**?

---

## <a name="esempio-address-book"></a>7. Primo Esempio: Address Book

`00:59:00 - fine`

### <a name="signature-address-book"></a>7.1 Definizione delle Signature

`00:59:00 - 01:02:35`

> **📖 Esempio: Rubrica Indirizzi**
> 
> Iniziamo con un esempio molto semplice: modellare una **address book** (rubrica indirizzi).

#### Descrizione del Problema

`00:59:00 - 01:01:01`

**Requisito**:
- Una rubrica (book) contiene **indirizzi** (addresses)
- Gli indirizzi sono **collegati** ai corrispondenti **nomi** (names)

**Esempio**:

```
My Address Book:
├─ Professor Rossi → rossi@email.com
├─ Professor Camilli → camilli@email.com
└─ ...
```

Nel mio address book:
- Ho il **nome** "Professor Rossi"
- Collegato al suo **indirizzo email**
- Ho il **nome** "Professor Camilli"
- Collegato al suo **indirizzo email**

#### Identificazione delle Entità

`01:01:01 - 01:02:02`

> **🎯 Approccio Object-Oriented**
> 
> Alloy ha un "sapore" object-oriented (OO flavor).

**Entità naturali** dalla descrizione:
1. **Book** (Rubrica)
2. **Name** (Nome)
3. **Address** (Indirizzo)

**Rappresentazione in Alloy**:

Ogni entità può essere rappresentata come una **signature** in Alloy.

#### Sintassi: Signature Semplici

`01:02:02 - 01:02:35`

**Codice Alloy**:

```alloy
sig Name, Addr {}
```

**Spiegazione**:

> **📝 Sintassi Abbreviata**
> 
> Nelle slide, la sintassi:
> ```alloy
> sig Name, Addr {}
> ```
> 
> È **equivalente** a:
> ```alloy
> sig Name {}
> sig Addr {}
> ```

**Significato**:
- `sig` = signature (parola chiave)
- `Name` e `Addr` sono due **signature diverse**
- Entrambe sono **elementi base** (ground elements)
- Non specifichiamo caratteristiche interne (`{}` vuoto)
- Ma **esistono** nel modello

**Cosa significa "esistere"**:

> **🌍 Generazione di Mondi**
> 
> Se eseguiamo una specifica con queste due signature:
> - Possiamo generare **worlds** (mondi) dalla specifica
> - Nei mondi generati, troveremo **istanze** (instances) di Name e Addr

**Atoms**:
- Le istanze specifiche sono chiamate **atoms** (atomi)
- Ad esempio: `Name0`, `Name1`, `Addr0`, `Addr1`

### <a name="relazioni-cardinalita"></a>7.2 Relazioni e Cardinalità

`01:02:35 - 01:04:44`

#### Signature con Relazioni

**Codice Alloy**:

```alloy
sig Book {
  addr: Name -> lone Addr
}
```

**Anatomia della dichiarazione**:

```
sig Book {
  ┌─────────────────────────────┐
  │ addr: Name -> lone Addr     │
  │   ↑     ↑         ↑          │
  │   │     │         │          │
  │ nome  tipo    cardinalità    │
  │della  origine  constraint    │
  │rel.                          │
  └─────────────────────────────┘
}
```

**Componenti**:

1. **`addr`**: Nome della relazione
2. **`Name -> Addr`**: Tipo della relazione (da Name a Addr)
3. **`lone`**: Vincolo di cardinalità

#### Significato della Relazione

`01:02:35 - 01:03:38`

> **🔗 Relazione addr**
> 
> Stiamo dicendo: All'interno di un certo book, possiamo avere un **nome** e a questo nome possiamo associare **lone address**.

**Cosa significa `lone`?**

```
lone = 0 oppure 1
```

- Possiamo avere **zero** indirizzi associati al nome
- Possiamo avere **uno** indirizzo associato al nome
- **NON** possiamo avere più di un indirizzo

**Effetto del vincolo**:

> **⚠️ Importante**
> 
> Con questo vincolo (`lone`), dato un nome in un book, **NON** possiamo avere il caso di:
> - Un nome associato a **multipli indirizzi**

**Se eliminassimo `lone`**:

```alloy
sig Book {
  addr: Name -> Addr  // senza lone
}
```

Potremmo avere un nome associato a **qualsiasi numero** di indirizzi (incluso zero).

#### Altri Vincoli di Cardinalità

`01:03:38 - 01:04:44`

> **📊 Cardinality Constraints in Alloy**

Alloy offre diversi vincoli di cardinalità:

| Vincolo | Significato | Numero di elementi |
|---------|-------------|-------------------|
| **`lone`** | **0 o 1** | Zero oppure uno |
| **`one`** | **Esattamente 1** | Uno e solo uno |
| **`set`** | **Qualsiasi numero** | Zero, uno, o molti (insieme) |
| **`some`** | **1 o più** | Almeno uno, possibilmente molti |

**Differenze chiave**:

```
set:  0, 1, 2, 3, ... ∞  (anche vuoto)
some: 1, 2, 3, ... ∞     (mai vuoto)
lone: 0, 1               (al massimo uno)
one:  1                  (esattamente uno)
```

> **💡 Set vs Some**
> 
> - **`set`**: Può essere anche **vuoto** (insieme vuoto)
> - **`some`**: Deve avere **almeno un elemento**

#### Riepilogo delle Entità

`01:04:44 - 01:05:49`

**Signature definite**:

```alloy
sig Name {}
sig Addr {}
sig Book {
  addr: Name -> lone Addr
}
```

**Cosa abbiamo**:
- **Name**, **Addr**, **Book**: Entità nel modello
- **addr**: Relazione che collega Name ad Addr nel contesto di Book

**Significato concettuale**:

> **🔍 Relazione nel Contesto**
> 
> Il collegamento tra Name e Addr è eseguito **all'interno del contesto di un Book specifico**.

**Cosa significa questo?**

Nei mondi che soddisfano la specifica, possiamo avere:

```
World Example:
├─ Book: B1
│  └─ addr relations:
│     ├─ Name0 → Addr1
│     └─ Name1 → Addr3
│
└─ Book: B2
   └─ addr relations:
      ├─ Name0 → Addr2
      └─ Name1 → Addr3
```

- **Due book** diversi: B1 e B2
- **Relazioni addr diverse** in ciascun book
- Lo stesso Name può avere indirizzi diversi in book diversi

### <a name="operazione-join"></a>7.3 Operazione di Join

`01:05:49 - 01:23:05`

#### Relazioni Ternarie

`01:05:49 - 01:07:35`

> **🔢 Relazione Ternaria**
> 
> Come esprimiamo il fatto che la relazione addr è contestuale al book?

**Concettualmente, addr è una relazione ternaria**:

```
addr: Book × Name × Addr
```

Una **tripla** (triple) nella relazione contiene:
1. Un **Book**
2. Un **Name**
3. Un **Addr**

**Esempio di triple**:

```
addr = {
  (B1, N1, A0),
  (B1, N2, A3),
  (B2, N1, A2),
  (B2, N2, A3),
  ...
}
```

**Notazione nella specifica**:

```alloy
sig Book {
  addr: Name -> lone Addr
}
```

Vediamo come **campo** (field) di Book, ma è in realtà una **relazione ternaria**!

> **📝 Interpretazione**
> 
> Anche se nella specifica vediamo `addr` come un campo dell'entità Book, è in realtà una **relazione ternaria** che coinvolge Book, Name e Addr.

**Perché ternaria?**

Ogni elemento della relazione addr specifica:
- **Quale book** (es. B1)
- **Quale nome** in quel book (es. N1)  
- **Quale indirizzo** è associato (es. A0)

#### Concetto di Atom

`01:07:35 - 01:09:53`

> **⚛️ Atoms - Elementi Indivisibili**
> 
> Le istanze delle nostre signature sono chiamate **atoms** (atomi).

**Caratteristiche degli Atoms**:

1. **Indivisibili**: Non possono essere divisi
   - Non posso prendere `B1` e dividerlo

2. **Mutabili** (in Alloy 6): Possono cambiare stato
   - **NOVITÀ** rispetto ad Alloy 5

3. **Non interpretati**: Sono concetti ground (di base)
   - Non hanno un significato intrinseco
   - Sono "primitivi" nel modello

**Esempio di atoms**:

```
Book atoms: B0, B1
Name atoms: N0, N1, N2
Addr atoms: A0, A1, A2, A3
```

> **🔤 Naming Convention**
> 
> Alloy nomina automaticamente gli atoms:
> - Prima lettera del tipo (maiuscola)
> - Seguita da un numero progressivo
> - `B0`, `B1` per Book
> - `N0`, `N1`, `N2` per Name
> - `A0`, `A1`, `A2`, `A3` per Addr

#### Relazioni come Insiemi di Tuple

`01:09:53 - 01:11:25`

> **📦 Relazioni = Set di Tuple**
> 
> Le relazioni sono **insiemi di tuple** (set of tuples).

**Per la relazione addr (ternaria)**:

```
addr = {
  (B0, N0, A0),
  (B0, N1, A1),
  (B1, N1, A2),
  (B1, N2, A2)
}
```

Ogni tripla è una **tuple**.

**Caratteristica importante**:

> **📏 L'Ordine Conta**
> 
> All'interno di una relazione, l'**ordine** degli elementi nelle tuple è importante!

**Nella relazione ternaria addr**:
- Il **Book** è sempre il **primo elemento**
- Il **Name** è sempre il **secondo elemento**
- L'**Addr** è sempre il **terzo elemento**

```
(Book, Name, Addr)  ← Ordine fisso!
  ↑     ↑     ↑
  1°    2°    3°
```

#### Relazioni Tipate

`01:11:25 - 01:12:00`

> **🏷️ Type System**
> 
> Le relazioni sono **tipate** (typed): ogni elemento ha un tipo associato.

**Quando definiamo**:

```alloy
sig Book {
  addr: Name -> lone Addr
}
```

Stiamo dicendo:
- `addr` contiene triple
- **Primo elemento**: Tipo `Book`
- **Secondo elemento**: Tipo `Name`
- **Terzo elemento**: Tipo `Addr`

**Ogni tripla rispetta i tipi**:

```
✅ (B0, N1, A2)  - Tutti i tipi corretti
❌ (N0, B1, A2)  - Tipi nell'ordine sbagliato
❌ (B0, B1, N2)  - Tipi errati
```

#### Esempio di Relazione Binaria

`01:12:00 - 01:13:34`

> **🔗 Relazioni Binarie**
> 
> Come otteniamo una relazione con tuple di **due elementi** (coppie)?

**Domanda dalla lezione precedente**: Come creare una relazione binaria?

**Risposta**:

```alloy
sig Name {
  id: String
}

sig String {}
```

**Cosa abbiamo creato**:

> **💡 Relazione Binaria `id`**
> 
> `id` è una **relazione binaria** che contiene **coppie** (pairs):
> - **Primo elemento**: Name
> - **Secondo elemento**: String

**Esempio di tuple in id**:

```
id = {
  (N0, "Rossi"),
  (N1, "Camilli"),
  (N2, "Bianchi")
}
```

#### Visualizzazione Grafica di un Mondo

`01:13:34 - 01:15:57`

Consideriamo un mondo concreto che soddisfa la nostra specifica:

**Relazione addr (ternaria)**:

```
addr = {
  (B0, N0, A0),
  (B0, N1, A1),
  (B1, N1, A2),
  (B1, N2, A2)
}
```

**Visualizzazione grafica**:

```
     BOOK              NAME              ADDR
    ┌────┐            ┌────┐           ┌────┐
    │ B0 │────────────│ N0 │───────────│ A0 │
    └────┘      │     └────┘           └────┘
                │      ┌────┐           ┌────┐
                └─────►│ N1 │───────────│ A1 │
                       └────┘      │    └────┘
    ┌────┐                         │    ┌────┐
    │ B1 │─────────────────────────┼───►│ A2 │
    └────┘      │                  │    └────┘
                │      ┌────┐      │
                └─────►│ N2 │──────┘
                       └────┘
```

**Set corrispondenti**:

Per ogni signature definita, abbiamo un **set** contenente tutti gli atoms:

```
Set Name = {N0, N1, N2}
Set Addr = {A0, A1, A2}
Set Book = {B0, B1}
```

> **✅ Condizione**
> 
> Se un atom **appare** nella relazione addr, allora **esiste** nel set corrispondente.

**Esempio**:
- `N0`, `N1`, `N2` appaiono in addr → sono in Set Name
- `A0`, `A1`, `A2` appaiono in addr → sono in Set Addr
- `B0`, `B1` appaiono in addr → sono in Set Book

**Possibilità aggiuntiva**:

> **📝 Nota**
> 
> Potremmo anche avere atoms che **NON** appaiono in addr!
> 
> Ad esempio: `N3` potrebbe esistere in Set Name ma non avere indirizzi associati.
> 
> La specifica **non richiede** che ogni Name appartenga alla relazione addr.

#### Singleton Sets

`01:15:57 - 01:17:27`

Possiamo anche definire **singleton** (o scalari):

```alloy
one sig MyName extends Name {}
one sig YourName extends Name {}
```

**Cosa significa**:

> **1️⃣ Singleton**
> 
> `one sig` crea un set che contiene **un singolo elemento**.

**Esempio**:

```
Set MyName = {N1}    // Solo un elemento
Set YourName = {N2}  // Solo un elemento
```

> **📝 Uso**
> 
> Non useremo molto questo meccanismo nel corso, ma è utile saperlo per casi speciali.

#### Operazione Join (.)

`01:17:27 - 01:23:05`

> **⚙️ Join - Operazione Fondamentale**
> 
> Il **dot** (`.`) è una delle operazioni più importanti in Alloy: l'operazione di **join**.

**Relazione con Relational Calculus**:

> **📚 Background**
> 
> Se avete fatto calcolo relazionale (relational calculus), vedrete la corrispondenza tra questa operazione join e quella del calcolo relazionale.

**Esempio di espressione**:

```alloy
myName . book . addr
```

Abbiamo **due join** in questa espressione:
1. `book . addr`
2. `myName .` (risultato del primo join)

##### Primo Join: book.addr

`01:18:05 - 01:19:38`

**Mondo di esempio**:

```
Book = {B0, B1}

addr = {
  (B0, N0, A0),
  (B0, N1, A1),
  (B1, N1, A2),
  (B1, N2, A2)
}
```

**Calcolo di `book . addr`**:

1. Prendiamo elementi dal set **Book**: `B0`, `B1`
2. Prendiamo triple dalla relazione **addr**
3. Identifichiamo triple che **iniziano con** elementi in Book
4. Eseguiamo il **join** eliminando l'elemento di join

**Passo per passo**:

```
B0 nella relazione addr:
  (B0, N0, A0) → Join su B0 → (N0, A0)
  (B0, N1, A1) → Join su B0 → (N1, A1)

B1 nella relazione addr:
  (B1, N1, A2) → Join su B1 → (N1, A2)
  (B1, N2, A2) → Join su B1 → (N2, A2)
```

**Risultato di `book . addr`**:

```
book . addr = {
  (N0, A0),
  (N1, A1),
  (N1, A2),
  (N2, A2)
}
```

> **🔑 Regola del Join**
> 
> 1. Trova tuple che **iniziano con** l'elemento di join
> 2. **Elimina** l'elemento di join
> 3. Il risultato è una tupla più **corta** di un elemento

**Struttura generale**:

```
Join tra:
  Set/Relazione A: tuple che finiscono con tipo T
  Set/Relazione B: tuple che iniziano con tipo T
  
Risultato:
  - Match sull'elemento di tipo T
  - Elimina l'elemento T
  - Concatena il resto
```

##### Secondo Join: myName . (book.addr)

`01:19:38 - 01:22:33`

Ora abbiamo:

```
myName = {N1}  // Singleton

book . addr = {
  (N0, A0),
  (N1, A1),
  (N1, A2),
  (N2, A2)
}
```

**Calcolo di `myName . book . addr`**:

1. Prendiamo l'elemento da **myName**: `N1`
2. Cerchiamo in `book.addr` le coppie che **iniziano con** `N1`
3. Eliminiamo `N1` dal risultato

**Passo per passo**:

```
N1 in book.addr:
  (N1, A1) → Join su N1 → (A1)
  (N1, A2) → Join su N1 → (A2)
```

**Risultato finale**:

```
myName . book . addr = {A1, A2}
```

> **💡 Significato Intuitivo**
> 
> **`book . addr`**: Estrae tutte le coppie (Name, Addr) presenti in tutti i book
> 
> **`myName . book . addr`**: Estrae tutti gli indirizzi che corrispondono a `myName` in qualsiasi book

**In altre parole**:

Con l'operazione `myName . book . addr`, stiamo chiedendo al sistema:

> "Dammi tutti gli indirizzi che, in **qualsiasi book**, corrispondono a `myName`"

Nel nostro esempio:
- `myName` (che è `N1`) appare in `B0` con `A1`
- `myName` (che è `N1`) appare in `B1` con `A2`
- Risultato: `{A1, A2}`

#### Chiarezza dell'Esempio?

`01:22:33 - 01:23:05`

> **❓ È Chiaro?**
> 
> Questa è l'operazione di join, una delle operazioni fondamentali in Alloy!

**Recap veloce**:

```
Operazione Join (.):
1. Match tra ultimo elemento della prima tupla 
   e primo elemento della seconda tupla
2. Elimina l'elemento di match
3. Restituisce tuple concatenate senza l'elemento eliminato
```

**Esempio completo**:

```
myName = {(N1)}                    // Singleton set
book = {(B0), (B1)}                // Set di books
addr = {(B0,N0,A0), (B0,N1,A1),    // Relazione ternaria
        (B1,N1,A2), (B1,N2,A2)}

book . addr:
  B0 joins con (B0,N0,A0) → (N0,A0)
  B0 joins con (B0,N1,A1) → (N1,A1)
  B1 joins con (B1,N1,A2) → (N1,A2)
  B1 joins con (B1,N2,A2) → (N2,A2)
  
  Risultato: {(N0,A0), (N1,A1), (N1,A2), (N2,A2)}

myName . (book . addr):
  N1 joins con (N1,A1) → (A1)
  N1 joins con (N1,A2) → (A2)
  
  Risultato: {A1, A2}
```

### <a name="alloy-analyzer"></a>7.4 Esecuzione con Alloy Analyzer

`01:23:05 - fine`

#### Preparazione della Specifica

`01:23:05 - 01:26:30`

> **💻 Demo con Alloy Analyzer**
> 
> A questo punto, posso avviare Alloy e verificare cosa succede quando proviamo a definire la specifica che abbiamo creato.

**Codice della specifica**:

```alloy
sig Name {}
sig Addr {}

sig Book {
  addr: Name -> lone Addr
}
```

> **📝 Nota sulla Sintassi**
> 
> Nella slide abbiamo scritto:
> ```alloy
> sig Name, Addr {}
> ```
> 
> Questo è **equivalente** a scrivere separatamente:
> ```alloy
> sig Name {}
> sig Addr {}
> ```

**Zoom e Visualizzazione**:

_[Il professore ha difficoltà con lo zoom dell'IDE]_

Spero che vediate abbastanza bene! A volte copiare-incollare dalle slide aggiunge caratteri extra.

#### Introduzione ai Predicati

`01:24:39 - 01:25:46`

Oltre alle signature, introduciamo qualcosa di nuovo: **predicati**.

**Codice**:

```alloy
pred show {}
```

> **🎯 Predicato**
> 
> Un **predicato** (predicate) in Alloy:
> - È simile a una funzione booleana
> - Può contenere **vincoli** (constraints)
> - Restituisce implicitamente true o false

**Predicato vuoto**:

```alloy
pred show {}  // Predicato vuoto
```

Un predicato **vuoto** (senza vincoli al suo interno):
- Non contiene nessun constraint
- È **sempre vero** (always true)
- Non restringe il modello

#### Comando Run

`01:25:46 - 01:27:33`

Ora possiamo **eseguire** la specifica con un comando:

```alloy
run show for 3 but 1 Book
```

**Anatomia del comando**:

```
run show
  ↑    ↑
  |    └─ Nome del predicato da eseguire
  └─ Comando per eseguire
  
for 3
    ↑
    └─ Limite: massimo 3 istanze per ogni entità
    
but 1 Book
    ↑
    └─ Eccezione: esattamente 1 Book
```

> **📊 Significato del Comando**
> 
> "Esegui la specifica e crea mondi che soddisfano:
> - Il predicato `show` (che è sempre vero)
> - **Al massimo 3 istanze** per Name e Addr
> - **Esattamente 1 istanza** di Book"

**Esecuzione**:

Quando eseguiamo questo comando, l'Alloy Analyzer:
1. **Cerca** mondi che soddisfano la specifica
2. **Genera** istanze concrete
3. **Visualizza** un mondo trovato

#### Visualizzazione del Risultato

`01:27:33 - 01:30:45`

_[Demo con Alloy Analyzer]_

**Messaggio dell'Analyzer**:

```
✅ Instance found
```

Significa che esiste **almeno un mondo** che soddisfa la specifica!

**Visualizzazione grafica**:

```
┌─────────┐
│  Book   │ (senza numero perché è unico)
└────┬────┘
     │
     ├──────►┌──────────┐        ┌──────────┐
     │       │  Name0   │───────►│  Addr1   │
     │       └──────────┘        └──────────┘
     │
     └──────►┌──────────┐        ┌──────────┐
             │  Name1   │───────►│  Addr0   │
             └──────────┘        └──────────┘
```

**Interpretazione**:

- **1 Book** (chiamato solo "Book" nella visualizzazione, ma è `Book0` internamente)
- **2 Names**: `Name0`, `Name1`
- **2 Addrs**: `Addr0`, `Addr1`
- **Relazioni**:
  - Book → Name0 → Addr1
  - Book → Name1 → Addr0

> **⚠️ Bug di Visualizzazione**
> 
> Nella visualizzazione grafica, quando c'è un **singolo elemento**, non viene mostrato il numero (`Book` invece di `Book0`).
> 
> Nella **visualizzazione testuale** vedrete il nome completo: `Book0`.

**Triple nella relazione addr**:

```
addr = {
  (Book0, Name0, Addr1),
  (Book0, Name1, Addr0)
}
```

#### Visualizzazione Testuale

`01:29:41 - 01:31:47`

_[Il professore cambia modalità di visualizzazione]_

**Vista testuale** (più chiara per alcuni aspetti):

```
this/Name = {Name0, Name1}
this/Addr = {Addr0, Addr1}
this/Book = {Book0}

Set String = {}
Set Seq = {}
Set Int = {-8, -7, ..., 0, ..., 6, 7}
univ = {Name0, Name1, Addr0, Addr1, Book0, -8, ..., 7}
```

**Insiemi predefiniti**:

- **String**: Vuoto (nessuna stringa in questo mondo)
- **Seq**: Sequenze (vedremo più avanti)
- **Int**: Interi da -8 a 7 (range predefinito)
- **univ**: L'**universo** - tutti gli atoms possibili

> **🌍 Universo (univ)**
> 
> Il set `univ` include **tutti** gli atoms che esistono nel mondo:
> - Tutti i Name
> - Tutti gli Addr
> - Tutti i Book
> - Tutti gli Int
> - Qualsiasi altro atom

#### Altre Modalità di Visualizzazione

`01:31:47 - 01:32:50`

L'Alloy Analyzer offre multiple visualizzazioni:

1. **Tabular** (Tabulare):
   - Mostra le relazioni in formato tabella
   - Utile per vedere chiaramente le tuple

2. **Tree** (Albero):
   - Visualizzazione gerarchica
   - Mostra le relazioni padre-figlio

3. **Graphical** (Grafica):
   - Quella che abbiamo visto prima
   - Più intuitiva visualmente

#### Generazione di Altri Mondi

`01:32:19 - 01:33:21`

**Bottone "Next"**:

Premendo il bottone **Next**, l'analyzer cerca un altro mondo diverso che ancora soddisfa la specifica.

**Esempio di nuovo mondo**:

```
this/Name = {Name0, Name1, Name2}
this/Addr = {Addr0, Addr1, Addr2}
this/Book = {Book0}

addr = {
  (Book0, Name0, Addr1),
  (Book0, Name2, Addr0)
}
```

> **📝 Differenze**
> 
> Questo mondo ha:
> - **3 addresses** (Addr0, Addr1, Addr2)
> - **3 names** (Name0, Name1, Name2)
> - Ma solo **2 di 3 addresses** sono collegati a names
> - **Addr2** è "libero" (non collegato)
> - **Name1** è "libero" (senza indirizzo)

**È ancora valido?**

✅ **Sì!** Perché:
- La specifica **non richiede** che ogni Name abbia un Addr
- La specifica **non richiede** che ogni Addr sia usato
- Il vincolo `lone` è rispettato (ogni Name ha 0 o 1 Addr)

#### Esplorazione Continua

`01:32:50 - 01:33:21`

Premendo **Next** ripetutamente:
- Possiamo vedere **altri mondi** possibili
- Ogni mondo è **diverso** ma **valido**

**Messaggi dell'Analyzer**:

Quando l'analyzer **non trova** altri mondi significativamente diversi:

```
⚠️ No more instances found (or similar)
```

Se premiamo **Next** di nuovo:
- Ricomincia a mostrare mondi già visti
- Possono esserci variazioni minori

**Esempio di mondo minimale**:

```
this/Name = {Name0}
this/Addr = {Addr0}
this/Book = {Book0}

addr = {
  (Book0, Name0, Addr0)
}
```

Solo 1 address e 1 name - ancora valido!

#### Conclusione della Demo

`01:33:21 - fine`

> **🎮 Suggerimento**
> 
> Vi suggerisco di **giocare** con l'analyzer per vedere cosa succede!

**Prossima settimana**:
- Partiremo da qui
- Continueremo la nostra panoramica del linguaggio
- Vedremo costrutti più avanzati

---

## 📝 Note Finali e Concetti Chiave

### Riepilogo della Lezione

Abbiamo completato tre parti principali:

1. **UML Avanzato**
   - State Machine Diagram per modellare cicli di vita
   - Activity Diagram per descrivere processi
   - Confronto sistematico tra tutti i diagrammi UML

2. **Caso di Studio Airbus**
   - Applicazione pratica di UML
   - Identificazione delle **limitazioni di UML** per requisiti formali
   - Necessità di linguaggi formali come Alloy

3. **Introduzione ad Alloy**
   - Linguaggio dichiarativo per specifica formale
   - Signature, relazioni, cardinalità
   - Operazione di join
   - Primo esempio pratico con Alloy Analyzer

### 🎯 Concetti Chiave da Ricordare

#### UML - Quando Usare Cosa

- **Use Case**: Panoramica funzionalità (cosa fa il sistema)
- **Class Diagram**: Struttura del dominio (entità e relazioni)
- **State Machine**: Ciclo di vita di entità con stati importanti
- **Sequence**: Interazione tra oggetti (protocolli di comunicazione)
- **Activity**: Processi e flussi (workflow)

#### Limitazioni di UML

- ❌ Non può esprimere **vincoli formali**
- ❌ Non può definire **goals** in logica
- ❌ Non può specificare **domain properties** formalmente
- ✅ Ma è ottimo per **modellazione concettuale**

#### Alloy - Fondamentali

**Signature**:
```alloy
sig Name {}              // Entità semplice
sig Book {               // Entità con relazione
  addr: Name -> lone Addr
}
```

**Cardinalità**:
- `lone`: 0 o 1
- `one`: esattamente 1
- `some`: 1 o più
- `set`: qualsiasi numero

**Join (.)**: Operazione fondamentale
- Match sull'elemento comune
- Elimina l'elemento di join
- Restituisce tuple concatenate

**Relazioni**:
- Ternarie: `sig A { r: B -> C }`
- Binarie: `sig A { r: B }`
- Sempre tipate e ordinate

### 📚 Prossima Lezione

Continueremo con Alloy:
- **Predicati e Fatti** (predicates and facts)
- **Asserzioni** (assertions)
- **Operatori** avanzati
- **Esempi** più complessi
- **Verifica** di proprietà

### ⚠️ Consigli per lo Studio

1. **Provate Alloy Analyzer** personalmente
2. **Sperimentate** con l'esempio Address Book
3. **Modificate** le specifiche e vedete cosa succede
4. **Confrontate** visualizzazioni diverse (grafica, testuale, tabulare)
5. **Fate attenzione** alle differenze Alloy 5 vs Alloy 6 negli esami vecchi

---

**Fine Lezione 6**

*Prossima lezione: Alloy - Predicati, Fatti e Verifica di Proprietà*
