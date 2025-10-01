## Panoramica del Corso: Analisi Numerica per il Machine Learning
### Introduzione a Python
*   Una breve introduzione pratica a Python sarà fornita durante la prima settimana (lunedì, martedì e venerdì) per stabilire una base comune.
*   Questo è giustificato da un sondaggio che mostra che il 20% a un terzo degli studenti si sente competente, mentre i restanti due terzi sono nuovi o principianti. La partecipazione è facoltativa per gli studenti competenti.
*   L'introduzione si concentrerà su funzionalità numeriche e gestione dei dati rilevanti per il corso, non su Python generico. Un libro di riferimento e il suo sito web sono disponibili per questa parte.
### Comunicazione e Materiali del Corso
*   **Canale Ufficiale:** WeBeep è la piattaforma ufficiale per tutti i materiali del corso, annunci e comunicazioni.
*   **Contatto:** Gli studenti devono inviare email a entrambi i docenti per qualsiasi domanda per garantire una risposta tempestiva.
*   **Oggetto Email:** Tutte le email devono iniziare con la stringa `M8ML2025` nell'oggetto per un facile filtraggio.
*   **Calendario del Corso:** Un foglio Google, collegato su WeBeep, servirà come calendario del corso. Sarà aggiornato dopo ogni lezione con gli argomenti trattati e i riferimenti ai libri di testo.
*   **Formato delle Lezioni:** Le lezioni e i laboratori sono progettati per la partecipazione in presenza ma saranno registrati e trasmessi in streaming. Gli studenti possono guardare le registrazioni o seguire online in caso di emergenze.
*   **Attrezzatura Necessaria:** Gli studenti devono portare i propri laptop per i laboratori.
## Orario e Logistica
### Orario Settimanale
*   **Lezioni di Python (Solo Questa Settimana):**
    *   Lunedì, Martedì, Venerdì.
    *   Orario di inizio: 13:30, con due slot. La pausa potrebbe essere omessa venerdì.
*   **Orario Regolare (A Partire dalla Prossima Settimana):**
    *   **Lezione del Lunedì:** 16:30 - 18:00 (senza pausa).
    *   **Lezione del Martedì:** 13:30 - 16:00 (con una piccola pausa).
    *   **Laboratorio del Venerdì:** Il primo laboratorio sarà all'inizio di ottobre. L'orario e l'organizzazione saranno decisi con il Dr. Caldana durante quella sessione.
## Contenuto e Argomenti del Corso
### Capitoli Principali
1.  **Algebra Numerica per il Machine Learning:** Una parte significativa del corso.
2.  **Differenziazione Automatica:** Un capitolo cruciale per il deep learning.
3.  **Metodi di Ottimizzazione.**
4.  **Proprietà di Approssimazione delle Reti Neurali:** L'argomento più teorico.
### Ambito
*   Il corso copre argomenti chiave all'intersezione tra metodi numerici, machine learning e ottimizzazione, concentrandosi sui blocchi fondamentali degli algoritmi di machine learning.
## Esame e Struttura della Valutazione
### Componenti dell'Esame
*   **Studenti da 8 Crediti:** Esame scritto e un possibile esame orale.
*   **Studenti da 10 Crediti:** Esame scritto, un possibile esame orale e un progetto.
### Politica dell'Esame Scritto
*   **Formato:** L'esame scritto è "a libro aperto."
*   **Materiali Consentiti:** Gli studenti possono utilizzare i propri appunti, libri e qualsiasi materiale fornito durante il corso.
*   **Azioni Vietate:** L'uso di strumenti AI (es. ChatGPT) per completare l'esame è strettamente proibito e comporterà l'annullamento dell'esame. L'obiettivo è valutare la comprensione e la selezione degli algoritmi, non la memorizzazione.
### Politica dell'Esame Orale
*   L'esame orale è **obbligatorio solo se il punteggio dell'esame scritto è superiore a 28**.
*   Se uno studente ottiene 30, deve sostenere l'esame orale per confermare il voto. Il risultato può essere 30, 30 e lode, o un voto inferiore.
*   Se uno studente con un punteggio >28 sceglie di non sostenere l'esame orale, il suo voto sarà automaticamente abbassato a 28.
*   Le parti scritte e orali dell'esame devono essere sostenute nella stessa sessione.
### Calcolo del Voto Finale (Studenti da 10 Crediti)
*   Il voto finale è una media ponderata:
    *   **80%** dal voto dell'esame scritto/orale.
    *   **20%** dal voto del progetto.
## Dettagli del Progetto (Studenti da 10 Crediti)
### Tipi di Progetto
1.  **Riproduzione dei Risultati di un Articolo (Gruppi di 1-2):**
    *   Scegliere un articolo da un elenco fornito dal docente.
    *   Riprodurre alcuni o tutti i risultati dell'articolo.
    *   **Consegne:** Un rapporto (problema, metodo, risultati) e il codice sviluppato.
2.  **Esplorazione Approfondita di un Argomento (Solo Individuale):**
    *   Esplora un argomento che è stato trattato solo brevemente nel corso.
    *   **Consegne:** Un rapporto dettagliato e una presentazione sull'argomento scelto.
### Tempistiche e Consegna del Progetto
*   La presentazione del progetto può essere posticipata rispetto all'esame scritto/orale.
*   Gli studenti hanno un anno da quando viene deciso l'argomento per presentare il progetto (l'ultima sessione valida è il settembre successivo).
*   Se due studenti lavorano insieme ma hanno orari d'esame diversi, possono comunque presentare il progetto. Il voto sarà registrato e applicato al voto finale del secondo studente dopo il completamento dei suoi esami.
## Ambiente Python e Strumenti del Corso
*   **Lingua Principale:** Python.
*   **Opzioni di Configurazione:**
    *   **Installazione Locale:** Si consiglia vivamente la distribuzione Anaconda in quanto include la maggior parte dei moduli necessari.
    *   **Basato su Cloud (Principale per i Laboratori):** Google Colab sarà lo strumento principale utilizzato durante i laboratori. Richiede un'account Google e Wi-Fi ma elimina la necessità di installazione locale.
*   **Materiali per il Laboratorio:** Tutti i laboratori saranno forniti come notebook (`.ipynb`), compatibili sia con Google Colab che con le installazioni locali di Jupyter.
## Fondamenti di Python
### Nozioni di Base su Google Colab/Jupyter Notebook
*   **Tipi di Celle:** I notebook sono composti da celle di codice (per Python) e celle di testo (per testo formattato e formule matematiche utilizzando LaTeX).
*   **Esecuzione:** Usa `Shift + Enter` per eseguire il contenuto di qualsiasi cella.
### Primo Esempio: Conversione da Fahrenheit a Celsius
1.  **Input Utente:** Usa `input()` per ottenere dati come stringa.
2.  **Casting del Tipo:** Converti la stringa in un numero usando `float()`.
3.  **Calcolo:** Applica la formula: `celsius = (fahrenheit - 32) * 5/9`.
4.  **Output:** Visualizza il risultato con `print()`.
### Funzioni Python: Definizione e Utilizzo
*   **Definizione:** Usa la parola chiave `def` seguita dal nome della funzione e dai parametri, terminando con due punti (es. `def nome_funzione(param1):`).
*   **Indentazione:** I blocchi di codice all'interno di una funzione sono definiti da un'indentazione obbligatoria. Un'indentazione incoerente causerà un errore.
*   **Restituzione di Valori:** Il comando `return` specifica l'output della funzione. Una funzione può restituire più valori (es. `return lunghezza, area`).
### Argomenti Avanzati delle Funzioni
*   **Operatore di Potenza:** Usa il doppio asterisco `**` per l'elevamento a potenza (es. `V**2`).
*   **Argomenti Predefiniti (Opzionali):** Assegna un valore predefinito nella firma della funzione (es. `def func(n, a=0.1, b=0.03):`). Questo consente di chiamare la funzione con meno argomenti.
*   **Argomenti Posizionali vs. per Parola Chiave:**
    *   **Posizionali:** Argomenti richiesti senza valori predefiniti. Devono apparire per primi.
    *   **Per Parola Chiave:** Argomenti opzionali con valori predefiniti. Possono essere forniti fuori ordine se specificati per nome (es. `my_func(b=valore1, a=valore2)`).
    *   **Regola:** Gli argomenti posizionali devono sempre venire prima degli argomenti per parola chiave.
### Moduli Python e Documentazione del Codice
*   **Moduli:** Pacchetti che forniscono funzioni aggiuntive (come "cassetti degli attrezzi"). Devono essere esplicitamente importati.
*   **Importazione di Moduli:**
    *   `import math`: Importa l'intero modulo (accesso con `math.pi`).
    *   `import matplotlib.pyplot as plt`: Importa con un alias standard. **Questa è la pratica consigliata.**
    *   `from math import pi`: Importa solo un elemento specifico.
    *   `from math import *`: Importa tutto nello spazio dei nomi globale. **Fortemente sconsigliato** a causa dei potenziali conflitti di nomi.
*   **Docstring:** Una stringa tra virgolette triple (`"""..."""`) immediatamente dopo una definizione di funzione. Serve come documentazione ufficiale, accessibile tramite `help(nome_funzione)`.
## 📅 Prossimi Appuntamenti
- [ ] Partecipare alle lezioni introduttive di Python martedì e venerdì di questa settimana.
- [ ] Portare il proprio laptop per i laboratori, a partire dal primo all'inizio di ottobre.
- [ ] Quando si inviano email ai docenti, includere entrambi i destinatari e iniziare l'oggetto con `M8ML2025`.
- [ ] Monitorare WeBeep per le diapositive, gli annunci e il link al calendario del corso su Google Sheet.
- [ ] Durante il primo laboratorio all'inizio di ottobre, coordinarsi con il Dr. Caldana per organizzare la struttura del laboratorio.
- [ ] Scrivere una funzione Python per convertire una temperatura da Kelvin a Fahrenheit.
- [ ] Scrivere una funzione Python per calcolare la temperatura dall'equazione di Van der Waals, dati tutti gli altri parametri, utilizzando valori predefiniti per le costanti dei gas dell'idrogeno.
- [ ] (Studenti da 10 Crediti) Attendere l'elenco degli articoli da parte del docente per iniziare a considerare gli argomenti del progetto.
- [ ] La prossima sessione continuerà dall'esempio di calcolo del cerchio per discutere della gestione delle funzioni che restituiscono più output.