# Ricerca Operativa - Lezione 1

**Data:** 16 Settembre 2025 | **Prof. Federico Malucelli**

## Cos'è la Ricerca Operativa

La Ricerca Operativa è la disciplina delle scienze decisionali che aiuta a prendere decisioni ottimali usando modelli matematici e algoritmi efficienti. Combina matematica (modelli e metodi quantitativi) e informatica (algoritmi efficienti).

**Applicazioni pratiche:**
- Percorsi ottimali (Google Maps)
- Ottimizzazione risorse limitate
- Problemi ambientali
- Gestione logistica e produzione

## Storia

**Origini:** Seconda Guerra Mondiale (termine militare). Alan Turing usò tecniche di ottimizzazione per decifrare i messaggi nazisti, minimizzando le perdite senza rivelare la conoscenza del codice.

**Storia antica:** Il problema isoperimetrico della Regina Didone (814 a.C.) - massimizzare l'area con perimetro fisso usando una pelle di bue per fondare Cartagine. Dimostrazione formale solo nel XVII secolo.

## Livelli Decisionali

1. **Strategico:** lungo termine (es. pianificazione rete metropolitana)
2. **Tattico:** medio termine, 10-20 anni (es. acquisto treni)
3. **Operativo:** orizzonte stagionale (es. schedulazione orari)
4. **Tempo reale:** gestione quotidiana interruzioni

## Metodologia del Corso

**Principio chiave (Confucio):** "Sento e dimentico. Vedo e ricordo. Faccio e capisco."

**Active Learning:** imparare mettendo le mani sui problemi fin dall'inizio, con matematica semplice (solo operazioni base).

## Esempio: Problema dei Cellulari

**Setup:**
- Risorse limitate (pezzi rossi e gialli)
- Due tipi di telefoni (A e B)
- Obiettivo: massimizzare punteggio

**Modello matematico:**
- Variabili: x = telefoni A, y = telefoni B
- Vincoli: 2x + 3y ≤ 9 (rosso), x + y ≤ 4 (giallo), x,y ≥ 0
- Funzione obiettivo: max 5x + 8y

**Soluzione grafica:**
1. Disegnare regione ammissibile
2. Tracciare curve di livello
3. Soluzione ottimale: x=1, y=3
4. Valore ottimale: 29

## Organizzazione del Corso

**Struttura:**
- **Parte 1:** Programmazione Lineare (simplex, dualità, analisi sensibilità)
- **Parte 2:** Programmazione Lineare Intera (problemi NP-hard, branch-and-bound)

**Software:** CPLEX (gratuito per uso accademico), alternativa: GLPK

## Esame

**Struttura (2 ore):**
1. Formulazione modelli (25%)
2. Risoluzione grafica (20%)
3. Algoritmo simplex (25%)
4. Analisi sensibilità (15%)
5. Programmazione intera (15%)

**Materiale consentito:** calcolatrice, righello, matite. **NON consentiti:** computer, telefoni, appunti.

**Valutazione:**
- Voto base: 0-30
- Progetto opzionale: +3 punti bonus
- Appelli: Gennaio, Febbraio, Giugno, Luglio, Settembre

## Risorse e Supporto

- Slides e esercizi su BIP
- Libro: "Ricerca Operativa" di Martello
- Ricevimento: Mercoledì 14:00-16:00
- Email: federico.malucelli@polimi.it
- Forum su BIP per domande

## Consigli per il Successo

**Piano di studio:**
- Dopo ogni lezione: rivedere appunti (30 min)
- Ogni settimana: esercizi aggiuntivi e gruppi di studio (2 ore)
- Prima dell'esame: simulazioni complete (10-15 ore)

**Errori comuni da evitare:**
- Variabili mal definite
- Vincoli incompleti
- Confondere massimizzazione/minimizzazione
- Non verificare ammissibilità della soluzione

## Messaggio Finale

La ricerca operativa non è solo matematica: è un modo di pensare per prendere decisioni migliori. Vedere il mondo come insieme di problemi di ottimizzazione da risolvere.

**Compiti per la prossima lezione:**
1. Risolvere di nuovo il problema dei cellulari
2. Installare CPLEX
3. Leggere Capitolo 1 del libro