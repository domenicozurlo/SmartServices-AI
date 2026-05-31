## Obiettivo

L’obiettivo del progetto è sostituire l’attuale workflow di research con un nuovo **workflow principale operativo**, pensato per analizzare le richieste degli utenti e soddisfarle attraverso un gruppo di agenti specializzati.
Il custom ednpoint di librechat, non è più DemoAgent, ma diventa SmartServiceFlow. Gli handler sono l' agent di demo già presente e poi il nostro nuovo workflow

Il workflow deve essere in grado di:

1. Comprendere e classificare la richiesta in ingresso.
2. Applicare controlli iniziali di sicurezza e coerenza tramite guardrail.
3. Riscrivere la richiesta dell’utente in una forma più chiara e adatta al routing.
4. Selezionare in modo deterministico il ramo operativo corretto.
5. Attivare l’agente specializzato più adatto.
6. Rifiutare tutte le richieste che non rientrano nei casi d’uso previsti.

Il sistema dovrà quindi comportarsi come un orchestratore centrale capace di gestire richieste relative a knowledge base documentale, knowledge base strutturata e prenotazioni.

## Descrizione generale

Il nuovo workflow principale deve gestire tre macro-tipologie di richieste:

1. **Domande su knowledge base documentale interna**

   L’agente deve poter rispondere a domande basate su documenti interni, guide utente, procedure operative e contenuti che possono includere anche immagini.
   Per questa funzionalità, l’agente non implementa direttamente la logica RAG, ma utilizza il servizio esterno `rag_api`, seguendo lo stesso approccio già adottato da LibreChat.

2. **Domande su knowledge base strutturata**

   L’agente deve poter rispondere a domande basate su dati strutturati, in particolare dati SQL.
   Per questa parte è necessario riutilizzare l’approccio già presente nell’esempio `structured_data_qa`.

3. **Gestione prenotazioni**

   L’agente deve poter effettuare prenotazioni e inserirle in un sistema calendario o in un’integrazione equivalente. Qui sarà necessario lo user approval prima di procedere cn l'inserimento della prenotazione.

---

## Flusso principale

Il workflow deve seguire una logica chiara e deterministica.

### 1. Guardrail iniziale

All’inizio del workflow deve essere presente un controllo di guardrail.

Il guardrail deve verificare che la richiesta sia coerente con gli ambiti supportati dal sistema, senza utilizzare le API di moderazione.

Il controllo deve bloccare o respingere richieste non pertinenti, non supportate o fuori perimetro.

---

### 2. Classification Agent

Dopo il guardrail, la richiesta deve essere gestita da un classification agent.

Il classification agent ha il compito di:

* analizzare la richiesta dell’utente;
* effettuare una query rewrite della richiesta originale;
* identificare l’intento principale;
* decidere quale ramo del workflow deve essere attivato;
* produrre un output strutturato utile al routing deterministico.

---

### 3. Routing deterministico

Il workflow deve instradare la richiesta verso uno dei tre rami previsti:

#### A. Document Knowledge Base Agent

Questo ramo gestisce richieste relative a documenti interni, procedure, manuali, guide utente e contenuti multimodali che possono includere immagini.

L’agente deve interrogare il servizio `rag_api`, analogamente a quanto già avviene in LibreChat.

Esempi di richieste supportate:

* “Come si resetta questo dispositivo?”
* “Qual è la procedura per attivare il servizio?”
* “Nel manuale, cosa significa questo errore?”
* “Mostrami i passaggi della guida utente.”

---

#### B. Structured Data QA Agent

Questo ramo gestisce richieste basate su dati strutturati SQL.

L’agente deve poter tradurre la richiesta dell’utente in una interrogazione o in un processo di analisi sui dati disponibili, seguendo il modello dell’esempio `structured_data_qa`.

Esempi di richieste supportate:

* “Quanti clienti hanno prenotazioni attive?”
* “Qual è il numero di richieste aperte per categoria?”
* “Mostrami i dati relativi agli ordini dell’ultimo mese.”
* “Quali sono i clienti con stato pending?”

---

#### C. Booking Agent

Questo ramo gestisce richieste di prenotazione.

L’agente deve raccogliere le informazioni necessarie, validare i dati disponibili e creare una prenotazione all’interno di un calendario o sistema equivalente.

Esempi di richieste supportate:

* “Vorrei prenotare un appuntamento per domani mattina.”
* “Puoi fissarmi una chiamata con un operatore?”
* “Prenota uno slot disponibile nel calendario.”
* “Sposta il mio appuntamento a venerdì.”

---

## Gestione delle richieste fuori perimetro

Qualsiasi richiesta che non rientri nei tre task supportati deve essere respinta.

Il sistema deve quindi rifiutare richieste che non riguardano:

1. knowledge base documentale interna;
2. knowledge base strutturata SQL;
3. prenotazioni/calendario.

La risposta di rifiuto deve essere chiara, professionale e coerente, ad esempio:

> Mi dispiace, ma posso supportarti solo per richieste relative alla knowledge base interna, ai dati strutturati disponibili o alla gestione delle prenotazioni.

---

## Requisiti tecnici

### Workflow

Il workflow principale deve sostituire il workflow di research.

Il nuovo workflow deve essere costruito secondo una logica agentica basata su OpenAI Agent SDK, prendendo come riferimento gli esempi presenti nella cartella:

`agents_workflow_examples`

Questi esempi devono essere usati come guida per applicare le best practice relative a:

* orchestrazione degli agenti;
* gestione del contesto;
* output strutturato;
* routing condizionale;
* gestione degli errori;
* separazione delle responsabilità tra agenti;
* uso corretto dei tool.
* gurdrail

---

### RAG documentale

Per la knowledge base interna documentale, l’agente deve utilizzare il servizio:

`rag_api`

Questo servizio sarà responsabile della ricerca e del recupero delle informazioni dalla knowledge base, inclusi eventuali riferimenti a immagini presenti nei documenti.

L’agente deve quindi limitarsi a:

* ricevere la richiesta riscritta;
* inviarla a `rag_api`;
* ricevere il risultato;
* formulare la risposta finale all’utente, cosi come la manderebbe rag_api al front end librechat

---

### Dati strutturati SQL

Per la parte di interrogazione dati SQL, il riferimento implementativo deve essere l’esempio:

`structured_data_qa`

Questo componente deve essere adattato al nuovo workflow principale per consentire all’agente di rispondere a domande basate su dati strutturati.

---

### Prenotazioni

Per la gestione delle prenotazioni, deve essere previsto un agente dedicato che possa:

* comprendere la richiesta di prenotazione;
* raccogliere eventuali dati mancanti;
* verificare disponibilità;
* creare o aggiornare l’appuntamento;
* confermare l’esito all’utente.

L’integrazione potrà essere realizzata verso un calendario o altro sistema equivalente.

---

## Architettura logica

Il flusso logico previsto è il seguente:

1. Richiesta utente in ingresso.
2. Guardrail iniziale.
3. Classification Agent con query rewrite.
4. Routing deterministico.
5. Attivazione di uno dei tre agenti:

   * Document Knowledge Base Agent;
   * Structured Data QA Agent;
   * Booking Agent.
6. Generazione della risposta o completamento dell’azione.
7. Gestione fallback per richieste non supportate.

---

## Sintesi finale

Il nuovo workflow deve diventare il punto centrale di orchestrazione per richieste operative del cliente.

Non deve essere un semplice workflow di ricerca, ma un sistema capace di classificare, instradare e soddisfare richieste attraverso agenti specializzati.

Il perimetro funzionale iniziale è limitato a tre casi d’uso:

1. risposta su knowledge base documentale interna, anche con immagini;
2. risposta su knowledge base strutturata SQL;
3. gestione prenotazioni su calendario.

Tutto ciò che esula da questi tre ambiti deve essere respinto.
