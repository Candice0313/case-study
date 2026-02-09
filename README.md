# PartSelect Domain-Focused Chat Agent

A conversational agent for **refrigerator and dishwasher** parts support: find parts by model, check compatibility, get installation and troubleshooting guidance. The system is domain-locked, evidence-grounded, and built as a case study in structured agent design.

---

## Project overview

The repo implements a **single-domain chat assistant** aligned to a PartSelect-style product: users ask in natural language about appliance parts, model numbers, compatibility, and repair; the backend classifies scope, routes to the right capability (RAG troubleshooting, parts list, part lookup, compatibility, or clarification), and returns an answer plus optional **Sources** (citations) and **Suggested parts** (product cards). The stack is **FastAPI (Python)** for the agent and **Next.js (App Router)** for the chat UI; optional **PostgreSQL + pgvector** for RAG over repair guides and part catalog.

**Main components:**

| Layer        | Role |
|-------------|------|
| **Scope router** | Classifies each message as in-scope (appliance parts/support) or out-of-scope; ambiguous or off-topic requests get a polite redirect. |
| **Agent (LangGraph)** | State machine: triage or LLM planner → clarify / parts_list / part_lookup / compatibility / find_model_help / retrieve (RAG). Produces `answer`, `citations`, `product_cards`. |
| **Tools**   | `get_troubleshooting` (RAG), `part_lookup`, `search_parts`, `check_compatibility`, Serp for model/part pages. |
| **Web app** | Chat UI with markdown answers, Sources list, and product cards linking to PartSelect. |

---

## Product goal & non-goals

**Goals**

- Help users **find parts** for a given model (parts list, model overview, symptom-based suggestions).
- **Check compatibility** between a part (e.g. PS number) and a model.
- Provide **troubleshooting** guidance (not cooling, not draining, leaking, ice maker, etc.) grounded in ingested repair guides, with citations.
- Support **“Where is my model number?”** with short guidance and links to refrigerator/dishwasher locator pages.
- Stay **in-domain**: refrigerators and dishwashers only; refuse or redirect off-topic and out-of-scope requests.
- Prefer **evidence and structured data** (RAG, DB, Serp) over free-form LLM generation where possible.

**Non-goals**

- No support for other appliances (washer, dryer, oven, microwave, etc.).
- No order placement, cart, or checkout (informational only).
- No medical, legal, coding, or general-knowledge Q&A.
- No open-ended chitchat beyond a short welcome and redirect to parts/support.

---

## User experience flow

1. **User sends a message** in the chat (e.g. “My fridge is not cooling”, “parts for WRF535SWHZ”, “is PS123 compatible with model X?”).
2. **Scope** is classified: in-scope → agent; out-of-scope/ambiguous → fixed redirect message, no graph run.
3. **Agent** runs:
   - **Triage or LLM planner** sets `next_action` (clarify, parts_list, part_lookup, compatibility, find_model_help, retrieve).
   - **Clarify**: reply with a short question (e.g. “What’s your model number?”) and no links/cards.
   - **Parts list / part lookup / compatibility / find_model_help**: dedicated nodes return an answer plus citations and/or product_cards.
   - **Retrieve**: RAG over repair guides → evidence → compose answer with citations; no product cards for pure troubleshoot.
4. **Response** is returned as: **content** (markdown), **sources** (citations for “Sources” in UI), **product_cards** (for “Suggested parts” / part links). Frontend renders markdown, source links, and cards; PartSelect base URLs are resolved to model/part pages where possible.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Web (Next.js)                                                           │
│  Chat UI → /chat, /chat/stream → API                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  API (FastAPI)                                                           │
│  Scope router → run_agent(message, scope_label, history)                 │
│  Agent: build initial state → LangGraph.ainvoke → postprocess           │
│  → content, sources, product_cards                                       │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
   config/              app/agent_graph        app/tools
   scope_contract.json   (LangGraph)            RAG, part_lookup,
   source_policy.json   triage / llm_planner   search_parts, Serp
   state_guide_links     → nodes → END          check_compatibility
```

**Repository layout:**

```
case-study/
├── config/                 # Domain and routing config
│   ├── scope_contract.json # Intents, entities, patterns, forbidden topics
│   ├── source_policy.json  # State → symptom tags, appliance, adjacent states (RAG filter)
│   └── state_guide_links.json
├── apps/
│   ├── api/                # FastAPI app
│   │   ├── app/            # scope_router, agent, agent_graph, tools, evidence, retrieval, …
│   │   ├── main.py         # /chat, /chat/stream
│   │   ├── schema.sql      # Postgres + pgvector (parts, models, chunks, embeddings)
│   │   └── requirements.txt
│   └── web/                # Next.js (App Router)
│       ├── app/            # page, layout
│       ├── components/     # Chat, ProductCard, Header
│       └── package.json
└── scripts/
    └── ingest/             # RAG: chunk, embed, load; optional model/parts fetch
```

---

## Agent state & routing design

The agent is a state machine: each turn it carries forward context (user message, extracted model/part, appliance type, etc.), chooses one **next action**, runs the matching node, then merges the result back into state. The graph runs until it reaches an answer.

**What the agent tracks (state)**  
`TroubleshootingState` holds inputs (e.g. `message`, `scope_label`, `model_number`, `part_number`, `appliance_type`, `intent`, `current_state`), planner outputs (`next_action`, `planner_next_action`, `action_args`, `info_type`), and outputs (e.g. `evidence`, `answer`, `citations`, `product_cards`). Each node returns a partial update that gets merged into this state.

**How the next action is chosen**  
- **LLM planner** (default when `USE_LLM_ROUTER_PLANNER=1`): the graph sends the message to `llm_router` → `llm_planner`, which sets `next_action` (and optional slots like model/part).  
- **Rule-based triage**: otherwise, deterministic rules set `next_action`. For refrigerators, a **cooling_split** step can refine the path (e.g. “fridge and freezer both warm” vs “freezer cold, fridge warm”) before clarify or RAG.

**What each action does (routing)**  

| `next_action` | What happens | Output |
|---------------|--------------|--------|
| **ask_clarify** | Ask the user for missing info (e.g. model or part number). | Short question; no citations or product cards. |
| **parts_list_answer** | Answer “parts for model X” using Serp and optional DB. | Answer + product cards (model/symptom/part links). |
| **part_lookup_answer** | Look up a part number (e.g. PS…) via part_lookup or Serp. | Answer + optional single product card. |
| **compatibility_answer** | Check whether a part fits a model. | Yes/no + model link. |
| **find_model_help** | Explain where to find the model number. | Short text + Sources (refrigerator/dishwasher locator links); no product cards. |
| **retrieve** | Run RAG over repair guides: get_troubleshooting → evidence → compose answer with citations. | Answer + citations; no product cards for pure troubleshooting. |

**Source policy**  
`config/source_policy.json` defines, per diagnostic state, which symptom tags and appliance types are allowed or forbidden. That filters RAG and suggested links so refrigerator questions don’t get dishwasher links and vice versa.

---

## Data & knowledge strategy

- **Scope contract** (`config/scope_contract.json`): allowed intents, entities, brands, forbidden topics, patterns; used by the scope router and kept in sync with product scope.
- **RAG**: Repair guides (HTML) are chunked, embedded (OpenAI), and stored in Postgres (pgvector). `get_troubleshooting` retrieves by embedding + optional state/symptom filter; evidence is passed to a compose step (claim-based or fallback) with citations from chunks. Suggested links for troubleshoot (no model) are capped and filtered by appliance type.
- **Structured data**: `parts`, `models`, `part_fitment` (or compatibility), `model_sections`, `section_parts` for part lookup and compatibility. When DB has no hit, the agent can fall back to Serp + LLM summarization for model or part.
- **Serp**: Used for model/part page discovery, symptom-specific part lists, and URL resolution when only a title is available. No Playwright in the main answer path; optional live fetch behind a feature flag.

---

## Example conversations

| User | System behavior |
|------|------------------|
| “Where can I find my model number?” | find_model_help: short guidance + Sources (Refrigerator / Dishwasher locator links). No product card. |
| “Parts for WRF535SWHZ” | parts_list: answer + product_cards (model overview, symptom/part links as applicable). |
| “Is PS11752778 compatible with WDT780SAEM1?” | compatibility_answer: Yes/No + model link. |
| “My fridge is not making ice” | retrieve: RAG over repair guides → answer + citations (Suggested links). No suggested parts unless user later asks for parts for a model. |
| “Install PS12345678” (no model/part in message) | ask_clarify: “Share model number or part number so I can help.” |
| “Write me a poem” | Scope router: out-of-scope → redirect to appliance parts/support only. |

---

## Frontend & UX

- **Stack**: Next.js (App Router), Tailwind, React Markdown. Chat calls `POST /chat` or SSE `GET /chat/stream` with history; response includes `content`, `sources`, `product_cards`.
- **Rendering**: Markdown for the reply; **Sources** as a list of links below the answer; **product_cards** as compact cards (title, “View on PartSelect →”). PartSelect base URLs are rewritten client-side to model or part URLs when possible (using model from content or card name).
- **Scoping**: No change to UI for in/out-of-scope; the user sees either the agent reply or the single redirect message.

---

## Extensibility & future work

- **New appliances**: Extend `scope_contract` and `source_policy`, add appliance-specific states and routes, and extend RAG/ingest to new guide sets.
- **New intents**: Add nodes and `next_action` values (e.g. order status, warranty); wire planner/triage to new actions and keep citations/cards semantics.
- **Stronger RAG**: Finer-grained chunks, hybrid search, or re-ranking; keep citation-to-chunk contract so answers stay grounded.

---

## Quick start

**API (FastAPI)**

```bash
cd apps/api && python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Web (Next.js)**

From repo root:

```bash
npm run dev
```

Or from `apps/web`: `npm install && npm run dev`. Open http://localhost:3000. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` if the API runs elsewhere.

**Database (optional, for full RAG)**

```bash
createdb partselect
psql partselect -f apps/api/schema.sql
```

**Environment**

- `OPENAI_API_KEY`: used for scope (optional), LLM planner/slots (when `USE_LLM_ROUTER_PLANNER=1`), RAG compose, and Serp summarization.
- `SERPAPI_API_KEY`: **recommended** for full agent behavior. Used for PartSelect model/part page discovery, “find model number” links, parts list and part lookup fallbacks, and compatibility link resolution. Without it, those flows return empty or skip Serp; get a key at [serpapi.com](https://serpapi.com/search-api).
- `DATABASE_URL`: for RAG and part lookup (optional if using Serp/fallbacks only).
- See **`apps/api/.env.example`** for API env vars (copy to `apps/api/.env`); see `scripts/ingest/.env.example` for ingest-related variables.
