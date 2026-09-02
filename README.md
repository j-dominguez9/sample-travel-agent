# AI Travel Agent

A simple AI travel agent built on the Anthropic API. It helps users plan trips: searching flights and hotels, checking the weather, and assembling day-by-day itineraries. All travel data comes from local JSON fixtures in `data/` — there are no external API calls beyond the LLM.

It exposes two interfaces:

- an interactive **CLI chat** (`python -m agent.chat`)
- a **FastAPI endpoint** (`POST /chat`) with in-memory multi-turn conversations

## How it works

The agent is a standard Anthropic tool-calling loop, written plainly with no frameworks:

```
agent/
├── config.py   # env vars (model, data dir)
├── prompt.py   # system prompt
├── tools.py    # tool schemas + implementations backed by data/*.json
├── loop.py     # the tool-calling loop (emits a span per tool call)
└── chat.py     # CLI entrypoint
backend/
├── main.py     # FastAPI app
└── tracing.py  # Phoenix / OpenTelemetry setup
common/
├── logging.py       # JSON-lines logging
├── redaction.py     # strips PII from spans at export
└── capabilities.py  # what the agent can't do, enforced in code
data/
├── flights.json
├── hotels.json
└── weather.json
scripts/
└── generate_traffic.py   # sends 50 sample conversations to the API
```

The model can call four tools:

| Tool | What it does |
|---|---|
| `search_flights(origin, destination, date)` | Look up flights between two cities |
| `search_hotels(city, check_in, check_out)` | Look up hotels for a stay |
| `get_weather(city, date)` | Get a forecast for a city |
| `create_itinerary(destination, num_days, notes?)` | Assemble a day-by-day trip plan |

## Setup

Requires Python 3.11+ and an Anthropic API key.

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then configure your key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

Environment variables:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | — | Anthropic API key |
| `ANTHROPIC_MODEL` | no | `claude-haiku-4-5` | Model used by the agent |
| `PHOENIX_COLLECTOR_ENDPOINT` | yes (API) | — | Phoenix collector, e.g. `http://localhost:6006`. The API refuses to start without it; the CLI doesn't need it. |

Each turn's agent span carries `session.id` (the conversation id) and
`travel_agent.turn_index`, so multi-turn conversations group into sessions in
Phoenix and evaluators can tell an opening request from a follow-up. Note that
Phoenix nests dotted attribute namespaces on read — `travel_agent.turn_index`
comes back as `attributes.travel_agent = {"turn_index": N}`, while semantic
conventions like `session.id` stay flat.

The agent can search but not transact, and it says so. Two prompt versions
tried to make it say so by instruction and both measured worse than no rule at
all (72% -> 50% -> 53%), so the disclosure is a guardrail in
`common/capabilities.py`, applied in `agent/loop.py`.

It fires on **what the conversation has been shown**, not on what the user
typed: once, the first time a search puts flights or hotels in front of someone.
Intent detection by keyword was tried first and is not reliable enough — it
missed "I'll take it", "Go ahead with the first option" and "Take the 6:50am
one", which is exactly how a real user commits. `booking_limits_disclosed`
monitors the control the way `no_unredacted_pii` monitors redaction, and is
session-aware: a disclosure at turn 1 covers turn 3.

Cost and token usage come from Phoenix's own pricing table rather than a second
copy of the rates: `evals/core/cost.py` reads per-project cost over a window and
expresses it per conversation, alongside what the evaluation stack itself costs.
The monitoring DAG runs it as `cost_report`, and it is also a CLI:

```bash
uv run python -m evals.core.cost                # the agent's monitoring window
uv run python -m evals.core.cost --window 1440  # last 24h
```

Phoenix's own Dashboards page (per project, with a project selector) covers
totals, trends and the per-model split and is the better place to explore.
What it cannot show is a *rate* — cost per conversation is spend divided by
sessions, and Phoenix has no custom dashboards in this version — which is why
those numbers live here. Judge calls are traced into their
own project by `evals/core/tracing.py` — without that their spend is not
recorded anywhere and the report reads `$0.00`, which looks like "free" rather
than "not measured".

Spans are redacted before export: email addresses, card numbers, phone numbers,
national ID numbers and passport numbers are replaced with `[REDACTED_*]`
markers by `common/redaction.py`, which wraps the OTLP exporter so it also
covers spans created by the Anthropic instrumentor. Redaction is by pattern
rather than by field, because the evaluation framework reads `input.value` and
`output.value` — blanking those would satisfy the privacy requirement and leave
nothing to evaluate.

## Usage

### CLI chat

```bash
uv run python -m agent.chat
```

```
you> Find me a flight from New York to Miami on March 12, 2026.
agent> I found a few options for you! ...
```

Type `quit` (or Ctrl-D) to exit. Conversation history is kept for the session.

### API

Start the server:

```bash
uv run fastapi dev backend/main.py
```

Send a message:

```bash
curl -s localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "I need a hotel in Paris from June 10 to June 14, 2026."}'
```

```json
{"reply": "Here are some hotels in Paris for those dates! ...", "conversation_id": "1f0e..."}
```

To continue a conversation, pass the returned `conversation_id` back:

```bash
curl -s localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Which one is cheapest?", "conversation_id": "1f0e..."}'
```

Conversations are held in memory and reset when the server restarts. `GET /health` returns `{"status": "ok"}`.

### Traffic generator

With the API server running, send 50 varied conversations (~55 turns, including
several multi-turn ones):

```bash
uv run python scripts/generate_traffic.py
uv run python scripts/generate_traffic.py --repeat 2   # ~110 turns
```

The mix is weighted so every evaluator clears its `min_sample` floor — about a
third of the conversations end in nothing available or nothing in scope, which
is what keeps `graceful_alternative` reportable.

Point it at a different host with an argument or env var:

```bash
uv run python scripts/generate_traffic.py http://localhost:9000
# or
TRAVEL_AGENT_URL=http://localhost:9000 uv run python scripts/generate_traffic.py
```

## Notes

- Flight, hotel, and weather data are static fixtures — edit the files in `data/` to change what the agent can find.
- There is no database, auth, or persistence; this is intentionally a minimal service.
