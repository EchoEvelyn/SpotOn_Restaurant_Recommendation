# SpotOn 🍜

**Good food, without the endless searching.**

SpotOn is a conversational restaurant recommendation app. Tell it what you're in the mood for in plain language — a cuisine, a vibe, a budget, a neighborhood — and it finds the best nearby options, asks smart follow-up questions, and refines results as you talk.

Everything runs locally. No cloud AI subscriptions required.

---

## How it works

1. You type something like *"spicy Korean food near convoy"* or *"cheap ramen downtown"*
2. A local LLM (Ollama / qwen2.5:7b) extracts your preferences — cuisine, budget, location, dietary needs
3. Google Places API searches for matching restaurants
4. A local ranking algorithm scores results on rating, review count, cuisine match, budget fit, and preference signals
5. The app shows results and asks follow-up questions to tighten the search (neighborhood, budget, cuisine) when useful

---

## Features

- Natural language input in any language
- GPS-based nearby search (10 km radius) with browser location
- Multi-turn conversation — refine by budget, area, or cuisine after seeing results
- Smart follow-up prompts only when they add value
- Google Places photos, ratings, review counts, and Maps links
- Fully local LLM inference via Ollama — no OpenAI key needed

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Ollama (`qwen2.5:7b`) |
| Search | Google Places API (New) |
| Location | Browser GPS via `streamlit-js-eval` |
| Ranking | Custom scoring (local, no ML) |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- `qwen2.5:7b` model pulled in Ollama
- Google Places API key (New Places API, not the legacy one)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/spoton.git
cd spoton
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv what_to_eat
source what_to_eat/bin/activate   # macOS / Linux
what_to_eat\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Google API key

```bash
cp .env.example .env
```

Edit `.env` and fill in your key:

```
GOOGLE_API_KEY=your_google_places_api_key_here
```

Then export it before running:

```bash
export GOOGLE_API_KEY="your_google_places_api_key_here"
```

### 5. Start Ollama and pull the model

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 6. Run the app

```bash
streamlit run app.py
```

---

## Project structure

```
spoton/
├── app.py                  # Main Streamlit app — UI, search flow, conversation logic
├── agent/
│   └── extractor.py        # LLM calls via Ollama: preference extraction, follow-up parsing
├── schemas/
│   └── schemas.py          # Preference schema and default values
├── services/
│   ├── ranking.py          # Restaurant scoring and ranking algorithm
│   └── recommend.py        # CLI entrypoint for standalone testing
├── tools/
│   └── google_places.py    # Google Places API wrapper (text search + nearby search)
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Ranking algorithm

Each restaurant is scored across five dimensions:

| Signal | Weight |
|---|---|
| Google rating | 35% |
| Review count (log-scaled) | 17% |
| Cuisine match | 20% |
| Preference match (spicy, healthy, etc.) | 13% |
| Budget match | 15% |

The final score is multiplied by an avoid penalty (e.g. if the user said "no spicy").

---

## Google Places API setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Places API (New)**
3. Create an API key under Credentials
4. (Recommended) Restrict the key to Places API only

The app uses two endpoints:
- `places:searchText` — keyword + location text search
- `places:searchNearby` — GPS-based nearby search

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google Places API (New) key |