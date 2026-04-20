# SpotOn 🍜

Good food, without the endless searching.

SpotOn is a conversational restaurant recommendation app powered by local LLM (Ollama) and Google Places API. It uses your GPS location or a typed area to find and rank nearby restaurants based on your preferences.

## Features

- Natural language search ("spicy Korean food near convoy")
- GPS-based nearby search with 10km radius
- Conversational follow-up (budget, neighborhood, cuisine)
- Google Places photos, ratings, and Maps links
- Runs fully locally via Ollama

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `qwen2.5:7b` pulled
- Google Places API key (New Places API)

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/spoton.git
cd spoton
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv what_to_eat
source what_to_eat/bin/activate  # macOS/Linux
what_to_eat\Scripts\activate     # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Google API key**
```bash
cp .env.example .env
# Edit .env and add your key
export GOOGLE_API_KEY="your_key_here"
```

**5. Make sure Ollama is running**
```bash
ollama serve
ollama pull qwen2.5:7b
```

**6. Run the app**
```bash
streamlit run app.py
```

## Project Structure

```
spoton/
├── app.py                 # Main Streamlit app
├── agent/
│   └── extractor.py       # LLM-based preference extraction (Ollama)
├── schemas/
│   └── schemas.py         # Preference schema and defaults
├── services/
│   ├── ranking.py         # Restaurant ranking logic
│   └── recommend.py       # Recommendation helpers
├── tools/
│   └── google_places.py   # Google Places API wrapper
└── requirements.txt
```

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google Places API (New) key |
