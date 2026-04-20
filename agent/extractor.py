import json
import requests
from copy import deepcopy

from schemas.schemas import DEFAULT_PREFERENCES


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()


def extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start:end + 1]


def build_extraction_prompt(user_input: str) -> str:
    return f"""
You are an information extraction system for a restaurant recommendation agent.

Extract dining preferences from the user request into JSON.

Rules:
- Return JSON only.
- Do not include any explanation.
- Use only the allowed values below.
- If information is missing, use null for scalar fields and [] for list fields.
- Do not invent information.
- Keep cuisine values lowercase.
- Put positive food or meal preferences into "preferences".
- Examples of positive preferences: spicy, healthy, light, warm, comforting, quick, crispy.
- Use "avoid" only for things the user explicitly does not want.
- If the user says "I want something spicy", put "spicy" in "preferences", not in "avoid".
- If the user says "not spicy", put "spicy" in "avoid".

Allowed schema:
{{
  "cuisine": [],
  "budget_level": "cheap|moderate|expensive|null",
  "budget_max": null,
  "distance_preference": "near|flexible|null",
  "party_size": null,
  "preferences": [],
  "avoid": [],
  "location_text": ""
}}

Allowed preferences values:
["spicy", "healthy", "light", "warm", "comforting", "quick", "crispy"]

Allowed avoid values:
["spicy", "cold_food", "fried"]

Examples:

User: cheap sushi for two people
Output:
{{
  "cuisine": ["sushi"],
  "budget_level": "cheap",
  "budget_max": null,
  "distance_preference": null,
  "party_size": 2,
  "preferences": [],
  "avoid": [],
  "location_text": ""
}}

User: I want something healthy and light, maybe Vietnamese
Output:
{{
  "cuisine": ["vietnamese"],
  "budget_level": null,
  "budget_max": null,
  "distance_preference": null,
  "party_size": null,
  "preferences": ["healthy", "light"],
  "avoid": [],
  "location_text": ""
}}

User: something not too expensive
Output:
{{
  "cuisine": [],
  "budget_level": "cheap",
  "budget_max": null,
  "distance_preference": null,
  "party_size": null,
  "preferences": [],
  "avoid": [],
  "location_text": ""
}}

User: under 20 dollars
Output:
{{
  "cuisine": [],
  "budget_level": "cheap",
  "budget_max": 20,
  "distance_preference": null,
  "party_size": null,
  "preferences": [],
  "avoid": [],
  "location_text": ""
}}

User: I want something spicy
Output:
{{
  "cuisine": [],
  "budget_level": null,
  "budget_max": null,
  "distance_preference": null,
  "party_size": null,
  "preferences": ["spicy"],
  "avoid": [],
  "location_text": ""
}}

User: I do not want anything spicy
Output:
{{
  "cuisine": [],
  "budget_level": null,
  "budget_max": null,
  "distance_preference": null,
  "party_size": null,
  "preferences": [],
  "avoid": ["spicy"],
  "location_text": ""
}}

Now extract from this request:

User: {user_input}
Output:
""".strip()


def build_followup_prompt(user_input: str, question_type: str) -> str:
    if question_type == "budget":
        schema = """
{
  "budget_level": "cheap|moderate|expensive|null",
  "budget_max": null
}
"""
        instruction = """
Extract only budget-related information.
If the user gives a numeric budget, put it in budget_max as an integer.
If the user gives only a vague budget preference, fill budget_level.
"""

    elif question_type == "location":
        schema = """
{
  "location_text": ""
}
"""
        instruction = """
Extract only the location text from the user's reply.
Return an empty string if missing.
"""

    elif question_type == "cuisine":
        schema = """
{
  "cuisine": []
}
"""
        instruction = """
Extract only cuisine information from the user's reply.
Return cuisine values as lowercase strings in a list.
"""

    else:
        raise ValueError(f"Unsupported question_type: {question_type}")

    return f"""
You are extracting follow-up information for a restaurant recommendation agent.

Return JSON only.
Do not include any explanation.

{instruction}

Schema:
{schema}

User reply:
{user_input}

Output:
""".strip()


def normalize_preferences(raw_prefs: dict) -> dict:
    prefs = deepcopy(DEFAULT_PREFERENCES)

    # cuisine
    cuisine = raw_prefs.get("cuisine", [])
    if isinstance(cuisine, list):
        cleaned_cuisine = []
        for item in cuisine:
            item = str(item).strip().lower()
            if item:
                cleaned_cuisine.append(item)
        prefs["cuisine"] = list(dict.fromkeys(cleaned_cuisine))

    # budget_level
    budget = raw_prefs.get("budget_level")
    if isinstance(budget, str):
        budget = budget.strip().lower()

    budget_map = {
        "cheap": "cheap",
        "budget": "cheap",
        "low": "cheap",
        "affordable": "cheap",
        "not too expensive": "cheap",
        "moderate": "moderate",
        "mid-range": "moderate",
        "midrange": "moderate",
        "expensive": "expensive",
        "high-end": "expensive",
        "upscale": "expensive",
        None: None,
    }
    if budget in budget_map:
        prefs["budget_level"] = budget_map[budget]

    # budget_max
    budget_max = raw_prefs.get("budget_max")
    if isinstance(budget_max, int) and budget_max > 0:
        prefs["budget_max"] = budget_max
    elif isinstance(budget_max, str):
        digits = "".join(ch for ch in budget_max if ch.isdigit())
        if digits:
            value = int(digits)
            if value > 0:
                prefs["budget_max"] = value

    # distance_preference
    distance = raw_prefs.get("distance_preference")
    if isinstance(distance, str):
        distance = distance.strip().lower()

    distance_map = {
        "near": "near",
        "close": "near",
        "nearby": "near",
        "flexible": "flexible",
        "far is okay": "flexible",
        None: None,
    }
    if distance in distance_map:
        prefs["distance_preference"] = distance_map[distance]

    # party_size
    party_size = raw_prefs.get("party_size")
    if isinstance(party_size, int) and party_size > 0:
        prefs["party_size"] = party_size
    elif isinstance(party_size, str) and party_size.isdigit():
        value = int(party_size)
        if value > 0:
            prefs["party_size"] = value

    # preferences
    allowed_preferences = {
        "spicy",
        "healthy",
        "light",
        "warm",
        "comforting",
        "quick",
        "crispy",
    }
    preference_aliases = {
        "comfort": "comforting",
        "comfort food": "comforting",
        "cozy": "comforting",
        "hot": "warm",
        "fast": "quick",
        "healthy food": "healthy",
    }

    preferences = raw_prefs.get("preferences", [])
    if isinstance(preferences, list):
        cleaned = []
        for item in preferences:
            item = str(item).strip().lower()
            item = preference_aliases.get(item, item)
            if item in allowed_preferences:
                cleaned.append(item)
        prefs["preferences"] = list(dict.fromkeys(cleaned))

    # avoid
    allowed_avoid = {"spicy", "cold_food", "fried"}
    avoid_aliases = {
        "spicy food": "spicy",
        "cold": "cold_food",
        "cold dish": "cold_food",
        "oily": "fried",
    }

    avoid = raw_prefs.get("avoid", [])
    if isinstance(avoid, list):
        cleaned = []
        for item in avoid:
            item = str(item).strip().lower()
            item = avoid_aliases.get(item, item)
            if item in allowed_avoid:
                cleaned.append(item)
        prefs["avoid"] = list(dict.fromkeys(cleaned))

    # location_text
    location_text = raw_prefs.get("location_text", "")
    if isinstance(location_text, str):
        prefs["location_text"] = location_text.strip()

    return prefs


def apply_sanity_checks(prefs: dict, user_input: str) -> dict:
    """
    Small guardrails for high-risk semantic mistakes.
    """
    text = user_input.lower()

    positive_spicy_patterns = [
        "want something spicy",
        "want spicy",
        "something spicy",
        "spicy food",
        "like spicy",
        "love spicy",
        "craving spicy",
    ]
    negative_spicy_patterns = [
        "not spicy",
        "no spice",
        "avoid spicy",
        "don't want spicy",
        "do not want spicy",
        "nothing spicy",
    ]

    if any(p in text for p in positive_spicy_patterns):
        prefs["avoid"] = [x for x in prefs["avoid"] if x != "spicy"]
        if "spicy" not in prefs["preferences"]:
            prefs["preferences"].append("spicy")

    if any(p in text for p in negative_spicy_patterns):
        prefs["preferences"] = [x for x in prefs["preferences"] if x != "spicy"]
        if "spicy" not in prefs["avoid"]:
            prefs["avoid"].append("spicy")

    prefs["preferences"] = list(dict.fromkeys(prefs["preferences"]))
    prefs["avoid"] = list(dict.fromkeys(prefs["avoid"]))

    return prefs


def extract_preferences_llm(user_input: str, model: str = OLLAMA_MODEL) -> dict:
    prompt = build_extraction_prompt(user_input)
    raw_output = call_ollama(prompt, model=model)
    json_text = extract_json_block(raw_output)
    parsed = json.loads(json_text)
    normalized = normalize_preferences(parsed)
    normalized = apply_sanity_checks(normalized, user_input)
    return normalized


def extract_followup_update_llm(
    user_input: str,
    question_type: str,
    model: str = OLLAMA_MODEL
) -> dict:
    prompt = build_followup_prompt(user_input, question_type)
    raw_output = call_ollama(prompt, model=model)
    json_text = extract_json_block(raw_output)
    parsed = json.loads(json_text)

    if question_type == "budget":
        normalized = normalize_preferences(parsed)
        return {
            "budget_level": normalized.get("budget_level"),
            "budget_max": normalized.get("budget_max"),
        }

    if question_type == "location":
        location_text = parsed.get("location_text", "")
        if not isinstance(location_text, str):
            location_text = ""
        return {"location_text": location_text.strip()}

    if question_type == "cuisine":
        cuisine = parsed.get("cuisine", [])
        if not isinstance(cuisine, list):
            cuisine = []
        cuisine = [str(x).strip().lower() for x in cuisine if str(x).strip()]
        return {"cuisine": list(dict.fromkeys(cuisine))}

    return {}