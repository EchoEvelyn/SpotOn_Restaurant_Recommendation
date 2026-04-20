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


def build_clarification_question_prompt(question_type: str, prefs: dict) -> str:
    if question_type == "location":
        instruction = """
The missing information is the user's search location.
Ask where they are searching.
You may include a few short examples such as La Jolla, San Diego, Irvine, or Seattle downtown.
"""

    elif question_type == "cuisine":
        instruction = """
The missing information is the cuisine.
Ask what cuisine they are in the mood for.
If the existing preferences are broad or generic, ask for a more specific cuisine.
You may include examples such as Japanese, Korean, Chinese, Thai, Mexican, or Italian.
"""

    else:
        instruction = f"""
The missing information is: {question_type}
Ask one concise follow-up question about it.
"""

    return f"""
You are a friendly restaurant recommendation assistant.

Your job is to ask ONE natural follow-up question to collect missing information before searching.

Current collected preferences:
{json.dumps(prefs, ensure_ascii=False)}

Instructions:
{instruction}

Rules:
- Ask only one question.
- Be concise and natural.
- Do not mention JSON, schema, or internal state.
- Do not ask multiple questions at once.
- Return only the question text.

Question:
""".strip()

def generate_clarification_question_llm(
    question_type: str,
    prefs: dict,
    model: str | None = None
) -> str:

    if model is None:
        model = OLLAMA_MODEL

    prompt = build_clarification_question_prompt(question_type, prefs)

    raw_output = call_ollama(prompt, model=model)

    question = raw_output.strip().strip('"').strip("'").strip()

    if not question:
        return fallback_clarification_question(question_type, prefs)

    return question

def fallback_clarification_question(question_type: str, prefs: dict) -> str:
    if question_type == "location":
        return "Where should I look? A city or neighborhood is enough."

    if question_type == "cuisine":
        return "Any cuisine in mind?"

    if question_type == "budget":
        return "Want me to keep it cheaper, moderate, or a bit nicer?"

    return "Tell me a little more and I can refine this."

def build_review_summary_prompt(name: str, reviews: list[dict]) -> str:
    review_lines = []

    for r in reviews:
        text = (r.get("text") or "").strip()
        rating = r.get("rating")
        if not text:
            continue

        prefix = f"[{rating} stars] " if rating is not None else ""
        review_lines.append(f"- {prefix}{text}")

    joined_reviews = "\n".join(review_lines[:5])

    return f"""
You are summarizing customer reviews for a restaurant recommendation card.

Restaurant: {name}

Customer reviews:
{joined_reviews}

Task:
Write a short, natural summary for a restaurant card.

Requirements:
- 1 to 2 sentences only
- Mention the most common positives
- Mention one important downside only if it appears in the reviews
- Do not invent facts
- Do not use hype, marketing language, or exaggeration
- Keep it concise and useful for someone deciding where to eat
- Output only the summary text
""".strip()


def build_review_quote_selection_prompt(name: str, reviews: list[dict]) -> str:
    review_lines = []

    for idx, r in enumerate(reviews):
        text = (r.get("text") or "").strip()
        if not text:
            continue

        rating = r.get("rating")
        relative_time = r.get("relative_time", "")
        author_name = r.get("author_name", "Anonymous")

        review_lines.append(
            f"""[{idx}]
author: {author_name}
rating: {rating}
time: {relative_time}
text: {text}
"""
        )

    joined_reviews = "\n".join(review_lines[:5])

    return f"""
You are selecting ONE customer review quote for a restaurant recommendation card.

Restaurant: {name}

Reviews:
{joined_reviews}

Task:
Choose the single best review to display as a short quote on the card.

Selection criteria:
- should be representative of the restaurant
- should be clear and readable
- should mention food, service, atmosphere, or wait time if possible
- should not be overly specific to a private situation unless still broadly useful
- should not be too short or too long

Return JSON only in this format:
{{"selected_index": 0}}

Rules:
- Choose only from the provided reviews
- Do not rewrite the review
- Do not explain your choice
- Output JSON only
""".strip()