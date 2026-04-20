import math
import os
import html
from copy import deepcopy

import requests
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

from agent.extractor import (
    extract_followup_update_llm,
    extract_preferences_llm,
)
from schemas.schemas import DEFAULT_PREFERENCES
from services.ranking import rank_restaurants
from tools.google_places import GooglePlacesTool


# =========================
# Session state and chat
# =========================

def initialize_state() -> None:
    defaults = {
        "search_text": "",
        "parsed_prefs": deepcopy(DEFAULT_PREFERENCES),
        "results": [],
        "search_query": "",
        "status_message": "",
        "has_searched": False,
        "user_lat": None,
        "user_lng": None,
        "location_error": "",
        "awaiting_followup": False,
        "pending_question_type": None,
        "assistant_prompt": "",
        "last_user_request": "",
        "decision_trace": {},
        "last_decision": None,
        "chat_history": [],
        "asked_about": set(),
        "input_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def push_chat(role: str, text: str) -> None:
    text = (text or "").strip()
    # Filter out empty or junk messages (e.g. stray punctuation)
    if not text or (len(text) <= 3 and not text[0].isalnum()):
        return

    history = st.session_state.chat_history
    if history and history[-1]["role"] == role and history[-1]["text"] == text:
        return

    history.append({"role": role, "text": text})


def sync_assistant_message() -> None:
    message = (st.session_state.assistant_prompt or st.session_state.status_message or "").strip()
    if message:
        push_chat("assistant", message)


def reset_followup_state() -> None:
    st.session_state.awaiting_followup = False
    st.session_state.pending_question_type = None
    st.session_state.assistant_prompt = ""


def reset_all_state() -> None:
    st.session_state.results = []
    st.session_state.search_query = ""
    st.session_state.parsed_prefs = deepcopy(DEFAULT_PREFERENCES)
    st.session_state.has_searched = False
    st.session_state.chat_history = []
    st.session_state.asked_about = set()
    st.session_state.input_counter += 1
    reset_followup_state()


# =========================
# Location and query helpers
# =========================

def try_get_browser_location() -> None:
    if st.session_state.user_lat is not None and st.session_state.user_lng is not None:
        return

    loc = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
                (pos) => resolve({
                    lat: pos.coords.latitude,
                    lng: pos.coords.longitude
                }),
                (err) => resolve({
                    error: err.message
                }),
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 60000
                }
            );
        })
        """,
        key="geo_location_once",
        want_output=True,
    )

    if isinstance(loc, dict):
        if "lat" in loc and "lng" in loc:
            st.session_state.user_lat = loc["lat"]
            st.session_state.user_lng = loc["lng"]
            st.session_state.location_error = ""
        elif "error" in loc:
            st.session_state.location_error = loc["error"]


def has_geo_context() -> bool:
    return (
        st.session_state.user_lat is not None
        and st.session_state.user_lng is not None
    )


def has_text_location(prefs: dict) -> bool:
    return bool((prefs.get("location_text") or "").strip())


def build_search_query(prefs: dict) -> str:
    terms: list[str] = []

    budget_max = prefs.get("budget_max")
    budget_level = prefs.get("budget_level")
    location_text = (prefs.get("location_text") or "").strip()

    if budget_max is not None:
        terms.append(f"under ${budget_max}")
    elif budget_level == "cheap":
        terms.append("cheap")
    elif budget_level == "moderate":
        terms.append("moderately priced")
    elif budget_level == "expensive":
        terms.append("upscale")

    preferences = prefs.get("preferences") or []
    if isinstance(preferences, str):
        preferences = [preferences]
    terms.extend(preferences)

    cuisines = prefs.get("cuisine") or []
    if isinstance(cuisines, str):
        cuisines = [cuisines]
    terms.extend(cuisines)

    terms.append("restaurant")

    if location_text:
        if prefs.get("distance_preference") == "near":
            terms.append(f"near {location_text}")
        else:
            terms.append(location_text)

    return " ".join(terms).strip()


def make_followup_message(question_type: str, prefs: dict) -> str:
    if question_type == "location":
        if has_geo_context():
            return "Want me to narrow it to a neighborhood?"
        return "What area should I look in?"

    if question_type == "cuisine":
        return "Any cuisine in mind?"

    if question_type == "budget":
        return "Want to keep it cheaper or a bit nicer?"

    return "Tell me a bit more and I’ll tighten this up."


def detect_missing_requirements(prefs: dict) -> list[str]:
    missing = []

    if not has_text_location(prefs) and not has_geo_context():
        missing.append("location")

    return missing


def get_quick_replies(question_type: str | None) -> list[str]:
    if question_type == "budget":
        return [
            "budget-friendly",
            "mid-range",
            "upscale",
            "fine dining",
        ]

    if question_type == "location":
        return []  # no hardcoded options, user types freely

    if question_type == "cuisine":
        return []  # no hardcoded options, user types freely

    return []


def detect_query_type(text: str) -> str:
    text = text.lower().strip()

    location_markers = [" near ", " in ", " around ", " at "]
    if any(marker in f" {text} " for marker in location_markers):
        return "NEW_QUERY"

    budget_words = {
        "cheap", "cheaper", "moderate", "upscale",
        "expensive", "budget", "affordable"
    }
    location_words = {
        "convoy", "utc", "la jolla", "downtown"
    }
    cuisine_words = {
        "sushi", "ramen", "korean", "chinese", "italian",
        "thai", "japanese", "bbq", "brunch", "dimsum", "dim sum"
    }

    if text in budget_words or text in location_words or text in cuisine_words:
        return "REFINE"

    if len(text.split()) >= 2:
        return "NEW_QUERY"

    return "REFINE"


def build_prefs_from_tag(tag: str) -> dict:
    """直接从 tag 关键词构建偏好，跳过 LLM 解析。"""
    prefs = deepcopy(DEFAULT_PREFERENCES)
    prefs["cuisine"] = [tag.lower()]
    return prefs


def merge_preferences(old: dict, new: dict) -> dict:
    merged = deepcopy(old)

    new_cuisine = new.get("cuisine")
    if new_cuisine:
        if isinstance(new_cuisine, str):
            new_cuisine = [new_cuisine]
        merged["cuisine"] = new_cuisine

    new_preferences = new.get("preferences")
    if new_preferences:
        if isinstance(new_preferences, str):
            new_preferences = [new_preferences]
        merged["preferences"] = new_preferences

    new_location = (new.get("location_text") or "").strip()
    if new_location:
        merged["location_text"] = new_location

    if new.get("distance_preference"):
        merged["distance_preference"] = new["distance_preference"]

    if new.get("budget_level"):
        merged["budget_level"] = new["budget_level"]
        merged["budget_max"] = None

    if new.get("budget_max") is not None:
        merged["budget_max"] = new["budget_max"]
        merged["budget_level"] = None

    new_avoid = new.get("avoid")
    if new_avoid:
        if isinstance(new_avoid, str):
            new_avoid = [new_avoid]
        merged["avoid"] = new_avoid

    return merged


# =========================
# Google Places fetch and enrichment
# =========================

def fetch_place_reviews(
    tool: GooglePlacesTool,
    place_id: str,
    reviews_per_place: int = 5,
) -> list[dict]:
    headers = tool._build_headers("reviews")
    url = f"{tool.details_base_url}/{place_id}"
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    data = response.json()

    reviews = []
    for review in data.get("reviews", [])[:reviews_per_place]:
        text_field = review.get("text")
        if isinstance(text_field, dict):
            text = text_field.get("text", "")
        else:
            text = text_field or ""

        reviews.append(
            {
                "text": text.strip(),
                "author_name": (review.get("authorAttribution") or {}).get("displayName", "Anonymous"),
                "rating": review.get("rating"),
                "relative_time": review.get("relativePublishTimeDescription", ""),
            }
        )
    return reviews


def call_places_text_search(
    api_key: str,
    query: str,
    max_results: int = 12,
    include_reviews: bool = True,
    reviews_per_place: int = 5,
    lat: float | None = None,
    lng: float | None = None,
    radius_meters: int = 10000,
) -> list[dict]:
    tool = GooglePlacesTool(api_key)

    if hasattr(tool, "search_restaurants") and lat is None:
        try:
            return tool.search_restaurants(
                query=query,
                max_results=max_results,
                include_reviews=include_reviews,
                reviews_per_place=reviews_per_place,
            )
        except TypeError:
            return tool.search_restaurants(query=query, max_results=max_results)

    field_mask = [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.types",
        "places.priceLevel",
        "places.googleMapsUri",
        "places.photos",
    ]
    headers = tool._build_headers(",".join(field_mask))
    payload = {"textQuery": query, "pageSize": max_results}
    if lat is not None and lng is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius_meters),
            }
        }

    response = requests.post(tool.search_url, headers=headers, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()

    restaurants: list[dict] = []
    for place in data.get("places", []):
        photo_url = None
        photos = place.get("photos", [])
        if photos:
            photo_name = photos[0].get("name")
            if photo_name:
                photo_url = tool.build_photo_url(photo_name)

        loc = place.get("location") or {}
        restaurants.append(
            {
                "place_id": place.get("id"),
                "name": (place.get("displayName") or {}).get("text"),
                "address": place.get("formattedAddress"),
                "lat": loc.get("latitude"),
                "lng": loc.get("longitude"),
                "rating": place.get("rating"),
                "user_rating_count": place.get("userRatingCount"),
                "types": place.get("types", []),
                "price_level": place.get("priceLevel"),
                "google_maps_uri": place.get("googleMapsUri"),
                "photo_url": photo_url,
                "reviews": [],
            }
        )

    if include_reviews:
        for restaurant in restaurants:
            place_id = restaurant.get("place_id")
            if not place_id:
                continue
            restaurant["reviews"] = fetch_place_reviews(
                tool=tool,
                place_id=place_id,
                reviews_per_place=reviews_per_place,
            )

    return restaurants


def pick_best_review_quote(reviews: list[dict]) -> dict | None:
    candidates = []
    for review in reviews:
        text = (review.get("text") or "").strip()
        if not text or len(text) < 25:
            continue

        score = 0
        rating = review.get("rating") or 0
        score += rating * 10

        text_len = len(text)
        if 40 <= text_len <= 180:
            score += 15
        elif 20 <= text_len <= 240:
            score += 8

        candidates.append((score, review))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    return {
        "text": (best.get("text") or "").strip(),
        "author_name": best.get("author_name", "Anonymous"),
        "rating": best.get("rating"),
        "relative_time": best.get("relative_time", ""),
    }


def enrich_restaurants(restaurants: list[dict], top_k_for_llm: int = 5) -> list[dict]:
    for idx, restaurant in enumerate(restaurants):
        restaurant["review_summary"] = ""
        restaurant["selected_review_quote"] = None

        if idx >= top_k_for_llm:
            continue

        reviews = restaurant.get("reviews", [])
        if not reviews:
            continue

        restaurant["review_summary"] = ""
        restaurant["selected_review_quote"] = reviews[0] if reviews else None

    return restaurants


def extract_price_bucket(price_level: str | None) -> str | None:
    mapping = {
        "PRICE_LEVEL_INEXPENSIVE": "cheap",
        "PRICE_LEVEL_MODERATE": "moderate",
        "PRICE_LEVEL_EXPENSIVE": "expensive",
        "PRICE_LEVEL_VERY_EXPENSIVE": "very_expensive",
    }
    return mapping.get(price_level)


def result_price_buckets(results: list[dict]) -> list[str]:
    return [
        bucket for result in results
        if (bucket := extract_price_bucket(result.get("price_level")))
    ]


def unique_locations_from_results(results: list[dict]) -> set[str]:
    locations = set()
    for result in results:
        addr = (result.get("address") or "").lower()
        if not addr:
            continue

        for token in ["convoy", "la jolla", "downtown", "irvine", "seattle", "utc", "kearny mesa"]:
            if token in addr:
                locations.add(token)
    return locations


def analyze_result_set(results: list[dict], prefs: dict) -> dict:
    if not results:
        return {
            "result_count": 0,
            "score_spread": 0.0,
            "top_score": 0.0,
            "price_span_large": False,
            "location_span_large": False,
            "has_budget": prefs.get("budget_level") is not None or prefs.get("budget_max") is not None,
            "has_location": has_text_location(prefs),
            "has_cuisine": bool(prefs.get("cuisine")),
        }

    scores = [result.get("_score", 0.0) for result in results]
    top_score = max(scores) if scores else 0.0
    score_spread = (max(scores) - min(scores)) if len(scores) >= 2 else 0.0

    price_buckets = set(result_price_buckets(results))
    location_markers = unique_locations_from_results(results)

    return {
        "result_count": len(results),
        "score_spread": score_spread,
        "top_score": top_score,
        "price_span_large": len(price_buckets) >= 2,
        "location_span_large": len(location_markers) >= 2,
        "has_budget": prefs.get("budget_level") is not None or prefs.get("budget_max") is not None,
        "has_location": has_text_location(prefs),
        "has_cuisine": bool(prefs.get("cuisine")),
    }


def next_best_action(prefs: dict, results: list[dict]) -> dict:
    analysis = analyze_result_set(results, prefs)

    if analysis["result_count"] == 0:
        return {
            "action": "recover",
            "question_type": None,
            "message": "I couldn’t find strong matches. Want to broaden the area or try another cuisine?",
            "reason": "No usable results were returned.",
        }

    # Location always comes first — if no text location, always soft-hint it before anything else
    if (not analysis["has_location"]) and has_geo_context():
        if analysis["location_span_large"]:
            return {
                "action": "show_with_optional_hint",
                "question_type": "location",
                "message": "These are near you but spread across different areas. Add a neighborhood for tighter picks.",
                "reason": "Only browser location is available and the results appear geographically spread out.",
            }
        return {
            "action": "show_with_optional_hint",
            "question_type": "location",
            "message": "These are near you right now. Add a neighborhood if you want tighter picks.",
            "reason": "Results are usable with browser location, but a neighborhood could make them more precise.",
        }

    if (not analysis["has_cuisine"]) and analysis["score_spread"] < 0.12 and analysis["result_count"] >= 5:
        return {
            "action": "refine",
            "question_type": "cuisine",
            "message": "I can make this more specific if you want a cuisine in mind.",
            "reason": "Cuisine is missing and the current candidates are broad and similarly scored.",
        }

    if (not analysis["has_budget"]) and analysis["price_span_large"]:
        return {
            "action": "show_with_optional_hint",
            "question_type": "budget",
            "message": "These vary quite a bit in price. Want me to narrow this down by budget?",
            "reason": "Budget is missing and the returned results span multiple price buckets.",
        }

    if (not analysis["has_budget"]) and analysis["result_count"] >= 5:
        return {
            "action": "show_with_optional_hint",
            "question_type": "budget",
            "message": "I can also narrow this down by budget if you want.",
            "reason": "Results are already usable, but budget could still help personalize reranking.",
        }

    if (not analysis["has_cuisine"]) and analysis["result_count"] >= 5:
        return {
            "action": "show_with_optional_hint",
            "question_type": "cuisine",
            "message": "I can make this more specific if you want a cuisine in mind.",
            "reason": "Results are usable, but cuisine could make the list more targeted.",
        }

    return {
        "action": "show_only",
        "question_type": None,
        "message": "",
        "reason": "Current results already look specific enough, so no follow-up is needed.",
    }


def search_with_geo_fallback(prefs: dict) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "error": 'Missing GOOGLE_API_KEY. Please run: export GOOGLE_API_KEY="your_key_here"'
        }

    lat = st.session_state.user_lat
    lng = st.session_state.user_lng

    if lat is None or lng is None:
        return {"error": "Location is unavailable right now."}

    cuisines = prefs.get("cuisine") or []
    if isinstance(cuisines, str):
        cuisines = [cuisines]
    preferences = prefs.get("preferences") or []
    if isinstance(preferences, str):
        preferences = [preferences]

    has_specific_query = bool(cuisines or preferences)

    if has_specific_query:
        # Use text search with locationBias so cuisine query is honoured and results stay nearby
        query = build_search_query(prefs)
        try:
            restaurants = call_places_text_search(
                api_key=api_key,
                query=query,
                max_results=18,
                include_reviews=False,
                lat=lat,
                lng=lng,
                radius_meters=10000,
            )
        except Exception as exc:
            return {"error": f"Could not search restaurants: {exc}"}
        cuisine_label = ", ".join(cuisines) if cuisines else ""
        search_query = f"{cuisine_label} near you" if cuisine_label else "restaurants near you"
    else:
        # No specific query — fall back to generic nearby search
        try:
            tool = GooglePlacesTool(api_key)
            restaurants = tool.nearby_restaurants(
                lat=lat,
                lng=lng,
                max_results=18,
                radius_meters=4000,
                include_reviews=False,
            )
        except Exception as exc:
            return {"error": f"Could not search nearby restaurants: {exc}"}
        search_query = "restaurants near you"

    if not restaurants:
        return {"error": "No nearby restaurants found."}

    ranked = rank_restaurants(restaurants, prefs)
    final_results = enrich_restaurants(ranked[:5], top_k_for_llm=5)

    return {"results": final_results, "search_query": search_query}


def search_with_prefs(prefs: dict) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": 'Missing GOOGLE_API_KEY. Please run: export GOOGLE_API_KEY="your_key_here"'}

    query = build_search_query(prefs)
    if not query:
        return {"error": "Please enter a search query."}

    # If GPS is available, use it as locationBias so ambiguous names like
    # "North Park" resolve to the neighbourhood near the user, not across the country
    bias_lat = st.session_state.get("user_lat")
    bias_lng = st.session_state.get("user_lng")
    restaurants = call_places_text_search(
        api_key=api_key,
        query=query,
        max_results=12,
        include_reviews=False,
        lat=bias_lat,
        lng=bias_lng,
        radius_meters=50000,  # generous bias radius when user typed a location
    )

    if not restaurants:
        return {"error": "No restaurants found. Try another cuisine, area, or budget."}

    ranked = rank_restaurants(restaurants, prefs)
    final_results = enrich_restaurants(ranked[:5], top_k_for_llm=5)

    return {"results": final_results, "search_query": query}


def apply_search_output(output: dict, prefs: dict) -> None:
    if "error" in output:
        st.session_state.results = []
        st.session_state.search_query = ""
        st.session_state.status_message = output["error"]
        st.session_state.assistant_prompt = output["error"]
        push_chat("assistant", output["error"])
        return

    st.session_state.results = output["results"]
    st.session_state.search_query = output["search_query"]

    analysis = analyze_result_set(output["results"], prefs)
    decision = next_best_action(prefs, output["results"])

    st.session_state.decision_trace = {
        "analysis": analysis,
        "decision": decision,
    }
    st.session_state.last_decision = decision.get("action")

    if decision["action"] == "refine":
        st.session_state.awaiting_followup = True
        st.session_state.pending_question_type = decision["question_type"]
        st.session_state.assistant_prompt = decision["message"]
        st.session_state.status_message = decision["message"]
        sync_assistant_message()
    elif decision["action"] in ("recover", "show_with_optional_hint"):
        st.session_state.awaiting_followup = False
        st.session_state.pending_question_type = decision.get("question_type")
        st.session_state.assistant_prompt = decision["message"]
        st.session_state.status_message = decision["message"]
        sync_assistant_message()
    else:
        reset_followup_state()
        st.session_state.assistant_prompt = ""
        st.session_state.status_message = ""


def run_search(search_text: str, skip_llm: bool = False) -> None:
    search_text = search_text.strip()
    if not search_text:
        return

    st.session_state.last_user_request = search_text
    push_chat("user", search_text)
    st.session_state.has_searched = True
    st.session_state.asked_about = set()

    query_type = detect_query_type(search_text)

    try:
        if skip_llm:
            parsed = build_prefs_from_tag(search_text)
        elif st.session_state.pending_question_type:
            # 有待回答的追问时，永远走 merge，不开新查询
            update = extract_followup_update_llm(
                search_text,
                st.session_state.pending_question_type,
            )
            parsed = merge_preferences(
                st.session_state.parsed_prefs,
                update,
            )
        elif query_type == "NEW_QUERY":
            parsed = extract_preferences_llm(search_text)
        else:
            update = extract_preferences_llm(search_text)
            parsed = merge_preferences(
                st.session_state.parsed_prefs,
                update,
            )

    except Exception as exc:
        st.session_state.results = []
        st.session_state.search_query = ""
        st.session_state.status_message = f"Could not understand that request: {exc}"
        st.session_state.assistant_prompt = st.session_state.status_message
        push_chat("assistant", st.session_state.status_message)
        return

    st.session_state.search_text = search_text
    st.session_state.parsed_prefs = parsed

    missing = detect_missing_requirements(parsed)

    if missing:
        question_type = missing[0]
        st.session_state.results = []
        st.session_state.search_query = ""
        st.session_state.awaiting_followup = True
        st.session_state.pending_question_type = question_type
        st.session_state.assistant_prompt = make_followup_message(question_type, parsed)
        st.session_state.status_message = st.session_state.assistant_prompt
        sync_assistant_message()
        return

    if has_text_location(parsed):
        output = search_with_prefs(parsed)
    else:
        output = search_with_geo_fallback(parsed)

    apply_search_output(output, parsed)


def rerun_with_current_filters() -> None:
    prefs = deepcopy(st.session_state.parsed_prefs)

    if has_text_location(prefs):
        output = search_with_prefs(prefs)
    elif has_geo_context():
        output = search_with_geo_fallback(prefs)
    else:
        st.session_state.results = []
        st.session_state.search_query = ""
        st.session_state.awaiting_followup = True
        st.session_state.pending_question_type = "location"
        st.session_state.assistant_prompt = "Where should I look? A city or neighborhood is enough."
        st.session_state.status_message = st.session_state.assistant_prompt
        sync_assistant_message()
        return

    apply_search_output(output, prefs)


def apply_refinement(user_reply: str) -> None:
    user_reply = user_reply.strip()
    if not user_reply:
        return

    push_chat("user", user_reply)
    question_type = st.session_state.get("pending_question_type")

    try:
        if question_type:
            update = extract_followup_update_llm(user_reply, question_type)
        else:
            update = extract_preferences_llm(user_reply)

        merged = merge_preferences(st.session_state.parsed_prefs, update)
        st.session_state.parsed_prefs = merged
    except Exception as exc:
        st.session_state.status_message = f"Could not apply that update: {exc}"
        st.session_state.assistant_prompt = st.session_state.status_message
        push_chat("assistant", st.session_state.status_message)
        return

    rerun_with_current_filters()
    sync_assistant_message()


# =========================
# Display helpers
# =========================

def build_reason(restaurant: dict, prefs: dict, rank_idx: int) -> str:
    reasons = []

    rating = restaurant.get("rating")
    reviews = restaurant.get("user_rating_count")
    price = restaurant.get("price_level")
    types = restaurant.get("types", [])

    if rating is not None:
        if rating >= 4.7:
            reasons.append("it has an excellent rating")
        elif rating >= 4.5:
            reasons.append("it is highly rated")
        elif rating >= 4.2:
            reasons.append("it has a solid rating")

    if reviews:
        if reviews >= 2000:
            reasons.append("it is backed by a large number of reviews")
        elif reviews >= 500:
            reasons.append("it has a strong review base")

    if prefs.get("budget_max") is not None:
        reasons.append(f"it fits your budget target of under ${prefs['budget_max']}")
    elif prefs.get("budget_level") == "cheap":
        if price == "PRICE_LEVEL_INEXPENSIVE":
            reasons.append("it fits a lower budget")
        elif price == "PRICE_LEVEL_MODERATE":
            reasons.append("it still looks reasonably priced")
    elif prefs.get("budget_level") == "moderate":
        if price == "PRICE_LEVEL_MODERATE":
            reasons.append("it fits a moderate price range")
    elif prefs.get("budget_level") == "expensive":
        if price == "PRICE_LEVEL_EXPENSIVE":
            reasons.append("it matches a more upscale option")
    elif prefs.get("budget_level") == "very_expensive":
        if price == "PRICE_LEVEL_VERY_EXPENSIVE":
            reasons.append("it matches a fine dining experience")

    requested = prefs.get("cuisine") or []
    if isinstance(requested, str):
        requested = [requested]

    if requested:
        joined_types = " ".join(types).lower()
        if any(c.lower() in joined_types for c in requested):
            reasons.append("it aligns well with your cuisine preference")
        else:
            reasons.append(f"it is worth considering for your {' / '.join(requested)} search")

    preferences = prefs.get("preferences") or []
    if isinstance(preferences, str):
        preferences = [preferences]

    if preferences:
        reasons.append(f"it may match your preference for {', '.join(preferences)}")

    if not reasons:
        fallbacks = [
            "A reasonable nearby option based on your request.",
            "This looks like a decent match for what you asked for.",
            "This seems worth considering for your current search.",
        ]
        return fallbacks[(rank_idx - 1) % len(fallbacks)]

    openers = [
        "Why it stands out",
        "Why this could work",
        "Why you might like it",
        "Why it made the list",
    ]
    opener = openers[(rank_idx - 1) % len(openers)]

    if len(reasons) == 1:
        body = reasons[0]
    elif len(reasons) == 2:
        body = f"{reasons[0]} and {reasons[1]}"
    else:
        body = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    return f"{opener}: {body}."


def render_assistant_drawer() -> None:
    question_type = st.session_state.get("pending_question_type")
    quick_replies = get_quick_replies(question_type)
    is_refine = st.session_state.get("awaiting_followup", False)

    # hint config for free-text question types
    free_text_hints = {
        "location": {
            "label": "Narrow by area",
            "hint": "Type any neighborhood, city, or street — or just keep searching as-is.",
            "placeholder": "e.g. North Park, downtown, near UCSD...",
        },
        "cuisine": {
            "label": "Narrow by cuisine",
            "hint": "Type any cuisine or craving — or just keep searching as-is.",
            "placeholder": "e.g. sushi, korean bbq, pasta, dim sum...",
        },
    }

    with st.sidebar:
        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

        history = st.session_state.get("chat_history", [])
        if history:
            st.markdown('<div class="sidebar-section-label">Conversation</div>', unsafe_allow_html=True)
            bubbles_html = "".join(
                f'<div class="assistant-bubble">{item["text"]}</div>'
                if item["role"] == "assistant"
                else f'<div class="user-bubble">{item["text"]}</div>'
                for item in history
            )
            st.markdown(
                f'''<div id="chat-scroll" style="max-height:340px;overflow-y:auto;display:flex;flex-direction:column;gap:0;">
{bubbles_html}
</div>
<script>
(function(){{
    var el = document.getElementById("chat-scroll");
    if(el) el.scrollTop = el.scrollHeight;
}})();
</script>''',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # free-text hint for location / cuisine
        if question_type in free_text_hints and not quick_replies:
            hint_config = free_text_hints[question_type]
            st.markdown(f'<div class="sidebar-section-label">{hint_config["label"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sidebar-location-hint">{hint_config["hint"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        elif quick_replies:
            if is_refine:
                st.markdown('<div class="sidebar-section-label sidebar-section-label--required">↩ Please reply to continue</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sidebar-section-label">Quick replies</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, option in enumerate(quick_replies):
                if cols[i % 2].button(
                    option,
                    key=f"quick_{option}",
                    use_container_width=True,
                ):
                    with st.spinner("Updating results..."):
                        apply_refinement(option)
                    st.session_state.input_counter += 1
                    st.rerun()
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Search</div>', unsafe_allow_html=True)

        # use context-aware placeholder when hinting
        if question_type in free_text_hints:
            input_placeholder = free_text_hints[question_type]["placeholder"]
        else:
            input_placeholder = 'Try "spicy ramen near downtown" or "cozy brunch spot"'

        user_text = st.text_input(
            "Chat search",
            placeholder=input_placeholder,
            label_visibility="collapsed",
            key=f"sidebar_chat_input_{st.session_state.get('input_counter', 0)}",
        )

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="send-btn-wrap">', unsafe_allow_html=True)
        clicked = st.button("Send", use_container_width=True, key="sidebar_send_button")
        st.markdown('</div>', unsafe_allow_html=True)
        if clicked and (text := user_text.strip()):
            with st.spinner("Finding good spots..."):
                if st.session_state.get("pending_question_type"):
                    apply_refinement(text)
                elif detect_query_type(text) == "NEW_QUERY":
                    run_search(text)
                else:
                    apply_refinement(text)
            st.session_state.input_counter += 1
            st.rerun()


def render_restaurant_card(restaurant: dict, prefs: dict, idx: int) -> None:
    name = html.escape(str(restaurant.get("name", "Unknown")))
    rating = restaurant.get("rating")
    review_count = restaurant.get("user_rating_count")
    address_raw = restaurant.get("address", "") or ""
    address = html.escape(str(address_raw))
    maps_uri = restaurant.get("google_maps_uri") or ""
    photo_url = restaurant.get("photo_url")
    quote = restaurant.get("selected_review_quote")
    summary = (restaurant.get("review_summary") or "").strip()

    short_address = address
    if len(short_address) > 100:
        short_address = short_address[:97].rstrip() + "..."

    quote_text = ""
    quote_meta = ""

    if quote and quote.get("text"):
        quote_text = html.escape((quote.get("text") or "").strip())
        author_name = html.escape(str(quote.get("author_name", "Anonymous")))
        quote_meta = author_name

        if quote.get("rating") is not None:
            quote_meta += f" · ⭐ {quote['rating']}"
        if quote.get("relative_time"):
            quote_meta += f" · {html.escape(str(quote['relative_time']))}"

    content_text = quote_text if quote_text else html.escape(summary)

    preview_limit = 240
    if content_text:
        display_text = content_text[:preview_limit].rstrip()
        if len(content_text) > preview_limit:
            display_text += "..."
    else:
        display_text = html.escape(build_reason(restaurant, prefs, idx))

    rating_html = ""
    if rating is not None:
        meta_text = f"⭐ {rating}"
        if review_count:
            meta_text += f" ({review_count} reviews)"
        rating_html = f'<div class="restaurant-card-rating">{meta_text}</div>'

    maps_html = ""
    if maps_uri:
        maps_html = (
            f'<div class="maps-link">'
            f'<a href="{maps_uri}" target="_blank">📍 Open in Google Maps</a>'
            f'</div>'
        )

    if photo_url:
        image_html = f'<img src="{photo_url}" class="restaurant-card-image" alt="{name}">'
    else:
        image_html = (
            '<div class="restaurant-card-image restaurant-card-image-fallback">'
            'No image'
            '</div>'
        )

    card_html = f"""
    <div class="restaurant-card">
        <div class="restaurant-card-left">
            {image_html}
        </div>
        <div class="restaurant-card-right">
            <div class="restaurant-card-title">{idx}. {name}</div>
            {rating_html}
            <div class="restaurant-card-address">{short_address}</div>
            {maps_html}
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 78vh;
            text-align: center;
            gap: 0;
        ">
            <div style="
                display: flex;
                gap: 1.8rem;
                font-size: 2rem;
                opacity: 0.45;
                margin-bottom: 2.5rem;
                letter-spacing: 4px;
            ">
                🍜 🌮 🍣 ☕ 🍕 🥞 🍔 🍛
            </div>
            <div style="display:flex; align-items:center; justify-content:center; gap:16px; margin-bottom: 0.3rem;">
                <div style="font-size:7rem;">✨</div>
                <div style="
                    font-family: 'Parisienne', cursive;
                    font-size: 9rem;
                    color: #7a322e;
                    line-height: 1;
                ">SpotOn</div>
            </div>
            <div style="font-size: 1rem; color: #a08880; margin-bottom: 3rem;">
                Good food, without the endless searching.
            </div>
            <div style="
                font-size: 2.4rem;
                font-weight: 800;
                color: #4b2b2a;
                letter-spacing: -0.6px;
                line-height: 1.15;
                margin-bottom: 1rem;
            ">
                What are you craving?
            </div>
            <div style="
                font-size: 1.05rem;
                color: #a08880;
                max-width: 360px;
                line-height: 1.7;
                margin-bottom: 2.2rem;
            ">
                Describe a cuisine, vibe, or neighborhood — we'll find the right spot.
            </div>
            <div style="
                font-size: 0.88rem;
                color: #c8aca5;
                display: flex;
                align-items: center;
                gap: 8px;
                border: 1px solid rgba(180,140,130,0.2);
                border-radius: 20px;
                padding: 0.5rem 1.1rem;
                background: rgba(255,255,255,0.5);
            ">
                <span style="font-size:1rem;">←</span>
                <span>Start typing in the sidebar</span>
            </div>
            <div style="
                display: flex;
                gap: 1.8rem;
                font-size: 2rem;
                opacity: 0.45;
                margin-top: 2.5rem;
                letter-spacing: 4px;
            ">
                🥗 🍱 🥩 🍷 🧆 🥐 🍦 🥟
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Main app
# =========================

def main() -> None:
    st.set_page_config(page_title="✨SpotOn", layout="wide")
    initialize_state()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Parisienne&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: #f8f2ee;
            color: #4b2e2b;
        }

        section[data-testid="stSidebar"] {
            background: #f3ebe6;
            border-right: 1px solid rgba(122, 80, 72, 0.08);
            width: 320px !important;
            min-width: 320px !important;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 2.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
        }

        div[data-testid="stToolbar"] {
            right: 1rem !important;
            top: 0.25rem !important;
        }

        .block-container {
            padding-top: 0.15rem !important;
            padding-left: 2.1rem !important;
            padding-right: 2.1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1320px;
        }

        .page-shell {
            max-width: 980px;
            margin: 0 auto;
            padding-top: 0;
        }

        .brand-title {
            font-family: 'Parisienne', cursive;
            font-size: 6rem;
            font-weight: 550;
            color: #7a322e;
            line-height: 1.5;
            letter-spacing: 0;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 500;
            color: #7a4a44;
            margin-bottom: 0.35rem;
            letter-spacing: 0.2px;
            line-height: 1.08;
        }

        .hero-title {
            text-align: center;
            font-size: 1.6rem;
            line-height: 1.2;
            font-weight: 600;
            color: #6a4a46;
            margin-top: 0;
            margin-bottom: 0.6rem;
        }

        .hero-subtitle {
            text-align: center;
            font-size: 0.88rem;
            color: #a08880;
            margin-bottom: 1.1rem;
            font-weight: 400;
        }

        .small-muted {
            font-size: 0.88rem;
            line-height: 1.6;
            color: #b09088;
            margin-bottom: 0.7rem;
            font-weight: 400;
            letter-spacing: 0.1px;
        }

        .assistant-bubble {
            background: rgba(255,255,255,0.52);
            border: 1px solid rgba(122, 80, 72, 0.13);
            border-radius: 14px 14px 14px 4px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.5rem;
            font-size: 0.92rem;
            color: #6d4a44;
            line-height: 1.55;
            box-shadow: 0 1px 4px rgba(80, 45, 40, 0.05);
        }

        .user-bubble {
            background: linear-gradient(90deg, #ece6f7 0%, #f7e8ee 100%);
            border: 1px solid rgba(154, 120, 145, 0.18);
            border-radius: 14px 14px 4px 14px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.5rem;
            font-size: 0.92rem;
            color: #6a2d2a;
            line-height: 1.55;
        }

        .sidebar-copy {
            font-size: 0.88rem;
            line-height: 1.6;
            color: #b09088;
            margin-bottom: 1.1rem;
            font-weight: 400;
            letter-spacing: 0.1px;
        }

        .sidebar-spacer {
            height: 4px;
        }

        .sidebar-divider {
            border: none;
            border-top: 1px solid rgba(122, 80, 72, 0.12);
            margin: 1rem 0;
        }

        .sidebar-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #c0a098;
            margin-bottom: 0.55rem;
            margin-top: 0.1rem;
        }

        .sidebar-section-label--required {
            color: #c0735a;
        }

        .sidebar-location-hint {
            font-size: 0.88rem;
            color: #9a7570;
            line-height: 1.55;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stTextInput"] {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] > div {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] > div > div {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] input {
            background: rgba(255, 252, 249, 0.7) !important;
            border: 1px solid rgba(190, 160, 150, 0.45) !important;
            border-radius: 12px !important;
            height: 2.8rem !important;
            color: #5c3733 !important;
            box-shadow: none !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
            font-size: 0.95rem !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border: 1px solid rgba(200, 140, 125, 0.7) !important;
            box-shadow: 0 0 0 2px rgba(238, 157, 176, 0.06) !important;
            background: rgba(255, 253, 251, 0.85) !important;
        }

        div[data-testid="stButton"] > button {
            border-radius: 16px !important;
            border: 1px solid #e6cec2 !important;
            color: #7a322e !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            min-height: 52px !important;
            transition: all 0.16s ease !important;
            background: rgba(255,250,246,0.82) !important;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #d8b3a3 !important;
            background: #fff1e8 !important;
            color: #6a2d2a !important;
            transform: translateY(-1px);
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            border-radius: 12px !important;
            min-height: 40px !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            color: #9a7570 !important;
            background: rgba(255, 248, 244, 0.6) !important;
            border: 1px solid rgba(190, 155, 145, 0.3) !important;
            box-shadow: none !important;
        }

        div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(255, 243, 236, 0.85) !important;
            border-color: rgba(200, 150, 138, 0.45) !important;
            color: #7a4a44 !important;
            transform: none !important;
        }

        .send-btn-wrap div[data-testid="stButton"] > button {
            background: linear-gradient(90deg, #f8c37d 0%, #f4b58d 35%, #ef9cb0 70%, #f38ab7 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 18px !important;
            min-height: 56px !important;
            box-shadow: 0 8px 18px rgba(239, 157, 176, 0.22) !important;
        }

        .send-btn-wrap div[data-testid="stButton"] > button:hover {
            background: linear-gradient(90deg, #f5bb74 0%, #f0aa84 35%, #eb92aa 70%, #ee81b1 100%) !important;
            color: white !important;
            border: none !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 10px 22px rgba(239, 157, 176, 0.26) !important;
        }

        .send-btn-wrap div[data-testid="stButton"] > button:active {
            transform: scale(0.98) !important;
            box-shadow: 0 4px 10px rgba(239, 157, 176, 0.25) !important;
        }

        div[data-testid="stButton"]:has(button[kind="secondary"]#start_over_btn) > button,
        button[key="start_over_btn"] {
            background: transparent !important;
            border: 1px solid rgba(122, 80, 72, 0.2) !important;
            color: #a08880 !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
            min-height: 36px !important;
            box-shadow: none !important;
        }

        .maps-link a {
            display: inline-block;
            background: linear-gradient(90deg, #f6c07d 0%, #f3b58b 28%, #ef9db0 68%, #f38bb8 100%);
            color: white !important;
            text-decoration: none !important;
            padding: 12px 22px;
            border-radius: 15px;
            font-weight: 700;
            margin-top: 10px;
            box-shadow: 0 8px 18px rgba(239, 157, 176, 0.18);
        }

        .restaurant-card {
            display: flex;
            align-items: stretch;
            gap: 28px;
            background: rgba(255,255,255,0.42);
            border: 1px solid rgba(122, 80, 72, 0.12);
            border-radius: 24px;
            padding: 22px;
            margin-bottom: 24px;
            box-shadow: 0 6px 18px rgba(80, 45, 40, 0.04);
        }

        .restaurant-card-left {
            flex: 0 0 36%;
            min-width: 280px;
        }

        .restaurant-card-right {
            flex: 1 1 auto;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .restaurant-card-image {
            width: 100%;
            height: 260px;
            object-fit: cover;
            border-radius: 18px;
            display: block;
            flex-shrink: 0;
        }

        .restaurant-card-left {
            flex: 0 0 340px;
            width: 340px;
            min-width: 0;
        }

        .restaurant-card-image-fallback {
            background: #f3f0ed;
            color: #a08b84;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .restaurant-card-title {
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.08;
            margin-bottom: 12px;
            color: #4b2b2a;
            letter-spacing: -0.7px;
        }

        .restaurant-card-rating {
            font-size: 1rem;
            color: #7b5a54;
            margin-bottom: 10px;
        }

        .restaurant-card-address {
            font-size: 0.95rem;
            color: #7e6c67;
            margin-bottom: 16px;
            line-height: 1.55;
        }

        .restaurant-card-quote-meta {
            font-size: 0.92rem;
            color: #866d67;
            margin-bottom: 10px;
        }

        .restaurant-card-text {
            font-size: 1rem;
            color: #52423d;
            line-height: 1.8;
            margin-bottom: 10px;
        }

        .restaurant-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(80, 45, 40, 0.06);
            transition: all 0.18s ease;
        }

        @media (max-width: 980px) {
            .restaurant-card {
                flex-direction: column;
                gap: 18px;
                padding: 18px;
            }

            .restaurant-card-left {
                min-width: 100%;
            }

            .restaurant-card-image {
                height: 220px;
            }

            .restaurant-card-title {
                font-size: 1.7rem;
            }
        }

        .popular-label {
            text-align: center;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #c0a098;
            margin-bottom: 1rem;
            margin-top: 0.5rem;
        }

        .tag-row {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }

                .nearby-card-shell {
            background: rgba(255,255,255,0.22);
            border: 1px solid rgba(122, 80, 72, 0.10);
            border-radius: 18px;
            padding: 1rem;
            height: 100%;
        }

        .nearby-card-image {
            width: 100%;
            height: 170px;
            object-fit: cover;
            border-radius: 14px;
            margin-bottom: 14px;
            display: block;
        }

        .nearby-card-meta {
            font-size: 0.94rem;
            color: #7b5a54;
            margin-bottom: 6px;
        }

        .nearby-card-address {
            font-size: 0.92rem;
            color: #7e6c67;
            line-height: 1.5;
            min-height: 0;
            margin-bottom: 8px;
        }

        .nearby-card-title {
            font-weight: 800;
            font-size: 1.1rem;
            color: #4b2b2a;
            line-height: 1.3;
            margin-bottom: 6px;
            min-height: 0;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
            letter-spacing: -0.3px;
        }

        .nearby-card-link a {
            color: #7a322e;
            font-size: 0.95rem;
            font-weight: 600;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try_get_browser_location()

    st.markdown('<div class="page-shell">', unsafe_allow_html=True)

    if st.session_state.results:
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:0.8rem;">
                <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
                    <div style="font-size:5rem;">✨</div>
                    <div class="brand-title">SpotOn</div>
                </div>
                <div style="margin-top:6px; color:#8c5a54; font-size:1rem; margin-bottom: 0.5rem;">
                    Good food, without the endless searching.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_title, col_btn = st.columns([5, 1])
        with col_title:
            st.markdown(
                '<div class="section-title" style="margin-top:0.4rem;">Top picks for you</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            st.markdown('<div style="padding-top:0.5rem;">', unsafe_allow_html=True)
            if st.button("↩ Start over", key="start_over_btn", use_container_width=True):
                reset_all_state()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        for idx, restaurant in enumerate(st.session_state.results, start=1):
            render_restaurant_card(restaurant, st.session_state.parsed_prefs, idx)
    elif not st.session_state.has_searched:
        render_empty_state()

    st.markdown('</div>', unsafe_allow_html=True)

    render_assistant_drawer()


if __name__ == "__main__":
    main()