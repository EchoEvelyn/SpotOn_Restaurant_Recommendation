import math


def normalize_rating(rating: float | None) -> float:
    if rating is None:
        return 0.5
    return max(0.0, min(rating / 5.0, 1.0))


def normalize_review_count(count: int | None) -> float:
    if not count:
        return 0.2
    return min(math.log(count + 1, 10) / 3.0, 1.0)


def cuisine_match_score(types: list[str], prefs: dict) -> float:
    if not types:
        return 0.5

    requested = prefs.get("cuisine", [])
    if not requested:
        return 0.7

    cuisine_map = {
        "chinese": ["chinese_restaurant"],
        "japanese": ["japanese_restaurant"],
        "sushi": ["sushi_restaurant", "japanese_restaurant"],
        "korean": ["korean_restaurant"],
        "thai": ["thai_restaurant"],
        "vietnamese": ["vietnamese_restaurant"],
        "mexican": ["mexican_restaurant"],
        "italian": ["italian_restaurant"],
        "indian": ["indian_restaurant"],
        "mediterranean": ["mediterranean_restaurant"],
        "american": ["american_restaurant"],
        "poke": ["hawaiian_restaurant", "restaurant"],
        "salad": ["salad_shop", "restaurant"],
        "asian": ["asian_restaurant", "restaurant"],
    }

    types_set = set(types)
    for cuisine in requested:
        if any(t in types_set for t in cuisine_map.get(cuisine, [])):
            return 1.0

    return 0.45


def preference_match_score(restaurant: dict, prefs: dict) -> float:
    preferences = prefs.get("preferences", [])
    if not preferences:
        return 0.7

    types_text = " ".join(restaurant.get("types", [])).lower()
    name_text = (restaurant.get("name") or "").lower()
    score = 0.5

    if "healthy" in preferences:
        if any(x in types_text for x in ["salad_shop", "vegetarian_restaurant"]):
            score += 0.2
        if any(x in name_text for x in ["sweetgreen", "urban plates", "cava"]):
            score += 0.15

    if "quick" in preferences:
        if any(x in types_text for x in ["meal_takeaway", "fast_food_restaurant"]):
            score += 0.2

    if "spicy" in preferences:
        score += 0.05
    if "light" in preferences:
        score += 0.05
    if "warm" in preferences or "comforting" in preferences:
        score += 0.05
    if "crispy" in preferences:
        score += 0.03

    return min(score, 1.0)


def budget_match_score(restaurant: dict, prefs: dict) -> float:
    budget_level = prefs.get("budget_level")
    if not budget_level:
        return 0.7

    price_level = restaurant.get("price_level")
    if not price_level:
        return 0.7

    exact_match = {
        "cheap": "PRICE_LEVEL_INEXPENSIVE",
        "moderate": "PRICE_LEVEL_MODERATE",
        "expensive": "PRICE_LEVEL_EXPENSIVE",
        "very_expensive": "PRICE_LEVEL_VERY_EXPENSIVE",
    }
    adjacent = {
        "cheap": {"PRICE_LEVEL_MODERATE"},
        "moderate": {"PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_EXPENSIVE"},
        "expensive": {"PRICE_LEVEL_MODERATE", "PRICE_LEVEL_VERY_EXPENSIVE"},
        "very_expensive": {"PRICE_LEVEL_EXPENSIVE"},
    }

    if price_level == exact_match.get(budget_level):
        return 1.0
    if price_level in adjacent.get(budget_level, set()):
        return 0.6
    return 0.3


def score_restaurant(restaurant: dict, prefs: dict) -> float:
    return (
        0.35 * normalize_rating(restaurant.get("rating"))
        + 0.17 * normalize_review_count(restaurant.get("user_rating_count"))
        + 0.20 * cuisine_match_score(restaurant.get("types", []), prefs)
        + 0.13 * preference_match_score(restaurant, prefs)
        + 0.15 * budget_match_score(restaurant, prefs)
    )


def rank_restaurants(restaurants: list[dict], prefs: dict) -> list[dict]:
    scored = [{**r, "_score": score_restaurant(r, prefs)} for r in restaurants]
    return sorted(scored, key=lambda r: r["_score"], reverse=True)