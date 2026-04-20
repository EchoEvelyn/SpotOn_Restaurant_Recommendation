import math


def normalize_rating(rating: float | None) -> float:
    """
    Map Google rating (0-5) to 0-1.
    Missing rating gets a neutral-but-not-strong score.
    """
    if rating is None:
        return 0.5
    return max(0.0, min(rating / 5.0, 1.0))


def normalize_review_count(count: int | None) -> float:
    """
    Compress review count with log scaling.
    This helps distinguish between:
    - very few reviews
    - moderate review base
    - highly reviewed places
    """
    if not count:
        return 0.2
    return min(math.log(count + 1, 10) / 3.0, 1.0)


def cuisine_match_score(types: list[str], prefs: dict) -> float:
    """
    Match requested cuisine against Google Places types.
    """
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
        expected_types = cuisine_map.get(cuisine, [])
        if any(t in types_set for t in expected_types):
            return 1.0

    return 0.45


def preference_match_score(restaurant: dict, prefs: dict) -> float:
    """
    Lightweight preference matching.
    Since Places metadata is limited, this should only gently boost candidates.
    """
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
    """
    Score how well the restaurant's price level matches user's budget preference.
    Returns 1.0 for exact match, 0.6 for adjacent, 0.3 for mismatch.
    If no budget preference or no price data, return neutral 0.7.
    """
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
    rating_component = normalize_rating(restaurant.get("rating"))
    review_component = normalize_review_count(restaurant.get("user_rating_count"))
    cuisine_component = cuisine_match_score(restaurant.get("types", []), prefs)
    preference_component = preference_match_score(restaurant, prefs)
    budget_component = budget_match_score(restaurant, prefs)

    return (
        0.35 * rating_component
        + 0.17 * review_component
        + 0.20 * cuisine_component
        + 0.13 * preference_component
        + 0.15 * budget_component
    )


def attach_scores(restaurants: list[dict], prefs: dict) -> list[dict]:
    scored = []
    for restaurant in restaurants:
        item = dict(restaurant)
        item["_score"] = score_restaurant(restaurant, prefs)
        scored.append(item)
    return scored


def rank_restaurants(restaurants: list[dict], prefs: dict) -> list[dict]:
    scored = attach_scores(restaurants, prefs)
    return sorted(scored, key=lambda r: r["_score"], reverse=True)