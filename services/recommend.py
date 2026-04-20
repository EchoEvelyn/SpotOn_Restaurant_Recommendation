import os

from tools.google_places import GooglePlacesTool
from ranking import rank_restaurants
from agent.extractor import summarize_reviews_llm, select_review_quote_llm


def build_query(user_request: str) -> str:
    return f"{user_request} near UCSD"


def enrich_restaurants_with_review_insights(
    restaurants: list[dict],
    top_k_for_llm: int = 5,
) -> list[dict]:
    """
    For the top-ranked restaurants, generate:
    - review_summary
    - selected_review_quote

    This avoids doing LLM calls inside the UI rendering layer.
    """
    for idx, restaurant in enumerate(restaurants):
        # default fields for all restaurants
        restaurant["review_summary"] = ""
        restaurant["selected_review_quote"] = None

        # only enrich the top K to keep latency reasonable
        if idx >= top_k_for_llm:
            continue

        reviews = restaurant.get("reviews", [])
        if not reviews:
            continue

        try:
            summary = summarize_reviews_llm(
                name=restaurant.get("name", "Unknown"),
                reviews=reviews,
            )
            restaurant["review_summary"] = summary or ""
        except Exception:
            restaurant["review_summary"] = ""

        try:
            selected = select_review_quote_llm(
                name=restaurant.get("name", "Unknown"),
                reviews=reviews,
            )

            if selected:
                restaurant["selected_review_quote"] = {
                    "text": (selected.get("text") or "").strip(),
                    "author_name": selected.get("author_name", "Anonymous"),
                    "rating": selected.get("rating"),
                    "relative_time": selected.get("relative_time", ""),
                }
            else:
                restaurant["selected_review_quote"] = None
        except Exception:
            restaurant["selected_review_quote"] = None

    return restaurants


def print_recommendations(restaurants: list[dict], top_k: int = 5) -> None:
    if not restaurants:
        print("No restaurants found.")
        return

    print("\nTop recommendations:\n")
    for i, r in enumerate(restaurants[:top_k], start=1):
        print(f"{i}. {r.get('name', 'Unknown')}")
        print(f"   Rating: {r.get('rating', 'N/A')}")
        print(f"   Price: {r.get('price_level', 'N/A')}")
        print(f"   Address: {r.get('address', 'N/A')}")

        summary = (r.get("review_summary") or "").strip()
        quote = r.get("selected_review_quote")

        if summary:
            print(f"   Why people like it: {summary}")

        if quote and quote.get("text"):
            quote_text = quote["text"]
            if len(quote_text) > 180:
                quote_text = quote_text[:177].rstrip() + "..."

            meta = quote.get("author_name", "Anonymous")
            if quote.get("rating") is not None:
                meta += f" · {quote['rating']} stars"
            if quote.get("relative_time"):
                meta += f" · {quote['relative_time']}"

            print(f'   Customer quote: "{quote_text}"')
            print(f"   Source: {meta}")

        print()


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please run:\n"
            'export GOOGLE_API_KEY="your_key_here"'
        )

    user_request = input("What do you want to eat? ").strip()
    if not user_request:
        print("Please enter a valid request.")
        return

    query = build_query(user_request)

    places_tool = GooglePlacesTool(api_key)
    restaurants = places_tool.search_restaurants(
        query=query,
        max_results=10,
        include_reviews=True,
        reviews_per_place=5,
    )

    ranked_restaurants = rank_restaurants(restaurants)
    ranked_restaurants = enrich_restaurants_with_review_insights(
        ranked_restaurants,
        top_k_for_llm=5,
    )

    print(f"\nSearch query used: {query}")
    print_recommendations(ranked_restaurants, top_k=5)


if __name__ == "__main__":
    main()