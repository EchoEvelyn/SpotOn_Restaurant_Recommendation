import requests


class GooglePlacesTool:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing Google API key.")

        self.api_key = api_key
        self.search_url = "https://places.googleapis.com/v1/places:searchText"
        self.nearby_url = "https://places.googleapis.com/v1/places:searchNearby"
        self.details_base_url = "https://places.googleapis.com/v1/places"

    def _build_headers(self, field_mask: str) -> dict:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

    def build_photo_url(self, photo_name: str, max_height_px: int = 400) -> str:
        return (
            f"https://places.googleapis.com/v1/{photo_name}/media"
            f"?maxHeightPx={max_height_px}&key={self.api_key}"
        )

    def _extract_photo_url(self, place: dict) -> str | None:
        photos = place.get("photos", [])
        if not photos:
            return None

        photo_name = photos[0].get("name")
        if not photo_name:
            return None

        return self.build_photo_url(photo_name)

    def _normalize_place(self, place: dict, reviews: list[dict] | None = None) -> dict:
        return {
            "place_id": place.get("id"),
            "name": (place.get("displayName") or {}).get("text"),
            "address": place.get("formattedAddress"),
            "rating": place.get("rating"),
            "user_rating_count": place.get("userRatingCount"),
            "types": place.get("types", []),
            "price_level": place.get("priceLevel"),
            "google_maps_uri": place.get("googleMapsUri"),
            "photo_url": self._extract_photo_url(place),
            "reviews": reviews or [],
        }

    def get_place_reviews(self, place_id: str, reviews_per_place: int = 5) -> list[dict]:
        if not place_id:
            return []

        headers = self._build_headers("reviews")
        url = f"{self.details_base_url}/{place_id}"

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
                    "author_name": (review.get("authorAttribution") or {}).get(
                        "displayName",
                        "Anonymous",
                    ),
                    "rating": review.get("rating"),
                    "relative_time": review.get("relativePublishTimeDescription", ""),
                }
            )

        return reviews

    def search_restaurants(
        self,
        query: str,
        max_results: int = 10,
        include_reviews: bool = True,
        reviews_per_place: int = 5,
    ) -> list[dict]:
        if not query or not query.strip():
            return []

        headers = self._build_headers(
            ",".join(
                [
                    "places.id",
                    "places.displayName",
                    "places.formattedAddress",
                    "places.rating",
                    "places.userRatingCount",
                    "places.types",
                    "places.priceLevel",
                    "places.googleMapsUri",
                    "places.photos",
                ]
            )
        )

        payload = {
            "textQuery": query.strip(),
            "pageSize": max_results,
        }

        response = requests.post(
            self.search_url,
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        restaurants = []
        for place in data.get("places", []):
            place_id = place.get("id")
            reviews = []

            if include_reviews and place_id:
                try:
                    reviews = self.get_place_reviews(
                        place_id=place_id,
                        reviews_per_place=reviews_per_place,
                    )
                except Exception:
                    reviews = []

            restaurants.append(self._normalize_place(place, reviews=reviews))

        return restaurants

    def nearby_restaurants(
        self,
        lat: float,
        lng: float,
        max_results: int = 10,
        radius_meters: float = 1500.0,
        include_reviews: bool = False,
        reviews_per_place: int = 5,
    ) -> list[dict]:
        headers = self._build_headers(
            ",".join(
                [
                    "places.id",
                    "places.displayName",
                    "places.formattedAddress",
                    "places.rating",
                    "places.userRatingCount",
                    "places.types",
                    "places.priceLevel",
                    "places.googleMapsUri",
                    "places.location",
                    "places.photos",
                ]
            )
        )

        payload = {
            "includedTypes": ["restaurant"],
            "maxResultCount": max_results,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng,
                    },
                    "radius": radius_meters,
                }
            },
        }

        response = requests.post(
            self.nearby_url,
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        restaurants = []
        for place in data.get("places", []):
            place_id = place.get("id")
            reviews = []

            if include_reviews and place_id:
                try:
                    reviews = self.get_place_reviews(
                        place_id=place_id,
                        reviews_per_place=reviews_per_place,
                    )
                except Exception:
                    reviews = []

            restaurants.append(self._normalize_place(place, reviews=reviews))

        return restaurants