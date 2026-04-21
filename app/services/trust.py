from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from app import config
from app.models import ReviewEntry
from app.services.common import clamp, coerce_bool


_VADER_ANALYZER = None


def get_vader_analyzer():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is False:
        return None
    if _VADER_ANALYZER is not None:
        return _VADER_ANALYZER

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
    except Exception:
        _VADER_ANALYZER = False
        return None

    try:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
        return _VADER_ANALYZER
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER_ANALYZER = False

    return _VADER_ANALYZER


def comment_sentiment_score(comment: Optional[str]) -> Optional[float]:
    if comment is None or not str(comment).strip():
        return None
    analyzer = get_vader_analyzer()
    if analyzer is None:
        return None
    scores = analyzer.polarity_scores(str(comment))
    return float(scores.get("compound", 0.0))


def coerce_review_payload(payload: Dict[str, Any]) -> Optional[ReviewEntry]:
    rating = payload.get("rating")
    if rating is None:
        rating = payload.get("score")
    verified = payload.get("verified_purchase")
    if verified is None:
        verified = payload.get("verifiedPurchase")

    helpful = payload.get("helpful_votes")
    if helpful is None:
        helpful = payload.get("helpfulVotes")

    reported = payload.get("reported")
    if reported is None:
        reported = payload.get("flagged")

    review_date = payload.get("review_date")
    if review_date is None:
        review_date = payload.get("createdAt")

    comment = payload.get("comment")
    if comment is None:
        comment = payload.get("review")
    if comment is None:
        comment = payload.get("reviewText")
    if comment is None:
        comment = payload.get("text")

    rating_value = None
    if rating is not None:
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            rating_value = None

    sentiment = comment_sentiment_score(comment)
    text_rating = None
    if sentiment is not None:
        text_rating = clamp(3.0 + (2.0 * sentiment), 1.0, 5.0)

    if rating_value is None and text_rating is None:
        return None

    if rating_value is None:
        rating_value = text_rating
    elif text_rating is not None:
        rating_value = (0.7 * rating_value) + (0.3 * text_rating)

    helpful_value = None
    if helpful is not None:
        try:
            helpful_value = int(helpful)
        except (TypeError, ValueError):
            helpful_value = None

    return ReviewEntry(
        rating=rating_value,
        comment=comment,
        verified_purchase=coerce_bool(verified),
        helpful_votes=helpful_value,
        reported=coerce_bool(reported),
        review_date=review_date,
    )


def placeholder_reviews(user_id: str) -> List[ReviewEntry]:
    return [
        ReviewEntry(
            rating=4.6,
            comment="Quick delivery and the produce quality was excellent.",
            verified_purchase=True,
            helpful_votes=12,
            reported=False,
            review_date="2025-06-12",
        ),
        ReviewEntry(
            rating=4.2,
            comment="Good experience overall, but packaging could improve.",
            verified_purchase=True,
            helpful_votes=5,
            reported=False,
            review_date="2025-07-03",
        ),
        ReviewEntry(
            rating=3.8,
            comment="Decent service, average quality, nothing special.",
            verified_purchase=False,
            helpful_votes=2,
            reported=False,
            review_date="2025-09-01",
        ),
        ReviewEntry(
            rating=4.9,
            comment="Fantastic support and very fresh produce!",
            verified_purchase=True,
            helpful_votes=18,
            reported=False,
            review_date="2025-11-15",
        ),
        ReviewEntry(
            rating=2.6,
            comment="Late delivery and items were damaged.",
            verified_purchase=False,
            helpful_votes=1,
            reported=True,
            review_date="2026-01-10",
        ),
    ]


def compute_trust_score(reviews: List[ReviewEntry]) -> Dict[str, Any]:
    if not reviews:
        return {
            "trust_score": 2.5,
            "review_count": 0,
            "average_rating": 0.0,
            "weighted_average": 0.0,
            "reported_ratio": 0.0,
            "verified_ratio": 0.0,
            "note": "no reviews available; returning neutral trust score",
        }

    ratings = []
    weights = []
    reported_count = 0
    verified_count = 0

    for review in reviews:
        rating = clamp(float(review.rating), 1.0, 5.0)
        helpful_votes = review.helpful_votes or 0
        helpful_votes = max(0, helpful_votes)
        weight = 1.0 + (min(helpful_votes, 20) / 20.0)

        if review.verified_purchase:
            weight += 0.25
            verified_count += 1

        if review.reported:
            weight *= 0.6
            reported_count += 1

        ratings.append(rating)
        weights.append(weight)

    weighted_avg = float(np.average(
        ratings, weights=weights)) if weights else 0.0
    reported_ratio = reported_count / len(reviews)
    trust = weighted_avg - (reported_ratio * 0.8)
    trust = clamp(trust, 1.0, 5.0)
    average_rating = float(np.mean(ratings)) if ratings else 0.0
    verified_ratio = verified_count / len(reviews)

    return {
        "trust_score": round(trust, 2),
        "review_count": len(reviews),
        "average_rating": round(average_rating, 2),
        "weighted_average": round(weighted_avg, 2),
        "reported_ratio": round(reported_ratio, 2),
        "verified_ratio": round(verified_ratio, 2),
    }


async def fetch_user_reviews(user_id: str) -> Tuple[List[ReviewEntry], str, Optional[str]]:
    if config.USE_REVIEW_PLACEHOLDER or not config.SPRING_BOOT_BASE_URL:
        return placeholder_reviews(user_id), "placeholder", None

    path = config.SPRING_BOOT_REVIEWS_PATH or "/api/reviews/user/{user_id}"
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{config.SPRING_BOOT_BASE_URL}{path.format(user_id=user_id)}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return placeholder_reviews(user_id), "placeholder_fallback", f"spring_boot_error: {exc.__class__.__name__}"

    raw_reviews = data
    if isinstance(data, dict) and "reviews" in data:
        raw_reviews = data["reviews"]

    reviews: List[ReviewEntry] = []
    if isinstance(raw_reviews, list):
        for item in raw_reviews:
            if not isinstance(item, dict):
                continue
            review = coerce_review_payload(item)
            if review is not None:
                reviews.append(review)

    return reviews, "spring_boot", None
