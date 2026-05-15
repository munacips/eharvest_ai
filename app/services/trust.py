from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from app import config
from app.models import ReviewEntry
from app.services.common import (
    clamp,
    coerce_bool,
    coerce_float,
    coerce_int,
    extract_list_payload,
    get_first_present,
    parse_date,
)


DEFAULT_TRUST_SCORE = 2.5
DEFAULT_REVIEWS_PATH = "/api/reviews/user/{user_id}"
MAX_HELPFUL_VOTES_WEIGHT = 20
_VADER_ANALYZER = None


def get_vader_analyzer() -> Optional[Any]:
    """Lazily load VADER sentiment analysis and cache failures."""
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
    """Return a VADER compound sentiment score in the range [-1, 1]."""
    if comment is None or not str(comment).strip():
        return None
    analyzer = get_vader_analyzer()
    if analyzer is None:
        return None
    scores = analyzer.polarity_scores(str(comment))
    return float(scores.get("compound", 0.0))


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _normalize_rating_value(value: Any) -> Optional[float]:
    numeric_value = coerce_float(value)
    if numeric_value is None or numeric_value <= 0:
        return None

    # Upstream systems sometimes provide ratings on 0-1, 1-10, or 0-100 scales.
    if numeric_value <= 1.0:
        numeric_value *= 5.0
    elif numeric_value <= 5.0:
        pass
    elif numeric_value <= 10.0:
        numeric_value /= 2.0
    elif numeric_value <= 100.0:
        numeric_value /= 20.0

    return clamp(numeric_value, 1.0, 5.0)


def _normalize_review_date(value: Any) -> Optional[str]:
    parsed = parse_date(value)
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).strftime("%Y-%m-%d")
    return _clean_optional_text(value)


def _review_recency_weight(review_date: Optional[str]) -> float:
    parsed = parse_date(review_date)
    if parsed is None:
        return 1.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    age_days = max(0, (datetime.now(UTC) - parsed).days)
    if age_days <= 30:
        return 1.15
    if age_days <= 90:
        return 1.1
    if age_days <= 180:
        return 1.05
    if age_days <= 365:
        return 1.0
    return 0.95


def _extract_review_items(data: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return None

    reviews_value = data.get("reviews")
    if isinstance(reviews_value, list):
        return [item for item in reviews_value if isinstance(item, dict)]
    if "reviews" in data and reviews_value is None:
        return []

    extracted = extract_list_payload(data)
    if extracted or any(
        key in data for key in ("content", "items", "results", "records", "rows")
    ):
        return extracted

    for key in ("data", "payload", "result"):
        if key not in data:
            continue
        nested_items = _extract_review_items(data.get(key))
        if nested_items is not None:
            return nested_items

    return None


def _build_reviews_url(user_id: str) -> Tuple[Optional[str], Optional[str]]:
    path = config.SPRING_BOOT_REVIEWS_PATH or DEFAULT_REVIEWS_PATH
    if not path.startswith("/"):
        path = f"/{path}"

    if "{user_id}" in path:
        try:
            path = path.format(user_id=user_id)
        except (KeyError, IndexError, ValueError):
            return None, "spring_boot_reviews_path_invalid"

    return f"{config.SPRING_BOOT_BASE_URL}{path}", None


def coerce_review_payload(payload: Dict[str, Any]) -> Optional[ReviewEntry]:
    """Normalize inconsistent review payloads into the shared ReviewEntry model."""
    if not isinstance(payload, dict):
        return None

    rating_value = _normalize_rating_value(
        get_first_present(
            payload,
            ("rating", "score", "stars", "overallRating", "overall_rating"),
        )
    )
    comment = _clean_optional_text(
        get_first_present(
            payload,
            ("comment", "review", "reviewText", "text", "message", "body"),
        )
    )
    verified = coerce_bool(
        get_first_present(
            payload,
            ("verified_purchase", "verifiedPurchase", "verified", "isVerified"),
        )
    )
    helpful_value = coerce_int(
        get_first_present(
            payload,
            ("helpful_votes", "helpfulVotes", "helpfulCount", "upvotes"),
        )
    )
    reported = coerce_bool(
        get_first_present(
            payload,
            ("reported", "flagged", "isReported", "isFlagged"),
        )
    )
    review_date = _normalize_review_date(
        get_first_present(
            payload,
            (
                "review_date",
                "reviewDate",
                "createdAt",
                "created_at",
                "submittedAt",
                "date",
            ),
        )
    )

    sentiment = comment_sentiment_score(comment)
    text_rating = None
    if sentiment is not None:
        # When there is both structured and unstructured feedback, keep the
        # explicit rating dominant and let the text act as a small adjustment.
        text_rating = clamp(3.0 + (2.0 * sentiment), 1.0, 5.0)

    if rating_value is None and text_rating is None:
        return None
    if rating_value is None:
        rating_value = text_rating
    elif text_rating is not None:
        rating_value = (0.7 * rating_value) + (0.3 * text_rating)

    helpful_value = max(
        0, helpful_value) if helpful_value is not None else None

    return ReviewEntry(
        rating=rating_value,
        comment=comment,
        verified_purchase=verified,
        helpful_votes=helpful_value,
        reported=reported,
        review_date=review_date,
    )


def compute_trust_score(reviews: List[ReviewEntry]) -> Dict[str, Any]:
    """Compute a bounded trust score from normalized reviews."""
    if not reviews:
        return {
            "trust_score": DEFAULT_TRUST_SCORE,
            "review_count": 0,
            "average_rating": 0.0,
            "weighted_average": 0.0,
            "reported_ratio": 0.0,
            "verified_ratio": 0.0,
            "note": "no reviews available; returning neutral trust score",
        }

    ratings: List[float] = []
    weights: List[float] = []
    reported_count = 0
    verified_count = 0

    for review in reviews:
        rating = clamp(float(review.rating), 1.0, 5.0)
        helpful_votes = max(0, review.helpful_votes or 0)

        # Helpful, verified, and recent reviews count a bit more because they
        # tend to be the strongest trust signals for marketplace buyers.
        weight = 1.0 + \
            (min(helpful_votes, MAX_HELPFUL_VOTES_WEIGHT) / MAX_HELPFUL_VOTES_WEIGHT)
        weight *= _review_recency_weight(review.review_date)

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
    average_rating = float(np.mean(ratings)) if ratings else 0.0
    reported_ratio = reported_count / len(reviews)
    verified_ratio = verified_count / len(reviews)

    trust = weighted_avg - (reported_ratio * 0.8)
    trust = clamp(trust, 1.0, 5.0)

    return {
        "trust_score": round(trust, 2),
        "review_count": len(reviews),
        "average_rating": round(average_rating, 2),
        "weighted_average": round(weighted_avg, 2),
        "reported_ratio": round(reported_ratio, 2),
        "verified_ratio": round(verified_ratio, 2),
    }


async def fetch_user_reviews(user_id: str) -> Tuple[List[ReviewEntry], str, Optional[str]]:
    """Fetch reviews from the live Spring Boot reviews service."""
    if not config.SPRING_BOOT_BASE_URL:
        return [], "spring_boot_error", "spring_boot_base_url_not_set"

    url, warning = _build_reviews_url(user_id)
    if url is None:
        return [], "spring_boot_error", warning

    try:
        async with httpx.AsyncClient(
            timeout=config.PLATFORM_API_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        return [], "spring_boot_error", "spring_boot_error: TimeoutException"
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return [], "spring_boot_error", f"spring_boot_error: HTTPStatusError:{status_code}"
    except httpx.RequestError as exc:
        return [], "spring_boot_error", f"spring_boot_error: {exc.__class__.__name__}"

    if response.status_code == httpx.codes.NO_CONTENT or not response.content:
        return [], "spring_boot", None

    try:
        data = response.json()
    except ValueError:
        return [], "spring_boot_error", "spring_boot_error: InvalidJSON"

    raw_reviews = _extract_review_items(data)
    if raw_reviews is None:
        return [], "spring_boot_error", "spring_boot_error: InvalidPayload"

    reviews: List[ReviewEntry] = []
    skipped_count = 0
    for item in raw_reviews:
        review = coerce_review_payload(item)
        if review is None:
            skipped_count += 1
            continue
        reviews.append(review)

    warning = None
    if skipped_count:
        warning = f"spring_boot_warning: skipped_{skipped_count}_invalid_reviews"

    return reviews, "spring_boot", warning
