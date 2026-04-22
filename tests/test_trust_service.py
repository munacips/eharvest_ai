import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import ReviewEntry
from app.services import trust as trust_service


def _install_fake_async_client(monkeypatch, response_factory):
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            return response_factory(url)

    monkeypatch.setattr(trust_service.httpx, "AsyncClient", DummyAsyncClient)


def test_coerce_review_payload_normalizes_alt_fields(monkeypatch):
    monkeypatch.setattr(trust_service, "comment_sentiment_score", lambda comment: 1.0)

    review = trust_service.coerce_review_payload(
        {
            "score": 8,
            "reviewText": "  Great produce and fast delivery.  ",
            "verifiedPurchase": "yes",
            "helpfulVotes": "7",
            "flagged": 0,
            "createdAt": "2026-04-01T10:00:00Z",
        }
    )

    assert review is not None
    assert round(review.rating, 2) == 4.3
    assert review.comment == "Great produce and fast delivery."
    assert review.verified_purchase is True
    assert review.helpful_votes == 7
    assert review.reported is False
    assert review.review_date == "2026-04-01"


def test_compute_trust_score_weights_verified_and_reported_reviews():
    reviews = [
        ReviewEntry(
            rating=5.0,
            verified_purchase=True,
            helpful_votes=20,
            reported=False,
            review_date=None,
        ),
        ReviewEntry(
            rating=1.0,
            verified_purchase=False,
            helpful_votes=0,
            reported=True,
            review_date=None,
        ),
    ]

    details = trust_service.compute_trust_score(reviews)

    assert details == {
        "trust_score": 3.76,
        "review_count": 2,
        "average_rating": 3.0,
        "weighted_average": 4.16,
        "reported_ratio": 0.5,
        "verified_ratio": 0.5,
    }


def test_fetch_user_reviews_returns_error_for_invalid_json(monkeypatch):
    def response_factory(url):
        request = httpx.Request("GET", url)
        return httpx.Response(200, content=b"not-json", request=request)

    monkeypatch.setattr(trust_service.config, "USE_REVIEW_PLACEHOLDER", False)
    monkeypatch.setattr(trust_service.config, "SPRING_BOOT_BASE_URL", "https://reviews.example.com")
    monkeypatch.setattr(
        trust_service.config,
        "SPRING_BOOT_REVIEWS_PATH",
        "/api/reviews/user/{user_id}",
    )
    _install_fake_async_client(monkeypatch, response_factory)

    reviews, source, warning = asyncio.run(
        trust_service.fetch_user_reviews("user-123")
    )

    assert reviews == []
    assert source == "spring_boot_error"
    assert warning == "spring_boot_error: InvalidJSON"


def test_fetch_user_reviews_returns_placeholder_only_when_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(trust_service.config, "USE_REVIEW_PLACEHOLDER", True)
    monkeypatch.setattr(trust_service.config, "SPRING_BOOT_BASE_URL", "")

    reviews, source, warning = asyncio.run(
        trust_service.fetch_user_reviews("user-123")
    )

    assert len(reviews) == 5
    assert source == "placeholder"
    assert warning == "trust_placeholder_enabled"


def test_fetch_user_reviews_requires_base_url_when_placeholder_disabled(monkeypatch):
    monkeypatch.setattr(trust_service.config, "USE_REVIEW_PLACEHOLDER", False)
    monkeypatch.setattr(trust_service.config, "SPRING_BOOT_BASE_URL", "")

    reviews, source, warning = asyncio.run(
        trust_service.fetch_user_reviews("user-123")
    )

    assert reviews == []
    assert source == "spring_boot_error"
    assert warning == "spring_boot_base_url_not_set"


def test_fetch_user_reviews_extracts_nested_reviews_and_warns_on_skips(monkeypatch):
    captured = {}

    def response_factory(url):
        captured["url"] = url
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "data": {
                    "reviews": [
                        {
                            "overallRating": 80,
                            "reviewText": "Excellent service",
                            "verified": "true",
                            "helpfulCount": "3",
                            "submittedAt": "2026-04-10T07:00:00Z",
                        },
                        {
                            "text": "Solid quality",
                            "flagged": "false",
                            "helpfulVotes": 1,
                        },
                        {
                            "rating": None,
                            "comment": "   ",
                        },
                        "bad-payload",
                    ]
                }
            },
            request=request,
        )

    monkeypatch.setattr(
        trust_service,
        "comment_sentiment_score",
        lambda comment: 0.5 if comment else None,
    )
    monkeypatch.setattr(trust_service.config, "USE_REVIEW_PLACEHOLDER", False)
    monkeypatch.setattr(trust_service.config, "SPRING_BOOT_BASE_URL", "https://reviews.example.com")
    monkeypatch.setattr(
        trust_service.config,
        "SPRING_BOOT_REVIEWS_PATH",
        "/api/reviews/user/{user_id}",
    )
    _install_fake_async_client(monkeypatch, response_factory)

    reviews, source, warning = asyncio.run(
        trust_service.fetch_user_reviews("grower-42")
    )

    assert captured["url"] == "https://reviews.example.com/api/reviews/user/grower-42"
    assert source == "spring_boot"
    assert warning == "spring_boot_warning: skipped_1_invalid_reviews"
    assert len(reviews) == 2
    assert reviews[0].review_date == "2026-04-10"
    assert reviews[0].verified_purchase is True
    assert round(reviews[0].rating, 2) == 4.0
    assert round(reviews[1].rating, 2) == 4.0
