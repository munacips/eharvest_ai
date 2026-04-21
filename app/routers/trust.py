from datetime import UTC, datetime

from fastapi import APIRouter

from app.services.trust import compute_trust_score, fetch_user_reviews

router = APIRouter()


@router.get("/trust-score/{user_id}")
async def trust_score(user_id: str):
    reviews, source, warning = await fetch_user_reviews(user_id)
    details = compute_trust_score(reviews)

    response = {
        "user_id": user_id,
        "trust_score": details["trust_score"],
        "scale": 5,
        "review_count": details["review_count"],
        "source": source,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "details": {
            "average_rating": details["average_rating"],
            "weighted_average": details["weighted_average"],
            "reported_ratio": details["reported_ratio"],
            "verified_ratio": details["verified_ratio"],
        },
    }

    if "note" in details:
        response["details"]["note"] = details["note"]

    if warning:
        response["warnings"] = [warning]

    return response
