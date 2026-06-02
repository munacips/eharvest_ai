"""Test for generating 20 sample reviews with user_id: 1."""
import json
from datetime import datetime, timedelta, UTC
from app.models import ReviewEntry
from app.services.trust import compute_trust_score


def generate_sample_reviews(user_id: int = 1, count: int = 20) -> list[dict]:
    """Generate sample reviews with user_id."""
    reviews = [
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Excellent quality and fast delivery. Very satisfied with the purchase!",
            "verified_purchase": True,
            "helpful_votes": 15,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.5,
            "comment": "Good product, reasonable price. Would recommend to others.",
            "verified_purchase": True,
            "helpful_votes": 12,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Perfect! Exceeded my expectations.",
            "verified_purchase": True,
            "helpful_votes": 8,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=15)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.0,
            "comment": "Good quality but took longer to arrive than expected.",
            "verified_purchase": True,
            "helpful_votes": 10,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=20)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Amazing seller! Very responsive and helpful.",
            "verified_purchase": True,
            "helpful_votes": 18,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=25)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 3.5,
            "comment": "Average quality, could be better for the price.",
            "verified_purchase": True,
            "helpful_votes": 5,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Best purchase I've made this month!",
            "verified_purchase": True,
            "helpful_votes": 20,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=35)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.0,
            "comment": "Decent product. No major complaints.",
            "verified_purchase": False,
            "helpful_votes": 6,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=40)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 3.0,
            "comment": "Product works but has some minor defects.",
            "verified_purchase": True,
            "helpful_votes": 4,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=45)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Highly recommend! Great value for money.",
            "verified_purchase": True,
            "helpful_votes": 22,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=50)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.5,
            "comment": "Very happy with my purchase. Fast shipping too.",
            "verified_purchase": True,
            "helpful_votes": 14,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=55)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 2.0,
            "comment": "Not as described. Quality is below average.",
            "verified_purchase": True,
            "helpful_votes": 3,
            "reported": True,
            "review_date": (datetime.now(UTC) - timedelta(days=60)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Absolutely love it! Worth every penny.",
            "verified_purchase": True,
            "helpful_votes": 25,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=65)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.0,
            "comment": "Good product, meets expectations.",
            "verified_purchase": True,
            "helpful_votes": 9,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=70)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 3.5,
            "comment": "Fair quality, acceptable price.",
            "verified_purchase": False,
            "helpful_votes": 2,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=75)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Excellent customer service and product quality!",
            "verified_purchase": True,
            "helpful_votes": 19,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=80)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.5,
            "comment": "Quality is good, delivery was fast.",
            "verified_purchase": True,
            "helpful_votes": 11,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=85)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 3.0,
            "comment": "Okay product. Not bad, not great.",
            "verified_purchase": True,
            "helpful_votes": 7,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 5.0,
            "comment": "Perfect purchase! Highly satisfied.",
            "verified_purchase": True,
            "helpful_votes": 16,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=95)).strftime("%Y-%m-%d")
        },
        {
            "user_id": user_id,
            "rating": 4.0,
            "comment": "Good value for money. Will buy again.",
            "verified_purchase": True,
            "helpful_votes": 13,
            "reported": False,
            "review_date": (datetime.now(UTC) - timedelta(days=100)).strftime("%Y-%m-%d")
        }
    ]
    
    return reviews[:count]


def test_generate_reviews_json():
    """Test generating 20 reviews and verify JSON response format."""
    reviews_data = generate_sample_reviews(user_id=1, count=20)
    
    # Verify we have 20 reviews
    assert len(reviews_data) == 20, f"Expected 20 reviews, got {len(reviews_data)}"
    
    # Verify all reviews have user_id = 1
    for review in reviews_data:
        assert review["user_id"] == 1, f"Review user_id mismatch: {review['user_id']}"
    
    # Verify required fields exist
    required_fields = ["rating", "comment", "verified_purchase", "helpful_votes", "reported", "review_date"]
    for i, review in enumerate(reviews_data):
        for field in required_fields:
            assert field in review, f"Review {i} missing field: {field}"
    
    print("✓ All reviews generated successfully with correct structure")
    return reviews_data


def test_compute_trust_score_from_reviews():
    """Test computing trust score from generated reviews."""
    reviews_data = generate_sample_reviews(user_id=1, count=20)
    
    # Convert to ReviewEntry objects
    review_entries = [
        ReviewEntry(
            rating=r["rating"],
            comment=r["comment"],
            verified_purchase=r["verified_purchase"],
            helpful_votes=r["helpful_votes"],
            reported=r["reported"],
            review_date=r["review_date"]
        )
        for r in reviews_data
    ]
    
    # Compute trust score
    trust_details = compute_trust_score(review_entries)
    
    # Verify trust score structure
    assert "trust_score" in trust_details
    assert "review_count" in trust_details
    assert "average_rating" in trust_details
    assert "weighted_average" in trust_details
    assert "reported_ratio" in trust_details
    assert "verified_ratio" in trust_details
    
    # Verify counts
    assert trust_details["review_count"] == 20
    
    print(f"✓ Trust score computed: {trust_details['trust_score']}")
    print(f"✓ Average rating: {trust_details['average_rating']}")
    print(f"✓ Weighted average: {trust_details['weighted_average']}")
    print(f"✓ Reported ratio: {trust_details['reported_ratio']}")
    print(f"✓ Verified ratio: {trust_details['verified_ratio']}")
    
    return trust_details


def get_response_json(user_id: int = 1) -> dict:
    """Generate the final JSON response with reviews and trust score."""
    reviews_data = generate_sample_reviews(user_id=user_id, count=20)
    
    # Convert to ReviewEntry objects for trust computation
    review_entries = [
        ReviewEntry(
            rating=r["rating"],
            comment=r["comment"],
            verified_purchase=r["verified_purchase"],
            helpful_votes=r["helpful_votes"],
            reported=r["reported"],
            review_date=r["review_date"]
        )
        for r in reviews_data
    ]
    
    # Compute trust score
    trust_details = compute_trust_score(review_entries)
    
    # Build the response
    response = {
        "user_id": user_id,
        "reviews": reviews_data,
        "trust_analysis": {
            "trust_score": trust_details["trust_score"],
            "scale": 5,
            "review_count": trust_details["review_count"],
            "details": {
                "average_rating": trust_details["average_rating"],
                "weighted_average": trust_details["weighted_average"],
                "reported_ratio": trust_details["reported_ratio"],
                "verified_ratio": trust_details["verified_ratio"],
            }
        },
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    
    return response


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing Review Generation")
    print("=" * 60)
    
    test_generate_reviews_json()
    print()
    
    test_compute_trust_score_from_reviews()
    print()
    
    # Generate and print final JSON response
    print("=" * 60)
    print("Final JSON Response with 20 Reviews (user_id: 1)")
    print("=" * 60)
    response = get_response_json(user_id=1)
    print(json.dumps(response, indent=2))
