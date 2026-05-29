import csv
import random
from datetime import datetime, timedelta

# 1. Expanded Geographic Database covering all 10 Provinces of Zimbabwe
REGIONS = [
    "Harare", "Bulawayo", "Manicaland", "Midlands", "Masvingo",
    "Mashonaland Central", "Mashonaland West", "Mashonaland East",
    "Matabeleland South", "Matabeleland North"
]

# 2. Complete list of requested commodities
COMMODITIES = [
    "Maize", "Maize meal", "Rice", "Sorghum", "Millet (finger millet)", "Wheat", "Barley",
    "Beans (sugar)", "Beans (round)", "Groundnuts (shelled)", "Soybeans", "Cowpeas", "Pigeon peas",
    "Tomatoes", "Onions", "Carrots", "Cabbages", "Spinach (rape)", "Sweet potatoes", "Irish potatoes",
    "Butternuts", "Pumpkin", "Green pepper", "Garlic", "Okra (derere)", "Cucumber",
    "Bananas", "Mangoes", "Avocados", "Oranges",
    "Oil (vegetable)", "Cooking oil (sunflower)",
    "Beef", "Fish (kapenta)", "Eggs (tray of 30)", "Chicken",
    "Salt", "Sugar", "Flour (wheat)"
]

# 3. Buyer Profiles and their typical purchasing behavior
BUYER_PROFILES = [
    {"type": "Household", "min_qty": 5.0, "max_qty": 150.0},
    {"type": "Retailer / Vendor", "min_qty": 200.0, "max_qty": 2500.0},
    {"type": "Wholesaler", "min_qty": 3000.0, "max_qty": 15000.0},
    {"type": "Institutional (GMB/Milling)",
     "min_qty": 20000.0, "max_qty": 50000.0}
]

# Helper function to determine Zimbabwe's agricultural season based on month


def get_zimbabwe_season(month):
    # May to October is the cool-to-warm dry season (main harvesting window)
    if 5 <= month <= 10:
        return "harvest/dry"
    # November to April is the primary rainy summer cropping season
    return "summer"


# 4. Timeline Configuration (Jan 1, 2022 to May 22, 2026)
start_date = datetime(2022, 1, 1)
end_date = datetime(2026, 5, 22)
delta_days = (end_date - start_date).days

FILENAME = "zimbabwe_crop_transactions.csv"
TOTAL_RECORDS = 1000000

print(f"Starting pipeline to generate {TOTAL_RECORDS:,} transaction rows...")
print(f"Target file destination: {FILENAME}")

# 5. Stream and Output directly to the local storage target
with open(FILENAME, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the custom schema header row
    writer.writerow(["date", "commodity", "region",
                    "quantity", "buyer_type", "season"])

    for i in range(TOTAL_RECORDS):
        # Determine random date
        random_days = random.randint(0, delta_days)
        current_date = start_date + timedelta(days=random_days)
        date_str = current_date.strftime("%Y-%m-%d")

        # Pull season programmatically from the month
        season = get_zimbabwe_season(current_date.month)

        # Select standard variables
        commodity = random.choice(COMMODITIES)
        region = random.choice(REGIONS)

        # Select buyer profile and scale volumes logically
        buyer = random.choice(BUYER_PROFILES)
        buyer_type = buyer["type"]

        # Generate logical float quantity based on buyer type, rounded to 1 decimal place
        quantity = round(random.uniform(buyer["min_qty"], buyer["max_qty"]), 1)

        # Introduce a real-world constraint: Households don't buy bulk industrial raw grain tons
        if buyer_type == "Institutional (GMB/Milling)" and commodity in ["Salt", "Eggs (tray of 30)", "Garlic", "Cucumber"]:
            # Re-scale industrial buyers downwards for specialty/table items, or pivot them to grains
            quantity = round(random.uniform(500.0, 3000.0), 1)

        # Write flat array row entry
        writer.writerow([
            date_str,
            commodity,
            region,
            quantity,
            buyer_type,
            season
        ])

        # Progress Logger
        if (i + 1) % 250000 == 0:
            print(
                f"Progress checkpoint: {i + 1:,} rows successfully committed...")

print(f"\nExecution Complete! Output file ready at: '{FILENAME}'")
