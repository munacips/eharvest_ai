import csv
import random
from datetime import datetime, timedelta

# 1. Ten Provinces of Zimbabwe
REGIONS = [
    "Harare", "Bulawayo", "Manicaland", "Midlands", "Masvingo", 
    "Mashonaland Central", "Mashonaland West", "Mashonaland East", 
    "Matabeleland South", "Matabeleland North"
]

# 2. Fully mapped Commodity Catalog
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

# 3. Supplier Profiles and their production capacities
SUPPLIER_PROFILES = [
    {"type": "Smallholder", "base_min": 500.0, "base_max": 8000.0},
    {"type": "A1 Commercial", "base_min": 10000.0, "base_max": 25000.0},
    {"type": "A2 Commercial", "base_min": 30000.0, "base_max": 75000.0},
    {"type": "Large-scale Commercial", "base_min": 80000.0, "base_max": 150000.0}
]

def get_zimbabwe_season(month):
    # May to October is the cool-to-warm dry season (primary harvest supply window)
    if 5 <= month <= 10:
        return "harvest/dry"
    # November to April is the primary rainy summer growing season
    return "summer"

# 4. Timeline Setup (Jan 1, 2022 to May 22, 2026)
start_date = datetime(2022, 1, 1)
end_date = datetime(2026, 5, 22)
delta_days = (end_date - start_date).days

FILENAME = "table3_supply_records.csv"
TOTAL_RECORDS = 1000000

print(f"Generating {TOTAL_RECORDS:,} rows for Table 3 — supply_records...")
print(f"Output File: {FILENAME}")

# 5. Direct disk stream generation
with open(FILENAME, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Matching your exact schema footprint layout
    writer.writerow(["date", "commodity", "region", "quantity_kg", "supplier_type", "season"])
    
    for i in range(TOTAL_RECORDS):
        # Programmatic timeline logic
        random_days = random.randint(0, delta_days)
        current_date = start_date + timedelta(days=random_days)
        date_str = current_date.strftime("%Y-%m-%d")
        
        season = get_zimbabwe_season(current_date.month)
        commodity = random.choice(COMMODITIES)
        region = random.choice(REGIONS)
        supplier = random.choice(SUPPLIER_PROFILES)
        
        supplier_type = supplier["type"]
        
        # Calculate dynamic localized output volume weights
        min_qty = supplier["base_min"]
        max_qty = supplier["base_max"]
        
        # Inject realistic agricultural trends: 
        # Staple grain supply volumes spike hard during dry harvest months
        if season == "harvest/dry" and commodity in ["Maize", "Sorghum", "Millet (finger millet)"]:
            min_qty *= 1.5
            max_qty *= 2.0
        # Conversely, supply drops slightly during the peak of rainy summer before harvest
        elif season == "summer" and commodity in ["Maize", "Sorghum"]:
            min_qty *= 0.6
            max_qty *= 0.7
            
        # Miscellaneous specialized items (like garlic or eggs) scale logically lower in total mass
        if commodity in ["Garlic", "Eggs (tray of 30)", "Salt", "Cucumber"]:
            min_qty *= 0.1
            max_qty *= 0.15
            
        quantity_kg = round(random.uniform(min_qty, max_qty), 1)
        
        # Write flat dataset row array
        writer.writerow([
            date_str,
            commodity,
            region,
            quantity_kg,
            supplier_type,
            season
        ])
        
        # Console output checkpoint intervals
        if (i + 1) % 250000 == 0:
            print(f"Database Stream: {i + 1:,} records successfully written...")

print(f"\nSuccess! File cleanly written to '{FILENAME}'. Ready for ingestion testing.")