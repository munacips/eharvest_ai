import csv
import random
from datetime import datetime, timedelta

# 1. Expanded Geographic Database covering all 10 Provinces of Zimbabwe
MARKET_DATA = [
    # Metropolitan Provinces
    {"region": "Harare", "market": "Mbare Musika"},
    {"region": "Harare", "market": "Jambanja Market (Chitungwiza)"},
    {"region": "Harare", "market": "Lusaka Market (Highfield)"},
    {"region": "Bulawayo", "market": "Renkini Market"},
    {"region": "Bulawayo", "market": "Shashe Market"},
    
    # Manicaland
    {"region": "Manicaland", "market": "Sakubva Market (Mutare)"},
    {"region": "Manicaland", "market": "Chipinge Town Market"},
    {"region": "Manicaland", "market": "Rusape Open Market"},
    
    # Midlands
    {"region": "Midlands", "market": "Kombayi Market (Gweru)"},
    {"region": "Midlands", "market": "Kwekwe Central Market"},
    {"region": "Midlands", "market": "Gokwe Center Market"},
    
    # Masvingo
    {"region": "Masvingo", "market": "Chitima Market (Masvingo City)"},
    {"region": "Masvingo", "market": "Chiredzi Open Market"},
    
    # Mashonaland Central
    {"region": "Mashonaland Central", "market": "Mazowe Market"},
    {"region": "Mashonaland Central", "market": "Bindura Tsungubvi Market"},
    
    # Mashonaland West
    {"region": "Mashonaland West", "market": "Chinhoyi Ombva Market"},
    {"region": "Mashonaland West", "market": "Kadoma Central Market"},
    {"region": "Mashonaland West", "market": "Karoi Agriculture Market"},
    
    # Mashonaland East
    {"region": "Mashonaland East", "market": "Marondera Chikwanha Market"},
    {"region": "Mashonaland East", "market": "Murehwa Center Market"},
    {"region": "Mashonaland East", "market": "Mutoko Vegetable Market"},
    
    # Matabeleland South
    {"region": "Matabeleland South", "market": "Gwanda Town Market"},
    {"region": "Matabeleland South", "market": "Beitbridge Border Market"},
    
    # Matabeleland North
    {"region": "Matabeleland North", "market": "Lupane Center Market"},
    {"region": "Matabeleland North", "market": "Hwange Open Market"},
    {"region": "Matabeleland North", "market": "Chinamora Market (Victoria Falls)"}
]

# 2. Comprehensive Commodity Catalog with targeted category, unit and baseline USD distributions
COMMODITIES = [
    # Cereals/Grains
    {"commodity": "Maize", "category": "cereals and tubers", "unit": "KG", "min_p": 0.25, "max_p": 0.50},
    {"commodity": "Maize meal", "category": "cereals and tubers", "unit": "10KG", "min_p": 4.50, "max_p": 7.50},
    {"commodity": "Rice", "category": "cereals and tubers", "unit": "KG", "min_p": 0.90, "max_p": 1.50},
    {"commodity": "Sorghum", "category": "cereals and tubers", "unit": "KG", "min_p": 0.40, "max_p": 0.80},
    {"commodity": "Millet (finger millet)", "category": "cereals and tubers", "unit": "KG", "min_p": 0.50, "max_p": 0.90},
    {"commodity": "Wheat", "category": "cereals and tubers", "unit": "KG", "min_p": 0.35, "max_p": 0.65},
    {"commodity": "Barley", "category": "cereals and tubers", "unit": "KG", "min_p": 0.30, "max_p": 0.60},
    
    # Pulses/Legumes
    {"commodity": "Beans (sugar)", "category": "pulses and legumes", "unit": "KG", "min_p": 1.20, "max_p": 2.20},
    {"commodity": "Beans (round)", "category": "pulses and legumes", "unit": "KG", "min_p": 1.50, "max_p": 2.50},
    {"commodity": "Groundnuts (shelled)", "category": "pulses and legumes", "unit": "KG", "min_p": 1.00, "max_p": 1.80},
    {"commodity": "Soybeans", "category": "pulses and legumes", "unit": "KG", "min_p": 0.40, "max_p": 0.75},
    {"commodity": "Cowpeas", "category": "pulses and legumes", "unit": "KG", "min_p": 0.60, "max_p": 1.20},
    {"commodity": "Pigeon peas", "category": "pulses and legumes", "unit": "KG", "min_p": 0.70, "max_p": 1.40},
    
    # Vegetables
    {"commodity": "Tomatoes", "category": "vegetables", "unit": "Crate", "min_p": 5.00, "max_p": 15.00},
    {"commodity": "Onions", "category": "vegetables", "unit": "Pocket", "min_p": 3.50, "max_p": 8.00},
    {"commodity": "Carrots", "category": "vegetables", "unit": "Bundle", "min_p": 0.50, "max_p": 1.50},
    {"commodity": "Cabbages", "category": "vegetables", "unit": "Head", "min_p": 0.50, "max_p": 1.20},
    {"commodity": "Spinach (rape)", "category": "vegetables", "unit": "Bundle", "min_p": 0.30, "max_p": 0.80},
    {"commodity": "Sweet potatoes", "category": "vegetables", "unit": "Bucket", "min_p": 2.50, "max_p": 6.00},
    {"commodity": "Irish potatoes", "category": "vegetables", "unit": "Pocket", "min_p": 5.00, "max_p": 11.00},
    {"commodity": "Butternuts", "category": "vegetables", "unit": "Pocket", "min_p": 3.00, "max_p": 7.00},
    {"commodity": "Pumpkin", "category": "vegetables", "unit": "Unit", "min_p": 1.00, "max_p": 3.00},
    {"commodity": "Green pepper", "category": "vegetables", "unit": "KG", "min_p": 0.80, "max_p": 2.00},
    {"commodity": "Garlic", "category": "vegetables", "unit": "KG", "min_p": 2.50, "max_p": 5.00},
    {"commodity": "Okra (derere)", "category": "vegetables", "unit": "KG", "min_p": 0.60, "max_p": 1.50},
    {"commodity": "Cucumber", "category": "vegetables", "unit": "Unit", "min_p": 0.20, "max_p": 0.50},
    
    # Fruits
    {"commodity": "Bananas", "category": "fruits", "unit": "Crate", "min_p": 7.00, "max_p": 14.00},
    {"commodity": "Mangoes", "category": "fruits", "unit": "Bucket", "min_p": 2.00, "max_p": 5.00},
    {"commodity": "Avocados", "category": "fruits", "unit": "Unit", "min_p": 0.20, "max_p": 0.60},
    {"commodity": "Oranges", "category": "fruits", "unit": "Pocket", "min_p": 3.00, "max_p": 6.50},
    
    # Oils/Fats
    {"commodity": "Oil (vegetable)", "category": "oils and fats", "unit": "2Litre", "min_p": 3.20, "max_p": 4.50},
    {"commodity": "Cooking oil (sunflower)", "category": "oils and fats", "unit": "2Litre", "min_p": 3.50, "max_p": 5.00},
    
    # Protein
    {"commodity": "Beef", "category": "protein", "unit": "KG", "min_p": 4.00, "max_p": 7.00},
    {"commodity": "Fish (kapenta)", "category": "protein", "unit": "KG", "min_p": 5.00, "max_p": 9.00},
    {"commodity": "Eggs (tray of 30)", "category": "protein", "unit": "Tray", "min_p": 3.50, "max_p": 5.50},
    {"commodity": "Chicken", "category": "protein", "unit": "KG", "min_p": 3.50, "max_p": 5.50},
    
    # Misc
    {"commodity": "Salt", "category": "misc", "unit": "KG", "min_p": 0.40, "max_p": 0.80},
    {"commodity": "Sugar", "category": "misc", "unit": "2KG", "min_p": 2.20, "max_p": 3.50},
    {"commodity": "Flour (wheat)", "category": "misc", "unit": "2KG", "min_p": 2.00, "max_p": 3.20}
]

# 3. Dynamic Timeline Configuration (Jan 1, 2022 up to current data scope May 2026)
start_date = datetime(2022, 1, 1)
end_date = datetime(2026, 5, 22)
delta_days = (end_date - start_date).days

FILENAME = "zimbabwe_provincial_market_data.csv"
TOTAL_RECORDS = 1000000

print(f"Starting pipeline to generate {TOTAL_RECORDS:,} rows...")
print(f"Target file destination: {FILENAME}")

# 4. Stream and Output directly to the local storage target
with open(FILENAME, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Exact header footprint match
    writer.writerow(["date", "commodity", "category", "region", "market", "price_usd", "unit", "price_type", "data_source"])
    
    for i in range(TOTAL_RECORDS):
        # Calculate timeline progression distribution
        random_days = random.randint(0, delta_days)
        record_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # Select entities from complete geo and commodity lists
        geo = random.choice(MARKET_DATA)
        item = random.choice(COMMODITIES)
        
        # Generate base prices across custom commodity variances
        price = round(random.uniform(item["min_p"], item["max_p"]), 2)
        
        # Factor realistic Retail vs Wholesale variances (Wholesale features a ~15% markdown index)
        price_type = "Retail" if random.random() > 0.25 else "Wholesale"
        if price_type == "Wholesale":
            price = round(price * 0.85, 2)
            
        writer.writerow([
            record_date,
            item["commodity"],
            item["category"],
            geo["region"],
            geo["market"],
            price,
            item["unit"],
            price_type,
            "synthetic"
        ])
        
        # Operational loop log milestones
        if (i + 1) % 250000 == 0:
            print(f"Successfully processed and written {i + 1:,} rows...")

print(f"\nCompleted! Generated clean execution log dataset file saved at: '{FILENAME}'")