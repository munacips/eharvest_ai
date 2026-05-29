import csv
import random
from datetime import datetime, timedelta

# 1. Ten Provinces of Zimbabwe with distinct climate profile templates
# Lowveld regions are hotter/drier; Highveld/Manicaland are wetter/cooler
REGION_CLIMATES = {
    "Harare":              {"rain_mult": 1.1, "temp_base": 26.0, "lowveld": False},
    "Bulawayo":            {"rain_mult": 0.8, "temp_base": 27.0, "lowveld": False},
    "Manicaland":          {"rain_mult": 1.4, "temp_base": 25.0, "lowveld": False}, # Eastern Highlands influence
    "Midlands":            {"rain_mult": 0.9, "temp_base": 28.0, "lowveld": False},
    "Masvingo":            {"rain_mult": 0.7, "temp_base": 31.0, "lowveld": True},  # Lowveld characteristics
    "Mashonaland Central": {"rain_mult": 1.2, "temp_base": 28.0, "lowveld": False},
    "Mashonaland West":    {"rain_mult": 1.1, "temp_base": 29.0, "lowveld": False},
    "Mashonaland East":    {"rain_mult": 1.2, "temp_base": 26.0, "lowveld": False},
    "Matabeleland South":  {"rain_mult": 0.6, "temp_base": 32.0, "lowveld": True},  # Hot & Arid
    "Matabeleland North":  {"rain_mult": 0.8, "temp_base": 30.0, "lowveld": True}
}

# 2. Timeline Configuration (Jan 1, 2022 to May 22, 2026)
start_date = datetime(2022, 1, 1)
end_date = datetime(2026, 5, 22)
delta_days = (end_date - start_date).days

FILENAME = "table4_weather_observations.csv"
TOTAL_RECORDS = 1000000

print(f"Generating {TOTAL_RECORDS:,} rows for Table 4 — weather_observations...")
print(f"Destination: {FILENAME}")

# 3. Generate data streams directly to local disk storage
with open(FILENAME, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Matching your exact schema blueprint layout fields
    writer.writerow(["date", "region", "rainfall_mm", "temp_max_c", "temp_min_c", "drought_index", "season"])
    
    regions_list = list(REGION_CLIMATES.keys())
    
    for i in range(TOTAL_RECORDS):
        # Pick a random date
        random_days = random.randint(0, delta_days)
        current_date = start_date + timedelta(days=random_days)
        date_str = current_date.strftime("%Y-%m-%d")
        month = current_date.month
        
        # Determine strict agricultural season grouping
        # Summer (Wet): Nov to Apr | Harvest/Dry: May to Oct
        if 5 <= month <= 10:
            season = "harvest/dry"
        else:
            season = "summer"
            
        # Select region and fetch its unique climate constraints
        region = random.choice(regions_list)
        geo = REGION_CLIMATES[region]
        
        # Base Temperature and Rain allocations conditioned strictly by Month
        if season == "summer":
            # High chance of rain storms during mid-summer months (Dec, Jan, Feb)
            if month in [12, 1, 2]:
                rain_chance = 0.65
                base_rain = random.uniform(5.0, 45.0)
            else: # Nov, Mar, Apr transition months
                rain_chance = 0.35
                base_rain = random.uniform(1.0, 15.0)
                
            rainfall = round(base_rain * geo["rain_mult"], 1) if random.random() < rain_chance else 0.0
            
            # Hot summer temperatures
            t_max = geo["temp_base"] + random.uniform(-3.0, 5.0)
            t_min = t_max - random.uniform(9.0, 14.0)
            
        else: # harvest/dry season (Cool winters / Hot spring in Sept-Oct)
            if month in [6, 7]: # Peak winter months
                t_max = (geo["temp_base"] - 5.0) + random.uniform(-2.0, 3.0)
                t_min = t_max - random.uniform(11.0, 16.0) # Cold clear nights
                rainfall = round(random.uniform(0.1, 2.0), 1) if random.random() < 0.03 else 0.0
            elif month in [9, 10]: # "Suicide month" spring heat spikes before rains
                t_max = (geo["temp_base"] + 4.0) + random.uniform(0.0, 6.0)
                t_min = t_max - random.uniform(10.0, 15.0)
                rainfall = round(random.uniform(0.5, 6.0), 1) if random.random() < 0.08 else 0.0
            else: # May, August
                t_max = geo["temp_base"] + random.uniform(-3.0, 2.0)
                t_min = t_max - random.uniform(10.0, 15.0)
                rainfall = 0.0
        
        # Adjust Lowveld region temperature ceilings upwards
        if geo["lowveld"]:
            t_max += random.uniform(2.0, 4.5)
            t_min += random.uniform(1.0, 3.0)
            
        # Generate Drought Index matching Standardized Precipitation Index rules (-3.0 to +3.0)
        # Correlate negative drought numbers with zero rainfall and soaring temperatures
        if rainfall == 0.0:
            if t_max > (geo["temp_base"] + 2):
                drought_index = round(random.uniform(-3.0, -1.1), 2)  # Severe to Moderate Dryness
            else:
                drought_index = round(random.uniform(-1.0, 0.2), 2)   # Near Normal Dryness
        else:
            if rainfall > 25.0:
                drought_index = round(random.uniform(1.5, 3.0), 2)    # Highly Wet
            else:
                drought_index = round(random.uniform(-0.5, 1.4), 2)   # Normal to Mild Wetness
                
        # Round off temperatures smoothly
        temp_max = round(t_max, 1)
        temp_min = round(t_min, 1)
        
        # Write clean record matrix list row
        writer.writerow([
            date_str,
            region,
            rainfall,
            temp_max,
            temp_min,
            drought_index,
            season
        ])
        
        # Operational loop feedback log milestones
        if (i + 1) % 250000 == 0:
            print(f"Database Core: Completed {i + 1:,} weather row block arrays...")

print(f"\nCompleted! Generated observation dataset matrix ready at: '{FILENAME}'")