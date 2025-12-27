import os
import time
import pandas as pd
from pvlib.iotools import get_pvgis_tmy

locations = {
    "Phoenix_AZ": (33.4484, -112.0740),
    "Houston_TX": (29.7604, -95.3698),
    "Seattle_WA": (47.6062, -122.3321)
}

output_dir = "data/weather"
os.makedirs(output_dir, exist_ok=True)

print("--- Day 1 Task: Acquiring TMY3 Data ---")

for name, (lat, lon) in locations.items():
    print(f"Fetching data for {name}...")
    try:
        result = get_pvgis_tmy(lat, lon, outputformat='csv')
        data = result[0]
        filename = f"{output_dir}/{name.lower()}_tmy.csv"
        data.to_csv(filename)
        print(f"✅ Saved: {filename}")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error fetching {name}: {e}")

print("--- TMY3 Data Acquisition Complete ---")
