import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.physics.atmosphere import calculate_sky_conditions

def validate_location(weather_file, location_name):
    print(f"Loading weather data for {location_name}...")
    try:
        df = pd.read_csv(weather_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {weather_file}")
        return

    temp_col = 'temp_air'
    rh_col = 'relative_humidity'
    
    if temp_col not in df.columns or rh_col not in df.columns:
        print(f"❌ Error: Cannot find columns. Available: {df.columns}")
        return

    print("Data loaded. Calculating Sky Temperatures...")
    sky_temps = []
    
    for index, row in df.iterrows():
        t_amb = row[temp_col]
        rh = row[rh_col]
        if rh > 1.0:
            rh = rh / 100.0
        cloud_cover = 0.0 
        result = calculate_sky_conditions(t_amb, rh, cloud_cover)
        sky_temps.append(result['T_sky_C'])

    df['T_sky'] = sky_temps

    start_idx = 4000
    end_idx = 4072
    subset = df.iloc[start_idx:end_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(subset.index, subset[temp_col], label='Ambient Temp (°C)', color='red', linewidth=2)
    plt.plot(subset.index, subset['T_sky'], label='Sky Temp (°C)', color='blue', linestyle='--', linewidth=2)
    plt.fill_between(subset.index, subset['T_sky'], subset[temp_col], color='cyan', alpha=0.1, label='Cooling Potential')
    plt.title(f"{location_name}: Ambient vs. Sky Temperature (Berdahl-Martin)")
    plt.ylabel("Temperature (°C)")
    plt.xlabel("Hour of Year")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"figures/validation_{location_name.lower()}.png"
    plt.savefig(output_file)
    print(f"✅ Validation plot saved: {output_file}")

if __name__ == "__main__":
    validate_location("data/weather/phoenix_az_tmy.csv", "Phoenix")
