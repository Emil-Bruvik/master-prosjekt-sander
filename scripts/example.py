import pandas as pd
from wind_production import MonthSpec, WindCalculator
import config

ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5"]
LOCATION_COLS_WIND = ["City", "State/Province", "Local area (taluk, county)", "Subregion", "Region"]


def main():
	calculator = WindCalculator()
	monthly_results = []
	months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

	for month_number, month_name in enumerate(months, start=1):
		month = MonthSpec("2025", f"{month_number:02d}", month_name)
		weather = calculator.open_weather(month)
		try:
			indexer = calculator.GridIndexer(
				weather["latitude"].values,
				weather["longitude"].values,
			)
			timestamps = pd.to_datetime(weather["valid_time"].values) - pd.Timedelta(hours=1)
			month_result = pd.DataFrame(index=timestamps)
			for zone in ZONES:
				_, capacity_2026 = calculator.country_timeseries(
					month,
					weather,
					indexer,
					ZONES,
					LOCATION_COLS_WIND,
					"Norway",
					zone,
					return_farm_data=False,
				)
				month_result[zone] = capacity_2026
			monthly_results.append(month_result)
		finally:
			weather.close()

	result = pd.concat(monthly_results).sort_index()
	result.index.name = "time"
	config.WIND_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(config.WIND_OUTPUT_CSV)
	print(f"Wrote {len(result)} hourly records to {config.WIND_OUTPUT_CSV}")


if __name__ == "__main__":
	main()

    
