import requests
import os
import json

def fetch_and_save_adm_borders_geojson(q, output_folder, output_file=None):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": q, "format": "json", "polygon_geojson": 1, "limit": 1},
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()
    res = r.json()

    if not output_file:
        output_file = f"{q.replace(', ', '_').replace(' ', '_').lower()}_aoi.geojson"

    if res:
        geojson = res[0].get("geojson")
        print(geojson)  # print to console
        # Save to file
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)
        print(f"Saved to {output_path}")
    else:
        print("Not found")

# Example usage
# fetch_and_save_adm_borders_geojson("Tegucigalpa, Honduras", "/data/raw/aoi/")