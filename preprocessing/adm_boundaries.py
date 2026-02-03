import requests
import os
import json
import logging
import time

logger = logging.getLogger(__name__)

def fetch_and_save_adm_borders_geojson(q, output_folder, output_file=None, replace=False) -> str:
    try:
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "polygon_geojson": 1, "limit": 5},
            headers=headers,
            timeout=10
        )
        r.raise_for_status()
        res = r.json()

        if not output_file:
            output_file = f"{q.replace(', ', '_').replace(' ', '_').lower()}_aoi.geojson"

        output_path = os.path.join(output_folder, output_file)
        
        if os.path.exists(output_path) and not replace:
            return output_path
        
        if not res:
            raise ValueError(f"No results found for query: {q}")
        
        # Filter through results to find a Polygon/MultiPolygon (skip Points)
        geojson = None
        for result in res:
            candidate = result.get("geojson")
            if candidate and candidate.get("type") in ["Polygon", "MultiPolygon"]:
                geojson = candidate
                break
        
        if geojson is None:
            # Extract city name from query for the error message
            city_part = q.split(',')[0].strip().lower().replace(" ", "_").replace("-", "_")
            raise ValueError(
                f"Unable to fetch city boundary. Provide your own AOI in data/raw/aoi/{city_part}_aoi.geojson with wgs84 projection"
            )
        
        os.makedirs(output_folder, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

        return output_path
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch AOI boundaries: {str(e)[:100]}")
    except Exception as e:
        raise RuntimeError(f"AOI processing failed: {str(e)[:100]}")

# Example usage
# fetch_and_save_adm_borders_geojson("Tegucigalpa, Honduras", "/data/raw/aoi/")