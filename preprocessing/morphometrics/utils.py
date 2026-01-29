import geopandas as gpd
import rtree
from shapely.geometry import Polygon

def fill_insides(df):
    """
    Remove faulty polygons inside other. Close gaps.
    Shapely-only implementation to avoid PyGEOS version conflicts.
    """
    # Create exterior polygons
    polys = [Polygon(poly.exterior) for poly in df.geometry]

    # Build spatial index
    idx = rtree.index.Index()
    for i, geometry in enumerate(polys):
        idx.insert(i, geometry.bounds)

    to_remove = set()
    for i, poly in enumerate(polys):
        # Find potential candidates using spatial index
        candidates = list(idx.intersection(poly.bounds))
        for j in candidates:
            if i != j and poly.contains_properly(polys[j]):
                to_remove.add(j)

    # Remove contained polygons
    mask = [i not in to_remove for i in range(len(polys))]
    cleaner = [polys[i] for i in range(len(polys)) if mask[i]]
    return gpd.GeoSeries(cleaner, crs=df.crs)