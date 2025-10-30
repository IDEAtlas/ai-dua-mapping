# import geopandas as gpd
# import pygeos
# import numpy as np
# from libpysal.weights import Queen


# def _pysal_blocks(tessellation, edges, buildings, id_name='bID', unique_id='uID'):
#     cut = gpd.overlay(tessellation, gpd.GeoDataFrame(geometry=edges.buffer(0.001)), how='difference').explode()
#     W = Queen.from_dataframe(cut, silence_warnings=True)
#     cut['component'] = W.component_labels
#     buildings_c = buildings.copy()
#     buildings_c["geometry"] = buildings_c.representative_point()  # make points
#     centroids_tempID = gpd.sjoin(
#         buildings_c, cut[['geometry', 'component']], how="left", op="intersects"
#     )
#     cells_copy = tessellation[[unique_id, "geometry"]].merge(centroids_tempID[[unique_id, 'component']], on='uID', how="left")
#     blocks = cells_copy.dissolve(by='component').explode().reset_index(drop=True)
#     blocks[id_name] = range(len(blocks))
#     blocks["geometry"] = gpd.GeoSeries(pygeos.polygons(blocks.exterior.values.data), crs=blocks.crs)
#     blocks = blocks[[id_name, 'geometry']]
#     # if polygon is within another one, delete it
#     inp, res = blocks.sindex.query_bulk(blocks.geometry, predicate="within")
#     inp = inp[~(inp == res)]
#     mask = np.ones(len(blocks.index), dtype=bool)
#     mask[inp] = False
#     blocks = blocks.loc[mask, [id_name, "geometry"]]

#     centroids_w_bl_ID2 = gpd.sjoin(
#         buildings_c, blocks, how="left", op="intersects"
#     )
#     bl_ID_to_uID = centroids_w_bl_ID2[[unique_id, id_name]]

#     buildings_m = buildings[[unique_id]].merge(
#         bl_ID_to_uID, on=unique_id, how="left"
#     )
#     buildings_id = buildings_m[id_name]

#     cells_m = tessellation[[unique_id]].merge(
#         bl_ID_to_uID, on=unique_id, how="left"
#     )
#     tessellation_id = cells_m[id_name]
#     return (blocks, buildings_id, tessellation_id)

# def fill_insides(df):
#     """
#     Remove faulty polygons inside other. Close gaps.
    
#     requires pygeos and geopandas 0.8+
#     """
#     polys = pygeos.polygons(pygeos.get_exterior_ring(df.geometry.values.data))
#     inp, res = pygeos.STRtree(polys).query_bulk(polys, predicate="contains_properly")
#     cleaner = np.delete(polys, res)
#     return gpd.GeoSeries(cleaner, crs=df.crs)


import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
from libpysal.weights import Queen
import pygeos

def fill_insides(df):
    """
    Remove faulty polygons inside other. Close gaps.
    
    requires shapely and geopandas 0.8+
    """
    # Convert to PyGEOS geometries
    polys = pygeos.from_shapely([Polygon(poly.exterior) for poly in df.geometry])
    tree = pygeos.STRtree(polys)
    inp, res = tree.query_bulk(polys, predicate="contains_properly")
    cleaner = np.delete(polys, res)
    # Convert back to Shapely geometries
    cleaner_shapely = pygeos.to_shapely(cleaner)
    return gpd.GeoSeries(cleaner_shapely, crs=df.crs)


# from shapely.geometry import Polygon
# import geopandas as gpd
# import numpy as np

# def fill_insides(df):
#     """
#     Remove faulty polygons inside other. Close gaps.
#     Shapely-only implementation to avoid PyGEOS version conflicts.
#     """
#     # Create exterior polygons
#     polys = [Polygon(poly.exterior) for poly in df.geometry]
    
#     # Create a spatial index using rtree if available, otherwise fall back to brute force
#     try:
#         import rtree
#         # Build spatial index
#         idx = rtree.index.Index()
#         for i, geometry in enumerate(polys):
#             idx.insert(i, geometry.bounds)
        
#         to_remove = set()
#         for i, poly in enumerate(polys):
#             # Find potential candidates using spatial index
#             candidates = list(idx.intersection(poly.bounds))
#             for j in candidates:
#                 if i != j and poly.contains_properly(polys[j]):
#                     to_remove.add(j)
        
#         # Remove contained polygons
#         mask = [i not in to_remove for i in range(len(polys))]
#         cleaner = [polys[i] for i in range(len(polys)) if mask[i]]
        
#     except ImportError:
#         # Fallback: brute force method (slower but works without rtree)
#         logging.warning("rtree not available, using slower method for spatial queries")
#         to_remove = set()
#         for i in range(len(polys)):
#             for j in range(len(polys)):
#                 if i != j and polys[i].contains_properly(polys[j]):
#                     to_remove.add(j)
        
#         cleaner = [polys[i] for i in range(len(polys)) if i not in to_remove]
    
#     return gpd.GeoSeries(cleaner, crs=df.crs)