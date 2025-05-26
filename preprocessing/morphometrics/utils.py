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

def _pysal_blocks(tessellation, edges, buildings, id_name='bID', unique_id='uID'):
    cut = gpd.overlay(tessellation, gpd.GeoDataFrame(geometry=edges.buffer(0.001)), how='difference').explode()
    W = Queen.from_dataframe(cut, silence_warnings=True)
    cut['component'] = W.component_labels
    buildings_c = buildings.copy()
    buildings_c["geometry"] = buildings_c.representative_point()  # make points
    centroids_tempID = gpd.sjoin(
        buildings_c, cut[['geometry', 'component']], how="left", op="intersects"
    )
    cells_copy = tessellation[[unique_id, "geometry"]].merge(centroids_tempID[[unique_id, 'component']], on='uID', how="left")
    blocks = cells_copy.dissolve(by='component').explode().reset_index(drop=True)
    blocks[id_name] = range(len(blocks))
    blocks["geometry"] = gpd.GeoSeries([Polygon(block.exterior) for block in blocks.geometry], crs=blocks.crs)
    blocks = blocks[[id_name, 'geometry']]
    # if polygon is within another one, delete it
    inp, res = blocks.sindex.query_bulk(blocks.geometry, predicate="within")
    inp = inp[~(inp == res)]
    mask = np.ones(len(blocks.index), dtype=bool)
    mask[inp] = False
    blocks = blocks.loc[mask, [id_name, "geometry"]]

    centroids_w_bl_ID2 = gpd.sjoin(
        buildings_c, blocks, how="left", op="intersects"
    )
    bl_ID_to_uID = centroids_w_bl_ID2[[unique_id, id_name]]

    buildings_m = buildings[[unique_id]].merge(
        bl_ID_to_uID, on=unique_id, how="left"
    )
    buildings_id = buildings_m[id_name]

    cells_m = tessellation[[unique_id]].merge(
        bl_ID_to_uID, on=unique_id, how="left"
    )
    tessellation_id = cells_m[id_name]
    return (blocks, buildings_id, tessellation_id)

def fill_insides(df):
    """
    Remove faulty polygons inside other. Close gaps.
    
    requires shapely and geopandas 0.8+
    """
    polys = [Polygon(poly.exterior) for poly in df.geometry]
    tree = STRtree(polys)
    inp, res = tree.query_bulk(polys, predicate="contains_properly")
    cleaner = np.delete(polys, res)
    return gpd.GeoSeries(cleaner, crs=df.crs)
