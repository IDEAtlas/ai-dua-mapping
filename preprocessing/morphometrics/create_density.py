import numpy as np
from osgeo import gdal, ogr
import os

def georrefData(data, filename, metadata):
	dataset = gdal.Open(metadata)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(filename, width, height, 1,gdal.GDT_Float32)
	outdata.SetGeoTransform(dataset.GetGeoTransform())
	outdata.SetProjection(dataset.GetProjection())
	outdata.GetRasterBand(1).WriteArray(data)
	outdata.FlushCache()
	outdata = None
	band = None
	ds = None


def CreateDensity(sentinelImg, DensityShape, saveadd):
    """
    Create building density raster from sentinel image and building shapefile.
    """
    # Open sentinel image
    sentinel_img = gdal.Open(sentinelImg)
    band = sentinel_img.GetRasterBand(1)  # bands start at one
    b = band.ReadAsArray().astype(np.float32)
    print(b.shape)
    density = np.zeros_like(b)
    
    # Get geotransform parameters
    geoTransform = sentinel_img.GetGeoTransform()
    x_0 = geoTransform[0]
    y_0 = geoTransform[3]
    delta_x = geoTransform[1]
    delta_y = geoTransform[5]
    
    # Open shapefile
    file = ogr.Open(DensityShape)
    shape = file.GetLayer(0)

    # Loop through all features
    total_features = shape.GetFeatureCount()
    print(f"Total features: {total_features}")
    
    for feature in shape:
        geom = feature.GetGeometryRef()
        if geom is not None:
            a = geom.GetEnvelope()
            i_min = int(np.floor((a[0]-x_0)/delta_x))
            i_max = int(np.ceil((a[1]-x_0)/delta_x))
            j_max = int(np.ceil((a[2]-y_0)/delta_y))
            j_min = int(np.floor((a[3]-y_0)/delta_y))
                
            # Add building density to pixels
            for i in range(i_min-1, i_max+1):
                for j in range(j_min-1, j_max+1):
                    if i>0 and i<b.shape[1] and j>0 and j<b.shape[0]:
                        density[j,i] += 1
    
    # Scale the density raster to 0-1 for deep learning training
    density_min = np.min(density)
    density_max = np.max(density)
    density_range = density_max - density_min
    
    if density_range > 0:
        density = (density - density_min) / density_range
    else:
        density = np.zeros_like(density)
    
    print(f"Density range: {density_min:.2f} - {density_max:.2f}, normalized to [0, 1]")
    georrefData(density, os.path.join(saveadd), sentinelImg)
    print(f"Density raster saved to {saveadd}")

if __name__ == "__main__":
    # Inputs: 
    # 1) Raster that will be the basis (the one that will be used to georeference the density raster)
    # 2) The buildings/morphometrics shapefile
    # 3) address to save the rasters with the morphometrics

    city = 'jakarta'
    basedir = '/data/raw/'
    year = 2023

    CreateDensity(os.path.join(basedir, "sentinel", city, f'S2_{year}.tif'), 
                  os.path.join(basedir, f'buildings/{city}_bldg.gpkg'), 
                  os.path.join(basedir, f'buildings/density/{city}_bd.tif'))