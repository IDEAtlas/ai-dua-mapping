import numpy as np
from osgeo import gdal, ogr, osr
from glob import glob
import os
import tifffile
import time
import geopandas as gpd


def georrefData(data, filename, metadata):
	dataset = gdal.Open(metadata)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(filename, width, height, 1,gdal.GDT_Float32)
	outdata.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(data)
	outdata.FlushCache()  ##saves to disk!!
	outdata = None
	band = None
	ds = None


def load_img(imgPath):
	'''
	Load image
	:param imgPath: path of the image to load
	:return: numpy array of the image
	'''
	if imgPath.endswith('.tif'):
		img = tifffile.imread(imgPath)
	else:
		img = np.array(cv2.imread(imgPath))
	return img

def CreateDensity(sentinelImg, DensityShape, saveadd):
    #open sentinel image:
    sentinel_img = gdal.Open(sentinelImg)
    band = sentinel_img.GetRasterBand(1)  # bands start at one
    b = band.ReadAsArray().astype(np.float32)
    print(b.shape)
    density = np.zeros_like(b)
    # mean_area = np.zeros_like(b)
    # mean_perimeter = np.zeros_like(b)
    # mean_ssbCCo = np.zeros_like(b)
    # mean_ssbCor = np.zeros_like(b)
    # mean_ssbSqu = np.zeros_like(b)
    # mean_ssbERI = np.zeros_like(b)
    # mean_ssbElo = np.zeros_like(b)
    # mean_stbOri = np.zeros_like(b)
    # mean_stbCeA = np.zeros_like(b)
    # mean_stcOri = np.zeros_like(b)
    # mean_sdcLAL = np.zeros_like(b)
    # mean_sdcAre = np.zeros_like(b)
    # mean_sscCCo = np.zeros_like(b)
    # mean_sscERI = np.zeros_like(b)
    # mean_sicCAR = np.zeros_like(b)
    # std_stbOri = np.zeros_like(b, dtype=object)
    # mean_percentBuild = np.zeros_like(b)
    
    geoTransform = sentinel_img.GetGeoTransform()
    x_0 = geoTransform[0]
    y_0 = geoTransform[3]
    delta_x = geoTransform[1]
    delta_y = geoTransform[5]
    # print(x_0)
    # print(y_0)
    # print(delta_x)
    # print(delta_y)
    
    #open shape_file
    # Open the shapefile
    
    file = ogr.Open(DensityShape)
    shape = file.GetLayer(0)

    # Define the spatial reference for EPSG:4326 (WGS 84)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    # Get the spatial reference of the shapefile (input SRS)
    source_srs = shape.GetSpatialRef()

    # Create a coordinate transformation
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Loop through all features and apply the transformation
    total_features = shape.GetFeatureCount()
    print(f"Total features: {total_features}")
    count = 1
    for feature in shape:
        geom = feature.GetGeometryRef()
        #geom = feature.GetField(4)
        # print(geom)
        #geom = ogr.CreateGeometryFromWkt(geom)
        if geom is not None:
            a = geom.GetEnvelope()
            i_min = int(np.floor((a[0]-x_0)/delta_x))
            i_max = int(np.ceil((a[1]-x_0)/delta_x))
            j_max = int(np.ceil((a[2]-y_0)/delta_y))
            j_min = int(np.floor((a[3]-y_0)/delta_y))
            del_x = i_max-i_min+1
            del_y = j_max-j_min+1
            areaIndegree = geom.GetArea()
            percent = areaIndegree/(del_x*del_y*delta_x*delta_x)
            # print(count)
            count = count+1
            #print(del_x*del_y*delta_x*delta_x)
            #print(percent)
            if(percent>1):
                print('Error')
                return 0
            # print("minX: %d, maxX: %d, minY: %d, maxY: %d" % (i_min, i_max, j_min, j_max))
            for i in range(i_min-1, i_max+1):
                for j in range(j_min-1, j_max+1):
                    if i>0 and i<b.shape[1]:
                        if j>0 and j<b.shape[0]:
                            density[j,i]=density[j,i]+1
                            #print(feature.GetField('sdbAre'))
                            #print(feature.GetField('sdbPer'))
                            #mean_area[j,i] = mean_area[j,i] + feature.GetField(2)
                            # mean_area[j,i] = mean_area[j,i] + feature.GetField('sdbAre')
                            # mean_perimeter[j,i] = mean_perimeter[j,i] + feature.GetField('sdbPer')
                            # mean_ssbCCo[j,i] = mean_ssbCCo[j,i] + feature.GetField('ssbCCo')
                            # mean_ssbCor[j,i] = mean_ssbCor[j,i] + feature.GetField('ssbCor')
                            # # mean_ssbSqu[j,i] = mean_ssbSqu[j,i] + feature.GetField('ssbSqu')
                            # mean_ssbERI[j,i] = mean_ssbERI[j,i] + feature.GetField('ssbERI')
                            # mean_ssbElo[j,i] = mean_ssbElo[j,i] + feature.GetField('ssbElo')
                            # mean_stbOri[j,i] = mean_stbOri[j,i] + feature.GetField('stbOri')
                            # mean_stbCeA[j,i] = mean_stbCeA[j,i] + feature.GetField('stbCeA')
                            # mean_stcOri[j,i] = mean_stcOri[j,i] + feature.GetField('stcOri')
                            # mean_sdcLAL[j,i] = mean_sdcLAL[j,i] + feature.GetField('sdcLAL')
                            # mean_sdcAre[j,i] = mean_sdcAre[j,i] + feature.GetField('sdcAre')
                            # mean_sscCCo[j,i] = mean_sscCCo[j,i] + feature.GetField('sscCCo')
                            # mean_sscERI[j,i] = mean_sscERI[j,i] + feature.GetField('sscERI')
                            # mean_sicCAR[j,i] = mean_sicCAR[j,i] + feature.GetField('sicCAR')
                            # mean_percentBuild[j,i] = mean_percentBuild[j,i] + percent
                            
                            # if std_stbOri[j,i]==0:
                            #     std_stbOri[j,i] = []
                            # std_stbOri[j,i].append(feature.GetField('stbOri'))
                            
                            
    # for i in range(b.shape[0]):
    #     for j in range(b.shape[1]):
    #         a = np.array(std_stbOri[i,j])
    #         std_stbOri[i,j] = np.std(a)
    
    # percentBuild = mean_percentBuild
    # mean_percentBuild = np.divide(mean_percentBuild, density, out=np.zeros_like(density), where=density!=0)
    # mean_percentBuild[np.isnan(mean_percentBuild)] = 0
    # mean_area = np.divide(mean_area, density, out=np.zeros_like(density), where=density!=0)
    # mean_area[np.isnan(mean_area)] = 0
    # mean_perimeter = np.divide(mean_perimeter, density, out=np.zeros_like(density), where=density!=0)
    # mean_perimeter[np.isnan(mean_perimeter)] = 0
    # mean_ssbCCo = np.divide(mean_ssbCCo, density, out=np.zeros_like(density), where=density!=0)
    # mean_ssbCCo[np.isnan(mean_ssbCCo)] = 0
    # mean_ssbCor = np.divide(mean_ssbCor, density, out=np.zeros_like(density), where=density!=0)
    # mean_ssbCor[np.isnan(mean_ssbCor)] = 0
    # mean_ssbSqu = np.divide(mean_ssbSqu, density, out=np.zeros_like(density), where=density!=0)
    # mean_ssbSqu[np.isnan(mean_ssbSqu)] = 0
    # mean_ssbERI = np.divide(mean_ssbERI, density, out=np.zeros_like(density), where=density!=0)
    # mean_ssbERI[np.isnan(mean_ssbERI)] = 0
    # mean_ssbElo = np.divide(mean_ssbElo, density, out=np.zeros_like(density), where=density!=0)
    # mean_ssbElo[np.isnan(mean_ssbElo)] = 0
    # mean_stbOri = np.divide(mean_stbOri, density, out=np.zeros_like(density), where=density!=0)
    # mean_stbOri[np.isnan(mean_stbOri)] = 0
    # mean_stbCeA = np.divide(mean_stbCeA, density, out=np.zeros_like(density), where=density!=0)
    # mean_stbCeA[np.isnan(mean_stbCeA)] = 0
    # mean_stcOri = np.divide(mean_stcOri, density, out=np.zeros_like(density), where=density!=0)
    # mean_stcOri[np.isnan(mean_stcOri)] = 0
    # mean_sdcLAL = np.divide(mean_sdcLAL, density, out=np.zeros_like(density), where=density!=0)
    # mean_sdcLAL[np.isnan(mean_sdcLAL)] = 0
    # mean_sdcAre = np.divide(mean_sdcAre, density, out=np.zeros_like(density), where=density!=0)
    # mean_sdcAre[np.isnan(mean_sdcAre)] = 0
    # mean_sscCCo = np.divide(mean_sscCCo, density, out=np.zeros_like(density), where=density!=0)
    # mean_sscCCo[np.isnan(mean_sscCCo)] = 0
    # mean_sscERI = np.divide(mean_sscERI, density, out=np.zeros_like(density), where=density!=0)
    # mean_sscERI[np.isnan(mean_sscERI)] = 0
    # mean_sicCAR = np.divide(mean_sicCAR, density, out=np.zeros_like(density), where=density!=0)
    # mean_sicCAR[np.isnan(mean_sicCAR)] = 0

    #scale the density raster to 0-1
    density = (density - np.min(density)) / (np.max(density) - np.min(density))
    
    georrefData(density, os.path.join(saveadd), sentinelImg)
    # georrefData(mean_area, os.path.join(saveadd,'Mean_area.tif'), sentinelImg)
    # georrefData(mean_perimeter, os.path.join(saveadd,'Mean_perimeter.tif'), sentinelImg)
    # georrefData(mean_ssbCCo, os.path.join(saveadd,'mean_ssbCCo.tif'), sentinelImg)
    # georrefData(mean_ssbCor, os.path.join(saveadd,'mean_ssbCor.tif'), sentinelImg)
    # georrefData(mean_ssbSqu, os.path.join(saveadd,'mean_ssbSqu.tif'), sentinelImg)
    # georrefData(mean_ssbERI, os.path.join(saveadd,'mean_ssbERI.tif'), sentinelImg)
    # georrefData(mean_ssbElo, os.path.join(saveadd,'mean_ssbElo.tif'), sentinelImg)
    # georrefData(mean_stbOri, os.path.join(saveadd,'mean_stbOri.tif'), sentinelImg)
    # georrefData(mean_stbCeA, os.path.join(saveadd,'mean_stbCeA.tif'), sentinelImg)
    # georrefData(mean_stcOri, os.path.join(saveadd,'mean_stcOri.tif'), sentinelImg)
    # georrefData(mean_sdcLAL, os.path.join(saveadd,'mean_sdcLAL.tif'), sentinelImg)
    # georrefData(mean_sdcAre, os.path.join(saveadd,'mean_sdcAre.tif'), sentinelImg)
    # georrefData(mean_sscCCo, os.path.join(saveadd,'mean_sscCCo.tif'), sentinelImg)
    # georrefData(mean_sscERI, os.path.join(saveadd,'mean_sscERI.tif'), sentinelImg)
    # georrefData(mean_sicCAR, os.path.join(saveadd,'mean_sicCAR.tif'), sentinelImg)
    # georrefData(std_stbOri, os.path.join(saveadd,'std_stbOri.tif'), sentinelImg)
    # georrefData(mean_percentBuild, os.path.join(saveadd,'mean_percentBuild.tif'), sentinelImg)
    # georrefData(percentBuild, os.path.join(saveadd,'percentBuild.tif'), sentinelImg)
    
    #scale the density raster to 0-1
    # density = np.clip(density, 0, 1)  # Ensure values are between 0 and 1
    # density[np.isnan(density)] = 0  # Set NaN values to 0

    # completion message
    print(f"Density raster saved to {saveadd}")
    



#Inputs: 
#1) Raster that will be the basis (the one that will be used to georeference the density raster)
#2) The buildings/morphometrics shapefile
#3) address  to save the rasters with the morphometrics

city = 'buenos_aires'
basedir = '/data/raw/'
year = 2023


CreateDensity(os.path.join(basedir, "sentinel", city, f'S2_{year}.tif'), 
os.path.join(basedir, f'buildings/{city}_bldg.geojson'), 
os.path.join(basedir, f'buildings/density/{city}_bd.tif'))