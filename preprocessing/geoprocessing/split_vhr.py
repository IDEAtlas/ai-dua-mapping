
# Importing libraries
import glob
import copy
import numpy as np
from scipy import misc
import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
# import cv2
import time
from osgeo import gdal
import rasterio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gc
import tracemalloc
# from numba import cuda
from random import randint
import tifffile


def georrefData(data, output_path ,filename, gt=False):
	if (not os.path.isdir(output_path)):
		try:
			os.makedirs(output_path)
		except OSError:
			print ("Failed to create folder %s " % output_path)
		else:
			print ("Created Folder %s" % output_path) 
	size = data.shape
	#print(size)
	if gt:
		size = 1
	else:
		size=data.shape[2]
	width = data.shape[1]
	height = data.shape[0]
	driver = gdal.GetDriverByName("GTiff")
	filename = output_path+filename
	if gt:
		outdata = driver.Create(filename, width, height, size)
	else:
		outdata = driver.Create(filename, width, height, size, gdal.GDT_Float32)
	#outdata.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	#outdata.SetProjection(dataset.GetProjection())  ##sets same projection as input
	if gt:
		outdata.GetRasterBand(1).WriteArray(data)
	else:
		for i in range(size):
			outdata.GetRasterBand(i+1).WriteArray(data[:,:,i])
	outdata.FlushCache()  ##saves to disk!!
	outdata = None
	band = None
	ds = None

def crop_save(size, folder, S2,labelmap):
	if (not os.path.isdir(folder)):
		try:
			os.makedirs(folder)
		except OSError:
			print ("Falha ao criar diretorio %s " % folder)
		else:
			print ("Sucesso ao criar o diretorio %s" % folder)
	a = S2.shape
	count = 0
	for i in range(0, a[0], size):
		for j in range(0, a[1], size):
			if i+size<a[0]:
				if j+size<a[1]:
					crop_input2 = S2[i:i + size, j:j + size, :]
					crop_input3 = labelmap[i:i + size, j:j + size]
				else:
					crop_input2 = S2[i:i + size, a[1]-size:a[1], :]
					crop_input3 = labelmap[i:i + size, a[1]-size:a[1]]
			else:
				if j+size<a[1]:
					crop_input2 = S2[a[0]-size:a[0], j:j + size, :]
					crop_input3 = labelmap[a[0]-size:a[0], j:j + size]
				else:
					crop_input2 = S2[a[0]-size:a[0], a[1]-size:a[1], :]
					crop_input3 = labelmap[a[0]-size:a[0], a[1]-size:a[1]]
			georrefData(crop_input2, folder, "VHR_"+str(count)+".tif")
			georrefData(crop_input3, folder, "RF_"+str(count)+".tif", gt=True)
			count = count + 1

def dataset_create(address):

	#open VHR Reference data
	glob_path = os.path.join(address, 'RF_*.tif')
	label_paths = glob.glob(glob_path)
    
	for label_path in label_paths:
        

		#Open each Reference file
		labels_arr = tifffile.imread(label_path)
		filename = os.path.split(label_path)[-1]
		filename = filename.replace('RF', 'VHR')
		#Open VHR correspondent patch file.
		VHR = tifffile.imread(os.path.join(address,filename))
		folder = filename.split('.')[0]
		save_address = os.path.join(address,folder+'/')
		#Function will create images with 512x512 size from the VHR patch.
		crop_save(512, save_address, VHR, labels_arr)

if __name__ == '__main__':
	DIR = '/data/raw_data/image/vhr/pleiades/jakarta/pathces/train/'
	dataset_create(DIR)