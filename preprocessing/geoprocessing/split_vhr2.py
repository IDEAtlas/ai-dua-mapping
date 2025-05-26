import os
import glob
import random
import numpy as np
import tifffile
from osgeo import gdal


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


def choose_patches(main_folder):

	list_folders = [a[0] for a in os.walk(main_folder)]

	max_total_patches = 7000
	count = 0
	for folder in list_folders:
		print(folder)
		gt_file_list = glob.glob(os.path.join(folder,'RF_*.tif'))
		for gt_file in gt_file_list:
				gt = tifffile.imread(gt_file)
				filename = os.path.split(gt_file)[-1]
				filename = filename.replace('RF', 'VHR')
				wv3 = tifffile.imread(os.path.join(folder,filename))
				zer = wv3==[0,0,0,0]
				zer = np.sum(zer[:,:,0])/(512*512)
				#print(zer)
				if(zer<0.01):
					classCount = np.zeros(4)
					for i in range(4):
						classCount[i]=np.sum(gt==i)
					if classCount[2]>0:
						georrefData(wv3, main_folder, "VHR_"+str(count)+".tif")
						georrefData(gt, main_folder, "RF_"+str(count)+".tif", gt=True)
						count = count+1
					else:
						a=np.argmax(classCount)
						if a ==0:
							x = random.uniform(0,1)
							if x < 0.25:
								georrefData(wv3, main_folder, "VHR_"+str(count)+".tif")
								georrefData(gt, main_folder, "RF_"+str(count)+".tif", gt=True)
								count = count+1
						else:
							x = random.uniform(0,1)
							if x < 0.09:
								georrefData(wv3, main_folder, "VHR_"+str(count)+".tif")
								georrefData(gt, main_folder, "RF_"+str(count)+".tif", gt=True)
								count = count+1

valid = '/data/raw_data/image/vhr/pleiades/jakarta/pathces/valid/'
train = '/data/raw_data/image/vhr/pleiades/jakarta/pathces/train/'
choose_patches(valid)
choose_patches(train)