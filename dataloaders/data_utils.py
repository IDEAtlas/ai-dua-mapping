import os
import numpy as np
from tqdm import tqdm
import rasterio as rio
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from glob import glob
import geopandas as gpd


def norm_s2(s2):
    if np.max(s2) > 3:
        s2 = s2 / 10000.0
    return np.clip(s2, 0, 1)
    # return s2

def percentile_clip(image, lower_percentile=0.5, upper_percentile=99.5):
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    clipped_image = np.clip(image, lower_bound, upper_bound)
    return (clipped_image - lower_bound) / (upper_bound - lower_bound)

def load_data(img_dir, size_x, size_y, prefix, n_classes=None):
    images = []
    image_paths = sorted(glob(os.path.join(img_dir, f"{prefix}*.tif")))
    for img_path in image_paths:
        ds = rio.open(img_path)
        data = ds.read(window=((0, size_y), (0, size_x)))
        data = np.moveaxis(data, 0, -1)
        # Normalize the data if the input is Sentinel-2 and max value is greater than 3
        if 'S2' in img_path and np.max(data) > 3:
            data = norm_s2(data)
        if n_classes:
            data = to_categorical(data, n_classes)
        images.append(data)
        ds.close()
    return np.array(images)


def load_data_perBatch(data, label, batch_size):
    while True:
        N = data.shape[0]
        for i in range(0, N, batch_size):
            if i+batch_size<N:
                data_batch = data[i:i+batch_size]
                label_batch = label[i:i+batch_size]
            else:
                data_batch = data[N-batch_size:N]
                label_batch = label[N-batch_size:N]
            #print(label_batch)
            yield (data_batch, label_batch)


####----data batch loade------#

def get_batch_inds(idx, batch_size):
#Function to create the indexes base on the Training data'''
	N = len(idx)
	batchInds = []
	idx0 = 0
	toProcess = True
	while toProcess:
		idx1 = idx0 + batch_size
		if idx1 > N:
			idx1 = N
			idx0 = idx1 - batch_size
			toProcess = False
		batchInds.append(idx[idx0:idx1])
		idx0 = idx1
	return batchInds

def get_label_mask(labelPath, num_classes):
	currLabel = rio.open(labelPath)
	currLabel = currLabel.read(1)
      
	# currLabel=currLabel.astype(np.uint8)
	if num_classes > 1:
		currLabel = to_categorical(currLabel, num_classes=num_classes)
	else:
		currLabel=np.expand_dims(currLabel, axis=-1)
	return currLabel

def load_batch(inds, trainData, batch_size, input_patchSize, num_classes, label_address):
	'''Function that will effectivelly load the images on a numpy array to then feed it to the GPU
		inds: indexes for the batch
		trainData: list with the filenames of the trainData
	'''

	batchShape = (batch_size, input_patchSize[0], input_patchSize[1])
	numBands = input_patchSize[2] #Put here the number of Bands you want to use from Sentinel. Needs to adapt for your case.
	imgBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2], numBands))
	labelBatch = np.zeros((batchShape[0], batchShape[1], batchShape[2],num_classes))
	batchInd = 0
	for i in inds:
		currData = trainData[i]
		currImg = rio.open(currData) #open current Image
		currImg = currImg.read()
		currImg = np.transpose(currImg, (1, 2, 0))
		filename = os.path.split(currData)[-1]
		currlabelPath = os.path.join(label_address, filename.replace('HR', 'RF')) #open corresponding label file
		currlabel = get_label_mask(currlabelPath, num_classes)
		imgBatch[batchInd,:,:,:]=currImg
		labelBatch[batchInd,:,:,:]=currlabel
		batchInd = batchInd+1

	return imgBatch,labelBatch


def image_generator(trainData, batch_size, input_patchSize, num_classes, label_address):
	"""
	Generates training batches of image data and ground truth from either semantic or depth files
	:param trainData: training paths of CLS files (string replacement to get RGB and depth files) and starting x,y pixel positions (can be      non-zero if blocking is set to happen in params.py)
	:yield: current batch data
	"""
	idx = np.random.permutation(len(trainData))
	while True:
		batchInds = get_batch_inds(idx, batch_size)
		for inds in batchInds:
			imgBatch,labelBatch = load_batch(inds, trainData, batch_size, input_patchSize, num_classes, label_address)
			yield (imgBatch, labelBatch)

def patch_class_proportion(masks):
    count = 0
    for i in range(masks.shape[0]):
        if np.any(masks[i] == 2):  # Class indices start from 0, slum class three will be index 2
            count += 1
    print(f'Out of the total {len(masks)} patches, {count} patches contain slum class')


def calculate_class_weights(one_hot_encoded_masks):
    # Convert one-hot encoded mask to single class labels
    mask_labels = np.argmax(one_hot_encoded_masks, axis=-1).flatten()
    class_labels = np.unique(mask_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=mask_labels)
    
    return class_weights