import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import random

def percentile_clip(image, lower_percentile=0.5, upper_percentile=99.5):
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    clipped_image = np.clip(image, lower_bound, upper_bound)
    return (clipped_image - lower_bound) / (upper_bound - lower_bound)


colors = ['#636363', '#ced2d2', '#fd0006'] #non-builtup, formal, informal
customCmap = mcolors.ListedColormap(colors)
classes = ['NBA', 'NDUA', 'DUA']

def plot_confusion_matrix(cm, save_plot=False):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    if save_plot:
        plt.savefig(save_plot, dpi=300)
    plt.show()


def plot_samples(train_s2, train_masks):
    ## Randomly visualizing the original images and its label
    random_number = random.randint(0,len(train_s2)-1)
    print('the image index number is',random_number)
    plt.figure(figsize=(20, 12))
    plt.subplot(231)
    plt.title('Training image', fontsize = 12)
    plt.imshow(percentile_clip((train_s2[..., [2,1,0]])[random_number, :, :, :]))
    plt.subplot(232)
    plt.title('Label', fontsize = 12)
    plt.imshow(train_masks[random_number], cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
    plt.show()

def plot_samples_1hot(train_s2, train_masks):
    # Randomly visualizing images and one hot encoded label
    random_number = random.randint(0, len(train_s2) - 1)
    print('The image index number is', random_number)
    plt.figure(figsize=(20, 12))
    plt.subplot(231)
    plt.title('Training image', fontsize=12)
    plt.imshow(percentile_clip((train_s2[..., [2,1,0]])[random_number, :, :, :]))
    plt.subplot(232)
    plt.title('Label', fontsize=12)
    mask = np.argmax(train_masks[random_number], axis=-1) if train_masks[random_number].ndim == 3 else train_masks[random_number]
    plt.imshow(mask, cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
    # plt.colorbar(ticks=[0, 1, 2], label='Classes')
    
    plt.show()

def plot_samples_dataloader(train_datagen):
    for train_images, train_masks in train_datagen:
        print("Data batch shape:", train_images.shape)
        print("Label batch shape:", train_masks.shape)
        print("Min and Max value in train image: ", train_images.min(), train_images.max())
        print("Labels in the mask are : ", np.unique(train_masks))
        for i in range(8):
            data_sample = train_images[i]
            label_sample = train_masks[i]
            plt.figure()
            plt.subplot(121)  
            plt.imshow(data_sample[..., [2,1,0]]*3)  
            plt.title(f"Data Sample {i + 1}")
            plt.subplot(122)  
            plt.imshow(label_sample) 
            plt.title(f"Label Sample {i + 1}")
            plt.show()
        break


def plot_log(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].plot(history['loss'], label='training loss')
    ax[0].plot(history['val_loss'], label='validation loss')
    ax[0].legend()
    
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].plot(history['IoU'], label='training fscore')
    ax[1].plot(history['val_IoU'], label='validation fscore')
    ax[1].legend()

    plt.tight_layout()
    plt.show()




# ##### plot some random test images from the model prediction
def plot_prediction(test_images, test_masks, model):
    test_img_number = random.randint(0, len(test_images)-1)
    print(f'Test image number: {test_img_number}')
    test_img = test_images[test_img_number]
    ground_truth=test_masks[test_img_number]
    ground_truth = np.argmax(ground_truth, axis=-1) #if masks are one hot encoded
    test_img_input=np.expand_dims(test_img, 0)
    test_pred = model.predict(test_img_input) 
    test_prediction = np.argmax(test_pred, axis=3)[0,:,:] 

    plt.figure(figsize=(20, 12))
    plt.subplot(231)
    plt.title('Test image', fontsize = 25)
    plt.imshow(test_img[..., [2,1,0]]*3)
    plt.subplot(232)
    plt.title('Reference', fontsize = 25)
    plt.imshow(ground_truth, cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
    plt.subplot(233)
    plt.title('Prediction', fontsize = 25)
    plt.imshow(test_prediction, cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
    plt.show()

def plot_prediction_mbcnn(test_images, test_masks, model):
    fig, ax = plt.subplots(ncols=4, figsize=(16, 16), layout='compressed')
    test_img_number = random.randint(0, len(test_masks) - 1)
    print(f'Test image number: {test_img_number}')
    input1 = test_images[0][test_img_number]  # multispectral image
    input2 = test_images[1][test_img_number]  # ancillary input
    test_masks = np.argmax(test_masks, axis=-1) #if masks are one hot encoded
    ground_truth = test_masks[test_img_number]
    test_img_inputs = [np.expand_dims(input1, 0), np.expand_dims(input2, 0)]
    prob = model.predict(test_img_inputs)
    pred = np.argmax(prob, axis=3)[0, :, :]
    ax[0].set_title('Test image')
    ax[0].imshow(percentile_clip(input1[..., [2, 1, 0]])) 
    ax[1].set_title('Reference')
    ax[1].imshow(ground_truth, cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
    ax[2].set_title('Prediction')
    ax[2].imshow(pred, cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
    ax[3].set_title('Probability')
    prod = ax[3].imshow(prob[0,:,:,2], cmap='hot')
    plt.colorbar(prod, ax=ax)
    plt.show()
    plt.tight_layout()

def plot_prediction_dataloader(test_images, test_masks, model):
    # Loop over all test images
    for test_img_number in range(len(test_images)):
        print(f'Test image number: {test_img_number}')
        test_img = test_images[test_img_number]
        ground_truth = test_masks[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        test_pred = model.predict(test_img_input) 
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:] 

        plt.figure(figsize=(20, 12))
        plt.subplot(231)
        plt.imshow((test_img[..., [2,1,0]])*3)
        plt.subplot(232)
        plt.imshow(ground_truth[:,:,:], cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
        plt.subplot(233)
        plt.imshow(test_prediction, cmap=customCmap, alpha=0.75, vmin=0, vmax=2)
        plt.show()