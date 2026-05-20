import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Global Configurations
colors = ['#636363', '#ced2d2', '#fd0006'] # non-builtup, formal, informal
customCmap = mcolors.ListedColormap(colors)
classes = ['NBA', 'NDUA', 'DUA']

def percentile_clip(image, lower_percentile=0.5, upper_percentile=99.5):
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    clipped_image = np.clip(image, lower_bound, upper_bound)
    return (clipped_image - lower_bound) / (upper_bound - lower_bound)

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

def plot_samples(train_iterator, batch_size=None, cmap=customCmap):
    train_images, train_masks = next(train_iterator)    
    num_samples = batch_size if batch_size is not None else train_images.shape[1]
    
    for i in range(num_samples):
        data_sample = train_images[0][i]
        if len(train_masks[i].shape) == 3:
            label_sample = np.argmax(train_masks[i], axis=-1)
        else:
            label_sample = train_masks[i]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)  
        
        rgb_sample = tf.gather(data_sample, [2, 1, 0], axis=-1)
        plt.imshow(rgb_sample * 3)  
        plt.title(f"S2 sample {i + 1}")
        
        plt.subplot(122)  
        plt.imshow(label_sample, cmap=cmap, alpha=0.75, vmin=0, vmax=2) 
        plt.title(f"Label Sample {i + 1}")
        plt.show()

def plot_log(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].plot(history['loss'], label='training loss')
    ax[0].plot(history['val_loss'], label='validation loss')
    ax[0].legend()
    
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].plot(history['f1'], label='training fscore')
    ax[1].plot(history['val_f1'], label='validation fscore')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_prediction(test_iterator, model, batch_size=None, cmap=customCmap):
    test_images, test_masks = next(test_iterator)
    
    num_samples = batch_size if batch_size is not None else test_images.shape[1]
    
    for i in range(num_samples):
        print(f'Sample number: {i + 1}')
        
        data_sample_s2 = test_images[0][i]
        data_sample_bd = test_images[1][i]
        ground_truth = test_masks[i]
        
        test_img_input_s2 = tf.expand_dims(data_sample_s2, 0)
        test_img_input_bd = tf.expand_dims(data_sample_bd, 0)
        
        # Use model call directly to avoid fused convolution issues
        @tf.function(jit_compile=False)
        def predict_fn(x1, x2):
            return model((x1, x2), training=False)
        
        test_pred = predict_fn(test_img_input_s2, test_img_input_bd) 
        test_prediction = np.argmax(test_pred, axis=3)[0, :, :] 

        plt.figure(figsize=(20, 12))
        
        plt.subplot(231)
        rgb_sample = tf.gather(data_sample_s2, [2, 1, 0], axis=-1)
        plt.imshow(rgb_sample * 3)
        plt.title(f"S2 sample {i + 1}")
        
        plt.subplot(232)
        plt.imshow(ground_truth, cmap=cmap, alpha=0.75, vmin=0, vmax=2)
        plt.title("Ground Truth")
        
        plt.subplot(233)
        plt.imshow(test_prediction, cmap=cmap, alpha=0.75, vmin=0, vmax=2)
        plt.title("Prediction")
        
        plt.show()