import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, add, Activation, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import BinaryAccuracy, Precision, TruePositives, FalseNegatives 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv, decay=None):
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu', 
                          kernel_regularizer=decay, bias_regularizer=decay)(layer_in)
    # add max pooling layer
    layer_out = MaxPooling2D((2,2), strides=(2,2))(layer_in)
    return layer_out

# function for creating a naive inception block
def naive_inception_module(layer_in, f1, f2, f3, decay=None):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu', kernel_regularizer=decay, bias_regularizer=decay)(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2, (3,3), padding='same', activation='relu', kernel_regularizer=decay, bias_regularizer=decay)(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f3, (5,5), padding='same', activation='relu', kernel_regularizer=decay, bias_regularizer=decay)(layer_in)
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters, decay=None):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv1
    conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=decay, bias_regularizer=decay)(layer_in)
    # conv2
    conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=decay, bias_regularizer=decay)(conv1)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, merge_input])
    # activation function
    layer_out = Activation('relu')(layer_out)
    return layer_out

# function for reducing faeaures
def feature_reduce_block(layer_in, n_filters, stride):
    layer_out = Conv2D(n_filters, (1,1), padding='same', activation='relu')(layer_in)
    layer_out = MaxPooling2D(stride, strides=stride, padding='same')(layer_out)
    return layer_out

# function for classfiying labels
def classification_block(layer_in, middle_feature, flatten=True, decay=None, classes=2):
    if flatten:
        layer_in = Flatten()(layer_in)
    layer_out = Dense(middle_feature, activation='relu', kernel_initializer='he_uniform',
                      kernel_regularizer=decay, bias_regularizer=decay)(layer_in)
    if classes == 2:
        layer_out = Dense(1, activation='sigmoid')(layer_out)
    else:
        layer_out = Dense(classes, activation='softmax')(layer_out)
    return layer_out

# function for compling binary model
def binary_compile(model, rate, mom):
    opt = SGD(learning_rate=rate, momentum=mom)
    model.compile(optimizer=opt, 
                  loss='binary_crossentropy', 
                  metrics=[BinaryAccuracy(name='binary_accuracy'), Precision(name="precision"), 
                           TruePositives(name="true_positives"), FalseNegatives(name="false_negatives")])
    
# functions for saving model evaluation result
def show_result(model, x, y=None):    
    if y is None:
        val_loss, val_acc, val_pre, val_TP, val_FN = model.evaluate(x=x, steps=len(x), verbose=0)
    else:
        val_loss, val_acc, val_pre, val_TP, val_FN = model.evaluate(x=x, y=y, verbose=0)
    print("Test loss: {:.2f}".format(val_loss))
    print("Test Accuracy: {:.2f}".format(val_acc))
    print("Test Precision: {:.2f}".format(val_pre))
    print("Test True Positive Rate: {:.2f}".format(val_TP / (val_TP + val_FN)))

def get_avg_result(histories, start_epoch):
    avg_result = dict()
    for model_name in histories:
        val_loss = np.asarray(histories[model_name]['val_loss'])[start_epoch:].mean()
        val_acc = np.asarray(histories[model_name]['val_binary_accuracy'])[start_epoch:].mean()
        val_pre = np.asarray(histories[model_name]['val_precision'])[start_epoch:].mean()
        tpr_array = np.asarray(histories[model_name]['val_true_positives']) / (np.asarray(histories[model_name]['val_true_positives']) + np.asarray(histories[model_name]['val_false_negatives']))
        val_tpr = tpr_array[start_epoch:].mean()
        avg_result[model_name] = [val_loss, val_acc, val_pre, val_tpr]
    avg_result = pd.DataFrame(avg_result)
    avg_result.index = ["Loss", "Accuracy", "Precision", "TPR"]
    return avg_result

def get_percentile_result(histories, epoch):
    avg_result = dict()
    for model_name in histories:
        val_loss = histories[model_name]['val_loss'][epoch - 1]
        val_acc = histories[model_name]['val_binary_accuracy'][epoch - 1]
        val_pre = histories[model_name]['val_precision'][epoch - 1]
        val_tpr = histories[model_name]['val_true_positives'][epoch - 1] / (histories[model_name]['val_true_positives'][epoch - 1] + histories[model_name]['val_false_negatives'][epoch - 1])
        avg_result[model_name] = [val_loss, val_acc, val_pre, val_tpr]
    avg_result = pd.DataFrame(avg_result)
    avg_result.index = ["Loss", "Accuracy", "Precision", "TPR"]
    return avg_result

def compare_all_result(histories, filename, end=50, step=10):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=100, figsize=(8,14))    
    steps = np.arange(end + 1, step=step) - 1
    for model in histories:
        ax1.plot(np.asarray(histories[model]['val_loss'])[steps], label=model)
        ax2.plot(np.asarray(histories[model]['val_binary_accuracy'])[steps], label=model)
        ax3.plot(np.asarray(histories[model]['val_precision'])[steps], label=model)
        tpr = np.asarray(histories[model]['val_true_positives'])[steps] / (
            np.asarray(histories[model]['val_true_positives'])[steps] + 
            np.asarray(histories[model]['val_false_negatives'])[steps])
        ax4.plot(tpr, label=model)
    for ax, title in zip([ax1, ax2, ax3, ax4], ['Cross Entropy Loss', 
                       'Classification Accuracy', 'Precision', 'True Positive Rate']):
        ax.set_title(title)
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels(steps + 1)
        ax.legend(bbox_to_anchor=(1.05, 1.05))
    fig.savefig(filename)
    plt.show()
    
# plot diagnostic learning curves for binary model
def summarize_diagnostics_binary(history, filename):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=100, figsize=(8,14))
    # plot loss
    ax1.set_title('Cross Entropy Loss')
    ax1.plot(history['loss'], color='blue', label='train')
    ax1.plot(history['val_loss'], color='orange', label='test')
    ax1.legend(bbox_to_anchor=(1.05, 1.05))
    # plot accuracy
    ax2.set_title('Classification Accuracy')
    ax2.plot(history['binary_accuracy'], color='blue', label='train')
    ax2.plot(history['val_binary_accuracy'], color='orange', label='test')
    ax2.legend(bbox_to_anchor=(1.05, 1.05))
    # plot precision
    ax3.set_title('Precision')
    ax3.plot(history['precision'], color='blue', label='train')
    ax3.plot(history['val_precision'], color='orange', label='test')
    ax3.legend(bbox_to_anchor=(1.05, 1.05))
    # plot TPR
    ax4.set_title('True Positive Rate')
    tpr_train = np.asarray(history['true_positives']) / (np.asarray(history['true_positives']) + np.asarray(history['false_negatives']))
    tpr_test = np.asarray(history['val_true_positives']) / (np.asarray(history['val_true_positives']) + np.asarray(history['val_false_negatives']))
    ax4.plot(tpr_train, color='blue', label='train')
    ax4.plot(tpr_test, color='orange', label='test')
    ax4.legend(bbox_to_anchor=(1.05, 1.05))
    fig.savefig(filename)
    plt.show()
    
# load images from dirsctory
def _load_image_from_dir(path, label, size, color):
    image_list = sorted([file for file in os.listdir(path) if file.endswith("jpeg")])
    images = np.asarray([img_to_array(load_img(os.path.join(path, img), target_size=size, color_mode=color)) for img in image_list]).astype(int)
    labels = np.expand_dims(np.asarray([label for _ in image_list]).astype(int), axis=1)
    return images, labels

# load complete dataset
def load_binary_dataset(src, load_size, color_mode="grayscale"):
    subdirs = ['train/', 'test/']
    image_list = list()
    label_list = list()
    for subdir in subdirs:
        path_to_dataset = os.path.join(src, subdir)
        loaded_image = list()
        loaded_label = list()
        for crack_type in os.listdir(path_to_dataset):
            if not crack_type.startswith("."):
                path_to_crack_type = os.path.join(path_to_dataset, crack_type)
                label = 1 if crack_type == "cracked" else 0
                images, labels = _load_image_from_dir(path_to_crack_type, label, load_size, color_mode)
                loaded_image.append(images)   
                loaded_label.append(labels)   
        image_list.append(np.concatenate(loaded_image, axis=0))
        label_list.append(np.concatenate(loaded_label, axis=0))
    shuffle(image_list[0], label_list[0], random_state=100)
    shuffle(image_list[1], label_list[1], random_state=100)
    return image_list[0], image_list[1], label_list[0], label_list[1]

def save_history(histories, folder):
    new_histories = dict()
    for key in histories:
        data = histories[key]
        pd.DataFrame(data).to_csv(f"{folder}/{key}.csv")
        
def extract_in_batch(batch_num, data, model, label, pre_func):
    data_num = data.shape[0]
    split_index = np.linspace(0, data_num, batch_num + 1).astype(int)
    save_list = list()
    for i in range(batch_num):
        print(f"Processing {i + 1} batch of {label}-set...", end="\r")
        if i != (batch_num - 1):
            temp_data = data[split_index[i]: split_index[i + 1]]
        else: 
            temp_data = data[split_index[i]:]
        temp_data = pre_func(np.concatenate((temp_data,)*3, axis=-1))
        save_list.append(model.predict(temp_data))
    return np.concatenate(save_list)