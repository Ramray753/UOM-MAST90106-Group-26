import os
import shutil
import numpy as np
from tensorflow.image import resize
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import save_img
from sklearn.metrics import f1_score, accuracy_score, classification_report

MAP = {
    0: "none",
    1: "crack",
    2: "long",
    3: "lat",
    4: "croc",
    5: "rail",
    6: "diag",
}

def process_image(path, image_list, folder="temp"):
    images = [img_to_array(load_img(os.path.join(path, img))).astype(int) for img in image_list]
    ideal_shape = (1230, 410, 3)
    crop_bound = {
        "xmin": int((images[0].shape[1] - ideal_shape[1]) / 2),
        "xmax": int((images[0].shape[1] + ideal_shape[1]) / 2 - 1),
        "ymin": int((images[0].shape[0] - ideal_shape[0]) / 2),
        "ymax": int((images[0].shape[0] + ideal_shape[0]) / 2 -1)
    }
    images_cropped = [img[crop_bound["ymin"]:crop_bound["ymax"] + 1, crop_bound["xmin"]:crop_bound["xmax"] + 1] for img in images]
    y_space = np.linspace(0, ideal_shape[0], 4).astype(int)
    y_space[-1] += 1
    images_split = [[img[y_space[i]:y_space[i + 1],] for i in range(len(y_space) - 1)] for img in images_cropped]
    images_flatten = [split_img for img in images_split for split_img in img]
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    index = 0
    for img in images_flatten:
        name_index = int(index / 3)
        name = image_list[name_index]
        print("Now Processing Image: {}                    ".format(name), end="\r")
        name_appendix = index % 3 + 1
        save_img("{}/{}-{}.JPG".format(folder, name.split(".")[0], name_appendix), img)
        index += 1

def load_image(size, path="temp", color="grayscale"):
    image_list = sorted([file for file in os.listdir(path) if file.endswith("JPG")])
    images = np.asarray([img_to_array(load_img(os.path.join(path, img), target_size=size, color_mode=color)) for img in image_list]).astype(int)
    return images

def extract_in_batch(batch_num, data, model, pre_func):
    data_num = data.shape[0]
    split_index = np.linspace(0, data_num, batch_num + 1).astype(int)
    save_list = list()
    for i in range(batch_num):
        print(f"Processing {i + 1} batch...                ", end="\r")
        if i != (batch_num - 1):
            temp_data = data[split_index[i]: split_index[i + 1]]
        else: 
            temp_data = data[split_index[i]:]
        temp_data = pre_func(np.concatenate((temp_data,)*3, axis=-1))
        save_list.append(model.predict(temp_data))
    return np.concatenate(save_list)

def decode(y_pred):
    return [MAP[i] for i in y_pred]

def get_result(y_true, y_pred, mul=True):
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Test accuracy: {:.2f}".format(accuracy_score(y_true, y_pred)))
    if mul:
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        print("Test f1 micro: {:.2f}".format(f1_micro))
        print("Test f1 macro: {:.2f}".format(f1_macro))
        print("Test f1 weighted: {:.2f}".format(f1_weighted))