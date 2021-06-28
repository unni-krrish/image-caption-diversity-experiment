from tensorflow.keras.applications import EfficientNetB0
import requests
import os
import zipfile
import pickle

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def down_and_extract(url, file, dest):
    file_url = url
    r = requests.get(url, stream=True)

    with open(file, "wb") as f:
        for block in r.iter_content(chunk_size=1024):
            if block:
                f.write(block)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def show_image(img_path):
    with open(img_path, 'rb') as file:
        img = Image.open(file)
        img = np.array(img)
        img = img / 255.0
        if (len(img.shape) == 2):
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        print(f"Shape : {img.shape}")
        plt.imshow(img)


def predict_class(img_path):
    with open(img_path, 'rb') as file:
        img = Image.open(file)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        print(img.shape)
        y_pred = effnet.predict(img.reshape(1, *img.shape))
    return decode_predictions(y_pred)


def load_image(path, size):
    img = Image.open(path)
    img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


cocodir = "/tmp/cocodata"
os.mkdir(cocodir)

drive_root = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Processed Data"

train_file = "http://images.cocodataset.org/zips/val2014.zip"
annots_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
annots_file = cocodir + "/annots"
imgdir = "/tmp/cocodata/val2014"

transfer_vals_colab = cocodir + "/B0_transfer_vals_val14.pkl"
transfer_vals_drive = drive_root + "/B0_transfer_vals_val14.pkl"

img_list_colab = cocodir + "/val14_img_list.pkl"
img_list_drive = drive_root + "/val14_img_list.pkl"

annots_json_colab = cocodir + "/annotations/captions_val2014.json"
captions_list_drive = drive_root + "/captions_list_val14.pkl"
tokenizer_drive = drive_root + "/val_14_tokenizer.pkl"

imgs_zip_file = "/tmp/cocodata/downloaded"
down_and_extract(train_file, imgs_zip_file, cocodir)

effnet = EfficientNetB0(include_top=True, weights='imagenet')
effnet.summary()

final_layer = effnet.get_layer('top_dropout')
transfer_model = Model(inputs=effnet.input, outputs=final_layer.output)
img_size = (224, 224)
n_transfer_units = 1280


def process_img_batches(data_dir, filenames, batch_size=64):

    num_images = len(filenames)

    shape = (batch_size, *img_size, 3)
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    shape = (num_images, n_transfer_units)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    ind = 0
    while ind < num_images:
        end_index = ind + batch_size
        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - ind
        for i, filename in enumerate(filenames[ind:end_index]):
            path = os.path.join(data_dir, filename)
            img = load_image(path, size=img_size)
            image_batch[i] = img

        transfer_values_batch = transfer_model.predict(
            image_batch[0:current_batch_size])
        transfer_values[ind:end_index] = transfer_values_batch[0:current_batch_size]

        ind = end_index

    return transfer_values


img_list = sorted(os.listdir(imgdir))
all_transfer_vals = process_img_batches(imgdir, img_list)

with open(transfer_vals_colab, 'wb') as file:
    pickle.dump(all_transfer_vals, file)

copyfile(transfer_vals_colab, transfer_vals_drive)

with open(, 'wb') as file:
    pickle.dump(img_list, file)

copyfile(img_list_colab, img_list_drive)

show_image(img_dir + '/' + img_list[9])

if not os.path.isdir(imgdir):
    os.mkdir(imgdir)

down_and_extract(annots_url, annots_file, cocodir)

with open(annots_json_colab, 'r') as f:
    targ = f.read()
    targ = json.loads(targ)
all_caps_dct = sorted(targ['annotations'], key=lambda x: x['image_id'])

all_caps = [[]]
st_tok, end_tok = 'ssss ', ' eeee'
prev = all_caps_dct[0]['image_id']
for i in all_caps_dct:
    curr = i['image_id']
    if curr == prev:
        all_caps[-1].append(st_tok + i['caption'] + end_tok)
    else:
        all_caps.append([])
        all_caps[-1].append(st_tok + i['caption'] + end_tok)
    prev = curr

flattened_caps = [i for j in all_caps for i in j]
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(flattened_caps)
token_list = [tokenizer.texts_to_sequences(caplist) for caplist in all_caps]

with open(captions_list_drive, 'wb') as f:
    pickle.dump(token_list, f)

with open(tokenizer_drive, 'wb') as f:
    pickle.dump(tokenizer, f)