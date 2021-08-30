import matplotlib.pyplot as plt
import numpy as np
import json
import requests
import os
import zipfile
import pickle
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import EfficientNetB0


def down_and_extract(url, file, dest):
    """
    Downloads and saves zip files from any zipfile link.
    """
    r = requests.get(url, stream=True)

    with open(file, "wb") as f:
        for block in r.iter_content(chunk_size=1024):
            if block:
                f.write(block)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def show_image(img_path):
    """
    Simple function to display an image from path
    """
    with open(img_path, 'rb') as file:
        img = Image.open(file)
        img = np.array(img)
        img = img / 255.0
        if (len(img.shape) == 2):
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        print(f"Shape : {img.shape}")
        plt.imshow(img)


def predict_class(img_path):
    """
    Decodes the predictions from Effnet final layer.
    Uses the Effnet pretrained model.
    """
    with open(img_path, 'rb') as file:
        img = Image.open(file)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        print(img.shape)
        y_pred = effnet.predict(img.reshape(1, *img.shape))
    return decode_predictions(y_pred)


def load_image(path, size):
    """
    Loads an image into np array from path with a check on its shape.
    """
    img = Image.open(path)
    img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.array(img)
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


cocodir = ""
if not os.path.isdir(cocodir):
    os.mkdir(cocodir)

# Setup the directories and files to write and read from
train_file = "http://images.cocodataset.org/zips/val2017.zip"
annots_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
annots_file = cocodir + "/annots"
imgdir = os.path.join(cocodir, "val2017")
transfer_values = os.path.join(cocodir, "/B0_transfer_vals_val17.pkl")
annots_json = os.path.join(cocodir, "/annotations/captions_val2017.json")
imgs_zip_file = os.path.join(cocodir, "downloaded")
down_and_extract(train_file, imgs_zip_file, cocodir)

# Setup the Effnet Model extraction
effnet = EfficientNetB0(include_top=True, weights='imagenet')
final_layer = effnet.get_layer('top_dropout')
transfer_model = Model(inputs=effnet.input, outputs=final_layer.output)
img_size = (224, 224)
n_transfer_units = 1280


def process_img_batches(data_dir, filenames, batch_size=64):
    """
    To reduce memory use, the images are processed in batches.
    For each batch, transfer values are updated using the model.

    Returns the transfer values for all the images in [filenames] 
    with the specified [batch size] and a directory to flow from.
    """
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


# Get the list of all the images and corresponding transfer values
img_list = sorted(os.listdir(imgdir))
all_transfer_vals = process_img_batches(imgdir, img_list)
with open(transfer_values, 'wb') as file:
    pickle.dump(all_transfer_vals, file)
