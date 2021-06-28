import requests
import os
import zipfile
import json
import pickle
from shutil import copyfile
import sys

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.models import load_model

cocodir = "/tmp/cocodata"
if not os.path.isdir(cocodir):
    os.mkdir(cocodir)

drive_data_root = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Processed Data"
transfer_vals_drive = drive_data_root + "/B0_transfer_vals_val17.pkl"
captions_list_drive = drive_data_root + "/captions_list_val17.pkl"
img_list_drive = drive_data_root + "/val17_img_list.pkl"

drive_save_dir = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/saved model"
model_name = "Exp_3_model_at_20000"
model_path = drive_save_dir + '/' + model_name

exp_name = "Exp_2_"
last_step = 30000
drive_res_root = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Results"
res_file_drive = drive_res_root + "/mean_scores_recent_run.pkl"
gen_caps_drive = drive_res_root + f"/{exp_name}generated_caps.pkl"


def down_and_extract(url, file, dest):
    file_url = url
    r = requests.get(url, stream=True)

    with open(file, "wb") as f:
        for block in r.iter_content(chunk_size=1024):
            if block:
                f.write(block)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


decoder_model = load_model(
    "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/saved model/Exp_3_model_at_20000")


def gen_valid_captions_v2(valid_transfer_vals, max_tokens=30):
    batch_shape = 100
    caps_final = np.zeros(shape=(1, 30), dtype=np.int)
    for b in range(50):
        token_int = 2*np.ones(batch_shape)
        output_text = ''
        count_tokens = 0

        shape = (batch_shape, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)
        dec_out = np.zeros(shape=shape, dtype=np.int)
        for pos in range(max_tokens-2):
            decoder_input_data[:, pos] = token_int
            x_data = {
                'transfer_values_input': valid_transfer_vals[100*b:100*(b+1)],
                'decoder_input': decoder_input_data
            }

            decoder_output = decoder_model.predict(x_data)
            token_onehot = decoder_output[:, pos, :]
            token_int = np.argmax(token_onehot, axis=1)
            # sampled_words = tokenizer.sequences_to_texts([token_int])
            dec_out[:, pos] = token_int

        caps_final = np.concatenate([caps_final, dec_out])
        print(f"\rFinished {b+1}/50", end="")
    caps_final = caps_final[1:]
    ret_caps = []
    for i in range(caps_final.shape[0]):
        sent = tokenizer.sequences_to_texts(caps_final[i].reshape(1, 30))
        ret_caps.append(' '.join(sent)[:-1])
    return ret_caps


# Get the CNN transfer values and image list for validation data - saved separately from another notebook
with open(transfer_vals_drive, 'rb') as file:
    valid_transfer_vals = pickle.load(file)

with open("/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Processed Data/val_14_tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)


with open(gen_caps_drive, 'wb') as f:
    pickle.dump(valid_caps, f)
