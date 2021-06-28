import requests
import os
import zipfile
import json
import pickle
from shutil import copyfile

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.efficientnet import decode_predictions
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

cocodir = "/tmp/cocodata"
os.mkdir(cocodir)

drive_root = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Processed Data"
transfer_vals_drive = drive_root + "/B0_transfer_vals_val14.pkl"
captions_list_drive = drive_root + "/captions_list_val14.pkl"

drive_save_dir = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/saved model"
exp_name = "Exp_4_"

model_name = "Exp_3_model_at_20000"
load_model_from = drive_save_dir + '/' + model_name

with open(transfer_vals_drive, 'rb') as file:
    all_transfer_vals = pickle.load(file)

with open(captions_list_drive, 'rb') as file:
    token_list = pickle.load(file)

slice_st = 0
slice_end = 40000

# Slice the train set
# all_transfer_vals = all_transfer_vals[slice_st:slice_end]
# token_list = token_list[slice_st:slice_end]


def get_caption_batch(idx):
    result = []
    cap_per_img = {}
    for i in idx:
        result.extend(token_list[i])
        cap_per_img[i] = len(token_list[i])
    return result, cap_per_img


def get_img_batch(idx, cap_per_img):
    res = np.zeros((1, all_transfer_vals.shape[1]))
    for i in idx:
        arr = all_transfer_vals[i].reshape(1, -1)
        arr = np.tile(arr, (cap_per_img[i], 1))
        res = np.concatenate([res, arr])
    return res[1:]


def batch_generator(batch_size=64):
    st_id = 0
    upper_lim = all_transfer_vals.shape[0]
    while True:
        end_id = st_id + batch_size
        if end_id >= upper_lim:
            end_id = upper_lim
        idx = np.arange(st_id, end_id)

        tokens, cap_per_img = get_caption_batch(idx)
        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)

        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        decoder_input_img = get_img_batch(idx, cap_per_img)

        x_data = \
            {
                'decoder_input': decoder_input_data,
                'transfer_values_input': decoder_input_img
            }

        # Dict for the output-data.
        y_data = \
            {
                'decoder_output': decoder_output_data
            }

        if end_id >= upper_lim:
            st_id = 0
        else:
            st_id += batch_size

        yield (x_data, y_data)


batch_size = 64
total_samples = sum([len(j) for j in token_list])
steps_per_epoch = int(total_samples/batch_size)
state_size = 512
embedding_size = 128
steps_per_epoch


def create_model():
    n_transfer_units = all_transfer_vals.shape[1]
    transfer_values_input = Input(
        shape=(n_transfer_units,), name='transfer_values_input')
    decoder_transfer_map = Dense(state_size,
                                 activation='tanh',
                                 name='decoder_transfer_map')
    initial_state = decoder_transfer_map(transfer_values_input)

    decoder_input = Input(shape=(None, ), name='decoder_input')
    decoder_embedding = Embedding(input_dim=10000,
                                  output_dim=embedding_size,
                                  name='decoder_embedding')
    net = decoder_embedding(decoder_input)

    lstm_dec_1 = LSTM(state_size, name='lstm_decoder_1', return_sequences=True)
    lstm_dec_2 = LSTM(state_size, name='lstm_decoder_2', return_sequences=True)
    lstm_dec_3 = LSTM(state_size, name='lstm_decoder_3', return_sequences=True)

    net = lstm_dec_1(net, initial_state=[initial_state, initial_state])
    net = lstm_dec_2(net, initial_state=[initial_state, initial_state])
    net = lstm_dec_3(net, initial_state=[initial_state, initial_state])

    decoder_dense = Dense(10000,
                          activation='softmax',
                          name='decoder_output')
    decoder_output = decoder_dense(net)
    ret_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])
    return ret_model


train_type = 0
if train_type == 0:
    decoder_model = create_model()
else:
    decoder_model = load_model(load_model_from)

decoder_model.compile(optimizer=RMSprop(lr=1e-3),
                      loss='sparse_categorical_crossentropy')

decoder_model.save(f"{drive_save_dir}/{exp_name}model_at_{slice_end}")
