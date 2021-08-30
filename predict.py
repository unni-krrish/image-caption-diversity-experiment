import requests
import os
import zipfile
import pickle
import numpy as np
from tensorflow.keras.models import load_model

cocodir = ""
if not os.path.isdir(cocodir):
    os.mkdir(cocodir)

source_dir = ""
transfer_vals = source_dir + "/B0_transfer_vals_val17.pkl"
captions_list_drive = source_dir + "/captions_list_val17.pkl"
img_list_drive = source_dir + "/val17_img_list.pkl"

save_dir = ""
model_name = "Exp_3_model_at_20000"
model_path = save_dir + '/' + model_name
exp_name = "Exp_2_"
last_step = 30000

# Load the tokenizer instance used in the save_train_data.py
with open("", 'rb') as f:
    tokenizer = pickle.load(f)


def down_and_extract(url, file, dest):
    """
    Downloads and saves zip files from any zipfile link.
    """
    file_url = url
    r = requests.get(url, stream=True)

    with open(file, "wb") as f:
        for block in r.iter_content(chunk_size=1024):
            if block:
                f.write(block)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


# Load the final model saved by train_model.py
decoder_model = load_model("")


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
with open(transfer_vals, 'rb') as file:
    valid_transfer_vals = pickle.load(file)

valid_caps = gen_valid_captions_v2(valid_transfer_vals)
