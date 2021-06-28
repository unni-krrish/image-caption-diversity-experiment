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

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

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

exp_name = "Exp_2"
last_step = 40000
drive_res_root = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project"
res_file_drive = drive_res_root + "/mean_scores_recent_run.pkl"
gen_caps_drive = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Results/Exp_2_generated_caps.pkl"
img_list_drive = "/content/drive/MyDrive/Colab Notebooks/AI Mini Project/Processed Data/val17_img_list.pkl"


def down_and_extract(url, file, dest):
    file_url = url
    r = requests.get(url, stream=True)

    with open(file, "wb") as f:
        for block in r.iter_content(chunk_size=1024):
            if block:
                f.write(block)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


with open(gen_caps_drive, 'rb') as f:
    valid_caps = pickle.load(f)

with open(img_list_drive, 'rb') as f:
    img_list_valid = pickle.load(f)

# Convert image_id into approp. format for pycocoevalcap


def remove_padding(id):
    for i in range(len(id)):
        if id[i] != '0':
            return int(id[i:-4])


img_list_valid = list(map(remove_padding, img_list_valid))
valid_shape = len(img_list_valid)
valid_shape

valid_caps = [i[:-1] for i in valid_caps]

jsondir = cocodir + "/jsonresults1"
if not os.path.isdir(jsondir):
    os.mkdir(jsondir)

annots = []
for ind in range(len(img_list_valid)):
    annots.append(
        {'image_id': img_list_valid[ind], 'caption': valid_caps[ind]})

st_ind, batch, done = 0, 1, False
while not done:
    # print(st_ind, batch, done)
    if st_ind + 1000 < len(annots):
        item = annots[st_ind:st_ind+1000]
    else:
        item = annots[st_ind:]
    with open(jsondir + f"/json_res_{batch}.json", 'w') as f:
        json.dump(item, f)
    st_ind += 1000
    batch += 1
    if st_ind >= len(annots):
        done = True

url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
file = "/tmp/cocodata/annots"
down_and_extract(url, file, cocodir)

coco = COCO("/tmp/cocodata/annotations/captions_val2017.json")
scores = {}
for file in os.listdir(jsondir):
    coco_result = coco.loadRes(jsondir+'/'+file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()

    if len(scores) == 0:
        for metric, score in coco_eval.eval.items():
            scores[metric] = [score]
    else:
        for metric, score in coco_eval.eval.items():
            scores[metric].append(score)

mean_scores = {k: np.mean(scores[k]) for k in list(scores.keys())}
print(len(scores['SPICE']))
mean_scores

with open(res_file_drive, 'wb') as file:
    pickle.dump(mean_scores, file)
