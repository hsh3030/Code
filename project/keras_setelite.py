import numpy as np # linear algebra
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm

train_df = pd.read_csv("./setelite/train.csv")
sample_df = pd.read_csv("./setelite/sample_submission.csv")

print(f'There are {train_df.shape[0]} records in train.csv')
train_df['Image_Label'].apply(lambda x : x.split('_')[1]).value_counts().plot(kind='bar')

len_train = len(os.listdir("./setelite/train_images"))
len_test = len(os.listdir("./setelite/test_images"))
print(f'There are {len_train} images in train dataset')
print(f'There are {len_test} images in test dataset')

len(train_df[train_df['EncodedPixels'].isnull()])

train_df.loc[train_df['EncodedPixels'].isnull(), 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts().plot(kind="bar")

train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()

train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts().plot(kind="bar")

from collections import defaultdict
train_size_dict = defaultdict(int)
train_path = Path("./setelite/train_images/")

for img_name in train_path.iterdir():
    img = Image.open(img_name)
    train_size_dict[img.size] += 1

print(train_size_dict)    

test_size_dict = defaultdict(int)
test_path = Path("./setelite/test_images/")

for img_name in test_path.iterdir():
    img = Image.open(img_name)
    test_size_dict[img.size] += 1

print(test_size_dict)

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col+4, 1]
    mask = np.zeros((1400, 2100, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(2100*1400, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos:(pos+le)] = 1
            mask[:, :, idx] = mask_label.reshape(1400, 2100, order='F')
    return img_names[0], mask

def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_path / name))
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.show()

idx_no_class = []
idx_class_1 = []
idx_class_2 = []
idx_class_3 = []
idx_class_4 = []
idx_class_multi = []
idx_class_triple = []

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError
        
    labels = train_df.iloc[col:col+4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    elif (labels.isna() == [False, True, True, True]).all():
        idx_class_1.append(col)
    elif (labels.isna() == [True, False, True, True]).all():
        idx_class_2.append(col)
    elif (labels.isna() == [True, True, False, True]).all():
        idx_class_3.append(col)
    elif (labels.isna() == [True, True, True, False]).all():
        idx_class_4.append(col)
    elif labels.isna().sum() == 1:
        idx_class_triple.append(col)
    else:
        idx_class_multi.append(col)

for idx in idx_class_1[:5]:
    show_mask_image(idx)

for idx in idx_class_2[:5]:
    show_mask_image(idx)
    
for idx in idx_class_3[:5]:
    show_mask_image(idx)

for idx in idx_class_4[:5]:
    show_mask_image(idx)

for idx in idx_class_multi[:5]:
    show_mask_image(idx)

