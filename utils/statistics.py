import os
from dataload.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

pos_path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/pos'
neg_path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/neg'

label_path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/labels'
pos_dataset = Dataset(pos_path, label_path)

width_info = []
height_info = []

for each_idx, labels in pos_dataset.labels.items():
    for each_label in labels:
        if each_label['class'] == 'roi':
            continue
        w, h = each_label['w'], each_label['h']
        width_info.append(w)
        height_info.append(h)

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Width')
plt.hist(width_info, 50)
# ax[0, 1].hist(height_info, 50)
plt.subplot(1,2,2)
plt.hist(height_info, 50)
plt.title('Height')
# ax[0, 1].hist(height_info, 50)
plt.savefig('./statistics.png')
plt.close()





