import os
import numpy as np
import cv2
import argparse
import shutil
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA

features_root = '/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/data/features/512_ImgNetpretrained/pos'
heatmap_root = '/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/heatmap/512'
def get_args():
    parser = argparse.ArgumentParser(description="PCA")
    parser.add_argument('--input', '-i', type=str, default='/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/heatmap/512', help="Path of input")
    parser.add_argument('--output', '-o', type=str, default='./reveal_all', help="path of output")
    return parser.parse_args()


def pca_visual(args):
    # img_pos_path = os.path.join(args.input, 'pos')
    # img_name = set(["{}_{}_{}".format(*each.split('_')[:3]) for each in os.listdir(img_pos_path)])
    # img_name = list(img_name)
    with open(args.input, 'r') as f:
        info = json.load(f)
    img_name = info['name_list']
    print(len(img_name), img_name[0])
    for idx, each_img in enumerate(img_name):
        features_path = os.path.join(features_root, "{}.npy".format(each_img))
        weights_path = os.path.join(heatmap_root, 'weights', "{}.npy".format(each_img))
        features = np.load(features_path)
        # weights = np.load(weights_path)
        weights = np.array(info['labels'][idx])
        # weights = (weights >= 0.7).astype(np.uint8)
        pred_img_path = os.path.join(heatmap_root, 'pos', "{}_pred.jpg".format(each_img))

        print(features.shape, weights.shape)
        pca = PCA(n_components=2)
        pca = pca.fit(features)
        cur_labels = weights
        labels_name = ['neg', 'pos']
        new_features = pca.transform(features)

        colors = ['blue', 'red']
        plt.figure()
        for each_label in range(2):
            print(cur_labels == each_label)
            plt.scatter(new_features[cur_labels == each_label, 0], new_features[cur_labels == each_label, 1], alpha=0.7, c=colors[each_label], label=labels_name[each_label])
        plt.legend()
        os.makedirs(args.output, exist_ok=True)

        path = os.path.join(args.output, "{}_pca.jpg".format(each_img))
        plt.savefig(path)
        plt.close()
        shutil.copy(pred_img_path, args.output)



if __name__ == '__main__':
    args = get_args()
    pca_visual(args)