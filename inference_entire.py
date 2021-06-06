import os
import cv2
import math
import shutil
import heapq
import numpy as np
import argparse

from dataload.dataset_entire import MILDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from models.model import ABMIL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ec_tools import colorful_str as cs
from ec_tools import procedure


# original_pos_root = './data/img/pos_2560_overlap/'
# original_neg_root = './data/img/neg_2560_overlap/'
# labeled_path = './data/img/pos_2560_labeled/'
original_pos_root = './demo_sample_pos/'
original_neg_root = './demo_sample_neg/'
labeled_path = './demo_labeled'


def get_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--input', '-i', type=str, default='./data/bags_sim/train_vaild_split.json', help="path of input")
    parser.add_argument('--output', '-o', type=str, default='./heatmap/512_entire', help='path of output')
    parser.add_argument('--size', '-s', type=int, default=128, help="size of patch")
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='name of backbone')
    # parser.add_argument('--dims', '-d', type=int, default=2048, help='features dims')
    parser.add_argument('--ckpt', '-c', type=str, default='./ckpt/Att_entire_0119.pth', help='path of checkpoint')
    parser.add_argument("--find_badcase", '-find_bc', action='store_true', default=False, help="find badcase")
    args = parser.parse_args()
    return args


def inf():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    with procedure("Init Model") as p:

        model = ABMIL(encoder_name='mobilenetv2', split_index_list=None)
        model.relocate()

    with procedure("Load Param") as p:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["state_dict"])
        p.add_log(cs("(#g) load from {}".format(args.ckpt)))

    with procedure("Prepare Dataset") as p:
        test_dset = MILDataset(args.input, False)
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, pin_memory=True)

    with procedure("Start Inference") as p:
        model.eval()
        with torch.no_grad():
            for iter, (input, label, name) in enumerate(test_loader):
                input = input.squeeze().cuda()
                output, infer_results = model(input,  inference_only=True)

                pred = F.softmax(output, dim=1)
                weights = np.squeeze(infer_results['weights'].detach().cpu().numpy())
                cur_min = np.min(weights)
                cur_max = np.max(weights)
                weights = (weights - cur_min) / (cur_max - cur_min)
                pred = np.squeeze(pred.detach().cpu().numpy())

                name = name[0]
                original_img_path, img_name = parser_original_img_path(name, label)
                labeled_img_path = parser_gt_path(img_name) if label == 1 else None
                heat_img = heatmap(original_img_path, args.size, weights, args.size)

                output_path = os.path.join(args.output, "pos" if label == 1 else "neg")
                os.makedirs(output_path, exist_ok=True)
                shutil.copy(original_img_path, output_path)
                heat_img_path = os.path.join(output_path, "{}_color.jpg".format(img_name.split('.')[0]))
                cv2.imwrite(heat_img_path, heat_img)
                gt_path = None
                if label == 1 and labeled_img_path:
                    new_name = "{}_labeled.jpg".format(img_name.replace(".jpg", ""))
                    gt_path =  os.path.join(output_path, new_name)
                    shutil.copy(labeled_img_path, gt_path)

                if args.find_badcase:
                    badcase_output_path = os.path.join(args.output, 'badcase', "pos" if label == 1 else "neg")
                    os.makedirs(badcase_output_path, exist_ok=True)
                    find_badcase(pred, labbel, badcase_output_path, original_img_path, labeled_img_path, gt_path)


def parser_original_img_path(name, label):
    slide_name = '_'.join(name.split('_')[:2])
    img_name = "{}_pos.jpg".format(name) if label == 1 else "{}_neg.jpg".format(name)
    img_path = os.path.join(original_pos_root, "images", slide_name, img_name) if label == 1 else \
        os.path.join(original_neg_root, "images", slide_name, img_name)
    return img_path, img_name


def parser_gt_path(img_name):
    labeled_img_path = None
    for root, dirs, filenames in os.walk(labeled_path):
        for each_img in filenames:
            if each_img == img_name:
                labeled_img_path = os.path.join(root, each_img)
    return labeled_img_path


def find_badcase(pred, label, badcase_output_path, original_img_path=None, labeled_img_path=None, gt_path=None, threshold=0.5):  # If it is badcase, find the tested image and ground truth
    cur_pred = pred[1] if label == 1 else pred[0]
    is_badcase = cur_pred < threshold
    if is_badcase:
        shutil.copy(original_img_path, badcase_output_path)
        shutil.copy(labeled_img_path, badcase_output_path)
        if gt_path:
            shutil.copy(gt_path, badcase_output_path)


def heatmap(img_path, size,  weights, stride):
    img = cv2.imread(img_path)
    origin_img = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h, w])
    mask_cnt = np.zeros([h, w])
    blocks_per_row = math.ceil(1.0 * w / stride)
    blocks_per_col = math.ceil(1.0 * h / stride)
    assert blocks_per_col * blocks_per_row == weights.shape[0], print(blocks_per_col * blocks_per_row, weights.shape[0])

    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            cur_ex = j * stride + size     # 行
            cur_ey = i * stride + size   # 列
            cur_sx = j * stride if cur_ex <= w else w - size
            cur_sy = i * stride if cur_ey <= h else h - size
            mask[cur_sy:cur_ey, cur_sx:cur_ex] += round(weights[i * blocks_per_row + j], 2)
            mask_cnt[cur_sy:cur_ey, cur_sx:cur_ex] += 1
            color = (0, 255, 0)
            cv2.rectangle(img, (cur_sx, cur_sy), (cur_ex, cur_ey), color, 3)

            cv2.putText(img, str(round(weights[i * blocks_per_row + j], 2)), (cur_sx + 30, cur_sy + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, color, 1)
    mask = mask / mask_cnt
    mask = np.uint8(mask * 255)

    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heat_img = cv2.addWeighted(origin_img, 1, mask, 0.5, 0)
    return heat_img


if __name__ == '__main__':
    inf()









