import os
import cv2
import math
import shutil
import heapq
import numpy as np
import argparse

from dataload.dataset_entire import MILDataset
from models.attention_modelparallel import ABMIL, ModelParallelABMILLight
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from models.model import ABMIL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ec_tools import colorful_str as cs
from ec_tools import procedure


original_pos_root = './data/img/pos_2560_overlap/'
original_neg_root = './data/img/neg_2560_overlap/'
badcase_path = "./badcase/512_mobilenetv2_10"
labeled_path = './data/img/pos_2560_labeled/'
os.makedirs(badcase_path, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--input', '-i', type=str, default='./data/bags_sim/train_vaild_split.json', help="path of input")
    parser.add_argument('--output', '-o', type=str, default='./heatmap/512_entire', help='path of output')
    parser.add_argument('--size', '-s', type=int, default=128, help="size of patch")
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='name of backbone')
    parser.add_argument('--dims', '-d', type=int, default=2048, help='features dims')
    parser.add_argument('--ckpt', '-c', type=str, default='./ckpt/Att_entire_0119.pth', help='path of checkpoint')
    args = parser.parse_args()
    return args


def inf():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with procedure("Init Model") as p:
        # model = ABMIL(encoder_name='mobilenetv2_10', instance_loss_fn=nn.CrossEntropyLoss())
        # model.relocate(model_parallel=False)
        model = ABMIL(encoder_name="mobilenetv2_10", fc_input_dims=1280)
        # model = ModelParallelABMILLight(args.dims)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(torch.device("cuda"))

    with procedure("Load Param") as p:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["state_dict"])
        p.add_log(cs("(#g) load from {}".format(args.ckpt)))

    with procedure("Prepare Dataset") as p:
        val_dset = MILDataset(args.input, False)
        val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, pin_memory=True)
    vars = []
    with procedure("Start Inference") as p:
        model.eval()
        with torch.no_grad():
            for iter, (input, label, name) in enumerate(val_loader):
                input = input.squeeze().cuda()
                print(input.size())
                output, infer_results = model(input)
                #
                # output, infer_results = model(input, label=label, inference_only=True)

                pred = F.softmax(output, dim=1)
                weights = np.squeeze(infer_results['weights'].detach().cpu().numpy())
                cur_min = np.min(weights)
                cur_max = np.max(weights)
                weights = (weights - cur_min) / (cur_max - cur_min)
                if label == 1:
                    vars.append(np.std(weights))
                pred = np.squeeze(pred.detach().cpu().numpy())
                name = name[0]

                slide_name = '_'.join(name.split('_')[:2])
                img_name = "{}_roi.jpg".format(name) if label == 1 else "{}.jpg".format(name)
                img_path = os.path.join(original_pos_root, "images",  slide_name, img_name) if label == 1 else \
                    os.path.join(original_neg_root, "images", slide_name, img_name)

                cur_pred = pred[1] if label == 1 else pred[0]   # 当前类别的概率
                isbadcase = cur_pred < 0.5

                if cur_pred < 0.5:  # badcase
                    cur_badcase_path = os.path.join(badcase_path, "pos" if label == 1 else "neg")
                    os.makedirs(cur_badcase_path, exist_ok=True)
                #     shutil.copy(img_path, cur_badcase_path)

                output_path = os.path.join(args.output, "pos" if label == 1 else "neg")
                pred_output_path = os.path.join(args.output, 'weights')
                os.makedirs(pred_output_path, exist_ok=True)
                os.makedirs(output_path, exist_ok=True)
                np.save(os.path.join(pred_output_path, "{}.npy".format(name)), weights.squeeze())
                # output_name = "{}_{}.jpg".format(name, round(cur_pred, 2))
                output_name = "{}_pred.jpg".format(name)
                # find labeled img
                if label == 1:
                    labeled_img_path = None
                    for root, dirs, filenames in os.walk(labeled_path):
                        for each_img in filenames:
                            if each_img == img_name:
                                labeled_img_path = os.path.join(root, each_img)
                    if labeled_img_path is not None:
                        new_name = "{}_labeled.jpg".format(img_name.replace(".jpg", ""))
                        shutil.copy(labeled_img_path, os.path.join(output_path, new_name))
                        if isbadcase:
                            shutil.copy(labeled_img_path, os.path.join(cur_badcase_path, new_name))

                heatmap(img_path, args.size, weights, output_path, output_name, args.size//2, cur_badcase_path if isbadcase else None )
    print("var: {}".format(np.mean(vars)))


def heatmap(img_path, size,  weights, output_path, output_name, stride=None ,badcase_path=None, k_sample=10):

    top_k = heapq.nlargest(k_sample, range(len(weights)), weights.__getitem__)
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
            color = (255, 0, 0) if (i * blocks_per_row + j) in top_k else (0, 255, 0)
            cv2.rectangle(img, (cur_sx, cur_sy), (cur_ex, cur_ey), color, 3)

            cv2.putText(img, str(round(weights[i * blocks_per_row + j], 2)), (cur_sx + 30, cur_sy + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, color, 1)
    mask = mask / mask_cnt
    index = mask < 0.6
    # mask[index] = 0
    mask = np.uint8(mask * 255)
    # cv2.imwrite(os.path.join(output_path, "{}_gray.jpg".format(output_name.split('.')[0])), mask)
    # print(mask[128][128])
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    heat_img = cv2.addWeighted(origin_img, 1, mask, 0.5, 0)

    cv2.imwrite(os.path.join(output_path, output_name), img)
    cv2.imwrite(os.path.join(output_path, "{}_color.jpg".format(output_name.split('.')[0])), heat_img)
    # cv2.imwrite(os.path.join(output_path, "{}_mask.jpg".format(output_name.split('.')[0])), mask)

    if badcase_path is not None:
        os.makedirs(badcase_path, exist_ok=True)
        cv2.imwrite(os.path.join(badcase_path, output_name), img)


if __name__ == '__main__':
    inf()









