import os
import cv2
import math
import shutil
import numpy as np
import argparse

from dataload.dataset_mil import MILDataset
from models.aggregation import Attention

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ec_tools import colorful_str as cs
from ec_tools import procedure


original_pos_root = './data/img/pos_512_sim'
original_neg_root = './data/img/neg_512_sim'
badcase_path = "./badcase/512"
labeled_path = './data/img/pos_512_sim_labeled/'
os.makedirs(badcase_path, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument('--input', '-i', type=str, default='./data/features/512/modify.json', help="path of input")
    parser.add_argument('--output', '-o', type=str, default='./heatmap/512', help='path of output')
    parser.add_argument('--size', '-s', type=int, default=128, help="size of patch")
    parser.add_argument('--ckpt', '-c', type=str, default='./ckpt/Att_2048_sim.pth', help='path of checkpoint')
    args = parser.parse_args()
    return args


def inf():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with procedure("Init Model") as p:
        model = Attention(2048)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(torch.device("cuda"))

    with procedure("Load Param") as p:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["state_dict"])
        p.add_log(cs("(#g) load from {}".format(args.ckpt)))

    with procedure("Prepare Dataset") as p:
        # train_dset = MILDataset(args.input, True, 512)
        val_dset = MILDataset(args.input, False, 2048)
        # train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, pin_memory=True)

    with procedure("Start Inference") as p:
        model.eval()
        with torch.no_grad():
            for iter, (input, label, name) in enumerate(val_loader):
                input = input.squeeze().cuda()
                output, weights = model(input)
                pred = F.softmax(output, dim=1)
                weights = np.squeeze(weights.detach().cpu().numpy())
                pred = np.squeeze(pred.detach().cpu().numpy())
                name = name[0]
                # print(name)
                slide_name = '_'.join(name.split('_')[:2])
                img_name = "{}_roi.jpg".format(name) if label == 1 else "{}.jpg".format(name)
                img_path = os.path.join(original_pos_root, "images",  slide_name, img_name) if label == 1 else \
                    os.path.join(original_neg_root, "images", slide_name, img_name)

                # print(img_path)

                cur_pred = pred[1] if label == 1 else pred[0]   # 当前类别的概率
                isbadcase = cur_pred < 0.5

                if cur_pred < 0.5:  # badcase
                    cur_badcase_path = os.path.join(badcase_path, "pos" if label == 1 else "neg")
                    os.makedirs(cur_badcase_path, exist_ok=True)
                #     shutil.copy(img_path, cur_badcase_path)

                output_path = os.path.join(args.output, "pos" if label == 1 else "neg")
                os.makedirs(output_path, exist_ok=True)
                output_name = "{}_{}.jpg".format(name, round(cur_pred, 2))
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

                heatmap(img_path, args.size, weights, output_path, output_name, cur_badcase_path if isbadcase else None )


def heatmap(img_path, size,  weights, output_path, output_name, badcase_path=None):

    print(img_path)
    cur_min = np.min(weights)
    cur_max = np.max(weights)
    weights = (weights - cur_min) / (cur_max - cur_min + 1e-4)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    blocks_per_row = math.ceil(1.0 * w / size)
    blocks_per_col = math.ceil(1.0 * h / size)
    assert blocks_per_col * blocks_per_row == weights.shape[0], print(blocks_per_col * blocks_per_row, weights.shape[0])

    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            cur_ex = (j + 1) * size     # 行
            cur_ey = (i + 1) * size     # 列
            cur_sx = j * size if cur_ex <= w else w - size
            cur_sy = i * size if cur_ey <= h else h - size
            cv2.rectangle(img, (cur_sx, cur_sy), (cur_ex, cur_ey), (0, 0, 255), 3)
            cv2.putText(img, str(round(weights[i * blocks_per_row + j], 2)), (cur_sx + 30, cur_sy + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_path, output_name), img)
    if badcase_path is not None:
        os.makedirs(badcase_path, exist_ok=True)
        cv2.imwrite(os.path.join(badcase_path, output_name), img)


if __name__ == '__main__':
    inf()









