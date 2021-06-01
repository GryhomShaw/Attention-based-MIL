import os
import cv2
import argparse
import numpy as np


from models.backbone import Backbone
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ec_tools import colorful_str as cs
from ec_tools import procedure


def get_args():
    parser = argparse.ArgumentParser(description="Encoder")
    parser.add_argument('--input', '-i', type=str, default='./data/bags_sim', help="path of input")
    parser.add_argument('--output', '-o', type=str, default='./data/features_sim', help='path of output')
    parser.add_argument('--batch_size', '-bs', type=int, default=256, help='adjust batch_size')
    parser.add_argument('--ckpt', '-c', type=str, default='./ckpt/resnet101.pth', help='path of checkpoint')
    args = parser.parse_args()
    return args


class EncoderDataset(Dataset):
    def __init__(self, bag_path, transforms=None):
        self.transforms = transforms
        if self.transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    normalize
                ])

        self.bag_info = [os.path.join(bag_path, 'pos', each_bag) for each_bag in os.listdir(os.path.join(bag_path, 'pos'))] + \
                        [os.path.join(bag_path, 'neg', each_bag) for each_bag in os.listdir(os.path.join(bag_path, 'neg'))]

    def __len__(self):
        return len(self.bag_info)

    def __getitem__(self, item):
        cur_bag_path = self.bag_info[item]
        cur_bag_name = os.path.basename(cur_bag_path)
        cur_label = cur_bag_path.split('/')[-2]
        instance_num = len(os.listdir(cur_bag_path))
        imgs = []
        for each_idx in range(instance_num):
            cur_img_name = "{}_{}.jpg".format(cur_bag_name, str(each_idx))
            img = Image.open(os.path.join(cur_bag_path, cur_img_name))
            imgs.append(self.transforms(img))
        return torch.stack(imgs, dim=0), cur_bag_name, cur_label


def encoder():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    with procedure("Init Model") as p:
        model = Backbone.model_zoo['mobilenetv2_10']
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(torch.device("cuda"))

    with procedure("Load Param") as p:
        ckpt = torch.load(args.ckpt)
        state_dict = model.state_dict()
        for k in state_dict.keys():
            if "encoder."+ k in ckpt['state_dict'].keys():
                state_dict[k] = ckpt['state_dict']["encoder." + k]
                # print(k)
        model.load_state_dict(state_dict)
        p.add_log(cs("(#g) load from {}".format(args.ckpt)))

    with procedure("Prepare Dataset") as p:
        encode_dataset = EncoderDataset(bag_path=args.input)
        loader = DataLoader(encode_dataset, 1, shuffle=False, pin_memory=True)

    with procedure("Start Encoding") as p:
        model.eval()
        with torch.no_grad():
            for images, name, cur_label in loader:
                input = images.cuda().squeeze()
                features = model(input)
                features = features.detach().cpu().numpy()
                print(features.shape)
                dirname = os.path.join(args.output, cur_label[-1])
                os.makedirs(dirname, exist_ok=True)
                filename = "{}.npy".format(name[-1])
                path = os.path.join(dirname, filename)
                np.save(path, features.squeeze())
                print(cs("(#g) features saved in {}".format(path)))



if __name__ == '__main__':
    encoder()