import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from ec_tools import colorful_str as cs


class MILDataset(Dataset):
    def __init__(self, json_path, train=True, transforms=None):
        self.json_path = json_path
        self.train = train
        self.bag_info = []
        self.transforms = transforms
        if self.transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if self.train:
                self.transforms = T.Compose(
                    [
                        T.RandomHorizontalFlip(p=0.3),
                        T.RandomVerticalFlip(p=0.3),
                        T.ToTensor(),
                        normalize
                    ]
                )
            else:
                self.transforms = T.Compose(
                    [
                        T.ToTensor(),
                        normalize
                    ]
                )

        with open(self.json_path, 'r') as f:
            self.train_valid_info = json.load(f)
        self.phase = "train" if self.train else "vaild"
        # self.sample_num = 10 if self.train else 5
        # pos
        self.bag_info.extend([[cur_bag_path, os.path.basename(cur_bag_path), 1] \
                               for cur_bag_path in self.train_valid_info["{}_pos".format(self.phase)]])

        self.pos_num = len(self.bag_info)
        # neg
        self.bag_info.extend([[cur_bag_path, os.path.basename(cur_bag_path), 0] \
                              for cur_bag_path in self.train_valid_info["{}_neg".format(self.phase)]])
        self.neg_num = len(self.bag_info) - self.pos_num

        print(cs("(#y)[{}] Pos num:{}\t Neg num:{}".format(self.phase, self.pos_num, self.neg_num)))

    def __len__(self):
        return len(self.bag_info)

    def __getitem__(self, index):
        cur_bag_path, cur_bag_name, cur_bag_label = self.bag_info[index]
        instance_num = len(os.listdir(cur_bag_path))
        imgs = []
        for each_idx in range(instance_num):
            cur_img_name = "{}_{}.jpg".format(cur_bag_name, str(each_idx))
            img = Image.open(os.path.join(cur_bag_path, cur_img_name))
            imgs.append(self.transforms(img))
        return torch.stack(imgs, dim=0),  cur_bag_label, cur_bag_name


def visualization(img, name, path):
    os.makedirs(path, exist_ok=True)
    img = img.squeeze().numpy().transpose(1, 2, 0)

    print(img.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255
    img = img.astype(np.uint8)
    cv2.imwrite(os.path.join(path, name+'.jpg'), img)


if __name__ == '__main__':
    dset = MILDataset('../data/bags_sim/train_vaild_split.json', False)
    for each in range(0, 10):
        input, label, name = dset[each]
        visualization(input[1], name, './dataset_debug')
        print(label)







