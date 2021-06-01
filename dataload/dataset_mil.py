import os
import json
import numpy as np
from torch.utils.data import Dataset
from ec_tools import colorful_str as cs


class MILDataset(Dataset):
    def __init__(self, json_path, train=True, input_dims=2048):
        self.json_path = json_path
        self.input_dims = input_dims
        self.train = train
        self.bag_info = []
        with open(self.json_path, 'r') as f:
            self.train_valid_info = json.load(f)
        self.phase = "train" if self.train else "vaild"
        # pos
        for cur_bag_path in self.train_valid_info["{}_pos".format(self.phase)]:
            cur_dirname = os.path.basename(cur_bag_path)
            instance_nums = len(os.listdir(cur_bag_path))
            # print(cs("(#y) instrance nums: {}".format(instance_nums)))
            npy_vector = []
            for each_idx in range(instance_nums):  # 向量与文件名字对应
                cur_npy_name = "{}_{}.npy".format(cur_dirname, each_idx)
                cur_npy_path = os.path.join(cur_bag_path, cur_npy_name)
                npy_vector.append(np.load(cur_npy_path))
            self.bag_info.append([cur_dirname, np.asarray(npy_vector).reshape(-1, self.input_dims), 1])
        self.pos_num = len(self.bag_info)
        # neg
        for cur_bag_path in self.train_valid_info["{}_neg".format(self.phase)]:
            cur_dirname = os.path.basename(cur_bag_path)
            instance_nums = len(os.listdir(cur_bag_path))
            # print(cs("(#y) instrance nums: {}".format(instance_nums)))
            npy_vector = []
            for each_idx in range(instance_nums):  # 向量与文件名字对应
                cur_npy_name = "{}_{}.npy".format(cur_dirname, each_idx)
                cur_npy_path = os.path.join(cur_bag_path, cur_npy_name)
                npy_vector.append(np.load(cur_npy_path))
            self.bag_info.append([cur_dirname, np.asarray(npy_vector).reshape(-1, self.input_dims), 0])
        self.neg_num = len(self.bag_info) - self.pos_num

        print(cs("(#y)[{}] Pos num:{}\t Neg num:{}".format(self.phase, self.pos_num, self.neg_num)))

    def __len__(self):
        return len(self.bag_info)

    def __getitem__(self, index):
        return self.bag_info[index][1],  self.bag_info[index][2], self.bag_info[index][0]   #vector、label， 名字、


if __name__ == '__main__':
    dset = MILDataset('../data/features/train_vaild_split.json', False)
    for each in range(700, 710):
        input, label, name = dset[each]
        print(input.shape, label, name)







