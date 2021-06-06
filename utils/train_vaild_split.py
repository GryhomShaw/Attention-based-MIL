import json 
import os
import random

pos_path = "/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/data/bags/pos"
neg_path = "/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/data/bags/neg"

pos_bag_list = [os.path.join(pos_path, each) for each in os.listdir(pos_path)]
neg_bag_list = [os.path.join(neg_path, each) for each in os.listdir(neg_path)]

random.shuffle(pos_bag_list)
random.shuffle(neg_bag_list)

train_vaild_info = {}
ratios = 0.9

pos_train_num = int(ratios* len(pos_bag_list))
pos_vaild_num = len(pos_bag_list) - pos_train_num
neg_train_num = int(ratios * len(neg_bag_list))
neg_vaild_num = len(neg_bag_list) - neg_train_num

train_vaild_info["train_pos"] = pos_bag_list[:pos_train_num]
train_vaild_info["vaild_pos"] = pos_bag_list[pos_train_num:]
train_vaild_info ["train_neg"] = neg_bag_list[:neg_train_num]
train_vaild_info["vaild_neg"] = neg_bag_list[neg_train_num:]
print(train_vaild_info["train_pos"][0])
print("train_pos:{}\ttrain_neg:{}\tvaild_pos:{}\t vaild_neg:{}".format(pos_train_num, neg_train_num, pos_vaild_num, neg_vaild_num))

with open("./train_vaild_split.json",'w') as f:
    json.dump(train_vaild_info,f)



