import os
import shutil
import json

img_root = '/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/data/img/pos_512/images'

with open('./test_list.json','r') as f:
    info = json.load(f)
    name_list = info['name_list']

for each_slide in name_list:
    slide_name = "{}_{}".format(*each_slide.split('_')[:2])
    img_path = os.path.join(img_root, slide_name, "{}_roi.jpg".format(each_slide))
    shutil.copy(img_path, './origin_img')
