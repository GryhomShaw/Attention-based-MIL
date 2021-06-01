from kfbreader import kfbReader
import cv2
import numpy as np
import os

path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/pos/pos_8/T2019_108.kfb'
reader = kfbReader.reader()
kfbReader.reader.ReadInfo(reader, path, 20, True)

w, h = reader.getWidth(),  reader.getHeight()
print(w, h)
# patches_per_col = []
# patches_per_raw = []
#
# for sx in range(0, h, 5120):
#     for sy in range(0, w, 5120):
#         patch_h = min(5120, h - sx)
#         patch_w = min(5120, w - sy)
#         cur_patch = reader.ReadRoi(sy, sx, patch_w, patch_h, 20)
#         dst_w = patch_w // 4
#         dst_h = patch_h // 4
#         img = cv2.resize(cur_patch, (dst_w, dst_h))
#         del cur_patch
#         print(img.shape)
#         patches_per_col.append(img)
#     patches_per_raw.append(np.concatenate(patches_per_col, axis=1))
#     patches_per_col = []
#
#
#
# print(len(patches_per_raw))
# img = np.concatenate(patches_per_raw, axis=0)
# print(img.shape)
# cv2.imwrite('./T2019_108.jpg', img)
#
# output_path = './T2019_108'
# os.makedirs(output_path, exist_ok=True)
# for sx in range(0, h, 5120):
#     for sy in range(0, w, 5120):
#         sx = max(0, min(h - 5120, sx))
#         sy = max(0, min(w - 5120, sy))
#         patch = reader.ReadRoi(sy, sx, 5120, 5120, 20)
#         patch_name = "{}_{}_({}_{}_{}_{}).jpg".format('T2019', '108', sy, sx, str(5120), str(5120))
#
#         cv2.imwrite(os.path.join(output_path, patch_name), patch)
#         print(patch_name)
#
#




