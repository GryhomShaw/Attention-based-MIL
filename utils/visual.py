import os
import json
import cv2
import argparse
from ec_tools import colorful_str as cs


def get_args():
    parser = argparse.ArgumentParser(description="visual dataset")
    parser.add_argument('--input', '-i', type=str, default=None, help="path of input")
    parser.add_argument('--output', '-o', type=str, default=None, help="path of output")
    parser.add_argument('--nums', '-n', type=int, default=10, help="sample nums")
    args = parser.parse_args()
    return args


def visval(args):
    dir_list = os.listdir(args.input)
    if 'annotations' not in dir_list or 'images' not in dir_list:
        print(cs("(#r)[Error] path is invalid\t {}".format(args.input)))
        return
    json_path = {}
    json_path['train'] = os.path.join(args.input, 'annotations', 'train.json')
    json_path['valid'] = os.path.join(args.input, 'annotations', 'valid.json')

    for each_phase, each_path in json_path.items():
        cur_output_path = args.output
        os.makedirs(cur_output_path, exist_ok=True)
        # sample_num = args.nums
        with open(each_path, 'r') as f:
            info = json.load(f)
        annos = info['annotations']
        imgs = info['images']
        # for each_anno in annos[:sample_num]:
        #     img_id = each_anno['image_id']
        #     cur_img_info = imgs[img_id]
        #     assert cur_img_info['id'] == img_id + 1, print(cs("(#r)[Error] image_id doesn't match!!\t[{}--{}]"
        #                                                         .format(cur_img_info['id'], img_id+1)))
        #     cur_img_path = cur_img_info['file_name']
        #     cur_img = cv2.imread(cur_img_path)
        #     bbox = each_anno['bbox']
        #     ix, iy, jx, jy = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        #     cv2.rectangle(cur_img, (ix, iy), (jx, jy), (0, 0, 255), 3)
        #     img_name = os.path.basename(cur_img_path)
        #     path = os.path.join(cur_output_path, img_name)
        #     cv2.imwrite(path, cur_img)
        #     print(cs("(#g)img saved in {}".format(path)))

        for each_img_id in range(len(imgs)):
            img_id = imgs[each_img_id]['id']
            cur_img = cv2.imread(imgs[each_img_id]['file_name'])
            assert img_id == each_img_id + 1, print(cs("(#r)[Error] image_id doesn't match!!\t[{}--{}]"
                                                                .format(img_id, each_img_id+1)))

            cur_bboxes = [annos[idx]['bbox'] for idx in range(0, len(annos)) if annos[idx]['image_id'] == each_img_id]
            for each_bbox in cur_bboxes:
                ix, iy, jx, jy = each_bbox[0], each_bbox[1], each_bbox[0] + each_bbox[2], each_bbox[1] + each_bbox[3]
                cv2.rectangle(cur_img, (ix, iy), (jx, jy), (0, 0, 255), 3)
            img_name = os.path.basename(imgs[each_img_id]['file_name'])
            path = os.path.join(cur_output_path, img_name)
            cv2.imwrite(path, cur_img)
            print(cs("(#g)img saved in {}".format(path)))






if __name__ == '__main__':
    args = get_args()
    visval(args)



