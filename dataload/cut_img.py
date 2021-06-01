import os
import cv2
import argparse
import threadpool
from ec_tools import colorful_str as cs
from ec_tools import procedure


extension = ['jpg']


def get_args():
    parser = argparse.ArgumentParser(description="cut_img")
    parser.add_argument('--output', '-o', type=str, default='../data/bags', help='path of output')
    parser.add_argument('--size', '-s', type=int, default=128, help="size of patch")
    args = parser.parse_args()
    return args


def cut_img(p):
    img_path, prefix_name, output_root, label, size = p
    cur_output_path = os.path.join(output_root, label, prefix_name)
    os.makedirs(cur_output_path, exist_ok=True)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    cnt = 0
    for x in range(0, h, size):
        for y in range(0, w, size):
            ix = max(0, min(x, h - size))
            iy = max(0, min(y, w - size))
            cur_patch = img[ix: ix+size, iy: iy+size, :]
            cur_patch_name = "{}_{}.jpg".format(prefix_name, cnt)
            cv2.imwrite(os.path.join(cur_output_path, cur_patch_name), cur_patch)
            cnt += 1
            print(cs("(#g)img saved in {}".format(os.path.join(os.path.join(cur_output_path, cur_patch_name)))))


if __name__ == '__main__':
    args = get_args()
    pos_path = '../demo_sample_pos'
    neg_path = '../demo_sample_neg'
    pos_img_info = []
    neg_img_info = []
    output_root_path = args.output
    for root, dirs, filenames in os.walk(pos_path):
        for each_img in filenames:
            ext = each_img.split('.')[-1]
            if ext not in extension:
                continue
            img_path = os.path.join(root, each_img)
            prefix = each_img.replace('_pos.jpg', '')
            pos_img_info.append([img_path, prefix, output_root_path, 'pos', args.size])

    for root, dirs, filenames in os.walk(neg_path):
        for each_img in filenames:
            ext = each_img.split('.')[-1]
            if ext not in extension:
                continue
            img_path = os.path.join(root, each_img)
            prefix = each_img.replace('_neg.jpg', '')
            neg_img_info.append([img_path, prefix, output_root_path, 'neg', args.size])
    print(cs("(#y)pos nums:{}\t (#g)neg nums:{}".format(len(pos_img_info), len(neg_img_info))))
    params = pos_img_info + neg_img_info
    pool = threadpool.ThreadPool(8)
    reqs = threadpool.makeRequests(cut_img, params)
    [pool.putRequest(req) for req in reqs]
    pool.wait()


