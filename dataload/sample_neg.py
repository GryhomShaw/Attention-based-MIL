"""visual analysis of dataset."""
import argparse
import cv2
import dataset
import numpy as np
import os
from coco import COCODataset
from eic_utils import colorful_str, colorful_print as cp

LABELS = ['pos']

def get_args():
    """setup args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='./neg_100', type=str,
                        help='path of output images.')
    parser.add_argument('-t', '--times', default=100, type=int,
                        help='sampleing times.')
    parser.add_argument('-s', '--shape', default=512, type=int,
                        help='shape of samples.')
    return parser.parse_args()

def mkdir(x):
    if not os.path.isdir(x):
        os.makedirs(x)

def sample_neg(dataset, args):
    slides_id = list(dataset.slides.keys())
    total = len(slides_id)
    train_num = int(total * 0.9)
    dir_name = args.output
    coco_train = COCODataset()
    coco_val = COCODataset()
    np.random.shuffle(slides_id)

    for idx, slide_id in enumerate(slides_id):
        mkdir(os.path.join(dir_name, 'images'))
        mkdir(os.path.join(dir_name, 'annotations'))
        mkdir(os.path.join(dir_name, 'images', slide_id))
        #print(os.path.join(dir_name, 'annotations'))
        coco = coco_train if idx < train_num else coco_val
        for label in LABELS:
            coco.add_category(label)

        print(colorful_str.log('[{}/{}] running on {}'.format(
            idx, total, slide_id)))

        shape = dataset.get_shape(slide_id)
        shape = (shape[0] - args.shape, shape[1] - args.shape)
        cnt = 0
        cnt_max = 0
        while cnt < args.times:
            cnt_max += 1
            if cnt_max > 200:
                break
            x, y = map(np.random.randint, shape)
            img = dataset.get_roi(slide_id, x, y, args.shape, args.shape)
            otsu_mask = otsu(img.copy())
            img_gray = img.copy()
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
            # if np.std(img_gray) < 10:  # 剔除空白页
            #    continue
            #if np.sum(otsu_mask) // 255 < 0.5 * img.shape[0] * img.shape[1]:
            #    continue
            cnt += 1
            img_name = '{}_({}-{}-{}-{})_{}.jpg'.format(
                slide_id, x, y, x + args.shape, y + args.shape, 'neg')

            img_path = os.path.join(dir_name, 'images', slide_id, img_name)
            # print("\t display {}".format(img_name))
            # cv2.imshow(img_name, img)
            # cv2.waitKey(0)
            cv2.imwrite(img_path, np.copy(img))

            coco.add_image(img_path, args.shape, args.shape, absolute_path=True)
            # cp.log('save img {}'.format(img_path))

    coco_train.dump(os.path.join(dir_name,
                           'annotations',
                           '{}.json'.format('train')))
    coco_val.dump(os.path.join(dir_name,
                           'annotations',
                           '{}.json'.format('valid')))
    

def otsu (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 40, 150, cv2.THRESH_OTSU)
    mask = 255 - th
    return mask


def main():
    args = get_args()
    neg_path = '/home/gryhomshaw/ssd1g/xiaoguohong/Attention-based-MIL/demo/neg'
    neg_dataset = dataset.Dataset(neg_path)

    sample_neg(neg_dataset, args)


if __name__ == '__main__':
    main()


