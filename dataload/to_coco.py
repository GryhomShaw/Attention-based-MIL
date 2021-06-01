"""visual analysis of dataset."""

import argparse
import cv2
import dataset
import numpy as np
import os
from coco import COCODataset
from eic_utils import colorful_str


def get_args():
    """setup args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='data/patches1/',
                        type=str, help='images')
    return parser.parse_args()

def otsu (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = 255 - th
    return mask


def scan_pos(dataset, args):
    """scan pos dataset and output or display roi and target cell."""
    slides_id = sorted(list(dataset.slides.keys()))
    np.random.seed(233)
    np.random.shuffle(slides_id)
    
    total = len(slides_id)
    dh, dw = 5120, 5120

    coco_train = COCODataset()
    coco_valid = COCODataset()
    coco_train.add_category('pos')
    coco_valid.add_category('pos')
    train_num = int(len(slides_id) * 0.9)

    for idx, slide_id in enumerate(slides_id):
        coco = coco_train if idx < train_num else coco_valid
        print(colorful_str.log('[{}/{}] running on {}'.format(
            idx, total, slide_id)))

        if slide_id in dataset.slides and slide_id in dataset.labels:
            np.random.shuffle(dataset.labels[slide_id])
            for item in dataset.labels[slide_id]:
                if item['class'] != 'roi':
                    continue

                x, y, w, h = item['x'], item['y'], item['w'], item['h']

                res = []
                for target in dataset.labels[slide_id]:
                    if target['class'] == 'roi':
                        continue
                    ix, iy = target['x'], target['y']
                    iw, ih = target['w'], target['h']
                    jx = ix + iw
                    jy = iy + ih
                    ix = max(ix, x)
                    iy = max(iy, y)
                    jx = min(jx, x + w)
                    jy = min(jy, y + h)
                    if jx > ix and jy > iy:
                        res.append([ix, iy, jx, jy])

                for sx in range(x, x + w, dw//2):
                    for sy in range(y, y + h, dh//2):
                        filename = '{}_({}-{}-{}-{})_{}.jpg'.format(
                                slide_id, sx, sy, dw, dh, item['class'])
                        slide_w, slide_h = dataset.get_shape(slide_id)

                        # print(slide_w, slide_h, sx, sy, sx+dw, sy+dh)
                        if sx+dw >= slide_w or sy+dh >= slide_h:
                            continue
                        img = dataset.get_roi(slide_id, sx, sy, dw, dh)
                        # print(img.shape)
                        dst = np.copy(img)

                        dirname = os.path.join(args.output, 'images', slide_id)
                        if not os.path.isdir(dirname):
                            os.makedirs(dirname)

                        category_id = coco.query_category('pos')
                        marked = False
                        for target in res:
                            ix, iy, jx, jy = target
                            ix = max(ix, sx)
                            iy = max(iy, sy)
                            jx = min(jx, sx + dw)
                            jy = min(jy, sy + dh)

                            if ix < jx and iy < jy and jx - ix > 60 and jy - iy > 60:
                                cv2.rectangle(dst, (ix-sx, iy-sy), (jx-sx, jy-sy),
                                              (0, 0, 255), 3)
                                coco.add_annotation(
                                    [ix-sx, iy-sy, jx-ix, jy-iy],
                                    coco.images_id, category_id)
                                marked = True
                                print("Add annotation in {}".format(coco.images_id))
                        if marked:
                            print('  display {}'.format(filename))
                            # dst = cv2.resize(dst, (dst.shape[1] >> 2, dst.shape[0] >> 2))
                            # cv2.imshow('img', dst)
                            # cv2.waitKey(0)
                            path = os.path.join(dirname, filename)
                            cv2.imwrite(path, img)
                            height, width = img.shape[:2]
                            print(colorful_str.log('[{}]\timg saved in {}'.format(coco.images_id, path)))
                            _ = coco.add_image(path, height, width, absolute_path=True)


    dirname = os.path.join(args.output, 'annotations')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    train_path = os.path.join(dirname, 'train.json')
    valid_path = os.path.join(dirname, 'valid.json')
    coco_train.dump(train_path)
    coco_valid.dump(valid_path)
        

def main():
    args = get_args()
    pos_path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/pos'
    neg_path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/neg'
    label_path = '/home/gryhomshaw/ssd1g/xiaoguohong/classification/data/tianchi/labels'
    pos_dataset = dataset.Dataset(pos_path, label_path)
    # neg_dataset = dataset.Dataset(neg_path)

    scan_pos(pos_dataset, args)


if __name__ == '__main__':
    main()


