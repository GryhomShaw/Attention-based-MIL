"""Builds a coco-format dataset for detection tasks."""

import json
import os

from eic_utils.base import colorful_str


class COCODataset(object):
    """A dataset consists of images, categories and annotations."""

    def __init__(self, warning=False):
        self.warning = warning
        self.categories_id, self.images_id, self.annotations_id = 0, 0, 0
        self.categories, self.images, self.annotations = [], [], []
        self.categories_to_ids = {}

    def add_category(self, name):
        """Creates a new category."""

        self.categories_id += 1
        self.categories.append({'id': self.categories_id, 'name': name})
        self.categories_to_ids[name] = self.categories_id
        return self.categories_id

    def add_image(self, path, height, width, absolute_path=False):
        """Adds a new image."""

        self.images_id += 1
        self.images.append({
            'id': self.images_id,
            'file_name': os.path.abspath(path) if absolute_path else path,
            'height': height,
            'width': width,
        })
        return self.images_id

    def add_annotation(self, bbox, image_id, category_id):
        """Adds a new annotation.

        Args:
            bbox: A list of 4 integers, [min_x, min_y, delta_x, delta_y].
                  x stands for width coordinate, while y for height.
            image_id: The index of image which this annotation belongs to.
            category_id: The index of category which this annotation represents.
        """

        if self.warning and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            msg = colorful_str.wrn(
                '(#y)bbox format warning.(#)',
                bbox,
                'the last two items are larger than the first two.',
                'bbox consists of a list of 4 integer,',
                '[min_x, min_y, (#r)delta(#)_x, (#r)delta(#)_y].',
                '(#y)x(#) stands for (#y)width(#) coordinate,',
                'while (#y)y(#) for (#y)height(#).')
            print(msg)
        else:
            self.warning = False

        self.annotations_id += 1
        self.annotations.append({
            'id': self.annotations_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'iscrowd': 0,
            'area': bbox[2] * bbox[3],
        })
        return self.annotations_id
    
    def query_category(self, name):
        """Queries the category id of a specific category."""

        return self.categories_to_ids[name]

    def dump(self, path=None):
        """Saves coco datasets."""
        data = {
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.categories,
        }
        if path is None:
            return json.dumps(data)
        #print(path)
        with open(path, 'w') as f:
            json.dump(data, f)


def __test():
    coco = COCODataset()
    img1 = coco.add_image('1.jpg', 1024, 1024, absolute_path=True)
    img2 = coco.add_image('1.jpg', 1024, 1024)
    category1 = coco.add_category('hi')
    coco.add_annotation([1, 1, 2, 2], img1, category1)
    coco.add_annotation([400, 200, 10, 10], img2, category1)
    print(coco.dump())

if __name__ == '__main__':
    __test()
