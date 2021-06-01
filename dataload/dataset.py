"""load pos and neg data."""

import json
import os

from eic_utils import procedure
from .kfbreader import kfbReader

slides_extensions = ['kfb']
labels_extensions = ['json']

class Dataset(object):
    """dataset for slides."""

    def __init__(self, slides_dir_path, labels_dir_path=None, scale=20):

        self.setup_slides(slides_dir_path, labels_dir_path, scale=scale)

    def setup_slides(self, slides_dir_path, labels_dir_path, scale=20):
        """locate all slides and label configs."""
        
        with procedure('setup slides') as p:
            self.slides_path = {}
            self.slides = {}
            for dirpath, dirnames, filenames in os.walk(slides_dir_path):
                for filename in filenames:
                    if filename.split('.')[-1].lower() in slides_extensions:
                        slide_id = filename.split('.')[0]
                        filepath = os.path.join(dirpath, filename)
                        self.slides_path[slide_id] = filepath
                        reader = kfbReader.reader()
                        kfbReader.reader.ReadInfo(reader, filepath, scale, True)
                        self.slides[slide_id] = reader


            self.labels_path = {}
            self.labels = {}
            if labels_dir_path is not None:
                for dirpath, dirnames, filenames in os.walk(labels_dir_path):
                    for filename in filenames:

                        if filename.split('.')[-1].lower() in labels_extensions:
                            slide_id = filename.split('.')[0]
                            filepath = os.path.join(dirpath, filename)
                            self.labels_path[slide_id] = filepath
                            with open(filepath, 'r') as f:
                                self.labels[slide_id] = json.load(f)

                # assert (sorted(self.labels_path.keys()) == 
                #         sorted(self.slides_path.keys())), 'slides ID not match!'

            p.add_log('{} slides and {} labels loaded.'.format(
                    len(self.slides_path), len(self.labels_path)))
    
    def get_roi(self, slide_id, x, y, w, h, scale=20):
        """fetch roi in specific slide."""
        slide = self.slides[slide_id]
        slide_w, slide_h = self.get_shape(slide_id)
        x = max(min(x, slide_w - w - 50), 0)
        y = max(min(y, slide_h - h - 50), 0)
        # print(x, y)
        img = slide.ReadRoi(x, y, w, h, scale)
        return img

    def get_shape(self, slide_id):
        """fetch shape (width, height) of specific slide."""
        slide = self.slides[slide_id]
        w = slide.getWidth()
        h = slide.getHeight()
        return [w, h]


def main():
    pos_path = './data/pos/'
    neg_path = './data/neg/'
    label_path = './data/labels'

    pos_dataset = Dataset(pos_path, label_path)
    neg_dataset = Dataset(neg_path)

    


if __name__ == '__main__':
    main()

