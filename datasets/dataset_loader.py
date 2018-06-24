import numpy
import cv2
import os


class DatasetLoader:
    def __init__(self, pre_processors=None):
        if isinstance(pre_processors, list):
            self.pre_processors = pre_processors
        else:
            self.pre_processors = []

    def load(self, image_paths, verbose=False):
        data = []
        labels = []

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]  # assume image_path format: dataset/{class}/{image}.jpg
            for pre_processor in self.pre_processors:
                image = pre_processor.pre_precess(image)
                data.append(image)
                labels.append(label)

            if verbose:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

        return numpy.array(data), numpy.array(labels)
