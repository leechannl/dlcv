import cv2


class PreProcessor:
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def pre_process(self):
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
