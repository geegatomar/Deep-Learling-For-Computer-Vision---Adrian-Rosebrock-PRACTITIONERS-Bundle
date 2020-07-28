import cv2
import imutils

class AspectAwarePreprocessor: 

    def __init__(self, height, width, inter=cv2.INTER_AREA):
        self.height = height                # the height and width that we want after resizing
        self.width = width
        self.inter = inter
    
    def preprocess(self, image):
        (h, w) = image.shape[:2]        # the current height and width of the image 
        dh, dw = 0, 0
    
        if h < w:            # then resize h and crop along w
            image = imutils.resize(image, height = self.height, inter=self.inter)
            dw = int((image.shape[1] - self.width) / 2.0)
        else:
            image = imutils.resize(image, width = self.width, inter=self.inter)
            dh = int((image.shape[0] - self.height) / 2.0)
        
        (h, w) = image.shape[:2]
        
        image = image[dh : h - dh, dw : w - dw]
    
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)



