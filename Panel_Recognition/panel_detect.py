import cv2
import torch
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI
import time
from string import digits
import re
import imutils

class yolov5():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_number.pt')
        self.floor = 0
        self.prob = 0
        # self.dir = './samples/'
    def get_floor(self, image):
        # result = self.model(Image.open(self.dir+filename))
        result = self.model(image)
        try:
            self.floor = result.xyxy[0][0].unsqueeze(1)[-1][0]
            self.prob = result.xyxy[0][0].unsqueeze(1)[-2][0]
        except:
            self.floor = 0
            self.prob = 0
        return result, int(self.floor), self.prob

class OCR():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best_panel.pt')
        self.api = PyTessBaseAPI()
        self.floor = 0
        self.prob = 0
        # self.dir = './samples/'
    def get_bbox(self, image):
        result = self.model(image)
        xmin = int(result.xyxy[0][0][0])+80
        xmax = int(result.xyxy[0][0][2])-50
        ymin = int(result.xyxy[0][0][1])+10
        ymax = int(result.xyxy[0][0][3])-10
        return xmin, xmax, ymin, ymax
    def get_floor_api(self, image):
        # image = Image.open(self.dir+filename)
        xmin, xmax, ymin, ymax = self.get_bbox(image)
        bbox_img = image.crop((xmin, ymin, xmax, ymax))
        bbox_img = self.preprocessing(bbox_img)
        self.floor = re.sub("\D","",self.ocr_api(bbox_img))
        try:
            self.floor = int(self.floor)
        except:
            pass
        return self.floor, bbox_img
    def ocr_api(self, bbox_img):
        self.api.SetVariable('tessedit_char_whitelist', digits)
        self.api.SetImage(bbox_img)
        # return self.api.AllWordConfidences()
        return self.api.GetUTF8Text()
    def preprocessing(self, bbox_img):
        # kernels
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        kernel7 = np.ones((7, 7), np.uint8)
        kernel9 = np.ones((9, 9), np.uint8)
        kernel11 = np.ones((11, 11), np.uint8)
        kernel15 = np.ones((15, 15), np.uint8)
        kernel19 = np.ones((19, 19), np.uint8)

        """
            Numpy array로 변환 (w, h)
            : Red Channel만 사용
        """
        numpy_img = np.asarray(bbox_img, dtype='uint8')[:,:,0]

        # 상하좌우
        numpy_img = cv2.copyMakeBorder(numpy_img,10,10,80,50,cv2.BORDER_CONSTANT,value=0)




        """
            Binary Thresholding
        """
        _, numpy_img = cv2.threshold(numpy_img, 120, 255, cv2.THRESH_BINARY_INV)
        # _, numpy_img = cv2.threshold(numpy_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # numpy_img = cv2.adaptiveThreshold(numpy_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        # numpy_img = cv2.resize(numpy_img, dsize=(h, w), interpolation=5)
        
        """
            Edge Operator
        """


        w, h = numpy_img.shape[0], numpy_img.shape[1]
        numpy_img = cv2.resize(numpy_img, dsize=(h*3, w*3), interpolation=4)

        numpy_img = cv2.Canny(numpy_img, 1, 250)

        """
            Morpological Filters
        """
        # numpy_img = cv2.morphologyEx(numpy_img, cv2.MORPH_DILATE, kernel5)
        # numpy_img = cv2.morphologyEx(numpy_img, cv2.MORPH_ERODE, kernel5)

        # numpy_img = cv2.morphologyEx(numpy_img, cv2.MORPH_CLOSE, kernel3)
        # w, h = numpy_img.shape[0], numpy_img.shape[1]
        # numpy_img = cv2.resize(numpy_img, dsize=(h*3, w*3), interpolation=4)
        # numpy_img = cv2.morphologyEx(numpy_img, cv2.MORPH_OPEN, kernel3)
        # numpy_img = cv2.morphologyEx(numpy_img, cv2.MORPH_CLOSE, kernel3)


        numpy_img = cv2.dilate(numpy_img, None, iterations=4)
        numpy_img = cv2.erode(numpy_img, None, iterations=4)

        # _, numpy_img = cv2.threshold(numpy_img, 50, 255, cv2.THRESH_BINARY_INV)
        
        

        """
            Interpolation
            : 3베
        """
        # w, h = numpy_img.shape[0], numpy_img.shape[1]
        # numpy_img = cv2.resize(numpy_img, dsize=(h*3, w*6), interpolation=4)
        

        # numpy_img = cv2.morphologyEx(numpy_img, cv2.MORPH_CLOSE, kernel7, iterations=3)
        # numpy_img = cv2.resize(numpy_img, dsize=(h, w), interpolation=5)
        
        # PIL Image로 변환

        

        bbox_img = Image.fromarray(numpy_img, "L")
        return bbox_img