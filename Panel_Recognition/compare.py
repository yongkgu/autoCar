import cv2
from sympy import fps
import torch
from PIL import Image
from torchinfo import summary
import numpy as np
from panel_detect import yolov5, OCR
import constant as const
from string import digits
from tesserocr import PyTessBaseAPI
import time
from imutils import contours
import argparse

"""
    < Video Sample Test >
    샘플 영상 : 11s / 약 30fps
    Model 처리 속도 : 약 7.5fps
    설정 FPS : 5fps
"""
def show_video():
    capture = cv2.VideoCapture(const.SAMPLE_VIDEO_URL)
    while capture.isOpened():
        run, frame = capture.read()
        if not run:
            break
        if int(capture.get(1))%const.FPS==0:
            cv2.imshow('video', frame)
            if cv2.waitKey(30)&0xFF==ord('q'):
                break
            cv2.destroyAllWindows()

def YOLOv5():
    capture = cv2.VideoCapture(const.SAMPLE_VIDEO_URL)
    yolo = yolov5()
    while capture.isOpened():
        run, frame = capture.read()
        if not run:
            break
        if int(capture.get(1))%const.FPS==0:
            start = time.time()
            # Red Channel만 적용
            _, floor, _ = yolo.get_floor(frame[:,:,2])
            end = time.time()

            cv2.imshow('video', frame)
            if cv2.waitKey(30)&0xFF==ord('q'):
                break
            cv2.destroyAllWindows()

            print("현재 층:", const.FLOORS_NUM[floor], "층", end='')
            print("| 인식 시간:", round(end-start, 4), "초")
            

def OCRwithYOLO():
    capture = cv2.VideoCapture(const.SAMPLE_VIDEO_URL)
    ocr = OCR()
    while capture.isOpened():
        run, frame = capture.read()
        if not run:
            break
        if int(capture.get(1))%const.FPS==0:
            start = time.time()
            floor, _ = ocr.get_floor_api(Image.fromarray(frame))
            end = time.time()

            cv2.imshow('video', frame)
            if cv2.waitKey(30)&0xFF==ord('q'):
                break
            cv2.destroyAllWindows()

            print("현재 층:", floor, "층", end='')
            print("| 인식 시간:", round(end-start, 4), "초")

"""
    < Image Sample Test >
"""
# Yolov5(Number)
def YOLOv5_image():
    image = cv2.imread('./samples/sample5.png', cv2.IMREAD_COLOR)
    result, floor, prob = yolov5().get_floor(image)
    print(const.FLOORS_NUM[floor], '층')
    print('확률', prob)
    result.show()

# OCR + Yolov5(Panel)
def OCRwithYOLO_image():
    image = Image.open('./samples/sample4.png')
    floor, bbox_img = OCR().get_floor_api(image)
    print(floor)
    bbox_img.show()

if __name__=="__main__":
    parser  = argparse.ArgumentParser(description="Elevator Floor Recognition")
    parser.add_argument('--model', type=str, default='Yolo', choices=['Yolo','OCR'], help='Select Yolo or OCR')
    parser.add_argument('--data', type=str, default='Video', choices=['Video','Image'], help='Select Video data or Image data')
    args = parser.parse_args()

    print('Model :', args.model)
    print('Data :', args.data)

    if (args.model=='Yolo') & (args.data=='Video'):
        YOLOv5()
    elif (args.model=='OCR') & (args.data=='Video'):
        OCRwithYOLO()
    elif (args.model=='Yolo') & (args.data=='Image'):
        YOLOv5_image()
    elif (args.model=='OCR') & (args.data=='Image'):
        OCRwithYOLO_image()
    