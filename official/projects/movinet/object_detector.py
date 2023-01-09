import math

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

# from utils import config
import config
import sys 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model', type=str, default = 'efficientdet', help='od_model')
parser.add_argument('--metric', type=str, default = 'mae', help='od_evaluation_metric')
args = parser.parse_args()

sys.path.append('./yolov5')
    
if args.model == 'yolo':
    from detect import run

def run_efficientdet(clip, od_model, threshold=0.0):
    img = clip[:, 0, ...]
    result = od_model(img)
    count = [1 if c == 1 and s >= threshold else 0 for c, s in zip(result['detection_classes'][0], result['detection_scores'][0]) ]

    return count
    

def run_yolo(weights, source, imgsz=(224,224), conf_thres=0.25, max_det=100, classes=[0]):
    source = source[0]
    conf_list, cls_list = run(weights = weights, source = source, imgsz = imgsz, conf_thres = conf_thres,
        max_det=max_det, classes=classes)
    count = [1 if c == 0 else 0 for c in cls_list]

    return count

def people_exist(count):
    if sum(count) >= 1:
        return True
    else:
        return False 


def cal_confusion_matrix(pred, label, fn, fp, tn, tp):

    pred = people_exist(pred)
    label = label >= 1

    if pred == label:
        if pred == True:
            tp += 1
        else:
            tn += 1
    else:
        if pred == True:
            fp += 1
        else:
            fn += 1

    return fn, fp, tn, tp


def people_exist2(count):
    if sum(count) >= 2:
        return True
    else:
        return False 


def cal_confusion_matrix2(pred, label, fn, fp, tn, tp):

    pred = people_exist2(pred)
    label = label >= 2

    if pred == label:
        if pred == True:
            tp += 1
        else:
            tn += 1
    else:
        if pred == True:
            fp += 1
        else:
            fn += 1

    return fn, fp, tn, tp


def cal_mae(pred, label):
    pred = sum(pred)

    return abs(pred-label)





