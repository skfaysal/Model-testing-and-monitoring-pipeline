from flask import request
import urllib.request
import json
import ast
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


def evaluate(url):

    with urllib.request.urlopen(url) as url:
        s = url.read()
        print(type(s))

        data = json.loads(s.decode('utf-8'))
        c = 0
        pred_list = []
        lebel_list = []
        miss_img_id = []
        miss_pred = []
        miss_label = []
        for i in range(len(data)):
            if data[i]["RPathy"] is not None:
                if 'null' not in data[i]["RPathy"]:
                    if data[i]["DrPositive"] is True:
                        pred_list.append(1)
                    else:
                        pred_list.append(0)

                    if 'R0' in data[i]["RPathy"]:
                        lebel_list.append(0)
                    else:
                        lebel_list.append(1)

                    if pred_list[-1] != lebel_list[-1]:
                        miss_pred.append(pred_list[-1])
                        miss_label.append(lebel_list[-1])
                        miss_img_id.append(data[i]["id"])

        enc_pred_list = pred_list

        enc_lebel_list = lebel_list

        img_list = np.array(enc_pred_list)
        img_label = np.array(enc_lebel_list)
        predict = img_list.reshape(-1, 1)

        # get miss class image links
        img_link_list = []
        for i in range(len(data)):
            if data[i]["id"] in miss_img_id:
                for j in range(4):
                    img_link_list.append(data[i]["Image"+str(j)])

        return img_list, img_label, predict, miss_img_id, data, miss_pred, miss_label, img_link_list


def saveImage(img_link_list,save_path):
    img_count=0
    for i in img_link_list:

        try:

            with urllib.request.urlopen(i) as url:
                s = url.read()
                # use numpy to construct an array from the bytes
                x = np.fromstring(s, dtype='uint8')

                # decode the array into an image
                img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
                print (img.shape)

                cv2.imwrite(str(save_path) + 'Image ' + str(img_count) +'.png', img)

                img_count+=1
        except:
            continue

