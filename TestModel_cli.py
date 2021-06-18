# run on terminal
"""
python3 TestModel_cli.py --drmodel models/b5_newpreprocessed_full_fold4.h5
--lfmodel models/model_binary_right_leaft_retina.h5
--imgdata eyepacs_train --savepath output/
"""

from glob import glob
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from cv2 import cvtColor, COLOR_BGR2RGB
from numpy import array
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
from cv2 import cvtColor, COLOR_BGR2RGB
from keras_efficientnets import EfficientNetB4
import argparse

class Preprocess:
    def crop_image_from_gray(self, img):
        if img.ndim == 2:
            mask = img > self.tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > self.tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:  # image is too dark so that we crop out everything,
                return img  # return original image
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

    # for taking input image and do some preprocessing
    def load_gauss(self, im_path):  # load_ben_color
        img = cvtColor(array(Image.open(im_path)), COLOR_BGR2RGB)
        img = self.crop_image_from_gray(img)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.sigmaX), -4, 128)
        return img

    def load_left_right_keras(self, im_path):
        img = Image.open(im_path)
        img = img.convert('RGB')
        target_size = (self.IMG_SIZE, self.IMG_SIZE)
        img = img.resize(target_size, Image.NEAREST)
        # img = cv2.resize(img, target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img / 255, axis=0)
        img = img.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3)
        return img

    def keras_preprocessing(self, im_path, *p):
        if p == 1:
            img = Image.open(im_path)
        else:

            img = self.load_gauss(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_size = (self.IMG_SIZE, self.IMG_SIZE)
        img = cv2.resize(img, target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img / 255, axis=0)
        img = img.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3)
        return img


class Data(Preprocess):
    # Data needed to be preprocessed
    def import_data(self, im_path, csv_path):
        data_csv = csv_path
        base = os.path.basename(im_path)
        img_name = os.path.splitext(base)[0]
        img = self.keras_preprocessing(im_path)
        label = data_csv.loc[data_csv.image == img_name, 'level'].values
        if label <= 1:
            label = 0
        else:
            label = 1
        return img, label

    # already preprocessed data
    def import_data_preprocessed(self, im_path, csv_path):
        data_csv = csv_path
        base = os.path.basename(im_path)
        img_name = os.path.splitext(base)[0]
        img = self.keras_preprocessing(im_path,1)
        label = data_csv.loc[data_csv.image == img_name, 'level'].values
        if label <= 1:
            label = 0
        else:
            label = 1
        return img, label

    def labelextract_left_right(self, im_path):
        base = os.path.basename(im_path)
        filename = os.path.splitext(base)[0]
        label = filename.split('_')[1]
        if label == 'right':
            label = 1
        else:
            label = 0
        img = self.load_left_right_keras(im_path)
        return img, label


class Prediction(Data):
    def __init__(self, model4, model_lf, tol, IMG_SIZE, sigmaX):
        self.model0 = model4
        self.model_lf = model_lf
        # self.path = im_path
        # self.cs_path = cs_path
        self.tol = tol
        self.IMG_SIZE = IMG_SIZE
        self.sigmaX = sigmaX
        # self.labels_array = np.array(1)

    # images needed to be processed
    def save_image(self, im_path, predicted, actual, img_count, save_path):
        img = self.load_gauss(im_path)
        cv2.imwrite(str(save_path) + 'Image ' + str(img_count) +
                    ' predicted ' + str(predicted) + 'Actual ' + str(actual) + '.png', img)

    # already processed or raw images
    def save_image_preprocessed(self, im_path, predicted, actual, img_count, save_path):
        img = Image.open(im_path)
        cv2.imwrite(str(save_path) + 'Image ' + str(img_count) +
                    ' predicted ' + str(predicted) + 'Actual ' + str(actual) + '.png', np.array(img))

    # already processed or raw images
    def save_image_left_right(self, im_path, predicted, actual, img_count, save_path):
        img = Image.open(im_path)
        img = img.convert('RGB')
        target_size = (self.IMG_SIZE, self.IMG_SIZE)
        img = img.resize(target_size, Image.NEAREST)
        cv2.imwrite(str(save_path) + 'Image ' + str(img_count) +
                    ' predicted ' + str(predicted) + 'Actual ' + str(actual) + '.png', np.array(img))

    # images needed to be processed
    def prediction_raw(self, im_path, cs_path, img_count, save_path):
        img, lbl = self.import_data(im_path, cs_path)

        prediction = self.model0.predict(img, batch_size=16)
        y = np.argmax(prediction)
        if y <= 1:
            y = 0
        else:
            y = 1

        if y != lbl:
            self.save_image(im_path, [y], lbl, img_count, save_path)
        return lbl, y

    def prediction_preprocessed(self, im_path, cs_path, img_count, save_path):
        img, lbl = self.import_data(im_path, cs_path)
        prediction = self.model0.predict(img, batch_size=16)
        y = np.argmax(prediction)
        if y <= 1:
            y = 0
        else:
            y = 1

        if y != lbl:
            self.save_image_preprocessed(im_path, [y], lbl, img_count, save_path)
        return lbl, y

    # for left_right model
    def predicton_left_right(self, im_path, img_count, save_path):
        img, label = self.labelextract_left_right(im_path)
        prediction = self.model_lf.predict(img, batch_size=16)
        if prediction > 0.9:
            prediction = 1
        else:
            prediction = 0
        if prediction != label:
            self.save_image_left_right(im_path, [prediction], lbl, img_count, save_path)
        return label, prediction


# Models
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--drmodel", required=False,help="path to input dr model")
ap.add_argument("-lf", "--lfmodel", required=False,help="path to input left_right model")
ap.add_argument("-d", "--imgdata", required=True,help="path to input image_data")
ap.add_argument("-c", "--csdata", required=False,help="path to input csv_data")
ap.add_argument("-f", "--folder", required=False,help="path to input folder")
ap.add_argument("-s", "--savepath", required=True,help="path to save misclassified data")
ap.add_argument("-t", "--tol", required=False,help="hyper parameter tol")
ap.add_argument("-i", "--img_size", required=False,help="hyper parameter img_size")
ap.add_argument("-sig", "--sigmaX", required=False,help="hyper parameter sigmaX")
args = vars(ap.parse_args())

model4 = load_model(args['drmodel'])
model_lf = load_model(str(args['lfmodel']))

# image_data
img_data__path = glob(str(args['imgdata'])+'/*')
# label
#cs_path = pd.read_csv(args['csdata'])
#folder_path = args['folder']
save_path = args['savepath']

# hyper parameters
tol = 7
IMG_SIZE = int(args['img_size'])
sigmaX = 10

# Declare object
obj = Prediction(model4,model_lf, tol, IMG_SIZE, sigmaX)

img_list = list()
img_label = list()
img_count = 0
for im_path in img_data__path:
    img_count += 1
    if img_count > 300:
        break
    pre, lbl = obj.predicton_left_right(im_path, img_count, save_path)
    img_list.append(pre)
    img_label.append(lbl)

if __name__ == '__main__':
    img_list = np.array(img_list)
    img_label = np.array(img_label)
    predict = img_list.reshape(-1, 1)
    print(confusion_matrix(img_label, predict))
    print(classification_report(img_label, predict))
