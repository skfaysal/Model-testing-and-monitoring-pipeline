import cv2
from glob import glob
import io
import cv2
import shutil
import random
import warnings
import tensorflow
import numpy as np
import pandas as pd
import seaborn as sns
import os, sys
from PIL import Image
import warnings
from cv2 import cvtColor, COLOR_BGR2RGB
from numpy import array
# import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
from cv2 import cvtColor, COLOR_BGR2RGB
from keras_efficientnets import EfficientNetB4

model_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/test model/models/b5_newpreprocessed_full_fold0.h5'
model0 = load_model(model_path)
model_lf = load_model(
    '/home/faysal/PycharmProjects/DiabeticRetnopathy/test model/models/model_binary_right_leaft_retina.h5')

img_data__path = glob('/home/faysal/PycharmProjects/DiabeticRetnopathy/messidor/Messidor-2/img_all/*')
cs_path = "/home/faysal/PycharmProjects/DiabeticRetnopathy/messidor/Messidor-2/allCSV/messidorFull.csv"

save_path = '/home/faysal/PycharmProjects/DiabeticRetnopathy/missclass_30k/'

c = 0
img_list = list()
img_label = list()
img_count = 0

for im_path in img_data__path:
    img_count += 1
    if img_count >= 30:
        break


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
        def load_gauss(self):  # load_ben_color
            img = cvtColor(array(Image.open(self.path)), COLOR_BGR2RGB)
            img = self.crop_image_from_gray(img)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.sigmaX), -4, 128)
            return img

        def load_left_right_keras(self):
            img = Image.open(self.path)
            img = img.convert('RGB')
            target_size = (self.IMG_SIZE, self.IMG_SIZE)
            img = img.resize(target_size, Image.NEAREST)
            # img = cv2.resize(img, target_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img / 255, axis=0)
            img = img.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3)
            return img

        def keras_preprocessing(self, *p):
            if p == 1:

                img = cvtColor(array(Image.open(self.path)), COLOR_BGR2RGB)
            else:
                img = self.load_gauss()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target_size = (self.IMG_SIZE, self.IMG_SIZE)
            img = cv2.resize(img, target_size)
            plt.imshow(img)
            img = image.img_to_array(img)
            img = np.expand_dims(img / 255, axis=0)
            img = img.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3)
            return img


    class Data(Preprocess):
        # already preprocessed data
        def import_data_preprocessed(self, im_path,csv_path):
            data_csv = pd.read_csv(csv_path)
            base = os.path.basename(im_path)
            img_name = os.path.splitext(base)[0]
            img = self.keras_preprocessing(1)
            label = data_csv.loc[data_csv.image == img_name, 'level'].values
            return img, label

        # Data needed to be preprocessed
        def import_data(self, im_path,csv_path):
            data_csv = pd.read_csv(csv_path)
            base = os.path.basename(im_path)
            img_name = os.path.splitext(base)[0]
            img = self.keras_preprocessing()
            label = data_csv.loc[data_csv.image == img_name, 'level'].values
            return img, label

        def labelextract_left_right(self):
            base = os.path.basename(im_path)
            filename = os.path.splitext(base)[0]
            label = filename.split('_')[1]
            if label == 'right':
                label = 1
            else:
                label = 0
            img = self.load_left_right_keras()
            return img, label


    class Prediction(Data):
        def __init__(self):
            self.model0 = model0
            self.model_lf = model_lf
            self.path = im_path
            self.cs_path = cs_path
            self.tol = 7
            self.IMG_SIZE = 224
            self.sigmaX = 10
            self.save_path = save_path
            self.labels_array = np.array(1)

        def save_image_left_right(self, predicted, actual):  # already processed or raw images
            img = Image.open(self.path)
            img = img.convert('RGB')
            target_size = (self.IMG_SIZE, self.IMG_SIZE)
            img = img.resize(target_size, Image.NEAREST)
            cv2.imwrite(str(save_path)+
                        'Image ' + str(img_count) + ' predicted ' + str(predicted) + 'Actual ' + str(actual) + '.png',
                        np.array(img))

        def save_image_preprocessed(self, predicted, actual):  # already processed or raw images
            img = cvtColor(array(Image.open(self.path)), COLOR_BGR2RGB)
            cv2.imwrite(str(save_path)+
                        'Image ' + str(img_count) + ' predicted ' + str(predicted) + 'Actual ' + str(actual) + '.png',
                        np.array(img))

        def save_image(self, predicted, actual):  # images needed to be processed
            imag = self.load_gauss()
            cv2.imwrite(str(save_path)+
                        '/predicted ' + str(predicted) + 'Actual ' + str(actual) + '.png', imag)

        # already processed or raw images
        def prediction_preprocessed(self):
            img, lbl = self.import_data_preprocessed(self.path, self.cs_path)

            prediction = self.model0.predict(img, batch_size=16)
            y = np.argmax(prediction)
            if y != lbl:
                self.save_image_preprocessed([y], lbl)
            return lbl, y

        # images needed to be processed
        def prediction(self):
            img, lbl = self.import_data(self.path, self.cs_path)

            prediction = self.model0.predict(img, batch_size=16)
            y = np.argmax(prediction)
            if y != lbl:
                self.save_image([y], lbl)
            return lbl, y

        # for left_right model
        def predicton_left_right(self):
            img, label = self.labelextract_left_right()
            prediction = self.model_lf.predict(img, batch_size=16)
            if prediction > 0.9:
                prediction = 1
            else:
                prediction = 0
            if prediction != label:
                self.save_image_left_right([prediction], label)
            return label, prediction


    obj = Prediction()
    pre, lbl = obj.prediction_preprocessed()
    img_list.append(pre)
    img_label.append(lbl)

if __name__ == '__main__':
    img_list = np.array(img_list)
    img_label = np.array(img_label)
    predict = img_list.reshape(-1, 1)
    print(confusion_matrix(img_label, predict))
    print(classification_report(img_label, predict))
