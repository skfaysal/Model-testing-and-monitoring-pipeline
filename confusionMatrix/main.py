import urllib.request
import json
import ast
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from cm import evaluate
from cm import saveImage

url = "http://52.146.29.32/patientimage/?"

save_path = str(os.getcwd()) + '/missclass images/'

img_list, img_label, predict, miss_img_id, data, miss_pred, miss_label, img_link_list = evaluate(url)

# save miss class images
saveImage(img_link_list, save_path)

# Confusion marix
cm = confusion_matrix(img_label, predict)
labels = ['No-DR', 'DR']
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
ax.figure.savefig('confusion matrix for 43 patients.png')
plt.close()

cr = classification_report(img_label, predict, target_names=labels)
print(cr)
