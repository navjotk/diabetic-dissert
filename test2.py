'''
Created on 14 Jul 2015

@author: navjotkukreja
'''
import os
from scipy import misc
from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import data, color, exposure
import math
from tc import TerminalController
import sys
from sklearn.svm import SVC
import csv
from sklearn.grid_search import GridSearchCV
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import confusion_matrix
import optunity
import optunity.metrics

def get_hog_fd(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    return fd

def chunks(lst,n):
    return [ lst[i::n] for i in xrange(n) ]
  
def print_replace(text):
    sys.stdout.write(text)
    sys.stdout.flush()
    sys.stdout.write(term.BOL + term.CLEAR_EOL)

def load_file(file):
    with open(file, 'rb') as csvfile:
        filereader = csv.reader(csvfile)
        lines = []
        for row in filereader:
            lines.append(row)
    return lines

def get_class(name):
    name_ar = name.split('.')
    name=name_ar[0]
    global labels
    if len(labels)==0:
        labels = load_file(label_file)
    
    for entry in labels:
        if entry[0]==name:
            if entry[1]=='0':
                return 0
            else:
                return 1
    return -1

def process_image(image_path):
    global imgIndex
    imgIndex+=1
    img_class = get_class(image_path)
    print_replace("Processing image "+str(imgIndex)+" with name "+image_path+" and class "+str(img_class))
    image = misc.imread('images/'+image_path)
    image_gray = rgb2gray(image)
    hog_fd = get_hog_fd(image_gray)
    return (hog_fd, img_class)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



    
num_images=5000
n_fold_cv = 10
label_file = 'trainLabels.csv'

term = TerminalController()
contents = os.listdir("images")

print "Found "+str(len(contents))+" images, using "+str(num_images)

n_folds = chunks(contents[0:num_images], n_fold_cv)

labels = []
train_paths = []
test_paths = []
selected_fold = 1
for i in range(0, len(n_folds)):
    if i==selected_fold:
        test_paths=n_folds[i]
    else:
        train_paths+=n_folds[i]
        
print "Processing training images"
imgIndex = 0
train = Parallel(n_jobs=6)(delayed(process_image)(i) for i in train_paths)    
print "Processing test images"
test = Parallel(n_jobs=6)(delayed(process_image)(i) for i in test_paths)


train_t = zip(*train)
train_x = np.array(train_t[0])
train_y = list(train_t[1])

test_t = zip(*test)
test_x = np.array(test_t[0])
test_y = list(test_t[1])


space = {'kernel': {'linear': {'C': [0, 2]},
                    'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                    'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                    }
         }
print "Training model"
cv_decorator = optunity.cross_validated(x=train_x, y=train_y, num_folds=n_fold_cv)

def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=C)
    elif kernel == 'poly':
        model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, C=C, gamma=10 ** logGamma)
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    model.fit(x_train, y_train)
    return model

def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, logGamma):
    model = SVC(C=C, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    auc = optunity.metrics.roc_auc(y_test, decision_values)
    return auc

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
    model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
#svm_rbf_tuned_auroc = cv_decorator(svm_rbf_tuned_auroc)
#print svm_rbf_tuned_auroc(C=1.0, logGamma=0.0)

optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)
print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)

