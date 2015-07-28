'''
Created on 22 Jul 2015

@author: navjotkukreja
'''
from dataloader import dataloader
import sys
from skimage.io import imread
import skimage.filters
import skimage.morphology
from matplotlib import pyplot as plt
import pylab
import numpy as np
import cv2



num_images=1
n_fold_cv = 20
label_file = 'trainLabels.csv'
image_directory = 'images'


#export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib:/opt/intel/composer_xe_2015.2.132/compiler/lib:/opt/intel/composer_xe_2015.2.132/mkl/lib


#Make sure we are running inside a virtualenv
if not hasattr(sys, 'real_prefix'):
    print "No virtualenv set. Please activate a virtualenv before running."
    sys.exit()

loader = dataloader(label_file, image_directory, num_images)
data = loader.get_data()

image_paths, labels=zip(*data)
f = pylab.figure()

#image = imread(image_paths[6])
print "Reading image "+image_paths[0]
image = imread("images/16_left.jpeg")
arr=np.asarray(image)
f.add_subplot(2, 1, 1)
pylab.imshow(arr)
image = image[:,:,1]
image_o = image
image = skimage.filters.median(image, skimage.morphology.rectangle(1, 1))
print "Applying gaussian filter"
image = skimage.filters.gaussian_filter(image, 1)
print "Applying opening"
image = skimage.morphology.binary_opening(image, selem=skimage.morphology.rectangle(1,1))
print "Applying closing"
image = skimage.morphology.binary_closing(image)
image = image_o-image

image = skimage.exposure.equalize_adapthist(image,clip_limit=0.05, nbins=512)
image=(image*255).astype(np.uint8)
image_o = image
image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,30)

image = image_o-image
arr=np.asarray(image)
f.add_subplot(2, 1, 2)
pylab.imshow(arr)
pylab.show()