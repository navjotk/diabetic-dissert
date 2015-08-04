'''
Created on 16 Jul 2015

@author: navjotkukreja
'''

from tc import TerminalController, ProgressBar
from joblib import Parallel, delayed
import cv2
from skimage.feature import hog
import sys
import skimage
from skimage.color import rgb2gray
from ctypes import c_int
from multiprocessing import Value, Lock, Process
import surf
from skimage.io import imread
import numpy as np
import detector

class preprocess:
    def __init__(self, image_paths):
        if not detector.GLOBAL_WINDOWS:
            self.__term_ = TerminalController()
        self.imgIndex = 0
        self.__image_paths_ = image_paths
        
    def process_images(self):
        self.print_replace("Starting processing")
        self.__progress_ = ProgressBar(self.__term_, 'Processing images')
        images = Parallel(n_jobs=20)(delayed(process_image2)(self, i) for i in self.__image_paths_) 
        return images
       
    def rgb2singch(self, image):
        #return rgb2gray(image)       #---Normal grayscale normalisation
        return image[:,:,1]            #Green channel
        #return image[:,:,0]         #Red Channel
        #return image[:,:,2]        #Blue Channel
    
    def get_hog_fd(self, image):
        fd = hog(image, orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(1, 1), normalise=True)
        return fd
    
    def meta(self, image):
        return surf.meta_descriptor(image)
    
    def hough(self, image):
        print "H"
        return skimage.transform.hough_ellipse(image)
    
    def median_filter(self, image):
        return skimage.filters.median(image, skimage.morphology.rectangle(3, 3))
    
    def gaussian_filter(self, image):
        return skimage.filters.gaussian_filter(image, 2)
    
    def print_replace(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        if not detector.GLOBAL_WINDOWS:
            sys.stdout.write(self.__term_.BOL + self.__term_.CLEAR_EOL)
        
    def update_progress(self, filename, c):
        if detector.GLOBAL_WINDOWS:
            print "Working on filename "+str(c)+"/"+str(len(self.__image_paths_))
        else:
            self.__progress_.update(float(c.value)/len(self.__image_paths_), 'working on %s' % filename)  
#Class ends


counter = Value(c_int)  # defaults to 0
counter_lock = Lock()
def increment(obj, image_path):
    with counter_lock:
        counter.value += 1
    obj.update_progress(image_path, counter)
    
        
def process_image(obj, image_path):
    image = cv2.imread(image_path)
    image = obj.rgb2singch(image)
    image_o = image
    image = skimage.filters.median(image, skimage.morphology.rectangle(1, 1))
    image = skimage.filters.gaussian_filter(image, 1)
    image = skimage.morphology.binary_opening(image, selem=skimage.morphology.rectangle(1,1))
    image = skimage.morphology.binary_closing(image, selem=skimage.morphology.rectangle(1,1))
    image = image_o-image
    image = skimage.exposure.equalize_adapthist(image,clip_limit=0.05, nbins=512)
    image=(image*255).astype(np.uint8)
    hog_fd = obj.meta(image)
    increment(obj, image_path)
    return hog_fd   

def process_image2(obj, image_path):
    image = imread(image_path)
    image = obj.rgb2singch(image)
    image_o = image
    image = skimage.filters.median(image, skimage.morphology.rectangle(50, 50))
    image = skimage.filters.gaussian_filter(image, 1)
    image = skimage.morphology.binary_opening(image, selem=skimage.morphology.rectangle(1,1))
    image = skimage.morphology.binary_closing(image)
    image = image_o-image
    image = skimage.filters.median(image, skimage.morphology.rectangle(10, 10))
    kernel_size = 120
    image = skimage.exposure.equalize_adapthist(image,clip_limit=0.05, nbins=512, ntiles_x=kernel_size, ntiles_y=kernel_size)
    image=(image*255).astype(np.uint8)
    image_o = image
    
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,30)
    
    image = image_o-image
    #hog_fd = obj.get_hog_fd(image)
    increment(obj, image_path)
    return image   