'''
Created on 28 Jul 2015

@author: navjotkukreja
'''
from ctypes import c_int
from multiprocessing import Value, Lock, Process
from tc import TerminalController, ProgressBar
from joblib import Parallel, delayed
import numpy as np
from skimage.feature import hog
import surf
import detector

class feature:
    def __init__(self):
        if not detector.GLOBAL_WINDOWS:
            self.__term_ = TerminalController()
    
    def extract_features(self, images, algorithm='hog'):
        self.__total_ = len(images)
        if not detector.GLOBAL_WINDOWS:
            self.__progress_ = ProgressBar(self.__term_, 'Extracting features')
            self.update_progress(c_int(0))
            p_jobs=10
        else:
            p_jobs=5
        
        if algorithm=='hog':
            images = Parallel(n_jobs=-1)(delayed(process_image_hog)(self, i) for i in images)
        else:
            if algorithm=='surf':
                self.__surf_extractor_=surf.surf()
                images = self.__surf_extractor_.extract_features(images)
        
        return images
    
    def get_hog_fd(self, image):
        fd = hog(image, orientations=8, pixels_per_cell=(64, 64),
                        cells_per_block=(1, 1), normalise=True)
        return fd
    
    def update_progress(self, c):
        if detector.GLOBAL_WINDOWS:
            print "Working on "+str(c.value)+"/"+str(self.__total_)
        else:
            self.__progress_.update(float(c.value)/self.__total_, 'working on %i' % c.value)
    
counter = Value(c_int)  # defaults to 0
counter_lock = Lock()
def increment(obj):
    with counter_lock:
        counter.value += 1
    obj.update_progress(counter)
    
def process_image_hog(obj, image):
    fd = obj.get_hog_fd(image)
    increment(obj)
    return fd

