'''
Created on 16 Jul 2015

@author: navjotkukreja
'''

from tc import TerminalController, ProgressBar
from joblib import Parallel, delayed
from scipy import misc
from skimage.feature import hog
import sys
from skimage.color import rgb2gray


class preprocess:
    def __init__(self, image_paths):
        self.__term_ = TerminalController()
        self.imgIndex = 0
        self.__image_paths_ = image_paths
        self.__progress_ = ProgressBar(self.__term_, 'Processing images')
        
    def process_images(self):
        self.print_replace("Starting processing")
        images = Parallel(n_jobs=6)(delayed(process_image)(self, i) for i in self.__image_paths_) 
        return images
       
    def rgb2singch(self, image):
        #return rgb2gray(image)       #---Normal grayscale normalisation
        #return image[:,:,1]            #Green channel
        return image[:,:,0]         #Red Channel
        #return image[:,:,2]        #Blue Channel
    
    def get_hog_fd(self, image):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
        return fd
    
    def print_replace(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        sys.stdout.write(self.__term_.BOL + self.__term_.CLEAR_EOL)
        
    def update_progress(self, filename):
        self.__progress_.update(float(self.imgIndex)/len(self.__image_paths_), 'working on %s' % filename)  
        
def process_image(obj, image_path):
    obj.imgIndex+=1
    obj.update_progress(image_path)
    image = misc.imread(image_path)
    image_gray = obj.rgb2singch(image)
    hog_fd = obj.get_hog_fd(image_gray)
    return hog_fd   