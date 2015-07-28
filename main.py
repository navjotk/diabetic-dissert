'''
Created on 14 Jul 2015

@author: navjotkukreja
'''
from dataloader import dataloader
import crossvalidation
from preprocess import preprocess
from model import model
import sys
import log
    
num_images=500
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

process = preprocess(image_paths)
processed = process.process_images()
#train_paths, test_paths = cv(data, n_fold_cv)
model = model(processed, labels, n_fold_cv)
#print model.fixed_params(6.306319345763202, -0.5151215990956187)
model.optimise()