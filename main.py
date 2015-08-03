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
import numpy as np

reprocess = True    
num_images=500
n_fold_cv = 5
label_file = 'trainLabels.csv'
image_directory = 'images'
#export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/cuda/lib:$HOME/anaconda/lib:/usr/local/lib:/usr/lib:/opt/intel/composer_xe_2015.2.132/compiler/lib:/opt/intel/composer_xe_2015.2.132/mkl/lib


#Make sure we are running inside a virtualenv
if not hasattr(sys, 'real_prefix'):
    print "No virtualenv set. Please activate a virtualenv before running."
    sys.exit()
if __name__=='__main__':
    
    loader = dataloader(label_file, image_directory, num_images)
    if(reprocess):    
        data = loader.get_data()
    
        image_paths, labels=zip(*data)
    
        process = preprocess(image_paths)
        processed = process.process_images()
        print type(processed[0])
        loader.write_csv('vecs.csv', map(list, zip(labels, processed)))
    else:
        rows = loader.load_features('vecs.csv')
        print np.array(rows).shape
        labels, processed = zip(*rows)
        print processed
    #train_paths, test_paths = cv(data, n_fold_cv)
    model = model(processed, labels, n_fold_cv)
    print model.fixed_params(15, 0.5)
    #model.optimise_sgd()