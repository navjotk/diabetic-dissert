'''
Created on 14 Jul 2015

@author: navjotkukreja
'''
from dataloader import dataloader
from preprocess import preprocess
import model
import sys
import log
import numpy as np
import feature
import detector

reprocess = True    
num_images=100
n_fold_cv = 5
label_file = 'trainLabels.csv'
image_directory = 'images'



#Make sure we are running inside a virtualenv
if not hasattr(sys, 'real_prefix') and detector.GLOBAL_WINDOWS==False:
    print "No virtualenv set. Please activate a virtualenv before running."
    sys.exit()
if __name__=='__main__':
    
    loader = dataloader(label_file, image_directory, num_images)
    if(reprocess):    
        data = loader.get_data()
    
        image_paths, labels=zip(*data)
    
        process = preprocess(image_paths)
        processed = process.process_images()
        feature_extractor = feature.feature()
        features = feature_extractor.extract_features(processed)
        #loader.write_csv('vecs.csv', map(list, zip(labels, features)))
    else:
        rows = loader.load_features('vecs.csv')
        print np.array(rows).shape
        labels, features = zip(*rows)
        print features
    
    
    model = model.model_w_cv(features, labels, n_fold_cv)
    model.process(12000, 1.5)
    #model = model(features, labels, n_fold_cv)
    #print model.fixed_params(0.4, 1.5)
    #model.optimise_sgd()