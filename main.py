'''
Created on 14 Jul 2015

@author: navjotkukreja
'''
from dataloader import dataloader
from crossvalidation import cv
from preprocess import preprocess
from model import model

    
num_images=500
n_fold_cv = 5
label_file = 'trainLabels.csv'
image_directory = 'images'

loader = dataloader(label_file, image_directory, num_images)
data = loader.get_data()

image_paths, labels=zip(*data)

process = preprocess(image_paths)
processed = process.process_images()
#train_paths, test_paths = cv(data, n_fold_cv)
model = model(processed, labels, n_fold_cv)
model.optimise()
