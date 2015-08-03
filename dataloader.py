'''
Created on 16 Jul 2015

@author: navjotkukreja
'''
import os
import csv
import numpy as np

class dataloader:
    def __init__(self, csv_file, image_directory, num_to_load):
        self.__csv_file_ = csv_file
        self.__image_directory_ = image_directory
        self.__num_to_load_=num_to_load
        self.__lines_ = self.__load_file_(csv_file)
        self.__load_image_file_names_()
        self.__load_labels_()
        self.__clean_data_()
        print np.asarray(np.unique(np.array(self.__labels_), return_counts=True)).T
    
    def __clean_data_(self):
        for i, j in enumerate(self.__labels_):
            if j == -1:
                del self.__labels_[i]
                del self.__image_file_names_[i]
                del self.__image_full_file_names_[i]
        
    def __load_file_(self, file):
        with open(file, 'rb') as csvfile:
            filereader = csv.reader(csvfile)
            lines = []
            for row in filereader:
                lines.append(row)
        return lines
    
    def load_features(self, file):
        return self.__load_file_(file)
    
    def write_csv(self, file, rows):
        with open(file, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)
    
    def __load_image_file_names_(self):
        contents = os.listdir(self.__image_directory_)
        if self.__num_to_load_==0:
            self.__num_to_load_=len(contents)
            
        print "Found "+str(len(contents))+" images, using "+str(self.__num_to_load_)
        self.__image_file_names_ = contents[0:self.__num_to_load_]
        self.__image_full_file_names_ = map(lambda x: (self.__image_directory_+"/"+x), contents[0:self.__num_to_load_])
    
    def __load_labels_(self):
        self.__labels_ = map(lambda x: self.__get_label_(x), self.__image_file_names_)
    
    def __get_label_(self, name):  
        name_ar = name.split('.')
        name=name_ar[0]
        for entry in self.__lines_:
            if entry[0]==name:
                #return entry[1] #Multi-class
                if entry[1]=='0' or entry[1]=='1':
                    return 0    #0 for original class 0
                else:
                    return 1    #For any of the original classes 1,2,3,4, return 1
        print "No label found for name "+name
        return -1               #Name not found
    
    def get_data(self):
        return zip(self.__image_full_file_names_, self.__labels_)