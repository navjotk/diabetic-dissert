'''
Created on 22 Jul 2015

@author: navjotkukreja
'''


import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import scipy
import sklearn.cluster

def partial_vector(image, threshold):
    surf = cv2.SURF(threshold)
    kp, des = surf.detectAndCompute(image,None)
    xsum = 0
    ysum = 0
    sizesum = 0
    anglesum = 0
    responsesum = 0
    octaves = dict()
    for keypoint in kp:
        #print str(keypoint.pt[0])+","+str(keypoint.pt[1])+", size="+str(keypoint.size)+", angle="+str(keypoint.angle)+", response="+str(keypoint.response)+", octave="+str(keypoint.octave)
        xsum += keypoint.pt[0]
        ysum += keypoint.pt[1]
        sizesum += keypoint.size
        anglesum += keypoint.angle
        responsesum += keypoint.response
        octaves[keypoint.octave] = octaves.get(keypoint.octave,0)+1
    
    if len(kp)>0:    
        xavg = xsum/len(kp)
        yavg = ysum/len(kp)
        sizeavg = sizesum/len(kp)
        angleavg = anglesum/len(kp)
        responseavg = responsesum/len(kp)
    else:
        xavg = 0
        yavg = 0
        sizeavg = 0
        angleavg = 0
        responseavg = 0
    vector = []
    vector.append(xavg)
    vector.append(yavg)
    vector.append(sizeavg)
    vector.append(angleavg)
    vector.append(responseavg)
    combined_octaves = 0
    for ind, el in iter(sorted(octaves.iteritems())):
        combined_octaves += el*(ind+1)
    vector.append(combined_octaves)
    return vector

def meta_descriptor(image):
    threshold_levels = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
    
    complete_vector = []
    for threshold in threshold_levels:
        partial_vec = partial_vector(image, threshold)
        complete_vector = complete_vector+partial_vec
    return complete_vector

class surf:
    def extract_features(self, images, n_clusters):
        descriptors = np.array([])
        surf = cv2.SURF(500)
        print "Extracting features"
        self.__n_clusters_=n_clusters
        image_des=[]
        for image in images:
            image=(image*255).astype(np.uint8)
            kp, des = surf.detectAndCompute(image, None)
            image_des.append(des)
            descriptors = np.append(descriptors, des)
        desc = np.reshape(descriptors, (len(descriptors)/128, 128))
        print "Total of "+str(len(desc))+" features"
        #desc = np.float32(desc)
        print "Clustering"
        self.__kmeans_model_ = sklearn.cluster.KMeans(n_clusters=self.__n_clusters_, n_jobs=-1)
        self.__kmeans_model_.fit(desc)
        #codebook, dist = scipy.cluster.vq.kmeans(desc, k_or_guess=100, iter=20, thresh=1e-05)
       
        print "Vector quantisation"
        vectors = []
        for i in image_des:
            vector = self.vectorize(i)
            print vector.shape
            vectors.append(vector)
        return vectors
    
    def vectorize(self, des):
        vector = {}
        for feature in des:
            y = self.__kmeans_model_.predict(feature)
            vector[y] = vector.get(y, 0)+1
        
        
        vector_list = []
        for i in range(0,self.__n_clusters_):
            vector_list.append(vector.get(i,0))
        
        return vector_list


#img = cv2.imread("16_right.jpeg")


#vector = meta_descriptor(img)

#img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#plt.imshow(img2),plt.show()
