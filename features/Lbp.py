import DataSet as ds
from multiprocessing import Process,Manager
from skimage import feature
import numpy as np
import cv2


def describe(image,numPoints=8*3,radius=3,eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, numPoints,radius, method="nri_uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, numPoints + 3),
			range=(0, numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist
    
def getFeatureVectorProcess(image,featureVectors,directions,index):
    
    featureVectors[index]=describe(image)
    #print(len(featureVectors[index]),featureVectors[index])
    
def getFeatureVector(image):
    # TODO: Write our method for extracting the feature vector.
    featureVector=describe(image)
    return featureVector

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    manager=Manager()
    featureVectors=manager.list([[] for i in range(len(trainingDataImages))])
    #featureVectors=[[] for i in range(len(trainingDataImages))] 
    processes = [createProcess(trainingDataImages[i],8,i,featureVectors) for i in range(len(trainingDataImages))]
    for p in processes:
        p[0].start()
    for p in processes:
        p[0].join()
        #print("thread index = "+str(p[1]),str(len(featureVectors[p[1]])))
        p[0].terminate()
    featureVectorsTemp=featureVectors
    del featureVectors
    return [],featureVectorsTemp

def getFeatureVectorsNoProcess(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    #manager=Manager()
    #featureVectors=manager.list([[] for i in range(len(trainingDataImages))])    
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
        
    return [],featureVectors

def createProcess(image,directions,index,featureVectors):
    #manager=Manager()
    #featureVector = manager.list()
    process = Process(target=getFeatureVectorProcess, args=(image,featureVectors,directions,index))
    return (process,index)
