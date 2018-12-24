from multiprocessing import Process,Manager
from skimage import feature
import numpy as np


def describe(image,numPoints=8*3,radius=3,eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, numPoints,radius, method="default")
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
    
def getFeatureVector(image):
    featureVector=describe(image)
    return featureVector

def getFeatureVectors(trainingDataImages):
    manager=Manager()
    featureVectors=manager.list([[] for i in range(len(trainingDataImages))])
    processes = [createProcess(trainingDataImages[i],8,i,featureVectors) for i in range(len(trainingDataImages))]
    for p in processes:
        p[0].start()
    for p in processes:
        p[0].join()
        p[0].terminate()
    featureVectorsTemp=featureVectors
    del featureVectors
    return [],featureVectorsTemp


def createProcess(image,directions,index,featureVectors):
    process = Process(target=getFeatureVectorProcess, args=(image,featureVectors,directions,index))
    return (process,index)
