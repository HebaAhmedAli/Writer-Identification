import FeatureExtraction as featureExtraction
from sklearn.svm import SVC

def identifyWriterSVM(svclassifier,method,writerImage,classifiedCO3=[]):
    featureVector=featureExtraction.extractFeaturesDuringIdentification(method,writerImage,classifiedCO3)
    nearestWriter=svclassifier.predict([featureVector])
    return nearestWriter

def trainSvmModel(featureVectors,tariningDataWriters):
    svclassifier = SVC(kernel='rbf')  # Gaussian.
    #svclassifier = SVC(kernel='poly', degree=8)  
    svclassifier.fit(featureVectors, tariningDataWriters)  
    return svclassifier