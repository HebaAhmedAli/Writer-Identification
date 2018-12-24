import FeatureExtraction as featureExtraction
from sklearn.svm import SVC,LinearSVC

def identifyWriterSVM(svclassifier,methods,writerImage,writerImageGray,classifiedCO3=[],classifiedSift=[]):
    featureVector=featureExtraction.extractAndConcatinateFeauturesDuringIdentification(methods,writerImage,writerImageGray,classifiedCO3,classifiedSift)
    nearestWriter=svclassifier.predict([featureVector])
    return nearestWriter

def trainSvmModel(featureVectors,tariningDataWriters):
    svclassifier = SVC(kernel='rbf')  # Gaussian.
    svclassifier.fit(featureVectors, tariningDataWriters)  
    return svclassifier