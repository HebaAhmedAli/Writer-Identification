import FeatureExtraction as featureExtraction
from sklearn.svm import SVC,LinearSVC

def identifyWriterSVM(svclassifier,methods,writerImage,writerImageGray,classifiedCO3=[],classifiedSift=[]):
    featureVector=featureExtraction.extractAndConcatinateFeauturesDuringIdentification(methods,writerImage,writerImageGray,classifiedCO3,classifiedSift)
    nearestWriter=svclassifier.predict([featureVector])
    return nearestWriter

def trainSvmModel(featureVectors,tariningDataWriters):
    svclassifier = SVC(kernel='rbf')  # Gaussian.
    #svclassifier = SVC(kernel='poly', degree=8)  # Polynomial.
    #svclassifier = LinearSVC(random_state=0,tol=1e-5,dual=False)  # Gaussian.
    svclassifier.fit(featureVectors, tariningDataWriters)  
    return svclassifier