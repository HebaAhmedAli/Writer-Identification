from features.CO3 import co3 as co3
import features.EdgeHinge as edgeHinge
import features.ContorBasedOrientation as contorBasedOrientation
import features.PolygonApproximation as polygonApproximation
import features.HorizontalScan as horizontalScan
import features.HMM as hmm

# Extract the features for all writers during training or testing phase.
def extractFeatures(method,trainingDataImages):
    if method=="co3":
        return co3.getFeatureVectors(trainingDataImages)
    elif method=="edgeHinge":
        return edgeHinge.getFeatureVectors(trainingDataImages)
    elif method=="contorBasedOrientation":
        return contorBasedOrientation.getFeatureVectors(trainingDataImages)
    elif method=="polygonApproximation":
        return polygonApproximation.getFeatureVectors(trainingDataImages)
    elif method=="horizontalScan":
        return horizontalScan.getFeatureVectors(trainingDataImages)
    elif method=="hmm":
        return hmm.getFeatureVectors(trainingDataImages)

# Extract the features for one writer during identification phase.
def extractFeaturesDuringIdentification(method,writerImage,classifiedCO3=[]):
    if method=="co3":
        return co3.getFeatureVector(classifiedCO3,[],False,writerImage)
    elif method=="edgeHinge":
        return edgeHinge.getFeatureVector(writerImage)
    elif method=="contorBasedOrientation":
        return contorBasedOrientation.getFeatureVector(writerImage)
    elif method=="polygonApproximation":
        return polygonApproximation.getFeatureVector(writerImage)
    elif method=="horizontalScan":
        return horizontalScan.getFeatureVector(writerImage)
    elif method=="hmm":
        return hmm.getFeatureVector(writerImage)
    
def extractAndConcatinateFeautures(methods,trainingDataImages):
    methodsFeatureVectors=[]
    classifiedCO3=[]
    for i in range(len(methods)):
        featureVectors=[]
        if methods[i]=="co3":
            classifiedCO3,featureVectors=extractFeatures(methods[i],trainingDataImages)
        else:
            _,featureVectors=extractFeatures(methods[i],trainingDataImages)
        methodsFeatureVectors.append(featureVectors)
    featureVectors=methodsFeatureVectors[0]
    for i in range(1,len(methodsFeatureVectors),1):
        for j in range(len(methodsFeatureVectors[0])):
            featureVectors[j]+=methodsFeatureVectors[i][j]
    return classifiedCO3,featureVectors

def extractAndConcatinateFeauturesDuringIdentification(methods,writerImage,classifiedCO3=[]):
    methodsFeatureVector=[]
    for i in range(len(methods)):
        methodsFeatureVector.append(extractFeaturesDuringIdentification(methods[i],writerImage,classifiedCO3))    
    featureVector=methodsFeatureVector[0]
    for i in range(1,len(methodsFeatureVector),1):
        featureVector+=methodsFeatureVector[i]   
    return featureVector