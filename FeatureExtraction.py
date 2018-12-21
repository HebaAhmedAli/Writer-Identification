from features.CO3 import co3 as co3
import features.EdgeHinge as edgeHinge
import features.ContorBasedOrientation as contorBasedOrientation
import features.PolygonApproximation as polygonApproximation
import features.HorizontalScan as horizontalScan
from features.Sift import sift as sift

# Extract the features for all writers during training or testing phase.
def extractFeatures(method,trainingDataImages,trainingDataImagesGray):
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
    elif method=="sift":
        return sift.getFeatureVectors(trainingDataImagesGray)

# Extract the features for one writer during identification phase.
def extractFeaturesDuringIdentification(method,writerImage,writerImageGray,classifiedCO3=[],classifiedSift=[]):
    if method=="co3":
        return co3.getFeatureVector(classifiedCO3,[],False,writerImage)
    elif method=="sift":
        return co3.getFeatureVector(classifiedSift,[],False,writerImageGray)
    elif method=="edgeHinge":
        return edgeHinge.getFeatureVector(writerImage)
    elif method=="contorBasedOrientation":
        return contorBasedOrientation.getFeatureVector(writerImage)
    elif method=="polygonApproximation":
        return polygonApproximation.getFeatureVector(writerImage)
    elif method=="horizontalScan":
        return horizontalScan.getFeatureVector(writerImage)
    elif method=="sift":
        return sift.getFeatureVector(writerImage)
    
def extractAndConcatinateFeautures(methods,trainingDataImages,trainingDataImagesGray):
    methodsFeatureVectors=[]
    classifiedCO3=[]
    classifiedSift=[]
    for i in range(len(methods)):
        featureVectors=[]
        if methods[i]=="co3":
            classifiedCO3,featureVectors=extractFeatures(methods[i],trainingDataImages,trainingDataImagesGray)
        elif methods[i]=="sift":
            classifiedSift,featureVectors=extractFeatures(methods[i],trainingDataImages,trainingDataImagesGray)
        else:
            _,featureVectors=extractFeatures(methods[i],trainingDataImages,trainingDataImagesGray)
        methodsFeatureVectors.append(featureVectors)
    featureVectors=methodsFeatureVectors[0]
    for i in range(1,len(methodsFeatureVectors),1):
        for j in range(len(methodsFeatureVectors[0])):
            featureVectors[j]+=methodsFeatureVectors[i][j]
    return classifiedCO3,classifiedSift,featureVectors

def extractAndConcatinateFeauturesDuringIdentification(methods,writerImage,writerImageGray,classifiedCO3=[],classifiedSift=[]):
    methodsFeatureVector=[]
    for i in range(len(methods)):
        methodsFeatureVector.append(extractFeaturesDuringIdentification(methods[i],writerImage,writerImageGray,classifiedCO3,classifiedSift))    
    featureVector=methodsFeatureVector[0]
    for i in range(1,len(methodsFeatureVector),1):
        featureVector+=methodsFeatureVector[i]   
    return featureVector