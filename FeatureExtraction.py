import features.CO3 as co3
import features.EdgeHinge as edgeHinge
import features.ContorBasedOrientation as contorBasedOrientation
import features.PolygonApproximation as polygonApproximation
import features.HorizontalScan as horizontalScan
import features.HMM as hmm

# Extract the features for all writers during training or testing phase.
def extractFeatures(method,trainingDataImages,writersWithCorrespondingImages):
    if method=="co3":
        return co3.getFeatureVectors(trainingDataImages,writersWithCorrespondingImages)
    elif method=="edgeHinge":
        return edgeHinge.getFeatureVectors(writersWithCorrespondingImages)
    elif method=="contorBasedOrientation":
        return contorBasedOrientation.getFeatureVectors(writersWithCorrespondingImages)
    elif method=="polygonApproximation":
        return polygonApproximation.getFeatureVectors(writersWithCorrespondingImages)
    elif method=="horizontalScan":
        return horizontalScan.getFeatureVectors(writersWithCorrespondingImages)
    elif method=="hmm":
        return hmm.getFeatureVectors(writersWithCorrespondingImages)

# Extract the features for one writer during identification phase.
def extractFeaturesDuringIdentification(method,writerImages):
    if method=="co3":
        return co3.getFeatureVector(writerImages)
    elif method=="edgeHinge":
        return edgeHinge.getFeatureVector(writerImages)
    elif method=="contorBasedOrientation":
        return contorBasedOrientation.getFeatureVector(writerImages)
    elif method=="polygonApproximation":
        return polygonApproximation.getFeatureVector(writerImages)
    elif method=="horizontalScan":
        return horizontalScan.getFeatureVector(writerImages)
    elif method=="hmm":
        return hmm.getFeatureVector(writerImages)
