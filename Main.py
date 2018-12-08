from DataSet import dataSet as dataSet
import FeatureExtraction as featureExtraction
from features.CO3 import co3 as co3
import features.EdgeHinge as hinge
import PerformanceEvaluation as performanceEvaluation
import KnnIdentification as knnIdentification
import time
import numpy as np


start = time.time()
# TODO: Uncomment after testing.

# Load the dataset and devide it between training and testing and make the nedded preprocessing.
dataSet.loadDataset()
print("Time taken to loadDataset = "+str(time.time() - start))
sartExtractFeatures=time.time()
# Extract the featureVectors for all the writers may need additional preprocessing inside.
classifiedCO3,featureVectors=featureExtraction.extractFeatures("edgeHinge",dataSet.trainingDataImages)
#print(featureVectors) # TODO: Delete after testing
print("Time taken to extractFeatures = "+str(time.time() - sartExtractFeatures))

# Evaluate the performance on the training data.
sartEvaluatePerformance=time.time()
performanceEvaluation.evaluatePerformance("edgeHinge",dataSet.testingDataImages,dataSet.testingDataWriters,featureVectors,dataSet.tariningDataWriters,classifiedCO3)
print("Time taken to evaluatePerformance = "+str(time.time() - sartEvaluatePerformance))


# Identify any writer.
sartIdentifyWriter=time.time()
writerImages=knnIdentification.readWriterImages(1)
writerId=knnIdentification.identifyWriter("edgeHinge",featureVectors,dataSet.tariningDataWriters,writerImages[0],classifiedCO3)
print("This image belongs to writer = "+str(writerId))
print("Time taken to identifyWriter = "+str(time.time() - sartIdentifyWriter))

# Printing the time taken.
print("Time taken to excute the code = "+str(time.time() - start))
