from DataSet import dataSet as dataSet
import FeatureExtraction as featureExtraction
from features.CO3 import co3 as co3
import PerformanceEvaluation as performanceEvaluation
import KnnIdentification as knnIdentification
import SVMIdentification as SVMIdentification
import constants
import time


start = time.time()
# Load the dataset and devide it between training and testing and make the nedded preprocessing.
dataSet.loadDataset()
print("Time taken to loadDataset = "+str(time.time() - start))

sartExtractFeatures=time.time()
# Extract the featureVectors for all the writers may need additional preprocessing inside.
classifiedCO3,featureVectors=featureExtraction.extractFeatures("co3",dataSet.trainingDataImages)
#print(featureVectors) # TODO: Delete after testing
print("Time taken to extractFeatures = "+str(time.time() - sartExtractFeatures))

svclassifier=None
# Evaluate the performance on the training data.
sartEvaluatePerformance=time.time()
if constants.identification=="svm":
    svclassifier=SVMIdentification.trainSvmModel(featureVectors,dataSet.tariningDataWriters)
    performanceEvaluation.evaluatePerformanceSVM(svclassifier,"co3",dataSet.testingDataImages,dataSet.testingDataWriters,classifiedCO3)
else:
    performanceEvaluation.evaluatePerformance("co3",dataSet.testingDataImages,dataSet.testingDataWriters,featureVectors,dataSet.tariningDataWriters,classifiedCO3)
print("Time taken to evaluatePerformance = "+str(time.time() - sartEvaluatePerformance))

# Identify any writer.
writerId=-1
sartIdentifyWriter=time.time()
writerImages=knnIdentification.readWriterImages(1)
if constants.identification=="svm":
    writerId=SVMIdentification.identifyWriterSVM(svclassifier,"co3",writerImages[0],classifiedCO3)
else:
    writerId=knnIdentification.identifyWriter("co3",featureVectors,dataSet.tariningDataWriters,writerImages[0],classifiedCO3)
print("(Identification) This image belongs to writer = "+str(writerId)+" , Correct = "+str(26))
print("Time taken to identifyWriter = "+str(time.time() - sartIdentifyWriter))

# Printing the time taken.
print("Time taken to excute the code = "+str(time.time() - start))

# Print the weights.
co3.printWeights(classifiedCO3)

