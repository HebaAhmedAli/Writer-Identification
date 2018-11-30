import FeatureExtraction as featureExtraction
from DataSet import dataSet as dataSet
import PerformanceEvaluation as performanceEvaluation
import KnnIdentification as knnIdentification
import time

start = time.time()

# Load the dataset and devide it between training and testing and make any nedded preprocessing.
dataSet.loadDataset()

# TODO: Uncomment after testing.

# Extract the featureVectors for all the writers.
featureVectors=featureExtraction.extractFeatures("co3",dataSet.trainingDataImages,dataSet.writersWithCorrespondingImagesTraining)
#print(featureVectors) # TODO: Delete after testing

# Evaluate the performance on the training data.
performanceEvaluation.evaluatePerformance(dataSet.writersWithCorrespondingImagesTesting,featureVectors)


# Identify any writer.
knnIdentification.identifyWriter("co3",featureVectors,1,3)

###############################.

# Printing the time taken.
end = time.time()
print("Time taken to excute the code = "+str(end - start))