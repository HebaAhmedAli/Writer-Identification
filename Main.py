from DataSet import dataSet as dataSet
import FeatureExtraction as featureExtraction
from features.CO3 import co3 as co3
import PerformanceEvaluation as performanceEvaluation
import KnnIdentification as knnIdentification
import SVMIdentification as SVMIdentification
import constants
import time


directories=dataSet.readTestNumbers("/home/heba/Documents/cmp/fourth_year/pattern/Writer-Identification/data")
results=open("results.txt","w")
times=open("time.txt","w")
for i in range(len(directories)):
    start = time.time()
    # Load the dataset and devide it between training and testing and make the nedded preprocessing.
    dataSet.loadDataset(directories[i])
    print("Time taken to loadDataset = "+str(time.time() - start))

    sartExtractFeatures=time.time()
    # Extract the featureVectors for all the writers may need additional preprocessing inside.
    classifiedCO3,featureVectors=featureExtraction.extractAndConcatinateFeautures(constants.methods,dataSet.trainingDataImages)
    print("Time taken to extractFeatures = "+str(time.time() - sartExtractFeatures))
         
    # Identify any writer.
    writerId=-1
    sartIdentifyWriter=time.time()
    writerImage=dataSet.readWriterImage(directories[i])
    if constants.identification=="svm":
        svclassifier=SVMIdentification.trainSvmModel(featureVectors,dataSet.tariningDataWriters)
        writerId=SVMIdentification.identifyWriterSVM(svclassifier,constants.methods,writerImage,classifiedCO3)
    else:
        writerId=knnIdentification.identifyWriter(constants.methods,featureVectors,dataSet.tariningDataWriters,writerImage,classifiedCO3)
    print("(Identification) This image belongs to writer = "+str(writerId))
    print("Time taken to identifyWriter = "+str(time.time() - sartIdentifyWriter))
                    
    # Printing the time taken.
    print("Time taken to excute the code = "+str(time.time() - start))
    results.write(str(writerId[0])+"\n")
    times.write(str(time.time() - start)+"\n")

results.close()
times.close()