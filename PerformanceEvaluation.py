import KnnIdentification as knnIdentification

def evaluatePerformance(method,testingImages,testingImagesWriters,featureVectorsAfterTraining,tariningDataWriters,classifiedCO3=[]):
    correctImages=0
    for i in range(len(testingImages)):
        correctWriterId=testingImagesWriters[i]
        nearestWriter=knnIdentification.identifyWriter(method,featureVectorsAfterTraining,tariningDataWriters,testingImages[i],classifiedCO3)
        if nearestWriter==correctWriterId:
            correctImages+=1
    print("The performance = "+str(correctImages/len(testingImages))+" ,")
    print("As correcttly identified "+str(correctImages)+" out of "+str(len(testingImages))+" writers.")
