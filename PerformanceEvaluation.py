import KnnIdentification as knnIdentification
import SVMIdentification as SVMIdentification

def evaluatePerformance(methods,testingImages,testingImagesWriters,featureVectorsAfterTraining,tariningDataWriters,classifiedCO3=[]):
    correctImages=0
    for i in range(len(testingImages)):
        correctWriterId=testingImagesWriters[i]
        nearestWriter=knnIdentification.identifyWriter(methods,featureVectorsAfterTraining,tariningDataWriters,testingImages[i],classifiedCO3)
        print("(Performance) This image belongs to writer = "+str(nearestWriter)+" , Correct = "+str(correctWriterId))
        if nearestWriter==correctWriterId:
            correctImages+=1
    print("The performance = "+str(correctImages/len(testingImages))+" ,")
    print("As correcttly identified "+str(correctImages)+" out of "+str(len(testingImages))+" images.")


def evaluatePerformanceSVM(svclassifier,methods,testingImages,testingImagesWriters,classifiedCO3=[]):
    correctImages=0
    for i in range(len(testingImages)):
        correctWriterId=testingImagesWriters[i]
        nearestWriter=SVMIdentification.identifyWriterSVM(svclassifier,methods,testingImages[i],classifiedCO3)
        print("(Performance) This image belongs to writer = "+str(nearestWriter)+" , Correct = "+str(correctWriterId))
        if nearestWriter[0]==correctWriterId:
            correctImages+=1
    print("The performance = "+str(correctImages/len(testingImages))+" ,")
    print("As correcttly identified "+str(correctImages)+" out of "+str(len(testingImages))+" images.")
