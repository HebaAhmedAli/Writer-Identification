import KnnIdentification as knnIdentification
import SVMIdentification as SVMIdentification

def evaluatePerformance(method,testingImages,testingImagesWriters,featureVectorsAfterTraining,tariningDataWriters,classifiedCO3=[]):
    correctImages=0
    for i in range(len(testingImages)):
        correctWriterId=testingImagesWriters[i]
        nearestWriter=knnIdentification.identifyWriter(method,featureVectorsAfterTraining,tariningDataWriters,testingImages[i],classifiedCO3)
        print("(Performance) This image belongs to writer = "+str(nearestWriter)+" , Correct = "+str(correctWriterId))
        if nearestWriter==correctWriterId:
            correctImages+=1
    print("The performance = "+str(correctImages/len(testingImages))+" ,")
    print("As correcttly identified "+str(correctImages)+" out of "+str(len(testingImages))+" images.")


def evaluatePerformanceSVM(svclassifier,method,testingImages,testingImagesWriters,classifiedCO3=[]):
    correctImages=0
    for i in range(len(testingImages)):
        correctWriterId=testingImagesWriters[i]
        nearestWriter=SVMIdentification.identifyWriterSVM(svclassifier,method,testingImages[i],classifiedCO3)
        print("(Performance) This image belongs to writer = "+str(nearestWriter)+" , Correct = "+str(correctWriterId))
        if nearestWriter==correctWriterId:
            correctImages+=1
    print("The performance = "+str(correctImages/len(testingImages))+" ,")
    print("As correcttly identified "+str(correctImages)+" out of "+str(len(testingImages))+" images.")
