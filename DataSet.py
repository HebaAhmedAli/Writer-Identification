import Preprocessing as preprocessing
from matplotlib import pyplot as plt
import cv2

class dataSet:
    dataSize=1539
    trainingSize=800
    testingSize=739
    # We have only 657 writers but the biggest id = 671.
    writersIdSize=671 
    # Contains all the processed training images.
    trainingDataImages=[]
    # Contains all the corresponding training writers.
    tariningDataWriters=[]
    # Contains all the processed testing images.
    testingDataImages=[]
    # Contains all the corresponding testing writers.
    testingDataWriters=[]
    # The index coressponding to writer id 
    # and contains all the processed images writen by this writer in the training data.
    writersWithCorrespondingImagesTraining=[[] for i in range(671)] 
    # The index coressponding to writer id 
    # and contains all the processed images writen by this writer in the testing data.
    writersWithCorrespondingImagesTesting=[[] for i in range(671)] 
    
    # The index coressponding to writer id 
    # and contains all the processed images writen by this writer in all data to be used while reading.
    writersWithCorrespondingImages=[[] for i in range(671)] 
    
      
    def loadDataset():
        # Make any needed preprocessing
        with open("meta/forms/forms.txt", "r") as file:
            i=0
            for line in file:
                if line[0]!='#':
                    data=line.split()
                    image= cv2.imread('data/'+data[0]+'.png')
                    processedImage=preprocessing.processImage(image)
                    dataSet.writersWithCorrespondingImages[int(data[1])].append(processedImage)
                    if i==0:
                        dataSet.show(image)
                        dataSet.show(processedImage)
                        i+=1
        dataSet.splitBetweenTrainingAndTesting()
        print("finishing data load and split it between training = "+str(dataSet.trainingSize)+" and testing = "+str(dataSet.testingSize))
        return  
 
    def splitBetweenTrainingAndTesting():
        for i in range(dataSet.writersIdSize):
            splitingIndex=len(dataSet.writersWithCorrespondingImages[i])/2
            for j in range(len(dataSet.writersWithCorrespondingImages[i])):
                image=dataSet.writersWithCorrespondingImages[i][j]
                if j<splitingIndex:
                    dataSet.trainingDataImages.append(image)
                    dataSet.tariningDataWriters.append(i)
                    dataSet.writersWithCorrespondingImagesTraining[i].append(image)
                else:
                    dataSet.testingDataImages.append(image)
                    dataSet.testingDataWriters.append(i)
                    dataSet.writersWithCorrespondingImagesTesting[i].append(image)
        dataSet.trainingSize=len(dataSet.trainingDataImages)
        dataSet.testingSize=len(dataSet.testingDataImages)
        return
        
    # TODO: Delete after testing.
    def show(image):
        # Figure size in inches
        plt.figure(figsize=(10, 10))
        # Display an image on the axes, with nearest neighbour interpolation
        plt.imshow(image, interpolation='nearest')
        return


        