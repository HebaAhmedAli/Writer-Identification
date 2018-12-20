import Preprocessing as preprocessing
from matplotlib import pyplot as plt
import constants
import cv2
import os

class dataSet:
    # Contains all the processed training images.
    trainingDataImages=[]
    # Contains all the corresponding training writers.
    tariningDataWriters=[1,1,2,2,3,3]
    # Contains all the processed testing images.
    #testingDataImages=[]
    # Contains all the corresponding testing writers.
    #testingDataWriters=[]
    # The index coressponding to writer id 
    # and contains all the processed images writen by this writer in the training data.
    #writersWithCorrespondingImagesTraining=[[] for i in range(constants.writersIdSize)] 
    # The index coressponding to writer id 
    # and contains all the processed images writen by this writer in the testing data.
    #writersWithCorrespondingImagesTesting=[[] for i in range(constants.writersIdSize)] 
    
    # The index coressponding to writer id 
    # and contains all the processed images writen by this writer in all data to be used while reading.
    #writersWithCorrespondingImages=[[] for i in range(constants.writersIdSize)] 
    
      
    def loadDataset(file_name):
        dataSet.trainingDataImages.clear()
        # Make any needed preprocessing
        image= cv2.imread('data/'+file_name+'/1/1'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        #cv2.imwrite("img_cropped1.jpg", processedImage)
        dataSet.trainingDataImages.append(processedImage)
        image= cv2.imread('data/'+file_name+'/1/2'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        #cv2.imwrite("img_cropped2.jpg", processedImage)
        dataSet.trainingDataImages.append(processedImage)
        image= cv2.imread('data/'+file_name+'/2/1'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        #cv2.imwrite("img_cropped3.jpg", processedImage)
        dataSet.trainingDataImages.append(processedImage)
        image= cv2.imread('data/'+file_name+'/2/2'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        #cv2.imwrite("img_cropped4.jpg", processedImage)
        dataSet.trainingDataImages.append(processedImage)
        image= cv2.imread('data/'+file_name+'/3/1'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        #cv2.imwrite("img_cropped5.jpg", processedImage)
        dataSet.trainingDataImages.append(processedImage)
        image= cv2.imread('data/'+file_name+'/3/2'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        #cv2.imwrite("img_cropped6.jpg", processedImage)
        dataSet.trainingDataImages.append(processedImage)
        return  
 
    
        
    # TODO: Delete after testing.
    def show(image):
        # Figure size in inches
        plt.figure(figsize=(10, 10))
        # Display an image on the axes, with nearest neighbour interpolation
        plt.imshow(image, interpolation='nearest',cmap="gray")
        return
    
    def readTestNumbers(path):
        _,directories,_=next(os.walk(path))
        directories=sorted(directories)
        return directories
    
    def readWriterImage(file_name):
        image= cv2.imread('data/'+file_name+'/test'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        return processedImage
        


        