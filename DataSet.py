import Preprocessing as preprocessing
from matplotlib import pyplot as plt
import constants
import cv2
import os

class dataSet:
    # Contains all the processed binary training images.
    trainingDataImages=[]
    # Contains all the processed gray training images.
    trainingDataImagesGray=[]
    # Contains all the corresponding training writers.
    tariningDataWriters=[1,1,2,2,3,3]

      
    def loadDataset(file_name):
        dataSet.trainingDataImages.clear()
        dataSet.trainingDataImagesGray.clear()
        # Make any needed preprocessing
        image= cv2.imread('data/'+file_name+'/1/1'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        dataSet.trainingDataImages.append(processedImage)
        dataSet.trainingDataImagesGray.append(processedImageGray)
        image= cv2.imread('data/'+file_name+'/1/2'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        dataSet.trainingDataImages.append(processedImage)
        dataSet.trainingDataImagesGray.append(processedImageGray)
        image= cv2.imread('data/'+file_name+'/2/1'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        dataSet.trainingDataImages.append(processedImage)
        dataSet.trainingDataImagesGray.append(processedImageGray)
        image= cv2.imread('data/'+file_name+'/2/2'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        dataSet.trainingDataImages.append(processedImage)
        dataSet.trainingDataImagesGray.append(processedImageGray)
        image= cv2.imread('data/'+file_name+'/3/1'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        dataSet.trainingDataImages.append(processedImage)
        dataSet.trainingDataImagesGray.append(processedImageGray)
        image= cv2.imread('data/'+file_name+'/3/2'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        dataSet.trainingDataImages.append(processedImage)
        dataSet.trainingDataImagesGray.append(processedImageGray)
        return  
 
    
    def readTestNumbers(path):
        _,directories,_=next(os.walk(path))
        directories.sort(key=int)
        return directories
    
    def readWriterImage(file_name):
        image= cv2.imread('data/'+file_name+'/test'+constants.extension,0)
        processedImage=preprocessing.processImage(image)
        processedImageGray=preprocessing.processImage(image,"gray")
        return processedImage,processedImageGray
        


        