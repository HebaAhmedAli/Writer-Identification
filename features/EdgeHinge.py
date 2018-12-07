import DataSet as ds
import numpy as np
import cv2
def getEdges(writerImg):
    sobelImg = cv2.Sobel(writerImg,-1,dx=1,dy=1,ksize=5)
    ds.dataSet.show(sobelImg)
    return sobelImg
 
def deltaX(index,directions):
    if index>directions :
        index -= directions 
    if index <= directions/4:
        x = directions/4
    elif index >= 3*directions/4:
        x = -1*directions/4   
    elif index == (directions/4)+1:
        x = (directions/4) - 1      
    elif index == (directions/4)+2:
        x = (directions/4) - 2  
    elif index == (directions/4)+3:
        x = (directions/4) - 3
    elif index == (directions/4)+4:
        x = (directions/4) - 4    
    elif index == (directions/4)+5:
        x = (directions/4) - 5  
    elif index == (directions/4)+6:
        x = (directions/4) - 6  
    elif index == (directions/4)+7:
        x = (directions/4) - 7
    return x

def deltaY(index,directions):
    if index>directions :
        index -= directions
    if index >= directions/4 and index <= 3*directions/4:
        y = -1*directions/4      
    elif index == (directions/4) - 1 or index == (3*directions/4) + 1:
        y = -1*directions/4 + 1
    elif index == (directions/4) - 2 or index == (3*directions/4) + 2:
        y = -1*directions/4 + 2
    elif index == (directions/4) - 3 or index == (3*directions/4) + 3:
        y = -1*directions/4 + 3
    elif index == (directions/4) - 4 or index == (3*directions/4) + 4:
        y = -1*directions/4 + 4 
    return y
         
def getFeatureVector(writerWithCoresspondingImages):
    # TODO: Write our method for extracting the feature vector.
    for i in range(len(writerWithCoresspondingImages)):
        hist=np.zeros((12,),dtype=int)
        nb=0
        #for j in range (len(writerWithCoresspondingImages[i])):
        edgeImg = getEdges(writerWithCoresspondingImages[i])
        height=edgeImg.shape[0]
        width=edgeImg.shape[1]
        
        for x in range (0,width-1):
            for y in range (1,height):
                if edgeImg[x,y]:
                    for i in range (0,11):
                        nextX = x+deltaX(i,12)
                        nextY = y+deltaY(i,12)
                        if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                           if edgeImg[nextX,nextY]:
                              hist[i]=hist[i]+1
                              nb=nb+1
        hist=hist/nb
        print(hist)
        
        
    featureVector=[]
    '''
    featureVector.append((1,2))
    featureVector.append((2,3))
    featureVector.append((5))
    featureVector.append((0))
    '''
    return featureVector

def getFeatureVectors(writersWithCorrespondingImages):
    # Initialize the vectors of each writer with empty vector.
    # Largest Id in the dataset is 671 for 657 writer.
    featureVectors=[[] for i in range(671)] 
    for i in range(len(writersWithCorrespondingImages)):
        featureVectors[i]=getFeatureVector(writersWithCorrespondingImages[i])
    return featureVectors


