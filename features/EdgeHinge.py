import constants
import DataSet as ds
import numpy as np
import cv2
def getEdges(writerImg):
    sobelImg = cv2.Sobel(writerImg,-1,dx=1,dy=1,ksize=5)
    #ds.dataSet.show(sobelImg)
    return sobelImg
 
def deltaX(index,directions):
    oldIndex = index
    if index>directions :
        index -= directions
    if index <= directions//4:
        x = directions//4
    elif index >= 3*directions//4:
        x = -1*directions//4   
    elif index == (directions//4)+1:
        x = (directions//4) - 1      
    elif index == (directions//4)+2:
        x = (directions//4) - 2  
    elif index == (directions//4)+3:
        x = (directions//4) - 3
    elif index == (directions//4)+4:
        x = (directions//4) - 4    
    elif index == (directions//4)+5:
        x = (directions//4) - 5  
    elif index == (directions//4)+6:
        x = (directions//4) - 6  
    elif index == (directions//4)+7:
        x = (directions//4) - 7
    if oldIndex>directions :
        return -1*x
    return x

def deltaY(index,directions):
    oldIndex = index
    if index>directions :
        index -= directions
    if index >= directions//4 and index <= 3*directions//4:
        y = -1*directions//4      
    elif index == (directions//4) - 1 or index == (3*directions//4) + 1:
        y = -1*directions//4 + 1
    elif index == (directions//4) - 2 or index == (3*directions//4) + 2:
        y = -1*directions//4 + 2
    elif index == (directions//4) - 3 or index == (3*directions//4) + 3:
        y = -1*directions//4 + 3
    elif index == (directions//4) - 4 or index == (3*directions//4) + 4:
        y = -1*directions//4 + 4 
    if oldIndex>directions :
        return -1*y
    return y

def hinge(edgeImg,directions):
    
    hist=np.zeros(directions*(2*directions-3),dtype=int) 
    nb=0
    height=edgeImg.shape[0]
    width=edgeImg.shape[1] 
    lookforward=[]
    histIdx=[]

    lookforward.append(2*directions-3)
    for i in range (2*directions-2):
        lookforward.append(2*directions-3-i)
        #print (lookforward[i])
    
    histIdx.append(0)    
    for i in range (2*directions-2):
        histIdx.append(histIdx[i]+lookforward[i])
       # print(histIdx[i])
    for x in range (0,height-1):
       for y in range (1,width-1):
           if edgeImg[x][y]:
               for i in range (0,2*directions-3):               
                   nextX = x+deltaX(i,directions)
                   nextY = y+deltaY(i,directions)
                   if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                        if edgeImg[nextX][nextY]:
                            for j in range (0,lookforward[i]):
                                nextX = x+deltaX(i+2+j,directions)
                                nextY = y+deltaY(i+2+j,directions)
                                if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                                    if edgeImg[nextX][nextY]:
                                         hist[histIdx[i]+j] += 1
                                         nb+=1
                        
                   
    hist=hist/nb
    #print ("Histogram-Hinge")
    #print(hist)
    return hist   
           
def edgeDirection(edgeImg,directions):
    hist=np.zeros(directions,dtype=int)
    nb=0
    height=edgeImg.shape[0]
    width=edgeImg.shape[1]      
    for x in range (0,height-1):
        for y in range (1,width-1):
            if edgeImg[x][y]:
                #print("Edge Pixel")
                for i in range (0,directions-1):
                    nextX = x+deltaX(i,directions)
                    nextY = y+deltaY(i,directions)
                    if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                        if edgeImg[nextX][nextY]:
                            #print("Next Edge Pixel")
                            hist[i] +=1
                            #print (hist[i])
                            nb+=1
    hist=hist/nb
    #print ("Histogram-Direction")
    #print(hist)
    return hist
def getFeatureVector(image):
    # TODO: Write our method for extracting the feature vector.
    featureVector=[]
    edgeImg = getEdges(image)
    featureVector.append(hinge(edgeImg,12))
    
    featureVector.append(edgeDirection(edgeImg,12))
    return featureVector
def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
    return [],featureVectors


