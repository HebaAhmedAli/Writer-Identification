import DataSet as ds
import numpy as np
import cv2
from multiprocessing import Process,Manager


def getEdges(writerImg):
    #sobelImg = cv2.Sobel(writerImg,-1,dx=1,dy=1,ksize=5)
    cannyImage = cv2.Canny(writerImg,100,200)
    #ds.dataSet.show(sobelImg)
    return cannyImage
 
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
    #hist=[0 for i in range(directions*(2*directions-3))]  # Heba
    hist=np.zeros((directions*(2*directions-3),))
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
    for y in range (0,height-1):
       for x in range (0,width-1):
           if edgeImg[y,x]:
               for i in range (0,2*directions-3):               
                   nextX = x+deltaX(i,directions)
                   nextY = y+deltaY(i,directions)
                   if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                        if edgeImg[nextY,nextX]:
                            for j in range (0,lookforward[i]):
                                nextX = x+deltaX(i+2+j,directions)
                                nextY = y+deltaY(i+2+j,directions)
                                if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                                    if edgeImg[nextY,nextX]:
                                         hist[histIdx[i]+j] += 1
                                         nb+=1
                        
                   
    #hist=[hist[i]/nb for i in range(len(hist))]  # Heba
    hist/=nb
    #print ("Histogram-Hinge")
    #print(hist)
    return hist.tolist()   
    
def hingeOptim(edgeImg,directions):
    #hist=[0 for i in range(directions*(2*directions-3))]  # Heba
    hist=np.zeros((directions*(2*directions-3),))
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
    white_pixels=np.argwhere(edgeImg)
    for j in range (0,len(white_pixels)):
        for i in range (0,2*directions-3):               
            nextX = white_pixels[j][1]+deltaX(i,directions)
            nextY = white_pixels[j][0]+deltaY(i,directions)
            if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                        if edgeImg[nextY,nextX]:
                            for j in range (0,lookforward[i]):
                                nextX = white_pixels[j][1]+deltaX(i+2+j,directions)
                                nextY = white_pixels[j][0]+deltaY(i+2+j,directions)
                                if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                                    if edgeImg[nextY,nextX]:
                                         hist[histIdx[i]+j] += 1
                                         nb+=1
                        
                   
    #hist=[hist[i]/nb for i in range(len(hist))]  # Heba
    hist/=nb
    #print ("Histogram-Hinge")
    #print(hist)
    return hist.tolist()   
    
def edgeDirectionOptim(edgeImg,directions):
    #hist=[0 for i in range(directions)]  # Heba
    hist=np.zeros((directions,))

    nb=0
    height=edgeImg.shape[0]
    width=edgeImg.shape[1]  
    white_pixels=np.argwhere(edgeImg)
    #print(white_pixels)
    #print(white_pixels[0][=0])
    #print(white_pixels[0][1])
    for j in range (0,len(white_pixels),20):
        for i in range (0,directions):
            nextX = white_pixels[j][1]+deltaX(i,directions)
            nextY = white_pixels[j][0]+deltaY(i,directions)
            '''
            print("direction ",i)
            print("delta X ",deltaX(i,directions))
            print("delta Y ",deltaY(i,directions))
            '''
            if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                if edgeImg[nextY,nextX]:
                    #print("Next Edge Pixel")
                    hist[i] +=1
                    #print (hist[i])
                    nb+=1
    hist/=nb
    #print ("Histogram-Direction")
    #print(hist)
    return hist.tolist()       
def edgeDirection(edgeImg,directions):
    #hist=[0 for i in range(directions)]  # Heba
    hist=np.zeros((directions,))

    nb=0
    height=edgeImg.shape[0]
    width=edgeImg.shape[1]      
    for y in range (0,height-1):
        for x in range (0,width-1):
           if edgeImg[y,x]:
                #print("Edge Pixel")
                for i in range (0,directions):
                    nextX = x+deltaX(i,directions)
                    nextY = y+deltaY(i,directions)
                    if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                        if edgeImg[nextY,nextX]:
                            #print("Next Edge Pixel")
                            hist[i] +=1
                            #print (hist[i])
                            nb+=1
    hist/=nb
    #print ("Histogram-Direction")
    #print(hist)
    return hist.tolist()
def getFeatureVectorProcess(image,featureVectors,directions,index):
    edgeImg = getEdges(image)
    featureVectors[index]=edgeDirectionOptim(edgeImg,directions) 
    

    
def getFeatureVector(image):
    # TODO: Write our method for extracting the feature vector.
    featureVector=[]
    edgeImg = getEdges(image)
    #featureVector+=hingeOptim(edgeImg,12)  # Heba 
    featureVector+=edgeDirectionOptim(edgeImg,8)   # Heba
    return featureVector

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    manager=Manager()
    featureVectors=manager.list([[] for i in range(len(trainingDataImages))])
    #featureVectors=[[] for i in range(len(trainingDataImages))] 
    processes = [createProcess(trainingDataImages[i],8,i,featureVectors) for i in range(len(trainingDataImages))]
    for p in processes:
        p[0].start()
    for p in processes:
        p[0].join()
        #print("thread index = "+str(p[1]),str(len(featureVectors[p[1]])))
        p[0].terminate()
    featureVectorsTemp=featureVectors
    del featureVectors
    return [],featureVectorsTemp

def getFeatureVectorsNoProcess(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    #manager=Manager()
    #featureVectors=manager.list([[] for i in range(len(trainingDataImages))])    
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
        
    return [],featureVectors

def createProcess(image,directions,index,featureVectors):
    #manager=Manager()
    #featureVector = manager.list()
    process = Process(target=getFeatureVectorProcess, args=(image,featureVectors,directions,index))
    return (process,index)
