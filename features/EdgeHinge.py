from multiprocessing import Process,Manager
import numpy as np
import cv2


def getEdges(writerImg):
    cannyImage = cv2.Canny(writerImg,100,200)
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

def hingeOptim(edgeImg,directions):
    hist=np.zeros((directions*(2*directions-3),))
    nb=0
    height=edgeImg.shape[0]
    width=edgeImg.shape[1] 
    lookforward=[]
    histIdx=[]
    lookforward.append(2*directions-3)
    for i in range (2*directions-2):
        lookforward.append(2*directions-3-i)
    histIdx.append(0)    
    for i in range (2*directions-2):
        histIdx.append(histIdx[i]+lookforward[i])
    white_pixels=np.argwhere(edgeImg)
    for j in range (0,len(white_pixels),20):
        for i in range (0,2*directions-3):               
            nextX = white_pixels[j][1]+deltaX(i,directions)
            nextY = white_pixels[j][0]+deltaY(i,directions)
            if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                        if edgeImg[nextY,nextX]:
                            for k in range (0,lookforward[i]):
                                nextX = white_pixels[j][1]+deltaX(i+2+k,directions)
                                nextY = white_pixels[j][0]+deltaY(i+2+k,directions)
                                if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                                    if edgeImg[nextY,nextX]:
                                         hist[histIdx[i]+k] += 1
                                         nb+=1               
    hist/=nb
    return hist.tolist()   
    
def edgeDirectionOptim(edgeImg,directions):
    hist=np.zeros((directions,))
    nb=0
    height=edgeImg.shape[0]
    width=edgeImg.shape[1]  
    white_pixels=np.argwhere(edgeImg)
    for j in range (0,len(white_pixels),20):
        for i in range (0,directions):
            nextX = white_pixels[j][1]+deltaX(i,directions)
            nextY = white_pixels[j][0]+deltaY(i,directions)
            if (nextX >= 0) and (nextX < width) and (nextY >= 0) and (nextY < height):
                if edgeImg[nextY,nextX]:
                    hist[i] +=1
                    nb+=1
    hist/=nb
    return hist.tolist()       

def getFeatureVectorProcess(image,featureVectors,directions,index):
    edgeImg = getEdges(image)
    featureVectors[index]=edgeDirectionOptim(edgeImg,directions) 
    #featureVectors[index]+=hingeOptim(edgeImg,directions)

    
def getFeatureVector(image):
    featureVector=[]
    edgeImg = getEdges(image)
    featureVector+=edgeDirectionOptim(edgeImg,8) 
    #featureVector+=hingeOptim(edgeImg,8)
    return featureVector

def getFeatureVectors(trainingDataImages):
    manager=Manager()
    # Initialize the vectors of each image with empty vector(this vector is shared between processes)
    featureVectors=manager.list([[] for i in range(len(trainingDataImages))])
    processes = [createProcess(trainingDataImages[i],8,i,featureVectors) for i in range(len(trainingDataImages))]
    for p in processes:
        p[0].start()
    for p in processes:
        p[0].join()
        p[0].terminate()
    featureVectorsTemp=featureVectors
    del featureVectors
    return [],featureVectorsTemp

def createProcess(image,directions,index,featureVectors):
    process = Process(target=getFeatureVectorProcess, args=(image,featureVectors,directions,index))
    return (process,index)
