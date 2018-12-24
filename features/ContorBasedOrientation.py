import Preprocessing as preprocessing
import numpy as np

def getFeatureVector(image):
    
    _ ,allcontour = preprocessing.segmentCharactersUsingProjection(image,"contorBasedOrientation")
 
    histDirections=np.zeros(9)
    totalContourpixels=0
    
    for i in range (len(allcontour)):
        totalContourpixels+=len(allcontour[i])
        for j in range (len(allcontour[i])-1):
            xdiff = allcontour[i][j][0] - allcontour[i][j+1][0]
            ydiff = allcontour[i][j][1] - allcontour[i][j+1][1]
            
            if xdiff <0 and ydiff <0:
                   histDirections[1]+=1
            elif xdiff ==0 and ydiff >0 :
                   histDirections[2]+=1
            elif xdiff >0 and ydiff <0 :
                   histDirections[3]+=1
            elif xdiff >0 and ydiff == 0:
                  histDirections[4]+=1
            elif xdiff >0 and ydiff >0:
                  histDirections[5]+=1
            elif xdiff ==0 and ydiff >0:
                  histDirections[6]+=1
            elif xdiff <0  and ydiff > 0 :
                  histDirections[7]+=1
            elif xdiff <0 and ydiff == 0:
                  histDirections[8]+=1
                  
                  
    
    return (histDirections/totalContourpixels).tolist()

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
    return [],featureVectors