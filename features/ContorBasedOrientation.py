import Preprocessing as preprocessing


def getFeatureVector(image):
    
    _ ,allcontour = preprocessing.segmentCharactersUsingProjection(image,"contorBasedOrientation")

    
    f1Normalized =[0 for i in range(9)]
    f2Normalized =[0 for i in range(9)]
    f3Normalized =[0 for i in range(9)]
    f4Normalized =[[0 for j in range(9)] for i in range(9)]

    for i in range (len(allcontour)):
        
        f1 =[0 for i in range(9)]
        f2=[0 for i in range(9)]
        f3=[0 for i in range(9)]
        f4=[[0 for j in range(9)] for i in range(9)]
        angles= [0 for k in range(len(allcontour[i])-1)]
        anglesdiff= [0 for k in range(len(allcontour[i])-1)]
        
        for j in range (len(allcontour[i])-1):
            xdiff = allcontour[i][j+1][0] - allcontour[i][j][0]
            ydiff = allcontour[i][j+1][1] - allcontour[i][j][1]
           
            if xdiff ==-1 and ydiff == 1:
                   angles[j]=1 
            elif xdiff ==0 and ydiff == 1:
                   angles[j]=2
            elif xdiff ==1 and ydiff == 1:
                   angles[j]=3 
            elif xdiff ==1 and ydiff == 0:
                   angles[j]=4
            elif xdiff ==1 and ydiff == -1:
                   angles[j]=5
            elif xdiff ==0 and ydiff == -1:
                   angles[j]=6 
            elif xdiff ==-1 and ydiff == -1:
                   angles[j]=7        
            elif xdiff ==-1 and ydiff == 0:
                   angles[j]=8
            f1[angles[j]]+=1
           
            if j>0:
               anglesdiff[j-1]=abs(angles[j]-angles[j-1])
               f2[anglesdiff[j-1]]+=1
               
            if j>1:
               f3[abs(anglesdiff[j-1]-anglesdiff[j-2])]+=1
        for k in range (8):
            if f1[k+1] != 0:
                f1[k+1]/=len(allcontour[i])
            f1Normalized[k+1]+=f1[k+1]  
            
            if f2[k+1] != 0:
                f2[k+1]/=len(allcontour[i])
            f2Normalized[k+1]+=f2[k+1]                 
            
            if f3[k+1] != 0:
                f3[k+1]/=len(allcontour[i])
            f3Normalized[k+1]+=f3[k+1] 
            
        for m in range (len(angles)-1):
            f4[angles[m]][angles[m+1]]+=1
        for k in range (8):
            for l in range(8):
                if f4[k+1][l+1] != 0:
                    f4[k+1][l+1]/=len(allcontour[i])
                f4Normalized[k+1][l+1]+=f4[k+1][l+1]         
    
    # TO DO: Write our method for extracting the feature vector.
    featureVector=[]
    for i in range(8):
        featureVector.append(f1Normalized[i+1])
    for i in range(8):
        featureVector.append(f2Normalized[i+1])
    for i in range(8):
        featureVector.append(f3Normalized[i+1])
    for i in range(8):
        for j in range(8) :
            featureVector.append(f4Normalized[i+1][j+1])        
    
    return featureVector

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
    return [],featureVectors


