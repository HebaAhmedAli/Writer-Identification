from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt
import numpy as np
import random
import time
from threading import Thread

class sift:
    
    siftSize=50   # no of interest points in each image
                  # we will need to classify siftSize * no of images 
    vetorSize=128
    epochs=100 # 200
    radiousS=siftSize # Not sure
    radiousE=0 # 
    learningRateS=0.9
    learningRateE=0.015
    s=5 # Stepness factor.
   

    def getFeatureVector(classifiedSift,imageSift,training=True,image=None):
        # TO DO: Write our method for extracting the feature vector.
        if training==False:
            imageSift=sift.extractTheSift(image)
        featureVector=sift.matchTheSiftToCalcPDF(classifiedSift,imageSift)
        return featureVector
    
    def getFeatureVectorThread(featureVector,classifiedSift,imageSift):
        featureVector+=sift.matchTheSiftToCalcPDF(classifiedSift,imageSift)
        
    
    
    def matchTheSiftToCalcPDF(classifiedSift,allSift):
        if len(allSift) == 0:
            return []
        imagePDF=[0 for i in range(sift.siftSize)] 
        for i in range(len(allSift)):
            siftMinIndex=sift.getTheIndexOfMinContorDiff(classifiedSift,allSift[i])
            imagePDF[siftMinIndex]+=(1/len(allSift)) # TODO: Be sure from the devision.
        return imagePDF
        
    
    def extractTheSift(image):
        imageSift=[]
        # TODO: extract the Sift here.
        
        return imageSift
    
    def extractTheSiftAllImages(trainingDataImages):
        allSift=[]
        imagesSift=[[] for i in range(len(trainingDataImages))]
        for i in range(len(trainingDataImages)):
             imagesSift[i]=sift.extractTheSift(trainingDataImages[i])
             allSift+=imagesSift[i]
        return allSift,imagesSift
    
    def classifyTheSiftUsingKmean(trainingDataImages):
        start=time.time()
        allSift,imagesSift=sift.extractTheSiftAllImages(trainingDataImages)
        print("len of allSift "+str(len(allSift))+" time "+str(time.time()-start))
        # TODO: Run the kmean algorithm.
        classifiedSift=[]
        
        
        return classifiedSift,imagesSift
    
    def classifyTheSiftUsingKohenenMap(trainingDataImages):
        start=time.time()
        allSift,imagesSift=sift.extractTheSiftAllImages(trainingDataImages)
        print("len of allSift "+str(len(allSift))+" time "+str(time.time()-start))
        # Run the kohenen self organizing map algorithm.
        classifiedSift=[]
        classifiedSift=[[(random.uniform(-1, 1),random.uniform(-1, 1)) for j in range(sift.vetorSize)] for i in range(sift.siftSize)]
        radious=sift.radiousS
        rate=sift.learningRateS
        for k in range(sift.epochs):
            alpha=[False for i in range(sift.siftSize)]
            randomIndexArr = np.random.choice(len(allSift), 1,replace=False)
            randomIndex=randomIndexArr[0]
            randomInput=allSift[randomIndex]
            nearestContorIndex=sift.getTheIndexOfMinVectorDiff(classifiedSift,randomInput)
            sift.updateAlpha(alpha,radious,nearestContorIndex)
            sift.updateWeights(classifiedSift,alpha,rate,randomInput)
            radious,rate=sift.updateTrainingParamters(k+1)
        return classifiedSift,imagesSift
        
    
   
    def updateAlpha(alpha,radious,index):
        minRange=max(index-round(radious),0)
        maxRange=min(index+round(radious),len(alpha))
        for i in range(minRange,maxRange,1):
            alpha[i]=True
        
    def updateTrainingParamters(k):
        newRadious=pow((pow(sift.radiousE,(1/sift.s))-pow(sift.radiousS,(1/sift.s)))*(k/sift.epochs)+pow(sift.radiousS,(1/sift.s)),sift.s)
        newRate=pow((pow(sift.learningRateE,(1/sift.s))-pow(sift.learningRateS,(1/sift.s)))*(k/sift.epochs)+pow(sift.learningRateS,(1/sift.s)),sift.s)
        return newRadious,newRate
    
    def updateWeights(classifiedSift,alpha,rate,randomInput):
       for i in range(sift.siftSize):
           difference=[tuple(np.subtract(randomInput[j],classifiedSift[i][j])) for j in range(len(randomInput))]
           classifiedSift[i]=[tuple(np.add(classifiedSift[i][j],(rate*alpha[i]*difference[j][0],rate*alpha[i]*difference[j][1]))) for j in range(len(difference))]
    
    # TODO: Remove trainingDataImages.
    def getFeatureVectors(trainingDataImages):
        start = time.time()
        classifiedSift,imagesSift=sift.classifyTheSiftUsingKohenenMap(trainingDataImages)
        print("Time taken to excute the kohenent = "+str(time.time() - start))
        # Initialize the vectors of each image with empty vector.
        start2 = time.time()
        featureVectors=[[] for i in range(len(trainingDataImages))] 
        '''
        threads = [sift.createThread(classifiedSift,imagesSift[i],i) for i in range(len(trainingDataImages))]
        for t in threads:
            t[0].start()
        for t in threads:
            t[0].join()
            print("thread index = "+str(t[2]))
            featureVectors[t[2]] = t[1]
        '''
        for i in range(len(trainingDataImages)):
            start = time.time()
            featureVectors[i]=sift.getFeatureVector(classifiedSift,imagesSift[i])
            print("Time taken to excute the getFeatureVector = "+str(time.time() - start))
        
        print("Time taken to excute the featureVectors loop = "+str(time.time() - start2))
        return classifiedSift,featureVectors
        
    def createThread(classifiedSift,imageSift,index):
        featureVector = []
        thread = Thread(target=sift.getFeatureVectorThread, args=(featureVector,classifiedSift,imageSift))
        return (thread, featureVector ,index)

    


    def getTheIndexOfMinVectorDiff(allVectors,vector):
        minIndex=-1
        minDist=10000000
        for i in range(len(allVectors)):
            totalDist=sum([np.linalg.norm(np.array(allVectors[i][j])-np.array(vector[j])) for j in range(len(vector))])
            if totalDist<minDist:
                minDist=totalDist
                minIndex=i
        return minIndex
            
         
 
