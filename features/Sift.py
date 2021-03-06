from multiprocessing import Process,Manager,Pool
import numpy as np
import constants
import random
import time
import cv2

class sift:
    
    siftSize=50     # no of clusters
    siftLimit=2000     
    surfLimit=1000  # no of interest points in each image
                    # we will need to classify siftSize * no of images
    vetorSize=128
    epochs=100 # 200
    radiousS=siftSize
    radiousE=0 
    learningRateS=0.9
    learningRateE=0.015
    s=5 # Stepness factor.
    manager=Manager()
    processesNo=5;
  
   
    def getFeatureVector(classifiedSift,imageSift,training=True,image=None):
        if training==False:
            imageSift=sift.extractTheSift(image)
        featureVector=sift.matchTheSiftToCalcPDF(classifiedSift,imageSift)
        return featureVector
    
    def getFeatureVectorProcess(featureVectors,classifiedSift,imageSift,index):
        featureVectors[index]=sift.matchTheSiftToCalcPDF(classifiedSift,imageSift)
      
            
    def matchTheSiftToCalcPDF(classifiedSift,allSift):
        if len(allSift) == 0:
            return []
        imagePDF=[0 for i in range(sift.siftSize)] 
        for i in range(len(allSift)):
            siftMinIndex=sift.getTheIndexOfMinVectorDiff(classifiedSift,allSift[i])
            imagePDF[siftMinIndex]+=(1/len(allSift)) # TODO: Be sure from the devision.
        return imagePDF
        
    
    def extractTheSift(image):
        imageSift=[]       
        if constants.siftOrserf=="sift":
            siift = cv2.xfeatures2d.SIFT_create(sift.siftLimit)
            (kps,imageSift) = siift.detectAndCompute(image, None)
            imageSift/=255
            print("# kps: {}, descriptors: {}".format(len(kps), imageSift.shape))
        else:
            surf = cv2.xfeatures2d.SURF_create(sift.surfLimit)
            kps,imageSift = surf.detectAndCompute(image,None)
        return imageSift
    
    
    def extractTheSiftAllImages(trainingDataImages):
        allSift=[]
        imagesSift=[[] for i in range(len(trainingDataImages))]
        for i in range(len(trainingDataImages)):
             imagesSift[i]=sift.extractTheSift(trainingDataImages[i])
             if i==0:
                 allSift=imagesSift[i]
             else:
                 allSift=np.vstack(( allSift,imagesSift[i]))
        return allSift,imagesSift
    
    def classifyTheSiftUsingKmean(trainingDataImages):
        start=time.time()
        allSift,imagesSift=sift.extractTheSiftAllImages(trainingDataImages)
        print("len of allSift "+str(len(allSift))+" time "+str(time.time()-start))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(allSift,sift.siftSize,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        classifiedSift= center
        return classifiedSift,imagesSift
 
    def classifyTheSiftUsingKohenenMap(trainingDataImages):
        start=time.time()
        allSift,imagesSift=sift.extractTheSiftAllImages(trainingDataImages)
        print("len of allSift "+str(len(allSift))+" time "+str(time.time()-start))
        # Run the kohenen self organizing map algorithm.
        classifiedSift=[]
        if constants.siftOrserf=="sift":
            classifiedSift=[[random.uniform(0, 1) for j in range(sift.vetorSize)] for i in range(sift.siftSize)]
        else:
            classifiedSift=[[random.uniform(-1, 1) for j in range(sift.vetorSize)] for i in range(sift.siftSize)]
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
           difference=[np.subtract(randomInput[j],classifiedSift[i][j]) for j in range(len(randomInput))]
           classifiedSift[i]=[np.add(classifiedSift[i][j],rate*alpha[i]*difference[j]) for j in range(len(difference))]
           
    def getFeatureVectors(trainingDataImages):
        start = time.time()
        classifiedSift=[]
        imagesSift=[]
        if constants.clusteringMethod=="kohenent":   
            classifiedSift,imagesSift=sift.classifyTheSiftUsingKohenenMap(trainingDataImages)
        else:
            classifiedSift,imagesSift=sift.classifyTheSiftUsingKmean(trainingDataImages)
        print("Time taken to excute the kohenent = "+str(time.time() - start))
        # Initialize the vectors of each image with empty vector.
        start2 = time.time()        
        featureVectors=sift.manager.list([[] for i in range(len(trainingDataImages))])
        processes = [sift.createProcess(classifiedSift,imagesSift[i],i,featureVectors) for i in range(len(trainingDataImages))]
        for p in processes:
            p[0].start()
        for p in processes:
            p[0].join()
            print("thread index = "+str(p[1]),str(len(featureVectors[p[1]])))
            p[0].terminate()
        featureVectorsTemp=featureVectors
        del featureVectors
        print("Time taken to excute the featureVectors loop = "+str(time.time() - start2))
        return classifiedSift,featureVectorsTemp
        
    def createProcess(classifiedSift,imageSift,index,featureVectors):
        process = Process(target=sift.getFeatureVectorProcess, args=(featureVectors,classifiedSift,imageSift,index))
        return (process ,index)
    


    def getTheIndexOfMinVectorDiff(allVectors,vector):
        minIndex=-1
        minDist=10000000
        for i in range(len(allVectors)):
            totalDist=sum([abs(allVectors[i][j]-vector[j]) for j in range(len(vector))])
            if totalDist<minDist:
                minDist=totalDist
                minIndex=i
        return minIndex
            
         
 
