from scipy.interpolate import splprep, splev
from multiprocessing import Process,Manager,Pool
from matplotlib import pyplot as plt
import Preprocessing as preprocessing
import constants
import numpy as np
import random
import time

class co3:
    
    co3Size=50    
    contorSize=40
    epochs=100      
    radiousS=co3Size 
    radiousE=0 
    learningRateS=0.9
    learningRateE=0.015
    s=5 # Stepness factor.
    manager=Manager()
    processesNo=5;
  

    def getFeatureVector(classifiedCO3,imageCO3,training=True,image=None):
        if training==False:
            imageCO3=co3.extractTheCO3(image)
        featureVector=co3.matchTheCO3ToCalcPDF(classifiedCO3,imageCO3)
        return featureVector
        
    
    def getFeatureVectorProcess(featureVectors,classifiedCO3,imageCO3,index):
        #featureVectors[index]=co3.matchTheCO3ToCalcPDF(classifiedCO3,imageCO3)
         featureVector=co3.manager.list([0 for i in range(co3.co3Size)])
         steps=int(np.ceil(len(imageCO3)/co3.processesNo))
         processes = [Process(target=co3.matchTheCO3ToCalcPDFProcess,args= (classifiedCO3,imageCO3[i*steps:min(i*steps+steps,len(imageCO3))],featureVector)) for i in range(co3.processesNo)]
         for p in processes:
             p.start()
         for p in processes:
             p.join()
             p.terminate()   
         featureVectors[index]=featureVector
         del featureVector

    def matchTheCO3ToCalcPDF(classifiedCO3,allCO3):
        if len(allCO3) == 0:
            return []
        imagePDF=[0 for i in range(co3.co3Size)] 
        for i in range(len(allCO3)):
            co3MinIndex=co3.getTheIndexOfMinContorDiff(classifiedCO3,allCO3[i])
            imagePDF[co3MinIndex]+=(1/len(allCO3)) 
        return imagePDF

    def matchTheCO3ToCalcPDFProcess(classifiedCO3,allCO3,imagePDF):
        if len(allCO3) == 0:
            return []
        for i in range(len(allCO3)):
            co3MinIndex=co3.getTheIndexOfMinContorDiff(classifiedCO3,allCO3[i])
            imagePDF[co3MinIndex]+=(1/len(allCO3))
        return   
    
    
    def extractTheCO3(image):
        _,contors=preprocessing.segmentCharactersUsingProjection(image,"co3",True) 
        resampledContors=co3.getResampledContors(contors,co3.contorSize)
        return resampledContors
    
    def extractTheCO3AllImages(trainingDataImages):
        allCO3=[]
        imagesCO3=[[] for i in range(len(trainingDataImages))]
        for i in range(len(trainingDataImages)):
             imagesCO3[i]=co3.extractTheCO3(trainingDataImages[i])
             allCO3+=imagesCO3[i]
        return allCO3,imagesCO3
    
    
    def classifyTheCO3UsingKohenenMap(trainingDataImages):
        start=time.time()
        allCO3,imagesCO3=co3.extractTheCO3AllImages(trainingDataImages)
        print("len of allCO3 "+str(len(allCO3))+" time "+str(time.time()-start))
        # Run the kohenen self organizing map algorithm.
        classifiedCO3=[]
        if constants.continueTrainning == True:
            classifiedCO3=co3.readWeights()
        else:
            classifiedCO3=[[(random.uniform(-1, 1),random.uniform(-1, 1)) for j in range(co3.contorSize)] for i in range(co3.co3Size)]
        radious=co3.radiousS
        rate=co3.learningRateS
        for k in range(co3.epochs):
            alpha=[False for i in range(co3.co3Size)]
            randomIndexArr = np.random.choice(len(allCO3), 1,replace=False)
            randomIndex=randomIndexArr[0]
            randomInput=allCO3[randomIndex]
            nearestContorIndex=co3.getTheIndexOfMinContorDiff(classifiedCO3,randomInput)
            co3.updateAlpha(alpha,radious,nearestContorIndex)
            co3.updateWeights(classifiedCO3,alpha,rate,randomInput)
            radious,rate=co3.updateTrainingParamters(k+1)
        return classifiedCO3,imagesCO3
        
       
   
    def updateAlpha(alpha,radious,index):
        minRange=max(index-round(radious),0)
        maxRange=min(index+round(radious),len(alpha))
        for i in range(minRange,maxRange,1):
            alpha[i]=True
        
    def updateTrainingParamters(k):
        newRadious=pow((pow(co3.radiousE,(1/co3.s))-pow(co3.radiousS,(1/co3.s)))*(k/co3.epochs)+pow(co3.radiousS,(1/co3.s)),co3.s)
        newRate=pow((pow(co3.learningRateE,(1/co3.s))-pow(co3.learningRateS,(1/co3.s)))*(k/co3.epochs)+pow(co3.learningRateS,(1/co3.s)),co3.s)
        return newRadious,newRate
    
    def updateWeights(classifiedCO3,alpha,rate,randomInput):
       for i in range(co3.co3Size):
           difference=[tuple(np.subtract(randomInput[j],classifiedCO3[i][j])) for j in range(len(randomInput))]
           classifiedCO3[i]=[tuple(np.add(classifiedCO3[i][j],(rate*alpha[i]*difference[j][0],rate*alpha[i]*difference[j][1]))) for j in range(len(difference))]
    
    def getFeatureVectors(trainingDataImages):
        start = time.time()
        classifiedCO3,imagesCO3=co3.classifyTheCO3UsingKohenenMap(trainingDataImages)
        print("Time taken to excute the kohenent = "+str(time.time() - start))
        start2 = time.time()
        # Initialize the vectors of each image with empty vector.    
        featureVectors=co3.manager.list([[] for i in range(len(trainingDataImages))])
        processes = [co3.createProcess(classifiedCO3,imagesCO3[i],i,featureVectors) for i in range(len(trainingDataImages))]
        for p in processes:
            p[0].start()
        for p in processes:
            p[0].join()
            print("thread index = "+str(p[1]),str(len(featureVectors[p[1]])))
            p[0].terminate()
        featureVectorsTemp=featureVectors
        del featureVectors
        print("Time taken to excute the featureVectors loop = "+str(time.time() - start2))
        return classifiedCO3,featureVectorsTemp
    
        
    def createProcess(classifiedCO3,imageCO3,index,featureVectors):
        #featureVector = co3.manager.list()
        process = Process(target=co3.getFeatureVectorProcess, args=(featureVectors,classifiedCO3,imageCO3,index))
        return (process ,index)

 
    def getTheIndexOfMinContorDiff(allContors,contor):
        minIndex=-1
        minDist=10000000
        for i in range(len(allContors)):
            totalDist=sum([np.linalg.norm(np.array(allContors[i][j])-np.array(contor[j])) for j in range(len(contor))])
            if totalDist<minDist:
                minDist=totalDist
                minIndex=i
        return minIndex
    
    def getResampledContors(allContorsNormalized,contorRequiredSize):
        resampledContors=[]
        for i in range(len(allContorsNormalized)):
            contor=allContorsNormalized[i]  # 1 & 3 shabh ba3d awy. 2 la
            if len(contor)<contorRequiredSize:
                resampledContors.append(co3.interpolateAndSampleContor(contor,contorRequiredSize))
            else:
                resampledContors.append(co3.selectAndSampleContor(contor,contorRequiredSize))
        return resampledContors
         
    def interpolateAndSampleContor(contor,contorRequiredSize):
         x=[i[0] for i in contor]
         y=[i[1] for i in contor]
         tck, u = splprep([x,y], u=None, s=0.0, per=0)
         uNew = np.linspace(u.min(), u.max(), contorRequiredSize)
         xNew, yNew = splev(uNew, tck, der=0)
         newContor=[(xNew[i],yNew[i]) for i in range(len(xNew))]
         newContor=sorted(newContor,key=lambda x: (x[0], x[1]))
         return newContor
     
    def selectAndSampleContor(contor,contorRequiredSize):
        newContor=[]
        random_indices = np.random.choice(len(contor), contorRequiredSize,replace=False)
        for i in range(contorRequiredSize):
            newContor.append(contor[random_indices[i]])
        newContor=sorted(newContor,key=lambda x: (x[0], x[1]))
        return newContor
    