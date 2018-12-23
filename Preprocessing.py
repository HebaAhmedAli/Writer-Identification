from matplotlib import pyplot as plt
import numpy as np
import cv2

def processImage(image,typeRequired=""):
    if typeRequired=="gray":
        return image#[740:2950,350:2400]
       #return image[900:3400,0:2460]  # LLtslem
    value = (5, 5)
    blurred = cv2.GaussianBlur(image, value, 0)
    _,binarized= cv2.threshold(blurred,175,255,cv2.THRESH_BINARY)
    croppedBinarized=binarized#[740:2950,350:2400] # TODO: To change these values.
    #croppedBinarized=binarized[900:3400,0:2460] # TODO: LLtslem.
    return croppedBinarized

# It returns the cropped images after splitting.
# Apply a threshold on black pixels ration in the window = 0.02
def splitImageIntoSmallImages(img,windowHight,windowWidth):
    cropped_images=[]
    for y in range(0,img.shape[0],windowHight):      #y
        for x in range(0,img.shape[1],windowWidth):  #x
            cropped_image=img[int(y):int(min(y+windowHight,img.shape[0])),
                              int(x):int(min(x+windowWidth,img.shape[1]))]
            if returnBlackPixelsRatio(cropped_image) < 0.05:
                continue
            cropped_images.append(cropped_image)
            # TODO: Delete after testing.
            #cv2.imwrite("toTestWindow/"+str(y)+"_"+str(x)+".png",cropped_image)
            ############################.
    return cropped_images


# It returns all the contors in the cropped images after splitting.
# Apply a threshold on black pixels ration in the window = 0.02
# Apply a threshold on contor sizes above 10 pixels.
def splitImageIntoSmallImagesAndGetContors(img,windowHight,windowWidth):
    allContors=[]
    approx=[]
    for y in range(0,img.shape[0],windowHight):      #y
        for x in range(0,img.shape[1],windowWidth):  #x
            cropped_image=img[int(y):int(min(y+windowHight,img.shape[0])),
                              int(x):int(min(x+windowWidth,img.shape[1]))]
            if returnBlackPixelsRatio(cropped_image) < 0.02:
                continue
            # TODO: Delete after testing.
            #cv2.imwrite("toTestWindow/"+str(y)+"_"+str(x)+".png",cropped_image)
            ############################.
            getContorsAndDraw(cropped_image,x,y,allContors,False,approx) # Fatema only.
    # TODO: Delete after testing.       
    printContorsLengths(allContors)
    ############################.
    return allContors,approx


def returnBlackPixelsRatio(img):
    nonZeros = cv2.countNonZero(img)
    height, width = img.shape[:2]
    imgSize=height* width
    ratio=(imgSize-nonZeros)/float(imgSize)
    return ratio

def getHorizontalBlackHistogram(processedImg):
    processedImg=1-(processedImg/255)
    horizontalHist = np.sum(processedImg,axis=1).tolist()
    # TODO: Delete after test.
    #plt.plot(horizontalHist)
    #plt.show()
    #########################
    return horizontalHist

def getVerticalBlackHistogram(processedImg):
    processedImg=1-(processedImg/255)
    vertiaclHist = np.sum(processedImg,axis=0).tolist()
    # TODO: Delete after test.
    #plt.plot(vertiaclHist)
    #plt.show()
    #########################
    return vertiaclHist

def getHorizontalImageLines(processedImg,minHight):
    horizontalImageLines=[]
    horizontalHist=getHorizontalBlackHistogram(processedImg)
    start=0
    end=0
    for i in range(len(horizontalHist)):
        if horizontalHist[i]==0 and i+1 < len(horizontalHist) and horizontalHist[i+1]>0:
            start=i+1
        elif horizontalHist[i]>0 and i+1 < len(horizontalHist) and horizontalHist[i+1]==0:
            end=i+1
            if end-start < minHight:
                continue
            horizontalImageLines.append(processedImg[start:end,0:processedImg.shape[1]])
            # TODO: Delete after test.
            #cv2.imwrite("toTestLines/L"+str(i)+".png",processedImg[start:end,0:processedImg.shape[1]])
            #########################.
    return horizontalImageLines

# Segment the image and return all the segmented images and all the contors in them.
def segmentCharactersUsingProjection(processedImg,method,normalizeContors=False,threshOfConnectedPixels=3,minWidth=10,minHight=20,maxWidth=100):
    characterSegments=[]
    allContors=[]
    horizontalImageLines=getHorizontalImageLines(processedImg,minHight)
    for k in range(len(horizontalImageLines)):
        verticalHist=getVerticalBlackHistogram(horizontalImageLines[k])
        start=0
        end=0
        for i in range(len(verticalHist)):
            if verticalHist[i]<=threshOfConnectedPixels and i+1 < len(verticalHist) and verticalHist[i+1]>threshOfConnectedPixels:
                start=i+1
            elif verticalHist[i]>threshOfConnectedPixels and i+1 < len(verticalHist) and verticalHist[i+1]<=threshOfConnectedPixels:
                end=i+1
                if end-start < minWidth:
                    continue
                elif end-start > maxWidth:
                    resegmentedSegments=resegmantation(horizontalImageLines[k][0:horizontalImageLines[k].shape[0],start:end],method,minWidth,threshOfConnectedPixels+10,allContors,normalizeContors)
                    characterSegments+=resegmentedSegments
                    continue
                characterSegments.append(horizontalImageLines[k][0:horizontalImageLines[k].shape[0],start:end])
                # TODO: Delete after test.
                #cv2.imwrite("toTestLines/C"+str(k)+"_"+str(i)+".png",horizontalImageLines[k][0:horizontalImageLines[k].shape[0],start:end])
                #########################.
                getContorsAndDraw(horizontalImageLines[k][0:horizontalImageLines[k].shape[0],start:end],method,k,i,allContors,normalizeContors)
                
    return characterSegments,allContors

def resegmantation(processedImg,method,minWidth,threshOfConnectedPixels,allContors,normalizeContors):
    resegmentedSegments=[]
    verticalHist=getVerticalBlackHistogram(processedImg)
    start=0
    end=0
    for i in range(len(verticalHist)):
        if verticalHist[i]<=threshOfConnectedPixels and i+1 < len(verticalHist) and verticalHist[i+1]>threshOfConnectedPixels:
            start=i+1
        elif verticalHist[i]>threshOfConnectedPixels and i+1 < len(verticalHist) and verticalHist[i+1]<=threshOfConnectedPixels:
            end=i+1
            if end-start < minWidth:
                continue
            resegmentedSegments.append(processedImg[0:processedImg.shape[0],start:end])
             # TODO: Delete after test.
            #cv2.imwrite("toTestLines/CK"+str(i)+".png",processedImg[0:processedImg.shape[0],start:end])
            #########################
            getContorsAndDraw(processedImg[0:processedImg.shape[0],start:end],method,-1,i,allContors,normalizeContors)
    return resegmentedSegments


def printContorsLengths(allContors):
    # TODO: Delete after testing.       
    minLength=100000000
    maxLength=0
    for j in range(len(allContors)):
        #print("contors of image "+str(0)+" cnt no "+str(j)+" length= "+str(len(allContors[j])))
        minLength=min(minLength,len(allContors[j]))
        maxLength=max(maxLength,len(allContors[j]))
    print("min contor length = "+str(minLength)+" and max = "+str(maxLength))
    

# Get all the contors of image in a form of arrays of tuples(x,y)
# and ignore the contors with length <10
def getContorsAndDraw(image,method,x,y,allContors,normalizeContors=False,approx=[]):
    _, contors, _ = cv2.findContours(255-image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #drawing = np.zeros([image.shape[0],image.shape[1],3],np.uint8)
    approxcnt=[]
    for j in range(len(contors)):
        if (len(contors[j]) > 20):
            if normalizeContors==True:
                contors[j]=(contors[j]-np.mean(contors[j], axis=0))/[[image.shape[1],image.shape[0]]]
            contorAsTuple=[(contors[j][i][0][0],contors[j][i][0][1]) for i in range(len(contors[j]))]
            allContors.append(contorAsTuple) # TODO: Delete after testing.
        if method=="polygonApproximation":
            epsilon = 0.012*cv2.arcLength(contors[j],True)
            app=cv2.approxPolyDP(contors[j],epsilon,True)
            approxcnt.append(app)
            appr =[(app[i][0][0],app[i][0][1]) for i in range(len(app))]
            approx.append(appr)
        ############################.
        # TODO: Delete after test.
        #cv2.drawContours(drawing,approx, j, (0,255,0),4)
        #cv2.imwrite("toTest/c"+str(y)+"_"+str(x)+".png",drawing)
    #cv2.drawContours(drawing,approxcnt, -1, (0,255,0),1)
    #cv2.imwrite("toTest/c"+str(y)+"_"+str(x)+".png",drawing)
        #########################.
    