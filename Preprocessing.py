import cv2

def processImage(image):
    value = (5, 5)
    blurred = cv2.GaussianBlur(image, value, 0)
    grayImage = cv2.cvtColor(blurred,cv2.COLOR_RGB2GRAY)
    ret,binarized= cv2.threshold(grayImage,175,255,cv2.THRESH_BINARY)
    return binarized