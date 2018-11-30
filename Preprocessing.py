import cv2

def processImage(image):
    value = (5, 5)
    blurred = cv2.GaussianBlur(image, value, 0)
    ret,binarized= cv2.threshold(blurred,175,255,cv2.THRESH_BINARY)
    return binarized