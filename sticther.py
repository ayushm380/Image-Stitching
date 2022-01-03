import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
from skimage import color
from skimage import io

def detectAndDescribe(image):
    # performing feature detection using ORB
    descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)

def createMatcher():
    # finding matching key points using using brute force method
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    return bf

def matchKeyPointsBF(featuresA, featuresB):
    # matching key points
    bf = createMatcher()
    best_matches = bf.match(featuresA,featuresB)
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    return rawMatches


def getHomography(kpsA, kpsB, featuresA, featuresB, matches):
    # finding Homography using RANSAC
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)
        return (matches, H, status)
    else:
        return None    

trainImg = imageio.imread('campus1.jpg')
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
queryImg = imageio.imread('campus2.jpg')
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
kpsA, featuresA = detectAndDescribe(trainImg_gray)
kpsB, featuresB = detectAndDescribe(queryImg_gray)
fig = plt.figure(figsize=(20,8))
matches = matchKeyPointsBF(featuresA, featuresB)
img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
M = getHomography(kpsA, kpsB, featuresA, featuresB, matches)
if M is None:
    print("Error!")
(matches, H, status) = M
width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]
result = cv2.warpPerspective(trainImg, H, (width, height))
result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
plt.figure(figsize=(20,10))
gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = result[y:y+h,x:x+w]
# croping the black parts of contour
crop_int8 = crop.astype(np.uint8)
res = np.uint8(cv2.normalize(crop_int8, None, 0, 255, cv2.NORM_MINMAX))
# applying histogram equlization
plt.imshow(res)
plt.axis('off')
plt.show()