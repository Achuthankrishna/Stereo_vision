import cv2
import numpy as np
import glob
################################################### Given Data ############################################
#For dataset 1
camd0=np.array([[1733.74,0,792.27],[0,1733.74,541.89], [0,0,1]])
camd1=np.array([[1733.74,0,792.27],[0,1733.74,541.89],[0,0,1]])
doffs1=0
baseline1=536.62
width1=1920
height1=1080
ndisp1=170
vmin1=55
vmax1=142

#For Dataset 2
camd20=np.array([[1758.23,0,829.15],[0,1758.23,552.78],[0,0,1]])
camd21=np.array([[1758.23,0,829.15],[0,1758.23,552.78],[0,0,1]])
doffs2=0
baseline2=97.99
width2=1920
height2=1080
ndisp2=220
vmin2=65
vmax2=197

#For Dataset 3
cam0=np.array([[1734.16,0,333.49],[0,1734.16,958.05], [0,0,1]])
cam1=np.array([[1734.16,0,333.49],[0,1734.16,958.05], [0,0,1]])
doffs3=0
baseline3=228.38
width3=1920
height3=1080
ndisp3=110
vmin3=27
vmax3=85
############################################## Importing Files ##################################
#STEP1 : Getting Common Features using BFMatch

def SIFT(image):
    imgs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect and extract features from the image
    sift = cv2.SIFT_create()
    #Keypoints will be a list of keypoints and desc will be an array of Num of keypoinnts x 128
    #We use SIFT instead of haris corner as Haris corner detection uses fixed kernel size and sclaing
    #image might miss out the details.
    keyp1, descrip1 = sift.detectAndCompute(imgs, None)
    return keyp1,descrip1


def BFMatch(descrip1,descrip2):
    
    bf = cv2.BFMatcher()
    matches1 = bf.knnMatch(descrip1, descrip2, k=2)
    knn1=[]
    for m,n in matches1:
        if m.distance < n.distance * 0.75:
             knn1.append(m)
    return knn1

def match(k1, k2, d1, d2):
    #Now having all keypoints, we will use BF matcher to match descriptor to one image with other using distance calculation
    #All features are compared. We use euclidean distance to match descriptor
    AllMatches =BFMatch(d1,d2)
    matched_pairs=[]
        # construct the two sets of points
        #source points are co-ordinated of keypoints in original plane and dst points are keypoints of
        #matching decscriptors of the second image.
        #We use the DMatch() function to get keypoints values, as its part of output of keyp1. I
    pA = [k1[m.queryIdx].pt for m in AllMatches]
    pB = [k2[m.trainIdx].pt for m in AllMatches]
    matched_pairs.append([pA[0], pA[1], pB[0], pB[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    return (AllMatches, pA, pB,matched_pairs)

def draw_matches(img1,k1,img2,k2,match_pairs):

    match=cv2.drawMatches(img1,k1,img2,k2,match_pairs,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matching Features",match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dataset=int(input("enter dataset number"))
if dataset==1:
    img1=cv2.imread('depth_files/artroom/im0.png')
    img2=cv2.imread('depth_files/artroom/im1.png')
    img1=cv2.resize(img1,(0,0),fx=0.5,fy=0.5)
    img2=cv2.resize(img2,(0,0),fx=0.5,fy=0.5)
elif dataset==2:
    img1=cv2.imread('depth_files/chess/im0.png')
    img2=cv2.imread('depth_files/chess/im1.png')
    img1=cv2.resize(img1,(0,0),fx=0.5,fy=0.5)
    img2=cv2.resize(img2,(0,0),fx=0.5,fy=0.5)
elif dataset==3:
    img1=cv2.imread('depth_files/ladder/im0.png')
    img2=cv2.imread('depth_files/ladder/im1.png')
    img1=cv2.resize(img1,(0,0),fx=0.5,fy=0.5)
    img2=cv2.resize(img2,(0,0),fx=0.5,fy=0.5)
else:
    print("Sorry")
k1,d1=SIFT(img1)
k2,d2=SIFT(img2)
bm=BFMatch(d1,d2)
match,key1,key2,mp=match(k1,k2,d1,d2)
# print(len(match))
draw_matches(img1,k1,img2,k2,match)




