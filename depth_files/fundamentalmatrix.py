import cv2 
import matplotlib.pyplot as plt
import numpy as np
import random
from Code import *
########## ESTIMATING FUNDAMENTAL MATIRX ###########
#On paper F = K^-1 EK^-1 where E is estimate matrix
def fundamental_matrix(kp1,kp2):
    M=np.zeros(shape=((len(kp1)),9))
    for i in range(len(kp1)):
        x0,y0=kp1[i][0],kp1[i][1]
        x1,y1=kp2[i][0],kp2[i][1]
        M[i]=np.array([x0*x1,x0*y1,x0,y0*x1,y0*y1,y0,x1,y1,1])
    #Find SVD of M matrix 
    _,_,V=np.linalg.svd(M)
    F=V[-1,:]
    F=F.reshape(3,3) #Now rank is 3
    #To reduce F to a rank of 2, last singular value of F=0
    U,S,v=np.linalg.svd(F)
    #Last singular value=0
    S[-1]=0
    singular=np.zeros((3,3))
    for i in range(3):
        singular[i][i]=S[i] #Diagonal Singular Matrix
    #Now un normalize F
    F=np.dot(U,np.dot(singular,v))
    # print("Shape of F :",np.shape(F))
    return F
#When we get matches we filter out using RANSAC
def RANSAC(p1, p2, N=5000, t=0.02):
    Cindex = []
    F = 0
    for i in range(0,N):
        idx1, idx2 = [], []
        for j in range(8):
            inl = random.randint(0, len(p1) - 1)
            idx1.append(p1[inl])
            idx2.append(p2[inl])
        F_m = fundamental_matrix(idx1, idx2)
        S_m = []
        for index1, index2 in zip(p1, p2):
            target1 = np.array([index1[0], index1[1], 1])
            target2 = np.array([index2[0], index2[1], 1])
            t_p = abs(np.dot(target2.T, np.dot(F_m, target1)))
            # print(t_p)
            if t_p < t:
                S_m.append([index1,index2])
                 # Append index1 (or any other identifier) instead of [index1, index2]
        if len(S_m) >= 4: #We need atleast 4 pairs - 8 Points to get a fundamental matrix
            new_S = len(S_m)
            S_inlier=S_m
            F_inlier = F_m
            # if new_S > thresh:
            #     Cindex = S_inlier
            break
    else:
        print("No inliers found")
        return None, None
    S_m=np.array(S_m)
    S_m = S_m.reshape(-1, 4)
    print((S_m))
    return F_inlier, S_m

#Get Essential Matrix
def essential(M1,M2,F):
    Ematrix=M2.T.dot(F).dot(M1)
    U,S,V=np.linalg.svd(Ematrix)
    S=[1,1,0] #Forcing last singular value as 0
    Ematrix=np.dot(U,np.dot(np.diag(S),V))
    return Ematrix

#Decomposition of E matrix into T and R, we take sVD of E and R matrix as solution of SVD
def getcampose(E):
    U,S,V=np.linalg.svd(E)
    #We compute R=UWV.T where W takes Skew symm matrix along diagonal
    W_mat=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Rot=[]
    trans=[]
    #checking for negative dets
    R1=np.dot(U,np.dot(W_mat,V))
    R2=np.dot(U,np.dot(W_mat.T,V))
    R3=np.dot(U,np.dot(W_mat,V))
    R4=np.dot(U,np.dot(W_mat.T,V))
    t1=U[:,2]
    t2=-U[:,2]
    t3=U[:,2]
    t4=-U[:,2]
    Rot=[R1,R2,R3,R4]
    trans=[t1,t2,t3,t4]
    for i in range(4): #For all 4 values in the array
        if(np.linalg.det(Rot[i])<0):
            Rot[i] = -Rot[i]
            trans[i] = -trans[i]
    return Rot,trans
 
def checkcheiral(pts, R, T):
    n = 0
    for i in range(pts.shape[1]):
        # Homogeneous coordinates of the 3D point
        X_hom = np.hstack((pts[:,i], 1))

        # Convert homogeneous coordinates to 3D coordinates
        X = X_hom[:3] / X_hom[3]

        # Check if R3(X-T)>0
        if R[2,:].dot(X - T) > 0:
            n += 1

    return n


def getPoints(intrin1, intrin2, inlier, rotation, translation):
    pts = []
    Rotation1 = np.identity(3)
    Translation1 = np.zeros((3,1))
    I = np.identity(3)
    #Taking initial camera pose to be a unit value matrix 
    Camera1 = np.dot(intrin1, np.dot(Rotation1, np.hstack((I, -Translation1.reshape(3,1)))))

    for i in range(len(translation)):
        #here we best best projection poinnts to be the best inlying points from RANSAC
        #Our points shape is (X,4) and we initially take the first 2 column as X1(x1,y1) and second as X2(x2,y2)
        # print(inlier.shape)
        xl1 = inlier[:,0:2].T
        # print(x1.shape)
        xl2 = inlier[:,2:4].T
        # print(x1.shape)
        Camera2 = np.dot(intrin2, np.dot(rotation[i], np.hstack((I, -translation[i].reshape(3,1)))))
        X = cv2.triangulatePoints(Camera1, Camera2, xl1, xl2)  
        pts.append(X)
        
    return pts
def GetRotTrans(pts,rot_mat,trans_mat):
     max_p=0
     best_v=0 #Takes only when chirality test is passed
     for i in range(len(pts)):
        print(len(pts))
        ptsa = pts[i]
        ptsa = ptsa/ptsa[3, :] 
        num=checkcheiral(ptsa, rot_mat[i], trans_mat[i])
        # print("nums",num)
        if num > max_p:
            best_v = i
            max_p = num
        else:
            break
     Rot,Trans,X3D=rot_mat[best_v],trans_mat[best_v],pts[best_v]
     return Rot,Trans,X3D 

######################################################## RECTIFICATION ###############################################################
#Referece from : https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
def draw_lines(img, lines, keypoints,flag):
    for i, (x, y) in enumerate(keypoints):
        a, b, c = lines[i]
        if not flag:
            y0, y1 = 0, img.shape[0] - 1
            x0 = int(-(b*y0+c)/a)
            x1 = int(-(b*y1+c)/a)
        else:
            x0, x1 = 0, img.shape[1] - 1
            y0 = int(-(a*x0+c)/b)
            y1 = int(-(a*x1+c)/b)
        cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
    
    return img
def epipolar(k1,k2,img1,img2,F,recti_flag=False):
    line=cv2.computeCorrespondEpilines(k2.reshape(-1,1,2),2,F)
    line=line.reshape(-1,3)
    img5=draw_lines(img2,line,k2,recti_flag)
    line2=cv2.computeCorrespondEpilines(k1.reshape(-1,1,2),1,F)
    line2=line2.reshape(-1,3)
    img3=draw_lines(img1,line2,k1,recti_flag)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    concat_img = np.concatenate((img5, img3), axis = 1)
    concat_img = cv2.resize(concat_img, (1920, 660))
    return line,line2,concat_img
def rectified_lines(img1,img2,f,s1,s2):
    hei1,wid1=img1.shape[:2]
    hei2,wid2=img2.shape[:2]
    #Calculate Rectification using In-Built function 
    _,H1,H2=cv2.stereoRectifyUncalibrated(s1,s2,f,(wid1,hei1))
    print("Rectified H1 is ",H1)
    print("Rectified H2 is ",H2)
    #Warp along homoggraphy
    img1_r=cv2.warpPerspective(img1,H1,(wid1,hei1))
    img2_r=cv2.warpPerspective(img2,H2,(wid2,hei2))
    chan1=cv2.perspectiveTransform(s1.reshape(-1, 1, 2), H1).reshape(-1,2)
    chan2=cv2.perspectiveTransform(s2.reshape(-1, 1, 2), H2).reshape(-1,2)
    #Recrtified F = H2.T ^-1 . F.H1^-1 as K RT = H^-1
    H2inv=np.linalg.inv(H2.T)
    H1inv=np.linalg.inv(H1)
    Frect=np.dot(H2inv,np.dot(f,H1inv))
    return img1_r,img2_r,chan1,chan2,Frect

############################################ MAIN ####################
f,s=RANSAC(key1,key2)
# print("THIS IS THE LENGTH",len(s))
if dataset==1:
    E=essential(camd0,camd1,f)
    Rot,trans=getcampose(E)
    xf=getPoints(camd0,camd1,s,Rot,trans)
elif dataset==2:
    E=essential(camd20,camd21,f)
    Rot,trans=getcampose(E)
    xf=getPoints(camd0,camd1,s,Rot,trans)
elif dataset==3:
    E=essential(cam0,cam1,f)
    Rot,trans=getcampose(E)
    xf=getPoints(camd0,camd1,s,Rot,trans)

# print(Rot)
# xf=getPoints(camd0,camd1,s,Rot,trans)
print("############################################################## RESULTS ################################################")
R,T,X3=GetRotTrans(xf,Rot,trans)
print("Fundamental Matrix \n",f)
print("Essential Matrix \n",E)
print("Rotation Obtained: \n",R)
print("Translation Obtained: \n",T)

print(img2.shape)
imga=img1.copy()
imgb=img2.copy()

s1=s[:,0:2]
s2=s[:,2:4]
lines1, lines2, result = epipolar(s1, s2,img1, img2,f,recti_flag=False)
if dataset==1:
    cv2.imwrite("unrectified_epipolar_artroom.jpg", result)
elif dataset==2:
    cv2.imwrite("unrectified_epipolar_chess.jpg", result)
elif dataset==3:
    cv2.imwrite("unrectified_epipolar_ladder.jpg", result)

img1_r, img2_r,tranf1, tranf2,F_rect=rectified_lines(imga,imgb,f,s1,s2)

lines1_r, lines2_r, result_2 = epipolar(tranf1, tranf2, img1_r, img2_r, F_rect,recti_flag=True)
if dataset==1:
    cv2.imwrite("rectified_epipolar_artroom.jpg", result_2)
elif dataset==2:
    cv2.imwrite("rectified_epipolar_chess.jpg", result_2)
elif dataset==3:
    cv2.imwrite("rectified_epipolar_ladder.jpg", result_2)

