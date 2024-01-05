import cv2
import numpy as np
import matplotlib.pyplot as plt
from Code import *
from fundamentalmatrix import rectified_lines,img1_rect,img2_rect,F_rectified
################ DISPARITY ##########################1######
img1=cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)
#Sliding window technique
### In sliding window we need to give block size and also a maximum disparity as a number to denote
#the distance between the disparities
maxdisp=20
def disparity(img1,img2,blk=3):
    hn, wn = img1.shape[0],img1.shape[1]
    disp_map = np.zeros((hn, wn), dtype=np.float32)
    #We are taking the block size to be the size of matrix to compare image patches 
    #since our blk size is not constant , we take a range from blk to height of image - blk. Iterating over (2*blk+1)
    for i in range(blk,hn-blk-1):
        for j in range(blk+maxdisp,wn-blk-1):
            min_ssd = np.zeros([maxdisp,1])
            # min_norm=np.zeros([maxdisp,1])
            ha=img1[(i-blk):(i+blk),(j-blk):(j+blk)]
            for m in range(0,maxdisp):
                ra=img2[(i-blk):(i+blk),(j-m-blk):(j-m+blk)]
                min_ssd[m]=np.sum((ha[:,:]-ra[:,:])**2)
                # min_norm[m] = np.sum(ha * ra) / (np.sqrt(np.sum(ha**2)) * np.sqrt(np.sum(ra**2)) + (1e-5))
            disp_map[i, j] = np.argmin(min_ssd)
            # disp_map[i,j]=np.argmax(min_norm)
    disp_map = (disp_map / disp_map.max()) * 255
    disp_map = disp_map.astype(np.uint8)
    # cv2.imwrite('Gray_diparity_of_dataset.png', disp_map)
    return disp_map,min_ssd
def calcdepth(img1,dmap,min,baseline,focal):
    #Depth is given as ratio of product of baseline and focal len of camera and the disparity
    depth = np.zeros(shape=img1.shape).astype(float)
    depth[dmap>0]= (focal*baseline)/(dmap[dmap>0])
    img_depth = ((depth/depth.max())*255).astype(np.uint8)
    return img_depth

dmap,min=disparity(img1,img2)
print("THE DISPARITY MAP \n",min)

# print(min)
if dataset==1:
    cv2.imwrite('Gray_diparity_of_artrooma.png', dmap)
    heatmap = cv2.applyColorMap(dmap, cv2.COLORMAP_INFERNO)
    cv2.imwrite('HeatMap_disparity_of_artrooma.png',heatmap)
    d=calcdepth(img1,dmap,min,baseline1,camd0[0,0])
    d_heat=cv2.applyColorMap(d,cv2.COLORMAP_HOT)
    cv2.imwrite('Depth_of_artrooma.png',d)
    cv2.imwrite('HeatDepth_of_artrooma.png',d_heat)
elif dataset==2:
    cv2.imwrite('Gray_diparity_of_chess.png', dmap)
    heatmap = cv2.applyColorMap(dmap, cv2.COLORMAP_INFERNO)
    cv2.imwrite('HeatMap_disparity_of_chess.png',heatmap)
    d=calcdepth(img1,dmap,min,baseline1,camd20[0,0])
    cv2.imwrite('Depth_of_chess.png',d)
    d_heat=cv2.applyColorMap(d,cv2.COLORMAP_HOT)
    cv2.imwrite('HeatDepth_of_chess.png',d_heat)
elif dataset==3:
    cv2.imwrite('Gray_diparity_of_laddera.png', dmap)
    heatmap = cv2.applyColorMap(dmap, cv2.COLORMAP_INFERNO)
    cv2.imwrite('HeatMap_disparity_of_laddera.png',heatmap)
    d=calcdepth(img1,dmap,min,baseline1,cam0[0,0])
    cv2.imwrite('Depth_of_laddera.png',d)
    d_heat=cv2.applyColorMap(d,cv2.COLORMAP_HOT)
    cv2.imwrite('HeatDepth_of_laddera.png',d_heat)



