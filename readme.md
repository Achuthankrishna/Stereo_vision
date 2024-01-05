## Stereo Vision 
This repository provides a comprehensive implementation of fundamental processes for obtaining stereo vision using OpenCV, without relying on external libraries. Stereo vision is a crucial aspect of computer vision, enabling the calculation of depth, correspondence, and disparity from stereo image pairs.
![Figure [Stetro Vision]](./Results/rectified_epipolar_chess.jpg)

## Steps to Run
First clone the current repo into any directory. Make sure ou have latest version of python installed and have OpenCv version geaater than 3.4.1 

For only seeing SIFT Features use,
```bash
python3 Code.py
```
For only fundamental matrix, estimation matrix and epipolar lines , use 
```bash
Python3 fundamentalmatrix.py
```
For Running the whole process, use 
```bash
python3 correspondence.py
```
