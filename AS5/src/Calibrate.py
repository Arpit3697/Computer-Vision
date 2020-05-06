#Name: Arpit Patel
#Id : A20424085
'''
Non-planar Calibration

Usage:
    Calibrate.py filename
    filename : A points correspondence file (3D-2D)

Output:
    - print intrinsic and extrinsic parameters
    - print mean square error
'''
import cv2
import numpy as np
import sys
from PIL import Image
import os

#Funtion to calculate Intrinsic and Extrinsic parameter
def ComputeParameters(a1, a2, a3, b):
    np.set_printoptions(formatter={'float': "{0:.6f}".format})
    normalizationP = 1 / np.linalg.norm(a3.T)
    u0 = normalizationP ** 2 * (a1.T.dot(a3))
    v0 = normalizationP ** 2 * (a2.T.dot(a3))
    a22 = a2.T.dot(a2)
    av = np.sqrt(normalizationP ** 2 * a22 - v0 ** 2)
    a1xa3 = np.cross(a1.T, a3.T)
    a2xa3 = np.cross(a2.T, a3.T)
    s = (normalizationP ** 4) / av * a1xa3.dot(a2xa3.T)
    a12 = a1.T.dot(a1)
    au = np.sqrt(normalizationP ** 2 * a12 - s ** 2 - u0 ** 2)
    KStar = np.array([[au, s, u0],[0, av, v0],[0, 0, 1]])
    littleE = np.sign(b[2])
    TStar = littleE * normalizationP * np.linalg.inv(KStar).dot(b).T
    R3 = littleE * normalizationP * a3
    R1 = normalizationP ** 2 / av * a2xa3
    R2 = np.cross(R3, R1)
    RStar = np.array([R1.T, R2.T, R3.T])
    print("u0, v0 = %f, %f\n" % (u0, v0))
    print("alphaU,alphaV = %f, %f\n" % (au, av))
    print("s = %f\n" % s)
    print("K* = %s\n" % KStar)
    print("T* = %s\n" % TStar)
    print("R* = %s\n" % RStar)

#Function to calculate Mean Square Error
def MeanSquareError(M, object_point, image_point):
    #Define M1,M2 and M3
    M1 = M[0][:4]
    M2 = M[1][:4]
    M3 = M[2][:4]
    Mean_Sq_Er = 0
    for i, j in zip(object_point, image_point):
        Xi = j[0]
        Yi = j[1]
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        Exi = (M1.T.dot(pi)) / (M3.T.dot(pi))
        Eyi = (M2.T.dot(pi)) / (M3.T.dot(pi))
        Mean_Sq_Er += ((Xi - Exi) ** 2 + (Yi - Eyi) ** 2)
    Mean_Sq_Er = Mean_Sq_Er / len(object_point)
    print("Mean Square Error = %s\n" % Mean_Sq_Er)

#Function to calculate Value of Matrix A
def MatirxA(object_point, image_point):
    A = []
    zeros = np.zeros(4)
    for j, k in zip(object_point, image_point):
        pi = np.array(j)
        pi = np.concatenate([pi, [1]])
        Xipi = k[0] * pi
        Yipi = k[1] * pi
		#Append Value into A
        A.append(np.concatenate([pi, zeros, -Xipi]))
        A.append(np.concatenate([zeros, pi, -Yipi]))
    return np.array(A) #REturn Matrix A

#Function To calculate value of Matrix M
def MatrixM(A):
    M = []
    u, s, v = np.linalg.svd(A, full_matrices = True)
    M = v[-1].reshape(3, 4) #Defining M from v
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    b = []
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1))
    return a1, a2, a3, b, M

def ReadData():
    object_point,image_point = [],[] #Initialize Object Point and Image Point
    d = open('correspondingPoints_test.txt').readlines() #Read Data from file
    for i in d:
        point = i.split()
        object_point.append([float(p) for p in point[:3]])
        image_point.append([float(p) for p in point[3:]])
    return object_point, image_point


print(__doc__)
object_point, image_point = ReadData()
A = MatirxA(object_point, image_point)
a1, a2, a3, b, M = MatrixM(A)
ComputeParameters(a1, a2, a3, b)
MeanSquareError(M,object_point,image_point)

