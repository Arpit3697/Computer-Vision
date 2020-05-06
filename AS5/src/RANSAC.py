#Name: Arpit Patel
#Id: A20424085

'''
RANSAC

Usage:
    RANSAC.py filename configname
    filrname: A points correspondence file (3D-2D)
    configname: RANSAC parameters(probability, N_min, N_max, K_max)

Output:
    - print intrinsic and extrinsic parameters

'''
import cv2
import numpy as np
import sys
import random
import math


#Funtion to calculate Intrinsic and Extrinsic parameter
def ComputeParameters(M):
    #Get a value od a1,a2,a3 from Matrix M
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    b = [] #initialize Matrix b
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1)) #reshape Matrix b
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

def Ransac(object_point, image_point, probability, N_min, N_max, K_max):
    #intializing w,k,count,inlierNumber and best_M
    w = 0.5
    # k = math.log(1 - probability) / math.log(1 - (w ** N_min))  maximum number of iteration in algorithm
    k = K_max
    np.random.seed(0)
    count = 0
    inlierNumber = 0
    best_M = None
    a_init = MatrixA(object_point, image_point)
    m_init = MatrixM(a_init)
    full_Dist = distance(m_init, object_point, image_point)
    median_Dist = np.median(full_Dist)
    t = 1.5 * median_Dist
    n = random.randint(N_min, N_max)
    
    #to find best value of M
    while(count < k and count < K_max):
        
        index = np.random.choice(len(object_point), n)
        ransac_objectpoint, ransac_imagepoint = np.array(object_point)[index], np.array(image_point)[index]
        A = MatrixA(ransac_objectpoint, ransac_imagepoint)
        M = MatrixM(A)
        d = distance(M, object_point, image_point)
        inlier = []
        for i, d in enumerate(d):
            if d < t:
                inlier.append(i)
        if len(inlier) >= inlierNumber:
            inlierNumber = len(inlier)
            inlierOp, inlierIp = np.array(object_point)[inlier], np.array(image_point)[inlier]
            A = MatrixA(ransac_objectpoint, ransac_imagepoint) #calculate A
            best_M = MatrixM(A)
        if not (w == 0 ):
            w = float(len(inlier))/float(len(image_point))
            k = float(math.log(1 - probability)) / np.absolute(math.log(1 - (w ** n)))
        count += 1;
    return inlierNumber, best_M #return inlierNumber and best value M

#Function to calculate distance 
def distance(M, object_point, image_point):
    m1 = M[0][:4]
    m2 = M[1][:4]
    m3 = M[2][:4]
    d = []
    for i, j in zip(object_point, image_point):
        xi = j[0]
        yi = j[1]
        pi = np.array(i)
        pi = np.append(pi, 1)
        exi = (m1.T.dot(pi)) / (m3.T.dot(pi)) # m1^T*p_i/m3^T*p_i
        eyi = (m2.T.dot(pi)) / (m3.T.dot(pi)) # m2^T*p_i/m3^T*p_i
        di = np.sqrt(((xi - exi) ** 2 + (yi - eyi) ** 2)) # d = sqrt((xi-exi)^2+(yi-eyi)^2)
        d.append(di)
    return d

#Function to calculate Value of Matrix A
def MatrixA(object_point, image_point):
    A = []
    zeros = np.zeros(4)
    for i, j in zip(object_point, image_point):
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        Xipi = j[0] * pi
        Yipi = j[1] * pi
        A.append(np.concatenate([pi, zeros, -Xipi]))
        A.append(np.concatenate([zeros, pi, -Yipi]))
    # print(np.array(A))
    return np.array(A)

#Function To calculate value of Matrix M
def MatrixM(A):
    M = []
    u, s, v = np.linalg.svd(A, full_matrices = True)
    M = v[-1].reshape(3, 4)
    return M



def ReadData():
    object_point,image_point = [], [] #Initialize Object Point and Image Point
    d = open('correspondingPoints_test.txt').readlines() #Read Data from file
    for i in d:
        point = i.split()
        object_point.append([float(p) for p in point[:3]])
        image_point.append([float(p) for p in point[3:]])
    return object_point, image_point


print(__doc__)
object_point, image_point = ReadData()
#Reading Configuration from RANSAC.config file
with open('RANSAC.config', 'r') as config:
        probability = float(config.readline().split()[0])
        K_max = int(config.readline().split()[0])
        N_min = int(config.readline().split()[0])
        N_max = int(config.readline().split()[0])

inlierNumber, best_M= Ransac(object_point, image_point, probability, N_min, N_max, K_max)
ComputeParameters(best_M)

