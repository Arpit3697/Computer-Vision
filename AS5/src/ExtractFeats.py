#Name: Arpit Patel
#Id: A20424085

'''
extract feature points
use the openCV functions

Usage:
    ExtractFeats.py filename
    filename : an image contain 3d chessboard

Keys:
    select image window
    press any key to exit

Output:
    correspondencePoints.txt
    A point correspondence file (3D-2D)
'''
import cv2
import numpy as np
import sys


print(__doc__)
image = cv2.imread('chessboard.jpg')

    # count termination criteria
term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points like (0,0,0), (1,0,0) ....,(6,5,0)
obj_point = np.zeros((6*7,3), np.float32)
obj_point[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    #convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # now find the chess board corners using grayscale image
ret, corners = cv2.findChessboardCorners(gray_image, (7,6), None)

    # If corner found add object points, image points
if ret:
	#refine the corner using cornerSubPix 
    corners2=cv2.cornerSubPix(gray_image,corners, (11,11), (-1,-1), term_criteria)

        # Draw and display the corners
    cv2.drawChessboardCorners(image, (7,6), corners2, ret)
    cv2.imshow('image', image) #show the image with corners
    with open('correspondencePoints.txt', 'w') as f:
        for i, j in zip(obj_point, corners.reshape(-1,2)): #zip function concatenate both value 
            f.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(j[0]) + ' ' + str(j[1]) + '\n') # wirte the values in the file

    cv2.waitKey(0)
cv2.destroyAllWindows()


