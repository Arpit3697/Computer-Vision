import cv2
import sys
import numpy as np
import random
from scipy import ndimage
from matplotlib import pyplot as plt

img = cv2.imread("abc.jpg",1)
orig_img = img
cv2.imshow('img',img)

def sliderHandler():
    n = 3
    kernel = np.ones((n,n),np.float32)/(n*n)
    img = cv2.filter2D(img,-1,kernel)
    cv2.imshow('img',img)
        
 
while True:
    k = chr(cv2.waitKey())
    if k== 'i':
        print("You selected command : i")
        img=cv2.imread('abc.jpg',1)
        
    if k=='w':
        print("You selected command : w")
        cv2.imwrite('out.jpg',img)
        
    if k == 'g' :
        img = orig_img
        print("You selected command : g")
        img= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    if k=='G':
        img = orig_img
        print("You selected command : G")
        def grayConv(img):
            grayValue = 0.07 * img[:,:,2] + 0.72 * img[:,:,1] + 0.21 * img[:,:,0]
            gray_img = grayValue.astype(np.uint8)
            return gray_img
        img = grayConv(img)
        
    if k == 'c':
        img = orig_img
        print("You selected command : c")
        img=cv2.imread('abc.jpg',1)
        r = img.copy()
        r[:,:,0] = 0
        r[:,:,1] = 0
        
        b = img.copy()
        b[:,:,1] = 0
        b[:,:,2] = 0
        
        g = img.copy()
        g[:,:,0] = 0
        g[:,:,2] = 0
        

        img = random.choice([b,g,r])
        
    if k == 's':
        img = orig_img
        print("You selected command : s")
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
        img = cv2.blur(img_gray,(3,3))
        
    if k == 'S':
        img = orig_img
        print("You selected command : S")
        def convolve2d(image_smooth, kernel):
            kernel = np.flipud(np.fliplr(kernel))    
            output = np.zeros_like(image_smooth)            
            image_padded = np.zeros((image_smooth.shape[0] + 2, image_smooth.shape[1] + 2))   
            image_padded[1:-1, 1:-1] = image_smooth
            for x in range(image_smooth.shape[1]): 
                for y in range(image_smooth.shape[0]):
                    output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
            return output
        
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
        kernel = (np.array([[1,1,1],[1,1,1],[1,1,1]])/9)
        img= convolve2d(img_gray,kernel)
        
    if k == 'd':
        img = orig_img
        print("You selected command : d")
        img = cv2.pyrDown(img,2)
        
    if k == 'D':
        img = orig_img
        print("You selected command : D")
        image_withsmooth = cv2.blur(img,(3,3))
        img =cv2.pyrDown(image_withsmooth,2)
        
    if k == 'x':
        img = orig_img
        print("You selected command : x")
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        der=[[-1,0,1],[-2,0,2],[-1,0,1]]
        xder=np.array(der)
        normimage=np.zeros((1920,1080))
        newimagex=ndimage.convolve(img,xder,mode='constant')
        normaimagex=cv2.normalize(newimagex,normimage,0,255,cv2.NORM_MINMAX)
        
    if k== 'y':
        img = orig_img
        print("You selected command : y")
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        der=[[-1,-2,21],[0,0,0],[1,2,1]]
        xder=np.array(der)
        normimage=np.zeros((1920,1080))
        newimagex=ndimage.convolve(img,xder,mode='constant')
        normaimagex=cv2.normalize(newimagex,normimage,0,255,cv2.NORM_MINMAX)

    if k == 'p':
        img = orig_img
        print("You selected command : p")
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    if k =='r':
        img = orig_img
        print("You selected command : r")
        theta= float(input('Enter an angle --->' ))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), theta, 1)
        img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))        
        
    if k == 'm':
        img = orig_img
        print("You selected command : m")
        def gramag(x,y,img):
                sobel1=cv2.Sobel(img,cv2.CV_64F,x,y,ksize=5)
                abs_val=np.absolute(sobel1)
                s8U=np.uint8(abs_val)
                return s8U
        img = gramag(1,1,img)

    if k == 'h':
        print("You selected command : h")
        print('\n Press i to view the input img \n Press w to save the img \n Press g to view the grayscale img \n Press G to view the custom grayscale img \n Press c to view the img in various color channels\n Press s to smooth the img with trackbar functionality \n Press S to view custom smmothing of img \n Press d to downsample the img without smoothing \n Press D to smooth the img and then downsample \n Press x to perform convolution with x derivative filter with normalization\n Press y to perform convolution with y derivative filter with normalization \n Press m to show the magnitude of the gradient nomalized \n Press p to convert img to grayscale and plot the gradient vectors \n Press r to rotate the img with an angle theta \n')
        
        
    cv2.imshow('img',img)
