import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
	combine, image1, image2 = getImage()
	print("input key to Process image(press 'H' for help, press 'q' to quit):")
	key = str(input())
	while key != 'q':
		if key == 'h':
			n = input("Enter the varience of Gussian scale(n):")
			windowSize = input("Enter windowSize :")
			k = input("Enter the weight of the trace in the harris conner detector(k)[0, 0.5]:")
			threshold = input("Enter threshold:")
			print("processing...")
			rst = harris(combine, n, windowSize, k, threshold)
			showWin(rst)
		if key == 'f':
			rst = featureVector(image1, image2)
			showWin(rst)
		if key == 'b':
			rst = betterLocalization(combine)
			showWin(rst)
		if key == 'H':
			help()
		print("Input key to Process image(press 'H' for help, press 'q' to quit):")
		key = str(input())


def getImage():
	if len(sys.argv) == 3:
		image1 = cv2.imread(sys.argv[1])
		image2 = cv2.imread(sys.argv[2])
	else:
			capture = cv2.VideoCapture(0)
			for i in range(0,15):
				retval1,image1 = capture.read()
				retval2,image2 = capture.read()
			if retval1 and retval2:
				cv2.imwrite("capture1.jpg", image1)
				cv2.imwrite("capture2.jpg", image2)
	combine = np.concatenate((image1, image2), axis=1)
	return combine, image1, image2;


def showWin(img):
	plt.imshow(img, cmap='gray')
	plt.show()

def cvt2Gray(img):
	image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return image_bw

def smooth(img, n):
	kernel = np.ones((n, n), np.float32)/(n * n)
	img1 = cv2.filter2D(img, -1, kernel)
	return img1

def harris(img, n, windowSize, k, threshold):
	n = int(n)
	windowSize = int(windowSize)
	k = float(k)
	threshold = int(threshold)
	copy = img.copy()
	rList = []
	height = img.shape[0]
	width = img.shape[1]	
	offset = int(windowSize / 2)
	img = cvt2Gray(img)
	img = np.float32(img)
	img = smooth(img, n)
	dy, dx = np.gradient(img)
	Ixx = dx ** 2
	Ixy = dy * dx
	Iyy = dy ** 2

	for y in range(offset, height - offset):
			for x in range(offset, width - offset):
				windowIxx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIxy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				Sxx = windowIxx.sum()
				Sxy = windowIxy.sum()
				Syy = windowIyy.sum()
				det = (Sxx * Syy) - (Sxy ** 2)
				trace = Sxx + Syy
				r = det - k *(trace ** 2)
				rList.append([x, y, r])
				if r > threshold:
							copy.itemset((y, x, 0), 0)
							copy.itemset((y, x, 1), 0)
							copy.itemset((y, x, 2), 255)
							cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
	return copy

def featureVector(image1, image2):
	# Initiate SIFT detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(image1,None) # returns keypoints and descriptors
	kp2, des2 = orb.detectAndCompute(image2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	kp1List = []
	kp2List = []
	for x in matches:
		(x1, y1) = kp1[x.queryIdx].pt
		(x2, y2) = kp2[x.trainIdx].pt
		kp1List.append((x1, y1))
		kp2List.append((x2, y2))
	for i in range(0, 50):
		point1 = kp1List[i]
		point2 = kp2List[i]
		cv2.putText(image1, str(i), (int(point1[0]), int(point1[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
		cv2.putText(image2, str(i), (int(point2[0]), int(point2[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	rst = np.concatenate((image1, image2), axis=1)
	return rst

def betterLocalization(img):
	gray = cvt2Gray(img)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)

	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

	rst = np.hstack((centroids,corners))
	rst = np.int0(rst)
	img[rst[:,1],rst[:,0]]=[0,0,255]
	img[rst[:,3],rst[:,2]] = [0,255,0]
	return img

def help():
	print("'h': For a estimate image gradients and apply Harris corner detection algorithm.")
	print("'b': For better localization of each corner.")
	print("'f': For computing a feature vector for each corner were detected.\n")


if __name__ == '__main__':
	main()
