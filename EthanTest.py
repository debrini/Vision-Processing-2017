import cv2 as cv
import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool
import os
import pdb

path1 = "/home/pi/Field Images/Vision Images/Red Boiler"
path2 = "/home/pi/LineyImages"

listing = os.listdir(path1)     

#p = Pool(1) # process 1 images simultaneously

#for x in range(1,5):
	


for fileName in listing:
	#pdb.set_trace()
	img = cv.imread(path1 + '/' + fileName)

	gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	cv.imwrite('gray_image.jpg', gray_image)
	#cv.imshow('gray_image', gray_image)
	'''
	capture = cv.VideoCapture(0)
	capture.open(0)
	
	while True:
		ret, frame = capture.read()
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#   cv.imshow('frame', gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
	'''
	ret,thresh = cv.threshold(gray_image,57,255,cv.THRESH_BINARY)
	cv.imwrite('binaryImage.jpg', thresh)
	cv.imshow('binaryImage', thresh)


	edges = cv.Canny(thresh,50,150,apertureSize = 3)

	lines = cv.HoughLines(edges,1,np.pi/180,50)
	try:
		for rho,theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
	except TypeError:
		print "Found no Lines in\n" + fileName
	
	
	
	edges = cv.Canny(gray_image,50,150,apertureSize = 3)
	minLineLength = 100
	maxLineGap = 10
	lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
	
	
	#for x1,y1,x2,y2 in lines[0]:cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)


	#cv.imwrite('/home/pi/Pictures/houghlines5.jpg',img)
	#cv.imshow(img)
	
	cv.imwrite(os.path.join(path2,fileName), img)
	

cv.waitKey(0)
#cv.DestroyAllWindows()

