import cv2
import numpy as np
import pdb
from imutils import contours as cnts
import pdb


# capturing video through connected camera
#pdb.set_trace()
tempImage = cv2.imread('/home/pi/1ftH3ftD0Angle0Brightness.jpg')
# shows live video on computer screen
# convert BGR to HSV
#try:
grayImage = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
ret, binaryImage = cv2.threshold(grayImage,10,255,cv2.THRESH_BINARY)

#hsv = cv2.cvtColor(grayImage, cv2.COLOR_BGR2HSV)

'''
# define range of teal color in HSV
lowerTeal = np.array([70,50,50])
upperTeal = np.array([100,250,250])

# threshold the HSV image to get only teal colors
mask = cv2.inRange(hsv, lowerTeal, upperTeal)
'''

kernel = np.ones((5, 5), dtype = np.int8)
imageErode = cv2.erode(binaryImage, kernel, iterations=3)
'''
edged = cv2.Canny(binaryImage, 50, 100)
edged = cv2.dilate(edged, kernel, iterations=1)
edged = cv2.erode(edged, kernel, iterations=1)
'''
cnts = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
print(cnts)
points = []
	
# list of contour points converted to suitable format to pass into cv2.minEnclosingCircle()
'''for pair in cnts:
	points.append([[pair[0], pair[1]]])
points = np.array(binaryImage)
'''
#(cen, rad) = cv2.minEnclosingCircle(points);

cv2.imwrite('/home/pi/erodedImage.jpg',imageErode)
cv2.imwrite('/home/pi/grayImage.jpg',grayImage)
cv2.imshow('frame', tempImage)
cv2.imshow('imageErode', imageErode)
#cv2.imshow('mask', mask)
#except:
#	print('color not found')
 
cv2.destroyAllWindows()
