import cv2
import numpy as np


# capturing video through connected camera
capture = cv2.VideoCapture(0)
capture.open(0)

# shows live video on computer screen
while True:
	ret, frame = capture.read()
	tempImage = frame    
	# convert BGR to HSV
	hsv = cv2.cvtColor(tempImage, cv2.COLOR_BGR2HSV)

	# define range of teal color in HSV
	lowerTeal = np.array([38,1,230])
	upperTeal = np.array([55,6,255])

	# threshold the HSV image to get only teal colors
	mask = cv2.inRange(hsv, lowerTeal, upperTeal)


	#filter by erosion
	kernel = np.ones((5, 5), dtype = np.int8)
	imageErode = cv2.erode(mask, kernel, iterations=1)

	#res = cv2.bitwise_and(frame, frame, mask = mask)

	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('erodedImage', imageErode)
	#cv2.imshow('res', res)
	
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

 
capture.release()
cv2.destroyAllWindows()
