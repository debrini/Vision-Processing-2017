# imports
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# capturing video through connected camera
capture = cv2.VideoCapture(0)

# midpoint formula defined as method
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
	   
# define order points
def order_points(pts):

	xSorted = pts[np.argsort(pts[:, 0]), :]

	leftMost = xSorted[:2, :]
	rightmost = xSorted[2:, :]
	
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	return np.array([tl, tr, br, bl], dtype="float32")

# measures box parameters
def distances():
	
	# shows live video on computer screen	
	#ret, frame = capture.read()
	temp_image = ('/home/pi/1ftH3ftD0Angle0Brightness.jpg')
	
	# convert BGR to HSV
	hsv = cv2.cvtColor(temp_image, cv2.COLOR_BGR2HSV)

	# define range of teal color in HSV
	lower_teal = np.array([30,1,230])
	upper_teal = np.array([55,6,255])

	# threshold the HSV image to get only teal colors
	mask = cv2.inRange(hsv, lower_teal, upper_teal)

	# filter by erosion
	kernel = np.ones((1, 1), dtype = np.int8)
	image_erode = cv2.erode(mask, kernel, iterations=1)
	 
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(image_erode, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
		
	# tries to find contours
	try:
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		(cnts, _) = contours.sort_contours(cnts)
		
	# if it finds nothing it tries again
	except:
		distances()
		
	# looks at contours
	for c in cnts:
		
		# defines box and draws the contours on the boxes
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		
		# saves box coords. into numpy array
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		
		# makes a copy of the eroded image so that contours are not 
		# drawn on original image
		orig = image_erode.copy()
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		# marks different points on box
		for (x, y) in box:
			cv2.circle(image_erode, (int(x), int(y)), 5, (0, 0, 255), -1)
			
			# defines the points + box 
			(tl, tr, br, bl) = box
			
			# defines horizontal and vertical midpoints on each box 
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)
			
			# draw the midpoints on the image
			cv2.line(image_erode, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
			cv2.line(image_erode, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)
			
			# finds dimensions of each box based on coordinate points found
			dVert = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
			dHorz = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

			# vertical + horizontal distance
			dimVert = dVert
			dimHorz = dHorz
			
			# defines area 
			area = dimVert * dimHorz
			
			# draws vertical line + writes vertical distance on image  
			cv2.putText(image_erode, "{:.1f}px".format(dimVert),
						(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
						0.65, (255, 255, 255), 2)
			
			# draws horizontal line + writes horizontal distance on image  			
			cv2.putText(image_erode, "{:.1f}px".format(dimHorz),
						(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
								0.65, (255, 255, 255), 2)
			
			# displays images
			cv2.imshow('frame', frame)
			cv2.imshow('mask', mask)
			cv2.imshow('erodedImage', image_erode)
			
			# filters out boxes with small area and horizontal boxes
			if area > 1000 and dimVert > dimHorz:
				
				# saves parameters of each box into a dictionary
				dim_dict = {'dimVert':dimVert,'dimHorz':dimHorz, 'area':area, 'bottom midpoint': blbrX,}
				
				# saves dictionary for each box into a list
				dict_list = []
				dict_list.append(dim_dict)
				
				# prints dict_list
				#print "{0}".format(dict_list)
				
				
				return dict_list
				return dim_dict
 
				# if the left box is in [1] and the right  box is in [0], this code will switch them
				# pretty much a bubble sort	
				if dict_list[0].get('bottom midpoint') > dict_list[1].get('bottom midpoint'):
					temp_dict = dict_list[1]
					dict_list[1] = dict_list[0]
					dict_list[0] = temp_dict
					boxes_midpoint = abs(dict_list[1].get('bottom midpoint') - dict_list[0].get('bottom midpoint'))
					return dict_list
					return dim_dict
				
			
			# clears list before code runs again
			dict_list = []
			

# loops code
if __name__ == '__main__':
	while True:
		distances()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
capture.release()
cv2.destroyAllWindows()			
