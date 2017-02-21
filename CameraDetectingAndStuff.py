from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


# capturing video through connected camera
capture = cv2.VideoCapture(0)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
	   
def order_points(pts):

	xSorted = pts[np.argsort(pts[:, 0]), :]

	leftMost = xSorted[:2, :]
	rightmost = xSorted[2:, :]

	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	return np.array([tl, tr, br, bl], dtype="float32")

# shows live video on computer screen
def distances():
	
	i = 0

	ret, frame = capture.read()
	temp_image = frame
	# convert BGR to HSV
	hsv = cv2.cvtColor(temp_image, cv2.COLOR_BGR2HSV)

	# define range of teal color in HSV
	lower_teal = np.array([30,1,230])
	upper_teal = np.array([55,6,255])

	# threshold the HSV image to get only teal colors
	mask = cv2.inRange(hsv, lower_teal, upper_teal)

	#filter by erosion
	kernel = np.ones((1, 1), dtype = np.int8)
	image_erode = cv2.erode(mask, kernel, iterations=1)
			
	#start of ratios
	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable

	#Loop over the original points and draw them
	 
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(image_erode, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	try:
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		(cnts, _) = contours.sort_contours(cnts)
	except:
		distances()
		
	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		orig = image_erode.copy()
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)  

		if cv2.contourArea(c) < 200:
			continue		
		# compute the rotated bounding box of the contour

		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		for (x, y) in box:
			cv2.circle(image_erode, (int(x), int(y)), 5, (0, 0, 255), -1)
			#unpack the ordered bounding box, then compute the midpoint
			#Between the top-left and top-right coordinates, followed by
			#The midpoint between bottom-left and bottom-right coordinates
			(tl, tr, br, bl) = box
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			#Compute the midpoint between the top-left and top-right points
			#Followed by the midpoint between the top-right and bottom-right
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)
			#Draw the midpoints on the image

			cv2.line(image_erode, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
			cv2.line(image_erode, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)
			dVert = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
			dHorz = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

			#Vertical Distance
			dimVert = dVert / 1 # shouldn't it be 72???
			#Horizontal Distance
			dimHorz = dHorz / 1 # shouldn't it be 72???
			area = dimVert * dimHorz		  
			cv2.putText(image_erode, "{:.1f}px".format(dimVert),
						(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
						0.65, (255, 255, 255), 2)
			cv2.putText(image_erode, "{:.1f}px".format(dimHorz),
						(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
						0.65, (255, 255, 255), 2)
			cv2.imshow('frame', frame)
			cv2.imshow('mask', mask)
			cv2.imshow('erodedImage', image_erode)

			if area > 1000 and dimVert > dimHorz:
				dict_list = []
				dim_dict = {'dimVert':dimVert,'dimHorz':dimHorz, 'area':area, 'bottom midpoint': blbrX,}
				dict_list.append(dim_dict)
				print "{0}".format(dict_list)
				dict_list = []
				return dict_list

				# if left box is in [1] and right box is in [0], switch them		
				if dict_list[0].get('bottom midpoint') > dict_list[1].get('bottom midpoint'):
					temp_dict = dict_list[1]
					dict_list[1] = dict_list[0]
					dict_list[0] = temp_dict

 		#cv2.putText(image_erode, "Object #{}".format(i + 1), (int(box[0][0] - 15), int(box[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
		i = i + 1
		

'''
if dict_list[0].get('dimHorz') < dict_list[1].get('dimHorz'):
	#reposition to turn right
elif dict_list[0].get('dimHorz') >= dict_list[1].get('dimHorz'):
	#reposition to turn left or go forward
'''
	#except: 
		#Distances()
		
		#ratios and movement
		#if (4.8 < dimVert < 5.2 or 1.8 < dimHorz < 2.2):
		#		right_motor = 0.5
		
		
		#pixel per metric
		

ideal_size = 15000

if dict_list[0].get('area') < ideal_size:
	if dict_list[0].get('dimHorz') <= dict_list[1].get('dimHorz'):
		vp.go_forward_and_left()
	elif dict_list[0].get('dimHorz') > dict_list[1].get('dimHorz'):
		vp.go_forward_and_right()
elif dict_list[0].get('area') = ideal_size:
	vp.execute_gear_drop_off()


#dict_list[0].get('dimVert')

#area = dimA * dimB
#print "Rect Area: {0} sq in".format(area)

'''
if area > 0.08: #filters noise
	continue
	if area < should_be_area:
		# reposition
	elif area = should_be_area:
		# initiate ghost code??
	elif area > should_be_area:
		# reposition or something 
else:
	# keep checking
'''	




	
if __name__ == '__main__':
	while True:
		distances()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
capture.release()
cv2.destroyAllWindows()			
