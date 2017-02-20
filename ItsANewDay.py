from scipy.spatial import distance as dist
from imutils import perspective
import numpy as np
import cv2
import imutils
from imutils import contours
import os

capture = cv2.VideoCapture(0)
capture.open(0)

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

	
gray_image = cv2.cvtColor('frame', cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', gray_image)
cv2.imshow('gray_image', gray_image)
ret, thresh = cv.threshold(gray_image, 57, 255, cv.THRESH_BINARY)
cv2.imwrite('binaryImage.jpg', thresh)
cv2.imshow('binaryImage', thresh)

edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
cv2.imshow('Camera', frame)

	
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points_old(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-c#008D84#008D84#008D84oodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")
	
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7, 7), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(img, 50, 100)
edged1 = cv2.dilate(edged, None, iterations=1)
edged2 = cv2.erode(edged1, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
# sort the contours from left-to-right and initialize the bounding box
# point colors
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
# Loop over contours individually
objectList = []
for (i, c) in enumerate(cnts):
	# If the contour not sufficently large, then ignore it.
	if cv2.contourArea(c) < 50:
		continue
	
	elif cv2.contourArea (c) >= 150:
		objectList.append((i, c))
		
		orig = img.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.BoxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		
		print("Object #{}:".format(i + 1))
		print(box)
		rect = order_points_old(box)
		print(rect.astype("int"))
		#print("")
		for (x, y) in box:
			#cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
			
			# unpack the ordered bounding box, then compute the midpoint
			# between the top-left and top-right coordinates, followed by
			# the midpoint between bottom-left and bottom-right coordinates
			(tl, tr, br, bl) = box
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
		 
			# compute the midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-righ and bottom-right
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)
		 
			# draw the midpoints on the image
			'''
			cv2.midpoint(orig, (int(tlblX, int(tlblY)), 5, (255, 0, 0), -1)
			cv2.midpoint(orig, (int(trbrX, int(trbrY)), 5, (255, 0, 0), -1)
			cv2.midpoint(orig, (int(tltrX, int(tltrY)), 5, (255, 0, 0), -1)
			cv2.midpoint(orig, (int(blbrX, int(blbrY)), 5, (255, 0, 0), -1)
			'''
			
			#cv2.imshow(pt, image)
							
			cv2.rectangle = (orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
			cv2.rectangle = (orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
			cv2.rectangle = (orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
			cv2.rectangle = (orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	 
			# draw lines between the midpoints
			cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
				(255, 0, 255), 2)
			
			# show the original coordinates
			
				# order the points in the contour such that they appear
				# in top-left, top-right, bottom-right, and bottom-left
				# order, then draw the outline of the rotated bounding box
			 
				# check to see if the new method should be used for
				# ordering the coordinates
				#if args["new"] > 0:
					#rect = perspective.order_points(box)
			 
				# show the re-ordered coordinates
				
				# order the points in the contour such that they appear
				# in top-left, top-right, bottom-right, and bottom-left
				# order, then draw the outline of the rotated bounding
				# box
			rect = order_points_old(box)
			 
				# check to see if the new method should be used for
				# ordering the coordinates
				#if args["new"] > 0:
					#rect = perspective.order_points(box)
			 
				# show the re-ordered coordinates
			print(rect.astype("int"))
			print("")
				# loop over the original points and draw them
			for ((x, y), color) in zip(rect, colors):
				cv2.circle(img, (int(x), int(y)), 5, color, -1)
			 
				# draw the object num at the top-left corner
			cv2.putText(img, "Object #{}".format(i + 1),
				(int(rect[0][0] - 15), int(rect[0][1] - 15)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
			 
				# show the image
			cv2.imshow("Image", img)
			#cv2.waitKey(0)
		horizDist = (((tr - tl) ** 2) + ((br - bl) ** 2)) ** 0.5 
		vertDist = (((tr - br) ** 2) + ((tl - bl) ** 2 )) ** 0.5
		print(((vertDist) / 72) + ((horizDist) / 72))
		
if cv2.waitKey(1) & 0xFF == ord('q'):


	capture.release()
	cv2.destroyAllWindows()
