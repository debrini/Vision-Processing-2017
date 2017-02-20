# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2 as cv

'''
Beginning of actual camera vision processing, still need to draw lines and change threshhold; may not be formatted the right way. just copied img code.

cap = cv2.VideoCapture(input # of camera)

while True:
ret, frame=cap.read()
gray_image = cv2.cvtColor('frame', cv.COLOR_BGR2GRAY)
cv.imwrite('gray_image.jpg', gray_image)
cv.imshow('gray_image', gray_image)
ret, thresh = cv.threshold(gray_image, 57, 255, cv.THRESH_BINARY)
cv.imwrite('binaryImage.jpg', thresh)
cv.imshow('binaryImage', thresh)

edges = cv.Canny(thresh, 50, 150, apertureSize=3)

lines = cv.HoughLines(edges, 1, np.pi / 180, 50)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv2.imshow('Camera', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
break


'''

img = cv.imread('/home/pi/sampleImages/Vision Images/Blue Boiler/P.jpg')

gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('gray_image.jpg', gray_image)
cv.imshow('gray_image', gray_image)
ret, thresh = cv.threshold(gray_image, 57, 255, cv.THRESH_BINARY)
cv.imwrite('binaryImage.jpg', thresh)
cv.imshow('binaryImage', thresh)


edges = cv.Canny(thresh, 50, 150, apertureSize=3)

lines = cv.HoughLines(edges, 1, np.pi / 180, 50)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
'''
cv.imwrite('/home/pi/Pictures/houghlines3.jpg',img)
cv.imshow('/home/pi/Downloads/Vision Images/LED Peg/1ftH2ftD2Angle0Brightness.jpg', img)
'''
edges = cv.Canny(gray, 50, 150, apertureSize=3)
minLineLength = 100
maxLineGap = 10
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#   cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imwrite('/home/pi/Pictures/houghlines5.jpg', img)
cv.imshow('/home/pi/Downloads/Vision Images/LED Peg/1ftH2ftD2Angle0Brightness.jpg', img)

# beginning of Ratios
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#
ap.add_argument("-i", "-/home/pi/Downloads/Vision Images/LED Peg/1ftH2ftD2Angle0Brightness.jpg", required=True,
help="path to the input image")
ap.add_argument("-w", "-width", type=float, required=True,
help="-width (in inches)")
args = vars(ap.parse_args())
'''

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv.Canny(gray, 50, 100)
edged = cv.dilate(edged, None, iterations=1)
edged = cv.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] 
if	imutils.is_cv(): 
	else cnts[1]

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
gray = cv.GaussianBlur(gray, (7, 7), 0)

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv.contourArea(c) < 100:
    continue

# compute the rotated bounding box of the contour
orig = image.copy('/home/pi/Downloads/Vision Images/LED Peg/1ftH2ftD2Angle0Brightness.jpg')
box = cv.minAreaRect(c)
box = cv.cv.BoxPoints(box) 
if imutils.is_cv() 
else cv.boxPoints(box)
box = np.array(box, dtype="int")

# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box
box = perspective.order_points(box)
cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

# loop over the original points and draw them
for (x, y) in box:
cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

# unpack the ordered bounding box, then compute the midpoint
# between the top-left and top-right coordinates, followed by
# the midpoint between bottom-left and bottom-right coordinates
(tl, tr, br, bl) = box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)


# draw the midpoints on the image
cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

# draw lines between the midpoints
cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
(255, 0, 255), 2)
cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
(255, 0, 255), 2)

# compute the Euclidean distance between the midpoints
dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

# if the pixels per metric has not been initialized, then
# compute it as the ratio of pixels to supplied metric
# (in this case, inches)
if pixelsPerMetric is None:
pixelsPerMetric = dB / args["width"]

# compute the size of the object
dimA = dA / pixelsPerMetric
dimB = dB / pixelsPerMetric

# draw the object sizes on the image
cv.putText(orig, "{:.1f}in".format(dimA),
(int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX,
0.65, (255, 255, 255), 2)
cv.putText(orig, "{:.1f}in".format(dimB),
(int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
0.65, (255, 255, 255), 2)

# show the output image
cv.imshow('gray_image', orig)
cv.waitKey(0)
cv.destroyAllWindows()
