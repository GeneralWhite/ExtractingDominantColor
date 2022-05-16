import cv2
import numpy as np

# set red thresh
# lower_blue=np.array([156,43,46])
# upper_blue=np.array([180,255,255])

lower_blue = np.array([10, 43, 46])
upper_blue = np.array([100, 255, 255])

img = cv2.imread('Image/black-spot (7).jpg')

# get a frame and show

frame = img

# cv2.imshow('Capture', frame)

# change to hsv model
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# get mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('Mask', mask)

# detect red
res = cv2.bitwise_and(frame, frame, mask = mask)
cv2.imshow('Result', res)

cv2.waitKey(0)
cv2.destroyAllWindows()