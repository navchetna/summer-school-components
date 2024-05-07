import cv2 as cv
from main import poseDetector

input = cv.imread("pose2.jpg")
output = poseDetector(input)
print('Image generated successfully!')
cv.imwrite("estimated_pose1.jpg", output)