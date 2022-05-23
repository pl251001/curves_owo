import cv2
import np
import math


def getAngle(img):
  
  height=1800
  left = []
  right = []
  status = False
  for y in reversed(range(height)):
    for x in range(2000):
      b,g,r=img[y,x]
      if b>100:
        left=[x,y]
        break
    if status == True:
      break
      
  status = False
  for y in reversed(range(height)):
    for x in reversed(range(2000)):
      b,g,r=img[y,x]
      if b>100:
        right=[x,y]
        break
    if status == True:
      break
  distance = math.sqrt(pow((right[0]-left[0]),2) + pow((right[1]-left[1]),2))
  if distance < 300:
    if left[0]>1000:
      left=[0,left[1]]
    if right[0]<1000:
      right=[2000,right[1]]
      
  endpt = [(left[0]+right[0])//2, (left[1]+right[1])//2]
  
  img = cv2.arrowedLine(img, (1000,2000), endpt, (255, 0, 0), 10)
  
  xchng = endpt[0]-1000
  ychng = endpt[1]
  slope = ychng/xchng
  if slope > 0:
    angle = math.atan(slope)
  else:
    angle = np.pi - math.atan(abs(slope))
  return angle
   
def drawCircles(img,pts):
  for a in pts:
    x,y=a
    cv2.circle(img,(x,y),1,(0,0,255),100)
     
