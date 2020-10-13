#imports
import numpy as np
import cv2
import pytesseract
import time
import imutils
import argparse
from scipy.ndimage import interpolation as inter
from imutils.object_detection import non_max_suppression

#get sign classifier
sign_cascade = cv2.CascadeClassifier(r"C:\Users\orosm\Desktop\Speed Limit Python Env\Speedlimit_HAAR_ 16Stages.xml")

#set up EAST text detection
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

net = cv2.dnn.readNet(r"C:\Users\orosm\Desktop\Speed Limit Python Env\frozen_east_text_detection.pb")

#set up pytesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#set up webcam
cam = cv2.VideoCapture(1)

#used to correct text skewing
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

#used for EAST text detection algorithm
def decode_predictions(scores, geometry):

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < 0.4:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)

#main loop
while True:
	
	#read camera
	ret_val, img = cam.read()
	#convert to gray
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#detect signs
	signs = sign_cascade.detectMultiScale(gray, 1.05, 3)
	
	#loop through signs
	for (x,y,w,h) in signs:
		
		#copy and resize image
		image = img[y:y+h, x:x+w]
		orig = image.copy()
		(origH, origW) = image.shape[:2]

		(newW, newH) = (320, 320)
		rW = origW / float(newW)
		rH = origH / float(newH)

		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]

		#propagate EAST network
		blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		#handle EAST output
		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		
		#text output results
		results = []
		
		#loop through text lines within sign
		for (startX, startY, endX, endY) in boxes:
			
			#scale boxes and add padding
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			dX = int((endX - startX) * 0.15)
			dY = int((endY - startY) * 0.15)

			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))
			roi = orig[startY:endY, startX:endX]
			
			#correct skewed text
			cor = correct_skew(roi)
			
			#run pytesseract and get resulting text from EAST detection
			config = ("-l eng --oem 1 --psm 7 digits")
			text = pytesseract.image_to_string(cor[1], config=config)
			results.append(((startX, startY, endX, endY), text))
			
		#sort results
		results = sorted(results, key=lambda r:r[0][1])
		
		#loop through text results
		for ((startX, startY, endX, endY), text) in results:

			#print text result
			if "10" in text:
				print("Limit: 10")
			elif "15" in text:
				print("Limit: 15")
			elif "20" in text:
				print("Limit: 20")
			elif "25" in text:
				print("Limit: 25")
			elif "30" in text:
				print("Limit: 30")
			elif "35" in text:
				print("Limit: 35")
			elif "40" in text:
				print("Limit: 40")
			elif "45" in text:
				print("Limit: 45")
			elif "50" in text:
				print("Limit: 50")
			elif "55" in text:
				print("Limit: 55")
			elif "60" in text:
				print("Limit: 60")
			elif "65" in text:
				print("Limit: 65")
			elif "70" in text:
				print("Limit: 70")
			elif "75" in text:
				print("Limit: 75")
			elif "80" in text:
				print("Limit: 80")
			elif "85" in text:
				print("Limit: 85")
	
	#show webcam
	cv2.imshow('Webcam', img)
	
	if cv2.waitKey(1) == 27: 
		break  # esc to quit
		
cv2.destroyAllWindows()
