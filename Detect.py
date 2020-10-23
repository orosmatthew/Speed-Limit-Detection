# imports

import numpy as np
import cv2
import pytesseract
import time
from scipy.ndimage import interpolation as inter
import threading

# get sign classifier

sign_cascade = \
	cv2.CascadeClassifier(r"D:\STUFF\Programming\Speed-Limit-Detection\Speedlimit_HAAR_ 16Stages.xml"
						  )

# set up pytesseract for OCR

pytesseract.pytesseract.tesseract_cmd = \
	r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# set up webcam

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

detect_queue = []

limit = 0

end = False


# detect thread in separate thread

def detect_text():
	global detect_queue
	global limit
	global end
	while True:

		if end == True:
			return

		if len(detect_queue) != 0:
			img = detect_queue[0]
			config = '-l eng --oem 1 --psm 11 digits'
			text = pytesseract.image_to_string(cor[1], config=config)

			if '10' in text:
				print('Limit: 10')
				limit = 10
			elif '15' in text:
				print('Limit: 15')
				limit = 15
			elif '20' in text:
				print('Limit: 20')
				limit = 20
			elif '25' in text:
				print('Limit: 25')
				limit = 25
			elif '30' in text:
				print('Limit: 30')
				limit = 30
			elif '35' in text:
				print('Limit: 35')
				limit = 35
			elif '40' in text:
				print('Limit: 40')
				limit = 40
			elif '45' in text:
				print('Limit: 45')
				limit = 45
			elif '50' in text:
				print('Limit: 50')
				limit = 50
			elif '55' in text:
				print('Limit: 55')
				limit = 55
			elif '60' in text:
				print('Limit: 60')
				limit = 60
			elif '65' in text:
				print('Limit: 65')
				limit = 65
			elif '70' in text:
				print('Limit: 70')
				limit = 70
			elif '75' in text:
				print('Limit: 75')
				limit = 75
			elif '80' in text:
				print('Limit: 80')
				limit = 80
			elif '85' in text:
				print('Limit: 85')
				limit = 85

			detect_queue.pop(0)
		else:

			time.sleep(0.25)


# used to correct text skewing

def correct_skew(image, delta=1, limit=5):

	def determine_score(arr, angle):
		data = inter.rotate(arr, angle, reshape=False, order=0)
		histogram = np.sum(data, axis=1)
		score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
		return (histogram, score)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV
						   + cv2.THRESH_OTSU)[1]

	scores = []
	angles = np.arange(-limit, limit + delta, delta)
	for angle in angles:
		(histogram, score) = determine_score(thresh, angle)
		scores.append(score)

	best_angle = angles[scores.index(max(scores))]

	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
							 borderMode=cv2.BORDER_REPLICATE)

	return (best_angle, rotated)


d_thread = threading.Thread(target=detect_text, args=())
d_thread.start()

# main loop

while True:

	# read camera

	(ret_val, img) = cam.read()

	# convert to gray

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# detect signs

	signs = sign_cascade.detectMultiScale(gray, 1.05, 3)

	# loop through signs

	for (x, y, w, h) in signs:

		# scale image

		sign_img = img[y:y + h, x:x + w]

		x_scale_percent = 87
		y_scale_percent = 95
		width = int(sign_img.shape[1] * x_scale_percent / 100)
		height = int(sign_img.shape[0] * y_scale_percent / 100)
		dim = (width, height)
		resized = cv2.resize(sign_img, dim,
							 interpolation=cv2.INTER_AREA)

		# correct skewed text

		cor = correct_skew(resized)

		if len(detect_queue) >= 10:
			detect_queue.pop(0)
		detect_queue.append(cor[1])

		time.sleep(0.075)

		# draw red rectangle over sign

		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# draw speed limit

	font = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (50, 50)
	fontScale = 1
	fontColor = (0, 0, 255)
	lineType = 2

	cv2.rectangle(img, (25, 15), (325, 70), (0, 0, 0), -1)

	cv2.putText(
		img,
		'Speed Limit: ' + str(limit),
		bottomLeftCornerOfText,
		font,
		fontScale,
		fontColor,
		lineType,
		)

	# show webcam

	cv2.imshow('Webcam', img)

	if cv2.waitKey(1) == 27:
		cam.release()
		break  # esc to quit

cv2.destroyAllWindows()

end = True
