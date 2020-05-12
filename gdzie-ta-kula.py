from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# zdefiniuj jasno-pomaranczowy i ciemno-pomaranczowy
Lower_Orange = (1,140,81) # 29, 86, 6
Upper_Orange = (12,255,255) # 64, 255, 255
pts = deque(maxlen=args["buffer"])

# jesli brak argumentu --video to bierz z kamerki, jesli jest to bierz z pliku
if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)


while True:

	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# jesli brak frame tzn. koniec filmu -> break
	if frame is None:
		break


	frame = imutils.resize(frame, width=600) #zmien rozmiar
	blurred = cv2.GaussianBlur(frame, (11, 11), 0) # dodaj blur
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # paleta kolorow na HSV

	#stworz maske na podstawie rangi roznicy kolorow Lower_Orange - Upper_Orange
	mask = cv2.inRange(hsv, Lower_Orange, Upper_Orange)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	res = cv2.bitwise_and(frame,frame, mask= mask)

	#znajdz kontury i (x,y) srodka kuli
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	
	if len(cnts) > 0: #jesli znaleziono kontury cnts to ->
		c = max(cnts, key=cv2.contourArea) # c najwiekszy kontur na plaszczyznie
		((x, y), radius) = cv2.minEnclosingCircle(c) # okrag na podstawie pozycji (x,y) i srednicy konturu
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


		#jesli srednica >10 to
		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2) #rysuj kolo
			cv2.circle(frame, center, 5, (0, 0, 255), -1) #rysuj srodek
			text = str(int(x)) + " " + str(int(y)) + " " + str(int(radius))
			cv2.putText(frame, text, (int(x), int(y)-int(radius)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



	#kolejna punktow
	pts.appendleft(center)


	for i in range(1, len(pts)):

		if pts[i - 1] is None or pts[i] is None: #jesli brak punktow to continue
			continue

		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 1.8) #oblicz grubosc linii
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness) #rysuj linie pomiedzy punktami

	# wyswietl zmodyfikowany strumien
	cv2.imshow("Frame", frame)
	cv2.imshow("Mask", mask) #DEBUG mask
	cv2.imshow("final", res) #DEBUG final
	
    # wcisniecie klawisza 'q' przerwie petle
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# wyczysc smieci
if not args.get("video", False):
	vs.stop()
else:
	vs.release()

cv2.destroyAllWindows()