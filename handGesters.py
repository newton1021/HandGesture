#!/usr/bin/env python3

import cv2
import os, sys
import csv
import math
import numpy as np
import mediapipe as mp
from tensorflow import keras
import argparse


#Globas
writer = None
csv_file = None
model = None
label_encoder = None


keymap = {
	ord('r'): "rock",
	ord('p'): "paper",
	ord('s'): "scissors",
	ord('n'): "none"
}

# -------- init Fuctions ----------
def initWriter(filename="gestures.csv", reset = False):
	global writer, csv_file
	print("starting writer")
	mode = "w" if reset else "a"
	
	#open the file 
	csv_file = open(filename, mode, newline="")
	writer = csv.writer(csv_file)
	
	
	if reset or (os.path.exists(filename) and os.stat(filename).st_size==0):
		print("****** reseting *******")
		rowHeader = []
		for lm in range(21):
			rowHeader.extend([f"x{lm}", f"y{lm}", f"z{lm}"])
		rowHeader.append("label")
		print(rowHeader)
		writer.writerow(rowHeader)

def initModel(model_path="gesture_model.h5", label_path="gesture_labels.npy"):
	#Load model
	global model, label_encoder
	
	model = keras.models.load_model(model_path)
	label_encoder = np.load(label_path, allow_pickle=True)
	

def flip_hand_landmarks(landmarks):
	
		return [[1 - lm[0], lm[1], lm[2]] for lm in landmarks] #mirror x
	
	
def normalize(landmarks):
		
	wrist = landmarks[0] # wrist location for the zero point
	mcp = landmarks[9] #middle finger base used for scaling
	
	# calculate the scale 
	scale = math.sqrt(
		(mcp[0] - wrist[0])**2 + 
		(mcp[1] - wrist[1])**2 + 
		(mcp[2] - wrist[2])**2)
	if scale == 0:
		scale = 1
	if scale < 1e-6:
		print(f"too small of scale {scale}")
		return None
	
	# step through each of the 21 points and center and scale 
	norm_landmarks = []
	for lm in landmarks:
		norm_landmarks.extend([
			(lm[0] - wrist[0]) / scale, 
			(lm[1] - wrist[1]) / scale,
			(lm[2] - wrist[2]) / scale 
		])
		
	# return the set of normalized flatened landmarks
	return norm_landmarks
#-----------------------------
# predict returns label and confidence
#-----------------------------

def predict_gesture(landmarks):
	# landmarks must be normalized
	global model, label_encoder
	
	
	landmark_list = normalize(landmarks)
	if landmark_list is None:
		return "", 0
	
	X = np.array([landmark_list]) # (1,63)
	pred = model.predict(X, verbose = 0)
	idx = np.argmax(pred)
	return label_encoder[idx], pred[0][idx]


def saveLandmarkData(landmarks, label):
	global writer
	
	landmark_list = normalize(landmarks)
	print(landmark_list)
	if landmark_list is None:
		return
	landmark_list.append(label)
	writer.writerow(landmark_list)

def preprocess(frame):
	mp_hands = mp.solutions.hands
	mp_draw = mp.solutions.drawing_utils
	
	with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = hands.process(rgb)
		
		last_hand = None
		if result.multi_hand_landmarks:
			for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
				mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
				
				handedness = result.multi_handedness[idx].classification[0].label
				cv2.putText(frame, f"Hand: {handedness}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
				
				
				raw_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
				
				
				if handedness == "Left":
					raw_landmarks = flip_hand_landmarks(raw_landmarks)
				last_hand = raw_landmarks
		return last_hand

def analyseFrame(frame):
	if model is None: 
		initModel()
	landmarks = preprocess(frame)
	if landmarks is None:
		return None, 0 # no gesture detected
	result = predict_gesture(landmarks)
	return result

# ******************************************
# ******************************************
	
def captureImage(mode="collect"):
	print("=========== collecting data =========")
	"""
		mode = "collect" -> record data to csv file
		mode = "predict" -> run trained model prediction
	"""
	
	# Suppress AVFoundation warnings
	sys.stderr = open(os.devnull, "w")
		
	cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
	
	
		
	last_key = None
		
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame = cv2.flip(frame,1)
							
			last_hand = preprocess(frame)					
			
			key = cv2.waitKey(1) & 0xFF
							
			# Show mode and file name on top of the window
			cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
			
				
			if (key != 255 and key != last_key)  or mode == "predict":
				
				if key == 27: #esc:
					print("Quiting")
					break
				if not last_hand is None:				
					if mode == "collect":
						if key in keymap and last_hand:
							print(keymap[key])
							saveLandmarkData(last_hand, keymap[key])
						
						else:
							#unknown
							print("unknown: (r)ock, (p)aper or (s)cissors")
					elif mode == "predict":
						prediction,confidence = predict_gesture(last_hand)
						
						cv2.putText(frame, f"Prediction: {prediction} ({confidence:.2f})",(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
			last_key = key
			cv2.imshow("Hand Tracking", frame)
	finally:
		cap.release()
		cv2.destroyAllWindows()
		if csv_file:
			csv_file.close()
	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description="Hand gesture data collection / predictor")
	parser.add_argument(
		"mode", 
		choices=["collect", "predict"],
		nargs="?",
		default="collect",
		help="Mode to run: collect(default) or predict"
	)
	
	parser.add_argument(
		"-r", "--reset", 
		action="store_true",
		help="reset gesture.csv before collecting"
	)
	
	parser.add_argument("-f", "--file",
		default="gestures.csv",
		help="Name of gesture file (default: gestures.csv)"
	)
	args = parser.parse_args()
	
	print(f"mode: {args.mode}, reset: {args.reset}, file: {args.file}")
	if args.mode == "collect":
		initWriter(args.file, reset = args.reset)
		captureImage("collect")
	
	elif args.mode == "predict":
		initModel()
		captureImage("predict")
	else:
		print(f"{args.mode} is Invalid mode. Use 'collect' or 'predict'.")
		
		
	print("----Done----")