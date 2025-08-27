#!/usr/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import handGesters as hg
import os
import itertools
import random
import json
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense


seq_len = 5
num_features = 6 # 3 for throws + 3 for outcomes
num_classes = 3 #rock/paper/scissors
throw_map = {"rock": 0, "paper": 1, "scissors": 2}
throw_names = ["rock", "paper", "scissors"]

throw_buffer = [] # your throws: rock paper scissors  deque(maxlen=seq_len)
outcome_buffer = [] #outcomes: "win lose draw"  deque(maxlen=seq_len)
history_file = "history.json"
model_file = "rps_model.h5"

# Generator for random throws
def throw_stream():
	while True:
		yield random.choice(throw_names)
random_throw = throw_stream()


# --- Load history safely ---
if os.path.exists(history_file) and os.stat(history_file).st_size > 0:
	try:
		with open(history_file, "r") as f:
			history = json.load(f)
	except json.JSONDecodeError:
		print("⚠️ Warning: history file corrupted, starting fresh.")
		history = []
else:
	history = []
	
# --- Load or build model ---
if os.path.exists(model_file):
	try:
		next_throw_model = load_model(model_file)
		print("✅ Loaded existing model")
	except Exception as e:
		print(f"⚠️ Could not load model: {e}. Rebuilding fresh model.")
		next_throw_model = Sequential([
			LSTM(32, input_shape=(seq_len, num_features)),
			Dense(num_classes, activation="softmax")
		])
		next_throw_model.compile(
			optimizer="adam",
			loss="categorical_crossentropy",
			metrics=["accuracy"]
		)
else:
	print("ℹ️ No model found, building fresh one.")
	next_throw_model = Sequential([
		LSTM(32, input_shape=(seq_len, num_features)),
		Dense(num_classes, activation="softmax")
	])
	next_throw_model.compile(
		optimizer="adam",
		loss="categorical_crossentropy",
		metrics=["accuracy"]
	)
	
	
def one_hot_throw(throw):
	vec = np.zeros(3)
	vec[throw_map[throw]] = 1
	return vec

def one_hot_outcome(outcome):
	# outcome = win lose, draw
	mapping = {"win": 0, "lose":1, "draw": 2}
	vec = np.zeros(3)
	vec[mapping[outcome]] = 1
	return vec

def get_lstm_input():
	if len(throw_buffer) < seq_len:
		return None
	X_seq = np.hstack([throw_buffer, outcome_buffer])
	return np.expand_dims(X_seq, axis=0)


# +====================================

def predict_next_throw():
	if len(throw_buffer) < seq_len:
		return next(random_throw)
	N = 5
	recent_throws = throw_buffer[-N:]
	recent_outcomes = outcome_buffer[-N:]
	
	pattern = tuple(zip(recent_throws, recent_outcomes))
	
	
	next_move = []
	for i in range(len(throw_buffer)-N):
		past_pattern = tuple(zip(throw_buffer[i:i+N], outcome_buffer[i:i+N]))
		if past_pattern == pattern: 
			next_moves.append(thow_buffer(i+N))
			
	if not next_moves:
		return next(random_throw)
	
	# MARK: predict next move
	X_seq=get_lstm_input()
	if X_seq is None:
		return  next(random_throw)
	pred = next_throw_model.predict(X_seq, verbose=0)
	predicted_throw_idx = np.argmax(pred[0])
	return throw_names[predicted_throw_idx]

def counter_throw(predicted_throw):
	if predicted_throw is None:
		return  next(random_throw)	
	
	return throw_names[(throw_map[predicted_throw] + 1) % 3]


def prepare_training_data(history, seq_len=5):
	X, y =  [],[]
	
	for i in range(len(history) - seq_len):
		seq = []
		for j in range(seq_len):
			throw = history[i + j]["player"]
			outcome = history[i+j]["outcome"]
			seq.append(np.concatenate([
				one_hot_throw(throw_names[throw_map[throw]]),
				one_hot_outcome(outcome)
			]))
		X.append(seq)
		y.append(one_hot_throw(history[i+seq_len]["player"]))
	return np.array(X), np.array(y)

	
def build_model():
	X, y = prepare_training_data(history, seq_len = 5)
	
	if len(history) < seq_len:
		return
	# Train the LSTM
	next_throw_model.fit(X, y, epochs=5, batch_size=16, verbose=0)
	next_throw_model.save("next_throw_model.h5")

build_model()

# ++++++++++++++++++++++++++++++++
# get_outcome who won
# ++++++++++++++++++++++++++++++++

def get_outcome(player, ai):
	if player == ai:
		return "draw"
	if (player == "rock" and ai == "scissors") or \
		(player == "scissors" and ai == "paper") or \
		(player == "paper" and ai == "rock"):
		return "win"
	return "lose"
	


# ++++++++++++++++++++++++++++++++
# play_game is the main loop
# ++++++++++++++++++++++++++++++++

def play_game():
	readyForRound = False 
	
	
	cap = cv2.VideoCapture(0)
	gesture = None
	current_throw = "none"
	predicted_throw = None
	ai_throw = "thinking..."
	last_outcome = "none"
	
	while True: 
		ret, frame = cap.read() #capture image
		if not ret:
			break
		
		#check for key stroke for escape
		#if no throw report Ready? 
		
		# step 1 Detect
		
		gesture, confidence = hg.analyseFrame(frame) #analize the frame looking for hands when one is found send back the landmarks? why not return throw
			
				
		
#		if gesture and gesture in throw_map:
#			current_throw = gesture
#			throw_buffer.append(one_hot_throw(current_throw))
#			outcome = get_outcome(current_throw, ai_throw)
#			outcome_buffer.append(outcome)
#		else: 
#			current_throw = "none"
#			
#		
#			
#		# step 2 predict next throw
#		predicted_throw = predict_next_throw()
#		if predicted_throw: 
#			ai_throw = counter_throw(predicted_throw)
#		else: 
#			ai_throw = "thinking..."
#			
#			
#		if current_throw != "none" and ai_throw != "thinking...":
#			last_outcome = get_outcome(current_throw, ai_throw)
#			
#			throw_buffer.append(one_hot_throw(current_throw))
#			outcome_buffer.append(one_hot_outcome(last_outcome))
#			
#		round_data = {"player": current_throw, "ai": ai_throw, "outcome": last_outcome}
#		
#		history.append(round_data)
#		
#		with open(history_file, "w") as f:
#			json.dump(history,f)
#		
#		if len(history) % 20 == 0: 
#			build_model()
		
		# Step 3: display
		cv2.putText(frame, f"You: {current_throw}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		cv2.putText(frame, f"AI predicts: {predicted_throw}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
		cv2.putText(frame, f"AI throws: {ai_throw}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		cv2.putText(frame, f"Outcome: {last_outcome}", (10,160),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
		
		
		cv2.imshow("RPS AI", frame)
		
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()
	
	
if __name__ == "__main__":
	play_game()