#!/usr/bin/env python3
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
	from absl import logging as absl_logging
	absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
	pass
	
	
	
import cv2
import numpy as np
from collections import deque
import handGesters as hg
import itertools
import random
import json
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense



seq_len = 5
num_features = 6 # 3 for throws + 3 for outcomes
num_classes = 3 #rock/paper/scissors

throw_map = {"rock": 0, "paper": 1, "scissors": 2}
throw_names = ["rock", "paper", "scissors"]
outcome_map = {"win":0, "lose":1, "draw":2}
history = []
next_throw_model = None
history_file = "history.json"
model_file = "rps_model.h5"

throw_buffer = [] # your throws: rock paper scissors  deque(maxlen=seq_len)
outcome_buffer = [] #outcomes: "win lose draw"  deque(maxlen=seq_len)




# Generator for random throws
def throw_stream():
	while True:
		yield random.choice(throw_names)
random_throw = throw_stream()


# --- Load history safely ---
def safe_load_history():
	if not os.path.exists(history_file) or os.stat(history_file).st_size == 0:
		return []

	try:
		with open(history_file, "r") as f:
			history = json.load(f)
	except json.JSONDecodeError:
		print("⚠️ Warning: history file corrupted, starting fresh.")
		history = []
	return history
	
	

	
def one_hot_throw(throw):
	vec = np.zeros(3, dtype=np.float32)
	vec[throw_map[throw]] = 1.0
	return vec

def one_hot_outcome(outcome):
	# outcome = win lose, draw
	vec = np.zeros(3, dtype=np.float32)
	
	vec[outcome_map[outcome]] = 1.0
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
		if np.array_equal(past_pattern, pattern):
			next_move.append(thow_buffer(i+N))
			
	if not next_move:
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
	"""
		Returns X (num_samples, seq_len, num_features),
				y (num_samples, num_classes)
		If not enough valid samples, returns empty arrays with correct ndim/dtype.
	"""
	X, y =  [],[]
	
	for i in range(len(history) - seq_len):
		ok = True
		seq = []
		for j in range(seq_len):
			rec = history[i+j]
			throw = rec.get("player","").lower()
			outcome = rec.get("outcome","").lower()
			if throw not in throw_map or outcome not in outcome_map:	
				ok = False; break
				
			seq.append(np.concatenate([
					one_hot_throw(throw),
					one_hot_outcome(outcome)
				]))
		if not ok:
			continue
		next_rec = history[i+seq_len]
		next_throw = next_rec.get("player", "").lower()
		if next_throw not in throw_map:
			continue
		
		
		X.append(seq)
		y.append(one_hot_throw(next_throw))
			
	if len(X) == 0:
		return np.empty((0, seq_len, num_features), dtype = np.float32), np.empty((0,num_classes), dtype = np.float32)
	X = np.asarray(X, dtype=np.float32)  # shape (N, seq_len, num_features)
	y = np.asarray(y, dtype=np.float32) # shape (N, num_classes)
	
	return X, y
	
def build_model(history, min_samples = 20, retrain_epochs=5):
	"""
	Loads existing model if available; otherwise builds a fresh one.
	Trains only if we have >= min_samples training sequences.
	Returns True if trained/saved, False if skipped (not enough data).
	"""
	
	if history is None:
		print("There is no history file yet so no model was built")
		return False, None
	X, y = prepare_training_data(history, seq_len = 5)
	
	if os.path.exists(model_file):
		try:
			next_throw_model = load_model(model_file)
			print("Loaded existing mext-throw model.")
		except Exception as e:
			print(f"⚠️ Could not load model: {e}. Rebuilding fresh model.")
			next_throw_model = None
	else:
		next_throw_model = None
		
	if next_throw_model is None:
		next_throw_model = Sequential([Input(shape=(seq_len, num_features)),
			LSTM(32),
			Dense(num_classes, activation="softmax")
			])
		next_throw_model.compile(
				optimizer="adam",
				loss="categorical_crossentropy",
				metrics=["accuracy"])
	# load history and prepare data
	history = safe_load_history()
	X, y = prepare_training_data(history, seq_len = seq_len)
	print(f"Prepared training data: X.shape={X.shape}, y.shape={y.shape}")
		
	if X.shape[0] < min_samples: 
		print(f"Not enough training samples ({X.shape[0]}).  Need >= {min_samples} Skip training.")
		# still return the model object for predictions , but not trained
		return False, next_throw_model
	
	# Train the LSTM
	batch = min(16, max(1, X.shape[0]))
	
	next_throw_model.fit(X, y, epochs=retrain_epochs, batch_size=batch, verbose=1)
	next_throw_model.save(model_file)
	print("Trained and saved model to", model_file)
	return True, next_throw_model


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
	
	# load models and setup data strucure.
	history = safe_load_history()
	trained, next_throw_model = build_model(history, min_samples=20, retrain_epochs=5)
	
	

	
	
	
	cap = cv2.VideoCapture(0)
	
	
	round_active = False 
	gesture = None
	current_throw = "none"
	predicted_throw = None
	ai_throw = "Waiting..."
	last_outcome = "none"
	hold_counter = 100
	
	while True: 
		ret, frame = cap.read() #capture image
		if not ret:
			break
		
		#check for key stroke for escape
		#if no throw report Ready? 
		
		# step 1 Detect
		
		gesture, confidence = hg.analyseFrame(frame) #analize the frame looking for hands when one is found send back the landmarks? why not return throw
			
				
		
		if gesture and gesture in throw_map:
			if not round_active:
				print(f"History length = {len(history)}")
				current_throw = gesture
				throw_buffer.append(one_hot_throw(current_throw))
				
				predicted_throw = predict_next_throw()
				
				if predicted_throw: 
					ai_throw = counter_throw(predicted_throw)
				else: 
					ai_throw = "Error..."
					
				# determin outcome
				last_outcome = get_outcome(current_throw, ai_throw)
				outcome_buffer.append(last_outcome)
				
				round_data = {"player": current_throw, "ai": ai_throw, "outcome": last_outcome}
				
				history.append(round_data)
				
				with open(history_file, "w") as f:
					json.dump(history,f)
					
				if len(history) % 20 == 0: 
					trained, next_throw_model = build_model(history, min_samples=20, retrain_epochs=5)
					ai_throw = "Calculating"
				
				round_active = True
			hold_counter = 30
		
		else: 
			hold_counter = max(hold_counter - 1, 0)
			if hold_counter < 1:
				current_throw = "none"
				round_active = False
				ai_throw = "waiting..."
			
			
			
				
		# Step 3: display
		cv2.putText(frame, f"You: {current_throw}", (100,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		cv2.putText(frame, f"AI predicts: {predicted_throw}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
		cv2.putText(frame, f"AI throws: {ai_throw}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		cv2.putText(frame, f"Outcome: {last_outcome}", (10,160),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)
		
		
		cv2.imshow("RPS AI", frame)
		
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()
	
	
if __name__ == "__main__":
	play_game()