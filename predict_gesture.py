#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


#Load model
model = tf.keras.models.load_model("gesture_model.h5")
labels = np.load("gesture_label.npy")

def predict_gesture(landmarks):
	# landmarks must be normalized
	X = np.array([landmarks]) # (1,63)
	pred = model.predict(X, verbose = 0)
	idx = np.argmax(pred)
	return labels[idx], pred[0][idx]
