#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("gestures.csv")

# Separate data from label
X = data.drop("label", axis=1).values
y = data['label'].values

# Encode labels ("rock", "paper", "scissors", "none"
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# split data
X_train, X_val, y_train, y_val = train_test_split( X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

#----------------
# build model
#----------------

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(63,)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

#-----------------
# Train
#-----------------
history = model.fit(
	X_train, y_train,
	epochs=20,
	batch_size=16,
	verbose=1
)

#-----------------
# save model + encoder
#-----------------
model.save("gesture_model.h5")

np.save("gesture_labels.npy", encoder.classes_)

print("Model Trained and saved!")