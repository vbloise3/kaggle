import numpy as np
import tensorflow as tf
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
#from keras import optimizers
from tensorflow.keras import optimizers

# spread predictions

drive.mount('/content/drive')
#!ls "/content/drive/My Drive/NFL"
!cp "/content/drive/My Drive/NFL/week_4_over_under_combined.csv" "week_4_over_under_combined.csv"
!cp "/content/drive/My Drive/NFL/week_5_games.csv" "week_5_games.csv"
NFLgames = pd.read_csv("/content/drive/My Drive/NFL/week_4_over_under_combined.csv")
gamesPredict = pd.read_csv("/content/drive/My Drive/NFL/week_5_games.csv")

X = NFLgames.copy()
X.drop(['Home_Team', 'Visitor_Team', 'over_under'], axis=1, inplace=True)
y = NFLgames.pop('over_under')
gamesPredict.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)

print(X.head())
print(X.shape)

print(y.head())
print(y.shape)

#
# Set up the network
#
network = models.Sequential()
network.add(layers.Dense(24, activation='relu', input_shape=(96,)))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(1))
#
# Configure the network with optimizer, loss function and accuracy
#
network.compile(optimizer=optimizers.RMSprop(learning_rate=0.01),
                loss='mse',
                metrics=['mae'])
#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# Fit the network
#
history = network.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=18,
                    batch_size=20)

import matplotlib.pyplot as plt
 
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['mae']
val_accuracy = history_dict['val_mae']
 
epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
# Plot the model accuracy (MAE) vs Epochs
#
ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()
#
# Plot the loss vs Epochs
#
ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()

# make predictions
network.predict(gamesPredict)