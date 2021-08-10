# Exercise: Overfitting and Underfitting

# Introduction
# In this exercise, you’ll learn how to improve training outcomes by including an early stopping callback 
# to prevent overfitting.

# When you're ready, run this next cell to set everything up!
# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex4 import *

# First load the Spotify dataset. Your task will be to predict the popularity of a song  
# based on various audio features, like 'tempo', 'danceability', and 'mode'.
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

spotify = pd.read_csv('../input/dl-course-data/spotify.csv')

X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))
# Input shape: [18]

# Let's start with the simplest network, a linear model. This model has low capacity.

# Run this next cell without any changes to train a linear model on the Spotify dataset.
model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # suppress output since we'll plot the curves
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
# Minimum Validation Loss: 0.1956

# It's not uncommon for the curves to follow a "hockey stick" pattern like you see here. 
# This makes the final part of training hard to see, so let's start at epoch 10 instead:
# Start the plot at epoch 10
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
# Minimum Validation Loss: 0.1956

# 1) Evaluate Baseline
# What do you think? Would you say this model is underfitting, overfitting, just right?
# The gap between these curves is quite small and the validation loss never increases, 
# so it's more likely that the network is underfitting than overfitting. 
# It would be worth experimenting with more capacity to see if that's the case.

# Now let's add some capacity to our network. We'll add three hidden layers with 128 units each. 
# Run the next cell to train the network and see the learning curves.
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
# Minimum Validation Loss: 0.1965

# 2) Add Capacity
# What is your evaluation of these curves? Underfitting, overfitting, just right?
# Now the validation loss begins to rise very early, while the training loss continues to decrease. 
# This indicates that the network has begun to overfit. At this point, we would need to try something to prevent it, 
# either by reducing the number of units or through a method like early stopping. 
# (We'll see another in the next lesson!)

# 3) Define Early Stopping Callback
# Now define an early stopping callback that waits 5 epochs ('patience') 
# for a change in validation loss of at least 0.001 (min_delta) and keeps the weights with the best loss 
# (restore_best_weights).
from tensorflow.keras import callbacks

# YOUR CODE HERE: define an early stopping callback
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# Check your answer
q_3.check()

# Now run this cell to train the model and get the learning curves. Notice the callbacks argument in model.fit.
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),    
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
# Minimum Validation Loss: 0.1964
# See chart at https://www.kaggle.com/vbloise/exercise-overfitting-and-underfitting/edit

# 4) Train and Interpret
# Was this an improvement compared to training without early stopping?
# The early stopping callback did stop the training once the network began overfitting. 
# Moreover, by including restore_best_weights we still get to keep the model where validation loss was lowest.