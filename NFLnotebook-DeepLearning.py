from google.colab import drive
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

drive.mount('/content/drive')
!ls "/content/drive/My Drive/NFL"
!cp "/content/drive/My Drive/NFL/week_3_spread.csv" "week_3_over_under.csv"
X = pd.read_csv("/content/drive/My Drive/NFL/week_3_over_under.csv")
X_test_full = pd.read_csv("/content/drive/My Drive/NFL/week_4_games.csv")
X_full = X.copy()

y = X.over_under              
X.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)
X_test_full.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)
X_full.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)
train_dataset = X_full.sample(frac=0.8, random_state=0)
test_dataset = X_full.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('over_under')
test_labels = test_features.pop('over_under')
# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

PF = np.array(train_features['PF'])
PF_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
PF_normalizer.adapt(PF)

PF_model = tf.keras.Sequential([
    PF_normalizer,
    layers.Dense(units=1)
])

PF_model.summary()

# Run the model
PF_model.predict(PF[:10])

PF_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = PF_model.fit(
    train_features['PF'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [over_under]')
  plt.legend()
  plt.grid(True)

plot_loss(history)

test_results = {}

test_results['PF_model'] = PF_model.evaluate(
    test_features['PF'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = PF_model.predict(x)

def plot_PF(x, y):
  plt.scatter(train_features['PF'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('PF')
  plt.ylabel('over_under')
  plt.legend()

plot_PF(x,y)


linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_PF_model = build_and_compile_model(PF_normalizer)

dnn_PF_model.summary()

history = dnn_PF_model.fit(
    train_features['PF'], train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_PF_model.predict(x)

plot_PF(x, y)

test_results['dnn_PF_model'] = dnn_PF_model.evaluate(
    test_features['PF'], test_labels,
    verbose=0)


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [over_under]']).T

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [over_under]')
plt.ylabel('Predictions [over_under]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [over_under]')
_ = plt.ylabel('Count')

dnn_model.save('dnn_model')



'''input_shape = [X_train_full.shape[1]]
print("Input shape: {}".format(input_shape))
# Define the model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),    
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='sgd', # SGD is more sensitive to differences of scale
    loss='mse',
    metrics=['mse'],
)
history = model.fit(
    X_train_full, y_train,
    validation_data=(X_valid_full, y_valid),
    batch_size=64,
    epochs=100,
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_df['val_loss'].min()))
# Get predictions
predictions_2 = model.predict(X_valid_full)
# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:" , mae_2)

preds = model.predict(X_test_full)
best_preds = np.asarray([np.argmax(line) for line in preds])
preds'''