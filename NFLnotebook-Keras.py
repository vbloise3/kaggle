import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

drive.mount('/content/drive')
!ls "/content/drive/My Drive/NFL"
!cp "/content/drive/My Drive/NFL/week_3_spread.csv" "week_3_over_under.csv"
!cp "/content/drive/My Drive/NFL/week_4_games.csv" "week_4_games.csv"
X_full = pd.read_csv("/content/drive/My Drive/NFL/week_3_over_under.csv")
#dataTest = pd.read_csv("/content/drive/My Drive/NFL/week_4_games.csv")

dataTrain = X_full.sample(frac=0.8, random_state=0)
dataTest = X_full.drop(train_dataset.index)

# print(dataTrain.head())
# print(dataTrain.shape)

dataTrain.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)
dataTest.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)

print(dataTest.head())
print(dataTest.shape)

batch_size = 64
input_dim = 97

units = 64
output_size = 1  # labels are from Win or Loss

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(input_dim,1))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(input_dim,1)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model


x_train, y_train = dataTrain.iloc[:,:97].values,dataTrain.iloc[:,97:].values
x_train=np.reshape(x_train,(13,97,1))
x_test, y_test = dataTest.iloc[:,:97].values,dataTest.iloc[:,97:].values
x_test=np.reshape(x_test,(3,97,1))

model = build_model(allow_cudnn_kernel=True)

model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer="Adam",
    metrics=["categorical_accuracy"],
)

model.summary()

model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10
)