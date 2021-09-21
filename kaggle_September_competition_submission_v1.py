import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Embedding,  Activation, Flatten, Conv1D
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from tensorflow.keras import regularizers

from sklearn.preprocessing import QuantileTransformer,  KBinsDiscretizer
from tensorflow import keras
from sklearn import metrics
from sklearn.impute import SimpleImputer

import math
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

%%time
train = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
test  = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')
#sub   = pd.read_csv('../input/tabular-playground-series-sep-2021/sample_solution.csv')

validation = train.sample(frac = 0.3)
train = train.drop(validation.index)

print(train.shape)
print(train.claim.value_counts())
print(validation.shape)
print(validation.claim.value_counts())

train['missing'] = train.isna().sum(axis=1)
validation['missing'] = validation.isna().sum(axis=1)
test['missing'] = test.isna().sum(axis=1)

features = [col for col in train.columns if col not in ['claim', 'id']]

def modelPreprocessor():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', missing_values=np.nan)),
        ("scaler", QuantileTransformer(n_quantiles=256,output_distribution='uniform')),
        ('bin', KBinsDiscretizer(n_bins=256, encode='ordinal',strategy='uniform'))
        ])
    
    train[features] = pipe.fit_transform(train[features])
    test[features] = pipe.transform(test[features])
    validation[features] = pipe.transform(validation[features])

def model(shape):
    input = Input(shape)
    
    e = Embedding(input_dim=256, output_dim=4)(input)
    f2 = Flatten()(e)
    
    #d1 = Dense(128,  activation='relu', activity_regularizer=regularizers.l2(1e-5))(f2)
    #do1 = Dropout(0.2)(d1)
    
    d2 = Dense(64,  activation='relu', activity_regularizer=regularizers.l2(1e-5), bias_regularizer=regularizers.l2(1e-4))(f2)
    do2 = Dropout(0.5)(d2)
    
    d3 = Dense(32,  activation='relu', activity_regularizer=regularizers.l2(1e-5), bias_regularizer=regularizers.l2(1e-4))(do2)
    do3 = Dropout(0.5)(d3)
    
    output = Dense(1, activation='sigmoid')(do3)

    model = Model(inputs=input, outputs=output)

    auc = tf.keras.metrics.AUC(name='aucroc')
    optimizer = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=[auc])
    
    return model

modelPreprocessor()

x=train[features]
y=train['claim']

xval=train[features]
yval=train['claim']

x.head()

xval.head()

model = model(x.shape[1:])

history = model.fit(x = x, y = y, batch_size = 512, shuffle = True, validation_data=(xval, yval), epochs=20)

# plot training history
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['aucroc'], label='aucroc')
plt.plot(history.history['val_aucroc'], label='val_aucroc')
plt.legend()
plt.show()

sub=pd.DataFrame()
sub['id'] = test['id']
sub['claim'] = model.predict(test[features])
sub=sub.set_index('id')
sub.to_csv('submission.csv')

sub.head()