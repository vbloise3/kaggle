# Linear Units in Keras
# The easiest way to create a model in Keras is through keras.
# Sequential, which creates a neural network as a stack of layers. 
# We can create models like those above using a dense layer (which we'll learn more about in the next lesson).

# We could define a linear model accepting three input features ('sugars', 'fiber', and 'protein') a
# nd producing a single output ('calories') like so:
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
# With the first argument, units, we define how many outputs we want. 
# In this case we are just predicting 'calories', so we'll use units=1.

# With the second argument, input_shape, we tell Keras the dimensions of the inputs. S
# etting input_shape=[3] ensures the model will accept three features as input ('sugars', 'fiber', and 'protein').

# This model is now ready to be fit to training data!
# Why is input_shape a Python list?
# The data we'll use in this course will be tabular data, like in a Pandas dataframe. 
# We'll have one input for each feature in the dataset. 
# The features are arranged by column, so we'll always have input_shape=[num_columns]. 
# The reason Keras uses a list here is to permit use of more complex datasets. 
# Image data, for instance, might need three dimensions: [height, width, channels].
