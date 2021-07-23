# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex5 import *
print("Setup Complete")

# load the training and validation sets in X_train, X_valid, y_train, and y_valid. The test set is loaded in X_test.
# For simplicity, we drop categorical variables.
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

# print the first several rows of the data.
X.head()

"""
	MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	BsmtFinSF2	...	GarageArea	WoodDeckSF	OpenPorchSF	EnclosedPorch	3SsnPorch	ScreenPorch	PoolArea	MiscVal	MoSold	YrSold
Id																					
1	60	65.0	8450	7	5	2003	2003	196.0	706	0	...	548	0	61	0	0	0	0	0	2	2008
2	20	80.0	9600	6	8	1976	1976	0.0	978	0	...	460	298	0	0	0	0	0	0	5	2007
3	60	68.0	11250	7	5	2001	2002	162.0	486	0	...	608	0	42	0	0	0	0	0	9	2008
4	70	60.0	9550	7	5	1915	1970	0.0	216	0	...	642	0	35	272	0	0	0	0	2	2006
5	60	84.0	14260	8	5	2000	2000	350.0	655	0	...	836	192	84	0	0	0	0	0	12	2008
"""

# use SimpleImputer() to replace missing values in the data, 
# before using RandomForestRegressor() to train a random forest model to make predictions. 
# We set the number of trees in the random forest model with the n_estimators parameter, 
# and setting random_state ensures reproducibility.
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# use pipelines in cross-validation. 
# The code below uses the cross_val_score() function to obtain the mean absolute error (MAE), 
# averaged across five different folds. Recall we set the number of folds with the cv parameter.
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

# Average MAE score: 18276.410356164386

# Step 1: Write a useful function
# use cross-validation to select parameters for a machine learning model.
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=3,
                              scoring='neg_mean_absolute_error')
    return(scores.mean())
    pass

# Check your answer
step_1.check()

# Step 2: Test different parameter values
# use the function that you defined in Step 1 to evaluate the model performance corresponding to eight 
# different values for the number of trees in the random forest: 50, 100, 150, ..., 300, 350, 400.
# Store your results in a Python dictionary results, where results[i] is the average MAE returned by get_score(i).
results = {}
number_of_trees = {50: 50, 100: 100, 150: 150, 200: 200, 250: 250, 300: 300, 350: 350, 400: 400}
for tree in number_of_trees: 
    results[tree] = get_score(int(tree))# Your code here
    #print(results[tree])

# Check your answer
step_2.check()

# visualize your results from Step 2
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()

# cool plot of MAE to number of trees!

# Step 3: Find the best parameter value
# Given the results, which value for n_estimators seems best for the random forest model?
n_estimators_best = 200

# Check your answer
step_3.check()



