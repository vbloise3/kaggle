# Introduction
# In this exercise you'll start developing the features you identified in Exercise 2 as having the most potential. 
# As you work through this exercise, you might take a moment to look at the data documentation again and 
# consider whether the features we're creating make sense from a real-world perspective, 
# and whether there are any useful combinations that stand out to you.

# Run this cell to set everything up!
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex3 import *

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv("../input/fe-course-data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")

# Let's start with a few mathematical combinations. 
# We'll focus on features describing areas -- having the same units (square-feet) makes it easy 
# to combine them in sensible ways. 
# Since we're using XGBoost (a tree-based model), we'll focus on ratios and sums.

# 1) Create Mathematical Transforms
# Create the following features:
#   LivLotRatio: the ratio of GrLivArea to LotArea
#   Spaciousness: the sum of FirstFlrSF and SecondFlrSF divided by TotRmsAbvGrd
#   TotalOutsideSF: the sum of WoodDeckSF, OpenPorchSF, EnclosedPorch, Threeseasonporch, and ScreenPorch
# YOUR CODE HERE
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = df.GrLivArea / df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch


# Check your answer
q_1.check()

# If you've discovered an interaction effect between a numeric feature and a categorical feature, 
# you might want to model it explicitly using a one-hot encoding, like so:

# One-hot encode Categorical feature, adding a column prefix "Cat"
X_new = pd.get_dummies(df.Categorical, prefix="Cat")

# Multiply row-by-row
X_new = X_new.mul(df.Continuous, axis=0)

# Join the new features to the feature set
X = X.join(X_new)

# 2) Interaction with a Categorical
# We discovered an interaction between BldgType and GrLivArea in Exercise 2. Now create their interaction features
# YOUR CODE HERE
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg") 
# Multiply
X_2 = X_2.mul(df.GrLivArea, axis=0)


# Check your answer
q_2.check()

# 3) Count Feature
# Let's try creating a feature that describes how many kinds of outdoor areas a dwelling has. Create a feature PorchTypes that counts how many of the following are greater than 0.0:
#   WoodDeckSF
#   OpenPorchSF
#   EnclosedPorch
#   Threeseasonporch
#   ScreenPorch
X_3 = pd.DataFrame()

# YOUR CODE HERE
X_3["PorchTypes"] = df[[ "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]].gt(0).sum(axis=1)


# Check your answer
q_3.check()

# 4) Break Down a Categorical Feature
# MSSubClass describes the type of a dwelling:
df.MSSubClass.unique()
"""
array(['One_Story_1946_and_Newer_All_Styles', 'Two_Story_1946_and_Newer',
       'One_Story_PUD_1946_and_Newer',
       'One_and_Half_Story_Finished_All_Ages', 'Split_Foyer',
       'Two_Story_PUD_1946_and_Newer', 'Split_or_Multilevel',
       'One_Story_1945_and_Older', 'Duplex_All_Styles_and_Ages',
       'Two_Family_conversion_All_Styles_and_Ages',
       'One_and_Half_Story_Unfinished_All_Ages',
       'Two_Story_1945_and_Older', 'Two_and_Half_Story_All_Ages',
       'One_Story_with_Finished_Attic_All_Ages',
       'PUD_Multilevel_Split_Level_Foyer',
       'One_and_Half_Story_PUD_All_Ages'], dtype=object)
"""
# You can see that there is a more general categorization described (roughly) by the first word of each category. 
# Create a feature containing only these first words by splitting MSSubClass at the first underscore _. 
# (Hint: In the split method use an argument n=1.)
X_4 = pd.DataFrame()

X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]

# Check your answer
q_4.check()

# 5) Use a Grouped Transform
# The value of a home often depends on how it compares to typical homes in its neighborhood. 
# Create a feature MedNhbdArea that describes the median of GrLivArea grouped on Neighborhood.
X_5 = pd.DataFrame()

# YOUR CODE HERE
X_5["MedNhbdArea"] = (
    df.groupby("Neighborhood")  # for each Neighborhood
    ["GrLivArea"]                 # select the GrLivArea
    .transform("median")         # and compute its median
)

# Check your answer
q_5.check()

# Now you've made your first new feature set! If you like, you can run the cell below to score the model 
# with all of your new features added:
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)
# 0.13847331710099203