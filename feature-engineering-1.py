# Example - Concrete Formulations
# see how adding a few synthetic features to a dataset can improve the predictive performance 
# of a random forest model.
# The task for this dataset is to predict a concrete's compressive strength given its formulation

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../input/fe-course-data/concrete.csv")
df.head()
"""
	Cement	BlastFurnaceSlag	FlyAsh	Water	Superplasticizer	CoarseAggregate	FineAggregate	Age	CompressiveStrength
0	540.0	0.0	0.0	162.0	2.5	1040.0	676.0	28	79.99
1	540.0	0.0	0.0	162.0	2.5	1055.0	676.0	28	61.89
2	332.5	142.5	0.0	228.0	0.0	932.0	594.0	270	40.27
3	332.5	142.5	0.0	228.0	0.0	932.0	594.0	365	41.05
4	198.6	132.4	0.0	192.0	0.0	978.4	825.5	360	44.30
"""

# We'll see in a moment how adding some additional synthetic features derived from these can help a model 
# to learn important relationships among them.
# We'll first establish a baseline by training the model on the un-augmented dataset.
# This will help us determine whether our new features are actually useful.
# Baseline:
X = df.copy()
y = X.pop("CompressiveStrength")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="mae", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")
# MAE Baseline Score: 8.232

# If you ever cook at home, you might know that the ratio of ingredients in a recipe is usually a 
# better predictor of how the recipe turns out than their absolute amounts. 
# We might reason then that ratios of the features above would be a good predictor of CompressiveStrength.
# The cell below adds three new ratio features to the dataset.
X = df.copy()
y = X.pop("CompressiveStrength")

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features
model = RandomForestRegressor(criterion="mae", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")
# MAE Score with Ratio Features: 7.948

