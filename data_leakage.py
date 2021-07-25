# Data leakage (or leakage) happens when your training data contains information about the target, 
# but similar data will not be available when the model is used for prediction. 
# This leads to high performance on the training set (and possibly even the validation data), 
# but the model will perform poorly in production.
# There are two main types of leakage: target leakage and train-test contamination
# Target leakage
# Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
# It is important to think about target leakage in terms of the timing or chronological order 
# that data becomes available, not merely whether a feature helps make good predictions.
# To prevent this type of data leakage, any variable updated (or created) 
# after the target value is realized should be excluded.
# Train-Test Contamination
# If the validation data affects the preprocessing behavior, this is sometimes called train-test contamination.
# For example, imagine you run preprocessing (like fitting an imputer for missing values) 
# before calling train_test_split(). The end result? Your model may get good validation scores, 
# giving you great confidence in it, but perform poorly when you deploy it to make decisions.

# Example
# detect and remove target leakage
import pandas as pd

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()
# Number of rows in the dataset: 1319
"""
	reports	age	income	share	expenditure	owner	selfemp	dependents	months	majorcards	active
0	0	37.66667	4.5200	0.033270	124.983300	True	False	3	54	1	12
1	0	33.25000	2.4200	0.005217	9.854167	False	False	3	34	1	13
2	0	33.66667	4.5000	0.004156	15.000000	True	False	4	58	1	5
3	0	30.50000	2.5400	0.065214	137.869200	False	False	0	25	1	7
4	0	32.16667	9.7867	0.067051	546.503300	True	False	2	64	1	5
"""

# Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())
# Cross-validation accuracy: 0.980294

# At this point, basic data comparisons can be very helpful:
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))

# Fraction of those who did not receive a card and had no expenditures: 1.00
# Fraction of those who received a card and had no expenditures: 0.02

# Run a model without target leakage as follows:
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
# Cross-val accuracy: 0.836989

