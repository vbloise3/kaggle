from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

drive.mount('/content/drive')
!ls "/content/drive/My Drive/NFL"
!cp "/content/drive/My Drive/NFL/week_3_spread.csv" "week_3_over_under.csv"
X = pd.read_csv("/content/drive/My Drive/NFL/week_3_over_under.csv")
X_test_full = pd.read_csv("/content/drive/My Drive/NFL/week_4_games.csv")

y = X.over_under              
X.drop(['over_under', 'Home_Team', 'Visitor_Team'], axis=1, inplace=True)
X_test_full.drop(['Home_Team', 'Visitor_Team'], axis=1, inplace=True)
# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Define the model
my_model_2 = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=10,
    nthread=4)
# Fit the model
my_model_2.fit(X_train_full, y_train, 
             early_stopping_rounds=1, 
             eval_set=[(X_valid_full, y_valid)], 
             verbose=False)
# Get predictions
predictions_2 = my_model_2.predict(X_valid_full)
# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:" , mae_2)

preds = my_model_2.predict(X_test_full)
best_preds = np.asarray([np.argmax(line) for line in preds])
preds