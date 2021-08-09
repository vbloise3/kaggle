# TPS-LDA-Catboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import model_selection
import lightgbm as lgbm
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import tqdm

# Exploratory Data Analysis
train=pd.read_csv("../input/tabular-playground-series-aug-2021/train.csv")
test=pd.read_csv("../input/tabular-playground-series-aug-2021/test.csv")

train.head()

train.isnull().sum()

y = train['loss']
train.drop(['id','loss'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)

# Feature Distribution
fig = plt.figure(figsize = (15, 60))
for i in range(len(train.columns.tolist()[:100])):
    plt.subplot(20,5,i+1)
    sns.set_style("white")
    plt.title(train.columns.tolist()[:100][i], size = 12, fontname = 'monospace')
    a = sns.kdeplot(train[train.columns.tolist()[:100][i]], color = '#34675c', shade = True, alpha = 0.9, linewidth = 1.5, edgecolor = 'black')
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks(fontname = 'monospace')
    plt.yticks([])
    for j in ['right', 'left', 'top']:
        a.spines[j].set_visible(False)
        a.spines['bottom'].set_linewidth(1.2)
        
fig.tight_layout(h_pad = 3)

plt.show()

# Scaling and LDA
not_features = ['id', 'loss']
features = []
for feat in train.columns:
    if feat not in not_features:
        features.append(feat)

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

x = train

lda = LDA(n_components=42, solver='svd')
X_lda = lda.fit_transform(x, y)

EVR = lda.explained_variance_ratio_
for idx, R in enumerate(EVR):
    print("Component {}: {}% var".format(idx+1, np.round(R*100,2)))

# Model Tuning and Training
def objective(trial,data=x,target=y):
    lda = LDA(n_components=42, solver='svd')
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25,random_state=42)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.fit_transform(X_test, y_test)
    params = {'iterations':trial.suggest_int("iterations", 1000, 20000),
              'od_wait':trial.suggest_int('od_wait', 500, 2000),
             'loss_function':'RMSE',
              'task_type':"GPU",
              'eval_metric':'RMSE',
              'leaf_estimation_method':'Newton',
              'bootstrap_type': 'Bernoulli',
              'learning_rate' : trial.suggest_uniform('learning_rate',0.02,1),
              'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
              'subsample': trial.suggest_uniform('subsample',0,1),
              'random_strength': trial.suggest_uniform('random_strength',10,50),
              'depth': trial.suggest_int('depth',1,15),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
              'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
               }
    model = CatBoostRegressor(**params)  
    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
        
    y_preds = model.predict(X_test)
    loss = np.sqrt(mean_squared_error(y_test, y_preds))
    
    return loss

# use optima
OPTUNA_OPTIMIZATION = True

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

if OPTUNA_OPTIMIZATION:
    display(optuna.visualization.plot_optimization_history(study))
    display(optuna.visualization.plot_slice(study))
    display(optuna.visualization.plot_parallel_coordinate(study))

# Training
cat_params = study.best_trial.params
cat_params['loss_function'] = 'RMSE'
cat_params['eval_metric'] = 'RMSE'
cat_params['bootstrap_type']= 'Bernoulli'
cat_params['leaf_estimation_method'] = 'Newton'
cat_params['random_state'] = 42
cat_params['task_type']='GPU'
test_preds=None

print("\033[93mTraining........")

kf = StratifiedKFold(n_splits = 15 , shuffle = True , random_state = 42)
for fold, (tr_index , val_index) in enumerate(kf.split(x.values , y.values)):
    
    print("⁙" * 15)
    print(f"Fold {fold + 1}")
    
    x_train,x_val = x.values[tr_index] , x.values[val_index]
    y_train,y_val = y.values[tr_index] , y.values[val_index]
        
    eval_set = [(x_val, y_val)]
    
    model =CatBoostRegressor(**cat_params)
    model.fit(x_train, y_train, eval_set = eval_set, verbose = False)
    
    train_preds = model.predict(x_train)    
    val_preds = model.predict(x_val)
    
    print(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    if test_preds is None:
        test_preds = model.predict(test.values)
    else:
        test_preds += model.predict(test.values)

print("-" * 50)
print("\033[95mTraining Done")

test_preds /= 15

# Preparing Submission
submission = pd.read_csv("../input/tabular-playground-series-aug-2021/sample_submission.csv")

submission['loss']=test_preds

submission.to_csv("lda.csv",index=False)

