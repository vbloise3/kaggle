

# Setup clients
import boto3
import csv
from io import StringIO
s3client = boto3.client('s3')

import boto3

#sagemaker_session = sagemaker.Session()
bucket = 'eminisp500vbloise'
#role = sagemaker.get_execution_role()
#print(role)

base_job_prefix = "xgboost-example"
#default_bucket = sagemaker_session.default_bucket()
s3_prefix = base_job_prefix

training_instance_type = "ml.m5.xlarge"

import numpy as np
import pandas as pd
from numpy import loadtxt
train_data_key_raw = '15-Micro-Emini-SP500-MES-F-granular.csv'
train_data_key = '15-Micro-Emini-SP500-MES-F-granular-poped.csv'
test_data_key = '15-Micro-Emini-SP500-MES-F-granular-outcome-poped.csv'
inference_data_key_raw = 'daily-granular-15.csv'
inference_data_key = 'daily-granular-15-poped.csv'
train_data_raw_location = 's3://{}/{}'.format(bucket, train_data_key_raw)
inference_data_raw_location = 's3://{}/{}'.format(bucket, inference_data_key_raw)
train_data_location = 's3://{}/{}'.format(bucket, train_data_key)
test_data_location = 's3://{}/{}'.format(bucket, test_data_key)
inference_data_location = 's3://{}/{}'.format(bucket, inference_data_key)

'''
EminiSP = pd.read_csv(train_data_raw_location)
EminiSPpredict = pd.read_csv(inference_data_raw_location)
X = EminiSP.copy()
X.drop(['date-1', 'time-1', 'time-2', 'time-3', 'time-4', 'time-5', 'time-6', 'time-7', 'time-8', 'time-9', 'time-10', 'time-11', 'time-12', 'b_outcome'], axis=1, inplace=True) #,'percent-change-1', 'percent-change-2', 'percent-change-3'
y = EminiSP.pop('outcome')
EminiSPpredict.drop(['date-1', 'time-1', 'time-2', 'time-3', 'time-4', 'time-5', 'time-6', 'time-7', 'time-8', 'time-9', 'time-10', 'time-11', 'time-12'], axis=1, inplace=True) #,'percent-change-1', 'percent-change-2', 'percent-change-3'
'''
object_key = 'daily-granular-15-poped.csv'
csv_obj = s3client.get_object(Bucket='eminisp500vbloise', Key='15-Micro-Emini-SP500-MES-F-granular.csv')
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')
df = pd.read_csv(StringIO(csv_string))
df.drop(['date-1', 'time-1', 'time-2', 'time-3', 'time-4', 'time-5', 'time-6', 'time-7', 'time-8', 'time-9', 'time-10', 'time-11', 'time-12', 'b_outcome'], axis=1, inplace=True)
x = df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
vals = [','.join(ele.split()) for ele in x]
for row in vals:
    print(str(row))
df.to_csv('test_csv_file', sep=',', index=False, header=False)
#print(df.head())
#X.to_csv(train_data_location, sep=',')
#y.to_csv(test_data_location, sep=',')
#EminiSPpredict.to_csv(inference_data_location, sep=',')