#!/usr/bin/python3
# Import scraping modules
from __future__ import print_function
import string
import sys
import os
import shutil
from urllib.request import urlopen
from bs4 import BeautifulSoup
from pyicloud import PyiCloudService
from shutil import copyfileobj
import requests
import json
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from apiclient import discovery, errors
from httplib2 import Http
from oauth2client import client, file, tools
import logging
import boto3
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd

# Import data manipulation modules
import pandas as pd
import numpy as np
# Import data visualization modules
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
import schedule
from datetime import datetime
days_of_the_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
### use API to connect to iCloud
iCloudId = os.environ['ID']
iCloudPw = os.environ['PW']
#api = PyiCloudService(iCloudId, iCloudPw)
#iPhone = api.devices[3]
###
### Google Drive stuff
# define variables
credentials_file_path = 'credentials.json'
clientsecret_file_path = 'client_secret.json'
# get SNS client
SNSclient = boto3.client("sns")
# define Google Drive scope
SCOPE = 'https://www.googleapis.com/auth/drive'
###
def runIt():
    file_name = "/Users/vincentbloise/kaggle/automated_daily_granular.csv"
    with open(file_name, "a") as emini_file:
        now = datetime.now()
        day_of_week = now.weekday()
        what = days_of_the_week[day_of_week]
        if days_of_the_week[day_of_week] in weekdays:
            '''
            if now.hour == 9 and now.minute == 0 and now.second == 50:
                # call trade_station_API() in each of these to get tick data
                emini_file.write(now.strftime("%m/%d/%Y") + "," + "9:00," + 'test @ 9:00' + '\n')
            if now.hour == 9 and now.minute == 30 and now.second == 50:
                emini_file.write(now.strftime("%m/%d/%Y") + "," + "9:30," + 'test @ 9:30' + '\n')
            if now.hour == 10 and now.minute == 0 and now.second == 50:
                emini_file.write(now.strftime("%m/%d/%Y") + "," + "10:00," + 'test @ 10:00' + '\n')
            if now.hour == 10 and now.minute == 30 and now.second == 50:
                emini_file.write(now.strftime("%m/%d/%Y") + "," + "10:30," + 'test @ 10:30' + '\n')
            if now.hour == 11 and now.minute == 0 and now.second == 50:
                emini_file.write(now.strftime("%m/%d/%Y") + "," + "11:00," + 'test @ 11:00' + '\n')
            if now.hour == 18 and now.minute == 0 and now.second == 50:
                emini_file.write(now.strftime("%m/%d/%Y") + "," + "6:00," + 'test @ 6:00' + '\n')
            '''
            if now.hour == 11 and now.minute == 15 and now.second == 50:
                # upload daily-granular.csv from the iCloud drive
                if upload_to_iCloud('read_daily_granular_automated'):
                    print("uploaded")

def getData():
    eminiSP500 = urlopen("https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.html")
    stats_page = BeautifulSoup(eminiSP500, features="lxml")
    theDiv = stats_page.find("div", {"class": "main-table-wrapper"})
    the_table = stats_page.find("table", {"id": "team_stats"})
    # Collect table rows
    rows = the_table.findAll('tr')[1:]
    # take the first row
    data_row = rows[0]
    print(data_row)
    # Get data from the row
    emini_data = []
    #emini_data.append([col.getText() for col in data_row.findAll('td')])
    return emini_data

def connect_iCloud():
    # connect to iCloud
    iCloudId = os.environ['ID']
    iCloudPw = os.environ['PW']
    api = PyiCloudService(iCloudId, iCloudPw)
    if api.requires_2fa:
        print ("Two-factor authentication required.")
        code = input("Enter the code you received of one of your approved devices: ")
        result = api.validate_2fa_code(code)
        print("Code validation result: %s" % result)
        if not result:
            print("Failed to verify security code")
            sys.exit(1)
        if not api.is_trusted_session:
            print("Session is not trusted. Requesting trust...")
            result = api.trust_session()
            print("Session trust result %s" % result)
            if not result:
                print("Failed to request trust. You will likely be prompted for the code again in the coming weeks")
    elif api.requires_2sa:
        import click
        print ("Two-step authentication required. Your trusted devices are:")

        devices = api.trusted_devices
        for i, device in enumerate(devices):
            print("  %s: %s" % (i, device.get('deviceName', "SMS to %s" % device.get('phoneNumber'))))
        device = click.prompt('Which device would you like to use?', default=0)
        device = devices[device]
        if not api.send_verification_code(device):
            print("Failed to send verification code")
            sys.exit(1)
        code = click.prompt('Please enter validation code')
        if not api.validate_verification_code(device, code):
            print("Failed to verify verification code")
            sys.exit(1)
    macBook = api.devices[5]
    iPad = api.devices[4]
    iPhone = api.devices[3]
    appleWatch = api.devices[2]
    macMini = api.devices[0]

def upload_to_iCloud(command):
    connect_iCloud()
    if command == 'find_phone':
        print("phone location: " + str(iPhone.status()))
    elif command == 'play_sound':
        iPhone.play_sound("Dude, where's my phone?")
    elif command == 'get_files_dir':
        print(api.drive.dir())
    elif command == 'get_Futures_files':
        print(api.drive['Futures'].dir())
    elif command == 'read_daily_granular_automated':
        drive_file = api.drive['Futures']['daily-granular.csv']
        with drive_file.open(stream=True) as response:
            with open(drive_file.name, 'wb') as file_out:
                copyfileobj(response.raw, file_out)
        print(file_out.name)
        push_emini_data_to_S3('daily-granular.csv', 'eminisp500vbloise')
    elif command == 'read_daily_granular':
        push_emini_data_to_S3('daily-granular.csv', 'eminisp500vbloise')
    elif command == "get_ticks":
        # use TradeStation API
        print(trade_station_API())
        # parse JSON for Open, High, Low, Close, TotalVolume
    return True

def push_emini_data_to_Drive():
    connect_iCloud()
    drive_file = api.drive['Futures']['daily-granular.csv']
    print(drive_file.open().content.decode('ascii'))
    # now write it to the local file called daily-granular.csv
    with open('daily-granular.csv', 'w') as f:
        f.write(drive_file.open().content.decode('ascii'))
    # next upload daily-granular.csv to the Google drive My Drive/LSTM Futures
    # define store
    store = file.Storage(credentials_file_path)
    credentials = store.get()

    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(clientsecret_file_path, SCOPE)
        credentials = tools.run_flow(flow, store)

    # define API service
    http = credentials.authorize(Http())
    drive = discovery.build('drive', 'v3', http=http)

    filedirectory = '/Users/vincentbloise/kaggle/daily-granular.csv'
    filename = 'daily-granular.csv'
    folderid = 'Futures'
    access_token = credentials
    metadata = {
        "name": filename,
        "parents": [folderid]
    }
    files = {
        'data': ('metadata', json.dumps(metadata), 'application/json'),
        'file': open(filedirectory, "rb").read()  # or  open(filedirectory, "rb")
    }
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers={"Authorization": "Bearer " + access_token},
        files=files
    )
    print(r.text)

def push_emini_data_to_S3(file_name, bucket, object_name=None):
    phoneNumber = os.environ['PN']
    topic_arn = 'arn:aws:sns:us-east-1:001178231653:emini'
    theMessage= 'daily-granular.csv uploaded to S3'
    if object_name is None:
        object_name = os.path.basename(file_name)
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    SNSclient.publish(Message=theMessage, TopicArn=topic_arn)
    print('daily-granular.csv uploaded to S3')
    return True

def trade_station_API():
    url = "https://api.tradestation.com/v3/marketdata/barcharts/MSFT?unit=minute"
    headers = {"Authorization": "Bearer TOKEN"}
    response = requests.request("GET", url, headers=headers)
    return response.text

def convert_15_minute_ticks():
    theFeature = []
    in_file_name = "calc-daily-granular-15.csv"
    eminiSPticks = pd.read_csv(in_file_name)
    # calculate the 5 differences here 
    eminiCalculatedTicks = eminiSPticks.copy()
    eminiCalculatedTicks['open-1'] = eminiSPticks['open-2'] -  eminiSPticks['open-1']
    eminiCalculatedTicks['high-1'] = eminiSPticks['high-2'] -  eminiSPticks['high-1']
    eminiCalculatedTicks['low-1'] = eminiSPticks['low-2'] -  eminiSPticks['low-1']
    eminiCalculatedTicks['close-1'] = eminiSPticks['close-2'] -  eminiSPticks['close-1']
    eminiCalculatedTicks['volume-1'] = eminiSPticks['volume-2'] -  eminiSPticks['volume-1']
    ##########
    eminiCalculatedTicks['open-2'] = eminiSPticks['open-3'] -  eminiSPticks['open-2']
    eminiCalculatedTicks['high-2'] = eminiSPticks['high-3'] -  eminiSPticks['high-2']
    eminiCalculatedTicks['low-2'] = eminiSPticks['low-3'] -  eminiSPticks['low-2']
    eminiCalculatedTicks['close-2'] = eminiSPticks['close-3'] -  eminiSPticks['close-2']
    eminiCalculatedTicks['volume-2'] = eminiSPticks['volume-3'] -  eminiSPticks['volume-2']
    ##########
    eminiCalculatedTicks['open-3'] = eminiSPticks['open-4'] -  eminiSPticks['open-3']
    eminiCalculatedTicks['high-3'] = eminiSPticks['high-4'] -  eminiSPticks['high-3']
    eminiCalculatedTicks['low-3'] = eminiSPticks['low-4'] -  eminiSPticks['low-3']
    eminiCalculatedTicks['close-3'] = eminiSPticks['close-4'] -  eminiSPticks['close-3']
    eminiCalculatedTicks['volume-3'] = eminiSPticks['volume-4'] -  eminiSPticks['volume-3']
    ##########
    eminiCalculatedTicks['open-4'] = eminiSPticks['open-5'] -  eminiSPticks['open-4']
    eminiCalculatedTicks['high-4'] = eminiSPticks['high-5'] -  eminiSPticks['high-4']
    eminiCalculatedTicks['low-4'] = eminiSPticks['low-5'] -  eminiSPticks['low-4']
    eminiCalculatedTicks['close-4'] = eminiSPticks['close-5'] -  eminiSPticks['close-4']
    eminiCalculatedTicks['volume-4'] = eminiSPticks['volume-5'] -  eminiSPticks['volume-4']
    ##########
    eminiCalculatedTicks['open-5'] = eminiSPticks['open-6'] -  eminiSPticks['open-5']
    eminiCalculatedTicks['high-5'] = eminiSPticks['high-6'] -  eminiSPticks['high-5']
    eminiCalculatedTicks['low-5'] = eminiSPticks['low-6'] -  eminiSPticks['low-5']
    eminiCalculatedTicks['close-5'] = eminiSPticks['close-6'] -  eminiSPticks['close-5']
    eminiCalculatedTicks['volume-5'] = eminiSPticks['volume-6'] -  eminiSPticks['volume-5']
    ##########
    eminiCalculatedTicks['open-6'] = eminiSPticks['open-7'] -  eminiSPticks['open-6']
    eminiCalculatedTicks['high-6'] = eminiSPticks['high-7'] -  eminiSPticks['high-6']
    eminiCalculatedTicks['low-6'] = eminiSPticks['low-7'] -  eminiSPticks['low-6']
    eminiCalculatedTicks['close-6'] = eminiSPticks['close-7'] -  eminiSPticks['close-6']
    eminiCalculatedTicks['volume-6'] = eminiSPticks['volume-7'] -  eminiSPticks['volume-6']
    ##########
    eminiCalculatedTicks['open-7'] = eminiSPticks['open-8'] -  eminiSPticks['open-7']
    eminiCalculatedTicks['high-7'] = eminiSPticks['high-8'] -  eminiSPticks['high-7']
    eminiCalculatedTicks['low-7'] = eminiSPticks['low-8'] -  eminiSPticks['low-7']
    eminiCalculatedTicks['close-7'] = eminiSPticks['close-8'] -  eminiSPticks['close-7']
    eminiCalculatedTicks['volume-7'] = eminiSPticks['volume-8'] -  eminiSPticks['volume-7']
    ##########
    eminiCalculatedTicks['open-8'] = eminiSPticks['open-9'] -  eminiSPticks['open-8']
    eminiCalculatedTicks['high-8'] = eminiSPticks['high-9'] -  eminiSPticks['high-8']
    eminiCalculatedTicks['low-8'] = eminiSPticks['low-9'] -  eminiSPticks['low-8']
    eminiCalculatedTicks['close-8'] = eminiSPticks['close-9'] -  eminiSPticks['close-8']
    eminiCalculatedTicks['volume-8'] = eminiSPticks['volume-9'] -  eminiSPticks['volume-8']
    ##########
    eminiCalculatedTicks['open-9'] = eminiSPticks['open-10'] -  eminiSPticks['open-9']
    eminiCalculatedTicks['high-9'] = eminiSPticks['high-10'] -  eminiSPticks['high-9']
    eminiCalculatedTicks['low-9'] = eminiSPticks['low-10'] -  eminiSPticks['low-9']
    eminiCalculatedTicks['close-9'] = eminiSPticks['close-10'] -  eminiSPticks['close-9']
    eminiCalculatedTicks['volume-9'] = eminiSPticks['volume-10'] -  eminiSPticks['volume-9']
    ##########
    eminiCalculatedTicks['open-10'] = eminiSPticks['open-11'] -  eminiSPticks['open-10']
    eminiCalculatedTicks['high-10'] = eminiSPticks['high-11'] -  eminiSPticks['high-10']
    eminiCalculatedTicks['low-10'] = eminiSPticks['low-11'] -  eminiSPticks['low-10']
    eminiCalculatedTicks['close-10'] = eminiSPticks['close-11'] -  eminiSPticks['close-10']
    eminiCalculatedTicks['volume-10'] = eminiSPticks['volume-11'] -  eminiSPticks['volume-10']
    ##########
    eminiCalculatedTicks['open-11'] = eminiSPticks['open-12'] -  eminiSPticks['open-11']
    eminiCalculatedTicks['high-11'] = eminiSPticks['high-12'] -  eminiSPticks['high-11']
    eminiCalculatedTicks['low-11'] = eminiSPticks['low-12'] -  eminiSPticks['low-11']
    eminiCalculatedTicks['close-11'] = eminiSPticks['close-12'] -  eminiSPticks['close-11']
    eminiCalculatedTicks['volume-11'] = eminiSPticks['volume-12'] -  eminiSPticks['volume-11']
    ##########
    eminiCalculatedTicks['open-12'] = eminiSPticks['open-13'] -  eminiSPticks['open-12']
    eminiCalculatedTicks['high-12'] = eminiSPticks['high-13'] -  eminiSPticks['high-12']
    eminiCalculatedTicks['low-12'] = eminiSPticks['low-13'] -  eminiSPticks['low-12']
    eminiCalculatedTicks['close-12'] = eminiSPticks['close-13'] -  eminiSPticks['close-12']
    eminiCalculatedTicks['volume-12'] = eminiSPticks['volume-13'] -  eminiSPticks['volume-12']
    #eminiSPticks.drop(['date-1', 'time-1', 'time-2', 'time-3', 'time-4', 'time-5', 'time-6', 'time-7', 'time-8', 'time-9', 'time-10', 'time-11', 'time-12', 'time-13'], axis=1, inplace=True)
    theFeature.append(''.join(str(e) for e in eminiSPticks['date-1'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-1'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-1'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-1'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-1'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-1'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-1'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-2'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-2'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-2'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-2'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-2'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-2'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-3'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-3'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-3'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-3'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-3'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-3'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-4'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-4'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-4'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-4'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-4'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-4'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-5'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-5'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-5'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-5'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-5'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-5'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-6'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-6'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-6'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-6'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-6'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-6'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-7'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-7'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-7'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-7'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-7'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-7'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-8'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-8'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-8'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-8'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-8'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-8'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-9'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-9'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-9'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-9'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-9'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-9'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-10'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-10'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-10'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-10'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-10'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-10'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-11'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-11'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-11'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-11'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-11'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-11'].values))
    theFeature.append(''.join(str(e) for e in eminiSPticks['time-12'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['open-12'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['high-12'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['low-12'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['close-12'].values))
    theFeature.append(''.join(str(e) for e in eminiCalculatedTicks['volume-12'].values))
    print(','.join(map(str, theFeature)))
    theArticulatedFeature = ','.join(map(str, theFeature))

    doing_historicals = False

    if doing_historicals:
        out_file_name = "15-Micro-Emini-SP500-MES-F-granular.csv"
        with open(out_file_name, "a") as ticks_transformed_file:
            ticks_transformed_file.write(theArticulatedFeature)
    else:
        daily_file_name = "daily-granular-15.csv"
        with open(daily_file_name, "a") as ticks_transformed_daily_file:
            ticks_transformed_daily_file.write(theArticulatedFeature)

# thedata = getData()

#if upload_to_iCloud():
#    print("successful iCloud API call")

###### scheduler code
'''
def runTimeLoop():
    schedule.every(1).seconds.do(runIt)
    
    while True:
        schedule.run_pending()
'''
if __name__ == "__main__":
    command = sys.argv[1]
    print(command)
    # upload_to_iCloud(command)
    # runTimeLoop()
    convert_15_minute_ticks()