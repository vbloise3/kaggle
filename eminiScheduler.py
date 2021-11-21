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
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
### use API to connect to iCloud
iCloudId = os.environ['ID']
iCloudPw = os.environ['PW']
api = PyiCloudService(iCloudId, iCloudPw)
iPhone = api.devices[3]
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
            if now.hour == 15 and now.minute == 6 and now.second == 50:
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

# thedata = getData()

#if upload_to_iCloud():
#    print("successful iCloud API call")

###### scheduler code
def runTimeLoop():
    schedule.every(1).seconds.do(runIt)
    
    while True:
        schedule.run_pending()

if __name__ == "__main__":
    command = sys.argv[1]
    print(command)
    # upload_to_iCloud(command)
    runTimeLoop()