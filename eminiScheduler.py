#!/usr/bin/python3
# Import scraping modules
import string
import sys
import os
import shutil
from urllib.request import urlopen
from bs4 import BeautifulSoup
from pyicloud import PyiCloudService
from shutil import copyfileobj

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
def runIt():
    file_name = "/Users/vincentbloise/kaggle/automated_daily_granular.csv"
    with open(file_name, "a") as emini_file:
        now = datetime.now()
        day_of_week = now.weekday()
        what = days_of_the_week[day_of_week]
        if days_of_the_week[day_of_week] in weekdays:
            if now.hour == 9 and now.minute == 0 and now.second == 50:
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
            if now.hour == 14 and now.minute == 20 and now.second == 50:
                # emini_file.write(now.strftime("%m/%d/%Y") + "," + "6:00," + 'test @ 6:00' + '\n')
                # upload daily-granular.csv from the iCloud drive
                if upload_to_iCloud():
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

def upload_to_iCloud(command):
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
    if command == 'find_phone':
        print("phone location: " + str(iPhone.status()))
    elif command == 'play_sound':
        iPhone.play_sound("Dude, where's my phone?")
    elif command == 'get_files_dir':
        print(api.drive.dir())
    elif command == 'get_Futures_files':
        print(api.drive['Futures'].dir())
    elif command == 'read_daily_granular_old':
        drive_file = api.drive['Futures']['daily-granular.csv']
        with drive_file.open(stream=True) as response:
            with open(drive_file.name, 'wb') as file_out:
                copyfileobj(response.raw, file_out)
        print(file_out.name)
    elif command == 'read_daily_granular':
        drive_file = api.drive['Futures']['daily-granular.csv']
        print(drive_file.open().content.decode('ascii'))
        # now write it to a local file
        with open('downloaded_emini.csv', 'w') as f:
            f.write(drive_file.open().content.decode('ascii'))
    return True

# thedata = getData()

#if upload_to_iCloud():
#    print("successful iCloud API call")

###### scheduler code
'''
schedule.every(1).seconds.do(runIt)
 
while True:
    schedule.run_pending()
'''
if __name__ == "__main__":
    command = sys.argv[1]
    print(command)
    upload_to_iCloud(command)