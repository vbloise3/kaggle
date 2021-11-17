#!/usr/bin/python3
# Import scraping modules
import string
import sys
import os
import shutil
from urllib.request import urlopen
from bs4 import BeautifulSoup

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

thedata = getData()

schedule.every(1).seconds.do(runIt)
 
while True:
    schedule.run_pending()
