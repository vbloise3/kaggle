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

def runIt():
    file_name = "/Users/vincentbloise/kaggle/test_cron.csv"
    with open(file_name, "a") as emini_file:
        now = datetime.now()
        emini_file.write(str(now) + "," + "1,2,4" + '\n')
 
schedule.every(10).seconds.do(runIt)
 
while True:
    schedule.run_pending()
