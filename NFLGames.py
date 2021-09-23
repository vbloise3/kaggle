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

# https://www.pro-football-reference.com/years/2021/
# URL of page
url = 'https://www.pro-football-reference.com/years/2021/'
# Open URL for each team and pass to BeautifulSoup
# html = urlopen(url)
# stats_page = BeautifulSoup(html)
teams_urls = ['https://www.pro-football-reference.com/teams/mia/2021.htm', 'https://www.pro-football-reference.com/teams/nwe/2021.htm',
'https://www.pro-football-reference.com/teams/buf/2021.htm', 'https://www.pro-football-reference.com/teams/nyj/2021.htm',
'https://www.pro-football-reference.com/teams/cle/2021.htm', 'https://www.pro-football-reference.com/teams/rav/2021.htm',
'https://www.pro-football-reference.com/teams/cin/2021.htm', 'https://www.pro-football-reference.com/teams/pit/2021.htm',
'https://www.pro-football-reference.com/teams/oti/2021.htm', 'https://www.pro-football-reference.com/teams/htx/2021.htm',
'https://www.pro-football-reference.com/teams/clt/2021.htm', 'https://www.pro-football-reference.com/teams/jax/2021.htm',
'https://www.pro-football-reference.com/teams/den/2021.htm', 'https://www.pro-football-reference.com/teams/rai/2021.htm',
'https://www.pro-football-reference.com/teams/sdg/2021.htm', 'https://www.pro-football-reference.com/teams/kan/2021.htm',
'https://www.pro-football-reference.com/teams/phi/2021.htm', 'https://www.pro-football-reference.com/teams/was/2021.htm',
'https://www.pro-football-reference.com/teams/dal/2021.htm', 'https://www.pro-football-reference.com/teams/nyg/2021.htm',
'https://www.pro-football-reference.com/teams/gnb/2021.htm', 'https://www.pro-football-reference.com/teams/chi/2021.htm',
'https://www.pro-football-reference.com/teams/min/2021.htm', 'https://www.pro-football-reference.com/teams/det/2021.htm',
'https://www.pro-football-reference.com/teams/tam/2021.htm', 'https://www.pro-football-reference.com/teams/car/2021.htm',
'https://www.pro-football-reference.com/teams/nor/2021.htm', 'https://www.pro-football-reference.com/teams/atl/2021.htm',
'https://www.pro-football-reference.com/teams/ram/2021.htm', 'https://www.pro-football-reference.com/teams/sfo/2021.htm',
'https://www.pro-football-reference.com/teams/crd/2021.htm', 'https://www.pro-football-reference.com/teams/sea/2021.htm']

teams_names = ['mia_dolphins', 'ne_patriots',
'buf_bills', 'ny_jets',
'cle_browns', 'balt_ravens',
'cin_bengals', 'pit_steelers',
'ten_titans', 'hou_texans',
'ind_colts', 'jax_jaguars',
'den_broncos', 'lv_raiders',
'la_chargers', 'kc_chiefs',
'phil_eagles', 'wash_football',
'dal_cowboys', 'ny_giants',
'gb_packers', 'chi_bears',
'min_vikings', 'det_lions',
'tamp_bucs', 'car_panthers',
'no_saints', 'atl_falcons',
'la_rams', 'sf_49ers',
'az_cards', 'sea_seahawks']

def def_value():
    return "Not Present"

# dictionary for the teams names and urls
teams = defaultdict(def_value)
for i in range(len(teams_names)):
    teams[teams_names[i]] = teams_urls[i]
for team in teams_urls:
    current_team = urlopen(team)
    stats_page = BeautifulSoup(current_team, features="lxml")
    # Collect table headers
    column_headers = stats_page.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    column_headers = column_headers[1:]
    #print(column_headers)
    # Get weekly csv files for each team
    the_table = stats_page.find("table", {"id": "team_stats"})
    # Collect table rows
    rows = the_table.findAll('tr')[1:]
    # Get stats from each row
    team_stats = []
    for i in range(len(rows)):
        team_stats.append([col.getText() for col in rows[i].findAll('td')])
    # Create DataFrame from our scraped data
    data = pd.DataFrame(team_stats, columns=column_headers)
    # Replace time with int time
    (m, s) = str(data.iloc[1]['Time']).split(':')
    int_time_home = int(m) * 60 + int(s)
    data.iloc[1]['Time'] = data.iloc[1]['Time'].replace(data.iloc[1]['Time'], str(int_time_home))
    (m, s) = str(data.iloc[2]['Time']).split(':')
    int_time_opponent = int(m) * 60 + int(s)
    data.iloc[2]['Time'] = data.iloc[2]['Time'].replace(data.iloc[2]['Time'], str(int_time_opponent))
    # Replace start position by removing Own
    start_home = data.iloc[1]['Start']
    start_home = start_home.replace("Own ", "")
    data.iloc[1]['Start'] = data.iloc[1]['Start'].replace(data.iloc[1]['Start'], start_home)
    start_opponent = data.iloc[2]['Start']
    start_opponent = start_opponent.replace("Own ", "")
    data.iloc[2]['Start'] = data.iloc[2]['Start'].replace(data.iloc[2]['Start'], start_opponent)
    # Combine the two rows
    combined_row = ''
    for item in data.iloc[1]:
        combined_row = combined_row + ',' + item
    for item in data.iloc[2]:
        combined_row = combined_row + ',' + item
    combined_row = combined_row[1:]
    # next normalize the data
    # then write to training files (spread and over/under)
    with open("week_2_training.csv", "a") as training_file:
        training_file.write(combined_row + '\n')
shutil.move("week_2_training.csv", "../NFL_Model/week_2/week_2_training.csv")