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

class NFL(object):
    def __init__(self, theFunction = 'games', theWeek = 5):
        self.theFunction = theFunction
        self.theWeek = theWeek
        self.manual_header = "Team,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,3D%,4D%,RZ%,3D%,4D%,RZ%"
        self.home_manual_header = "Home_Team,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,3D%,4D%,RZ%,3D%,4D%,RZ%"
        self.visitor_manual_header = "Visitor_Team,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,3D%,4D%,RZ%,3D%,4D%,RZ%"

    def setup(self):
        def def_value():
            return "Not Present"
        # https://www.pro-football-reference.com/years/2021/
        # URL of page
        url = 'https://www.pro-football-reference.com/years/2021/'
        # Open URL for each team and pass to BeautifulSoup
        # html = urlopen(url)
        # stats_page = BeautifulSoup(html)
        self.teams_urls = ['https://www.pro-football-reference.com/teams/mia/2021.htm', 'https://www.pro-football-reference.com/teams/nwe/2021.htm',
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

        self.teams_names = ['mia_dolphins', 'ne_patriots',
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

        self.teams_proper_names = ['Miami Dolphins', 'New England Patriots',
        'Buffalo Bills', 'New York Jets',
        'Cleveland Browns', 'Baltimore Ravens',
        'Cincinnati Bengals', 'Pittsburgh Steelers',
        'Tennessee Titans', 'Houston Texans',
        'Indianapolis Colts', 'Jacksonville Jaguars',
        'Denver Broncos', 'Las Vegas Raiders',
        'Los Angeles Chargers', 'Kansas City Chiefs',
        'Philadelphia Eagles', 'Washington Football Team',
        'Dallas Cowboys', 'New York Giants',
        'Green Bay Packers', 'Chicago Bears',
        'Minnesota Vikings', 'Detroit Lions',
        'Tampa Bay Buccaneers', 'Carolina Panthers',
        'New Orleans Saints', 'Atlanta Falcons',
        'Los Angeles Rams', 'San Francisco 49ers',
        'Arizona Cardinals', 'Seattle Seahawks']

        # dictionary for the teams names and urls
        self.teams = defaultdict(def_value)
        self.team_names = []
        for i in range(len(self.teams_names)):
            self.teams[self.teams_names[i]] = self.teams_urls[i]

    def getStats(self):
        counter = 1
        name_counter = 0
        for team in self.teams_urls:
            current_team = urlopen(team)
            stats_page = BeautifulSoup(current_team, features="lxml")
            # Collect table headers
            column_headers = stats_page.findAll('tr')[1]
            column_headers = [i.getText() for i in column_headers.findAll('th')]
            column_headers = column_headers[1:]
            #print(column_headers)
            #
            # Get weekly team stats csv files for each team
            # 
            the_table = stats_page.find("table", {"id": "team_stats"})
            # Collect table rows
            rows = the_table.findAll('tr')[1:]
            # Get stats from each row
            team_stats = []
            for i in range(len(rows)):
                team_stats.append([col.getText() for col in rows[i].findAll('td')])
            # Create DataFrame from our scraped data
            data = pd.DataFrame(team_stats, columns=column_headers)
            # drop empty columns
            #data.replace('', np.nan, inplace=True)
            #data.dropna(how='all', axis=1, inplace=True)
            # Combine the two rows
            combined_row = ''
            for item in data.iloc[3]:
                if len(item) > 0:
                    combined_row = combined_row + ',' + str(item)
            for item in data.iloc[4]:
                if len(item) > 0:
                    combined_row = combined_row + ',' + str(item)
            combined_row = combined_row[1:]
            #
            # end get team stats
            #
            #
            # Get weekly team conversions csv files for each team
            # 
            the_conv_table = stats_page.find("table", {"id": "team_conversions"})
            # Collect table rows
            rows_conv = the_conv_table.findAll('tr')[1:]
            # Get stats from each row
            team_conv_stats = []
            for i in range(len(rows_conv)):
                team_conv_stats.append([col.getText() for col in rows_conv[i].findAll('td')])
            # Create DataFrame from our scraped data
            data_conv = pd.DataFrame(team_conv_stats)
            # drop empty columns
            #data_conv.replace('', np.nan, inplace=True)
            #data_conv.dropna(how='all', axis=1, inplace=True)
            # Combine the two rows
            combined_conv_row = ''
            for item in data_conv.iloc[3]:
                if len(item) > 0:
                    combined_conv_row = combined_conv_row + ',' + str(item)
            for item in data_conv.iloc[4]:
                if len(item) > 0:
                    combined_conv_row = combined_conv_row + ',' + str(item)
            combined_conv_row = combined_conv_row[1:]
            #
            # end get team conversions
            #
            # next normalize the data
            # then write to training files (spread and over_under)
            # self.manual_header = "Team,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,PF,Yds,TO,FL,1stD,PAtt,PYds,PTD,PInt,PNY/A,RAtt,RYds,RTD,RY/A,Sc%,TO%,AvDrvStart,AveDrvTime,AveDrvPlays,AveDriveYds,AveDrivePts,3D%,4D%,RZ%,3D%,4D%,RZ%"
            file_name = "week_" + str(self.theWeek) + "_teams.csv"
            with open(file_name, "a") as teams_file:
                if counter == 1:
                    teams_file.write(self.manual_header + '\n')
                teams_file.write(self.teams_proper_names[name_counter] + ',' + combined_row + ',' + combined_conv_row + '\n')
            name_counter += 1
            counter += 1
        file_path = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_teams.csv"
        shutil.move(file_name, file_path)
    def createGames(self):
        read_file_path = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_teams.csv"
        read_file_name = "week_" + str(self.theWeek) + "_teams.csv"
        games_url = "https://www.pro-football-reference.com/years/2021/week_" + str(self.theWeek) + ".htm"
        with open(read_file_path) as teams_file:
            teams_read = [team.rstrip() for team in teams_file]
        #for team in teams_read:
            # 
        # create the games
        current_week = urlopen(games_url)
        stats_page = BeautifulSoup(current_week, features="lxml")
        the_table = stats_page.find("div", {"class": "game_summaries"})
        rows = the_table.findAll('tr')
        game_stats = []
        for i in range(len(rows)):
            game_stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
        data = pd.DataFrame(game_stats)
        # get the away and home team names
        combined_row = ''
        home_counter = 0
        days_of_the_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for j in range(1, len(data)):
            for item in data.iloc[j]:
                #if item not in days_of_the_week and item != None and len(item) > 0 and "\n" not in item and "\t" not in item:
                if item in self.teams_proper_names and item != None and len(item) > 0:
                    combined_row = combined_row + ',' + str(item)
        combined_row = combined_row[1:]
        #print(combined_row)
        the_game_opponents = list(map(str, combined_row.split(",")))
        #print("\nNumber of opponents: ", len(the_game_opponents))
        # iterate over the_game_opponents, use teams_read rows
        # and write game rows to week_x_games_spread.csv and week_x_games_O-U.csv
        team_counter = 1
        game_row = ''
        write_row = False
        games_file_name = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_games.csv"
        for team in the_game_opponents:
            team_data = [item for item in teams_read if team in item]
            if team_counter % 2 != 0:
                game_row = str(team_data)
                write_row = False
            else:
                game_row = game_row + ',' + str(team_data)
                game_row = game_row.replace("['", "")
                game_row = game_row.replace("']", "")
                write_row = True
            if write_row:
                with open(games_file_name, "a") as games_file:
                    if team_counter == 2:
                        games_file.write(self.visitor_manual_header + ',' + self.home_manual_header + '\n')
                    games_file.write(game_row + '\n')
            team_counter += 1
    def combinedGameResults(self):
        # spread
        # read in last week's results
        results_file_name = "../NFL_Model/week_" + str(self.theWeek - 1) + "/week_" + str(self.theWeek - 1) + "_spread.csv"
        with open(results_file_name) as data_file:
            last_weeks_rows_read = [row.rstrip() for row in data_file]
        # append this week's results to last weeks results
        results_file_name = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_spread.csv"
        with open(results_file_name) as data_file:
            this_weeks_rows_read = [row.rstrip() for row in data_file]
        # write out combined results
        write_file_path = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_spread_combined.csv"
        with open(write_file_path, "a") as combined_file:
            for row in last_weeks_rows_read:
                combined_file.write(row + '\n')
        with open(write_file_path, "a") as combined_file:
            counter_rows = 0
            for row in this_weeks_rows_read:
                if counter_rows != 0:
                    combined_file.write(row + '\n')
                counter_rows += 1
        # over under
        # read in last week's results, make sure the files are named correctly
        results_file_name = "../NFL_Model/week_" + str(self.theWeek - 1) + "/week_" + str(self.theWeek - 1) + "_over_under.csv"
        with open(results_file_name) as data_file:
            last_weeks_rows_read = [row.rstrip() for row in data_file]
        # append this week's results to last weeks results
        results_file_name = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_over_under.csv"
        with open(results_file_name) as data_file:
            this_weeks_rows_read = [row.rstrip() for row in data_file]
        # write out combined results
        write_file_path = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_over_under_combined.csv"
        with open(write_file_path, "a") as combined_file:
            for row in last_weeks_rows_read:
                combined_file.write(row + '\n')
        with open(write_file_path, "a") as combined_file:
            counter_rows = 0
            for row in this_weeks_rows_read:
                if counter_rows != 0:
                    combined_file.write(row + '\n')
                counter_rows += 1
    def gameResults(self):
        read_file_path = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_games.csv"
        read_file_name = "week_" + str(self.theWeek) + "_games.csv"
        games_url = "https://www.pro-football-reference.com/years/2021/week_" + str(self.theWeek) + ".htm"
        with open(read_file_path) as games_file:
            games_read = [game.rstrip() for game in games_file]
        # create the games
        current_week = urlopen(games_url)
        stats_page = BeautifulSoup(current_week, features="lxml")
        the_table = stats_page.find("div", {"class": "game_summaries"})
        rows = the_table.findAll('tr')
        game_results = []
        for i in range(len(rows)):
            game_results.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
        data = pd.DataFrame(game_results)
        # get the team scores
        combined_row = ''
        home_counter = 0
        divider = ';'
        games_data = []
        individual_game = []
        # try placing game results in nested list game[[home team name:score][visitor team name:score]] versus separating via ';' and ','
        for j in range(1, len(data)):
            for item in data.iloc[j]:
                if item in self.teams_proper_names:
                    combined_row = combined_row + str(item) + ':'
                    home_counter += 1
                else:
                    try:
                        score = int(item)
                    except ValueError:
                        break
                    if 0 <= int(item) <= 100:
                        if home_counter % 2 != 0:
                            divider = ','
                            combined_row = combined_row + str(item)
                            individual_game.append(combined_row)
                            combined_row = ''
                        else:
                            divider = ';'
                            combined_row = combined_row + str(item)
                            individual_game.append(combined_row)
                            games_data.append(individual_game)
                            individual_game = []
                            combined_row = ''
        combined_row = combined_row[1:]
        # use read games.csv into a list here
        # already have games_read as a list
        games_output = games_read[1:]
        spread_file_name = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_spread.csv"
        over_under_file_name = "../NFL_Model/week_" + str(self.theWeek) + "/week_" + str(self.theWeek) + "_over_under.csv"
        row_counter = 1
        for game in games_data:
            home_score = game[1].split(':')[1:]
            home_team = game[1].split(':')[0]
            visitor_score = game[0].split(':')[1:]
            visitor_team = game[0].split(':')[0]
            spread = int(home_score[0]) - int(visitor_score[0])
            print(home_team + ' vs ' + visitor_team + ' spread: ' + str(spread))
            over_under = int(home_score[0]) + int(visitor_score[0])
            print(home_team + ' vs ' + visitor_team + ' over_under: ' + str(over_under))
            # add spread and over_under to individual files that extend the games.csv file
            # by searching for the game in the read in games.csv list
            # search for game row
            game_row_ = [game_spread for game_spread in games_output if home_team in game_spread]
            with open(spread_file_name, "a") as spread_file:
                if row_counter == 1:
                    spread_file.write(self.visitor_manual_header + ',' + self.home_manual_header + ',spread' + '\n')
                spread_file.write(game_row_[0] + ',' + str(spread) + '\n')
            with open(over_under_file_name, "a") as over_under_file:
                if row_counter == 1:
                    over_under_file.write(self.visitor_manual_header + ',' + self.home_manual_header + ',over_under' + '\n')
                over_under_file.write(game_row_[0] + ',' + str(over_under) + '\n')
            row_counter += 1

if __name__ == "__main__":
    funct = ''
    if len(sys.argv) == 1:
        print("Usage: NFLGames.py teams|games|results week")
        my_nfl = NFL()
    elif len(sys.argv) > 2:
        week = int(sys.argv[1])
        funct = sys.argv[2]
        my_nfl = NFL(funct, week)
    else:
        my_nfl = NFL()
    my_nfl.setup()
    if funct == "teams":
        my_nfl.getStats()
    elif funct == "games":
        my_nfl.createGames()
    elif funct == "results":
        my_nfl.gameResults()
    elif funct == "combine_results":
        my_nfl.combinedGameResults()
