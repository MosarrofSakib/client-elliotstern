import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from sklearn import datasets
from sklearn.cluster import KMeans
import os
import pulp
import numpy as np
import plotly.express as px
from dash_table import DataTable
#from multielo import MultiElo
import re
from datetime import timedelta, date, datetime
from typing import List
import math

# between -2 to 10 on elo of players

"""

You need the same column names

Handicap in your handicap and age dataframe will be the handicaps used in the algorithm



Save the files in CSV form



formula

Ai: A player of Team A.
RAi: Rating of player Ai.

RA / RB: Average rating of team A / average rating of team B.
EAi: Individual expectation value of player Ai for a given match—indicates the chances of success for the player and is calculated from RAi and RB.
MAi: Individual number of minutes played by Ai in a given match—indicates how long the player was on the pitch.
DAi: Individual goal difference of Ai for a given match—indicates the difference of the goals of the teams that were scored while the player was on the pitch.



Update:

no player minutes data and most players play the full game


Ai: A player of Team A.
RAi: Rating of player Ai.

RA / RB: Average rating of team A / average rating of team B.
EAi: Individual expectation value of player Ai for a given match—indicates the chances of success for the player and is calculated from RAi and RB.



no goal difference. Win
"""
#########################
# HANDICAP ELO
##########################
elo_band = 120

adjustment_multiplier = .2

#########################
# IMPORT
#########################
df = pd.read_csv(
    "./csv/uk_results_21.csv")
df_age_handicap = pd.read_csv(
    "./csv/handicap and age 2021.csv")

#########################
# CLEAN
#########################
# ------------------------------------------------------------------------------------
# Clean Player Age
# ------------------------------------------------------------------------------------
df_age_handicap = df_age_handicap[["Name", "Handicap", "Age"]]

# Clean names
# Remove paranthesized name
df_age_handicap["Name"] = df_age_handicap["Name"].str.replace("\(.*\)", "")
df_age_handicap["Name"] = df_age_handicap["Name"].str.replace("  ", " ")

# Create handicap elo
df_age_handicap["Handicap"] = df_age_handicap["Handicap"].str.replace(
    "[a-z]|[A-Z]|\(|\)", "")
df_age_handicap["Elo"] = 1000 + \
    (df_age_handicap["Handicap"].astype(float) + 2) * elo_band

# ------------------------------------------------------------------------------------
# Clean Elo
# ------------------------------------------------------------------------------------
# CLEAN to long
df = df.rename(columns={"Team A": "Team",
                        "Team B": "Opponent",
                        "Goals Scored TA": "Goals",
                        "Goals Scored TB": "Opponent Goals"})

# CLEAN names
# remove subs (place)
sub_columns = [k for k in df.columns if 'SUB' in k]
df[sub_columns[0]] = df[sub_columns[0]].str.replace("\(.*\)", "")
df[sub_columns[1]] = df[sub_columns[1]].str.replace("\(.*\)", "")

# split to names and ratings
regex = re.compile(r'P\d')
player_columns = list(filter(regex.search, df.columns))

# CREATE COLUMNS: HANDICAP
for i in range(len(player_columns)):
    df[player_columns[i] + " Handicap"] = df[player_columns[i]
                                             ].str.replace("\D", "")
for i in range(len(sub_columns)):
    df[sub_columns[i] + " Handicap"] = df[sub_columns[i]].str.replace("\D", "")
# CREATE COLUMNS: NAME (CLEANED)
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace("\(n\)", "")
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace("\(\)", "")
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace("\d", "")
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace(" - ", "-")
# Remove paranthesis
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace("\(.*\)", "")
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace("  ", " ")


for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace(
        "\(n\)|\(st\)|\(rd\)|\(th\)", "")
for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace("\d", "")
for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace("\(\)", "")
for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace("\d", "")
# Remove space before and after -
for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace(" - ", "-")
# Remove paranthesis
for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace("\(.*\)", "")
for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.replace("  ", " ")


# Clean names
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.strip()

for i in range(len(sub_columns)):
    df[sub_columns[i]] = df[sub_columns[i]].str.strip()

# Remove period to make it match
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace("\\.", "")

# Remove Double quotes for removing nicknames
for i in range(len(player_columns)):
    df[player_columns[i]] = df[player_columns[i]].str.replace(' “.*”', '')


# Set starting elo rating
default_elo = 1000.0
df[["Elo_B_P1_pre", "Elo_B_P2_pre", "Elo_B_P3_pre", "Elo_B_P4_pre", "Elo_B_P5_pre",
    "Elo_A_P1_pre", "Elo_A_P2_pre", "Elo_A_P3_pre", "Elo_A_P4_pre", "Elo_A_P5_pre",
    "Elo_B_P1", "Elo_B_P2", "Elo_B_P3", "Elo_B_P4", "Elo_B_P5",
    "Elo_A_P1", "Elo_A_P2", "Elo_A_P3", "Elo_A_P4", "Elo_A_P5",
    "Elo", "Opponent Elo"]] = default_elo

# Join Age
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team A P1", "Age": "age_A_P1", "Handicap": "Handicap_A_P1"}), on='Team A P1', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team A P2", "Age": "age_A_P2", "Handicap": "Handicap_A_P2"}), on='Team A P2', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team A P3", "Age": "age_A_P3", "Handicap": "Handicap_A_P3"}), on='Team A P3', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team A P4", "Age": "age_A_P4", "Handicap": "Handicap_A_P4"}), on='Team A P4', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "SUB Team A", "Age": "age_A_Sub", "Handicap": "Handicap_A_Sub"}), on='SUB Team A', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team B P1", "Age": "age_B_P1", "Handicap": "Handicap_B_P1"}), on='Team B P1', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team B P2", "Age": "age_B_P2", "Handicap": "Handicap_B_P2"}), on='Team B P2', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team B P3", "Age": "age_B_P3", "Handicap": "Handicap_B_P3"}), on='Team B P3', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "Team B P4", "Age": "age_B_P4", "Handicap": "Handicap_B_P4"}), on='Team B P4', how='left')
df = df.merge(df_age_handicap.rename(columns={
              "Name": "SUB Team B", "Age": "age_B_Sub", "Handicap": "Handicap_B_Sub"}), on='SUB Team B', how='left')

# Get a list of names that don't match
position_names = [k for k in df.columns if 'Team A' in k]
position_names.extend([k for k in df.columns if 'Team B' in k])
position_names = [k for k in position_names if 'Handicap' not in k]

game_log_names = []
for i in range(len(position_names)):
    game_log_names.extend(df[position_names[i]])

name_difference = list(set(game_log_names) -
                       set(list(df_age_handicap["Name"])))
name_difference = [x for x in name_difference if str(x) != 'nan']
name_difference_string = ", ".join(map(str, name_difference))

# CLEAN AGE
df[['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub",
    'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4", "age_B_Sub"]] = df[['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub",
                                                                        'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4", "age_B_Sub"]].replace("+", "")

for i in range(len(['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub", 'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4", "age_B_Sub"])):
    df[['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub",
        'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4", "age_B_Sub"][i]] = df[['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub",
                                                                               'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4", "age_B_Sub"][i]].str.replace("+", "")


df[['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub",
    'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4", "age_B_Sub"]] = df[
    ['age_A_P1', "age_A_P2", "age_A_P3", "age_A_P4", "age_A_Sub", 'age_B_P1', "age_B_P2", "age_B_P3", "age_B_P4",
     "age_B_Sub"]].replace(np.nan, 24).astype(int)

# clean handicap
df[['Handicap_A_P1', "Handicap_A_P2", "Handicap_A_P3", "Handicap_A_P4", "Handicap_A_Sub",
    'Handicap_B_P1', "Handicap_B_P2", "Handicap_B_P3", "Handicap_B_P4", "Handicap_B_Sub"]] = df[[
        'Handicap_A_P1', "Handicap_A_P2", "Handicap_A_P3", "Handicap_A_P4", "Handicap_A_Sub",
        'Handicap_B_P1', "Handicap_B_P2", "Handicap_B_P3", "Handicap_B_P4", "Handicap_B_Sub"]].replace(np.nan, 0).astype(int)

df[['Team A P1 Handicap',
    'Team A P2 Handicap', 'Team A P3 Handicap', 'Team A P4 Handicap',
    'Team B P1 Handicap', 'Team B P2 Handicap', 'Team B P3 Handicap',
    'Team B P4 Handicap', 'SUB Team A Handicap', 'SUB Team B Handicap']] = df[['Team A P1 Handicap',
                                                                               'Team A P2 Handicap', 'Team A P3 Handicap', 'Team A P4 Handicap',
                                                                               'Team B P1 Handicap', 'Team B P2 Handicap', 'Team B P3 Handicap',
                                                                               'Team B P4 Handicap', 'SUB Team A Handicap', 'SUB Team B Handicap']].replace("", 0)

df[['Team A P1 Handicap',
    'Team A P2 Handicap', 'Team A P3 Handicap', 'Team A P4 Handicap',
    'Team B P1 Handicap', 'Team B P2 Handicap', 'Team B P3 Handicap',
    'Team B P4 Handicap', 'SUB Team A Handicap', 'SUB Team B Handicap']] = df[['Team A P1 Handicap',
                                                                               'Team A P2 Handicap', 'Team A P3 Handicap', 'Team A P4 Handicap',
                                                                               'Team B P1 Handicap', 'Team B P2 Handicap', 'Team B P3 Handicap',
                                                                               'Team B P4 Handicap', 'SUB Team A Handicap', 'SUB Team B Handicap']].replace(np.nan, 0).astype(int)

# CLEAN HANDICAP


# df["k_A_P1"] = df["age_A_P1"]
# df["k_A_P2"] = df["age_A_P2"]
# df["k_A_P3"] = df["age_A_P3"]
# df["k_A_P4"] = df["age_A_P4"]
# df["k_A_P5"] = df["age_A_Sub"]


df["Date Time"] = df["Date"] + " " + df["TIME"]
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%m/%d/%y %H:%M")

df["Winning Team"] = np.where(
    df["Goals"] > df["Opponent Goals"], df["Team"], df["Opponent"])
df["Losing Team"] = np.where(
    df["Goals"] < df["Opponent Goals"], df["Team"], df["Opponent"])
df["Winning Team Elo"] = np.where(
    df["Goals"] > df["Opponent Goals"], df["Elo"], df["Opponent Elo"])
df["Losing Team Elo"] = np.where(
    df["Goals"] < df["Opponent Goals"], df["Elo"], df["Opponent Elo"])

# df["Winning Team Score"]
# df["Losing Team Score"]


df_results = df[["Date", "Team", "Goals", "Opponent Goals"]]
results_columns = [{'name': col, 'id': col} for col in df_results.columns]


#########################
# CALCULATE
#########################
def expected(A, B):
    """
    Expected score of A
    :param A: Elo rating for player A (average rating of team A)
    :param B: Elo rating for player B (average rating of team B)
    """
    return 1 / (1 + 10 ** ((B - A) / 400))


def elo(old, exp, score, k=32):
    """
    :param old: The Previous Elo rating
    :param exp: The expected score for this match
    :param score: The actual score for this match
    :param k: The k-factor for Elo (default: 32)
    """
    return old + k * (score - exp)


def elo_points(old, exp, goal_dif, k=32, w=1):
    """
    :param old: The Previous Elo rating
    :param exp: The expected score for this match
    :param goal_dif: Goal difference of the game
    :param k: The k-factor for Elo (default: 32)
    :param w: The weight of a game
    """
    if goal_dif > 0:
        game_score = 1
    elif goal_dif == 0:
        game_score = .5
    else:
        game_score = 0
    if goal_dif != 0:
        elo_change = w * (game_score - exp) * abs(goal_dif) ** (1 / 3)
    else:
        elo_change = w * (game_score - exp)
    return old + k * ((1 * elo_change) + ((1 - 1) * elo_change * 1))


def Probability(rating1, rating2):
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))


df["Goal Differential"] = df["Goals"] - df["Opponent Goals"]
df["Wins"] = np.where(df["Goals"] > df["Opponent Goals"], 1, 0)
df["Loses"] = np.where(df["Goals"] < df["Opponent Goals"], 1, 0)

df = df.sort_values("Date Time")
# add a lag cleaned on the first to prevent double counting games from both team's perspective


past_elo = pd.DataFrame()
# past_elo =
# find pre (0, 6)
for i in range(0, len(df)):
    # Get the player elo rating, default if no previous data
    try:
        # get player name of winner
        elo_a_P1 = past_elo[df["Team A P1"].iloc[i] == past_elo["player"]]
        elo_a_P1 = elo_a_P1[elo_a_P1["Date Time"] ==
                            max(elo_a_P1["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team A P1"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_a_P1 = df_age_handicap[df["Team A P1"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_a_P1 = default_elo
    try:
        elo_a_P2 = past_elo[df["Team A P2"].iloc[i] == past_elo["player"]]
        elo_a_P2 = elo_a_P2[elo_a_P2["Date Time"] ==
                            max(elo_a_P2["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team A P2"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_a_P2 = df_age_handicap[df["Team A P2"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_a_P2 = default_elo
    try:
        elo_a_P3 = past_elo[df["Team A P3"].iloc[i] == past_elo["player"]]
        elo_a_P3 = elo_a_P3[elo_a_P3["Date Time"] ==
                            max(elo_a_P3["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team A P3"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_a_P3 = df_age_handicap[df["Team A P3"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_a_P3 = default_elo
    try:
        elo_a_P4 = past_elo[df["Team A P4"].iloc[i] == past_elo["player"]]
        elo_a_P4 = elo_a_P4[elo_a_P4["Date Time"] ==
                            max(elo_a_P4["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team A P4"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_a_P4 = df_age_handicap[df["Team A P4"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_a_P4 = default_elo
    try:
        elo_a_P5 = past_elo[df["SUB Team A"].iloc[i] == past_elo["player"]]
        elo_a_P5 = elo_a_P5[elo_a_P5["Date Time"] ==
                            max(elo_a_P5["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["SUB Team A"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_a_P5 = df_age_handicap[df["SUB Team A"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_a_P5 = default_elo
    try:
        # get player name of winner
        elo_b_P1 = past_elo[df["Team B P1"].iloc[i] == past_elo["player"]]
        elo_b_P1 = elo_b_P1[elo_b_P1["Date Time"] ==
                            max(elo_b_P1["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team B P1"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_b_P1 = df_age_handicap[df["Team B P1"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_b_P1 = default_elo
    try:
        elo_b_P2 = past_elo[df["Team B P2"].iloc[i] == past_elo["player"]]
        elo_b_P2 = elo_b_P2[elo_b_P2["Date Time"] ==
                            max(elo_b_P2["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team B P2"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_b_P2 = df_age_handicap[df["Team B P2"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_b_P2 = default_elo
    try:
        elo_b_P3 = past_elo[df["Team B P3"].iloc[i] == past_elo["player"]]
        elo_b_P3 = elo_b_P3[elo_b_P3["Date Time"] ==
                            max(elo_b_P3["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team B P3"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_b_P3 = df_age_handicap[df["Team B P3"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_b_P3 = default_elo
    try:
        elo_b_P4 = past_elo[df["Team B P4"].iloc[i] == past_elo["player"]]
        elo_b_P4 = elo_b_P4[elo_b_P4["Date Time"] ==
                            max(elo_b_P4["Date Time"])]['Elo'].iloc[0]
    except:
        if len(df_age_handicap[df["Team B P4"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_b_P4 = df_age_handicap[df["Team B P4"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_b_P4 = default_elo
    try:
        elo_b_P5 = past_elo[df["SUB Team B"].iloc[i] == past_elo["player"]]
        elo_b_P5 = elo_b_P5[elo_b_P5["Date Time"] ==
                            max(elo_b_P5["Date Time"])]['Elo'].iloc[0]
    except:
        elo_b_P5 = default_elo
        if len(df_age_handicap[df["SUB Team B"].iloc[i] == df_age_handicap["Name"]]["Elo"]) > 0:
            elo_b_P5 = df_age_handicap[df["SUB Team B"].iloc[i]
                                       == df_age_handicap["Name"]]["Elo"].values[0]
        else:
            elo_b_P5 = default_elo

        # CALCUATE: New Elo
    if pd.isnull(df["SUB Team A"].iloc[i]):
        team_a_elo = (elo_a_P1 + elo_a_P2 + elo_a_P3 + elo_a_P4) / 4
    else:
        team_a_elo = (elo_a_P1 + elo_a_P2 + elo_a_P3 + elo_a_P4 + elo_a_P5) / 5
    if pd.isnull(df["SUB Team B"].iloc[i]):
        team_b_elo = (elo_b_P1 + elo_b_P2 + elo_b_P3 + elo_b_P4) / 4
    else:
        team_b_elo = (elo_b_P1 + elo_b_P2 + elo_b_P3 + elo_b_P4 + elo_b_P5) / 5

    df["Elo"].iloc[i] = team_a_elo
    df["Opponent Elo"].iloc[i] = team_b_elo

    p_1a_elo = elo_points(old=elo_a_P1, exp=expected(team_a_elo, team_b_elo),
                          goal_dif=df["Goals"].iloc[i] -
                          df["Opponent Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P1"].iloc[i] > 24]), w=1)
    p_2a_elo = elo_points(old=elo_a_P2, exp=expected(team_a_elo, team_b_elo),
                          goal_dif=df["Goals"].iloc[i] -
                          df["Opponent Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P2"].iloc[i] > 24]), w=1)
    p_3a_elo = elo_points(old=elo_a_P3, exp=expected(team_a_elo, team_b_elo),
                          goal_dif=df["Goals"].iloc[i] -
                          df["Opponent Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P3"].iloc[i] > 24]), w=1)
    p_4a_elo = elo_points(old=elo_a_P4, exp=expected(team_a_elo, team_b_elo),
                          goal_dif=df["Goals"].iloc[i] -
                          df["Opponent Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P4"].iloc[i] > 24]), w=1)
    p_5a_elo = elo_points(old=elo_a_P5, exp=expected(team_a_elo, team_b_elo),
                          goal_dif=df["Goals"].iloc[i] -
                          df["Opponent Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_Sub"].iloc[i] > 24]), w=1)
    p_1b_elo = elo_points(old=elo_b_P1, exp=expected(team_b_elo, team_a_elo),
                          goal_dif=df["Opponent Goals"].iloc[i] -
                          df["Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P1"].iloc[i] > 24]), w=1)
    p_2b_elo = elo_points(old=elo_b_P2, exp=expected(team_b_elo, team_a_elo),
                          goal_dif=df["Opponent Goals"].iloc[i] -
                          df["Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P2"].iloc[i] > 24]), w=1)
    p_3b_elo = elo_points(old=elo_b_P3, exp=expected(team_b_elo, team_a_elo),
                          goal_dif=df["Opponent Goals"].iloc[i] -
                          df["Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P3"].iloc[i] > 24]), w=1)
    p_4b_elo = elo_points(old=elo_b_P4, exp=expected(team_b_elo, team_a_elo),
                          goal_dif=df["Opponent Goals"].iloc[i] -
                          df["Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_P4"].iloc[i] > 24]), w=1)
    p_5b_elo = elo_points(old=elo_b_P5, exp=expected(team_b_elo, team_a_elo),
                          goal_dif=df["Opponent Goals"].iloc[i] -
                          df["Goals"].iloc[i],
                          k=adjustment_multiplier * (24 + (0, 16)[df["age_A_Sub"].iloc[i] > 24]), w=1)

    # Add The Updated Elo Ratings
    # df["Winning Team Elo"].iloc[i] = elo.get_new_ratings(np.array([w_team_past_elo, l_team_past_elo]))[0]
    # df["Losing Team Elo"].iloc[i] =  elo.get_new_ratings(np.array([w_team_past_elo, l_team_past_elo]))[1]

    df["Elo_A_P1_pre"].iloc[[i]] = df["Elo_A_P1"].iloc[[i]]
    df["Elo_A_P2_pre"].iloc[[i]] = df["Elo_A_P2"].iloc[[i]]
    df["Elo_A_P3_pre"].iloc[[i]] = df["Elo_A_P3"].iloc[[i]]
    df["Elo_A_P4_pre"].iloc[[i]] = df["Elo_A_P4"].iloc[[i]]
    df["Elo_A_P5_pre"].iloc[[i]] = df["Elo_A_P5"].iloc[[i]]
    df["Elo_B_P1_pre"].iloc[[i]] = df["Elo_B_P1"].iloc[[i]]
    df["Elo_B_P2_pre"].iloc[[i]] = df["Elo_B_P2"].iloc[[i]]
    df["Elo_B_P3_pre"].iloc[[i]] = df["Elo_B_P3"].iloc[[i]]
    df["Elo_B_P4_pre"].iloc[[i]] = df["Elo_B_P4"].iloc[[i]]
    df["Elo_B_P5_pre"].iloc[[i]] = df["Elo_B_P5"].iloc[[i]]

    df = df.copy()

    df.loc[i, "Elo_A_P1"] = p_1a_elo
    df.loc[i, "Elo_A_P2"] = p_2a_elo
    df.loc[i, "Elo_A_P3"] = p_3a_elo
    df.loc[i, "Elo_A_P4"] = p_4a_elo
    df.loc[i, "Elo_A_P5"] = p_5a_elo
    df.loc[i, "Elo_B_P1"] = p_1b_elo
    df.loc[i, "Elo_B_P2"] = p_2b_elo
    df.loc[i, "Elo_B_P3"] = p_3b_elo
    df.loc[i, "Elo_B_P4"] = p_4b_elo
    df.loc[i, "Elo_B_P5"] = p_5b_elo

    # Add To The Past Elo Table
    past_elo = past_elo.append(df[["Date Time", "Team A P1", "Elo_A_P1", "Team A P1 Handicap"]].iloc[[i]].rename(
        columns={"Team A P1": "player", "Elo_A_P1": "Elo", "Team A P1 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team A P2", "Elo_A_P2", "Team A P2 Handicap"]].iloc[[i]].rename(
        columns={"Team A P2": "player", "Elo_A_P2": "Elo", "Team A P2 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team A P3", "Elo_A_P3", "Team A P3 Handicap"]].iloc[[i]].rename(
        columns={"Team A P3": "player", "Elo_A_P3": "Elo", "Team A P3 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team A P4", "Elo_A_P4", "Team A P4 Handicap"]].iloc[[i]].rename(
        columns={"Team A P4": "player", "Elo_A_P4": "Elo", "Team A P4 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "SUB Team A", "Elo_A_P5", "SUB Team A Handicap"]].iloc[[i]].rename(
        columns={"SUB Team A": "player", "Elo_A_P5": "Elo", "SUB Team A Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team B P1", "Elo_B_P1", "Team B P1 Handicap"]].iloc[[i]].rename(
        columns={"Team B P1": "player", "Elo_B_P1": "Elo", "Team B P1 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team B P2", "Elo_B_P2", "Team B P2 Handicap"]].iloc[[i]].rename(
        columns={"Team B P2": "player", "Elo_B_P2": "Elo", "Team B P2 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team B P3", "Elo_B_P3", "Team B P3 Handicap"]].iloc[[i]].rename(
        columns={"Team B P3": "player", "Elo_B_P3": "Elo", "Team B P3 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "Team B P4", "Elo_B_P4", "Team B P4 Handicap"]].iloc[[i]].rename(
        columns={"Team B P4": "player", "Elo_B_P4": "Elo", "Team B P4 Handicap": "Handicap"}))
    past_elo = past_elo.append(df[["Date Time", "SUB Team B", "Elo_B_P5", "SUB Team A Handicap"]].iloc[[i]].rename(
        columns={"SUB Team B": "player", "Elo_B_P5": "Elo", "SUB Team A Handicap": "Handicap"}))
    # past_elo = past_elo.append(df[["Date Time", "Winning Team", "Winning Team Elo"]].iloc[[i-1]].rename(columns={"Winning Team": "Team", "Winning Team Elo": "Elo"}))
    # past_elo = past_elo.append(df[["Date Time", "Losing Team", "Losing Team Elo"]].iloc[[i-1]].rename(columns={"Losing Team": "Team", "Losing Team Elo": "Elo"}))
    # Remove nans from the past_elo table
    past_elo = past_elo[past_elo['player'].notnull()]

# CLEAN: Sub elos
df["Elo_A_P5"][df["SUB Team A"].isna()] = "-"
df["Elo_B_P5"][df["SUB Team B"].isna()] = "-"

df_cleaned = df[["Date", "Team", "Elo", "Goals", "Opponent Goals", "Tournament",
                 "Team A P1", "Elo_A_P1", "Team A P2", "Elo_A_P2", "Team A P3", "Elo_A_P3", "Team A P4", "Elo_A_P4",
                 "SUB Team A", "Elo_A_P5",
                 "Team A P1", "Elo_B_P1", "Team A P2", "Elo_B_P2", "Team A P3", "Elo_B_P3", "Team A P4", "Elo_B_P4",
                 "SUB Team B", "Elo_B_P5"
                 ]]
results_columns = [{'name': col, 'id': col} for col in df_cleaned.columns]

past_elo_cleaned = past_elo[past_elo['player'].notnull()]
# elo_columns = [{'name': col, 'id': col} for col in past_elo_cleaned.columns]
past_elo_cleaned = past_elo_cleaned.rename(
    index=str, columns={"player": "Player"})

# Get last elo
elo_total = past_elo_cleaned.groupby("Player").tail(1)

# Get the total amount of games per player
elo_games = past_elo_cleaned.groupby("Player").size().to_frame()
elo_games.reset_index(inplace=True)
elo_games = elo_games.rename(index=str, columns={0: "Games"})

# Join last elo, games
elo_total = elo_total.merge(elo_games, on='Player', how='left')
elo_total = elo_total[["Player", "Elo", "Games",
                       "Handicap"]].sort_values("Elo", ascending=False)
#df_age_join2 = df_age_join[["name", "Handicap"]].rename(columns={"name":"Player"})

#elo_total = elo_total[["Player", "Elo", "Games"]]

#elo_total = elo_total.merge(df_age_join2, on='Player', how='left')

elo_total["Handicap"] = elo_total["Handicap"].astype(float)
# CLEAN: Round Elo Ratings
#elo_total["Elo"] = elo_total["Elo"] + (4 - elo_total["Handicap"] * 80)

# CLEAN: Round Elo Ratings
elo_total["Elo"] = round(elo_total["Elo"], 1)
elo_columns = [{'name': col, 'id': col} for col in elo_total.columns]

line_graph = px.line(data_frame=past_elo_cleaned, x='Date Time',
                     y='Elo', color='Player', title='Polo Elo Ratings')
# line_graph.show()

past_elo2 = past_elo
past_elo2["test"] = list(range(0, len(past_elo2)))

#line_graph = px.line(data_frame=past_elo, x='Date Time', y='Elo', title='Total Sales by Month')
# line_graph.show()


#line_graph = px.line(data_frame=past_elo2, x='test', y='Elo', title='Total Sales by Month')
# line_graph.show()

# df = pd.DataFrame(dict(
#    x = [1, 3, 2, 4],
#    y = [1, 2, 3, 4]
# ))
#line_graph = px.line(df, x="x", y="y", title="Unsorted Input")
# line_graph.show()

#######
the_graph = dcc.Graph(
    id='example-graph',
    figure=line_graph
)

elo_datatable = DataTable(columns=elo_columns, data=elo_total.to_dict('records'), cell_selectable=False,
                          fill_width=False, editable=True,
                          style_cell={'font_size': '12px'}, filter_action="native"
                          )

the_datatable = DataTable(columns=results_columns, data=df_cleaned.to_dict('records'), cell_selectable=False,
                          fill_width=False, editable=True,
                          style_cell={'font_size': '12px'}
                          )

center_style = {"textAlign": "center"}


def section_header(text: str):
    return html.H4(children=text, style=center_style)


def win_probability_tab():
    return html.Div(children=[
        dbc.Row(children=[
            dbc.Col(width=3, children=[
                section_header("Player List"),
                dcc.Markdown(className="text-muted",
                             children="Select which players will participate."),
                dbc.Checklist(id="player-options", value=[]),
                html.Br(),
                dbc.Button(id="clear-button",
                           children="Clear selections", color="primary")
            ]),

            dbc.Col(width=9, children=[
                section_header("Probability of Results"),
                html.Div(id="win-probability-table")
            ])
        ]),
    ])


def tab1():
    return html.Div([
        dbc.Row(
            [
                dbc.Col(elo_datatable, md=0),
                dbc.Col(the_graph, md=0)
            ],
            align="center",
        ),
        html.Br(),
        html.Hr(),
        the_datatable
    ])


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# DF TRANSLATE to       [{"label": "Option 1", "value": 1}, {"label": "Option 2", "value": 2}]
player_checks = []
elo_total["id"] = 0
for i in range(len(elo_total)):
    player_checks.append({"label": elo_total["Player"][i], "value": i})
    elo_total["id"][i] = i


app.layout = html.Div([
    html.H1("Polo Rating", style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Current Elo Ratings', value='tab-1-example-graph'),
        dcc.Tab(label='Win Probabilities', value='tab-2-example-graph'),
    ]),
    html.Div(id='tabs-content-example-graph')
])


@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-example-graph':
        return html.Div([
            dbc.Row(
                [
                    dbc.Col(elo_datatable, md=0),
                    dbc.Col(the_graph, md=0)
                ],
                align="center",
            ),
            html.Br(),
            html.Hr(),
            the_datatable,
            html.H6(children=name_difference_string)
        ])
    elif tab == 'tab-2-example-graph':
        return html.Div([
            dbc.Row(children=[
                dbc.Col(width=3, children=[
                    section_header("Team 1"),
                    dcc.Markdown(className="text-muted",
                                 children="Select which players will participate on team 1."),
                    dbc.Checklist(options=player_checks, id="player-options"),
                    html.Br(),
                    dbc.Button(id="clear-button",
                               children="Clear selections", color="primary")
                ]),
                dbc.Col(width=3, children=[
                    section_header("Team 2"),
                    dcc.Markdown(className="text-muted",
                                 children="Select which players will participate on team 2."),
                    dbc.Checklist(options=player_checks, id="player-options2"),
                    html.Br(),
                    dbc.Button(id="clear-button",
                               children="Clear selections", color="primary")
                ]),
                dbc.Col(width=5, children=[
                    section_header("Projected Win Probability"),
                    html.Br(),
                    dcc.Markdown(
                        children="Odds of Team 1 Winning:"),
                    html.Div(id="win-probability-table"),
                    html.Br(),
                    dcc.Markdown(
                        children="Odds of Team 2 Winning:"),
                    html.Div(id="win-probability-table2"),
                ])
            ])
        ])


@app.callback(
    Output(component_id="win-probability-table",
           component_property="children"),
    Input(component_id="player-options", component_property="value"),
    Input(component_id="player-options2", component_property="value")

)
# def create_win_probability_table(players: List[tuple], players2: List[tuple]):
def create_win_probability_table(players, players2):
    # if not players:
    #    return None
    #n_players = len(players)
    #player_ids = [x[0] for x in players]
    #ratings = [x[1] for x in players]
    #players = elo_total[elo_total["id"] == players]["Elo"]
    #players2 = elo_total[elo_total["id"] == players2]["Elo"]
    prob = Probability(np.mean(elo_total.iloc[players2]["Elo"]), np.mean(
        elo_total.iloc[players]["Elo"])) * 100
    return str(round(prob, 1)) + "%"
    # return f'Output: {players}'

    # return str(round((Probability(np.mean(players), np.mean(players2)))*100, 1)) + "%"


@app.callback(
    Output(component_id="win-probability-table2",
           component_property="children"),
    Input(component_id="player-options", component_property="value"),
    Input(component_id="player-options2", component_property="value")

)
# def create_win_probability_table(players: List[tuple], players2: List[tuple]):
def create_win_probability_table(players, players2):
    # if not players:
    #    return None
    #n_players = len(players)
    #player_ids = [x[0] for x in players]
    #ratings = [x[1] for x in players]
    #players = elo_total[elo_total["id"] == players]["Elo"]
    #players2 = elo_total[elo_total["id"] == players2]["Elo"]
    prob = Probability(np.mean(elo_total.iloc[players2]["Elo"]), np.mean(
        elo_total.iloc[players]["Elo"])) * 100
    return str(round(100 - prob, 1)) + "%"
    # return f'Output: {players}'

    # return str(round((Probability(np.mean(players), np.mean(players2)))*100, 1)) + "%"


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=443)
