import pandas as pd 

df_games = pd.read_csv("Games.csv")
df_schedule = pd.read_csv("LeagueSchedule24_25.csv")
df_teamStats = pd.read_csv("TeamStatistics.csv")


#merging data frames
df = pd.merge(df_games, df_schedule, on="gameId", how="inner")
df = pd.merge(df, df_teamStats, on="gameId", how="inner")



#filtering out all games except those in the 2024-2025 Regular Season
start_date = 20241022
end_date = 20250413
df['gameDate'] = pd.to_datetime(df['gameDate_x'])
df['gameDate_int'] = df['gameDate'].dt.strftime('%Y%m%d').astype(int)
df = df[(df['gameDate_int'] >= start_date) & (df['gameDate_int'] <= end_date)]

columns_to_drop = ['gameDate_y', 'hometeamId_y', 'awayteamId_y', 'gameLabel_y', 
                   'gameSubLabel_y', 'seriesGameNumber_y', 'gameLabel_x', 'gameSubLabel_x', 
                   'seriesGameNumber_x', 'gameDay', 'gameSubtype', 'seriesText', 'coachId', 
                   'teamScore', 'opponentScore', 'teamName', 'teamId', 'opponentTeamCity', 
                   'opponentTeamName', 'opponentTeamId', 'opponentScore', 'winner', 
                   'teamCity']
df = df.drop(columns=columns_to_drop)

df = df.rename(columns={
    'gameDate_x': 'gameDate',
    'hometeamId_x': 'hometeamId',
    'awayteamId_x': 'awayteamId'
})

#drop the duplicate gameDate column (keeping the integer type column)
df = df.drop(columns=['gameDate'])

#combine home+away columns into one row for easy reference
home_df = df[df['home'] == 1].copy()
away_df = df[df['home'] == 0].copy()

df = home_df.merge(
    away_df,
    on='gameId',
    suffixes=('_home', '_away')
)


#feature engineering
df['fgpDiff'] = df['fieldGoalsPercentage_home'] - df['fieldGoalsPercentage_away']
df['3ppDiff'] = df['threePointersPercentage_home'] - df['threePointersPercentage_away']
df['reboundDiff'] = df['reboundsTotal_home'] - df['reboundsTotal_away']
df['turnoverDiff'] = df['turnovers_home'] - df['turnovers_away']

#creating a column for days since last game
df['gameDateTimeEst_home'] = pd.to_datetime(df['gameDateTimeEst_home'])
df = df.sort_values(by=['gameDateTimeEst_home'])

#for home
df['days_since_last_home_game'] = (
    df.groupby('hometeamId_home')['gameDateTimeEst_home']
      .diff()
      .dt.days
      .fillna(0)
)

#for away
df['days_since_last_away_game'] = (
    df.groupby('awayteamId_away')['gameDateTimeEst_home']
      .diff()
      .dt.days
      .fillna(0)
)

#more feature engineering
df['rest_diff'] = df['days_since_last_home_game'] = df['days_since_last_away_game']
df['back_to_back_home'] = (df['days_since_last_home_game'] == 1).astype(int)
df['back_to_back_away'] = (df['days_since_last_away_game'] == 1).astype(int)




#convert the merged and cleansed data frame into a CSV file (commented out the line below)
#df.to_csv('merged_filtered_games.csv', index=False)

print(df.columns.to_list())