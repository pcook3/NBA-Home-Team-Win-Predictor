import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('merged_filtered_games.csv')

# insert features here
features = [
    # Team performance
    'fieldGoalsPercentage_home', 'fieldGoalsPercentage_away',
    'threePointersPercentage_home', 'threePointersPercentage_away',
    'freeThrowsPercentage_home', 'freeThrowsPercentage_away',
    'reboundsTotal_home', 'reboundsTotal_away',
    'turnovers_home', 'turnovers_away',
    
    # Pre-game context
    'seasonWins_home', 'seasonLosses_home',
    'seasonWins_away', 'seasonLosses_away',
    'days_since_last_home_game', 'days_since_last_away_game',
    'rest_diff', 'back_to_back_home', 'back_to_back_away',
    'gameType_home', 'weekNumber_home', 'attendance_home',
]

# select features and target 
x = df[features]
y = df['win_home']