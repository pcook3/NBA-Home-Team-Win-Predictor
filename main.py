import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

#dropping non-numeric types in x
x = x.select_dtypes(include=['number'])
x = x.select_dtypes(include=['number']).fillna(0)


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

#Logistic Regression (scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

logreg_preds = logreg.predict(X_test_scaled)

print("\n--- LOGISTIC REGRESSION ---")
print("Accuracy:", accuracy_score(y_test, logreg_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, logreg_preds))
print(classification_report(y_test, logreg_preds))

#Random Forest (no scaling)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

print("\n--- RANDOM FOREST ---")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

#interpret coefficients
coef_df = pd.DataFrame({
    "feature": x.columns,
    "coefficient": logreg.coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\n--- COEFFICIENTS ---")
print(coef_df.head(10))

#feature importatnce
importance_df = pd.DataFrame({
    'feature': x.columns,
    'importance': rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance_df.head(10))

# Probability the home team wins
home_win_prob = logreg.predict_proba(X_test_scaled)[:, 1]

predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': logreg_preds,
    'home_win_probability': home_win_prob
})

print("\n--- HOME WIN PROBABILITY ---")
print(predictions_df.head())

#saving model to predict future games
import joblib

joblib.dump(logreg, "logistic_nba_model.pkl")
joblib.dump(scaler, "nba_scaler.pkl")

# df_future = insert future dataframe here

# df_future_scaled = scaler.transform(df_future[X.columns])
# future_probs = logreg.predict_proba(df_future_scaled)[:, 1]

predictions_df['PredictionCorrect'] = predictions_df['actual'] == predictions_df['predicted']
predictions_df['PredictionCorrect'] = predictions_df['PredictionCorrect'].map({True:'Correct', False:'Incorrect'})
accuracy = (predictions_df['PredictionCorrect'] == 'Correct').mean()
print(accuracy)
predictions_df.to_csv("nba_predictions_for_dashboard.csv", index=False)
