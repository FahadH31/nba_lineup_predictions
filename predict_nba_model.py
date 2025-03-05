# Currently obtaining 84% accuracy on 500 tests.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os, joblib, numpy as np

# Function to load and preprocess the data
def load_and_preprocess_data(data_dir):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = []

    for file in all_files:  # Append all csv files into one dataframe to be used later
        df = pd.read_csv(file)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    
    # Sort all players alphabetically in the dataframe
    def sort_players(row):
        home_players = sorted([row[f'home_{i}'] for i in range(5)])
        away_players = sorted([row[f'away_{i}'] for i in range(5)])
        for i in range(5):
            row[f'home_{i}'] = home_players[i]
            row[f'away_{i}'] = away_players[i]
        return row
    df = df.apply(sort_players, axis=1)
    
    return df

# Function to create roster for each team and season
def create_roster(df):
    # Create a roster for each home team and season. 
    # Ex. GSW 2015 would be one roster, in 2016 it would be a different one.
    roster = df.groupby(['home_team', 'season'])[[f'home_{i}' for i in range(5)]].apply(
        lambda x: set(pd.unique(x.values.ravel()))
    ).to_dict()
    return roster

# Function to encode all categorical features to numerical values 
def encode_features(df, player_encoder, team_encoder, season_encoder):
    df['home_team_encoded'] = team_encoder.transform(df['home_team'])
    df['away_team_encoded'] = team_encoder.transform(df['away_team'])
    df['season_encoded'] = season_encoder.transform(df['season'])
    for i in range(5):
        df[f'home_{i}_encoded'] = player_encoder.transform(df[f'home_{i}'])
        df[f'away_{i}_encoded'] = player_encoder.transform(df[f'away_{i}'])
    return df


# Main Code
data_dir = './csv_files'
df = load_and_preprocess_data(data_dir) # Initialize dataframe

# Create dictionary with rosters for each team and season
roster_dict = create_roster(df)

# Initialize encoders
player_encoder = LabelEncoder()
team_encoder = LabelEncoder()
season_encoder = LabelEncoder()

# Fit all encoders on the data
# Get all players from all columns (avoiding duplicates)
all_players = pd.unique(df[[f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]].values.ravel())
player_encoder.fit(all_players)

teams = pd.unique(pd.concat([df['home_team'], df['away_team']]))
team_encoder.fit(teams)

seasons = pd.unique(df['season'])
season_encoder.fit(seasons)

# Encode the dataframe
df = encode_features(df, player_encoder, team_encoder, season_encoder)

# Prepare features and target
features = ['home_team_encoded', 'season_encoded', 'starting_min']
for i in range(5):
    features.extend([f'home_{i}_encoded', f'away_{i}_encoded'])

X = df[features]
y = df['outcome'].apply(lambda x: 1 if x == 1 else 0)  # Convert outcome to binary

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'encoders/nba_lineup_model.pkl')
joblib.dump(player_encoder, 'encoders/player_encoder.pkl')
joblib.dump(team_encoder, 'encoders/team_encoder.pkl')
joblib.dump(season_encoder, 'encoders/season_encoder.pkl')

# Prediction function, predicts best 5th player to maximize winning. Uses maximum win probability.
def predict_fifth_player(home_team, season, home_players_4, away_players_5, k=5):
    # Load encoders and model
    model = joblib.load('encoders/nba_lineup_model.pkl')
    player_encoder = joblib.load('encoders/player_encoder.pkl')
    team_encoder = joblib.load('encoders/team_encoder.pkl')
    season_encoder = joblib.load('encoders/season_encoder.pkl')
    
    # Get eligible players for the input test data
    key = (home_team, season)
    eligible_players = roster_dict.get(key, set())
    eligible_players = eligible_players - set(home_players_4)
    if not eligible_players:
        return None
    eligible_players = list(eligible_players)
    
    # Encode base features from input test data
    try:
        home_team_enc = team_encoder.transform([home_team])[0]
        season_enc = season_encoder.transform([season])[0]
    except:
        return None 
    
    # Encode away players
    away_sorted = sorted(away_players_5)
    try:
        away_encoded = [player_encoder.transform([p])[0] for p in away_sorted]
    except ValueError as e:
        return None
    
    # Prepare each candidate
    candidates = []
    for candidate in eligible_players:
        home_lineup = sorted(home_players_4 + [candidate])
        try:
            home_encoded = [player_encoder.transform([p])[0] for p in home_lineup]
        except ValueError:
            continue 
        # Construct feature vector
        feature_vec = [home_team_enc, season_enc]
        for i in range(5):
            feature_vec.extend([home_encoded[i], away_encoded[i]])
        # Convert to DataFrame with feature names
        candidate_df = pd.DataFrame([feature_vec], columns=features)
        candidates.append(candidate_df)
    
    if not candidates:
        return None
    
    # Predict probabilities
    probas = model.predict_proba(pd.concat(candidates))[:, 1]
    top_k_indices = np.argsort(probas)[-k:][::-1]
    top_k_players = [eligible_players[i] for i in top_k_indices]
    return top_k_players

# Function to calculate top-k accuracy
def top_k_accuracy(true_player, predicted_players, k):
    return true_player in predicted_players[:k]

# Function to generate test cases automatically
def generate_test_cases(df, num_test_cases=5):
    # Filter rows where outcome is 1
    df_filtered = df[df['outcome'] == 1]
    
    test_cases = []
    for _ in range(num_test_cases):
        # Randomly select a row from the filtered dataset
        random_row = df_filtered.sample(1).iloc[0]
        
        # Extract the parameters to be used in the model
        home_team = random_row['home_team']
        season = random_row['season']
        home_players_4 = sorted([random_row[f'home_{i}'] for i in range(4)])  # First 4 home players
        away_players_5 = sorted([random_row[f'away_{i}'] for i in range(5)])  # All 5 away players
        
        # Store the actual 5th player (for calculating accuracy)
        true_fifth_player = random_row['home_4']
        
        # Append the test case
        test_cases.append({
            'home_team': home_team,
            'season': season,
            'home_players_4': home_players_4,
            'away_players_5': away_players_5,
            'true_fifth_player': true_fifth_player
        })
    return test_cases

def evaluate_top_k_accuracy(test_cases, k=3):
    top_k_accuracies = []
    for case in test_cases:
        # Predict top 3 players for the 5th spot
        top_k_players = predict_fifth_player(
            case['home_team'],
            case['season'],
            case['home_players_4'],
            case['away_players_5'],
            k
        )
        
        # Calculate if the real player in the top 3 choices
        accuracy = top_k_accuracy(case['true_fifth_player'], top_k_players, k)
        top_k_accuracies.append(accuracy)
        
        # Results for the current test case
        print(f"Test Case: {case['home_team']} ({case['season']})")
        print(f"Home Players (4): {case['home_players_4']}")
        print(f"Away Players (5): {case['away_players_5']}")
        print(f"True Fifth Player: {case['true_fifth_player']}")
        print(f"Top {k} Predicted Players: {top_k_players}")
        print(f"Top-{k} Accuracy: {accuracy}")
        print("-" * 50)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(top_k_accuracies)
    print(f"Overall Top-{k} Accuracy: {overall_accuracy:.2f}")

# Generate test cases
num_test_cases = 500
try:
    test_cases = generate_test_cases(df, num_test_cases)
except ValueError as e:
    print(e)
    test_cases = []

if test_cases:
    k = 3  # k value (how many players to find for each lineup)
    evaluate_top_k_accuracy(test_cases, k)