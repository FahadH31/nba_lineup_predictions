# Currently obtaining 84% accuracy on 500 tests.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os, joblib, numpy as np

# ------------- LOADING & PROCESSING DATA ------------- # 
# Function to load and preprocess the data
def load_data(data_dir):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = []

    for file in all_files:  # Append all csv files into one dataframe to be used later
        df = pd.read_csv(file)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    
    # Function to sort all players alphabetically in the dataframe
    def sort_players(row):
        home_players = sorted([row[f'home_{i}'] for i in range(5)])
        away_players = sorted([row[f'away_{i}'] for i in range(5)])
        for i in range(5):
            row[f'home_{i}'] = home_players[i]
            row[f'away_{i}'] = away_players[i]
        return row
    
    # Apply the sorting on the dataframe
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


# ------------- PREDICTION, TESTING ACCURACY ------------- #
# Prediction function, predicts 3 best options for 5th player to maximize winning.
def predict_fifth_player(home_team, season, home_players_4, away_players_5, k):
    # Load encoders and model
    model = joblib.load('encoders/nba_lineup_model.pkl')
    player_encoder = joblib.load('encoders/player_encoder.pkl')
    team_encoder = joblib.load('encoders/team_encoder.pkl')
    season_encoder = joblib.load('encoders/season_encoder.pkl')
    
    # Get the appropriate roster (eligible players) based on home team and season of input case
    key = (home_team, season)
    eligible_players = rosters_dict.get(key, set())
    eligible_players = eligible_players - set(home_players_4)
    if not eligible_players:
        return None
    eligible_players = list(eligible_players)
    
    # Encode features from input case
    home_team_enc = team_encoder.transform([home_team])[0]
    season_enc = season_encoder.transform([season])[0]
    away_encoded = [player_encoder.transform([p])[0] for p in away_players_5]
    
    # Evaluate all eligible players
    candidates = []
    for candidate in eligible_players:
        # Create and encode home lineup with the current eligible player
        home_lineup = sorted(home_players_4 + [candidate])  # Sorting here to ensure that when the candidate is appended, the lineup stays sorted as in the dataset and trained model
        home_encoded = [player_encoder.transform([p])[0] for p in home_lineup]

        # Features to be saved in the dataframe (encoded versions of all the input case info)
        features = [home_team_enc, season_enc]
        for i in range(5):
            features.extend([home_encoded[i]])
        for i in range(5):
            features.extend([away_encoded[i]])

        # Convert the candidate data into a dataframe (with column names) (this is necessary because the model expects the input as a dataframe)
        candidate_df = pd.DataFrame(
            [features], columns=['home_team_encoded', 'season_encoded', 
                                    'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 'home_3_encoded', 'home_4_encoded',
                                    'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 'away_3_encoded', 'away_4_encoded'])
        candidates.append(candidate_df) # Save each dataframe into the list of candidates
    
    if not candidates:
        return None
    
    # Use model to predict winning probabilities
    all_candidates = pd.concat((candidates)) # Combine each dataframe in the list into one large dataframe
    probs = model.predict_proba(all_candidates)[:, 1] # Model predicts the win probability for each candidate lineup.
    
    # Return the top-3 players that provide the highest winning %
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_players = [eligible_players[i] for i in top_k_indices]
    return top_k_players

# Function to automatically generate test cases
def generate_test_cases(df, num_test_cases):
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

def evaluate_accuracy(test_cases, k):
    top_k_accuracies = []
    for case in test_cases:
        # Return top 3 players for the current test case
        top_k_players = predict_fifth_player(
            case['home_team'],
            case['season'],
            case['home_players_4'],
            case['away_players_5'],
            k
        )
        
        # Calculate if the real player is in the top 3 choices
        if(case['true_fifth_player'] in top_k_players[:k]):
            success = True
        else:
            success = False

        top_k_accuracies.append(success)
        
        # Print results for the current test case
        print(f"Test Case: {case['home_team']} ({case['season']})")
        print(f"Home Players (4): {case['home_players_4']}")
        print(f"Away Players (5): {case['away_players_5']}")
        print(f"True Fifth Player: {case['true_fifth_player']}")
        print(f"Top {k} Predicted Players: {top_k_players}")
        print(f"Success: {success}")
        print("-" * 50)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(top_k_accuracies)
    print(f"Overall Top-{k} Accuracy: {overall_accuracy:.2f}")


# ------------- MAIN CODE ------------- #
data_dir = './csv_files'
df = load_data(data_dir) # Initialize dataframe

rosters_dict = create_roster(df) # Intialize the dictionary with all eligible players for each team + season

# Initialize encoders
player_encoder = LabelEncoder()
team_encoder = LabelEncoder()
season_encoder = LabelEncoder()

# Fit all encoders on the data
all_players = pd.unique(df[[f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]].values.ravel()) # Player encoder for all player columns
player_encoder.fit(all_players)
teams = pd.unique(pd.concat([df['home_team'], df['away_team']])) # Team encoder
team_encoder.fit(teams)
seasons = pd.unique(df['season']) # Season encoder
season_encoder.fit(seasons)

# Add encoded (numerical) versions of each column to the dataframe
df = encode_features(df, player_encoder, team_encoder, season_encoder)

    # ------------- TRAINING MODEL ------------- #
# Gather input features and target
X = df[['home_team_encoded', 'season_encoded', 
       'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 'home_3_encoded', 'home_4_encoded',
       'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 'away_3_encoded', 'away_4_encoded']]
y = df['outcome']

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'encoders/nba_lineup_model.pkl')
joblib.dump(player_encoder, 'encoders/player_encoder.pkl')
joblib.dump(team_encoder, 'encoders/team_encoder.pkl')
joblib.dump(season_encoder, 'encoders/season_encoder.pkl')

# Generate test cases
num_test_cases = 100
test_cases = generate_test_cases(df, num_test_cases)

# Evaluate model accuracy (with top-3 predicted players)
k = 3
evaluate_accuracy(test_cases, k)