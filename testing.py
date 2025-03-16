# Current Code:
# Works accurately on test data from 2007-2015 (90%+ accuracy)
# Not very accurately on unseen/trained test data from 2016 (55% accuracy)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os, joblib, numpy as np
from collections import defaultdict

TEAM_MAPPINGS = {
    'CHO': 'CHA',  	# Charlotte Hornets
    'NOP': 'NOK',  	# New Orleans Pelicans/Oklahoma City
    'NOH': 'NOK',   # New Orleans Hornets/Oklahoma City
    'NJN': 'BRK',	# New Jersey/Brooklyn Nets
    'SEA': 'OKC',	# Seattle Supersonics/Oklahoma City
}

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

# Function to create roster for each team and season - modified to handle test data
def create_rosters(train_df, test_df=None):
    # Create a roster for each team and season from training data
    roster = train_df.groupby(['home_team', 'season'])[[f'home_{i}' for i in range(5)]].apply(
        lambda x: set(pd.unique(x.values.ravel()))
    ).to_dict()
    
    # Supplement with away team data (important for completeness)
    for idx, row in train_df.iterrows():
        key = (row['away_team'], row['season'])
        if key not in roster:
            roster[key] = set()
        for i in range(5):
            roster[key].add(row[f'away_{i}'])
    
    # If test data is provided, build rosters for new seasons only
    if test_df is not None:
        # Identify new seasons in test data
        train_seasons = set(train_df['season'].unique())
        test_seasons = set(test_df['season'].unique())
        new_seasons = test_seasons - train_seasons
        
        if new_seasons:
            print(f"Building rosters for new seasons in test data: {new_seasons}")
            
            # Create temporary dataframe with only new seasons
            new_season_df = test_df[test_df['season'].isin(new_seasons)]
            
            # Build rosters for new seasons from test data
            for idx, row in new_season_df.iterrows():
                # Add home team players
                key = (row['home_team'], row['season'])
                if key not in roster:
                    roster[key] = set()
                for i in range(5):
                    player = row[f'home_{i}']
                    if player != '?':  # Skip unknown players
                        roster[key].add(player)
                
                # Add away team players
                key = (row['away_team'], row['season'])
                if key not in roster:
                    roster[key] = set()
                for i in range(5):
                    roster[key].add(row[f'away_{i}'])
                    
            # Add the true fifth player to the roster (from labels)
            if 'true_fifth_player' in new_season_df.columns:
                for idx, row in new_season_df.iterrows():
                    key = (row['home_team'], row['season'])
                    if key in roster and row['true_fifth_player'] != '?':
                        roster[key].add(row['true_fifth_player'])
    
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
# Prediction function, predicts k best options for 5th player to maximize winning.
def predict_fifth_player(home_team, season, home_players_4, away_players_5, k):
    # Load encoders and model
    model = joblib.load('encoders/nba_lineup_model.pkl')
    player_encoder = joblib.load('encoders/player_encoder.pkl')
    team_encoder = joblib.load('encoders/team_encoder.pkl')
    season_encoder = joblib.load('encoders/season_encoder.pkl')
    
    # Get the appropriate roster (eligible players) based on home team and season of input case
    key = (home_team, season)
    eligible_players = rosters_dict.get(key, set())
    
    if not eligible_players:
        print(f"Warning: No roster found for {home_team} in season {season}")
        return None
    
    # Remove the 4 players already in the lineup from eligible players
    eligible_players = eligible_players - set(home_players_4)
    
    if not eligible_players:
        print(f"Warning: No eligible players left for {home_team} in season {season} after removing existing players")
        return None
    
    eligible_players = list(eligible_players)
    
    try:
        home_team_enc = team_encoder.transform([home_team])[0]
    except ValueError:
        print(f"Unknown home team: {home_team}")
        return None
        
    try:
        season_enc = season_encoder.transform([season])[0]
    except ValueError:
        print(f"Unknown season: {season}")
        return None
    
    # Encode away players
    away_encoded = []
    for p in away_players_5:
        try:
            encoded_p = player_encoder.transform([p])[0]
        except ValueError:
            print(f"Unknown away player: {p}, using default encoding")
            encoded_p = -1  # Default for unknown player
        away_encoded.append(encoded_p)
    
    # Evaluate all eligible players
    candidates = []
    valid_candidates = []
    
    for candidate in eligible_players:
        try:
            # Create and encode home lineup with the current eligible player
            home_lineup = sorted(home_players_4 + [candidate])
            home_encoded = []
            valid_player = True
            
            for p in home_lineup:
                try:
                    encoded_p = player_encoder.transform([p])[0]
                except ValueError:
                    print(f"Unknown home player: {p}, skipping candidate")
                    valid_player = False
                    break
                home_encoded.append(encoded_p)
            
            if not valid_player:
                continue
                
            # Prepare features
            features = [home_team_enc, season_enc] + home_encoded + away_encoded
            
            # Create candidate DataFrame
            candidate_df = pd.DataFrame(
                [features], columns=['home_team_encoded', 'season_encoded', 
                                     'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 
                                     'home_3_encoded', 'home_4_encoded',
                                     'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 
                                     'away_3_encoded', 'away_4_encoded'])
            candidates.append(candidate_df)
            valid_candidates.append(candidate)
        except Exception as e:
            print(f"Error processing candidate {candidate}: {e}")
            continue
    
    if not candidates:
        print(f"No valid candidates found for {home_team} in season {season}")
        return None
    
    # Use model to predict winning probabilities
    all_candidates = pd.concat(candidates)
    probs = model.predict_proba(all_candidates)[:, 1]
    
    # Return the top-k players with highest probabilities
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_players = [valid_candidates[i] for i in top_k_indices]
    top_k_probs = probs[top_k_indices]
    
    return top_k_players

# Function to automatically generate test cases
def generate_test_cases(test_file, labels_file):
    test_df = pd.read_csv(test_file)
    labels_df = pd.read_csv(labels_file)
    
    # Combine the true labels into test DataFrame
    test_df['true_fifth_player'] = labels_df['removed_value']
    
    # Fix team names in test data
    test_df['home_team'] = test_df['home_team'].replace(TEAM_MAPPINGS)
    test_df['away_team'] = test_df['away_team'].replace(TEAM_MAPPINGS)

    test_cases = []
    for _, row in test_df.iterrows():
        # Extract home players and find missing position
        home_players = [row[f'home_{i}'] for i in range(5)]
        missing_idx = [i for i, player in enumerate(home_players) if player == '?'][0]
        
        # Get sorted known home players
        home_players_4 = sorted([p for i, p in enumerate(home_players) if i != missing_idx])
        
        # Get sorted away players
        away_players_5 = sorted([row[f'away_{i}'] for i in range(5)])
        
        test_cases.append({
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'season': row['season'],
            'home_players_4': home_players_4,
            'away_players_5': away_players_5,
            'true_fifth_player': row['true_fifth_player']
        })
    return test_cases

def evaluate_accuracy(test_cases, k):
    season_results = defaultdict(list)
    
    for case in test_cases:
        top_k_players = predict_fifth_player(
            case['home_team'],
            case['season'],
            case['home_players_4'],
            case['away_players_5'],
            k
        )
        
        success = False
        if top_k_players:
            success = case['true_fifth_player'] in top_k_players[:k]

        season_results[case['season']].append(success)
        
        print(f"Test Case: {case['home_team']} vs. {case['away_team']} ({case['season']})")
        print(f"Home Players (4): {case['home_players_4']}")
        print(f"Away Players (5): {case['away_players_5']}")
        print(f"True Fifth Player: {case['true_fifth_player']}")
        print(f"Top {k} Predicted Players: {top_k_players}")
        print(f"Success: {success}")
        print("-" * 50)
    
    # Calculate overall accuracy
    all_results = []
    for season, results in season_results.items():
        all_results.extend(results)
    overall_accuracy = np.mean(all_results)
    print(f"\nOverall Top-{k} Accuracy: {overall_accuracy:.2f}")
    
    # Calculate per-season accuracy
    print("\nSeason-wise Accuracy:")
    for season in sorted(season_results.keys()):
        acc = np.mean(season_results[season])
        count = len(season_results[season])
        print(f"Season {season}: {acc:.2f} (n={count})")



# ------------- MAIN CODE ------------- #
data_dir = './csv_files'
df = load_data(data_dir)  # Initialize training dataframe

# Load test data
test_file = 'test_files/NBA_test_2016.csv'
labels_file = 'test_files/NBA_test_labels_2016.csv'
test_df = pd.read_csv(test_file)
labels_df = pd.read_csv(labels_file)

# Fix team names in test data
test_df['home_team'] = test_df['home_team'].replace(TEAM_MAPPINGS)
test_df['away_team'] = test_df['away_team'].replace(TEAM_MAPPINGS)

# Combine the true labels into test DataFrame
test_df['true_fifth_player'] = labels_df['removed_value']

# Create rosters from training data and supplement with new seasons from test data
rosters_dict = create_rosters(df, test_df)

# Initialize encoders
player_encoder = LabelEncoder()
team_encoder = LabelEncoder()
season_encoder = LabelEncoder()

# Get unique values from both training and test data
all_players_train = pd.unique(df[[f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]].values.ravel())
all_players_test = pd.unique(test_df[[f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]].values.ravel())
all_players_test = np.append(all_players_test, labels_df['removed_value'].values)
all_players = np.unique(np.concatenate([all_players_train, all_players_test]))
all_players = all_players[all_players != '?']  # Remove placeholder

# Teams from both datasets
teams_train = pd.unique(pd.concat([df['home_team'], df['away_team']]))
teams_test = pd.unique(pd.concat([test_df['home_team'], test_df['away_team']]))
teams = np.unique(np.concatenate([teams_train, teams_test]))

# Seasons from both datasets
seasons_train = pd.unique(df['season'])
seasons_test = pd.unique(test_df['season'])
seasons = np.unique(np.concatenate([seasons_train, seasons_test]))

# Fit encoders on all data (including test data) to ensure we can encode all values
player_encoder.fit(all_players)
team_encoder.fit(teams)
season_encoder.fit(seasons)

# Add encoded versions of each column to the training dataframe ONLY
df = encode_features(df, player_encoder, team_encoder, season_encoder)

# ------------- TRAINING MODEL ------------- #
# Train only on training data
X = df[['home_team_encoded', 'season_encoded', 
       'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 'home_3_encoded', 'home_4_encoded',
       'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 'away_3_encoded', 'away_4_encoded']]
y = df['outcome']

# Train Random Forest model
model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=1, n_jobs=-1)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'encoders/nba_lineup_model.pkl')
joblib.dump(player_encoder, 'encoders/player_encoder.pkl')
joblib.dump(team_encoder, 'encoders/team_encoder.pkl')
joblib.dump(season_encoder, 'encoders/season_encoder.pkl')

# ------------- TESTING MODEL ------------- #
# Generate test cases from test data
test_cases = generate_test_cases(test_file, labels_file)

# Evaluate model accuracy with test cases
k = 4
evaluate_accuracy(test_cases, k)