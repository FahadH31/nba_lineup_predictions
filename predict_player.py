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

# Function to create roster for each team and season
def create_rosters(df):
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
# Prediction function, predicts 2 best options for 5th player to maximize winning.
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
    
    home_team_enc = team_encoder.transform([home_team])[0]
    season_enc = season_encoder.transform([season])[0]
    
    away_encoded = []
    for p in away_players_5:
        try:
            encoded_p = player_encoder.transform([p])[0]
        except ValueError:
            encoded_p = -1  # Default for unknown player
        away_encoded.append(encoded_p)
    
    # Evaluate all eligible players
    candidates = []
    for candidate in eligible_players:
        # Create and encode home lineup with the current eligible player
        home_lineup = sorted(home_players_4 + [candidate])
        home_encoded = []
        for p in home_lineup:
            try:
                encoded_p = player_encoder.transform([p])[0]
            except ValueError:
                encoded_p = -1  # Default for unknown player
            home_encoded.append(encoded_p)

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
    
    if not candidates:
        return None
    
    # Use model to predict winning probabilities
    all_candidates = pd.concat(candidates)
    probs = model.predict_proba(all_candidates)[:, 1]
    
    # Return the top-k players with highest probabilities
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_players = [eligible_players[i] for i in top_k_indices]
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
df = load_data(data_dir) # Initialize dataframe

rosters_dict = create_rosters(df) # Intialize the dictionary with all eligible players for each team + season

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
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=1, n_jobs=-1)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'encoders/nba_lineup_model.pkl')
joblib.dump(player_encoder, 'encoders/player_encoder.pkl')
joblib.dump(team_encoder, 'encoders/team_encoder.pkl')
joblib.dump(season_encoder, 'encoders/season_encoder.pkl')

    # ------------- TESTING MODEL ------------- #
#test_cases = generate_test_cases('test_files/NBA_test.csv', 'test_files/NBA_test_labels.csv')
test_cases = generate_test_cases('test_files/NBA_test_2016.csv', 'test_files/NBA_test_labels_2016.csv')

# Evaluate model accuracy with new test cases
k = 3
evaluate_accuracy(test_cases, k)

# I want to update this code to do the following:
# - if a new season appears in the test data (for example, I am training on 2007-2015. 2016 would be new/unseen):
    # build the rosters for the new season from the test data. (in my case, 2016). ONLY the rosters for the unseen season should be built from test data, since all others will already be built via training data.
    # don't train the model on the new season's data, but ensure the recommended players for test cases from the new season are only from the rosters of that new season. 

# some cases are def wrong. where the model only provides 1 option
# something to do with rosters. players aren't being recognized on the correct rosters.
