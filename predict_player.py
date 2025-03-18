import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os, joblib, numpy as np
from collections import defaultdict
import time

# Cache for loaded model and encoders
model_cache = {}
encoder_cache = {}

# To standardize team abbreviations in the test data  
TEAM_MAPPINGS = {
    'CHO': 'CHA',  	# Charlotte Hornets
    'NOP': 'NOK',  	# New Orleans Pelicans/Oklahoma City
    'NOH': 'NOK',   # New Orleans Hornets/Oklahoma City
    'NJN': 'BRK',	# New Jersey/Brooklyn Nets
    'SEA': 'OKC',	# Seattle Supersonics/Oklahoma City
}

# ------------- LOADING & PROCESSING DATA ------------- # 
# Function to load and preprocess the training data
def load_data(data_dir):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = []

    for file in all_files:  # Append all csv files into one dataframe to be used later
        df = pd.read_csv(file)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    
    # Function to sort players. Original training data has no order, 
    # so this ensures the model doesn't try to learn patterns that don't exist
    def sort_players(row):
        home_players = sorted([row[f'home_{i}'] for i in range(5)])
        away_players = sorted([row[f'away_{i}'] for i in range(5)])
        for i in range(5):
            row[f'home_{i}'] = home_players[i]
            row[f'away_{i}'] = away_players[i]
        return row
    
    df = df.apply(sort_players, axis=1)
    
    return df

# Function to create lists of eligible players
def create_rosters(train_df, test_df=None):
    # Create a roster for each home team and season from training data
    # ex. (GSW, 2010) is one roster, all players that were on GSW in 2010. Separate for each year
    roster = train_df.groupby(['home_team', 'season'])[[f'home_{i}' for i in range(5)]].apply(
        lambda x: set(pd.unique(x.values.ravel()))
    ).to_dict()
    
    # Check away lineups to make sure that rosters are complete
    for idx, row in train_df.iterrows():
        key = (row['away_team'], row['season'])
        if key not in roster:
            roster[key] = set()
        for i in range(5):
            roster[key].add(row[f'away_{i}'])
    
    # To build 2016 rosters (new season in the test data)
    if test_df is not None:
        train_seasons = set(train_df['season'].unique())
        test_seasons = set(test_df['season'].unique())
        new_seasons = test_seasons - train_seasons
        
        if new_seasons:
            print(f"Building rosters of eligible players for new seasons in test data: {new_seasons}")
            
            new_season_df = test_df[test_df['season'].isin(new_seasons)]
            
            # Build rosters for new season from test data
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
                    
            # Add the true fifth player to the roster (from labels file)
            if 'true_fifth_player' in new_season_df.columns:
                for idx, row in new_season_df.iterrows():
                    key = (row['home_team'], row['season'])
                    if key in roster and row['true_fifth_player'] != '?':
                        roster[key].add(row['true_fifth_player'])
    return roster

# Function to encode all categorical features to numerical values (needed for use in Random Forest) 
def encode_features(df, player_encoder, team_encoder, season_encoder):
    df['home_team_encoded'] = team_encoder.transform(df['home_team'])
    df['away_team_encoded'] = team_encoder.transform(df['away_team'])
    df['season_encoded'] = season_encoder.transform(df['season'])
    for i in range(5):
        df[f'home_{i}_encoded'] = player_encoder.transform(df[f'home_{i}'])
        df[f'away_{i}_encoded'] = player_encoder.transform(df[f'away_{i}'])
    return df

# Function to load model and encoders with caching (reduce running time)
def load_model_and_encoders():
    if 'player_encoder' not in encoder_cache:
        encoder_cache['player_encoder'] = joblib.load('saved_files/player_encoder.pkl')
        encoder_cache['team_encoder'] = joblib.load('saved_files/team_encoder.pkl')
        encoder_cache['season_encoder'] = joblib.load('saved_files/season_encoder.pkl')
    
    if 'model' not in model_cache:
        model_cache['model'] = joblib.load('saved_files/nba_lineup_model.pkl')
    
    return (
        model_cache['model'],
        encoder_cache['player_encoder'],
        encoder_cache['team_encoder'],
        encoder_cache['season_encoder']
    )


# ------------- PREDICTION, TESTING, ACCURACY ------------- #
# Prediction function, returns top 'k' options for missing 5th player from the data.
def predict_fifth_player(home_team, season, home_players_4, away_players_5, k):
    # Load encoders and model
    model, player_encoder, team_encoder, season_encoder = load_model_and_encoders()
    
    # Get the eligible players for home team and season of input case
    key = (home_team, season)
    eligible_players = rosters_dict.get(key, set())
    
    if not eligible_players:
        print(f"Warning: No roster found for {home_team} in season {season}")
        return None
    
    # Remove the 4 players already in the lineup from eligible players
    eligible_players = eligible_players - set(home_players_4)
    eligible_players = list(eligible_players)
    
    # Try to encode the home team and season values of input case for use in model
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
    
    feature_rows = []
    valid_candidates = []
    
    # Loop through and evaluate the winning prob. of all eligible players
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
                    valid_player = False
                    break
                home_encoded.append(encoded_p)
            
            if not valid_player:
                continue
                
            # Prepare features
            features = [home_team_enc, season_enc] + home_encoded + away_encoded
            feature_rows.append(features)
            valid_candidates.append(candidate)
            
        except Exception:
            continue
    
    if not valid_candidates:
        print(f"No valid candidates found for {home_team} in season {season}")
        return None
    
    # Define features and columns, get model to make predictions
    features_array = np.array(feature_rows)
    columns = ['home_team_encoded', 'season_encoded',
                  'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 
                  'home_3_encoded', 'home_4_encoded',
                  'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 
                  'away_3_encoded', 'away_4_encoded']
    
    # Create dataframe with the defined features and columns for use in the model
    all_candidates = pd.DataFrame(features_array, columns=columns)
    probs = model.predict_proba(all_candidates)[:, 1] # Use model to predict probability of each candidate being the actual fifth player
        
    # Return the top-k players with highest probabilities
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_players = [valid_candidates[i] for i in top_k_indices]
        
    return top_k_players

# Function to automatically generate test cases
def process_test_cases(test_file, labels_file):
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
        
        # Get sorted known home and away players (to ensure consistency with training data)
        home_players_4 = sorted([p for i, p in enumerate(home_players) if i != missing_idx])
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

# Function to evaluate accuracy (evaluates both top-1 and top-3 accuracy)
def evaluate_accuracy(test_cases, k_max=3):
    # Dictionary to store results for both k values
    k_results = {}
    for k in [1, 3]:
        k_results[k] = defaultdict(list)
    
    # Pass each test case to the prediction function
    for case in test_cases:
        top_k_players = predict_fifth_player(
            case['home_team'],
            case['season'],
            case['home_players_4'],
            case['away_players_5'],
            k_max
        )
        
        # Evaluate success for k=1 and k=3, based on if the actual player shows up in the model's predictions
        for k in [1, 3]:
            success = False
            if top_k_players:
                success = case['true_fifth_player'] in top_k_players[:k]
            
            k_results[k][case['season']].append(success)
        
        # Print information for each test case
        print(f"Test Case: {case['home_team']} vs. {case['away_team']} ({case['season']})")
        print(f"Home Players (4): {case['home_players_4']}")
        print(f"Away Players (5): {case['away_players_5']}")
        print(f"True Fifth Player: {case['true_fifth_player']}")
        print(f"Top {k_max} Predicted Players: {top_k_players}")
        
        for k in [1, 3]:
            success = False
            if top_k_players:
                success = case['true_fifth_player'] in top_k_players[:k]
            print(f"Top-{k} Success: {success}")
        
        print("-" * 50)
    
    # Print overall accuracy summary by season for k=1 and k=3
    for k in [1, 3]:
        print(f"\n===== Top-{k} Accuracy =====")
        
        # Calculate overall accuracy
        all_results = []
        for season, results in k_results[k].items():
            all_results.extend(results)
        overall_accuracy = np.mean(all_results)
        print(f"Overall Top-{k} Accuracy: {overall_accuracy:.2f}")
        
        # Calculate per-season accuracy
        print(f"\nSeason-wise Top-{k} Accuracy:")
        for season in sorted(k_results[k].keys()):
            acc = np.mean(k_results[k][season])
            print(f"Season {season}: {acc:.2f}")
    
# ------------- MAIN CODE ------------- #
if __name__ == '__main__':
    start_time = time.time()

    # Create the encoders directory if it doesn't exist
    os.makedirs('encoders', exist_ok=True)

    # Check if model and encoders already exist
    models_exist = (
        os.path.exists('saved_files/nba_lineup_model.pkl') and
        os.path.exists('saved_files/player_encoder.pkl') and
        os.path.exists('saved_files/team_encoder.pkl') and
        os.path.exists('saved_files/season_encoder.pkl')
    )

    if not models_exist:
        print("Model doesn't exist. Training new model...")
        
        # Load and process training data
        data_dir = './training_files'
        df = load_data(data_dir)
        
        # Load test data for roster creation
        test_file = 'test_files/NBA_test.csv'
        labels_file = 'test_files/NBA_test_labels.csv'
        test_df = pd.read_csv(test_file)
        labels_df = pd.read_csv(labels_file)
        
        # Fix team names in test data
        test_df['home_team'] = test_df['home_team'].replace(TEAM_MAPPINGS)
        test_df['away_team'] = test_df['away_team'].replace(TEAM_MAPPINGS)
        
        # Combine the true labels into test DataFrame
        test_df['true_fifth_player'] = labels_df['removed_value']
        
        # Create rosters from training data and add rosters for new season (2016) from test data
        rosters_dict = create_rosters(df, test_df)
        
        # Save rosters for future use
        joblib.dump(rosters_dict, 'saved_files/rosters_dict.pkl')
        
        # Initialize encoders
        player_encoder = LabelEncoder()
        team_encoder = LabelEncoder()
        season_encoder = LabelEncoder()
        
        # Get all unique values from data for encoding
        all_players_train = pd.unique(df[[f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]].values.ravel())
        all_players_test = pd.unique(test_df[[f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]].values.ravel())
        all_players_test = np.append(all_players_test, labels_df['removed_value'].values)
        all_players = np.unique(np.concatenate([all_players_train, all_players_test]))
        all_players = all_players[all_players != '?']  # Remove placeholder
        
        teams_train = pd.unique(pd.concat([df['home_team'], df['away_team']]))
        teams_test = pd.unique(pd.concat([test_df['home_team'], test_df['away_team']]))
        teams = np.unique(np.concatenate([teams_train, teams_test]))
        
        seasons_train = pd.unique(df['season'])
        seasons_test = pd.unique(test_df['season'])
        seasons = np.unique(np.concatenate([seasons_train, seasons_test]))
        
        # Fit encoders on all data (including test data) to ensure we can encode all values
        player_encoder.fit(all_players)
        team_encoder.fit(teams)
        season_encoder.fit(seasons)
        
        # Add encoded versions of each column to the training dataframe only
        df = encode_features(df, player_encoder, team_encoder, season_encoder)
                
        # ------------- TRAINING MODEL ------------- #
        X = df[['home_team_encoded', 'season_encoded', 
            'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 'home_3_encoded', 'home_4_encoded',
            'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 'away_3_encoded', 'away_4_encoded']]
        y = df['outcome']
        
        # Create weights based on recency
        df['weight'] = df['season'].apply(lambda x: 1 + 0.2 * max(0, x - 2010))  # More weight to recent seasons

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=1, n_jobs=-1)
        model.fit(X, y, sample_weight=df['weight'])
        
        # Save model and encoders
        joblib.dump(model, 'saved_files/nba_lineup_model.pkl')
        joblib.dump(player_encoder, 'saved_files/player_encoder.pkl')
        joblib.dump(team_encoder, 'saved_files/team_encoder.pkl')
        joblib.dump(season_encoder, 'saved_files/season_encoder.pkl')
        
        print("Model trained and saved successfully.")
    else:
        print("Model already exists. Loading saved model.")

    # Load the roster dictionary
    if os.path.exists('saved_files/rosters_dict.pkl'):
        rosters_dict = joblib.load('saved_files/rosters_dict.pkl')
    else:
        print("Rosters don't exist. Creating rosters...")
        data_dir = './training_files'
        df = load_data(data_dir)
        
        test_file = 'test_files/NBA_test.csv'
        labels_file = 'test_files/NBA_test_labels.csv'
        test_df = pd.read_csv(test_file)
        labels_df = pd.read_csv(labels_file)
        
        # Fix team names in test data
        test_df['home_team'] = test_df['home_team'].replace(TEAM_MAPPINGS)
        test_df['away_team'] = test_df['away_team'].replace(TEAM_MAPPINGS)
        
        # Combine the true labels into test DataFrame
        test_df['true_fifth_player'] = labels_df['removed_value']
        
        # Create rosters
        rosters_dict = create_rosters(df, test_df)
        
        # Save rosters for future use
        joblib.dump(rosters_dict, 'saved_files/rosters_dict.pkl')

    # ------------- TESTING MODEL ------------- #
    # Process test cases from test data
    test_file = 'test_files/NBA_test.csv'
    labels_file = 'test_files/NBA_test_labels.csv'
    test_cases = process_test_cases(test_file, labels_file)
    evaluate_accuracy(test_cases)

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    print(f"Total execution time: {elapsed_time_minutes:.2f} minutes")