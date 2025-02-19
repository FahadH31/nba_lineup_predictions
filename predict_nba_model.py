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

# Create roster for each team and season
def create_roster(df):
    roster = df.groupby(['home_team', 'season'])[[f'home_{i}' for i in range(5)]].apply(
        lambda x: set(pd.unique(x.values.ravel()))
    ).to_dict()
    return roster

# Encode all categorical features to numerical values 
def encode_features(df, player_encoder, team_encoder, season_encoder):
    df['home_team_encoded'] = team_encoder.transform(df['home_team'])
    df['away_team_encoded'] = team_encoder.transform(df['away_team'])
    df['season_encoded'] = season_encoder.transform(df['season'])
    for i in range(5):
        df[f'home_{i}_encoded'] = player_encoder.transform(df[f'home_{i}'])
        df[f'away_{i}_encoded'] = player_encoder.transform(df[f'away_{i}'])
    return df


data_dir = './csv_files'
df = load_and_preprocess_data(data_dir) # Dataframe to store data from all csv files

# Create dictionary with rosters for each team
roster_dict = create_roster(df)

# Initialize encoders
player_encoder = LabelEncoder()
team_encoder = LabelEncoder()
season_encoder = LabelEncoder()

# Fit all the encoders
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
joblib.dump(model, 'nba_lineup_model.pkl')
joblib.dump(player_encoder, 'player_encoder.pkl')
joblib.dump(team_encoder, 'team_encoder.pkl')
joblib.dump(season_encoder, 'season_encoder.pkl')

# Prediction function with top-k predictions
def predict_fifth_player(home_team, season, home_players_4, away_players_5, starting_min, k=5):
    # Load encoders and model
    model = joblib.load('nba_lineup_model.pkl')
    player_encoder = joblib.load('player_encoder.pkl')
    team_encoder = joblib.load('team_encoder.pkl')
    season_encoder = joblib.load('season_encoder.pkl')
    
    # Get eligible players
    key = (home_team, season)
    eligible_players = roster_dict.get(key, set())
    eligible_players = eligible_players - set(home_players_4)
    if not eligible_players:
        return None
    eligible_players = list(eligible_players)
    
    # Encode base features
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
        return None  # Handle unseen players
    
    # Prepare each candidate
    candidates = []
    for candidate in eligible_players:
        home_lineup = sorted(home_players_4 + [candidate])
        try:
            home_encoded = [player_encoder.transform([p])[0] for p in home_lineup]
        except ValueError:
            continue 
        # Construct feature vector
        feature_vec = [home_team_enc, season_enc, starting_min]
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
