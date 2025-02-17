import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

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