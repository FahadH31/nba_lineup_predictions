import pandas as pd
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

data_dir = './csv_files'
df = load_and_preprocess_data(data_dir) # Dataframe to store data from all csv files

# Create dictionary with rosters for each team
roster_dict = create_roster(df)