import pandas as pd
import glob


data_path = "Individual task data/"

# 1. LOAD ALL 5 SEASON FILES FOR BOTH AFC AND NFC CONFERENCES
# ----------------------------

# Use glob to find all relevant files
afc_files = glob.glob(f"{data_path}*_AFC.xlsx")
nfc_files = glob.glob(f"{data_path}*_NFC.xlsx")

# Function to load and tag conference and season
def load_and_tag(files, conference):
    dfs = []
    
    for file in files:
        df = pd.read_excel(file, index_col=None)
        
        # Extract year from filename
        year = int(file.split("\\")[1].split("_")[0])
        
        df["Season"] = year
        df["Conference"] = conference
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# Load and tag both conferences
afc_df = load_and_tag(afc_files, "AFC")
nfc_df = load_and_tag(nfc_files, "NFC")


# 2. COMBINE DATAFRAMES AND CREATE NEW FEATURES
# ----------------------------

# Combine AFC and NFC dataframes
teams_df = pd.concat([afc_df, nfc_df], ignore_index=True)

# Create new features
teams_df["Mean points for by game"] = teams_df['PF'] / (teams_df['W'] + teams_df['L'])
teams_df["Mean points against by game"] = teams_df['PA'] / (teams_df['W'] + teams_df['L'])
teams_df["Mean points difference by game"] = teams_df['PD'] / (teams_df['W'] + teams_df['L'])


# 3. CLEAN DATA (STRIP SPACES, CONVERT TO NUMERIC, REMOVE SPECIAL CHARACTERS, REMOVE UNNECESSARY COLUMNS, RENAME COLUMNS)
# ----------------------------

# Strip spaces from column names
teams_df.columns = teams_df.columns.str.strip()

# Ensure numeric columns are correct
for col in ["W", "L", "PF", "PA"]:
    teams_df[col] = pd.to_numeric(teams_df[col], errors="coerce")

# Clean team names (remove special characters and extra spaces)
teams_df["Tm"] = (
teams_df["Tm"]
.str.replace("*", "", regex=False)
.str.replace("+", "", regex=False)
.str.strip()
)

# Remove unnecessary columns
teams_df = teams_df.drop(columns=["MoV", "SoS", "SRS", "OSRS", "DSRS"], errors="ignore")

# Rename columns for clarity
teams_df.rename(columns={
    "PA": "Points Against",
    "PF": "Points For",
    "PD": "Points Difference",
    "W": "Wins",
    "L": "Losses",
    "T": "Ties",
    "Tm": "Team"
}, inplace=True)

# Once Washington Football Team was renamed to Washington Commanders in 2022, we need to standardize the name across the dataset
teams_df["Team"] = teams_df["Team"].str.replace("Washington Football Team", "Washington Commanders")

# Fill NaN values in "Ties" column with 0 (assuming that if it's NaN, it means there were no ties)
teams_df["Ties"] = teams_df["Ties"].fillna(0)


# 5. LOAD PLAYOFFS FILES (the goal is to create flags for "Made Playoffs" and "Is Champion")
# ----------------------------

# Use glob to find all playoff files
playoff_files = glob.glob(f"{data_path}*_pl.xlsx")

# Load and combine playoff data
playoff_dfs = []
for file in playoff_files:
    pl_df = pd.read_excel(file, index_col=None)
    
    year = int(file.split("\\")[1].split("_")[0])
    pl_df["Season"] = year
    
    playoff_dfs.append(pl_df)

# Combine all playoff dataframes
playoffs_df = pd.concat(playoff_dfs, ignore_index=True)


# 6. REMOVE UNNECESSARY COLUMNS AND RENAME COLUMNS IN PLAYOFFS DATAFRAME
# ----------------------------

# Remove unnecessary columns
playoffs_df = playoffs_df.drop(columns=["Unnamed: 4", "Data.1"], errors="ignore")

# Rename columns for clarity
playoffs_df.rename(columns={
    "Pts": "Winner pts",
    "Pts.1": "Loser pts"
}, inplace=True)


# 7. CREATE FLAGS (PLAYOFF + CHAMPION)
# ----------------------------

# Adjust column name depending on your file
playoff_teams = playoffs_df["Winner/tie"].unique()

# Create "Made_Playoffs" flag
teams_df["Made_Playoffs"] = teams_df["Team"].isin(playoff_teams)

# Strip spaces from "Week" column to ensure consistency
playoffs_df["Week"] = playoffs_df["Week"].str.strip()

# Create "Is_Champion" flag by identifying Super Bowl winner for each season
superbowl_winners = (
    playoffs_df[playoffs_df["Week"] == "SuperBowl"]
    .groupby("Season")["Winner/tie"]
    .first()
)

# Map the Super Bowl winners to the teams dataframe to create the "Is_Champion" flag
teams_df["Is_Champion"] = teams_df.apply(
    lambda row: row["Team"] == superbowl_winners.get(row["Season"]),
    axis=1
)

# Convert boolean flags to integers (1 for True, 0 for False)
teams_df["Is_Champion"] = teams_df["Is_Champion"].astype(int)

# Load divisions dataset and map divisions to teams
divisions_dataset = pd.read_csv(f"{data_path}nfl_conferences_dataset.csv")
teams_df["Division"] = teams_df["Team"].map(
    divisions_dataset.set_index("Team")["Division"]
)

# Re order columns to have "Division" after "Conference" and not in the end
cols = teams_df.columns.tolist()
cols.remove("Division")
conf_index = cols.index("Conference")
cols.insert(conf_index + 1, "Division")

teams_df = teams_df[cols]


# 8. FINAL OUTPUT
# ----------------------------

# Sort by season for better readability
teams_df = teams_df.sort_values(by="Season")

# Save final dataset
teams_df.to_csv("nfl_master_dataset.csv", index=False)


