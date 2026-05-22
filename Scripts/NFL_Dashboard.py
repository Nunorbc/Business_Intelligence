# Complete NFL Streamlit Dashboard (Enhanced Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import base64

# 1.PAGE CONFIGURATION
# =========================================================

# Set wide layout and page title
st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")


# 2.BACKGROUND
# =========================================================

# Load local image and convert to base64 for background
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64(r"Individual task data/NFL_Background.png")

# Define the opacity of the overlay
overlay_opacity = 0.90


# Background styles and configurations
st.markdown(f"""
<style>

.stApp {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, {overlay_opacity});
    z-index: 0;
}}

.block-container {{
    position: relative;
    z-index: 1;
}}

</style>
""", unsafe_allow_html=True)


# 3 TITLE
# =========================================================

# Main title and description
st.title("🏈 NFL Offense vs Defense Analysis (2021–2025)")
st.markdown("""
### Do winning teams rely more on offense or defense, and what defines Super Bowl champions?
This dashboard explores team performance patterns using only regular-season data from the 2021-2025 seasons.
""")


# LOAD DATA
# =========================================================

#Load previously prepared dataset (data_preparation.py)
df = pd.read_csv("nfl_master_dataset.csv")
# Strip spaces from column names to ensure consistency
df.columns = df.columns.str.strip()


# 4. CLEANING + FEATURES
# =========================================================

# Convert relevant columns to numeric and handle errors
numeric_cols = [
    "Wins",
    "Losses",
    "Points For",
    "Points Against",
    "Points Difference",
    "Mean points for by game",
    "Mean points against by game",
    "Mean points difference by game"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values in critical numeric columns
df = df.dropna(subset=numeric_cols)

# Create new features for visualization and modeling
 # Absolute point difference for size encoding
df["PD_Size"] = df["Points Difference"].abs() 

df["Champion_Label"] = np.where(df["Is_Champion"] == 1, df["Team"], "")
df["Champion_Status"] = df["Is_Champion"].map({
    1: "Champion 🏆",
    0: "Not Champion"
})

# Create playoff status flag (not used in final version but can be useful for future enhancements)
# df["Playoff_Status"] = np.where(
#     df["Made_Playoffs"] == 1,
#     "Playoff Team",
#     "Non-Playoff Team"
# )


# 5. SIDEBAR FILTERS (LINKED)
# =========================================================

st.sidebar.header("📌 Filters")


# 5.1 SEASONS (independent)
# -------------------------

# Get unique seasons from the dataset and create a multiselect filter
seasons = sorted(df["Season"].unique())
selected_seasons = st.sidebar.multiselect(
    "Select Seasons",
    seasons,
    default=seasons
)

# Filter dataframe based on selected seasons
df_season = df[df["Season"].isin(selected_seasons)]


# 5.2 CONFERENCES (depends on seasons)
# -------------------------

# Get unique conferences from the filtered dataframe and create a multiselect filter
confs = sorted(df_season["Conference"].unique())
selected_confs = st.sidebar.multiselect(
    "Select Conferences",
    confs,
    default=confs
)

# Filter dataframe based on selected conferences
df_conf = df_season[df_season["Conference"].isin(selected_confs)]


# 5.3 DIVISIONS (depends on conf + season)
# -------------------------

# Get unique divisions from the filtered dataframe and create a multiselect filter
divisions = sorted(df_conf["Division"].unique())
selected_divisions = st.sidebar.multiselect(
    "Select Divisions",
    divisions,
    default=divisions
)

# Filter dataframe based on selected divisions
df_div = df_conf[df_conf["Division"].isin(selected_divisions)]


# 5.4 TEAMS (fully dependent on all previous filters)
# -------------------------

# Get unique teams from the filtered dataframe and create a multiselect filter
teams = sorted(df_div["Team"].unique())
selected_teams = st.sidebar.multiselect(
    "Select Teams",
    teams,
    default=teams
)


# 5.5 FINAL FILTERED DATAFRAME
# -------------------------

# If no teams are selected, show all teams from the previous filters
if len(selected_teams) == 0:
    filtered_df = df_div.copy()
else:
    filtered_df = df_div[df_div["Team"].isin(selected_teams)]


# 6. KPI AND OVERVIEW SECTION
# =========================================================
st.subheader("📊 Overview")

# Calculate KPIs based on the currently filtered dataframe
col1, col2, col3, col4, col5 = st.columns(5)

# Calculate and display KPIs
col1.metric("Seasons", filtered_df["Season"].nunique())
col2.metric("Teams", filtered_df["Team"].nunique())

# Each team plays 17 games, but since each game involves 2 teams, we divide by 2 to avoid double counting
col3.metric("Matches", (len(filtered_df)*17)//2)

# Calculate average points for per game across the filtered dataset
col4.metric("Avg Points/Game", round(filtered_df["Mean points for by game"].mean(), 2))

# Calculate average points difference per game across the filtered dataset (using absolute value for better interpretability)
filtered_df["Abs Mean Points Difference per game"] = filtered_df["Mean points difference by game"].abs()
col5.metric("Avg Point Diff/Game", round(filtered_df["Abs Mean Points Difference per game"].mean(), 2))


# 7. WHAT DRIVES WINS? OFFENSE VS DEFENSE
# =========================================================

# This section explores the relationship between offensive and defensive performance and wins, by using scatter plots with trendlines and correlation coeficients
st.subheader("🔥 What Drives Wins in the NFL?")


# 7.1 CORRELATIONS
# -------------------------

# Calculate correlation coefficients for offense and defense against wins
corr_off = filtered_df["Points For"].corr(filtered_df["Wins"])
corr_def = filtered_df["Points Against"].corr(filtered_df["Wins"])

# Create two columns for side-by-side scatter plots
col1, col2 = st.columns(2)


# 7.2 OFFENSE VS WINS
# -------------------------

# Create scatter plot for offense vs wins with trendline and custom styling. 
# # Show correlation coefficient in the title and add an info box with interpretation.
with col1:
    fig_off = px.scatter(
        filtered_df,
        x="Points For",
        y="Wins",
        trendline="ols",
        color="Points Difference",
        size="Wins",
        size_max=14,
        hover_name="Team",
        title=f"🏈 More Points Scored = More Wins (r = {corr_off:.3f})"
    )

    for trace in fig_off.data:
        if trace.mode == "lines":
            trace.line.color = "black"
            trace.line.width = 1


    st.plotly_chart(fig_off, use_container_width=True)

    st.info(f"""
    Teams that score more points consistently win more games.  
    Correlation: **{corr_off:.3f}**
    """)


# 7.3 DEFENSE VS WINS
# -------------------------

# Create scatter plot for offense vs wins with trendline and custom styling, side by side with offense plot for easy comparison
# # Show correlation coefficient in the title and add an info box with interpretation.
with col2:
    fig_def = px.scatter(
        filtered_df,
        x="Points Against",
        y="Wins",
        trendline="ols",
        color="Points Difference",
        size="Wins",
        size_max=14,
        hover_name="Team",
        title=f"🛡️ Fewer Points Allowed = More Wins (r = {corr_def:.3f})"
    )

    for trace in fig_def.data:
        # trendline is usually a line trace with mode "lines"
        if trace.mode == "lines":
            trace.line.color = "black"
            trace.line.width = 1

    st.plotly_chart(fig_def, use_container_width=True)

    st.info(f"""
    Teams that allow fewer points consistently win more games.  
    Correlation: **{corr_def:.3f}**
    """)

st.success(f"""
**Winning is driven by both sides of the ball, but offense shows the stronger signal.**
\nTeams that score more win more often (r = {corr_off:.3f}) than teams that simply concede less (r = |{abs(corr_def):.3f}|).
""")


# 8. OFFENSE VS DEFENSE BALANCE 
# =========================================================

# Create a scatter plot with "Points For" on the x-axis and "Points Against" on the y-axis, colored by champion status
st.subheader("⚖️ Offensive vs Defensive Balance")

# Use the "Champion_Status" column to color the points, and "PD_Size" (representing points difference) for the size of the points.
# Add dashed lines to indicate league averages for points for and against.
fig_balance = px.scatter(
    filtered_df,
    x="Points For",
    y="Points Against",
    color="Champion_Status",
    size="PD_Size",
    text="Champion_Label",
    hover_name="Team",
    hover_data={
        "Team": False,
        "Division": True,
        "Conference": True,
        "Season": True,
        "Points For": True,
        "Points Against": True,
        "Points Difference": True,
        "Mean points difference by game": True,
        "PD_Size": False,
        "Champion_Status": False,
        "Champion_Label": False
    },
    title="Team Balance Map",
    color_discrete_map={
        "Champion 🏆": "gold",
        "Not Champion": "lightgray"
    },
    custom_data=[
        "Team",
        "Season",
        "Conference",
        "Division",
        "Points For",
        "Points Against",
        "Points Difference",
        "Mean points difference by game",
    ]
)


# Add dashed lines for league averages
avg_pf = filtered_df["Points For"].mean()
avg_pa = filtered_df["Points Against"].mean()
fig_balance.add_vline(x=avg_pf, line_dash="dash", line_width=0.5)
fig_balance.add_hline(y=avg_pa, line_dash="dash", line_width=0.5)

# Some styling adjustments
fig_balance.update_traces(textposition="top center")
fig_balance.update_layout(
    legend=dict(
        font=dict(size=16)  # 👈 increase this value
    )
)

# Move labels above markers and improve readability
fig_balance.update_traces(
    textposition="top center",
    textfont_size=15,
    textfont_color="black"
)

# Customize hover template to show all relevant information in a clear format
fig_balance.update_traces(
    hovertemplate=
    "<b>%{customdata[0]}</b><br><br>" +   # Team (bold + blank line)
    "Season: %{customdata[1]}<br>" +
    "Conference: %{customdata[2]}<br>" +
    "Division: %{customdata[3]}<br>" +
    "Points For: %{customdata[4]}<br>" +
    "Points Against: %{customdata[5]}<br>" +
    "Points Difference: %{customdata[6]}<br>" +
    "Mean points difference by game: %{customdata[7]}<br>" +
    "<extra></extra>"
)

st.plotly_chart(fig_balance, use_container_width=True)

st.success(f"""
**Champions cluster in the “high offense + strong defense” region.**  \n
Championship success is driven by strong point differential, not offense or defense alone.
Winning teams are not just high-scoring—they also tend to concede fewer points than league average.
The most successful teams sit near the upper-right offensive tier, but importantly, below the defensive average line, showing balance matters 
as is where the champions tend to cluster.
""")



# 9. TABLE OF TOP 10 NFL TEAMS BY WINS (2021–2025, DYNAMIC FILTER)
# =========================================================

st.subheader("🏅 Top 10 Teams by Wins - Aggregated Stats")

# Create a table showing the top 10 teams by total wins across the selected seasons, by including some of their key performance metrics and champion status.
top10_a = (
    filtered_df
    .groupby("Team", as_index=False)
    .agg({
        "Division": "first",
        "Wins": "sum",
        "Losses": "sum",
        "Ties": "sum",
        "Points For": "sum",
        "Points Against": "sum",
        "Points Difference": "sum",
        "Is_Champion": "sum"
    })
)

# Rename columns for clarity
top10_a = top10_a.rename(columns={
    "Points For": "Total Points For",
    "Points Difference": "Total Point Difference",
    "Is_Champion": "Championships"
})

# Sort by total wins in descending order and select the top 10 teams
top10_a = top10_a.sort_values("Total Point Difference", ascending=False).head(10)

# display the top 10 teams in a table with selected columns.
st.dataframe(
    top10_a[[
        "Team",
        "Division",
        "Wins",
        "Losses",
        "Ties",
        "Total Points For",
        "Points Against",
        "Total Point Difference",
        "Championships"
    ]],
    hide_index=True,
)

st.success(f"""
    **Across the top 10 teams by wins, the data shows consistently high offensive output paired with solid defensive performance.** 
    \nNotably, 4 Super Bowl champions appear within this top 10 across both conferences between 2021 and 2025, reinforcing that 
    championship teams typically combine elite offense with disciplined defense rather than relying on just one side of the game.
""")

# 10. TABLE OF TOP 10 TEAM PERFORMANCES BY POINT DIFFERENTIAL (2021-2025, DYNAMIC FILTER)
# =========================================================

# Create a table showing the top 10 teams by point differential, which is the strongest overall predictor of championship success, 
# across the selected seasons, by including some of their key performance metrics and champion status.
st.subheader("🏅 Top 10 performances by Point Differential")

# Sort the filtered dataframe by "Points Difference" in descending order and select the top 10 teams
top10_b = filtered_df.sort_values(
    "Points Difference",
    ascending=False
).head(10)

# Display the top 10 teams in a table with selected columns
st.dataframe(
    top10_b[[
        "Season",
        "Team",
        "Division",
        "Wins",
        "Losses",
        "Ties",\
        "Points For",
        "Points Against",
        "Points Difference",
        "Is_Champion"
    ]],
    hide_index=True
)

st.success(f"""
    **Only one team from the top 10 performances between 2021 and 2025, went on to become a champion.**
    \nThis suggests that although point differential is a strong indicator of championship success,
    it is not sufficient on its own to determine championship outcomes.
""")


st.markdown("""
## 🎯 Key Findings

All conclusions below are drawn exclusively from regular-season data:

- **Offensive strength shows a stronger correlation with wins than defensive performance.**
- **Defensive strength remains essential for maintaining a competitive edge.**
- **Champions are balanced elite teams with strong offense and disciplined defense.**
- **Point differential is the strongest overall predictor as champions cluster is present in the high offense + strong defense region.**
- **However, point differential alone is not enough to predict the champion, as many top-performing teams by this metric fail to win the championship.**
""")

###########################################################
# =========================================================
# EXTRAS - EXPLORATORY ANALYSIS AND MODELING (It was added to enrich the analysis))
# =========================================================



# 11. CHAMPION PROBABILITY MODEL
# =========================================================

# Build a simple logistic regression model to predict championship probability based on 
# points for, points against, and point differential.
st.subheader("🔮 NFL Champion Probability Model")

# preparation for modeling: select season for prediction, train on all data, 
# predict probabilities for the selected season, and identify actual vs predicted champions

# Create two columns: one for the season selection dropdown and the other to control layout width
col1, col2 = st.columns([1, 7])

with col1:
    model_season = st.selectbox(
        "Season",
        sorted(filtered_df["Season"].unique())
    )

# 11.1 TRAIN MODEL
# -------------------------

# Prepare training data using the currently filtered dataframe
train_df = filtered_df.copy()

# We will use "Points For", "Points Against", and "Points Difference" as features to predict "Is_Champion"
X_train = train_df[
    ["Points For", "Points Against", "Points Difference"]
]

# Target variable is "Is_Champion", which is binary (1 for champion, 0 for non-champion)
y_train = train_df["Is_Champion"].astype(int)


# 11.2 SAFETY CHECK
# -------------------------

# Ensure that there are both champions and non-champions in the training data to avoid errors during model fitting 
# and in visualization. If not, show a warning
if y_train.nunique() < 2:
    st.warning("⚠️ Not enough data to train model (need both champions and non-champions in current filters).")
    st.stop()

# Fit a logistic regression model to predict championship probability based on the selected features
model = LogisticRegression()
model.fit(X_train, y_train)


# 11.3 FILTER SEASON
# -------------------------

# Filter the dataframe to the selected season for prediction and visualization
model_df = filtered_df[filtered_df["Season"] == model_season].copy()

# Prepare test data for the selected season using the same features as the training data
X_test = model_df[
    ["Points For", "Points Against", "Points Difference"]
]

# Predict championship probabilities for the teams in the selected season using the trained model
model_df["Champion_Prob"] = model.predict_proba(X_test)[:, 1]


# 11.4 IDENTIFY ACTUAL VS PREDICTED CHAMPIONS
# # -------------------------

# Identify the actual champion(s) in the selected season, 
# the team with the highest predicted championship probability,
#  and the other teams for visualization purposes.
actual = model_df[model_df["Is_Champion"] == 1]
pred_idx = model_df["Champion_Prob"].idxmax()
predicted = model_df.loc[[pred_idx]]

exclude_idx = pd.concat([actual, predicted]).index
others = model_df.drop(exclude_idx)



# 11.5 MODEL'S EVALUATION - CHAMPIONSHIP PROBABILITY VS POINT DIFFERENTIAL
# # -------------------------

# Create a scatter plot showing the relationship between points difference and predicted championship probability 
# for the selected season.
fig_model = go.Figure()

# Add three traces to the scatter plot: one for the actual champion(s), one for the predicted champion, 
# and one for the other teams, with custom styling and informative hover templates.
# (gold for actual champions; red for predicted; grey for the others)

# ACTUAL CHAMPION
# -------------------------
fig_model.add_trace(go.Scatter(
    x=actual["Points Difference"],
    y=actual["Champion_Prob"],
    mode="markers+text",
    name="Actual Champion",
    text=actual["Team"],
    textposition="top center",
    textfont=dict(size=13),
    customdata=actual[["Team", "Division"]],
    hovertemplate=
        "<b>%{customdata[0]}</b><br><br>" +   # Team in bold
        "Division: %{customdata[1]}<br>" +
        "Points Difference: %{x}<br>" +
        "Win Prob: %{y:.3f}" +
        "<extra></extra>",
    marker=dict(
        size=24,
        color="gold",
        opacity=1,
        line=dict(width=0)
    )
))


# PREDICTED CHAMPION
# -------------------------
fig_model.add_trace(go.Scatter(
    x=predicted["Points Difference"],
    y=predicted["Champion_Prob"],
    mode="markers+text",
    name="Predicted Champion",
    text=predicted["Team"],
    textposition="top center",
    textfont=dict(size=13),
    customdata=predicted[["Team", "Division"]],
    hovertemplate=
        "<b>%{customdata[0]}</b><br><br>" +   # Team in bold
        "Division: %{customdata[1]}<br>" +
        "Points Difference: %{x}<br>" +
        "Win Prob: %{y:.3f}" +
        "<extra></extra>",
    marker=dict(
        size=20,
        color="red",
        opacity=0.6,
        line=dict(width=0)
    )
))


#NOT CHAMPIONS (OTHERS)
# -------------------------
fig_model.add_trace(go.Scatter(
    x=others["Points Difference"],
    y=others["Champion_Prob"],
    mode="markers",
    name="Not Champion",
    customdata=others[["Team", "Division"]],
    hovertemplate=
        "<b>%{customdata[0]}</b><br><br>" +   # Team in bold
        "Division: %{customdata[1]}<br>" +
        "Points Difference: %{x}<br>" +
        "Win Prob: %{y:.3f}" +
        "<extra></extra>",
    marker=dict(
        size=16,
        color="lightgray",
        opacity=0.6,
        line=dict(width=0)
    )
))


# 11.6 LAYOUT CUSTOMIZATION
# =========================================================

# Customize the layout of the figure with a descriptive title, axis labels, and a legend with larger font 
# for better readability.
fig_model.update_layout(
    title=f"Championship Probability vs Point Differential ({model_season})",
    legend=dict(
        font=dict(size=13),
        traceorder="normal"
    ),
    xaxis_title="Points Difference",
    yaxis_title="Champion Probability",
    xaxis=dict(title_font=dict(size=16)),
    yaxis=dict(title_font=dict(size=16))
)

st.plotly_chart(fig_model, use_container_width=True)

st.info("""**How can the champion be predicted based on the analyzed metrics?**
    \nThis logistic regression model is designed to estimate the likelihood of a team winning the championship based on key 
    performance indicators: points scored, points allowed, and point differential.
    By combining offensive output and defensive strength into a single probabilistic framework, the model quantifies 
    how overall team performance translates into championship success.
""")

# 12. ANIMATED EVOLUTION OF TEAMS ACROSS SEASONS
# =========================================================

# Create an animated scatter plot showing the evolution of teams across seasons, 
# with points for and against on the axes, colored by conference, and sized by wins.
st.subheader("📈 NFL Teams Evolution")

# Get the list of available teams from the currently filtered dataframe
teams_available = sorted(filtered_df["Team"].unique())

#Create two columns: one for the team selection dropdown and the other to control layout width
col1, col2 = st.columns([1, 4])
with col1:
    selected_teams = st.multiselect(
        "Teams (empty = ALL)",
        teams_available
    )

with col2:
    st.write("")

# If no teams are selected, use the entire dataframe for the animation; otherwise, filter to the selected teams.
if len(selected_teams) == 0:
    anim_df = filtered_df.copy()
else:
    anim_df = filtered_df[filtered_df["Team"].isin(selected_teams)]

# Fix axis ranges for consistent animation scaling
x_range = [
    filtered_df["Points For"].min() - 10,
    filtered_df["Points For"].max() + 10
]

y_range = [
    filtered_df["Points Against"].min() - 10,
    filtered_df["Points Against"].max() + 10
]

# Create animated scatter plot
fig_anim = px.scatter(
    anim_df,
    x="Points For",
    y="Points Against",
    animation_frame="Season",
    animation_group="Team",
    size="Wins",
    color="Conference",
    symbol="Conference",          # <-- enables shape mapping
    text="Team",                  # <-- show labels
    hover_name="Team",
    hover_data={
        "Team": False,
        "Conference": False,
        "Division": True,
        "Season": True,
        "Points For": True,
        "Points Against": True,
        "Points Difference": True,
    },
    range_x=x_range,
    range_y=y_range,
    title="League Evolution by Season (Filtered Teams)",
    color_discrete_map={
        "AFC": "darkgreen",
        "NFC": "darkorange"
    },
    symbol_map={
        "AFC": "circle",
        "NFC": "square"
    },
     custom_data=[
        "Team",
        "Season",
        "Division",
        "Points For",
        "Points Against",
        "Points Difference",
    ],
)

fig_anim.update_traces(
    hovertemplate=
    "<b>%{customdata[0]}</b><br><br>" +   # Team (bold + blank line)
    "Division: %{customdata[2]}<br>" +
    "Season: %{customdata[1]}<br>" +
    "Points For: %{customdata[3]}<br>" +
    "Points Against: %{customdata[4]}<br>" +
    "Points Difference: %{customdata[5]}<br>" +
    "<extra></extra>"
)

# move labels above markers + improve readability
fig_anim.update_traces(
    textposition="top center",
    textfont_size=11
)

st.plotly_chart(fig_anim, use_container_width=True)

st.info("""**How teams evolve over time by mapping offensive and defensive performance across multiple seasons ?**
    \nAs the animation progresses, it reveals shifts in team balance, showing how strong contenders typically combine high scoring with defensive control, 
    and how championship-level teams tend to cluster in the upper-performing regions of the chart.
""")


