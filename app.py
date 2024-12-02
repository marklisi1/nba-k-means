import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
from sklearn.manifold import TSNE 

st.set_page_config(layout="wide", page_title="NBA Player Analysis", initial_sidebar_state="expanded")

# Load data
@st.cache_data
def load_playoff_data():
    stat_df = pd.read_csv('playoff_stats.csv')
    stat_df = stat_df[stat_df['MPG'] > 10]
    return stat_df

@st.cache_data
def load_regular_season_data():
    reg_stats = pd.read_csv("reg_stats.csv")
    player_count = reg_stats.Player.value_counts()
    dupes = reg_stats[reg_stats['Player'].isin(player_count[player_count > 1].index)]
    dupes = dupes[dupes['Tm'] == 'TOT']
    reg_stats = pd.concat([reg_stats[~reg_stats['Player'].isin(dupes.Player)], dupes], ignore_index=True)
    return reg_stats.dropna()

def generate_color_sequence(n_clusters):
    """Generate a sequence of bright, distinct colors"""
    bright_colors = [
        '#FF4B4B',  # bright red
        '#45E6B0',  # bright turquoise
        '#FFB931',  # bright yellow
        '#3CB9FF',  # bright blue
        '#FF61D9',  # bright pink
        '#00FF00',  # bright green
        '#FF8C00',  # bright orange
        '#9D72FF'   # bright purple
    ]
    # Ensure we have enough colors by cycling if necessary
    while len(bright_colors) < n_clusters:
        bright_colors += bright_colors
    return bright_colors[:n_clusters]

# Sidebar controls
with st.sidebar:
    st.header("Analysis Options")
    
    # Dataset selection
    dataset_choice = st.radio(
        "Choose Dataset",
        ["Playoff Stats", "Regular Season Stats"]
    )
    
    # Dimensionality reduction technique
    dim_reduction = st.radio(
        "Choose Dimensionality Reduction",
        ["PCA", "t-SNE"]
    )
    
    # Number of clusters
    n_clusters = st.slider('Number of Clusters', 2, 8, 5)

# Load and process data based on selection
if dataset_choice == "Playoff Stats":
    df = load_playoff_data()
    name_col = 'NAME'
    drop_cols = ['RANK', 'TEAM', 'NAME']
    position_col = 'POS'
else:
    df = load_regular_season_data()
    name_col = 'Player'
    drop_cols = ['Player', 'Tm']
    position_col = 'Pos'

# Preprocessing
X = df.drop(drop_cols, axis=1)
X[position_col] = pd.factorize(X[position_col])[0]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns),
        ('cat', OneHotEncoder(), [position_col])
    ]
)

X_processed = preprocessor.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_processed)

# Dimensionality reduction
if dim_reduction == "PCA":
    reducer = PCA(n_components=2)
    X_reduced = reducer.fit_transform(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)
    explained_variance = reducer.explained_variance_ratio_
    st.write(f"Explained variance ratio: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
else:
    reducer = TSNE(n_components=2, random_state=42)
    X_reduced = reducer.fit_transform(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed)

# Add reduced dimensions to dataframe
df['Dim1'] = X_reduced[:, 0]
df['Dim2'] = X_reduced[:, 1]


st.markdown('One of my friends thinks Jamal Murray is way better than I do. Here, we use both PCA and t-SNE on some statistical metrics from both regular season and playoff performance in the 2023-24 NBA season to see which NBA players are most statistically similar.')

# Player selection
selected_player = st.selectbox(
    'Select a player to highlight:',
    df[name_col].tolist()
)

# Create visualization
df['point_size'] = df[name_col].map(lambda x: 15 if x == selected_player else 1)

# convert cluster labels to strings
df['cluster'] = 'Cluster ' + df['cluster'].astype(str)

fig = px.scatter(
    df,
    x='Dim1',
    y='Dim2',
    color='cluster',
    hover_name=df[name_col],
    size='point_size',
    title=f'NBA Players Clustering - {dataset_choice} ({dim_reduction})',
    labels={
        'Dim1': f'{dim_reduction} Component 1', 
        'Dim2': f'{dim_reduction} Component 2',
        'cluster': 'Cluster'
    },
    color_discrete_sequence=generate_color_sequence(n_clusters),
    template='plotly_dark'
)

# Update layout with larger height
fig.update_layout(
    width=800, 
    height=900,  # Increased height
    paper_bgcolor='rgba(0,0,0,0.0)',
    plot_bgcolor='rgba(0,0,0,0.0)',
    font_color='white',
    showlegend=True,  # Ensure legend is shown
    legend=dict(
        title_font_color='white',
        font_color='white',
        bgcolor='rgba(0,0,0,0.0)'
    )
)

# More explicit axis styling
fig.update_traces(
    marker=dict(
        line=dict(width=1, color='white')
    )
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    linecolor='white',
    linewidth=1
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    linecolor='white',
    linewidth=1
)

# Use a different Plotly theme setting when displaying the chart
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Display player stats
if selected_player:
    st.subheader(f"Stats for {selected_player}")
    player_stats = df[df[name_col] == selected_player]
    st.dataframe(player_stats)

# Add explanation in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### About the Visualization Methods
    
    **PCA (Principal Component Analysis)**
    - Linear dimensionality reduction
    - Preserves global structure
    - Faster computation
    
    **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
    - Non-linear dimensionality reduction
    - Better at preserving local structure
    - Can reveal more complex patterns
    """)