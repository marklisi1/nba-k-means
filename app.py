import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
from sklearn.manifold import TSNE 

st.title('NBA Player Analysis')

# Add tabs for playoff and regular season stats
tab1, tab2 = st.tabs(["Playoff Stats", "Regular Season Stats"])

with tab1:
    st.header("Playoff Statistics")
    
    # Load and process playoff data
    @st.cache_data
    def load_playoff_data():
        stat_df = pd.read_csv('playoff_stats.csv')
        stat_df = stat_df[stat_df['MPG'] > 10]
        return stat_df

    stat_df = load_playoff_data()
    
    # Process playoff data
    names = stat_df['NAME']
    position = stat_df['POS']
    X = stat_df.drop(['RANK', 'TEAM','NAME'], axis=1)
    X.POS = pd.factorize(X['POS'])[0]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns),
            ('cat', OneHotEncoder(), ['POS'])
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Clustering
    n_clusters_playoff = st.slider('Number of Playoff Clusters', 2, 8, 5, key='playoff_clusters')
    kmeans = KMeans(n_clusters=n_clusters_playoff, random_state=42)
    stat_df['cluster'] = kmeans.fit_predict(X_processed)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    stat_df['PCA1'] = X_pca[:, 0]
    stat_df['PCA2'] = X_pca[:, 1]

    # Player selection for playoffs
    selected_playoff_player = st.selectbox(
        'Select a playoff player to highlight:',
        stat_df['NAME'].tolist(),
        key='playoff_player'
    )

    # Create playoff visualization
    stat_df['point_size'] = stat_df['NAME'].map(lambda x: 15 if x == selected_playoff_player else 1)
    
    fig1 = px.scatter(
        stat_df,
        x='PCA1',
        y='PCA2',
        color='cluster',
        hover_name=stat_df['NAME'],
        size='point_size',
        title=f'Playoff Players Clustering (Highlight: {selected_playoff_player})',
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
    )

    fig1.update_layout(width=800, height=600)
    st.plotly_chart(fig1)

    # Display playoff player stats
    if selected_playoff_player:
        st.subheader(f"Stats for {selected_playoff_player}")
        player_stats = stat_df[stat_df['NAME'] == selected_playoff_player]
        st.dataframe(player_stats)

with tab2:
    st.header("Regular Season Statistics")
    
    # Load and process regular season data
    @st.cache_data
    def load_regular_season_data():
        reg_stats = pd.read_csv("reg_stats.csv")
        player_count = reg_stats.Player.value_counts()
        dupes = reg_stats[reg_stats['Player'].isin(player_count[player_count > 1].index)]
        dupes = dupes[dupes['Tm'] == 'TOT']
        reg_stats = pd.concat([reg_stats[~reg_stats['Player'].isin(dupes.Player)], dupes], ignore_index=True)
        return reg_stats.dropna()

    reg_stats = load_regular_season_data()

    # Process regular season data
    X = reg_stats.drop(['Player', 'Tm'], axis=1)
    X.Pos = pd.factorize(X['Pos'])[0]

    reg_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns),
            ('cat', OneHotEncoder(), ['Pos'])
        ]
    )

    X_processed = reg_preprocessor.fit_transform(X)

    # Clustering
    n_clusters_regular = st.slider('Number of Regular Season Clusters', 2, 8, 4, key='regular_clusters')
    kmeans = KMeans(n_clusters=n_clusters_regular, random_state=42)
    reg_stats['cluster'] = kmeans.fit_predict(X_processed)

    # t-SNE
    @st.cache_data
    def perform_tsne(X_processed):
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(X_processed)

    X_tsne = perform_tsne(X_processed)
    reg_stats['TSNE1'] = X_tsne[:, 0]
    reg_stats['TSNE2'] = X_tsne[:, 1]

    # Player selection for regular season
    selected_regular_player = st.selectbox(
        'Select a regular season player to highlight:',
        reg_stats['Player'].tolist(),
        key='regular_player'
    )

    # Create regular season visualization
    reg_stats['point_size'] = reg_stats['Player'].map(lambda x: 15 if x == selected_regular_player else 1)
    
    fig2 = px.scatter(
        reg_stats,
        x='TSNE1',
        y='TSNE2',
        color='cluster',
        hover_name=reg_stats['Player'],
        size='point_size',
        title=f'Regular Season Players Clustering (Highlight: {selected_regular_player})',
        labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
    )

    fig2.update_layout(width=800, height=600)
    st.plotly_chart(fig2)

    # Display regular season player stats
    if selected_regular_player:
        st.subheader(f"Stats for {selected_regular_player}")
        player_stats = reg_stats[reg_stats['Player'] == selected_regular_player]
        st.dataframe(player_stats)

# Add sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("""
    This app analyzes NBA player statistics using machine learning clustering techniques.
    - The playoff analysis uses PCA for dimensionality reduction
    - The regular season analysis uses t-SNE for dimensionality reduction
    - You can select the number of clusters and highlight specific players
    """)