import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
from sklearn.manifold import TSNE 


# ### Playoff Stats

stat_df = pd.read_csv('playoff_stats.csv')
stat_df.columns

# Filter the df to contain only players that played significant minutes
stat_df = stat_df[stat_df['MPG'] > 10]

names = stat_df['NAME']
position = stat_df['POS']
X = stat_df.drop(['RANK', 'TEAM','NAME'], axis=1)
X


# preprocessing for position (categorical)
X.POS = pd.factorize(X['POS'])[0]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns),
        ('cat', OneHotEncoder(), ['POS'])
    ]
)

X_processed = preprocessor.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
stat_df['cluster'] = kmeans.fit_predict(X_processed)

# Optionally, reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Add the PCA components to the DataFrame for plotting
stat_df['PCA1'] = X_pca[:, 0]
stat_df['PCA2'] = X_pca[:, 1]

fig = px.scatter(
    stat_df,
    x='PCA1',
    y='PCA2',
    color='cluster',
    hover_name=stat_df['NAME'],  # Assuming player names are the index, otherwise use df['player_name']
    title='NBA Players Clustering',
    labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
)

fig.update_layout(
    width=800,  # Width of the plot in pixels
    height=600  # Height of the plot in pixels
)
# Show the plot
fig.show()


# In[31]:


import plotly.express as px

def plot_player_highlight(df, player_name):
    """
    Generates an interactive scatter plot with a specific player highlighted.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the player data, including PCA components and cluster labels.
    player_name (str): The name of the player to highlight.
    
    Returns:
    plotly.graph_objects.Figure: The generated plot with the highlighted player.
    """
    
    # Check if the player_name exists in the DataFrame
    if not df['NAME'].str.contains(player_name).any():
        raise ValueError(f"Player '{player_name}' not found in the DataFrame index.")
    
    # Create a column to determine the size of each point (increase size for the highlighted player)
    df['point_size'] = df['NAME'].map(lambda x: 15 if x == player_name else 1)
    
    # Create a column to determine the color of the point (distinct color for the highlighted player)
    df['highlight_color'] = df['NAME'].map(lambda x: 'red' if x == player_name else 'blue')

    # Plot using Plotly
    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='cluster',
        hover_name=df['NAME'],  # Assuming player names are the index, otherwise use df['player_name']
        size='point_size',  # Size of points, highlight player will be larger
        color_discrete_sequence=df['highlight_color'],  # Custom colors
        title=f'NBA Players Clustering (Highlight: {player_name})',
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
    )

    fig.update_layout(
    width=800,  # Width of the plot in pixels
    height=600  # Height of the plot in pixels
    )

    # Show the plot
    fig.show()
    print(stat_df['point_size'])

# Example usage
# plot_player_highlight(df, 'LeBron James')

plot_player_highlight(stat_df, 'LeBron James')


# ### Regular Season Stats

# In[32]:


reg_stats = pd.read_csv("reg_stats.csv")
player_count = reg_stats.Player.value_counts() 

dupes = reg_stats[reg_stats['Player'].isin(player_count[player_count > 1].index)]
dupes = dupes[dupes['Tm'] == 'TOT']

reg_stats = pd.concat([reg_stats[~reg_stats['Player'].isin(dupes.Player)], dupes], ignore_index=True)
reg_stats = reg_stats.dropna()
reg_stats.head()


# In[33]:


names = reg_stats['Player']
position = reg_stats['Pos']
X = reg_stats.drop(['Player', 'Tm'], axis=1)

# preprocessing for position (categorical)
X.Pos = pd.factorize(X['Pos'])[0]
X


# In[ ]:


reg_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns),
        ('cat', OneHotEncoder(), ['Pos'])
    ]
)


X_processed = reg_preprocessor.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
reg_stats['cluster'] = kmeans.fit_predict(X_processed)

# Reduce dimensions for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42) 
X_tsne = tsne.fit_transform(X_processed) 

# Add the t-SNE components to the DataFrame for plotting
reg_stats['TSNE1'] = X_tsne[:, 0]  
reg_stats['TSNE2'] = X_tsne[:, 1]

# Create a scatter plot using Plotly
fig = px.scatter(
    reg_stats,
    x='TSNE1',
    y='TSNE2',
    color='cluster',
    hover_name=reg_stats['Player'], 
    title='NBA Players Clustering (t-SNE)',
    labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
)

# Update the layout for the plot dimensions
fig.update_layout(
    width=800,  # Width of the plot in pixels
    height=600  # Height of the plot in pixels
)

# Show the plot
fig.show()


# In[ ]:


def plot_player_highlight(df, player_name):
    """
    Generates an interactive scatter plot with a specific player highlighted.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the player data, including PCA components and cluster labels.
    player_name (str): The name of the player to highlight.
    
    Returns:
    plotly.graph_objects.Figure: The generated plot with the highlighted player.
    """
    
    # Check if the player_name exists in the DataFrame
    if not df['Player'].str.contains(player_name).any():
        raise ValueError(f"Player '{player_name}' not found in the DataFrame index.")
    
    # increase size for the highlighted player
    df['point_size'] = df['Player'].map(lambda x: 15 if x == player_name else 1)
    
    # distinct color for the highlighted player
    df['highlight_color'] = df['Player'].map(lambda x: 'red' if x == player_name else 'blue')

    # Plot using Plotly
    fig = px.scatter(
        df,
        x='TSNE1',
        y='TSNE2',
        color='cluster',
        hover_name=df['Player'], 
        size='point_size', 
        color_discrete_sequence=df['highlight_color'],  # Custom colors
        title=f'NBA Players Clustering (Highlight: {player_name})',
        labels={'TSNE1': 'TSNE Component 1', 'TSNE2': 'TSNE Component 2'},
    )

    fig.update_layout(
    width=800,  # Width of the plot in pixels
    height=600  # Height of the plot in pixels
    )

    # Show the plot
    fig.show()

# Example usage
# plot_player_highlight(df, 'LeBron James')

plot_player_highlight(reg_stats, 'Jamal Murray')


# In[37]:


reg_stats.columns


# In[ ]:




