import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


try:
    file_path = r'C:\Users\yash3\OneDrive\Desktop\spotify dataset.csv'
    spotify_data = pd.read_csv(file_path)

    # Data Preprocessing
    print("Performing data pre-processing operations...")
    
    numeric_cols = spotify_data.select_dtypes(include=['float64', 'int64']).columns
    spotify_data_numeric = spotify_data[numeric_cols]

    spotify_data_numeric.dropna(inplace=True)

    # Data Analysis and Visualizations
    print("Performing data analysis and visualizations...")
    sns.set(style="whitegrid")
    
    #Distribution of Playlist Genres
    plt.figure(figsize=(10, 6))
    sns.countplot(y='playlist_genre', data=spotify_data, order=spotify_data['playlist_genre'].value_counts().index)
    plt.title('Distribution of Playlist Genres')
    plt.xlabel('Count')
    plt.ylabel('Playlist Genre')
    plt.show()

    #Correlation Matrix
    print("Calculating correlation matrix...")
    plt.figure(figsize=(12, 8))
    correlation_matrix = spotify_data_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.show()

    #Scatter plot for tempo vs. loudness by playlist genre
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tempo', y='loudness', data=spotify_data, hue='playlist_genre')
    plt.title('Tempo vs Loudness by Playlist Genre')
    plt.xlabel('Tempo')
    plt.ylabel('Loudness')
    plt.show()

    #Clustering based on Playlist Genres and Playlist Names
    print("Performing clustering on the dataset...")
    features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']  # Features for clustering
    X = spotify_data[features]

    #Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Perform KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    spotify_data['cluster'] = kmeans.fit_predict(X_scaled)

    #Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=spotify_data['cluster'], palette='viridis')
    plt.title('KMeans Clusters based on Danceability and Energy')
    plt.xlabel('Danceability (scaled)')
    plt.ylabel('Energy (scaled)')
    plt.show()

    #Evaluate Clusters with Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, spotify_data['cluster'])
    print(f'Silhouette Score for KMeans Clustering: {silhouette_avg:.2f}')

    print("Building recommendation model...")

    def recommend_songs(genre):
        # Recommending based on playlist genre
        recommended_songs = spotify_data[spotify_data['playlist_genre'] == genre]['track_name'].sample(5)
        return recommended_songs

    genre_to_recommend = 'pop'  # Example genre
    print(f"Top 5 song recommendations for genre '{genre_to_recommend}':")
    print(recommend_songs(genre_to_recommend))

except Exception as e:
    print(f"An error occurred: {e}")
